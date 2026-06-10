"""Hybrid SPAD integrators with PerPixelBayesian-compatible streaming API."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.quanta_neural_networks.ops.array_ops import torch_quantile
from ultralytics.quanta_neural_networks.ops.image import nearest_neighbor_inpaint


class SpatioTemporalEvidenceAccumulation(nn.Module):
    """
    Spatio-temporal evidence accumulation (STEA) for SPAD photon cubes.

    The tensor path is fully convolutional along time: causal 1D temporal
    bases, Bernoulli variance-normalized evidence, causal 3D evidence smoothing,
    then sigmoid soft routing between slow and fast rates.
    """

    def __init__(
        self,
        fast_window: int = 16,
        slow_window: int = 128,
        temporal_window: int = 5,
        fast_tau: float | None = None,
        sharpness: float = 1.0,
        bias: float = 3.0,
        eps: float = 1e-5,
        chunk_size: int = 320,
        subsampling: int = 1,
        hot_pixel_mask: np.ndarray | None = None,
        normalize: bool = False,
        quantile: float = 1.0,
    ):
        super().__init__()
        self.fast_window = max(int(fast_window), 1)
        self.slow_window = max(int(slow_window), 1)
        self.temporal_window = max(int(temporal_window), 1)
        self.fast_tau = float(fast_tau) if fast_tau is not None else max(self.fast_window / 4.0, 1.0)
        self.sharpness = float(sharpness)
        self.bias = float(bias)
        self.eps = float(eps)
        self.chunk_size = max(int(chunk_size), 1)
        self.subsampling = max(int(subsampling), 1)
        self.hot_pixel_mask = hot_pixel_mask
        self.normalize = bool(normalize)
        self.quantile = float(quantile)

        self.t_absolute = 0
        self._h, self._w, self._t = None, None, None
        self.register_buffer("photon_history", None)
        self.register_buffer("kl_history", None)
        self._rebuild_kernels()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(fast_window={self.fast_window}, "
            f"slow_window={self.slow_window}, temporal_window={self.temporal_window}, "
            f"sharpness={self.sharpness}, bias={self.bias})"
        )

    @staticmethod
    def _normalize_kernel(taps: Tensor) -> Tensor:
        return taps / taps.sum().clamp(min=1e-12)

    def _rebuild_kernels(self, device: torch.device | str | None = None) -> None:
        """Build causal 1D temporal bases and the 3D evidence smoother."""
        if device is None and hasattr(self, "fast_kernel"):
            device = self.fast_kernel.device
        device = device or "cpu"

        # F.conv1d is cross-correlation. With left padding, kernel[-1] touches
        # the current bin, so taps are stored oldest -> newest.
        gamma_age = torch.arange(self.fast_window, 0, -1, dtype=torch.float32)
        fast = gamma_age * torch.exp(-gamma_age / max(self.fast_tau, 1e-6))
        slow = torch.ones(self.slow_window, dtype=torch.float32)
        stea = torch.ones(
            (1, 1, self.temporal_window, 3, 3),
            dtype=torch.float32,
        )

        self.register_buffer("fast_kernel", self._normalize_kernel(fast).view(1, 1, -1).to(device))
        self.register_buffer("slow_kernel", self._normalize_kernel(slow).view(1, 1, -1).to(device))
        self.register_buffer("stea_kernel", self._normalize_kernel(stea).to(device))
        self.register_buffer("fast_noise_gain", self.fast_kernel.square().sum().view(1, 1, 1))
        self.register_buffer("slow_noise_gain", self.slow_kernel.square().sum().view(1, 1, 1))

    def update_hyperparams(self, **kwargs) -> None:
        """Update STEA attributes; rebuild convolution kernels when needed."""
        rebuild_keys = {"fast_window", "slow_window", "temporal_window", "fast_tau"}
        # Compatibility with the previous hybrid CLI/API naming.
        if kwargs.get("kernel_size") is not None and kwargs.get("slow_window") is None:
            kwargs["slow_window"] = kwargs.pop("kernel_size")
        kwargs.pop("prior_strength", None)
        kwargs.pop("gating_tau", None)
        kwargs.pop("max_filter_size", None)
        kwargs.pop("min_filter_size", None)

        needs_rebuild = False
        for name, value in kwargs.items():
            if hasattr(self, name) and value is not None:
                if name in {"fast_window", "slow_window", "temporal_window", "chunk_size", "subsampling"}:
                    value = max(int(value), 1)
                elif name in {"fast_tau", "sharpness", "bias", "eps", "quantile"}:
                    value = float(value)
                elif name == "normalize":
                    value = bool(value)
                setattr(self, name, value)
                needs_rebuild = needs_rebuild or name in rebuild_keys
        if needs_rebuild:
            self._rebuild_kernels(device=self.fast_kernel.device)
            self._clear_histories()

    def _clear_histories(self) -> None:
        self.photon_history = None
        self.kl_history = None

    def set_cube(self, photon_cube: Tensor) -> None:
        self._h, self._w, self._t = map(int, photon_cube.shape)

    def clamp_recons(self, recons: Tensor) -> Tensor:
        if recons.numel() == 0:
            return recons.float()
        recons = recons.float()
        max_value = 1.0
        if self.normalize:
            max_value = torch_quantile(recons, self.quantile).clamp(min=1e-6)
        return (recons / max_value).clamp(0, 1)

    def _history_or_zeros(self, history: Tensor | None, h: int, w: int, length: int, x: Tensor) -> Tensor:
        if length <= 0:
            return x.new_zeros(h, w, 0)
        if history is None or tuple(history.shape[:2]) != (h, w):
            return x.new_zeros(h, w, length)
        if int(history.shape[-1]) >= length:
            return history[..., -length:].to(device=x.device, dtype=x.dtype)
        pad = x.new_zeros(h, w, length - int(history.shape[-1]))
        return torch.cat([pad, history.to(device=x.device, dtype=x.dtype)], dim=-1)

    def _causal_conv1d(self, x_flat: Tensor, kernel: Tensor, history_hwt: Tensor) -> Tensor:
        hlen = int(kernel.shape[-1]) - 1
        hist_flat = history_hwt.reshape(-1, 1, hlen)
        return F.conv1d(torch.cat([hist_flat, x_flat], dim=-1), kernel)

    def _temporal_basis(self, photon_cube: Tensor) -> tuple[Tensor, Tensor]:
        h, w, t = map(int, photon_cube.shape)
        x = photon_cube.float()
        x_flat = x.reshape(h * w, 1, t)

        fast_hist = self._history_or_zeros(self.photon_history, h, w, self.fast_window - 1, x)
        slow_hist = self._history_or_zeros(self.photon_history, h, w, self.slow_window - 1, x)
        y_fast = self._causal_conv1d(x_flat, self.fast_kernel, fast_hist)
        y_slow = self._causal_conv1d(x_flat, self.slow_kernel, slow_hist)
        return y_fast.clamp(self.eps, 1.0 - self.eps), y_slow.clamp(self.eps, 1.0 - self.eps)

    def _smooth_kl(self, k_raw_hwt: Tensor) -> Tensor:
        h, w, _ = map(int, k_raw_hwt.shape)
        hist = self._history_or_zeros(self.kl_history, h, w, self.temporal_window - 1, k_raw_hwt)
        k_dhw = torch.cat([hist, k_raw_hwt], dim=-1).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        k_dhw = F.pad(k_dhw, (1, 1, 1, 1, 0, 0))
        return F.conv3d(k_dhw, self.stea_kernel)

    @torch.no_grad()
    def _integrate_full_with_debug(self, photon_cube: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        h, w, t = map(int, photon_cube.shape)
        if t == 0:
            empty = photon_cube.new_zeros(h, w, 0, dtype=torch.float32)
            debug = {
                "y_fast": empty,
                "y_slow": empty,
                "k_raw": empty,
                "k_smoothed": empty,
                "route_weight": empty,
                "fused": empty,
                "fused_blocks": empty,
                "motion_blocks": empty,
                "doe_blocks": empty,
                "weight_gamma_blocks": empty,
                "fused_mean": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "fused_last": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "motion_peak": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "motion_blend": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "y_scales_last": photon_cube.new_zeros(h, w, 2, dtype=torch.float32),
                "scores_last": photon_cube.new_zeros(h, w, 2, dtype=torch.float32),
                "k_smoothed_last": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "route_weight_last": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "recons_prenorm": empty,
            }
            return empty, debug

        y_fast, y_slow = self._temporal_basis(photon_cube)
        # Bernoulli variance-normalized fast/slow disagreement. This keeps
        # bright static regions from looking dynamic only because their absolute
        # photon variance is larger.
        delta = y_fast - y_slow
        p_ref = y_slow.clamp(self.eps, 1.0 - self.eps)
        var_delta = p_ref * (1.0 - p_ref) * (self.fast_noise_gain + self.slow_noise_gain)
        k_raw_flat = delta.square() / var_delta.clamp(min=self.eps)
        k_raw_hwt = k_raw_flat.reshape(h, w, t)
        k_smoothed = self._smooth_kl(k_raw_hwt)
        k_s_flat = k_smoothed.squeeze(0).squeeze(0).permute(1, 2, 0).reshape(h * w, 1, t)
        weights = torch.sigmoid(self.sharpness * (k_s_flat - self.bias))
        fused_flat = (1.0 - weights) * y_slow + weights * y_fast
        fused = fused_flat.reshape(h, w, t)

        max_photon_hist = max(self.fast_window, self.slow_window) - 1
        max_kl_hist = self.temporal_window - 1
        self.photon_history = photon_cube.float()[..., -max_photon_hist:].detach() if max_photon_hist > 0 else None
        self.kl_history = k_raw_hwt[..., -max_kl_hist:].detach() if max_kl_hist > 0 else None

        y_fast_hwt = y_fast.reshape(h, w, t)
        y_slow_hwt = y_slow.reshape(h, w, t)
        weight_hwt = weights.reshape(h, w, t)
        k_s_hwt = k_smoothed.squeeze(0).squeeze(0).permute(1, 2, 0)
        debug = {
            "y_fast": y_fast_hwt,
            "y_slow": y_slow_hwt,
            "k_raw": k_raw_hwt,
            "k_smoothed": k_s_hwt,
            "route_weight": weight_hwt,
            "fused": fused,
            # Compatibility field names used by existing visualization code.
            "fused_blocks": fused,
            "motion_blocks": weight_hwt,
            "doe_blocks": k_s_hwt,
            "weight_gamma_blocks": weight_hwt,
            "fused_mean": fused.mean(dim=-1),
            "fused_last": fused[..., -1],
            "motion_peak": weight_hwt.max(dim=-1).values,
            "motion_blend": weight_hwt[..., -1],
            "y_scales_last": torch.stack([y_fast_hwt[..., -1], y_slow_hwt[..., -1]], dim=-1),
            "scores_last": torch.stack([k_s_hwt[..., -1], weight_hwt[..., -1]], dim=-1),
            "k_smoothed_last": k_s_hwt[..., -1],
            "route_weight_last": weight_hwt[..., -1],
            "recons_prenorm": fused,
        }
        return fused, debug

    def _subsample_reconstruction(self, fused_hwt: Tensor) -> Tensor:
        h, w, t = map(int, fused_hwt.shape)
        if t <= 0:
            return fused_hwt.new_zeros(h, w, 0, dtype=torch.float32)
        if t < self.subsampling:
            return fused_hwt[..., -1:].float()
        return fused_hwt[..., self.subsampling - 1 :: self.subsampling].float()

    @torch.no_grad()
    def process_photon_cube(
        self,
        photon_cube: Tensor,
        subsampling: int | None = None,
        hot_pixel_mask: np.ndarray | None = None,
        quantile: float | None = None,
        normalize: bool | None = None,
        clear_states: bool = True,
        chunk_size: int | None = None,
        fast_window: int | None = None,
        slow_window: int | None = None,
        temporal_window: int | None = None,
        fast_tau: float | None = None,
        sharpness: float | None = None,
        bias: float | None = None,
        eps: float | None = None,
        **kwargs,
    ) -> Tensor:
        if clear_states:
            self.t_absolute = 0
            self._clear_histories()

        self.update_hyperparams(
            subsampling=subsampling,
            hot_pixel_mask=hot_pixel_mask,
            normalize=normalize,
            quantile=quantile,
            chunk_size=chunk_size,
            fast_window=fast_window,
            slow_window=slow_window,
            temporal_window=temporal_window,
            fast_tau=fast_tau,
            sharpness=sharpness,
            bias=bias,
            eps=eps,
            **kwargs,
        )

        self.set_cube(photon_cube)
        fused, _ = self._integrate_full_with_debug(photon_cube)
        recons = self._subsample_reconstruction(fused)
        if self.hot_pixel_mask is not None:
            recons = nearest_neighbor_inpaint(recons, self.hot_pixel_mask)
        recons = self.clamp_recons(recons)
        self.t_absolute += self._t
        return recons

    @torch.no_grad()
    def process_photon_cube_with_motion(
        self,
        photon_cube: Tensor,
        subsampling: int | None = None,
        hot_pixel_mask: np.ndarray | None = None,
        quantile: float | None = None,
        normalize: bool | None = None,
        clear_states: bool = True,
        chunk_size: int | None = None,
        fast_window: int | None = None,
        slow_window: int | None = None,
        temporal_window: int | None = None,
        fast_tau: float | None = None,
        sharpness: float | None = None,
        bias: float | None = None,
        eps: float | None = None,
        **kwargs,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if clear_states:
            self.t_absolute = 0
            self._clear_histories()

        self.update_hyperparams(
            subsampling=subsampling,
            hot_pixel_mask=hot_pixel_mask,
            normalize=normalize,
            quantile=quantile,
            chunk_size=chunk_size,
            fast_window=fast_window,
            slow_window=slow_window,
            temporal_window=temporal_window,
            fast_tau=fast_tau,
            sharpness=sharpness,
            bias=bias,
            eps=eps,
            **kwargs,
        )

        self.set_cube(photon_cube)
        fused, motion_debug = self._integrate_full_with_debug(photon_cube)
        recons = self._subsample_reconstruction(fused)
        motion_debug["recons_prenorm"] = self._subsample_reconstruction(motion_debug["recons_prenorm"])
        if self.hot_pixel_mask is not None:
            recons = nearest_neighbor_inpaint(recons, self.hot_pixel_mask)
        recons = self.clamp_recons(recons)
        self.t_absolute += self._t
        return recons, motion_debug

    def forward(self, photon_cube: Tensor) -> Tensor:
        return self._integrate_full_with_debug(photon_cube)[0]


class GatedMultiScaleEMA(nn.Module):
    """
    White-box spatio-temporal integration for SPAD sensors.

    Uses a heterogeneous 1D Filter Bank (Gamma, EMA, Boxcar) and threshold-free
    Bayesian soft-routing: D_KL(boxcar || p_m) with boxcar as the stable reference.
    Exposes the same ``process_photon_cube`` contract as
    :class:`~ultralytics.quanta_neural_networks.integrator.PerPixelBayesian`.
    """

    def __init__(
        self,
        alphas: list[float] | None = None,
        chunk_size: int = 320,
        kernel_size: int = 64,
        prior_strength: float = 1.0,
        gating_tau: float = 0.1,
        subsampling: int = 1,
        hot_pixel_mask: np.ndarray | None = None,
        normalize: bool = False,
        quantile: float = 1.0,
        spatial_batch_size: int = 16384,
        max_filter_size: int = 3,
    ):
        """
        :param alphas: Slowness proxies for scales. Used to define prior rewards.
        :param chunk_size: Nominal raw window length (``det_spad`` slicing).
        :param kernel_size: FIR length; non-overlapping segments contain this many bins.
        :param prior_strength: Bayesian prior weight pushing toward slower (smoother) EMAs.
        :param gating_tau: Softmax temperature for KL-divergence routing.
        :param subsampling: Output temporal subsampling.
        :param hot_pixel_mask: Optional hot-pixel mask for inpainting.
        :param normalize: If True, normalize reconstruction by quantile.
        :param quantile: Upper quantile used when ``normalize`` is True.
        :param spatial_batch_size: Pixels per ``conv1d`` batch (lower uses less VRAM).
        :param max_filter_size: Odd max-pool on gamma weight for motion/blend only (1 = off).
        """
        super().__init__()

        max_filter_size = max(int(max_filter_size), 1)
        if max_filter_size % 2 == 0:
            raise ValueError("max_filter_size must be odd (or 1 to disable max-pool)")

        if alphas is None:
            alphas = [1, 0.05, 0.02, 0.01, 0]

        self.chunk_size = max(int(chunk_size), 1)
        self.kernel_size = max(int(kernel_size), 1)
        self.prior_strength = float(prior_strength)
        self.gating_tau = float(gating_tau)
        self.subsampling = max(int(subsampling), 1)
        self.hot_pixel_mask = hot_pixel_mask
        self.normalize = bool(normalize)
        self.quantile = float(quantile)
        self.spatial_batch_size = max(int(spatial_batch_size), 1)
        self.max_filter_size = max_filter_size

        self.alphas = sorted(alphas, reverse=True)
        self.num_scales = len(self.alphas)

        # Prior logits (-log alpha): slower scales get higher prior reward.
        # Centered on boxcar in routing so q=boxcar is the score origin.
        prior_tensor = -torch.log(torch.tensor(self.alphas, dtype=torch.float32).clamp(min=1e-5))
        prior_centered = prior_tensor - prior_tensor[-1]
        self.register_buffer("prior_logits", prior_centered.view(1, self.num_scales, 1))

        self._rebuild_ema_kernel()

        self.t_absolute = 0
        self._h, self._w, self._t = None, None, None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
            f"prior_strength={self.prior_strength}, tau={self.gating_tau})"
        )

    @staticmethod
    def _mean_rate_kernel(taps: Tensor) -> Tensor:
        """L1-normalize taps to unit sum (same per-bin rate scale as boxcar mean)."""
        return taps / taps.sum().clamp(min=1e-12)

    def _rebuild_ema_kernel(self, device: torch.device | str | None = None) -> None:
        """Rebuild heterogeneous FIR kernels: Gamma (fast), EMA (mid), Boxcar (slow)."""
        kernel = torch.zeros(self.num_scales, 1, self.kernel_size, dtype=torch.float32)

        for m, alpha in enumerate(self.alphas):
            # lags=0 on the newest bin, lags=K-1 on the oldest (causal conv indexing)
            lags = torch.arange(self.kernel_size - 1, -1, -1, dtype=torch.float32)

            if m == 0:
                # Causal exponential; newest bin must be non-zero for fair rate comparison
                tau_gamma = 2.0
                taps = torch.exp(-lags / tau_gamma)
            elif m == self.num_scales - 1:
                taps = torch.ones_like(lags)
            else:
                # Finite-window causal EMA: sum = 1 - (1-alpha)^K before L1 norm
                taps = alpha * torch.pow(1 - alpha, lags)

            kernel[m, 0, :] = self._mean_rate_kernel(taps)

        if device is None and hasattr(self, "ema_kernel"):
            device = self.ema_kernel.device
        kernel = kernel.to(device or "cpu")
        
        if hasattr(self, "ema_kernel"):
            del self.ema_kernel
        self.register_buffer("ema_kernel", kernel)

    def update_hyperparams(self, **kwargs) -> None:
        """Dynamically update attributes; rebuild FIR kernels when ``kernel_size`` changes."""
        # Legacy aliases from pre-KL min-pool API
        if kwargs.get("min_filter_size") is not None and kwargs.get("max_filter_size") is None:
            kwargs["max_filter_size"] = kwargs.pop("min_filter_size")
        kwargs.pop("peak_min_filter_size", None)
        kwargs.pop("peak_max_filter_size", None)

        old_kernel_size = int(self.kernel_size)
        for name, value in kwargs.items():
            if hasattr(self, name) and value is not None:
                setattr(self, name, value)
        if int(self.kernel_size) != old_kernel_size:
            self.kernel_size = max(int(self.kernel_size), 1)
            self._rebuild_ema_kernel()
        if kwargs.get("max_filter_size") is not None:
            val = max(int(self.max_filter_size), 1)
            if val % 2 == 0:
                raise ValueError("max_filter_size must be odd (or 1 to disable max-pool)")
            self.max_filter_size = val

    def set_cube(self, photon_cube: Tensor) -> None:
        """Record input cube shape."""
        self._h, self._w, self._t = map(int, photon_cube.shape)

    def clamp_recons(self, recons: Tensor) -> Tensor:
        """Clamp and optionally normalize reconstruction."""
        if recons.numel() == 0:
            return recons.float()
        recons = recons.float()
        max_value = 1.0
        if self.normalize:
            max_value = torch_quantile(recons, self.quantile).clamp(min=1e-6)
        return (recons / max_value).clamp(0, 1)

    @staticmethod
    def max_pool2d(x: Tensor, kernel_size: int) -> Tensor:
        """2D max-pool on ``[H, W]`` with replicate padding (dilate motion scores)."""
        if kernel_size <= 1:
            return x
        x_batched = x.unsqueeze(0).unsqueeze(0)
        padding = (kernel_size - 1) // 2
        x_batched = F.pad(x_batched, (padding, padding, padding, padding), mode="replicate")
        pooled = F.max_pool2d(x_batched, kernel_size=kernel_size, stride=1, padding=0)
        return pooled.squeeze(0).squeeze(0)

    @torch.no_grad()
    def _gate_conv_block(
        self,
        photon_block: Tensor,
        spatial_batch_size: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Threshold-Free Bayesian Convolution Block.
        :return: fused, motion_prob (pooled), kl_gamma, weight_gamma_raw, y_scales, scores [H, W, M]
        """
        h, w, t_seg = map(int, photon_block.shape)
        num_pixels = h * w
        batch_size = (
            self.spatial_batch_size
            if spatial_batch_size is None
            else max(int(spatial_batch_size), 1)
        )
        scores_all = torch.empty(
            (num_pixels, self.num_scales), device=photon_block.device, dtype=torch.float32
        )
        y_all_flat = torch.empty(
            (num_pixels, self.num_scales), device=photon_block.device, dtype=torch.float32
        )
        kl_gamma_raw = torch.empty((num_pixels,), device=photon_block.device, dtype=torch.float32)

        x_flat = photon_block.reshape(num_pixels, 1, self.kernel_size).float()
        eps = 1e-5

        for i in range(0, num_pixels, batch_size):
            end_i = min(i + batch_size, num_pixels)
            x_batch = x_flat[i:end_i]

            # y_all: [batch, num_scales, 1]
            y_all = F.conv1d(x_batch, self.ema_kernel)

            # q = Boxcar (stable reference); p_m = Gamma / EMA / Boxcar hypotheses
            q = y_all[:, -1:, :].clamp(eps, 1.0 - eps)
            p = y_all.clamp(eps, 1.0 - eps)

            # Bernoulli KL: D_KL(q || p_m); larger when p_m deviates from slow reference
            kl_div = q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))

            # score_m = KL_m/τ + prior (relative to boxcar); static → all KL≈0, slow scales win
            scores = (kl_div / self.gating_tau).squeeze(-1) + (
                self.prior_strength * self.prior_logits.squeeze(-1)
            )

            scores_all[i:end_i] = scores
            y_all_flat[i:end_i] = y_all.squeeze(-1)
            kl_gamma_raw[i:end_i] = kl_div[:, 0, :].squeeze(-1)

        weights = F.softmax(scores_all, dim=1)
        fused_flat = (weights * y_all_flat).sum(dim=1)
        weight_gamma_raw = weights[:, 0].view(h, w)
        motion_map = weight_gamma_raw
        # Pool motion only: inflating gamma *scores* before fusion pulls hand-edge
        # pixels toward low local y_gamma and causes a dark silhouette ring.
        if self.max_filter_size > 1:
            motion_map = self.max_pool2d(motion_map, self.max_filter_size)

        return (
            fused_flat.view(h, w),
            motion_map,
            kl_gamma_raw.view(h, w),
            weight_gamma_raw,
            y_all_flat.view(h, w, self.num_scales),
            scores_all.view(h, w, self.num_scales),
        )

    @torch.no_grad()
    def _integrate_blocks_with_motion(
        self, photon_cube: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Integrate ``[H, W, T]`` and return fused output plus per-pixel motion debug maps.
        """
        h, w, t = map(int, photon_cube.shape)
        ks = self.kernel_size
        n_blocks = t // ks
        empty_debug = {
            "motion_blocks": photon_cube.new_zeros(h, w, 0),
            "doe_blocks": photon_cube.new_zeros(h, w, 0),
            "weight_gamma_blocks": photon_cube.new_zeros(h, w, 0),
            "fused_blocks": photon_cube.new_zeros(h, w, 0),
            "fused_mean": photon_cube.new_zeros(h, w),
            "fused_last": photon_cube.new_zeros(h, w),
            "y_scales_last": photon_cube.new_zeros(h, w, self.num_scales),
            "scores_last": photon_cube.new_zeros(h, w, self.num_scales),
            "motion_peak_raw": photon_cube.new_zeros(h, w),
            "motion_peak": photon_cube.new_zeros(h, w),
            "doe_peak_raw": photon_cube.new_zeros(h, w),
            "doe_peak": photon_cube.new_zeros(h, w),
            "motion_blend": photon_cube.new_zeros(h, w),
            "recons_prenorm": photon_cube.new_zeros(h, w, 0),
        }
        if n_blocks == 0:
            out_t = 1 if t > 0 else 0
            return photon_cube.new_zeros(h, w, out_t, dtype=torch.float32), empty_debug

        fused_stack, motion_stack, doe_stack, weight_stack, y_scales_last, scores_last = [], [], [], [], None, None
        for b in range(n_blocks):
            seg = photon_cube[:, :, b * ks : (b + 1) * ks]
            fused_b, motion_b, doe_b, weight_b, y_scales_b, scores_b = self._gate_conv_block(seg)
            fused_stack.append(fused_b)
            motion_stack.append(motion_b)
            doe_stack.append(doe_b)
            weight_stack.append(weight_b)
            y_scales_last = y_scales_b
            scores_last = scores_b

        fused_hwb = torch.stack(fused_stack, dim=-1)
        motion_hwb = torch.stack(motion_stack, dim=-1)
        doe_hwb = torch.stack(doe_stack, dim=-1)
        weight_hwb = torch.stack(weight_stack, dim=-1)
        fused_mean_hw = fused_hwb.mean(dim=-1)
        fused_last_hw = fused_hwb[..., -1]

        doe_peak_raw = doe_hwb.max(dim=-1).values
        doe_peak = doe_peak_raw

        if n_blocks == 1:
            motion_peak_raw = motion_hwb[..., 0]
            motion_peak = motion_peak_raw

            # Direct soft-blend, no threshold
            motion_blend = motion_peak
            debug = {
                "motion_blocks": motion_hwb,
                "doe_blocks": doe_hwb,
                "weight_gamma_blocks": weight_hwb,
                "fused_blocks": fused_hwb,
                "fused_mean": fused_mean_hw,
                "fused_last": fused_last_hw,
                "y_scales_last": y_scales_last,
                "scores_last": scores_last,
                "motion_peak_raw": motion_peak_raw,
                "motion_peak": motion_peak,
                "doe_peak_raw": doe_peak_raw,
                "doe_peak": doe_peak,
                "motion_blend": motion_blend,
                "recons_prenorm": fused_hwb[..., -1:],
            }
            return fused_hwb[..., -1:], debug

        fused_mean = fused_mean_hw.unsqueeze(-1)
        fused_last = fused_last_hw.unsqueeze(-1)

        motion_peak_raw = motion_hwb.max(dim=-1).values
        motion_peak = motion_peak_raw
        motion_blend = motion_peak.unsqueeze(-1)
        out = (1.0 - motion_blend) * fused_mean + motion_blend * fused_last

        debug = {
            "motion_blocks": motion_hwb,
            "doe_blocks": doe_hwb,
            "weight_gamma_blocks": weight_hwb,
            "fused_blocks": fused_hwb,
            "fused_mean": fused_mean_hw,
            "fused_last": fused_last_hw,
            "y_scales_last": y_scales_last,
            "scores_last": scores_last,
            "motion_peak_raw": motion_peak_raw,
            "motion_peak": motion_peak,
            "doe_peak_raw": doe_peak_raw,
            "doe_peak": doe_peak,
            "motion_blend": motion_blend.squeeze(-1),
            "recons_prenorm": out,
        }
        return out, debug

    @torch.no_grad()
    def _integrate_blocks(self, photon_cube: Tensor) -> Tensor:
        fused, _ = self._integrate_blocks_with_motion(photon_cube)
        return fused

    def _subsample_reconstruction(self, fused_hwt: Tensor) -> Tensor:
        h, w, t = map(int, fused_hwt.shape)
        if t <= 0:
            return fused_hwt.new_zeros(h, w, 0, dtype=torch.float32)
        if t < self.subsampling:
            out = fused_hwt.new_zeros(h, w, 1, dtype=torch.float32)
            out[..., 0] = fused_hwt[..., -1].float()
            return out
        return fused_hwt[..., self.subsampling - 1 :: self.subsampling]

    @torch.no_grad()
    def process_photon_cube(
        self,
        photon_cube: Tensor,
        subsampling: int | None = None,
        hot_pixel_mask: np.ndarray | None = None,
        quantile: float | None = None,
        normalize: bool | None = None,
        clear_states: bool = True,
        chunk_size: int | None = None,
        kernel_size: int | None = None,
        prior_strength: float | None = None,
        gating_tau: float | None = None,
        max_filter_size: int | None = None,
        **kwargs,
    ) -> Tensor:
        if clear_states:
            self.t_absolute = 0

        self.update_hyperparams(
            subsampling=subsampling, hot_pixel_mask=hot_pixel_mask,
            normalize=normalize, quantile=quantile,
            chunk_size=chunk_size, kernel_size=kernel_size,
            prior_strength=prior_strength, gating_tau=gating_tau,
            max_filter_size=max_filter_size,
            **kwargs,
        )

        self.set_cube(photon_cube)
        fused = self._integrate_blocks(photon_cube)
        recons = self._subsample_reconstruction(fused)

        if self.hot_pixel_mask is not None:
            recons = nearest_neighbor_inpaint(recons, self.hot_pixel_mask)

        recons = self.clamp_recons(recons)
        self.t_absolute += self._t
        return recons

    @torch.no_grad()
    def process_photon_cube_with_motion(
        self,
        photon_cube: Tensor,
        subsampling: int | None = None,
        hot_pixel_mask: np.ndarray | None = None,
        quantile: float | None = None,
        normalize: bool | None = None,
        clear_states: bool = True,
        chunk_size: int | None = None,
        kernel_size: int | None = None,
        prior_strength: float | None = None,
        gating_tau: float | None = None,
        max_filter_size: int | None = None,
        **kwargs,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if clear_states:
            self.t_absolute = 0

        self.update_hyperparams(
            subsampling=subsampling, hot_pixel_mask=hot_pixel_mask,
            normalize=normalize, quantile=quantile,
            chunk_size=chunk_size, kernel_size=kernel_size,
            prior_strength=prior_strength, gating_tau=gating_tau,
            max_filter_size=max_filter_size,
            **kwargs,
        )

        self.set_cube(photon_cube)
        fused, motion_debug = self._integrate_blocks_with_motion(photon_cube)
        recons = self._subsample_reconstruction(fused)
        motion_debug["recons_prenorm"] = self._subsample_reconstruction(motion_debug["recons_prenorm"])

        if self.hot_pixel_mask is not None:
            recons = nearest_neighbor_inpaint(recons, self.hot_pixel_mask)

        recons = self.clamp_recons(recons)
        self.t_absolute += self._t
        return recons, motion_debug

    def forward(self, photon_cube: Tensor) -> Tensor:
        return self._integrate_blocks(photon_cube)