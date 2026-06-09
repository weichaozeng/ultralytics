"""Hybrid SPAD integrators with PerPixelBayesian-compatible streaming API."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.quanta_neural_networks.ops.array_ops import torch_quantile
from ultralytics.quanta_neural_networks.ops.image import nearest_neighbor_inpaint


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