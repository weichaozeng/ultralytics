"""STEA SPAD integrators with PerPixelBayesian-compatible streaming API."""

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

    The tensor path follows the 5-stage STEA pipeline: causal 1D temporal
    bases, pointwise Bernoulli KL, causal 3D spatio-temporal smoothing,
    time-reversed cummax masking, and inverse-length Bayesian fusion.
    """

    def __init__(
        self,
        fast_window: int = 16,
        slow_window: int = 128,
        temporal_window: int = 5,
        fast_tau: float | None = None,
        motion_sharpness: float = 60.0,
        motion_threshold: float = 0.05,
        eps: float = 1e-5,
        stable_prior: float = 16.0,
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
        self.motion_sharpness = float(motion_sharpness)
        self.motion_threshold = float(motion_threshold)
        self.eps = float(eps)
        self.stable_prior = float(stable_prior)
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
            f"motion_sharpness={self.motion_sharpness}, "
            f"motion_threshold={self.motion_threshold}, stable_prior={self.stable_prior})"
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
        stea = torch.ones((1, 1, self.temporal_window, 3, 3), dtype=torch.float32)

        self.register_buffer("fast_kernel", self._normalize_kernel(fast).view(1, 1, -1).to(device))
        self.register_buffer("slow_kernel", self._normalize_kernel(slow).view(1, 1, -1).to(device))
        self.register_buffer("stea_kernel", self._normalize_kernel(stea).to(device))

    def update_hyperparams(self, **kwargs) -> None:
        """Update STEA attributes; rebuild convolution kernels when needed."""
        rebuild_keys = {"fast_window", "slow_window", "temporal_window", "fast_tau"}
        # Compatibility with earlier STEA CLI/API aliases.
        if kwargs.get("kernel_size") is not None and kwargs.get("slow_window") is None:
            kwargs["slow_window"] = kwargs.pop("kernel_size")
        kwargs.pop("prior_strength", None)
        kwargs.pop("gating_tau", None)
        kwargs.pop("max_filter_size", None)
        kwargs.pop("min_filter_size", None)

        needs_rebuild = False
        for name, value in kwargs.items():
            # Legacy aliases from earlier STEA drafts.
            if name == "sharpness":
                name = "motion_sharpness"
            elif name == "bias":
                name = "motion_threshold"
            if hasattr(self, name) and value is not None:
                if name in {
                    "fast_window", "slow_window", "temporal_window", "chunk_size", "subsampling",
                }:
                    value = max(int(value), 1)
                if name in {"fast_tau", "motion_sharpness", "motion_threshold", "eps", "stable_prior", "quantile"}:
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
        k_context = torch.cat([hist, k_raw_hwt], dim=-1)
        k_5d = k_context.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        k_5d = F.pad(k_5d, (1, 1, 1, 1, 0, 0))
        return F.conv3d(k_5d, self.stea_kernel).squeeze(0).squeeze(0).permute(1, 2, 0)

    def _smooth_kl_last(self, k_raw_hwt: Tensor) -> Tensor:
        """Return only the final causal 3D-smoothed evidence frame."""
        h, w, _ = map(int, k_raw_hwt.shape)
        hist = self._history_or_zeros(self.kl_history, h, w, self.temporal_window - 1, k_raw_hwt)
        context = torch.cat([hist, k_raw_hwt], dim=-1)[..., -self.temporal_window :]
        k_thw = context.permute(2, 0, 1).unsqueeze(1)
        k_5d = k_thw.unsqueeze(0).transpose(1, 2)
        k_5d = F.pad(k_5d, (1, 1, 1, 1, 0, 0))
        return F.conv3d(k_5d, self.stea_kernel).squeeze(0).squeeze(0).squeeze(0)

    @staticmethod
    def _append_temporal_history(history: Tensor, frame_hw1: Tensor, maxlen: int) -> Tensor:
        if maxlen <= 0:
            return history
        if int(history.shape[-1]) == 0:
            out = frame_hw1
        else:
            out = torch.cat([history, frame_hw1], dim=-1)
        if int(out.shape[-1]) > maxlen:
            out = out[..., -maxlen:]
        return out

    def _smooth_kl_step(self, k_raw_hw1: Tensor, kl_hist: Tensor) -> Tensor:
        """Causal 3D-smoothed KL evidence for one new raw-KL frame."""
        context = torch.cat([kl_hist, k_raw_hw1], dim=-1)[..., -self.temporal_window :]
        k_thw = context.permute(2, 0, 1).unsqueeze(1)
        k_5d = k_thw.unsqueeze(0).transpose(1, 2)
        k_5d = F.pad(k_5d, (1, 1, 1, 1, 0, 0))
        return F.conv3d(k_5d, self.stea_kernel).squeeze(0).squeeze(0).squeeze(0)

    @staticmethod
    def _bernoulli_kl(y_fast: Tensor, y_slow: Tensor) -> Tensor:
        return y_fast * torch.log(y_fast / y_slow) + (1.0 - y_fast) * torch.log((1.0 - y_fast) / (1.0 - y_slow))

    @torch.no_grad()
    def _integrate_last(self, photon_cube: Tensor) -> Tensor:
        """Memory-efficient streaming integration that emits only the final fused frame."""
        h, w, t = map(int, photon_cube.shape)
        if t == 0:
            return photon_cube.new_zeros(h, w, 0, dtype=torch.float32)

        x = photon_cube.float()
        max_photon_hist = max(self.fast_window, self.slow_window) - 1
        max_kl_hist = self.temporal_window - 1

        fast_hist = self._history_or_zeros(self.photon_history, h, w, self.fast_window - 1, x)
        slow_hist = self._history_or_zeros(self.photon_history, h, w, self.slow_window - 1, x)
        kl_hist = self._history_or_zeros(self.kl_history, h, w, max_kl_hist, x)

        p_motion = x.new_zeros(h, w, t)
        y_fast_last = x.new_zeros(h, w)

        for ti in range(t):
            x_t = x[..., ti : ti + 1]
            x_flat = x_t.reshape(h * w, 1, 1)

            y_fast_t = self._causal_conv1d(x_flat, self.fast_kernel, fast_hist).reshape(h, w).clamp(
                self.eps, 1.0 - self.eps
            )
            y_slow_t = self._causal_conv1d(x_flat, self.slow_kernel, slow_hist).reshape(h, w).clamp(
                self.eps, 1.0 - self.eps
            )
            y_fast_last = y_fast_t

            k_raw_t = self._bernoulli_kl(y_fast_t, y_slow_t).unsqueeze(-1)
            k_smoothed_t = self._smooth_kl_step(k_raw_t, kl_hist)
            p_motion[..., ti] = torch.sigmoid(self.motion_sharpness * (k_smoothed_t - self.motion_threshold))

            fast_hist = self._append_temporal_history(fast_hist, x_t, self.fast_window - 1)
            slow_hist = self._append_temporal_history(slow_hist, x_t, self.slow_window - 1)
            kl_hist = self._append_temporal_history(kl_hist, k_raw_t, max_kl_hist)

        running_max = x.new_zeros(h, w)
        stable_num = x.new_zeros(h, w)
        stable_den = x.new_zeros(h, w)
        for ti in range(t - 1, -1, -1):
            running_max = torch.maximum(running_max, p_motion[..., ti])
            valid_weight = 1.0 - running_max
            stable_num += valid_weight * x[..., ti]
            stable_den += valid_weight

        mean_stable = stable_num / stable_den.clamp(min=self.eps)
        w_mean = stable_den / (stable_den + max(self.stable_prior, self.eps))
        fused_last = w_mean * mean_stable + (1.0 - w_mean) * y_fast_last

        if max_photon_hist > 0:
            self.photon_history = x[..., -max_photon_hist:].detach()
        else:
            self.photon_history = None
        self.kl_history = kl_hist.detach() if max_kl_hist > 0 else None
        return fused_last.unsqueeze(-1)

    @torch.no_grad()
    def _integrate_last_with_debug(self, photon_cube: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Memory-efficient path for chunked scripts that only emit the final frame."""
        h, w, t = map(int, photon_cube.shape)
        if t == 0:
            empty, debug = self._integrate_full_with_debug(photon_cube)
            return empty, debug

        y_fast, y_slow = self._temporal_basis(photon_cube)
        y_fast_last = y_fast[..., -1].reshape(h, w)
        y_slow_last = y_slow[..., -1].reshape(h, w)

        k_raw_flat = y_fast * torch.log(y_fast / y_slow) + (1.0 - y_fast) * torch.log(
            (1.0 - y_fast) / (1.0 - y_slow)
        )
        k_raw_hwt = k_raw_flat.reshape(h, w, t)
        k_smoothed_hwt = self._smooth_kl(k_raw_hwt)

        p_motion = torch.sigmoid(self.motion_sharpness * (k_smoothed_hwt - self.motion_threshold))
        p_motion_raw = p_motion
        future_motion = torch.flip(torch.flip(p_motion, dims=(-1,)).cummax(dim=-1).values, dims=(-1,))
        valid_weight = 1.0 - future_motion
        stable_support = valid_weight.sum(dim=-1)
        mean_stable = (valid_weight * photon_cube.float()).sum(dim=-1) / stable_support.clamp(min=self.eps)
        w_mean = stable_support / (stable_support + max(self.stable_prior, self.eps))
        fused_last = w_mean * mean_stable + (1.0 - w_mean) * y_fast_last
        fused = fused_last.unsqueeze(-1)

        max_photon_hist = max(self.fast_window, self.slow_window) - 1
        max_kl_hist = self.temporal_window - 1
        self.photon_history = photon_cube.float()[..., -max_photon_hist:].detach() if max_photon_hist > 0 else None
        self.kl_history = k_raw_hwt[..., -max_kl_hist:].detach() if max_kl_hist > 0 else None

        debug = {
            "y_fast": y_fast_last.unsqueeze(-1),
            "y_slow": y_slow_last.unsqueeze(-1),
            "k_raw": k_raw_hwt[..., -1:],
            "k_smoothed": k_smoothed_hwt,
            "route_weight": future_motion,
            "route_weight_raw": p_motion_raw,
            "p_motion": p_motion,
            "p_motion_raw": p_motion_raw,
            "future_motion": future_motion,
            "valid_weight": valid_weight,
            "fused": fused,
            "fused_blocks": fused,
            "motion_blocks": future_motion,
            "doe_blocks": k_smoothed_hwt,
            "weight_gamma_blocks": future_motion,
            "fused_mean": fused_last,
            "fused_last": fused_last,
            "motion_peak": future_motion.max(dim=-1).values,
            "motion_blend": future_motion[..., -1],
            "y_scales_last": torch.stack([y_fast_last, y_slow_last], dim=-1),
            "scores_last": torch.stack([k_smoothed_hwt[..., -1], future_motion[..., -1]], dim=-1),
            "k_smoothed_last": k_smoothed_hwt[..., -1],
            "route_weight_last": future_motion[..., -1],
            "route_weight_raw_last": p_motion_raw[..., -1],
            "p_motion_last": p_motion[..., -1],
            "future_motion_last": future_motion[..., -1],
            "valid_weight_last": valid_weight[..., -1],
            "mean_stable": mean_stable,
            "stable_support": stable_support,
            "w_mean": w_mean,
            "recons_prenorm": fused,
        }
        return fused, debug

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
                "route_weight_raw": empty,
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
                "route_weight_raw_last": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "recons_prenorm": empty,
            }
            return empty, debug

        return self._integrate_last_with_debug(photon_cube)

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
        motion_sharpness: float | None = None,
        motion_threshold: float | None = None,
        eps: float | None = None,
        stable_prior: float | None = None,
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
            motion_sharpness=motion_sharpness,
            motion_threshold=motion_threshold,
            eps=eps,
            stable_prior=stable_prior,
            **kwargs,
        )

        self.set_cube(photon_cube)
        fused = self._integrate_last(photon_cube)
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
        motion_sharpness: float | None = None,
        motion_threshold: float | None = None,
        eps: float | None = None,
        stable_prior: float | None = None,
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
            motion_sharpness=motion_sharpness,
            motion_threshold=motion_threshold,
            eps=eps,
            stable_prior=stable_prior,
            **kwargs,
        )

        self.set_cube(photon_cube)
        if self._t <= self.subsampling:
            fused, motion_debug = self._integrate_last_with_debug(photon_cube)
        else:
            fused, motion_debug = self._integrate_full_with_debug(photon_cube)
        recons = self._subsample_reconstruction(fused)
        motion_debug["recons_prenorm"] = self._subsample_reconstruction(motion_debug["recons_prenorm"])
        if self.hot_pixel_mask is not None:
            recons = nearest_neighbor_inpaint(recons, self.hot_pixel_mask)
        recons = self.clamp_recons(recons)
        self.t_absolute += self._t
        return recons, motion_debug

    def forward(self, photon_cube: Tensor) -> Tensor:
        return self._integrate_last(photon_cube)


class SpatioTemporalEvidenceFrame(SpatioTemporalEvidenceAccumulation):
    """STEA variant that collapses each raw chunk into one frame plus a lightweight `w_mean` map."""

    @torch.no_grad()
    def _integrate_last_with_w_mean(self, photon_cube: Tensor) -> tuple[Tensor, Tensor]:
        h, w, t = map(int, photon_cube.shape)
        if t == 0:
            empty = photon_cube.new_zeros(h, w, 0, dtype=torch.float32)
            return empty, photon_cube.new_zeros(h, w, dtype=torch.float32)

        x = photon_cube.float()
        max_photon_hist = max(self.fast_window, self.slow_window) - 1
        max_kl_hist = self.temporal_window - 1

        fast_hist = self._history_or_zeros(self.photon_history, h, w, self.fast_window - 1, x)
        slow_hist = self._history_or_zeros(self.photon_history, h, w, self.slow_window - 1, x)
        kl_hist = self._history_or_zeros(self.kl_history, h, w, max_kl_hist, x)

        p_motion = x.new_zeros(h, w, t)
        y_fast_last = x.new_zeros(h, w)

        for ti in range(t):
            x_t = x[..., ti : ti + 1]
            x_flat = x_t.reshape(h * w, 1, 1)

            y_fast_t = self._causal_conv1d(x_flat, self.fast_kernel, fast_hist).reshape(h, w).clamp(
                self.eps, 1.0 - self.eps
            )
            y_slow_t = self._causal_conv1d(x_flat, self.slow_kernel, slow_hist).reshape(h, w).clamp(
                self.eps, 1.0 - self.eps
            )
            y_fast_last = y_fast_t

            k_raw_t = self._bernoulli_kl(y_fast_t, y_slow_t).unsqueeze(-1)
            k_smoothed_t = self._smooth_kl_step(k_raw_t, kl_hist)
            p_motion[..., ti] = torch.sigmoid(self.motion_sharpness * (k_smoothed_t - self.motion_threshold))

            fast_hist = self._append_temporal_history(fast_hist, x_t, self.fast_window - 1)
            slow_hist = self._append_temporal_history(slow_hist, x_t, self.slow_window - 1)
            kl_hist = self._append_temporal_history(kl_hist, k_raw_t, max_kl_hist)

        running_max = x.new_zeros(h, w)
        stable_num = x.new_zeros(h, w)
        stable_den = x.new_zeros(h, w)
        for ti in range(t - 1, -1, -1):
            running_max = torch.maximum(running_max, p_motion[..., ti])
            valid_weight = 1.0 - running_max
            stable_num += valid_weight * x[..., ti]
            stable_den += valid_weight

        mean_stable = stable_num / stable_den.clamp(min=self.eps)
        w_mean = stable_den / (stable_den + max(self.stable_prior, self.eps))
        fused_last = w_mean * mean_stable + (1.0 - w_mean) * y_fast_last

        if max_photon_hist > 0:
            self.photon_history = x[..., -max_photon_hist:].detach()
        else:
            self.photon_history = None
        self.kl_history = kl_hist.detach() if max_kl_hist > 0 else None
        return fused_last.unsqueeze(-1), w_mean

    @staticmethod
    def _w_mean_to_frame_space(w_mean_hw: Tensor) -> Tensor:
        if w_mean_hw.ndim != 2:
            raise ValueError(f"Expected w_mean (H,W), got shape={tuple(w_mean_hw.shape)}")
        return F.avg_pool2d(w_mean_hw.unsqueeze(0).unsqueeze(0).float(), kernel_size=2, stride=2).squeeze(0)

    @torch.no_grad()
    def process_photon_cube_to_frame(
        self,
        photon_cube: Tensor,
        hot_pixel_mask: np.ndarray | None = None,
        quantile: float | None = None,
        normalize: bool | None = None,
        clear_states: bool = True,
        chunk_size: int | None = None,
        fast_window: int | None = None,
        slow_window: int | None = None,
        temporal_window: int | None = None,
        fast_tau: float | None = None,
        motion_sharpness: float | None = None,
        motion_threshold: float | None = None,
        eps: float | None = None,
        stable_prior: float | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        if clear_states:
            self.t_absolute = 0
            self._clear_histories()

        self.update_hyperparams(
            hot_pixel_mask=hot_pixel_mask,
            normalize=normalize,
            quantile=quantile,
            chunk_size=chunk_size,
            fast_window=fast_window,
            slow_window=slow_window,
            temporal_window=temporal_window,
            fast_tau=fast_tau,
            motion_sharpness=motion_sharpness,
            motion_threshold=motion_threshold,
            eps=eps,
            stable_prior=stable_prior,
            **kwargs,
        )

        self.set_cube(photon_cube)
        fused, w_mean = self._integrate_last_with_w_mean(photon_cube)
        if self.hot_pixel_mask is not None:
            fused = nearest_neighbor_inpaint(fused, self.hot_pixel_mask)
        fused = self.clamp_recons(fused)
        self.t_absolute += self._t
        return fused, self._w_mean_to_frame_space(w_mean).to(device=fused.device, dtype=fused.dtype)
