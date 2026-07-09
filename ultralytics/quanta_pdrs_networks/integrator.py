"""
Poisson Dual-Rate Split (PDRS) for SPAD photon cubes.

Theory
------
Motion is **transient divergence** between two causal rate estimators, not deviation
from a frozen reference (the failure mode of chunk-entry Poisson deviance).

  λ_f(t)  fast causal conv  — reacts in O(fast_window) bins (PPB-like responsiveness)
  λ_s(t)  slow causal conv  — long memory baseline (STEA-like stability)

  score(t) = D_Pois(λ_f(t) || λ_s(t))   generalized Poisson KL, zero when rates agree

After a changepoint λ_f moves first → score spikes → reverse-cummax fusion uses λ_f.
Once λ_s catches up, score relaxes → stable photon integration resumes without drag.

Parallel path: dual conv1d over the full chunk (STEA-style). Streaming fallback for long T.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.quanta_neural_networks.ops.array_ops import torch_quantile
from ultralytics.quanta_neural_networks.ops.image import nearest_neighbor_inpaint


def poisson_rate_kl(lam_f: Tensor, lam_s: Tensor, eps: float) -> Tensor:
    """Generalized KL between Poisson means: λ_f log(λ_f/λ_s) + λ_s - λ_f."""
    lf = lam_f.float().clamp(min=eps)
    ls = lam_s.float().clamp(min=eps)
    return lf * torch.log(lf / ls) + ls - lf


class PoissonDualRateSplit(nn.Module):
    def __init__(
        self,
        fast_window: int = 32,
        slow_window: int = 128,
        temporal_window: int = 5,
        fast_tau: float | None = None,
        motion_sharpness: float = 60.0,
        motion_threshold: float = 0.07,
        motion_pool_size: int = 7,
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
        self.motion_pool_size = max(int(motion_pool_size), 1)
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
        self.register_buffer("score_history", None)
        self.register_buffer("sample_weight", None)
        self._rebuild_kernels()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(fast_window={self.fast_window}, "
            f"slow_window={self.slow_window}, temporal_window={self.temporal_window}, "
            f"motion_threshold={self.motion_threshold}, motion_pool_size={self.motion_pool_size})"
        )

    @staticmethod
    def _normalize_kernel(taps: Tensor) -> Tensor:
        return taps / taps.sum().clamp(min=1e-12)

    def _rebuild_kernels(self, device: torch.device | str | None = None) -> None:
        if device is None and hasattr(self, "fast_kernel"):
            device = self.fast_kernel.device
        device = device or "cpu"
        gamma_age = torch.arange(self.fast_window, 0, -1, dtype=torch.float32)
        fast = gamma_age * torch.exp(-gamma_age / max(self.fast_tau, 1e-6))
        slow = torch.ones(self.slow_window, dtype=torch.float32)
        stea = torch.ones((1, 1, self.temporal_window, 1, 1), dtype=torch.float32)
        self.register_buffer("fast_kernel", self._normalize_kernel(fast).view(1, 1, -1).to(device))
        self.register_buffer("slow_kernel", self._normalize_kernel(slow).view(1, 1, -1).to(device))
        self.register_buffer("stea_kernel", self._normalize_kernel(stea).to(device))

    def update_hyperparams(self, **kwargs) -> None:
        rebuild_keys = {"fast_window", "slow_window", "temporal_window", "fast_tau"}
        if kwargs.get("kernel_size") is not None and kwargs.get("slow_window") is None:
            kwargs["slow_window"] = kwargs.pop("kernel_size")
        for alias, target in (("sharpness", "motion_sharpness"), ("bias", "motion_threshold")):
            if kwargs.get(alias) is not None and kwargs.get(target) is None:
                kwargs[target] = kwargs.pop(alias)
        needs_rebuild = False
        for name, value in kwargs.items():
            if not hasattr(self, name) or value is None:
                continue
            if name in {"fast_window", "slow_window", "temporal_window", "chunk_size", "subsampling", "motion_pool_size"}:
                value = max(int(value), 1)
            elif name in {"fast_tau", "motion_sharpness", "motion_threshold", "eps", "stable_prior", "quantile"}:
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
        self.score_history = None
        self.sample_weight = None

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
        lam_f = self._causal_conv1d(x_flat, self.fast_kernel, fast_hist).reshape(h, w, t).clamp(min=self.eps)
        lam_s = self._causal_conv1d(x_flat, self.slow_kernel, slow_hist).reshape(h, w, t).clamp(min=self.eps)
        return lam_f, lam_s

    def _max_pool_score_hw(self, score_hw: Tensor) -> Tensor:
        """Spatial max-pool on D — equivalent to PPB min-pool on run-length proxy."""
        k = int(self.motion_pool_size)
        if k <= 1:
            return score_hw
        pad = (k - 1) // 2
        x = score_hw.unsqueeze(0).unsqueeze(0)
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad).squeeze(0).squeeze(0)

    def _max_pool_score_hwt(self, score_hwt: Tensor) -> Tensor:
        k = int(self.motion_pool_size)
        if k <= 1:
            return score_hwt
        pad = (k - 1) // 2
        x = score_hwt.permute(2, 0, 1).unsqueeze(1)
        out = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
        return out.squeeze(1).permute(1, 2, 0)

    def _motion_from_score(self, score: Tensor) -> Tensor:
        if score.ndim == 2:
            score = self._max_pool_score_hw(score)
        elif score.ndim == 3:
            score = self._max_pool_score_hwt(score)
        else:
            raise ValueError(f"Expected score (H,W) or (H,W,T), got shape={tuple(score.shape)}")
        return torch.sigmoid(self.motion_sharpness * (score - self.motion_threshold))

    def _smooth_score(self, score_hwt: Tensor) -> Tensor:
        hist = self._history_or_zeros(
            self.score_history, int(score_hwt.shape[0]), int(score_hwt.shape[1]), self.temporal_window - 1, score_hwt
        )
        context = torch.cat([hist, score_hwt], dim=-1)
        s_5d = context.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        return F.conv3d(s_5d, self.stea_kernel).squeeze(0).squeeze(0).permute(1, 2, 0)

    @staticmethod
    def _append_temporal_history(history: Tensor, frame_hw1: Tensor, maxlen: int) -> Tensor:
        if maxlen <= 0:
            return history
        out = frame_hw1 if int(history.shape[-1]) == 0 else torch.cat([history, frame_hw1], dim=-1)
        return out[..., -maxlen:] if int(out.shape[-1]) > maxlen else out

    def _smooth_score_step(self, score_hw1: Tensor, score_hist: Tensor) -> Tensor:
        context = torch.cat([score_hist, score_hw1], dim=-1)[..., -self.temporal_window :]
        s_thw = context.permute(2, 0, 1).unsqueeze(1)
        s_5d = s_thw.unsqueeze(0).transpose(1, 2)
        return F.conv3d(s_5d, self.stea_kernel).squeeze(0).squeeze(0).squeeze(0)

    def _fuse_reverse_cummax(
        self, x: Tensor, p_motion: Tensor, lam_f_last: Tensor
    ) -> tuple[Tensor, Tensor]:
        h, w, t = map(int, x.shape)
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
        fused_last = w_mean * mean_stable + (1.0 - w_mean) * lam_f_last
        return fused_last, w_mean

    def _save_histories(self, x: Tensor, score_hwt: Tensor | None) -> None:
        max_photon_hist = max(self.fast_window, self.slow_window) - 1
        max_score_hist = self.temporal_window - 1
        self.photon_history = x[..., -max_photon_hist:].detach() if max_photon_hist > 0 else None
        if score_hwt is not None and max_score_hist > 0:
            self.score_history = score_hwt[..., -max_score_hist:].detach()
        else:
            self.score_history = None

    @torch.no_grad()
    def _integrate_chunk_parallel(self, photon_cube: Tensor) -> Tensor:
        """Full-chunk dual conv1d path (default for T <= chunk_size)."""
        h, w, t = map(int, photon_cube.shape)
        if t == 0:
            return photon_cube.new_zeros(h, w, 0, dtype=torch.float32)

        x = photon_cube.float()
        lam_f, lam_s = self._temporal_basis(x)
        lam_f_last = lam_f[..., -1]
        score_raw = poisson_rate_kl(lam_f, lam_s, self.eps)
        del lam_s
        score_smooth = self._smooth_score(score_raw)
        p_motion = self._motion_from_score(score_smooth)
        del score_smooth
        fused_last, w_mean = self._fuse_reverse_cummax(x, p_motion, lam_f_last)
        self.sample_weight = w_mean.detach()
        self._save_histories(x, score_raw)
        del p_motion, score_raw, lam_f
        return fused_last.unsqueeze(-1)

    @torch.no_grad()
    def _integrate_streaming(self, photon_cube: Tensor) -> Tensor:
        """Bin-wise streaming path for long cubes (same logic, lower peak memory)."""
        h, w, t = map(int, photon_cube.shape)
        if t == 0:
            return photon_cube.new_zeros(h, w, 0, dtype=torch.float32)

        x = photon_cube.float()
        max_photon_hist = max(self.fast_window, self.slow_window) - 1
        max_score_hist = self.temporal_window - 1
        fast_hist = self._history_or_zeros(self.photon_history, h, w, self.fast_window - 1, x)
        slow_hist = self._history_or_zeros(self.photon_history, h, w, self.slow_window - 1, x)
        score_hist = self._history_or_zeros(self.score_history, h, w, max_score_hist, x)

        p_motion = x.new_zeros(h, w, t)
        lam_f_last = x.new_zeros(h, w)
        score_tail = x.new_zeros(h, w, 0)

        for ti in range(t):
            x_t = x[..., ti : ti + 1]
            x_flat = x_t.reshape(h * w, 1, 1)
            lam_f_t = self._causal_conv1d(x_flat, self.fast_kernel, fast_hist).reshape(h, w).clamp(min=self.eps)
            lam_s_t = self._causal_conv1d(x_flat, self.slow_kernel, slow_hist).reshape(h, w).clamp(min=self.eps)
            lam_f_last = lam_f_t
            score_t = poisson_rate_kl(lam_f_t, lam_s_t, self.eps).unsqueeze(-1)
            score_sm = self._smooth_score_step(score_t, score_hist)
            p_motion[..., ti] = self._motion_from_score(score_sm)
            fast_hist = self._append_temporal_history(fast_hist, x_t, self.fast_window - 1)
            slow_hist = self._append_temporal_history(slow_hist, x_t, self.slow_window - 1)
            score_hist = self._append_temporal_history(score_hist, score_t, max_score_hist)
            score_tail = score_t if int(score_tail.shape[-1]) == 0 else torch.cat([score_tail, score_t], dim=-1)

        fused_last, w_mean = self._fuse_reverse_cummax(x, p_motion, lam_f_last)
        self.sample_weight = w_mean.detach()
        self._save_histories(x, score_tail)
        return fused_last.unsqueeze(-1)

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
        motion_pool_size: int | None = None,
        eps: float | None = None,
        stable_prior: float | None = None,
        parallel: bool | None = None,
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
            motion_pool_size=motion_pool_size,
            eps=eps,
            stable_prior=stable_prior,
            **kwargs,
        )

        self.set_cube(photon_cube)
        use_parallel = False if parallel is None else bool(parallel)
        fused = self._integrate_chunk_parallel(photon_cube) if use_parallel else self._integrate_streaming(photon_cube)
        recons = self._subsample_reconstruction(fused)
        if self.hot_pixel_mask is not None:
            recons = nearest_neighbor_inpaint(recons, self.hot_pixel_mask)
        recons = self.clamp_recons(recons)
        self.t_absolute += self._t
        return recons

    def forward(self, photon_cube: Tensor) -> Tensor:
        return self.process_photon_cube(photon_cube, clear_states=False)


class PoissonDualRateSplitFrame(PoissonDualRateSplit):
    """Frame-mode wrapper emitting one reconstruction plus a confidence map."""

    @staticmethod
    def _confidence_to_frame_space(confidence_hw: Tensor) -> Tensor:
        if confidence_hw.ndim != 2:
            raise ValueError(f"Expected confidence map (H,W), got shape={tuple(confidence_hw.shape)}")
        return F.avg_pool2d(confidence_hw.unsqueeze(0).unsqueeze(0).float(), kernel_size=2, stride=2).squeeze(0)

    @torch.no_grad()
    def process_photon_cube_to_frame(
        self, photon_cube: Tensor, clear_states: bool = True, **kwargs
    ) -> tuple[Tensor, Tensor]:
        recons = self.process_photon_cube(photon_cube, clear_states=clear_states, **kwargs)
        if recons.ndim != 3 or int(recons.shape[-1]) == 0:
            h, w = map(int, photon_cube.shape[:2])
            recons = photon_cube.new_zeros((h, w, 1), dtype=torch.float32)
        else:
            recons = recons[..., -1:].contiguous()
        sample_weight = getattr(self, "sample_weight", None)
        if sample_weight is None:
            confidence_hw = photon_cube.new_ones(photon_cube.shape[0], photon_cube.shape[1], dtype=torch.float32)
        else:
            confidence_hw = sample_weight.detach().float().clamp(0.0, 1.0)
        confidence = self._confidence_to_frame_space(confidence_hw).to(device=recons.device, dtype=recons.dtype)
        return recons, confidence
