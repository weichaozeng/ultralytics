"""PG-DR path 2: Poisson-Gamma deviance with per-bin Gamma soft-reset."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.quanta_neural_networks.ops.array_ops import torch_quantile
from ultralytics.quanta_neural_networks.ops.image import nearest_neighbor_inpaint
from ultralytics.quanta_pgdr_networks.common import (
    append_temporal_history,
    build_fast_kernel,
    build_stea_smooth_kernel,
    causal_window_sum_step,
    compute_motion_weights_parallel,
    history_or_zeros,
    motion_gate,
    poisson_deviance,
    smooth_deviance_step,
)


class PoissonGammaDevianceGamma(nn.Module):
    """
    PG-DR gamma path (path 2).

    Motion scoring matches the fusion path (fast-window Poisson deviance vs chunk-entry
    slow rate). Slow branch applies per-bin Gamma soft-reset and display EMA buffering.
    """

    def __init__(
        self,
        fast_window: int = 16,
        temporal_window: int = 5,
        fast_tau: float | None = None,
        motion_sharpness: float = 60.0,
        motion_threshold: float = 0.05,
        eps: float = 1e-5,
        stable_prior: float = 16.0,
        alpha_prior: float = 1.0,
        beta_min: float = 32.0,
        eta_min: float = 0.02,
        eta_max: float = 0.85,
        w_eps: float = 1e-4,
        chunk_size: int = 320,
        subsampling: int = 1,
        hot_pixel_mask: np.ndarray | None = None,
        normalize: bool = False,
        quantile: float = 1.0,
        cold_start_chunks: int = 1,
    ):
        super().__init__()
        self.fast_window = max(int(fast_window), 1)
        self.temporal_window = max(int(temporal_window), 1)
        self.fast_tau = float(fast_tau) if fast_tau is not None else max(self.fast_window / 4.0, 1.0)
        self.motion_sharpness = float(motion_sharpness)
        self.motion_threshold = float(motion_threshold)
        self.eps = float(eps)
        self.stable_prior = float(stable_prior)
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(stable_prior)
        self.beta_min = float(beta_min)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.w_eps = float(w_eps)
        self.chunk_size = max(int(chunk_size), 1)
        self.subsampling = max(int(subsampling), 1)
        self.hot_pixel_mask = hot_pixel_mask
        self.normalize = bool(normalize)
        self.quantile = float(quantile)
        self.cold_start_chunks = max(int(cold_start_chunks), 0)

        self.t_absolute = 0
        self._chunk_index = 0
        self._h, self._w, self._t = None, None, None

        self.register_buffer("alpha_s", None)
        self.register_buffer("beta_s", None)
        self.register_buffer("y", None)
        self.register_buffer("photon_history", None)
        self.register_buffer("d_history", None)
        self.register_buffer("sample_weight", None)
        self._rebuild_kernels()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(fast_window={self.fast_window}, "
            f"temporal_window={self.temporal_window}, motion_sharpness={self.motion_sharpness}, "
            f"motion_threshold={self.motion_threshold}, stable_prior={self.stable_prior})"
        )

    def _rebuild_kernels(self, device: torch.device | str | None = None) -> None:
        if device is None and hasattr(self, "fast_kernel"):
            device = self.fast_kernel.device
        device = device or "cpu"
        self.register_buffer("fast_kernel", build_fast_kernel(self.fast_window, self.fast_tau, device))
        self.register_buffer("stea_kernel", build_stea_smooth_kernel(self.temporal_window, device))

    def update_hyperparams(self, **kwargs) -> None:
        rebuild_keys = {"fast_window", "temporal_window", "fast_tau"}
        if kwargs.get("sharpness") is not None and kwargs.get("motion_sharpness") is None:
            kwargs["motion_sharpness"] = kwargs.pop("sharpness")
        if kwargs.get("bias") is not None and kwargs.get("motion_threshold") is None:
            kwargs["motion_threshold"] = kwargs.pop("bias")

        needs_rebuild = False
        for name, value in kwargs.items():
            if not hasattr(self, name) or value is None:
                continue
            if name in {"fast_window", "temporal_window", "chunk_size", "subsampling", "cold_start_chunks"}:
                value = max(int(value), 1)
            elif name in {
                "fast_tau",
                "motion_sharpness",
                "motion_threshold",
                "eps",
                "stable_prior",
                "alpha_prior",
                "beta_min",
                "eta_min",
                "eta_max",
                "w_eps",
                "quantile",
            }:
                value = float(value)
            elif name == "normalize":
                value = bool(value)
            setattr(self, name, value)
            if name == "stable_prior":
                self.beta_prior = float(value)
            needs_rebuild = needs_rebuild or name in rebuild_keys
        if needs_rebuild:
            self._rebuild_kernels(device=self.fast_kernel.device)
            self._clear_histories()

    def _clear_histories(self) -> None:
        self.alpha_s = None
        self.beta_s = None
        self.y = None
        self.photon_history = None
        self.d_history = None
        self.sample_weight = None

    def init_state(self, h: int, w: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.alpha_s = torch.full((h, w), self.alpha_prior, device=device, dtype=dtype)
        self.beta_s = torch.full((h, w), self.beta_prior, device=device, dtype=dtype)
        self.y = torch.zeros((h, w), device=device, dtype=dtype)
        self.photon_history = None
        self.d_history = None
        self.sample_weight = torch.zeros((h, w), device=device, dtype=dtype)

    def _ensure_state(self, photon_cube: Tensor) -> None:
        h, w, _ = map(int, photon_cube.shape)
        device = photon_cube.device
        dtype = torch.float32
        if self.alpha_s is None or tuple(self.alpha_s.shape) != (h, w):
            self.init_state(h, w, device=device, dtype=dtype)

    def _cold_start_active(self) -> bool:
        return self._chunk_index < self.cold_start_chunks or (
            self.beta_s is not None and bool(torch.any(self.beta_s < self.beta_min))
        )

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

    def _mu_reference(self) -> Tensor:
        return (self.fast_window * self.alpha_s / self.beta_s.clamp(min=self.eps)).clamp(min=self.eps)

    def _gamma_micro_step(self, x_t: Tensor, w_t: Tensor) -> None:
        a0 = self.alpha_s.new_tensor(self.alpha_prior)
        b0 = self.alpha_s.new_tensor(self.beta_prior)
        self.alpha_s = (1.0 - w_t) * self.alpha_s + w_t * a0 + x_t
        self.beta_s = (1.0 - w_t) * self.beta_s + w_t * b0 + 1.0
        eta_t = self.eta_min + (self.eta_max - self.eta_min) * (1.0 - w_t)
        self.y = (1.0 - eta_t) * self.y + eta_t * x_t
        self.sample_weight = (1.0 - w_t).detach()

    def _integrate_motion_gamma_streaming(self, x: Tensor) -> Tensor | None:
        h, w, t = map(int, x.shape)
        photon_hist = history_or_zeros(self.photon_history, h, w, self.fast_window - 1, x)
        d_hist = history_or_zeros(self.d_history, h, w, self.temporal_window - 1, x)

        mu_ref = self._mu_reference()
        cold = self._cold_start_active()

        for ti in range(t):
            x_t = x[..., ti]
            k_fast_t = causal_window_sum_step(x[..., ti : ti + 1], photon_hist, self.fast_window)
            d_t = poisson_deviance(k_fast_t, mu_ref, eps=self.eps)
            d_raw_t = d_t.unsqueeze(-1)
            d_smooth_t = smooth_deviance_step(d_raw_t, d_hist, self.stea_kernel)
            w_t = motion_gate(d_smooth_t, self.motion_sharpness, self.motion_threshold)
            if cold:
                w_t = w_t * 0.0
            self._gamma_micro_step(x_t, w_t)
            photon_hist = append_temporal_history(photon_hist, x[..., ti : ti + 1], self.fast_window - 1)
            d_hist = append_temporal_history(d_hist, d_raw_t, self.temporal_window - 1)

        return d_hist if self.temporal_window > 1 else None

    def _finalize_chunk_histories(self, x: Tensor, d_hist: Tensor | None) -> None:
        if self.fast_window > 1:
            self.photon_history = x[..., -self.fast_window + 1 :].detach()
        else:
            self.photon_history = None
        if self.temporal_window > 1 and d_hist is not None:
            self.d_history = d_hist.detach()
        else:
            self.d_history = None

    @torch.no_grad()
    def _integrate_last(self, photon_cube: Tensor) -> Tensor:
        h, w, t = map(int, photon_cube.shape)
        if t == 0:
            return photon_cube.new_zeros(h, w, 0, dtype=torch.float32)

        self._ensure_state(photon_cube)
        x = photon_cube.float()
        d_hist = self._integrate_motion_gamma_streaming(x)
        self._finalize_chunk_histories(x, d_hist)
        self._chunk_index += 1
        return self.y.unsqueeze(-1)

    @torch.no_grad()
    def _integrate_parallel(self, photon_cube: Tensor) -> Tensor:
        h, w, t = map(int, photon_cube.shape)
        if t == 0:
            return photon_cube.new_zeros(h, w, 0, dtype=torch.float32)

        self._ensure_state(photon_cube)
        x = photon_cube.float()
        p_motion, _, d_hist_tail = compute_motion_weights_parallel(
            x,
            mu_ref=self._mu_reference(),
            fast_kernel=self.fast_kernel,
            stea_kernel=self.stea_kernel,
            photon_history=self.photon_history,
            d_history=self.d_history,
            fast_window=self.fast_window,
            motion_sharpness=self.motion_sharpness,
            motion_threshold=self.motion_threshold,
            eps=self.eps,
            cold_start=self._cold_start_active(),
        )
        self._apply_gamma_parallel(x, p_motion)
        del p_motion
        self._finalize_chunk_histories(x, d_hist_tail)
        self._chunk_index += 1
        return self.y.unsqueeze(-1)

    def _apply_gamma_parallel(self, x: Tensor, p_motion: Tensor) -> None:
        if float(p_motion.max()) < self.w_eps:
            self.alpha_s = self.alpha_s + x.sum(dim=-1)
            self.beta_s = self.beta_s + float(x.shape[-1])
            self.y = x.mean(dim=-1)
            self.sample_weight = torch.ones_like(self.sample_weight)
            return
        h, w, t = map(int, x.shape)
        for ti in range(t):
            self._gamma_micro_step(x[..., ti], p_motion[..., ti])

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
        temporal_window: int | None = None,
        fast_tau: float | None = None,
        motion_sharpness: float | None = None,
        motion_threshold: float | None = None,
        eps: float | None = None,
        stable_prior: float | None = None,
        alpha_prior: float | None = None,
        beta_min: float | None = None,
        eta_min: float | None = None,
        eta_max: float | None = None,
        w_eps: float | None = None,
        cold_start_chunks: int | None = None,
        parallel: bool | None = None,
        **kwargs,
    ) -> Tensor:
        if clear_states:
            self.t_absolute = 0
            self._chunk_index = 0
            self._clear_histories()

        self.update_hyperparams(
            subsampling=subsampling,
            hot_pixel_mask=hot_pixel_mask,
            normalize=normalize,
            quantile=quantile,
            chunk_size=chunk_size,
            fast_window=fast_window,
            temporal_window=temporal_window,
            fast_tau=fast_tau,
            motion_sharpness=motion_sharpness,
            motion_threshold=motion_threshold,
            eps=eps,
            stable_prior=stable_prior,
            alpha_prior=alpha_prior,
            beta_min=beta_min,
            eta_min=eta_min,
            eta_max=eta_max,
            w_eps=w_eps,
            cold_start_chunks=cold_start_chunks,
            **kwargs,
        )

        self.set_cube(photon_cube)
        use_parallel = False if parallel is None else bool(parallel)
        fused = self._integrate_parallel(photon_cube) if use_parallel else self._integrate_last(photon_cube)
        recons = self._subsample_reconstruction(fused)
        if self.hot_pixel_mask is not None:
            recons = nearest_neighbor_inpaint(recons, self.hot_pixel_mask)
        recons = self.clamp_recons(recons)
        self.t_absolute += self._t
        return recons

    def forward(self, photon_cube: Tensor) -> Tensor:
        return self._integrate_last(photon_cube)


class PoissonGammaDevianceGammaFrame(nn.Module):
    """Frame-mode wrapper that emits one reconstructed frame plus a confidence map."""

    def __init__(self, **kwargs):
        super().__init__()
        self.core = PoissonGammaDevianceGamma(**kwargs)

    @property
    def subsampling(self) -> int:
        return int(getattr(self.core, "subsampling", 1) or 1)

    @staticmethod
    def _confidence_to_frame_space(confidence_hw: Tensor) -> Tensor:
        if confidence_hw.ndim != 2:
            raise ValueError(f"Expected confidence map (H,W), got shape={tuple(confidence_hw.shape)}")
        return F.avg_pool2d(confidence_hw.unsqueeze(0).unsqueeze(0).float(), kernel_size=2, stride=2).squeeze(0)

    @torch.no_grad()
    def process_photon_cube_to_frame(
        self, photon_cube: Tensor, clear_states: bool = True, **kwargs
    ) -> tuple[Tensor, Tensor]:
        recons = self.core.process_photon_cube(photon_cube, clear_states=clear_states, **kwargs)
        if recons.ndim != 3 or int(recons.shape[-1]) == 0:
            h, w = map(int, photon_cube.shape[:2])
            recons = photon_cube.new_zeros((h, w, 1), dtype=torch.float32)
        else:
            recons = recons[..., -1:].contiguous()
        sample_weight = getattr(self.core, "sample_weight", None)
        if sample_weight is None:
            confidence_hw = photon_cube.new_ones(photon_cube.shape[0], photon_cube.shape[1], dtype=torch.float32)
        else:
            confidence_hw = sample_weight.detach().float().clamp(0.0, 1.0)
        confidence = self._confidence_to_frame_space(confidence_hw).to(device=recons.device, dtype=recons.dtype)
        return recons, confidence
