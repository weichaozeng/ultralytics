"""HIRE: Hardened IIR Rate Estimation with soft τ-routing."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def zoh_alpha(sample_rate_hz: float, tau_s: float) -> float:
    """Zero-order-hold discrete retention α = exp(-1/(f_s · τ))."""
    fs = float(sample_rate_hz)
    tau = float(tau_s)
    if fs <= 0.0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
    if tau <= 0.0:
        raise ValueError(f"tau_s must be > 0, got {tau_s}")
    return float(math.exp(-1.0 / (fs * tau)))


def taus_from_presets(
    *,
    fast_bins: int = 16,
    slow_bins: int = 128,
    surprise_bins: int = 8,
    ref_rate_hz: float = 8000.0,
) -> tuple[float, float, float]:
    """Map reference-bin presets to wall-clock taus (independent of current f_s)."""
    ref = float(ref_rate_hz)
    if ref <= 0.0:
        raise ValueError(f"ref_rate_hz must be > 0, got {ref_rate_hz}")
    fb = max(int(fast_bins), 1)
    sb = max(int(slow_bins), 1)
    qb = max(int(surprise_bins), 1)
    return fb / ref, sb / ref, qb / ref


def _resolve_tau(
    *,
    bins: int,
    ref_rate_hz: float,
    tau_override: float | None,
) -> float:
    if tau_override is not None and float(tau_override) > 0.0:
        return float(tau_override)
    return float(max(int(bins), 1)) / float(ref_rate_hz)


class HIRE(nn.Module):
    """τ-anchored dual-rate IIR with Bernoulli-KL soft routing.

    Primary user knob is ``sample_rate_hz`` (SPAD bin rate). Wall-clock
    ``tau_*`` are derived from reference-bin presets at ``ref_rate_hz`` unless
    explicit ``tau_*`` overrides are set ``> 0``.
    """

    def __init__(
        self,
        subsampling: int = 1,
        sample_rate_hz: float = 8000.0,
        bin_rate_hz: float | None = None,
        ref_rate_hz: float = 8000.0,
        fast_bins: int = 16,
        slow_bins: int = 128,
        surprise_bins: int = 8,
        tau_fast: float | None = None,
        tau_slow: float | None = None,
        tau_surprise: float | None = None,
        gate_theta: float = 0.05,
        spatial_kernel: int = 3,
        eps: float = 1e-5,
    ):
        super().__init__()
        fs = float(bin_rate_hz) if bin_rate_hz is not None else float(sample_rate_hz)
        if fs <= 0.0:
            raise ValueError(f"sample_rate_hz must be > 0, got {fs}")
        ref = float(ref_rate_hz)
        if ref <= 0.0:
            raise ValueError(f"ref_rate_hz must be > 0, got {ref_rate_hz}")

        self.subsampling = max(int(subsampling), 1)
        self.sample_rate_hz = fs
        self.ref_rate_hz = ref
        self.fast_bins = max(int(fast_bins), 1)
        self.slow_bins = max(int(slow_bins), 1)
        self.surprise_bins = max(int(surprise_bins), 1)
        self.gate_theta = float(gate_theta)
        if self.gate_theta <= 0.0:
            raise ValueError(f"gate_theta must be > 0, got {gate_theta}")
        k = int(spatial_kernel)
        if k < 1 or k % 2 == 0:
            raise ValueError(f"spatial_kernel must be odd and >= 1, got {spatial_kernel}")
        self.spatial_kernel = k
        self.eps = float(eps)

        self.tau_fast = _resolve_tau(bins=self.fast_bins, ref_rate_hz=ref, tau_override=tau_fast)
        self.tau_slow = _resolve_tau(bins=self.slow_bins, ref_rate_hz=ref, tau_override=tau_slow)
        self.tau_surprise = _resolve_tau(bins=self.surprise_bins, ref_rate_hz=ref, tau_override=tau_surprise)
        if not (self.tau_fast > 0.0 and self.tau_slow > 0.0 and self.tau_surprise > 0.0):
            raise ValueError("All taus must be > 0")
        if self.tau_fast > self.tau_slow:
            # Soft-routing still works; keep the user's / preset values.
            pass

        self._rebuild_alphas()

        self.i_fast: Tensor | None = None
        self.i_out: Tensor | None = None
        self.s_tilde: Tensor | None = None
        self.gate: Tensor | None = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(fs={self.sample_rate_hz:g}, "
            f"tau_fast={self.tau_fast:g}, tau_slow={self.tau_slow:g}, "
            f"tau_surprise={self.tau_surprise:g}, gate_theta={self.gate_theta:g}, "
            f"subsampling={self.subsampling})"
        )

    def _rebuild_alphas(self) -> None:
        self.alpha_fast = zoh_alpha(self.sample_rate_hz, self.tau_fast)
        self.alpha_surprise = zoh_alpha(self.sample_rate_hz, self.tau_surprise)
        self.log_tau_fast = float(math.log(self.tau_fast))
        self.log_tau_slow = float(math.log(self.tau_slow))

    def clear_states(self) -> None:
        self.i_fast = None
        self.i_out = None
        self.s_tilde = None
        self.gate = None

    def reset(self) -> None:
        self.clear_states()

    def recon_t_indices(self, t_raw: int, num_frames: int) -> list[int]:
        if t_raw <= 0 or num_frames <= 0:
            return []
        if t_raw <= self.subsampling:
            return [int(t_raw)]
        idx: list[int] = []
        for t1 in range(self.subsampling, t_raw + 1, self.subsampling):
            idx.append(int(min(t1, t_raw)))
        if idx and idx[-1] != int(t_raw):
            idx.append(int(t_raw))
        return idx[:num_frames]

    def update_hyperparams(self, **kwargs: Any) -> None:
        """Update HIRE attributes and rebuild ZOH alphas when needed."""
        if "subsampling" in kwargs and kwargs["subsampling"] is not None:
            self.subsampling = max(int(kwargs["subsampling"]), 1)

        alias_map = {
            "bin_rate_hz": "sample_rate_hz",
            "hire_ref_rate_hz": "ref_rate_hz",
            "hire_fast_bins": "fast_bins",
            "hire_slow_bins": "slow_bins",
            "hire_surprise_bins": "surprise_bins",
            "hire_gate_theta": "gate_theta",
            "hire_spatial_kernel": "spatial_kernel",
            "hire_tau_fast": "tau_fast",
            "hire_tau_slow": "tau_slow",
            "hire_tau_surprise": "tau_surprise",
        }
        normalized: dict[str, Any] = {}
        for key, value in kwargs.items():
            if value is None or key == "subsampling":
                continue
            normalized[alias_map.get(key, key)] = value
        if not normalized:
            return

        preset_changed = False
        rate_changed = False

        if "sample_rate_hz" in normalized:
            fs = float(normalized["sample_rate_hz"])
            if fs <= 0.0:
                raise ValueError(f"sample_rate_hz must be > 0, got {fs}")
            self.sample_rate_hz = fs
            rate_changed = True

        if "ref_rate_hz" in normalized:
            ref = float(normalized["ref_rate_hz"])
            if ref <= 0.0:
                raise ValueError(f"ref_rate_hz must be > 0, got {ref}")
            self.ref_rate_hz = ref
            preset_changed = True

        for bins_key in ("fast_bins", "slow_bins", "surprise_bins"):
            if bins_key in normalized:
                setattr(self, bins_key, max(int(normalized[bins_key]), 1))
                preset_changed = True

        if "gate_theta" in normalized:
            theta = float(normalized["gate_theta"])
            if theta <= 0.0:
                raise ValueError(f"gate_theta must be > 0, got {theta}")
            self.gate_theta = theta

        if "spatial_kernel" in normalized:
            k = int(normalized["spatial_kernel"])
            if k < 1 or k % 2 == 0:
                raise ValueError(f"spatial_kernel must be odd and >= 1, got {k}")
            self.spatial_kernel = k

        if "eps" in normalized:
            eps = float(normalized["eps"])
            if eps <= 0.0:
                raise ValueError(f"eps must be > 0, got {eps}")
            self.eps = eps

        tau_set = False
        for tau_key, bins_attr in (
            ("tau_fast", "fast_bins"),
            ("tau_slow", "slow_bins"),
            ("tau_surprise", "surprise_bins"),
        ):
            if tau_key not in normalized:
                continue
            val = float(normalized[tau_key])
            if val > 0.0:
                setattr(self, tau_key, val)
            else:
                setattr(self, tau_key, getattr(self, bins_attr) / self.ref_rate_hz)
            tau_set = True

        if preset_changed and not tau_set:
            self.tau_fast = self.fast_bins / self.ref_rate_hz
            self.tau_slow = self.slow_bins / self.ref_rate_hz
            self.tau_surprise = self.surprise_bins / self.ref_rate_hz

        if rate_changed or preset_changed or tau_set:
            self._rebuild_alphas()

    @staticmethod
    def _bernoulli_kl(p: Tensor, q: Tensor, eps: float) -> Tensor:
        p = p.clamp(eps, 1.0 - eps)
        q = q.clamp(eps, 1.0 - eps)
        return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))

    def _spatial_mean(self, surprise_hw: Tensor) -> Tensor:
        k = self.spatial_kernel
        if k == 1:
            return surprise_hw
        pad = k // 2
        x = surprise_hw.unsqueeze(0).unsqueeze(0)
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad).squeeze(0).squeeze(0)

    def _step(self, xt: Tensor, i_fast: Tensor, i_out: Tensor, s_tilde: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One-bin HIRE update; returns (i_fast, i_out, s_tilde, gate)."""
        a_f = self.alpha_fast
        a_s = self.alpha_surprise
        eps = self.eps
        theta = self.gate_theta

        i_fast = a_f * i_fast + (1.0 - a_f) * xt
        s_raw = self._bernoulli_kl(i_fast, i_out, eps)
        s_spat = self._spatial_mean(s_raw)
        s_tilde = a_s * s_tilde + (1.0 - a_s) * s_spat
        gate = s_tilde / (s_tilde + theta)

        # τ soft-interpolation in log space, then ZOH α map.
        log_tau = (1.0 - gate) * self.log_tau_slow + gate * self.log_tau_fast
        tau = torch.exp(log_tau).clamp(min=eps)
        alpha = torch.exp(-1.0 / (self.sample_rate_hz * tau))
        i_out = alpha * i_out + (1.0 - alpha) * xt
        return i_fast, i_out, s_tilde, gate

    def _update_causal(self, photon_cube: Tensor, *, clear_states: bool) -> Tensor:
        if photon_cube.ndim != 3:
            raise ValueError(f"Expected photon_cube (H,W,T), got shape={tuple(photon_cube.shape)}")
        if clear_states:
            self.clear_states()
        if photon_cube.shape[-1] == 0:
            h, w = map(int, photon_cube.shape[:2])
            return photon_cube.new_zeros((h, w, 0), dtype=torch.float32)

        raw = photon_cube.float()
        t_raw = int(raw.shape[-1])
        i_fast = self.i_fast
        i_out = self.i_out
        s_tilde = self.s_tilde
        gate = self.gate
        frames: list[Tensor] = []

        for t0 in range(0, t_raw, self.subsampling):
            t1 = min(t_raw, t0 + self.subsampling)
            for t in range(t0, t1):
                xt = raw[..., t]
                if i_fast is None or i_out is None or s_tilde is None:
                    i_fast = xt.clone()
                    i_out = xt.clone()
                    s_tilde = xt.new_zeros(xt.shape)
                    gate = xt.new_zeros(xt.shape)
                else:
                    i_fast, i_out, s_tilde, gate = self._step(xt, i_fast, i_out, s_tilde)
            frames.append(i_out.unsqueeze(-1))

        self.i_fast = None if i_fast is None else i_fast.detach()
        self.i_out = None if i_out is None else i_out.detach()
        self.s_tilde = None if s_tilde is None else s_tilde.detach()
        self.gate = None if gate is None else gate.detach()
        return torch.cat(frames, dim=-1)

    @torch.no_grad()
    def process_photon_cube(
        self,
        photon_cube: Tensor,
        clear_states: bool = True,
        subsampling: int | None = None,
        **kwargs: Any,
    ) -> Tensor:
        prev = self.subsampling
        if subsampling is not None or kwargs:
            self.update_hyperparams(subsampling=subsampling, **kwargs)
        try:
            return self._update_causal(photon_cube, clear_states=clear_states)
        finally:
            self.subsampling = prev


class HIREFrame(HIRE):
    """Frame-mode HIRE: last emitted recon + Bayer-pooled confidence ``1 - g``."""

    @staticmethod
    def _confidence_to_frame_space(confidence_hw: Tensor) -> Tensor:
        if confidence_hw.ndim != 2:
            raise ValueError(f"Expected confidence map (H,W), got shape={tuple(confidence_hw.shape)}")
        return F.avg_pool2d(confidence_hw.unsqueeze(0).unsqueeze(0).float(), kernel_size=2, stride=2).squeeze(0)

    @torch.no_grad()
    def process_photon_cube_to_frame(
        self,
        photon_cube: Tensor,
        clear_states: bool = True,
        subsampling: int | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        prev = self.subsampling
        if subsampling is not None or kwargs:
            self.update_hyperparams(subsampling=subsampling, **kwargs)
        try:
            recons = self._update_causal(photon_cube, clear_states=clear_states)
        finally:
            self.subsampling = prev

        if recons.ndim != 3 or int(recons.shape[-1]) == 0:
            h, w = map(int, photon_cube.shape[:2])
            empty = photon_cube.new_zeros((h, w, 0), dtype=torch.float32)
            confidence = photon_cube.new_zeros((1, h // 2, w // 2), dtype=torch.float32)
            return empty, confidence

        frame = recons[..., -1:].contiguous()
        if self.gate is None:
            conf_hw = photon_cube.new_ones(photon_cube.shape[0], photon_cube.shape[1], dtype=torch.float32)
        else:
            conf_hw = (1.0 - self.gate.float()).clamp(0.0, 1.0)
        confidence = self._confidence_to_frame_space(conf_hw).to(device=frame.device, dtype=frame.dtype)
        return frame, confidence
