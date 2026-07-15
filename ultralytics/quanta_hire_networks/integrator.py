"""HIRE v0.3: dual-rate I^f / I^s with 1/n cold-start and hysteresis change-point reset.

Designed around discrete bin-time at the operating SPAD rate (default 2 kHz,
``chunk_size`` / ``subsampling`` = 80). Retention alphas are **bin-direct**:

    α = exp(-1 / W)     # W = fast_bins | slow_bins | surprise_bins

so ``*_bins`` plug straight into the EMA formulas (no 8 kHz ref-rate fold-back).
Optional ``tau_* > 0`` (seconds) still maps via ZOH ``α=exp(-1/(f_s·τ))`` for later
rate-transfer experiments.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.quanta_neural_networks.ops.array_ops import torch_quantile


def zoh_alpha(sample_rate_hz: float, tau_s: float) -> float:
    """Zero-order-hold discrete retention α = exp(-1/(f_s · τ))."""
    fs = float(sample_rate_hz)
    tau = float(tau_s)
    if fs <= 0.0:
        raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
    if tau <= 0.0:
        raise ValueError(f"tau_s must be > 0, got {tau_s}")
    return float(math.exp(-1.0 / (fs * tau)))


def bin_retention(window_bins: int) -> float:
    """Discrete-time EMA retention α = exp(-1/W); new-sample weight β = 1-α."""
    w = max(int(window_bins), 1)
    return float(math.exp(-1.0 / w))


def taus_from_presets(
    *,
    fast_bins: int = 16,
    slow_bins: int = 160,
    surprise_bins: int = 8,
    sample_rate_hz: float = 2000.0,
    ref_rate_hz: float | None = None,
) -> tuple[float, float, float]:
    """Map bin windows to wall-clock taus at ``sample_rate_hz`` (τ = W / f_s).

    ``ref_rate_hz`` is accepted for back-compat but ignored; bin time is native.
    """
    fs = float(sample_rate_hz if ref_rate_hz is None else sample_rate_hz)
    if fs <= 0.0:
        raise ValueError(f"sample_rate_hz must be > 0, got {fs}")
    return (
        max(int(fast_bins), 1) / fs,
        max(int(slow_bins), 1) / fs,
        max(int(surprise_bins), 1) / fs,
    )


class HIRE(nn.Module):
    """I^f / I^s rate estimator with hold+exp output mix and hard I^s reset.

    Roles (per bin)::

        I^f  — fast probe (always EMA)
        I^s  — slow bank: normal EMA; hard ``I^s←I^f`` when max-pooled S exceeds θ_on
        I_out — hold full I^f for H bins after reset, then exp-decay toward I^s

    Formulas::

        β_f = max(1/n_f, 1-α_f),   I^f ← (1-β_f) I^f + β_f x
        β_s = max(1/n_s, 1-α_s),   I^s ← (1-β_s) I^s + β_s x
        S   ← α_S S + (1-α_S) BernKL(I^f || I^s)
        S̄  = pool_k(S)   (``gate_pool`` = max | avg);  enter if S̄>θ_on; leave if S̄<θ_off
        at c==C_min: I^s←I^f, n_s←W_f, S←0, t_mix←0 (+ short cooldown)
        t_mix ← t_mix+1 (else);  g_reset = 1 if t_mix<H else exp(-(t_mix-H)/τ)
        g_soft = S̄/(S̄+θ_mix)   (θ_mix = ``mix_theta`` > 0, else 0 → disabled)
        g = max(g_reset, g_soft);   I_out = (1-g) I^s + g I^f

    ``H = mix_hold_bins`` (≤0 → ``subsampling`` / chunk_size), ``τ = mix_bins``.

    ``g_soft`` is a safety-net: wherever current pooled surprise is high the output
    leans to I^f **without** waiting for a confirmed hard reset — this rescues brief /
    fast motion that never sustains long enough to earn a reset (else it fragments and
    ghosts). Background S is low → g_soft≈0 → clean I^s. Set ``mix_theta<=0`` to disable.
    """

    def __init__(
        self,
        subsampling: int = 80,
        sample_rate_hz: float = 2000.0,
        bin_rate_hz: float | None = None,
        ref_rate_hz: float = 2000.0,
        fast_bins: int = 16,
        slow_bins: int = 160,
        surprise_bins: int = 8,
        tau_fast: float | None = None,
        tau_slow: float | None = None,
        tau_surprise: float | None = None,
        mix_hold_bins: int = 0,
        mix_bins: float = 16.0,
        mix_theta: float = 0.1,
        mix_kappa: float | None = None,  # legacy → mix_bins
        gate_theta: float | None = None,  # legacy → mix_bins
        theta_on: float = 0.15,
        theta_off: float = 0.06,
        confirm_bins: int = 1,
        cooldown_bins: int = 3,
        spatial_kernel: int = 3,
        gate_pool: str = "max",
        eps: float = 1e-5,
        normalize: bool = False,
        quantile: float = 1.0,
    ):
        super().__init__()
        fs = float(bin_rate_hz) if bin_rate_hz is not None else float(sample_rate_hz)
        if fs <= 0.0:
            raise ValueError(f"sample_rate_hz must be > 0, got {fs}")
        # ref_rate_hz kept for kwargs/back-compat; alphas use bin_retention unless tau override.
        ref = float(ref_rate_hz) if float(ref_rate_hz) > 0.0 else fs

        self.subsampling = max(int(subsampling), 1)
        self.sample_rate_hz = fs
        self.ref_rate_hz = ref
        self.fast_bins = max(int(fast_bins), 1)
        self.slow_bins = max(int(slow_bins), 1)
        self.surprise_bins = max(int(surprise_bins), 1)
        self.mix_hold_bins = int(mix_hold_bins)
        tau_mix = mix_bins
        if mix_kappa is not None:
            tau_mix = mix_kappa
        if gate_theta is not None:
            tau_mix = gate_theta
        self.mix_bins = float(tau_mix)
        if self.mix_bins <= 0.0:
            raise ValueError(f"mix_bins must be > 0, got {self.mix_bins}")
        # Back-compat attribute used by older logs/vis.
        self.mix_kappa = self.mix_bins
        # Soft output gate threshold (<=0 disables the safety-net gate).
        self.mix_theta = float(mix_theta)
        self.theta_on = float(theta_on)
        self.theta_off = float(theta_off)
        if not (self.theta_on > self.theta_off > 0.0):
            raise ValueError(
                f"require theta_on > theta_off > 0, got on={theta_on}, off={theta_off}"
            )
        self.confirm_bins = max(int(confirm_bins), 1)
        self.cooldown_bins = max(int(cooldown_bins), 0)
        k = int(spatial_kernel)
        if k < 1 or k % 2 == 0:
            raise ValueError(f"spatial_kernel must be odd and >= 1, got {spatial_kernel}")
        self.spatial_kernel = k
        gp = str(gate_pool).lower()
        if gp not in ("max", "avg"):
            raise ValueError(f"gate_pool must be 'max' or 'avg', got {gate_pool}")
        self.gate_pool = gp
        self.eps = float(eps)
        self.normalize = bool(normalize)
        self.quantile = float(quantile)

        self._tau_fast_override = float(tau_fast) if tau_fast is not None and float(tau_fast) > 0.0 else None
        self._tau_slow_override = float(tau_slow) if tau_slow is not None and float(tau_slow) > 0.0 else None
        self._tau_surprise_override = (
            float(tau_surprise) if tau_surprise is not None and float(tau_surprise) > 0.0 else None
        )
        self._sync_tau_display()
        self._rebuild_alphas()

        self.i_fast: Tensor | None = None
        self.i_slow: Tensor | None = None
        self.n_fast: Tensor | None = None
        self.n_slow: Tensor | None = None
        self.s_tilde: Tensor | None = None
        self.in_change: Tensor | None = None
        self.confirm_count: Tensor | None = None
        self.cooldown: Tensor | None = None
        self.t_mix: Tensor | None = None
        self.w_slow: Tensor | None = None

    def effective_mix_hold_bins(self) -> int:
        """Hold length H; ``mix_hold_bins<=0`` means use ``subsampling`` (chunk_size)."""
        h = int(self.mix_hold_bins)
        return int(self.subsampling) if h <= 0 else h

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(fs={self.sample_rate_hz:g}, "
            f"W_f/s/S={self.fast_bins}/{self.slow_bins}/{self.surprise_bins}, "
            f"α_f/s/S={self.alpha_fast:.4f}/{self.alpha_slow:.4f}/{self.alpha_surprise:.4f}, "
            f"mix_hold={self.effective_mix_hold_bins()} mix_τ={self.mix_bins:g} mix_θ={self.mix_theta:g}, "
            f"theta_on/off={self.theta_on:g}/{self.theta_off:g}, "
            f"confirm={self.confirm_bins}, cooldown={self.cooldown_bins}, "
            f"gate_pool={self.gate_pool}, subsampling={self.subsampling})"
        )

    def _sync_tau_display(self) -> None:
        """Wall-clock τ for logging: override or W/f_s."""
        fs = self.sample_rate_hz
        self.tau_fast = (
            self._tau_fast_override if self._tau_fast_override is not None else self.fast_bins / fs
        )
        self.tau_slow = (
            self._tau_slow_override if self._tau_slow_override is not None else self.slow_bins / fs
        )
        self.tau_surprise = (
            self._tau_surprise_override
            if self._tau_surprise_override is not None
            else self.surprise_bins / fs
        )

    def _rebuild_alphas(self) -> None:
        # Primary path: α = exp(-1/W). Override: ZOH from wall-clock τ (rate experiments).
        self.alpha_fast = (
            zoh_alpha(self.sample_rate_hz, self._tau_fast_override)
            if self._tau_fast_override is not None
            else bin_retention(self.fast_bins)
        )
        self.alpha_slow = (
            zoh_alpha(self.sample_rate_hz, self._tau_slow_override)
            if self._tau_slow_override is not None
            else bin_retention(self.slow_bins)
        )
        self.alpha_surprise = (
            zoh_alpha(self.sample_rate_hz, self._tau_surprise_override)
            if self._tau_surprise_override is not None
            else bin_retention(self.surprise_bins)
        )
        self.beta_fast_floor = 1.0 - self.alpha_fast
        self.beta_slow_floor = 1.0 - self.alpha_slow

    def clear_states(self) -> None:
        self.i_fast = None
        self.i_slow = None
        self.n_fast = None
        self.n_slow = None
        self.s_tilde = None
        self.in_change = None
        self.confirm_count = None
        self.cooldown = None
        self.t_mix = None
        self.w_slow = None

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
            "hire_mix_hold_bins": "mix_hold_bins",
            "hire_mix_bins": "mix_bins",
            "hire_mix_kappa": "mix_bins",  # legacy → τ
            "hire_gate_theta": "mix_bins",  # legacy
            "mix_kappa": "mix_bins",  # legacy
            "gate_theta": "mix_bins",  # legacy
            "hire_theta_on": "theta_on",
            "hire_theta_off": "theta_off",
            "hire_confirm_bins": "confirm_bins",
            "hire_cooldown_bins": "cooldown_bins",
            "hire_spatial_kernel": "spatial_kernel",
            "hire_mix_theta": "mix_theta",
            "hire_gate_pool": "gate_pool",
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

        if "mix_hold_bins" in normalized:
            self.mix_hold_bins = int(normalized["mix_hold_bins"])
        if "mix_bins" in normalized:
            tau = float(normalized["mix_bins"])
            if tau <= 0.0:
                raise ValueError(f"mix_bins must be > 0, got {tau}")
            self.mix_bins = tau
            self.mix_kappa = tau
        if "mix_theta" in normalized:
            self.mix_theta = float(normalized["mix_theta"])

        if "theta_on" in normalized:
            self.theta_on = float(normalized["theta_on"])
        if "theta_off" in normalized:
            self.theta_off = float(normalized["theta_off"])
        if "theta_on" in normalized or "theta_off" in normalized:
            if not (self.theta_on > self.theta_off > 0.0):
                raise ValueError(
                    f"require theta_on > theta_off > 0, got on={self.theta_on}, off={self.theta_off}"
                )

        if "confirm_bins" in normalized:
            self.confirm_bins = max(int(normalized["confirm_bins"]), 1)
        if "cooldown_bins" in normalized:
            self.cooldown_bins = max(int(normalized["cooldown_bins"]), 0)

        if "spatial_kernel" in normalized:
            k = int(normalized["spatial_kernel"])
            if k < 1 or k % 2 == 0:
                raise ValueError(f"spatial_kernel must be odd and >= 1, got {k}")
            self.spatial_kernel = k

        if "gate_pool" in normalized:
            gp = str(normalized["gate_pool"]).lower()
            if gp not in ("max", "avg"):
                raise ValueError(f"gate_pool must be 'max' or 'avg', got {gp}")
            self.gate_pool = gp

        if "eps" in normalized:
            eps = float(normalized["eps"])
            if eps <= 0.0:
                raise ValueError(f"eps must be > 0, got {eps}")
            self.eps = eps

        if "normalize" in normalized:
            self.normalize = bool(normalized["normalize"])
        if "quantile" in normalized:
            self.quantile = float(normalized["quantile"])

        tau_set = False
        for tau_key, override_attr in (
            ("tau_fast", "_tau_fast_override"),
            ("tau_slow", "_tau_slow_override"),
            ("tau_surprise", "_tau_surprise_override"),
        ):
            if tau_key not in normalized:
                continue
            val = float(normalized[tau_key])
            setattr(self, override_attr, val if val > 0.0 else None)
            tau_set = True

        if rate_changed or preset_changed or tau_set:
            self._sync_tau_display()
            self._rebuild_alphas()

    @staticmethod
    def _bernoulli_kl(p: Tensor, q: Tensor, eps: float) -> Tensor:
        p = p.clamp(eps, 1.0 - eps)
        q = q.clamp(eps, 1.0 - eps)
        return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))

    def _spatial_mean(self, surprise_hw: Tensor) -> Tensor:
        """Local pool on S before in_change / soft-gate thresholds.

        ``gate_pool='max'`` dilates/connects motion blobs (but propagates isolated
        spikes); ``gate_pool='avg'`` requires spatial mass, killing single-pixel
        background noise while filling small holes in real blobs.
        """
        k = self.spatial_kernel
        if k == 1:
            return surprise_hw
        pad = k // 2
        x = surprise_hw.unsqueeze(0).unsqueeze(0)
        if self.gate_pool == "avg":
            pooled = F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
        else:
            pooled = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
        return pooled.squeeze(0).squeeze(0)

    def _step(
        self,
        xt: Tensor,
        i_fast: Tensor,
        i_slow: Tensor,
        n_fast: Tensor,
        n_slow: Tensor,
        s_tilde: Tensor,
        in_change: Tensor,
        confirm_count: Tensor,
        cooldown: Tensor,
        t_mix: Tensor,
    ) -> tuple[Tensor, ...]:
        """One-bin HIRE update.

        Returns
        ``(i_fast, i_slow, n_fast, n_slow, s_tilde, in_change, confirm_count,
          cooldown, t_mix, i_out, w_slow, g_fast, s_raw, s_spat, did_reset)``.
        """
        eps = self.eps
        c_min = self.confirm_bins
        t_cd = float(self.cooldown_bins)
        n_f_max = float(self.fast_bins)
        n_s_max = float(self.slow_bins)
        hold_h = float(self.effective_mix_hold_bins())
        tau_mix = float(self.mix_bins)

        # --- fast always updates ---
        beta_f = torch.maximum(1.0 / n_fast, xt.new_tensor(self.beta_fast_floor))
        i_fast = (1.0 - beta_f) * i_fast + beta_f * xt
        n_fast = torch.minimum(n_fast + 1.0, xt.new_tensor(n_f_max))

        # --- slow bank: normal EMA only ---
        beta_s = torch.maximum(1.0 / n_slow, xt.new_tensor(self.beta_slow_floor))
        i_slow = (1.0 - beta_s) * i_slow + beta_s * xt
        n_slow = torch.minimum(n_slow + 1.0, xt.new_tensor(n_s_max))

        # --- evidence (S only for hard reset) ---
        s_raw = self._bernoulli_kl(i_fast, i_slow, eps)
        s_spat = s_raw
        a_s = self.alpha_surprise
        s_tilde = a_s * s_tilde + (1.0 - a_s) * s_spat

        # --- hysteresis on pooled S (also feeds soft output gate below) ---
        s_chg = self._spatial_mean(s_tilde)
        enter = s_chg > self.theta_on
        leave = s_chg < self.theta_off
        in_change = torch.where(enter, xt.new_ones(xt.shape), in_change)
        in_change = torch.where(leave, xt.new_zeros(xt.shape), in_change)

        confirm_count = torch.where(in_change > 0.5, confirm_count + 1.0, xt.new_zeros(xt.shape))
        can_reset = (
            (in_change > 0.5)
            & (cooldown <= 0.0)
            & (confirm_count >= float(c_min) - 1e-6)
            & (confirm_count < float(c_min) + 1.0 - 1e-6)
        )
        i_slow = torch.where(can_reset, i_fast, i_slow)
        n_slow = torch.where(can_reset, xt.new_full(xt.shape, n_f_max), n_slow)
        s_tilde = torch.where(can_reset, xt.new_zeros(xt.shape), s_tilde)
        in_change = torch.where(can_reset, xt.new_zeros(xt.shape), in_change)
        confirm_count = torch.where(can_reset, xt.new_zeros(xt.shape), confirm_count)
        if t_cd > 0.0:
            cooldown = torch.where(can_reset, xt.new_full(xt.shape, t_cd), cooldown)
        cooldown = torch.clamp(cooldown - 1.0, min=0.0)

        # --- hold+exp mix age (independent of n_s) ---
        t_mix = torch.where(can_reset, xt.new_zeros(xt.shape), t_mix + 1.0)
        # g_reset=1 for t<H (full I^f); then exp(-(t-H)/τ) toward I^s
        over = torch.clamp(t_mix - hold_h, min=0.0)
        g_reset = torch.where(
            t_mix < hold_h,
            xt.new_ones(xt.shape),
            torch.exp(-over / tau_mix),
        )
        # Soft safety-net: lean I^f wherever current pooled surprise is high, even
        # without a confirmed reset (rescues brief/fast motion from fragmentation).
        if self.mix_theta > 0.0:
            g_soft = s_chg / (s_chg + self.mix_theta)
            g_fast = torch.maximum(g_reset, g_soft)
        else:
            g_fast = g_reset
        w_slow = 1.0 - g_fast
        i_out = w_slow * i_slow + g_fast * i_fast

        did_reset = can_reset.to(dtype=xt.dtype)
        return (
            i_fast,
            i_slow,
            n_fast,
            n_slow,
            s_tilde,
            in_change,
            confirm_count,
            cooldown,
            t_mix,
            i_out,
            w_slow,
            g_fast,
            s_raw,
            s_spat,
            did_reset,
        )

    def _update_causal(self, photon_cube: Tensor, *, clear_states: bool) -> Tensor:
        recons, _ = self._update_causal_with_debug(photon_cube, clear_states=clear_states, record_debug=False)
        return recons

    def _update_causal_with_debug(
        self,
        photon_cube: Tensor,
        *,
        clear_states: bool,
        record_debug: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if photon_cube.ndim != 3:
            raise ValueError(f"Expected photon_cube (H,W,T), got shape={tuple(photon_cube.shape)}")
        if clear_states:
            self.clear_states()

        h, w, t_raw = map(int, photon_cube.shape)
        empty_debug: dict[str, Tensor] = {}
        if t_raw == 0:
            empty = photon_cube.new_zeros((h, w, 0), dtype=torch.float32)
            return empty, empty_debug

        raw = photon_cube.float()
        i_fast = self.i_fast
        i_slow = self.i_slow
        n_fast = self.n_fast
        n_slow = self.n_slow
        s_tilde = self.s_tilde
        in_change = self.in_change
        confirm_count = self.confirm_count
        cooldown = self.cooldown
        t_mix = self.t_mix
        w_slow = self.w_slow
        frames: list[Tensor] = []

        dbg_keys = (
            "i_fast",
            "i_slow",
            "i_out",
            "s_raw",
            "s_spat",
            "s_tilde",
            "w_slow",
            "g_fast",
            "t_mix",
            "n_slow",
            "n_fast",
            "in_change",
            "confirm",
            "cooldown",
            "did_reset",
        )
        dbg_lists: dict[str, list[Tensor]] = {k: [] for k in dbg_keys} if record_debug else {}

        # Large age ⇒ g≈0 (full slow) until the first hard reset.
        t_mix_init = float(max(self.effective_mix_hold_bins(), 1) + 10.0 * self.mix_bins)

        for t0 in range(0, t_raw, self.subsampling):
            t1 = min(t_raw, t0 + self.subsampling)
            for t in range(t0, t1):
                xt = raw[..., t]
                if i_fast is None or i_slow is None or s_tilde is None:
                    i_fast = xt.clone()
                    i_slow = xt.clone()
                    n_fast = xt.new_ones(xt.shape)
                    n_slow = xt.new_ones(xt.shape)
                    s_tilde = xt.new_zeros(xt.shape)
                    in_change = xt.new_zeros(xt.shape)
                    confirm_count = xt.new_zeros(xt.shape)
                    cooldown = xt.new_zeros(xt.shape)
                    t_mix = xt.new_full(xt.shape, t_mix_init)
                    g_fast = xt.new_zeros(xt.shape)
                    w_slow = xt.new_ones(xt.shape)
                    s_raw = xt.new_zeros(xt.shape)
                    s_spat = xt.new_zeros(xt.shape)
                    i_out = i_slow
                    did_reset = xt.new_zeros(xt.shape)
                else:
                    (
                        i_fast,
                        i_slow,
                        n_fast,
                        n_slow,
                        s_tilde,
                        in_change,
                        confirm_count,
                        cooldown,
                        t_mix,
                        i_out,
                        w_slow,
                        g_fast,
                        s_raw,
                        s_spat,
                        did_reset,
                    ) = self._step(
                        xt,
                        i_fast,
                        i_slow,
                        n_fast,
                        n_slow,
                        s_tilde,
                        in_change,
                        confirm_count,
                        cooldown,
                        t_mix,
                    )
                if record_debug:
                    dbg_lists["i_fast"].append(i_fast)
                    dbg_lists["i_slow"].append(i_slow)
                    dbg_lists["i_out"].append(i_out)
                    dbg_lists["s_raw"].append(s_raw)
                    dbg_lists["s_spat"].append(s_spat)
                    dbg_lists["s_tilde"].append(s_tilde)
                    dbg_lists["w_slow"].append(w_slow)
                    dbg_lists["g_fast"].append(g_fast)
                    dbg_lists["t_mix"].append(t_mix)
                    dbg_lists["n_slow"].append(n_slow)
                    dbg_lists["n_fast"].append(n_fast)
                    dbg_lists["in_change"].append(in_change)
                    dbg_lists["confirm"].append(confirm_count)
                    dbg_lists["cooldown"].append(cooldown)
                    dbg_lists["did_reset"].append(did_reset)
            frames.append(i_out.unsqueeze(-1))

        self.i_fast = None if i_fast is None else i_fast.detach()
        self.i_slow = None if i_slow is None else i_slow.detach()
        self.n_fast = None if n_fast is None else n_fast.detach()
        self.n_slow = None if n_slow is None else n_slow.detach()
        self.s_tilde = None if s_tilde is None else s_tilde.detach()
        self.in_change = None if in_change is None else in_change.detach()
        self.confirm_count = None if confirm_count is None else confirm_count.detach()
        self.cooldown = None if cooldown is None else cooldown.detach()
        self.t_mix = None if t_mix is None else t_mix.detach()
        self.w_slow = None if w_slow is None else w_slow.detach()

        recons_prenorm = torch.cat(frames, dim=-1)
        recons = self.clamp_recons(recons_prenorm)

        if not record_debug:
            return recons, empty_debug

        debug: dict[str, Tensor] = {"recons_prenorm": recons_prenorm, "recons": recons}
        for key, parts in dbg_lists.items():
            vol = torch.stack(parts, dim=-1)
            debug[f"{key}_hwt"] = vol
            debug[f"{key}_last"] = vol[..., -1]
            debug[f"{key}_peak"] = vol.amax(dim=-1)
            debug[f"{key}_mean"] = vol.mean(dim=-1)
        debug["w_fast_last"] = debug["g_fast_last"]
        # Chunk summaries: any hard reset / min age over the processed volume.
        debug["reset_any"] = debug["did_reset_peak"]
        debug["n_slow_min"] = debug["n_slow_hwt"].amin(dim=-1)
        # Back-compat aliases for older vis scripts.
        debug["gate_last"] = debug["w_slow_last"]
        debug["gate_hwt"] = debug["w_slow_hwt"]
        debug["conf_last"] = debug["w_fast_last"]
        return recons, debug

    def clamp_recons(self, recons: Tensor) -> Tensor:
        if recons.numel() == 0:
            return recons.float()
        recons = recons.float()
        max_value = 1.0
        if self.normalize:
            max_value = torch_quantile(recons, self.quantile).clamp(min=1e-6)
        return (recons / max_value).clamp(0, 1)

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

    @torch.no_grad()
    def process_photon_cube_with_debug(
        self,
        photon_cube: Tensor,
        clear_states: bool = True,
        subsampling: int | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return reconstructions plus per-bin intermediate maps for visualization."""
        prev = self.subsampling
        if subsampling is not None or kwargs:
            self.update_hyperparams(subsampling=subsampling, **kwargs)
        try:
            return self._update_causal_with_debug(photon_cube, clear_states=clear_states, record_debug=True)
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
        if self.w_slow is None:
            conf_hw = photon_cube.new_ones(photon_cube.shape[0], photon_cube.shape[1], dtype=torch.float32)
        else:
            # High when recently reset (prefer I^f).
            conf_hw = (1.0 - self.w_slow.float()).clamp(0.0, 1.0)
        confidence = self._confidence_to_frame_space(conf_hw).to(device=frame.device, dtype=frame.dtype)
        return frame, confidence
