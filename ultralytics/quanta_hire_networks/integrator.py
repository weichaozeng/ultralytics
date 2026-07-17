"""HIRE v0.3: dual-rate I^f / I^s with 1/n cold-start and hysteresis change-point reset.

Retention alphas are **bin-direct**:

    α = exp(-1 / W)     # W = fast_bins | slow_bins | surprise_bins

so ``*_bins`` plug straight into the EMA formulas.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.quanta_neural_networks.ops.array_ops import torch_quantile


def bin_retention(window_bins: int) -> float:
    """Discrete-time EMA retention α = exp(-1/W); new-sample weight β = 1-α."""
    w = max(int(window_bins), 1)
    return float(math.exp(-1.0 / w))


class HIRE(nn.Module):
    """I^f / I^s rate estimator with hold+exp output mix and hard I^s reset.

    Roles (per bin)::

        I^f  — fast probe (always EMA); age set to W_f with I^s on hard reset
        I^s  — slow bank: EMA when idle, frozen while in_change; hard ``I^s←I^f`` + n_s←n_f←W_f;
               leave-without-reset keeps frozen I^s and resumes EMA next bin
        I_out — short hold of I^f after reset, then exp-decay toward I^s; plus deadzoned soft gate

    Formulas::

        β_f = max(1/n_f, 1-α_f),   I^f ← (1-β_f) I^f + β_f x
        if not in_change:  β_s = max(1/n_s, 1-α_s), I^s ← (1-β_s) I^s + β_s x   # freeze I^s during change
        S   ← α_S S + (1-α_S) BernKL(I^f || I^s)
        S̄  = pool_k(S)   (``gate_pool`` = max | avg);  enter if S̄>θ_on; leave if S̄<θ_off
        at c>=C_min: seed → geodesic inside {S̄>θ_grow}; each bin while armed:
                     I^s←I^f, n_s←n_f←W_f  (level trigger; fills motion over time)
        leave & ¬reset: keep I^s (frozen value); next bin resumes EMA
                     (leave via S̄<θ_off clears in_change / confirm)
        t_mix ← t_mix+1 (else);  g_reset = 1 if t_mix<H else exp(-(t-H)/τ)
        S₊ = relu(S̄ - θ_floor);  g_soft = S₊/(S₊+θ_mix)   (θ_mix<=0 disables)
        g = max(g_reset, g_soft);   I_out = (1-g) I^s + g I^f

    Hold ``H = mix_hold_bins``: ``H<0`` → chunk/subsampling (legacy); ``H>=0`` → that many
    bins (default 80 ≈ one 2 kHz chunk). Soft floor defaults to ``θ_off``; grow threshold to ``θ_off``.

    Design notes (using KL well)::

    - KL / S is typically a **thin edge map**. Blind dilate into background creates false
      resets; soft ``S/(S+θ)`` without a floor mixes background noise into I_out.
    - **Freeze I^s while in_change** so KL is vs pre-change bank (seeds stay sharp).
    - Updates resume when ``in_change`` clears: leave keeps frozen I^s then EMA;
      hard reset does ``I^s←I^f`` and sets both ages to ``W_f`` (avoids n←1 dark rims).
    - **Geodesic grow** expands confirmed high-S seeds only through moderate-S support.
    - Hard reset is **level-triggered** (``confirm≥C`` every bin while in_change), not
      a one-shot edge — geodesic can fill motion bands across successive bins.
    - Ages at ``W_f`` keep β≈1/W_f so I^f≈I^s after each refresh — replaces cooldown.
    - Soft gate uses the **same deadzone** so low-S salt does not pull I^f into I_out.
    """

    def __init__(
        self,
        subsampling: int = 80,
        sample_rate_hz: float = 2000.0,
        bin_rate_hz: float | None = None,
        fast_bins: int = 24,
        slow_bins: int = 160,
        surprise_bins: int = 4,
        mix_hold_bins: int = 80,
        mix_bins: float = 12.0,
        mix_theta: float = 0.06,
        mix_floor: float | None = None,
        mix_kappa: float | None = None,  # legacy → mix_bins
        gate_theta: float | None = None,  # legacy → mix_bins
        theta_on: float = 0.08,
        theta_off: float = 0.02,
        theta_grow: float | None = None,
        confirm_bins: int = 4,
        cooldown_bins: int = 0,  # unused (n←W_f after reset replaces anti-chatter)
        spatial_kernel: int = 5,
        gate_pool: str = "max",
        reset_open: int = 15,
        reset_grow: int = 6,
        reset_dilate: int | None = None,  # legacy → reset_grow steps
        eps: float = 1e-5,
        normalize: bool = False,
        quantile: float = 1.0,
    ):
        super().__init__()
        fs = float(bin_rate_hz) if bin_rate_hz is not None else float(sample_rate_hz)
        if fs <= 0.0:
            raise ValueError(f"sample_rate_hz must be > 0, got {fs}")

        self.subsampling = max(int(subsampling), 1)
        self.sample_rate_hz = fs
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
        self.mix_floor = self._optional_threshold(mix_floor)
        self.theta_on = float(theta_on)
        self.theta_off = float(theta_off)
        if not (self.theta_on > self.theta_off > 0.0):
            raise ValueError(
                f"require theta_on > theta_off > 0, got on={theta_on}, off={theta_off}"
            )
        self.theta_grow = self._optional_threshold(theta_grow)
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
        self.reset_open = self._validate_odd_kernel(reset_open, "reset_open")
        # Geodesic grow steps; legacy reset_dilate (odd kernel) ≈ half-width in steps.
        if reset_dilate is not None:
            rd = int(reset_dilate)
            grow = max(rd // 2, 0) if rd > 1 else max(int(reset_grow), 0)
        else:
            grow = max(int(reset_grow), 0)
        self.reset_grow = grow
        self.reset_dilate = max(2 * grow + 1, 1)  # back-compat attr for logs/vis
        self.eps = float(eps)
        self.normalize = bool(normalize)
        self.quantile = float(quantile)

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
        """Hold length H. ``H<0`` → chunk/subsampling (legacy); ``H>=0`` → that many bins."""
        h = int(self.mix_hold_bins)
        return int(self.subsampling) if h < 0 else max(h, 0)

    def effective_mix_floor(self) -> float:
        """Soft-gate deadzone; default ``theta_off`` so background S does not mix I^f."""
        if self.mix_floor is None:
            return float(self.theta_off)
        return float(self.mix_floor)

    def effective_theta_grow(self) -> float:
        """Geodesic support threshold; default ``theta_off`` (moderate evidence corridor)."""
        if self.theta_grow is None:
            return float(self.theta_off)
        return float(self.theta_grow)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(fs={self.sample_rate_hz:g}, "
            f"W_f/s/S={self.fast_bins}/{self.slow_bins}/{self.surprise_bins}, "
            f"α_f/s/S={self.alpha_fast:.4f}/{self.alpha_slow:.4f}/{self.alpha_surprise:.4f}, "
            f"mix_hold={self.effective_mix_hold_bins()} mix_τ={self.mix_bins:g} "
            f"mix_θ/floor={self.mix_theta:g}/{self.effective_mix_floor():g}, "
            f"theta_on/off/grow={self.theta_on:g}/{self.theta_off:g}/{self.effective_theta_grow():g}, "
            f"confirm={self.confirm_bins}, cooldown={self.cooldown_bins}, "
            f"gate_pool={self.gate_pool}, reset_open/grow={self.reset_open}/{self.reset_grow}, "
            f"subsampling={self.subsampling})"
        )

    @staticmethod
    def _validate_odd_kernel(value: int, name: str) -> int:
        k = int(value)
        if k < 1 or k % 2 == 0:
            raise ValueError(f"{name} must be odd and >= 1, got {value}")
        return k

    @staticmethod
    def _optional_threshold(value: float | None) -> float | None:
        """``None`` or ``<0`` means follow the linked default (θ_off)."""
        if value is None:
            return None
        v = float(value)
        return None if v < 0.0 else v

    def _rebuild_alphas(self) -> None:
        """α = exp(-1/W) from bin windows."""
        self.alpha_fast = bin_retention(self.fast_bins)
        self.alpha_slow = bin_retention(self.slow_bins)
        self.alpha_surprise = bin_retention(self.surprise_bins)
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
        """Update HIRE attributes and rebuild bin-retention alphas when needed."""
        if "subsampling" in kwargs and kwargs["subsampling"] is not None:
            self.subsampling = max(int(kwargs["subsampling"]), 1)

        alias_map = {
            "bin_rate_hz": "sample_rate_hz",
            "hire_fast_bins": "fast_bins",
            "hire_slow_bins": "slow_bins",
            "hire_surprise_bins": "surprise_bins",
            "hire_mix_hold_bins": "mix_hold_bins",
            "hire_mix_bins": "mix_bins",
            "hire_mix_kappa": "mix_bins",  # legacy → mix_bins
            "hire_gate_theta": "mix_bins",  # legacy
            "mix_kappa": "mix_bins",  # legacy
            "gate_theta": "mix_bins",  # legacy
            "hire_mix_theta": "mix_theta",
            "hire_mix_floor": "mix_floor",
            "hire_theta_on": "theta_on",
            "hire_theta_off": "theta_off",
            "hire_theta_grow": "theta_grow",
            "hire_confirm_bins": "confirm_bins",
            "hire_cooldown_bins": "cooldown_bins",
            "hire_spatial_kernel": "spatial_kernel",
            "hire_gate_pool": "gate_pool",
            "hire_reset_open": "reset_open",
            "hire_reset_grow": "reset_grow",
            "hire_reset_dilate": "reset_dilate",  # legacy → grow steps
        }
        normalized: dict[str, Any] = {}
        for key, value in kwargs.items():
            if value is None or key == "subsampling":
                continue
            # Silently ignore removed legacy keys so old configs/fingerprints do not crash.
            if key in {"ref_rate_hz", "hire_ref_rate_hz", "tau_fast", "tau_slow", "tau_surprise",
                       "hire_tau_fast", "hire_tau_slow", "hire_tau_surprise"}:
                continue
            normalized[alias_map.get(key, key)] = value
        if not normalized:
            return

        preset_changed = False

        if "sample_rate_hz" in normalized:
            fs = float(normalized["sample_rate_hz"])
            if fs <= 0.0:
                raise ValueError(f"sample_rate_hz must be > 0, got {fs}")
            self.sample_rate_hz = fs

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
        if "mix_floor" in normalized:
            self.mix_floor = self._optional_threshold(normalized["mix_floor"])

        if "theta_on" in normalized:
            self.theta_on = float(normalized["theta_on"])
        if "theta_off" in normalized:
            self.theta_off = float(normalized["theta_off"])
        if "theta_grow" in normalized:
            self.theta_grow = self._optional_threshold(normalized["theta_grow"])
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

        if "reset_open" in normalized:
            self.reset_open = self._validate_odd_kernel(normalized["reset_open"], "reset_open")
        if "reset_grow" in normalized:
            self.reset_grow = max(int(normalized["reset_grow"]), 0)
            self.reset_dilate = max(2 * self.reset_grow + 1, 1)
        if "reset_dilate" in normalized:
            # Legacy odd kernel → approximate geodesic step count.
            rd = int(normalized["reset_dilate"])
            if rd > 1:
                self.reset_grow = max(rd // 2, 0)
            self.reset_dilate = max(2 * self.reset_grow + 1, 1)

        if "eps" in normalized:
            eps = float(normalized["eps"])
            if eps <= 0.0:
                raise ValueError(f"eps must be > 0, got {eps}")
            self.eps = eps

        if "normalize" in normalized:
            self.normalize = bool(normalized["normalize"])
        if "quantile" in normalized:
            self.quantile = float(normalized["quantile"])

        if preset_changed:
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

    def _open_mask(self, seed: Tensor) -> Tensor:
        """Morphological opening (optional); kills isolated 1-px seeds when kernel > 1."""
        if self.reset_open <= 1:
            return seed
        k = self.reset_open
        pad = k // 2
        x = seed.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=pad)
        x = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
        return x.squeeze(0).squeeze(0) > 0.5

    def _geodesic_grow(self, seed: Tensor, support: Tensor) -> Tensor:
        """Grow confirmed seeds only through moderate-evidence support (geodesic dilate).

        Unlike blind dilate, pixels without KL support never join the reset region —
        so motion edge bands fill while quiet background stays untouched.
        """
        region = seed & support
        steps = int(self.reset_grow)
        if steps <= 0:
            return region
        for _ in range(steps):
            dil = F.max_pool2d(region.float().unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze(0).squeeze(0) > 0.5
            region = dil & support
        return region

    def _expand_reset_mask(self, seed: Tensor, s_chg: Tensor) -> Tensor:
        """Open (optional) then geodesic-grow confirmed reset seeds inside {S̄>θ_grow}."""
        seed = self._open_mask(seed)
        support = s_chg > self.effective_theta_grow()
        return self._geodesic_grow(seed, support)

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
        n_f_max = float(self.fast_bins)
        n_s_max = float(self.slow_bins)
        hold_h = float(self.effective_mix_hold_bins())
        tau_mix = float(self.mix_bins)
        in_change_prev = in_change

        # --- fast always updates ---
        beta_f = torch.maximum(1.0 / n_fast, xt.new_tensor(self.beta_fast_floor))
        i_fast = (1.0 - beta_f) * i_fast + beta_f * xt
        n_fast = torch.minimum(n_fast + 1.0, xt.new_tensor(n_f_max))

        # --- slow bank: freeze while in_change (keep pre-change I^s for sharp KL) ---
        beta_s = torch.maximum(1.0 / n_slow, xt.new_tensor(self.beta_slow_floor))
        i_slow_upd = (1.0 - beta_s) * i_slow + beta_s * xt
        n_slow_upd = torch.minimum(n_slow + 1.0, xt.new_tensor(n_s_max))
        static = in_change_prev < 0.5
        i_slow = torch.where(static, i_slow_upd, i_slow)
        n_slow = torch.where(static, n_slow_upd, n_slow)

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
        # Level trigger: once confirmed, keep resetting every bin while still in_change
        # (fills motion bands over time; edge-only left thin ridges).
        can_reset = (in_change > 0.5) & (confirm_count >= float(c_min) - 1e-6)
        # Expand seeds through moderate-S support (geodesic); not blind dilate.
        can_reset = self._expand_reset_mask(can_reset, s_chg)
        i_slow = torch.where(can_reset, i_fast, i_slow)
        # Both ages → W_f: I^s matches I^f content without n←1 under-integration dark rims.
        n_age = xt.new_full(xt.shape, n_f_max)
        n_slow = torch.where(can_reset, n_age, n_slow)
        n_fast = torch.where(can_reset, n_age, n_fast)
        # Do not clear S / in_change here: zeroing S would trip leave and disarm the level latch.
        # Leave (S̄<θ_off) alone ends the sustained-reset episode.
        cooldown = xt.new_zeros(xt.shape)  # kept for debug/vis; anti-chatter unused

        # --- hold+exp mix age (independent of n_s) ---
        t_mix = torch.where(can_reset, xt.new_zeros(xt.shape), t_mix + 1.0)
        # g_reset=1 for t<H (full I^f); then exp(-(t-H)/τ) toward I^s
        over = torch.clamp(t_mix - hold_h, min=0.0)
        if hold_h > 0.0:
            g_reset = torch.where(
                t_mix < hold_h,
                xt.new_ones(xt.shape),
                torch.exp(-over / tau_mix),
            )
        else:
            g_reset = torch.exp(-t_mix / tau_mix)
        # Deadzoned soft gate: background below θ_floor contributes 0 (no I^f bleed).
        if self.mix_theta > 0.0:
            s_eff = torch.clamp(s_chg - self.effective_mix_floor(), min=0.0)
            g_soft = s_eff / (s_eff + self.mix_theta)
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

        def _dbg_cpu(x: Tensor) -> Tensor:
            """Debug H×W frames on CPU — keeps algorithm tensors on the compute device."""
            return x.detach().to(device="cpu", dtype=torch.float32)

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
                    dbg_lists["i_fast"].append(_dbg_cpu(i_fast))
                    dbg_lists["i_slow"].append(_dbg_cpu(i_slow))
                    dbg_lists["i_out"].append(_dbg_cpu(i_out))
                    dbg_lists["s_raw"].append(_dbg_cpu(s_raw))
                    dbg_lists["s_spat"].append(_dbg_cpu(s_spat))
                    dbg_lists["s_tilde"].append(_dbg_cpu(s_tilde))
                    dbg_lists["w_slow"].append(_dbg_cpu(w_slow))
                    dbg_lists["g_fast"].append(_dbg_cpu(g_fast))
                    dbg_lists["t_mix"].append(_dbg_cpu(t_mix))
                    dbg_lists["n_slow"].append(_dbg_cpu(n_slow))
                    dbg_lists["n_fast"].append(_dbg_cpu(n_fast))
                    dbg_lists["in_change"].append(_dbg_cpu(in_change))
                    dbg_lists["confirm"].append(_dbg_cpu(confirm_count))
                    dbg_lists["cooldown"].append(_dbg_cpu(cooldown))
                    dbg_lists["did_reset"].append(_dbg_cpu(did_reset))
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

        debug: dict[str, Tensor] = {
            "recons_prenorm": recons_prenorm.detach().cpu(),
            "recons": recons.detach().cpu(),
        }
        for key, parts in dbg_lists.items():
            vol = torch.stack(parts, dim=-1)  # already on CPU
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
