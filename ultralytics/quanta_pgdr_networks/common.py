"""Shared PG-DR helpers used by fusion and gamma integrator paths."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def poisson_deviance(k: Tensor, mu: Tensor, eps: float = 1e-5) -> Tensor:
    """Poisson deviance D(k, mu) with stable k=0 branch."""
    k = k.float()
    mu = mu.float().clamp(min=eps)
    while mu.ndim < k.ndim:
        mu = mu.unsqueeze(-1)
    k_pos = k > eps
    k_safe = k.clamp(min=eps)
    dev = 2.0 * (k_safe * torch.log(k_safe / mu) + mu - k_safe)
    return torch.where(k_pos, dev, 2.0 * mu)


def normalize_kernel(taps: Tensor) -> Tensor:
    return taps / taps.sum().clamp(min=1e-12)


def build_fast_kernel(fast_window: int, fast_tau: float, device: torch.device | str) -> Tensor:
    gamma_age = torch.arange(fast_window, 0, -1, dtype=torch.float32, device=device)
    fast = gamma_age * torch.exp(-gamma_age / max(float(fast_tau), 1e-6))
    return normalize_kernel(fast).view(1, 1, -1)


def build_stea_smooth_kernel(temporal_window: int, device: torch.device | str) -> Tensor:
    stea = torch.ones((1, 1, temporal_window, 3, 3), dtype=torch.float32, device=device)
    return normalize_kernel(stea)


def history_or_zeros(
    history: Tensor | None, h: int, w: int, length: int, x: Tensor
) -> Tensor:
    if length <= 0:
        return x.new_zeros(h, w, 0)
    if history is None or tuple(history.shape[:2]) != (h, w):
        return x.new_zeros(h, w, length)
    if int(history.shape[-1]) >= length:
        return history[..., -length:].to(device=x.device, dtype=x.dtype)
    pad = x.new_zeros(h, w, length - int(history.shape[-1]))
    return torch.cat([pad, history.to(device=x.device, dtype=x.dtype)], dim=-1)


def append_temporal_history(history: Tensor, frame_hw1: Tensor, maxlen: int) -> Tensor:
    if maxlen <= 0:
        return history
    if int(history.shape[-1]) == 0:
        out = frame_hw1
    else:
        out = torch.cat([history, frame_hw1], dim=-1)
    if int(out.shape[-1]) > maxlen:
        out = out[..., -maxlen:]
    return out


def causal_conv1d_step(x_hw1: Tensor, kernel: Tensor, hist_hwt: Tensor) -> Tensor:
    h, w, _ = map(int, x_hw1.shape)
    hlen = int(kernel.shape[-1]) - 1
    hist_flat = hist_hwt.reshape(-1, 1, hlen)
    x_flat = x_hw1.reshape(h * w, 1, 1)
    return F.conv1d(torch.cat([hist_flat, x_flat], dim=-1), kernel).reshape(h, w)


def causal_conv1d_full(x_hwt: Tensor, kernel: Tensor, hist_hwt: Tensor) -> Tensor:
    h, w, t = map(int, x_hwt.shape)
    hlen = int(kernel.shape[-1]) - 1
    hist_flat = hist_hwt.reshape(-1, 1, hlen)
    x_flat = x_hwt.reshape(h * w, 1, t)
    return F.conv1d(torch.cat([hist_flat, x_flat], dim=-1), kernel).reshape(h, w, t)


def causal_window_sum_step(x_hw1: Tensor, hist_hwt: Tensor, window: int) -> Tensor:
    context = append_temporal_history(hist_hwt, x_hw1, window - 1)
    return context.sum(dim=-1)


def causal_window_sum_full(x_hwt: Tensor, hist_hwt: Tensor, window: int) -> Tensor:
    """Sliding window sum via conv1d (no Python loop over time)."""
    h, w, t = map(int, x_hwt.shape)
    window = max(int(window), 1)
    hist = history_or_zeros(hist_hwt, h, w, window - 1, x_hwt)
    seq = torch.cat([hist, x_hwt], dim=-1)
    seq_flat = seq.reshape(h * w, 1, window - 1 + t)
    kernel = x_hwt.new_ones(1, 1, window)
    out = F.conv1d(seq_flat, kernel)[..., :t]
    return out.reshape(h, w, t)


def smooth_deviance_step(d_raw_hw1: Tensor, d_hist: Tensor, stea_kernel: Tensor) -> Tensor:
    context = torch.cat([d_hist, d_raw_hw1], dim=-1)[..., -int(stea_kernel.shape[2]) :]
    d_thw = context.permute(2, 0, 1).unsqueeze(1)
    d_5d = d_thw.unsqueeze(0).transpose(1, 2)
    d_5d = F.pad(d_5d, (1, 1, 1, 1, 0, 0))
    return F.conv3d(d_5d, stea_kernel).squeeze(0).squeeze(0).squeeze(0)


def smooth_deviance_full(d_raw_hwt: Tensor, d_hist: Tensor, stea_kernel: Tensor) -> Tensor:
    h, w, t = map(int, d_raw_hwt.shape)
    hist = history_or_zeros(d_hist, h, w, int(stea_kernel.shape[2]) - 1, d_raw_hwt)
    context = torch.cat([hist, d_raw_hwt], dim=-1)
    d_5d = context.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    d_5d = F.pad(d_5d, (1, 1, 1, 1, 0, 0))
    return F.conv3d(d_5d, stea_kernel).squeeze(0).squeeze(0).permute(1, 2, 0)


def motion_gate(d_smooth: Tensor, sharpness: float, threshold: float) -> Tensor:
    return torch.sigmoid(float(sharpness) * (d_smooth - float(threshold)))


def fuse_stea_style(
    x: Tensor,
    p_motion: Tensor,
    y_fast_last: Tensor,
    *,
    stable_prior: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reverse running-max fusion without materializing (H,W,T) mask tensors."""
    h, w, t = map(int, x.shape)
    running_max = x.new_zeros(h, w)
    stable_num = x.new_zeros(h, w)
    stable_den = x.new_zeros(h, w)
    for ti in range(t - 1, -1, -1):
        running_max = torch.maximum(running_max, p_motion[..., ti])
        valid_weight = 1.0 - running_max
        stable_num += valid_weight * x[..., ti]
        stable_den += valid_weight
    mean_stable = stable_num / stable_den.clamp(min=eps)
    w_mean = stable_den / (stable_den + max(float(stable_prior), eps))
    fused_last = w_mean * mean_stable + (1.0 - w_mean) * y_fast_last
    return fused_last, stable_num, stable_den, w_mean


def compute_motion_weights_parallel(
    x: Tensor,
    *,
    mu_ref: Tensor,
    fast_kernel: Tensor,
    stea_kernel: Tensor,
    photon_history: Tensor | None,
    d_history: Tensor | None,
    fast_window: int,
    motion_sharpness: float,
    motion_threshold: float,
    eps: float,
    cold_start: bool,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Compact parallel motion path: keep only p_motion, y_fast_last, d_hist tail."""
    h, w, t = map(int, x.shape)
    sum_hist = history_or_zeros(photon_history, h, w, fast_window - 1, x)
    k_fast = causal_window_sum_full(x, sum_hist, fast_window)
    d_raw = poisson_deviance(k_fast, mu_ref, eps=eps)
    del k_fast
    d_hist_tail = (
        d_raw[..., -int(stea_kernel.shape[2]) + 1 :].detach()
        if int(stea_kernel.shape[2]) > 1
        else None
    )
    d_smooth = smooth_deviance_full(d_raw, d_history, stea_kernel)
    del d_raw
    p_motion = motion_gate(d_smooth, motion_sharpness, motion_threshold)
    del d_smooth
    if cold_start:
        p_motion = p_motion * 0.0
    y_fast_last = causal_conv1d_full(x, fast_kernel, sum_hist)[..., -1].clamp(eps, 1.0 - eps)
    return p_motion, y_fast_last, d_hist_tail
