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
    h, w, t = map(int, x_hwt.shape)
    hist = history_or_zeros(hist_hwt, h, w, window - 1, x_hwt)
    context = torch.cat([hist, x_hwt], dim=-1)
    sums = []
    for ti in range(t):
        sums.append(context[..., ti : ti + window].sum(dim=-1, keepdim=True))
    return torch.cat(sums, dim=-1)


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


def reverse_future_motion(p_motion_hwt: Tensor) -> Tensor:
    return torch.flip(torch.flip(p_motion_hwt, dims=(-1,)).cummax(dim=-1).values, dims=(-1,))
