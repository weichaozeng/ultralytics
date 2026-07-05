"""Lightweight frame-mode adapters for SPAD detector conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class IdentityFrameAdapter(nn.Module):
    """No-op image-space frame adapter."""

    def forward(self, x: Tensor, confidence: Tensor | None = None) -> Tensor:
        return x


class ResidualFrameAdapter(nn.Module):
    """Lightweight image-space residual adapter without uncertainty conditioning."""

    def __init__(self, in_channels: int = 3, kernel_size: int = 3, alpha_init: float = 0.0):
        super().__init__()
        padding = int(kernel_size) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.act = nn.SiLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, x: Tensor, confidence: Tensor | None = None) -> Tensor:
        delta = self.pointwise(self.act(self.depthwise(x)))
        return x + self.alpha * delta


class UncertaintyAwareFrameAdapter(nn.Module):
    """Lightweight image-space residual adapter gated by a single confidence map."""

    def __init__(self, in_channels: int = 3, kernel_size: int = 3, alpha_init: float = 0.0):
        super().__init__()
        padding = int(kernel_size) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.act = nn.SiLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gate_proj = nn.Conv2d(1, 1, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, x: Tensor, confidence: Tensor | None = None) -> Tensor:
        if confidence is None:
            raise ValueError("UncertaintyAwareFrameAdapter requires a conditioning confidence map.")
        if confidence.ndim != 4 or confidence.shape[1] != 1:
            raise ValueError(f"Expected confidence map (B,1,H,W), got shape={tuple(confidence.shape)}")
        delta = self.pointwise(self.act(self.depthwise(x)))
        gate = torch.sigmoid(self.gate_proj(confidence))
        return x + self.alpha * gate * delta


def build_frame_adapter(name: str, *, in_channels: int = 3, kernel_size: int = 3, alpha_init: float = 0.0) -> nn.Module:
    """Build a frame-mode adapter."""
    name = str(name).strip().lower()
    if name == "none":
        return IdentityFrameAdapter()
    if name == "residual":
        return ResidualFrameAdapter(in_channels=in_channels, kernel_size=kernel_size, alpha_init=alpha_init)
    if name == "ua":
        return UncertaintyAwareFrameAdapter(in_channels=in_channels, kernel_size=kernel_size, alpha_init=alpha_init)
    raise ValueError(f"Unsupported frame adapter: {name!r}")
