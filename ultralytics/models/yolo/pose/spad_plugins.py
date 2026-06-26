"""Detector-side SPAD feature plugins for multi-scale temporal experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def _compatible_head_dim(channels: int, preferred: int) -> int:
    head_dim = max(min(int(preferred), int(channels)), 1)
    while channels % head_dim != 0 and head_dim > 1:
        head_dim -= 1
    return head_dim


def _flatten_tb(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
    t, b, c, h, w = x.shape
    return x.permute(1, 0, 2, 3, 4).reshape(b * t, c, h, w).contiguous(), (t, b, c, h, w)


def _unflatten_tb(x: torch.Tensor, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
    t, b, c, h, w = shape
    return x.reshape(b, t, c, h, w).permute(1, 0, 2, 3, 4).contiguous()


class IdentityPlugin(nn.Module):
    """No-op detector plugin."""

    def forward(self, x: torch.Tensor, t_index_ll: list[int]):
        return x, t_index_ll


class TemporalSSDPlugin(nn.Module):
    """Pure temporal SSD plugin that preserves the legacy behavior."""

    def __init__(self, *, in_dim: int, state_dim: int, head_dim: int, ssd_kwargs: dict[str, Any] | None = None):
        super().__init__()
        try:
            from ultralytics.quanta_neural_networks.ssd import SSD
        except ImportError as exc:
            raise ImportError(
                "TemporalSSDPlugin requires `ultralytics.quanta_neural_networks.ssd.SSD`, "
                "but that implementation is not present in the current code tree."
            ) from exc

        self.core = SSD(in_dim=in_dim, state_dim=state_dim, head_dim=head_dim, **dict(ssd_kwargs or {}))

    def set_bin_rate_hz(
        self,
        *,
        current_bin_rate_hz: float | None = None,
        reference_bin_rate_hz: float | None = None,
    ) -> None:
        """Update the detector-side SSD time base for runtime frequency changes."""
        self.core.set_bin_rate_hz(
            current_bin_rate_hz=current_bin_rate_hz,
            reference_bin_rate_hz=reference_bin_rate_hz,
        )

    def forward(self, x: torch.Tensor, t_index_ll: list[int]):
        if not torch.is_tensor(x) or x.ndim != 5:
            raise ValueError(f"TemporalSSDPlugin expected T,B,C,H,W tensor, got {type(x)}")
        t, b, c, h, w = x.shape
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(t, b * h * w, c).contiguous()
        out, out_t_index_ll = self.core(x_flat, t_index_ll)
        out_t = out.shape[0]
        out = out.reshape(out_t, b, h, w, c).permute(0, 1, 4, 2, 3).contiguous()
        return out, out_t_index_ll


class SpatialOnlyPlugin(nn.Module):
    """Lightweight spatial residual adapter."""

    def __init__(self, *, in_dim: int, reduce_ratio: int = 2, kernel_size: int = 3, alpha_init: float = 0.0):
        super().__init__()
        hidden_dim = max(int(in_dim // max(int(reduce_ratio), 1)), 1)
        padding = int(kernel_size) // 2
        self.reduce = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, groups=hidden_dim)
        self.act = nn.SiLU(inplace=True)
        self.project = nn.Conv2d(hidden_dim, in_dim, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor, t_index_ll: list[int]):
        flat, shape = _flatten_tb(x)
        delta = self.project(self.act(self.depthwise(self.reduce(flat))))
        out = flat + self.alpha * delta
        return _unflatten_tb(out, shape), t_index_ll


class SpatialTemporalPlugin(nn.Module):
    """Spatial adapter followed by a temporal SSD core and residual fusion."""

    def __init__(
        self,
        *,
        in_dim: int,
        state_dim: int,
        head_dim: int,
        reduce_ratio: int = 2,
        kernel_size: int = 3,
        alpha_init: float = 0.0,
        temporal_core: str = "ssd",
        ssd_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        hidden_dim = max(int(in_dim // max(int(reduce_ratio), 1)), 1)
        padding = int(kernel_size) // 2
        self.reduce = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, groups=hidden_dim)
        self.act = nn.SiLU(inplace=True)
        self.project = nn.Conv2d(hidden_dim, in_dim, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

        temporal_core = str(temporal_core).strip().lower()
        if temporal_core != "ssd":
            raise ValueError(f"Unsupported temporal_core={temporal_core!r}; only 'ssd' is implemented right now.")
        self.temporal = TemporalSSDPlugin(
            in_dim=hidden_dim,
            state_dim=state_dim,
            head_dim=_compatible_head_dim(hidden_dim, head_dim),
            ssd_kwargs=ssd_kwargs,
        )

    def set_bin_rate_hz(
        self,
        *,
        current_bin_rate_hz: float | None = None,
        reference_bin_rate_hz: float | None = None,
    ) -> None:
        """Update the temporal SSD time base while leaving the spatial adapter untouched."""
        self.temporal.set_bin_rate_hz(
            current_bin_rate_hz=current_bin_rate_hz,
            reference_bin_rate_hz=reference_bin_rate_hz,
        )

    def forward(self, x: torch.Tensor, t_index_ll: list[int]):
        residual = x
        flat, shape = _flatten_tb(x)
        spatial = self.act(self.depthwise(self.reduce(flat)))
        spatial = _unflatten_tb(spatial, (shape[0], shape[1], spatial.shape[1], shape[3], shape[4]))
        temporal, out_t_index_ll = self.temporal(spatial, t_index_ll)
        temporal_flat, _ = _flatten_tb(temporal)
        delta = self.project(temporal_flat)
        delta = _unflatten_tb(delta, (temporal.shape[0], temporal.shape[1], delta.shape[1], temporal.shape[3], temporal.shape[4]))
        if residual.shape[0] != delta.shape[0]:
            residual = residual[: delta.shape[0]]
        out = residual + self.alpha * delta
        return out, out_t_index_ll


def build_spad_plugin(
    name: str,
    *,
    in_dim: int,
    state_dim: int,
    head_dim: int,
    reduce_ratio: int = 2,
    kernel_size: int = 3,
    alpha_init: float = 0.0,
    temporal_core: str = "ssd",
    ssd_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Build a detector-side SPAD plugin."""
    name = str(name).strip().lower()
    if name == "none":
        return IdentityPlugin()
    if name == "temporal_ssd":
        return TemporalSSDPlugin(in_dim=in_dim, state_dim=state_dim, head_dim=head_dim, ssd_kwargs=ssd_kwargs)
    if name == "spatial_only":
        return SpatialOnlyPlugin(in_dim=in_dim, reduce_ratio=reduce_ratio, kernel_size=kernel_size, alpha_init=alpha_init)
    if name == "spatial_temporal":
        return SpatialTemporalPlugin(
            in_dim=in_dim,
            state_dim=state_dim,
            head_dim=head_dim,
            reduce_ratio=reduce_ratio,
            kernel_size=kernel_size,
            alpha_init=alpha_init,
            temporal_core=temporal_core,
            ssd_kwargs=ssd_kwargs,
        )
    raise ValueError(f"Unsupported detector plugin: {name!r}")
