"""Detector-side SPAD feature plugins for multi-scale temporal experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

ATTN_PLUGIN_NAMES = frozenset({"temporal_attn", "sr_attn", "window_st_attn"})
ATTN_TEMPORAL_CORES = frozenset({"temporal_attn", "sr_attn", "window_st_attn"})
SSD_TEMPORAL_CORES = frozenset({"ssd"})


def uses_attn_temporal(*, plugin: str, temporal_core: str = "ssd") -> bool:
    """Return True when the active temporal core should read attn_* hyperparameters."""
    plugin = str(plugin).strip().lower()
    temporal_core = str(temporal_core).strip().lower()
    if plugin in ATTN_PLUGIN_NAMES:
        return True
    return plugin == "spatial_temporal" and temporal_core in ATTN_TEMPORAL_CORES


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


def _build_temporal_attention_core(
    core_name: str,
    *,
    in_dim: int,
    state_dim: int,
    head_dim: int,
    attn_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    kwargs = dict(attn_kwargs or {})
    if core_name == "temporal_attn":
        from ultralytics.quanta_stea_networks.attn import TemporalAttention

        return TemporalAttention(in_dim=in_dim, state_dim=state_dim, head_dim=head_dim, **kwargs)
    if core_name == "sr_attn":
        from ultralytics.quanta_stea_networks.attn import SRAttention

        return SRAttention(in_dim=in_dim, state_dim=state_dim, head_dim=head_dim, **kwargs)
    if core_name == "window_st_attn":
        from ultralytics.quanta_stea_networks.attn import WindowSTAttention

        return WindowSTAttention(in_dim=in_dim, state_dim=state_dim, head_dim=head_dim, **kwargs)
    raise ValueError(f"Unsupported temporal attention core: {core_name!r}")


class _TemporalCorePluginMixin:
    """Shared lifecycle hooks for temporal core plugins."""

    core: nn.Module
    online_mode: bool

    def set_online_mode(self, enabled: bool) -> None:
        self.online_mode = bool(enabled)

    def clear_temporal_state(self) -> None:
        self.core.clear_hidden_state()

    def set_bin_rate_hz(
        self,
        *,
        current_bin_rate_hz: float | None = None,
        reference_bin_rate_hz: float | None = None,
    ) -> None:
        self.core.set_bin_rate_hz(
            current_bin_rate_hz=current_bin_rate_hz,
            reference_bin_rate_hz=reference_bin_rate_hz,
        )


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
        self.online_mode = False

    def set_online_mode(self, enabled: bool) -> None:
        """Enable one-step online SSD updates that carry hidden state across forwards."""
        self.online_mode = bool(enabled)

    def clear_temporal_state(self) -> None:
        """Reset SSD hidden state before a new streaming sequence."""
        self.core.clear_hidden_state()

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
        if self.online_mode:
            if t != 1:
                raise ValueError(
                    f"TemporalSSDPlugin online mode expects exactly one timestep per forward, got T={t}"
                )
            if len(t_index_ll) != 1:
                raise ValueError(
                    f"TemporalSSDPlugin online mode expects one t_index, got {len(t_index_ll)}"
                )
            out = self.core.forward_online(x_flat[0], time_instant=float(t_index_ll[0]))
            out = out.reshape(b, h, w, c).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
            return out, list(t_index_ll)
        out, out_t_index_ll = self.core(x_flat, t_index_ll)
        out_t = out.shape[0]
        out = out.reshape(out_t, b, h, w, c).permute(0, 1, 4, 2, 3).contiguous()
        return out, out_t_index_ll


class TemporalAttentionPlugin(_TemporalCorePluginMixin, nn.Module):
    """Pure temporal causal attention plugin with SSD-compatible I/O."""

    def __init__(
        self,
        *,
        in_dim: int,
        state_dim: int,
        head_dim: int,
        attn_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.core = _build_temporal_attention_core(
            "temporal_attn",
            in_dim=in_dim,
            state_dim=state_dim,
            head_dim=head_dim,
            attn_kwargs=attn_kwargs,
        )
        self.online_mode = False

    def forward(self, x: torch.Tensor, t_index_ll: list[int]):
        if not torch.is_tensor(x) or x.ndim != 5:
            raise ValueError(f"TemporalAttentionPlugin expected T,B,C,H,W tensor, got {type(x)}")
        t, b, c, h, w = x.shape
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(t, b * h * w, c).contiguous()
        if self.online_mode:
            if t != 1:
                raise ValueError(
                    f"TemporalAttentionPlugin online mode expects exactly one timestep per forward, got T={t}"
                )
            if len(t_index_ll) != 1:
                raise ValueError(
                    f"TemporalAttentionPlugin online mode expects one t_index, got {len(t_index_ll)}"
                )
            out = self.core.forward_online(x_flat[0], time_instant=float(t_index_ll[0]))
            out = out.reshape(b, h, w, c).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
            return out, list(t_index_ll)
        out, out_t_index_ll = self.core(x_flat, t_index_ll)
        out_t = out.shape[0]
        out = out.reshape(out_t, b, h, w, c).permute(0, 1, 4, 2, 3).contiguous()
        return out, out_t_index_ll


class SRAttentionPlugin(_TemporalCorePluginMixin, nn.Module):
    """Spatial-reduction spatio-temporal attention plugin."""

    def __init__(
        self,
        *,
        in_dim: int,
        state_dim: int,
        head_dim: int,
        attn_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.core = _build_temporal_attention_core(
            "sr_attn",
            in_dim=in_dim,
            state_dim=state_dim,
            head_dim=head_dim,
            attn_kwargs=attn_kwargs,
        )
        self.online_mode = False

    def forward(self, x: torch.Tensor, t_index_ll: list[int]):
        if not torch.is_tensor(x) or x.ndim != 5:
            raise ValueError(f"SRAttentionPlugin expected T,B,C,H,W tensor, got {type(x)}")
        t, b, c, h, w = x.shape
        if self.online_mode:
            if t != 1:
                raise ValueError(f"SRAttentionPlugin online mode expects T=1, got T={t}")
            if len(t_index_ll) != 1:
                raise ValueError(f"SRAttentionPlugin online mode expects one t_index, got {len(t_index_ll)}")
            frame = x[0].contiguous()
            out = self.core.forward_online(frame, time_instant=float(t_index_ll[0]))
            return out.unsqueeze(0), list(t_index_ll)
        out, out_t_index_ll = self.core(x.contiguous(), t_index_ll)
        from ultralytics.quanta_stea_networks.attn import TBCHWMeta, unflatten_tbchw

        out = unflatten_tbchw(out, TBCHWMeta(t, b, c, h, w))
        return out, out_t_index_ll


class WindowSTAttentionPlugin(_TemporalCorePluginMixin, nn.Module):
    """Local window spatio-temporal attention plugin."""

    def __init__(
        self,
        *,
        in_dim: int,
        state_dim: int,
        head_dim: int,
        attn_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.core = _build_temporal_attention_core(
            "window_st_attn",
            in_dim=in_dim,
            state_dim=state_dim,
            head_dim=head_dim,
            attn_kwargs=attn_kwargs,
        )
        self.online_mode = False

    def forward(self, x: torch.Tensor, t_index_ll: list[int]):
        if not torch.is_tensor(x) or x.ndim != 5:
            raise ValueError(f"WindowSTAttentionPlugin expected T,B,C,H,W tensor, got {type(x)}")
        t, b, c, h, w = x.shape
        if self.online_mode:
            if t != 1:
                raise ValueError(f"WindowSTAttentionPlugin online mode expects T=1, got T={t}")
            if len(t_index_ll) != 1:
                raise ValueError(f"WindowSTAttentionPlugin online mode expects one t_index, got {len(t_index_ll)}")
            frame = x[0].contiguous()
            out = self.core.forward_online(frame, time_instant=float(t_index_ll[0]))
            return out.unsqueeze(0), list(t_index_ll)
        out, out_t_index_ll = self.core(x.contiguous(), t_index_ll)
        from ultralytics.quanta_stea_networks.attn import TBCHWMeta, unflatten_tbchw

        out = unflatten_tbchw(out, TBCHWMeta(t, b, c, h, w))
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
        attn_kwargs: dict[str, Any] | None = None,
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
        if temporal_core == "ssd":
            self.temporal = TemporalSSDPlugin(
                in_dim=hidden_dim,
                state_dim=state_dim,
                head_dim=_compatible_head_dim(hidden_dim, head_dim),
                ssd_kwargs=dict(ssd_kwargs or {}),
            )
        elif temporal_core in ATTN_TEMPORAL_CORES:
            plugin_cls = {
                "temporal_attn": TemporalAttentionPlugin,
                "sr_attn": SRAttentionPlugin,
                "window_st_attn": WindowSTAttentionPlugin,
            }[temporal_core]
            self.temporal = plugin_cls(
                in_dim=hidden_dim,
                state_dim=state_dim,
                head_dim=_compatible_head_dim(hidden_dim, head_dim),
                attn_kwargs=dict(attn_kwargs or {}),
            )
        else:
            raise ValueError(
                f"Unsupported temporal_core={temporal_core!r}; "
                f"expected one of: ssd, {', '.join(sorted(ATTN_TEMPORAL_CORES))}."
            )
        self.online_mode = False

    def set_online_mode(self, enabled: bool) -> None:
        self.online_mode = bool(enabled)
        self.temporal.set_online_mode(enabled)

    def clear_temporal_state(self) -> None:
        self.temporal.clear_temporal_state()

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
    attn_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Build a detector-side SPAD plugin."""
    name = str(name).strip().lower()
    ssd_kwargs = dict(ssd_kwargs or {})
    attn_kwargs = dict(attn_kwargs or {})
    if name == "none":
        return IdentityPlugin()
    if name == "temporal_ssd":
        return TemporalSSDPlugin(in_dim=in_dim, state_dim=state_dim, head_dim=head_dim, ssd_kwargs=ssd_kwargs)
    if name == "temporal_attn":
        return TemporalAttentionPlugin(
            in_dim=in_dim,
            state_dim=state_dim,
            head_dim=head_dim,
            attn_kwargs=attn_kwargs,
        )
    if name == "sr_attn":
        return SRAttentionPlugin(
            in_dim=in_dim,
            state_dim=state_dim,
            head_dim=head_dim,
            attn_kwargs=attn_kwargs,
        )
    if name == "window_st_attn":
        return WindowSTAttentionPlugin(
            in_dim=in_dim,
            state_dim=state_dim,
            head_dim=head_dim,
            attn_kwargs=attn_kwargs,
        )
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
            attn_kwargs=attn_kwargs,
        )
    raise ValueError(f"Unsupported detector plugin: {name!r}")
