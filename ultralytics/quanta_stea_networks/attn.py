"""Spatio-temporal attention cores for SPAD detector plugins."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

_CAUSAL_MASK_CACHE: dict[tuple[int, str, torch.dtype], Tensor] = {}


class TBCHWMeta(NamedTuple):
    t: int
    b: int
    c: int
    h: int
    w: int


def flatten_tbchw(x: Tensor) -> tuple[Tensor, TBCHWMeta]:
    """(T, B, C, H, W) -> (T, B*H*W, C)."""
    t, b, c, h, w = x.shape
    flat = rearrange(x, "t b c h w -> t (b h w) c")
    return flat.contiguous(), TBCHWMeta(t, b, c, h, w)


def unflatten_tbchw(x: Tensor, meta: TBCHWMeta) -> Tensor:
    """(T, B*H*W, C) -> (T, B, C, H, W)."""
    return rearrange(
        x,
        "t (b h w) c -> t b c h w",
        b=meta.b,
        h=meta.h,
        w=meta.w,
    ).contiguous()


def build_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Lower-triangular additive mask (T, T): future positions are -inf."""
    key = (seq_len, str(device), dtype)
    cached = _CAUSAL_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    _CAUSAL_MASK_CACHE[key] = mask
    return mask


def build_temporal_causal_mask(
    q_len: int,
    k_len: int,
    q_times: Tensor,
    k_times: Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build (q_len, k_len) mask where k_time > q_time is masked to -inf."""
    mask = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    invalid = k_times.unsqueeze(0) > q_times.unsqueeze(1)
    mask = mask.masked_fill(invalid, float("-inf"))
    return mask


def _split_dim_3way(dim: int) -> tuple[int, int, int]:
    """Split embedding dim across T/H/W axes (each gets an even count when possible)."""
    base = dim // 3
    rem = dim - base * 3
    dims = [base, base, base]
    for i in range(rem):
        dims[i] += 1
    return dims[0], dims[1], dims[2]


def _sinusoidal_1d(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if dim == 0:
        return torch.empty(length, 0, device=device, dtype=dtype)
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(dim, 1))
    )
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: (dim // 2)])
    return pe


class AbsolutePositionalEncoding3D(nn.Module):
    """Independent sinusoidal encodings for T, H, W axes, summed before use."""

    def __init__(self, dim: int, max_t: int = 512, max_h: int = 256, max_w: int = 256):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.max_t = int(max_t)
        self.max_h = int(max_h)
        self.max_w = int(max_w)
        dim_t, dim_h, dim_w = _split_dim_3way(self.dim)
        self.dim_t = dim_t
        self.dim_h = dim_h
        self.dim_w = dim_w
        pe_t = _sinusoidal_1d(self.max_t, dim_t, torch.device("cpu"), torch.float32)
        pe_h = _sinusoidal_1d(self.max_h, dim_h, torch.device("cpu"), torch.float32)
        pe_w = _sinusoidal_1d(self.max_w, dim_w, torch.device("cpu"), torch.float32)
        self.register_buffer("pe_t", pe_t, persistent=False)
        self.register_buffer("pe_h", pe_h, persistent=False)
        self.register_buffer("pe_w", pe_w, persistent=False)

    def _encode_tbchw(self, x: Tensor, *, t_start: int = 0) -> Tensor:
        t, b, c, h, w = x.shape
        device, dtype = x.device, x.dtype
        t_start = int(t_start)
        t_end = t_start + t
        if t_end > self.max_t:
            raise ValueError(f"t_start+t={t_end} exceeds max_t={self.max_t}")
        enc_t = self.pe_t[t_start:t_end, : self.dim_t].to(device=device, dtype=dtype)
        enc_h = self.pe_h[:h, : self.dim_h].to(device=device, dtype=dtype)
        enc_w = self.pe_w[:w, : self.dim_w].to(device=device, dtype=dtype)
        enc = torch.cat(
            [
                repeat(enc_t, "t dt -> t b h w dt", b=b, h=h, w=w),
                repeat(enc_h, "h dh -> t b h w dh", t=t, b=b, w=w),
                repeat(enc_w, "w dw -> t b h w dw", t=t, b=b, h=h),
            ],
            dim=-1,
        )
        if enc.shape[-1] < c:
            pad = torch.zeros(t, b, h, w, c - enc.shape[-1], device=device, dtype=dtype)
            enc = torch.cat([enc, pad], dim=-1)
        elif enc.shape[-1] > c:
            enc = enc[..., :c]
        return x + enc.permute(0, 1, 4, 2, 3)

    def _encode_t_only(self, x: Tensor) -> Tensor:
        """Add T-axis PE to (T, batch, C)."""
        t, batch, c = x.shape
        device, dtype = x.device, x.dtype
        enc_t = self.pe_t[:t, : min(self.dim_t, c)].to(device=device, dtype=dtype)
        enc = repeat(enc_t, "t dt -> t b dt", b=batch)
        if enc.shape[-1] < c:
            pad = torch.zeros(t, batch, c - enc.shape[-1], device=device, dtype=dtype)
            enc = torch.cat([enc, pad], dim=-1)
        out = x + enc
        return out

    def forward(
        self,
        x: Tensor,
        *,
        layout: str = "TBCHW",
        temporal_only: bool = False,
        t_start: int = 0,
    ) -> Tensor:
        if temporal_only or layout.upper() in {"TLBC", "TBC"}:
            if x.ndim != 3:
                raise ValueError(f"Expected (T, batch, C) for temporal-only PE, got shape={tuple(x.shape)}")
            if t_start != 0:
                raise ValueError("t_start is only supported for TBCHW positional encoding")
            return self._encode_t_only(x)
        if layout.upper() == "TBCHW":
            if x.ndim != 5:
                raise ValueError(f"Expected (T, B, C, H, W) for TBCHW PE, got shape={tuple(x.shape)}")
            return self._encode_tbchw(x, t_start=t_start)
        raise ValueError(f"Unsupported layout={layout!r}")


class TemporalCoreBase(nn.Module):
    """Shared lifecycle API aligned with SSD temporal cores."""

    in_dim: int
    reference_bin_rate_hz: float
    current_bin_rate_hz: float

    def forward(self, in_vector_ll: Tensor, t_index_ll: list[int]) -> tuple[Tensor, list[int]]:
        raise NotImplementedError

    def forward_online(self, in_vector: Tensor, time_instant: float) -> Tensor:
        raise NotImplementedError

    def clear_hidden_state(self) -> None:
        raise NotImplementedError

    def set_bin_rate_hz(
        self,
        *,
        current_bin_rate_hz: float | None = None,
        reference_bin_rate_hz: float | None = None,
    ) -> None:
        if reference_bin_rate_hz is not None:
            reference_bin_rate_hz = float(reference_bin_rate_hz)
            if reference_bin_rate_hz <= 0:
                raise ValueError(f"reference_bin_rate_hz must be positive, got {reference_bin_rate_hz}")
            self.reference_bin_rate_hz = reference_bin_rate_hz
        if current_bin_rate_hz is None:
            current_bin_rate_hz = self.reference_bin_rate_hz
        current_bin_rate_hz = float(current_bin_rate_hz)
        if current_bin_rate_hz <= 0:
            raise ValueError(f"current_bin_rate_hz must be positive, got {current_bin_rate_hz}")
        self.current_bin_rate_hz = current_bin_rate_hz


class TemporalAttention(TemporalCoreBase):
    """Pure temporal causal attention with per-pixel batching."""

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        head_dim: int,
        dropout: float = 0.0,
        reference_bin_rate_hz: float = 8000.0,
        current_bin_rate_hz: float | None = None,
    ):
        super().__init__()
        if in_dim % head_dim != 0:
            raise ValueError(f"in_dim={in_dim} must be divisible by head_dim={head_dim}")
        self.in_dim = int(in_dim)
        self.state_dim = int(state_dim)
        self.head_dim = int(head_dim)
        self.num_heads = self.in_dim // self.head_dim
        self.attn_dim = self.num_heads * self.state_dim
        self.dropout = float(dropout)
        self.reference_bin_rate_hz = float(reference_bin_rate_hz)
        self.current_bin_rate_hz = float(
            current_bin_rate_hz if current_bin_rate_hz is not None else reference_bin_rate_hz
        )

        self.norm = nn.LayerNorm(self.in_dim)
        self.pos_enc = AbsolutePositionalEncoding3D(self.in_dim)
        self.q_proj = nn.Linear(self.in_dim, self.attn_dim, bias=True)
        self.k_proj = nn.Linear(self.in_dim, self.attn_dim, bias=True)
        self.v_proj = nn.Linear(self.in_dim, self.attn_dim, bias=True)
        self.out_proj = nn.Linear(self.attn_dim, self.in_dim, bias=True)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        self._k_cache: Tensor | None = None
        self._v_cache: Tensor | None = None

    @staticmethod
    def spatial_to_embedding(tensor: Tensor) -> tuple[Tensor, tuple[int, int]]:
        height, width = tensor.shape[-2:]
        flat = rearrange(tensor, "... c h w -> ... (h w) c")
        return flat, (height, width)

    @staticmethod
    def embedding_to_spatial(tensor: Tensor, height: int, width: int) -> Tensor:
        return rearrange(tensor, "... (h w) c -> ... c h w", h=height, w=width)

    def _apply_temporal_pe_step(self, x: Tensor, time_index: int) -> Tensor:
        device, dtype = x.device, x.dtype
        enc_t = self.pos_enc.pe_t[time_index : time_index + 1, : self.pos_enc.dim_t].to(device=device, dtype=dtype)
        enc = repeat(enc_t, "1 dt -> b dt", b=x.shape[0])
        if enc.shape[-1] < self.in_dim:
            pad = torch.zeros(x.shape[0], self.in_dim - enc.shape[-1], device=device, dtype=dtype)
            enc = torch.cat([enc, pad], dim=-1)
        elif enc.shape[-1] > self.in_dim:
            enc = enc[..., : self.in_dim]
        return x + enc

    def clear_hidden_state(self) -> None:
        self._k_cache = None
        self._v_cache = None

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None) -> Tensor:
        out, _ = self.attn(q, k, v, attn_mask=attn_mask, need_weights=False)
        return self.out_proj(out)

    def forward(
        self,
        in_vector_ll: Tensor,
        t_index_ll: list[int],
    ) -> tuple[Tensor, list[int]]:
        is_spatial_input = False
        height = width = 0
        if in_vector_ll.ndim == 5:
            is_spatial_input = True
            t, b, c, h, w = in_vector_ll.shape
            in_vector_ll = rearrange(in_vector_ll, "t b c h w -> t (b h w) c")
            height, width = h, w

        if in_vector_ll.ndim != 3:
            raise ValueError(f"Expected (T, batch, C) input, got shape={tuple(in_vector_ll.shape)}")
        if in_vector_ll.shape[-1] != self.in_dim:
            raise ValueError(f"Expected in_dim={self.in_dim}, got {in_vector_ll.shape[-1]}")

        residual = in_vector_ll
        seq_len, batch, _ = in_vector_ll.shape
        x = self.norm(in_vector_ll)
        x = self.pos_enc(x, layout="TBC", temporal_only=True)
        q, k, v = self._project_qkv(x)
        q = rearrange(q, "t b d -> b t d")
        k = rearrange(k, "t b d -> b t d")
        v = rearrange(v, "t b d -> b t d")
        mask = build_causal_mask(seq_len, q.device, q.dtype)
        out = self._attend(q, k, v, mask)
        out = rearrange(out, "b t d -> t b d") + residual

        if is_spatial_input:
            out = rearrange(out, "t (b h w) c -> t b c h w", b=b, h=height, w=width)
        return out, list(t_index_ll)

    def forward_online(self, in_vector: Tensor, time_instant: float) -> Tensor:
        del time_instant
        is_spatial_input = False
        height = width = 0
        if in_vector.ndim == 4:
            is_spatial_input = True
            in_vector, (height, width) = self.spatial_to_embedding(in_vector)
        if in_vector.ndim != 2 or in_vector.shape[-1] != self.in_dim:
            raise ValueError(f"Expected (batch, in_dim) online input, got shape={tuple(in_vector.shape)}")

        residual = in_vector
        x = self.norm(in_vector)
        time_index = 0 if self._k_cache is None else self._k_cache.shape[1]
        x = self._apply_temporal_pe_step(x, time_index)
        q, k, v = self._project_qkv(x)
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        if self._k_cache is None:
            self._k_cache = k
            self._v_cache = v
        else:
            self._k_cache = torch.cat([self._k_cache, k], dim=1)
            self._v_cache = torch.cat([self._v_cache, v], dim=1)

        out = self._attend(q, self._k_cache, self._v_cache, attn_mask=None)
        out = out.squeeze(1) + residual
        if is_spatial_input:
            out = self.embedding_to_spatial(out, height, width)
        return out


class SRAttention(TemporalCoreBase):
    """Spatial-reduction spatio-temporal attention with temporal causality."""

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        head_dim: int,
        sr_ratio: int = 2,
        dropout: float = 0.0,
        reference_bin_rate_hz: float = 8000.0,
        current_bin_rate_hz: float | None = None,
    ):
        super().__init__()
        if in_dim % head_dim != 0:
            raise ValueError(f"in_dim={in_dim} must be divisible by head_dim={head_dim}")
        self.in_dim = int(in_dim)
        self.state_dim = int(state_dim)
        self.head_dim = int(head_dim)
        self.sr_ratio = max(int(sr_ratio), 1)
        self.num_heads = self.in_dim // self.head_dim
        self.attn_dim = self.num_heads * self.state_dim
        self.dropout = float(dropout)
        self.reference_bin_rate_hz = float(reference_bin_rate_hz)
        self.current_bin_rate_hz = float(
            current_bin_rate_hz if current_bin_rate_hz is not None else reference_bin_rate_hz
        )

        self.norm = nn.LayerNorm(self.in_dim)
        self.pos_enc = AbsolutePositionalEncoding3D(self.in_dim)
        self.sr_dw = nn.Conv2d(
            self.in_dim,
            self.in_dim,
            kernel_size=self.sr_ratio,
            stride=self.sr_ratio,
            groups=self.in_dim,
            bias=False,
        )
        self.q_proj = nn.Linear(self.in_dim, self.attn_dim, bias=True)
        self.k_proj = nn.Linear(self.in_dim, self.attn_dim, bias=True)
        self.v_proj = nn.Linear(self.in_dim, self.attn_dim, bias=True)
        self.out_proj = nn.Linear(self.attn_dim, self.in_dim, bias=True)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        # Online K/V cache layout: (B, Tc * H_down * W_down, attn_dim)
        self._k_cache: Tensor | None = None
        self._v_cache: Tensor | None = None
        self._cache_hw: tuple[int, int] | None = None  # downsampled (H', W')
        self._time_index = 0

    def clear_hidden_state(self) -> None:
        self._k_cache = None
        self._v_cache = None
        self._cache_hw = None
        self._time_index = 0

    def _to_tbchw(self, in_vector_ll: Tensor, meta: TBCHWMeta | None) -> tuple[Tensor, TBCHWMeta]:
        if in_vector_ll.ndim == 5:
            t, b, c, h, w = in_vector_ll.shape
            return in_vector_ll, TBCHWMeta(t, b, c, h, w)
        if in_vector_ll.ndim == 3 and meta is not None:
            return unflatten_tbchw(in_vector_ll, meta), meta
        raise ValueError("SRAttention requires (T,B,C,H,W) or (T,B*H*W,C) with meta.")

    def _build_sr_causal_mask(
        self,
        t: int,
        h: int,
        w: int,
        h_down: int,
        w_down: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        q_len = t * h * w
        k_len = t * h_down * w_down
        q_times = torch.arange(t, device=device).repeat_interleave(h * w)
        k_times = torch.arange(t, device=device).repeat_interleave(h_down * w_down)
        return build_temporal_causal_mask(q_len, k_len, q_times, k_times, device, dtype)

    def forward(
        self,
        in_vector_ll: Tensor,
        t_index_ll: list[int],
        *,
        meta: TBCHWMeta | None = None,
    ) -> tuple[Tensor, list[int]]:
        x5d, meta = self._to_tbchw(in_vector_ll, meta)
        t, b, c, h, w = meta.t, meta.b, meta.c, meta.h, meta.w
        residual, _ = flatten_tbchw(x5d)

        x = norm_tbchw(x5d, self.norm)
        x = self.pos_enc(x, layout="TBCHW", t_start=0)
        q = rearrange(x, "t b c h w -> b (t h w) c")
        q = self.q_proj(q)

        x_down = rearrange(x, "t b c h w -> (t b) c h w")
        x_down = self.sr_dw(x_down)
        _, _, h_down, w_down = x_down.shape
        x_down = rearrange(x_down, "(t b) c h w -> b t c h w", t=t, b=b)
        x_down = rearrange(x_down, "b t c h w -> b (t h w) c")
        k = self.k_proj(x_down)
        v = self.v_proj(x_down)

        mask = self._build_sr_causal_mask(t, h, w, h_down, w_down, q.device, q.dtype)
        out = self.out_proj(self.attn(q, k, v, attn_mask=mask, need_weights=False)[0])
        out = rearrange(out, "b (t h w) c -> t (b h w) c", t=t, h=h, w=w)
        out = out + residual
        return out, list(t_index_ll)

    def forward_online(self, in_vector: Tensor, time_instant: float) -> Tensor:
        """One-step streaming update with growing reduced-spatial K/V cache.

        Matches batch semantics for prefix ``0..t``: current-frame queries attend to
        all cached keys from times ``<= t`` (no future keys exist online, so no mask).
        """
        del time_instant
        if in_vector.ndim != 4 or in_vector.shape[1] != self.in_dim:
            raise ValueError(f"SRAttention online expects (B,C,H,W), got shape={tuple(in_vector.shape)}")
        b, c, h, w = in_vector.shape
        residual = in_vector
        time_index = self._time_index

        x5d = in_vector.unsqueeze(0)
        x = norm_tbchw(x5d, self.norm)
        x = self.pos_enc(x, layout="TBCHW", t_start=time_index)
        q = rearrange(x, "t b c h w -> b (t h w) c")
        q = self.q_proj(q)

        # Use norm+PE features for spatial reduction (same as batch), not raw input.
        x_down = self.sr_dw(x[0])
        h_down, w_down = x_down.shape[-2], x_down.shape[-1]
        x_down_seq = rearrange(x_down, "b c h w -> b (h w) c")
        k = self.k_proj(x_down_seq)
        v = self.v_proj(x_down_seq)

        if self._k_cache is None:
            self._k_cache = k
            self._v_cache = v
            self._cache_hw = (h_down, w_down)
        else:
            if self._cache_hw != (h_down, w_down):
                raise ValueError(
                    f"SRAttention online spatial size changed from {self._cache_hw} to {(h_down, w_down)}"
                )
            self._k_cache = torch.cat([self._k_cache, k], dim=1)
            self._v_cache = torch.cat([self._v_cache, v], dim=1)

        # Current queries only; cache holds times 0..t, so attention is already causal.
        out = self.out_proj(self.attn(q, self._k_cache, self._v_cache, attn_mask=None, need_weights=False)[0])
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        self._time_index = time_index + 1
        return out + residual


class RelativePositionBias3D(nn.Module):
    """Learnable relative position bias indexed by (dt, dx, dy)."""

    def __init__(self, window_size: tuple[int, int, int], num_heads: int):
        super().__init__()
        self.window_t, self.window_h, self.window_w = window_size
        self.num_heads = int(num_heads)
        self.bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_t - 1) * (2 * self.window_h - 1) * (2 * self.window_w - 1),
                self.num_heads,
            )
        )
        coords_t = torch.arange(self.window_t)
        coords_h = torch.arange(self.window_h)
        coords_w = torch.arange(self.window_w)
        coords = torch.stack(torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij"))
        coords = rearrange(coords, "d t h w -> d (t h w)")
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rearrange(rel, "d i j -> i j d")
        rel[..., 0] += self.window_t - 1
        rel[..., 1] += self.window_h - 1
        rel[..., 2] += self.window_w - 1
        rel_t = rel[..., 0] * (2 * self.window_h - 1) * (2 * self.window_w - 1)
        rel_h = rel[..., 1] * (2 * self.window_w - 1)
        rel_w = rel[..., 2]
        index = rel_t + rel_h + rel_w
        self.register_buffer("relative_position_index", index, persistent=False)

    def forward(self) -> Tensor:
        return self.bias_table[self.relative_position_index.view(-1)].view(
            self.window_t * self.window_h * self.window_w,
            self.window_t * self.window_h * self.window_w,
            self.num_heads,
        )

    def bias_from_deltas(self, dt: Tensor, dh: Tensor, dw: Tensor) -> Tensor:
        """Lookup per-head bias for relative offsets.

        Args:
            dt, dh, dw: Integer tensors of identical shape (broadcastable).

        Returns:
            Bias tensor with shape ``(*delta_shape, num_heads)``.
        """
        dt_i = dt + (self.window_t - 1)
        dh_i = dh + (self.window_h - 1)
        dw_i = dw + (self.window_w - 1)
        max_t = 2 * self.window_t - 1
        max_h = 2 * self.window_h - 1
        max_w = 2 * self.window_w - 1
        dt_i = dt_i.clamp(0, max_t - 1)
        dh_i = dh_i.clamp(0, max_h - 1)
        dw_i = dw_i.clamp(0, max_w - 1)
        index = dt_i * (max_h * max_w) + dh_i * max_w + dw_i
        return self.bias_table[index.long()]


class WindowSTAttention(TemporalCoreBase):
    """Causal sliding-window spatio-temporal attention with relative bias.

    For each query at time ``t``, keys are restricted to
    ``[max(0, t - window_t + 1), t]`` (frame 0 sees itself, frame 1 sees two
    frames, later frames see a full ``window_t`` history). Relative position
    bias is applied for every valid ``(dt, dx, dy)`` pair. Online inference
    keeps a rolling K/V cache of at most ``window_t`` frames with the same
    attention rule.
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int,
        head_dim: int,
        window_size: tuple[int, int, int] = (3, 7, 7),
        dropout: float = 0.0,
        reference_bin_rate_hz: float = 8000.0,
        current_bin_rate_hz: float | None = None,
    ):
        super().__init__()
        if in_dim % head_dim != 0:
            raise ValueError(f"in_dim={in_dim} must be divisible by head_dim={head_dim}")
        self.in_dim = int(in_dim)
        self.state_dim = int(state_dim)
        self.head_dim = int(head_dim)
        self.window_size = tuple(int(v) for v in window_size)
        self.window_t, self.window_h, self.window_w = self.window_size
        self.num_heads = self.in_dim // self.head_dim
        self.attn_dim = self.num_heads * self.state_dim
        self.dropout = float(dropout)
        self.reference_bin_rate_hz = float(reference_bin_rate_hz)
        self.current_bin_rate_hz = float(
            current_bin_rate_hz if current_bin_rate_hz is not None else reference_bin_rate_hz
        )

        self.norm = nn.LayerNorm(self.in_dim)
        self.pos_enc = AbsolutePositionalEncoding3D(self.in_dim)
        self.q_proj = nn.Linear(self.in_dim, self.attn_dim, bias=True)
        self.k_proj = nn.Linear(self.in_dim, self.attn_dim, bias=True)
        self.v_proj = nn.Linear(self.in_dim, self.attn_dim, bias=True)
        self.out_proj = nn.Linear(self.attn_dim, self.in_dim, bias=True)
        self.rel_pos_bias = RelativePositionBias3D(self.window_size, self.num_heads)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        self._k_cache: Tensor | None = None  # (n_windows, Tc, S, D)
        self._v_cache: Tensor | None = None
        self._cache_hw: tuple[int, int] | None = None
        self._time_index = 0

    def clear_hidden_state(self) -> None:
        self._k_cache = None
        self._v_cache = None
        self._cache_hw = None
        self._time_index = 0

    def _to_tbchw(self, in_vector_ll: Tensor, meta: TBCHWMeta | None) -> tuple[Tensor, TBCHWMeta]:
        if in_vector_ll.ndim == 5:
            t, b, c, h, w = in_vector_ll.shape
            return in_vector_ll, TBCHWMeta(t, b, c, h, w)
        if in_vector_ll.ndim == 3 and meta is not None:
            return unflatten_tbchw(in_vector_ll, meta), meta
        raise ValueError("WindowSTAttention requires (T,B,C,H,W) or (T,B*H*W,C) with meta.")

    def _pad_spatial(self, x: Tensor) -> tuple[Tensor, TBCHWMeta, tuple[int, int]]:
        """Pad H/W to multiples of the spatial window; do not pad time."""
        t, b, c, h, w = x.shape
        wh, ww = self.window_h, self.window_w
        pad_h = (wh - h % wh) % wh
        pad_w = (ww - w % ww) % ww
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0, 0, 0, 0, 0))
        t2, _, _, h2, w2 = x.shape
        return x, TBCHWMeta(t2, b, c, h2, w2), (pad_h, pad_w)

    def _spatial_token_coords(self, device: torch.device) -> tuple[Tensor, Tensor]:
        wh, ww = self.window_h, self.window_w
        ys = torch.arange(wh, device=device).repeat_interleave(ww)
        xs = torch.arange(ww, device=device).repeat(wh)
        return ys, xs

    def _build_sliding_attn_mask(
        self,
        n_q_frames: int,
        n_k_frames: int,
        q_time0: int,
        k_time0: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build ``(num_heads, Q, K)`` additive mask with causal local window + rel bias.

        Token layout is ``(time, spatial)`` with spatial = ``window_h * window_w``.
        ``q_time0`` / ``k_time0`` are absolute frame indices of the first query/key frame.
        """
        wh, ww = self.window_h, self.window_w
        s = wh * ww
        ys, xs = self._spatial_token_coords(device)

        q_t = (q_time0 + torch.arange(n_q_frames, device=device)).repeat_interleave(s)
        k_t = (k_time0 + torch.arange(n_k_frames, device=device)).repeat_interleave(s)
        q_y = ys.repeat(n_q_frames)
        q_x = xs.repeat(n_q_frames)
        k_y = ys.repeat(n_k_frames)
        k_x = xs.repeat(n_k_frames)

        dt = q_t.unsqueeze(1) - k_t.unsqueeze(0)
        dh = q_y.unsqueeze(1) - k_y.unsqueeze(0)
        dw = q_x.unsqueeze(1) - k_x.unsqueeze(0)
        invalid = (dt < 0) | (dt >= self.window_t)

        rel = self.rel_pos_bias.bias_from_deltas(dt, dh, dw).to(dtype=dtype)
        # (Q, K, heads) -> (heads, Q, K)
        mask = rel.permute(2, 0, 1).contiguous()
        mask = mask.masked_fill(invalid.unsqueeze(0), float("-inf"))
        return mask

    def _attend_with_rel_mask(self, q: Tensor, k: Tensor, v: Tensor, mask_heads: Tensor) -> Tensor:
        """Run MHA with per-head additive mask of shape ``(num_heads, Q, K)``."""
        n = q.shape[0]
        attn_mask = mask_heads.unsqueeze(0).expand(n, -1, -1, -1).reshape(
            n * self.num_heads, mask_heads.shape[-2], mask_heads.shape[-1]
        )
        out, _ = self.attn(q, k, v, attn_mask=attn_mask, need_weights=False)
        return self.out_proj(out)

    def _window_partition(self, x: Tensor) -> tuple[Tensor, int, int, int]:
        """(T,B,C,H,W) -> (B*hp*wp, T, wh*ww, C)."""
        t, b, c, h, w = x.shape
        wh, ww = self.window_h, self.window_w
        hp, wp = h // wh, w // ww
        x = rearrange(
            x,
            "t b c (hp wh) (wp ww) -> (b hp wp) t (wh ww) c",
            wh=wh,
            ww=ww,
            hp=hp,
            wp=wp,
        )
        return x, b, hp, wp

    def forward(
        self,
        in_vector_ll: Tensor,
        t_index_ll: list[int],
        *,
        meta: TBCHWMeta | None = None,
    ) -> tuple[Tensor, list[int]]:
        x5d, orig_meta = self._to_tbchw(in_vector_ll, meta)
        residual, _ = flatten_tbchw(x5d)
        x5d, meta, _pads = self._pad_spatial(x5d)
        t, b, c, h, w = meta.t, meta.b, meta.c, meta.h, meta.w
        wh, ww = self.window_h, self.window_w
        s = wh * ww

        x = norm_tbchw(x5d, self.norm)
        x = self.pos_enc(x, layout="TBCHW", t_start=0)
        x, _, hp, wp = self._window_partition(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "n t s d -> n (t s) d")
        k = rearrange(k, "n t s d -> n (t s) d")
        v = rearrange(v, "n t s d -> n (t s) d")

        mask = self._build_sliding_attn_mask(
            n_q_frames=t,
            n_k_frames=t,
            q_time0=0,
            k_time0=0,
            device=q.device,
            dtype=q.dtype,
        )
        out = self._attend_with_rel_mask(q, k, v, mask)
        out = rearrange(out, "n (t s) d -> n t s d", t=t, s=s)
        out = rearrange(
            out,
            "(b hp wp) t (wh ww) c -> t b c (hp wh) (wp ww)",
            b=b,
            hp=hp,
            wp=wp,
            wh=wh,
            ww=ww,
        )
        out = out[: orig_meta.t, :, :, : orig_meta.h, : orig_meta.w]
        out_flat, _ = flatten_tbchw(out)
        return out_flat + residual, list(t_index_ll)

    def forward_online(self, in_vector: Tensor, time_instant: float) -> Tensor:
        del time_instant  # PE / cache use monotonic _time_index from stream start
        if in_vector.ndim != 4:
            raise ValueError(f"WindowSTAttention online expects (B,C,H,W), got shape={tuple(in_vector.shape)}")
        b, c, h, w = in_vector.shape
        residual = in_vector
        x5d = in_vector.unsqueeze(0)
        x5d, meta, _pads = self._pad_spatial(x5d)
        _, _, _, h_pad, w_pad = meta.t, meta.b, meta.c, meta.h, meta.w
        wh, ww = self.window_h, self.window_w
        s = wh * ww
        time_index = self._time_index

        x = norm_tbchw(x5d, self.norm)
        x = self.pos_enc(x, layout="TBCHW", t_start=time_index)
        x, _, hp, wp = self._window_partition(x)  # (n, 1, S, C)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self._k_cache is None:
            self._k_cache = k
            self._v_cache = v
            self._cache_hw = (h_pad, w_pad)
        else:
            if self._cache_hw != (h_pad, w_pad):
                raise ValueError(
                    f"WindowSTAttention online spatial size changed from {self._cache_hw} to {(h_pad, w_pad)}"
                )
            self._k_cache = torch.cat([self._k_cache, k], dim=1)
            self._v_cache = torch.cat([self._v_cache, v], dim=1)
            if self._k_cache.shape[1] > self.window_t:
                self._k_cache = self._k_cache[:, -self.window_t :].contiguous()
                self._v_cache = self._v_cache[:, -self.window_t :].contiguous()

        tc = self._k_cache.shape[1]
        k_time0 = time_index - tc + 1
        q_flat = rearrange(q, "n t s d -> n (t s) d")
        k_flat = rearrange(self._k_cache, "n t s d -> n (t s) d")
        v_flat = rearrange(self._v_cache, "n t s d -> n (t s) d")
        mask = self._build_sliding_attn_mask(
            n_q_frames=1,
            n_k_frames=tc,
            q_time0=time_index,
            k_time0=k_time0,
            device=q.device,
            dtype=q.dtype,
        )
        out = self._attend_with_rel_mask(q_flat, k_flat, v_flat, mask)
        out = rearrange(out, "n (t s) d -> n t s d", t=1, s=s)
        out = rearrange(
            out,
            "(b hp wp) t (wh ww) c -> t b c (hp wh) (wp ww)",
            b=b,
            hp=hp,
            wp=wp,
            wh=wh,
            ww=ww,
        )
        out = out[0, :, :, :h, :w]
        self._time_index = time_index + 1
        return out + residual


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def norm_tbchw(x: Tensor, norm: nn.LayerNorm) -> Tensor:
    """Apply LayerNorm on the channel axis of (T, B, C, H, W)."""
    x = rearrange(x, "t b c h w -> t b h w c")
    x = norm(x)
    return rearrange(x, "t b h w c -> t b c h w")
