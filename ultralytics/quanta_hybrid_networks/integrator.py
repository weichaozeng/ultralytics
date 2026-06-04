"""Hybrid SPAD integrators with PerPixelBayesian-compatible streaming API."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ultralytics.quanta_neural_networks.ops.array_ops import torch_quantile
from ultralytics.quanta_neural_networks.ops.image import nearest_neighbor_inpaint


class GatedMultiScaleEMA(nn.Module):
    """
    White-box spatio-temporal integration for SPAD sensors.

    Parallel causal 1D convolutions compute multi-scale EMAs; Difference-of-EMAs
    drives RBF soft-routing across scales. Exposes the same ``process_photon_cube``
    contract as :class:`~ultralytics.quanta_neural_networks.integrator.PerPixelBayesian`.
    """

    def __init__(
        self,
        alphas: list[float] | None = None,
        chunk_size: int = 320,
        kernel_size: int = 64,
        v_threshold: float = 0.1,
        gating_sharpness: float = 20.0,
        subsampling: int = 1,
        hot_pixel_mask: np.ndarray | None = None,
        normalize: bool = False,
        quantile: float = 1.0,
        gating_tau: float = 0.1,
        spatial_batch_size: int = 16384,
    ):
        """
        :param alphas: Per-scale EMA decay rates (larger alpha = faster response; default fastest ≈10-bin FIR memory).
        :param chunk_size: Temporal block length for block-wise streaming integration.
        :param kernel_size: Causal FIR length used by ``conv1d`` (EMA memory horizon).
        :param v_threshold: DoE magnitude threshold for motion gating.
        :param gating_sharpness: Sigmoid sharpness on motion score.
        :param subsampling: Output temporal subsampling (same role as PerPixelBayesian).
        :param hot_pixel_mask: Optional hot-pixel mask for inpainting.
        :param normalize: If True, normalize reconstruction by quantile.
        :param quantile: Upper quantile used when ``normalize`` is True.
        :param gating_tau: RBF softmax temperature for scale routing.
        :param spatial_batch_size: Pixels per ``conv1d`` batch in ``_forward_fused`` (lower uses less VRAM).
        """
        super().__init__()

        if alphas is None:
            # Fastest α≈0.07 → ~10-bin half-mass memory at kernel_size=64 (was 0.5 ≈ 1 bin).
            alphas = [0.07, 0.05, 0.02, 0.01, 0.005]

        self.chunk_size = max(int(chunk_size), 1)
        self.kernel_size = max(int(kernel_size), 1)
        self.v_threshold = float(v_threshold)
        self.gating_sharpness = float(gating_sharpness)
        self.subsampling = max(int(subsampling), 1)
        self.hot_pixel_mask = hot_pixel_mask
        self.normalize = bool(normalize)
        self.quantile = float(quantile)
        self.gating_tau = float(gating_tau)
        self.spatial_batch_size = max(int(spatial_batch_size), 1)

        self.alphas = sorted(alphas, reverse=True)
        self.num_scales = len(self.alphas)

        centers = torch.linspace(1.0, 0.0, self.num_scales).view(1, self.num_scales, 1)
        self.register_buffer("channel_centers", centers)
        self._rebuild_ema_kernel()

        self.t_absolute = 0
        self._h, self._w, self._t = None, None, None
        self._stream_padding: Tensor | None = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(chunk_size={self.chunk_size}, kernel_size={self.kernel_size}, "
            f"subsampling={self.subsampling}, num_scales={self.num_scales}, v_threshold={self.v_threshold})"
        )

    def _rebuild_ema_kernel(self, device: torch.device | str | None = None) -> None:
        """Rebuild causal FIR kernels after ``kernel_size`` changes.

        Each scale uses ``alpha * (1-alpha)^lag`` (fast → recent-heavy, slow → flatter),
        then normalizes so taps sum to 1 within ``kernel_size`` (unit DC gain for 0/1 input).
        """
        kernel = torch.zeros(self.num_scales, 1, self.kernel_size, dtype=torch.float32)
        for m, alpha in enumerate(self.alphas):
            lags = torch.arange(self.kernel_size - 1, -1, -1, dtype=torch.float32)
            taps = alpha * torch.pow(1 - alpha, lags)
            kernel[m, 0, :] = taps / taps.sum().clamp(min=1e-12)
        if device is None and hasattr(self, "ema_kernel"):
            device = self.ema_kernel.device
        kernel = kernel.to(device or "cpu")
        if hasattr(self, "ema_kernel"):
            del self.ema_kernel
        self.register_buffer("ema_kernel", kernel)

    def update_hyperparams(self, **kwargs) -> None:
        """Dynamically update attributes; rebuild FIR kernels when ``kernel_size`` changes."""
        old_kernel_size = int(self.kernel_size)
        for name, value in kwargs.items():
            if hasattr(self, name) and value is not None:
                setattr(self, name, value)
        if int(self.kernel_size) != old_kernel_size:
            self.kernel_size = max(int(self.kernel_size), 1)
            self._rebuild_ema_kernel()
        if "chunk_size" in kwargs and kwargs["chunk_size"] is not None:
            self.chunk_size = max(int(self.chunk_size), 1)
        if "spatial_batch_size" in kwargs and kwargs["spatial_batch_size"] is not None:
            self.spatial_batch_size = max(int(self.spatial_batch_size), 1)

    def set_cube(self, photon_cube: Tensor) -> None:
        """Record input cube shape (mirrors PerPixelBayesian)."""
        self._h, self._w, self._t = map(int, photon_cube.shape)

    def clamp_recons(self, recons: Tensor) -> Tensor:
        """Clamp and optionally normalize reconstruction."""
        if recons.numel() == 0:
            return recons
        max_value = 1.0
        if self.normalize:
            max_value = torch_quantile(recons, self.quantile).clamp(min=1e-6)
        return (recons / max_value).clamp(0, 1)

    @staticmethod
    def _tail_padding(x: Tensor, x_padded: Tensor, *, pad_tail: int) -> Tensor:
        """Return the last ``pad_tail`` raw samples for cross-block FIR continuity."""
        if pad_tail <= 0:
            return x.new_zeros(x.shape[0], 1, 0)
        if x.shape[-1] >= pad_tail:
            return x[:, :, -pad_tail:].contiguous()
        if x.shape[-1] > 0:
            return x_padded[:, :, -pad_tail:].contiguous()
        return x.new_zeros(x.shape[0], 1, 0)

    @torch.no_grad()
    def _forward_fused(
        self,
        photon_cube: Tensor,
        last_chunk_padding: Tensor | None = None,
        spatial_batch_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Memory-efficient spatial-batched gated multi-scale EMA integration."""
        h, w, t = map(int, photon_cube.shape)
        num_pixels = h * w
        pad_left = self.kernel_size - 1
        batch_size = (
            self.spatial_batch_size
            if spatial_batch_size is None
            else max(int(spatial_batch_size), 1)
        )

        out_fused_flat = torch.empty((num_pixels, t), device=photon_cube.device, dtype=torch.float32)
        pad_len = pad_left if t > 0 else 0
        next_padding_out = (
            torch.empty((num_pixels, 1, pad_len), device=photon_cube.device, dtype=torch.float32)
            if pad_len > 0
            else None
        )

        x_flat = photon_cube.reshape(num_pixels, 1, t).float()
        pad_flat = (
            last_chunk_padding.view(num_pixels, 1, -1)
            if last_chunk_padding is not None and last_chunk_padding.numel()
            else None
        )

        for i in range(0, num_pixels, batch_size):
            end_i = min(i + batch_size, num_pixels)
            x_batch = x_flat[i:end_i]
            pad_batch = pad_flat[i:end_i] if pad_flat is not None else None

            if pad_batch is not None:
                x_padded = torch.cat([pad_batch, x_batch], dim=-1)
            elif pad_left > 0:
                x_padded = F.pad(x_batch, (pad_left, 0))
            else:
                x_padded = x_batch

            y_all = F.conv1d(x_padded, self.ema_kernel)

            y_fast = y_all[:, 0:1, :]
            y_slow = y_all[:, -1:, :]
            motion_score = torch.sigmoid(
                self.gating_sharpness * (torch.abs(y_fast - y_slow) - self.v_threshold)
            )

            distance_sq = torch.pow(motion_score - self.channel_centers, 2)
            weights = F.softmax(-distance_sq / self.gating_tau, dim=1)
            out_fused_flat[i:end_i] = torch.sum(weights * y_all, dim=1)

            if next_padding_out is not None:
                next_padding_out[i:end_i] = self._tail_padding(x_batch, x_padded, pad_tail=pad_left)

        return out_fused_flat.view(h, w, t), next_padding_out

    def _integrate_timeline(self, photon_cube: Tensor, *, carry_stream: bool) -> Tensor:
        """Integrate ``[H, W, T]`` in blocks of ``chunk_size``; carry ``kernel_size`` FIR padding."""
        h, w, t = map(int, photon_cube.shape)
        if t == 0:
            return photon_cube.new_zeros(h, w, 0)

        padding = self._stream_padding if carry_stream else None
        parts: list[Tensor] = []
        pos = 0
        while pos < t:
            end = min(pos + self.chunk_size, t)
            fused, padding = self._forward_fused(photon_cube[:, :, pos:end], padding)
            parts.append(fused)
            pos = end

        if carry_stream:
            self._stream_padding = padding
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _subsample_reconstruction(self, fused_hwt: Tensor) -> Tensor:
        """Apply PerPixelBayesian-style temporal subsampling to a dense timeline."""
        h, w, t = map(int, fused_hwt.shape)
        if t <= 0:
            return fused_hwt.new_zeros(h, w, 0)
        if t < self.subsampling:
            out = fused_hwt.new_zeros(h, w, 1)
            out[..., 0] = fused_hwt[..., -1]
            return out

        # Equivalent to (t_index + 1) % subsampling == 0 at t_index = subsampling-1, 2*subsampling-1, ...
        return fused_hwt[..., self.subsampling - 1 :: self.subsampling]

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
        kernel_size: int | None = None,
        v_threshold: float | None = None,
        gating_sharpness: float | None = None,
        gating_tau: float | None = None,
        **kwargs,
    ) -> Tensor:
        """
        Integrate a photon cube and return subsampled reconstruction ``[H, W, T']``.

        API and temporal subsampling match
        :meth:`ultralytics.quanta_neural_networks.integrator.PerPixelBayesian.process_photon_cube`.
        """
        del kwargs  # BOCPD-only kwargs ignored for hybrid integrator

        if clear_states:
            self.t_absolute = 0
            self._stream_padding = None

        self.update_hyperparams(
            subsampling=subsampling,
            hot_pixel_mask=hot_pixel_mask,
            normalize=normalize,
            quantile=quantile,
            chunk_size=chunk_size,
            kernel_size=kernel_size,
            v_threshold=v_threshold,
            gating_sharpness=gating_sharpness,
            gating_tau=gating_tau,
        )

        self.set_cube(photon_cube)
        fused = self._integrate_timeline(photon_cube, carry_stream=not clear_states)
        recons = self._subsample_reconstruction(fused)

        if self.hot_pixel_mask is not None:
            recons = nearest_neighbor_inpaint(recons, self.hot_pixel_mask)

        recons = self.clamp_recons(recons)
        self.t_absolute += self._t
        return recons

    def forward(
        self,
        photon_cube: Tensor,
        last_chunk_padding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Low-level forward on one segment (dense ``[H, W, T]``). Prefer ``process_photon_cube`` for QNN training.
        """
        return self._forward_fused(photon_cube, last_chunk_padding)
