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

    Non-overlapping ``kernel_size`` bins are convolved (no cross-block padding) to one
    gated value per block; static regions average all blocks for denoising, while
    moving regions use the latest block only to avoid multi-block motion ghosts.
    Exposes the same ``process_photon_cube`` contract as
    :class:`~ultralytics.quanta_neural_networks.integrator.PerPixelBayesian`.
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
        min_filter_size: int = 7,
        peak_min_filter_size: int = 7,
    ):
        """
        :param alphas: Per-scale EMA decay rates (larger alpha = faster response; default fastest ≈10-bin FIR memory).
        :param chunk_size: Nominal raw window length (``det_spad`` slicing); integration tiles by ``kernel_size`` only.
        :param kernel_size: FIR length; each non-overlapping segment must contain exactly this many bins.
        :param v_threshold: DoE magnitude threshold for motion gating.
        :param gating_sharpness: Sigmoid sharpness on motion score.
        :param subsampling: Output temporal subsampling (same role as PerPixelBayesian).
        :param hot_pixel_mask: Optional hot-pixel mask for inpainting.
        :param normalize: If True, normalize reconstruction by quantile.
        :param quantile: Upper quantile used when ``normalize`` is True.
        :param gating_tau: RBF temperature for per-scale routing within each time block.
        :param spatial_batch_size: Pixels per ``conv1d`` batch (lower uses less VRAM).
        :param min_filter_size: Odd min-pool on per-block ``motion_score`` before scale routing (1 = off).
        :param peak_min_filter_size: Odd min-pool on chunk ``motion_peak`` before block mean/last blend (1 = off).
        """
        super().__init__()

        min_filter_size = max(int(min_filter_size), 1)
        peak_min_filter_size = max(int(peak_min_filter_size), 1)
        if min_filter_size % 2 == 0 or peak_min_filter_size % 2 == 0:
            raise ValueError(
                "min_filter_size and peak_min_filter_size must be odd (or 1 to disable min-pool)"
            )

        if alphas is None:
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
        self.min_filter_size = min_filter_size
        self.peak_min_filter_size = peak_min_filter_size

        self.alphas = sorted(alphas, reverse=True)
        self.num_scales = len(self.alphas)

        centers = torch.linspace(1.0, 0.0, self.num_scales).view(1, self.num_scales, 1)
        self.register_buffer("channel_centers", centers)
        self._rebuild_ema_kernel()

        self.t_absolute = 0
        self._h, self._w, self._t = None, None, None

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
        if "min_filter_size" in kwargs and kwargs["min_filter_size"] is not None:
            mfs = max(int(self.min_filter_size), 1)
            if mfs % 2 == 0:
                raise ValueError("min_filter_size must be odd (or 1 to disable min-pool)")
            self.min_filter_size = mfs
        if "peak_min_filter_size" in kwargs and kwargs["peak_min_filter_size"] is not None:
            pmfs = max(int(self.peak_min_filter_size), 1)
            if pmfs % 2 == 0:
                raise ValueError("peak_min_filter_size must be odd (or 1 to disable min-pool)")
            self.peak_min_filter_size = pmfs

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
    def min_pool2d(x: Tensor, kernel_size: int) -> Tensor:
        """2D min-pool on ``[H, W]`` (same implementation as PerPixelBayesian)."""
        x_batched = x.unsqueeze(0).unsqueeze(0)
        padding = (kernel_size - 1) // 2
        pooled = -F.max_pool2d(
            -x_batched, kernel_size=kernel_size, stride=1, padding=padding
        )
        return pooled.squeeze(0).squeeze(0)

    @torch.no_grad()
    def _gate_conv_block(
        self,
        photon_block: Tensor,
        spatial_batch_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Convolve one non-overlapping ``[H, W, kernel_size]`` segment (no padding).

        :return: ``fused [H, W]``, ``motion_score [H, W]`` (spatially min-pooled when configured).
        """
        h, w, t_seg = map(int, photon_block.shape)
        if t_seg != self.kernel_size:
            raise ValueError(
                f"Expected segment length kernel_size={self.kernel_size}, got T={t_seg}"
            )

        num_pixels = h * w
        batch_size = (
            self.spatial_batch_size
            if spatial_batch_size is None
            else max(int(spatial_batch_size), 1)
        )
        fused_flat = torch.empty((num_pixels,), device=photon_block.device, dtype=torch.float32)
        motion_raw = torch.empty((num_pixels,), device=photon_block.device, dtype=torch.float32)
        y_cache = torch.empty(
            (num_pixels, self.num_scales),
            device=photon_block.device,
            dtype=torch.float32,
        )
        x_flat = photon_block.reshape(num_pixels, 1, self.kernel_size).float()

        for i in range(0, num_pixels, batch_size):
            end_i = min(i + batch_size, num_pixels)
            x_batch = x_flat[i:end_i]
            y_all = F.conv1d(x_batch, self.ema_kernel)

            y_fast = y_all[:, 0:1, :]
            y_slow = y_all[:, -1:, :]
            motion_score = torch.sigmoid(
                self.gating_sharpness * (torch.abs(y_fast - y_slow) - self.v_threshold)
            )

            y_cache[i:end_i] = y_all.squeeze(-1)
            motion_raw[i:end_i] = motion_score.squeeze(-1).squeeze(-1)

        motion_map = motion_raw.view(h, w)
        if self.min_filter_size > 1:
            motion_map = self.min_pool2d(motion_map, self.min_filter_size)
        motion_flat = motion_map.reshape(num_pixels)

        for i in range(0, num_pixels, batch_size):
            end_i = min(i + batch_size, num_pixels)
            motion_batch = motion_flat[i:end_i].view(-1, 1, 1)
            y_batch = y_cache[i:end_i].unsqueeze(-1)

            distance_sq = torch.pow(motion_batch - self.channel_centers, 2)
            scale_weights = F.softmax(-distance_sq / self.gating_tau, dim=1)
            fused_flat[i:end_i] = torch.sum(scale_weights * y_batch, dim=1).squeeze(-1)

        return fused_flat.view(h, w), motion_map

    @torch.no_grad()
    def _integrate_blocks_with_motion(
        self, photon_cube: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Integrate ``[H, W, T]`` and return fused output plus per-pixel motion debug maps.

        Debug tensors (raw Bayer resolution ``[H, W]`` unless noted):

        - ``motion_blocks``: block ``motion_score`` after scale min-pool, ``[H, W, B]``
        - ``motion_peak_raw``: ``max`` over blocks before peak min-pool
        - ``motion_peak``: peak score after ``peak_min_filter_size`` min-pool
        - ``motion_blend``: chunk-level blend toward last block, ``[H, W]`` in ``[0, 1]``
        """
        h, w, t = map(int, photon_cube.shape)
        ks = self.kernel_size
        n_blocks = t // ks
        empty_debug = {
            "motion_blocks": photon_cube.new_zeros(h, w, 0),
            "motion_peak_raw": photon_cube.new_zeros(h, w),
            "motion_peak": photon_cube.new_zeros(h, w),
            "motion_blend": photon_cube.new_zeros(h, w),
        }
        if n_blocks == 0:
            return photon_cube.new_zeros(h, w, 1 if t > 0 else 0), empty_debug

        fused_stack = []
        motion_stack = []
        for b in range(n_blocks):
            seg = photon_cube[:, :, b * ks : (b + 1) * ks]
            fused_b, motion_b = self._gate_conv_block(seg)
            fused_stack.append(fused_b)
            motion_stack.append(motion_b)

        fused_hwb = torch.stack(fused_stack, dim=-1)
        motion_hwb = torch.stack(motion_stack, dim=-1)

        if n_blocks == 1:
            motion_peak_raw = motion_hwb[..., 0]
            motion_peak = motion_peak_raw
            if self.peak_min_filter_size > 1:
                motion_peak = self.min_pool2d(motion_peak, self.peak_min_filter_size)
            motion_blend = torch.sigmoid(
                self.gating_sharpness * (motion_peak - self.v_threshold)
            )
            debug = {
                "motion_blocks": motion_hwb,
                "motion_peak_raw": motion_peak_raw,
                "motion_peak": motion_peak,
                "motion_blend": motion_blend,
            }
            return fused_hwb[..., -1:], debug

        fused_mean = fused_hwb.mean(dim=-1, keepdim=True)
        fused_last = fused_hwb[..., -1:]
        motion_peak_raw = motion_hwb.max(dim=-1).values
        motion_peak = motion_peak_raw
        if self.peak_min_filter_size > 1:
            motion_peak = self.min_pool2d(motion_peak, self.peak_min_filter_size)
        motion_blend = torch.sigmoid(
            self.gating_sharpness * (motion_peak.unsqueeze(-1) - self.v_threshold)
        ).squeeze(-1)
        out = (1.0 - motion_blend.unsqueeze(-1)) * fused_mean + motion_blend.unsqueeze(-1) * fused_last
        debug = {
            "motion_blocks": motion_hwb,
            "motion_peak_raw": motion_peak_raw,
            "motion_peak": motion_peak,
            "motion_blend": motion_blend,
        }
        return out, debug

    @torch.no_grad()
    def _integrate_blocks(self, photon_cube: Tensor) -> Tensor:
        """
        Tile ``[H, W, T]`` into non-overlapping ``kernel_size`` segments; aggregate block outputs.

        Each block: ``kernel_size``-bin conv → per-pixel gated ``fused_b`` and ``motion_score_b``.
        Chunk output blends block means vs. the latest block (per pixel):

        - low peak ``motion_score`` across blocks → ``mean(fused_b)`` (denoise, all windows contribute)
        - high peak motion → ``fused_{B-1}`` only (avoid ghosting from misaligned block snapshots)

        ``motion_peak`` is min-pooled spatially (``peak_min_filter_size``) before the blend, analogous to PPB
        runlength pooling: static neighbours pull pixels toward block averaging.

        Remainder bins ``T % kernel_size`` are ignored (no padding).
        """
        fused, _ = self._integrate_blocks_with_motion(photon_cube)
        return fused

    def _subsample_reconstruction(self, fused_hwt: Tensor) -> Tensor:
        """Apply PerPixelBayesian-style temporal subsampling to a dense timeline."""
        h, w, t = map(int, fused_hwt.shape)
        if t <= 0:
            return fused_hwt.new_zeros(h, w, 0)
        if t < self.subsampling:
            out = fused_hwt.new_zeros(h, w, 1)
            out[..., 0] = fused_hwt[..., -1]
            return out

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
        min_filter_size: int | None = None,
        peak_min_filter_size: int | None = None,
        **kwargs,
    ) -> Tensor:
        """
        Integrate a photon cube and return subsampled reconstruction ``[H, W, T']``.

        Each call is stateless across chunks (``clear_states`` only resets ``t_absolute``).
        """
        del kwargs  # BOCPD-only kwargs ignored for hybrid integrator

        if clear_states:
            self.t_absolute = 0

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
            min_filter_size=min_filter_size,
            peak_min_filter_size=peak_min_filter_size,
        )

        self.set_cube(photon_cube)
        fused = self._integrate_blocks(photon_cube)
        recons = self._subsample_reconstruction(fused)

        if self.hot_pixel_mask is not None:
            recons = nearest_neighbor_inpaint(recons, self.hot_pixel_mask)

        recons = self.clamp_recons(recons)
        self.t_absolute += self._t
        return recons

    @torch.no_grad()
    def process_photon_cube_with_motion(
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
        min_filter_size: int | None = None,
        peak_min_filter_size: int | None = None,
        **kwargs,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Like :meth:`process_photon_cube`, also returning motion debug maps at raw resolution.

        See :meth:`_integrate_blocks_with_motion` for debug tensor keys.
        """
        del kwargs

        if clear_states:
            self.t_absolute = 0

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
            min_filter_size=min_filter_size,
            peak_min_filter_size=peak_min_filter_size,
        )

        self.set_cube(photon_cube)
        fused, motion_debug = self._integrate_blocks_with_motion(photon_cube)
        recons = self._subsample_reconstruction(fused)

        if self.hot_pixel_mask is not None:
            recons = nearest_neighbor_inpaint(recons, self.hot_pixel_mask)

        recons = self.clamp_recons(recons)
        self.t_absolute += self._t
        return recons, motion_debug

    def forward(self, photon_cube: Tensor) -> Tensor:
        """Low-level forward: block-tiled integration → ``[H, W, 1]`` (or ``[H,W,0]``)."""
        return self._integrate_blocks(photon_cube)
