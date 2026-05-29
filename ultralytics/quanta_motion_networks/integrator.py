"""Motion-compensated SPAD integration utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VelIntegrator(nn.Module):
    """Integrate SPAD photon cubes with velocity-compensated averaging.

    Velocity is estimated from the previous two integrated chunks via block
    matching (min MSE). With ``patch_size <= 0``, one global shift is used for
    the whole frame; otherwise the frame is tiled into non-overlapping patches
    and each patch gets its own shift (batched in parallel). The first two
    chunks use zero velocity. Within a chunk, frames are shifted toward the
    chunk-end reference before averaging.
    """

    def __init__(
        self,
        chunk_size: int = 320,
        max_shift: int = 16,
        estimate_downsample: int = 4,
        patch_size: int = 0,
        normalize: bool = False,
        quantile: float = 1.0,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.max_shift = int(max_shift)
        self.estimate_downsample = max(int(estimate_downsample), 1)
        self.patch_size = int(patch_size)
        self.normalize = bool(normalize)
        self.quantile = float(quantile)

        self.prev_frame_2: Tensor | None = None
        self.prev_frame_1: Tensor | None = None
        self.last_velocity: tuple[int, int] = (0, 0)
        self.last_velocity_vx: Tensor | None = None
        self.last_velocity_vy: Tensor | None = None

    @property
    def local_patches(self) -> bool:
        return self.patch_size > 0

    def reset(self) -> None:
        """Clear streaming state."""
        self.prev_frame_2 = None
        self.prev_frame_1 = None
        self.last_velocity = (0, 0)
        self.last_velocity_vx = None
        self.last_velocity_vy = None

    @staticmethod
    def _pad_to_multiple(x: Tensor, block: int) -> Tensor:
        h, w = x.shape
        ph = ((h + block - 1) // block) * block
        pw = ((w + block - 1) // block) * block
        if ph == h and pw == w:
            return x
        return F.pad(x, (0, pw - w, 0, ph - h))

    @staticmethod
    def _image_to_patches(x: Tensor, patch_size: int) -> tuple[Tensor, int, int]:
        x = VelIntegrator._pad_to_multiple(x, patch_size)
        h, w = x.shape
        nh, nw = h // patch_size, w // patch_size
        patches = x.view(nh, patch_size, nw, patch_size).permute(0, 2, 1, 3).reshape(nh * nw, patch_size, patch_size)
        return patches, nh, nw

    @staticmethod
    def _patches_to_image(patches: Tensor, nh: int, nw: int, patch_size: int, h: int, w: int) -> Tensor:
        img = patches.view(nh, nw, patch_size, patch_size).permute(0, 2, 1, 3).reshape(nh * patch_size, nw * patch_size)
        return img[:h, :w]

    @staticmethod
    def _shift2d(x: Tensor, dx: int, dy: int) -> Tensor:
        """Shift a 2D tensor with zero padding instead of wraparound."""
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape={tuple(x.shape)}")
        h, w = x.shape
        out = torch.zeros_like(x)

        src_x0 = max(-dx, 0)
        src_x1 = min(w - dx, w)
        dst_x0 = max(dx, 0)
        dst_x1 = min(w + dx, w)

        src_y0 = max(-dy, 0)
        src_y1 = min(h - dy, h)
        dst_y0 = max(dy, 0)
        dst_y1 = min(h + dy, h)

        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return out
        out[dst_y0:dst_y1, dst_x0:dst_x1] = x[src_y0:src_y1, src_x0:src_x1]
        return out

    @staticmethod
    def _shift2d_batch_multi(patches: Tensor, dx: Tensor, dy: Tensor) -> Tensor:
        """Shift a batch of 2D patches; ``dx``/``dy`` are length-B integer tensors."""
        out = torch.zeros_like(patches)
        for i in range(patches.shape[0]):
            out[i] = VelIntegrator._shift2d(patches[i], int(dx[i].item()), int(dy[i].item()))
        return out

    @staticmethod
    def _block_match_patches(patches_a: Tensor, patches_b: Tensor, max_shift: int, scale: int) -> tuple[Tensor, Tensor]:
        """Return per-patch integer shifts (vx, vy) at full resolution."""
        device = patches_a.device
        n_patches = patches_a.shape[0]
        best_score = torch.full((n_patches,), float("inf"), device=device)
        best_vx = torch.zeros(n_patches, dtype=torch.long, device=device)
        best_vy = torch.zeros(n_patches, dtype=torch.long, device=device)
        max_shift = max(int(max_shift), 0)

        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                shifted = VelIntegrator._shift2d_batch_multi(
                    patches_a,
                    torch.full((n_patches,), dx, dtype=torch.long, device=device),
                    torch.full((n_patches,), dy, dtype=torch.long, device=device),
                )
                score = (shifted - patches_b).pow(2).mean(dim=(-2, -1))
                better = score < best_score
                best_score = torch.where(better, score, best_score)
                best_vx = torch.where(better, torch.tensor(dx, device=device), best_vx)
                best_vy = torch.where(better, torch.tensor(dy, device=device), best_vy)

        return best_vx * scale, best_vy * scale

    def _prepare_estimate_frames(self) -> tuple[Tensor, Tensor, int] | None:
        if self.prev_frame_2 is None or self.prev_frame_1 is None:
            return None
        k = self.estimate_downsample
        a = self.prev_frame_2.float()
        b = self.prev_frame_1.float()
        if k > 1:
            a = F.avg_pool2d(a[None, None], kernel_size=k, stride=k).squeeze()
            b = F.avg_pool2d(b[None, None], kernel_size=k, stride=k).squeeze()
        max_shift = max(self.max_shift // k, 0)
        return a, b, max_shift

    def _estimate_velocity_global(self) -> tuple[int, int]:
        prepared = self._prepare_estimate_frames()
        if prepared is None:
            return (0, 0)
        a, b, max_shift = prepared
        k = self.estimate_downsample
        patches_a = a.unsqueeze(0)
        patches_b = b.unsqueeze(0)
        vx, vy = self._block_match_patches(patches_a, patches_b, max_shift, k)
        return int(vx[0].item()), int(vy[0].item())

    def _estimate_velocity_local(self, h: int, w: int) -> tuple[Tensor, Tensor]:
        k = self.estimate_downsample
        ps = max(self.patch_size // k, 4)
        nh = ((h + self.patch_size - 1) // self.patch_size)
        nw = ((w + self.patch_size - 1) // self.patch_size)
        device = self.prev_frame_1.device if self.prev_frame_1 is not None else torch.device("cpu")
        zero_vx = torch.zeros((nh, nw), dtype=torch.long, device=device)
        zero_vy = torch.zeros((nh, nw), dtype=torch.long, device=device)

        prepared = self._prepare_estimate_frames()
        if prepared is None:
            return zero_vx, zero_vy
        a, b, max_shift = prepared
        patches_a, nh, nw = self._image_to_patches(a, ps)
        patches_b, _, _ = self._image_to_patches(b, ps)
        vx, vy = self._block_match_patches(patches_a, patches_b, max_shift, k)
        return vx.view(nh, nw), vy.view(nh, nw)

    def _integrate_global(self, cube: Tensor, vx: int, vy: int) -> Tensor:
        t = int(cube.shape[2])
        accum = torch.zeros_like(cube[..., 0])
        denom = max(t - 1, 1)
        for ti in range(t):
            alpha = (t - 1 - ti) / denom
            dx = int(round(alpha * vx))
            dy = int(round(alpha * vy))
            accum += self._shift2d(cube[..., ti], dx, dy)
        return accum / float(t)

    def _integrate_local(self, cube: Tensor, vx_grid: Tensor, vy_grid: Tensor) -> Tensor:
        h, w = cube.shape[:2]
        ps = self.patch_size
        t = int(cube.shape[2])
        nh, nw = vx_grid.shape
        accum = torch.zeros(h, w, device=cube.device, dtype=cube.dtype)
        denom = max(t - 1, 1)
        vx_flat = vx_grid.reshape(-1)
        vy_flat = vy_grid.reshape(-1)

        for ti in range(t):
            alpha = (t - 1 - ti) / denom
            frame = cube[..., ti]
            padded = self._pad_to_multiple(frame, ps)
            patches, _, _ = self._image_to_patches(padded, ps)
            dx = (alpha * vx_flat.float()).round().long()
            dy = (alpha * vy_flat.float()).round().long()
            shifted = self._shift2d_batch_multi(patches, dx, dy)
            accum += self._patches_to_image(shifted, nh, nw, ps, h, w)

        return accum / float(t)

    def _clamp_recons(self, recons: Tensor) -> Tensor:
        if recons.numel() == 0:
            return recons
        if self.normalize:
            scale = torch.quantile(recons.reshape(-1), self.quantile).clamp(min=1e-6)
            recons = recons / scale
        return recons.clamp(0, 1)

    @torch.no_grad()
    def process_photon_cube(self, photon_cube: Tensor, clear_states: bool = False, **kwargs) -> Tensor:
        """Process one photon cube.

        Args:
            photon_cube: Boolean-like tensor shaped `(H, W, T)`.
            clear_states: If True, reset previous chunk history.

        Returns:
            Tensor shaped `(H, W, 1)` containing one integrated frame.
        """
        if clear_states:
            self.reset()
        if photon_cube.ndim != 3:
            raise ValueError(f"Expected photon cube (H,W,T), got shape={tuple(photon_cube.shape)}")
        if photon_cube.shape[2] <= 0:
            return torch.zeros((*photon_cube.shape[:2], 0), device=photon_cube.device, dtype=torch.float32)

        if "normalize" in kwargs and kwargs["normalize"] is not None:
            self.normalize = bool(kwargs["normalize"])
        if "quantile" in kwargs and kwargs["quantile"] is not None:
            self.quantile = float(kwargs["quantile"])

        cube = photon_cube.float()
        h, w = int(cube.shape[0]), int(cube.shape[1])

        if self.local_patches:
            vx_grid, vy_grid = self._estimate_velocity_local(h, w)
            self.last_velocity_vx = vx_grid
            self.last_velocity_vy = vy_grid
            self.last_velocity = (int(vx_grid.float().mean().item()), int(vy_grid.float().mean().item()))
            integrated = self._integrate_local(cube, vx_grid, vy_grid)
        else:
            vx, vy = self._estimate_velocity_global()
            self.last_velocity = (vx, vy)
            self.last_velocity_vx = None
            self.last_velocity_vy = None
            integrated = self._integrate_global(cube, vx, vy)

        integrated = self._clamp_recons(integrated)

        self.prev_frame_2 = self.prev_frame_1
        self.prev_frame_1 = integrated.detach()
        return integrated[..., None]
