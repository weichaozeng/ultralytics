"""Motion-compensated SPAD integration utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VelIntegrator(nn.Module):
    """Integrate SPAD photon cubes with a simple velocity-compensated model.

    This first version estimates one global integer velocity from the previous two
    integrated chunks. The first two chunks use zero velocity. For each new chunk,
    frames are shifted toward the chunk-end reference before averaging.
    """

    def __init__(
        self,
        chunk_size: int = 320,
        max_shift: int = 16,
        estimate_downsample: int = 4,
        normalize: bool = False,
        quantile: float = 1.0,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.max_shift = int(max_shift)
        self.estimate_downsample = max(int(estimate_downsample), 1)
        self.normalize = bool(normalize)
        self.quantile = float(quantile)

        self.prev_frame_2: Tensor | None = None
        self.prev_frame_1: Tensor | None = None
        self.last_velocity: tuple[int, int] = (0, 0)

    def reset(self) -> None:
        """Clear streaming state."""
        self.prev_frame_2 = None
        self.prev_frame_1 = None
        self.last_velocity = (0, 0)

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

    def _estimate_velocity(self) -> tuple[int, int]:
        """Estimate global displacement from prev_frame_2 to prev_frame_1."""
        if self.prev_frame_2 is None or self.prev_frame_1 is None:
            return (0, 0)

        a = self.prev_frame_2.float()
        b = self.prev_frame_1.float()
        if self.estimate_downsample > 1:
            k = self.estimate_downsample
            a = F.avg_pool2d(a[None, None], kernel_size=k, stride=k).squeeze()
            b = F.avg_pool2d(b[None, None], kernel_size=k, stride=k).squeeze()

        best_score = None
        best = (0, 0)
        max_shift = max(self.max_shift // self.estimate_downsample, 0)
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                shifted = self._shift2d(a, dx, dy)
                score = torch.mean((shifted - b) ** 2)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (dx, dy)

        return (best[0] * self.estimate_downsample, best[1] * self.estimate_downsample)

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
        t = int(cube.shape[2])
        vx, vy = self._estimate_velocity()
        self.last_velocity = (vx, vy)

        accum = torch.zeros_like(cube[..., 0])
        denom = max(t - 1, 1)
        for ti in range(t):
            alpha = (t - 1 - ti) / denom
            dx = int(round(alpha * vx))
            dy = int(round(alpha * vy))
            accum += self._shift2d(cube[..., ti], dx, dy)

        integrated = accum / float(t)
        integrated = self._clamp_recons(integrated)

        self.prev_frame_2 = self.prev_frame_1
        self.prev_frame_1 = integrated.detach()
        return integrated[..., None]
