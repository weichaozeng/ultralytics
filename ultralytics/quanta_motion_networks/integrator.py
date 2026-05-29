"""Motion-compensated SPAD integration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class _DetSnapshot:
    """Track centers in full-resolution raw (Bayer) coordinates."""

    ids: np.ndarray
    centers: np.ndarray  # (N, 2) as (x, y)


class VelIntegrator(nn.Module):
    """Integrate SPAD photon cubes with detection-guided motion compensation.

    Velocity is estimated from the previous two detection frames: for each track ID
    present in both, displacement between box centers defines motion. The median
    displacement (global) or per-patch median among tracks inside each patch is
    used. The first two chunks use zero velocity.

    Call :meth:`push_detection` after running the detector on each integrated frame
    before processing the next chunk.
    """

    def __init__(
        self,
        chunk_size: int = 320,
        max_shift: int = 16,
        patch_size: int = 0,
        normalize: bool = False,
        quantile: float = 1.0,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.max_shift = int(max_shift)
        self.patch_size = int(patch_size)
        self.normalize = bool(normalize)
        self.quantile = float(quantile)

        self.prev_det_2: _DetSnapshot | None = None
        self.prev_det_1: _DetSnapshot | None = None
        self.last_velocity: tuple[int, int] = (0, 0)
        self.last_velocity_vx: Tensor | None = None
        self.last_velocity_vy: Tensor | None = None

    @property
    def local_patches(self) -> bool:
        return self.patch_size > 0

    def reset(self) -> None:
        """Clear streaming state."""
        self.prev_det_2 = None
        self.prev_det_1 = None
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
        return torch.nn.functional.pad(x, (0, pw - w, 0, ph - h))

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
        out = torch.zeros_like(patches)
        for i in range(patches.shape[0]):
            out[i] = VelIntegrator._shift2d(patches[i], int(dx[i].item()), int(dy[i].item()))
        return out

    @staticmethod
    def _match_displacements(prev2: _DetSnapshot, prev1: _DetSnapshot) -> np.ndarray:
        """Return (M, 2) displacements for track IDs present in both snapshots."""
        id_to_center = {int(tid): center for tid, center in zip(prev2.ids, prev2.centers)}
        disps = []
        for tid, center in zip(prev1.ids, prev1.centers):
            key = int(tid)
            if key in id_to_center:
                disps.append(center - id_to_center[key])
        if not disps:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack(disps, axis=0).astype(np.float32)

    def _clamp_shift(self, dx: float, dy: float) -> tuple[int, int]:
        ms = max(int(self.max_shift), 0)
        vx = int(np.clip(round(dx), -ms, ms))
        vy = int(np.clip(round(dy), -ms, ms))
        return vx, vy

    def _aggregate_displacement(self, disps: np.ndarray) -> tuple[int, int]:
        if disps.size == 0:
            return (0, 0)
        d = np.median(disps, axis=0)
        return self._clamp_shift(float(d[0]), float(d[1]))

    def _estimate_velocity_global(self) -> tuple[int, int]:
        if self.prev_det_2 is None or self.prev_det_1 is None:
            return (0, 0)
        disps = self._match_displacements(self.prev_det_2, self.prev_det_1)
        return self._aggregate_displacement(disps)

    def _estimate_velocity_local(self, h: int, w: int) -> tuple[Tensor, Tensor]:
        ps = self.patch_size
        nh = (h + ps - 1) // ps
        nw = (w + ps - 1) // ps
        device = torch.device("cpu")
        vx_grid = torch.zeros((nh, nw), dtype=torch.long, device=device)
        vy_grid = torch.zeros((nh, nw), dtype=torch.long, device=device)

        if self.prev_det_2 is None or self.prev_det_1 is None:
            return vx_grid, vy_grid

        id_to_disp: dict[int, np.ndarray] = {}
        for tid, center in zip(self.prev_det_1.ids, self.prev_det_1.centers):
            key = int(tid)
            prev_center = self._center_for_id(self.prev_det_2, key)
            if prev_center is not None:
                id_to_disp[key] = center - prev_center
        if not id_to_disp:
            return vx_grid, vy_grid

        all_disps = np.stack(list(id_to_disp.values()), axis=0)
        global_vx, global_vy = self._aggregate_displacement(all_disps)

        for pi in range(nh):
            for pj in range(nw):
                y0, x0 = pi * ps, pj * ps
                y1, x1 = min(y0 + ps, h), min(x0 + ps, w)
                patch_disps = []
                for tid, disp in id_to_disp.items():
                    center = self._center_for_id(self.prev_det_1, tid)
                    if center is None:
                        continue
                    cx, cy = float(center[0]), float(center[1])
                    if x0 <= cx < x1 and y0 <= cy < y1:
                        patch_disps.append(disp)
                if patch_disps:
                    vx, vy = self._aggregate_displacement(np.stack(patch_disps, axis=0))
                else:
                    vx, vy = global_vx, global_vy
                vx_grid[pi, pj] = vx
                vy_grid[pi, pj] = vy

        return vx_grid, vy_grid

    @staticmethod
    def _center_for_id(snapshot: _DetSnapshot, track_id: int) -> np.ndarray | None:
        for tid, center in zip(snapshot.ids, snapshot.centers):
            if int(tid) == track_id:
                return center
        return None

    def push_detection(
        self,
        track_ids: np.ndarray,
        centers_xy: np.ndarray,
        *,
        det_hw: tuple[int, int],
        raw_hw: tuple[int, int],
    ) -> None:
        """Record one detection frame; centers are mapped from det image space to raw."""
        det_h, det_w = map(int, det_hw)
        raw_h, raw_w = map(int, raw_hw)
        if det_h <= 0 or det_w <= 0:
            raise ValueError(f"Invalid det_hw={det_hw}")
        sx = raw_w / float(det_w)
        sy = raw_h / float(det_h)

        ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        centers = np.asarray(centers_xy, dtype=np.float32).reshape(-1, 2)
        if ids.size == 0:
            snapshot = _DetSnapshot(ids=np.zeros(0, dtype=np.int64), centers=np.zeros((0, 2), dtype=np.float32))
        else:
            scaled = centers.copy()
            scaled[:, 0] *= sx
            scaled[:, 1] *= sy
            snapshot = _DetSnapshot(ids=ids, centers=scaled)

        self.prev_det_2 = self.prev_det_1
        self.prev_det_1 = snapshot

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
        """Process one photon cube using detection-guided velocity when available."""
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
        return integrated[..., None]
