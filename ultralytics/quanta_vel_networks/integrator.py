"""Velocity-guided SPAD integration utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VelIntegrator(nn.Module):
    """Integrate SPAD photon cubes using an externally supplied dense velocity field."""

    def __init__(
        self,
        chunk_size: int = 320,
        max_shift: int = 16,
        patch_size: int = 0,
        compensate_space: str = "rgb",
        normalize: bool = False,
        quantile: float = 1.0,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.max_shift = int(max_shift)
        self.patch_size = int(patch_size)
        self.compensate_space = str(compensate_space).lower()
        if self.compensate_space not in {"rgb", "raw"}:
            raise ValueError(f"compensate_space must be 'rgb' or 'raw', got {compensate_space!r}")
        self.normalize = bool(normalize)
        self.quantile = float(quantile)
        self.outputs_rgb: bool = self.compensate_space == "rgb"

        self.velocity_field: Tensor | None = None
        self.velocity_field_space: str = "rgb"
        self.last_velocity: tuple[float, float] = (0.0, 0.0)
        self.last_velocity_vx: Tensor | None = None
        self.last_velocity_vy: Tensor | None = None

    def reset(self) -> None:
        """Clear cached velocity-field state."""
        self.velocity_field = None
        self.velocity_field_space = "rgb"
        self.last_velocity = (0.0, 0.0)
        self.last_velocity_vx = None
        self.last_velocity_vy = None

    def set_velocity_field(self, field: Tensor | np.ndarray | None, *, source_space: str = "rgb") -> None:
        """Cache the dense velocity field that will be used for the next chunk."""
        if field is None:
            self.velocity_field = None
            self.velocity_field_space = str(source_space).lower()
            self.last_velocity = (0.0, 0.0)
            self.last_velocity_vx = None
            self.last_velocity_vy = None
            return

        source_space = str(source_space).lower()
        if source_space not in {"rgb", "raw"}:
            raise ValueError(f"source_space must be 'rgb' or 'raw', got {source_space!r}")

        field_t = torch.as_tensor(field, dtype=torch.float32)
        if field_t.ndim != 3 or field_t.shape[-1] != 2:
            raise ValueError(f"Velocity field must be shaped (H,W,2), got {tuple(field_t.shape)}")
        self.velocity_field = field_t.detach().cpu().contiguous()
        self.velocity_field_space = source_space
        self.last_velocity_vx = self.velocity_field[..., 0]
        self.last_velocity_vy = self.velocity_field[..., 1]
        self.last_velocity = (
            float(self.last_velocity_vx.mean().item()),
            float(self.last_velocity_vy.mean().item()),
        )

    @staticmethod
    def _photon_cube_to_rgb_tchw(cube: Tensor, packed_nch: int) -> Tensor:
        """Convert ``(H, W, T)`` Bayer photon cube to ``(T, 3, H/2, W/2)`` float RGB."""
        from ultralytics.data.spad_packed import raw_hwt_to_rgb_float

        return raw_hwt_to_rgb_float(cube.float(), packed_nch=int(packed_nch))

    @staticmethod
    def _base_grid(h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((xx, yy), dim=-1)

    @staticmethod
    def _normalize_flow(flow_hw2: Tensor) -> Tensor:
        h, w = int(flow_hw2.shape[0]), int(flow_hw2.shape[1])
        scale_x = 0.0 if w <= 1 else 2.0 / float(w - 1)
        scale_y = 0.0 if h <= 1 else 2.0 / float(h - 1)
        flow = flow_hw2.clone()
        flow[..., 0] *= scale_x
        flow[..., 1] *= scale_y
        return flow

    def _resized_velocity_field(
        self,
        target_hw: tuple[int, int],
        *,
        target_space: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        h, w = map(int, target_hw)
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid target_hw={target_hw}")
        if self.velocity_field is None:
            return torch.zeros((h, w, 2), device=device, dtype=dtype)

        field = self.velocity_field.to(device=device, dtype=dtype)
        src_h, src_w = int(field.shape[0]), int(field.shape[1])
        field_chw = field.permute(2, 0, 1).unsqueeze(0)
        if (src_h, src_w) != (h, w):
            field_chw = F.interpolate(field_chw, size=(h, w), mode="bilinear", align_corners=True)
        field = field_chw.squeeze(0).permute(1, 2, 0).contiguous()

        scale_x = float(w) / max(float(src_w), 1.0)
        scale_y = float(h) / max(float(src_h), 1.0)
        if self.velocity_field_space != target_space:
            field[..., 0] *= scale_x
            field[..., 1] *= scale_y
        elif (src_h, src_w) != (h, w):
            field[..., 0] *= scale_x
            field[..., 1] *= scale_y
        return field

    def _warp_parallel(self, frames_tchw: Tensor, flow_hw2: Tensor) -> Tensor:
        if frames_tchw.ndim != 4:
            raise ValueError(f"Expected frames_tchw (T,C,H,W), got shape={tuple(frames_tchw.shape)}")
        t, _, h, w = frames_tchw.shape
        if t <= 0:
            return frames_tchw
        base_grid = self._base_grid(h, w, device=frames_tchw.device, dtype=frames_tchw.dtype)
        flow_norm = self._normalize_flow(flow_hw2.to(device=frames_tchw.device, dtype=frames_tchw.dtype))
        if t == 1:
            alpha = torch.zeros((1, 1, 1, 1), device=frames_tchw.device, dtype=frames_tchw.dtype)
        else:
            alpha = torch.linspace(1.0, 0.0, t, device=frames_tchw.device, dtype=frames_tchw.dtype).view(t, 1, 1, 1)
        grids = base_grid.unsqueeze(0) - alpha * flow_norm.unsqueeze(0)
        return F.grid_sample(
            frames_tchw,
            grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def _integrate_rgb(self, cube: Tensor, packed_nch: int) -> Tensor:
        rgb_tchw = self._photon_cube_to_rgb_tchw(cube, packed_nch)
        flow = self._resized_velocity_field(
            target_hw=(int(rgb_tchw.shape[2]), int(rgb_tchw.shape[3])),
            target_space="rgb",
            device=rgb_tchw.device,
            dtype=rgb_tchw.dtype,
        )
        self.last_velocity_vx = flow[..., 0].detach().cpu()
        self.last_velocity_vy = flow[..., 1].detach().cpu()
        self.last_velocity = (
            float(flow[..., 0].mean().item()),
            float(flow[..., 1].mean().item()),
        )
        warped = self._warp_parallel(rgb_tchw, flow)
        return warped.mean(dim=0)

    def _integrate_raw(self, cube: Tensor) -> Tensor:
        raw_tchw = cube.permute(2, 0, 1).unsqueeze(1).float()
        flow = self._resized_velocity_field(
            target_hw=(int(raw_tchw.shape[2]), int(raw_tchw.shape[3])),
            target_space="raw",
            device=raw_tchw.device,
            dtype=raw_tchw.dtype,
        )
        self.last_velocity_vx = flow[..., 0].detach().cpu()
        self.last_velocity_vy = flow[..., 1].detach().cpu()
        self.last_velocity = (
            float(flow[..., 0].mean().item()),
            float(flow[..., 1].mean().item()),
        )
        warped = self._warp_parallel(raw_tchw, flow)
        return warped.mean(dim=0).squeeze(0)

    def _clamp_recons(self, recons: Tensor) -> Tensor:
        if recons.numel() == 0:
            return recons
        if self.normalize:
            scale = torch.quantile(recons.reshape(-1), self.quantile).clamp(min=1e-6)
            recons = recons / scale
        return recons.clamp(0, 1)

    @torch.no_grad()
    def process_photon_cube(self, photon_cube: Tensor, clear_states: bool = False, **kwargs) -> Tensor:
        """Warp all time slices using the cached field and average them."""
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
        if "compensate_space" in kwargs and kwargs["compensate_space"] is not None:
            space = str(kwargs["compensate_space"]).lower()
            if space not in {"rgb", "raw"}:
                raise ValueError(f"compensate_space must be 'rgb' or 'raw', got {space!r}")
            self.compensate_space = space
            self.outputs_rgb = space == "rgb"

        cube = photon_cube.float()
        if self.compensate_space == "rgb":
            packed_nch = int(kwargs.get("packed_nch", 4))
            integrated = self._integrate_rgb(cube, packed_nch)
            return self._clamp_recons(integrated)

        integrated = self._integrate_raw(cube)
        integrated = self._clamp_recons(integrated)
        return integrated[..., None]
