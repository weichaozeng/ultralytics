"""Builders and adapters for SPAD training preprocessors."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.quanta_hyb_networks.integrator import HybridSpatioTemporalEvidenceAccumulation
from ultralytics.quanta_neural_networks.integrator import PerPixelBayesian
from ultralytics.quanta_pdrs_networks.integrator import PoissonDualRateSplit, PoissonDualRateSplitFrame
from ultralytics.quanta_stea_networks.integrator import SpatioTemporalEvidenceAccumulation, SpatioTemporalEvidenceFrame


class SumPreprocessor(nn.Module):
    """Temporal-mean preprocessor that emits one frame per subsampling window."""

    def __init__(self, subsampling: int = 1):
        super().__init__()
        self.subsampling = max(int(subsampling), 1)

    def process_photon_cube(self, photon_cube: torch.Tensor, clear_states: bool = True, **kwargs) -> torch.Tensor:
        if photon_cube.ndim != 3:
            raise ValueError(f"Expected photon_cube (H,W,T), got shape={tuple(photon_cube.shape)}")
        if photon_cube.shape[-1] == 0:
            h, w = map(int, photon_cube.shape[:2])
            return photon_cube.new_zeros((h, w, 0), dtype=torch.float32)
        raw = photon_cube.float()
        t_raw = int(raw.shape[-1])
        if t_raw <= self.subsampling:
            return raw.mean(dim=-1, keepdim=True)

        frames = []
        for t0 in range(0, t_raw, self.subsampling):
            t1 = min(t_raw, t0 + self.subsampling)
            frames.append(raw[..., t0:t1].mean(dim=-1, keepdim=True))
        return torch.cat(frames, dim=-1)

    def recon_t_indices(self, t_raw: int, num_frames: int) -> list[int]:
        if t_raw <= 0 or num_frames <= 0:
            return []
        if t_raw <= self.subsampling:
            return [int(t_raw)]
        idx = []
        for t1 in range(self.subsampling, t_raw + 1, self.subsampling):
            idx.append(int(min(t1, t_raw)))
        if idx and idx[-1] != int(t_raw):
            idx.append(int(t_raw))
        return idx[:num_frames]

    @staticmethod
    def _confidence_to_frame_space(confidence_hw: torch.Tensor) -> torch.Tensor:
        if confidence_hw.ndim != 2:
            raise ValueError(f"Expected confidence map (H,W), got shape={tuple(confidence_hw.shape)}")
        return F.avg_pool2d(confidence_hw.unsqueeze(0).unsqueeze(0).float(), kernel_size=2, stride=2).squeeze(0)

    @staticmethod
    def _sobel_edge_confidence(frame_hw: torch.Tensor) -> torch.Tensor:
        if frame_hw.ndim != 2:
            raise ValueError(f"Expected frame (H,W), got shape={tuple(frame_hw.shape)}")
        frame_11hw = frame_hw.unsqueeze(0).unsqueeze(0).float()
        sobel_x = frame_hw.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = frame_hw.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        grad_x = F.conv2d(frame_11hw, sobel_x, padding=1)
        grad_y = F.conv2d(frame_11hw, sobel_y, padding=1)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12).squeeze(0).squeeze(0)
        edge_max = edge.amax().clamp(min=1e-6)
        return (edge / edge_max).clamp(0.0, 1.0)

    @torch.no_grad()
    def process_photon_cube_to_frame(self, photon_cube: torch.Tensor, clear_states: bool = True, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if photon_cube.ndim != 3:
            raise ValueError(f"Expected photon_cube (H,W,T), got shape={tuple(photon_cube.shape)}")
        if photon_cube.shape[-1] == 0:
            h, w = map(int, photon_cube.shape[:2])
            empty = photon_cube.new_zeros((h, w, 0), dtype=torch.float32)
            confidence = photon_cube.new_zeros((1, h // 2, w // 2), dtype=torch.float32)
            return empty, confidence
        frame = photon_cube.float().mean(dim=-1, keepdim=True)
        confidence_hw = self._sobel_edge_confidence(frame.squeeze(-1))
        confidence = self._confidence_to_frame_space(confidence_hw).to(device=frame.device, dtype=frame.dtype)
        return frame, confidence


class PerPixelBayesianFrame(nn.Module):
    """Frame-mode wrapper around PerPixelBayesian that exposes a lightweight confidence map."""

    def __init__(self, **kwargs):
        super().__init__()
        self.core = PerPixelBayesian(**kwargs)

    @property
    def subsampling(self) -> int:
        return int(getattr(self.core, "subsampling", 1) or 1)

    @staticmethod
    def _confidence_to_frame_space(confidence_hw: torch.Tensor) -> torch.Tensor:
        if confidence_hw.ndim != 2:
            raise ValueError(f"Expected confidence map (H,W), got shape={tuple(confidence_hw.shape)}")
        return F.avg_pool2d(confidence_hw.unsqueeze(0).unsqueeze(0).float(), kernel_size=2, stride=2).squeeze(0)

    @torch.no_grad()
    def process_photon_cube_to_frame(self, photon_cube: torch.Tensor, clear_states: bool = True, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        recons = self.core.process_photon_cube(photon_cube, clear_states=clear_states, **kwargs)
        if recons.ndim != 3 or int(recons.shape[-1]) == 0:
            h, w = map(int, photon_cube.shape[:2])
            recons = photon_cube.new_zeros((h, w, 1), dtype=torch.float32)
        else:
            recons = recons[..., -1:].contiguous()
        sample_weight = getattr(self.core, "sample_weight", None)
        if sample_weight is None:
            confidence_hw = photon_cube.new_ones(photon_cube.shape[0], photon_cube.shape[1], dtype=torch.float32)
        else:
            confidence_hw = (1.0 - sample_weight.detach().float()).clamp(0.0, 1.0)
        confidence = self._confidence_to_frame_space(confidence_hw).to(device=recons.device, dtype=recons.dtype)
        return recons, confidence


def build_spad_preprocessor(name: str, *, kwargs: dict[str, Any]) -> nn.Module:
    """Build a SPAD preprocessor module from a short name."""
    name = str(name).strip().lower()
    if name == "ppb":
        return PerPixelBayesian(**kwargs)
    if name == "stea":
        return SpatioTemporalEvidenceAccumulation(**kwargs)
    if name in {"pdrs", "dual_rate", "dual-rate"}:
        return PoissonDualRateSplit(**kwargs)
    if name == "hyb":
        return HybridSpatioTemporalEvidenceAccumulation(**kwargs)
    if name == "sum":
        return SumPreprocessor(**kwargs)
    raise ValueError(f"Unsupported SPAD preprocessor: {name!r}")


def build_spad_frame_preprocessor(name: str, *, kwargs: dict[str, Any]) -> nn.Module:
    """Build a frame-mode SPAD preprocessor that emits one detector frame per raw chunk."""
    name = str(name).strip().lower()
    if name == "ppb":
        return PerPixelBayesianFrame(**kwargs)
    if name == "stea":
        return SpatioTemporalEvidenceFrame(**kwargs)
    if name in {"pdrs", "dual_rate", "dual-rate"}:
        return PoissonDualRateSplitFrame(**kwargs)
    if name == "sum":
        return SumPreprocessor(**kwargs)
    raise ValueError(f"Unsupported frame-mode SPAD preprocessor: {name!r}")
