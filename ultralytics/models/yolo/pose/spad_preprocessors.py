"""Builders and adapters for SPAD training preprocessors."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ultralytics.quanta_hyb_networks.integrator import HybridSpatioTemporalEvidenceAccumulation
from ultralytics.quanta_neural_networks.integrator import PerPixelBayesian
from ultralytics.quanta_stea_networks.integrator import SpatioTemporalEvidenceAccumulation


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


def build_spad_preprocessor(name: str, *, kwargs: dict[str, Any]) -> nn.Module:
    """Build a SPAD preprocessor module from a short name."""
    name = str(name).strip().lower()
    if name == "ppb":
        return PerPixelBayesian(**kwargs)
    if name == "stea":
        return SpatioTemporalEvidenceAccumulation(**kwargs)
    if name == "hyb":
        return HybridSpatioTemporalEvidenceAccumulation(**kwargs)
    if name == "sum":
        return SumPreprocessor(**kwargs)
    raise ValueError(f"Unsupported SPAD preprocessor: {name!r}")
