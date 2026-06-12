"""Unpack VisionSIM / real-device packed SPAD frames for integrator pipelines."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import Tensor

PACKED_WIDTHS = (64, 128)


def infer_packed_nch(arr: np.ndarray) -> int:
    """Infer packed channel semantics from a SPAD array layout."""
    if arr.ndim == 4 and arr.shape[-1] in (3, 4) and arr.shape[2] in PACKED_WIDTHS:
        return int(arr.shape[-1])
    return 4


def is_packed_spad(arr: np.ndarray) -> bool:
    """True for `(T, H, Wpacked, 3)` synthetic or `(T, H, Wpacked, 4)` real RGGB planes."""
    return (
        arr.ndim == 4
        and arr.shape[-1] in (3, 4)
        and arr.shape[2] in PACKED_WIDTHS
    )


def is_synthetic_packed(packed_nch: int) -> bool:
    """Synthetic VisionSIM packed data uses three native R/G/B planes (no Bayer expand)."""
    return int(packed_nch) == 3


def unpack_packed_frames(frames_packed: np.ndarray, *, expected_w: int = 512) -> np.ndarray:
    """Unpack bit-packed width to boolean planes ``(T, H, W, C)`` with ``C`` in {3, 4}."""
    if frames_packed.ndim != 4 or frames_packed.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected packed frames (T,H,Wpacked,3|4), got shape={frames_packed.shape}"
        )

    unpacked = np.unpackbits(frames_packed, axis=2)
    if unpacked.shape[2] > expected_w:
        unpacked = unpacked[:, :, :expected_w, :]
    return unpacked.astype(bool, copy=False)


def _unpacked4_to_bayer(unpacked: np.ndarray, ch_order: str) -> np.ndarray:
    """Map unpacked four-channel planes to a full-resolution Bayer mosaic ``(T, 2H, 2W)``."""
    t, h, w, _ = unpacked.shape
    raw = np.zeros((t, h * 2, w * 2), dtype=np.uint8)
    if ch_order.upper() == "BGR":
        r_ch, g1_ch, g2_ch, b_ch = 3, 1, 2, 0
    else:
        r_ch, g1_ch, g2_ch, b_ch = 0, 1, 2, 3
    raw[:, 0::2, 0::2] = unpacked[:, :, :, r_ch]
    raw[:, 0::2, 1::2] = unpacked[:, :, :, g1_ch]
    raw[:, 1::2, 0::2] = unpacked[:, :, :, g2_ch]
    raw[:, 1::2, 1::2] = unpacked[:, :, :, b_ch]
    return raw


def packed_frames_to_raw_bayer(
    frames_packed: np.ndarray,
    *,
    expected_w: int = 512,
    ch_order: str = "RGB",
) -> np.ndarray:
    """Expand real-device packed frames to Bayer raw ``(T, 2H, 2W)`` uint8.

  For synthetic ``C=3`` data use :func:`packed_frames_to_raw_video` instead; do not
  interleave RGB into a fake Bayer grid.
    """
    unpacked = unpack_packed_frames(frames_packed, expected_w=expected_w)
    n_ch = int(unpacked.shape[-1])
    if n_ch == 3:
        raise ValueError(
            "packed_frames_to_raw_bayer is for 4-channel real Bayer data only; "
            "use packed_frames_to_raw_video for synthetic 3-channel frames.npy"
        )
    return _unpacked4_to_bayer(unpacked, ch_order)


def packed_frames_to_raw_video(
    frames_packed: np.ndarray,
    *,
    expected_w: int = 512,
    ch_order: str = "RGB",
) -> np.ndarray:
    """Convert packed frames to integrator-ready raw video.

    - ``C=3`` (synthetic): ``(T, H, W, 3)`` native R/G/B photon planes (no 2× Bayer expand).
    - ``C=4`` (real): ``(T, 2H, 2W, 1)`` full Bayer mosaic.
    """
    unpacked = unpack_packed_frames(frames_packed, expected_w=expected_w)
    n_ch = int(unpacked.shape[-1])
    if n_ch == 3:
        return unpacked.astype(np.uint8, copy=False)
    raw = _unpacked4_to_bayer(unpacked, ch_order)
    return raw[:, :, :, None]


def raw_chunk_plane(raw_chunk: np.ndarray, channel: int, *, packed_nch: int) -> np.ndarray:
    """Return one photon plane as ``(T, H, W)``."""
    if int(packed_nch) == 3:
        return np.ascontiguousarray(raw_chunk[..., int(channel)])
    return np.ascontiguousarray(raw_chunk[..., 0])


def raw_plane_to_photon_cube(
    plane_thw: np.ndarray,
    *,
    device: torch.device | str,
    as_bool: bool = True,
) -> Tensor:
    """Map ``(T, H, W)`` to integrator layout ``(H, W, T)``."""
    tensor = torch.from_numpy(plane_thw).to(device)
    cube = tensor.permute(1, 2, 0)
    if as_bool:
        return cube.bool()
    return cube.float()


def make_integrator_triplet(integrator: Any) -> torch.nn.ModuleList:
    """Three independent integrator instances for streaming per-channel 3ch synthetic data."""
    return torch.nn.ModuleList([copy.deepcopy(integrator) for _ in range(3)])


def _last_recon_hw(recons_hwt: Tensor) -> Tensor:
    if int(recons_hwt.shape[-1]) <= 0:
        raise ValueError("integrator returned empty reconstruction")
    return recons_hwt[..., -1]


def integrate_raw_chunk_to_rgb(
    integrator: Any,
    raw_chunk: np.ndarray,
    *,
    packed_nch: int,
    device: torch.device | str,
    clear_states: bool,
    integrators_3ch: torch.nn.ModuleList | None = None,
    **integrator_kwargs: Any,
) -> Tensor:
    """Run a stateful integrator on a raw chunk and return ``(1, 3, H, W)`` float RGB."""
    if int(packed_nch) == 3:
        integrators = integrators_3ch
        if integrators is None:
            raise ValueError("integrators_3ch is required for synthetic 3-channel raw chunks")
        channels = []
        for ch in range(3):
            cube = raw_plane_to_photon_cube(
                raw_chunk_plane(raw_chunk, ch, packed_nch=3),
                device=device,
                as_bool=True,
            )
            recons = integrators[ch].process_photon_cube(cube, clear_states=clear_states, **integrator_kwargs)
            channels.append(_last_recon_hw(recons.float()))
        rgb_hw3 = torch.stack(channels, dim=-1)
        return rgb_hw3.permute(2, 0, 1).unsqueeze(0).contiguous()

    cube = raw_plane_to_photon_cube(
        raw_chunk_plane(raw_chunk, 0, packed_nch=4),
        device=device,
        as_bool=True,
    )
    recons = integrator.process_photon_cube(cube, clear_states=clear_states, **integrator_kwargs)
    return raw_hwt_to_rgb_float(recons.float(), packed_nch=4)


def sum_raw_chunk_to_rgb(raw_chunk: np.ndarray, *, packed_nch: int, device: torch.device | str) -> Tensor:
    """Temporal mean over a raw chunk -> ``(1, 3, H, W)`` float RGB."""
    if int(packed_nch) == 3:
        raw = torch.from_numpy(raw_chunk).to(device).float()
        mean_hw3 = raw.mean(dim=0)
        return mean_hw3.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

    raw = torch.from_numpy(raw_chunk_plane(raw_chunk, 0, packed_nch=4)).to(device).float()
    raw_mean = raw.mean(dim=0, keepdim=True).permute(1, 2, 0)
    return raw_hwt_to_rgb_float(raw_mean, packed_nch=4)


def raw_hwt_to_rgb_float(raw_hwt: Tensor, *, packed_nch: int) -> Tensor:
    """Convert integrator output ``(H, W, T)`` to ``(T, 3, H, W)`` float RGB.

    - ``packed_nch=3``: native-resolution mono planes are stacked as R=G=B grayscale RGB.
    - ``packed_nch=4``: demosaic full-resolution Bayer then resize to half resolution.
    """
    if not torch.is_tensor(raw_hwt):
        raise TypeError(f"Expected torch.Tensor, got {type(raw_hwt)}")
    if raw_hwt.ndim != 3:
        raise ValueError(f"Expected raw_hwt (H,W,T), got shape={tuple(raw_hwt.shape)}")

    h_raw, w_raw, t = map(int, raw_hwt.shape)
    if t <= 0:
        out_h = h_raw if is_synthetic_packed(packed_nch) else h_raw // 2
        out_w = w_raw if is_synthetic_packed(packed_nch) else w_raw // 2
        return raw_hwt.new_zeros((0, 3, out_h, out_w))

    if is_synthetic_packed(packed_nch):
        mono = raw_hwt.float()
        rgb = mono.unsqueeze(2).expand(-1, -1, 3, -1)
        return rgb.permute(3, 2, 0, 1).contiguous()

    if h_raw % 2 != 0 or w_raw % 2 != 0:
        raise ValueError(f"Bayer raw reconstruction must have even H/W, got {(h_raw, w_raw)}")

    import cv2

    raw_np = raw_hwt.detach().float().cpu().numpy()
    frames = []
    for ti in range(t):
        raw_u8 = np.clip(raw_np[:, :, ti] * 255.0, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(raw_u8, cv2.COLOR_BAYER_RG2RGB)
        rgb = cv2.resize(rgb, (w_raw // 2, h_raw // 2), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    return torch.stack(frames, dim=0).to(raw_hwt.device)


def stack_native_recons_to_rgb(recons_by_channel: list[Tensor] | tuple[Tensor, ...]) -> Tensor:
    """Stack per-channel ``(H, W, T)`` reconstructions into ``(T, 3, H, W)``."""
    if not recons_by_channel:
        raise ValueError("recons_by_channel must not be empty")
    stacked = torch.stack([recons.float() for recons in recons_by_channel], dim=2)
    return stacked.permute(3, 2, 0, 1).contiguous()


def bayer_plane_to_rgb_u8(raw_hw: np.ndarray, *, packed_nch: int) -> np.ndarray:
    """Map one reconstructed frame to RGB uint8 at native output resolution.

    - ``packed_nch=3``: ``(H, W)`` mono or ``(H, W, 3)`` native RGB (no Bayer subsample).
    - ``packed_nch=4``: ``(2H, 2W)`` Bayer demosaic to ``(H, W, 3)``.
    """
    if raw_hw.ndim == 3 and int(raw_hw.shape[-1]) == 3:
        rgb = raw_hw if raw_hw.dtype == np.uint8 else np.clip(raw_hw, 0, 255).astype(np.uint8)
        return rgb

    if raw_hw.ndim != 2:
        raise ValueError(f"Expected raw frame (H,W) or (H,W,3), got shape={raw_hw.shape}")

    raw_u8 = raw_hw if raw_hw.dtype == np.uint8 else np.clip(raw_hw, 0, 255).astype(np.uint8)
    h, w = raw_u8.shape

    if is_synthetic_packed(packed_nch):
        return np.stack([raw_u8, raw_u8, raw_u8], axis=2)

    import cv2

    rgb = cv2.cvtColor(raw_u8, cv2.COLOR_BAYER_RG2RGB)
    target = (w // 2, h // 2)
    if rgb.shape[1] != target[0] or rgb.shape[0] != target[1]:
        rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)
    return rgb


def raw_video_mean_to_rgb_u8(raw_video: np.ndarray, *, packed_nch: int) -> np.ndarray:
    """Temporal mean of a raw video chunk, returned as RGB uint8 ``(H, W, 3)``."""
    if raw_video.ndim != 4:
        raise ValueError(f"Expected raw video (T,H,W,C), got shape={raw_video.shape}")
    t = max(int(raw_video.shape[0]), 1)
    if is_synthetic_packed(packed_nch):
        mean_hw3 = raw_video.astype(np.float32).mean(axis=0)
        return np.clip(mean_hw3 * 255.0, 0, 255).astype(np.uint8)

    if raw_video.shape[-1] != 1:
        raise ValueError(f"Expected Bayer raw video (T,H,W,1), got shape={raw_video.shape}")
    raw_mean = raw_video[..., 0].astype(np.float32).mean(axis=0)
    raw_u8 = np.clip(raw_mean * 255.0, 0, 255).astype(np.uint8)
    return bayer_plane_to_rgb_u8(raw_u8, packed_nch=4)
