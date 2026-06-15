"""Unpack VisionSIM / real-device packed SPAD frames for integrator pipelines."""

from __future__ import annotations

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
    """Synthetic VisionSIM packed data (3ch) uses RGGB Bayer expand + plane subsample."""
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


def _unpacked3_to_bayer(unpacked: np.ndarray, ch_order: str) -> np.ndarray:
    """Map synthetic RGB planes to an RGGB Bayer mosaic ``(T, 2H, 2W)``.

    G is written to both Bayer G sites (G1 and G2).
    """
    t, h, w, _ = unpacked.shape
    raw = np.zeros((t, h * 2, w * 2), dtype=np.uint8)
    if ch_order.upper() == "BGR":
        r_ch, g_ch, b_ch = 2, 1, 0
    else:
        r_ch, g_ch, b_ch = 0, 1, 2
    raw[:, 0::2, 0::2] = unpacked[:, :, :, r_ch]
    raw[:, 0::2, 1::2] = unpacked[:, :, :, g_ch]
    raw[:, 1::2, 0::2] = unpacked[:, :, :, g_ch]
    raw[:, 1::2, 1::2] = unpacked[:, :, :, b_ch]
    return raw


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
    """Expand packed frames to Bayer raw ``(T, 2H, 2W)`` uint8."""
    unpacked = unpack_packed_frames(frames_packed, expected_w=expected_w)
    n_ch = int(unpacked.shape[-1])
    if n_ch == 3:
        return _unpacked3_to_bayer(unpacked, ch_order)
    return _unpacked4_to_bayer(unpacked, ch_order)


def packed_frames_to_raw_video(
    frames_packed: np.ndarray,
    *,
    expected_w: int = 512,
    ch_order: str = "RGB",
) -> np.ndarray:
    """Convert packed frames to integrator-ready Bayer raw ``(T, 2H, 2W, 1)`` uint8."""
    raw = packed_frames_to_raw_bayer(frames_packed, expected_w=expected_w, ch_order=ch_order)
    return raw[:, :, :, None]


def raw_chunk_plane(raw_chunk: np.ndarray, *, packed_nch: int) -> np.ndarray:
    """Return the Bayer photon plane as ``(T, H, W)``."""
    if raw_chunk.ndim != 4 or raw_chunk.shape[-1] != 1:
        raise ValueError(f"Expected Bayer raw chunk (T,H,W,1), got shape={raw_chunk.shape}")
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


def _bayer_rggb_hwt_to_rgb_tchw(raw_hwt: Tensor) -> Tensor:
    """Subsample RGGB Bayer ``(2H, 2W, T)`` to ``(T, 3, H, W)`` with ``G=(G1+G2)/2``."""
    r = raw_hwt[0::2, 0::2, :]
    g1 = raw_hwt[0::2, 1::2, :]
    g2 = raw_hwt[1::2, 0::2, :]
    b = raw_hwt[1::2, 1::2, :]
    g = 0.5 * (g1 + g2)
    return torch.stack((r, g, b), dim=0).permute(3, 0, 1, 2).contiguous()


def _bayer_rggb_hw_to_rgb_hw3(bayer_hw: Tensor) -> Tensor:
    """Subsample one RGGB Bayer frame ``(2H, 2W)`` to ``(3, H, W)``."""
    r = bayer_hw[0::2, 0::2]
    g = 0.5 * (bayer_hw[0::2, 1::2] + bayer_hw[1::2, 0::2])
    b = bayer_hw[1::2, 1::2]
    return torch.stack((r, g, b), dim=0)


def integrate_raw_chunk_to_rgb(
    integrator: Any,
    raw_chunk: np.ndarray,
    *,
    packed_nch: int,
    device: torch.device | str,
    clear_states: bool,
    **integrator_kwargs: Any,
) -> Tensor:
    """Run a stateful integrator on Bayer raw and return ``(1, 3, H/2, W/2)`` float RGB."""
    cube = raw_plane_to_photon_cube(raw_chunk_plane(raw_chunk, packed_nch=packed_nch), device=device, as_bool=True)
    recons = integrator.process_photon_cube(cube, clear_states=clear_states, **integrator_kwargs)
    rgb_tchw = raw_hwt_to_rgb_float(recons.float(), packed_nch=int(packed_nch))
    if int(rgb_tchw.shape[0]) <= 0:
        return rgb_tchw
    return rgb_tchw[-1:].contiguous()


def sum_raw_chunk_to_rgb(raw_chunk: np.ndarray, *, packed_nch: int, device: torch.device | str) -> Tensor:
    """Temporal mean over Bayer raw -> ``(1, 3, H/2, W/2)`` float RGB."""
    raw = torch.from_numpy(raw_chunk_plane(raw_chunk, packed_nch=packed_nch)).to(device).float()
    raw_mean = raw.mean(dim=0)
    if is_synthetic_packed(packed_nch):
        return _bayer_rggb_hw_to_rgb_hw3(raw_mean).unsqueeze(0).clamp(0, 1)
    return raw_hwt_to_rgb_float(raw_mean.unsqueeze(-1), packed_nch=4)


def raw_hwt_to_rgb_float(raw_hwt: Tensor, *, packed_nch: int) -> Tensor:
    """Convert integrator output ``(H, W, T)`` to ``(T, 3, H_out, W_out)`` float RGB.

    - ``packed_nch=3``: RGGB plane subsample, ``G=(G1+G2)/2``.
    - ``packed_nch=4``: OpenCV demosaic then resize to half resolution.
    """
    if not torch.is_tensor(raw_hwt):
        raise TypeError(f"Expected torch.Tensor, got {type(raw_hwt)}")
    if raw_hwt.ndim != 3:
        raise ValueError(f"Expected raw_hwt (H,W,T), got shape={tuple(raw_hwt.shape)}")

    h_raw, w_raw, t = map(int, raw_hwt.shape)
    if t <= 0:
        return raw_hwt.new_zeros((0, 3, h_raw // 2, w_raw // 2))

    if h_raw % 2 != 0 or w_raw % 2 != 0:
        raise ValueError(f"Bayer raw reconstruction must have even H/W, got {(h_raw, w_raw)}")

    if is_synthetic_packed(packed_nch):
        return _bayer_rggb_hwt_to_rgb_tchw(raw_hwt.float())

    import cv2

    raw_np = raw_hwt.detach().float().cpu().numpy()
    frames = []
    for ti in range(t):
        raw_u8 = np.clip(raw_np[:, :, ti] * 255.0, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(raw_u8, cv2.COLOR_BAYER_RG2RGB)
        rgb = cv2.resize(rgb, (w_raw // 2, h_raw // 2), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    return torch.stack(frames, dim=0).to(raw_hwt.device)


def bayer_plane_to_rgb_u8(raw_hw: np.ndarray, *, packed_nch: int) -> np.ndarray:
    """Map one Bayer frame to RGB uint8 at half resolution.

    - ``packed_nch=3``: RGGB subsample with ``G=(G1+G2)/2``.
    - ``packed_nch=4``: demosaic then resize if needed.
    """
    if raw_hw.ndim != 2:
        raise ValueError(f"Expected Bayer frame (H,W), got shape={raw_hw.shape}")

    raw_u8 = raw_hw if raw_hw.dtype == np.uint8 else np.clip(raw_hw, 0, 255).astype(np.uint8)
    h, w = raw_u8.shape

    if is_synthetic_packed(packed_nch):
        r = raw_u8[0::2, 0::2]
        g = (0.5 * (raw_u8[0::2, 1::2].astype(np.float32) + raw_u8[1::2, 0::2].astype(np.float32))).astype(np.uint8)
        b = raw_u8[1::2, 1::2]
        return np.stack((r, g, b), axis=2)

    import cv2

    rgb = cv2.cvtColor(raw_u8, cv2.COLOR_BAYER_RG2RGB)
    target = (w // 2, h // 2)
    if rgb.shape[1] != target[0] or rgb.shape[0] != target[1]:
        rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)
    return rgb


def raw_video_mean_to_rgb_u8(raw_video: np.ndarray, *, packed_nch: int) -> np.ndarray:
    """Temporal mean of a Bayer raw video chunk, returned as RGB uint8 ``(H/2, W/2, 3)``."""
    if raw_video.ndim != 4 or raw_video.shape[-1] != 1:
        raise ValueError(f"Expected Bayer raw video (T,H,W,1), got shape={raw_video.shape}")
    raw_mean = raw_video[..., 0].astype(np.float32).mean(axis=0)
    raw_u8 = np.clip(raw_mean * 255.0, 0, 255).astype(np.uint8)
    return bayer_plane_to_rgb_u8(raw_u8, packed_nch=int(packed_nch))
