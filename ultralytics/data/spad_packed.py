"""Unpack VisionSIM / real-device packed SPAD frames to full-resolution Bayer raw."""

from __future__ import annotations

import numpy as np

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


def packed_frames_to_raw_bayer(
    frames_packed: np.ndarray,
    *,
    expected_w: int = 512,
    ch_order: str = "RGB",
) -> np.ndarray:
    """Expand packed bits to Bayer raw `(T, 2H, 2W)` uint8.

    - ``C=3`` (synthetic): ch0/ch1/ch2 are R/G/B; G is written to both Bayer G sites.
    - ``C=4`` (real): ch0/ch1/ch2/ch3 are R/G1/G2/B mapped to RGGB sites separately.
    """
    if frames_packed.ndim != 4 or frames_packed.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected packed frames (T,H,Wpacked,3|4), got shape={frames_packed.shape}"
        )

    unpacked = np.unpackbits(frames_packed, axis=2)
    if unpacked.shape[2] > expected_w:
        unpacked = unpacked[:, :, :expected_w, :]

    n_ch = int(unpacked.shape[-1])
    t, h, w, _ = unpacked.shape
    raw = np.zeros((t, h * 2, w * 2), dtype=np.uint8)

    if n_ch == 4:
        if ch_order.upper() == "BGR":
            r_ch, g1_ch, g2_ch, b_ch = 3, 1, 2, 0
        else:
            r_ch, g1_ch, g2_ch, b_ch = 0, 1, 2, 3
        raw[:, 0::2, 0::2] = unpacked[:, :, :, r_ch]
        raw[:, 0::2, 1::2] = unpacked[:, :, :, g1_ch]
        raw[:, 1::2, 0::2] = unpacked[:, :, :, g2_ch]
        raw[:, 1::2, 1::2] = unpacked[:, :, :, b_ch]
        return raw

    if ch_order.upper() == "BGR":
        r_ch, g_ch, b_ch = 2, 1, 0
    else:
        r_ch, g_ch, b_ch = 0, 1, 2
    raw[:, 0::2, 0::2] = unpacked[:, :, :, r_ch]
    raw[:, 0::2, 1::2] = unpacked[:, :, :, g_ch]
    raw[:, 1::2, 0::2] = unpacked[:, :, :, g_ch]
    raw[:, 1::2, 1::2] = unpacked[:, :, :, b_ch]
    return raw


def bayer_plane_to_rgb_u8(raw_hw: np.ndarray, *, packed_nch: int) -> np.ndarray:
    """Map one full-resolution Bayer frame to RGB uint8 at half resolution.

    - ``packed_nch=3`` (synthetic): direct RGGB plane subsample (no demosaic).
    - ``packed_nch=4`` (real / true Bayer): OpenCV RG demosaic then resize if needed.
    """
    if raw_hw.ndim != 2:
        raise ValueError(f"Expected Bayer frame (H,W), got shape={raw_hw.shape}")
    raw_u8 = raw_hw if raw_hw.dtype == np.uint8 else np.clip(raw_hw, 0, 255).astype(np.uint8)
    h, w = raw_u8.shape
    if int(packed_nch) == 3:
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
