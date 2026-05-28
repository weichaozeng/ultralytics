"""Unpack VisionSIM / real-device packed SPAD frames to full-resolution Bayer raw."""

from __future__ import annotations

import numpy as np

PACKED_WIDTHS = (64, 128)


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
