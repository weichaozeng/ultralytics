#!/usr/bin/env python3
"""Export binary 0/1 SPAD bin frames from ``frames.npy`` to a folder.

Loads packed VisionSIM / device dumps ``(T, H, Wpacked, C)`` (or already-unpacked
``(T, H, W)`` / ``(T, H, W, C)``), selects ``--num`` frames, and writes one binary
image per bin.

Examples
--------
# First 32 bins → PNG (0 / 255)
python ultralytics/vis_spad_bins.py \\
  --in_path /path/to/frames.npy \\
  --save_dir /tmp/spad_bins \\
  --num 32

# Start at bin 1000, every 10th bin, R plane only
python ultralytics/vis_spad_bins.py \\
  --in_path /path/to/frames.npy \\
  --save_dir /tmp/spad_bins_r \\
  --start 1000 --num 16 --stride 10 --channel r
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics.data.spad_packed import is_packed_spad, unpack_packed_frames


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Save binary 0/1 SPAD bins from frames.npy")
    ap.add_argument("--in_path", type=Path, required=True, help="Path to frames.npy (or dir with frames.npy)")
    ap.add_argument("--save_dir", type=Path, required=True, help="Output folder for binary frames")
    ap.add_argument("--num", type=int, required=True, help="Number of bins/frames to save")
    ap.add_argument("--start", type=int, default=0, help="First temporal index (default: 0)")
    ap.add_argument("--stride", type=int, default=1, help="Step between saved bins (default: 1)")
    ap.add_argument(
        "--channel",
        type=str,
        default="any",
        help=(
            "How to collapse multi-channel planes to one binary map: "
            "any|or (OR of channels), r|g|b|0|1|2, or bayer (RGGB mosaic). "
            "Ignored for single-channel input."
        ),
    )
    ap.add_argument(
        "--expected_w",
        type=int,
        default=0,
        help="Crop unpacked width; 0 = keep Wpacked*8",
    )
    ap.add_argument("--flip_x", action="store_true")
    ap.add_argument("--flip_y", action="store_true")
    ap.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "npy"],
        help="png: uint8 0/255; npy: uint8 0/1",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="bin",
        help="Filename prefix (default: bin → bin_0000000.png)",
    )
    return ap.parse_args()


def _resolve_npy(in_path: Path) -> Path:
    if in_path.is_dir():
        for name in ("frames.npy", "binary.npy"):
            cand = in_path / name
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No frames.npy/binary.npy under {in_path}")
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if in_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected .npy, got {in_path}")
    return in_path


def _channel_index(name: str, n_ch: int) -> int | None:
    key = name.strip().lower()
    if key in ("any", "or", "bayer"):
        return None
    aliases = {"r": 0, "g": 1, "b": 2, "g1": 1, "g2": 2}
    if key in aliases:
        idx = aliases[key]
    else:
        idx = int(key)
    if idx < 0 or idx >= n_ch:
        raise ValueError(f"channel={name!r} out of range for C={n_ch}")
    return idx


def _to_binary_hw(frame: np.ndarray, *, channel: str) -> np.ndarray:
    """Return ``(H, W)`` uint8 with values in {0, 1}."""
    if frame.ndim == 2:
        return (frame > 0).astype(np.uint8)

    if frame.ndim != 3:
        raise ValueError(f"Expected HW or HWC frame, got shape={frame.shape}")

    h, w, c = frame.shape
    key = channel.strip().lower()
    if key == "bayer":
        if c not in (3, 4):
            raise ValueError(f"bayer needs C in {{3,4}}, got C={c}")
        mosaic = np.zeros((h * 2, w * 2), dtype=np.uint8)
        if c == 3:
            mosaic[0::2, 0::2] = frame[:, :, 0] > 0
            mosaic[0::2, 1::2] = frame[:, :, 1] > 0
            mosaic[1::2, 0::2] = frame[:, :, 1] > 0
            mosaic[1::2, 1::2] = frame[:, :, 2] > 0
        else:
            mosaic[0::2, 0::2] = frame[:, :, 0] > 0
            mosaic[0::2, 1::2] = frame[:, :, 1] > 0
            mosaic[1::2, 0::2] = frame[:, :, 2] > 0
            mosaic[1::2, 1::2] = frame[:, :, 3] > 0
        return mosaic

    idx = _channel_index(key, c)
    if idx is None:
        return (frame > 0).any(axis=-1).astype(np.uint8)
    return (frame[:, :, idx] > 0).astype(np.uint8)


def _unpack_slice(packed: np.ndarray, *, expected_w: int) -> np.ndarray:
    """Unpack a temporal slice to bool/uint ``(T, H, W)`` or ``(T, H, W, C)``."""
    # Already-unpacked binary (bool) or single-plane video.
    if packed.dtype == np.bool_ or packed.ndim == 3:
        return packed

    # Bit-packed VisionSIM / device dumps: uint8 (T, H, Wpacked, 3|4).
    if is_packed_spad(packed) and np.issubdtype(packed.dtype, np.integer):
        return unpack_packed_frames(packed, expected_w=expected_w if expected_w > 0 else None)

    if packed.ndim == 4 and packed.shape[-1] in (1, 3, 4):
        return packed

    raise ValueError(
        f"Unsupported frames.npy shape {packed.shape} dtype={packed.dtype}. "
        "Expected packed (T,H,Wpacked,3|4) or unpacked (T,H,W)/(T,H,W,C)."
    )


def main() -> None:
    args = _parse_args()
    if int(args.num) <= 0:
        raise ValueError("--num must be > 0")
    if int(args.stride) <= 0:
        raise ValueError("--stride must be > 0")
    if int(args.start) < 0:
        raise ValueError("--start must be >= 0")

    path = _resolve_npy(args.in_path)
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    t_total = int(arr.shape[0])
    print(f"loaded {path}")
    print(f"shape={arr.shape} dtype={arr.dtype}")

    indices = [int(args.start) + i * int(args.stride) for i in range(int(args.num))]
    if indices[-1] >= t_total:
        raise IndexError(
            f"Requested last index {indices[-1]} >= T={t_total} "
            f"(start={args.start}, num={args.num}, stride={args.stride})"
        )

    out_dir = args.save_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Contiguous slab when stride==1; otherwise unpack one bin at a time.
    slab_frames = None
    if int(args.stride) == 1:
        t0, t1 = indices[0], indices[-1] + 1
        slab = np.asarray(arr[t0:t1])
        unpacked = _unpack_slice(slab, expected_w=int(args.expected_w))
        if args.flip_x:
            unpacked = np.flip(unpacked, axis=2)
        if args.flip_y:
            unpacked = np.flip(unpacked, axis=1)
        slab_frames = np.ascontiguousarray(unpacked)

    n_written = 0
    for i, t in enumerate(indices):
        if slab_frames is not None:
            frame = slab_frames[i]
        else:
            slab = np.asarray(arr[t : t + 1])
            unpacked = _unpack_slice(slab, expected_w=int(args.expected_w))
            if args.flip_x:
                unpacked = np.flip(unpacked, axis=2)
            if args.flip_y:
                unpacked = np.flip(unpacked, axis=1)
            frame = unpacked[0]

        binary = _to_binary_hw(np.asarray(frame), channel=str(args.channel))
        stem = f"{args.prefix}_{t:07d}"
        if args.format == "npy":
            out_path = out_dir / f"{stem}.npy"
            np.save(out_path, binary.astype(np.uint8, copy=False))
        else:
            out_path = out_dir / f"{stem}.png"
            # Viewable binary: 0 / 255
            ok = cv2.imwrite(str(out_path), (binary * 255).astype(np.uint8))
            if not ok:
                raise RuntimeError(f"Failed to write {out_path}")
        n_written += 1

    print(f"wrote {n_written} binary frames → {out_dir}")
    print(f"indices: {indices[0]} … {indices[-1]} (stride={args.stride})")
    print(f"channel={args.channel} format={args.format}")


if __name__ == "__main__":
    main()
