# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""inspect_spad_packed.py

Small utility to validate how VisionSIM SPAD packed data is interpreted.

It loads a packed array (typically frames.npy shaped (N,H,Wpacked,3)), unpacks width bits,
then visualizes:
- each of the 3 unpacked channels (as grayscale)
- the reduced single-channel photon observation used by det_qnns.py (any/sum)

Outputs PNGs into --save_dir for quick inspection.

Example
-------
python ultralytics/inspect_spad_packed.py \
  --in_path /path/to/dataset_or_frames.npy \
  --save_dir /tmp/inspect \
  --reduce any \
  --max_frames 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _load_packed(in_path: Path) -> np.ndarray:
    """Load packed VisionSIM frames.

    Accepts:
    - directory containing frames.npy
    - direct .npy path

    Returns:
        ndarray: (N,H,Wpacked,3) uint8
    """
    if in_path.is_dir():
        npy = in_path / "frames.npy"
        if not npy.exists():
            raise FileNotFoundError(f"Directory input requires frames.npy, not found: {npy}")
        arr = np.load(npy)
        src = npy
    else:
        if in_path.suffix.lower() != ".npy":
            raise ValueError(f"Unsupported input: {in_path} (expect directory or .npy)")
        arr = np.load(in_path)
        src = in_path

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected packed frames (N,H,Wpacked,3) in {src}, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _unpack(frames_packed: np.ndarray, *, expected_w: int = 512) -> np.ndarray:
    """Unpack bit-packed width. (N,H,Wpacked,3) -> (N,H,W,3) bool."""
    unpacked = np.unpackbits(frames_packed, axis=2)
    if unpacked.shape[2] > expected_w:
        unpacked = unpacked[:, :, :expected_w, :]
    return unpacked.astype(bool, copy=False)


def _reduce(unpacked: np.ndarray, *, mode: str) -> np.ndarray:
    """Reduce (N,H,W,3) bool -> (N,H,W) bool."""
    if mode == "any":
        return unpacked.any(axis=3)
    if mode == "sum":
        return (unpacked.sum(axis=3) > 0)
    raise ValueError(f"Unsupported reduce mode: {mode}")


def _to_u8(x: np.ndarray) -> np.ndarray:
    """bool/int/float -> uint8 image."""
    if x.dtype == bool:
        return (x.astype(np.uint8) * 255)
    x = x.astype(np.float32, copy=False)
    mx = float(x.max())
    if mx <= 0:
        return np.zeros_like(x, dtype=np.uint8)
    return np.clip(x / mx * 255.0, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Inspect VisionSIM bit-packed SPAD frames")
    ap.add_argument("--in_path", type=str, required=True, help="Dataset directory containing frames.npy or the .npy path")
    ap.add_argument("--save_dir", type=str, required=True, help="Output directory for PNGs")
    ap.add_argument("--expected_w", type=int, default=512)
    ap.add_argument("--reduce", type=str, default="any", choices=["any", "sum"], help="Channel-reduction used by det_qnns")
    ap.add_argument("--t0", type=int, default=0, help="Start time index to dump")
    ap.add_argument("--max_frames", type=int, default=1, help="How many time frames to dump (default: 1)")
    ap.add_argument("--stride", type=int, default=1, help="Stride over time frames")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    packed = _load_packed(in_path)
    unpacked = _unpack(packed, expected_w=args.expected_w)  # (N,H,W,3) bool
    reduced = _reduce(unpacked, mode=args.reduce)  # (N,H,W) bool

    n, h, w, c = unpacked.shape
    print(f"packed:   shape={packed.shape} dtype={packed.dtype}")
    print(f"unpacked: shape={unpacked.shape} dtype={unpacked.dtype} (W={w})")
    print(f"reduced:  shape={reduced.shape} dtype={reduced.dtype} reduce={args.reduce}")

    # Simple statistics to see whether channels differ
    ch_rates = [float(unpacked[..., k].mean()) for k in range(3)]
    red_rate = float(reduced.mean())
    print(f"mean occupancy: ch0={ch_rates[0]:.6f} ch1={ch_rates[1]:.6f} ch2={ch_rates[2]:.6f} reduced={red_rate:.6f}")

    dumped = 0
    for t in tqdm(range(args.t0, n, args.stride), desc="Dumping frames"):
        if dumped >= args.max_frames:
            break

        # Per-channel grayscale
        for k in range(3):
            u8 = _to_u8(unpacked[t, :, :, k])
            out = save_dir / f"t{t:06d}_ch{k}.png"
            cv2.imwrite(str(out), u8)

        # Reduced photon map
        u8r = _to_u8(reduced[t])
        out = save_dir / f"t{t:06d}_reduced_{args.reduce}.png"
        cv2.imwrite(str(out), u8r)

        # RGB-composite visualization (treat channels as RGB just for inspection)
        rgb = np.stack([unpacked[t, :, :, 0], unpacked[t, :, :, 1], unpacked[t, :, :, 2]], axis=2).astype(np.uint8) * 255
        bgr = rgb[:, :, ::-1]
        out = save_dir / f"t{t:06d}_rgb_composite.png"
        cv2.imwrite(str(out), bgr)

        dumped += 1


if __name__ == "__main__":
    main()
