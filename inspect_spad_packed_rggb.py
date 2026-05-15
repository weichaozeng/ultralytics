# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""inspect_spad_packed_rggb.py

Utility to validate how VisionSIM SPAD packed data is interpreted, with extra Bayer/RGGB
mapping diagnostics.

It loads a packed array (typically frames.npy shaped (N,H,Wpacked,3)), unpacks width bits,
then visualizes:
- each of the 3 unpacked channels (as grayscale)
- a reduced single-channel photon observation (any/sum)
- optional RGGB RAW sampling (rggb_raw)
- optional RGGB expand (2H,2W) visualization (rggb_expand)
- optional expand->pack-back roundtrip XOR image to sanity-check the mapping assumption

Notes
-----
- This tool is for debugging channel order, spatial mapping, and raw/Bayer assumptions.
- Roundtrip currently implements pack-back for RGGB only.

Example
-------
python ultralytics/inspect_spad_packed_rggb.py \
  --in_path /path/to/frames.npy \
  --save_dir /tmp/inspect \
  --reduce rggb_raw \
  --packed_ch_order RGB \
  --dump_expand --roundtrip \
  --max_frames 4 --stride 200
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
        ndarray: (N,H,Wpacked,3) uint8 (memmap if possible)
    """
    if in_path.is_dir():
        npy = in_path / "frames.npy"
        if not npy.exists():
            raise FileNotFoundError(f"Directory input requires frames.npy, not found: {npy}")
        arr = np.load(npy, mmap_mode="r")
        src = npy
    else:
        if in_path.suffix.lower() != ".npy":
            raise ValueError(f"Unsupported input: {in_path} (expect directory or .npy)")
        arr = np.load(in_path, mmap_mode="r")
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


def _rgb_to_bgr_u8(rgb_u8: np.ndarray) -> np.ndarray:
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3) image, got {rgb_u8.shape}")
    return rgb_u8[:, :, ::-1]


def _to_u8(x: np.ndarray) -> np.ndarray:
    """bool/int/float -> uint8 image."""
    if x.dtype == bool:
        return x.astype(np.uint8) * 255
    x = x.astype(np.float32, copy=False)
    mx = float(x.max())
    if mx <= 0:
        return np.zeros_like(x, dtype=np.uint8)
    return np.clip(x / mx * 255.0, 0, 255).astype(np.uint8)


def _channels_from_order(packed_ch_order: str) -> tuple[int, int, int]:
    """Return (r_ch,g_ch,b_ch) indices for packed order."""
    if packed_ch_order == "BGR":
        return 2, 1, 0
    return 0, 1, 2


def _rggb_sample_raw(unpacked: np.ndarray, *, packed_ch_order: str) -> np.ndarray:
    """Sample (N,H,W,3) -> (N,H,W) RGGB mosaic raw."""
    r_ch, g_ch, b_ch = _channels_from_order(packed_ch_order)

    n, h, w, _ = unpacked.shape
    raw = np.zeros((n, h, w), dtype=bool)

    # RGGB mosaic positions
    raw[:, 0::2, 0::2] = unpacked[:, 0::2, 0::2, r_ch]
    raw[:, 0::2, 1::2] = unpacked[:, 0::2, 1::2, g_ch]
    raw[:, 1::2, 0::2] = unpacked[:, 1::2, 0::2, g_ch]
    raw[:, 1::2, 1::2] = unpacked[:, 1::2, 1::2, b_ch]
    return raw


def _rggb_expand(unpacked: np.ndarray, *, packed_ch_order: str) -> np.ndarray:
    """Expand (N,H,W,3) -> (N,2H,2W) single-channel RGGB raw plane.

    Each (y,x) creates a 2x2 block:
      (2y,2x)=R, (2y,2x+1)=G, (2y+1,2x)=G, (2y+1,2x+1)=B
    """
    r_ch, g_ch, b_ch = _channels_from_order(packed_ch_order)
    n, h, w, _ = unpacked.shape
    raw = np.zeros((n, h * 2, w * 2), dtype=bool)

    raw[:, 0::2, 0::2] = unpacked[:, :, :, r_ch]
    raw[:, 0::2, 1::2] = unpacked[:, :, :, g_ch]
    raw[:, 1::2, 0::2] = unpacked[:, :, :, g_ch]
    raw[:, 1::2, 1::2] = unpacked[:, :, :, b_ch]

    return raw


def _pack_back_from_expand(expanded_raw: np.ndarray, *, bayer_pattern: str) -> np.ndarray:
    """Pack expanded (N,2H,2W) back to standard (N,H,W) mosaic.

    Mapping sanity check only.

    NOTE: RGGB only for now.
    """
    if bayer_pattern != "RGGB":
        raise NotImplementedError("pack-back currently implemented for RGGB only")

    n, hh, ww = expanded_raw.shape
    if hh % 2 != 0 or ww % 2 != 0:
        raise ValueError(f"expanded_raw must have even spatial dims, got {expanded_raw.shape}")

    h, w = hh // 2, ww // 2
    raw = np.zeros((n, h, w), dtype=bool)

    # Each original pixel expanded to a 2x2 block, so the expanded frame is 2H x 2W.
    # When packing back to a standard mosaic, we must select samples for each mosaic cell.
    # The selection below mirrors the earlier discussion used for the PPB roundtrip path.
    raw[:, 0::2, 0::2] = expanded_raw[:, 0::4, 0::4]  # R
    raw[:, 0::2, 1::2] = expanded_raw[:, 0::4, 3::4]  # G (top-right)
    raw[:, 1::2, 0::2] = expanded_raw[:, 3::4, 0::4]  # G (bottom-left)
    raw[:, 1::2, 1::2] = expanded_raw[:, 3::4, 3::4]  # B

    return raw


def _reduce(unpacked: np.ndarray, *, mode: str, packed_ch_order: str) -> np.ndarray:
    """Reduce (N,H,W,3) bool -> (N,H,W) bool (or raw mode)."""
    if mode == "any":
        return unpacked.any(axis=3)
    if mode == "sum":
        return unpacked.sum(axis=3) > 0
    if mode == "rggb_raw":
        return _rggb_sample_raw(unpacked, packed_ch_order=packed_ch_order)
    raise ValueError(f"Unsupported reduce mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect VisionSIM bit-packed SPAD frames (with RGGB diagnostics)")
    ap.add_argument("--in_path", type=str, required=True, help="Dataset directory containing frames.npy or the .npy path")
    ap.add_argument("--save_dir", type=str, required=True, help="Output directory for PNGs")
    ap.add_argument("--expected_w", type=int, default=512)
    ap.add_argument(
        "--reduce",
        type=str,
        default="any",
        choices=["any", "sum", "rggb_raw"],
        help="Channel-reduction similar to det_qnns (plus rggb_raw for Bayer sampling checks)",
    )
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"], help="Packed 3-channel order")
    ap.add_argument(
        "--bayer_pattern",
        type=str,
        default="RGGB",
        choices=["RGGB", "BGGR", "GRBG", "GBRG"],
        help="Bayer pattern for roundtrip check (pack-back currently supports RGGB only)",
    )
    ap.add_argument("--dump_expand", action="store_true", help="Dump rggb_expand (2H,2W) raw visualization")
    ap.add_argument("--roundtrip", action="store_true", help="Run expand->pack-back XOR check (requires --dump_expand)")
    ap.add_argument("--t0", type=int, default=0, help="Start time index to dump")
    ap.add_argument("--max_frames", type=int, default=1, help="How many time frames to dump")
    ap.add_argument("--stride", type=int, default=1, help="Stride over time frames")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    packed = _load_packed(in_path)
    unpacked = _unpack(packed, expected_w=args.expected_w)  # (N,H,W,3) bool
    reduced = _reduce(unpacked, mode=args.reduce, packed_ch_order=args.packed_ch_order)  # (N,H,W) bool

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
            cv2.imwrite(str(save_dir / f"t{t:06d}_ch{k}.png"), u8)

        # Reduced photon map
        cv2.imwrite(str(save_dir / f"t{t:06d}_reduced_{args.reduce}.png"), _to_u8(reduced[t]))

        # RGB-composite visualization (treat channels as RGB just for inspection)
        rgb = (
            np.stack([unpacked[t, :, :, 0], unpacked[t, :, :, 1], unpacked[t, :, :, 2]], axis=2)
            .astype(np.uint8)
            * 255
        )
        cv2.imwrite(str(save_dir / f"t{t:06d}_rgb_composite.png"), _rgb_to_bgr_u8(rgb))

        if args.dump_expand:
            exp = _rggb_expand(unpacked[t : t + 1], packed_ch_order=args.packed_ch_order)[0]
            cv2.imwrite(str(save_dir / f"t{t:06d}_rggb_expand.png"), _to_u8(exp))

            if args.roundtrip:
                try:
                    packed_back = _pack_back_from_expand(exp[None, ...], bayer_pattern=args.bayer_pattern)[0]
                    cv2.imwrite(str(save_dir / f"t{t:06d}_rggb_packback.png"), _to_u8(packed_back))

                    src_raw = _rggb_sample_raw(unpacked[t : t + 1], packed_ch_order=args.packed_ch_order)[0]
                    diff = np.logical_xor(src_raw, packed_back)
                    cv2.imwrite(str(save_dir / f"t{t:06d}_rggb_roundtrip_xor.png"), _to_u8(diff))
                except Exception as e:
                    print(f"[roundtrip] t={t} failed: {e}")

        dumped += 1


if __name__ == "__main__":
    main()
