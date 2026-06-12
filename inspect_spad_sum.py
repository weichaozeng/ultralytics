# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Inspect summed VisionSIM SPAD chunks.

This utility is for generated VisionSIM packed `frames.npy` data, e.g.
`renders-spc8kHz/.../frames.npy`, where frames are shaped `(T, H, Wpacked, 3)`.
It unpacks native R/G/B planes without Bayer expansion, then visualizes
`sum / chunk_size` over a temporal chunk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics.data.spad_packed import packed_frames_to_raw_video, raw_video_mean_to_rgb_u8


def _load_frames(path: Path) -> np.ndarray:
    """Load a VisionSIM `frames.npy` array with mmap."""
    npy = path / "frames.npy" if path.is_dir() else path
    if not npy.exists():
        raise FileNotFoundError(f"frames.npy not found: {npy}")
    arr = np.load(npy, mmap_mode="r")
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected synthetic packed frames (T,H,Wpacked,3), got {arr.shape} from {npy}")
    return arr


def _to_bgr_u8(rgb: np.ndarray, *, mode: str, gamma: float, percentile: float) -> np.ndarray:
    """Map linear RGB float image to BGR uint8 for OpenCV."""
    rgb = rgb.astype(np.float32, copy=False)
    if mode == "linear":
        vis = np.clip(rgb, 0.0, 1.0)
    elif mode == "gamma":
        vis = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / gamma)
    elif mode == "percentile":
        scale = float(np.percentile(rgb, percentile))
        vis = np.clip(rgb / max(scale, 1e-6), 0.0, 1.0)
    elif mode == "percentile_gamma":
        scale = float(np.percentile(rgb, percentile))
        vis = np.clip(rgb / max(scale, 1e-6), 0.0, 1.0)
        vis = np.power(vis, 1.0 / gamma)
    else:
        raise ValueError(f"Unsupported vis mode: {mode}")
    return np.ascontiguousarray((vis * 255.0).round().astype(np.uint8)[:, :, ::-1])


def _print_stats(name: str, x: np.ndarray) -> None:
    """Print useful brightness statistics."""
    flat = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x.reshape(-1)
    print(f"{name}: shape={x.shape} dtype={x.dtype}")
    if x.ndim == 3:
        for idx, ch in enumerate("RGB"):
            values = flat[:, idx]
            pct = np.percentile(values, [50, 90, 95, 99, 99.5, 99.9, 100])
            print(
                f"  {ch}: mean={values.mean():.6f} min={values.min():.6f} "
                f"p50={pct[0]:.6f} p90={pct[1]:.6f} p95={pct[2]:.6f} "
                f"p99={pct[3]:.6f} p99.5={pct[4]:.6f} p99.9={pct[5]:.6f} max={pct[6]:.6f}"
            )
    else:
        pct = np.percentile(flat, [50, 90, 95, 99, 99.5, 99.9, 100])
        print(
            f"  mean={flat.mean():.6f} min={flat.min():.6f} p50={pct[0]:.6f} "
            f"p90={pct[1]:.6f} p95={pct[2]:.6f} p99={pct[3]:.6f} "
            f"p99.5={pct[4]:.6f} p99.9={pct[5]:.6f} max={pct[6]:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect sum/chunk_size visualization for synthetic VisionSIM SPAD data")
    parser.add_argument("--in_path", type=Path, required=True, help="VisionSIM sample directory containing frames.npy, or frames.npy path")
    parser.add_argument("--save_dir", type=Path, required=True, help="Directory to save inspection PNGs")
    parser.add_argument("--t0", type=int, default=0, help="Chunk start index")
    parser.add_argument("--chunk_size", type=int, default=320, help="Number of SPAD frames to average")
    parser.add_argument("--expected_w", type=int, default=512, help="Unpacked synthetic RGB width")
    parser.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    parser.add_argument(
        "--vis_mode",
        type=str,
        default="linear",
        choices=["linear", "gamma", "percentile", "percentile_gamma"],
        help="Display mapping for the saved image. The underlying summed image is always sum/chunk_size.",
    )
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma for gamma display modes")
    parser.add_argument("--percentile", type=float, default=99.5, help="Percentile for percentile display modes")
    args = parser.parse_args()

    frames = _load_frames(args.in_path)
    t0 = int(args.t0)
    t1 = min(t0 + int(args.chunk_size), int(frames.shape[0]))
    if t0 < 0 or t0 >= frames.shape[0] or t1 <= t0:
        raise ValueError(f"Invalid chunk [{t0}:{t1}] for frames length {frames.shape[0]}")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    packed = np.asarray(frames[t0:t1])
    raw = packed_frames_to_raw_video(packed, expected_w=args.expected_w, ch_order=args.packed_ch_order)
    rgb_u8 = raw_video_mean_to_rgb_u8(raw, packed_nch=3)
    rgb_mean = rgb_u8.astype(np.float32) / 255.0

    _print_stats("packed_chunk", packed)
    _print_stats("rgb_mean", rgb_mean)

    bgr = _to_bgr_u8(rgb_mean, mode=args.vis_mode, gamma=float(args.gamma), percentile=float(args.percentile))
    out = args.save_dir / f"sum_t{t0:06d}_{t1:06d}_div{args.chunk_size}_{args.vis_mode}.png"
    cv2.imwrite(str(out), bgr)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
