#!/usr/bin/env python3
"""Make side-by-side videos from Vis/0730 frame dumps with optional slow-mo.

Reads (from ``vis_0730_pose.py``)::

    <vis_root>/<sample>/{qnn,hire}/
      recon/frame_XXXXXXX.png
      pose/frame_XXXXXXX.png      # BGRA → composited on white
      heatmap/frame_XXXXXXX.png   # BGRA → use BGR only (no bg fill)

For each sample × method, stitches panels left→right::

    recon | pose (white) | heatmap

and writes a constant-fps MP4 under::

    <vis_root>/video/<sample>_{ppb,hire}.mp4

``qnn`` dumps are labeled ``ppb`` in the output name.

Slow-mo: in ``--slow_ranges`` (source frame index ranges), consecutive
frame pairs are expanded by linear (or hold) interpolation so those
segments play slower at the same output ``--fps``.

Examples
--------
# One sample, slow frames 40–80 at 4×, clip frames 0–200
python ultralytics/vis_0730_video.py \\
  --samples acq00001 \\
  --frame_start 0 --frame_end 200 \\
  --slow_ranges 40-80 \\
  --slow_factor 4 --fps 25

# Multiple slow windows; both methods
python ultralytics/vis_0730_video.py \\
  --samples acq00001 acq00002 \\
  --methods ppb,hire \\
  --slow_ranges 10-30,100-140 \\
  --slow_factor 8 --interp linear --overwrite
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_VIS_ROOT = Path("/home/zvc/Project/SPADHand/Vis/0730")
DEFAULT_VIDEO_SUBDIR = "video"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# folder name on disk → name in output mp4 stem
METHOD_ALIASES = {
    "ppb": "qnn",
    "qnn": "qnn",
    "hire": "hire",
}
METHOD_OUT_NAME = {
    "qnn": "ppb",
    "hire": "hire",
}


def _natural_key(path: Path) -> list:
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_frame_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=_natural_key)


def parse_slow_ranges(spec: str) -> list[tuple[int, int]]:
    """Parse ``'10-30,100-140'`` → inclusive ``[(10,30), (100,140)]``."""
    spec = (spec or "").strip()
    if not spec:
        return []
    ranges: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Bad slow range {part!r}; want start-end")
        a, b = part.split("-", 1)
        lo, hi = int(a.strip()), int(b.strip())
        if hi < lo:
            lo, hi = hi, lo
        ranges.append((lo, hi))
    return ranges


def frame_in_slow(i: int, ranges: list[tuple[int, int]]) -> bool:
    """True if transition ``i → i+1`` should be slowed (``i`` in a slow range)."""
    for lo, hi in ranges:
        if lo <= i < hi:
            return True
    return False


def _to_bgr_u8(
    img: np.ndarray,
    *,
    alpha_bg: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Array → BGR uint8.

    If ``alpha_bg`` is set and image has alpha, composite over that color.
    Otherwise drop alpha and keep BGR (no background fill).
    """
    if img is None:
        raise ValueError("empty image")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return img
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        if alpha_bg is None:
            return np.ascontiguousarray(bgr)
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        base = np.full_like(bgr, alpha_bg, dtype=np.float32)
        out = bgr.astype(np.float32) * a + base * (1.0 - a)
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image shape {img.shape}")


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    new_w = max(int(round(w * (height / float(h)))), 1)
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def stitch_panels(recon: np.ndarray, pose: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """Left→right: recon | pose (white bg) | heatmap (raw BGR, no fill)."""
    r = _to_bgr_u8(recon)
    h = int(r.shape[0])
    p = _resize_to_height(_to_bgr_u8(pose, alpha_bg=(255, 255, 255)), h)
    hm = _resize_to_height(_to_bgr_u8(heat, alpha_bg=None), h)
    return np.concatenate([r, p, hm], axis=1)


def lerp_bgr(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = a.astype(np.float32) * (1.0 - t) + b.astype(np.float32) * t
    return np.clip(out, 0, 255).astype(np.uint8)


def expand_with_slowmo(
    panels: list[np.ndarray],
    *,
    slow_ranges: list[tuple[int, int]],
    slow_factor: int,
    interp: str,
) -> list[np.ndarray]:
    """Insert interpolated frames on slow transitions; keep constant-time steps at output fps."""
    if not panels:
        return []
    factor = max(int(slow_factor), 1)
    interp = str(interp).strip().lower()
    if interp not in {"linear", "hold"}:
        raise ValueError(f"interp must be linear|hold, got {interp!r}")

    out: list[np.ndarray] = [panels[0]]
    for i in range(len(panels) - 1):
        a, b = panels[i], panels[i + 1]
        n_steps = factor if frame_in_slow(i, slow_ranges) else 1
        if n_steps <= 1:
            out.append(b)
            continue
        for s in range(1, n_steps + 1):
            t = s / float(n_steps)
            if interp == "hold" or s == n_steps:
                out.append(b if s == n_steps else a)
            else:
                out.append(lerp_bgr(a, b, t))
    return out


def load_stitched_sequence(
    method_dir: Path,
    *,
    frame_start: int,
    frame_end: int | None,
) -> list[np.ndarray]:
    recon_paths = list_frame_paths(method_dir / "recon")
    pose_paths = list_frame_paths(method_dir / "pose")
    heat_paths = list_frame_paths(method_dir / "heatmap")
    n = min(len(recon_paths), len(pose_paths), len(heat_paths))
    if n <= 0:
        raise FileNotFoundError(
            f"No aligned recon/pose/heatmap frames under {method_dir} "
            f"(recon={len(recon_paths)}, pose={len(pose_paths)}, heat={len(heat_paths)})"
        )
    lo = max(int(frame_start), 0)
    hi = n if frame_end is None or int(frame_end) < 0 else min(int(frame_end), n)
    if hi <= lo:
        raise ValueError(f"Empty frame window [{lo}, {hi}) for {method_dir} (n={n})")

    panels: list[np.ndarray] = []
    for i in range(lo, hi):
        recon = cv2.imread(str(recon_paths[i]), cv2.IMREAD_UNCHANGED)
        pose = cv2.imread(str(pose_paths[i]), cv2.IMREAD_UNCHANGED)
        heat = cv2.imread(str(heat_paths[i]), cv2.IMREAD_UNCHANGED)
        if recon is None or pose is None or heat is None:
            raise RuntimeError(f"Failed to read frame index {i} under {method_dir}")
        panels.append(stitch_panels(recon, pose, heat))
    return panels


def write_mp4(frames: list[np.ndarray], path: Path, *, fps: float) -> None:
    if not frames:
        raise ValueError("no frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")
    try:
        for fr in frames:
            if fr.shape[0] != h or fr.shape[1] != w:
                fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(fr)
    finally:
        writer.release()


def discover_samples(vis_root: Path, samples: list[str] | None) -> list[Path]:
    if samples:
        dirs = [vis_root / s for s in samples]
        missing = [p for p in dirs if not p.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing sample dirs: {missing}")
        return dirs
    skip = {DEFAULT_VIDEO_SUBDIR}
    dirs = sorted(p for p in vis_root.iterdir() if p.is_dir() and p.name not in skip)
    if not dirs:
        raise FileNotFoundError(f"No sample folders under {vis_root}")
    return dirs


def resolve_methods(spec: str) -> list[str]:
    names = [x.strip().lower() for x in spec.split(",") if x.strip()]
    if not names:
        raise ValueError("empty --methods")
    out: list[str] = []
    for n in names:
        if n not in METHOD_ALIASES:
            raise ValueError(f"Unknown method {n!r}; use ppb|qnn|hire")
        folder = METHOD_ALIASES[n]
        if folder not in out:
            out.append(folder)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Vis/0730 → side-by-side videos with optional slow-mo interpolation"
    )
    ap.add_argument(
        "--vis_root",
        type=Path,
        default=DEFAULT_VIS_ROOT,
        help="Root with <sample>/{qnn,hire}/{recon,pose,heatmap}",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output dir (default: <vis_root>/video)",
    )
    ap.add_argument(
        "--samples",
        type=str,
        nargs="*",
        default=None,
        help="Sample folder names under vis_root; default = all",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="ppb,hire",
        help="Comma list: ppb (qnn dumps), hire",
    )
    ap.add_argument(
        "--frame_start",
        type=int,
        default=0,
        help="First source frame index (inclusive) to include",
    )
    ap.add_argument(
        "--frame_end",
        type=int,
        default=-1,
        help="End source frame index (exclusive); -1 = all",
    )
    ap.add_argument(
        "--slow_ranges",
        type=str,
        default="",
        help="Slow-mo source ranges, e.g. '40-80' or '10-30,100-140' (inclusive start, exclusive end for transitions)",
    )
    ap.add_argument(
        "--slow_factor",
        type=int,
        default=4,
        help="Time stretch in slow ranges: insert (factor-1) intermediates per step (default 4)",
    )
    ap.add_argument(
        "--interp",
        type=str,
        default="linear",
        choices=["linear", "hold"],
        help="Interpolation inside slow ranges",
    )
    ap.add_argument("--fps", type=float, default=25.0, help="Output video FPS (constant)")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    vis_root = Path(args.vis_root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else vis_root / DEFAULT_VIDEO_SUBDIR
    if not vis_root.is_dir():
        raise FileNotFoundError(vis_root)
    if float(args.fps) <= 0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")
    if int(args.slow_factor) < 1:
        raise ValueError(f"--slow_factor must be >= 1, got {args.slow_factor}")

    methods = resolve_methods(args.methods)
    slow_ranges = parse_slow_ranges(args.slow_ranges)
    sample_dirs = discover_samples(vis_root, args.samples)
    frame_end = None if int(args.frame_end) < 0 else int(args.frame_end)

    print(
        f"vis_root={vis_root}\n"
        f"out_dir={out_dir}\n"
        f"samples={[p.name for p in sample_dirs]}\n"
        f"methods={methods} (out names={[METHOD_OUT_NAME[m] for m in methods]})\n"
        f"frame_start={args.frame_start} frame_end={frame_end}\n"
        f"slow_ranges={slow_ranges} slow_factor={args.slow_factor} interp={args.interp}\n"
        f"fps={args.fps}",
        flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for sample_dir in sample_dirs:
        sample = sample_dir.name
        for folder in methods:
            method_dir = sample_dir / folder
            out_name = METHOD_OUT_NAME[folder]
            out_path = out_dir / f"{sample}_{out_name}.mp4"
            if out_path.is_file() and not args.overwrite:
                print(f"skip (exists): {out_path}", flush=True)
                continue
            if not method_dir.is_dir():
                print(f"skip (missing method dir): {method_dir}", flush=True)
                continue

            panels = load_stitched_sequence(
                method_dir,
                frame_start=int(args.frame_start),
                frame_end=frame_end,
            )
            # Remap slow ranges into the clipped window (indices relative to panels).
            lo = max(int(args.frame_start), 0)
            rel_slow = [(max(a - lo, 0), max(b - lo, 0)) for a, b in slow_ranges]
            rel_slow = [(a, b) for a, b in rel_slow if b > a]
            expanded = expand_with_slowmo(
                panels,
                slow_ranges=rel_slow,
                slow_factor=int(args.slow_factor),
                interp=str(args.interp),
            )
            print(
                f"{sample}/{out_name}: src_frames={len(panels)} → out_frames={len(expanded)} → {out_path}",
                flush=True,
            )
            for _ in tqdm(range(1), desc=f"write {sample}_{out_name}", leave=False):
                write_mp4(expanded, out_path, fps=float(args.fps))
            n_written += 1

    print(f"Done. wrote={n_written} → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
