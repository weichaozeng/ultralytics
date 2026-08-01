#!/usr/bin/env python3
"""Make combined PPB+HIRE videos from pose-export frame dumps with optional slow-mo.

Reads dumps from ``vis_0730_pose.py`` / ``vis_spad_pose_export.py``::

    <vis_root>/<sample>/{qnn,hire}/
      recon/frame_XXXXXXX.png
      pose/frame_XXXXXXX.png      # BGRA → composited on white
      heatmap/frame_XXXXXXX.png   # BGRA → use BGR only (no bg fill)

Raw SPAD (left column) is taken from the matching ``frames.npy`` chunk::

    <data_root>/<sample>/frames.npy
    viz frame i  ↔  bins [i*chunk_size : (i+1)*chunk_size]

Each sample becomes one MP4. Layout (top→bottom)::

    PPB :  spad | recon | pose (white) | heatmap
    HIRE:  spad | recon | pose (white) | heatmap

Writes::

    <vis_root>/video/<sample>.mp4

Examples
--------
python ultralytics/vis_0730_video.py \\
  --samples acq00001 \\
  --frame_start 0 --frame_end 200 \\
  --slow_ranges 40-80 \\
  --slow_factor 4 --fps 25 --overwrite

# Slow-mo draws e.g. 0.02x on pose (top-left) when --slow_factor 50
python ultralytics/vis_0730_video.py \\
  --samples acq00019 --frame_start 0 --frame_end 33 \\
  --slow_ranges 10-15,20-26 --slow_factor 50 --interp hold --overwrite
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from ultralytics.data.spad_packed import packed_frames_to_raw_bayer

DEFAULT_VIS_ROOT = Path("/home/zvc/Project/SPADHand/Vis/0730")
DEFAULT_DATA_ROOT = Path("/home/zvc/Data/SPADHand/0730/spad/capture-spc8kHz")
DEFAULT_VIDEO_SUBDIR = "video"
DEFAULT_CHUNK_SIZE = 320
FRAMES_NPY = "frames.npy"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _natural_key(path: Path) -> list:
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_frame_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=_natural_key)


def parse_slow_ranges(spec: str) -> list[tuple[int, int]]:
    """Parse ``'10-30,100-140'`` → ``[(10,30), (100,140)]`` (hi exclusive for transitions).

    Accepts ASCII ``,`` / ``;`` and full-width ``，`` / ``；`` as separators.
    """
    spec = (spec or "").strip()
    if not spec:
        return []
    for sep in ("，", "；", ";", "|"):
        spec = spec.replace(sep, ",")
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


def format_playback_speed(slow_factor: int) -> str:
    """``slow_factor=50`` → ``'0.02x'``; ``10`` → ``'0.1x'``."""
    factor = max(int(slow_factor), 1)
    speed = 1.0 / float(factor)
    text = f"{speed:.4g}x"
    return text


def draw_speed_label(img: np.ndarray, text: str, *, x: int, y: int) -> np.ndarray:
    """Draw playback-speed text at top-left of a pose panel (in-place + return)."""
    if not text:
        return img
    h = int(img.shape[0])
    scale = max(0.55, min(1.4, h / 512.0 * 0.85))
    thickness = max(1, int(round(scale * 2)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (int(x) + 8, int(y) + int(28 * scale) + 4)
    # White halo then dark text — readable on white pose bg and dark recon.
    cv2.putText(img, text, org, font, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return img


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


def packed_slice_to_spad_gray(
    packed: np.ndarray,
    t0: int,
    t1: int,
    *,
    mode: str = "last",
    ch_order: str = "RGB",
) -> np.ndarray:
    """Read one bin window from mmap'd ``frames.npy`` → Bayer gray BGR.

    Supports packed ``C=3`` (VisionSIM) and ``C=4`` (real RGGB). Uses
    ``packed_frames_to_raw_bayer`` so expand matches the detector path.

    ``last`` / ``binary``: single last bin → 0/255.
    ``any``: OR over the chunk (often near-white at 320 bins).
    ``sum``: hit-count / T → 0..255 gray.
    """
    if packed.ndim != 4 or packed.shape[-1] not in (3, 4):
        raise ValueError(f"Expected packed (T,H,Wp,3|4), got {packed.shape}")
    t0 = max(int(t0), 0)
    t1 = min(int(t1), int(packed.shape[0]))
    if t1 <= t0:
        h = int(packed.shape[1]) * 2
        w = int(packed.shape[2]) * 8 * 2
        return np.zeros((h, w, 3), dtype=np.uint8)

    mode = str(mode).strip().lower()
    order = str(ch_order)

    def _bayer_u8(slice_thwpc: np.ndarray) -> np.ndarray:
        raw = packed_frames_to_raw_bayer(np.asarray(slice_thwpc), ch_order=order)
        # raw: (T, 2H, 2W) uint8 {0,1}
        return raw

    if mode in {"last", "binary"}:
        raw = _bayer_u8(packed[t1 - 1 : t1])  # (1,2H,2W)
        gray = (raw[0] > 0).astype(np.uint8) * 255
    elif mode == "any":
        ored = np.bitwise_or.reduce(packed[t0:t1], axis=0)  # (H,Wp,C)
        raw = _bayer_u8(ored[None, ...])
        gray = (raw[0] > 0).astype(np.uint8) * 255
    elif mode == "sum":
        raw = _bayer_u8(packed[t0:t1]).astype(np.float32)
        t = max(int(raw.shape[0]), 1)
        gray = np.clip(raw.sum(axis=0) / float(t) * 255.0, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"spad mode must be last|binary|any|sum, got {mode!r}")
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def load_spad_panel_sequence(
    frames_npy: Path,
    *,
    frame_start: int,
    frame_end: int | None,
    n_viz: int,
    chunk_size: int,
    mode: str,
    ch_order: str = "RGB",
) -> list[np.ndarray]:
    """Map viz frame i → bins ``[i*chunk_size:(i+1)*chunk_size]`` via mmap."""
    if not frames_npy.is_file():
        raise FileNotFoundError(frames_npy)
    packed = np.load(str(frames_npy), mmap_mode="r")
    if packed.ndim != 4 or packed.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected frames.npy (T,H,Wp,3|4), got {packed.shape} in {frames_npy}"
        )
    t_bins = int(packed.shape[0])
    cs = int(chunk_size)
    if cs <= 0:
        raise ValueError(f"chunk_size must be > 0, got {cs}")

    lo = max(int(frame_start), 0)
    hi = n_viz if frame_end is None or int(frame_end) < 0 else min(int(frame_end), n_viz)
    panels: list[np.ndarray] = []
    for i in range(lo, hi):
        t0 = i * cs
        t1 = min(t0 + cs, t_bins)
        panels.append(
            packed_slice_to_spad_gray(packed, t0, t1, mode=mode, ch_order=ch_order)
        )
    return panels


def stitch_panels(
    recon: np.ndarray,
    pose: np.ndarray,
    heat: np.ndarray,
    *,
    spad: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Left→right: [spad |] recon | pose (white) | heatmap.

    Returns ``(row_bgr, pose_x0)`` where ``pose_x0`` is the pose panel left edge.
    """
    r = _to_bgr_u8(recon)
    h = int(r.shape[0])
    parts: list[np.ndarray] = []
    if spad is not None:
        parts.append(_resize_to_height(_to_bgr_u8(spad), h))
    parts.append(r)
    pose_x0 = int(sum(int(p.shape[1]) for p in parts))
    parts.append(_resize_to_height(_to_bgr_u8(pose, alpha_bg=(255, 255, 255)), h))
    parts.append(_resize_to_height(_to_bgr_u8(heat, alpha_bg=None), h))
    return np.concatenate(parts, axis=1), pose_x0


def _pad_to_width(img: np.ndarray, width: int, *, fill: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    if w > width:
        return img[:, :width]
    out = np.full((h, width, 3), fill, dtype=np.uint8)
    out[:, :w] = img
    return out


def stack_ppb_hire(
    ppb_row: np.ndarray, hire_row: np.ndarray, *, pose_x0: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Top=PPB row, bottom=HIRE row; pad to common width.

    Returns stacked frame and pose-panel top-left origins ``[(x,y), ...]``.
    """
    w = max(int(ppb_row.shape[1]), int(hire_row.shape[1]))
    top = _pad_to_width(ppb_row, w)
    bot = _pad_to_width(hire_row, w)
    top_h = int(top.shape[0])
    stacked = np.concatenate([top, bot], axis=0)
    origins = [(int(pose_x0), 0), (int(pose_x0), top_h)]
    return stacked, origins


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
    pose_origins: list[tuple[int, int]] | None = None,
) -> list[np.ndarray]:
    """Insert interpolated frames on slow transitions; keep constant-time steps at output fps.

    During slowed source frames, draws playback speed (e.g. ``0.02x``) on each pose panel.
    """
    if not panels:
        return []
    factor = max(int(slow_factor), 1)
    interp = str(interp).strip().lower()
    if interp not in {"linear", "hold"}:
        raise ValueError(f"interp must be linear|hold, got {interp!r}")
    speed_text = format_playback_speed(factor)
    origins = list(pose_origins or [])

    def _emit(frame: np.ndarray, src_idx: int) -> np.ndarray:
        out_fr = np.ascontiguousarray(frame.copy())
        if origins and frame_in_slow(src_idx, slow_ranges):
            for x, y in origins:
                draw_speed_label(out_fr, speed_text, x=x, y=y)
        return out_fr

    out: list[np.ndarray] = [_emit(panels[0], 0)]
    for i in range(len(panels) - 1):
        a, b = panels[i], panels[i + 1]
        n_steps = factor if frame_in_slow(i, slow_ranges) else 1
        if n_steps <= 1:
            out.append(_emit(b, i + 1))
            continue
        for s in range(1, n_steps + 1):
            t = s / float(n_steps)
            if interp == "hold" or s == n_steps:
                fr = b if s == n_steps else a
            else:
                fr = lerp_bgr(a, b, t)
            src = i + 1 if s == n_steps else i
            out.append(_emit(fr, src))
    return out


def count_viz_frames(method_dir: Path) -> int:
    recon_paths = list_frame_paths(method_dir / "recon")
    pose_paths = list_frame_paths(method_dir / "pose")
    heat_paths = list_frame_paths(method_dir / "heatmap")
    return min(len(recon_paths), len(pose_paths), len(heat_paths))


def load_stitched_sequence(
    method_dir: Path,
    *,
    frame_start: int,
    frame_end: int | None,
    spad_panels: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Load stitched rows. Returns ``(panels, pose_x0)``."""
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
    if spad_panels is not None and len(spad_panels) != (hi - lo):
        raise ValueError(
            f"spad_panels length {len(spad_panels)} != viz window {hi - lo}"
        )

    panels: list[np.ndarray] = []
    pose_x0 = 0
    for j, i in enumerate(range(lo, hi)):
        recon = cv2.imread(str(recon_paths[i]), cv2.IMREAD_UNCHANGED)
        pose = cv2.imread(str(pose_paths[i]), cv2.IMREAD_UNCHANGED)
        heat = cv2.imread(str(heat_paths[i]), cv2.IMREAD_UNCHANGED)
        if recon is None or pose is None or heat is None:
            raise RuntimeError(f"Failed to read frame index {i} under {method_dir}")
        spad = spad_panels[j] if spad_panels is not None else None
        row, pose_x0 = stitch_panels(recon, pose, heat, spad=spad)
        panels.append(row)
    return panels, pose_x0


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


def load_combined_sequence(
    sample_dir: Path,
    *,
    frame_start: int,
    frame_end: int | None,
    data_root: Path | None,
    chunk_size: int,
    spad_mode: str,
    use_spad: bool,
    packed_ch_order: str = "RGB",
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Per frame: top=PPB (qnn) row, bottom=HIRE row; optional raw SPAD on the left.

    Returns ``(frames, pose_origins)`` with pose top-lefts for speed-label overlay.
    """
    ppb_dir = sample_dir / "qnn"
    hire_dir = sample_dir / "hire"
    if not ppb_dir.is_dir():
        raise FileNotFoundError(f"Missing PPB/qnn dumps: {ppb_dir}")
    if not hire_dir.is_dir():
        raise FileNotFoundError(f"Missing HIRE dumps: {hire_dir}")

    n_ppb = count_viz_frames(ppb_dir)
    n_hire = count_viz_frames(hire_dir)
    n_viz = min(n_ppb, n_hire)
    if n_viz <= 0:
        raise ValueError(f"No overlapping frames for {sample_dir.name}")

    spad_panels: list[np.ndarray] | None = None
    if use_spad:
        if data_root is None:
            raise ValueError("data_root is required when using SPAD panel")
        frames_npy = Path(data_root) / sample_dir.name / FRAMES_NPY
        spad_panels = load_spad_panel_sequence(
            frames_npy,
            frame_start=frame_start,
            frame_end=frame_end,
            n_viz=n_viz,
            chunk_size=chunk_size,
            mode=spad_mode,
            ch_order=packed_ch_order,
        )

    ppb_rows, pose_x0 = load_stitched_sequence(
        ppb_dir, frame_start=frame_start, frame_end=frame_end, spad_panels=spad_panels
    )
    hire_rows, pose_x0_hire = load_stitched_sequence(
        hire_dir, frame_start=frame_start, frame_end=frame_end, spad_panels=spad_panels
    )
    if pose_x0_hire != pose_x0:
        print(
            f"  warn: {sample_dir.name} pose_x0 qnn={pose_x0} hire={pose_x0_hire}; using qnn",
            flush=True,
        )
    n = min(len(ppb_rows), len(hire_rows))
    if len(ppb_rows) != len(hire_rows):
        print(
            f"  warn: {sample_dir.name} ppb_frames={len(ppb_rows)} hire_frames={len(hire_rows)}; "
            f"using first {n}",
            flush=True,
        )
    frames: list[np.ndarray] = []
    pose_origins: list[tuple[int, int]] = []
    for i in range(n):
        stacked, origins = stack_ppb_hire(ppb_rows[i], hire_rows[i], pose_x0=pose_x0)
        frames.append(stacked)
        if not pose_origins:
            pose_origins = origins
    return frames, pose_origins


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Vis/0730 → PPB (top) + HIRE (bottom) videos with optional slow-mo"
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
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Packed frames.npy root (same as vis_0730_pose --data_root)",
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Bins per viz frame (must match vis_0730_pose; default 320)",
    )
    ap.add_argument(
        "--spad_mode",
        type=str,
        default="last",
        choices=["last", "binary", "any", "sum"],
        help="SPAD panel: last/binary=single-bin 0/255; any=OR over chunk (often white); sum=count/T",
    )
    ap.add_argument(
        "--no-spad",
        action="store_true",
        help="Skip raw SPAD left column",
    )
    ap.add_argument(
        "--packed_ch_order",
        type=str,
        default="RGB",
        choices=["RGB", "BGR"],
        help="Channel order for C=3 VisionSIM packed frames (same as export)",
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
        "--slow_range",
        type=str,
        default="",
        help="Slow-mo source ranges, e.g. '40-80' or '10-30,100-140' (also accepts Chinese ，)",
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

    slow_ranges = parse_slow_ranges(args.slow_ranges)
    sample_dirs = discover_samples(vis_root, args.samples)
    frame_end = None if int(args.frame_end) < 0 else int(args.frame_end)

    use_spad = not bool(args.no_spad)
    print(
        f"vis_root={vis_root}\n"
        f"out_dir={out_dir}\n"
        f"samples={[p.name for p in sample_dirs]}\n"
        f"layout=PPB(top) / HIRE(bottom); each row = "
        f"{'spad|' if use_spad else ''}recon|pose|heatmap\n"
        f"data_root={args.data_root} chunk_size={args.chunk_size} "
        f"spad_mode={args.spad_mode} use_spad={use_spad}\n"
        f"frame_start={args.frame_start} frame_end={frame_end}\n"
        f"slow_ranges={slow_ranges} slow_factor={args.slow_factor} "
        f"({format_playback_speed(int(args.slow_factor))} on pose) interp={args.interp}\n"
        f"fps={args.fps}",
        flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for sample_dir in sample_dirs:
        sample = sample_dir.name
        out_path = out_dir / f"{sample}.mp4"
        if out_path.is_file() and not args.overwrite:
            print(f"skip (exists): {out_path}", flush=True)
            continue

        panels, pose_origins = load_combined_sequence(
            sample_dir,
            frame_start=int(args.frame_start),
            frame_end=frame_end,
            data_root=Path(args.data_root),
            chunk_size=int(args.chunk_size),
            spad_mode=str(args.spad_mode),
            use_spad=use_spad,
            packed_ch_order=str(args.packed_ch_order),
        )
        lo = max(int(args.frame_start), 0)
        rel_slow = [(max(a - lo, 0), max(b - lo, 0)) for a, b in slow_ranges]
        rel_slow = [(a, b) for a, b in rel_slow if b > a]
        expanded = expand_with_slowmo(
            panels,
            slow_ranges=rel_slow,
            slow_factor=int(args.slow_factor),
            interp=str(args.interp),
            pose_origins=pose_origins,
        )
        print(
            f"{sample}: src_frames={len(panels)} → out_frames={len(expanded)} → {out_path}",
            flush=True,
        )
        write_mp4(expanded, out_path, fps=float(args.fps))
        n_written += 1

    print(f"Done. wrote={n_written} → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
