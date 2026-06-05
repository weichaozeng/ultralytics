# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stitch det_spad outputs (sum / ppb / vel / hyb) into side-by-side comparison images.

Expected layout from ``det_spad.py``::

    {save_dir}/{sample}/video00000/cube00000_t000000_000320_frame0000000_sum_recon.png
    {save_dir}/{sample}/video00000/cube00000_t000000_000320_frame0000000_hyb_overlay.png

For sample ``0428_wc/acq00002``, writes to ``0428_wc/acq00002_compare`` (sibling folder).

Example
-------
python ultralytics/vis_compare_det_spad.py \\
  --save_dir /path/to/det_spad_outputs \\
  --sample_glob "0428_wc/acq00002"
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

FNAME_RE = re.compile(
    r"^(?P<stem>cube\d+_t\d+_\d+_frame\d+)_(?P<pre>sum|ppb|vel|hyb)_(?P<kind>recon|overlay)\.png$",
    re.IGNORECASE,
)

ALL_METHODS = ("sum", "ppb", "vel", "hyb")

LABEL_COLORS = {
    "sum": (0, 255, 255),
    "ppb": (0, 255, 0),
    "vel": (255, 128, 0),
    "hyb": (255, 0, 255),
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Side-by-side comparison for det_spad preprocess outputs")
    ap.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Root directory that contains sample folders (same as det_spad --save_dir)",
    )
    ap.add_argument(
        "--in_dir",
        type=str,
        default=None,
        help="Single sample folder, e.g. 0428_wc/acq00002 (overrides --sample_glob)",
    )
    ap.add_argument(
        "--sample_glob",
        type=str,
        default="*/*",
        help="Glob under --save_dir for sample folders (default: */*)",
    )
    ap.add_argument(
        "--pre",
        type=str,
        default=",".join(ALL_METHODS),
        help="Comma-separated methods (sum,ppb,vel,hyb), left-to-right order",
    )
    ap.add_argument("--kinds", type=str, default="recon,overlay", help="Comma-separated: recon, overlay")
    ap.add_argument("--gap", type=int, default=8, help="Pixels between panels")
    ap.add_argument("--label_h", type=int, default=28, help="Header height for method labels")
    ap.add_argument("--font_scale", type=float, default=0.7)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing comparison PNGs")
    return ap.parse_args()


def _compare_out_dir(sample_dir: Path) -> Path:
    return sample_dir.parent / f"{sample_dir.name}_compare"


def _discover_samples(save_dir: Path, in_dir: str | None, sample_glob: str) -> list[Path]:
    if in_dir:
        path = Path(in_dir)
        if not path.is_absolute():
            path = save_dir / path
        if not path.is_dir():
            raise FileNotFoundError(f"Sample directory not found: {path}")
        return [path]
    samples = sorted(p for p in save_dir.glob(sample_glob) if p.is_dir())
    if not samples:
        raise FileNotFoundError(f"No sample folders matched: {save_dir}/{sample_glob}")
    return samples


def _label_panel(img_bgr: np.ndarray, text: str, color: tuple[int, int, int], label_h: int, font_scale: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    header = np.zeros((label_h, w, 3), dtype=np.uint8)
    cv2.putText(
        header,
        text.upper(),
        (8, int(label_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )
    return np.vstack([header, img_bgr])


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = max(int(round(w * target_h / h)), 1)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _stitch_row(
    panels: list[np.ndarray],
    *,
    gap: int,
    labels: list[str],
    colors: list[tuple[int, int, int]],
    label_h: int,
    font_scale: float,
) -> np.ndarray:
    target_h = max(p.shape[0] for p in panels)
    resized = [_resize_to_height(p, target_h) for p in panels]
    labeled = [
        _label_panel(img, lab, col, label_h, font_scale)
        for img, lab, col in zip(resized, labels, colors)
    ]
    sep = np.full((labeled[0].shape[0], gap, 3), 32, dtype=np.uint8)
    out = labeled[0]
    for nxt in labeled[1:]:
        out = np.hstack([out, sep, nxt])
    return out


def _index_video_dir(video_dir: Path, methods: list[str], kinds: list[str]) -> dict[tuple[str, str], dict[str, Path]]:
    """Map (stem, kind) -> {method: path}."""
    grouped: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for path in sorted(video_dir.glob("*.png")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        pre = m.group("pre").lower()
        kind = m.group("kind").lower()
        if pre not in methods or kind not in kinds:
            continue
        grouped[(m.group("stem"), kind)][pre] = path
    return grouped


def _missing_panel(h: int, w: int, text: str) -> np.ndarray:
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        panel,
        text,
        (16, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (64, 64, 64),
        2,
        cv2.LINE_AA,
    )
    return panel


def _build_compare(
    paths_by_method: dict[str, Path],
    methods: list[str],
    *,
    gap: int,
    label_h: int,
    font_scale: float,
) -> np.ndarray | None:
    imgs: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[tuple[int, int, int]] = []
    ref_h, ref_w = 256, 256

    for method in methods:
        path = paths_by_method.get(method)
        if path is None or not path.exists():
            imgs.append(_missing_panel(ref_h, ref_w, f"missing {method}"))
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                imgs.append(_missing_panel(ref_h, ref_w, f"read fail {method}"))
            else:
                ref_h, ref_w = img.shape[:2]
                imgs.append(img)
        labels.append(method)
        colors.append(LABEL_COLORS.get(method, (255, 255, 255)))

    if not imgs:
        return None
    return _stitch_row(imgs, gap=gap, labels=labels, colors=colors, label_h=label_h, font_scale=font_scale)


def _process_sample(sample_dir: Path, methods: list[str], kinds: list[str], args: argparse.Namespace) -> int:
    out_root = _compare_out_dir(sample_dir)
    n_written = 0
    video_dirs = sorted(p for p in sample_dir.iterdir() if p.is_dir() and p.name.startswith("video"))
    if not video_dirs:
        return 0

    for video_dir in video_dirs:
        grouped = _index_video_dir(video_dir, methods, kinds)
        if not grouped:
            continue
        out_video = out_root / video_dir.name
        out_video.mkdir(parents=True, exist_ok=True)

        for (stem, kind), paths_by_method in tqdm(
            sorted(grouped.items()),
            desc=f"{sample_dir.name}/{video_dir.name}",
            leave=False,
        ):
            out_path = out_video / f"{stem}_compare_{kind}.png"
            if out_path.exists() and not args.overwrite:
                continue
            mosaic = _build_compare(
                paths_by_method,
                methods,
                gap=int(args.gap),
                label_h=int(args.label_h),
                font_scale=float(args.font_scale),
            )
            if mosaic is None:
                continue
            cv2.imwrite(str(out_path), mosaic)
            n_written += 1
    return n_written


def main() -> None:
    args = _parse_args()
    save_dir = Path(args.save_dir)
    if not save_dir.is_dir():
        raise FileNotFoundError(f"save_dir not found: {save_dir}")

    methods = [x.strip().lower() for x in args.pre.split(",") if x.strip()]
    kinds = [x.strip().lower() for x in args.kinds.split(",") if x.strip()]
    invalid_pre = sorted(set(methods) - set(ALL_METHODS))
    invalid_kind = sorted(set(kinds) - {"recon", "overlay"})
    if invalid_pre:
        raise ValueError(f"Unsupported methods: {invalid_pre}")
    if invalid_kind:
        raise ValueError(f"Unsupported kinds: {invalid_kind}")

    samples = _discover_samples(save_dir, args.in_dir, args.sample_glob)
    total = 0
    for sample_dir in samples:
        n = _process_sample(sample_dir, methods, kinds, args)
        total += n
        print(f"{_compare_out_dir(sample_dir)}: wrote {n} comparison image(s)")
    print(f"Done. Total {total} image(s).")


if __name__ == "__main__":
    main()
