#!/usr/bin/env python3
"""3D spatiotemporal scatter from ``vis_spad_bins.py`` binary exports.

Loads ``bin_XXXXXXX.png`` / ``.npy`` (0/1 or 0/255), collects photon hits as
``(x, y, t)``, and draws a 3D scatter (optional save).

Examples
--------
# Interactive view
python ultralytics/vis_spad_bins_3d.py --in_dir /tmp/spad_bins

# Dense Bayer: low alpha + small markers
python ultralytics/vis_spad_bins_3d.py \\
  --in_dir /tmp/spad_bins_bayer \\
  --alpha 0.03 --point_size 0.1 \\
  --max_points 250000 \\
  --save /tmp/spad_bins_3d.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


_BIN_RE = re.compile(r"^(?P<prefix>.+?)_(?P<idx>\d+)\.(?P<ext>png|npy)$", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3D scatter of binary SPAD bins (x, y, t)")
    ap.add_argument("--in_dir", type=Path, required=True, help="Folder from vis_spad_bins.py")
    ap.add_argument(
        "--glob",
        type=str,
        default="*",
        help="Optional filename glob under in_dir (default: all bin_*.png/npy)",
    )
    ap.add_argument(
        "--max_points",
        type=int,
        default=200_000,
        help="Random subsample if more hits than this (0 = keep all)",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for subsample")
    ap.add_argument(
        "--point_size",
        type=float,
        default=0.15,
        help="Scatter marker size (matplotlib s); keep small for dense Bayer",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Marker alpha; default is low for dense Bayer hit clouds",
    )
    ap.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Color by time index",
    )
    ap.add_argument(
        "--elev",
        type=float,
        default=25.0,
        help="3D view elevation (degrees)",
    )
    ap.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="3D view azimuth (degrees)",
    )
    ap.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save figure instead of (or in addition to) interactive show",
    )
    ap.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open interactive window (use with --save)",
    )
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--figsize", type=float, nargs=2, default=(9.0, 7.0))
    ap.add_argument(
        "--t_as_index",
        action="store_true",
        help="Use 0..N-1 as t instead of filename bin indices",
    )
    return ap.parse_args()


def _list_bin_files(in_dir: Path, pattern: str) -> list[tuple[int, Path]]:
    if not in_dir.is_dir():
        raise FileNotFoundError(in_dir)

    files: list[tuple[int, Path]] = []
    for path in sorted(in_dir.glob(pattern)):
        if not path.is_file():
            continue
        m = _BIN_RE.match(path.name)
        if m is None:
            continue
        if m.group("ext").lower() not in ("png", "npy"):
            continue
        files.append((int(m.group("idx")), path))

    files.sort(key=lambda x: x[0])
    if not files:
        raise FileNotFoundError(
            f"No bin_XXXXXXX.png/.npy under {in_dir} (glob={pattern!r})"
        )
    return files


def _load_binary_hw(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        img = np.load(path)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D binary map, got {path} shape={getattr(img, 'shape', None)}")
    return (np.asarray(img) > 0)


def _collect_hits(
    files: list[tuple[int, Path]],
    *,
    t_as_index: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []

    for i, (bin_idx, path) in enumerate(files):
        mask = _load_binary_hw(path)
        yy, xx = np.nonzero(mask)
        if xx.size == 0:
            continue
        t_val = float(i if t_as_index else bin_idx)
        xs.append(xx.astype(np.float32))
        ys.append(yy.astype(np.float32))
        ts.append(np.full(xx.shape, t_val, dtype=np.float32))

    if not xs:
        raise RuntimeError("No photon hits found in the loaded binary frames")

    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ts),
    )


def _subsample(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    if max_points <= 0 or n <= max_points:
        return x, y, t
    rng = np.random.default_rng(int(seed))
    sel = rng.choice(n, size=int(max_points), replace=False)
    return x[sel], y[sel], t[sel]


def main() -> None:
    args = _parse_args()
    files = _list_bin_files(args.in_dir, args.glob)
    print(f"found {len(files)} frames in {args.in_dir}")
    print(f"bin index range: {files[0][0]} … {files[-1][0]}")

    x, y, t = _collect_hits(files, t_as_index=bool(args.t_as_index))
    n_all = int(x.shape[0])
    x, y, t = _subsample(x, y, t, max_points=int(args.max_points), seed=int(args.seed))
    print(f"hits: {n_all:,} → plot {int(x.shape[0]):,} points")

    import matplotlib

    if args.save is not None and args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=tuple(args.figsize))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        x,
        y,
        t,
        c=t,
        cmap=args.cmap,
        s=float(args.point_size),
        alpha=float(args.alpha),
        linewidths=0,
    )
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    ax.set_zlabel("t (bin)")
    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    # Image coords: y grows downward; flip for a more natural view.
    ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08, label="t")
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=int(args.dpi), bbox_inches="tight")
        print(f"saved {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
