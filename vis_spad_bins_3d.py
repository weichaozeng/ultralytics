#!/usr/bin/env python3
"""3D spatiotemporal scatter from ``vis_spad_bins.py`` binary exports.

Loads ``bin_XXXXXXX.png`` / ``.npy`` (0/1 or 0/255) and draws a 3D scatter.

Modes
-----
- ``full`` (default): both 0 (black) and 1 (white) voxels; xy plane upright, t along floor
- ``hits``: only photon hits (value>0), colored by time

Full Bayer volumes are huge (H×W×T); use ``--stride_xy`` / ``--max_points``.

Examples
--------
python ultralytics/vis_spad_bins_3d.py \\
  --in_dir /tmp/spad_bins_bayer \\
  --mode full --stride_xy 4 \\
  --save /tmp/spad_cube.png --no_show
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
        "--mode",
        type=str,
        default="full",
        choices=["full", "hits"],
        help="full=0/1 as black/white; hits=only value>0 (colored by t)",
    )
    ap.add_argument(
        "--stride_xy",
        type=int,
        default=2,
        help="Spatial stride when collecting voxels (full mode; also applied in hits). "
        "Bayer 1024² often needs 4–8.",
    )
    ap.add_argument(
        "--max_points",
        type=int,
        default=400_000,
        help="Random subsample if more points than this (0 = keep all)",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for subsample")
    ap.add_argument(
        "--point_size",
        type=float,
        default=0.2,
        help="Scatter marker size (matplotlib s)",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.08,
        help="Marker alpha",
    )
    ap.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Color by time (hits mode only)",
    )
    ap.add_argument(
        "--bg",
        type=str,
        default="none",
        help="Figure / axes background; 'none' = transparent (default)",
    )
    ap.add_argument(
        "--elev",
        type=float,
        default=18.0,
        help="View elevation; xy stands upright with default mapping",
    )
    ap.add_argument(
        "--azim",
        type=float,
        default=-70.0,
        help="View azimuth; t runs along the floor / depth",
    )
    ap.add_argument(
        "--color0",
        type=str,
        default="#000000",
        help="Color for binary 0 (default black)",
    )
    ap.add_argument(
        "--color1",
        type=str,
        default="#ffffff",
        help="Color for binary 1 / photon (default white)",
    )
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--no_show", action="store_true")
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
    """Return bool HW: True = photon (1/255), False = empty (0)."""
    if path.suffix.lower() == ".npy":
        img = np.load(path)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D binary map, got {path} shape={getattr(img, 'shape', None)}")
    return np.asarray(img) > 0


def _collect_voxels(
    files: list[tuple[int, Path]],
    *,
    mode: str,
    stride_xy: int,
    t_as_index: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, t, v where v is 0/1 binary value."""
    if stride_xy < 1:
        raise ValueError("--stride_xy must be >= 1")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    vs: list[np.ndarray] = []

    for i, (bin_idx, path) in enumerate(files):
        mask = _load_binary_hw(path)
        if stride_xy > 1:
            mask = mask[::stride_xy, ::stride_xy]
        t_val = float(i if t_as_index else bin_idx)

        if mode == "hits":
            yy, xx = np.nonzero(mask)
            if xx.size == 0:
                continue
            val = np.ones(xx.shape, dtype=np.uint8)
        else:
            # full: every voxel after spatial stride
            h, w = mask.shape
            yy, xx = np.mgrid[0:h, 0:w]
            yy = yy.ravel()
            xx = xx.ravel()
            val = mask.ravel().astype(np.uint8)

        # Map strided coords back to original pixel units for axis readability
        scale = float(stride_xy)
        xs.append((xx.astype(np.float32) * scale))
        ys.append((yy.astype(np.float32) * scale))
        ts.append(np.full(xx.shape, t_val, dtype=np.float32))
        vs.append(val.astype(np.uint8))

    if not xs:
        raise RuntimeError("No voxels collected from the loaded binary frames")

    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ts),
        np.concatenate(vs),
    )


def _subsample(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    v: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    if max_points <= 0 or n <= max_points:
        return x, y, t, v
    rng = np.random.default_rng(int(seed))
    sel = rng.choice(n, size=int(max_points), replace=False)
    return x[sel], y[sel], t[sel], v[sel]


def main() -> None:
    args = _parse_args()
    files = _list_bin_files(args.in_dir, args.glob)
    print(f"found {len(files)} frames in {args.in_dir}")
    print(f"bin index range: {files[0][0]} … {files[-1][0]}")
    print(f"mode={args.mode} stride_xy={args.stride_xy}")

    x, y, t, v = _collect_voxels(
        files,
        mode=str(args.mode),
        stride_xy=int(args.stride_xy),
        t_as_index=bool(args.t_as_index),
    )
    n_all = int(x.shape[0])
    n0 = int((v == 0).sum())
    n1 = int((v == 1).sum())
    x, y, t, v = _subsample(
        x, y, t, v, max_points=int(args.max_points), seed=int(args.seed)
    )
    print(
        f"voxels: {n_all:,} (0={n0:,}, 1={n1:,}) → plot {int(x.shape[0]):,} "
        f"(0={int((v == 0).sum()):,}, 1={int((v == 1).sum()):,})"
    )

    import matplotlib

    if args.save is not None and args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    face = "none" if str(args.bg).lower() in {"none", "transparent"} else args.bg
    fig = plt.figure(figsize=tuple(args.figsize), facecolor=face)
    ax = fig.add_subplot(111, projection="3d", facecolor=face)

    # Orientation: xy image plane upright (Z=y up, X=x), t along floor depth (Y=t).
    px, py, pz = x, t, y

    if args.mode == "full":
        cmap = ListedColormap([args.color0, args.color1])
        ax.scatter(
            px,
            py,
            pz,
            c=v,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=float(args.point_size),
            alpha=float(args.alpha),
            linewidths=0,
        )
    else:
        ax.scatter(
            px,
            py,
            pz,
            c=t,
            cmap=args.cmap,
            s=float(args.point_size),
            alpha=float(args.alpha),
            linewidths=0,
        )

    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    # Image row increases downward → flip vertical axis for natural upright view.
    ax.invert_zaxis()

    # Cube only: no axes, ticks, panes, or colorbar.
    ax.set_axis_off()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0.0))
        axis.pane.set_alpha(0.0)
        axis.line.set_color((1, 1, 1, 0.0))
    fig.tight_layout(pad=0)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        # Transparent PNG needs facecolor/edgecolor='none' + transparent=True
        fig.savefig(
            args.save,
            dpi=int(args.dpi),
            bbox_inches="tight",
            facecolor=face,
            edgecolor="none",
            transparent=(face == "none"),
        )
        print(f"saved {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
