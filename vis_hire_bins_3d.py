#!/usr/bin/env python3
"""3D cubes from ``vis_hire_bins.py`` exports (out / s_raw / n_slow).

- ``s_raw`` / ``n_slow``: dense per-bin cube (same layout as ``vis_spad_bins_3d``),
  colored by map value (turbo), alpha=0.4, no axes / colorbar.
- ``out``: keep every ``--out_emit``-th bin (default 320), place slices at their
  true bin index so gaps along ``t`` show the downsampled / lower frame-rate feel.

Examples
--------
python ultralytics/vis_hire_bins_3d.py \\
  --in_dir /tmp/hire_bins \\
  --save_dir /tmp/hire_bins_3d \\
  --no_show

# Only out, with its own alpha / stride / point size
python ultralytics/vis_hire_bins_3d.py \\
  --in_dir /tmp/hire_bins \\
  --save_dir /tmp/hire_bins_3d \\
  --which out --out_emit 320 \\
  --out_alpha 0.25 --out_stride_xy 4 --out_point_size 0.5 \\
  --no_show
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


_BIN_RE = re.compile(r"^(?P<prefix>.+?)_(?P<idx>\d+)\.(?P<ext>png|npy)$", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3D cubes for HIRE bin maps (out/s_raw/n_slow)")
    ap.add_argument(
        "--in_dir",
        type=Path,
        required=True,
        help="Root from vis_hire_bins.py (contains out/, s_raw/, n_slow/)",
    )
    ap.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Where to write *_3d.png (default: in_dir)",
    )
    ap.add_argument(
        "--which",
        type=str,
        default="s_raw,n_slow,out",
        help="Comma-separated: s_raw,n_slow,out",
    )
    ap.add_argument(
        "--out_emit",
        type=int,
        default=320,
        help="Temporal downsample for out: keep bins where index %% emit == first%%emit "
        "(default 320). Slices stay at true bin t so gaps show lower rate.",
    )
    ap.add_argument(
        "--out_slab",
        type=int,
        default=1,
        help="Thickness of each out slice along t in bin units (default 1)",
    )
    ap.add_argument(
        "--out_alpha",
        type=float,
        default=0.4,
        help="Marker alpha for out cube only (default 0.4; s_raw/n_slow use --alpha)",
    )
    ap.add_argument(
        "--out_stride_xy",
        type=int,
        default=0,
        help="Spatial stride for out only; 0 = use --stride_xy",
    )
    ap.add_argument(
        "--out_point_size",
        type=float,
        default=0.0,
        help="Point size for out only; 0 = use --point_size",
    )
    ap.add_argument(
        "--out_max_points",
        type=int,
        default=0,
        help="Max points for out only; 0 = use --max_points",
    )
    ap.add_argument("--stride_xy", type=int, default=2, help="Spatial stride for s_raw/n_slow (and out if --out_stride_xy 0)")
    ap.add_argument("--max_points", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--point_size", type=float, default=0.2, help="Marker size for s_raw/n_slow (and out if --out_point_size 0)")
    ap.add_argument("--alpha", type=float, default=0.4, help="Marker alpha for s_raw/n_slow (default 0.4)")
    ap.add_argument("--cmap", type=str, default="turbo", help="Colormap for continuous values")
    ap.add_argument(
        "--n_slow_vmax",
        type=float,
        default=160.0,
        help="Color scale max for n_slow (default hire_slow_bins=160)",
    )
    ap.add_argument(
        "--s_raw_percentile",
        type=float,
        default=99.5,
        help="vmax = this percentile of s_raw (0 = use --s_raw_vmax)",
    )
    ap.add_argument("--s_raw_vmax", type=float, default=0.0, help="Fixed s_raw vmax if >0")
    ap.add_argument("--out_vmin", type=float, default=0.0)
    ap.add_argument("--out_vmax", type=float, default=1.0)
    ap.add_argument("--bg", type=str, default="none")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-70.0)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--figsize", type=float, nargs=2, default=(9.0, 7.0))
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument(
        "--t_as_index",
        action="store_true",
        help="Use 0..N-1 as t instead of filename bin indices (s_raw/n_slow only)",
    )
    return ap.parse_args()


def _list_bin_files(in_dir: Path) -> list[tuple[int, Path]]:
    if not in_dir.is_dir():
        raise FileNotFoundError(in_dir)
    files: list[tuple[int, Path]] = []
    for path in sorted(in_dir.iterdir()):
        if not path.is_file():
            continue
        m = _BIN_RE.match(path.name)
        if m is None or m.group("ext").lower() != "npy":
            continue
        files.append((int(m.group("idx")), path))
    files.sort(key=lambda x: x[0])
    if not files:
        raise FileNotFoundError(f"No bin_XXXXXXX.npy under {in_dir}")
    return files


def _load_hw(path: Path) -> np.ndarray:
    img = np.load(path)
    if img.ndim != 2:
        raise ValueError(f"Expected (H,W), got {path} shape={img.shape}")
    return img.astype(np.float32, copy=False)


def _collect_float_voxels(
    files: list[tuple[int, Path]],
    *,
    stride_xy: int,
    t_as_index: bool,
    t_override: list[float] | None = None,
    slab: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, t, value for every strided pixel."""
    if stride_xy < 1:
        raise ValueError("--stride_xy must be >= 1")
    if slab < 1:
        raise ValueError("--out_slab must be >= 1")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    vs: list[np.ndarray] = []

    for i, (bin_idx, path) in enumerate(files):
        arr = _load_hw(path)
        if stride_xy > 1:
            arr = arr[::stride_xy, ::stride_xy]
        h, w = arr.shape
        yy, xx = np.mgrid[0:h, 0:w]
        xx = (xx.ravel().astype(np.float32) * float(stride_xy))
        yy = (yy.ravel().astype(np.float32) * float(stride_xy))
        val = arr.ravel().astype(np.float32)

        if t_override is not None:
            t0 = float(t_override[i])
        elif t_as_index:
            t0 = float(i)
        else:
            t0 = float(bin_idx)

        if slab == 1:
            xs.append(xx)
            ys.append(yy)
            ts.append(np.full(xx.shape, t0, dtype=np.float32))
            vs.append(val)
        else:
            for dt in range(slab):
                xs.append(xx)
                ys.append(yy)
                ts.append(np.full(xx.shape, t0 + float(dt), dtype=np.float32))
                vs.append(val)

    if not xs:
        raise RuntimeError("No voxels collected")
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


def _select_out_files(
    files: list[tuple[int, Path]],
    *,
    emit: int,
) -> list[tuple[int, Path]]:
    """Keep bins on an emit grid (true indices), so gaps remain along t."""
    if emit < 1:
        raise ValueError("--out_emit must be >= 1")
    if not files:
        return []
    phase = int(files[0][0]) % int(emit)
    selected = [(idx, p) for idx, p in files if int(idx) % int(emit) == phase]
    if not selected:
        # Fallback: every emit-th file in sorted order, placed at true indices
        selected = files[::emit]
    return selected


def _process_channel(
    name: str,
    files: list[tuple[int, Path]],
    args: argparse.Namespace,
    save_dir: Path,
) -> None:
    t_as_index = bool(args.t_as_index)
    slab = 1
    use_files = files

    if name == "out":
        use_files = _select_out_files(files, emit=int(args.out_emit))
        slab = int(args.out_slab)
        t_as_index = False  # keep true bin indices → empty gaps along t
        stride_xy = int(args.out_stride_xy) if int(args.out_stride_xy) > 0 else int(args.stride_xy)
        point_size = float(args.out_point_size) if float(args.out_point_size) > 0 else float(args.point_size)
        alpha = float(args.out_alpha)
        max_points = int(args.out_max_points) if int(args.out_max_points) > 0 else int(args.max_points)
        print(
            f"[{name}] emit={args.out_emit}: {len(files)} bins → {len(use_files)} slices "
            f"(t gaps show {args.out_emit}× downsample) "
            f"alpha={alpha:g} stride_xy={stride_xy} point_size={point_size:g}"
        )
        if use_files:
            print(f"[{name}] slice indices: {use_files[0][0]} … {use_files[-1][0]}")
    else:
        stride_xy = int(args.stride_xy)
        point_size = float(args.point_size)
        alpha = float(args.alpha)
        max_points = int(args.max_points)
        print(
            f"[{name}] {len(use_files)} bins, dense along t "
            f"alpha={alpha:g} stride_xy={stride_xy} point_size={point_size:g}"
        )

    if not use_files:
        print(f"[{name}] skip (no frames)")
        return

    x, y, t, v = _collect_float_voxels(
        use_files,
        stride_xy=stride_xy,
        t_as_index=t_as_index,
        slab=slab,
    )
    n_all = int(x.shape[0])
    x, y, t, v = _subsample(
        x, y, t, v, max_points=max_points, seed=int(args.seed)
    )
    print(f"[{name}] voxels {n_all:,} → plot {int(x.shape[0]):,}")

    if name == "n_slow":
        vmin, vmax = 0.0, float(args.n_slow_vmax)
    elif name == "s_raw":
        if float(args.s_raw_vmax) > 0:
            vmax = float(args.s_raw_vmax)
        else:
            vmax = float(np.nanpercentile(v, float(args.s_raw_percentile)))
        vmin, vmax = 0.0, max(vmax, 1e-6)
    else:  # out
        vmin, vmax = float(args.out_vmin), float(args.out_vmax)
        # If values look unnormalized, stretch by percentile for display
        if float(np.nanmax(v)) > 1.5 and vmax <= 1.0 + 1e-6:
            vmax = float(np.nanpercentile(v, 99.5))
            vmin = 0.0

    # Match overall t extent of dense cube when rendering out with gaps:
    # set t limits from full file list if available.
    t_lim = None
    if name == "out" and len(files) >= 2 and not args.t_as_index:
        t_lim = (float(files[0][0]), float(files[-1][0]))

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    face = "none" if str(args.bg).lower() in {"none", "transparent"} else args.bg
    fig = plt.figure(figsize=tuple(args.figsize), facecolor=face)
    ax = fig.add_subplot(111, projection="3d", facecolor=face)
    ax.scatter(
        x,
        t,
        y,
        c=v,
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        s=float(point_size),
        alpha=float(alpha),
        linewidths=0,
    )
    if t_lim is not None:
        ax.set_ylim(t_lim[0], t_lim[1])
    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    ax.invert_zaxis()
    ax.set_axis_off()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0.0))
        axis.pane.set_alpha(0.0)
        axis.line.set_color((1, 1, 1, 0.0))
    fig.tight_layout(pad=0)

    out_path = save_dir / f"{name}_3d.png"
    fig.savefig(
        out_path,
        dpi=int(args.dpi),
        bbox_inches="tight",
        facecolor=face,
        edgecolor="none",
        transparent=(face == "none"),
    )
    print(f"saved {out_path}  vmin/vmax={vmin:g}/{vmax:g}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = _parse_args()
    in_root = args.in_dir
    save_dir = args.save_dir if args.save_dir is not None else in_root
    save_dir.mkdir(parents=True, exist_ok=True)

    which = [w.strip() for w in str(args.which).split(",") if w.strip()]
    for name in which:
        if name not in {"out", "s_raw", "n_slow"}:
            raise ValueError(f"Unknown channel {name!r}; expected out/s_raw/n_slow")
        sub = in_root / name
        files = _list_bin_files(sub)
        print(f"\n=== {name} ===")
        print(f"found {len(files)} maps in {sub}  bins {files[0][0]}…{files[-1][0]}")
        _process_channel(name, files, args, save_dir)

    print(f"\nDone → {save_dir}")


if __name__ == "__main__":
    main()
