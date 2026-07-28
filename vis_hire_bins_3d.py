#!/usr/bin/env python3
"""3D cubes from ``vis_hire_bins.py`` exports (out / s_raw / n_slow).

Memory-safe: streams frames with a fixed-size reservoir (``--max_points``),
mmap-loads ``.npy``, and frees each figure before the next channel.

- ``s_raw`` / ``n_slow``: dense per-bin cube, colored by value (turbo).
- ``out``: only ``--out_n_slices`` frames (default 5) at true ``t``, full t-extent
  kept so gaps show lower frame-rate while the volume stays cube-shaped.
- ``out_dense``: all ``out/`` bins dense along ``t`` (same ``--out_*`` knobs as ``out``).

Examples
--------
python ultralytics/vis_hire_bins_3d.py \\
  --in_dir /tmp/hire_bins \\
  --save_dir /tmp/hire_bins_3d \\
  --which s_raw,n_slow,out,out_dense \\
  --out_n_slices 5 --max_points 300000 --no_show
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import numpy as np


_BIN_RE = re.compile(r"^(?P<prefix>.+?)_(?P<idx>\d+)\.(?P<ext>png|npy)$", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3D cubes for HIRE bin maps (out/s_raw/n_slow)")
    ap.add_argument("--in_dir", type=Path, required=True)
    ap.add_argument("--save_dir", type=Path, default=None)
    ap.add_argument(
        "--which",
        type=str,
        default="s_raw,n_slow,out,out_dense",
        help="Comma-separated: s_raw,n_slow,out,out_dense",
    )
    ap.add_argument(
        "--out_n_slices",
        type=int,
        default=5,
        help="Sparse out: how many frames to keep (default 5); 0 = use --out_emit instead. "
        "out_dense ignores this",
    )
    ap.add_argument(
        "--out_emit",
        type=int,
        default=320,
        help="Legacy temporal stride for sparse out when --out_n_slices 0 (default 320); "
        "out_dense ignores this",
    )
    ap.add_argument("--out_slab", type=int, default=1)
    ap.add_argument("--out_alpha", type=float, default=0.4)
    ap.add_argument("--out_stride_xy", type=int, default=0, help="0 = use --stride_xy")
    ap.add_argument("--out_point_size", type=float, default=0.0, help="0 = use --point_size")
    ap.add_argument("--out_max_points", type=int, default=0, help="0 = use --max_points")
    ap.add_argument("--stride_xy", type=int, default=2)
    ap.add_argument(
        "--max_points",
        type=int,
        default=300_000,
        help="Hard cap on plotted points (reservoir). Always bounded — never loads full volume.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--point_size", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--cmap", type=str, default="turbo")
    ap.add_argument("--n_slow_vmax", type=float, default=160.0)
    ap.add_argument("--s_raw_percentile", type=float, default=99.5)
    ap.add_argument("--s_raw_vmax", type=float, default=0.0)
    ap.add_argument("--out_vmin", type=float, default=0.0)
    ap.add_argument("--out_vmax", type=float, default=1.0)
    ap.add_argument("--bg", type=str, default="none")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-70.0)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--figsize", type=float, nargs=2, default=(9.0, 7.0))
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--t_as_index", action="store_true")
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


def _load_hw_mmap(path: Path) -> np.ndarray:
    img = np.load(path, mmap_mode="r")
    if img.ndim != 2:
        raise ValueError(f"Expected (H,W), got {path} shape={img.shape}")
    return img


def _select_out_files(
    files: list[tuple[int, Path]],
    *,
    n_slices: int,
    emit: int,
) -> list[tuple[int, Path]]:
    """Pick sparse out frames at true bin indices (gaps show lower rate)."""
    if not files:
        return []
    if int(n_slices) > 0:
        n = int(n_slices)
        if len(files) <= n:
            return list(files)
        # Evenly spaced across the full sequence (include first & last).
        idxs = np.round(np.linspace(0, len(files) - 1, n)).astype(int)
        seen: set[int] = set()
        out: list[tuple[int, Path]] = []
        for i in idxs:
            ii = int(i)
            if ii in seen:
                continue
            seen.add(ii)
            out.append(files[ii])
        return out

    if emit < 1:
        raise ValueError("--out_emit must be >= 1 when --out_n_slices is 0")
    phase = int(files[0][0]) % int(emit)
    selected = [(idx, p) for idx, p in files if int(idx) % int(emit) == phase]
    return selected if selected else files[::emit]


class _Reservoir4:
    """Fixed-capacity reservoir for (x, y, t, v) float32 points."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("reservoir capacity must be >= 1")
        self.k = int(capacity)
        self.rng = np.random.default_rng(int(seed))
        self.x = np.empty(self.k, dtype=np.float32)
        self.y = np.empty(self.k, dtype=np.float32)
        self.t = np.empty(self.k, dtype=np.float32)
        self.v = np.empty(self.k, dtype=np.float32)
        self.filled = 0
        self.seen = 0

    def add_batch(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, v: np.ndarray) -> None:
        m = int(x.shape[0])
        if m == 0:
            return
        # Fill phase
        if self.filled < self.k:
            take = min(m, self.k - self.filled)
            sl = slice(self.filled, self.filled + take)
            self.x[sl] = x[:take]
            self.y[sl] = y[:take]
            self.t[sl] = t[:take]
            self.v[sl] = v[:take]
            self.filled += take
            self.seen += take
            if take == m:
                return
            x, y, t, v = x[take:], y[take:], t[take:], v[take:]
            m = int(x.shape[0])

        # Replacement phase (vectorized per-point)
        # For i-th new point, seen becomes seen+i+1; replace with p = k/seen.
        for start in range(0, m, 65536):
            end = min(start + 65536, m)
            xb = x[start:end]
            yb = y[start:end]
            tb = t[start:end]
            vb = v[start:end]
            b = int(xb.shape[0])
            seen0 = self.seen
            # indices into reservoir to maybe replace
            # j ~ U{0..seen_i-1}; keep if j < k
            seen_i = seen0 + np.arange(1, b + 1, dtype=np.int64)
            j = self.rng.integers(0, seen_i, endpoint=False, dtype=np.int64)
            mask = j < self.k
            if mask.any():
                slots = j[mask]
                self.x[slots] = xb[mask]
                self.y[slots] = yb[mask]
                self.t[slots] = tb[mask]
                self.v[slots] = vb[mask]
            self.seen = int(seen_i[-1])

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        n = self.filled
        return self.x[:n], self.y[:n], self.t[:n], self.v[:n], self.seen


def _xy_grid(h: int, w: int, stride_xy: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx.astype(np.float32) * float(stride_xy)).ravel()
    yy = (yy.astype(np.float32) * float(stride_xy)).ravel()
    return xx, yy


def _collect_reservoir(
    files: list[tuple[int, Path]],
    *,
    stride_xy: int,
    t_as_index: bool,
    slab: int,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Stream frames into a reservoir of size ``max_points`` (peak mem ~ O(max_points + H*W/stride²))."""
    if stride_xy < 1:
        raise ValueError("--stride_xy must be >= 1")
    if slab < 1:
        raise ValueError("--out_slab must be >= 1")
    if max_points < 1:
        raise ValueError("--max_points must be >= 1 (memory cap)")

    res = _Reservoir4(max_points, seed)
    xx_cache: np.ndarray | None = None
    yy_cache: np.ndarray | None = None
    cache_hw: tuple[int, int] | None = None

    for i, (bin_idx, path) in enumerate(files):
        arr = _load_hw_mmap(path)
        # Copy only the strided view into RAM (small)
        if stride_xy > 1:
            view = np.asarray(arr[::stride_xy, ::stride_xy], dtype=np.float32)
        else:
            view = np.asarray(arr, dtype=np.float32)
        h, w = view.shape
        if cache_hw != (h, w):
            xx_cache, yy_cache = _xy_grid(h, w, stride_xy)
            cache_hw = (h, w)
        assert xx_cache is not None and yy_cache is not None
        val = view.ravel()
        t0 = float(i if t_as_index else bin_idx)

        if slab == 1:
            res.add_batch(xx_cache, yy_cache, np.full(val.shape, t0, dtype=np.float32), val)
        else:
            for dt in range(slab):
                res.add_batch(
                    xx_cache,
                    yy_cache,
                    np.full(val.shape, t0 + float(dt), dtype=np.float32),
                    val,
                )
        # Drop frame ASAP
        del view, val, arr

    x, y, t, v, seen = res.arrays()
    if seen == 0:
        raise RuntimeError("No voxels collected")
    return x.copy(), y.copy(), t.copy(), v.copy(), seen


def _estimate_s_raw_vmax(files: list[tuple[int, Path]], *, stride_xy: int, percentile: float, seed: int) -> float:
    """Light pass: sample strided pixels from up to 32 frames for vmax."""
    rng = np.random.default_rng(int(seed) + 1)
    if not files:
        return 1.0
    pick = files if len(files) <= 32 else [files[i] for i in rng.choice(len(files), size=32, replace=False)]
    samples: list[np.ndarray] = []
    budget = 200_000
    per = max(1, budget // max(len(pick), 1))
    for _, path in pick:
        arr = _load_hw_mmap(path)
        view = np.asarray(arr[::stride_xy, ::stride_xy], dtype=np.float32).ravel()
        if view.size > per:
            view = view[rng.choice(view.size, size=per, replace=False)]
        samples.append(view)
        del arr
    if not samples:
        return 1.0
    cat = np.concatenate(samples)
    return float(max(np.nanpercentile(cat, float(percentile)), 1e-6))


def _process_channel(
    name: str,
    files: list[tuple[int, Path]],
    args: argparse.Namespace,
    save_dir: Path,
) -> None:
    t_as_index = bool(args.t_as_index)
    slab = 1
    use_files = files
    is_out_family = name in {"out", "out_dense"}

    if is_out_family:
        slab = int(args.out_slab)
        t_as_index = False
        stride_xy = int(args.out_stride_xy) if int(args.out_stride_xy) > 0 else int(args.stride_xy)
        point_size = float(args.out_point_size) if float(args.out_point_size) > 0 else float(args.point_size)
        alpha = float(args.out_alpha)
        max_points = int(args.out_max_points) if int(args.out_max_points) > 0 else int(args.max_points)
        if name == "out":
            use_files = _select_out_files(
                files,
                n_slices=int(args.out_n_slices),
                emit=int(args.out_emit),
            )
            if int(args.out_n_slices) > 0:
                how = f"n_slices={args.out_n_slices}"
            else:
                how = f"emit={args.out_emit}"
            print(
                f"[{name}] sparse {how}: {len(files)} → {len(use_files)} slices | "
                f"alpha={alpha:g} stride_xy={stride_xy} point_size={point_size:g} max_points={max_points}"
            )
        else:
            print(
                f"[{name}] dense: {len(use_files)} bins | "
                f"alpha={alpha:g} stride_xy={stride_xy} point_size={point_size:g} max_points={max_points}"
            )
    else:
        stride_xy = int(args.stride_xy)
        point_size = float(args.point_size)
        alpha = float(args.alpha)
        max_points = int(args.max_points)
        print(
            f"[{name}] {len(use_files)} bins | "
            f"alpha={alpha:g} stride_xy={stride_xy} point_size={point_size:g} max_points={max_points}"
        )

    if not use_files:
        print(f"[{name}] skip (no frames)")
        return

    # Resolve vmax before building the big scatter (s_raw needs a light pass).
    if name == "n_slow":
        vmin, vmax = 0.0, float(args.n_slow_vmax)
    elif name == "s_raw":
        if float(args.s_raw_vmax) > 0:
            vmax = float(args.s_raw_vmax)
        else:
            vmax = _estimate_s_raw_vmax(
                use_files,
                stride_xy=stride_xy,
                percentile=float(args.s_raw_percentile),
                seed=int(args.seed),
            )
        vmin = 0.0
    else:
        vmin, vmax = float(args.out_vmin), float(args.out_vmax)

    x, y, t, v, seen = _collect_reservoir(
        use_files,
        stride_xy=stride_xy,
        t_as_index=t_as_index,
        slab=slab,
        max_points=max_points,
        seed=int(args.seed),
    )
    print(f"[{name}] scanned ~{seen:,} voxels → plot {int(x.shape[0]):,} (cap={max_points})")

    if is_out_family and float(np.nanmax(v)) > 1.5 and vmax <= 1.0 + 1e-6:
        vmax = float(np.nanpercentile(v, 99.5))
        vmin = 0.0

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
        import matplotlib.pyplot as plt

        plt.show()

    # Release plot + point buffers before next channel
    plt.close(fig)
    plt.close("all")
    del fig, ax, x, y, t, v
    gc.collect()


def main() -> None:
    args = _parse_args()
    if int(args.max_points) < 1:
        raise ValueError("--max_points must be >= 1")
    in_root = args.in_dir
    save_dir = args.save_dir if args.save_dir is not None else in_root
    save_dir.mkdir(parents=True, exist_ok=True)

    # Agg backend for the whole run when --no_show (set once)
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    which = [w.strip() for w in str(args.which).split(",") if w.strip()]
    for name in which:
        if name not in {"out", "out_dense", "s_raw", "n_slow"}:
            raise ValueError(f"Unknown channel {name!r}; expected out/out_dense/s_raw/n_slow")
        # out_dense reads the same maps as out/
        sub = in_root / ("out" if name == "out_dense" else name)
        files = _list_bin_files(sub)
        print(f"\n=== {name} ===")
        print(f"found {len(files)} maps in {sub}  bins {files[0][0]}…{files[-1][0]}")
        _process_channel(name, files, args, save_dir)
        gc.collect()

    print(f"\nDone → {save_dir}")


if __name__ == "__main__":
    main()
