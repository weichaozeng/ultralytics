#!/usr/bin/env python3
"""Stack ``vis_hire_pose.py`` PNGs into a 3D image-plane cube.

Each pose visualization is placed as a textured plane. Filenames may have
uneven ``t0``; by default planes are laid out **uniformly** along ``t`` over
the full sequence extent (first ``t0`` → last ``t1``), with only
``--max_frames`` slices kept so equal gaps show a lower rate while the volume
stays cube-shaped. Pass ``--t_from_name`` to use raw filename ``t0`` instead.

Same camera convention as ``vis_spad_bins_3d`` / ``vis_hire_bins_3d``:
``X=x``, ``Y=t``, ``Z=y`` (xy upright, t along floor). No axes / colorbar.

Examples
--------
python ultralytics/vis_hire_pose_3d.py \\
  --in_dir /tmp/hire_pose/sample_name \\
  --save /tmp/hire_pose_3d.png \\
  --max_frames 5 --stride_xy 2 --alpha 0.2 --no_show
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import cv2
import numpy as np


_POSE_RE = re.compile(
    r"^cube(?P<video>\d+)_t(?P<t0>\d+)_(?P<t1>\d+)_frame(?P<frame>\d+)(?P<suffix>|_recon)\.png$",
    re.IGNORECASE,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="3D cube from vis_hire_pose PNGs (image planes)")
    ap.add_argument(
        "--in_dir",
        type=Path,
        required=True,
        help="Folder with cube*_t*_frame*.png (exclude *_recon by default)",
    )
    ap.add_argument("--save", type=Path, default=None, help="Output PNG (default: in_dir/pose_3d.png)")
    ap.add_argument(
        "--which",
        type=str,
        default="pose",
        choices=["pose", "recon"],
        help="pose = overlay PNG; recon = *_recon.png",
    )
    ap.add_argument(
        "--max_frames",
        type=int,
        default=5,
        help="Keep this many slices (default 5); 0 = all. Selection is even in "
        "file order; placement along t is uniform by default",
    )
    ap.add_argument(
        "--stride_xy",
        type=int,
        default=2,
        help="Spatial downsample of each image before placing as a plane",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Plane opacity (default 0.85)",
    )
    ap.add_argument(
        "--t_from_name",
        action="store_true",
        help="Place each plane at its filename t0 (may be uneven). "
        "Default: uniform spacing over the full t extent",
    )
    ap.add_argument(
        "--t_gap",
        type=float,
        default=0.0,
        help="If >0, place slices at 0, t_gap, 2*t_gap, … "
        "(overrides uniform / filename t). 0 = auto uniform (or t_from_name)",
    )
    ap.add_argument("--bg", type=str, default="none")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-70.0)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--figsize", type=float, nargs=2, default=(9.0, 7.0))
    ap.add_argument("--no_show", action="store_true")
    return ap.parse_args()


def _list_pose_pngs(in_dir: Path, *, which: str) -> list[tuple[int, int, Path]]:
    """Return ``(t0, frame_idx, path)`` sorted by t0 then frame."""
    if not in_dir.is_dir():
        raise FileNotFoundError(in_dir)
    want_recon = which == "recon"
    items: list[tuple[int, int, Path]] = []
    for path in sorted(in_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        m = _POSE_RE.match(path.name)
        if m is None:
            continue
        is_recon = m.group("suffix").lower() == "_recon"
        if is_recon != want_recon:
            continue
        items.append((int(m.group("t0")), int(m.group("frame")), path))
    items.sort(key=lambda x: (x[0], x[1]))
    if not items:
        raise FileNotFoundError(
            f"No matching PNGs under {in_dir} (which={which}). "
            "Expected cubeXXXXX_tTTTTTT_TTTTTT_frameFFFFFFF.png"
        )
    return items


def _select_even_slices(
    items: list[tuple[int, int, Path]],
    n: int,
) -> list[tuple[int, int, Path]]:
    """Evenly spaced subset including first & last when possible."""
    if n <= 0 or len(items) <= n:
        return list(items)
    idxs = np.round(np.linspace(0, len(items) - 1, n)).astype(int)
    seen: set[int] = set()
    out: list[tuple[int, int, Path]] = []
    for i in idxs:
        ii = int(i)
        if ii in seen:
            continue
        seen.add(ii)
        out.append(items[ii])
    return out


def _t_extent(items: list[tuple[int, int, Path]]) -> tuple[float, float]:
    """Full cube t span from first t0 to last t1 (fallback: last t0)."""
    t0 = float(items[0][0])
    m = _POSE_RE.match(items[-1][2].name)
    t_end = float(m.group("t1")) if m is not None else float(items[-1][0])
    return t0, max(t_end, float(items[-1][0]) + 1.0)


def _uniform_t_positions(
    n: int,
    t_lim: tuple[float, float] | None,
) -> list[float]:
    """Evenly space n planes over ``t_lim`` (inclusive)."""
    if n <= 0:
        return []
    if t_lim is None:
        return [float(i) for i in range(n)]
    t0, t1 = float(t_lim[0]), float(t_lim[1])
    if n == 1:
        return [t0]
    return [float(x) for x in np.linspace(t0, t1, n)]


def _load_rgb_plane(path: Path, *, stride_xy: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {path}")
    if stride_xy > 1:
        bgr = bgr[::stride_xy, ::stride_xy]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def main() -> None:
    args = _parse_args()
    if int(args.stride_xy) < 1:
        raise ValueError("--stride_xy must be >= 1")

    items_all = _list_pose_pngs(args.in_dir, which=str(args.which))
    t_lim = _t_extent(items_all) if len(items_all) >= 1 else None
    items = _select_even_slices(items_all, int(args.max_frames))
    if len(items) != len(items_all):
        extent_msg = (
            f", full extent {t_lim[0]:g}…{t_lim[1]:g}" if t_lim is not None else ""
        )
        print(
            f"max_frames={args.max_frames}: {len(items_all)} → {len(items)} slices{extent_msg}"
        )
    else:
        print(f"found {len(items)} slices in {args.in_dir} (which={args.which})")

    if float(args.t_gap) > 0:
        t_positions = [float(i * args.t_gap) for i in range(len(items))]
        place_mode = f"t_gap={args.t_gap:g}"
    elif bool(args.t_from_name):
        t_positions = [float(t0) for t0, _, _ in items]
        place_mode = "filename t0"
    else:
        t_positions = _uniform_t_positions(len(items), t_lim)
        place_mode = "uniform t"
    print(f"placement={place_mode}  t: {t_positions[0]:g} … {t_positions[-1]:g}")

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    face = "none" if str(args.bg).lower() in {"none", "transparent"} else args.bg
    fig = plt.figure(figsize=tuple(args.figsize), facecolor=face)
    ax = fig.add_subplot(111, projection="3d", facecolor=face)

    alpha = float(np.clip(args.alpha, 0.0, 1.0))

    for (t0, _frame, path), t_pos in zip(items, t_positions):
        rgb = _load_rgb_plane(path, stride_xy=int(args.stride_xy))
        h, w, _ = rgb.shape

        # Quad corners: X=x, Y=t, Z=y  (image row → Z, col → X)
        # One textured plane via many small quads would be huge; use plot_surface.
        xs = np.linspace(0.0, float(w - 1), w, dtype=np.float32)
        zs = np.linspace(0.0, float(h - 1), h, dtype=np.float32)
        xx, zz = np.meshgrid(xs, zs)
        yy = np.full_like(xx, float(t_pos), dtype=np.float32)

        # facecolors needs (H, W, 4)
        rgba = np.concatenate([rgb, np.full((h, w, 1), alpha, dtype=np.float32)], axis=-1)
        ax.plot_surface(
            xx,
            yy,
            zz,
            facecolors=rgba,
            rstride=1,
            cstride=1,
            shade=False,
            linewidth=0,
            antialiased=False,
        )
        del rgb, rgba, xx, yy, zz
        print(
            f"  + plane t={t_pos:g} (file t0={t0})  {path.name}  ({w}x{h})",
            flush=True,
        )

    # Cube t-extent: full sequence for uniform / filename modes; t_gap uses its own span.
    if float(args.t_gap) > 0 and t_positions:
        ax.set_ylim(min(t_positions), max(t_positions) + max(float(args.t_gap), 1.0))
    elif t_lim is not None:
        ax.set_ylim(t_lim[0], t_lim[1])
    elif t_positions:
        ax.set_ylim(min(t_positions), max(t_positions) + 1.0)

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

    save_path = args.save if args.save is not None else args.in_dir / "pose_3d.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path,
        dpi=int(args.dpi),
        bbox_inches="tight",
        facecolor=face,
        edgecolor="none",
        transparent=(face == "none"),
    )
    print(f"saved {save_path}")

    if not args.no_show:
        plt.show()
    plt.close(fig)
    plt.close("all")
    gc.collect()


if __name__ == "__main__":
    main()
