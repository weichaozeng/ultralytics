#!/usr/bin/env python3
"""3D spatiotemporal pose + bbox-center trajectories from ``vis_main_pose.py``.

Loads ``{gt,rgb,qnn,hire}/poses.npy`` and draws:
- hand skeletons at ~6–8 uniformly spaced frames
- bbox-center polylines over the full ``[start_frame, end_frame)`` window

Camera / axes match ``vis_spad_bins_3d.py``:
  plot (X, Y, Z) = (x_img, t, y_img), elev=18, azim=-70, invert_zaxis, axes off.
No image planes or SPAD voxels — pose sticks + trajectories only.

Examples
--------
python ultralytics/vis_main_pose_3d.py \\
  --in_dir /tmp/main_pose \\
  --start_frame 0 --end_frame 40 \\
  --n_poses 7 \\
  --save /tmp/pose_traj_3d.png --no_show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Distinct RGB colors per method (matplotlib)
METHOD_COLORS = {
    "gt": (0.15, 0.15, 0.15),       # near-black
    "rgb": (0.20, 0.45, 0.95),      # blue (pure detector)
    "qnn": (0.90, 0.45, 0.10),      # orange
    "hire": (0.15, 0.70, 0.35),     # green
}

METHOD_ORDER = ("gt", "rgb", "qnn", "hire")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="3D pose skeletons + bbox-center trajectories from vis_main_pose output"
    )
    ap.add_argument("--in_dir", type=Path, required=True, help="Output dir from vis_main_pose.py")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1, help="Exclusive; <0 = until end")
    ap.add_argument(
        "--n_poses",
        type=int,
        default=7,
        help="Uniformly spaced skeleton snapshots in [start,end) (clamped to 6–8)",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="gt,rgb,qnn,hire",
        help="Comma-separated subset of gt,rgb,qnn,hire",
    )
    ap.add_argument(
        "--hand",
        type=str,
        default="both",
        choices=["both", "left", "right"],
        help="Which hand class(es) to draw",
    )
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-70.0)
    ap.add_argument("--figsize", type=float, nargs=2, default=[10.0, 8.0])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--bg", type=str, default="none", help="Figure facecolor; 'none' = transparent")
    ap.add_argument("--bone_lw", type=float, default=1.6)
    ap.add_argument("--traj_lw", type=float, default=2.0)
    ap.add_argument("--joint_size", type=float, default=8.0)
    ap.add_argument("--kpt_thresh", type=float, default=0.5)
    ap.add_argument("--pale", type=float, default=0.55, help="Early-pose mix toward white")
    ap.add_argument("--deep", type=float, default=0.55, help="Late-pose brightness scale")
    ap.add_argument("--legend", action="store_true", help="Show method legend")
    ap.add_argument("--per_method", action="store_true", help="Also save one PNG per method")
    ap.add_argument("--save", type=Path, default=None, help="Output PNG path")
    ap.add_argument("--no_show", action="store_true")
    return ap.parse_args()


def _uniform_sample_indices(n_total: int, n_keep: int) -> list[int]:
    n_total = int(n_total)
    n_keep = int(n_keep)
    if n_total <= 0:
        return []
    if n_keep <= 0 or n_keep >= n_total:
        return list(range(n_total))
    if n_keep == 1:
        return [n_total - 1]
    raw = np.linspace(0, n_total - 1, num=n_keep)
    idxs = [int(round(float(x))) for x in raw]
    out: list[int] = []
    for i in idxs:
        i = int(np.clip(i, 0, n_total - 1))
        if not out or i != out[-1]:
            out.append(i)
    if len(out) < n_keep:
        out = [
            int(round(i * (n_total - 1) / float(n_keep - 1)))
            for i in range(n_keep)
        ]
        out = sorted(set(out))
    return out


def _shade_rgb(
    base: tuple[float, float, float],
    shade: float,
    *,
    pale: float,
    deep: float,
) -> tuple[float, float, float]:
    shade = float(np.clip(shade, 0.0, 1.0))
    base_a = np.asarray(base, dtype=np.float64)
    light = base_a * (1.0 - pale) + 1.0 * pale
    dark = base_a * deep
    color = (1.0 - shade) * light + shade * dark
    return tuple(float(np.clip(v, 0.0, 1.0)) for v in color)


def _load_poses(method_dir: Path) -> list[dict[str, Any]]:
    path = method_dir / "poses.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, allow_pickle=True)
    frames = list(arr)
    if not frames:
        return []
    # Normalize numpy arrays inside
    out = []
    for fr in frames:
        hands = []
        for h in fr.get("hands", []):
            hands.append(
                {
                    "cls": int(h["cls"]),
                    "score": float(h.get("score", 1.0)),
                    "bbox_xyxy": np.asarray(h["bbox_xyxy"], dtype=np.float32).reshape(4),
                    "bbox_center": np.asarray(h["bbox_center"], dtype=np.float32).reshape(2),
                    "keypoints": np.asarray(h["keypoints"], dtype=np.float32),
                }
            )
        out.append(
            {
                "frame_idx": int(fr["frame_idx"]),
                "chunk_start_bin": int(fr.get("chunk_start_bin", 0)),
                "chunk_end_bin": int(fr.get("chunk_end_bin", 0)),
                "image_shape": tuple(int(x) for x in fr["image_shape"]),
                "hands": hands,
            }
        )
    return out


def _hand_allowed(cls_id: int, hand_mode: str) -> bool:
    if hand_mode == "both":
        return True
    if hand_mode == "left":
        return int(cls_id) == 0
    if hand_mode == "right":
        return int(cls_id) == 1
    return True


def _pick_hand(frame: dict[str, Any], cls_id: int) -> dict[str, Any] | None:
    cands = [h for h in frame["hands"] if int(h["cls"]) == int(cls_id)]
    if not cands:
        return None
    # highest score
    return max(cands, key=lambda h: float(h.get("score", 0.0)))


def _trajectory(
    frames: list[dict[str, Any]],
    cls_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ts = [], [], []
    for fr in frames:
        hand = _pick_hand(fr, cls_id)
        if hand is None:
            continue
        c = hand["bbox_center"]
        xs.append(float(c[0]))
        ys.append(float(c[1]))
        ts.append(float(fr["frame_idx"]))
    if not xs:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(ts, dtype=np.float64),
    )


def _draw_skeleton(
    ax,
    hand: dict[str, Any],
    t: float,
    color: tuple[float, float, float],
    *,
    kpt_thresh: float,
    bone_lw: float,
    joint_size: float,
) -> None:
    kpts = np.asarray(hand["keypoints"], dtype=np.float64)
    if kpts.shape != (21, 3):
        if kpts.shape == (21, 2):
            kpts = np.concatenate([kpts, np.ones((21, 1), dtype=np.float64)], axis=1)
        else:
            return
    for s, e in BONE_CONNECTIONS:
        ks = kpts[s]
        ke = kpts[e]
        if ks[2] > kpt_thresh and ke[2] > kpt_thresh:
            # (X,Y,Z) = (x, t, y)
            ax.plot(
                [ks[0], ke[0]],
                [t, t],
                [ks[1], ke[1]],
                color=color,
                linewidth=float(bone_lw),
                solid_capstyle="round",
            )
    vis = kpts[:, 2] > float(kpt_thresh)
    if np.any(vis):
        ax.scatter(
            kpts[vis, 0],
            np.full(int(vis.sum()), float(t)),
            kpts[vis, 1],
            c=[color],
            s=float(joint_size),
            depthshade=False,
            linewidths=0,
        )


def _style_axes(ax, face: str) -> None:
    ax.set_axis_off()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0.0))
        axis.pane.set_alpha(0.0)
        axis.line.set_color((1, 1, 1, 0.0))


def _draw_methods(
    ax,
    method_frames: dict[str, list[dict[str, Any]]],
    *,
    pose_local_idxs: list[int],
    hand_mode: str,
    kpt_thresh: float,
    bone_lw: float,
    traj_lw: float,
    joint_size: float,
    pale: float,
    deep: float,
    legend: bool,
) -> None:
    from matplotlib.lines import Line2D

    legend_handles = []
    n_pose = max(len(pose_local_idxs), 1)

    for method, frames in method_frames.items():
        base = METHOD_COLORS.get(method, (0.5, 0.5, 0.5))
        legend_handles.append(Line2D([0], [0], color=base, lw=2.5, label=method))

        cls_ids = []
        if hand_mode in {"both", "left"}:
            cls_ids.append(0)
        if hand_mode in {"both", "right"}:
            cls_ids.append(1)

        for cls_id in cls_ids:
            xs, ys, ts = _trajectory(frames, cls_id)
            if xs.size >= 2:
                ax.plot(
                    xs,
                    ts,
                    ys,
                    color=base,
                    linewidth=float(traj_lw),
                    alpha=0.9,
                    solid_capstyle="round",
                )
            elif xs.size == 1:
                ax.scatter([xs[0]], [ts[0]], [ys[0]], c=[base], s=20, depthshade=False)

        for pi, local_i in enumerate(pose_local_idxs):
            if local_i < 0 or local_i >= len(frames):
                continue
            fr = frames[local_i]
            t = float(fr["frame_idx"])
            shade = 0.0 if n_pose <= 1 else float(pi) / float(n_pose - 1)
            color = _shade_rgb(base, shade, pale=pale, deep=deep)
            for hand in fr["hands"]:
                if not _hand_allowed(int(hand["cls"]), hand_mode):
                    continue
                _draw_skeleton(
                    ax,
                    hand,
                    t,
                    color,
                    kpt_thresh=kpt_thresh,
                    bone_lw=bone_lw,
                    joint_size=joint_size,
                )

    if legend and legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", frameon=False)


def _axis_limits(method_frames: dict[str, list[dict[str, Any]]]) -> tuple[float, float, float, float, float, float]:
    xs, ys, ts = [], [], []
    for frames in method_frames.values():
        for fr in frames:
            ts.append(float(fr["frame_idx"]))
            for hand in fr["hands"]:
                k = np.asarray(hand["keypoints"], dtype=np.float64)
                if k.size == 0:
                    continue
                xs.extend(k[:, 0].tolist())
                ys.extend(k[:, 1].tolist())
                c = hand["bbox_center"]
                xs.append(float(c[0]))
                ys.append(float(c[1]))
    if not xs:
        return 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
    pad_x = max((max(xs) - min(xs)) * 0.05, 1.0)
    pad_y = max((max(ys) - min(ys)) * 0.05, 1.0)
    pad_t = max((max(ts) - min(ts)) * 0.05, 0.5)
    return (
        min(xs) - pad_x,
        max(xs) + pad_x,
        min(ts) - pad_t,
        max(ts) + pad_t,
        min(ys) - pad_y,
        max(ys) + pad_y,
    )


def main() -> None:
    args = _parse_args()
    in_dir = args.in_dir
    if not in_dir.is_dir():
        raise FileNotFoundError(in_dir)

    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_COLORS:
            raise ValueError(f"Unknown method {m!r}; expected one of {list(METHOD_COLORS)}")
    # stable order
    methods = [m for m in METHOD_ORDER if m in methods]

    meta_path = in_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    loaded: dict[str, list[dict[str, Any]]] = {}
    for m in methods:
        loaded[m] = _load_poses(in_dir / m)

    # Common length
    lengths = [len(v) for v in loaded.values() if v]
    if not lengths:
        raise RuntimeError(f"No poses.npy found under {in_dir} for methods {methods}")
    n_all = min(lengths)
    for m in list(loaded):
        loaded[m] = loaded[m][:n_all]

    start = max(0, int(args.start_frame))
    end = int(args.end_frame)
    if end < 0:
        end = n_all
    end = min(end, n_all)
    if start >= end:
        raise ValueError(f"Empty frame range [{start}, {end}) with n_frames={n_all}")

    windowed = {m: [fr for fr in frames if start <= int(fr["frame_idx"]) < end] for m, frames in loaded.items()}
    # Ensure contiguous local list; re-index not needed — keep original frame_idx for t axis
    for m, frames in windowed.items():
        if not frames:
            raise RuntimeError(f"Method {m}: no frames in [{start}, {end})")

    n_window = len(next(iter(windowed.values())))
    n_poses = int(np.clip(int(args.n_poses), 6, 8))
    pose_local_idxs = _uniform_sample_indices(n_window, n_poses)

    print(
        f"in_dir={in_dir} frames=[{start},{end}) n={n_window} n_poses={n_poses} "
        f"idxs={pose_local_idxs} methods={methods} hand={args.hand}",
        flush=True,
    )
    if meta:
        print(
            f"meta: chunk={meta.get('chunk_size')} rate={meta.get('spad_bin_rate_hz')} "
            f"fps={meta.get('frame_rate')}",
            flush=True,
        )

    import matplotlib

    if args.save is not None and args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    face = "none" if str(args.bg).lower() in {"none", "transparent"} else args.bg

    def _make_fig(method_subset: dict[str, list[dict[str, Any]]]):
        fig = plt.figure(figsize=tuple(args.figsize), facecolor=face)
        ax = fig.add_subplot(111, projection="3d", facecolor=face)
        _draw_methods(
            ax,
            method_subset,
            pose_local_idxs=pose_local_idxs,
            hand_mode=str(args.hand),
            kpt_thresh=float(args.kpt_thresh),
            bone_lw=float(args.bone_lw),
            traj_lw=float(args.traj_lw),
            joint_size=float(args.joint_size),
            pale=float(args.pale),
            deep=float(args.deep),
            legend=bool(args.legend),
        )
        xmin, xmax, tmin, tmax, ymin, ymax = _axis_limits(method_subset)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(tmin, tmax)
        ax.set_zlim(ymin, ymax)
        ax.view_init(elev=float(args.elev), azim=float(args.azim))
        ax.invert_zaxis()
        _style_axes(ax, face)
        fig.tight_layout(pad=0)
        return fig

    fig = _make_fig(windowed)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            args.save,
            dpi=int(args.dpi),
            bbox_inches="tight",
            facecolor=face,
            edgecolor="none",
            transparent=(face == "none"),
        )
        print(f"saved {args.save}", flush=True)

    if args.per_method:
        base = args.save if args.save is not None else in_dir / "pose_traj_3d.png"
        for m, frames in windowed.items():
            fig_m = _make_fig({m: frames})
            out_m = base.with_name(f"{base.stem}_{m}{base.suffix}")
            fig_m.savefig(
                out_m,
                dpi=int(args.dpi),
                bbox_inches="tight",
                facecolor=face,
                edgecolor="none",
                transparent=(face == "none"),
            )
            print(f"saved {out_m}", flush=True)
            plt.close(fig_m)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
