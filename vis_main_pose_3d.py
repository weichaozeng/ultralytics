#!/usr/bin/env python3
"""3D spatiotemporal pose + bbox-center trajectories from ``vis_main_pose.py``.

Loads ``{gt,rgb,qnn,hire}/poses.npy`` and draws **one figure per method**:
- hand skeletons at ~6–8 uniformly spaced frames
- bbox-center polylines over ``[start_frame, end_frame)``

Camera matches ``vis_spad_bins_3d.py``:
  plot (X, Y, Z) = (x_img, t_ms, y_img), elev=18, azim=-70, invert_zaxis.
Time axis is milliseconds: ``t_ms = frame_idx * (1000 / fps)`` (fps from meta,
default 25 → 40 ms/frame). Simple axis lines/labels (no grid). No image planes.

Also saves RGB first / mid / last frames in the same window via ``--rgb_path``
(``frames.npy`` dir or file, same as ``vis_main_pose`` / ``test_rgb_pose``).

Examples
--------
python ultralytics/vis_main_pose_3d.py \\
  --in_dir /tmp/main_pose \\
  --rgb_path /path/to/renders-rgb25fps-8kHz/sample \\
  --start_frame 0 --end_frame 40 \\
  --n_poses 7 \\
  --save /tmp/pose_traj_3d --no_show
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

METHOD_COLORS = {
    "gt": (0.15, 0.15, 0.15),
    "rgb": (0.20, 0.45, 0.95),
    "qnn": (0.90, 0.45, 0.10),
    "hire": (0.15, 0.70, 0.35),
}

METHOD_ORDER = ("gt", "rgb", "qnn", "hire")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-method 3D pose trajectories from vis_main_pose + RGB keyframes"
    )
    ap.add_argument("--in_dir", type=Path, required=True, help="Output dir from vis_main_pose.py")
    ap.add_argument(
        "--rgb_path",
        type=Path,
        required=True,
        help="RGB dir with frames.npy, frames.npy path, image dir, or video",
    )
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
    ap.add_argument("--bg", type=str, default="white", help="Figure facecolor; 'none' = transparent")
    ap.add_argument("--bone_lw", type=float, default=1.6)
    ap.add_argument("--traj_lw", type=float, default=2.0)
    ap.add_argument("--joint_size", type=float, default=8.0)
    ap.add_argument("--kpt_thresh", type=float, default=0.5)
    ap.add_argument("--pale", type=float, default=0.55, help="Early-pose mix toward white")
    ap.add_argument("--deep", type=float, default=0.55, help="Late-pose brightness scale")
    ap.add_argument(
        "--save",
        type=Path,
        required=True,
        help="Output folder for per-method traj PNGs + RGB first/mid/last frames",
    )
    ap.add_argument("--no_show", action="store_true")
    return ap.parse_args()


def _load_vis_main_pose_rgb():
    path = Path(__file__).resolve().parent / "vis_main_pose.py"
    spec = importlib.util.spec_from_file_location("vis_main_pose", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Avoid colliding with a package named vis_main_pose if imported elsewhere.
    sys.modules["vis_main_pose_for_3d"] = mod
    spec.loader.exec_module(mod)
    return mod._load_rgb_frames


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
    return max(cands, key=lambda h: float(h.get("score", 0.0)))


def _frame_ms_per_frame(meta: dict[str, Any], *, default_fps: float = 25.0) -> float:
    """Milliseconds per 25 fps emit frame from meta (or default)."""
    fps = meta.get("frame_rate", None)
    if fps is not None and float(fps) > 0:
        return 1000.0 / float(fps)
    chunk = meta.get("chunk_size", None)
    rate = meta.get("spad_bin_rate_hz", None)
    if chunk is not None and rate is not None and float(rate) > 0:
        return 1000.0 * float(chunk) / float(rate)
    return 1000.0 / float(default_fps)


def _frame_to_ms(frame_idx: int | float, ms_per_frame: float) -> float:
    return float(frame_idx) * float(ms_per_frame)


def _trajectory(
    frames: list[dict[str, Any]],
    cls_id: int,
    *,
    ms_per_frame: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ts = [], [], []
    for fr in frames:
        hand = _pick_hand(fr, cls_id)
        if hand is None:
            continue
        c = hand["bbox_center"]
        xs.append(float(c[0]))
        ys.append(float(c[1]))
        ts.append(_frame_to_ms(fr["frame_idx"], ms_per_frame))
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


def _resolve_times_font():
    """Return ``(FontProperties, font_path_or_name)`` for Times New Roman.

    Reasons the previous ``fontname=...`` often looked unchanged:
    1. Linux servers usually lack Times New Roman → silent fallback to DejaVu.
    2. mplot3d rebuilds tick/axis text after ``view_init`` / ``tight_layout`` /
       ``savefig(bbox_inches='tight')``, wiping per-label font settings.

    We bind an explicit TTF path when possible and re-apply after draw.
    """
    from matplotlib import font_manager as fm
    from matplotlib.font_manager import FontProperties

    preferred_names = ("Times New Roman", "TimesNewRoman", "Times")
    # Common install locations (Linux msttcorefonts / macOS Supplemental).
    preferred_files = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
    ]
    for path in preferred_files:
        p = Path(path)
        if p.is_file():
            try:
                fm.fontManager.addfont(str(p))
            except (ValueError, RuntimeError):
                pass
            return FontProperties(fname=str(p), size=10), str(p)

    for name in preferred_names:
        try:
            path = fm.findfont(FontProperties(family=name), fallback_to_default=False)
        except (ValueError, RuntimeError):
            continue
        # findfont may still return DejaVu when fallback is allowed; reject that.
        if path and "dejavu" not in Path(path).name.lower():
            return FontProperties(fname=path, size=10), path

    # Last resort: Liberation Serif / STIX (Times-like metrics), else DejaVu Serif.
    for name in ("Liberation Serif", "STIXGeneral", "DejaVu Serif"):
        path = fm.findfont(FontProperties(family=name))
        print(
            f"Warning: Times New Roman not found; using {name} ({path}). "
            "On Linux install fonts with: sudo apt-get install ttf-mscorefonts-installer",
            flush=True,
        )
        return FontProperties(fname=path, size=10), path

    fp = FontProperties(family="serif", size=10)
    return fp, "serif"


def _apply_font_to_3d_ax(ax, font_prop) -> None:
    """Force font on 3D axis labels + ticks (call after draw / tight_layout)."""
    tick_prop = font_prop.copy()
    tick_prop.set_size(8)
    label_prop = font_prop.copy()
    label_prop.set_size(10)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_fontproperties(label_prop)
        for t in axis.get_ticklabels():
            t.set_fontproperties(tick_prop)


def _style_axes_simple(ax, font_prop=None) -> None:
    """Keep axis lines + labels; hide grid and pane fill."""
    ax.grid(False)
    if font_prop is not None:
        ax.set_xlabel("x (px)", fontproperties=font_prop)
        ax.set_ylabel("t (ms)", fontproperties=font_prop)
        ax.set_zlabel("y (px)", fontproperties=font_prop)
    else:
        ax.set_xlabel("x (px)")
        ax.set_ylabel("t (ms)")
        ax.set_zlabel("y (px)")
    ax.tick_params(labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.75, 0.75, 0.75, 0.35))
        axis.pane.set_alpha(0.0)
        axis.line.set_color((0.25, 0.25, 0.25, 1.0))
    if font_prop is not None:
        _apply_font_to_3d_ax(ax, font_prop)


def _draw_one_method(
    ax,
    method: str,
    frames: list[dict[str, Any]],
    *,
    pose_local_idxs: list[int],
    hand_mode: str,
    kpt_thresh: float,
    bone_lw: float,
    traj_lw: float,
    joint_size: float,
    pale: float,
    deep: float,
    ms_per_frame: float,
) -> None:
    base = METHOD_COLORS.get(method, (0.5, 0.5, 0.5))
    n_pose = max(len(pose_local_idxs), 1)

    cls_ids = []
    if hand_mode in {"both", "left"}:
        cls_ids.append(0)
    if hand_mode in {"both", "right"}:
        cls_ids.append(1)

    for cls_id in cls_ids:
        xs, ys, ts = _trajectory(frames, cls_id, ms_per_frame=ms_per_frame)
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
        t = _frame_to_ms(fr["frame_idx"], ms_per_frame)
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


def _axis_limits(
    frames: list[dict[str, Any]],
    *,
    ms_per_frame: float,
) -> tuple[float, float, float, float, float, float]:
    xs, ys, ts = [], [], []
    for fr in frames:
        ts.append(_frame_to_ms(fr["frame_idx"], ms_per_frame))
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
    pad_t = max((max(ts) - min(ts)) * 0.05, float(ms_per_frame) * 0.5)
    return (
        min(xs) - pad_x,
        max(xs) + pad_x,
        min(ts) - pad_t,
        max(ts) + pad_t,
        min(ys) - pad_y,
        max(ys) + pad_y,
    )


def _first_mid_last_indices(start: int, end: int) -> tuple[int, int, int]:
    """Inclusive frame indices for first / mid / last in ``[start, end)``."""
    if end <= start:
        raise ValueError(f"Empty range [{start}, {end})")
    first = int(start)
    last = int(end - 1)
    mid = int(start + (end - start - 1) // 2)
    return first, mid, last


def _save_rgb_keyframes(
    *,
    rgb_path: Path,
    save_dir: Path,
    start: int,
    end: int,
) -> list[Path]:
    load_rgb = _load_vis_main_pose_rgb()
    frames = load_rgb(rgb_path)
    n_rgb = len(frames)
    if n_rgb <= 0:
        raise RuntimeError(f"No RGB frames loaded from {rgb_path}")

    first, mid, last = _first_mid_last_indices(start, end)
    # Map 25 fps indices into RGB length (1:1 when RGB is already 25 fps).
    picks = {
        "first": int(np.clip(first, 0, n_rgb - 1)),
        "mid": int(np.clip(mid, 0, n_rgb - 1)),
        "last": int(np.clip(last, 0, n_rgb - 1)),
    }
    saved = []
    for tag, idx in picks.items():
        out = save_dir / f"rgb_{tag}_frame{idx:07d}.png"
        ok = cv2.imwrite(str(out), frames[idx])
        if not ok:
            raise RuntimeError(f"Failed to write {out}")
        saved.append(out)
        print(f"saved {out} (rgb idx={idx})", flush=True)
    return saved


def main() -> None:
    args = _parse_args()
    in_dir = args.in_dir
    if not in_dir.is_dir():
        raise FileNotFoundError(in_dir)

    save_dir = args.save
    if save_dir.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        raise ValueError(f"--save must be a folder path, got file-like {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_COLORS:
            raise ValueError(f"Unknown method {m!r}; expected one of {list(METHOD_COLORS)}")
    methods = [m for m in METHOD_ORDER if m in methods]

    meta_path = in_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    loaded: dict[str, list[dict[str, Any]]] = {}
    for m in methods:
        loaded[m] = _load_poses(in_dir / m)

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

    windowed = {
        m: [fr for fr in frames if start <= int(fr["frame_idx"]) < end]
        for m, frames in loaded.items()
    }
    for m, frames in windowed.items():
        if not frames:
            raise RuntimeError(f"Method {m}: no frames in [{start}, {end})")

    n_window = len(next(iter(windowed.values())))
    n_poses = int(np.clip(int(args.n_poses), 6, 8))
    pose_local_idxs = _uniform_sample_indices(n_window, n_poses)
    first_i, mid_i, last_i = _first_mid_last_indices(start, end)
    ms_per_frame = _frame_ms_per_frame(meta)
    t0_ms = _frame_to_ms(start, ms_per_frame)
    t1_ms = _frame_to_ms(end - 1, ms_per_frame)

    print(
        f"in_dir={in_dir} frames=[{start},{end}) n={n_window} n_poses={n_poses} "
        f"idxs={pose_local_idxs} methods={methods} hand={args.hand}",
        flush=True,
    )
    print(
        f"t axis: {ms_per_frame:g} ms/frame → [{t0_ms:g}, {t1_ms:g}] ms",
        flush=True,
    )
    print(f"rgb keyframes: first={first_i} mid={mid_i} last={last_i}", flush=True)
    if meta:
        print(
            f"meta: chunk={meta.get('chunk_size')} rate={meta.get('spad_bin_rate_hz')} "
            f"fps={meta.get('frame_rate')}",
            flush=True,
        )

    # RGB first / mid / last
    _save_rgb_keyframes(
        rgb_path=args.rgb_path,
        save_dir=save_dir,
        start=start,
        end=end,
    )

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font_prop, font_id = _resolve_times_font()
    plt.rcParams.update(
        {
            "font.family": font_prop.get_name(),
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            # Embed TrueType in vector outputs; avoids bitmap fallbacks looking "wrong".
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    print(f"font: {font_prop.get_name()} ← {font_id}", flush=True)

    face = "none" if str(args.bg).lower() in {"none", "transparent"} else args.bg

    shown = None
    for method, frames in windowed.items():
        fig = plt.figure(figsize=tuple(args.figsize), facecolor=face)
        ax = fig.add_subplot(111, projection="3d", facecolor=face)
        _draw_one_method(
            ax,
            method,
            frames,
            pose_local_idxs=pose_local_idxs,
            hand_mode=str(args.hand),
            kpt_thresh=float(args.kpt_thresh),
            bone_lw=float(args.bone_lw),
            traj_lw=float(args.traj_lw),
            joint_size=float(args.joint_size),
            pale=float(args.pale),
            deep=float(args.deep),
            ms_per_frame=ms_per_frame,
        )
        xmin, xmax, tmin, tmax, ymin, ymax = _axis_limits(frames, ms_per_frame=ms_per_frame)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(tmin, tmax)
        ax.set_zlim(ymin, ymax)
        ax.view_init(elev=float(args.elev), azim=float(args.azim))
        ax.invert_zaxis()
        _style_axes_simple(ax, font_prop)
        fig.tight_layout(pad=0.4)
        # mplot3d regenerates text during draw; re-apply font immediately before save.
        fig.canvas.draw()
        _apply_font_to_3d_ax(ax, font_prop)

        out_path = save_dir / f"{method}_traj3d.png"
        fig.savefig(
            out_path,
            dpi=int(args.dpi),
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor=face,
            edgecolor="none",
            transparent=(face == "none"),
        )
        print(f"saved {out_path}", flush=True)

        if not args.no_show:
            if shown is not None:
                plt.close(shown)
            shown = fig
        else:
            plt.close(fig)

    if not args.no_show:
        plt.show()
    elif shown is not None:
        plt.close(shown)

    print(f"Done → {save_dir}", flush=True)


if __name__ == "__main__":
    main()
