#!/usr/bin/env python3
"""3D pose trajectories + SPAD/RGB spatiotemporal cubes from ``vis_main_pose.py``.

Outputs under ``--save`` (folder):
- ``{gt,rgb,qnn,hire}_traj3d.png`` — skeletons + bbox-center traj (axes, Times New Roman)
- ``spad_cube3d.png`` — SPAD bins in ``[start_frame, end_frame)`` as a voxel cube (no axes/text)
- ``rgb_cube3d.png`` — RGB frames in the same window as stacked image planes (no axes/text)

Camera: ``(X,Y,Z)=(x, t_ms, y)``, elev=18, azim=-70, invert_zaxis.
Default figure background is transparent.

Examples
--------
python ultralytics/vis_main_pose_3d.py \\
  --in_dir /tmp/main_pose \\
  --rgb_path /path/renders-rgb25fps-8kHz/sample \\
  --spad_path /path/spc-8kHz/sample \\
  --start_frame 0 --end_frame 40 \\
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

from ultralytics.data.spad_packed import is_packed_spad, unpack_packed_frames


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
        description="Per-method 3D pose traj + SPAD/RGB cubes from vis_main_pose"
    )
    ap.add_argument("--in_dir", type=Path, required=True, help="Output dir from vis_main_pose.py")
    ap.add_argument(
        "--rgb_path",
        type=Path,
        required=True,
        help="RGB dir with frames.npy / frames.npy path (25 fps)",
    )
    ap.add_argument(
        "--spad_path",
        type=Path,
        required=True,
        help="SPAD dir with frames.npy / frames.npy path (raw bins)",
    )
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1, help="Exclusive; <0 = until end")
    ap.add_argument(
        "--n_poses",
        type=int,
        default=7,
        help="Uniform skeleton snapshots in [start,end) (clamped to 6–8)",
    )
    ap.add_argument("--methods", type=str, default="gt,rgb,qnn,hire")
    ap.add_argument("--hand", type=str, default="both", choices=["both", "left", "right"])
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-70.0)
    ap.add_argument("--figsize", type=float, nargs=2, default=[10.0, 8.0])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument(
        "--bg",
        type=str,
        default="none",
        help="Figure facecolor; 'none'/transparent = transparent (default)",
    )
    ap.add_argument("--bone_lw", type=float, default=1.6)
    ap.add_argument("--traj_lw", type=float, default=2.0)
    ap.add_argument("--joint_size", type=float, default=8.0)
    ap.add_argument("--kpt_thresh", type=float, default=0.5)
    ap.add_argument("--pale", type=float, default=0.55)
    ap.add_argument("--deep", type=float, default=0.55)
    # SPAD cube — defaults match vis_spad_bins_3d (full = black/white 0/1 photons)
    ap.add_argument(
        "--spad_stride_xy",
        type=int,
        default=2,
        help="Spatial stride (same default as vis_spad_bins_3d)",
    )
    ap.add_argument("--spad_stride_t", type=int, default=8, help="Temporal bin stride for SPAD cube")
    ap.add_argument(
        "--spad_alpha",
        type=float,
        default=0.08,
        help="Marker alpha (vis_spad_bins_3d default 0.08)",
    )
    ap.add_argument(
        "--spad_point_size",
        type=float,
        default=0.2,
        help="Scatter marker size (vis_spad_bins_3d default 0.2)",
    )
    ap.add_argument(
        "--spad_mode",
        type=str,
        default="full",
        choices=["full", "hits"],
        help="full=0/1 black/white (default, same as vis_spad_bins_3d); hits=photons only colored by t",
    )
    ap.add_argument("--spad_color0", type=str, default="#000000", help="Color for binary 0 (no photon)")
    ap.add_argument("--spad_color1", type=str, default="#ffffff", help="Color for binary 1 (photon)")
    ap.add_argument("--spad_cmap", type=str, default="viridis", help="hits-mode colormap by t")
    ap.add_argument("--spad_max_points", type=int, default=400_000)
    ap.add_argument(
        "--spad_channel",
        type=str,
        default="any",
        help="Collapse packed channels: any|r|g|b|bayer (same as vis_spad_bins)",
    )
    ap.add_argument("--spad_expected_w", type=int, default=0, help="Unpack crop width; 0=full")
    ap.add_argument("--spad_seed", type=int, default=0)
    # RGB cube
    ap.add_argument("--rgb_stride_xy", type=int, default=2, help="Spatial downsample for RGB planes")
    ap.add_argument(
        "--rgb_stride_t",
        type=int,
        default=1,
        help="Keep every Nth RGB frame in the window for the cube",
    )
    ap.add_argument(
        "--rgb_alpha_start",
        type=float,
        default=0.7,
        help="RGB plane opacity at first slice (default 0.7)",
    )
    ap.add_argument(
        "--rgb_alpha_mid",
        type=float,
        default=0.2,
        help="RGB plane opacity at middle slice (default 0.2)",
    )
    ap.add_argument(
        "--rgb_alpha_end",
        type=float,
        default=0.7,
        help="RGB opacity at second-to-last via start→mid→end blend; "
        "the final slice is always 1.0",
    )
    ap.add_argument(
        "--save",
        type=Path,
        required=True,
        help="Output folder for traj PNGs + spad_cube3d.png + rgb_cube3d.png",
    )
    ap.add_argument("--no_show", action="store_true")
    return ap.parse_args()


def _load_vis_main_pose_rgb():
    path = Path(__file__).resolve().parent / "vis_main_pose.py"
    spec = importlib.util.spec_from_file_location("vis_main_pose", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
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
        out = [int(round(i * (n_total - 1) / float(n_keep - 1))) for i in range(n_keep)]
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


def _bin_to_ms(bin_idx: int | float, bin_rate_hz: float) -> float:
    return 1000.0 * float(bin_idx) / float(bin_rate_hz)


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
    from matplotlib import font_manager as fm
    from matplotlib.font_manager import FontProperties

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

    for name in ("Times New Roman", "TimesNewRoman", "Times"):
        try:
            path = fm.findfont(FontProperties(family=name), fallback_to_default=False)
        except (ValueError, RuntimeError):
            continue
        if path and "dejavu" not in Path(path).name.lower():
            return FontProperties(fname=path, size=10), path

    for name in ("Liberation Serif", "STIXGeneral", "DejaVu Serif"):
        path = fm.findfont(FontProperties(family=name))
        print(
            f"Warning: Times New Roman not found; using {name} ({path}). "
            "On Linux: sudo apt-get install ttf-mscorefonts-installer",
            flush=True,
        )
        return FontProperties(fname=path, size=10), path

    return FontProperties(family="serif", size=10), "serif"


def _apply_font_to_3d_ax(ax, font_prop) -> None:
    tick_prop = font_prop.copy()
    tick_prop.set_size(8)
    label_prop = font_prop.copy()
    label_prop.set_size(10)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_fontproperties(label_prop)
        for t in axis.get_ticklabels():
            t.set_fontproperties(tick_prop)


def _style_axes_simple(ax, font_prop=None) -> None:
    ax.grid(False)
    if font_prop is not None:
        ax.set_xlabel("x (px)", fontproperties=font_prop)
        ax.set_ylabel("t (ms)", fontproperties=font_prop)
        ax.set_zlabel("y (px)", fontproperties=font_prop)
        _apply_font_to_3d_ax(ax, font_prop)
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


def _style_axes_off(ax) -> None:
    """No text / ticks / panes (cube figures)."""
    ax.set_axis_off()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0.0))
        axis.pane.set_alpha(0.0)
        axis.line.set_color((1, 1, 1, 0.0))


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
            ax.plot(xs, ts, ys, color=base, linewidth=float(traj_lw), alpha=0.9, solid_capstyle="round")
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


def _resolve_npy(path: Path) -> Path:
    if path.is_dir():
        for name in ("frames.npy", "binary.npy"):
            cand = path / name
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"No frames.npy under {path}")
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Expected .npy, got {path}")
    return path


def _channel_index(name: str, n_ch: int) -> int | None:
    key = name.strip().lower()
    if key in ("any", "or", "bayer"):
        return None
    aliases = {"r": 0, "g": 1, "b": 2, "g1": 1, "g2": 2}
    if key in aliases:
        idx = aliases[key]
    else:
        idx = int(key)
    if idx < 0 or idx >= n_ch:
        raise ValueError(f"channel={name!r} out of range for C={n_ch}")
    return idx


def _to_binary_hw(frame: np.ndarray, *, channel: str) -> np.ndarray:
    if frame.ndim == 2:
        return (frame > 0).astype(np.uint8)
    if frame.ndim != 3:
        raise ValueError(f"Expected HW or HWC, got {frame.shape}")
    h, w, c = frame.shape
    key = channel.strip().lower()
    if key == "bayer":
        if c not in (3, 4):
            raise ValueError(f"bayer needs C in {{3,4}}, got C={c}")
        mosaic = np.zeros((h * 2, w * 2), dtype=np.uint8)
        if c == 3:
            mosaic[0::2, 0::2] = frame[:, :, 0] > 0
            mosaic[0::2, 1::2] = frame[:, :, 1] > 0
            mosaic[1::2, 0::2] = frame[:, :, 1] > 0
            mosaic[1::2, 1::2] = frame[:, :, 2] > 0
        else:
            mosaic[0::2, 0::2] = frame[:, :, 0] > 0
            mosaic[0::2, 1::2] = frame[:, :, 1] > 0
            mosaic[1::2, 0::2] = frame[:, :, 2] > 0
            mosaic[1::2, 1::2] = frame[:, :, 3] > 0
        return mosaic
    idx = _channel_index(key, c)
    if idx is None:
        return (frame > 0).any(axis=-1).astype(np.uint8)
    return (frame[:, :, idx] > 0).astype(np.uint8)


def _unpack_slice(packed: np.ndarray, *, expected_w: int) -> np.ndarray:
    if packed.dtype == np.bool_ or packed.ndim == 3:
        return packed
    if is_packed_spad(packed) and np.issubdtype(packed.dtype, np.integer):
        return unpack_packed_frames(packed, expected_w=expected_w if expected_w > 0 else None)
    if packed.ndim == 4 and packed.shape[-1] in (1, 3, 4):
        return packed
    raise ValueError(f"Unsupported SPAD shape {packed.shape} dtype={packed.dtype}")


def _collect_spad_voxels(
    spad_npy: Path,
    *,
    bin_start: int,
    bin_end: int,
    stride_xy: int,
    stride_t: int,
    mode: str,
    channel: str,
    expected_w: int,
    bin_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x,y,t_ms,v for SPAD bins in ``[bin_start, bin_end)``."""
    if stride_xy < 1 or stride_t < 1:
        raise ValueError("spad strides must be >= 1")
    arr = np.load(spad_npy, mmap_mode="r", allow_pickle=False)
    t_total = int(arr.shape[0])
    bin_start = max(0, int(bin_start))
    bin_end = min(int(bin_end), t_total)
    if bin_start >= bin_end:
        raise ValueError(f"Empty SPAD bin range [{bin_start}, {bin_end}) T={t_total}")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    vs: list[np.ndarray] = []

    for b in range(bin_start, bin_end, stride_t):
        slab = np.asarray(arr[b : b + 1])
        unpacked = _unpack_slice(slab, expected_w=expected_w)
        frame = unpacked[0]
        mask = _to_binary_hw(frame, channel=channel).astype(bool)
        if stride_xy > 1:
            mask = mask[::stride_xy, ::stride_xy]
        t_ms = _bin_to_ms(b, bin_rate_hz)

        if mode == "hits":
            yy, xx = np.nonzero(mask)
            if xx.size == 0:
                continue
            val = np.ones(xx.shape, dtype=np.uint8)
        else:
            h, w = mask.shape
            yy, xx = np.mgrid[0:h, 0:w]
            yy = yy.ravel()
            xx = xx.ravel()
            val = mask.ravel().astype(np.uint8)

        scale = float(stride_xy)
        xs.append(xx.astype(np.float32) * scale)
        ys.append(yy.astype(np.float32) * scale)
        ts.append(np.full(xx.shape, t_ms, dtype=np.float32))
        vs.append(val)

    if not xs:
        raise RuntimeError(f"No SPAD voxels in bins [{bin_start}, {bin_end})")
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ts),
        np.concatenate(vs),
    )


def _subsample_points(
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


def _savefig_transparent(fig, path: Path, *, dpi: int, face: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor=face,
        edgecolor="none",
        transparent=(face == "none"),
    )


def _render_spad_cube(
    *,
    spad_path: Path,
    bin_start: int,
    bin_end: int,
    bin_rate_hz: float,
    args: argparse.Namespace,
    save_path: Path,
    face: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    npy = _resolve_npy(spad_path)
    print(
        f"SPAD cube: {npy} bins=[{bin_start},{bin_end}) "
        f"stride_xy={args.spad_stride_xy} stride_t={args.spad_stride_t} mode={args.spad_mode}",
        flush=True,
    )
    x, y, t, v = _collect_spad_voxels(
        npy,
        bin_start=bin_start,
        bin_end=bin_end,
        stride_xy=int(args.spad_stride_xy),
        stride_t=int(args.spad_stride_t),
        mode=str(args.spad_mode),
        channel=str(args.spad_channel),
        expected_w=int(args.spad_expected_w),
        bin_rate_hz=float(bin_rate_hz),
    )
    n_all = int(x.shape[0])
    x, y, t, v = _subsample_points(
        x, y, t, v, max_points=int(args.spad_max_points), seed=int(args.spad_seed)
    )
    print(f"  voxels {n_all:,} → plot {int(x.shape[0]):,}", flush=True)

    fig = plt.figure(figsize=tuple(args.figsize), facecolor=face)
    ax = fig.add_subplot(111, projection="3d", facecolor=face)
    px, py, pz = x, t, y  # (X,Y,Z)=(x,t,y)

    if args.spad_mode == "full":
        # Same as vis_spad_bins_3d: binary 0 → black, 1 → white.
        cmap = ListedColormap([str(args.spad_color0), str(args.spad_color1)])
        ax.scatter(
            px,
            py,
            pz,
            c=v,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=float(args.spad_point_size),
            alpha=float(args.spad_alpha),
            linewidths=0,
        )
    else:
        ax.scatter(
            px,
            py,
            pz,
            c=t,
            cmap=str(args.spad_cmap),
            s=float(args.spad_point_size),
            alpha=float(args.spad_alpha),
            linewidths=0,
        )

    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    ax.invert_zaxis()
    _style_axes_off(ax)
    fig.tight_layout(pad=0)
    _savefig_transparent(fig, save_path, dpi=int(args.dpi), face=face)
    print(f"saved {save_path}", flush=True)
    plt.close(fig)


def _rgb_slice_alpha(
    i: int,
    n: int,
    *,
    alpha_start: float,
    alpha_mid: float,
    alpha_end: float,
) -> float:
    """Opacity for slice ``i`` in ``[0, n)``.

    Slices ``[0, n-2]`` blend start → mid → end (piecewise linear);
    the **last** slice is always fully opaque (1.0).
    """
    a0 = float(np.clip(alpha_start, 0.0, 1.0))
    a1 = float(np.clip(alpha_mid, 0.0, 1.0))
    a2 = float(np.clip(alpha_end, 0.0, 1.0))
    if n <= 1:
        return 1.0
    if i >= n - 1:
        return 1.0
    # Map first .. second-to-last onto [0, 1] for start→mid→end.
    t = float(i) / float(max(n - 2, 1))
    if t <= 0.5:
        u = t / 0.5
        return float(a0 + (a1 - a0) * u)
    u = (t - 0.5) / 0.5
    return float(a1 + (a2 - a1) * u)


def _render_rgb_cube(
    *,
    rgb_path: Path,
    start: int,
    end: int,
    ms_per_frame: float,
    args: argparse.Namespace,
    save_path: Path,
    face: str,
) -> None:
    import matplotlib.pyplot as plt

    load_rgb = _load_vis_main_pose_rgb()
    frames = load_rgb(rgb_path)
    n_rgb = len(frames)
    if n_rgb <= 0:
        raise RuntimeError(f"No RGB frames from {rgb_path}")

    stride_xy = int(args.rgb_stride_xy)
    stride_t = int(args.rgb_stride_t)
    if stride_xy < 1 or stride_t < 1:
        raise ValueError("rgb strides must be >= 1")
    alpha_start = float(args.rgb_alpha_start)
    alpha_mid = float(args.rgb_alpha_mid)
    alpha_end = float(args.rgb_alpha_end)

    frame_idxs = list(range(int(start), int(end), stride_t))
    if not frame_idxs:
        raise ValueError(f"Empty RGB frame range [{start}, {end}) stride_t={stride_t}")
    n_planes = len(frame_idxs)
    print(
        f"RGB cube: {rgb_path} frames={n_planes} "
        f"stride_xy={stride_xy} stride_t={stride_t} "
        f"alpha_start={alpha_start:g} mid={alpha_mid:g} end={alpha_end:g}",
        flush=True,
    )

    fig = plt.figure(figsize=tuple(args.figsize), facecolor=face)
    ax = fig.add_subplot(111, projection="3d", facecolor=face)

    for plane_i, fi in enumerate(frame_idxs):
        idx = int(np.clip(fi, 0, n_rgb - 1))
        bgr = frames[idx]
        if stride_xy > 1:
            bgr = bgr[::stride_xy, ::stride_xy]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w, _ = rgb.shape
        t_ms = _frame_to_ms(fi, ms_per_frame)
        alpha = _rgb_slice_alpha(
            plane_i,
            n_planes,
            alpha_start=alpha_start,
            alpha_mid=alpha_mid,
            alpha_end=alpha_end,
        )

        # Map strided plane back to original pixel units (match pose coords).
        xs = np.linspace(0.0, float((w - 1) * stride_xy), w, dtype=np.float32)
        zs = np.linspace(0.0, float((h - 1) * stride_xy), h, dtype=np.float32)
        xx, zz = np.meshgrid(xs, zs)
        yy = np.full_like(xx, t_ms, dtype=np.float32)
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

    ax.set_ylim(_frame_to_ms(start, ms_per_frame), _frame_to_ms(end - 1, ms_per_frame))
    ax.view_init(elev=float(args.elev), azim=float(args.azim))
    ax.invert_zaxis()
    _style_axes_off(ax)
    fig.tight_layout(pad=0)
    _savefig_transparent(fig, save_path, dpi=int(args.dpi), face=face)
    print(f"saved {save_path}", flush=True)
    plt.close(fig)


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
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    loaded: dict[str, list[dict[str, Any]]] = {m: _load_poses(in_dir / m) for m in methods}
    lengths = [len(v) for v in loaded.values() if v]
    if not lengths:
        raise RuntimeError(f"No poses.npy under {in_dir} for {methods}")
    n_all = min(lengths)
    for m in list(loaded):
        loaded[m] = loaded[m][:n_all]

    start = max(0, int(args.start_frame))
    end = int(args.end_frame)
    if end < 0:
        end = n_all
    end = min(end, n_all)
    if start >= end:
        raise ValueError(f"Empty frame range [{start}, {end})")

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
    ms_per_frame = _frame_ms_per_frame(meta)
    bin_rate = float(meta.get("spad_bin_rate_hz", 8000.0))
    chunk_size = int(meta.get("chunk_size", 320))

    # Bin span covering [start_frame, end_frame) from pose records (fallback: frame*chunk).
    ref = next(iter(windowed.values()))
    bin_start = int(ref[0].get("chunk_start_bin", start * chunk_size))
    bin_end = int(ref[-1].get("chunk_end_bin", end * chunk_size))
    if bin_end <= bin_start:
        bin_start = start * chunk_size
        bin_end = end * chunk_size

    print(
        f"in_dir={in_dir} frames=[{start},{end}) bins=[{bin_start},{bin_end}) "
        f"n_poses={n_poses} methods={methods}",
        flush=True,
    )
    print(
        f"t: {ms_per_frame:g} ms/frame; spad_rate={bin_rate:g} Hz",
        flush=True,
    )

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    face = "none" if str(args.bg).lower() in {"none", "transparent"} else args.bg

    # --- SPAD + RGB cubes (no axes / text) ---
    _render_spad_cube(
        spad_path=args.spad_path,
        bin_start=bin_start,
        bin_end=bin_end,
        bin_rate_hz=bin_rate,
        args=args,
        save_path=save_dir / "spad_cube3d.png",
        face=face,
    )
    _render_rgb_cube(
        rgb_path=args.rgb_path,
        start=start,
        end=end,
        ms_per_frame=ms_per_frame,
        args=args,
        save_path=save_dir / "rgb_cube3d.png",
        face=face,
    )

    # --- Per-method traj (axes + Times New Roman) ---
    font_prop, font_id = _resolve_times_font()
    plt.rcParams.update(
        {
            "font.family": font_prop.get_name(),
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    print(f"font: {font_prop.get_name()} ← {font_id}", flush=True)

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
        fig.canvas.draw()
        _apply_font_to_3d_ax(ax, font_prop)

        out_path = save_dir / f"{method}_traj3d.png"
        _savefig_transparent(fig, out_path, dpi=int(args.dpi), face=face)
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
