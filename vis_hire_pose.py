#!/usr/bin/env python3
"""Per-chunk HIRE SPAD pose inference → pose-only overlay PNGs.

Runs a trained ``SpadPoseModel`` (HIRE preprocessor) on packed/raw ``frames.npy``
in **causal streaming** mode: preprocessor + detector temporal plugins carry state
across chunks (no mid-video reset).

Inference emit cadence follows ``--chunk_size`` only (one recon + one pose per chunk).
Train-time ``spad_subsampling`` is not used for how often frames are emitted.

Visualization: neon skeleton + bbox (no ID/conf text). Each track/detection ID
shares one fluorescent color for bones and box.

Output naming (for ``vis_hire_pose_3d.py``)::

    {save_dir}/{sample}/cubeXXXXX_tTTTTTT_TTTTTT_frameFFFFFFF.png

Trail composite (``--trail_composite``)::
    one transparent RGBA PNG over ``[--start_bin, --end_bin)``; pose only (no
    boxes). Color lightness is pre-scheduled from light→dark by bin range /
    chunk_size; stroke alpha is always 1.

Examples
--------
python ultralytics/vis_hire_pose.py \\
  --in_path /path/to/frames.npy \\
  --ckpt /path/to/last.pt \\
  --save_dir /tmp/hire_pose \\
  --chunk_size 320 \\
  --start_bin 960

python ultralytics/vis_hire_pose.py \\
  --in_path /path/to/frames.npy \\
  --ckpt /path/to/last.pt \\
  --save_dir /tmp/hire_pose \\
  --trail_composite \\
  --start_bin 960 --end_bin 3200 \\
  --chunk_size 320
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO


def _load_det_spad_pose():
    """Import sibling ``det_spad_pose.py`` helpers."""
    path = Path(__file__).resolve().parent / "det_spad_pose.py"
    spec = importlib.util.spec_from_file_location("det_spad_pose", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["det_spad_pose"] = mod
    spec.loader.exec_module(mod)
    return mod


dsp = _load_det_spad_pose()

# Neon / fluorescent BGR palette (high-sat, high-value) — one color per hand ID.
_NEON_COLORS_BGR = [
    (0, 255, 255),  # neon yellow
    (255, 0, 255),  # neon magenta / hot pink
    (0, 255, 80),  # neon lime
    (255, 255, 0),  # neon cyan
    (0, 128, 255),  # neon orange
    (255, 0, 180),  # electric violet
    (60, 255, 0),  # chartreuse
    (255, 80, 80),  # neon blue
    (0, 60, 255),  # fluorescent red-orange
    (180, 255, 0),  # aqua-lime
]


def _id_color_bgr(track_id: int) -> tuple[int, int, int]:
    return _NEON_COLORS_BGR[int(track_id) % len(_NEON_COLORS_BGR)]


def _draw_bbox_neon(
    img_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    color: tuple[int, int, int],
    *,
    thickness: int = 3,
) -> np.ndarray:
    """Draw bbox only (no ID / conf text), with dark outline for contrast."""
    x1, y1, x2, y2 = map(int, box_xyxy[:4])
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    return img_bgr


def _draw_pose_id(
    img_bgr: np.ndarray,
    pose_kpts: np.ndarray,
    *,
    color: tuple[int, int, int],
    thresh: float = 0.5,
    k: int = 21,
    thickness: int = 4,
) -> np.ndarray:
    """Draw one hand skeleton in a single neon color (dark outline + bright fill)."""
    if pose_kpts.shape != (k, 3):
        raise ValueError(f"Pose shape must be ({k}, 3), but got {pose_kpts.shape}")

    outline = (0, 0, 0)
    for s, e in dsp.BONE_CONNECTIONS:
        ks = pose_kpts[s]
        ke = pose_kpts[e]
        if ks[2] > thresh and ke[2] > thresh:
            p0 = (int(ks[0]), int(ks[1]))
            p1 = (int(ke[0]), int(ke[1]))
            cv2.line(img_bgr, p0, p1, outline, thickness + 3, lineType=cv2.LINE_AA)
            cv2.line(img_bgr, p0, p1, color, thickness, lineType=cv2.LINE_AA)

    for i in range(k):
        kk = pose_kpts[i]
        if kk[2] > thresh:
            center = (int(kk[0]), int(kk[1]))
            r = 7 if i == 0 else 5
            cv2.circle(img_bgr, center, r + 2, outline, -1, lineType=cv2.LINE_AA)
            cv2.circle(img_bgr, center, r, color, -1, lineType=cv2.LINE_AA)

    return img_bgr


def _draw_result_poses(
    img_bgr: np.ndarray,
    result,
    *,
    kpt_thresh: float,
    draw_boxes: bool = True,
    color_for_id=None,
) -> np.ndarray:
    """Draw skeletons (and optional boxes) from one Ultralytics result onto ``img_bgr``.

    ``color_for_id(tid) -> (B,G,R)`` overrides the default neon palette when set.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return img_bgr
    track_ids = result.boxes.id
    if track_ids is None:
        track_ids = torch.arange(len(result.boxes), device=result.boxes.data.device)
    track_id = track_ids.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    poses = None
    if getattr(result, "keypoints", None) is not None:
        poses = result.keypoints.data.cpu().numpy()
    for j, tid in enumerate(track_id):
        if color_for_id is not None:
            color = color_for_id(int(tid))
        else:
            color = _id_color_bgr(int(tid))
        if draw_boxes and j < len(boxes):
            img_bgr = _draw_bbox_neon(img_bgr, boxes[j], color)
        if poses is not None and j < len(poses):
            img_bgr = _draw_pose_id(
                img_bgr,
                poses[j],
                color=color,
                thresh=float(kpt_thresh),
            )
    return img_bgr


def _composite_layer_rgba(
    canvas_rgba: np.ndarray,
    layer_bgr: np.ndarray,
    *,
    opacity: float = 1.0,
) -> None:
    """Alpha-composite a BGR drawing layer onto float RGBA canvas in-place.

    Coverage is inferred from drawn intensity so anti-aliased strokes keep soft edges.
    """
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity <= 0.0:
        return
    cov = layer_bgr.max(axis=2).astype(np.float32) / 255.0
    if float(cov.max()) <= 0.0:
        return
    src_a = cov * opacity
    src_rgb = layer_bgr.astype(np.float32) / 255.0
    dst_a = canvas_rgba[:, :, 3]
    out_a = src_a + dst_a * (1.0 - src_a)
    for c in range(3):
        canvas_rgba[:, :, c] = np.where(
            out_a > 1e-8,
            (src_rgb[:, :, c] * src_a + canvas_rgba[:, :, c] * dst_a * (1.0 - src_a)) / np.maximum(out_a, 1e-8),
            0.0,
        )
    canvas_rgba[:, :, 3] = out_a


def _planned_trail_steps(
    t_begin: int,
    t_end: int,
    chunk_t: int,
    stride: int,
    max_chunks: int,
) -> int:
    """Number of chunk steps in ``[t_begin, t_end)`` (one pose layer per chunk)."""
    n = 0
    for t0 in range(int(t_begin), int(t_end), int(stride)):
        if int(max_chunks) > 0 and n >= int(max_chunks):
            break
        t1 = min(int(t_end), t0 + int(chunk_t))
        if t1 <= t0:
            break
        n += 1
    return n


def _trail_shade(step_i: int, n_steps: int) -> float:
    """0 = lightest (earliest), 1 = darkest (latest)."""
    if n_steps <= 1:
        return 1.0
    return float(np.clip(step_i, 0, n_steps - 1)) / float(n_steps - 1)


def _shade_bgr(
    base_bgr: tuple[int, int, int],
    shade: float,
    *,
    pale: float = 0.72,
    deep: float = 0.40,
) -> tuple[int, int, int]:
    """Lerp neon base from pale (mix toward white) to deep (scaled toward black)."""
    shade = float(np.clip(shade, 0.0, 1.0))
    base = np.asarray(base_bgr, dtype=np.float32)
    light = base * (1.0 - pale) + 255.0 * pale
    dark = base * deep
    color = (1.0 - shade) * light + shade * dark
    return tuple(int(np.clip(round(v), 0, 255)) for v in color)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HIRE chunk pose-only viz (one color per hand ID)")
    ap.add_argument("--in_path", type=Path, required=True, help="frames.npy or dir with frames.npy")
    ap.add_argument("--save_dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="Trained SpadPoseModel .pt")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Letterbox recon frames to this square size for the detector (matches train). "
        "0 = native resolution (may break FPN if H/W not divisible by stride).",
    )
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument(
        "--tracker",
        type=str,
        default="none",
        choices=["none", "bytetrack", "botsort", "spad_tracker", "posetrack", "spad_posetrack"],
        help="Tracker for stable IDs across frames (default none = per-frame det index)",
    )
    ap.add_argument("--frame_rate", type=int, default=25)
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--spad_bin_rate_hz", type=float, default=8000.0)
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=320,
        help="Bins per chunk (8 kHz @ 25 FPS default). 0 = checkpoint spad_chunk_size",
    )
    ap.add_argument("--chunk_stride", type=int, default=0, help="0 = chunk_size (non-overlapping)")
    ap.add_argument("--max_chunks", type=int, default=0, help="0 = all chunks from start")
    ap.add_argument(
        "--start_bin",
        type=int,
        default=0,
        help="Start streaming from this raw SPAD bin index (e.g. 960).",
    )
    ap.add_argument(
        "--end_bin",
        type=int,
        default=0,
        help="Stop before this raw bin (exclusive). 0 = end of video. Used by --trail_composite.",
    )
    ap.add_argument("--tail_pad", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--vis_bg",
        type=str,
        default="recon",
        choices=["sum", "recon", "black"],
        help="Background under the skeleton (default recon). Ignored by --trail_composite.",
    )
    ap.add_argument(
        "--trail_composite",
        action="store_true",
        help="Composite poses (no boxes) in [--start_bin,--end_bin) onto one transparent "
        "RGBA PNG. Color goes light→dark by precomputed chunk schedule; alpha=1.",
    )
    ap.add_argument(
        "--trail_pale",
        type=float,
        default=0.72,
        help="How much the earliest trail pose mixes toward white (0–1).",
    )
    ap.add_argument(
        "--trail_deep",
        type=float,
        default=0.40,
        help="Brightness scale of the latest trail pose relative to neon base (0–1).",
    )
    ap.add_argument(
        "--trail_name",
        type=str,
        default="",
        help="Optional output filename under sample dir (default: trail_tSTART_END.png).",
    )
    ap.add_argument("--kpt_thresh", type=float, default=0.5)
    ap.add_argument(
        "--preprocessor_override",
        type=str,
        default="none",
        choices=["none", "hire"],
        help="Force HIRE preprocessor (none = keep checkpoint)",
    )
    # HIRE override knobs (match vis_pre / sequence_hire_attn_wst_8kHz)
    ap.add_argument("--hire_fast_bins", type=int, default=24)
    ap.add_argument("--hire_slow_bins", type=int, default=160)
    ap.add_argument("--hire_surprise_bins", type=int, default=4)
    ap.add_argument("--hire_mix_hold_bins", type=int, default=80)
    ap.add_argument("--hire_mix_bins", type=float, default=12.0)
    ap.add_argument("--hire_mix_theta", type=float, default=0.06)
    ap.add_argument("--hire_mix_floor", type=float, default=-1.0)
    ap.add_argument("--hire_theta_on", type=float, default=0.08)
    ap.add_argument("--hire_theta_off", type=float, default=0.02)
    ap.add_argument("--hire_theta_grow", type=float, default=-1.0)
    ap.add_argument("--hire_confirm_bins", type=int, default=4)
    ap.add_argument("--hire_cooldown_bins", type=int, default=0)
    ap.add_argument("--hire_spatial_kernel", type=int, default=5)
    ap.add_argument("--hire_gate_pool", type=str, default="max", choices=["max", "avg"])
    ap.add_argument("--hire_reset_open", type=int, default=15)
    ap.add_argument("--hire_reset_grow", type=int, default=6)
    return ap.parse_args()


def _resolve_in_npy(in_path: Path) -> Path:
    if in_path.is_dir():
        for name in ("frames.npy", "binary.npy"):
            cand = in_path / name
            if cand.exists():
                return cand
        npy = sorted(p for p in in_path.iterdir() if p.suffix.lower() == ".npy")
        if npy:
            return npy[0]
        raise FileNotFoundError(f"No .npy under {in_path}")
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    return in_path


def _build_hire_override(args: argparse.Namespace, spad_subsampling: int):
    from ultralytics.models.yolo.pose.spad_preprocessors import build_spad_preprocessor

    kwargs = {
        "subsampling": int(spad_subsampling),
        "sample_rate_hz": float(args.spad_bin_rate_hz),
        "fast_bins": int(args.hire_fast_bins),
        "slow_bins": int(args.hire_slow_bins),
        "surprise_bins": int(args.hire_surprise_bins),
        "mix_hold_bins": int(args.hire_mix_hold_bins),
        "mix_bins": float(args.hire_mix_bins),
        "mix_theta": float(args.hire_mix_theta),
        "mix_floor": float(args.hire_mix_floor),
        "theta_on": float(args.hire_theta_on),
        "theta_off": float(args.hire_theta_off),
        "theta_grow": float(args.hire_theta_grow),
        "confirm_bins": int(args.hire_confirm_bins),
        "cooldown_bins": int(args.hire_cooldown_bins),
        "spatial_kernel": int(args.hire_spatial_kernel),
        "gate_pool": str(args.hire_gate_pool),
        "reset_open": int(args.hire_reset_open),
        "reset_grow": int(args.hire_reset_grow),
    }
    return build_spad_preprocessor("hire", kwargs=kwargs)


def main() -> None:
    args = _parse_args()
    npy_path = _resolve_in_npy(args.in_path)
    sample_name = npy_path.parent.name if npy_path.name in {"frames.npy", "binary.npy"} else npy_path.stem
    out_dir = args.save_dir / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = dsp._resolve_device(args.device)
    yolo = YOLO(str(args.ckpt))
    spad_model = yolo.model
    if not getattr(spad_model, "spad_enabled", False) or not hasattr(spad_model, "preprocessor"):
        raise TypeError(
            f"Checkpoint {args.ckpt} is not a SpadPoseModel "
            f"(got {spad_model.__class__.__name__})"
        )
    spad_model.to(device)
    spad_model.eval()
    dsp._configure_model_spad_bin_rate(spad_model, current_bin_rate_hz=float(args.spad_bin_rate_hz))
    spad_model.spad_detect_imgsz = int(args.imgsz)
    if int(args.imgsz) > 0:
        print(f"Detector letterbox imgsz={int(args.imgsz)} (preds scaled back to native recon)")

    names = yolo.names
    kpt_shape = getattr(spad_model, "kpt_shape", (21, 3))
    trained_chunk_t = dsp._trained_chunk_t(spad_model)
    pre = getattr(spad_model, "preprocessor", None)
    subsampling = int(getattr(pre, "subsampling", 1) or 1)

    if args.preprocessor_override == "hire":
        emit = int(args.chunk_size) if int(args.chunk_size) > 0 else max(subsampling, 1)
        spad_model.preprocessor = _build_hire_override(args, spad_subsampling=emit).to(device)
        spad_model.preprocessor_name = "hire"
        subsampling = emit
        print(f"preprocessor override → hire (subsampling={emit})")

    if int(args.chunk_size) > 0:
        chunk_t = int(args.chunk_size)
    elif trained_chunk_t is not None:
        chunk_t = int(trained_chunk_t)
    else:
        chunk_t = max(subsampling, 320)
    stride = int(args.chunk_stride) if int(args.chunk_stride) > 0 else chunk_t

    t_begin = max(0, int(args.start_bin))
    t_end_arg = int(args.end_bin)
    trail_mode = bool(args.trail_composite)
    trail_pale = float(args.trail_pale)
    trail_deep = float(args.trail_deep)
    if trail_mode and not (0.0 <= trail_deep <= 1.0 and 0.0 <= trail_pale <= 1.0):
        raise ValueError(
            f"Need trail_pale/trail_deep in [0,1], got pale={trail_pale}, deep={trail_deep}"
        )

    tracker = None
    if args.tracker != "none":
        tracker = dsp._init_tracker(args.tracker, frame_rate=args.frame_rate, class_names=names)

    sources = list(dsp._iter_raw_video_sources_from_sample_path(npy_path))
    if not sources:
        raise RuntimeError(f"No SPAD sources in {npy_path}")

    if stride < chunk_t:
        print(
            f"Warning: chunk_stride={stride} < chunk_size={chunk_t} overlaps bins; "
            "streaming will double-count overlapping bins. Prefer stride == chunk_size."
        )

    mode_desc = (
        f"trail_composite pose-only light→dark (pale={trail_pale:g}, deep={trail_deep:g}, alpha=1)"
        if trail_mode
        else "pose-only per-frame, 1 color / ID"
    )
    print(
        f"ckpt={args.ckpt} device={device} pre={getattr(spad_model, 'preprocessor_name', '?')} "
        f"chunk={chunk_t} stride={stride} bins={dsp._video_num_bins(sources[0])} "
        f"start_bin={t_begin} end_bin={t_end_arg or 'EOF'} | {mode_desc}"
    )
    print(f"save → {out_dir}")

    global_frame_idx = 0
    for video_idx, source in enumerate(sources):
        total_bins = dsp._video_num_bins(source)
        t_end = total_bins if t_end_arg <= 0 else min(total_bins, t_end_arg)
        if t_begin >= t_end:
            raise ValueError(f"Empty bin range: start_bin={t_begin} end_bin={t_end}")
        n_trail_steps = _planned_trail_steps(
            t_begin, t_end, chunk_t, stride, int(args.max_chunks)
        )
        if trail_mode:
            print(
                f"  planned trail steps={n_trail_steps} "
                f"over bins [{t_begin},{t_end}) chunk={chunk_t} stride={stride}",
                flush=True,
            )
            if n_trail_steps <= 0:
                raise ValueError(
                    f"No trail steps planned for bins [{t_begin},{t_end}) "
                    f"chunk={chunk_t} stride={stride}"
                )
        if tracker is not None:
            tracker.reset()

        # True streaming: HIRE + temporal plugins continue across chunks.
        if hasattr(spad_model, "spad_begin_stream"):
            spad_model.spad_begin_stream()
        else:
            spad_model.spad_set_online_inference(True)
            spad_model.spad_clear_plugin_states()

        trail_layers: list[np.ndarray] = []
        trail_hw: tuple[int, int] | None = None
        chunk_i = 0
        try:
            for t0 in range(t_begin, t_end, stride):
                if int(args.max_chunks) > 0 and chunk_i >= int(args.max_chunks):
                    break
                t1 = min(t_end, t0 + chunk_t)
                if t1 <= t0:
                    break
                raw_chunk = dsp._slice_raw_chunk(source, t0, t1, packed_ch_order=args.packed_ch_order)
                # Do not pad short tails when streaming — fake repeated bins would pollute state.
                pad_tail = bool(args.tail_pad) and (t1 >= total_bins)
                raw_chunk = dsp._prepare_raw_chunk_for_spad(
                    raw_chunk, chunk_t=chunk_t, tail_pad_full=pad_tail
                )
                if raw_chunk is None:
                    chunk_i += 1
                    continue

                spad_model.spad_packed_nch = int(source.packed_nch)
                if tracker is not None:
                    dsp._set_velocity_field_on_preprocessor(spad_model.preprocessor, tracker)

                with torch.inference_mode():
                    video_tensor = torch.from_numpy(raw_chunk).unsqueeze(0).to(device)
                    raw_preds = spad_model(video_tensor)
                    preds = dsp._postprocess_pose_predictions(
                        raw_preds,
                        conf=args.det_thresh,
                        iou=args.iou,
                        nc=len(names),
                        max_det=args.max_det,
                        kpt_shape=kpt_shape,
                    )
                    preds = dsp._scale_pose_preds_to_native(
                        preds,
                        scale_meta=getattr(spad_model, "spad_scale_meta", None),
                        kpt_shape=kpt_shape,
                    )
                    recon_frames_bgr = dsp._recon_frames_bgr(spad_model, batch_index=0)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                results = dsp._results_from_preds(
                    preds,
                    recon_frames_bgr,
                    names,
                    prefix=f"{sample_name}_cube{video_idx:05d}_t{t0:06d}_{t1:06d}",
                    kpt_shape=kpt_shape,
                )

                bg_bgr = None
                if (not trail_mode) and args.vis_bg == "sum":
                    bg_bgr = dsp._raw_sum_bgr(raw_chunk, packed_nch=source.packed_nch)

                shade = _trail_shade(chunk_i, n_trail_steps) if trail_mode else 0.0

                def _trail_color_for_id(tid: int, _shade: float = shade) -> tuple[int, int, int]:
                    return _shade_bgr(
                        _id_color_bgr(int(tid)),
                        _shade,
                        pale=trail_pale,
                        deep=trail_deep,
                    )

                saved_stem = None
                if trail_mode:
                    if not results:
                        chunk_i += 1
                        continue
                    result = results[-1]
                    if tracker is not None:
                        # Step tracker on every emit so IDs stay consistent; keep last.
                        tracked = None
                        for res in results:
                            tracked = dsp._apply_tracker(res, tracker)
                        result = tracked
                    recon = (
                        np.ascontiguousarray(result.orig_img.copy())
                        if getattr(result, "orig_img", None) is not None
                        else np.zeros((512, 512, 3), dtype=np.uint8)
                    )
                    h, w = recon.shape[:2]
                    if trail_hw is None:
                        trail_hw = (h, w)
                    elif trail_hw != (h, w):
                        raise ValueError(
                            f"Trail frame size changed from {trail_hw} to {(h, w)} at bins [{t0},{t1})"
                        )
                    layer = np.zeros((h, w, 3), dtype=np.uint8)
                    layer = _draw_result_poses(
                        layer,
                        result,
                        kpt_thresh=float(args.kpt_thresh),
                        draw_boxes=False,
                        color_for_id=_trail_color_for_id,
                    )
                    trail_layers.append(layer)
                    global_frame_idx += 1
                    chunk_i += 1
                    print(
                        f"  [{t0:06d},{t1:06d}) → trail step {chunk_i}/{n_trail_steps} "
                        f"shade={shade:.3f}",
                        flush=True,
                    )
                    continue

                for result in results:
                    if tracker is not None:
                        result = dsp._apply_tracker(result, tracker)

                    recon = (
                        np.ascontiguousarray(result.orig_img.copy())
                        if getattr(result, "orig_img", None) is not None
                        else np.zeros((512, 512, 3), dtype=np.uint8)
                    )
                    if args.vis_bg == "black":
                        vis = np.zeros_like(recon)
                    elif args.vis_bg == "recon":
                        vis = recon.copy()
                    else:
                        vis = bg_bgr.copy() if bg_bgr is not None else np.zeros_like(recon)

                    vis = _draw_result_poses(vis, result, kpt_thresh=float(args.kpt_thresh))

                    saved_stem = f"cube{video_idx:05d}_t{t0:06d}_{t1:06d}_frame{global_frame_idx:07d}"
                    cv2.imwrite(str(out_dir / f"{saved_stem}.png"), vis)
                    global_frame_idx += 1

                chunk_i += 1
                if saved_stem is not None:
                    print(
                        f"  [{t0:06d},{t1:06d}) → {saved_stem}.png "
                        f"(chunk {chunk_i}, frame {global_frame_idx - 1})",
                        flush=True,
                    )
        finally:
            if hasattr(spad_model, "spad_end_stream"):
                spad_model.spad_end_stream()

        if trail_mode:
            if not trail_layers or trail_hw is None:
                print(f"Warning: video {video_idx}: no trail layers in [{t_begin},{t_end})")
                continue
            h, w = trail_hw
            canvas = np.zeros((h, w, 4), dtype=np.float32)
            for layer in trail_layers:
                _composite_layer_rgba(canvas, layer, opacity=1.0)
            out_u8 = np.clip(np.round(canvas * 255.0), 0, 255).astype(np.uint8)
            if str(args.trail_name).strip():
                trail_name = str(args.trail_name).strip()
                if not trail_name.lower().endswith(".png"):
                    trail_name = f"{trail_name}.png"
            else:
                trail_name = f"trail_t{t_begin:06d}_{t_end:06d}.png"
            if len(sources) > 1:
                trail_name = f"cube{video_idx:05d}_{trail_name}"
            trail_path = out_dir / trail_name
            cv2.imwrite(str(trail_path), out_u8)
            print(
                f"  trail → {trail_path} ({len(trail_layers)} poses, RGBA, alpha=1)",
                flush=True,
            )

    if trail_mode:
        print(f"Done. Trail composite from {global_frame_idx} poses → {out_dir}")
    else:
        print(f"Done. Wrote {global_frame_idx} pose frames → {out_dir}")


if __name__ == "__main__":
    main()
