#!/usr/bin/env python3
"""Per-chunk HIRE SPAD pose inference → pose overlay PNGs on reconstruction.

Runs a trained ``SpadPoseModel`` (HIRE preprocessor) on packed/raw ``frames.npy``
in **causal streaming** mode: preprocessor + detector temporal plugins carry state
across chunks (no mid-video reset). Saves pose visualizations on the recon frame.

Output naming (for ``vis_hire_pose_3d.py``)::

    {save_dir}/{sample}/cubeXXXXX_tTTTTTT_TTTTTT_frameFFFFFFF.png

Examples
--------
python ultralytics/vis_hire_pose.py \\
  --in_path /path/to/frames.npy \\
  --ckpt /path/to/last.pt \\
  --save_dir /tmp/hire_pose \\
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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HIRE chunk pose viz on recon frames")
    ap.add_argument("--in_path", type=Path, required=True, help="frames.npy or dir with frames.npy")
    ap.add_argument("--save_dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True, help="Trained SpadPoseModel .pt")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument(
        "--tracker",
        type=str,
        default="none",
        choices=["none", "bytetrack", "botsort", "spad_tracker", "posetrack", "spad_posetrack"],
        help="Tracker; default none (per-chunk independent IDs)",
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
    ap.add_argument("--max_chunks", type=int, default=0, help="0 = all chunks")
    ap.add_argument("--start_bin", type=int, default=0, help="Skip bins before this index")
    ap.add_argument("--tail_pad", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vis_bg", type=str, default="recon", choices=["sum", "recon"])
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

    print(
        f"ckpt={args.ckpt} device={device} pre={getattr(spad_model, 'preprocessor_name', '?')} "
        f"chunk={chunk_t} stride={stride} bins={dsp._video_num_bins(sources[0])} "
        f"stream=causal (state carries across chunks)"
    )
    print(f"save → {out_dir}")

    global_frame_idx = 0
    for video_idx, source in enumerate(sources):
        total_bins = dsp._video_num_bins(source)
        t_begin = max(0, int(args.start_bin))
        if tracker is not None:
            tracker.reset()

        # True streaming: HIRE + temporal plugins continue across chunks.
        if hasattr(spad_model, "spad_begin_stream"):
            spad_model.spad_begin_stream()
        else:
            spad_model.spad_set_online_inference(True)
            spad_model.spad_clear_plugin_states()

        chunk_i = 0
        try:
            for t0 in range(t_begin, total_bins, stride):
                if int(args.max_chunks) > 0 and chunk_i >= int(args.max_chunks):
                    break
                t1 = min(total_bins, t0 + chunk_t)
                raw_chunk = dsp._slice_raw_chunk(source, t0, t1, packed_ch_order=args.packed_ch_order)
                # Do not pad short tails when streaming — fake repeated bins would pollute state.
                pad_tail = bool(args.tail_pad) and (t1 >= total_bins)
                raw_chunk = dsp._prepare_raw_chunk_for_spad(
                    raw_chunk, chunk_t=chunk_t, tail_pad_full=pad_tail
                )
                if raw_chunk is None:
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
                if args.vis_bg == "sum":
                    bg_bgr = dsp._raw_sum_bgr(raw_chunk, packed_nch=source.packed_nch)

                saved_stem = None
                for result in results:
                    if tracker is not None:
                        result = dsp._apply_tracker(result, tracker)
                    recon = (
                        np.ascontiguousarray(result.orig_img.copy())
                        if getattr(result, "orig_img", None) is not None
                        else np.zeros((512, 512, 3), dtype=np.uint8)
                    )
                    if args.vis_bg == "recon":
                        vis = recon.copy()
                    else:
                        vis = bg_bgr.copy() if bg_bgr is not None else np.zeros_like(recon)

                    if result.boxes is not None and len(result.boxes):
                        track_ids = result.boxes.id
                        if track_ids is None:
                            track_ids = torch.arange(len(result.boxes), device=result.boxes.data.device)
                        track_id = track_ids.cpu().numpy()
                        boxes = result.boxes.xyxy.cpu().numpy()
                        box_confs = result.boxes.conf.cpu().numpy()
                        handedness = result.boxes.cls.cpu().numpy()
                        poses = None
                        if getattr(result, "keypoints", None) is not None:
                            poses = result.keypoints.data.cpu().numpy()

                        for j, tid in enumerate(track_id):
                            box_xyxyc = np.concatenate([boxes[j], [box_confs[j]]], axis=0)
                            vis = dsp.draw_bbox(vis, int(tid), box_xyxyc, float(handedness[j]))
                            if poses is not None and j < len(poses):
                                vis = dsp.draw_pose(vis, poses[j], thresh=float(args.kpt_thresh))

                    saved_stem = f"cube{video_idx:05d}_t{t0:06d}_{t1:06d}_frame{global_frame_idx:07d}"
                    cv2.imwrite(str(out_dir / f"{saved_stem}.png"), vis)
                    cv2.imwrite(str(out_dir / f"{saved_stem}_recon.png"), recon)
                    global_frame_idx += 1

                chunk_i += 1
                if saved_stem is not None:
                    print(
                        f"  [{t0:06d},{t1:06d}) → {saved_stem}.png "
                        f"(chunk {chunk_i}, frames {global_frame_idx})",
                        flush=True,
                    )
        finally:
            if hasattr(spad_model, "spad_end_stream"):
                spad_model.spad_end_stream()

    print(f"Done. Wrote {global_frame_idx} pose frames → {out_dir}")


if __name__ == "__main__":
    main()
