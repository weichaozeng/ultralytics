# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Frequency-sweep sequence SPAD pose inference (raw + test_json only).

Controls prediction cadence via ``--pred_fps`` / ``--stride_bins`` while keeping the
checkpoint chunk window. Emits one pose per window at the chunk end bin.

Example (8 kHz, 50 fps readout):
  python test_spad_pose_sequence_freq.py \\
    --ckpt <hire_8kHz.pt> --test_json <test_8kHz.json> \\
    --spad-bin-rate-hz 8000 --pred_fps 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO

from det_spad_pose import (
    _apply_tracker,
    _configure_model_spad_bin_rate,
    _init_tracker,
    _iter_raw_video_sources_from_sample_path,
    _postprocess_pose_predictions,
    _prepare_raw_chunk_for_spad,
    _recon_frames_bgr,
    _resolve_device,
    _results_from_preds,
    _scale_pose_preds_to_native,
    _set_velocity_field_on_preprocessor,
    _slice_raw_chunk,
    _video_num_bins,
)
from test_spad_pose_sequence import (
    PREPROCESSOR_CHOICES,
    TRACKER_CHOICES,
    RenderChunkRecord,
    _apply_sequence_preprocessor_override,
    _build_preprocessor_kwargs,
    _build_vis_frame,
    _choose_effective_preprocessor,
    _dated_model_name,
    _default_test_name_from_json,
    _eval_dir,
    _frame_record_from_result,
    _iter_chunk_outputs,
    _json_default,
    _load_gt_frame_count,
    _maybe_set_velocity_field,
    _model_name_from_ckpt,
    _names_to_dict,
    _resolve_chunk_size,
    _resolve_input_gamma,
    _resolve_spad_bin_rate_hz,
    _resolve_spad_bins_per_gt,
    _resolve_spad_subsampling,
    _sample_specs_from_json,
    _vis_dir,
)


def _resolve_pred_cadence(args, *, bin_rate_hz: float) -> tuple[float, int]:
    """Return (pred_fps, stride_bins). Prefer explicit --stride_bins when set."""
    stride_bins = int(getattr(args, "stride_bins", 0) or 0)
    pred_fps_cli = float(getattr(args, "pred_fps", 0.0) or 0.0)
    if stride_bins > 0:
        pred_fps = float(bin_rate_hz) / float(stride_bins)
        if pred_fps_cli > 0 and abs(pred_fps - pred_fps_cli) > 1e-3:
            print(
                f"Warning: --stride_bins={stride_bins} implies pred_fps={pred_fps:g}, "
                f"ignoring --pred_fps={pred_fps_cli:g}"
            )
        return pred_fps, stride_bins
    if pred_fps_cli <= 0:
        raise ValueError("Provide --pred_fps > 0 or --stride_bins > 0")
    stride_bins = max(1, int(round(float(bin_rate_hz) / pred_fps_cli)))
    pred_fps = float(bin_rate_hz) / float(stride_bins)
    return pred_fps, stride_bins


def _hire_cli_kwargs(args, *, bin_rate_hz: float) -> dict[str, Any]:
    """Collect CLI HIRE hyperparams for rebuild / in-place ``update_hyperparams``."""
    return {
        "sample_rate_hz": float(bin_rate_hz),
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
        "normalize": bool(args.hire_normalize),
        "quantile": float(args.hire_quantile),
    }


def _hire_config_snapshot(preprocessor) -> dict[str, Any] | None:
    """Serialize live HIRE attributes for prediction metadata."""
    if preprocessor is None or not hasattr(preprocessor, "update_hyperparams"):
        return None
    if not all(hasattr(preprocessor, k) for k in ("fast_bins", "slow_bins", "theta_on")):
        return None
    mix_floor = getattr(preprocessor, "mix_floor", None)
    theta_grow = getattr(preprocessor, "theta_grow", None)
    return {
        "sample_rate_hz": float(getattr(preprocessor, "sample_rate_hz", 0.0)),
        "fast_bins": int(preprocessor.fast_bins),
        "slow_bins": int(preprocessor.slow_bins),
        "surprise_bins": int(preprocessor.surprise_bins),
        "mix_hold_bins": int(preprocessor.mix_hold_bins),
        "mix_bins": float(preprocessor.mix_bins),
        "mix_theta": float(preprocessor.mix_theta),
        "mix_floor": None if mix_floor is None else float(mix_floor),
        "theta_on": float(preprocessor.theta_on),
        "theta_off": float(preprocessor.theta_off),
        "theta_grow": None if theta_grow is None else float(theta_grow),
        "confirm_bins": int(preprocessor.confirm_bins),
        "cooldown_bins": int(preprocessor.cooldown_bins),
        "spatial_kernel": int(preprocessor.spatial_kernel),
        "gate_pool": str(preprocessor.gate_pool),
        "reset_open": int(preprocessor.reset_open),
        "reset_grow": int(preprocessor.reset_grow),
        "normalize": bool(getattr(preprocessor, "normalize", False)),
        "quantile": float(getattr(preprocessor, "quantile", 1.0)),
    }


def _apply_hire_cli_overrides(spad_model, args, *, bin_rate_hz: float) -> dict[str, Any] | None:
    """Apply ``--hire-*`` CLI onto the live HIRE preprocessor (ckpt or rebuilt)."""
    prep = getattr(spad_model, "preprocessor", None)
    if prep is None or not hasattr(prep, "update_hyperparams"):
        return None
    name = str(getattr(spad_model, "preprocessor_name", "") or "").strip().lower()
    if name and name != "hire" and prep.__class__.__name__ not in {"HIRE", "HIREFrame"}:
        return None
    kwargs = _hire_cli_kwargs(args, bin_rate_hz=bin_rate_hz)
    prep.update_hyperparams(**kwargs)
    snap = _hire_config_snapshot(prep)
    if snap is not None:
        print(
            "HIRE CLI overrides applied: "
            f"W_f/s/S={snap['fast_bins']}/{snap['slow_bins']}/{snap['surprise_bins']}, "
            f"θ_on/off={snap['theta_on']:g}/{snap['theta_off']:g}, "
            f"mix_hold={snap['mix_hold_bins']}, mix_τ={snap['mix_bins']:g}, "
            f"normalize={snap['normalize']}, quantile={snap['quantile']:g}"
        )
    return snap


def _freq_chunk_records(
    *,
    total_bins: int,
    chunk_size: int,
    stride_bins: int,
    spad_bins_per_gt: int,
    n_gt: int,
    packed_nch: int,
) -> list[RenderChunkRecord]:
    """Bin-based GT-aligned windows: first emit at end_bin=chunk_size, then +stride_bins."""
    if chunk_size <= 0 or stride_bins <= 0 or spad_bins_per_gt <= 0:
        raise ValueError(
            f"chunk_size, stride_bins, spad_bins_per_gt must be > 0, got "
            f"{chunk_size}, {stride_bins}, {spad_bins_per_gt}"
        )
    max_end = min(int(total_bins), int(n_gt * spad_bins_per_gt))
    if max_end < chunk_size:
        return []

    records: list[RenderChunkRecord] = []
    for chunk_index, end_bin in enumerate(range(chunk_size, max_end + 1, stride_bins)):
        start_bin = int(end_bin - chunk_size)
        records.append(
            RenderChunkRecord(
                chunk_index=int(chunk_index),
                spad_start_bin=int(start_bin),
                spad_end_bin=int(end_bin),
                chunk_size=int(chunk_size),
                target_gt_time=float(end_bin) / float(spad_bins_per_gt),
                packed_nch=int(packed_nch),
            )
        )
    return records


def parse_args():
    ap = argparse.ArgumentParser(
        description="Frequency-sweep sequence SPAD pose test (raw + test_json). "
        "Prediction cadence via --pred_fps / --stride_bins; chunk window from ckpt."
    )
    ap.add_argument("--test_json", type=str, required=True, help="VisionSIM test split JSON")
    ap.add_argument("--test_name", type=str, default=None, help="Base output test-set folder name")
    ap.add_argument("--ckpt", type=str, required=True, help="Trained sequence-mode SPAD pose checkpoint")
    ap.add_argument("--device", type=str, default="", help="Torch device, e.g. cpu / cuda:0 / mps")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--tracker", type=str, default="botsort", choices=TRACKER_CHOICES)
    ap.add_argument("--frame_rate", type=int, default=25, help="Tracker / video FPS hint")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument(
        "--spad-bin-rate-hz",
        type=float,
        default=8000.0,
        help="Raw-bin frequency of the inference input (default 8000).",
    )
    ap.add_argument(
        "--pred_fps",
        type=float,
        default=0.0,
        help="Desired prediction FPS. stride_bins = round(bin_rate / pred_fps). "
        "Ignored when --stride_bins > 0.",
    )
    ap.add_argument(
        "--stride_bins",
        type=int,
        default=0,
        help="Explicit bin stride between successive chunk end bins. 0 = derive from --pred_fps.",
    )
    ap.add_argument("--preprocessor", type=str, default="model", choices=PREPROCESSOR_CHOICES)
    ap.add_argument(
        "--preprocessor-override",
        type=str,
        default="none",
        choices=["none", "ppb", "sum", "ema", "stea", "hyb", "hire"],
        help="Deprecated alias for --preprocessor when --preprocessor model.",
    )
    ap.add_argument(
        "--spad_chunk_t",
        "--cube_chunk_t",
        dest="cube_chunk_t",
        type=int,
        default=0,
        help="Raw-bin chunk size (integration window). 0 = from checkpoint.",
    )
    ap.add_argument("--spad_bins_per_gt", type=int, default=0, help="0 = derive from bin rate / 125 Hz GT")
    ap.add_argument(
        "--spad_subsampling",
        type=int,
        default=0,
        help="Train-time preprocessor emit interval (0 = from checkpoint).",
    )
    ap.add_argument("--input_gamma", type=float, default=0.0)
    ap.add_argument(
        "--spad_online",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Carry detector temporal state across chunks (default).",
    )
    ap.add_argument("--ppb-bocpd-gamma", type=float, default=1e-3)
    ap.add_argument("--ppb-memory-size", type=int, default=10)
    ap.add_argument("--ppb-quantile", type=float, default=1.0)
    ap.add_argument("--ppb-normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppb-min-filter-size", type=int, default=5)
    ap.add_argument("--ema-alpha", type=float, default=0.01)
    ap.add_argument("--stea-fast-window", type=int, default=16)
    ap.add_argument("--stea-slow-window", type=int, default=128)
    ap.add_argument("--stea-temporal-window", type=int, default=5)
    ap.add_argument("--stea-fast-tau", type=float, default=6.0)
    ap.add_argument("--stea-motion-sharpness", type=float, default=60.0)
    ap.add_argument("--stea-motion-threshold", type=float, default=0.07)
    ap.add_argument("--stea-stable-prior", type=float, default=16.0)
    ap.add_argument("--stea-normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stea-quantile", type=float, default=1.0)
    # HIRE hyperparams (applied to live HIRE via update_hyperparams; also used by --preprocessor hire)
    ap.add_argument("--hire-fast-bins", type=int, default=24, help="HIRE fast EMA window W_f (bins)")
    ap.add_argument("--hire-slow-bins", type=int, default=160, help="HIRE slow EMA window W_s (bins)")
    ap.add_argument("--hire-surprise-bins", type=int, default=4, help="HIRE surprise EMA window W_S (bins)")
    ap.add_argument("--hire-mix-hold-bins", type=int, default=80, help="HIRE I^f hold length H after hard reset")
    ap.add_argument("--hire-mix-bins", type=float, default=12.0, help="HIRE mix decay τ (bins)")
    ap.add_argument("--hire-mix-theta", type=float, default=0.06, help="HIRE soft-gate mix threshold")
    ap.add_argument(
        "--hire-mix-floor",
        type=float,
        default=-1.0,
        help="HIRE soft-gate deadzone; <0 follows theta_off",
    )
    ap.add_argument("--hire-theta-on", type=float, default=0.08, help="HIRE change-point enter threshold")
    ap.add_argument("--hire-theta-off", type=float, default=0.02, help="HIRE change-point leave threshold")
    ap.add_argument(
        "--hire-theta-grow",
        type=float,
        default=-1.0,
        help="HIRE geodesic grow threshold; <0 follows theta_off",
    )
    ap.add_argument("--hire-confirm-bins", type=int, default=4, help="HIRE confirm bins before hard reset")
    ap.add_argument("--hire-cooldown-bins", type=int, default=0, help="HIRE cooldown (legacy / unused)")
    ap.add_argument("--hire-spatial-kernel", type=int, default=5, help="HIRE spatial pool kernel (odd)")
    ap.add_argument("--hire-gate-pool", type=str, default="max", choices=["max", "avg"], help="HIRE gate pool")
    ap.add_argument("--hire-reset-open", type=int, default=15, help="HIRE reset open kernel (odd)")
    ap.add_argument("--hire-reset-grow", type=int, default=6, help="HIRE geodesic grow steps")
    ap.add_argument(
        "--hire-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantile-normalize HIRE recon (default true; --no-hire-normalize to disable)",
    )
    ap.add_argument("--hire-quantile", type=float, default=1.0, help="HIRE normalize quantile")
    ap.add_argument("--hyb-motion-sharpness", type=float, default=60.0)
    ap.add_argument("--hyb-motion-threshold", type=float, default=0.05)
    ap.add_argument("--hyb-warp-block-size", type=int, default=16)
    ap.add_argument("--hyb-source-space", type=str, default="rgb", choices=["rgb", "raw"])
    ap.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Letterbox reconstructed frames for the detector. 0 disables.",
    )
    ap.add_argument(
        "--expected_w",
        type=int,
        default=0,
        help="Unpacked RGB-plane width for packed SPAD. 0 = auto.",
    )
    ap.add_argument("--vis", nargs="?", const="video", default="none", choices=["none", "image", "video"])
    ap.add_argument("--vis_bg", type=str, default="recon", choices=["sum", "recon"])
    ap.add_argument("--save_readrgb", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt)
    model_name = _model_name_from_ckpt(ckpt)
    run_name = _dated_model_name(model_name)

    test_json = Path(args.test_json)
    if not test_json.is_file():
        raise FileNotFoundError(f"Test split JSON not found: {test_json}")

    yolo = YOLO(str(ckpt))
    spad_model = yolo.model
    if not getattr(spad_model, "spad_enabled", False) or not hasattr(spad_model, "preprocessor"):
        raise TypeError(
            f"Checkpoint {args.ckpt} is not a trained SPAD pose model. "
            f"Loaded type: {spad_model.__class__.__name__}"
        )
    if hasattr(spad_model, "frame_adapter_name"):
        raise TypeError(
            f"Checkpoint {args.ckpt} is a frame-mode SPAD pose model. "
            "Use test_spad_pose_frame.py instead."
        )

    device = _resolve_device(args.device)
    spad_model.to(device)
    spad_model.eval()
    bin_rate_hz = float(args.spad_bin_rate_hz) if float(args.spad_bin_rate_hz) > 0 else 8000.0
    _configure_model_spad_bin_rate(spad_model, current_bin_rate_hz=bin_rate_hz)
    spad_model.spad_detect_imgsz = int(args.imgsz)

    names = yolo.names
    kpt_shape = getattr(spad_model, "kpt_shape", (21, 3))
    effective_preprocessor = _choose_effective_preprocessor(spad_model, args)
    if effective_preprocessor == "hyb" and args.tracker not in {"spad_tracker", "spad_posetrack"}:
        raise ValueError("--preprocessor hyb requires --tracker spad_tracker or spad_posetrack")

    sample_specs = _sample_specs_from_json(test_json)
    base_test_name = args.test_name or _default_test_name_from_json(test_json)

    chunk_size = _resolve_chunk_size(spad_model, args)
    if chunk_size <= 0:
        raise ValueError(f"Sequence chunk size must be positive, got {chunk_size}")
    # Prefer CLI bin rate for GT mapping in this frequency-sweep script so a
    # mismatched ckpt train rate cannot silently remap 8 kHz data to 2 kHz bins/GT.
    if int(args.spad_bins_per_gt) > 0:
        spad_bins_per_gt = int(args.spad_bins_per_gt)
    else:
        spad_bins_per_gt = max(1, int(round(float(bin_rate_hz) / 125.0)))
        ckpt_bins = _resolve_spad_bins_per_gt(spad_model, args, ckpt=yolo.ckpt)
        if ckpt_bins != spad_bins_per_gt:
            print(
                f"Warning: using spad_bins_per_gt={spad_bins_per_gt} from "
                f"--spad-bin-rate-hz={bin_rate_hz:g} (ckpt would suggest {ckpt_bins})"
            )
    pred_fps, stride_bins = _resolve_pred_cadence(args, bin_rate_hz=bin_rate_hz)
    # Pretty folder tag: prefer CLI pred_fps when it was the source of truth.
    fps_tag = int(round(float(args.pred_fps))) if float(args.pred_fps) > 0 else int(round(pred_fps))
    test_name = f"{base_test_name}_pred{fps_tag}fps"

    print(
        f"Freq sequence windowing: chunk_size={chunk_size}, "
        f"spad_bins_per_gt={spad_bins_per_gt}, pred_fps={pred_fps:g}, "
        f"stride_bins={stride_bins}, bin_rate_hz={bin_rate_hz:g}, cache_mode=raw"
    )

    spad_subsampling = _resolve_spad_subsampling(spad_model, args, ckpt=yolo.ckpt)
    input_gamma = _resolve_input_gamma(spad_model, args)
    _ = _build_preprocessor_kwargs(
        args, preprocessor_name=effective_preprocessor, spad_subsampling=spad_subsampling
    )

    _apply_sequence_preprocessor_override(
        spad_model,
        args,
        device,
        effective_preprocessor,
        spad_subsampling=spad_subsampling,
    )
    if effective_preprocessor == "ema" and hasattr(spad_model, "preprocessor") and spad_model.preprocessor is not None:
        alpha = float(args.ema_alpha)
        if alpha <= 0.0:
            alpha = 2.0 / (max(int(spad_subsampling), 1) + 1)
        if hasattr(spad_model.preprocessor, "ema_alpha"):
            old = float(getattr(spad_model.preprocessor, "ema_alpha", alpha))
            spad_model.preprocessor.ema_alpha = float(alpha)
            if abs(old - alpha) > 1e-12:
                print(f"EMA inference: override ema_alpha {old:g} -> {alpha:g}")

    hire_config = _apply_hire_cli_overrides(spad_model, args, bin_rate_hz=bin_rate_hz)
    if hire_config is None and effective_preprocessor == "hire":
        # Fallback snapshot after rebuild if update path was skipped.
        hire_config = _hire_config_snapshot(getattr(spad_model, "preprocessor", None))

    spad_model.spad_cache_mode = "raw"
    spad_model.spad_set_online_inference(bool(args.spad_online))

    out_dir = _eval_dir(run_name, test_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_root = _vis_dir(run_name, test_name) if args.vis != "none" else None
    if vis_root is not None:
        vis_root.mkdir(parents=True, exist_ok=True)

    tracker = None if args.tracker == "none" else _init_tracker(args.tracker, frame_rate=args.frame_rate, class_names=names)

    for sample_spec in sample_specs:
        sample_name = str(sample_spec["sample_name"])
        sample_path = Path(sample_spec["sample_path"])
        out_path = out_dir / f"{sample_name}.json"
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing file: {out_path}")
            continue

        if tracker is not None:
            tracker.reset()
        if args.spad_online and hasattr(spad_model, "spad_clear_plugin_states"):
            spad_model.spad_clear_plugin_states()

        global_frame_idx = 0
        sample_vis_dir = None
        video_writer = None
        if vis_root is not None and args.vis == "image":
            sample_vis_dir = vis_root / sample_name
            sample_vis_dir.mkdir(parents=True, exist_ok=True)
        video_path = vis_root / f"{sample_name}.mp4" if vis_root is not None and args.vis == "video" else None

        sample_record: dict[str, Any] = {
            "metadata": {
                "format": "spadhand_spad_predictions_v1",
                "ckpt": str(ckpt),
                "model_name": model_name,
                "run_name": run_name,
                "test_name": test_name,
                "sample_name": sample_name,
                "det_thresh": float(args.det_thresh),
                "iou": float(args.iou),
                "max_det": int(args.max_det),
                "tracker": args.tracker,
                "frame_rate": int(args.frame_rate),
                "packed_ch_order": args.packed_ch_order,
                "tail_pad": False,
                "drop_tail": True,
                "vis_mode": str(args.vis),
                "names": _names_to_dict(names),
                "kpt_shape": list(map(int, kpt_shape)),
                "spad_cache_mode": "raw",
                "preprocessor": effective_preprocessor,
                "spad_chunk_size": int(chunk_size),
                "chunk_t": int(chunk_size),
                "spad_bins_per_gt": int(spad_bins_per_gt),
                "spad_bin_rate_hz": float(bin_rate_hz),
                "pred_fps": float(pred_fps),
                "stride_bins": int(stride_bins),
                "input_gamma": float(input_gamma),
                "spad_online": bool(args.spad_online),
                "source_bin_convention": "chunk_end_bin",
                "imgsz": int(args.imgsz),
                "expected_w": int(args.expected_w),
                "freq_sweep": True,
                "hire": hire_config,
                "hire_fast_bins": int(args.hire_fast_bins),
                "hire_slow_bins": int(args.hire_slow_bins),
                "hire_surprise_bins": int(args.hire_surprise_bins),
                "hire_mix_hold_bins": int(args.hire_mix_hold_bins),
                "hire_mix_bins": float(args.hire_mix_bins),
                "hire_mix_theta": float(args.hire_mix_theta),
                "hire_mix_floor": float(args.hire_mix_floor),
                "hire_theta_on": float(args.hire_theta_on),
                "hire_theta_off": float(args.hire_theta_off),
                "hire_theta_grow": float(args.hire_theta_grow),
                "hire_confirm_bins": int(args.hire_confirm_bins),
                "hire_cooldown_bins": int(args.hire_cooldown_bins),
                "hire_spatial_kernel": int(args.hire_spatial_kernel),
                "hire_gate_pool": str(args.hire_gate_pool),
                "hire_reset_open": int(args.hire_reset_open),
                "hire_reset_grow": int(args.hire_reset_grow),
                "hire_normalize": bool(args.hire_normalize),
                "hire_quantile": float(args.hire_quantile),
            },
            "source": {
                "path": str(sample_path),
                "is_dir": bool(sample_path.is_dir()),
                "split_record": sample_spec["record"],
            },
            "videos": [],
        }

        n_gt = _load_gt_frame_count(sample_spec["record"])
        video_iter = _iter_raw_video_sources_from_sample_path(
            sample_path, expected_w=int(args.expected_w)
        )
        for video_idx, source in enumerate(tqdm(video_iter, desc=f"Testing [{sample_name}]")):
            total_bins = _video_num_bins(source)
            chunk_records = _freq_chunk_records(
                total_bins=total_bins,
                chunk_size=chunk_size,
                stride_bins=stride_bins,
                spad_bins_per_gt=spad_bins_per_gt,
                n_gt=n_gt,
                packed_nch=source.packed_nch,
            )
            train_subsampling = int(
                getattr(getattr(spad_model, "preprocessor", None), "subsampling", spad_subsampling) or 1
            )

            video_record: dict[str, Any] = {
                "video_idx": int(video_idx),
                "layout": source.layout,
                "packed_nch": int(source.packed_nch),
                "expected_w": int(getattr(source, "expected_w", 0) or 0),
                "total_bins": int(total_bins),
                "chunk_t": int(chunk_size),
                "chunk_stride": int(stride_bins),
                "subsampling": int(chunk_size),
                "train_subsampling": int(train_subsampling),
                "pred_fps": float(pred_fps),
                "stride_bins": int(stride_bins),
                "chunks": [],
            }

            if hasattr(spad_model, "spad_begin_stream"):
                spad_model.spad_begin_stream()
                if not args.spad_online:
                    spad_model.spad_set_online_inference(False)
            else:
                spad_model.spad_set_online_inference(bool(args.spad_online))
                if args.spad_online and hasattr(spad_model, "spad_clear_plugin_states"):
                    spad_model.spad_clear_plugin_states()

            try:
                for chunk in chunk_records:
                    t0 = int(chunk.spad_start_bin)
                    t1 = int(chunk.spad_end_bin)
                    raw_chunk = _slice_raw_chunk(source, t0, t1, packed_ch_order=args.packed_ch_order)
                    raw_chunk = _prepare_raw_chunk_for_spad(raw_chunk, chunk_t=chunk_size, tail_pad_full=False)
                    if raw_chunk is None:
                        continue

                    spad_model.spad_packed_nch = int(source.packed_nch)
                    spad_model.spad_cached_confidence_batch = None
                    if tracker is not None:
                        _set_velocity_field_on_preprocessor(spad_model.preprocessor, tracker)
                    else:
                        _maybe_set_velocity_field(spad_model.preprocessor, tracker)

                    with torch.inference_mode():
                        video_tensor = torch.from_numpy(np.ascontiguousarray(raw_chunk)).unsqueeze(0).to(device)
                        if getattr(spad_model, "spad_stream_mode", False):
                            spad_model.spad_stream_bin_offset = int(t0)
                            spad_model.spad_pending_t_index_ll = [int(t1)]
                        raw_preds = spad_model(video_tensor)
                        preds = _postprocess_pose_predictions(
                            raw_preds,
                            conf=args.det_thresh,
                            iou=args.iou,
                            nc=len(names),
                            max_det=args.max_det,
                            kpt_shape=kpt_shape,
                        )
                        preds = _scale_pose_preds_to_native(
                            preds,
                            scale_meta=getattr(spad_model, "spad_scale_meta", None),
                            kpt_shape=kpt_shape,
                        )
                        recon_frames_bgr = _recon_frames_bgr(spad_model, batch_index=0)

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    results = _results_from_preds(
                        preds,
                        recon_frames_bgr,
                        names,
                        prefix=f"{sample_name}_cube{video_idx:05d}_t{t0:06d}_{t1:06d}",
                        kpt_shape=kpt_shape,
                    )

                    chunk_record: dict[str, Any] = {
                        "chunk_idx": int(chunk.chunk_index),
                        "start_bin": int(t0),
                        "end_bin": int(t1),
                        "input_bins": int(t1 - t0),
                        "model_input_bins": int(raw_chunk.shape[0]),
                        "target_gt_time": chunk.target_gt_time,
                        "frames": [],
                    }

                    for result, output_frame_idx, source_bin in _iter_chunk_outputs(
                        results,
                        gt_aligned=True,
                        t0=t0,
                        t1=t1,
                        subsampling=train_subsampling,
                    ):
                        if tracker is not None:
                            result = _apply_tracker(result, tracker)
                        if args.vis != "none":
                            vis, recon, readrgb = _build_vis_frame(
                                result,
                                vis_bg=args.vis_bg,
                                raw_chunk=raw_chunk,
                                packed_nch=source.packed_nch,
                                save_readrgb=bool(args.save_readrgb),
                            )
                            if args.vis == "image" and sample_vis_dir is not None:
                                stem = f"cube{video_idx:05d}_t{t0:06d}_{t1:06d}_frame{global_frame_idx:07d}"
                                cv2.imwrite(str(sample_vis_dir / f"{stem}.png"), vis)
                                cv2.imwrite(str(sample_vis_dir / f"{stem}_recon.png"), recon)
                                if readrgb is not None:
                                    cv2.imwrite(str(sample_vis_dir / f"{stem}_readrgb.png"), readrgb)
                            elif args.vis == "video" and video_path is not None:
                                if video_writer is None:
                                    h, w = vis.shape[:2]
                                    video_writer = cv2.VideoWriter(
                                        str(video_path),
                                        cv2.VideoWriter_fourcc(*"mp4v"),
                                        float(pred_fps),
                                        (w, h),
                                    )
                                video_writer.write(vis)
                        chunk_record["frames"].append(
                            _frame_record_from_result(
                                result,
                                names=names,
                                global_frame_idx=global_frame_idx,
                                video_idx=video_idx,
                                chunk_idx=int(chunk.chunk_index),
                                chunk_start=t0,
                                chunk_end=t1,
                                output_frame_idx=output_frame_idx,
                                source_bin=source_bin,
                            )
                        )
                        global_frame_idx += 1

                    if chunk_record["frames"]:
                        video_record["chunks"].append(chunk_record)
            finally:
                if hasattr(spad_model, "spad_end_stream"):
                    spad_model.spad_end_stream()

            sample_record["videos"].append(video_record)

        sample_record["metadata"]["num_frames"] = int(global_frame_idx)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(sample_record, f, ensure_ascii=False, indent=2, default=_json_default)
            f.write("\n")
        if video_writer is not None:
            video_writer.release()
        print(f"Wrote {out_path} ({global_frame_idx} frames @ {pred_fps:g} fps)")


if __name__ == "__main__":
    main()
