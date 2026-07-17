# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""External SPAD preprocessor + standard YOLO detector evaluation.

Sibling of ``test_spad_pose_sequence.py`` / ``test_rgb_pose.py``:
- ``--preprocessor`` selects sum / ema / ppb / stea / hire.
- ``--cache_mode`` auto|raw|rendered: use cached renders from the split JSON when
  available, otherwise reconstruct from raw SPAD.
- Predictions come from the pretrained detector (``det.py`` / ``model.track``).
- Outputs: ``Evals/<YYYYMMDD>_detector_<pre>/<test_name>/<sample>.json``
  and optional ``Vis/...``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data.spad_packed import (
    raw_chunk_plane,
    raw_hwt_to_rgb_float,
    raw_plane_to_photon_cube,
)
from ultralytics.data.spad_render_cache import build_render_config, render_config_fingerprint
from ultralytics.models.yolo.pose.spad_preprocessors import build_spad_frame_preprocessor

from det import TRACKER_CHOICES, _tracker_yaml
from det_spad import _reset_tracker
from det_spad_pose import (
    _iter_raw_video_sources_from_sample_path,
    _prepare_raw_chunk_for_spad,
    _raw_sum_bgr,
    _raw_sum_readrgb_like_bgr,
    _resolve_device,
    _resize_to_shape_bgr,
    _slice_raw_chunk,
    _video_num_bins,
)
from test_spad_pose_sequence import (
    RenderChunkRecord,
    _build_preprocessor_kwargs,
    _cfg_get,
    _default_test_name,
    _default_test_name_from_json,
    _draw_bbox,
    _draw_pose,
    _eval_dir,
    _frame_record_from_result,
    _gt_aligned_chunk_records,
    _json_default,
    _load_gt_frame_count,
    _load_render_meta,
    _names_to_dict,
    _render_paths_from_direct_path,
    _render_paths_from_record,
    _resolve_cache_mode,
    _sample_paths_from_args,
    _sample_record_has_explicit_render,
    _sample_specs_from_json,
    _sample_specs_from_render_path,
    _sliding_chunk_records,
    _vis_dir,
)

SUPPORTED_PREPROCESSORS = ("sum", "ema", "ppb", "stea", "hire")


def _dated_run_name(preprocessor: str) -> str:
    return f"{datetime.now():%Y%m%d}_detector_{preprocessor}"


def _apply_input_gamma(frames_tchw: torch.Tensor, gamma: float) -> torch.Tensor:
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError(f"--input_gamma must be positive, got {gamma}")
    if abs(gamma - 1.0) < 1e-8:
        return frames_tchw
    return torch.pow(torch.clamp(frames_tchw, 0.0, 1.0), 1.0 / gamma)


def _tensor_frame_to_bgr(frame_chw: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a CHW float/[0,1] or HWC RGB frame to contiguous uint8 BGR."""
    if isinstance(frame_chw, torch.Tensor):
        arr = frame_chw.detach().float().cpu().numpy()
    else:
        arr = np.array(frame_chw, copy=True)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D frame, got shape={tuple(arr.shape)}")
    if arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 3:
        arr = arr[..., ::-1]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr)


def _render_raw_chunk_to_bgr(
    *,
    preprocessor,
    raw_chunk: np.ndarray,
    packed_nch: int,
    input_gamma: float,
    device: torch.device,
) -> np.ndarray:
    """One raw SPAD chunk -> one detector BGR frame (same path as cache_spad_renders)."""
    cube = raw_plane_to_photon_cube(raw_chunk_plane(raw_chunk, packed_nch=packed_nch), device=device, as_bool=True)
    recons, _confidence = preprocessor.process_photon_cube_to_frame(cube, clear_states=True)
    rgb = raw_hwt_to_rgb_float(recons.float(), packed_nch=int(packed_nch))
    if int(rgb.shape[0]) <= 0:
        raise ValueError("Frame preprocessor emitted zero frames for one chunk.")
    rgb = _apply_input_gamma(rgb[-1:].contiguous(), input_gamma).squeeze(0)
    return _tensor_frame_to_bgr(rgb)


def _build_vis_frame(
    frame_bgr: np.ndarray,
    result,
    *,
    vis_bg: str,
    raw_chunk: np.ndarray | None,
    packed_nch: int,
    save_readrgb: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    recon = np.ascontiguousarray(frame_bgr.copy())
    if vis_bg == "recon" or raw_chunk is None:
        vis = recon.copy()
    else:
        bg_bgr = _raw_sum_bgr(raw_chunk, packed_nch=packed_nch)
        vis = _resize_to_shape_bgr(bg_bgr, recon.shape[:2])
    readrgb = None
    if save_readrgb and raw_chunk is not None:
        readrgb_bgr = _raw_sum_readrgb_like_bgr(raw_chunk, packed_nch=packed_nch)
        readrgb = _resize_to_shape_bgr(readrgb_bgr, vis.shape[:2])
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
            xy = result.keypoints.xy.cpu().numpy()
            kconf = result.keypoints.conf.cpu().numpy()
            poses = np.concatenate([xy, kconf[..., None]], axis=2)
        for j, tid in enumerate(track_id):
            box_xyxyc = np.concatenate([boxes[j], [box_confs[j]]], axis=0)
            vis = _draw_bbox(vis, int(tid), box_xyxyc, float(handedness[j]))
            if poses is not None and j < len(poses):
                vis = _draw_pose(vis, poses[j])
    return vis, recon, readrgb


def _track_frames(model: YOLO, frames_bgr: list[np.ndarray], args) -> list:
    track_kwargs: dict[str, Any] = {
        "conf": float(args.det_thresh),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
        "persist": True,
        "tracker": _tracker_yaml(args.tracker),
        "verbose": False,
    }
    if args.device:
        track_kwargs["device"] = args.device
    return model.track(frames_bgr, **track_kwargs)


def parse_args():
    ap = argparse.ArgumentParser(
        description="External SPAD preprocessor (+ optional render cache) -> YOLO detector -> JSON predictions"
    )
    ap.add_argument("--ckpt", type=str, required=True, help="Standard pretrained YOLO pose checkpoint (e.g. weights/detector.pt)")
    ap.add_argument("--preprocessor", type=str, required=True, choices=SUPPORTED_PREPROCESSORS)
    ap.add_argument("--test_json", type=str, default=None, help="VisionSIM test split JSON")
    ap.add_argument("--in_path", type=str, default=None, help="Raw sample path/root, or rendered sample path/root when --cache_mode rendered")
    ap.add_argument("--in_glob", type=str, default=None, help="Optional glob when --in_path is a root folder")
    ap.add_argument("--test_name", type=str, default=None, help="Output test-set folder name")
    ap.add_argument("--device", type=str, default="", help="Torch device, e.g. cpu / cuda:0 / mps")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--tracker", type=str, default="botsort", choices=list(TRACKER_CHOICES))
    ap.add_argument("--frame_rate", type=int, default=25, help="Tracker frame-rate hint")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--cache_mode", type=str, default="auto", choices=["auto", "raw", "rendered"])
    ap.add_argument("--render_root", type=str, default=None, help="Optional explicit root for cached rendered frames")
    ap.add_argument("--source_render_dirname", type=str, default="renders-spc8kHz")
    ap.add_argument(
        "--render_contains_confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, require confidence.npy in rendered cache (2 kHz caches are frames+meta only).",
    )
    ap.add_argument(
        "--input_gamma",
        type=float,
        default=2.2,
        help="Gamma for detector input after SPAD reconstruction / fingerprinting. Use 1.0 to disable.",
    )
    ap.add_argument(
        "--spad_chunk_t",
        "--cube_chunk_t",
        dest="cube_chunk_t",
        type=int,
        default=320,
        help="Raw-bin chunk size per reconstructed frame (raw mode). Matches cache_spad_renders --chunk-size default.",
    )
    ap.add_argument(
        "--cube_chunk_stride",
        type=int,
        default=0,
        help="Stride in raw bins between chunks (raw mode). Default = cube_chunk_t.",
    )
    ap.add_argument("--spad_stride_frames", type=int, default=0, help="If >0 with GT alignment, stride = this * spad_bins_per_gt")
    ap.add_argument("--spad_bins_per_gt", type=int, default=64)
    ap.add_argument(
        "--spad_subsampling",
        type=int,
        default=0,
        help="Preprocessor subsampling. Default uses spad_bins_per_gt (same as cache_spad_renders).",
    )
    ap.add_argument("--ppb-bocpd-gamma", type=float, default=1e-3)
    ap.add_argument("--ppb-quantile", type=float, default=1.0)
    ap.add_argument("--ppb-normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppb-min-filter-size", type=int, default=7)
    ap.add_argument(
        "--ema-alpha",
        type=float,
        default=0.0,
        help="EMA new-sample weight. <=0 uses 2/(subsampling+1) SMA-equivalent default.",
    )
    ap.add_argument("--stea-fast-window", type=int, default=16)
    ap.add_argument("--stea-slow-window", type=int, default=128)
    ap.add_argument("--stea-temporal-window", type=int, default=5)
    ap.add_argument("--stea-fast-tau", type=float, default=6.0)
    ap.add_argument("--stea-motion-sharpness", type=float, default=60.0)
    ap.add_argument("--stea-motion-threshold", type=float, default=0.07)
    ap.add_argument("--stea-stable-prior", type=float, default=16.0)
    ap.add_argument("--stea-normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stea-quantile", type=float, default=1.0)
    ap.add_argument("--spad-bin-rate-hz", type=float, default=8000.0)
    ap.add_argument("--hire-ref-rate-hz", type=float, default=8000.0)
    ap.add_argument("--hire-fast-bins", type=int, default=16)
    ap.add_argument("--hire-slow-bins", type=int, default=128)
    ap.add_argument("--hire-surprise-bins", type=int, default=8)
    ap.add_argument("--hire-tau-fast", type=float, default=0.0)
    ap.add_argument("--hire-tau-slow", type=float, default=0.0)
    ap.add_argument("--hire-tau-surprise", type=float, default=0.0)
    ap.add_argument("--hire-gate-theta", type=float, default=0.05)
    ap.add_argument("--hire-spatial-kernel", type=int, default=3)
    ap.add_argument(
        "--vis",
        nargs="?",
        const="video",
        default="none",
        choices=["none", "image", "video"],
        help="Optional visualization. `--vis` defaults to video; use `--vis image` for PNG frames.",
    )
    ap.add_argument("--vis_bg", type=str, default="recon", choices=["sum", "recon"])
    ap.add_argument("--save_readrgb", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    preprocessor_name = str(args.preprocessor).strip().lower()
    model_name = "detector"
    run_name = _dated_run_name(preprocessor_name)

    if args.test_json:
        test_json = Path(args.test_json)
        if not test_json.is_file():
            raise FileNotFoundError(f"Test split JSON not found: {test_json}")
        sample_specs = _sample_specs_from_json(test_json)
        test_name = args.test_name or _default_test_name_from_json(test_json)
    else:
        if not args.in_path:
            raise ValueError("Either --test_json or --in_path must be provided.")
        in_path = Path(args.in_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Input path not found: {in_path}")
        requested_cache_mode = str(args.cache_mode).strip().lower()
        if requested_cache_mode == "rendered":
            sample_specs = _sample_specs_from_render_path(in_path, args.in_glob)
        else:
            sample_specs = [
                {"sample_name": p.name if p.is_dir() else p.stem, "sample_path": p, "record": None}
                for p in _sample_paths_from_args(in_path, args.in_glob)
            ]
        test_name = args.test_name or _default_test_name(in_path, args.in_glob)

    resolved_cache_mode = _resolve_cache_mode(args.cache_mode, sample_specs, preprocessor_name)
    print(f"Resolved cache_mode={resolved_cache_mode}, preprocessor={preprocessor_name}, run_name={run_name}")

    chunk_size = int(args.cube_chunk_t)
    if chunk_size <= 0:
        raise ValueError(f"cube_chunk_t must be positive, got {chunk_size}")
    spad_bins_per_gt = int(args.spad_bins_per_gt)
    if spad_bins_per_gt <= 0:
        raise ValueError(f"spad_bins_per_gt must be positive, got {spad_bins_per_gt}")
    if int(args.spad_stride_frames) > 0:
        stride_bins = int(args.spad_stride_frames) * spad_bins_per_gt
        stride_frames = int(args.spad_stride_frames)
    else:
        stride_bins = int(args.cube_chunk_stride) if int(args.cube_chunk_stride) > 0 else chunk_size
        stride_frames = max(stride_bins // spad_bins_per_gt, 1)
    spad_subsampling = int(args.spad_subsampling) if int(args.spad_subsampling) > 0 else spad_bins_per_gt
    input_gamma = float(args.input_gamma)

    out_dir = _eval_dir(run_name, test_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_root = _vis_dir(run_name, test_name) if args.vis != "none" else None
    if vis_root is not None:
        vis_root.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    model = YOLO(str(ckpt))
    names = model.names
    kpt_shape = getattr(model.model, "kpt_shape", (21, 3))

    preprocessor = None
    preprocessor_kwargs = _build_preprocessor_kwargs(
        args, preprocessor_name=preprocessor_name, spad_subsampling=spad_subsampling
    )
    if resolved_cache_mode == "raw":
        preprocessor = build_spad_frame_preprocessor(preprocessor_name, kwargs=preprocessor_kwargs).to(device)

    expected_render_fingerprint = None
    if resolved_cache_mode == "rendered" and not all(
        _sample_record_has_explicit_render(spec.get("record"), preprocessor_name) for spec in sample_specs
    ):
        expected_render_config = build_render_config(
            preprocessor=preprocessor_name,
            chunk_size=chunk_size,
            stride_bins=stride_bins,
            spad_bins_per_gt=spad_bins_per_gt,
            packed_ch_order=args.packed_ch_order,
            input_gamma=input_gamma,
            extra_kwargs=preprocessor_kwargs,
        )
        expected_render_fingerprint = render_config_fingerprint(expected_render_config)

    for sample_spec in sample_specs:
        sample_name = str(sample_spec["sample_name"])
        sample_path = Path(sample_spec["sample_path"])
        out_path = out_dir / f"{sample_name}.json"
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing file: {out_path}")
            continue

        _reset_tracker(model)

        global_frame_idx = 0
        sample_vis_dir = None
        video_writer = None
        if vis_root is not None and args.vis == "image":
            sample_vis_dir = vis_root / sample_name
            sample_vis_dir.mkdir(parents=True, exist_ok=True)
        video_path = vis_root / f"{sample_name}.mp4" if vis_root is not None and args.vis == "video" else None

        sample_record: dict[str, Any] = {
            "metadata": {
                "format": "spadhand_external_pre_predictions_v1",
                "ckpt": str(ckpt),
                "model_name": model_name,
                "run_name": run_name,
                "test_name": test_name,
                "sample_name": sample_name,
                "preprocessor": preprocessor_name,
                "det_thresh": float(args.det_thresh),
                "iou": float(args.iou),
                "max_det": int(args.max_det),
                "tracker": args.tracker,
                "frame_rate": int(args.frame_rate),
                "packed_ch_order": args.packed_ch_order,
                "input_gamma": float(input_gamma),
                "vis_mode": str(args.vis),
                "spad_cache_mode": resolved_cache_mode,
                "requested_cache_mode": str(args.cache_mode),
                "spad_chunk_size": int(chunk_size),
                "spad_bins_per_gt": int(spad_bins_per_gt),
                "spad_stride_frames": int(stride_frames),
                "cube_chunk_stride": int(stride_bins),
                "spad_subsampling": int(spad_subsampling),
                "names": _names_to_dict(names),
                "kpt_shape": list(map(int, kpt_shape)),
                "source_bin_convention": "chunk_end_bin",
            },
            "source": {
                "path": str(sample_path),
                "is_dir": bool(sample_path.is_dir()),
            },
            "videos": [],
        }
        if sample_spec.get("record") is not None:
            sample_record["source"]["split_record"] = sample_spec["record"]

        if resolved_cache_mode == "raw":
            assert preprocessor is not None
            video_iter = _iter_raw_video_sources_from_sample_path(sample_path)
            for video_idx, source in enumerate(tqdm(video_iter, desc=f"Testing [{sample_name}]")):
                total_bins = _video_num_bins(source)
                gt_aligned = sample_spec.get("record") is not None
                if gt_aligned:
                    n_gt = _load_gt_frame_count(sample_spec["record"])
                    chunk_records = _gt_aligned_chunk_records(
                        total_bins=total_bins,
                        chunk_size=chunk_size,
                        spad_bins_per_gt=spad_bins_per_gt,
                        stride_frames=stride_frames,
                        n_gt=n_gt,
                        packed_nch=source.packed_nch,
                    )
                else:
                    chunk_records = _sliding_chunk_records(
                        total_bins=total_bins,
                        chunk_size=chunk_size,
                        stride=stride_bins,
                        packed_nch=source.packed_nch,
                    )

                video_record: dict[str, Any] = {
                    "video_idx": int(video_idx),
                    "layout": source.layout,
                    "packed_nch": int(source.packed_nch),
                    "total_bins": int(total_bins),
                    "chunk_t": int(chunk_size),
                    "chunk_stride": int(stride_bins),
                    "subsampling": int(spad_subsampling),
                    "chunks": [],
                }

                frames_bgr: list[np.ndarray] = []
                chunk_meta: list[RenderChunkRecord] = []
                raw_chunks: list[np.ndarray] = []
                for chunk in chunk_records:
                    t0 = int(chunk.spad_start_bin)
                    t1 = int(chunk.spad_end_bin)
                    raw_chunk = _slice_raw_chunk(source, t0, t1, packed_ch_order=args.packed_ch_order)
                    raw_chunk = _prepare_raw_chunk_for_spad(raw_chunk, chunk_t=chunk_size, tail_pad_full=False)
                    if raw_chunk is None:
                        continue
                    with torch.inference_mode():
                        frame_bgr = _render_raw_chunk_to_bgr(
                            preprocessor=preprocessor,
                            raw_chunk=raw_chunk,
                            packed_nch=int(source.packed_nch),
                            input_gamma=input_gamma,
                            device=device,
                        )
                    frames_bgr.append(frame_bgr)
                    chunk_meta.append(chunk)
                    raw_chunks.append(raw_chunk)

                if not frames_bgr:
                    sample_record["videos"].append(video_record)
                    continue

                results = _track_frames(model, frames_bgr, args)
                if len(results) != len(frames_bgr):
                    raise RuntimeError(
                        f"Detector returned {len(results)} results for {len(frames_bgr)} frames in {sample_name}"
                    )

                for chunk, result, frame_bgr, raw_chunk in zip(chunk_meta, results, frames_bgr, raw_chunks):
                    t0 = int(chunk.spad_start_bin)
                    t1 = int(chunk.spad_end_bin)
                    chunk_record: dict[str, Any] = {
                        "chunk_idx": int(chunk.chunk_index),
                        "start_bin": int(t0),
                        "end_bin": int(t1),
                        "input_bins": int(t1 - t0),
                        "model_input_bins": int(raw_chunk.shape[0]),
                        "target_gt_time": chunk.target_gt_time,
                        "frames": [],
                    }
                    if args.vis != "none":
                        vis, recon, readrgb = _build_vis_frame(
                            frame_bgr,
                            result,
                            vis_bg=args.vis_bg,
                            raw_chunk=raw_chunk,
                            packed_nch=int(source.packed_nch),
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
                                    float(args.frame_rate),
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
                            output_frame_idx=0,
                            source_bin=t1,
                        )
                    )
                    global_frame_idx += 1
                    video_record["chunks"].append(chunk_record)

                sample_record["videos"].append(video_record)
        else:
            if sample_spec.get("record") is not None:
                render_dir, frames_path, confidence_path, meta_path = _render_paths_from_record(
                    sample_spec["record"],
                    preprocessor=preprocessor_name,
                    render_root=args.render_root,
                    source_render_dirname=args.source_render_dirname,
                )
            else:
                render_dir = Path(sample_spec["render_dir"])
                frames_path = Path(sample_spec["frames_path"])
                confidence_path = (
                    None if sample_spec.get("confidence_path") is None else Path(sample_spec["confidence_path"])
                )
                meta_path = Path(sample_spec["meta_path"])

            frames, _confidence, chunk_records, meta = _load_render_meta(
                sample_name=sample_name,
                sample_path=sample_path,
                record=sample_spec.get("record"),
                render_dir=render_dir,
                frames_path=frames_path,
                confidence_path=confidence_path,
                meta_path=meta_path,
                require_confidence=bool(args.render_contains_confidence),
                expected_preprocessor=preprocessor_name,
                expected_fingerprint=expected_render_fingerprint,
            )
            sample_record["source"]["render_dir"] = str(render_dir)
            sample_record["source"]["render_meta"] = str(meta_path)
            sample_record["metadata"]["render_config_fingerprint"] = meta.get("config_fingerprint")
            meta_config = meta.get("config") if isinstance(meta.get("config"), dict) else {}

            total_bins = int(meta.get("total_raw_bins", 0) or 0)
            if total_bins <= 0 and chunk_records:
                total_bins = int(max(chunk.spad_end_bin for chunk in chunk_records))
            video_record = {
                "video_idx": 0,
                "layout": "rendered",
                "packed_nch": int(chunk_records[0].packed_nch if chunk_records else 3),
                "total_bins": total_bins,
                "chunk_t": int(_cfg_get(meta_config, "chunk_size", chunk_size) or chunk_size),
                "chunk_stride": int(
                    _cfg_get(meta_config, "stride_bins", _cfg_get(meta, "stride_bins", stride_bins)) or stride_bins
                ),
                "subsampling": int(
                    _cfg_get(meta_config, "extra_kwargs", {}).get("subsampling", meta.get("subsampling", spad_subsampling))
                    or spad_subsampling
                ),
                "chunks": [],
            }

            frames_bgr = [_tensor_frame_to_bgr(frames[chunk.chunk_index]) for chunk in chunk_records]
            results = _track_frames(model, frames_bgr, args)
            if len(results) != len(frames_bgr):
                raise RuntimeError(
                    f"Detector returned {len(results)} results for {len(frames_bgr)} cached frames in {sample_name}"
                )

            for chunk, result, frame_bgr in tqdm(
                zip(chunk_records, results, frames_bgr),
                total=len(chunk_records),
                desc=f"Testing [{sample_name}]",
            ):
                chunk_record = {
                    "chunk_idx": int(chunk.chunk_index),
                    "start_bin": int(chunk.spad_start_bin),
                    "end_bin": int(chunk.spad_end_bin),
                    "input_bins": int(chunk.chunk_size),
                    "model_input_bins": int(chunk.chunk_size),
                    "target_gt_time": chunk.target_gt_time,
                    "frames": [],
                }
                if args.vis != "none":
                    vis, recon, _ = _build_vis_frame(
                        frame_bgr,
                        result,
                        vis_bg="recon",
                        raw_chunk=None,
                        packed_nch=int(chunk.packed_nch),
                        save_readrgb=False,
                    )
                    if args.vis == "image" and sample_vis_dir is not None:
                        stem = f"cube00000_t{chunk.spad_start_bin:06d}_{chunk.spad_end_bin:06d}_frame{global_frame_idx:07d}"
                        cv2.imwrite(str(sample_vis_dir / f"{stem}.png"), vis)
                        cv2.imwrite(str(sample_vis_dir / f"{stem}_recon.png"), recon)
                    elif args.vis == "video" and video_path is not None:
                        if video_writer is None:
                            h, w = vis.shape[:2]
                            video_writer = cv2.VideoWriter(
                                str(video_path),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                float(args.frame_rate),
                                (w, h),
                            )
                        video_writer.write(vis)

                chunk_record["frames"].append(
                    _frame_record_from_result(
                        result,
                        names=names,
                        global_frame_idx=global_frame_idx,
                        video_idx=0,
                        chunk_idx=int(chunk.chunk_index),
                        chunk_start=int(chunk.spad_start_bin),
                        chunk_end=int(chunk.spad_end_bin),
                        output_frame_idx=0,
                        source_bin=int(chunk.spad_end_bin),
                    )
                )
                global_frame_idx += 1
                video_record["chunks"].append(chunk_record)

            sample_record["videos"].append(video_record)

        sample_record["metadata"]["num_frames"] = int(global_frame_idx)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(sample_record, f, ensure_ascii=False, indent=2, default=_json_default)
            f.write("\n")
        if video_writer is not None:
            video_writer.release()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
