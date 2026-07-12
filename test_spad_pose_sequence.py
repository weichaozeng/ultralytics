# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run sequence-mode SPAD pose inference on a test set and save structured predictions.

Supports raw SPAD clips and cached rendered frames (PPB/STEA), mirroring the cache
workflow in ``test_spad_pose_frame.py``. This is the non-visualization sibling of
``det_spad_pose.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data.spad_pose_dataset import load_visionsim_split_json
from ultralytics.data.spad_render_cache import (
    CACHE_META_VERSION,
    build_render_config,
    render_config_fingerprint,
    sample_render_dir,
    sibling_sample_render_dir,
)

from det_spad_pose import (
    _apply_tracker,
    _build_override_preprocessor,
    _configure_model_spad_bin_rate,
    _get_finger_color,
    _init_tracker,
    _iter_raw_video_sources_from_sample_path,
    _postprocess_pose_predictions,
    _prepare_raw_chunk_for_spad,
    _raw_sum_bgr,
    _raw_sum_readrgb_like_bgr,
    _recon_frames_bgr,
    _resolve_device,
    _results_from_preds,
    _resize_to_shape_bgr,
    _set_velocity_field_on_preprocessor,
    _slice_raw_chunk,
    _trained_chunk_t,
    _video_num_bins,
)

BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
COLOR_KEYPOINT = (255, 255, 255)
COLOR_WRIST = (255, 165, 0)

TRACKER_CHOICES = ["none", "bytetrack", "botsort", "spad_tracker", "posetrack", "spad_posetrack"]
PREPROCESSOR_CHOICES = ["model", "ppb", "sum", "stea", "hyb"]


@dataclass(frozen=True)
class RenderChunkRecord:
    chunk_index: int
    spad_start_bin: int
    spad_end_bin: int
    chunk_size: int
    target_gt_time: float | None
    packed_nch: int


def _json_default(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sample_paths_from_args(in_path: Path, in_glob: str | None) -> list[Path]:
    if in_path.is_dir() and in_glob:
        sample_paths = [p for p in sorted(in_path.glob(in_glob)) if p.is_dir()]
        if not sample_paths:
            raise FileNotFoundError(f"No sample/video subfolders matched: {in_path}/{in_glob}")
        return sample_paths
    return [in_path]


def _sample_specs_from_json(test_json: Path) -> list[dict[str, Any]]:
    specs = []
    for record in load_visionsim_split_json(test_json):
        spad_path = Path(record["spad"])
        sample_name = str(record.get("name") or spad_path.stem)
        specs.append({"sample_name": sample_name, "sample_path": spad_path, "record": record})
    return specs


def _default_test_name(in_path: Path, in_glob: str | None) -> str:
    if in_path.is_dir() and in_glob:
        return in_path.name
    return in_path.name if in_path.is_dir() else in_path.stem


def _default_test_name_from_json(test_json: Path) -> str:
    return test_json.stem


def _model_name_from_ckpt(ckpt: Path) -> str:
    if ckpt.parent.name == "weights":
        return ckpt.parent.parent.name
    return ckpt.parent.name


def _dated_model_name(model_name: str) -> str:
    return f"{datetime.now():%Y%m%d}_{model_name}"


def _eval_dir(run_name: str, test_name: str) -> Path:
    return Path("Evals") / run_name / test_name


def _vis_dir(run_name: str, test_name: str) -> Path:
    return Path("Vis") / run_name / test_name


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _names_to_dict(names) -> dict[str, str]:
    if isinstance(names, dict):
        return {str(k): str(v) for k, v in names.items()}
    return {str(i): str(v) for i, v in enumerate(names)}


def _sample_record_has_explicit_render(record: dict[str, Any] | None, preprocessor: str) -> bool:
    if not record:
        return False
    key = str(preprocessor).strip().lower()
    return bool(
        record.get(key)
        or record.get(f"render_{key}")
        or record.get(f"render_{key}_frames")
    )


def _resolve_cache_mode(requested: str, sample_specs: list[dict[str, Any]], preprocessor: str) -> str:
    requested = str(requested).strip().lower()
    if requested in {"raw", "rendered"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported --cache_mode={requested!r}; expected raw, rendered, or auto.")
    if preprocessor == "hyb":
        return "raw"
    if not sample_specs:
        return "raw"
    has_all = all(_sample_record_has_explicit_render(spec.get("record"), preprocessor) for spec in sample_specs)
    return "rendered" if has_all else "raw"


def _choose_effective_preprocessor(spad_model, args) -> str:
    requested = str(args.preprocessor).strip().lower()
    if requested != "model":
        return requested
    legacy = str(getattr(args, "preprocessor_override", "none")).strip().lower()
    if legacy not in {"", "none"}:
        return legacy
    return str(getattr(spad_model, "preprocessor_name", "stea")).strip().lower()


def _resolve_chunk_size(spad_model, args) -> int:
    if int(args.cube_chunk_t) > 0:
        return int(args.cube_chunk_t)
    trained = _trained_chunk_t(spad_model)
    if trained:
        return int(trained)
    train_args = getattr(spad_model, "args", None)
    subsampling = _cfg_get(train_args, "spad_subsampling", getattr(getattr(spad_model, "preprocessor", None), "subsampling", 64))
    subsampling = _cfg_get(train_args, "spad_chunk_size", subsampling)
    return int(subsampling)


def _resolve_spad_stride_frames(spad_model, args, *, ckpt: dict | None = None) -> int:
    if int(args.spad_stride_frames) > 0:
        return int(args.spad_stride_frames)
    for src in ((ckpt or {}).get("train_args"), getattr(spad_model, "args", None)):
        val = _cfg_get(src, "spad_stride_frames", None)
        if val not in {None, 0, ""}:
            return int(val)
    return 5


def _resolve_spad_bins_per_gt(spad_model, args) -> int:
    if int(args.spad_bins_per_gt) > 0:
        return int(args.spad_bins_per_gt)
    train_args = getattr(spad_model, "args", None)
    return int(_cfg_get(train_args, "spad_bins_per_gt", 64) or 64)


def _resolve_spad_subsampling(spad_model, args) -> int:
    if int(args.spad_subsampling) > 0:
        return int(args.spad_subsampling)
    train_args = getattr(spad_model, "args", None)
    return int(_cfg_get(train_args, "spad_subsampling", getattr(getattr(spad_model, "preprocessor", None), "subsampling", 64)) or 64)


def _resolve_input_gamma(spad_model, args) -> float:
    if float(args.input_gamma) > 0:
        return float(args.input_gamma)
    train_args = getattr(spad_model, "args", None)
    return float(_cfg_get(train_args, "spad_input_gamma", getattr(spad_model, "spad_input_gamma", 1.0)))


def _build_preprocessor_kwargs(args, *, preprocessor_name: str, spad_subsampling: int) -> dict[str, Any]:
    name = str(preprocessor_name).strip().lower()
    if name == "ppb":
        return {
            "subsampling": spad_subsampling,
            "bocpd_gamma": float(args.ppb_bocpd_gamma),
            "normalize": bool(args.ppb_normalize),
            "quantile": float(args.ppb_quantile),
            "min_filter_size": int(args.ppb_min_filter_size),
        }
    if name == "stea":
        return {
            "subsampling": spad_subsampling,
            "fast_window": int(args.stea_fast_window),
            "slow_window": int(args.stea_slow_window),
            "temporal_window": int(args.stea_temporal_window),
            "fast_tau": None if args.stea_fast_tau in {None, 0} else float(args.stea_fast_tau),
            "motion_sharpness": float(args.stea_motion_sharpness),
            "motion_threshold": float(args.stea_motion_threshold),
            "stable_prior": float(args.stea_stable_prior),
            "normalize": bool(args.stea_normalize),
            "quantile": float(args.stea_quantile),
        }
    if name == "sum":
        return {"subsampling": spad_subsampling}
    if name == "hyb":
        return {
            "subsampling": spad_subsampling,
            "fast_window": int(args.stea_fast_window),
            "slow_window": int(args.stea_slow_window),
            "temporal_window": int(args.stea_temporal_window),
            "fast_tau": None if args.stea_fast_tau in {None, 0} else float(args.stea_fast_tau),
            "motion_sharpness": float(args.stea_motion_sharpness),
            "motion_threshold": float(args.stea_motion_threshold),
            "stable_prior": float(args.stea_stable_prior),
            "normalize": bool(args.stea_normalize),
            "quantile": float(args.stea_quantile),
            "warp_block_size": int(args.hyb_warp_block_size),
            "source_space": str(args.hyb_source_space),
        }
    raise ValueError(f"Unsupported preprocessor: {name!r}")


def _render_paths_from_record(
    record: dict[str, Any],
    *,
    preprocessor: str,
    render_root: str | Path | None,
    source_render_dirname: str,
) -> tuple[Path, Path, Path | None, Path]:
    sample_name = str(record["name"])
    explicit_frames = record.get(preprocessor) or record.get(f"render_{preprocessor}_frames")
    explicit_render_dir = record.get(f"render_{preprocessor}")
    explicit_conf = record.get(f"{preprocessor}_confidence") or record.get(f"render_{preprocessor}_confidence")
    explicit_meta = record.get(f"{preprocessor}_meta") or record.get(f"render_{preprocessor}_meta")

    if explicit_frames or explicit_render_dir:
        if explicit_frames:
            frames_path = Path(explicit_frames)
            render_dir = frames_path.parent
        else:
            render_dir = Path(explicit_render_dir)
            frames_path = render_dir / "frames.npy"
        meta_path = Path(explicit_meta) if explicit_meta else render_dir / "meta.json"
        confidence_path = Path(explicit_conf) if explicit_conf else render_dir / "confidence.npy"
        return render_dir, frames_path, confidence_path, meta_path

    if render_root not in {None, ""}:
        render_dir = sample_render_dir(render_root, sample_name)
    else:
        render_dir = sibling_sample_render_dir(
            record["spad"],
            preprocessor=preprocessor,
            sample_name=sample_name,
            source_render_dirname=source_render_dirname,
        )
    return render_dir, render_dir / "frames.npy", render_dir / "confidence.npy", render_dir / "meta.json"


def _render_paths_from_direct_path(path: Path) -> tuple[Path, Path, Path | None, Path]:
    if path.is_dir():
        render_dir = path
        frames_path = render_dir / "frames.npy"
        meta_path = render_dir / "meta.json"
        confidence_path = render_dir / "confidence.npy"
    elif path.is_file() and path.name == "frames.npy":
        frames_path = path
        render_dir = path.parent
        meta_path = render_dir / "meta.json"
        confidence_path = render_dir / "confidence.npy"
    else:
        raise ValueError(f"Unsupported rendered input path: {path}")
    return render_dir, frames_path, confidence_path, meta_path


def _sample_specs_from_render_path(in_path: Path, in_glob: str | None) -> list[dict[str, Any]]:
    paths = _sample_paths_from_args(in_path, in_glob)
    specs = []
    for path in paths:
        render_dir, frames_path, confidence_path, meta_path = _render_paths_from_direct_path(path)
        if not frames_path.is_file():
            raise FileNotFoundError(f"Rendered frames not found: {frames_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Rendered metadata not found: {meta_path}")
        specs.append(
            {
                "sample_name": render_dir.name,
                "sample_path": path,
                "record": None,
                "render_dir": render_dir,
                "frames_path": frames_path,
                "confidence_path": confidence_path,
                "meta_path": meta_path,
            }
        )
    return specs


def _load_gt_frame_count(record: dict[str, Any]) -> int:
    with Path(record["gt"]).open("r", encoding="utf-8") as f:
        ann = json.load(f)
    return len(ann)


def _gt_aligned_chunk_records(
    *,
    total_bins: int,
    chunk_size: int,
    spad_bins_per_gt: int,
    stride_frames: int,
    n_gt: int,
    packed_nch: int,
) -> list[RenderChunkRecord]:
    gt_offset = chunk_size / float(spad_bins_per_gt)
    max_start = int(np.floor((n_gt - 1) - gt_offset))
    if max_start < 0:
        return []

    records = []
    for chunk_index, gt_start in enumerate(range(0, max_start + 1, stride_frames)):
        spad_start = int(gt_start * spad_bins_per_gt)
        spad_end = spad_start + chunk_size
        if spad_end > total_bins:
            break
        records.append(
            RenderChunkRecord(
                chunk_index=int(chunk_index),
                spad_start_bin=int(spad_start),
                spad_end_bin=int(spad_end),
                chunk_size=int(chunk_size),
                target_gt_time=float(gt_start + gt_offset),
                packed_nch=int(packed_nch),
            )
        )
    return records


def _sliding_chunk_records(*, total_bins: int, chunk_size: int, stride: int, packed_nch: int) -> list[RenderChunkRecord]:
    records = []
    for chunk_index, t0 in enumerate(range(0, total_bins, stride)):
        t1 = min(total_bins, t0 + chunk_size)
        if (t1 - t0) < chunk_size:
            continue
        records.append(
            RenderChunkRecord(
                chunk_index=int(chunk_index),
                spad_start_bin=int(t0),
                spad_end_bin=int(t1),
                chunk_size=int(chunk_size),
                target_gt_time=None,
                packed_nch=int(packed_nch),
            )
        )
    return records


def _load_render_meta(
    *,
    sample_name: str,
    sample_path: Path,
    record: dict[str, Any] | None,
    render_dir: Path,
    frames_path: Path,
    confidence_path: Path | None,
    meta_path: Path,
    require_confidence: bool,
    expected_preprocessor: str,
    expected_fingerprint: str | None,
) -> tuple[np.ndarray, np.ndarray | None, list[RenderChunkRecord], dict[str, Any]]:
    if not frames_path.is_file():
        raise FileNotFoundError(f"Cached frames not found for {sample_name!r}: {frames_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Cached metadata not found for {sample_name!r}: {meta_path}")
    if require_confidence and (confidence_path is None or not confidence_path.is_file()):
        raise FileNotFoundError(f"Cached confidence required but not found for {sample_name!r}: {confidence_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    version = int(meta.get("version", -1))
    if version != CACHE_META_VERSION:
        raise ValueError(f"Unsupported cache meta version for {meta_path}: {version}")
    cached_preprocessor = str(meta.get("preprocessor", "")).strip().lower()
    if cached_preprocessor and cached_preprocessor != expected_preprocessor:
        raise ValueError(
            f"Cached render preprocessor mismatch for {sample_name!r}: expected {expected_preprocessor!r}, "
            f"got {cached_preprocessor!r}"
        )
    if expected_fingerprint is not None:
        cached_fp = str(meta.get("config_fingerprint", "")).strip()
        if cached_fp != expected_fingerprint:
            raise ValueError(
                f"Cached render fingerprint mismatch for {sample_name!r}: expected {expected_fingerprint}, "
                f"got {cached_fp or '<missing>'}"
            )
    if record is not None:
        source_gt = meta.get("source_gt")
        if source_gt and Path(source_gt).resolve() != Path(record["gt"]).resolve():
            raise ValueError(f"Cached render GT mismatch for {sample_name!r}: {source_gt} != {record['gt']}")
        source_spad = meta.get("source_spad")
        if source_spad and Path(source_spad).resolve() != Path(record["spad"]).resolve():
            raise ValueError(f"Cached render SPAD mismatch for {sample_name!r}: {source_spad} != {record['spad']}")
    chunks = meta.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Cached render metadata must include a non-empty chunks list: {meta_path}")

    frames = np.load(frames_path, mmap_mode="r")
    confidence = np.load(confidence_path, mmap_mode="r") if confidence_path is not None and confidence_path.is_file() else None
    if len(frames) < len(chunks):
        raise ValueError(f"Cached frames length {len(frames)} is smaller than chunk count {len(chunks)} for {sample_name!r}")
    if confidence is not None and len(confidence) < len(chunks):
        raise ValueError(
            f"Cached confidence length {len(confidence)} is smaller than chunk count {len(chunks)} for {sample_name!r}"
        )

    chunk_records = [
        RenderChunkRecord(
            chunk_index=int(chunk["chunk_index"]),
            spad_start_bin=int(chunk["spad_start_bin"]),
            spad_end_bin=int(chunk["spad_end_bin"]),
            chunk_size=int(chunk["chunk_size"]),
            target_gt_time=None if chunk.get("target_gt_time") is None else float(chunk["target_gt_time"]),
            packed_nch=int(chunk.get("packed_nch", 3)),
        )
        for chunk in chunks
    ]
    return frames, confidence, chunk_records, meta


def _frame_to_tensor(frame: np.ndarray, *, device: torch.device) -> torch.Tensor:
    arr = np.array(frame, copy=True)
    tensor = torch.from_numpy(arr)
    if tensor.ndim != 3:
        raise ValueError(f"Expected cached frame with 3 dimensions, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] not in {1, 3} and tensor.shape[-1] in {1, 3}:
        tensor = tensor.permute(2, 0, 1).contiguous()
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
    return tensor.unsqueeze(0).to(device)


def _frame_to_sequence_tensor(frame_tensor: torch.Tensor) -> torch.Tensor:
    if frame_tensor.ndim != 4:
        raise ValueError(f"Expected frame tensor B,C,H,W, got shape={tuple(frame_tensor.shape)}")
    return frame_tensor.unsqueeze(1)


def _confidence_to_tensor(confidence: np.ndarray | None, *, device: torch.device, h: int, w: int) -> torch.Tensor:
    if confidence is None:
        return torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
    conf = torch.from_numpy(np.array(confidence, copy=True)).float()
    if conf.ndim == 2:
        conf = conf.unsqueeze(0)
    if conf.ndim == 3:
        conf = conf.unsqueeze(0)
    if conf.ndim != 4:
        raise ValueError(f"Expected confidence with 2/3/4 dimensions, got shape={tuple(conf.shape)}")
    return conf.to(device)


def _detections_from_result(result, names) -> list[dict[str, Any]]:
    if result.boxes is None or not len(result.boxes):
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    track_ids = result.boxes.id
    if track_ids is None:
        track_ids_np = np.arange(len(result.boxes), dtype=np.int64)
    else:
        track_ids_np = track_ids.cpu().numpy()

    keypoints = None
    if getattr(result, "keypoints", None) is not None:
        keypoints = result.keypoints.data.cpu().numpy()

    detections = []
    for det_idx in range(len(boxes)):
        cls_id = int(classes[det_idx])
        class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        det = {
            "det_idx": det_idx,
            "track_id": int(track_ids_np[det_idx]),
            "bbox_xyxy": boxes[det_idx].astype(float).tolist(),
            "score": float(scores[det_idx]),
            "cls": cls_id,
            "class_name": class_name,
        }
        if keypoints is not None and det_idx < len(keypoints):
            det["keypoints"] = keypoints[det_idx].astype(float).tolist()
        detections.append(det)
    return detections


def _frame_record_from_result(
    result,
    *,
    names,
    global_frame_idx: int,
    video_idx: int,
    chunk_idx: int,
    chunk_start: int,
    chunk_end: int,
    output_frame_idx: int,
    source_bin: int,
) -> dict[str, Any]:
    image_shape = None
    if getattr(result, "orig_img", None) is not None:
        h, w = result.orig_img.shape[:2]
        image_shape = [int(h), int(w)]

    return {
        "frame_idx": int(global_frame_idx),
        "video_idx": int(video_idx),
        "chunk_idx": int(chunk_idx),
        "chunk_start_bin": int(chunk_start),
        "chunk_end_bin": int(chunk_end),
        "output_frame_idx": int(output_frame_idx),
        "source_bin": int(source_bin),
        "source_time_index": float(source_bin),
        "image_shape": image_shape,
        "detections": _detections_from_result(result, names),
    }


def _draw_pose(img_bgr: np.ndarray, pose_kpts: np.ndarray, thresh: float = 0.5, k: int = 21) -> np.ndarray:
    if pose_kpts.shape != (k, 3):
        raise ValueError(f"Pose shape must be ({k}, 3), but got {pose_kpts.shape}")
    for i, (s, e) in enumerate(BONE_CONNECTIONS):
        ks = pose_kpts[s]
        ke = pose_kpts[e]
        if ks[2] > thresh and ke[2] > thresh:
            cv2.line(img_bgr, (int(ks[0]), int(ks[1])), (int(ke[0]), int(ke[1])), _get_finger_color(i), 3)
    for i in range(k):
        kk = pose_kpts[i]
        if kk[2] > thresh:
            center = (int(kk[0]), int(kk[1]))
            color, radius = (COLOR_WRIST, 6) if i == 0 else (COLOR_KEYPOINT, 4)
            cv2.circle(img_bgr, center, radius, color, -1)
    return img_bgr


def _draw_bbox(img_bgr: np.ndarray, track_id: int, box_xyxyc: np.ndarray, handedness: float) -> np.ndarray:
    x1, y1, x2, y2, conf = box_xyxyc
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    color = (0, 0, 255) if handedness > 0 else (255, 0, 0)
    text = f"ID: {int(track_id)}"
    font_scale = 0.8
    thickness = 2
    text_color = (255, 255, 255)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pt1 = (x1, y1)
    pt2 = (x1 + tw, y1 + th + baseline)
    text_org = (x1, y1 + baseline + baseline)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(img_bgr, pt1, pt2, color, -1)
    cv2.putText(img_bgr, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    return img_bgr


def _build_vis_frame(result, *, vis_bg: str, raw_chunk: np.ndarray, packed_nch: int, save_readrgb: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    recon = (
        np.ascontiguousarray(result.orig_img.copy())
        if getattr(result, "orig_img", None) is not None
        else np.zeros((512, 512, 3), dtype=np.uint8)
    )
    if vis_bg == "recon" and getattr(result, "orig_img", None) is not None:
        vis = recon.copy()
    else:
        bg_bgr = _raw_sum_bgr(raw_chunk, packed_nch=packed_nch)
        vis = _resize_to_shape_bgr(bg_bgr, recon.shape[:2])
    readrgb = None
    if save_readrgb:
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
        poses = result.keypoints.data.cpu().numpy() if getattr(result, "keypoints", None) is not None else None
        for j, tid in enumerate(track_id):
            box_xyxyc = np.concatenate([boxes[j], [box_confs[j]]], axis=0)
            vis = _draw_bbox(vis, int(tid), box_xyxyc, float(handedness[j]))
            if poses is not None and j < len(poses):
                vis = _draw_pose(vis, poses[j])
    return vis, recon, readrgb


def _build_cached_vis_frame(result) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    recon = (
        np.ascontiguousarray(result.orig_img.copy())
        if getattr(result, "orig_img", None) is not None
        else np.zeros((512, 512, 3), dtype=np.uint8)
    )
    vis = recon.copy()
    if result.boxes is not None and len(result.boxes):
        track_ids = result.boxes.id
        if track_ids is None:
            track_ids = torch.arange(len(result.boxes), device=result.boxes.data.device)
        track_id = track_ids.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        box_confs = result.boxes.conf.cpu().numpy()
        handedness = result.boxes.cls.cpu().numpy()
        poses = result.keypoints.data.cpu().numpy() if getattr(result, "keypoints", None) is not None else None
        for j, tid in enumerate(track_id):
            box_xyxyc = np.concatenate([boxes[j], [box_confs[j]]], axis=0)
            vis = _draw_bbox(vis, int(tid), box_xyxyc, float(handedness[j]))
            if poses is not None and j < len(poses):
                vis = _draw_pose(vis, poses[j])
    return vis, recon, None


def _maybe_set_velocity_field(preprocessor, tracker) -> None:
    if preprocessor is None or not hasattr(preprocessor, "set_velocity_field"):
        return
    field = None if tracker is None else getattr(tracker, "last_velocity_field", None)
    preprocessor.set_velocity_field(field, source_space="rgb")


def _iter_chunk_outputs(
    results,
    *,
    gt_aligned: bool,
    t0: int,
    t1: int,
    subsampling: int,
) -> list[tuple[Any, int, int]]:
    if not results:
        return []
    if gt_aligned:
        output_frame_idx = len(results) - 1
        return [(results[-1], output_frame_idx, int(t1))]
    outputs = []
    for output_frame_idx, result in enumerate(results):
        source_bin = int(t0 + output_frame_idx * subsampling)
        outputs.append((result, output_frame_idx, source_bin))
    return outputs


def _run_sequence_chunk_forward(spad_model, seq_tensor: torch.Tensor, *, t_index_ll: list[int]):
    """Run one streaming chunk through SpadPoseModel (B=1, T=1)."""
    if seq_tensor.ndim != 5 or int(seq_tensor.shape[0]) != 1 or int(seq_tensor.shape[1]) != 1:
        raise ValueError(
            f"Sequence streaming inference expects shape (1, 1, C, H, W), got {tuple(seq_tensor.shape)}"
        )
    spad_model.spad_pending_t_index_ll = [int(v) for v in t_index_ll]
    return spad_model(seq_tensor)


def _apply_sequence_preprocessor_override(spad_model, args, device, effective_preprocessor: str) -> None:
    if effective_preprocessor == "model":
        return
    args.preprocessor_override = effective_preprocessor
    override_name, override_preprocessor = _build_override_preprocessor(args)
    if override_preprocessor is None:
        return
    if override_name == "hyb" and args.tracker not in {"spad_tracker", "spad_posetrack"}:
        raise ValueError("--preprocessor hyb requires --tracker spad_tracker or spad_posetrack")
    spad_model.preprocessor = override_preprocessor.to(device)
    spad_model.preprocessor_name = override_name


def parse_args():
    ap = argparse.ArgumentParser(
        description="Sequence-mode SPAD test set -> trained SpadPoseModel -> JSON predictions and optional visualizations"
    )
    ap.add_argument("--in_path", type=str, default=None, help="Raw sample path/root, or rendered sample path/root when --cache_mode rendered")
    ap.add_argument("--test_json", type=str, default=None, help="VisionSIM test split JSON; when set, samples are read from its `spad` paths.")
    ap.add_argument("--in_glob", type=str, default=None, help="Optional glob when --in_path is a root folder")
    ap.add_argument("--test_name", type=str, default=None, help="Output test-set folder name; defaults from --test_json or --in_path")
    ap.add_argument("--ckpt", type=str, required=True, help="Trained sequence-mode SPAD pose checkpoint")
    ap.add_argument("--device", type=str, default="", help="Torch device, e.g. cpu / cuda:0 / mps")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--tracker", type=str, default="botsort", choices=TRACKER_CHOICES)
    ap.add_argument("--frame_rate", type=int, default=25, help="Tracker frame-rate hint")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument(
        "--spad-bin-rate-hz",
        type=float,
        default=8000.0,
        help="Raw-bin frequency of the current inference input. Detector-side SSD time deltas are scaled relative to training.",
    )
    ap.add_argument("--cache_mode", type=str, default="auto", choices=["auto", "raw", "rendered"])
    ap.add_argument("--preprocessor", type=str, default="model", choices=PREPROCESSOR_CHOICES)
    ap.add_argument(
        "--preprocessor-override",
        type=str,
        default="none",
        choices=["none", "ppb", "sum", "stea", "hyb"],
        help="Deprecated alias for --preprocessor when --preprocessor model.",
    )
    ap.add_argument("--render_root", type=str, default=None, help="Optional explicit root for cached rendered frames")
    ap.add_argument("--source_render_dirname", type=str, default="renders-spc8kHz")
    ap.add_argument("--render_contains_confidence", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--spad_chunk_t", "--cube_chunk_t", dest="cube_chunk_t", type=int, default=0)
    ap.add_argument("--cube_chunk_stride", type=int, default=0, help="Only used for raw direct-path evaluation without GT alignment")
    ap.add_argument("--spad_stride_frames", type=int, default=0)
    ap.add_argument("--spad_bins_per_gt", type=int, default=0)
    ap.add_argument("--spad_subsampling", type=int, default=0)
    ap.add_argument("--input_gamma", type=float, default=0.0, help="Used for rendered-cache fingerprinting and override preprocessors")
    ap.add_argument(
        "--spad_online",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Serial/online inference: carry detector temporal state across frames (default). "
        "Use --no-spad_online for training-style windowed batch plugins without state carry.",
    )
    ap.add_argument("--ppb-bocpd-gamma", type=float, default=5e-4)
    ap.add_argument("--ppb-quantile", type=float, default=1.0)
    ap.add_argument("--ppb-normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppb-min-filter-size", type=int, default=7)
    ap.add_argument("--stea-fast-window", type=int, default=16)
    ap.add_argument("--stea-slow-window", type=int, default=128)
    ap.add_argument("--stea-temporal-window", type=int, default=5)
    ap.add_argument("--stea-fast-tau", type=float, default=6.0)
    ap.add_argument("--stea-motion-sharpness", type=float, default=60.0)
    ap.add_argument("--stea-motion-threshold", type=float, default=0.07)
    ap.add_argument("--stea-stable-prior", type=float, default=16.0)
    ap.add_argument("--stea-normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stea-quantile", type=float, default=1.0)
    ap.add_argument("--hyb-motion-sharpness", type=float, default=60.0)
    ap.add_argument("--hyb-motion-threshold", type=float, default=0.05)
    ap.add_argument("--hyb-warp-block-size", type=int, default=16)
    ap.add_argument("--hyb-source-space", type=str, default="rgb", choices=["rgb", "raw"])
    ap.add_argument(
        "--tail_pad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deprecated: final chunks shorter than cube_chunk_t are always dropped.",
    )
    ap.add_argument(
        "--drop_tail",
        action="store_true",
        help="Deprecated: final chunks shorter than cube_chunk_t are always dropped.",
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

    if args.test_json:
        test_json = Path(args.test_json)
        if not test_json.is_file():
            raise FileNotFoundError(f"Test split JSON not found: {test_json}")
    elif not args.in_path:
        raise ValueError("Either --test_json or --in_path must be provided.")

    yolo = YOLO(str(ckpt))
    spad_model = yolo.model
    if not getattr(spad_model, "spad_enabled", False) or not hasattr(spad_model, "preprocessor"):
        raise TypeError(
            f"Checkpoint {args.ckpt} is not a trained SPAD pose model. "
            f"Loaded type: {spad_model.__class__.__name__}"
        )
    if hasattr(spad_model, "frame_adapter_name"):
        raise TypeError(
            f"Checkpoint {args.ckpt} is a frame-mode SPAD pose model. Use test_spad_pose_frame.py instead."
        )

    device = _resolve_device(args.device)
    spad_model.to(device)
    spad_model.eval()
    _configure_model_spad_bin_rate(spad_model, current_bin_rate_hz=float(args.spad_bin_rate_hz))

    names = yolo.names
    kpt_shape = getattr(spad_model, "kpt_shape", (21, 3))
    effective_preprocessor = _choose_effective_preprocessor(spad_model, args)
    if effective_preprocessor == "hyb" and str(args.cache_mode).strip().lower() == "rendered":
        raise ValueError("HYB is supported only for raw-SPAD evaluation.")

    if args.test_json:
        sample_specs = _sample_specs_from_json(test_json)
        test_name = args.test_name or _default_test_name_from_json(test_json)
    else:
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

    resolved_cache_mode = _resolve_cache_mode(args.cache_mode, sample_specs, effective_preprocessor)
    if effective_preprocessor == "hyb" and resolved_cache_mode != "raw":
        raise ValueError("HYB is supported only for raw-SPAD evaluation.")

    chunk_size = _resolve_chunk_size(spad_model, args)
    if chunk_size <= 0:
        raise ValueError(f"Sequence chunk size must be positive, got {chunk_size}")
    spad_bins_per_gt = _resolve_spad_bins_per_gt(spad_model, args)
    stride_frames = _resolve_spad_stride_frames(spad_model, args, ckpt=yolo.ckpt)
    print(
        f"Resolved sequence windowing: chunk_size={chunk_size}, "
        f"spad_bins_per_gt={spad_bins_per_gt}, stride_frames={stride_frames}, "
        f"chunk_stride_bins={int(stride_frames * spad_bins_per_gt)}, cache_mode={resolved_cache_mode}"
    )
    spad_subsampling = _resolve_spad_subsampling(spad_model, args)
    input_gamma = _resolve_input_gamma(spad_model, args)
    preprocessor_kwargs = _build_preprocessor_kwargs(
        args, preprocessor_name=effective_preprocessor, spad_subsampling=spad_subsampling
    )

    if resolved_cache_mode == "raw":
        _apply_sequence_preprocessor_override(spad_model, args, device, effective_preprocessor)
        spad_model.spad_cache_mode = "raw"
    else:
        spad_model.spad_cache_mode = "rendered"

    spad_model.spad_set_online_inference(bool(args.spad_online))
    if args.spad_online:
        print(
            f"Sequence inference: online/serial streaming "
            f"(cache_mode={resolved_cache_mode}; detector plugins carry state across frames)"
        )
        if resolved_cache_mode == "raw":
            print(
                "Raw mode: each chunk may reconstruct multiple frames; "
                "detector plugins step those frames serially within and across chunks."
            )
    else:
        print("Sequence inference: windowed batch plugins (state not carried across forwards)")

    out_dir = _eval_dir(run_name, test_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_root = _vis_dir(run_name, test_name) if args.vis != "none" else None
    if vis_root is not None:
        vis_root.mkdir(parents=True, exist_ok=True)

    tracker = None if args.tracker == "none" else _init_tracker(args.tracker, frame_rate=args.frame_rate, class_names=names)

    expected_render_fingerprint = None
    if resolved_cache_mode == "rendered" and not all(
        _sample_record_has_explicit_render(spec.get("record"), effective_preprocessor) for spec in sample_specs
    ):
        stride_bins = int(stride_frames * spad_bins_per_gt)
        expected_render_config = build_render_config(
            preprocessor=effective_preprocessor,
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
                "spad_cache_mode": resolved_cache_mode,
                "requested_cache_mode": str(args.cache_mode),
                "preprocessor": effective_preprocessor,
                "spad_chunk_size": int(chunk_size),
                "spad_bins_per_gt": int(spad_bins_per_gt),
                "spad_stride_frames": int(stride_frames),
                "input_gamma": float(input_gamma),
                "spad_online": bool(args.spad_online),
                "requested_spad_online": bool(args.spad_online),
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
                    stride_bins = int(stride_frames * spad_bins_per_gt)
                else:
                    stride_bins = int(args.cube_chunk_stride) if int(args.cube_chunk_stride) > 0 else chunk_size
                    chunk_records = _sliding_chunk_records(
                        total_bins=total_bins,
                        chunk_size=chunk_size,
                        stride=stride_bins,
                        packed_nch=source.packed_nch,
                    )

                subsampling = int(getattr(getattr(spad_model, "preprocessor", None), "subsampling", spad_subsampling) or 1)
                if not gt_aligned and chunk_size < subsampling:
                    raise ValueError(
                        f"cube_chunk_t={chunk_size} is shorter than preprocessor subsampling={subsampling}, "
                        "which would produce zero reconstructed frames."
                    )

                video_record: dict[str, Any] = {
                    "video_idx": int(video_idx),
                    "layout": source.layout,
                    "packed_nch": int(source.packed_nch),
                    "total_bins": int(total_bins),
                    "chunk_t": int(chunk_size),
                    "chunk_stride": int(stride_bins),
                    "subsampling": int(subsampling),
                    "chunks": [],
                }

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
                        video_tensor = torch.from_numpy(raw_chunk).unsqueeze(0).to(device)
                        raw_preds = spad_model(video_tensor)
                        preds = _postprocess_pose_predictions(
                            raw_preds,
                            conf=args.det_thresh,
                            iou=args.iou,
                            nc=len(names),
                            max_det=args.max_det,
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
                        results, gt_aligned=gt_aligned, t0=t0, t1=t1, subsampling=subsampling
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
                                output_frame_idx=output_frame_idx,
                                source_bin=source_bin,
                            )
                        )
                        global_frame_idx += 1

                    if chunk_record["frames"]:
                        video_record["chunks"].append(chunk_record)

                sample_record["videos"].append(video_record)
        else:
            if sample_spec.get("record") is not None:
                render_dir, frames_path, confidence_path, meta_path = _render_paths_from_record(
                    sample_spec["record"],
                    preprocessor=effective_preprocessor,
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

            frames, confidence, chunk_records, meta = _load_render_meta(
                sample_name=sample_name,
                sample_path=sample_path,
                record=sample_spec.get("record"),
                render_dir=render_dir,
                frames_path=frames_path,
                confidence_path=confidence_path,
                meta_path=meta_path,
                require_confidence=bool(args.render_contains_confidence),
                expected_preprocessor=effective_preprocessor,
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
                    _cfg_get(meta_config, "stride_bins", _cfg_get(meta, "stride_bins", stride_frames * spad_bins_per_gt))
                    or (stride_frames * spad_bins_per_gt)
                ),
                "subsampling": int(
                    _cfg_get(meta_config, "extra_kwargs", {}).get(
                        "subsampling",
                        meta.get("subsampling", getattr(getattr(spad_model, "preprocessor", None), "subsampling", spad_subsampling) or 1),
                    )
                ),
                "chunks": [],
            }

            for chunk in tqdm(chunk_records, desc=f"Testing [{sample_name}]"):
                frame = frames[chunk.chunk_index]
                frame_tensor = _frame_to_tensor(frame, device=device)
                seq_tensor = _frame_to_sequence_tensor(frame_tensor)
                spad_model.spad_packed_nch = int(chunk.packed_nch)
                spad_model.spad_cached_confidence_batch = None

                with torch.inference_mode():
                    raw_preds = _run_sequence_chunk_forward(
                        spad_model,
                        seq_tensor,
                        t_index_ll=[int(chunk.spad_end_bin)],
                    )
                    preds = _postprocess_pose_predictions(
                        raw_preds,
                        conf=args.det_thresh,
                        iou=args.iou,
                        nc=len(names),
                        max_det=args.max_det,
                        kpt_shape=kpt_shape,
                    )
                    recon_frames_bgr = _recon_frames_bgr(spad_model, batch_index=0)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

                results = _results_from_preds(
                    preds,
                    recon_frames_bgr,
                    names,
                    prefix=f"{sample_name}_cube00000_t{chunk.spad_start_bin:06d}_{chunk.spad_end_bin:06d}",
                    kpt_shape=kpt_shape,
                )
                if len(results) != 1:
                    raise ValueError(
                        f"Rendered cache evaluation expects one result per chunk, got {len(results)} "
                        f"for chunk {chunk.chunk_index}"
                    )

                chunk_record = {
                    "chunk_idx": int(chunk.chunk_index),
                    "start_bin": int(chunk.spad_start_bin),
                    "end_bin": int(chunk.spad_end_bin),
                    "input_bins": int(chunk.chunk_size),
                    "model_input_bins": int(chunk.chunk_size),
                    "target_gt_time": chunk.target_gt_time,
                    "frames": [],
                }

                result = results[0]
                if tracker is not None:
                    result = _apply_tracker(result, tracker)
                source_bin = int(chunk.spad_end_bin)

                if args.vis != "none":
                    vis, recon, _ = _build_cached_vis_frame(result)
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
                        source_bin=source_bin,
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
