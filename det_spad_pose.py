# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""det_spad_pose.py

Run inference with a trained SPAD pose checkpoint on raw SPAD clips.

Unlike the older SPAD predictor workflow, this script does not reconstruct frames in the
predictor. Instead it feeds raw SPAD clips directly into the checkpoint's `SpadPoseModel`, which
already contains the trained `PerPixelBayesian + SSD + YOLO pose head` pipeline.

Expected input formats
----------------------
- A root directory plus `--in_glob`, where each matched subfolder is one independent
  video/sample and contains its `.npy` data file.
- A directory containing one or more `.npy` files, each treated in sorted order.
- OR a single `.npy` file shaped:
    * (T, H, W, 1)       raw SPAD clip
    * (T, H, W)          raw SPAD clip
    * (H, W, T)          legacy raw photon cube
    * (N, H, W, T)       sequence of legacy raw photon cubes
    * (T, H, Wpacked, 3) synthetic packed R,G,B (G duplicated to both Bayer G sites on load)
    * (T, H, Wpacked, 4) real packed R,G1,G2,B (RGGB sites filled separately on load)

The checkpoint reconstructs `T'` RGB-like frames internally. This script runs the model in
**causal streaming** mode across chunks (preprocessor + detector temporal plugins carry state),
postprocesses detections, applies optional ByteTrack/BoT-SORT tracking, and saves per-frame
visualizations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data.spad_packed import (
    infer_packed_expected_w,
    infer_packed_nch,
    is_packed_spad,
    packed_frames_to_raw_video,
    raw_video_mean_to_rgb_u8,
)
from ultralytics.engine.results import Results
from ultralytics.models.yolo.pose.spad_preprocessors import build_spad_preprocessor
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.trackers.utils.result_layout import apply_pose_tracks_to_result
from ultralytics.utils import IterableSimpleNamespace, YAML, nms
from ultralytics.utils.checks import check_yaml


def _np_load(path: Path) -> np.ndarray:
    """Load .npy with memory mapping (keeps most data on disk)."""
    return np.load(path, mmap_mode="r")


BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGER_COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]
COLOR_KEYPOINT = (255, 255, 255)
COLOR_WRIST = (255, 165, 0)


def _get_finger_color(bone_index: int) -> tuple[int, int, int]:
    if bone_index < 4:
        return FINGER_COLORS[0]
    if bone_index < 8:
        return FINGER_COLORS[1]
    if bone_index < 12:
        return FINGER_COLORS[2]
    if bone_index < 16:
        return FINGER_COLORS[3]
    return FINGER_COLORS[4]


def draw_bbox(img_bgr: np.ndarray, track_id: int, box_xyxyc: np.ndarray, handedness: float) -> np.ndarray:
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


def draw_pose(img_bgr: np.ndarray, pose_kpts: np.ndarray, thresh: float = 0.5, k: int = 21) -> np.ndarray:
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


def _packed_frames_to_raw_video(
    frames_packed: np.ndarray, *, expected_w: int | None = None, ch_order: str = "RGB"
) -> np.ndarray:
    """Convert packed `(T,H,Wpacked,3|4)` to raw SPAD video (full width when expected_w<=0/None)."""
    return packed_frames_to_raw_video(frames_packed, expected_w=expected_w, ch_order=ch_order)


def _looks_like_hwt(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[0] == arr.shape[1] and arr.shape[2] != arr.shape[1]


@dataclass(frozen=True)
class RawVideoSource:
    """One lazily-sliced video source."""

    array: np.ndarray
    layout: str
    packed_nch: int = 4
    expected_w: int = 0  # 0 => full Wpacked*8 for packed layouts


def _video_sources_from_array(arr: np.ndarray, *, expected_w: int = 0) -> list[RawVideoSource]:
    """Describe supported layouts without materializing the full raw video."""
    if is_packed_spad(arr):
        ew = int(expected_w) if int(expected_w) > 0 else infer_packed_expected_w(arr)
        return [RawVideoSource(arr, "packed", packed_nch=infer_packed_nch(arr), expected_w=ew)]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return [RawVideoSource(arr, "thwc1", packed_nch=4)]
    if arr.ndim == 3:
        return [RawVideoSource(arr, "hwt" if _looks_like_hwt(arr) else "thw", packed_nch=4)]
    if arr.ndim == 4:
        return [RawVideoSource(arr[i], "hwt", packed_nch=4) for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported input shape: {arr.shape}")


def _iter_raw_video_sources_from_sample_path(in_path: Path, *, expected_w: int = 0):
    """Yield lazy video sources from a sample path."""
    if in_path.is_dir():
        files = sorted([p for p in in_path.iterdir() if p.suffix.lower() == ".npy"])
        for p in files:
            yield from _video_sources_from_array(_np_load(p), expected_w=expected_w)
        return

    if in_path.suffix.lower() == ".npy":
        yield from _video_sources_from_array(_np_load(in_path), expected_w=expected_w)
        return

    raise ValueError(f"Unsupported input path: {in_path}")


def _video_num_bins(source: RawVideoSource) -> int:
    if source.layout in {"packed", "thwc1", "thw"}:
        return int(source.array.shape[0])
    if source.layout == "hwt":
        return int(source.array.shape[2])
    raise ValueError(f"Unsupported source layout: {source.layout}")


def _slice_raw_chunk(source: RawVideoSource, t0: int, t1: int, *, packed_ch_order: str) -> np.ndarray:
    """Load one raw `(T,H,W,1)` chunk from a lazy source."""
    if source.layout == "packed":
        packed = np.asarray(source.array[t0:t1])
        ew = int(source.expected_w) if int(source.expected_w) > 0 else None
        return _packed_frames_to_raw_video(packed, expected_w=ew, ch_order=packed_ch_order)
    if source.layout == "thwc1":
        return np.ascontiguousarray(source.array[t0:t1].astype(np.uint8, copy=False))
    if source.layout == "thw":
        return np.ascontiguousarray(source.array[t0:t1, :, :, None].astype(np.uint8, copy=False))
    if source.layout == "hwt":
        return np.ascontiguousarray(np.transpose(source.array[:, :, t0:t1], (2, 0, 1))[:, :, :, None].astype(np.uint8, copy=False))
    raise ValueError(f"Unsupported source layout: {source.layout}")


def _pad_raw_chunk_repeat_last(raw_chunk: np.ndarray, target_t: int) -> np.ndarray:
    """Pad a raw chunk to `target_t` by repeating its last frame."""
    if raw_chunk.shape[0] >= target_t:
        return raw_chunk
    if raw_chunk.shape[0] == 0:
        return raw_chunk
    pad_t = int(target_t) - int(raw_chunk.shape[0])
    last = raw_chunk[-1:, :, :, :]
    pad = np.repeat(last, pad_t, axis=0)
    return np.ascontiguousarray(np.concatenate((raw_chunk, pad), axis=0))


def _prepare_raw_chunk_for_spad(raw_chunk: np.ndarray, *, chunk_t: int, tail_pad_full: bool) -> np.ndarray | None:
    """Prepare one raw chunk for SPAD inference."""
    if raw_chunk.shape[0] == 0:
        return None
    if tail_pad_full and raw_chunk.shape[0] < chunk_t:
        raw_chunk = _pad_raw_chunk_repeat_last(raw_chunk, chunk_t)
    return raw_chunk


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _trained_chunk_t(model) -> int | None:
    model_chunk_size = getattr(model, "spad_chunk_size", None)
    if model_chunk_size:
        return int(model_chunk_size)
    train_args = getattr(model, "args", None)
    chunk_size = _cfg_get(train_args, "spad_chunk_size", None)
    if chunk_size:
        return int(chunk_size)
    output_frames = _cfg_get(train_args, "spad_output_frames", None)
    subsampling = getattr(getattr(model, "preprocessor", None), "subsampling", None)
    if output_frames is None or subsampling is None:
        return None
    return int(output_frames) * int(subsampling)


def _reference_bin_rate_hz(model) -> float:
    if hasattr(model, "spad_reference_bin_rate_hz"):
        return float(getattr(model, "spad_reference_bin_rate_hz"))
    train_args = getattr(model, "args", None)
    return float(_cfg_get(train_args, "spad_bin_rate_hz", 2000.0))


def _configure_model_spad_bin_rate(model, *, current_bin_rate_hz: float) -> None:
    current_bin_rate_hz = float(current_bin_rate_hz)
    if current_bin_rate_hz <= 0:
        raise ValueError(f"spad_bin_rate_hz must be positive, got {current_bin_rate_hz}")
    reference_bin_rate_hz = _reference_bin_rate_hz(model)
    if hasattr(model, "set_spad_bin_rate_hz"):
        model.set_spad_bin_rate_hz(
            current_bin_rate_hz=current_bin_rate_hz,
            reference_bin_rate_hz=reference_bin_rate_hz,
        )
        return
    setattr(model, "spad_reference_bin_rate_hz", reference_bin_rate_hz)
    setattr(model, "spad_current_bin_rate_hz", current_bin_rate_hz)
    plugins = getattr(model, "plugins_by_layer", None)
    if plugins is None:
        return
    for plugin in plugins.values():
        if hasattr(plugin, "set_bin_rate_hz"):
            plugin.set_bin_rate_hz(
                current_bin_rate_hz=current_bin_rate_hz,
                reference_bin_rate_hz=reference_bin_rate_hz,
            )


def _build_override_preprocessor(args) -> tuple[str | None, object | None]:
    name = str(getattr(args, "preprocessor_override", "none")).strip().lower()
    if name in {"", "none"}:
        return None, None

    spad_subsampling = int(getattr(args, "spad_subsampling", 64))
    if spad_subsampling <= 0:
        spad_subsampling = 64
    kwargs: dict[str, Any] = {"subsampling": spad_subsampling}
    if name == "ppb":
        kwargs.update(
            {
                "bocpd_gamma": float(args.ppb_bocpd_gamma),
                "memory_size": int(args.ppb_memory_size),
                "normalize": bool(args.ppb_normalize),
                "quantile": float(args.ppb_quantile),
                "min_filter_size": int(args.ppb_min_filter_size),
            }
        )
    elif name == "stea":
        kwargs.update(
            {
                "fast_window": int(args.stea_fast_window),
                "slow_window": int(args.stea_slow_window),
                "temporal_window": int(args.stea_temporal_window),
                "fast_tau": float(args.stea_fast_tau),
                "motion_sharpness": float(args.stea_motion_sharpness),
                "motion_threshold": float(args.stea_motion_threshold),
                "stable_prior": float(args.stea_stable_prior),
                "normalize": bool(args.stea_normalize),
                "quantile": float(args.stea_quantile),
            }
        )
    elif name == "hyb":
        # HYB shares STEA temporal/motion knobs; only warp_* are HYB-specific.
        kwargs.update(
            {
                "fast_window": int(args.stea_fast_window),
                "slow_window": int(args.stea_slow_window),
                "temporal_window": int(args.stea_temporal_window),
                "fast_tau": float(args.stea_fast_tau),
                "motion_sharpness": float(args.stea_motion_sharpness),
                "motion_threshold": float(args.stea_motion_threshold),
                "stable_prior": float(args.stea_stable_prior),
                "normalize": bool(args.stea_normalize),
                "quantile": float(args.stea_quantile),
                "warp_block_size": int(args.hyb_warp_block_size),
                "source_space": str(args.hyb_source_space),
            }
        )
    elif name == "sum":
        kwargs = {"subsampling": spad_subsampling}
    elif name == "ema":
        kwargs = {"subsampling": spad_subsampling, "ema_alpha": float(getattr(args, "ema_alpha", 0.01))}
    elif name == "hire":
        kwargs.update(
            {
                "sample_rate_hz": float(getattr(args, "spad_bin_rate_hz", 2000.0)),
                "fast_bins": int(getattr(args, "hire_fast_bins", 24)),
                "slow_bins": int(getattr(args, "hire_slow_bins", 160)),
                "surprise_bins": int(getattr(args, "hire_surprise_bins", 4)),
                "mix_hold_bins": int(getattr(args, "hire_mix_hold_bins", 80)),
                "mix_bins": float(
                    getattr(
                        args,
                        "hire_mix_bins",
                        getattr(args, "hire_mix_kappa", getattr(args, "hire_gate_theta", 12.0)),
                    )
                ),
                "mix_theta": float(getattr(args, "hire_mix_theta", 0.06)),
                "mix_floor": float(getattr(args, "hire_mix_floor", -1.0)),
                "theta_on": float(getattr(args, "hire_theta_on", 0.08)),
                "theta_off": float(getattr(args, "hire_theta_off", 0.02)),
                "theta_grow": float(getattr(args, "hire_theta_grow", -1.0)),
                "confirm_bins": int(getattr(args, "hire_confirm_bins", 4)),
                "cooldown_bins": int(getattr(args, "hire_cooldown_bins", 0)),
                "spatial_kernel": int(getattr(args, "hire_spatial_kernel", 5)),
                "gate_pool": str(getattr(args, "hire_gate_pool", "max")),
                "reset_open": int(getattr(args, "hire_reset_open", 15)),
                "reset_grow": int(getattr(args, "hire_reset_grow", 6)),
                "normalize": bool(getattr(args, "hire_normalize", True)),
                "quantile": float(getattr(args, "hire_quantile", 1.0)),
            }
        )
    else:
        raise ValueError(f"Unsupported --preprocessor-override: {name!r}")

    return name, build_spad_preprocessor(name, kwargs=kwargs)


def _set_velocity_field_on_preprocessor(preprocessor, tracker) -> None:
    if preprocessor is None or not hasattr(preprocessor, "set_velocity_field"):
        return
    if tracker is None:
        preprocessor.set_velocity_field(None, source_space="rgb")
        return
    if not hasattr(tracker, "last_velocity_field"):
        raise TypeError("This preprocessor requires a tracker that exposes 'last_velocity_field'.")
    preprocessor.set_velocity_field(getattr(tracker, "last_velocity_field", None), source_space="rgb")


def _recon_frames_bgr(model, batch_index: int = 0) -> list[np.ndarray]:
    frames = getattr(model, "spad_last_recon_frames", None)
    if frames is None:
        return []
    if torch.is_tensor(frames) and frames.ndim == 4:
        rgb = frames[batch_index : batch_index + 1].detach().float().cpu().permute(0, 2, 3, 1).numpy()
    else:
        rgb = frames[:, batch_index].detach().float().cpu().permute(0, 2, 3, 1).numpy()
    bgr = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)[:, :, :, ::-1]
    return [np.ascontiguousarray(frame) for frame in bgr]


def _letterbox_tbchw(
    frames_t_b_c_h_w: torch.Tensor, imgsz: int
) -> tuple[torch.Tensor, dict[str, float | int | tuple[int, int]]]:
    """Letterbox ``T,B,3,H,W`` float frames to square ``imgsz`` (preserve aspect)."""
    if frames_t_b_c_h_w.ndim != 5 or int(frames_t_b_c_h_w.shape[2]) != 3:
        raise ValueError(f"Expected T,B,3,H,W frames, got shape={tuple(frames_t_b_c_h_w.shape)}")
    imgsz = int(imgsz)
    if imgsz <= 0:
        raise ValueError(f"imgsz must be positive, got {imgsz}")

    t, b, c, h, w = map(int, frames_t_b_c_h_w.shape)
    if h == imgsz and w == imgsz:
        meta = {
            "ratio": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
            "native_hw": (h, w),
            "imgsz": imgsz,
        }
        return frames_t_b_c_h_w, meta

    ratio = min(imgsz / float(h), imgsz / float(w))
    new_h = max(int(round(h * ratio)), 1)
    new_w = max(int(round(w * ratio)), 1)
    flat = frames_t_b_c_h_w.reshape(t * b, c, h, w)
    resized = torch.nn.functional.interpolate(flat, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = imgsz - new_h
    pad_w = imgsz - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    boxed = torch.nn.functional.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    out = boxed.reshape(t, b, c, imgsz, imgsz)
    meta = {
        "ratio": float(ratio),
        "pad_x": float(pad_left),
        "pad_y": float(pad_top),
        "native_hw": (h, w),
        "imgsz": imgsz,
    }
    return out, meta


def _scale_pose_preds_to_native(
    preds: list[torch.Tensor],
    *,
    scale_meta: dict[str, float | int | tuple[int, int]] | None,
    kpt_shape,
) -> list[torch.Tensor]:
    """Map detector-space pose preds (letterboxed imgsz) back to native recon coordinates."""
    if not scale_meta:
        return preds
    ratio = float(scale_meta["ratio"])
    pad_x = float(scale_meta["pad_x"])
    pad_y = float(scale_meta["pad_y"])
    native_h, native_w = map(int, scale_meta["native_hw"])
    if ratio <= 0:
        return preds

    nk, nd = int(kpt_shape[0]), int(kpt_shape[1])
    out = []
    for pred in preds:
        if pred is None or pred.numel() == 0:
            out.append(pred)
            continue
        scaled = pred.clone()
        scaled[:, 0] = (scaled[:, 0] - pad_x) / ratio
        scaled[:, 1] = (scaled[:, 1] - pad_y) / ratio
        scaled[:, 2] = (scaled[:, 2] - pad_x) / ratio
        scaled[:, 3] = (scaled[:, 3] - pad_y) / ratio
        scaled[:, 0].clamp_(0, native_w - 1)
        scaled[:, 1].clamp_(0, native_h - 1)
        scaled[:, 2].clamp_(0, native_w - 1)
        scaled[:, 3].clamp_(0, native_h - 1)
        if scaled.shape[1] > 6 and nk > 0 and nd >= 2:
            kpts = scaled[:, 6:].reshape(-1, nk, nd).clone()
            kpts[..., 0] = (kpts[..., 0] - pad_x) / ratio
            kpts[..., 1] = (kpts[..., 1] - pad_y) / ratio
            kpts[..., 0].clamp_(0, native_w - 1)
            kpts[..., 1].clamp_(0, native_h - 1)
            scaled[:, 6:] = kpts.reshape(scaled.shape[0], -1)
        out.append(scaled)
    return out


def _resize_to_shape_bgr(img_bgr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = map(int, shape_hw)
    if img_bgr.shape[:2] == (target_h, target_w):
        return np.ascontiguousarray(img_bgr.copy())
    return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _raw_sum_bgr(raw_video: np.ndarray, *, packed_nch: int) -> np.ndarray:
    rgb = raw_video_mean_to_rgb_u8(raw_video, packed_nch=int(packed_nch))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _raw_sum_readrgb_like_bgr(raw_video: np.ndarray, *, packed_nch: int) -> np.ndarray:
    rgb = raw_video_mean_to_rgb_u8(raw_video, packed_nch=int(packed_nch))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _postprocess_pose_predictions(
    raw_preds,
    *,
    conf: float,
    iou: float,
    nc: int,
    max_det: int,
    kpt_shape,
) -> list[torch.Tensor]:
    raw = raw_preds[0] if isinstance(raw_preds, (list, tuple)) and torch.is_tensor(raw_preds[0]) else raw_preds
    preds = nms.non_max_suppression(raw, conf, iou, nc=nc, multi_label=True, max_det=max_det)
    return [pred if pred is not None else raw.new_zeros((0, 6 + int(np.prod(kpt_shape)))) for pred in preds]


def _results_from_preds(preds: list[torch.Tensor], recon_frames_bgr: list[np.ndarray], names, *, prefix: str, kpt_shape) -> list[Results]:
    results = []
    empty_kpts = (0, int(kpt_shape[0]), int(kpt_shape[1]))
    for i, pred in enumerate(preds):
        orig = recon_frames_bgr[i] if i < len(recon_frames_bgr) else np.zeros((512, 512, 3), dtype=np.uint8)
        boxes = pred[:, :6] if pred.numel() else pred.new_zeros((0, 6))
        keypoints = pred[:, 6:].view(-1, int(kpt_shape[0]), int(kpt_shape[1])) if pred.numel() else pred.new_zeros(empty_kpts)
        results.append(Results(orig_img=orig, path=f"{prefix}_frame{i:06d}.png", names=names, boxes=boxes, keypoints=keypoints))
    return results


def _init_tracker(tracker_name: str, *, frame_rate: int, class_names=None):
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(f"{tracker_name}.yaml")))
    tracker_cls = TRACKER_MAP[cfg.tracker_type]
    if cfg.tracker_type in {"posetrack", "spad_posetrack"}:
        return tracker_cls(args=cfg, frame_rate=frame_rate, class_names=class_names)
    return tracker_cls(args=cfg, frame_rate=frame_rate)


def _apply_tracker(result: Results, tracker) -> Results:
    if result.boxes is None or len(result.boxes) == 0:
        return result
    det = result.boxes.cpu().numpy()
    keypoints = None
    if getattr(result, "keypoints", None) is not None and len(result.keypoints):
        keypoints = result.keypoints.data.cpu().numpy()
    if keypoints is not None and hasattr(tracker, "n_keypoints"):
        tracks = tracker.update(det, result.orig_img, getattr(result, "feats", None), keypoints=keypoints)
        return apply_pose_tracks_to_result(
            result,
            tracks,
            n_keypoints=int(getattr(tracker, "n_keypoints", 21)),
            kpt_dims=int(getattr(tracker, "kpt_dims", 3)),
        )
    tracks = tracker.update(det, result.orig_img, getattr(result, "feats", None))
    if len(tracks) == 0:
        return result
    idx = tracks[:, -1].astype(int)
    valid = (idx >= 0) & (idx < len(result))
    if not np.any(valid):
        return result[:0]
    idx = idx[valid]
    tracks = tracks[valid]
    tracked = result[idx]
    tracked.update(boxes=torch.as_tensor(tracks[:, :-1], device=result.boxes.data.device))
    return tracked


def main():
    ap = argparse.ArgumentParser(description="Raw SPAD clip -> trained SpadPoseModel -> pose tracking")
    ap.add_argument("--in_path", type=str, required=True, help="Sample/video folder, root folder, or a .npy file")
    ap.add_argument("--in_glob", type=str, default=None, help="Optional glob when --in_path is a root folder containing many sample/video subfolders")
    ap.add_argument("--save_dir", type=str, required=True, help="Directory to save visualized frames")
    ap.add_argument("--ckpt", type=str, default="weights/detector.pt")
    ap.add_argument("--device", type=str, default="", help="Torch device, e.g. cpu / cuda:0 / mps")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--tracker", type=str, default="spad_posetrack", choices=["bytetrack", "botsort", "spad_tracker", "posetrack", "spad_posetrack"])
    ap.add_argument("--frame_rate", type=int, default=25, help="Tracker frame-rate hint")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument(
        "--spad-bin-rate-hz",
        type=float,
        default=8000.0,
        help=(
            "Raw-bin frequency of the current inference input. Detector-side SSD time deltas "
            "are scaled relative to the checkpoint's training reference frequency."
        ),
    )
    ap.add_argument(
        "--preprocessor-override",
        type=str,
        default="none",
        choices=["none", "ppb", "sum", "ema", "stea", "hyb", "hire"],
        help="Optionally override the checkpoint's internal SPAD preprocessor at inference time.",
    )
    ap.add_argument("--spad-subsampling", type=int, default=320, help="Temporal subsampling used by override preprocessors.")
    ap.add_argument("--ppb-bocpd-gamma", type=float, default=1e-3)
    ap.add_argument("--ppb-memory-size", type=int, default=10)
    ap.add_argument("--ppb-quantile", type=float, default=1.0)
    ap.add_argument("--ppb-normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppb-min-filter-size", type=int, default=5)
    ap.add_argument(
        "--ema-alpha",
        type=float,
        default=0.01,
        help="EMA new-sample weight (inference standard 0.01). <=0 uses 2/(subsampling+1) SMA-equivalent.",
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
    ap.add_argument("--hire-fast-bins", type=int, default=16)
    ap.add_argument("--hire-slow-bins", type=int, default=128)
    ap.add_argument("--hire-surprise-bins", type=int, default=8)
    ap.add_argument("--hire-gate-theta", type=float, default=0.2)
    ap.add_argument("--hire-spatial-kernel", type=int, default=3)
    ap.add_argument(
        "--hyb-motion-sharpness",
        type=float,
        default=None,
        help="Deprecated: HYB now uses --stea-motion-sharpness. Kept for CLI compatibility.",
    )
    ap.add_argument(
        "--hyb-motion-threshold",
        type=float,
        default=None,
        help="Deprecated: HYB now uses --stea-motion-threshold. Kept for CLI compatibility.",
    )
    ap.add_argument("--hyb-warp-block-size", type=int, default=16)
    ap.add_argument("--hyb-source-space", type=str, default="rgb", choices=["rgb", "raw"])
    ap.add_argument(
        "--cube_chunk_t",
        type=int,
        default=0,
        help="If >0, split each raw video into chunks of this many bins. If 0, use the train-time SPAD window length when available, else full video.",
    )
    ap.add_argument("--cube_chunk_stride", type=int, default=0, help="Stride for chunking; default uses cube_chunk_t (no overlap)")
    ap.add_argument(
        "--tail_pad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pad final short chunk to cube_chunk_t; with --no-tail_pad, run short tails at natural length.",
    )
    ap.add_argument("--vis_bg", type=str, default="recon", choices=["sum", "recon"], help="Visualization background")
    ap.add_argument(
        "--save_readrgb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save the direct read_rgb-style sum visualization with suffix '_readrgb'.",
    )
    ap.add_argument("--save_ppb_demosaic_compare", action=argparse.BooleanOptionalAction, default=False, help=argparse.SUPPRESS)

    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir() and args.in_glob:
        sample_paths = [p for p in sorted(in_path.glob(args.in_glob)) if p.is_dir()]
        if not sample_paths:
            raise FileNotFoundError(f"No sample/video subfolders matched: {in_path}/{args.in_glob}")
    else:
        sample_paths = [in_path]

    yolo = YOLO(args.ckpt)
    spad_model = yolo.model
    if not getattr(spad_model, "spad_enabled", False) or not hasattr(spad_model, "preprocessor"):
        raise TypeError(
            f"Checkpoint {args.ckpt} is not a trained SPAD pose model. "
            f"Loaded type: {spad_model.__class__.__name__}"
        )

    device = _resolve_device(args.device)
    spad_model.to(device)
    spad_model.eval()
    _configure_model_spad_bin_rate(spad_model, current_bin_rate_hz=float(args.spad_bin_rate_hz))
    names = yolo.names
    tracker = _init_tracker(args.tracker, frame_rate=args.frame_rate, class_names=names)
    override_name, override_preprocessor = _build_override_preprocessor(args)
    if override_preprocessor is not None:
        if override_name == "hyb" and args.tracker not in {"spad_tracker", "spad_posetrack"}:
            raise ValueError("--preprocessor-override hyb requires --tracker spad_tracker or spad_posetrack")
        spad_model.preprocessor = override_preprocessor.to(device)
        spad_model.preprocessor_name = override_name
    kpt_shape = getattr(spad_model, "kpt_shape", (21, 3))
    trained_chunk_t = _trained_chunk_t(spad_model)
    global_frame_idx = 0

    for sample_path in sample_paths:
        sample_name = sample_path.name if sample_path.is_dir() else sample_path.stem
        out_dir = save_dir / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)

        tracker.reset()
        video_iter = _iter_raw_video_sources_from_sample_path(sample_path)

        for video_idx, source in enumerate(tqdm(video_iter, desc=f"Processing video [{sample_name}]")):
            total_bins = _video_num_bins(source)
            if int(args.cube_chunk_t) > 0:
                chunk_t = int(args.cube_chunk_t)
            elif trained_chunk_t is not None:
                chunk_t = int(trained_chunk_t)
            else:
                raise ValueError(
                    "Unable to infer train-time SPAD window length from checkpoint. "
                    "Pass --cube_chunk_t explicitly, e.g. --cube_chunk_t 64 for spad_output_frames=1,spad_subsampling=64. "
                    f"Refusing to process the full video as one chunk (T={total_bins}), which is likely to OOM."
                )
            if chunk_t <= 0:
                raise ValueError(f"cube_chunk_t must be positive, got {chunk_t}")
            subsampling = int(getattr(getattr(spad_model, "preprocessor", None), "subsampling", 1) or 1)
            requires_multiframe_chunk = not hasattr(spad_model, "frame_adapter_name")
            if requires_multiframe_chunk and chunk_t < subsampling:
                raise ValueError(
                    f"cube_chunk_t={chunk_t} is shorter than preprocessor subsampling={subsampling}, "
                    "which would produce zero reconstructed frames. Increase --cube_chunk_t or use the checkpoint default."
                )
            stride = int(args.cube_chunk_stride) if int(args.cube_chunk_stride) > 0 else chunk_t

            # Causal streaming across chunks (HIRE/STEA + detector temporal plugins).
            if hasattr(spad_model, "spad_begin_stream"):
                spad_model.spad_begin_stream()
            else:
                spad_model.spad_set_online_inference(True)
                spad_model.spad_clear_plugin_states()

            try:
                for t0 in range(0, total_bins, stride):
                    t1 = min(total_bins, t0 + chunk_t)
                    raw_chunk = _slice_raw_chunk(source, t0, t1, packed_ch_order=args.packed_ch_order)
                    pad_tail = bool(args.tail_pad) and (t1 >= total_bins)
                    raw_chunk = _prepare_raw_chunk_for_spad(
                        raw_chunk, chunk_t=chunk_t, tail_pad_full=pad_tail
                    )
                    if raw_chunk is None:
                        continue

                    spad_model.spad_packed_nch = int(source.packed_nch)
                    _set_velocity_field_on_preprocessor(spad_model.preprocessor, tracker)

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

                    bg_bgr = None
                    if args.vis_bg == "sum":
                        bg_bgr = _raw_sum_bgr(raw_chunk, packed_nch=source.packed_nch)
                    readrgb_bgr = (
                        _raw_sum_readrgb_like_bgr(raw_chunk, packed_nch=source.packed_nch)
                        if bool(args.save_readrgb)
                        else None
                    )

                    for result in results:
                        result = _apply_tracker(result, tracker)
                        recon = (
                            np.ascontiguousarray(result.orig_img.copy())
                            if getattr(result, "orig_img", None) is not None
                            else np.zeros((512, 512, 3), dtype=np.uint8)
                        )
                        if args.vis_bg == "recon" and getattr(result, "orig_img", None) is not None:
                            vis = recon.copy()
                        else:
                            vis = bg_bgr.copy() if bg_bgr is not None else np.zeros_like(recon)
                        readrgb = _resize_to_shape_bgr(readrgb_bgr, vis.shape[:2]) if readrgb_bgr is not None else None

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
                                vis = draw_bbox(vis, int(tid), box_xyxyc, float(handedness[j]))
                                if poses is not None and j < len(poses):
                                    vis = draw_pose(vis, poses[j])

                        out_path = out_dir / f"cube{video_idx:05d}_t{t0:06d}_{t1:06d}_frame{global_frame_idx:07d}.png"
                        cv2.imwrite(str(out_path), vis)
                        recon_path = out_dir / f"cube{video_idx:05d}_t{t0:06d}_{t1:06d}_frame{global_frame_idx:07d}_recon.png"
                        cv2.imwrite(str(recon_path), recon)
                        if readrgb is not None:
                            readrgb_path = (
                                out_dir / f"cube{video_idx:05d}_t{t0:06d}_{t1:06d}_frame{global_frame_idx:07d}_readrgb.png"
                            )
                            cv2.imwrite(str(readrgb_path), readrgb)
                        global_frame_idx += 1
            finally:
                if hasattr(spad_model, "spad_end_stream"):
                    spad_model.spad_end_stream()


if __name__ == "__main__":
    main()
