#!/usr/bin/env python3
"""Multi-method 25 fps pose dump: GT / pure-detector(rgb) / QNN-SSD / HIRE-SSD.

Streams packed/raw ``frames.npy`` at ``spad_bin_rate_hz`` with non-overlapping
``chunk_size`` windows (default 8000 Hz / 320 bins → 25 fps). Saves a shared
``poses.npy`` schema plus RGB-overlay PNGs under::

    save_dir/
      meta.json
      gt/{poses.npy, vis/frame_XXXXXXX.png}
      rgb/{poses.npy, vis/frame_XXXXXXX.png}   # pure detector
      qnn/{poses.npy, vis/frame_XXXXXXX.png}
      hire/{poses.npy, vis/frame_XXXXXXX.png}

All sequence models run with ``cache_mode=raw`` (online preprocessor from bins).

Examples
--------
python ultralytics/vis_main_pose.py \\
  --in_path /path/frames.npy \\
  --rgb_path /path/rgb \\
  --gt_path /path/hand_ann.json \\
  --save_dir /tmp/main_pose \\
  --detector_ckpt /path/detector.pt \\
  --qnn_ckpt /path/qnn_ssd.pt \\
  --hire_ckpt /path/hire_ssd.pt \\
  --qnn_pre_override ppb --ppb_bocpd_gamma 0.001 \\
  --hire_pre_override hire --hire_fast_bins 24 --hire_slow_bins 160
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.spad_packed import (
    raw_chunk_plane,
    raw_hwt_to_rgb_float,
    raw_plane_to_photon_cube,
)
from ultralytics.models.yolo.pose.spad_preprocessors import (
    build_spad_frame_preprocessor,
    build_spad_preprocessor,
)


_DSP_MOD = None
_VHP_MOD = None


def _dsp():
    """Lazy-load sibling ``det_spad_pose`` (avoids tracker deps for ``--help``)."""
    global _DSP_MOD
    if _DSP_MOD is None:
        path = Path(__file__).resolve().parent / "det_spad_pose.py"
        spec = importlib.util.spec_from_file_location("det_spad_pose", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["det_spad_pose"] = mod
        spec.loader.exec_module(mod)
        _DSP_MOD = mod
    return _DSP_MOD


def _vhp():
    """Lazy-load sibling ``vis_hire_pose`` draw helpers."""
    global _VHP_MOD
    if _VHP_MOD is None:
        path = Path(__file__).resolve().parent / "vis_hire_pose.py"
        spec = importlib.util.spec_from_file_location("vis_hire_pose", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["vis_hire_pose"] = mod
        spec.loader.exec_module(mod)
        _VHP_MOD = mod
    return _VHP_MOD


HAND_TO_CLASS = {"left_hand": 0, "right_hand": 1}
CLASS_TO_HAND = {0: "left_hand", 1: "right_hand"}
_FRAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Dump 25 fps GT / detector(rgb) / QNN / HIRE poses as npy + RGB overlays"
    )
    ap.add_argument("--in_path", type=Path, required=True, help="frames.npy or sample dir")
    ap.add_argument(
        "--rgb_path",
        type=Path,
        required=True,
        help="RGB dir with frames.npy (VisionSIM rgb25fps), frames.npy path, image dir, or video",
    )
    ap.add_argument("--gt_path", type=Path, required=True, help="GT hand annotation JSON")
    ap.add_argument("--save_dir", type=Path, required=True)
    ap.add_argument("--detector_ckpt", type=Path, required=True, help="Frozen YOLO pose detector .pt")
    ap.add_argument("--qnn_ckpt", type=Path, required=True, help="QNN/SSD sequence SpadPoseModel .pt")
    ap.add_argument("--hire_ckpt", type=Path, required=True, help="HIRE/SSD sequence SpadPoseModel .pt")
    ap.add_argument(
        "--detector_pre",
        type=str,
        default="sum",
        choices=["sum", "ema", "ppb", "stea", "hire"],
        help="External preprocessor for pure detector (rgb track)",
    )
    ap.add_argument(
        "--qnn_pre_override",
        type=str,
        default="none",
        choices=["none", "ppb", "ema", "sum", "stea", "hire"],
        help="Rebuild QNN sequence preprocessor from CLI knobs (none = keep ckpt)",
    )
    ap.add_argument(
        "--hire_pre_override",
        type=str,
        default="none",
        choices=["none", "hire", "ppb", "ema", "sum", "stea"],
        help="Rebuild HIRE sequence preprocessor from CLI knobs (none = keep ckpt)",
    )
    ap.add_argument("--chunk_size", type=int, default=320)
    ap.add_argument("--spad_bin_rate_hz", type=float, default=8000.0)
    ap.add_argument("--spad_bins_per_gt", type=int, default=64, help="GT@125Hz → bins (8000/125=64)")
    ap.add_argument("--gt_image_size", type=int, default=512, help="Pixel space of GT JSON coords")
    ap.add_argument("--start_bin", type=int, default=0)
    ap.add_argument("--end_bin", type=int, default=0, help="Exclusive; 0 = EOF")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--kpt_thresh", type=float, default=0.5)
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--input_gamma", type=float, default=2.2)
    # EMA
    ap.add_argument("--ema_alpha", type=float, default=0.01)
    ap.add_argument("--ema_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ema_quantile", type=float, default=1.0)
    # PPB (QNN)
    ap.add_argument("--ppb_bocpd_gamma", type=float, default=0.001)
    ap.add_argument("--ppb_memory_size", type=int, default=10)
    ap.add_argument("--ppb_quantile", type=float, default=1.0)
    ap.add_argument("--ppb_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppb_min_filter_size", type=int, default=5)
    # HIRE (defaults match sequence_hire_*_8kHz / vis_hire_pose)
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
    ap.add_argument("--hire_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hire_quantile", type=float, default=1.0)
    # STEA (optional for detector_pre / overrides)
    ap.add_argument("--stea_fast_window", type=int, default=8)
    ap.add_argument("--stea_slow_window", type=int, default=64)
    ap.add_argument("--stea_temporal_window", type=int, default=16)
    ap.add_argument("--stea_fast_tau", type=float, default=0.2)
    ap.add_argument("--stea_motion_sharpness", type=float, default=8.0)
    ap.add_argument("--stea_motion_threshold", type=float, default=0.05)
    ap.add_argument("--stea_stable_prior", type=float, default=0.7)
    ap.add_argument("--stea_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stea_quantile", type=float, default=1.0)
    ap.add_argument("--no_vis", action="store_true", help="Skip PNG overlays (npy + meta only)")
    ap.add_argument("--methods", type=str, default="gt,rgb,qnn,hire", help="Comma subset to run")
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


def _apply_input_gamma(rgb: torch.Tensor, gamma: float) -> torch.Tensor:
    g = float(gamma)
    if abs(g - 1.0) < 1e-6:
        return rgb
    return torch.clamp(rgb, 0.0, 1.0).pow(1.0 / g)


def _tensor_frame_to_bgr(frame_chw: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frame_chw, torch.Tensor):
        arr = frame_chw.detach().float().cpu().numpy()
    else:
        arr = np.array(frame_chw, copy=True)
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
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
    cube = raw_plane_to_photon_cube(
        raw_chunk_plane(raw_chunk, packed_nch=packed_nch), device=device, as_bool=True
    )
    recons, _confidence = preprocessor.process_photon_cube_to_frame(cube, clear_states=True)
    rgb = raw_hwt_to_rgb_float(recons.float(), packed_nch=int(packed_nch))
    if int(rgb.shape[0]) <= 0:
        raise ValueError("Frame preprocessor emitted zero frames for one chunk.")
    rgb = _apply_input_gamma(rgb[-1:].contiguous(), input_gamma).squeeze(0)
    return _tensor_frame_to_bgr(rgb)


def _frames_npy_to_bgr_list(frames: np.ndarray, *, src: Path) -> list[np.ndarray]:
    """Convert RGB ``frames.npy`` (NHWC or NCHW) to contiguous BGR uint8 list.

    Same layout rules as ``test_rgb_pose._frames_to_bgr_list``.
    """
    arr = np.asarray(frames)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D RGB frames in {src}, got shape={arr.shape}")

    # NHWC RGB (VisionSIM renders-rgb25fps*/frames.npy)
    if arr.shape[-1] == 3:
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames_bgr = arr[..., ::-1]
        return [np.ascontiguousarray(frames_bgr[i]) for i in range(frames_bgr.shape[0])]

    # NCHW RGB (offline sum/ema/... style caches)
    if arr.shape[1] == 3:
        out = []
        for i in range(arr.shape[0]):
            chw = arr[i]
            if np.issubdtype(chw.dtype, np.floating):
                rgb = np.clip(chw * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb = np.clip(chw, 0, 255).astype(np.uint8)
            bgr = np.transpose(rgb, (1, 2, 0))[..., ::-1]
            out.append(np.ascontiguousarray(bgr))
        return out

    raise ValueError(f"Unsupported frames layout {arr.shape} in {src}; expected NHWC or NCHW RGB")


def _load_rgb_frames(rgb_path: Path) -> list[np.ndarray]:
    """Load RGB frames as BGR uint8 list (OpenCV convention).

    Prefer ``frames.npy`` (dir or file) like ``det_rgb._load_rgb_frames`` /
    ``test_rgb_pose``; fall back to image directory or video.
    """
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)

    # Directory with frames.npy (VisionSIM rgb25fps layout)
    if rgb_path.is_dir():
        npy = rgb_path / "frames.npy"
        if npy.is_file():
            return _frames_npy_to_bgr_list(np.load(npy), src=npy)
        paths = sorted(
            p for p in rgb_path.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )
        if not paths:
            raise FileNotFoundError(
                f"Directory input requires frames.npy (or images), not found under {rgb_path}"
            )
        frames = []
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read {p}")
            frames.append(img)
        return frames

    # Direct .npy path
    if rgb_path.is_file() and rgb_path.suffix.lower() == ".npy":
        return _frames_npy_to_bgr_list(np.load(rgb_path), src=rgb_path)

    # Video file
    cap = cv2.VideoCapture(str(rgb_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video/path: {rgb_path}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {rgb_path}")
    return frames


def _load_gt_ann(gt_path: Path) -> dict[str, Any]:
    with gt_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)
    if not isinstance(ann, dict):
        raise ValueError(f"GT JSON must be a dict of frame_XXXXXX.png → hands, got {type(ann)}")
    return ann


def _gt_max_index(ann: dict[str, Any]) -> int:
    idxs = []
    for key in ann:
        m = _FRAME_RE.match(str(key))
        if m:
            idxs.append(int(m.group(1)))
    if not idxs:
        raise ValueError(f"No frame_XXXXXX.png keys in {ann.keys()!r}")
    return max(idxs)


def _hand_record(
    *,
    cls_id: int,
    score: float,
    bbox_xyxy: np.ndarray,
    keypoints: np.ndarray,
) -> dict[str, Any]:
    bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)
    kpts = np.asarray(keypoints, dtype=np.float32)
    if kpts.ndim != 2 or kpts.shape[0] != 21:
        raise ValueError(f"keypoints must be (21, 2|3), got {kpts.shape}")
    if kpts.shape[1] == 2:
        kpts = np.concatenate([kpts, np.ones((21, 1), dtype=np.float32)], axis=1)
    elif kpts.shape[1] != 3:
        raise ValueError(f"keypoints must be (21, 2|3), got {kpts.shape}")
    cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
    cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
    return {
        "cls": int(cls_id),
        "score": float(score),
        "bbox_xyxy": bbox.astype(np.float32),
        "bbox_center": np.asarray([cx, cy], dtype=np.float32),
        "keypoints": kpts.astype(np.float32),
    }


def _empty_frame(
    *,
    frame_idx: int,
    chunk_start_bin: int,
    chunk_end_bin: int,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    return {
        "frame_idx": int(frame_idx),
        "chunk_start_bin": int(chunk_start_bin),
        "chunk_end_bin": int(chunk_end_bin),
        "image_shape": (int(image_shape[0]), int(image_shape[1])),
        "hands": [],
    }


def _scale_hand_to_hw(hand: dict[str, Any], from_hw: tuple[int, int], to_hw: tuple[int, int]) -> dict[str, Any]:
    fh, fw = map(int, from_hw)
    th, tw = map(int, to_hw)
    if fh <= 0 or fw <= 0:
        raise ValueError(f"Invalid from_hw={from_hw}")
    if (fh, fw) == (th, tw):
        return {
            "cls": int(hand["cls"]),
            "score": float(hand["score"]),
            "bbox_xyxy": np.asarray(hand["bbox_xyxy"], dtype=np.float32).copy(),
            "bbox_center": np.asarray(hand["bbox_center"], dtype=np.float32).copy(),
            "keypoints": np.asarray(hand["keypoints"], dtype=np.float32).copy(),
        }
    sx = tw / float(fw)
    sy = th / float(fh)
    bbox = np.asarray(hand["bbox_xyxy"], dtype=np.float32).copy()
    bbox[0] *= sx
    bbox[2] *= sx
    bbox[1] *= sy
    bbox[3] *= sy
    kpts = np.asarray(hand["keypoints"], dtype=np.float32).copy()
    kpts[:, 0] *= sx
    kpts[:, 1] *= sy
    cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
    cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
    return {
        "cls": int(hand["cls"]),
        "score": float(hand["score"]),
        "bbox_xyxy": bbox,
        "bbox_center": np.asarray([cx, cy], dtype=np.float32),
        "keypoints": kpts,
    }


def _scale_frame_to_hw(frame: dict[str, Any], to_hw: tuple[int, int]) -> dict[str, Any]:
    from_hw = tuple(frame["image_shape"])
    hands = [_scale_hand_to_hw(h, from_hw, to_hw) for h in frame["hands"]]
    return {
        "frame_idx": int(frame["frame_idx"]),
        "chunk_start_bin": int(frame["chunk_start_bin"]),
        "chunk_end_bin": int(frame["chunk_end_bin"]),
        "image_shape": (int(to_hw[0]), int(to_hw[1])),
        "hands": hands,
    }


def _hands_from_result(result, *, conf_min: float = 0.0) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    kpts = None
    if getattr(result, "keypoints", None) is not None:
        kpts = result.keypoints.data.cpu().numpy()
    hands = []
    for i in range(len(boxes)):
        if float(scores[i]) < float(conf_min):
            continue
        if kpts is None or i >= len(kpts):
            continue
        hands.append(
            _hand_record(
                cls_id=int(classes[i]),
                score=float(scores[i]),
                bbox_xyxy=boxes[i],
                keypoints=kpts[i],
            )
        )
    return hands


def _gt_hands_at_index(
    ann: dict[str, Any],
    gt_idx: int,
    *,
    gt_image_size: int,
    target_hw: tuple[int, int],
) -> list[dict[str, Any]]:
    key = f"frame_{int(gt_idx):06d}.png"
    frame = ann.get(key, {})
    if not isinstance(frame, dict):
        return []
    from_hw = (int(gt_image_size), int(gt_image_size))
    hands = []
    for hand_name, cls_id in HAND_TO_CLASS.items():
        hand = frame.get(hand_name)
        if not isinstance(hand, dict) or "bbox" not in hand or "keypoints_2d" not in hand:
            continue
        bbox = np.asarray(hand["bbox"], dtype=np.float32).reshape(4)
        kpts = np.asarray(hand["keypoints_2d"], dtype=np.float32)
        rec = _hand_record(cls_id=cls_id, score=1.0, bbox_xyxy=bbox, keypoints=kpts)
        hands.append(_scale_hand_to_hw(rec, from_hw, target_hw))
    return hands


def _select_rgb_index(n_rgb: int, n_gt: int, frame_idx: int, gt_idx: int) -> int:
    """Map 25 fps frame to an RGB frame.

    Prefer 1:1 with 25 fps count; if RGB length matches GT@125Hz, use end-aligned gt_idx.
    """
    if n_rgb <= 0:
        raise ValueError("Empty RGB sequence")
    # Exact / near match to emitted 25 fps length handled by caller via clamp.
    # If RGB appears to be the 125 Hz GT timeline, index by gt_idx.
    if n_gt > 0 and n_rgb >= max(n_gt - 2, 1) and abs(n_rgb - n_gt) <= max(2, n_gt // 50):
        return int(np.clip(gt_idx, 0, n_rgb - 1))
    return int(np.clip(frame_idx, 0, n_rgb - 1))


def _cls_color_bgr(cls_id: int) -> tuple[int, int, int]:
    # left=blue-ish, right=orange (BGR)
    return (255, 128, 0) if int(cls_id) == 0 else (0, 128, 255)


def _draw_frame_overlay(
    rgb_bgr: np.ndarray,
    frame: dict[str, Any],
    *,
    kpt_thresh: float,
) -> np.ndarray:
    target_hw = tuple(frame["image_shape"])
    vis = _dsp()._resize_to_shape_bgr(rgb_bgr, target_hw)
    for hand in frame["hands"]:
        color = _cls_color_bgr(int(hand["cls"]))
        bbox = np.asarray(hand["bbox_xyxy"], dtype=np.float32)
        vis = _vhp()._draw_bbox_neon(vis, bbox, color)
        vis = _vhp()._draw_pose_id(
            vis,
            np.asarray(hand["keypoints"], dtype=np.float32),
            color=color,
            thresh=float(kpt_thresh),
        )
    return vis


def _save_method(
    save_dir: Path,
    method: str,
    frames: list[dict[str, Any]],
    rgb_frames: list[np.ndarray],
    *,
    n_gt: int,
    bins_per_gt: int,
    chunk_size: int,
    no_vis: bool,
    kpt_thresh: float,
) -> None:
    method_dir = save_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    # numpy object array for list-of-dicts
    np.save(method_dir / "poses.npy", np.asarray(frames, dtype=object), allow_pickle=True)

    if no_vis:
        return
    vis_dir = method_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for fr in frames:
        fi = int(fr["frame_idx"])
        gt_idx = int(round((fi + 1) * (chunk_size / float(bins_per_gt))))
        rgb_i = _select_rgb_index(len(rgb_frames), n_gt, fi, gt_idx)
        overlay = _draw_frame_overlay(rgb_frames[rgb_i], fr, kpt_thresh=kpt_thresh)
        cv2.imwrite(str(vis_dir / f"frame_{fi:07d}.png"), overlay)


def _pre_kwargs(name: str, *, subsampling: int, args: argparse.Namespace) -> dict[str, Any]:
    """Build preprocessor kwargs from CLI (shared by detector frame-pre and sequence override)."""
    name = str(name).strip().lower()
    kwargs: dict[str, Any] = {"subsampling": int(subsampling)}
    if name == "sum":
        return kwargs
    if name == "ema":
        kwargs.update(
            {
                "ema_alpha": float(args.ema_alpha),
                "normalize": bool(args.ema_normalize),
                "quantile": float(args.ema_quantile),
            }
        )
        return kwargs
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
        return kwargs
    if name == "stea":
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
        return kwargs
    if name == "hire":
        kwargs.update(
            {
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
                "normalize": bool(args.hire_normalize),
                "quantile": float(args.hire_quantile),
            }
        )
        return kwargs
    raise ValueError(f"Unsupported preprocessor name for kwargs: {name!r}")


def _pre_kwargs_for_meta(name: str, *, subsampling: int, args: argparse.Namespace) -> dict[str, Any] | None:
    name = str(name).strip().lower()
    if name in {"", "none", "?"}:
        return None
    return _pre_kwargs(name, subsampling=subsampling, args=args)


def _run_detector(
    *,
    ckpt: Path,
    source,
    device: torch.device,
    args: argparse.Namespace,
    chunk_size: int,
    t_begin: int,
    t_end: int,
    target_hw: tuple[int, int],
) -> list[dict[str, Any]]:
    yolo = YOLO(str(ckpt))
    yolo.model.to(device)
    yolo.model.eval()
    names = yolo.names
    pre_name = str(args.detector_pre)
    pre_kw = _pre_kwargs(pre_name, subsampling=chunk_size, args=args)
    print(f"  rgb/detector: pre={pre_name} kwargs={pre_kw}", flush=True)
    preprocessor = build_spad_frame_preprocessor(pre_name, kwargs=pre_kw).to(device)
    preprocessor.eval()

    frames_out: list[dict[str, Any]] = []
    frame_idx = 0
    for t0 in range(t_begin, t_end, chunk_size):
        t1 = min(t_end, t0 + chunk_size)
        if t1 <= t0:
            break
        raw_chunk = _dsp()._slice_raw_chunk(source, t0, t1, packed_ch_order=args.packed_ch_order)
        raw_chunk = _dsp()._prepare_raw_chunk_for_spad(raw_chunk, chunk_t=chunk_size, tail_pad_full=False)
        if raw_chunk is None or raw_chunk.shape[0] < chunk_size:
            # skip short tail to keep strict 25 fps alignment
            break
        with torch.inference_mode():
            frame_bgr = _render_raw_chunk_to_bgr(
                preprocessor=preprocessor,
                raw_chunk=raw_chunk,
                packed_nch=int(source.packed_nch),
                input_gamma=float(args.input_gamma),
                device=device,
            )
            results = yolo.predict(
                frame_bgr,
                conf=float(args.conf),
                iou=float(args.iou),
                max_det=int(args.max_det),
                imgsz=int(args.imgsz),
                device=str(device),
                verbose=False,
            )
        result = results[0]
        native_hw = frame_bgr.shape[:2]
        fr = _empty_frame(
            frame_idx=frame_idx,
            chunk_start_bin=t0,
            chunk_end_bin=t1,
            image_shape=native_hw,
        )
        fr["hands"] = _hands_from_result(result, conf_min=0.0)
        frames_out.append(_scale_frame_to_hw(fr, target_hw))
        frame_idx += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _ = names  # kept for future class-name logging
    return frames_out


def _warn_if_not_ssd(spad_model, ckpt: Path) -> None:
    plugin = getattr(spad_model, "spad_plugin", None) or getattr(spad_model, "plugin", None)
    train_args = getattr(spad_model, "args", None)
    if plugin is None and train_args is not None:
        plugin = _dsp()._cfg_get(train_args, "spad_plugin", None)
    if plugin is None:
        # inspect plugins_by_layer class names
        plugins = getattr(spad_model, "plugins_by_layer", None) or {}
        kinds = {type(p).__name__.lower() for p in plugins.values()}
        if any("ssd" in k for k in kinds):
            return
        print(f"Warning: could not confirm temporal_ssd plugin on {ckpt}", flush=True)
        return
    if str(plugin).lower() not in {"temporal_ssd", "ssd"}:
        print(f"Warning: expected temporal_ssd plugin, got {plugin!r} on {ckpt}", flush=True)


def _run_sequence(
    *,
    ckpt: Path,
    source,
    device: torch.device,
    args: argparse.Namespace,
    chunk_size: int,
    t_begin: int,
    t_end: int,
    target_hw: tuple[int, int],
    label: str,
) -> list[dict[str, Any]]:
    yolo = YOLO(str(ckpt))
    spad_model = yolo.model
    if not getattr(spad_model, "spad_enabled", False) or not hasattr(spad_model, "preprocessor"):
        raise TypeError(
            f"{label} checkpoint {ckpt} is not a SpadPoseModel "
            f"(got {spad_model.__class__.__name__})"
        )
    _warn_if_not_ssd(spad_model, ckpt)
    spad_model.to(device)
    spad_model.eval()
    _dsp()._configure_model_spad_bin_rate(spad_model, current_bin_rate_hz=float(args.spad_bin_rate_hz))
    spad_model.spad_detect_imgsz = int(args.imgsz)
    if hasattr(spad_model, "spad_cache_mode"):
        spad_model.spad_cache_mode = "raw"

    override = "none"
    if label == "qnn":
        override = str(args.qnn_pre_override).strip().lower()
    elif label == "hire":
        override = str(args.hire_pre_override).strip().lower()
    if override not in {"", "none"}:
        pre_kw = _pre_kwargs(override, subsampling=chunk_size, args=args)
        spad_model.preprocessor = build_spad_preprocessor(override, kwargs=pre_kw).to(device)
        spad_model.preprocessor_name = override
        print(f"  {label}: preprocessor override → {override} kwargs={pre_kw}", flush=True)
    else:
        print(
            f"  {label}: pre={getattr(spad_model, 'preprocessor_name', '?')} "
            f"(ckpt) cache_mode=raw chunk={chunk_size}",
            flush=True,
        )

    names = yolo.names
    kpt_shape = getattr(spad_model, "kpt_shape", (21, 3))

    frames_out: list[dict[str, Any]] = []
    frame_idx = 0
    if hasattr(spad_model, "spad_begin_stream"):
        spad_model.spad_begin_stream()
    else:
        spad_model.spad_set_online_inference(True)
        spad_model.spad_clear_plugin_states()

    try:
        for t0 in range(t_begin, t_end, chunk_size):
            t1 = min(t_end, t0 + chunk_size)
            if t1 <= t0:
                break
            raw_chunk = _dsp()._slice_raw_chunk(source, t0, t1, packed_ch_order=args.packed_ch_order)
            raw_chunk = _dsp()._prepare_raw_chunk_for_spad(raw_chunk, chunk_t=chunk_size, tail_pad_full=False)
            if raw_chunk is None or raw_chunk.shape[0] < chunk_size:
                break

            spad_model.spad_packed_nch = int(source.packed_nch)
            with torch.inference_mode():
                video_tensor = torch.from_numpy(np.ascontiguousarray(raw_chunk)).unsqueeze(0).to(device)
                if getattr(spad_model, "spad_stream_mode", False):
                    spad_model.spad_stream_bin_offset = int(t0)
                    spad_model.spad_pending_t_index_ll = [int(t1)]
                raw_preds = spad_model(video_tensor)
                preds = _dsp()._postprocess_pose_predictions(
                    raw_preds,
                    conf=float(args.conf),
                    iou=float(args.iou),
                    nc=len(names),
                    max_det=int(args.max_det),
                    kpt_shape=kpt_shape,
                )
                preds = _dsp()._scale_pose_preds_to_native(
                    preds,
                    scale_meta=getattr(spad_model, "spad_scale_meta", None),
                    kpt_shape=kpt_shape,
                )
                recon_frames_bgr = _dsp()._recon_frames_bgr(spad_model, batch_index=0)

            results = _dsp()._results_from_preds(
                preds,
                recon_frames_bgr,
                names,
                prefix=f"{label}_t{t0:06d}_{t1:06d}",
                kpt_shape=kpt_shape,
            )
            result = results[-1] if results else None
            if result is not None and getattr(result, "orig_img", None) is not None:
                native_hw = result.orig_img.shape[:2]
            elif recon_frames_bgr:
                native_hw = recon_frames_bgr[-1].shape[:2]
            else:
                native_hw = target_hw

            fr = _empty_frame(
                frame_idx=frame_idx,
                chunk_start_bin=t0,
                chunk_end_bin=t1,
                image_shape=native_hw,
            )
            if result is not None:
                fr["hands"] = _hands_from_result(result, conf_min=0.0)
            frames_out.append(_scale_frame_to_hw(fr, target_hw))
            frame_idx += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if hasattr(spad_model, "spad_end_stream"):
            spad_model.spad_end_stream()

    return frames_out


def _build_gt_frames(
    *,
    ann: dict[str, Any],
    n_frames: int,
    chunk_size: int,
    bins_per_gt: int,
    t_begin: int,
    gt_image_size: int,
    target_hw: tuple[int, int],
) -> list[dict[str, Any]]:
    frames = []
    for i in range(n_frames):
        t0 = int(t_begin + i * chunk_size)
        t1 = int(t0 + chunk_size)
        # end-aligned: pred i ↔ GT time (i+1) * (chunk_size / bins_per_gt)
        gt_idx = int(round((i + 1) * (chunk_size / float(bins_per_gt))))
        fr = _empty_frame(
            frame_idx=i,
            chunk_start_bin=t0,
            chunk_end_bin=t1,
            image_shape=target_hw,
        )
        fr["hands"] = _gt_hands_at_index(
            ann, gt_idx, gt_image_size=gt_image_size, target_hw=target_hw
        )
        frames.append(fr)
    return frames


def main() -> None:
    args = _parse_args()
    methods = {m.strip().lower() for m in str(args.methods).split(",") if m.strip()}
    allowed = {"gt", "rgb", "qnn", "hire"}
    unknown = methods - allowed
    if unknown:
        raise ValueError(f"Unknown methods {sorted(unknown)}; expected subset of {sorted(allowed)}")
    if not methods:
        raise ValueError("No methods selected")

    chunk_size = int(args.chunk_size)
    bins_per_gt = int(args.spad_bins_per_gt)
    if chunk_size <= 0 or bins_per_gt <= 0:
        raise ValueError("chunk_size and spad_bins_per_gt must be positive")

    npy_path = _resolve_in_npy(args.in_path)
    rgb_frames = _load_rgb_frames(args.rgb_path)
    ann = _load_gt_ann(args.gt_path)
    n_gt = _gt_max_index(ann) + 1
    target_hw = rgb_frames[0].shape[:2]

    device = _dsp()._resolve_device(args.device)
    sources = list(_dsp()._iter_raw_video_sources_from_sample_path(npy_path))
    if not sources:
        raise RuntimeError(f"No SPAD sources in {npy_path}")
    source = sources[0]
    if len(sources) > 1:
        print(f"Warning: multiple SPAD sources found; using the first ({len(sources)} total)")

    total_bins = _dsp()._video_num_bins(source)
    t_begin = max(0, int(args.start_bin))
    t_end = total_bins if int(args.end_bin) <= 0 else min(total_bins, int(args.end_bin))
    if t_begin >= t_end:
        raise ValueError(f"Empty bin range [{t_begin}, {t_end})")

    n_chunks = (t_end - t_begin) // chunk_size
    print(
        f"in={npy_path} bins=[{t_begin},{t_end}) chunk={chunk_size} → {n_chunks} frames @ "
        f"{float(args.spad_bin_rate_hz) / chunk_size:.3g} fps | rgb={target_hw} gt_frames~{n_gt} "
        f"device={device} methods={sorted(methods)}",
        flush=True,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Infer n_frames from a prediction pass when possible; GT alone needs n_chunks.
    n_frames = n_chunks
    results: dict[str, list[dict[str, Any]]] = {}

    if "rgb" in methods:
        print("Running pure detector (rgb)…", flush=True)
        results["rgb"] = _run_detector(
            ckpt=args.detector_ckpt,
            source=source,
            device=device,
            args=args,
            chunk_size=chunk_size,
            t_begin=t_begin,
            t_end=t_end,
            target_hw=target_hw,
        )
        n_frames = min(n_frames, len(results["rgb"]))

    if "qnn" in methods:
        print("Running QNN sequence (raw)…", flush=True)
        results["qnn"] = _run_sequence(
            ckpt=args.qnn_ckpt,
            source=source,
            device=device,
            args=args,
            chunk_size=chunk_size,
            t_begin=t_begin,
            t_end=t_end,
            target_hw=target_hw,
            label="qnn",
        )
        n_frames = min(n_frames, len(results["qnn"]))

    if "hire" in methods:
        print("Running HIRE sequence (raw)…", flush=True)
        results["hire"] = _run_sequence(
            ckpt=args.hire_ckpt,
            source=source,
            device=device,
            args=args,
            chunk_size=chunk_size,
            t_begin=t_begin,
            t_end=t_end,
            target_hw=target_hw,
            label="hire",
        )
        n_frames = min(n_frames, len(results["hire"]))

    if n_frames <= 0:
        raise RuntimeError("No 25 fps frames produced; check bin range / chunk_size")

    # Truncate all model outputs to common length
    for key in list(results):
        results[key] = results[key][:n_frames]
        for i, fr in enumerate(results[key]):
            fr["frame_idx"] = i

    if "gt" in methods:
        print("Building end-aligned GT @ 25 fps…", flush=True)
        results["gt"] = _build_gt_frames(
            ann=ann,
            n_frames=n_frames,
            chunk_size=chunk_size,
            bins_per_gt=bins_per_gt,
            t_begin=t_begin,
            gt_image_size=int(args.gt_image_size),
            target_hw=target_hw,
        )

    meta = {
        "format": "spadhand_main_pose_v1",
        "in_path": str(npy_path),
        "rgb_path": str(args.rgb_path),
        "gt_path": str(args.gt_path),
        "detector_ckpt": str(args.detector_ckpt),
        "qnn_ckpt": str(args.qnn_ckpt),
        "hire_ckpt": str(args.hire_ckpt),
        "detector_pre": str(args.detector_pre),
        "detector_pre_kwargs": _pre_kwargs_for_meta(
            str(args.detector_pre), subsampling=chunk_size, args=args
        ),
        "qnn_pre_override": str(args.qnn_pre_override),
        "qnn_pre_kwargs": _pre_kwargs_for_meta(
            str(args.qnn_pre_override), subsampling=chunk_size, args=args
        ),
        "hire_pre_override": str(args.hire_pre_override),
        "hire_pre_kwargs": _pre_kwargs_for_meta(
            str(args.hire_pre_override), subsampling=chunk_size, args=args
        ),
        "chunk_size": int(chunk_size),
        "spad_bin_rate_hz": float(args.spad_bin_rate_hz),
        "spad_bins_per_gt": int(bins_per_gt),
        "frame_rate": float(args.spad_bin_rate_hz) / float(chunk_size),
        "start_bin": int(t_begin),
        "end_bin": int(t_end),
        "n_frames": int(n_frames),
        "image_shape": [int(target_hw[0]), int(target_hw[1])],
        "gt_image_size": int(args.gt_image_size),
        "gt_align": "end",
        "gt_index_formula": "(frame_idx + 1) * (chunk_size / spad_bins_per_gt)",
        "cache_mode": "raw",
        "methods": sorted(results.keys()),
        "names": {0: "left_hand", 1: "right_hand"},
    }
    with (args.save_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for method, frames in results.items():
        print(f"Saving {method}: {len(frames)} frames → {args.save_dir / method}", flush=True)
        _save_method(
            args.save_dir,
            method,
            frames,
            rgb_frames,
            n_gt=n_gt,
            bins_per_gt=bins_per_gt,
            chunk_size=chunk_size,
            no_vis=bool(args.no_vis),
            kpt_thresh=float(args.kpt_thresh),
        )

    print(f"Done. n_frames={n_frames} → {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
