# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""det_qnns.py

Run inference with the trained QNN pose checkpoint on SPAD raw clips.

Unlike the older SPAD predictor workflow, this script does not reconstruct frames in the
predictor. Instead it feeds raw SPAD clips directly into the checkpoint's `QNNPoseModel`, which
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
    * (T, H, Wpacked, 3) VisionSIM packed frames (expanded to raw RGGB on load)

The checkpoint reconstructs `T'` RGB-like frames internally. This script runs the model,
postprocesses detections, applies optional ByteTrack/BoT-SORT tracking, and saves per-frame
visualizations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace, YAML, nms
from ultralytics.utils.checks import check_yaml


def _np_load(path: Path) -> np.ndarray:
    """Load .npy with memory mapping (keeps most data on disk)."""
    return np.load(path, mmap_mode="r")


# ----------------------------
# Visualization (copied from det.py)
# ----------------------------
BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Mid
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]
FINGER_COLORS = [
    (0, 0, 255),  # Thumb - Red
    (255, 0, 0),  # Index - Blue
    (0, 255, 0),  # Mid - Green
    (0, 255, 255),  # Ring - Yellow
    (255, 0, 255),  # Pinky - magenta
]
COLOR_KEYPOINT = (255, 255, 255)  # Joint - White
COLOR_WRIST = (255, 165, 0)  # Wrist - Orange


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
    # pose_kpts: (K,3) with (x,y,conf)
    if pose_kpts.shape != (k, 3):
        raise ValueError(f"Pose shape must be ({k}, 3), but got {pose_kpts.shape}")

    for i, (s, e) in enumerate(BONE_CONNECTIONS):
        ks = pose_kpts[s]
        ke = pose_kpts[e]
        if ks[2] > thresh and ke[2] > thresh:
            cv2.line(
                img_bgr,
                (int(ks[0]), int(ks[1])),
                (int(ke[0]), int(ke[1])),
                _get_finger_color(i),
                3,
            )

    for i in range(k):
        kk = pose_kpts[i]
        if kk[2] > thresh:
            center = (int(kk[0]), int(kk[1]))
            if i == 0:
                color, radius = COLOR_WRIST, 6
            else:
                color, radius = COLOR_KEYPOINT, 4
            cv2.circle(img_bgr, center, radius, color, -1)

    return img_bgr


# ----------------------------
# Data loading utilities
# ----------------------------

def _packed_frames_to_raw_video(
    frames_packed: np.ndarray, *, expected_w: int = 512, ch_order: str = "RGB"
) -> np.ndarray:
    """Convert VisionSIM packed frames `(T,H,Wpacked,3)` to raw `(T,2H,2W,1)`."""
    if frames_packed.ndim != 4 or frames_packed.shape[-1] != 3:
        raise ValueError(f"Expected packed frames (T,H,Wpacked,3), got shape={frames_packed.shape}")

    unpacked = np.unpackbits(frames_packed, axis=2)
    if unpacked.shape[2] > expected_w:
        unpacked = unpacked[:, :, :expected_w, :]

    if ch_order.upper() == "BGR":
        r_ch, g_ch, b_ch = 2, 1, 0
    else:
        r_ch, g_ch, b_ch = 0, 1, 2

    t, h, w, _ = unpacked.shape
    raw = np.zeros((t, h * 2, w * 2), dtype=np.uint8)
    raw[:, 0::2, 0::2] = unpacked[:, :, :, r_ch]
    raw[:, 0::2, 1::2] = unpacked[:, :, :, g_ch]
    raw[:, 1::2, 0::2] = unpacked[:, :, :, g_ch]
    raw[:, 1::2, 1::2] = unpacked[:, :, :, b_ch]
    return raw[:, :, :, None]


def _is_packed_spad(arr: np.ndarray) -> bool:
    return arr.ndim == 4 and arr.shape[-1] == 3 and arr.shape[2] in (64, 128)


def _looks_like_hwt(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[0] == arr.shape[1] and arr.shape[2] != arr.shape[1]


@dataclass(frozen=True)
class RawVideoSource:
    """One lazily-sliced video source."""

    array: np.ndarray
    layout: str


def _video_sources_from_array(arr: np.ndarray) -> list[RawVideoSource]:
    """Describe supported layouts without materializing the full raw video."""
    if _is_packed_spad(arr):
        return [RawVideoSource(arr, "packed")]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return [RawVideoSource(arr, "thwc1")]
    if arr.ndim == 3:
        return [RawVideoSource(arr, "hwt" if _looks_like_hwt(arr) else "thw")]
    if arr.ndim == 4:
        return [RawVideoSource(arr[i], "hwt") for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported input shape: {arr.shape}")


def _iter_raw_video_sources_from_sample_path(in_path: Path):
    """Yield lazy video sources from a sample path."""
    if in_path.is_dir():
        files = sorted([p for p in in_path.iterdir() if p.suffix.lower() == ".npy"])
        for p in files:
            yield from _video_sources_from_array(_np_load(p))
        return

    if in_path.suffix.lower() == ".npy":
        yield from _video_sources_from_array(_np_load(in_path))
        return

    raise ValueError(f"Unsupported input path: {in_path}")


def _video_num_bins(source: RawVideoSource) -> int:
    """Return the raw time length of a source."""
    if source.layout in {"packed", "thwc1", "thw"}:
        return int(source.array.shape[0])
    if source.layout == "hwt":
        return int(source.array.shape[2])
    raise ValueError(f"Unsupported source layout: {source.layout}")


def _slice_raw_chunk(source: RawVideoSource, t0: int, t1: int, *, packed_ch_order: str) -> np.ndarray:
    """Load one raw `(T,H,W,1)` chunk from a lazy source."""
    if source.layout == "packed":
        packed = np.asarray(source.array[t0:t1])
        return _packed_frames_to_raw_video(packed, ch_order=packed_ch_order)
    if source.layout == "thwc1":
        return np.ascontiguousarray(source.array[t0:t1].astype(np.uint8, copy=False))
    if source.layout == "thw":
        return np.ascontiguousarray(source.array[t0:t1, :, :, None].astype(np.uint8, copy=False))
    if source.layout == "hwt":
        return np.ascontiguousarray(np.transpose(source.array[:, :, t0:t1], (2, 0, 1))[:, :, :, None].astype(np.uint8, copy=False))
    raise ValueError(f"Unsupported source layout: {source.layout}")


def _maybe_zero_pad_raw_chunk(raw_chunk: np.ndarray, target_t: int, *, enabled: bool) -> np.ndarray:
    """Optionally zero-pad the tail chunk to the target time length."""
    if not enabled or raw_chunk.shape[0] >= target_t:
        return raw_chunk
    pad_t = int(target_t) - int(raw_chunk.shape[0])
    if pad_t <= 0:
        return raw_chunk
    pad_shape = (pad_t, raw_chunk.shape[1], raw_chunk.shape[2], raw_chunk.shape[3])
    pad = np.zeros(pad_shape, dtype=raw_chunk.dtype)
    return np.ascontiguousarray(np.concatenate((raw_chunk, pad), axis=0))


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
    train_args = getattr(model, "args", None)
    output_frames = _cfg_get(train_args, "qnn_output_frames", None)
    subsampling = getattr(getattr(model, "integrator", None), "subsampling", None)
    if output_frames is None or subsampling is None:
        return None
    return int(output_frames) * int(subsampling)


def _recon_frames_bgr(model, batch_index: int = 0) -> list[np.ndarray]:
    frames = getattr(model, "qnn_last_recon_frames", None)
    if frames is None:
        return []
    rgb = frames[:, batch_index].detach().float().cpu().permute(0, 2, 3, 1).numpy()
    bgr = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)[:, :, :, ::-1]
    return [np.ascontiguousarray(frame) for frame in bgr]


def _raw_sum_bgr(raw_video: np.ndarray) -> np.ndarray:
    raw_sum = raw_video[..., 0].astype(np.float32).sum(axis=0)
    r = raw_sum[0::2, 0::2]
    g = 0.5 * (raw_sum[0::2, 1::2] + raw_sum[1::2, 0::2])
    b = raw_sum[1::2, 1::2]
    rgb = np.stack((r, g, b), axis=2)
    rgb /= rgb.max() + 1e-6
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb_u8[:, :, ::-1])


def _postprocess_pose_predictions(raw_preds, *, conf: float, iou: float, nc: int, max_det: int, kpt_shape) -> list[torch.Tensor]:
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


def _init_tracker(tracker_name: str, *, frame_rate: int):
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(f"{tracker_name}.yaml")))
    return TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=frame_rate)


def _apply_tracker(result: Results, tracker) -> Results:
    det = result.boxes.cpu().numpy()
    tracks = tracker.update(det, result.orig_img, getattr(result, "feats", None))
    if len(tracks) == 0:
        return result
    idx = tracks[:, -1].astype(int)
    tracked = result[idx]
    tracked.update(boxes=torch.as_tensor(tracks[:, :-1], device=result.boxes.data.device))
    return tracked


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Raw SPAD clip -> trained QNNPoseModel -> pose tracking")
    ap.add_argument("--in_path", type=str, required=True, help="Sample/video folder, root folder, or a .npy file")
    ap.add_argument("--in_glob", type=str, default=None,
                    help="Optional glob (e.g. '*') when --in_path is a root folder containing many sample/video subfolders")
    ap.add_argument("--save_dir", type=str, required=True, help="Directory to save visualized frames")
    ap.add_argument("--ckpt", type=str, default="weights/detector.pt")
    ap.add_argument("--device", type=str, default="", help="Torch device, e.g. cpu / cuda:0 / mps")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--tracker", type=str, default="botsort", choices=["bytetrack", "botsort"])
    ap.add_argument("--frame_rate", type=int, default=30, help="Tracker frame-rate hint")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])

    # Raw clip chunking
    ap.add_argument(
        "--cube_chunk_t",
        type=int,
        default=0,
        help="If >0, split each raw video into chunks of this many bins. If 0, use the train-time QNN window length when available, else full video.",
    )
    ap.add_argument("--cube_chunk_stride", type=int, default=0, help="Stride for chunking; default uses cube_chunk_t (no overlap)")
    ap.add_argument(
        "--tail_pad_zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero-pad the final short chunk up to `cube_chunk_t` before inference.",
    )
    ap.add_argument("--vis_bg", type=str, default="recon", choices=["sum", "recon"], help="Visualization background")

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
    qnn_model = yolo.model
    if not getattr(qnn_model, "qnn_enabled", False) or not hasattr(qnn_model, "integrator"):
        raise TypeError(
            f"Checkpoint {args.ckpt} is not a trained QNN pose model. "
            f"Loaded type: {qnn_model.__class__.__name__}"
        )

    device = _resolve_device(args.device)
    qnn_model.to(device)
    qnn_model.eval()
    tracker = _init_tracker(args.tracker, frame_rate=args.frame_rate)
    names = yolo.names
    kpt_shape = getattr(qnn_model, "kpt_shape", (21, 3))
    trained_chunk_t = _trained_chunk_t(qnn_model)
    global_frame_idx = 0

    for sample_path in sample_paths:
        sample_name = sample_path.name if sample_path.is_dir() else sample_path.stem
        out_dir = save_dir / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)

        tracker.reset()
        video_iter = _iter_raw_video_sources_from_sample_path(sample_path)

        for video_idx, source in enumerate(tqdm(video_iter, desc=f"Processing video [{sample_name}]")):
            T = _video_num_bins(source)
            if int(args.cube_chunk_t) > 0:
                chunk_t = int(args.cube_chunk_t)
            elif trained_chunk_t is not None:
                chunk_t = min(int(trained_chunk_t), T)
            else:
                raise ValueError(
                    "Unable to infer train-time QNN window length from checkpoint. "
                    "Pass --cube_chunk_t explicitly, e.g. --cube_chunk_t 64 for qnn_output_frames=1,qnn_subsampling=64. "
                    f"Refusing to process the full video as one chunk (T={T}), which is likely to OOM."
                )
            stride = int(args.cube_chunk_stride) if int(args.cube_chunk_stride) > 0 else chunk_t

            for t0 in range(0, T, stride):
                t1 = min(T, t0 + chunk_t)
                raw_chunk = _slice_raw_chunk(source, t0, t1, packed_ch_order=args.packed_ch_order)
                raw_chunk = _maybe_zero_pad_raw_chunk(raw_chunk, chunk_t, enabled=bool(args.tail_pad_zero))
                if raw_chunk.shape[0] == 0:
                    continue

                with torch.inference_mode():
                    video_tensor = torch.from_numpy(raw_chunk).unsqueeze(0).to(device)
                    raw_preds = qnn_model(video_tensor)
                    preds = _postprocess_pose_predictions(
                        raw_preds,
                        conf=args.det_thresh,
                        iou=args.iou,
                        nc=len(names),
                        max_det=args.max_det,
                        kpt_shape=kpt_shape,
                    )
                    recon_frames_bgr = _recon_frames_bgr(qnn_model, batch_index=0)
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
                    bg_bgr = _raw_sum_bgr(raw_chunk)

                for i, r in enumerate(results):
                    r = _apply_tracker(r, tracker)
                    if args.vis_bg == "recon" and getattr(r, "orig_img", None) is not None:
                        vis = np.ascontiguousarray(r.orig_img.copy())
                    else:
                        vis = bg_bgr.copy() if bg_bgr is not None else np.zeros_like(r.orig_img)

                    if r.boxes is not None and len(r.boxes):
                        track_ids = r.boxes.id
                        if track_ids is None:
                            track_ids = torch.arange(len(r.boxes), device=r.boxes.data.device)
                        track_id = track_ids.cpu().numpy()
                        boxes = r.boxes.xyxy.cpu().numpy()
                        box_confs = r.boxes.conf.cpu().numpy()
                        handedness = r.boxes.cls.cpu().numpy()

                        poses = None
                        if getattr(r, "keypoints", None) is not None:
                            poses = r.keypoints.data.cpu().numpy()

                        for j, tid in enumerate(track_id):
                            box_xyxyc = np.concatenate([boxes[j], [box_confs[j]]], axis=0)
                            vis = draw_bbox(vis, int(tid), box_xyxyc, float(handedness[j]))
                            if poses is not None and j < len(poses):
                                vis = draw_pose(vis, poses[j])

                    out_path = out_dir / f"cube{video_idx:05d}_t{t0:06d}_{t1:06d}_frame{global_frame_idx:07d}.png"
                    cv2.imwrite(str(out_path), vis)
                    global_frame_idx += 1


if __name__ == "__main__":
    main()