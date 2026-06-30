# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run SPAD pose inference on a test set and save structured predictions.

This is the non-visualization sibling of ``det_spad_pose.py``. It feeds raw SPAD clips
into a trained SPAD pose checkpoint, applies the same NMS and optional tracker, and
writes one JSON file per input sample.
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
from ultralytics.data.spad_pose_dataset import load_visionsim_split_json

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


def _eval_dir(model_name: str, test_name: str) -> Path:
    return Path("Evals") / model_name / test_name


def _vis_dir(model_name: str, test_name: str) -> Path:
    return Path("Vis") / model_name / test_name


def _names_to_dict(names) -> dict[str, str]:
    if isinstance(names, dict):
        return {str(k): str(v) for k, v in names.items()}
    return {str(i): str(v) for i, v in enumerate(names)}


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


def parse_args():
    ap = argparse.ArgumentParser(description="SPAD test set -> trained SpadPoseModel -> JSON predictions and optional visualizations")
    ap.add_argument("--in_path", type=str, default=None, help="Sample/video folder, root folder, or a .npy file")
    ap.add_argument("--test_json", type=str, default=None, help="VisionSIM test split JSON; when set, samples are read from its `spad` paths.")
    ap.add_argument(
        "--in_glob",
        type=str,
        default=None,
        help="Optional glob when --in_path is a root folder containing many sample/video subfolders",
    )
    ap.add_argument("--test_name", type=str, default=None, help="Output test-set folder name; defaults from --in_path")
    ap.add_argument("--ckpt", type=str, required=True, help="Trained SPAD pose checkpoint")
    ap.add_argument("--device", type=str, default="", help="Torch device, e.g. cpu / cuda:0 / mps")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--tracker", type=str, default="botsort", choices=["bytetrack", "botsort", "spad_tracker"])
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
        choices=["none", "ppb", "sum", "stea", "hyb"],
        help="Optionally override the checkpoint's internal SPAD preprocessor at inference time.",
    )
    ap.add_argument("--spad-subsampling", type=int, default=320, help="Temporal subsampling used by override preprocessors.")
    ap.add_argument("--ppb-bocpd-gamma", type=float, default=5e-4)
    ap.add_argument("--ppb-quantile", type=float, default=1.0)
    ap.add_argument("--ppb-normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppb-min-filter-size", type=int, default=7)
    ap.add_argument("--stea-fast-window", type=int, default=16)
    ap.add_argument("--stea-slow-window", type=int, default=128)
    ap.add_argument("--stea-temporal-window", type=int, default=5)
    ap.add_argument("--stea-fast-tau", type=float, default=6.0)
    ap.add_argument("--stea-motion-sharpness", type=float, default=60.0)
    ap.add_argument("--stea-motion-threshold", type=float, default=0.05)
    ap.add_argument("--stea-stable-prior", type=float, default=16.0)
    ap.add_argument("--stea-normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stea-quantile", type=float, default=1.0)
    ap.add_argument("--hyb-motion-sharpness", type=float, default=60.0)
    ap.add_argument("--hyb-motion-threshold", type=float, default=0.05)
    ap.add_argument("--hyb-warp-block-size", type=int, default=16)
    ap.add_argument("--hyb-source-space", type=str, default="rgb", choices=["rgb", "raw"])
    ap.add_argument(
        "--cube_chunk_t",
        type=int,
        default=0,
        help="If >0, split each raw video into chunks of this many bins. If 0, use the train-time SPAD window length.",
    )
    ap.add_argument("--cube_chunk_stride", type=int, default=0, help="Stride for chunking; default uses cube_chunk_t")
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
    ap.add_argument(
        "--vis",
        nargs="?",
        const="video",
        default="none",
        choices=["none", "image", "video"],
        help="Optional visualization output. `--vis` defaults to video; use `--vis image` for PNG frames.",
    )
    ap.add_argument("--vis_bg", type=str, default="recon", choices=["sum", "recon"], help="Visualization background")
    ap.add_argument(
        "--save_readrgb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save the direct read_rgb-style sum visualization for image-mode visualization.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt)
    model_name = _model_name_from_ckpt(ckpt)

    if args.test_json:
        test_json = Path(args.test_json)
        if not test_json.is_file():
            raise FileNotFoundError(f"Test split JSON not found: {test_json}")
        test_name = args.test_name or _default_test_name_from_json(test_json)
        sample_specs = _sample_specs_from_json(test_json)
    else:
        if not args.in_path:
            raise ValueError("Either --test_json or --in_path must be provided.")
        in_path = Path(args.in_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Input path not found: {in_path}")
        test_name = args.test_name or _default_test_name(in_path, args.in_glob)
        sample_specs = [{"sample_name": p.name if p.is_dir() else p.stem, "sample_path": p, "record": None} for p in _sample_paths_from_args(in_path, args.in_glob)]

    out_dir = _eval_dir(model_name, test_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_root = _vis_dir(model_name, test_name) if args.vis != "none" else None
    if vis_root is not None:
        vis_root.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(ckpt))
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

    tracker = _init_tracker(args.tracker, frame_rate=args.frame_rate)
    override_name, override_preprocessor = _build_override_preprocessor(args)
    if override_preprocessor is not None:
        if override_name == "hyb" and args.tracker != "spad_tracker":
            raise ValueError("--preprocessor-override hyb requires --tracker spad_tracker")
        spad_model.preprocessor = override_preprocessor.to(device)
        spad_model.preprocessor_name = override_name
    names = yolo.names
    kpt_shape = getattr(spad_model, "kpt_shape", (21, 3))
    trained_chunk_t = _trained_chunk_t(spad_model)

    for sample_spec in sample_specs:
        sample_name = str(sample_spec["sample_name"])
        sample_path = Path(sample_spec["sample_path"])
        out_path = out_dir / f"{sample_name}.json"
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing file: {out_path}")
            continue

        tracker.reset()
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
            },
            "source": {
                "path": str(sample_path),
                "is_dir": bool(sample_path.is_dir()),
            },
            "videos": [],
        }
        if sample_spec["record"] is not None:
            sample_record["source"]["split_record"] = sample_spec["record"]

        video_iter = _iter_raw_video_sources_from_sample_path(sample_path)
        for video_idx, source in enumerate(tqdm(video_iter, desc=f"Testing [{sample_name}]")):
            total_bins = _video_num_bins(source)
            if int(args.cube_chunk_t) > 0:
                chunk_t = int(args.cube_chunk_t)
            elif trained_chunk_t is not None:
                chunk_t = int(trained_chunk_t)
            else:
                raise ValueError(
                    "Unable to infer train-time SPAD window length from checkpoint. "
                    "Pass --cube_chunk_t explicitly."
                )
            if chunk_t <= 0:
                raise ValueError(f"cube_chunk_t must be positive, got {chunk_t}")

            subsampling = int(getattr(getattr(spad_model, "preprocessor", None), "subsampling", 1) or 1)
            if chunk_t < subsampling:
                raise ValueError(
                    f"cube_chunk_t={chunk_t} is shorter than preprocessor subsampling={subsampling}, "
                    "which would produce zero reconstructed frames."
                )
            stride = int(args.cube_chunk_stride) if int(args.cube_chunk_stride) > 0 else chunk_t

            video_record: dict[str, Any] = {
                "video_idx": int(video_idx),
                "layout": source.layout,
                "packed_nch": int(source.packed_nch),
                "total_bins": int(total_bins),
                "chunk_t": int(chunk_t),
                "chunk_stride": int(stride),
                "subsampling": int(subsampling),
                "chunks": [],
            }

            for chunk_idx, t0 in enumerate(range(0, total_bins, stride)):
                t1 = min(total_bins, t0 + chunk_t)
                if (t1 - t0) < chunk_t:
                    continue
                raw_chunk = _slice_raw_chunk(source, t0, t1, packed_ch_order=args.packed_ch_order)
                raw_chunk = _prepare_raw_chunk_for_spad(raw_chunk, chunk_t=chunk_t, tail_pad_full=False)
                if raw_chunk is None:
                    continue

                spad_model.spad_packed_nch = int(source.packed_nch)
                _set_velocity_field_on_preprocessor(spad_model.preprocessor, tracker)

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
                    "chunk_idx": int(chunk_idx),
                    "start_bin": int(t0),
                    "end_bin": int(t1),
                    "input_bins": int(t1 - t0),
                    "model_input_bins": int(raw_chunk.shape[0]),
                    "frames": [],
                }

                for output_frame_idx, result in enumerate(results):
                    result = _apply_tracker(result, tracker)
                    source_bin = int(t0 + output_frame_idx * subsampling)
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
                            chunk_idx=chunk_idx,
                            chunk_start=t0,
                            chunk_end=t1,
                            output_frame_idx=output_frame_idx,
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
