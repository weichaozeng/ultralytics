# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run QNNS pose inference on a test set and save structured predictions.

This is the non-visualization sibling of ``det_qnns.py``. It feeds raw SPAD clips
into a trained QNN pose checkpoint, applies the same NMS and optional tracker, and
writes one JSON file per input sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO

from det_qnns import (
    _apply_tracker,
    _init_tracker,
    _iter_raw_video_sources_from_sample_path,
    _postprocess_pose_predictions,
    _prepare_raw_chunk_for_qnn,
    _recon_frames_bgr,
    _resolve_device,
    _results_from_preds,
    _slice_raw_chunk,
    _trained_chunk_t,
    _video_num_bins,
)


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


def _default_test_name(in_path: Path, in_glob: str | None) -> str:
    if in_path.is_dir() and in_glob:
        return in_path.name
    return in_path.name if in_path.is_dir() else in_path.stem


def _output_dir_from_ckpt(ckpt: Path, test_name: str) -> Path:
    if ckpt.parent.name == "weights":
        run_dir = ckpt.parent.parent
    else:
        run_dir = ckpt.parent
    return run_dir / "test" / test_name


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


def parse_args():
    ap = argparse.ArgumentParser(description="Raw SPAD test set -> trained QNNPoseModel -> JSON predictions")
    ap.add_argument("--in_path", type=str, required=True, help="Sample/video folder, root folder, or a .npy file")
    ap.add_argument(
        "--in_glob",
        type=str,
        default=None,
        help="Optional glob when --in_path is a root folder containing many sample/video subfolders",
    )
    ap.add_argument("--test_name", type=str, default=None, help="Output test-set folder name; defaults from --in_path")
    ap.add_argument("--ckpt", type=str, required=True, help="Trained QNN pose checkpoint")
    ap.add_argument("--device", type=str, default="", help="Torch device, e.g. cpu / cuda:0 / mps")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--tracker", type=str, default="botsort", choices=["bytetrack", "botsort"])
    ap.add_argument("--frame_rate", type=int, default=25, help="Tracker frame-rate hint")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument(
        "--cube_chunk_t",
        type=int,
        default=0,
        help="If >0, split each raw video into chunks of this many bins. If 0, use the train-time QNN window length.",
    )
    ap.add_argument("--cube_chunk_stride", type=int, default=0, help="Stride for chunking; default uses cube_chunk_t")
    ap.add_argument(
        "--tail_pad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pad final short chunk to cube_chunk_t; with --no-tail_pad, run short tails at natural length.",
    )
    ap.add_argument(
        "--drop_tail",
        action="store_true",
        help="Drop the final chunk when it has fewer than cube_chunk_t bins.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files")
    return ap.parse_args()


def main():
    args = parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    ckpt = Path(args.ckpt)
    test_name = args.test_name or _default_test_name(in_path, args.in_glob)
    out_dir = _output_dir_from_ckpt(ckpt, test_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = _sample_paths_from_args(in_path, args.in_glob)

    yolo = YOLO(str(ckpt))
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

    for sample_path in sample_paths:
        sample_name = sample_path.name if sample_path.is_dir() else sample_path.stem
        out_path = out_dir / f"{sample_name}.json"
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing file: {out_path}")
            continue

        tracker.reset()
        global_frame_idx = 0
        sample_record: dict[str, Any] = {
            "metadata": {
                "format": "spadhand_qnns_predictions_v1",
                "ckpt": str(ckpt),
                "test_name": test_name,
                "sample_name": sample_name,
                "det_thresh": float(args.det_thresh),
                "iou": float(args.iou),
                "max_det": int(args.max_det),
                "tracker": args.tracker,
                "frame_rate": int(args.frame_rate),
                "packed_ch_order": args.packed_ch_order,
                "tail_pad": bool(args.tail_pad),
                "drop_tail": bool(args.drop_tail),
                "names": _names_to_dict(names),
                "kpt_shape": list(map(int, kpt_shape)),
            },
            "source": {
                "path": str(sample_path),
                "is_dir": bool(sample_path.is_dir()),
            },
            "videos": [],
        }

        video_iter = _iter_raw_video_sources_from_sample_path(sample_path)
        for video_idx, source in enumerate(tqdm(video_iter, desc=f"Testing [{sample_name}]")):
            total_bins = _video_num_bins(source)
            if int(args.cube_chunk_t) > 0:
                chunk_t = int(args.cube_chunk_t)
            elif trained_chunk_t is not None:
                chunk_t = int(trained_chunk_t)
            else:
                raise ValueError(
                    "Unable to infer train-time QNN window length from checkpoint. "
                    "Pass --cube_chunk_t explicitly."
                )
            if chunk_t <= 0:
                raise ValueError(f"cube_chunk_t must be positive, got {chunk_t}")

            subsampling = int(getattr(getattr(qnn_model, "integrator", None), "subsampling", 1) or 1)
            if chunk_t < subsampling:
                raise ValueError(
                    f"cube_chunk_t={chunk_t} is shorter than integrator subsampling={subsampling}, "
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
                if bool(args.drop_tail) and (t1 - t0) < chunk_t:
                    continue
                raw_chunk = _slice_raw_chunk(source, t0, t1, packed_ch_order=args.packed_ch_order)
                raw_chunk = _prepare_raw_chunk_for_qnn(
                    raw_chunk,
                    chunk_t=chunk_t,
                    tail_pad_full=bool(args.tail_pad),
                )
                if raw_chunk is None:
                    continue

                qnn_model.qnn_packed_nch = int(source.packed_nch)

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
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
