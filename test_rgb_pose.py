# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run standard YOLO pose tracking on RGB or offline-rendered ``frames.npy``.

Sibling of ``test_spad_pose_sequence.py`` / ``test_spad_pose_pre.py``:
- ``--source rgb`` (default): VisionSIM RGB clips (``renders-rgb25fps*/.../frames.npy``).
- ``--source sum`` (also ema/ppb/stea/hire): offline cache from the split JSON
  (``sum`` / ``render_sum``) or sibling ``renders-{src}-2kHz``.
- Runs the detector via Ultralytics ``model.track`` (same path as ``det.py`` / ``det_rgb.py``).
- Writes ``Evals/<YYYYMMDD>_detector[_<src>]/<test_name>/<sample>.json`` and optional ``Vis/...``.
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
from ultralytics.data.spad_pose_dataset import load_visionsim_split_json

from det import TRACKER_CHOICES, _tracker_yaml
from det_rgb import _load_rgb_frames
from det_spad import _reset_tracker
from test_spad_pose_sequence import (
    _default_test_name,
    _default_test_name_from_json,
    _draw_bbox,
    _draw_pose,
    _eval_dir,
    _frame_record_from_result,
    _json_default,
    _names_to_dict,
    _render_paths_from_direct_path,
    _render_paths_from_record,
    _sample_paths_from_args,
    _vis_dir,
)


RENDER_SOURCES = ("sum", "ema", "ppb", "stea", "hire")
FRAME_SOURCES = ("rgb", *RENDER_SOURCES)


def _dated_run_name(source: str) -> str:
    source = str(source).strip().lower()
    if source in {"", "rgb"}:
        return f"{datetime.now():%Y%m%d}_detector"
    return f"{datetime.now():%Y%m%d}_detector_{source}"


def _sample_specs_from_json(test_json: Path, *, source: str, render_root: str | None, source_render_dirname: str, render_tag: str) -> list[dict[str, Any]]:
    source = str(source).strip().lower()
    specs = []
    for record in load_visionsim_split_json(test_json):
        sample_name = str(record.get("name") or Path(record["id"]).name)
        if source == "rgb":
            rgb = record.get("rgb")
            if not rgb:
                raise ValueError(
                    f"Sample {record.get('id') or sample_name!r} has no `rgb` path in {test_json}. "
                    "Rebuild the split JSON with build_visionsim_split.py so each sample includes rgb."
                )
            sample_path = Path(rgb)
            specs.append(
                {
                    "sample_name": sample_name,
                    "sample_path": sample_path,
                    "frames_path": None,
                    "record": record,
                    "source": "rgb",
                }
            )
            continue

        render_dir, frames_path, _conf, _meta = _render_paths_from_record(
            record,
            preprocessor=source,
            render_root=render_root,
            source_render_dirname=source_render_dirname,
            render_tag=render_tag,
        )
        if not frames_path.is_file():
            raise FileNotFoundError(
                f"Cached {source} frames not found for {sample_name!r}: {frames_path}. "
                f"Run cache_spad_renders_2kHz.py (or pass --render_root)."
            )
        specs.append(
            {
                "sample_name": sample_name,
                "sample_path": render_dir,
                "frames_path": frames_path,
                "record": record,
                "source": source,
            }
        )
    return specs


def _resolve_frames_path(sample_path: Path) -> Path:
    """Accept a frames.npy file, or a directory containing frames.npy."""
    if sample_path.is_file():
        if sample_path.suffix.lower() != ".npy":
            raise ValueError(f"Expected frames.npy, got: {sample_path}")
        return sample_path
    if sample_path.is_dir():
        frames_path = sample_path / "frames.npy"
        if not frames_path.is_file():
            raise FileNotFoundError(f"Directory input requires frames.npy, not found: {frames_path}")
        return frames_path
    raise FileNotFoundError(f"Sample path not found: {sample_path}")


def _frames_to_bgr_list(frames: np.ndarray) -> list[np.ndarray]:
    """Convert RGB HWC uint8 or CHW/NCHW float cache frames to contiguous BGR uint8 list."""
    arr = np.asarray(frames)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D frames array, got shape={arr.shape}")

    # NHWC RGB
    if arr.shape[-1] == 3:
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames_bgr = arr[..., ::-1]
        return [np.ascontiguousarray(frames_bgr[i]) for i in range(frames_bgr.shape[0])]

    # NCHW RGB (offline sum/ema/... caches)
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

    raise ValueError(f"Unsupported frames layout {arr.shape}; expected NHWC or NCHW RGB")


def _load_frames_bgr(frames_path: Path, *, source: str) -> list[np.ndarray]:
    if source == "rgb":
        return _frames_to_bgr_list(_load_rgb_frames(frames_path))
    # mmap-friendly for large caches; convert frame-by-frame
    arr = np.load(frames_path, mmap_mode="r")
    return _frames_to_bgr_list(arr)


def _build_vis_frame(frame_bgr: np.ndarray, result) -> np.ndarray:
    vis = np.ascontiguousarray(frame_bgr.copy())
    if result.boxes is None or not len(result.boxes):
        return vis
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
    return vis


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "RGB or offline-rendered frames.npy -> YOLO pose detector -> JSON predictions "
            "(aligned with test_spad_pose_sequence / sum_pose_2kHz training)"
        )
    )
    ap.add_argument("--in_path", type=str, default=None, help="Sample path/root (dir with frames.npy, or frames.npy itself)")
    ap.add_argument(
        "--test_json",
        type=str,
        default=None,
        help="VisionSIM test split JSON; paths come from `rgb` or offline cache keys depending on --source.",
    )
    ap.add_argument("--in_glob", type=str, default=None, help="Optional glob when --in_path is a root folder")
    ap.add_argument("--test_name", type=str, default=None, help="Output test-set folder name; defaults from --test_json or --in_path")
    ap.add_argument("--ckpt", type=str, required=True, help="Standard YOLO pose checkpoint (e.g. weights/detector.pt or Runs/rgb_pose/.../best.pt)")
    ap.add_argument(
        "--source",
        type=str,
        default="rgb",
        choices=list(FRAME_SOURCES),
        help="Frame source: rgb (default) or offline cache sum/ema/ppb/stea/hire (e.g. renders-sum-2kHz).",
    )
    ap.add_argument("--render_root", type=str, default=None, help="Optional explicit root for offline rendered frames")
    ap.add_argument(
        "--source_render_dirname",
        type=str,
        default="renders-spc2kHz",
        help="Sibling packed-SPAD dirname used to infer renders-{source}-{tag}",
    )
    ap.add_argument(
        "--render_tag",
        type=str,
        default="2kHz",
        help="Cache dir suffix: renders-{source}-{tag}. Empty => renders-{source}.",
    )
    ap.add_argument("--device", type=str, default="", help="Torch device, e.g. cpu / cuda:0 / mps")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--tracker", type=str, default="botsort", choices=list(TRACKER_CHOICES))
    ap.add_argument("--frame_rate", type=int, default=25, help="Tracker / video FPS hint (RGB/sum chunks are typically 25)")
    ap.add_argument(
        "--vis",
        nargs="?",
        const="video",
        default="none",
        choices=["none", "image", "video"],
        help="Optional visualization. `--vis` defaults to video; use `--vis image` for PNG frames.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    source = str(args.source).strip().lower()
    model_name = "detector" if source == "rgb" else f"detector_{source}"
    run_name = _dated_run_name(source)

    if args.test_json:
        test_json = Path(args.test_json)
        if not test_json.is_file():
            raise FileNotFoundError(f"Test split JSON not found: {test_json}")
        sample_specs = _sample_specs_from_json(
            test_json,
            source=source,
            render_root=args.render_root,
            source_render_dirname=args.source_render_dirname,
            render_tag=args.render_tag,
        )
        test_name = args.test_name or _default_test_name_from_json(test_json)
    else:
        if not args.in_path:
            raise ValueError("Either --test_json or --in_path must be provided.")
        in_path = Path(args.in_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Input path not found: {in_path}")
        sample_specs = []
        for p in _sample_paths_from_args(in_path, args.in_glob):
            if source == "rgb":
                sample_specs.append(
                    {
                        "sample_name": p.name if p.is_dir() else (p.parent.name if p.name == "frames.npy" else p.stem),
                        "sample_path": p,
                        "frames_path": None,
                        "record": None,
                        "source": "rgb",
                    }
                )
            else:
                render_dir, frames_path, _conf, _meta = _render_paths_from_direct_path(p)
                sample_specs.append(
                    {
                        "sample_name": render_dir.name,
                        "sample_path": render_dir,
                        "frames_path": frames_path,
                        "record": None,
                        "source": source,
                    }
                )
        test_name = args.test_name or _default_test_name(in_path, args.in_glob)

    out_dir = _eval_dir(run_name, test_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_root = _vis_dir(run_name, test_name) if args.vis != "none" else None
    if vis_root is not None:
        vis_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(ckpt))
    names = model.names
    kpt_shape = getattr(model.model, "kpt_shape", (21, 3))
    tracker_cfg = _tracker_yaml(args.tracker)

    for sample_spec in sample_specs:
        sample_name = str(sample_spec["sample_name"])
        sample_path = Path(sample_spec["sample_path"])
        out_path = out_dir / f"{sample_name}.json"
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing file: {out_path}")
            continue

        frames_path = Path(sample_spec["frames_path"]) if sample_spec.get("frames_path") else _resolve_frames_path(sample_path)
        frames_bgr = _load_frames_bgr(frames_path, source=source)
        num_frames = len(frames_bgr)

        _reset_tracker(model)

        global_frame_idx = 0
        sample_vis_dir = None
        video_writer = None
        if vis_root is not None and args.vis == "image":
            sample_vis_dir = vis_root / sample_name
            sample_vis_dir.mkdir(parents=True, exist_ok=True)
        video_path = vis_root / f"{sample_name}.mp4" if vis_root is not None and args.vis == "video" else None

        # End-of-chunk GT index: RGB/cache frame i <-> GT (i+1)*gt_per_rgb (matches RgbPoseFrameDataset).
        gt_fps = 125.0
        rgb_fps = float(args.frame_rate)
        gt_per_rgb = max(int(round(gt_fps / rgb_fps)), 1)
        sample_record: dict[str, Any] = {
            "metadata": {
                "format": "spadhand_rgb_predictions_v1",
                "ckpt": str(ckpt),
                "model_name": model_name,
                "run_name": run_name,
                "test_name": test_name,
                "sample_name": sample_name,
                "frame_source": source,
                "det_thresh": float(args.det_thresh),
                "iou": float(args.iou),
                "max_det": int(args.max_det),
                "tracker": args.tracker,
                "frame_rate": int(args.frame_rate),
                "rgb_fps": int(args.frame_rate),
                "gt_fps": float(gt_fps),
                "gt_per_rgb": int(gt_per_rgb),
                "vis_mode": str(args.vis),
                "names": _names_to_dict(names),
                "kpt_shape": list(map(int, kpt_shape)),
                "source_bin_convention": "rgb_frame_idx" if source == "rgb" else f"{source}_chunk_idx",
            },
            "source": {
                "path": str(frames_path),
                "is_dir": bool(sample_path.is_dir()),
                "modality": source,
            },
            "videos": [],
        }
        if sample_spec.get("record") is not None:
            sample_record["source"]["split_record"] = sample_spec["record"]

        track_kwargs: dict[str, Any] = {
            "conf": float(args.det_thresh),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "persist": True,
            "tracker": tracker_cfg,
            "verbose": False,
        }
        if args.device:
            track_kwargs["device"] = args.device
        results = model.track(frames_bgr, **track_kwargs)
        if len(results) != num_frames:
            raise RuntimeError(
                f"Detector returned {len(results)} results for {num_frames} frames in {frames_path}"
            )

        video_record: dict[str, Any] = {
            "video_idx": 0,
            "layout": source,
            "packed_nch": 3,
            "total_bins": int(num_frames),
            "chunk_t": 1,
            "chunk_stride": 1,
            "subsampling": 1,
            "chunks": [],
        }

        for frame_idx, (result, frame_bgr) in enumerate(
            tqdm(zip(results, frames_bgr), total=num_frames, desc=f"Testing [{sample_name}] ({source})")
        ):
            chunk_record: dict[str, Any] = {
                "chunk_idx": int(frame_idx),
                "start_bin": int(frame_idx),
                "end_bin": int(frame_idx + 1),
                "input_bins": 1,
                "model_input_bins": 1,
                "target_gt_time": float((frame_idx + 1) * gt_per_rgb),
                "frames": [],
            }

            if args.vis != "none":
                vis = _build_vis_frame(frame_bgr, result)
                if args.vis == "image" and sample_vis_dir is not None:
                    stem = f"frame{global_frame_idx:07d}"
                    cv2.imwrite(str(sample_vis_dir / f"{stem}.png"), vis)
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
                    chunk_idx=frame_idx,
                    chunk_start=frame_idx,
                    chunk_end=frame_idx + 1,
                    output_frame_idx=0,
                    source_bin=frame_idx,
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
