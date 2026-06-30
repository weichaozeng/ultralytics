"""Run external SPAD preprocessors before a standard YOLO pose detector.

This script evaluates classic preprocessor + original detector baselines against a
VisionSIM split JSON. It mirrors ``test_spad_pose.py`` output conventions:

- JSON results: ``Evals/<model_name>/<test_name>/<sample>.json``
- Optional visualizations: ``Vis/<model_name>/<test_name>/...``
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
from ultralytics.data.spad_packed import raw_chunk_plane, raw_hwt_to_rgb_float, raw_plane_to_photon_cube
from ultralytics.models.yolo.pose.spad_preprocessors import build_spad_preprocessor

from det_spad import _reset_tracker, _run_detector_on_frames
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
from test_spad_pose import (
    _default_test_name,
    _default_test_name_from_json,
    _draw_bbox,
    _draw_pose,
    _eval_dir,
    _frame_record_from_result,
    _json_default,
    _model_name_from_ckpt,
    _names_to_dict,
    _sample_paths_from_args,
    _sample_specs_from_json,
    _vis_dir,
)


SUPPORTED_PREPROCESSORS = {"sum", "ppb", "stea"}


def _recon_t_indices(t_raw: int, subsampling: int, num_frames: int) -> list[int]:
    if num_frames <= 0 or t_raw <= 0:
        return []
    subsampling = max(int(subsampling), 1)
    t_raw = int(t_raw)
    if t_raw < subsampling:
        return [t_raw][:num_frames]
    full = t_raw // subsampling
    idx = [(i + 1) * subsampling for i in range(full)]
    if t_raw % subsampling != 0:
        idx.append(t_raw)
    return idx[:num_frames]


def _build_vis_frame(
    frame_bgr: np.ndarray,
    result,
    *,
    vis_bg: str,
    raw_chunk: np.ndarray,
    packed_nch: int,
    save_readrgb: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    recon = np.ascontiguousarray(frame_bgr.copy())
    if vis_bg == "recon":
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


def _process_window_with_preprocessor(
    preprocessor_name: str,
    preprocessor,
    raw_chunk: np.ndarray,
    *,
    packed_nch: int,
    device: torch.device,
    clear_states: bool,
) -> tuple[torch.Tensor, list[int]]:
    plane_thw = raw_chunk_plane(raw_chunk, packed_nch=packed_nch)
    photon_cube = raw_plane_to_photon_cube(plane_thw, device=device, as_bool=True)
    subsampling = int(getattr(preprocessor, "subsampling", raw_chunk.shape[0]) or raw_chunk.shape[0])
    t_raw = int(photon_cube.shape[-1])

    if preprocessor_name == "sum":
        recons = preprocessor.process_photon_cube(photon_cube, clear_states=clear_states)
        t_indices = list(preprocessor.recon_t_indices(t_raw, int(recons.shape[-1])))
    elif preprocessor_name == "ppb":
        recons = preprocessor.process_photon_cube(photon_cube, clear_states=clear_states)
        t_indices = _recon_t_indices(t_raw, subsampling, int(recons.shape[-1]))
    elif preprocessor_name == "stea":
        recons_ll = []
        t_indices = []
        first_local_chunk = bool(clear_states)
        for local_t0 in range(0, t_raw, subsampling):
            local_t1 = min(t_raw, local_t0 + subsampling)
            recon = preprocessor.process_photon_cube(
                photon_cube[..., local_t0:local_t1],
                clear_states=first_local_chunk,
            )
            first_local_chunk = False
            if int(recon.shape[-1]) > 0:
                recons_ll.append(recon)
                local_idx = _recon_t_indices(local_t1 - local_t0, subsampling, int(recon.shape[-1]))
                t_indices.extend([local_t0 + idx for idx in local_idx])
        if recons_ll:
            recons = torch.cat(recons_ll, dim=-1)
        else:
            h, w = map(int, photon_cube.shape[:2])
            recons = photon_cube.new_zeros((h, w, 0), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported preprocessor: {preprocessor_name!r}")

    frames = raw_hwt_to_rgb_float(recons.float(), packed_nch=int(packed_nch))
    return frames, t_indices[: int(frames.shape[0])]


def parse_args():
    ap = argparse.ArgumentParser(description="External SPAD preprocessor + original YOLO pose detector evaluation")
    ap.add_argument("--ckpt", type=str, required=True, help="Standard pretrained YOLO pose checkpoint")
    ap.add_argument("--preprocessor", type=str, required=True, choices=sorted(SUPPORTED_PREPROCESSORS))
    ap.add_argument("--test_json", type=str, default=None, help="VisionSIM test split JSON")
    ap.add_argument("--in_path", type=str, default=None, help="Optional sample/video folder, root folder, or .npy file")
    ap.add_argument("--in_glob", type=str, default=None, help="Optional glob when --in_path is a root folder")
    ap.add_argument("--test_name", type=str, default=None, help="Output test-set folder name")
    ap.add_argument("--device", type=str, default="", help="Torch device, e.g. cpu / cuda:0 / mps")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--tracker", type=str, default="botsort", choices=["bytetrack", "botsort", "spad_tracker"])
    ap.add_argument("--frame_rate", type=int, default=25, help="Tracker frame-rate hint")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument(
        "--spad-output-frames",
        type=int,
        default=8,
        help="Metadata only for external-preprocessor evaluation; default runtime chunking is serial at spad_subsampling.",
    )
    ap.add_argument("--spad-subsampling", type=int, default=320, help="Raw-bin aggregation per reconstructed frame")
    ap.add_argument(
        "--cube_chunk_t",
        type=int,
        default=0,
        help="If >0, split each raw video into chunks of this many bins. If 0, default to serial online-style chunking at spad_subsampling.",
    )
    ap.add_argument("--cube_chunk_stride", type=int, default=0, help="Stride for chunking; default uses cube_chunk_t")
    ap.add_argument(
        "--tail_pad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deprecated: final chunks shorter than cube_chunk_t are always dropped.",
    )
    ap.add_argument("--drop_tail", action="store_true", help="Deprecated: final chunks shorter than cube_chunk_t are always dropped.")
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


def _build_preprocessor(args, device: torch.device):
    name = str(args.preprocessor).strip().lower()
    kwargs: dict[str, Any] = {"subsampling": int(args.spad_subsampling)}
    if name == "ppb":
        kwargs.update(
            {
                "bocpd_gamma": float(args.ppb_bocpd_gamma),
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
    return build_spad_preprocessor(name, kwargs=kwargs).to(device)


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

    device = _resolve_device(args.device)
    model = YOLO(str(ckpt))
    preprocessor = _build_preprocessor(args, device)

    chunk_t = int(args.cube_chunk_t) if int(args.cube_chunk_t) > 0 else int(args.spad_subsampling)
    if chunk_t <= 0:
        raise ValueError(f"cube_chunk_t must be positive, got {chunk_t}")
    stride = int(args.cube_chunk_stride) if int(args.cube_chunk_stride) > 0 else chunk_t
    subsampling = int(getattr(preprocessor, "subsampling", args.spad_subsampling) or args.spad_subsampling)

    tracker_cfg = f"{args.tracker}.yaml"

    for sample_spec in sample_specs:
        sample_name = str(sample_spec["sample_name"])
        sample_path = Path(sample_spec["sample_path"])
        out_path = out_dir / f"{sample_name}.json"
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing file: {out_path}")
            continue

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
                "test_name": test_name,
                "sample_name": sample_name,
                "preprocessor": str(args.preprocessor),
                "det_thresh": float(args.det_thresh),
                "iou": float(args.iou),
                "max_det": int(args.max_det),
                "tracker": args.tracker,
                "frame_rate": int(args.frame_rate),
                "packed_ch_order": args.packed_ch_order,
                "tail_pad": False,
                "drop_tail": True,
                "vis_mode": str(args.vis),
                "spad_output_frames": int(args.spad_output_frames),
                "spad_subsampling": int(args.spad_subsampling),
                "cube_chunk_t": int(chunk_t),
                "cube_chunk_stride": int(stride),
                "names": _names_to_dict(model.names),
                "kpt_shape": list(map(int, getattr(model.model, "kpt_shape", (21, 3)))),
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
            _reset_tracker(model)
            first_chunk = True
            total_bins = _video_num_bins(source)
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

                with torch.inference_mode():
                    frames = _process_window_with_preprocessor(
                        str(args.preprocessor).strip().lower(),
                        preprocessor,
                        raw_chunk,
                        packed_nch=int(source.packed_nch),
                        device=device,
                        clear_states=first_chunk,
                    )
                frames_tchw, local_t_indices = frames
                frames_bgr = [
                    np.ascontiguousarray((np.clip(frame.permute(1, 2, 0).detach().float().cpu().numpy(), 0, 1) * 255.0).astype(np.uint8)[:, :, ::-1])
                    for frame in frames_tchw
                ]
                if not frames_bgr:
                    first_chunk = False
                    continue

                results = _run_detector_on_frames(
                    model,
                    frames_bgr,
                    conf=float(args.det_thresh),
                    tracker_cfg=tracker_cfg,
                    device=args.device,
                )

                chunk_record: dict[str, Any] = {
                    "chunk_idx": int(chunk_idx),
                    "start_bin": int(t0),
                    "end_bin": int(t1),
                    "input_bins": int(t1 - t0),
                    "model_input_bins": int(raw_chunk.shape[0]),
                    "frames": [],
                }

                for output_frame_idx, (result, frame_bgr) in enumerate(zip(results, frames_bgr)):
                    source_bin = int(t0 + (local_t_indices[output_frame_idx] if output_frame_idx < len(local_t_indices) else (output_frame_idx + 1) * subsampling))
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
                            names=model.names,
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
                first_chunk = False

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
