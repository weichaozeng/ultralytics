# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compare SPAD preprocessing methods before a standard YOLO pose detector.

This script does not use QNNPoseModel. It converts SPAD chunks into RGB-like
frames with one of:
- sum: temporal mean over the chunk
- ppb: PerPixelBayesian reconstruction
- vel: detection-guided velocity-compensated integration

- hyb: STEA (causal temporal bases + KL spatio-temporal soft routing)

The resulting frames are passed to an unmodified pretrained YOLO pose model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils.ops import Profile
from ultralytics.data.spad_packed import (
    infer_packed_nch,
    integrate_raw_chunk_to_rgb,
    is_packed_spad,
    packed_frames_to_raw_video,
    raw_chunk_plane,
    raw_hwt_to_rgb_float,
    raw_plane_to_photon_cube,
    sum_raw_chunk_to_rgb,
)
from ultralytics.quanta_hybrid_networks.integrator import SpatioTemporalEvidenceAccumulation
from ultralytics.quanta_motion_networks.integrator import VelIntegrator
from ultralytics.quanta_neural_networks.integrator import PerPixelBayesian


BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGER_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
COLOR_KEYPOINT = (255, 255, 255)
COLOR_WRIST = (255, 165, 0)


@dataclass(frozen=True)
class SpadSource:
    array: np.ndarray
    layout: str
    packed_nch: int


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
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img_bgr, (x1, y1), (x1 + tw, y1 + th + baseline), color, -1)
    cv2.putText(img_bgr, text, (x1, y1 + baseline * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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
            color, radius = (COLOR_WRIST, 6) if i == 0 else (COLOR_KEYPOINT, 4)
            cv2.circle(img_bgr, (int(kk[0]), int(kk[1])), radius, color, -1)
    return img_bgr


def _np_load(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def _looks_like_hwt(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[0] == arr.shape[1] and arr.shape[2] != arr.shape[1]


def _sources_from_array(arr: np.ndarray) -> list[SpadSource]:
    if is_packed_spad(arr):
        return [SpadSource(arr, "packed", infer_packed_nch(arr))]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return [SpadSource(arr, "thwc1", 4)]
    if arr.ndim == 3:
        return [SpadSource(arr, "hwt" if _looks_like_hwt(arr) else "thw", 4)]
    if arr.ndim == 4:
        return [SpadSource(arr[i], "hwt", 4) for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported SPAD input shape: {arr.shape}")


def _iter_sources(path: Path):
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".npy")
        if not files and (path / "frames.npy").exists():
            files = [path / "frames.npy"]
        for file in files:
            yield from _sources_from_array(_np_load(file))
        return
    if path.suffix.lower() == ".npy":
        yield from _sources_from_array(_np_load(path))
        return
    raise ValueError(f"Unsupported input path: {path}")


def _num_bins(source: SpadSource) -> int:
    if source.layout in {"packed", "thwc1", "thw"}:
        return int(source.array.shape[0])
    if source.layout == "hwt":
        return int(source.array.shape[2])
    raise ValueError(f"Unsupported layout: {source.layout}")


def _slice_raw(source: SpadSource, t0: int, t1: int, *, packed_ch_order: str) -> np.ndarray:
    if source.layout == "packed":
        packed = np.asarray(source.array[t0:t1])
        return packed_frames_to_raw_video(packed, ch_order=packed_ch_order)
    if source.layout == "thwc1":
        return np.ascontiguousarray(source.array[t0:t1].astype(np.uint8, copy=False))
    if source.layout == "thw":
        return np.ascontiguousarray(source.array[t0:t1, :, :, None].astype(np.uint8, copy=False))
    if source.layout == "hwt":
        return np.ascontiguousarray(np.transpose(source.array[:, :, t0:t1], (2, 0, 1))[:, :, :, None].astype(np.uint8, copy=False))
    raise ValueError(f"Unsupported layout: {source.layout}")


def _rgb_tensor_to_bgr_u8(frames_tchw: torch.Tensor, *, vis_mode: str, percentile: float, gamma: float) -> list[np.ndarray]:
    rgb = frames_tchw.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    out = []
    for frame in rgb:
        if vis_mode == "linear":
            vis = np.clip(frame, 0, 1)
        elif vis_mode == "gamma":
            vis = np.power(np.clip(frame, 0, 1), 1.0 / gamma)
        elif vis_mode == "percentile":
            scale = float(np.percentile(frame, percentile))
            vis = np.clip(frame / max(scale, 1e-6), 0, 1)
        elif vis_mode == "percentile_gamma":
            scale = float(np.percentile(frame, percentile))
            vis = np.clip(frame / max(scale, 1e-6), 0, 1)
            vis = np.power(vis, 1.0 / gamma)
        else:
            raise ValueError(f"Unsupported vis mode: {vis_mode}")
        out.append(np.ascontiguousarray((vis * 255.0).round().astype(np.uint8)[:, :, ::-1]))
    return out


def _preprocess_sum(raw_chunk: np.ndarray, *, packed_nch: int, device: torch.device, **kwargs) -> torch.Tensor:
    return sum_raw_chunk_to_rgb(raw_chunk, packed_nch=packed_nch, device=device)


def _preprocess_ppb(
    raw_chunk: np.ndarray,
    *,
    packed_nch: int,
    device: torch.device,
    integrator: PerPixelBayesian,
    clear_states: bool,
    **kwargs,
) -> torch.Tensor:
    return integrate_raw_chunk_to_rgb(
        integrator,
        raw_chunk,
        packed_nch=packed_nch,
        device=device,
        clear_states=clear_states,
    )


def _preprocess_hyb(
    raw_chunk: np.ndarray,
    *,
    packed_nch: int,
    device: torch.device,
    integrator: SpatioTemporalEvidenceAccumulation,
    clear_states: bool,
    **kwargs,
) -> torch.Tensor:
    return integrate_raw_chunk_to_rgb(
        integrator,
        raw_chunk,
        packed_nch=packed_nch,
        device=device,
        clear_states=clear_states,
    )


def _preprocess_vel(
    raw_chunk: np.ndarray,
    *,
    packed_nch: int,
    device: torch.device,
    integrator: VelIntegrator,
    clear_states: bool,
    **kwargs,
) -> torch.Tensor:
    cube = raw_plane_to_photon_cube(raw_chunk_plane(raw_chunk, packed_nch=packed_nch), device=device, as_bool=True)
    recons = integrator.process_photon_cube(cube, clear_states=clear_states, packed_nch=packed_nch)
    if integrator.outputs_rgb:
        return recons.unsqueeze(0)
    rgb_tchw = raw_hwt_to_rgb_float(recons.float(), packed_nch=packed_nch)
    if int(rgb_tchw.shape[0]) <= 0:
        return rgb_tchw
    return rgb_tchw[-1:].contiguous()


def _extract_track_centers(result) -> tuple[np.ndarray, np.ndarray]:
    """Return track IDs and box centers (x, y) in detection image coordinates."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.float32)
    track_ids = result.boxes.id
    if track_ids is None:
        track_ids = torch.arange(len(result.boxes), device=result.boxes.data.device)
    ids = track_ids.detach().cpu().numpy().astype(np.int64).reshape(-1)
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1).astype(np.float32)
    return ids, centers


def _get_live_tracker(model: YOLO):
    if not hasattr(model, "predictor") or model.predictor is None:
        return None
    trackers = getattr(model.predictor, "trackers", None)
    if not trackers:
        return None
    return trackers[0]


def _set_vel_field_from_tracker(model: YOLO, integrator: VelIntegrator) -> None:
    tracker = _get_live_tracker(model)
    if tracker is None:
        integrator.set_velocity_field(None, source_space="rgb")
        return
    if not hasattr(tracker, "last_velocity_field"):
        raise TypeError("The 'vel' preprocessor requires a tracker that exposes 'last_velocity_field'.")
    integrator.set_velocity_field(getattr(tracker, "last_velocity_field", None), source_space="rgb")


def _run_detector_on_frames(model: YOLO, frames_bgr: list[np.ndarray], *, conf: float, tracker_cfg: str, device: str):
    kwargs = {"conf": conf, "persist": True, "tracker": tracker_cfg, "verbose": False}
    if device:
        kwargs["device"] = device
    return model.track(frames_bgr, **kwargs)


def _draw_results(frame_bgr: np.ndarray, result) -> np.ndarray:
    vis = np.ascontiguousarray(frame_bgr.copy())
    if result.boxes is None or len(result.boxes) == 0:
        return vis
    track_ids = result.boxes.id
    if track_ids is None:
        track_ids = torch.arange(len(result.boxes), device=result.boxes.data.device)
    ids = track_ids.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()

    poses = None
    if getattr(result, "keypoints", None) is not None:
        xy = result.keypoints.xy.cpu().numpy()
        kconf = result.keypoints.conf.cpu().numpy()
        poses = np.concatenate([xy, kconf[..., None]], axis=2)

    for i, tid in enumerate(ids):
        vis = draw_bbox(vis, int(tid), np.concatenate([boxes[i], [confs[i]]]), float(cls[i]))
        if poses is not None and i < len(poses):
            vis = draw_pose(vis, poses[i])
    return vis


def _reset_tracker(model: YOLO) -> None:
    if hasattr(model, "predictor") and model.predictor is not None and getattr(model.predictor, "trackers", None):
        for tracker in model.predictor.trackers:
            tracker.reset()


def _resolve_device(device: str) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _output_dir(save_root: Path, sample_name: str, video_idx: int) -> Path:
    """Per-video output folder, e.g. ``sample/video00000``."""
    return save_root / sample_name / f"video{video_idx:05d}"


@dataclass
class _MethodChunkTiming:
    chunk_secs: list[float] = field(default_factory=list)
    chunk_bins: list[int] = field(default_factory=list)
    first_chunk_secs: list[float] = field(default_factory=list)
    rest_chunk_secs: list[float] = field(default_factory=list)


class PreprocessTiming:
    """Accumulate per-chunk preprocess wall time (integrator + raw→RGB only)."""

    def __init__(self, *, warmup_chunks: int = 0) -> None:
        self.warmup_chunks = max(int(warmup_chunks), 0)
        self.global_stats: dict[str, _MethodChunkTiming] = defaultdict(_MethodChunkTiming)
        self.per_video_secs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def record(
        self,
        method: str,
        dt_sec: float,
        *,
        video_key: str,
        chunk_bins: int,
        cube_idx: int,
        is_first_chunk: bool,
    ) -> None:
        if cube_idx < self.warmup_chunks:
            return
        stats = self.global_stats[method]
        stats.chunk_secs.append(float(dt_sec))
        stats.chunk_bins.append(int(chunk_bins))
        if is_first_chunk:
            stats.first_chunk_secs.append(float(dt_sec))
        else:
            stats.rest_chunk_secs.append(float(dt_sec))
        self.per_video_secs[video_key][method].append(float(dt_sec))

    @staticmethod
    def _summarize_chunks(secs: list[float], bins: list[int] | None = None) -> dict[str, float | int] | None:
        if not secs:
            return None
        ms = np.asarray(secs, dtype=np.float64) * 1000.0
        total_bins = int(sum(bins)) if bins else 0
        out: dict[str, float | int] = {
            "chunks": len(secs),
            "total_s": float(np.sum(secs)),
            "mean_ms": float(np.mean(ms)),
            "p50_ms": float(np.percentile(ms, 50)),
            "p90_ms": float(np.percentile(ms, 90)),
        }
        if total_bins > 0:
            out["ms_per_bin"] = float(np.sum(secs) * 1000.0 / total_bins)
        return out

    def global_summary(self) -> dict[str, dict]:
        summary: dict[str, dict] = {}
        for method, stats in sorted(self.global_stats.items()):
            entry: dict = {}
            overall = self._summarize_chunks(stats.chunk_secs, stats.chunk_bins)
            if overall is not None:
                entry["overall"] = overall
            first = self._summarize_chunks(stats.first_chunk_secs)
            if first is not None:
                entry["first_chunk"] = first
            rest = self._summarize_chunks(stats.rest_chunk_secs)
            if rest is not None:
                entry["rest_chunks"] = rest
            if entry:
                summary[method] = entry
        return summary

    def per_video_summary(self) -> dict[str, dict[str, dict]]:
        out: dict[str, dict[str, dict]] = {}
        for video_key in sorted(self.per_video_secs):
            methods: dict[str, dict] = {}
            for method, secs in sorted(self.per_video_secs[video_key].items()):
                row = self._summarize_chunks(secs)
                if row is not None:
                    methods[method] = row
            if methods:
                out[video_key] = methods
        return out

    def to_dict(self, *, chunk_size: int, device: str) -> dict:
        return {
            "chunk_size_nominal": int(chunk_size),
            "device": device,
            "warmup_chunks_skipped": self.warmup_chunks,
            "note": "Times cover preprocess only (_preprocess_*: integrator + raw→RGB), not vis/det/imwrite.",
            "global": self.global_summary(),
            "per_video": self.per_video_summary(),
        }

    def print_summary(self) -> None:
        global_summary = self.global_summary()
        if not global_summary:
            print("Preprocess timing: no chunks recorded (increase data or lower --time_pre_warmup_chunks).")
            return

        print("\n=== Preprocess timing (per chunk, global) ===")
        header = f"{'method':<6} {'chunks':>7} {'total_s':>9} {'mean_ms':>9} {'p50_ms':>9} {'p90_ms':>9} {'ms/bin':>9}"
        print(header)
        print("-" * len(header))
        for method, entry in global_summary.items():
            row = entry["overall"]
            ms_per_bin = row.get("ms_per_bin", float("nan"))
            print(
                f"{method:<6} {row['chunks']:>7d} {row['total_s']:>9.3f} "
                f"{row['mean_ms']:>9.2f} {row['p50_ms']:>9.2f} {row['p90_ms']:>9.2f} {ms_per_bin:>9.4f}"
            )
            if "first_chunk" in entry and "rest_chunks" in entry:
                f_ms = entry["first_chunk"]["mean_ms"]
                r_ms = entry["rest_chunks"]["mean_ms"]
                print(f"       first_chunk mean_ms={f_ms:.2f}  rest_chunks mean_ms={r_ms:.2f}")

        per_video = self.per_video_summary()
        if per_video:
            print("\n=== Preprocess timing (per video, mean_ms/chunk) ===")
            for video_key, methods in per_video.items():
                parts = [f"{m}={methods[m]['mean_ms']:.2f}ms" for m in sorted(methods)]
                print(f"  {video_key}: " + ", ".join(parts))

    def write_json(self, path: Path, *, chunk_size: int, device: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(chunk_size=chunk_size, device=device), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote preprocess timing JSON: {path}")


def _preprocess_chunk(
    name: str,
    raw_chunk: np.ndarray,
    *,
    packed_nch: int,
    device: torch.device,
    first_chunk: bool,
    ppb: PerPixelBayesian | None,
    hyb: SpatioTemporalEvidenceAccumulation | None,
    vel: VelIntegrator | None,
) -> torch.Tensor:
    if name == "sum":
        return _preprocess_sum(raw_chunk, packed_nch=packed_nch, device=device)
    if name == "ppb":
        return _preprocess_ppb(
            raw_chunk,
            packed_nch=packed_nch,
            device=device,
            integrator=ppb,
            clear_states=first_chunk,
        )
    if name == "hyb":
        return _preprocess_hyb(
            raw_chunk,
            packed_nch=packed_nch,
            device=device,
            integrator=hyb,
            clear_states=first_chunk,
        )
    return _preprocess_vel(
        raw_chunk, packed_nch=packed_nch, device=device, integrator=vel, clear_states=first_chunk
    )


def main():
    ap = argparse.ArgumentParser(description="SPAD preprocess comparison before standard YOLO pose detection")
    ap.add_argument("--in_path", type=str, required=True, help="SPAD sample directory, root directory, or .npy path")
    ap.add_argument("--in_glob", type=str, default=None, help="Optional glob for a root folder containing sample directories")
    ap.add_argument("--ckpt", type=str, required=True, help="Standard pretrained YOLO pose checkpoint")
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--pre", type=str, default="sum,ppb,vel", help="Comma-separated preprocessors: sum,ppb,vel,hyb")
    ap.add_argument("--chunk_size", type=int, default=320)
    ap.add_argument("--chunk_stride", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--tracker", type=str, default="spad_tracker", choices=["bytetrack", "botsort", "spad_tracker"])
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
   # perpixelbayesian
    ap.add_argument("--ppb_gamma", type=float, default=5e-4)
    ap.add_argument("--ppb_quantile", type=float, default=1.0)
    ap.add_argument("--ppb_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppb_min_filter_size", type=int, default=7)
    # STEA hybrid preprocessor
    ap.add_argument("--hyb_fast_window", type=int, default=16, help="Fast Gamma temporal basis length for STEA")
    ap.add_argument("--hyb_slow_window", type=int, default=128, help="Slow boxcar temporal basis length for STEA")
    ap.add_argument("--hyb_temporal_window", type=int, default=5, help="Causal evidence time blur window for STEA")
    ap.add_argument("--hyb_fast_tau", type=float, default=4.0, help="Gamma kernel tau for the fast STEA basis")
    ap.add_argument("--hyb_motion_sharpness", type=float, default=60.0, help="Sigmoid sharpness for KL motion probability")
    ap.add_argument("--hyb_motion_threshold", type=float, default=0.05, help="KL threshold for sigmoid motion probability")
    ap.add_argument("--hyb_eps", type=float, default=1e-5, help="Clamp epsilon for Bernoulli rates")
    ap.add_argument("--hyb_blend_const", type=float, default=16.0, help="C in W_mean=L/(L+C) for stable mean confidence")
    ap.add_argument("--hyb_kernel_size", type=int, default=None, help="Deprecated alias for --hyb_slow_window")
    ap.add_argument(
        "--hyb_prior_strength",
        type=float,
        default=1.0,
        help="Deprecated; ignored by STEA",
    )
    ap.add_argument(
        "--hyb_gating_tau",
        type=float,
        default=0.1,
        help="Deprecated; ignored by STEA",
    )
    ap.add_argument("--hyb_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hyb_quantile", type=float, default=1.0)
    ap.add_argument(
        "--hyb_max_filter_size",
        type=int,
        default=3,
        help="Deprecated; ignored by STEA",
    )
    # velintegrator
    ap.add_argument("--vel_max_shift", type=int, default=16)
    ap.add_argument("--vel_patch_size", type=int, default=0, help="Deprecated; ignored by the dense-field integrator")
    ap.add_argument("--vel_compensate", type=str, default="rgb", choices=["rgb", "raw"], help="Shift in RGB (avoids Bayer color fringing) or raw")
    ap.add_argument("--vel_quantile", type=float, default=1.0)
    ap.add_argument("--vel_normalize", action=argparse.BooleanOptionalAction, default=False)
    # vis
    ap.add_argument("--vis_mode", type=str, default="linear", choices=["linear", "gamma", "percentile", "percentile_gamma"])
    ap.add_argument("--vis_percentile", type=float, default=99.5)
    ap.add_argument("--vis_gamma", type=float, default=2.2)
    ap.add_argument(
        "--time_pre",
        action="store_true",
        help="Record preprocess wall time per chunk for each --pre method (integrator + raw→RGB only)",
    )
    ap.add_argument(
        "--time_pre_warmup_chunks",
        type=int,
        default=0,
        help="Skip timing for the first N chunks per video (e.g. 1 to exclude clear_states warmup)",
    )
    ap.add_argument(
        "--time_pre_out",
        type=str,
        default="",
        help="JSON output path for timing stats (default: {save_dir}/preprocess_timing.json)",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")
    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir() and args.in_glob:
        sample_paths = [p for p in sorted(in_path.glob(args.in_glob)) if p.is_dir()]
    else:
        sample_paths = [in_path]
    if not sample_paths:
        raise FileNotFoundError(f"No inputs matched: {in_path}/{args.in_glob}")

    preprocessors = [x.strip() for x in args.pre.split(",") if x.strip()]
    invalid = sorted(set(preprocessors) - {"sum", "ppb", "vel", "hyb"})
    if invalid:
        raise ValueError(f"Unsupported preprocessors: {invalid}")
    if "vel" in preprocessors and args.tracker != "spad_tracker":
        raise ValueError("The 'vel' preprocessor now requires '--tracker spad_tracker'.")

    device = _resolve_device(args.device)
    tracker_cfg = f"{args.tracker}.yaml"

    models = {name: YOLO(args.ckpt) for name in preprocessors}
    ppb = (
        PerPixelBayesian(
            subsampling=int(args.chunk_size),
            bocpd_gamma=float(args.ppb_gamma),
            normalize=bool(args.ppb_normalize),
            quantile=float(args.ppb_quantile),
            min_filter_size=int(args.ppb_min_filter_size),
        ).to(device)
        if "ppb" in preprocessors
        else None
    )
    vel = (
        VelIntegrator(
            chunk_size=int(args.chunk_size),
            max_shift=int(args.vel_max_shift),
            patch_size=int(args.vel_patch_size),
            compensate_space=str(args.vel_compensate),
            normalize=bool(args.vel_normalize),
            quantile=float(args.vel_quantile),
        ).to(device)
        if "vel" in preprocessors
        else None
    )
    hyb = (
        SpatioTemporalEvidenceAccumulation(
            chunk_size=int(args.chunk_size),
            fast_window=int(args.hyb_fast_window),
            slow_window=int(args.hyb_kernel_size or args.hyb_slow_window),
            temporal_window=int(args.hyb_temporal_window),
            fast_tau=float(args.hyb_fast_tau),
            motion_sharpness=float(args.hyb_motion_sharpness),
            motion_threshold=float(args.hyb_motion_threshold),
            eps=float(args.hyb_eps),
            stable_prior=float(args.hyb_blend_const),
            subsampling=int(args.chunk_size),
            normalize=bool(args.hyb_normalize),
            quantile=float(args.hyb_quantile),
        ).to(device)
        if "hyb" in preprocessors
        else None
    )
    timing = PreprocessTiming(warmup_chunks=int(args.time_pre_warmup_chunks)) if args.time_pre else None
    device_str = str(device)

    for sample_path in sample_paths:
        sample_name = sample_path.name if sample_path.is_dir() else sample_path.stem
        for model in models.values():
            _reset_tracker(model)
        if vel is not None:
            vel.reset()

        sources = list(_iter_sources(sample_path))
        for video_idx, source in enumerate(tqdm(sources, desc=f"Processing [{sample_name}]")):
            n_bins = _num_bins(source)
            stride = int(args.chunk_stride) if int(args.chunk_stride) > 0 else int(args.chunk_size)

            if "vel" in preprocessors:
                vel.reset()
                _reset_tracker(models["vel"])

            first_chunk = True
            cube_idx = 0
            frame_idx_by_pre = {name: 0 for name in preprocessors}
            out_dir = _output_dir(save_root, sample_name, video_idx)
            out_dir.mkdir(parents=True, exist_ok=True)
            video_key = f"{sample_name}/video{video_idx:05d}"

            for t0 in range(0, n_bins, stride):
                t1 = min(t0 + int(args.chunk_size), n_bins)
                raw_chunk = _slice_raw(source, t0, t1, packed_ch_order=args.packed_ch_order)
                if raw_chunk.shape[0] == 0:
                    continue

                chunk_bins = int(t1 - t0)
                for name in preprocessors:
                    if timing is not None:
                        with Profile(device=device) as pre_timer:
                            frames = _preprocess_chunk(
                                name,
                                raw_chunk,
                                packed_nch=source.packed_nch,
                                device=device,
                                first_chunk=first_chunk,
                                ppb=ppb,
                                hyb=hyb,
                                vel=vel,
                            )
                        timing.record(
                            name,
                            pre_timer.t,
                            video_key=video_key,
                            chunk_bins=chunk_bins,
                            cube_idx=cube_idx,
                            is_first_chunk=first_chunk,
                        )
                    else:
                        frames = _preprocess_chunk(
                            name,
                            raw_chunk,
                            packed_nch=source.packed_nch,
                            device=device,
                            first_chunk=first_chunk,
                            ppb=ppb,
                            hyb=hyb,
                            vel=vel,
                        )

                    frames_bgr = _rgb_tensor_to_bgr_u8(
                        frames,
                        vis_mode=args.vis_mode,
                        percentile=float(args.vis_percentile),
                        gamma=float(args.vis_gamma),
                    )
                    results = _run_detector_on_frames(
                        models[name],
                        frames_bgr,
                        conf=float(args.det_thresh),
                        tracker_cfg=tracker_cfg,
                        device=args.device,
                    )

                    for frame_bgr, result in zip(frames_bgr, results):
                        stem = (
                            f"cube{cube_idx:05d}_t{t0:06d}_{t1:06d}"
                            f"_frame{frame_idx_by_pre[name]:07d}_{name}"
                        )
                        recon_path = out_dir / f"{stem}_recon.png"
                        overlay_path = out_dir / f"{stem}_overlay.png"
                        cv2.imwrite(str(recon_path), frame_bgr)
                        cv2.imwrite(str(overlay_path), _draw_results(frame_bgr, result))
                        frame_idx_by_pre[name] += 1
                    if name == "vel":
                        _set_vel_field_from_tracker(models["vel"], vel)

                first_chunk = False
                cube_idx += 1

    if timing is not None:
        timing.print_summary()
        time_out = Path(args.time_pre_out) if args.time_pre_out else save_root / "preprocess_timing.json"
        timing.write_json(time_out, chunk_size=int(args.chunk_size), device=device_str)


if __name__ == "__main__":
    main()
