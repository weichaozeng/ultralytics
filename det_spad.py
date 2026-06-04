# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compare SPAD preprocessing methods before a standard YOLO pose detector.

This script does not use QNNPoseModel. It converts SPAD chunks into RGB-like
frames with one of:
- sum: temporal mean over the chunk
- ppb: PerPixelBayesian reconstruction
- vel: detection-guided velocity-compensated integration

- hyb: GatedMultiScaleEMA (parallel multi-scale FIR + soft routing)

The resulting frames are passed to an unmodified pretrained YOLO pose model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data.spad_packed import infer_packed_nch, is_packed_spad, packed_frames_to_raw_bayer
from ultralytics.quanta_hybrid_networks.integrator import GatedMultiScaleEMA
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
        raw = packed_frames_to_raw_bayer(packed, ch_order=packed_ch_order)
        return raw[:, :, :, None]
    if source.layout == "thwc1":
        return np.ascontiguousarray(source.array[t0:t1].astype(np.uint8, copy=False))
    if source.layout == "thw":
        return np.ascontiguousarray(source.array[t0:t1, :, :, None].astype(np.uint8, copy=False))
    if source.layout == "hwt":
        return np.ascontiguousarray(np.transpose(source.array[:, :, t0:t1], (2, 0, 1))[:, :, :, None].astype(np.uint8, copy=False))
    raise ValueError(f"Unsupported layout: {source.layout}")


def _raw_hwt_to_rgb_float(raw_hwt: torch.Tensor, *, packed_nch: int) -> torch.Tensor:
    """Convert raw Bayer H,W,T float [0,1] to T,3,H/2,W/2."""
    h_raw, w_raw, t = map(int, raw_hwt.shape)
    if int(packed_nch) == 3:
        r = raw_hwt[0::2, 0::2, :]
        g = 0.5 * (raw_hwt[0::2, 1::2, :] + raw_hwt[1::2, 0::2, :])
        b = raw_hwt[1::2, 1::2, :]
        return torch.stack((r, g, b), dim=0).permute(3, 0, 1, 2).contiguous()

    raw_np = raw_hwt.detach().float().cpu().numpy()
    frames = []
    for ti in range(t):
        raw_u8 = np.clip(raw_np[:, :, ti] * 255.0, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(raw_u8, cv2.COLOR_BAYER_RG2RGB)
        rgb = cv2.resize(rgb, (w_raw // 2, h_raw // 2), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    return torch.stack(frames, dim=0).to(raw_hwt.device)


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
    raw = torch.from_numpy(raw_chunk[:, :, :, 0]).to(device).permute(1, 2, 0).float()
    raw_mean = raw.mean(dim=2, keepdim=True).clamp(0, 1)
    return _raw_hwt_to_rgb_float(raw_mean, packed_nch=packed_nch)


def _preprocess_ppb(raw_chunk: np.ndarray, *, packed_nch: int, device: torch.device, integrator: PerPixelBayesian, clear_states: bool, **kwargs) -> torch.Tensor:
    raw = torch.from_numpy(raw_chunk[:, :, :, 0]).to(device).permute(1, 2, 0).bool()
    recons = integrator.process_photon_cube(raw, clear_states=clear_states)
    return _raw_hwt_to_rgb_float(recons, packed_nch=packed_nch)


def _preprocess_hyb(
    raw_chunk: np.ndarray,
    *,
    packed_nch: int,
    device: torch.device,
    integrator: GatedMultiScaleEMA,
    clear_states: bool,
    **kwargs,
) -> torch.Tensor:
    raw = torch.from_numpy(raw_chunk[:, :, :, 0]).to(device).permute(1, 2, 0).bool()
    recons = integrator.process_photon_cube(raw, clear_states=clear_states)
    return _raw_hwt_to_rgb_float(recons, packed_nch=packed_nch)


def _preprocess_vel(raw_chunk: np.ndarray, *, packed_nch: int, device: torch.device, integrator: VelIntegrator, clear_states: bool, **kwargs) -> torch.Tensor:
    raw = torch.from_numpy(raw_chunk[:, :, :, 0]).to(device).permute(1, 2, 0).bool()
    recons = integrator.process_photon_cube(raw, clear_states=clear_states, packed_nch=packed_nch)
    if integrator.outputs_rgb:
        return recons.unsqueeze(0)
    return _raw_hwt_to_rgb_float(recons, packed_nch=packed_nch)


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
    ap.add_argument("--tracker", type=str, default="botsort", choices=["bytetrack", "botsort"])
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
   # perpixelbayesian
    ap.add_argument("--ppb_gamma", type=float, default=5e-4)
    ap.add_argument("--ppb_quantile", type=float, default=1.0)
    ap.add_argument("--ppb_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppb_min_filter_size", type=int, default=7)
    # gated multi-scale EMA (hybrid)
    ap.add_argument("--hyb_kernel_size", type=int, default=64, help="FIR kernel length for hyb integrator")
    ap.add_argument("--hyb_v_threshold", type=float, default=0.1)
    ap.add_argument("--hyb_gating_sharpness", type=float, default=20.0)
    ap.add_argument("--hyb_gating_tau", type=float, default=0.1)
    ap.add_argument("--hyb_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hyb_quantile", type=float, default=1.0)
    # velintegrator
    ap.add_argument("--vel_max_shift", type=int, default=16)
    ap.add_argument("--vel_patch_size", type=int, default=0, help="Per-patch vel from tracks in patch (0 = global median)")
    ap.add_argument("--vel_compensate", type=str, default="rgb", choices=["rgb", "raw"], help="Shift in RGB (avoids Bayer color fringing) or raw")
    ap.add_argument("--vel_quantile", type=float, default=1.0)
    ap.add_argument("--vel_normalize", action=argparse.BooleanOptionalAction, default=False)
    # vis
    ap.add_argument("--vis_mode", type=str, default="linear", choices=["linear", "gamma", "percentile", "percentile_gamma"])
    ap.add_argument("--vis_percentile", type=float, default=99.5)
    ap.add_argument("--vis_gamma", type=float, default=2.2)
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

    device = _resolve_device(args.device)
    tracker_cfg = f"{args.tracker}.yaml"

    models = {name: YOLO(args.ckpt) for name in preprocessors}
    ppb = PerPixelBayesian(
        subsampling=int(args.chunk_size),
        bocpd_gamma=float(args.ppb_gamma),
        normalize=bool(args.ppb_normalize),
        quantile=float(args.ppb_quantile),
        min_filter_size=int(args.ppb_min_filter_size),
    ).to(device)
    vel = VelIntegrator(
        chunk_size=int(args.chunk_size),
        max_shift=int(args.vel_max_shift),
        patch_size=int(args.vel_patch_size),
        compensate_space=str(args.vel_compensate),
        normalize=bool(args.vel_normalize),
        quantile=float(args.vel_quantile),
    ).to(device)
    hyb = GatedMultiScaleEMA(
        chunk_size=int(args.chunk_size),
        kernel_size=int(args.hyb_kernel_size),
        subsampling=int(args.chunk_size),
        v_threshold=float(args.hyb_v_threshold),
        gating_sharpness=float(args.hyb_gating_sharpness),
        gating_tau=float(args.hyb_gating_tau),
        normalize=bool(args.hyb_normalize),
        quantile=float(args.hyb_quantile),
    ).to(device)

    for sample_path in sample_paths:
        sample_name = sample_path.name if sample_path.is_dir() else sample_path.stem
        for model in models.values():
            _reset_tracker(model)
        ppb.reset() if hasattr(ppb, "reset") else None
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

            for t0 in range(0, n_bins, stride):
                t1 = min(t0 + int(args.chunk_size), n_bins)
                raw_chunk = _slice_raw(source, t0, t1, packed_ch_order=args.packed_ch_order)
                if raw_chunk.shape[0] == 0:
                    continue

                for name in preprocessors:
                    if name == "sum":
                        frames = _preprocess_sum(raw_chunk, packed_nch=source.packed_nch, device=device)
                    elif name == "ppb":
                        frames = _preprocess_ppb(raw_chunk, packed_nch=source.packed_nch, device=device, integrator=ppb, clear_states=first_chunk)
                    elif name == "hyb":
                        frames = _preprocess_hyb(raw_chunk, packed_nch=source.packed_nch, device=device, integrator=hyb, clear_states=first_chunk)
                    else:
                        frames = _preprocess_vel(raw_chunk, packed_nch=source.packed_nch, device=device, integrator=vel, clear_states=first_chunk)

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

                    raw_hw = (int(raw_chunk.shape[1]), int(raw_chunk.shape[2]))
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
                            track_ids, centers = _extract_track_centers(result)
                            vel.push_detection(
                                track_ids,
                                centers,
                                det_hw=frame_bgr.shape[:2],
                                raw_hw=raw_hw,
                            )

                first_chunk = False
                cube_idx += 1


if __name__ == "__main__":
    main()
