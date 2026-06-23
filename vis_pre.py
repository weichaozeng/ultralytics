"""Visualize SPAD preprocessors on packed binary `.npy` inputs.

This script is detector-free. It unpacks bit-packed binary arrays, runs one or
more preprocessors (`sum`, `ppb`, `vel`, `stea`, `hyb`), saves per-method
reconstructions, and writes side-by-side comparison mosaics.

Supported input layouts
-----------------------
- Single-channel packed video: `(T, H, Wpacked)` via `--bitdim`
- Three-channel packed video: `(T, H, Wpacked, 3)` via `--bitdim`

Examples
--------
python ultralytics/vis_pre.py \
  --in_path /path/to/binary.npy \
  --save_dir /tmp/vis_pre \
  --bitdim 2 \
  --expected_w 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data.spad_packed import (
    integrate_raw_chunk_to_rgb,
    raw_plane_to_photon_cube,
    sum_raw_chunk_to_rgb,
)
from ultralytics.quanta_hyb_networks.integrator import HybridSpatioTemporalEvidenceAccumulation
from ultralytics.quanta_neural_networks.integrator import PerPixelBayesian
from ultralytics.quanta_stea_networks.integrator import SpatioTemporalEvidenceAccumulation
from ultralytics.quanta_vel_networks.integrator import VelIntegrator


ALL_METHODS = ("sum", "ppb", "vel", "stea", "hyb")
LABEL_COLORS = {
    "sum": (0, 255, 255),
    "ppb": (0, 255, 0),
    "vel": (255, 128, 0),
    "stea": (255, 0, 255),
    "hyb": (128, 0, 255),
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize SPAD preprocessors on packed binary npy inputs")
    ap.add_argument("--in_path", type=Path, required=True, help="Directory containing binary.npy or a direct .npy path")
    ap.add_argument("--save_dir", type=Path, required=True, help="Output root")
    ap.add_argument("--pre", type=str, default="sum,ppb,vel,stea,hyb", help="Comma-separated preprocessors")
    ap.add_argument("--bitdim", type=int, default=2, help="0-based axis to unpack with np.unpackbits")
    ap.add_argument("--expected_w", type=int, default=512, help="Crop unpacked bit dimension to this width")
    ap.add_argument("--bitorder", type=str, default="big", choices=["big", "little"])
    ap.add_argument("--transpose_hw", action="store_true", help="Swap H/W after unpack for debugging geometry")
    ap.add_argument("--flip_x", action="store_true", help="Flip unpacked frames left-right for debugging geometry")
    ap.add_argument("--flip_y", action="store_true", help="Flip unpacked frames top-bottom for debugging geometry")
    ap.add_argument("--dump_unpacked", action="store_true", help="Save unpacked frames before any preprocessing")
    ap.add_argument("--dump_only", action="store_true", help="Only dump unpacked frames, skip preprocessors")
    ap.add_argument("--t0", type=int, default=0, help="First frame index to dump when --dump_unpacked is set")
    ap.add_argument("--max_dump_frames", type=int, default=8, help="Max unpacked frames to dump")
    ap.add_argument("--chunk_size", type=int, default=320)
    ap.add_argument("--chunk_stride", type=int, default=0, help="0 means equal to chunk_size")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--vis_mode", type=str, default="linear", choices=["linear", "gamma", "percentile", "percentile_gamma"])
    ap.add_argument("--vis_percentile", type=float, default=99.5)
    ap.add_argument("--vis_gamma", type=float, default=2.2)
    ap.add_argument("--gap", type=int, default=8, help="Pixels between compare panels")
    ap.add_argument("--label_h", type=int, default=28, help="Header height for compare labels")
    ap.add_argument("--font_scale", type=float, default=0.7)
    # PerPixelBayesian
    ap.add_argument("--ppb_gamma", type=float, default=5e-4)
    ap.add_argument("--ppb_quantile", type=float, default=1.0)
    ap.add_argument("--ppb_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppb_min_filter_size", type=int, default=7)
    # STEA
    ap.add_argument("--stea_fast_window", type=int, default=64)
    ap.add_argument("--stea_slow_window", type=int, default=128)
    ap.add_argument("--stea_temporal_window", type=int, default=5)
    ap.add_argument("--stea_fast_tau", type=float, default=6.0)
    ap.add_argument("--stea_motion_sharpness", type=float, default=60.0)
    ap.add_argument("--stea_motion_threshold", type=float, default=0.05)
    ap.add_argument("--stea_eps", type=float, default=1e-5)
    ap.add_argument("--stea_blend_const", type=float, default=16.0)
    ap.add_argument("--stea_kernel_size", type=int, default=None)
    ap.add_argument("--stea_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stea_quantile", type=float, default=1.0)
    # HYB
    ap.add_argument("--hyb_fast_window", type=int, default=64)
    ap.add_argument("--hyb_slow_window", type=int, default=128)
    ap.add_argument("--hyb_temporal_window", type=int, default=5)
    ap.add_argument("--hyb_fast_tau", type=float, default=6.0)
    ap.add_argument("--hyb_motion_sharpness", type=float, default=60.0)
    ap.add_argument("--hyb_motion_threshold", type=float, default=0.05)
    ap.add_argument("--hyb_eps", type=float, default=1e-5)
    ap.add_argument("--hyb_blend_const", type=float, default=16.0)
    ap.add_argument("--hyb_kernel_size", type=int, default=None)
    ap.add_argument("--hyb_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hyb_quantile", type=float, default=1.0)
    ap.add_argument("--hyb_warp_block_size", type=int, default=16)
    # VEL
    ap.add_argument("--vel_max_shift", type=int, default=16)
    ap.add_argument("--vel_patch_size", type=int, default=0)
    ap.add_argument("--vel_quantile", type=float, default=1.0)
    ap.add_argument("--vel_normalize", action=argparse.BooleanOptionalAction, default=False)
    return ap.parse_args()


def _resolve_input_npy(in_path: Path) -> Path:
    if in_path.is_dir():
        npy = in_path / "binary.npy"
        if not npy.exists():
            raise FileNotFoundError(f"Directory input requires binary.npy, not found: {npy}")
        return npy
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")
    if in_path.suffix.lower() != ".npy":
        raise ValueError(f"Unsupported input: {in_path} (expect directory or .npy)")
    return in_path


def _sample_name(path: Path) -> str:
    if path.is_dir():
        return path.name
    if path.name == "binary.npy":
        return path.parent.name
    return path.stem


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"bitdim={axis} is out of range for ndim={ndim}")
    return axis


def _crop_axis(arr: np.ndarray, axis: int, size: int) -> np.ndarray:
    if size <= 0 or arr.shape[axis] <= size:
        return arr
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(0, size)
    return arr[tuple(slices)]


def _load_unpacked(path: Path, *, bitdim: int, expected_w: int, bitorder: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected ndarray in {path}, got {type(arr)}")
    axis = _normalize_axis(int(bitdim), int(arr.ndim))
    unpacked = np.unpackbits(arr, axis=axis, bitorder=bitorder)
    unpacked = _crop_axis(unpacked, axis=axis, size=int(expected_w))
    return unpacked.astype(np.uint8, copy=False)


def _infer_input_kind(unpacked: np.ndarray) -> str:
    if unpacked.ndim == 3:
        return "single"
    if unpacked.ndim == 4 and unpacked.shape[-1] == 3:
        return "rgb_planes"
    if unpacked.ndim == 4 and unpacked.shape[-1] == 1:
        return "single"
    raise ValueError(
        f"Unsupported unpacked shape {unpacked.shape}. "
        "Expected THW for single-channel or THWC with C=3 for three-channel input."
    )


def _apply_spatial_debug_ops(
    unpacked: np.ndarray,
    *,
    transpose_hw: bool,
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    out = unpacked
    if transpose_hw:
        if out.ndim == 3:
            out = np.transpose(out, (0, 2, 1))
        elif out.ndim == 4:
            out = np.transpose(out, (0, 2, 1, 3))
        else:
            raise ValueError(f"Unsupported ndim for transpose_hw: {out.ndim}")
    if flip_x:
        out = np.flip(out, axis=2)
    if flip_y:
        out = np.flip(out, axis=1)
    return np.ascontiguousarray(out)


def _rgb_planes_to_raw_video(unpacked: np.ndarray, *, ch_order: str) -> np.ndarray:
    if unpacked.ndim != 4 or unpacked.shape[-1] != 3:
        raise ValueError(f"Expected unpacked RGB planes (T,H,W,3), got {unpacked.shape}")
    t, h, w, _ = unpacked.shape
    raw = np.zeros((t, h * 2, w * 2), dtype=np.uint8)
    if ch_order.upper() == "BGR":
        r_ch, g_ch, b_ch = 2, 1, 0
    else:
        r_ch, g_ch, b_ch = 0, 1, 2
    raw[:, 0::2, 0::2] = unpacked[:, :, :, r_ch]
    raw[:, 0::2, 1::2] = unpacked[:, :, :, g_ch]
    raw[:, 1::2, 0::2] = unpacked[:, :, :, g_ch]
    raw[:, 1::2, 1::2] = unpacked[:, :, :, b_ch]
    return raw[:, :, :, None]


def _single_to_raw_video(unpacked: np.ndarray) -> np.ndarray:
    if unpacked.ndim == 4 and unpacked.shape[-1] == 1:
        return np.ascontiguousarray(unpacked.astype(np.uint8, copy=False))
    if unpacked.ndim != 3:
        raise ValueError(f"Expected single-channel unpacked THW, got {unpacked.shape}")
    return np.ascontiguousarray(unpacked[:, :, :, None].astype(np.uint8, copy=False))


def _resolve_device(device: str) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _build_integrators(args: argparse.Namespace, device: torch.device, preprocessors: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    if "ppb" in preprocessors:
        out["ppb"] = PerPixelBayesian(
            subsampling=int(args.chunk_size),
            bocpd_gamma=float(args.ppb_gamma),
            normalize=bool(args.ppb_normalize),
            quantile=float(args.ppb_quantile),
            min_filter_size=int(args.ppb_min_filter_size),
        ).to(device)
    if "vel" in preprocessors:
        out["vel_rgb"] = VelIntegrator(
            chunk_size=int(args.chunk_size),
            max_shift=int(args.vel_max_shift),
            patch_size=int(args.vel_patch_size),
            compensate_space="rgb",
            normalize=bool(args.vel_normalize),
            quantile=float(args.vel_quantile),
        ).to(device)
        out["vel_raw"] = VelIntegrator(
            chunk_size=int(args.chunk_size),
            max_shift=int(args.vel_max_shift),
            patch_size=int(args.vel_patch_size),
            compensate_space="raw",
            normalize=bool(args.vel_normalize),
            quantile=float(args.vel_quantile),
        ).to(device)
    if "stea" in preprocessors:
        out["stea"] = SpatioTemporalEvidenceAccumulation(
            chunk_size=int(args.chunk_size),
            fast_window=int(args.stea_fast_window),
            slow_window=int(args.stea_kernel_size or args.stea_slow_window),
            temporal_window=int(args.stea_temporal_window),
            fast_tau=float(args.stea_fast_tau),
            motion_sharpness=float(args.stea_motion_sharpness),
            motion_threshold=float(args.stea_motion_threshold),
            eps=float(args.stea_eps),
            stable_prior=float(args.stea_blend_const),
            subsampling=int(args.chunk_size),
            normalize=bool(args.stea_normalize),
            quantile=float(args.stea_quantile),
        ).to(device)
    if "hyb" in preprocessors:
        out["hyb"] = HybridSpatioTemporalEvidenceAccumulation(
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
            warp_block_size=int(args.hyb_warp_block_size),
        ).to(device)
    return out


def _reset_integrator_states(integrators: dict[str, object]) -> None:
    for integrator in integrators.values():
        reset = getattr(integrator, "reset", None)
        if callable(reset):
            reset()


def _gray_hwt_to_chw(recons: torch.Tensor) -> torch.Tensor:
    if recons.ndim != 3:
        raise ValueError(f"Expected grayscale reconstructions (H,W,T), got {tuple(recons.shape)}")
    if int(recons.shape[-1]) <= 0:
        h, w = map(int, recons.shape[:2])
        return recons.new_zeros((0, 1, h, w))
    return recons.permute(2, 0, 1).unsqueeze(1).contiguous().float()


def _gray_chunk_to_cube(raw_chunk: np.ndarray, device: torch.device) -> torch.Tensor:
    plane = np.ascontiguousarray(raw_chunk[..., 0])
    return raw_plane_to_photon_cube(plane, device=device, as_bool=True)


def _preprocess_sum_gray(raw_chunk: np.ndarray, *, device: torch.device) -> torch.Tensor:
    raw = torch.from_numpy(np.ascontiguousarray(raw_chunk[..., 0])).to(device).float()
    mean_hw = raw.mean(dim=0, keepdim=True).unsqueeze(1)
    return mean_hw.clamp(0, 1)


def _preprocess_ppb_gray(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: PerPixelBayesian,
    clear_states: bool,
) -> torch.Tensor:
    cube = _gray_chunk_to_cube(raw_chunk, device)
    recons = integrator.process_photon_cube(cube, clear_states=clear_states)
    return _gray_hwt_to_chw(recons)[-1:].contiguous()


def _preprocess_stea_gray(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: SpatioTemporalEvidenceAccumulation,
    clear_states: bool,
) -> torch.Tensor:
    cube = _gray_chunk_to_cube(raw_chunk, device)
    recons = integrator.process_photon_cube(cube, clear_states=clear_states)
    return _gray_hwt_to_chw(recons)[-1:].contiguous()


def _preprocess_hyb_gray(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: HybridSpatioTemporalEvidenceAccumulation,
    clear_states: bool,
) -> torch.Tensor:
    integrator.set_velocity_field(None, source_space="rgb")
    cube = _gray_chunk_to_cube(raw_chunk, device)
    recons = integrator.process_photon_cube(cube, clear_states=clear_states)
    return _gray_hwt_to_chw(recons)[-1:].contiguous()


def _preprocess_vel_gray(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: VelIntegrator,
    clear_states: bool,
) -> torch.Tensor:
    integrator.set_velocity_field(None, source_space="raw")
    cube = _gray_chunk_to_cube(raw_chunk, device)
    recons = integrator.process_photon_cube(cube, clear_states=clear_states)
    if recons.ndim == 2:
        return recons.unsqueeze(0).unsqueeze(0).float().contiguous()
    if recons.ndim == 3 and int(recons.shape[-1]) == 1:
        return recons.permute(2, 0, 1).unsqueeze(1).float().contiguous()
    raise ValueError(f"Unexpected vel grayscale output shape: {tuple(recons.shape)}")


def _preprocess_sum_rgb(raw_chunk: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return sum_raw_chunk_to_rgb(raw_chunk, packed_nch=3, device=device)


def _preprocess_ppb_rgb(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: PerPixelBayesian,
    clear_states: bool,
) -> torch.Tensor:
    return integrate_raw_chunk_to_rgb(integrator, raw_chunk, packed_nch=3, device=device, clear_states=clear_states)


def _preprocess_stea_rgb(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: SpatioTemporalEvidenceAccumulation,
    clear_states: bool,
) -> torch.Tensor:
    return integrate_raw_chunk_to_rgb(integrator, raw_chunk, packed_nch=3, device=device, clear_states=clear_states)


def _preprocess_hyb_rgb(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: HybridSpatioTemporalEvidenceAccumulation,
    clear_states: bool,
) -> torch.Tensor:
    integrator.set_velocity_field(None, source_space="rgb")
    return integrate_raw_chunk_to_rgb(integrator, raw_chunk, packed_nch=3, device=device, clear_states=clear_states)


def _preprocess_vel_rgb(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: VelIntegrator,
    clear_states: bool,
) -> torch.Tensor:
    integrator.set_velocity_field(None, source_space="rgb")
    cube = _gray_chunk_to_cube(raw_chunk, device)
    recons = integrator.process_photon_cube(cube, clear_states=clear_states, packed_nch=3)
    if integrator.outputs_rgb:
        return recons.unsqueeze(0).contiguous()
    raise ValueError("RGB velocity path expected outputs_rgb=True")


def _apply_vis_scaling(x: np.ndarray, *, mode: str, gamma: float, percentile: float) -> np.ndarray:
    if mode == "linear":
        vis = np.clip(x, 0.0, 1.0)
    elif mode == "gamma":
        vis = np.power(np.clip(x, 0.0, 1.0), 1.0 / gamma)
    elif mode == "percentile":
        scale = float(np.percentile(x, percentile))
        vis = np.clip(x / max(scale, 1e-6), 0.0, 1.0)
    elif mode == "percentile_gamma":
        scale = float(np.percentile(x, percentile))
        vis = np.clip(x / max(scale, 1e-6), 0.0, 1.0)
        vis = np.power(vis, 1.0 / gamma)
    else:
        raise ValueError(f"Unsupported vis mode: {mode}")
    return vis


def _frames_tensor_to_bgr_u8(
    frames_tchw: torch.Tensor,
    *,
    vis_mode: str,
    percentile: float,
    gamma: float,
) -> list[np.ndarray]:
    if frames_tchw.ndim != 4 or int(frames_tchw.shape[1]) not in {1, 3}:
        raise ValueError(f"Expected (N,1|3,H,W) tensor, got shape={tuple(frames_tchw.shape)}")
    frames = frames_tchw.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    out: list[np.ndarray] = []
    for frame in frames:
        vis = _apply_vis_scaling(frame, mode=vis_mode, gamma=gamma, percentile=percentile)
        if vis.shape[2] == 1:
            vis = np.repeat(vis, 3, axis=2)
        bgr = np.ascontiguousarray((vis * 255.0).round().astype(np.uint8)[:, :, ::-1])
        out.append(bgr)
    return out


def _unpacked_frame_to_bgr(frame: np.ndarray, *, kind: str) -> np.ndarray:
    if kind == "single":
        if frame.ndim != 2:
            raise ValueError(f"Expected single frame HW, got {frame.shape}")
        u8 = (frame.astype(np.uint8) * 255) if frame.dtype != np.uint8 or frame.max() <= 1 else frame.astype(np.uint8)
        return np.repeat(u8[:, :, None], 3, axis=2)

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB-plane frame HWC, got {frame.shape}")
    rgb = (frame.astype(np.uint8) * 255) if frame.max() <= 1 else frame.astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _dump_unpacked_frames(
    unpacked: np.ndarray,
    *,
    kind: str,
    out_dir: Path,
    t0: int,
    max_dump_frames: int,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(unpacked.shape[0])
    start = max(int(t0), 0)
    stop = min(start + max(int(max_dump_frames), 0), n)
    n_written = 0
    for ti in range(start, stop):
        frame = unpacked[ti] if kind == "single" else unpacked[ti, :, :, :]
        bgr = _unpacked_frame_to_bgr(frame, kind=kind)
        cv2.imwrite(str(out_dir / f"t{ti:06d}_unpacked.png"), bgr)
        n_written += 1
    return n_written


def _label_panel(img_bgr: np.ndarray, text: str, color: tuple[int, int, int], label_h: int, font_scale: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    header = np.zeros((label_h, w, 3), dtype=np.uint8)
    cv2.putText(
        header,
        text.upper(),
        (8, int(label_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )
    return np.vstack([header, img_bgr])


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = max(int(round(w * target_h / h)), 1)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _stitch_row(
    panels: list[np.ndarray],
    labels: list[str],
    *,
    gap: int,
    label_h: int,
    font_scale: float,
) -> np.ndarray:
    target_h = max(p.shape[0] for p in panels)
    resized = [_resize_to_height(p, target_h) for p in panels]
    labeled = [
        _label_panel(img, label, LABEL_COLORS.get(label, (255, 255, 255)), label_h, font_scale)
        for img, label in zip(resized, labels)
    ]
    sep = np.full((labeled[0].shape[0], gap, 3), 32, dtype=np.uint8)
    out = labeled[0]
    for nxt in labeled[1:]:
        out = np.hstack([out, sep, nxt])
    return out


def _preprocess_chunk(
    method: str,
    raw_chunk: np.ndarray,
    *,
    kind: str,
    device: torch.device,
    first_chunk: bool,
    integrators: dict[str, object],
) -> torch.Tensor:
    if kind == "single":
        if method == "sum":
            return _preprocess_sum_gray(raw_chunk, device=device)
        if method == "ppb":
            return _preprocess_ppb_gray(raw_chunk, device=device, integrator=integrators["ppb"], clear_states=first_chunk)
        if method == "stea":
            return _preprocess_stea_gray(raw_chunk, device=device, integrator=integrators["stea"], clear_states=first_chunk)
        if method == "hyb":
            return _preprocess_hyb_gray(raw_chunk, device=device, integrator=integrators["hyb"], clear_states=first_chunk)
        return _preprocess_vel_gray(raw_chunk, device=device, integrator=integrators["vel_raw"], clear_states=first_chunk)

    if method == "sum":
        return _preprocess_sum_rgb(raw_chunk, device=device)
    if method == "ppb":
        return _preprocess_ppb_rgb(raw_chunk, device=device, integrator=integrators["ppb"], clear_states=first_chunk)
    if method == "stea":
        return _preprocess_stea_rgb(raw_chunk, device=device, integrator=integrators["stea"], clear_states=first_chunk)
    if method == "hyb":
        return _preprocess_hyb_rgb(raw_chunk, device=device, integrator=integrators["hyb"], clear_states=first_chunk)
    return _preprocess_vel_rgb(raw_chunk, device=device, integrator=integrators["vel_rgb"], clear_states=first_chunk)


def main() -> None:
    args = _parse_args()
    npy = _resolve_input_npy(args.in_path)
    preprocessors = [x.strip().lower() for x in args.pre.split(",") if x.strip()]
    invalid = sorted(set(preprocessors) - set(ALL_METHODS))
    if invalid:
        raise ValueError(f"Unsupported preprocessors: {invalid}")
    if not preprocessors:
        raise ValueError("No preprocessors selected")

    unpacked = _load_unpacked(npy, bitdim=int(args.bitdim), expected_w=int(args.expected_w), bitorder=str(args.bitorder))
    unpacked = _apply_spatial_debug_ops(
        unpacked,
        transpose_hw=bool(args.transpose_hw),
        flip_x=bool(args.flip_x),
        flip_y=bool(args.flip_y),
    )
    kind = _infer_input_kind(unpacked)
    raw_video = _single_to_raw_video(unpacked) if kind == "single" else _rgb_planes_to_raw_video(unpacked, ch_order=args.packed_ch_order)
    n_bins = int(raw_video.shape[0])
    stride = int(args.chunk_stride) if int(args.chunk_stride) > 0 else int(args.chunk_size)
    device = _resolve_device(args.device)
    integrators = _build_integrators(args, device, preprocessors)
    _reset_integrator_states(integrators)

    sample = _sample_name(args.in_path if args.in_path.exists() else npy)
    out_dir = args.save_dir / sample / "video00000"
    compare_dir = args.save_dir / f"{sample}_compare" / "video00000"
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {npy}")
    print(f"unpacked: shape={unpacked.shape} kind={kind}")
    print(f"raw_video: shape={raw_video.shape}")
    print(f"writing outputs to: {out_dir}")
    print(
        "debug_ops:"
        f" bitorder={args.bitorder}"
        f" transpose_hw={bool(args.transpose_hw)}"
        f" flip_x={bool(args.flip_x)}"
        f" flip_y={bool(args.flip_y)}"
    )

    if args.dump_unpacked:
        dump_dir = args.save_dir / sample / "unpacked_debug"
        dumped = _dump_unpacked_frames(
            unpacked,
            kind=kind,
            out_dir=dump_dir,
            t0=int(args.t0),
            max_dump_frames=int(args.max_dump_frames),
        )
        print(f"Dumped {dumped} unpacked frame(s) to {dump_dir}")
        if args.dump_only:
            return

    first_chunk = True
    frame_idx = 0
    cube_idx = 0
    for t0 in range(0, n_bins, stride):
        t1 = min(t0 + int(args.chunk_size), n_bins)
        raw_chunk = np.ascontiguousarray(raw_video[t0:t1])
        if raw_chunk.shape[0] == 0:
            continue

        stem = f"cube{cube_idx:05d}_t{t0:06d}_{t1:06d}_frame{frame_idx:07d}"
        labels: list[str] = []
        panels: list[np.ndarray] = []
        for method in preprocessors:
            frames = _preprocess_chunk(
                method,
                raw_chunk,
                kind=kind,
                device=device,
                first_chunk=first_chunk,
                integrators=integrators,
            )
            frames_bgr = _frames_tensor_to_bgr_u8(
                frames,
                vis_mode=args.vis_mode,
                percentile=float(args.vis_percentile),
                gamma=float(args.vis_gamma),
            )
            if not frames_bgr:
                continue
            frame_bgr = frames_bgr[-1]
            cv2.imwrite(str(out_dir / f"{stem}_{method}_recon.png"), frame_bgr)
            labels.append(method)
            panels.append(frame_bgr)

        if panels:
            mosaic = _stitch_row(
                panels,
                labels,
                gap=int(args.gap),
                label_h=int(args.label_h),
                font_scale=float(args.font_scale),
            )
            cv2.imwrite(str(compare_dir / f"{stem}_compare_recon.png"), mosaic)

        first_chunk = False
        frame_idx += 1
        cube_idx += 1

    print(f"Done. Wrote method PNGs to {out_dir}")
    print(f"Done. Wrote comparison PNGs to {compare_dir}")


if __name__ == "__main__":
    main()
