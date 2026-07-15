"""Visualize SPAD preprocessors on packed binary `.npy` inputs.

This script is detector-free. It unpacks bit-packed binary arrays, runs one or
more preprocessors (`sum`, `ema`, `ppb`, `stea`, `hire`), saves per-method
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
    is_packed_spad,
    packed_frames_to_raw_video,
    raw_plane_to_photon_cube,
)
from ultralytics.models.yolo.pose.spad_preprocessors import EmaPreprocessor, SumPreprocessor
from ultralytics.quanta_hire_networks.integrator import HIRE
from ultralytics.quanta_neural_networks.integrator import PerPixelBayesian
from ultralytics.quanta_stea_networks.integrator import SpatioTemporalEvidenceAccumulation


ALL_METHODS = ("sum", "ema", "ppb", "stea", "hire")
LABEL_COLORS = {
    "sum": (0, 255, 255),
    "ema": (255, 165, 0),
    "ppb": (0, 255, 0),
    "stea": (255, 0, 255),
    "hire": (0, 200, 255),
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize SPAD preprocessors on packed binary npy inputs")
    ap.add_argument("--in_path", type=Path, required=True, help="Directory containing binary.npy or a direct .npy path")
    ap.add_argument("--save_dir", type=Path, required=True, help="Output root")
    ap.add_argument("--pre", type=str, default="sum,ema,ppb,stea,hire", help="Comma-separated preprocessors")
    ap.add_argument("--bitdim", type=int, default=2, help="0-based axis to unpack with np.unpackbits")
    ap.add_argument("--expected_w", type=int, default=512, help="Crop unpacked bit dimension to this width")
    ap.add_argument("--bitorder", type=str, default="big", choices=["big", "little"])
    ap.add_argument("--flip_x", action="store_true", help="Flip frames left-right before preprocessing")
    ap.add_argument("--flip_y", action="store_true", help="Flip frames top-bottom before preprocessing")
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=320,
        help=(
            "Raw bins per output frame (= preprocessor emit interval). "
            "Example: 2 kHz SPAD at 25 FPS => chunk_size=80; 8 kHz at 125 FPS => 64."
        ),
    )
    ap.add_argument("--chunk_stride", type=int, default=0, help="0 means equal to chunk_size")
    ap.add_argument(
        "--independent-chunks",
        action="store_true",
        help=(
            "Reset PPB/STEA/EMA/HIRE state on every chunk (each window independent, like sum). "
            "Default keeps causal state across contiguous non-overlapping chunks so background "
            "statistics can stabilize — the intended online behavior."
        ),
    )
    ap.add_argument("--max_bins", type=int, default=0, help="Process at most this many time bins (0 = all)")
    ap.add_argument(
        "--max_chunks",
        type=int,
        default=0,
        help=(
            "Save at most this many output frames per sample (0 = all). "
            "When >0, chunks are evenly spaced (non-contiguous) and state is always reset."
        ),
    )
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
    # Shared dynamic-range stretch (align brightness across methods)
    ap.add_argument("--sum_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sum_quantile", type=float, default=1.0)
    ap.add_argument("--ema_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ema_quantile", type=float, default=1.0)
    ap.add_argument("--hire_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hire_quantile", type=float, default=1.0)
    # EMA
    ap.add_argument(
        "--ema_alpha",
        type=float,
        default=0.0,
        help="EMA new-sample weight. <=0 uses 2/(chunk_size+1) SMA-equivalent default.",
    )
    # STEA
    ap.add_argument("--stea_fast_window", type=int, default=16)
    ap.add_argument("--stea_slow_window", type=int, default=128)
    ap.add_argument("--stea_temporal_window", type=int, default=5)
    ap.add_argument("--stea_fast_tau", type=float, default=6.0)
    ap.add_argument("--stea_motion_sharpness", type=float, default=60.0)
    ap.add_argument("--stea_motion_threshold", type=float, default=0.07)
    ap.add_argument("--stea_eps", type=float, default=1e-5)
    ap.add_argument("--stea_blend_const", type=float, default=16.0)
    ap.add_argument("--stea_kernel_size", type=int, default=None)
    ap.add_argument("--stea_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stea_quantile", type=float, default=1.0)
    # HIRE (primary knob: SPAD bin rate; taus from reference-bin presets)
    ap.add_argument(
        "--bin_rate_hz",
        type=float,
        default=8000.0,
        help="SPAD bin sample rate f_s for HIRE ZOH alphas.",
    )
    ap.add_argument("--hire_ref_rate_hz", type=float, default=8000.0)
    ap.add_argument("--hire_fast_bins", type=int, default=16)
    ap.add_argument("--hire_slow_bins", type=int, default=128)
    ap.add_argument("--hire_surprise_bins", type=int, default=8)
    ap.add_argument("--hire_tau_fast", type=float, default=0.0)
    ap.add_argument("--hire_tau_slow", type=float, default=0.0)
    ap.add_argument("--hire_tau_surprise", type=float, default=0.0)
    ap.add_argument("--hire_gate_theta", type=float, default=0.2)
    ap.add_argument("--hire_theta_on", type=float, default=0.15)
    ap.add_argument("--hire_theta_off", type=float, default=0.06)
    ap.add_argument("--hire_confirm_bins", type=int, default=5)
    ap.add_argument("--hire_cooldown_bins", type=int, default=8)
    ap.add_argument("--hire_spatial_kernel", type=int, default=3)
    ap.add_argument("--hire_eps", type=float, default=1e-5)
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


def _load_packed_mmap(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected ndarray in {path}, got {type(arr)}")
    return arr


def _load_unpacked(path: Path, *, bitdim: int, expected_w: int, bitorder: str) -> np.ndarray:
    arr = _load_packed_mmap(path)
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


def _apply_spatial_preprocess_ops(
    unpacked: np.ndarray,
    *,
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    out = unpacked
    if flip_x:
        out = np.flip(out, axis=2)
    if flip_y:
        out = np.flip(out, axis=1)
    return np.ascontiguousarray(out)


def _apply_spatial_preprocess_ops_raw(
    raw_chunk: np.ndarray,
    *,
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    out = raw_chunk
    if flip_x:
        out = np.flip(out, axis=2)
    if flip_y:
        out = np.flip(out, axis=1)
    return np.ascontiguousarray(out)


def _generic_packed_slice_to_raw_chunk(
    packed_slice: np.ndarray,
    *,
    bitdim: int,
    expected_w: int,
    bitorder: str,
    ch_order: str,
) -> tuple[np.ndarray, str]:
    axis = _normalize_axis(int(bitdim), int(packed_slice.ndim))
    unpacked = np.unpackbits(packed_slice, axis=axis, bitorder=bitorder)
    unpacked = _crop_axis(unpacked, axis=axis, size=int(expected_w))
    unpacked = unpacked.astype(np.uint8, copy=False)
    kind = _infer_input_kind(unpacked)
    if kind == "single":
        raw_chunk = _single_to_raw_video(unpacked)
    else:
        raw_chunk = _rgb_planes_to_raw_video(unpacked, ch_order=ch_order)
    return raw_chunk, kind


def _packed_time_range_to_raw_chunk(
    packed: np.ndarray,
    t0: int,
    t1: int,
    *,
    bitdim: int,
    expected_w: int,
    bitorder: str,
    ch_order: str,
) -> tuple[np.ndarray, str]:
    packed_slice = np.ascontiguousarray(packed[t0:t1])
    if packed_slice.shape[0] == 0:
        raise ValueError(f"Empty packed slice [{t0}:{t1})")
    if is_packed_spad(packed):
        raw_chunk = packed_frames_to_raw_video(
            packed_slice,
            expected_w=int(expected_w),
            ch_order=ch_order,
        )
        return raw_chunk, "rgb_planes"
    return _generic_packed_slice_to_raw_chunk(
        packed_slice,
        bitdim=bitdim,
        expected_w=expected_w,
        bitorder=bitorder,
        ch_order=ch_order,
    )


def _infer_layout_from_packed(packed: np.ndarray, *, bitdim: int, expected_w: int, bitorder: str, ch_order: str) -> str:
    if is_packed_spad(packed):
        return "rgb_planes"
    _, kind = _packed_time_range_to_raw_chunk(
        packed,
        0,
        1,
        bitdim=bitdim,
        expected_w=expected_w,
        bitorder=bitorder,
        ch_order=ch_order,
    )
    return kind


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
    """Build stateful preprocessors.

    In vis_pre, ``--chunk_size`` is the emit interval (bins per output frame), e.g.
    80 for 2 kHz -> 25 FPS. That is exactly what PPB/STEA/EMA/HIRE ``subsampling`` means
    here — not the training ``spad_subsampling=64`` (8 kHz / 125 Hz alignment).
    """
    emit = max(int(args.chunk_size), 1)
    out: dict[str, object] = {}
    if "sum" in preprocessors:
        out["sum"] = SumPreprocessor(
            subsampling=emit,
            normalize=bool(args.sum_normalize),
            quantile=float(args.sum_quantile),
        ).to(device)
    if "ema" in preprocessors:
        out["ema"] = EmaPreprocessor(
            subsampling=emit,
            ema_alpha=float(args.ema_alpha),
            normalize=bool(args.ema_normalize),
            quantile=float(args.ema_quantile),
        ).to(device)
    if "ppb" in preprocessors:
        out["ppb"] = PerPixelBayesian(
            subsampling=emit,
            bocpd_gamma=float(args.ppb_gamma),
            normalize=bool(args.ppb_normalize),
            quantile=float(args.ppb_quantile),
            min_filter_size=int(args.ppb_min_filter_size),
        ).to(device)
    if "stea" in preprocessors:
        out["stea"] = SpatioTemporalEvidenceAccumulation(
            chunk_size=emit,
            fast_window=int(args.stea_fast_window),
            slow_window=int(args.stea_kernel_size or args.stea_slow_window),
            temporal_window=int(args.stea_temporal_window),
            fast_tau=float(args.stea_fast_tau),
            motion_sharpness=float(args.stea_motion_sharpness),
            motion_threshold=float(args.stea_motion_threshold),
            eps=float(args.stea_eps),
            stable_prior=float(args.stea_blend_const),
            subsampling=emit,
            normalize=bool(args.stea_normalize),
            quantile=float(args.stea_quantile),
        ).to(device)
    if "hire" in preprocessors:
        def _tau_or_none(val: float) -> float | None:
            return None if float(val) <= 0.0 else float(val)

        out["hire"] = HIRE(
            subsampling=emit,
            sample_rate_hz=float(args.bin_rate_hz),
            ref_rate_hz=float(args.hire_ref_rate_hz),
            fast_bins=int(args.hire_fast_bins),
            slow_bins=int(args.hire_slow_bins),
            surprise_bins=int(args.hire_surprise_bins),
            tau_fast=_tau_or_none(args.hire_tau_fast),
            tau_slow=_tau_or_none(args.hire_tau_slow),
            tau_surprise=_tau_or_none(args.hire_tau_surprise),
            gate_theta=float(args.hire_gate_theta),
            theta_on=float(args.hire_theta_on),
            theta_off=float(args.hire_theta_off),
            confirm_bins=int(args.hire_confirm_bins),
            cooldown_bins=int(args.hire_cooldown_bins),
            spatial_kernel=int(args.hire_spatial_kernel),
            eps=float(args.hire_eps),
            normalize=bool(args.hire_normalize),
            quantile=float(args.hire_quantile),
        ).to(device)
    return out


def _emit_subsampling_for_chunk(raw_chunk: np.ndarray) -> int:
    """Emit one reconstruction at the end of this chunk (handles short tails)."""
    return max(int(raw_chunk.shape[0]), 1)


def _reset_integrator_states(integrators: dict[str, object]) -> None:
    for integrator in integrators.values():
        reset = getattr(integrator, "reset", None)
        if callable(reset):
            reset()
            continue
        # PPB / STEA: drop absolute time so the next clear_states path re-inits.
        if hasattr(integrator, "t_absolute"):
            integrator.t_absolute = 0
        clear = getattr(integrator, "clear_states", None)
        if callable(clear):
            clear()


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


def _preprocess_sum_gray(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: SumPreprocessor,
    clear_states: bool,
) -> torch.Tensor:
    cube = _gray_chunk_to_cube(raw_chunk, device)
    recons = integrator.process_photon_cube(
        cube,
        clear_states=clear_states,
        subsampling=_emit_subsampling_for_chunk(raw_chunk),
    )
    return _gray_hwt_to_chw(recons)[-1:].contiguous()


def _preprocess_ema_gray(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: EmaPreprocessor,
    clear_states: bool,
) -> torch.Tensor:
    cube = _gray_chunk_to_cube(raw_chunk, device)
    recons = integrator.process_photon_cube(
        cube,
        clear_states=clear_states,
        subsampling=_emit_subsampling_for_chunk(raw_chunk),
    )
    return _gray_hwt_to_chw(recons)[-1:].contiguous()


def _preprocess_ppb_gray(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: PerPixelBayesian,
    clear_states: bool,
) -> torch.Tensor:
    cube = _gray_chunk_to_cube(raw_chunk, device)
    recons = integrator.process_photon_cube(
        cube,
        clear_states=clear_states,
        subsampling=_emit_subsampling_for_chunk(raw_chunk),
    )
    return _gray_hwt_to_chw(recons)[-1:].contiguous()


def _preprocess_stea_gray(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: SpatioTemporalEvidenceAccumulation,
    clear_states: bool,
) -> torch.Tensor:
    cube = _gray_chunk_to_cube(raw_chunk, device)
    recons = integrator.process_photon_cube(
        cube,
        clear_states=clear_states,
        subsampling=_emit_subsampling_for_chunk(raw_chunk),
    )
    return _gray_hwt_to_chw(recons)[-1:].contiguous()


def _preprocess_hire_gray(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: HIRE,
    clear_states: bool,
) -> torch.Tensor:
    cube = _gray_chunk_to_cube(raw_chunk, device)
    recons = integrator.process_photon_cube(
        cube,
        clear_states=clear_states,
        subsampling=_emit_subsampling_for_chunk(raw_chunk),
    )
    return _gray_hwt_to_chw(recons)[-1:].contiguous()


def _preprocess_sum_rgb(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: SumPreprocessor,
    clear_states: bool,
) -> torch.Tensor:
    return integrate_raw_chunk_to_rgb(
        integrator,
        raw_chunk,
        packed_nch=3,
        device=device,
        clear_states=clear_states,
        subsampling=_emit_subsampling_for_chunk(raw_chunk),
    )


def _preprocess_ema_rgb(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: EmaPreprocessor,
    clear_states: bool,
) -> torch.Tensor:
    return integrate_raw_chunk_to_rgb(
        integrator,
        raw_chunk,
        packed_nch=3,
        device=device,
        clear_states=clear_states,
        subsampling=_emit_subsampling_for_chunk(raw_chunk),
    )


def _preprocess_ppb_rgb(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: PerPixelBayesian,
    clear_states: bool,
) -> torch.Tensor:
    return integrate_raw_chunk_to_rgb(
        integrator,
        raw_chunk,
        packed_nch=3,
        device=device,
        clear_states=clear_states,
        subsampling=_emit_subsampling_for_chunk(raw_chunk),
    )


def _preprocess_stea_rgb(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: SpatioTemporalEvidenceAccumulation,
    clear_states: bool,
) -> torch.Tensor:
    return integrate_raw_chunk_to_rgb(
        integrator,
        raw_chunk,
        packed_nch=3,
        device=device,
        clear_states=clear_states,
        subsampling=_emit_subsampling_for_chunk(raw_chunk),
    )


def _preprocess_hire_rgb(
    raw_chunk: np.ndarray,
    *,
    device: torch.device,
    integrator: HIRE,
    clear_states: bool,
) -> torch.Tensor:
    return integrate_raw_chunk_to_rgb(
        integrator,
        raw_chunk,
        packed_nch=3,
        device=device,
        clear_states=clear_states,
        subsampling=_emit_subsampling_for_chunk(raw_chunk),
    )


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
            return _preprocess_sum_gray(
                raw_chunk, device=device, integrator=integrators["sum"], clear_states=first_chunk
            )
        if method == "ema":
            return _preprocess_ema_gray(raw_chunk, device=device, integrator=integrators["ema"], clear_states=first_chunk)
        if method == "ppb":
            return _preprocess_ppb_gray(raw_chunk, device=device, integrator=integrators["ppb"], clear_states=first_chunk)
        if method == "stea":
            return _preprocess_stea_gray(raw_chunk, device=device, integrator=integrators["stea"], clear_states=first_chunk)
        if method == "hire":
            return _preprocess_hire_gray(raw_chunk, device=device, integrator=integrators["hire"], clear_states=first_chunk)
        raise ValueError(f"Unsupported grayscale preprocessor: {method!r}")

    if method == "sum":
        return _preprocess_sum_rgb(
            raw_chunk, device=device, integrator=integrators["sum"], clear_states=first_chunk
        )
    if method == "ema":
        return _preprocess_ema_rgb(raw_chunk, device=device, integrator=integrators["ema"], clear_states=first_chunk)
    if method == "ppb":
        return _preprocess_ppb_rgb(raw_chunk, device=device, integrator=integrators["ppb"], clear_states=first_chunk)
    if method == "stea":
        return _preprocess_stea_rgb(raw_chunk, device=device, integrator=integrators["stea"], clear_states=first_chunk)
    if method == "hire":
        return _preprocess_hire_rgb(raw_chunk, device=device, integrator=integrators["hire"], clear_states=first_chunk)
    raise ValueError(f"Unsupported RGB preprocessor: {method!r}")


def _chunk_start_indices(
    n_bins: int,
    *,
    chunk_size: int,
    stride: int,
    max_chunks: int,
) -> list[int]:
    if int(max_chunks) > 0:
        if n_bins <= 0:
            return []
        max_t0 = max(0, n_bins - int(chunk_size))
        if int(max_chunks) == 1:
            return [0]
        return [int(round(i * max_t0 / (int(max_chunks) - 1))) for i in range(int(max_chunks))]
    return list(range(0, n_bins, int(stride)))


def main() -> None:
    args = _parse_args()
    npy = _resolve_input_npy(args.in_path)
    preprocessors = [x.strip().lower() for x in args.pre.split(",") if x.strip()]
    invalid = sorted(set(preprocessors) - set(ALL_METHODS))
    if invalid:
        raise ValueError(f"Unsupported preprocessors: {invalid}")
    if not preprocessors:
        raise ValueError("No preprocessors selected")

    packed = _load_packed_mmap(npy)
    n_bins = int(packed.shape[0])
    if int(args.max_bins) > 0:
        n_bins = min(n_bins, int(args.max_bins))
    kind = _infer_layout_from_packed(
        packed,
        bitdim=int(args.bitdim),
        expected_w=int(args.expected_w),
        bitorder=str(args.bitorder),
        ch_order=str(args.packed_ch_order),
    )
    stride = int(args.chunk_stride) if int(args.chunk_stride) > 0 else int(args.chunk_size)
    max_chunks = int(args.max_chunks)
    chunk_starts = _chunk_start_indices(
        n_bins,
        chunk_size=int(args.chunk_size),
        stride=stride,
        max_chunks=max_chunks,
    )
    # Contiguous non-overlapping chunks: keep causal state (PPB background run-lengths, etc.).
    # Reset when windows are independent by request, non-contiguous (max_chunks), or overlapping
    # (stride < chunk_size would otherwise double-count bins under carried state).
    contiguous_stream = (
        max_chunks == 0
        and stride >= int(args.chunk_size)
        and not bool(args.independent_chunks)
    )
    reset_each_chunk = not contiguous_stream
    device = _resolve_device(args.device)
    integrators = _build_integrators(args, device, preprocessors)
    _reset_integrator_states(integrators)

    sample = _sample_name(args.in_path if args.in_path.exists() else npy)
    out_dir = args.save_dir / sample / "video00000"
    compare_dir = args.save_dir / f"{sample}_compare" / "video00000"
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {npy}")
    print(f"packed: shape={packed.shape} kind={kind} n_bins={n_bins}")
    print(
        f"chunk_size={int(args.chunk_size)} (=emit interval) stride={stride} "
        f"carry_state={contiguous_stream} reset_each_chunk={reset_each_chunk}"
    )
    if "hire" in preprocessors:
        print(
            f"hire: bin_rate_hz={float(args.bin_rate_hz):g} "
            f"ref_rate_hz={float(args.hire_ref_rate_hz):g} "
            f"bins={int(args.hire_fast_bins)}/{int(args.hire_slow_bins)}/{int(args.hire_surprise_bins)} "
            f"gate_theta={float(args.hire_gate_theta):g} "
            f"theta_on/off={float(args.hire_theta_on):g}/{float(args.hire_theta_off):g} "
            f"confirm={int(args.hire_confirm_bins)} cooldown={int(args.hire_cooldown_bins)}"
        )
    if "ppb" in preprocessors:
        print(
            f"ppb: gamma={float(args.ppb_gamma):g} "
            f"min_filter_size={int(args.ppb_min_filter_size)} "
            f"normalize={bool(args.ppb_normalize)}"
        )
    if max_chunks > 0:
        print(f"max_chunks={max_chunks} (evenly spaced; always reset)")
    elif stride < int(args.chunk_size):
        print("Note: stride < chunk_size (overlapping windows) => state reset each chunk to avoid double-counting.")
    elif bool(args.independent_chunks):
        print("Note: --independent-chunks resets state every window (sum-like; background stats cannot accumulate).")
    else:
        print("State: causal carry across chunks (PPB/STEA/EMA/HIRE online). Use --independent-chunks to disable.")
    print(f"writing outputs to: {out_dir}")
    print(f"preprocess flips: flip_x={bool(args.flip_x)} flip_y={bool(args.flip_y)} bitorder={args.bitorder}")

    frame_idx = 0
    cube_idx = 0
    for t0 in chunk_starts:
        t1 = min(t0 + int(args.chunk_size), n_bins)
        raw_chunk, _ = _packed_time_range_to_raw_chunk(
            packed,
            t0,
            t1,
            bitdim=int(args.bitdim),
            expected_w=int(args.expected_w),
            bitorder=str(args.bitorder),
            ch_order=str(args.packed_ch_order),
        )
        raw_chunk = _apply_spatial_preprocess_ops_raw(
            raw_chunk,
            flip_x=bool(args.flip_x),
            flip_y=bool(args.flip_y),
        )
        if raw_chunk.shape[0] == 0:
            continue

        stem = f"cube{cube_idx:05d}_t{t0:06d}_{t1:06d}_frame{frame_idx:07d}"
        labels: list[str] = []
        panels: list[np.ndarray] = []
        clear_states = reset_each_chunk or cube_idx == 0
        for method in preprocessors:
            frames = _preprocess_chunk(
                method,
                raw_chunk,
                kind=kind,
                device=device,
                first_chunk=clear_states,
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

        frame_idx += 1
        cube_idx += 1

    print(f"Done. Wrote method PNGs to {out_dir}")
    print(f"Done. Wrote comparison PNGs to {compare_dir}")


if __name__ == "__main__":
    main()
