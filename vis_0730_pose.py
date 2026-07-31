#!/usr/bin/env python3
"""0730 capture: QNN + HIRE raw streaming pose viz.

Loops ``<data_root>/<sample>/frames.npy`` (default capture-spc8kHz), runs each
sample with QNN and HIRE SpadPoseModels in **causal online** raw mode
(``cache_mode=raw``, non-overlapping ``chunk_size=320`` @ 8000 Hz → 25 fps).

Detector RGB uses raw-sum SpadDSC (same as preview): unpack frames.npy → sum
→ full spad_dsc (p≤319/320, DCR, gain, bad-pixel inpaint) → BGGR ISP.
QNN/HIRE still run for temporal state / heatmaps. Use ``--no-dsc`` to disable.

Per sample / method writes::

    <save_root>/<sample>/{qnn,hire}/
      recon/frame_XXXXXXX.png
      heatmap/frame_XXXXXXX.png   # TURBO of min_run_length (QNN) or n_slow (HIRE), 0..100 → RGBA
      pose/frame_XXXXXXX.png      # transparent RGBA + 2D pose (vis_main_pose cls colors)

Examples
--------
python ultralytics/vis_0730_pose.py \\
  --qnn_ckpt /path/qnn_ssd.pt \\
  --hire_ckpt /path/hire_ssd.pt

python ultralytics/vis_0730_pose.py \\
  --qnn_ckpt /path/qnn.pt --hire_ckpt /path/hire.pt \\
  --folders acq00001 acq00002 --overwrite
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.models.yolo.pose.spad_preprocessors import build_spad_preprocessor

DEFAULT_DATA_ROOT = Path("/home/zvc/Data/SPADHand/0730/spad/capture-spc8kHz")
DEFAULT_SAVE_ROOT = Path("/home/zvc/Project/SPADHand/Vis/0730")
FRAMES_NPY = "frames.npy"

# vis_main_pose._cls_color_bgr: left=blue-ish, right=orange (BGR)
_CLS_COLOR_BGR = {
    0: (255, 128, 0),
    1: (0, 128, 255),
}

_DSP_MOD = None
_VHP_MOD = None


def _dsp():
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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="0730 QNN/HIRE raw streaming pose viz (recon + heatmap + transparent pose)"
    )
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--save_root", type=Path, default=DEFAULT_SAVE_ROOT)
    ap.add_argument("--qnn_ckpt", type=Path, required=True, help="QNN/SSD sequence SpadPoseModel .pt")
    ap.add_argument("--hire_ckpt", type=Path, required=True, help="HIRE/SSD sequence SpadPoseModel .pt")
    ap.add_argument(
        "--qnn_pre_override",
        type=str,
        default="ppb",
        choices=["none", "ppb", "ema", "sum", "stea", "hire"],
        help="Rebuild QNN preprocessor from CLI (none = keep ckpt); default ppb for run-length maps",
    )
    ap.add_argument(
        "--hire_pre_override",
        type=str,
        default="hire",
        choices=["none", "hire", "ppb", "ema", "sum", "stea"],
        help="Rebuild HIRE preprocessor from CLI (none = keep ckpt); default hire for n_slow maps",
    )
    ap.add_argument("--chunk_size", type=int, default=320)
    ap.add_argument("--spad_bin_rate_hz", type=float, default=8000.0)
    ap.add_argument("--start_bin", type=int, default=0)
    ap.add_argument("--end_bin", type=int, default=0, help="Exclusive; 0 = EOF")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=20)
    ap.add_argument("--kpt_thresh", type=float, default=0.5)
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument(
        "--folders",
        type=str,
        nargs="*",
        default=None,
        help="Optional sample folder names; default = all with frames.npy",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--methods",
        type=str,
        default="qnn,hire",
        help="Comma subset: qnn,hire",
    )
    # PPB (QNN)
    ap.add_argument("--ppb_bocpd_gamma", type=float, default=0.001)
    ap.add_argument("--ppb_memory_size", type=int, default=10)
    ap.add_argument("--ppb_quantile", type=float, default=1.0)
    ap.add_argument("--ppb_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppb_min_filter_size", type=int, default=5)
    # HIRE
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
    # STEA (optional override)
    ap.add_argument("--stea_fast_window", type=int, default=8)
    ap.add_argument("--stea_slow_window", type=int, default=64)
    ap.add_argument("--stea_temporal_window", type=int, default=16)
    ap.add_argument("--stea_fast_tau", type=float, default=0.2)
    ap.add_argument("--stea_motion_sharpness", type=float, default=8.0)
    ap.add_argument("--stea_motion_threshold", type=float, default=0.05)
    ap.add_argument("--stea_stable_prior", type=float, default=0.7)
    ap.add_argument("--stea_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stea_quantile", type=float, default=1.0)
    # Heatmap normalization → display range 0..100
    ap.add_argument(
        "--run_length_vmax",
        type=float,
        default=100.0,
        help="QNN min_run_length cap before mapping to 0..100",
    )
    ap.add_argument(
        "--n_slow_vmax",
        type=float,
        default=0.0,
        help="HIRE n_slow cap before mapping to 0..100; 0 = hire_slow_bins",
    )
    ap.add_argument("--ema_alpha", type=float, default=0.01)
    ap.add_argument("--ema_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ema_quantile", type=float, default=1.0)
    # DSC on raw sum → color → detector (same as preview_video)
    ap.add_argument(
        "--resources",
        type=Path,
        default=Path("/home/zvc/Data/SPADHand/NoiseCorrection/resources/resources"),
        help="SpadDSC calibration dir (bad_pixel / gain / dcr / wb / ccm)",
    )
    ap.add_argument("--no-dsc", action="store_true", help="Skip raw-sum DSC + BGGR ISP (use QNN/HIRE recon RGB)")
    ap.add_argument("--no-dsc-gain", action="store_true", help="Skip gain_map_cfa inside spad_dsc")
    ap.add_argument(
        "--dsc-isp-divide",
        type=float,
        default=320.0,
        help="Divide before WB in BGGR ISP after DSC (default=320, same as preview)",
    )
    ap.add_argument(
        "--dsc-srgb",
        action="store_true",
        help="Apply sRGB OETF in DSC ISP (default: linear RGB + spad_input_gamma)",
    )
    ap.add_argument(
        "--dsc-max-sum-count",
        type=int,
        default=319,
        help="Cap binary hits before -log (p <= count/num_frames; default 319/320)",
    )
    return ap.parse_args()


def _cls_color_bgr(cls_id: int) -> tuple[int, int, int]:
    return _CLS_COLOR_BGR.get(int(cls_id), _CLS_COLOR_BGR[0])


def _pre_kwargs(name: str, *, subsampling: int, args: argparse.Namespace) -> dict[str, Any]:
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


def _discover_samples(data_root: Path, folders: list[str] | None) -> list[Path]:
    if folders:
        sample_dirs = [data_root / name for name in folders]
        missing = [p for p in sample_dirs if not (p / FRAMES_NPY).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {FRAMES_NPY} in: {missing}")
        return sample_dirs
    sample_dirs = sorted(
        p for p in data_root.iterdir() if p.is_dir() and (p / FRAMES_NPY).is_file()
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No <sample>/{FRAMES_NPY} under {data_root}")
    return sample_dirs


def _warn_if_not_ssd(spad_model, ckpt: Path) -> None:
    try:
        plugin = None
        plugins = getattr(spad_model, "plugins_by_layer", None) or {}
        for p in plugins.values():
            name = getattr(p, "name", None) or p.__class__.__name__
            plugin = name
            break
        if plugin is None:
            print(f"Warning: could not confirm temporal_ssd plugin on {ckpt}", flush=True)
            return
        if str(plugin).lower() not in {"temporal_ssd", "ssd"}:
            print(f"Warning: expected temporal_ssd plugin, got {plugin!r} on {ckpt}", flush=True)
    except Exception:
        print(f"Warning: could not confirm temporal_ssd plugin on {ckpt}", flush=True)


def _load_sequence_model(
    ckpt: Path,
    *,
    label: str,
    args: argparse.Namespace,
    device: torch.device,
    chunk_size: int,
) -> tuple[Any, Any]:
    """Return ``(yolo, spad_model)`` configured for raw online streaming."""
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

    if not bool(getattr(args, "no_dsc", False)):
        noise_dir = Path(__file__).resolve().parents[1] / "DataProcess" / "noisecorrection"
        if str(noise_dir) not in sys.path:
            sys.path.insert(0, str(noise_dir))
        from attach_dsc import attach_spad_dsc_corrector

        corr = attach_spad_dsc_corrector(
            spad_model,
            args.resources,
            bin_rate_hz=float(args.spad_bin_rate_hz),
            num_frames=int(chunk_size),
            apply_gain=not bool(args.no_dsc_gain),
            isp_divide=float(args.dsc_isp_divide),
            use_srgb=bool(args.dsc_srgb),
            max_sum_count=int(args.dsc_max_sum_count),
        )
        print(f"  {label}: DSC attached ({corr.calib_summary().splitlines()[0]})", flush=True)
    else:
        spad_model.spad_dsc_corrector = None
        print(f"  {label}: DSC disabled", flush=True)

    return yolo, spad_model


def _unwrap_preprocessor(pre: Any) -> Any:
    """Prefer core integrator when wrapped (e.g. PerPixelBayesianFrame)."""
    if pre is None:
        return None
    core = getattr(pre, "core", None)
    return core if core is not None else pre


def _tensor_hw_to_npy(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if torch.is_tensor(x):
        return np.ascontiguousarray(x.detach().float().cpu().numpy())
    arr = np.asarray(x)
    if arr.ndim < 2:
        return None
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _extract_aux_map(spad_model, *, label: str) -> np.ndarray | None:
    """QNN → min_run_length (estimated_run_length_hw); HIRE → n_slow."""
    pre = _unwrap_preprocessor(getattr(spad_model, "preprocessor", None))
    if pre is None:
        return None
    if label == "qnn":
        if hasattr(pre, "estimated_run_length_hw"):
            return _tensor_hw_to_npy(pre.estimated_run_length_hw())
        return None
    if label == "hire":
        return _tensor_hw_to_npy(getattr(pre, "n_slow", None))
    return None


def _resize_map_to_hw(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if arr.ndim != 2:
        raise ValueError(f"Expected (H,W) map, got {arr.shape}")
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr.astype(np.float32, copy=False)
    return cv2.resize(arr.astype(np.float32), (int(w), int(h)), interpolation=cv2.INTER_LINEAR)


def _map_to_heatmap_bgra(arr: np.ndarray, *, vmax: float) -> np.ndarray:
    """Normalize map to 0..100, TURBO colormap, alpha from value → BGRA uint8 (OpenCV PNG)."""
    vmax = float(max(vmax, 1e-6))
    score01 = np.clip(arr.astype(np.float32) / vmax, 0.0, 1.0)
    score100 = score01 * 100.0
    u8 = np.clip(np.round(score100 * (255.0 / 100.0)), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    alpha = np.clip(np.round(score01 * 255.0), 0, 255).astype(np.uint8)
    # Zero-value pixels fully transparent (true transparent background)
    alpha = np.where(u8 > 0, alpha, 0).astype(np.uint8)
    return np.dstack((bgr, alpha))


def _pose_to_bgra(
    result,
    *,
    hw: tuple[int, int],
    kpt_thresh: float,
) -> np.ndarray:
    """Transparent BGRA with 2D pose; colors from vis_main_pose cls palette."""
    h, w = hw
    layer = np.zeros((h, w, 3), dtype=np.uint8)
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return np.zeros((h, w, 4), dtype=np.uint8)

    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    poses = None
    if getattr(result, "keypoints", None) is not None:
        poses = result.keypoints.data.cpu().numpy()

    vhp = _vhp()
    for j, cls_id in enumerate(cls_ids):
        color = _cls_color_bgr(int(cls_id))
        if poses is not None and j < len(poses):
            layer = vhp._draw_pose_id(
                layer,
                poses[j],
                color=color,
                thresh=float(kpt_thresh),
                draw_outline=True,
                trail_style=False,
            )
    alpha = np.where(layer.max(axis=2) > 0, 255, 0).astype(np.uint8)
    return np.dstack((layer, alpha))


def _sample_done(out_method: Path, n_expected: int) -> bool:
    """Skip if recon frames already exist for this method (unless overwrite)."""
    recon_dir = out_method / "recon"
    if not recon_dir.is_dir():
        return False
    n = len(list(recon_dir.glob("frame_*.png")))
    return n >= int(n_expected) > 0


def _run_sample_method(
    *,
    yolo,
    spad_model,
    source,
    sample_name: str,
    label: str,
    out_method: Path,
    args: argparse.Namespace,
    device: torch.device,
    chunk_size: int,
    t_begin: int,
    t_end: int,
    heatmap_vmax: float,
) -> int:
    """Stream one sample with one model; save recon / heatmap / pose. Returns frame count."""
    names = yolo.names
    kpt_shape = getattr(spad_model, "kpt_shape", (21, 3))
    recon_dir = out_method / "recon"
    heat_dir = out_method / "heatmap"
    pose_dir = out_method / "pose"
    for d in (recon_dir, heat_dir, pose_dir):
        d.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    if hasattr(spad_model, "spad_begin_stream"):
        spad_model.spad_begin_stream()
    else:
        spad_model.spad_set_online_inference(True)
        spad_model.spad_clear_plugin_states()

    try:
        n_bins = t_end - t_begin
        n_chunks = max(n_bins // chunk_size, 0)
        pbar = tqdm(total=n_chunks, desc=f"{sample_name}/{label}", leave=False)
        for t0 in range(t_begin, t_end, chunk_size):
            t1 = min(t_end, t0 + chunk_size)
            if t1 <= t0:
                break
            raw_chunk = _dsp()._slice_raw_chunk(source, t0, t1, packed_ch_order=args.packed_ch_order)
            raw_chunk = _dsp()._prepare_raw_chunk_for_spad(
                raw_chunk, chunk_t=chunk_size, tail_pad_full=False
            )
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
            if recon_frames_bgr:
                recon_bgr = recon_frames_bgr[-1]
            else:
                # Fallback empty canvas
                recon_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
            hw = recon_bgr.shape[:2]

            stem = f"frame_{frame_idx:07d}"
            cv2.imwrite(str(recon_dir / f"{stem}.png"), recon_bgr)

            aux = _extract_aux_map(spad_model, label=label)
            if aux is not None:
                aux = _resize_map_to_hw(aux, hw)
                heat_bgra = _map_to_heatmap_bgra(aux, vmax=heatmap_vmax)
            else:
                heat_bgra = np.zeros((hw[0], hw[1], 4), dtype=np.uint8)
            cv2.imwrite(str(heat_dir / f"{stem}.png"), heat_bgra)

            pose_bgra = _pose_to_bgra(result, hw=hw, kpt_thresh=float(args.kpt_thresh))
            cv2.imwrite(str(pose_dir / f"{stem}.png"), pose_bgra)

            frame_idx += 1
            pbar.update(1)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pbar.close()
    finally:
        if hasattr(spad_model, "spad_end_stream"):
            spad_model.spad_end_stream()

    return frame_idx


def main() -> int:
    args = _parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"data_root not found: {args.data_root}")
    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"--chunk_size must be > 0, got {chunk_size}")
    if float(args.spad_bin_rate_hz) <= 0:
        raise ValueError(f"--spad_bin_rate_hz must be > 0, got {args.spad_bin_rate_hz}")

    methods = {m.strip().lower() for m in str(args.methods).split(",") if m.strip()}
    allowed = {"qnn", "hire"}
    unknown = methods - allowed
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; allowed={sorted(allowed)}")
    if not methods:
        raise ValueError("--methods is empty")

    n_slow_vmax = float(args.n_slow_vmax) if float(args.n_slow_vmax) > 0 else float(args.hire_slow_bins)
    run_length_vmax = float(args.run_length_vmax)

    sample_dirs = _discover_samples(args.data_root, args.folders)
    device = _dsp()._resolve_device(args.device)
    args.save_root.mkdir(parents=True, exist_ok=True)

    print(f"data_root={args.data_root}")
    print(f"save_root={args.save_root}")
    print(
        f"samples={len(sample_dirs)} methods={sorted(methods)} "
        f"chunk={chunk_size} bin_rate={args.spad_bin_rate_hz} device={device}"
    )

    models: dict[str, tuple[Any, Any]] = {}
    if "qnn" in methods:
        print(f"Loading QNN: {args.qnn_ckpt}", flush=True)
        models["qnn"] = _load_sequence_model(
            args.qnn_ckpt, label="qnn", args=args, device=device, chunk_size=chunk_size
        )
    if "hire" in methods:
        print(f"Loading HIRE: {args.hire_ckpt}", flush=True)
        models["hire"] = _load_sequence_model(
            args.hire_ckpt, label="hire", args=args, device=device, chunk_size=chunk_size
        )

    for sample_dir in tqdm(sample_dirs, desc="Samples"):
        npy_path = sample_dir / FRAMES_NPY
        sources = list(_dsp()._iter_raw_video_sources_from_sample_path(npy_path))
        if not sources:
            print(f"skip (no source): {npy_path}", flush=True)
            continue
        source = sources[0]
        n_bins = _dsp()._video_num_bins(source)
        t_begin = max(int(args.start_bin), 0)
        t_end = int(args.end_bin) if int(args.end_bin) > 0 else n_bins
        t_end = min(t_end, n_bins)
        n_expected = max((t_end - t_begin) // chunk_size, 0)

        sample_out = args.save_root / sample_dir.name
        sample_out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {sample_dir.name}: bins=[{t_begin},{t_end}) n_chunks≈{n_expected} ===", flush=True)

        for label in ("qnn", "hire"):
            if label not in methods:
                continue
            yolo, spad_model = models[label]
            out_method = sample_out / label
            if not args.overwrite and _sample_done(out_method, n_expected):
                print(f"  skip (exists): {out_method}", flush=True)
                continue
            vmax = run_length_vmax if label == "qnn" else n_slow_vmax
            n_frames = _run_sample_method(
                yolo=yolo,
                spad_model=spad_model,
                source=source,
                sample_name=sample_dir.name,
                label=label,
                out_method=out_method,
                args=args,
                device=device,
                chunk_size=chunk_size,
                t_begin=t_begin,
                t_end=t_end,
                heatmap_vmax=vmax,
            )
            print(
                f"  {label}: {n_frames} frames → {out_method} "
                f"(heatmap={'min_run_length' if label == 'qnn' else 'n_slow'} vmax={vmax})",
                flush=True,
            )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
