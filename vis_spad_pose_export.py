#!/usr/bin/env python3
"""Export recon / pose / heatmap from any packed ``frames.npy`` dataset.

Pipeline (no DSC / ISP correction): preprocessor recon → RGB → ``x**(1/γ)``
(default ``γ=2.2``) → detector. Same layout as ``vis_0730_pose.py`` so
``vis_0730_video.py`` can compose::

    <save_root>/<sample>/{qnn,hire}/
      recon/frame_XXXXXXX.png
      pose/frame_XXXXXXX.png
      heatmap/frame_XXXXXXX.png

Input::

    <data_root>/<sample>/frames.npy

Works for 0730, 0428, HamNoSys, or any capture tree with the same packing.
PPB parameters use the ``--ppb_*`` / QNN override; HIRE uses ``--hire_*``.

Examples
--------
# Generic dataset
python ultralytics/vis_spad_pose_export.py \\
  --data_root /home/zvc/Data/SPADHand/0428_wc/spad/capture-spc8kHz \\
  --save_root /home/zvc/Project/SPADHand/Vis/0428 \\
  --qnn_ckpt /path/qnn.pt --hire_ckpt /path/hire.pt \\
  --overwrite

# Then make videos (same layout)
python ultralytics/vis_0730_video.py \\
  --vis_root /home/zvc/Project/SPADHand/Vis/0428 \\
  --data_root /home/zvc/Data/SPADHand/0428_wc/spad/capture-spc8kHz \\
  --samples SAMPLE_A --slow_ranges 40-80 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

# Reuse 0730 implementation (model load / stream / save panels).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import vis_0730_pose as _v0730  # noqa: E402

FRAMES_NPY = _v0730.FRAMES_NPY


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="frames.npy → recon/pose/heatmap (QNN+HIRE); dataset-agnostic"
    )
    ap.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Root with <sample>/frames.npy",
    )
    ap.add_argument(
        "--save_root",
        type=Path,
        required=True,
        help="Output root: <save_root>/<sample>/{qnn,hire}/{recon,pose,heatmap}",
    )
    ap.add_argument("--qnn_ckpt", type=Path, required=True, help="QNN/SSD SpadPoseModel .pt")
    ap.add_argument("--hire_ckpt", type=Path, required=True, help="HIRE/SSD SpadPoseModel .pt")
    ap.add_argument(
        "--qnn_pre_override",
        type=str,
        default="ppb",
        choices=["none", "ppb", "ema", "sum", "stea", "hire"],
        help="QNN preprocessor (default ppb)",
    )
    ap.add_argument(
        "--hire_pre_override",
        type=str,
        default="hire",
        choices=["none", "hire", "ppb", "ema", "sum", "stea"],
        help="HIRE preprocessor (default hire)",
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
        "--input_gamma",
        type=float,
        default=2.2,
        help="After recon RGB: apply pow(x, 1/gamma) before detector (default 2.2)",
    )
    ap.add_argument(
        "--folders",
        type=str,
        nargs="*",
        default=None,
        help="Optional sample names; default = all with frames.npy",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--methods",
        type=str,
        default="qnn,hire",
        help="Comma subset: qnn,hire",
    )
    # PPB
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
    # STEA / EMA (if overrides used)
    ap.add_argument("--stea_fast_window", type=int, default=8)
    ap.add_argument("--stea_slow_window", type=int, default=64)
    ap.add_argument("--stea_temporal_window", type=int, default=16)
    ap.add_argument("--stea_fast_tau", type=float, default=0.2)
    ap.add_argument("--stea_motion_sharpness", type=float, default=8.0)
    ap.add_argument("--stea_motion_threshold", type=float, default=0.05)
    ap.add_argument("--stea_stable_prior", type=float, default=0.7)
    ap.add_argument("--stea_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stea_quantile", type=float, default=1.0)
    ap.add_argument("--ema_alpha", type=float, default=0.01)
    ap.add_argument("--ema_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ema_quantile", type=float, default=1.0)
    # Heatmap display
    ap.add_argument("--run_length_vmax", type=float, default=100.0)
    ap.add_argument("--n_slow_vmax", type=float, default=0.0, help="0 = hire_slow_bins")
    return ap.parse_args()


def export_spad_pose_panels(
    *,
    data_root: Path,
    save_root: Path,
    qnn_ckpt: Path,
    hire_ckpt: Path,
    args: argparse.Namespace | None = None,
) -> int:
    """Programmatic entry: same as CLI ``main`` with explicit roots/ckpts.

    If ``args`` is None, builds defaults from ``_parse_args``-compatible Namespace
    (call CLI for full control). Returns number of sample×method runs written.
    """
    if args is None:
        # Minimal namespace for library use — prefer CLI for full knobs.
        args = argparse.Namespace(
            data_root=data_root,
            save_root=save_root,
            qnn_ckpt=qnn_ckpt,
            hire_ckpt=hire_ckpt,
            qnn_pre_override="ppb",
            hire_pre_override="hire",
            chunk_size=320,
            spad_bin_rate_hz=8000.0,
            start_bin=0,
            end_bin=0,
            device="",
            imgsz=512,
            conf=0.25,
            iou=0.7,
            max_det=20,
            kpt_thresh=0.5,
            packed_ch_order="RGB",
            input_gamma=2.2,
            folders=None,
            overwrite=False,
            methods="qnn,hire",
            ppb_bocpd_gamma=0.001,
            ppb_memory_size=10,
            ppb_quantile=1.0,
            ppb_normalize=True,
            ppb_min_filter_size=5,
            hire_fast_bins=24,
            hire_slow_bins=160,
            hire_surprise_bins=4,
            hire_mix_hold_bins=80,
            hire_mix_bins=12.0,
            hire_mix_theta=0.06,
            hire_mix_floor=-1.0,
            hire_theta_on=0.08,
            hire_theta_off=0.02,
            hire_theta_grow=-1.0,
            hire_confirm_bins=4,
            hire_cooldown_bins=0,
            hire_spatial_kernel=5,
            hire_gate_pool="max",
            hire_reset_open=15,
            hire_reset_grow=6,
            hire_normalize=True,
            hire_quantile=1.0,
            stea_fast_window=8,
            stea_slow_window=64,
            stea_temporal_window=16,
            stea_fast_tau=0.2,
            stea_motion_sharpness=8.0,
            stea_motion_threshold=0.05,
            stea_stable_prior=0.7,
            stea_normalize=True,
            stea_quantile=1.0,
            ema_alpha=0.01,
            ema_normalize=True,
            ema_quantile=1.0,
            run_length_vmax=100.0,
            n_slow_vmax=0.0,
            no_dsc=True,
        )
    else:
        args.data_root = Path(data_root)
        args.save_root = Path(save_root)
        args.qnn_ckpt = Path(qnn_ckpt)
        args.hire_ckpt = Path(hire_ckpt)

    return _run_export(args)


def _configure_export_model(spad_model: Any, *, input_gamma: float) -> None:
    """Export path: no DSC; recon RGB → pow(x, 1/gamma) before detector."""
    spad_model.spad_dsc_corrector = None
    gamma = float(input_gamma)
    if gamma <= 0:
        raise ValueError(f"--input_gamma must be > 0, got {gamma}")
    spad_model.spad_input_gamma = gamma


def _run_export(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    save_root = Path(args.save_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    chunk_size = int(args.chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"--chunk_size must be > 0, got {chunk_size}")
    if float(args.spad_bin_rate_hz) <= 0:
        raise ValueError(f"--spad_bin_rate_hz must be > 0, got {args.spad_bin_rate_hz}")

    # Export never uses DSC (recon → gamma → detector).
    args.no_dsc = True
    input_gamma = float(getattr(args, "input_gamma", 2.2))

    methods = {m.strip().lower() for m in str(args.methods).split(",") if m.strip()}
    allowed = {"qnn", "hire"}
    unknown = methods - allowed
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; allowed={sorted(allowed)}")
    if not methods:
        raise ValueError("--methods is empty")

    n_slow_vmax = float(args.n_slow_vmax) if float(args.n_slow_vmax) > 0 else float(args.hire_slow_bins)
    run_length_vmax = float(args.run_length_vmax)

    sample_dirs = _v0730._discover_samples(data_root, args.folders)
    device = _v0730._dsp()._resolve_device(args.device)
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"data_root={data_root}")
    print(f"save_root={save_root}")
    print(
        f"samples={len(sample_dirs)} methods={sorted(methods)} "
        f"chunk={chunk_size} bin_rate={args.spad_bin_rate_hz} "
        f"input_gamma={input_gamma} dsc=off device={device}"
    )

    models: dict[str, tuple[Any, Any]] = {}
    if "qnn" in methods:
        print(f"Loading QNN: {args.qnn_ckpt}", flush=True)
        models["qnn"] = _v0730._load_sequence_model(
            Path(args.qnn_ckpt), label="qnn", args=args, device=device, chunk_size=chunk_size
        )
        _configure_export_model(models["qnn"][1], input_gamma=input_gamma)
    if "hire" in methods:
        print(f"Loading HIRE: {args.hire_ckpt}", flush=True)
        models["hire"] = _v0730._load_sequence_model(
            Path(args.hire_ckpt), label="hire", args=args, device=device, chunk_size=chunk_size
        )
        _configure_export_model(models["hire"][1], input_gamma=input_gamma)

    n_written = 0
    for sample_dir in tqdm(sample_dirs, desc="Samples"):
        npy_path = sample_dir / FRAMES_NPY
        sources = list(_v0730._dsp()._iter_raw_video_sources_from_sample_path(npy_path))
        if not sources:
            print(f"skip (no source): {npy_path}", flush=True)
            continue
        source = sources[0]
        n_bins = _v0730._dsp()._video_num_bins(source)
        t_begin = max(int(args.start_bin), 0)
        t_end = int(args.end_bin) if int(args.end_bin) > 0 else n_bins
        t_end = min(t_end, n_bins)
        n_expected = max((t_end - t_begin) // chunk_size, 0)

        sample_out = save_root / sample_dir.name
        sample_out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {sample_dir.name}: bins=[{t_begin},{t_end}) n_chunks≈{n_expected} ===", flush=True)

        for label in ("qnn", "hire"):
            if label not in methods:
                continue
            yolo, spad_model = models[label]
            out_method = sample_out / label
            if not args.overwrite and _v0730._sample_done(out_method, n_expected):
                print(f"  skip (exists): {out_method}", flush=True)
                continue
            vmax = run_length_vmax if label == "qnn" else n_slow_vmax
            n_frames = _v0730._run_sample_method(
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
            n_written += 1

    print(f"Done. wrote_runs={n_written}")
    return n_written


def main() -> int:
    args = _parse_args()
    _run_export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
