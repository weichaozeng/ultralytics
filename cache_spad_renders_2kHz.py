"""Offline 2 kHz SPAD render cache: sum / ema / ppb / hire.

Pipeline
--------
1. Build split JSON with spad under ``renders-spc2kHz``::

     python DataProcess/traindata/build_visionsim_split_2kHz.py \\
       --reuse-split-from /home/zvc/Data/visionsim/outputs/train.json \\
                          /home/zvc/Data/visionsim/outputs/test.json

2. Cache preprocessed frames (no confidence)::

     python cache_spad_renders_2kHz.py \\
       --split-json /home/zvc/Data/visionsim/outputs/train_2kHz.json \\
                    /home/zvc/Data/visionsim/outputs/test_2kHz.json \\
       --json-output /home/zvc/Data/visionsim/outputs/train_2kHz.json \\
                     /home/zvc/Data/visionsim/outputs/test_2kHz.json \\
       --update-json \\
       --preprocessors sum ema ppb hire

Reads packed SPAD from sibling ``renders-spc2kHz``, writes ``frames.npy`` + ``meta.json``
under ``renders-{method}-2kHz``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data.spad_packed import (
    infer_packed_nch,
    packed_frames_to_raw_video,
    raw_chunk_plane,
    raw_hwt_to_rgb_float,
    raw_plane_to_photon_cube,
)
from ultralytics.data.spad_pose_dataset import load_visionsim_split_json
from ultralytics.data.spad_render_cache import (
    CACHE_META_VERSION,
    build_render_config,
    render_config_fingerprint,
    sample_render_dir,
    sibling_sample_render_dir,
)
from ultralytics.models.yolo.pose.spad_preprocessors import build_spad_frame_preprocessor

_PREPROCESSORS = ("sum", "ema", "ppb", "hire")
_DEFAULT_PREPROCESSORS = ("sum", "ema", "ppb", "hire")


def parse_args():
    ap = argparse.ArgumentParser(
        description="2 kHz SPAD cache: frames.npy + meta.json under renders-{method}-2kHz."
    )
    ap.add_argument(
        "--split-json",
        type=str,
        nargs="+",
        required=True,
        help="VisionSIM split JSON path(s), e.g. train.json test.json",
    )
    ap.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Optional explicit cache root. Default: sibling renders-{prep}-2kHz beside renders-spc2kHz.",
    )
    ap.add_argument(
        "--preprocessors",
        type=str,
        nargs="+",
        default=list(_DEFAULT_PREPROCESSORS),
        choices=list(_PREPROCESSORS),
        help="Preprocessors to cache (default: sum ema ppb hire).",
    )
    ap.add_argument("--chunk-size", type=int, default=80, help="Bins per cached frame (=5 GT @ 125 Hz).")
    ap.add_argument("--stride-bins", type=int, default=80)
    ap.add_argument(
        "--spad-bins-per-gt",
        type=int,
        default=16,
        help="Raw bins per GT frame: 2000 Hz / 125 Hz GT = 16 (same GT rate as 8 kHz@64).",
    )
    ap.add_argument("--packed-ch-order", type=str, default="RGB")
    ap.add_argument("--input-gamma", type=float, default=2.2)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit-samples", type=int, default=0)
    ap.add_argument(
        "--update-json",
        action="store_true",
        help="Write frames/meta paths into output split JSON(s).",
    )
    ap.add_argument(
        "--json-output",
        type=str,
        nargs="*",
        default=[],
        help="Output JSON path(s) parallel to --split-json (e.g. train_2kHz.json test_2kHz.json).",
    )
    ap.add_argument(
        "--source-render-dirname",
        type=str,
        default="renders-spc2kHz",
        help="Source packed-SPAD directory name (default: renders-spc2kHz).",
    )
    ap.add_argument(
        "--render-tag",
        type=str,
        default="2kHz",
        help="Cache dir suffix: renders-{preprocessor}-{tag}.",
    )
    # EMA
    ap.add_argument("--ema-alpha", type=float, default=0.01)
    ap.add_argument("--ema-normalize", type=str, default="true")
    ap.add_argument("--ema-quantile", type=float, default=1.0)
    # PPB
    ap.add_argument("--ppb-bocpd-gamma", type=float, default=2e-3)
    ap.add_argument("--ppb-min-filter-size", type=int, default=5)
    ap.add_argument("--ppb-normalize", type=str, default="true")
    ap.add_argument("--ppb-quantile", type=float, default=1.0)
    # sum
    ap.add_argument("--sum-normalize", type=str, default="true")
    ap.add_argument("--sum-quantile", type=float, default=1.0)
    # HIRE (current project defaults)
    ap.add_argument("--spad-bin-rate-hz", type=float, default=2000.0)
    ap.add_argument("--hire-ref-rate-hz", type=float, default=2000.0)
    ap.add_argument("--hire-fast-bins", type=int, default=24)
    ap.add_argument("--hire-slow-bins", type=int, default=160)
    ap.add_argument("--hire-surprise-bins", type=int, default=4)
    ap.add_argument("--hire-tau-fast", type=float, default=0.0)
    ap.add_argument("--hire-tau-slow", type=float, default=0.0)
    ap.add_argument("--hire-tau-surprise", type=float, default=0.0)
    ap.add_argument("--hire-mix-hold-bins", type=int, default=80)
    ap.add_argument("--hire-mix-bins", type=float, default=12.0)
    ap.add_argument("--hire-mix-theta", type=float, default=0.06)
    ap.add_argument("--hire-mix-floor", type=float, default=-1.0)
    ap.add_argument("--hire-theta-on", type=float, default=0.08)
    ap.add_argument("--hire-theta-off", type=float, default=0.02)
    ap.add_argument("--hire-theta-grow", type=float, default=-1.0)
    ap.add_argument("--hire-confirm-bins", type=int, default=4)
    ap.add_argument("--hire-cooldown-bins", type=int, default=0)
    ap.add_argument("--hire-spatial-kernel", type=int, default=5)
    ap.add_argument("--hire-gate-pool", type=str, default="max", choices=["max", "avg"])
    ap.add_argument("--hire-reset-open", type=int, default=15)
    ap.add_argument("--hire-reset-grow", type=int, default=6)
    ap.add_argument("--hire-normalize", type=str, default="true")
    ap.add_argument("--hire-quantile", type=float, default=1.0)
    return ap.parse_args()


def _as_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    lowered = str(text).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean-like value, got {text!r}")


def _apply_input_gamma(frame_tchw: torch.Tensor, gamma: float) -> torch.Tensor:
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError(f"input_gamma must be positive, got {gamma}")
    if abs(gamma - 1.0) < 1e-8:
        return frame_tchw
    return torch.pow(torch.clamp(frame_tchw, 0.0, 1.0), 1.0 / gamma)


def _build_preprocessor_kwargs(args, preprocessor: str) -> dict[str, Any]:
    name = str(preprocessor).strip().lower()
    subsampling = int(args.chunk_size)
    if name == "sum":
        return {
            "subsampling": subsampling,
            "normalize": _as_bool(args.sum_normalize),
            "quantile": float(args.sum_quantile),
        }
    if name == "ema":
        return {
            "subsampling": subsampling,
            "ema_alpha": float(args.ema_alpha),
            "normalize": _as_bool(args.ema_normalize),
            "quantile": float(args.ema_quantile),
        }
    if name == "ppb":
        return {
            "subsampling": subsampling,
            "bocpd_gamma": float(args.ppb_bocpd_gamma),
            "normalize": _as_bool(args.ppb_normalize),
            "quantile": float(args.ppb_quantile),
            "min_filter_size": int(args.ppb_min_filter_size),
        }
    if name == "hire":

        def _tau_or_none(val: float) -> float | None:
            return None if float(val) <= 0.0 else float(val)

        return {
            "subsampling": subsampling,
            "sample_rate_hz": float(args.spad_bin_rate_hz),
            "ref_rate_hz": float(args.hire_ref_rate_hz),
            "fast_bins": int(args.hire_fast_bins),
            "slow_bins": int(args.hire_slow_bins),
            "surprise_bins": int(args.hire_surprise_bins),
            "tau_fast": _tau_or_none(args.hire_tau_fast),
            "tau_slow": _tau_or_none(args.hire_tau_slow),
            "tau_surprise": _tau_or_none(args.hire_tau_surprise),
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
            "normalize": _as_bool(args.hire_normalize),
            "quantile": float(args.hire_quantile),
        }
    raise ValueError(f"Unsupported preprocessor for 2 kHz cache: {name!r}")


def _load_annotation_len(gt_path: str | Path) -> int:
    with Path(gt_path).open("r", encoding="utf-8") as f:
        return len(json.load(f))


def _build_chunk_records(
    *, total_raw_bins: int, n_gt: int, chunk_size: int, stride_bins: int, spad_bins_per_gt: int
) -> list[dict[str, Any]]:
    if chunk_size <= 0 or stride_bins <= 0 or spad_bins_per_gt <= 0:
        raise ValueError("chunk_size, stride_bins and spad_bins_per_gt must be positive.")
    if chunk_size % spad_bins_per_gt != 0:
        raise ValueError(
            f"chunk_size={chunk_size} must be divisible by spad_bins_per_gt={spad_bins_per_gt}."
        )
    if stride_bins % spad_bins_per_gt != 0:
        raise ValueError(
            f"stride_bins={stride_bins} must be divisible by spad_bins_per_gt={spad_bins_per_gt}."
        )
    gt_chunk = chunk_size / spad_bins_per_gt
    records = []
    for chunk_index, spad_start in enumerate(range(0, max(total_raw_bins - chunk_size + 1, 0), stride_bins)):
        spad_end = spad_start + chunk_size
        gt_start = spad_start // spad_bins_per_gt
        target_gt_time = gt_start + gt_chunk
        if target_gt_time > (n_gt - 1):
            break
        records.append(
            {
                "chunk_index": int(chunk_index),
                "gt_start": int(gt_start),
                "spad_start_bin": int(spad_start),
                "spad_end_bin": int(spad_end),
                "target_gt_time": float(target_gt_time),
                "chunk_size": int(chunk_size),
            }
        )
    return records


def _render_frame(
    *,
    preprocessor,
    packed_chunk: np.ndarray,
    packed_nch: int,
    packed_ch_order: str,
    input_gamma: float,
    device: torch.device,
) -> np.ndarray:
    """Return one CHW float32 RGB frame (no confidence)."""
    raw_chunk = packed_frames_to_raw_video(packed_chunk, ch_order=packed_ch_order)
    cube = raw_plane_to_photon_cube(raw_chunk_plane(raw_chunk, packed_nch=packed_nch), device=device, as_bool=True)
    recons, _confidence = preprocessor.process_photon_cube_to_frame(cube, clear_states=True)
    rgb = raw_hwt_to_rgb_float(recons.float(), packed_nch=packed_nch)
    if int(rgb.shape[0]) <= 0:
        raise ValueError("Frame preprocessor emitted zero frames for one chunk.")
    rgb = _apply_input_gamma(rgb[-1:].contiguous(), input_gamma).squeeze(0)
    return rgb.detach().cpu().numpy().astype(np.float32, copy=False)


def _load_split_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), dict):
        raise ValueError(f"Expected split JSON with top-level 'samples' mapping: {path}")
    return payload


def _update_payload_entry(
    payload: dict[str, Any], sample: dict[str, str], *, preprocessor: str, render_dir: Path
) -> None:
    """Write frames + meta paths only (no confidence keys)."""
    samples = payload["samples"]
    sample_entry = samples.get(str(sample["id"]))
    if not isinstance(sample_entry, dict):
        raise KeyError(f"Missing sample entry for id={sample['id']!r} while updating JSON.")
    prefix = str(preprocessor).strip().lower()
    sample_entry[prefix] = str(render_dir / "frames.npy")
    sample_entry[f"{prefix}_meta"] = str(render_dir / "meta.json")
    sample_entry[f"render_{prefix}"] = str(render_dir)
    sample_entry[f"render_{prefix}_frames"] = str(render_dir / "frames.npy")
    sample_entry[f"render_{prefix}_meta"] = str(render_dir / "meta.json")
    # Drop stale confidence keys if re-running on an older JSON.
    for key in (f"{prefix}_confidence", f"render_{prefix}_confidence"):
        sample_entry.pop(key, None)


def _write_sample_cache(
    *,
    sample: dict[str, str],
    args,
    preprocessor_name: str,
    preprocessor,
    render_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any] | None:
    sample_name = str(sample["name"])
    tag = str(args.render_tag).strip()
    if str(args.output_root).strip():
        leaf = f"renders-{preprocessor_name}-{tag}" if tag else f"renders-{preprocessor_name}"
        render_dir = sample_render_dir(Path(args.output_root) / leaf, sample_name)
    else:
        render_dir = sibling_sample_render_dir(
            sample["spad"],
            preprocessor=preprocessor_name,
            sample_name=sample_name,
            source_render_dirname=args.source_render_dirname,
            render_tag=tag,
        )

    frames_path = render_dir / "frames.npy"
    meta_path = render_dir / "meta.json"
    if not args.overwrite and frames_path.exists() and meta_path.exists():
        print(f"[skip] {preprocessor_name} {sample_name}: cache already exists")
        return {"render_dir": render_dir}

    packed = np.load(sample["spad"], mmap_mode="r")
    n_gt = _load_annotation_len(sample["gt"])
    chunk_records = _build_chunk_records(
        total_raw_bins=int(packed.shape[0]),
        n_gt=n_gt,
        chunk_size=int(args.chunk_size),
        stride_bins=int(args.stride_bins),
        spad_bins_per_gt=int(args.spad_bins_per_gt),
    )
    if not chunk_records:
        print(f"[skip] {preprocessor_name} {sample_name}: no valid chunks")
        return None

    packed_nch = infer_packed_nch(packed)
    first = chunk_records[0]
    first_chunk = np.asarray(packed[first["spad_start_bin"] : first["spad_end_bin"]])
    first_frame = _render_frame(
        preprocessor=preprocessor,
        packed_chunk=first_chunk,
        packed_nch=packed_nch,
        packed_ch_order=args.packed_ch_order,
        input_gamma=float(args.input_gamma),
        device=device,
    )

    render_dir.mkdir(parents=True, exist_ok=True)
    frames_mm = np.lib.format.open_memmap(
        frames_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(chunk_records),) + tuple(first_frame.shape),
    )
    frames_mm[0] = first_frame

    for chunk in chunk_records[1:]:
        packed_chunk = np.asarray(packed[chunk["spad_start_bin"] : chunk["spad_end_bin"]])
        frame = _render_frame(
            preprocessor=preprocessor,
            packed_chunk=packed_chunk,
            packed_nch=packed_nch,
            packed_ch_order=args.packed_ch_order,
            input_gamma=float(args.input_gamma),
            device=device,
        )
        frames_mm[chunk["chunk_index"]] = frame

    del frames_mm

    for chunk in chunk_records:
        chunk["packed_nch"] = int(packed_nch)

    meta = {
        "version": CACHE_META_VERSION,
        "sample_id": str(sample["id"]),
        "sample_name": sample_name,
        "render_dir": str(render_dir),
        "source_spad": str(sample["spad"]),
        "source_gt": str(sample["gt"]),
        "preprocessor": str(preprocessor_name).strip().lower(),
        "render_tag": tag,
        "source_render_dirname": str(args.source_render_dirname),
        "num_frames": len(chunk_records),
        "frame_shape": list(first_frame.shape),
        "frame_dtype": "float32",
        "has_confidence": False,
        "packed_nch": int(packed_nch),
        "config": render_config,
        "config_fingerprint": render_config_fingerprint(render_config),
        "chunks": chunk_records,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[ok] {preprocessor_name} {sample_name}: {len(chunk_records)} chunks -> {render_dir} "
        f"shape={tuple(first_frame.shape)}"
    )
    return {"render_dir": str(render_dir)}


def main():
    args = parse_args()
    preprocessors = [str(x).strip().lower() for x in args.preprocessors]
    split_paths = [Path(p) for p in args.split_json]
    json_outputs = [Path(p) for p in args.json_output] if args.json_output else []
    if json_outputs and len(json_outputs) != len(split_paths):
        raise ValueError(
            f"--json-output length ({len(json_outputs)}) must match --split-json ({len(split_paths)})."
        )

    device = torch.device(args.device)
    output_root = Path(args.output_root) if str(args.output_root).strip() else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"2kHz cache: preprocessors={preprocessors} chunk={args.chunk_size} "
        f"stride={args.stride_bins} bins_per_gt={args.spad_bins_per_gt} "
        f"source={args.source_render_dirname} tag={args.render_tag!r} "
        f"ema_alpha={args.ema_alpha} ppb_gamma={args.ppb_bocpd_gamma} "
        f"ppb_min_filter={args.ppb_min_filter_size} (frames+meta only, no confidence)"
    )

    for split_idx, split_path in enumerate(split_paths):
        payload = _load_split_payload(split_path) if args.update_json else None
        samples = load_visionsim_split_json(split_path)
        if int(args.limit_samples) > 0:
            samples = samples[: int(args.limit_samples)]

        written = 0
        for prep_name in preprocessors:
            kwargs = _build_preprocessor_kwargs(args, prep_name)
            preprocessor = build_spad_frame_preprocessor(prep_name, kwargs=kwargs).to(device)
            render_config = build_render_config(
                preprocessor=prep_name,
                chunk_size=int(args.chunk_size),
                stride_bins=int(args.stride_bins),
                spad_bins_per_gt=int(args.spad_bins_per_gt),
                packed_ch_order=args.packed_ch_order,
                input_gamma=float(args.input_gamma),
                extra_kwargs=kwargs,
            )
            print(
                f"=== {split_path.name} / {prep_name} "
                f"fingerprint={render_config_fingerprint(render_config)} ==="
            )
            for sample in samples:
                meta = _write_sample_cache(
                    sample=sample,
                    args=args,
                    preprocessor_name=prep_name,
                    preprocessor=preprocessor,
                    render_config=render_config,
                    device=device,
                )
                if meta is not None:
                    written += 1
                    if payload is not None:
                        _update_payload_entry(
                            payload,
                            sample,
                            preprocessor=prep_name,
                            render_dir=Path(meta["render_dir"]),
                        )

        if payload is not None:
            out_path = json_outputs[split_idx] if json_outputs else split_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Updated split JSON: {out_path}")

        print(
            f"Done split={split_path} samples={len(samples)} write_ops={written} "
            f"dirs=renders-{{method}}-{args.render_tag} from {args.source_render_dirname}"
        )


if __name__ == "__main__":
    main()
