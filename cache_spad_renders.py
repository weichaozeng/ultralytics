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


def parse_args():
    ap = argparse.ArgumentParser(description="Offline cache for SPAD chunk renders.")
    ap.add_argument("--split-json", type=str, required=True, help="Path to VisionSIM split JSON.")
    ap.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Optional explicit cache root. If omitted, writes beside each sample's renders-spc8kHz tree.",
    )
    ap.add_argument("--preprocessor", type=str, choices=["sum", "ppb", "stea", "pdrs"], required=True)
    ap.add_argument("--chunk-size", type=int, default=320, help="Raw-bin chunk size per rendered frame.")
    ap.add_argument("--stride-bins", type=int, default=320, help="Stride in raw bins between cached chunks.")
    ap.add_argument("--spad-bins-per-gt", type=int, default=64, help="Raw bins corresponding to one GT frame.")
    ap.add_argument("--packed-ch-order", type=str, default="RGB", help="Packed SPAD channel order.")
    ap.add_argument("--input-gamma", type=float, default=2.2, help="Gamma correction applied to cached RGB frames.")
    ap.add_argument("--device", type=str, default="cuda:0", help="Torch device for preprocessing.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite per-sample cache outputs if they exist.")
    ap.add_argument("--limit-samples", type=int, default=0, help="Optional number of samples to process for smoke tests.")
    ap.add_argument(
        "--update-json",
        action="store_true",
        help="Write explicit render cache paths back into the split JSON after caching.",
    )
    ap.add_argument(
        "--json-output",
        type=str,
        default="",
        help="Optional output path for the updated split JSON. Defaults to in-place update when --update-json is set.",
    )
    ap.add_argument(
        "--source-render-dirname",
        type=str,
        default="renders-spc8kHz",
        help="Name of the source packed-SPAD render directory used to infer sibling render roots.",
    )
    ap.add_argument("--ppb-bocpd-gamma", type=float, default=5e-4)
    ap.add_argument("--ppb-quantile", type=float, default=1.0)
    ap.add_argument("--ppb-normalize", type=str, default="true")
    ap.add_argument("--ppb-min-filter-size", type=int, default=7)
    ap.add_argument("--stea-fast-window", type=int, default=32)
    ap.add_argument("--stea-slow-window", type=int, default=128)
    ap.add_argument("--stea-temporal-window", type=int, default=5)
    ap.add_argument("--stea-fast-tau", type=float, default=6.0)
    ap.add_argument("--stea-motion-sharpness", type=float, default=60.0)
    ap.add_argument("--stea-motion-threshold", type=float, default=0.07)
    ap.add_argument("--stea-stable-prior", type=float, default=16.0)
    ap.add_argument("--stea-normalize", type=str, default="true")
    ap.add_argument("--stea-quantile", type=float, default=1.0)
    ap.add_argument("--pdrs-fast-window", type=int, default=32)
    ap.add_argument("--pdrs-slow-window", type=int, default=128)
    ap.add_argument("--pdrs-temporal-window", type=int, default=5)
    ap.add_argument("--pdrs-fast-tau", type=float, default=6.0)
    ap.add_argument("--pdrs-motion-sharpness", type=float, default=60.0)
    ap.add_argument("--pdrs-motion-threshold", type=float, default=0.07)
    ap.add_argument("--pdrs-stable-prior", type=float, default=16.0)
    ap.add_argument("--pdrs-normalize", type=str, default="true")
    ap.add_argument("--pdrs-quantile", type=float, default=1.0)
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


def _build_pdrs_kwargs(args, *, subsampling: int, chunk_size: int) -> dict[str, Any]:
    return {
        "subsampling": int(subsampling),
        "chunk_size": int(chunk_size),
        "fast_window": int(args.pdrs_fast_window),
        "slow_window": int(args.pdrs_slow_window),
        "temporal_window": int(args.pdrs_temporal_window),
        "fast_tau": float(args.pdrs_fast_tau),
        "motion_sharpness": float(args.pdrs_motion_sharpness),
        "motion_threshold": float(args.pdrs_motion_threshold),
        "stable_prior": float(args.pdrs_stable_prior),
        "normalize": _as_bool(args.pdrs_normalize),
        "quantile": float(args.pdrs_quantile),
    }


def _build_preprocessor_kwargs(args) -> dict[str, Any]:
    name = str(args.preprocessor).strip().lower()
    subsampling = int(args.spad_bins_per_gt)
    chunk_size = int(args.chunk_size)
    if name == "sum":
        return {"subsampling": subsampling}
    if name == "ppb":
        return {
            "subsampling": subsampling,
            "bocpd_gamma": float(args.ppb_bocpd_gamma),
            "normalize": _as_bool(args.ppb_normalize),
            "quantile": float(args.ppb_quantile),
            "min_filter_size": int(args.ppb_min_filter_size),
        }
    if name == "stea":
        return {
            "subsampling": subsampling,
            "fast_window": int(args.stea_fast_window),
            "slow_window": int(args.stea_slow_window),
            "temporal_window": int(args.stea_temporal_window),
            "fast_tau": float(args.stea_fast_tau),
            "motion_sharpness": float(args.stea_motion_sharpness),
            "motion_threshold": float(args.stea_motion_threshold),
            "stable_prior": float(args.stea_stable_prior),
            "normalize": _as_bool(args.stea_normalize),
            "quantile": float(args.stea_quantile),
        }
    if name == "pdrs":
        return _build_pdrs_kwargs(args, subsampling=subsampling, chunk_size=chunk_size)
    raise ValueError(f"Unsupported preprocessor: {name!r}")


def _load_annotation_len(gt_path: str | Path) -> int:
    with Path(gt_path).open("r", encoding="utf-8") as f:
        ann = json.load(f)
    return len(ann)


def _build_chunk_records(*, total_raw_bins: int, n_gt: int, chunk_size: int, stride_bins: int, spad_bins_per_gt: int) -> list[dict[str, Any]]:
    if chunk_size <= 0 or stride_bins <= 0 or spad_bins_per_gt <= 0:
        raise ValueError("chunk_size, stride_bins and spad_bins_per_gt must be positive.")
    if chunk_size % spad_bins_per_gt != 0:
        raise ValueError(
            f"chunk_size={chunk_size} must be divisible by spad_bins_per_gt={spad_bins_per_gt} "
            "to preserve integer GT frame alignment."
        )
    if stride_bins % spad_bins_per_gt != 0:
        raise ValueError(
            f"stride_bins={stride_bins} must be divisible by spad_bins_per_gt={spad_bins_per_gt} "
            "to preserve integer GT frame alignment."
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


def _render_chunk(*, preprocessor, packed_chunk: np.ndarray, packed_nch: int, packed_ch_order: str, input_gamma: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    raw_chunk = packed_frames_to_raw_video(packed_chunk, ch_order=packed_ch_order)
    cube = raw_plane_to_photon_cube(raw_chunk_plane(raw_chunk, packed_nch=packed_nch), device=device, as_bool=True)
    recons, confidence = preprocessor.process_photon_cube_to_frame(cube, clear_states=True)
    rgb = raw_hwt_to_rgb_float(recons.float(), packed_nch=packed_nch)
    if int(rgb.shape[0]) <= 0:
        raise ValueError("Frame preprocessor emitted zero frames for one chunk.")
    rgb = _apply_input_gamma(rgb[-1:].contiguous(), input_gamma).squeeze(0)
    conf = confidence.float()
    return rgb.detach().cpu().numpy().astype(np.float32, copy=False), conf.detach().cpu().numpy().astype(np.float32, copy=False)


def _load_split_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), dict):
        raise ValueError(f"Expected split JSON with top-level 'samples' mapping: {path}")
    return payload


def _update_payload_entry(payload: dict[str, Any], sample: dict[str, str], *, preprocessor: str, render_dir: Path) -> None:
    samples = payload["samples"]
    sample_entry = samples.get(str(sample["id"]))
    if not isinstance(sample_entry, dict):
        raise KeyError(f"Missing sample entry for id={sample['id']!r} while updating JSON.")
    prefix = str(preprocessor).strip().lower()
    sample_entry[prefix] = str(render_dir / "frames.npy")
    sample_entry[f"{prefix}_confidence"] = str(render_dir / "confidence.npy")
    sample_entry[f"{prefix}_meta"] = str(render_dir / "meta.json")
    # Backward-compatible explicit directory key for older cache consumers.
    sample_entry[f"render_{prefix}"] = str(render_dir)
    sample_entry[f"render_{prefix}_frames"] = str(render_dir / "frames.npy")
    sample_entry[f"render_{prefix}_confidence"] = str(render_dir / "confidence.npy")
    sample_entry[f"render_{prefix}_meta"] = str(render_dir / "meta.json")


def _write_sample_cache(*, sample: dict[str, str], args, preprocessor, render_config: dict[str, Any], device: torch.device) -> dict[str, Any] | None:
    sample_name = str(sample["name"])
    if str(args.output_root).strip():
        render_dir = sample_render_dir(args.output_root, sample_name)
    else:
        render_dir = sibling_sample_render_dir(
            sample["spad"],
            preprocessor=args.preprocessor,
            sample_name=sample_name,
            source_render_dirname=args.source_render_dirname,
        )
    frames_path = render_dir / "frames.npy"
    confidence_path = render_dir / "confidence.npy"
    meta_path = render_dir / "meta.json"
    if not args.overwrite and frames_path.exists() and confidence_path.exists() and meta_path.exists():
        print(f"[skip] {sample_name}: cache already exists")
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
        print(f"[skip] {sample_name}: no valid chunks")
        return None

    packed_nch = infer_packed_nch(packed)
    first = chunk_records[0]
    first_chunk = np.asarray(packed[first["spad_start_bin"] : first["spad_end_bin"]])
    first_frame, first_conf = _render_chunk(
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
    confidence_mm = np.lib.format.open_memmap(
        confidence_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(chunk_records),) + tuple(first_conf.shape),
    )
    frames_mm[0] = first_frame
    confidence_mm[0] = first_conf

    for chunk in chunk_records[1:]:
        packed_chunk = np.asarray(packed[chunk["spad_start_bin"] : chunk["spad_end_bin"]])
        frame, conf = _render_chunk(
            preprocessor=preprocessor,
            packed_chunk=packed_chunk,
            packed_nch=packed_nch,
            packed_ch_order=args.packed_ch_order,
            input_gamma=float(args.input_gamma),
            device=device,
        )
        frames_mm[chunk["chunk_index"]] = frame
        confidence_mm[chunk["chunk_index"]] = conf

    del frames_mm
    del confidence_mm

    for chunk in chunk_records:
        chunk["packed_nch"] = int(packed_nch)

    meta = {
        "version": CACHE_META_VERSION,
        "sample_id": str(sample["id"]),
        "sample_name": sample_name,
        "render_dir": str(render_dir),
        "source_spad": str(sample["spad"]),
        "source_gt": str(sample["gt"]),
        "preprocessor": str(args.preprocessor).strip().lower(),
        "num_frames": len(chunk_records),
        "frame_shape": list(first_frame.shape),
        "confidence_shape": list(first_conf.shape),
        "frame_dtype": "float32",
        "confidence_dtype": "float32",
        "packed_nch": int(packed_nch),
        "config": render_config,
        "config_fingerprint": render_config_fingerprint(render_config),
        "chunks": chunk_records,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[ok] {sample_name}: {len(chunk_records)} chunks -> {render_dir} "
        f"shape={tuple(first_frame.shape)} conf={tuple(first_conf.shape)}"
    )
    meta["render_dir"] = str(render_dir)
    return meta


def main():
    args = parse_args()
    payload = _load_split_payload(args.split_json) if args.update_json else None
    samples = load_visionsim_split_json(args.split_json)
    if int(args.limit_samples) > 0:
        samples = samples[: int(args.limit_samples)]

    device = torch.device(args.device)
    preprocessor_kwargs = _build_preprocessor_kwargs(args)
    preprocessor = build_spad_frame_preprocessor(args.preprocessor, kwargs=preprocessor_kwargs).to(device)
    render_config = build_render_config(
        preprocessor=args.preprocessor,
        chunk_size=int(args.chunk_size),
        stride_bins=int(args.stride_bins),
        spad_bins_per_gt=int(args.spad_bins_per_gt),
        packed_ch_order=args.packed_ch_order,
        input_gamma=float(args.input_gamma),
        extra_kwargs=preprocessor_kwargs,
    )
    output_root = Path(args.output_root) if str(args.output_root).strip() else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    written = 0
    for sample in samples:
        meta = _write_sample_cache(
            sample=sample,
            args=args,
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
                    preprocessor=args.preprocessor,
                    render_dir=Path(meta["render_dir"]),
                )

    if payload is not None:
        json_output = Path(args.json_output) if str(args.json_output).strip() else Path(args.split_json)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        with json_output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Updated split JSON: {json_output}")

    print(
        f"Done. processed_samples={len(samples)} newly_written={written} "
        f"fingerprint={render_config_fingerprint(render_config)} "
        f"output_root={output_root if output_root is not None else '<sibling-to-renders-spc8kHz>'}"
    )


if __name__ == "__main__":
    main()
