# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Visualize STEA routing evidence and reconstructions.

Writes per-chunk outputs under ``{save_dir}/{sample}/videoXXXXX/``:

- ``{stem}_stea_stats.txt`` — percentile summary of STEA evidence tensors
- ``{stem}_stea_compare.png`` — sum vs stea reconstruction (side-by-side)
- ``{stem}_stea_scores.png`` — heatmaps of KL evidence, routing weights, temporal peaks, and temporal bases

Example
-------
python ultralytics/vis_stea_det.py \\
  --in_path /path/to/sample \\
  --save_dir /path/to/stea_vis \\
  --chunk_size 320
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics.data.spad_packed import (
    infer_packed_nch,
    is_packed_spad,
    packed_frames_to_raw_video,
    raw_hwt_to_rgb_float,
    raw_plane_to_photon_cube,
    sum_raw_chunk_to_rgb,
)
from ultralytics.quanta_stea_networks.integrator import SpatioTemporalEvidenceAccumulation

COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "hot": cv2.COLORMAP_HOT,
}

@dataclass(frozen=True)
class SpadSource:
    array: np.ndarray
    layout: str
    packed_nch: int


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
        return np.ascontiguousarray(
            np.transpose(source.array[:, :, t0:t1], (2, 0, 1))[:, :, :, None].astype(np.uint8, copy=False)
        )
    raise ValueError(f"Unsupported layout: {source.layout}")


def _rgb_tensor_to_bgr_u8(
    frames_tchw: torch.Tensor, *, vis_mode: str, percentile: float, gamma: float
) -> list[np.ndarray]:
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


def _resolve_device(device: str) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _output_dir(save_root: Path, sample_name: str, video_idx: int) -> Path:
    return save_root / sample_name / f"video{video_idx:05d}"


def _resize_map_to_display(score_hw: np.ndarray, display_hw: tuple[int, int]) -> np.ndarray:
    disp_h, disp_w = display_hw
    return cv2.resize(score_hw.astype(np.float32), (disp_w, disp_h), interpolation=cv2.INTER_AREA)


def _value_to_heatmap(values: np.ndarray, cmap_id: int, *, vmax: float) -> np.ndarray:
    vmax = max(float(vmax), 1e-8)
    u8 = (np.clip(values / vmax, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return cv2.applyColorMap(u8, cmap_id)


def _value_to_gray_bgr(values: np.ndarray, *, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Monotonic visualization: black is low, white is high."""
    denom = max(float(vmax) - float(vmin), 1e-8)
    u8 = (np.clip((values - float(vmin)) / denom, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def _resolve_vmax(values: np.ndarray, *, fixed_vmax: float, percentile: float) -> float:
    if fixed_vmax > 0.0:
        return float(fixed_vmax)
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 1.0
    return float(np.percentile(flat, percentile))


def _percentile_summary(values: np.ndarray, *, name: str, percentiles: tuple[float, ...]) -> list[str]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return [f"[{name}] n=0 (empty)"]
    lines = [f"[{name}] n={flat.size} min={flat.min():.6f} max={flat.max():.6f} mean={flat.mean():.6f}"]
    for p in percentiles:
        lines.append(f"  p{p:g} = {float(np.percentile(flat, p)):.6f}")
    return lines


def _label_panel(img_bgr: np.ndarray, text: str, label_h: int = 26) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    header = np.zeros((label_h, w, 3), dtype=np.uint8)
    cv2.putText(
        header,
        text,
        (8, int(label_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, img_bgr])


def _stitch_panels(panels: list[np.ndarray], labels: list[str], gap: int = 6) -> np.ndarray:
    target_h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != target_h:
            new_w = max(int(round(p.shape[1] * target_h / p.shape[0])), 1)
            p = cv2.resize(p, (new_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(p)
    labeled = [_label_panel(img, lab) for img, lab in zip(resized, labels)]
    sep = np.full((labeled[0].shape[0], gap, 3), 32, dtype=np.uint8)
    out = labeled[0]
    for nxt in labeled[1:]:
        out = np.hstack([out, sep, nxt])
    return out


def _select_time_indices(length: int, max_slices: int) -> list[int]:
    if length <= 0:
        return []
    max_slices = max(int(max_slices), 1)
    if length <= max_slices:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, num=max_slices).round().astype(int).tolist()))


def _panel_from_map(
    score_map: np.ndarray,
    *,
    display_hw: tuple[int, int],
    cmap_id: int,
    mode: str,
    vmax: float,
    vmin: float = 0.0,
) -> np.ndarray:
    disp = _resize_map_to_display(score_map, display_hw)
    if mode == "heatmap":
        return _value_to_heatmap(disp, cmap_id, vmax=vmax)
    if mode == "gray":
        return _value_to_gray_bgr(disp, vmin=vmin, vmax=vmax)
    raise ValueError(f"Unsupported panel mode: {mode}")


def _temporal_strip(
    volume_hwt: np.ndarray,
    *,
    display_hw: tuple[int, int],
    cmap_id: int,
    label_prefix: str,
    mode: str,
    vmax: float,
    vmin: float = 0.0,
    max_slices: int,
) -> np.ndarray | None:
    if volume_hwt.ndim != 3 or int(volume_hwt.shape[-1]) <= 0:
        return None
    panels = []
    labels = []
    for ti in _select_time_indices(int(volume_hwt.shape[-1]), max_slices):
        panels.append(
            _panel_from_map(
                volume_hwt[..., ti],
                display_hw=display_hw,
                cmap_id=cmap_id,
                mode=mode,
                vmax=vmax,
                vmin=vmin,
            )
        )
        labels.append(f"{label_prefix}[t={ti}]")
    if not panels:
        return None
    return _stitch_panels(panels, labels)


def _process_chunk_with_full_debug(
    stea: SpatioTemporalEvidenceAccumulation,
    raw: torch.Tensor,
    *,
    clear_states: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run full temporal STEA integration so per-chunk route_weight peaks are available."""
    if clear_states:
        stea.t_absolute = 0
        stea._clear_histories()
    stea.set_cube(raw)
    fused, motion_debug = stea._integrate_full_with_debug(raw)
    recons = stea._subsample_reconstruction(fused)
    motion_debug["recons_prenorm"] = stea._subsample_reconstruction(motion_debug["recons_prenorm"])
    recons = stea.clamp_recons(recons)
    stea.t_absolute += int(raw.shape[-1])
    return recons, motion_debug


def _save_visuals(
    *,
    out_dir: Path,
    stem: str,
    recon_bgr: np.ndarray,
    sum_bgr: np.ndarray,
    motion_debug: dict[str, torch.Tensor],
    cmap_id: int,
    score_vmax: float,
    score_percentile: float,
    fast_window: int,
    slow_window: int,
    temporal_window: int,
    fast_tau: float,
    motion_sharpness: float,
    motion_threshold: float,
    blend_const: float,
    temporal_slices: int,
) -> None:
    percentiles = (1.0, 5.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)
    display_hw = recon_bgr.shape[:2]

    p_motion_hwt = motion_debug["p_motion"].detach().float().cpu().numpy()
    route_weight_hwt = motion_debug["route_weight"].detach().float().cpu().numpy()
    valid_weight_hwt = motion_debug["valid_weight"].detach().float().cpu().numpy()
    k_raw_hwt = motion_debug["k_raw"].detach().float().cpu().numpy()
    k_smoothed_hwt = motion_debug["k_smoothed"].detach().float().cpu().numpy()
    future_gain_hwt = np.clip(route_weight_hwt - p_motion_hwt, 0.0, 1.0)
    maps = {
        "k_raw": motion_debug["k_raw"][..., -1].detach().float().cpu().numpy(),
        "k_smoothed": motion_debug["k_smoothed_last"].detach().float().cpu().numpy(),
        "k_smoothed_peak": k_smoothed_hwt.max(axis=-1),
        "p_motion": motion_debug["p_motion_last"].detach().float().cpu().numpy(),
        "future_motion": motion_debug["future_motion_last"].detach().float().cpu().numpy(),
        "future_minus_p": future_gain_hwt[..., -1],
        "p_motion_peak": p_motion_hwt.max(axis=-1),
        "route_weight_peak": route_weight_hwt.max(axis=-1),
        "valid_weight": motion_debug["valid_weight_last"].detach().float().cpu().numpy(),
        "valid_weight_sum": motion_debug["stable_support"].detach().float().cpu().numpy() / max(
            float(motion_debug["valid_weight"].shape[-1]), 1.0
        ),
        "y_fast": motion_debug["y_scales_last"][..., 0].detach().float().cpu().numpy(),
        "y_slow": motion_debug["y_scales_last"][..., 1].detach().float().cpu().numpy(),
        "mean_stable": motion_debug["mean_stable"].detach().float().cpu().numpy(),
        "w_mean": motion_debug["w_mean"].detach().float().cpu().numpy(),
        "fused": motion_debug["fused_last"].detach().float().cpu().numpy(),
    }
    k_vmax_scale = _resolve_vmax(k_smoothed_hwt, fixed_vmax=score_vmax, percentile=score_percentile)

    score_panels = []
    score_labels = []
    for label, score_map in maps.items():
        if label in {"k_raw", "k_smoothed", "k_smoothed_peak"}:
            panel = _panel_from_map(
                score_map,
                display_hw=display_hw,
                cmap_id=cmap_id,
                mode="heatmap",
                vmax=k_vmax_scale,
            )
        else:
            panel = _panel_from_map(
                score_map,
                display_hw=display_hw,
                cmap_id=cmap_id,
                mode="gray",
                vmax=1.0,
            )
        score_panels.append(panel)
        score_labels.append(label)
    cv2.imwrite(str(out_dir / f"{stem}_stea_scores.png"), _stitch_panels(score_panels, score_labels))

    temporal_rows = []
    temporal_rows.append(
        _temporal_strip(
            k_smoothed_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="k_smoothed",
            mode="heatmap",
            vmax=k_vmax_scale,
            max_slices=temporal_slices,
        )
    )
    temporal_rows.append(
        _temporal_strip(
            p_motion_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="p_motion",
            mode="gray",
            vmax=1.0,
            max_slices=temporal_slices,
        )
    )
    temporal_rows.append(
        _temporal_strip(
            route_weight_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="future_motion",
            mode="gray",
            vmax=1.0,
            max_slices=temporal_slices,
        )
    )
    temporal_rows.append(
        _temporal_strip(
            future_gain_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="future_minus_p",
            mode="gray",
            vmax=1.0,
            max_slices=temporal_slices,
        )
    )
    temporal_rows.append(
        _temporal_strip(
            valid_weight_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="valid_weight",
            mode="gray",
            vmax=1.0,
            max_slices=temporal_slices,
        )
    )
    temporal_rows = [row for row in temporal_rows if row is not None]
    if temporal_rows:
        row_w = max(row.shape[1] for row in temporal_rows)
        padded_rows = []
        for row in temporal_rows:
            if row.shape[1] < row_w:
                pad = np.full((row.shape[0], row_w - row.shape[1], 3), 24, dtype=np.uint8)
                row = np.hstack([row, pad])
            padded_rows.append(row)
        sep_h = np.full((8, row_w, 3), 24, dtype=np.uint8)
        temporal_canvas = padded_rows[0]
        for row in padded_rows[1:]:
            temporal_canvas = np.vstack([temporal_canvas, sep_h, row])
        cv2.imwrite(str(out_dir / f"{stem}_stea_temporal.png"), temporal_canvas)

    cv2.imwrite(
        str(out_dir / f"{stem}_stea_compare.png"),
        _stitch_panels([sum_bgr, recon_bgr], ["sum", "stea"]),
    )

    route_from_k = 1.0 / (
        1.0 + np.exp(-float(motion_sharpness) * (maps["k_smoothed"] - float(motion_threshold)))
    )
    route_abs_err = np.abs(route_from_k - maps["p_motion"])
    future_gain_last = maps["future_minus_p"]
    stats_lines = [
        f"stem={stem}",
        f"fast_window={fast_window} slow_window={slow_window} temporal_window={temporal_window}",
        f"fast_tau={fast_tau} motion_sharpness={motion_sharpness} "
        f"motion_threshold={motion_threshold} blend_const={blend_const}",
        f"k_smoothed_vmax={k_vmax_scale:.6f} (fixed={score_vmax:g}, percentile={score_percentile:g})",
        "k_smoothed is causal-smoothed Bernoulli KL evidence.",
        "All grayscale routing maps use a fixed [0, 1] range so brightness is monotonic.",
        "p_motion = sigmoid(motion_sharpness * (k_smoothed - motion_threshold))",
        "future_motion = flip(cummax(flip(P_motion)))",
        "future_minus_p highlights where reverse-cummax expands motion support beyond the raw sigmoid map.",
        "w_mean = L / (L + blend_const), where L=sum(1-Future_Motion)",
        "k_smoothed_peak = max(k_smoothed) over all frames in the chunk",
        f"route_raw_from_k_abs_err_max={float(route_abs_err.max()):.8f} mean={float(route_abs_err.mean()):.8f}",
        f"future_minus_p_last_max={float(future_gain_last.max()):.8f} mean={float(future_gain_last.mean()):.8f}",
        "",
    ]
    for label, values in maps.items():
        stats_lines.extend(_percentile_summary(values, name=label, percentiles=percentiles))
        stats_lines.append("")
    for label, values in {
        "k_smoothed_hwt": k_smoothed_hwt,
        "p_motion_hwt": p_motion_hwt,
        "future_motion_hwt": route_weight_hwt,
        "future_minus_p_hwt": future_gain_hwt,
        "valid_weight_hwt": valid_weight_hwt,
    }.items():
        stats_lines.extend(_percentile_summary(values, name=label, percentiles=percentiles))
        stats_lines.append("")
    (out_dir / f"{stem}_stea_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize STEA integrator scores and reconstructions")
    ap.add_argument("--in_path", type=str, required=True, help="SPAD sample directory or .npy path")
    ap.add_argument("--save_dir", type=str, required=True, help="Output root directory")
    ap.add_argument("--chunk_size", type=int, default=320)
    ap.add_argument("--chunk_stride", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--stea_fast_window", type=int, default=16)
    ap.add_argument("--stea_slow_window", type=int, default=128)
    ap.add_argument("--stea_temporal_window", type=int, default=5)
    ap.add_argument("--stea_fast_tau", type=float, default=4.0)
    ap.add_argument("--stea_motion_sharpness", type=float, default=60.0)
    ap.add_argument("--stea_motion_threshold", type=float, default=0.05)
    ap.add_argument("--stea_eps", type=float, default=1e-5)
    ap.add_argument("--stea_blend_const", type=float, default=16.0)
    ap.add_argument("--stea_kernel_size", type=int, default=None, help="Deprecated alias for --stea_slow_window")
    ap.add_argument("--stea_prior_strength", type=float, default=1.0, help="Deprecated; ignored by STEA")
    ap.add_argument("--stea_gating_tau", type=float, default=0.1, help="Deprecated; ignored by STEA")
    ap.add_argument("--stea_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stea_quantile", type=float, default=1.0)
    ap.add_argument("--stea_max_filter_size", type=int, default=3, help="Deprecated; ignored by STEA")
    ap.add_argument("--vis_mode", type=str, default="linear", choices=["linear", "gamma", "percentile", "percentile_gamma"])
    ap.add_argument("--vis_percentile", type=float, default=99.5)
    ap.add_argument("--vis_gamma", type=float, default=2.2)
    ap.add_argument("--colormap", type=str, default="turbo", choices=sorted(COLORMAPS))
    ap.add_argument(
        "--score_vmax",
        type=float,
        default=0.0,
        help="Fixed max for score heatmaps (0 = use --score_percentile on all scales)",
    )
    ap.add_argument(
        "--score_percentile",
        type=float,
        default=99.5,
        help="Percentile vmax for score heatmaps when --score_vmax=0",
    )
    ap.add_argument(
        "--temporal_slices",
        type=int,
        default=6,
        help="How many evenly spaced time slices to show in the temporal STEA debug sheet",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    cmap_id = COLORMAPS[args.colormap]

    stea = SpatioTemporalEvidenceAccumulation(
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

    sample_name = in_path.name if in_path.is_dir() else in_path.stem
    stride = int(args.chunk_stride) if int(args.chunk_stride) > 0 else int(args.chunk_size)
    sources = list(_iter_sources(in_path))

    for video_idx, source in enumerate(tqdm(sources, desc=f"stea vis [{sample_name}]")):
        n_bins = _num_bins(source)
        out_dir = _output_dir(save_root, sample_name, video_idx)
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        for cube_idx, t0 in enumerate(range(0, n_bins, stride)):
            t1 = min(t0 + int(args.chunk_size), n_bins)
            raw_chunk = _slice_raw(source, t0, t1, packed_ch_order=args.packed_ch_order)
            if raw_chunk.shape[0] == 0:
                continue

            raw = raw_plane_to_photon_cube(
                raw_chunk[..., 0], device=device, as_bool=True
            )
            recons, motion_debug = _process_chunk_with_full_debug(
                stea, raw, clear_states=cube_idx == 0
            )
            if int(recons.shape[-1]) == 0:
                tqdm.write(f"Skip cube {cube_idx} (no temporal blocks): t{t0:06d}_{t1:06d}")
                continue

            vis_kw = dict(
                vis_mode=args.vis_mode,
                percentile=float(args.vis_percentile),
                gamma=float(args.vis_gamma),
            )
            recon_rgb = raw_hwt_to_rgb_float(recons.float(), packed_nch=source.packed_nch)
            sum_rgb = sum_raw_chunk_to_rgb(raw_chunk, packed_nch=source.packed_nch, device=device)

            recon_bgr = _rgb_tensor_to_bgr_u8(recon_rgb, **vis_kw)[0]
            sum_bgr = _rgb_tensor_to_bgr_u8(sum_rgb, **vis_kw)[0]

            stem = f"cube{cube_idx:05d}_t{t0:06d}_{t1:06d}_frame{frame_idx:07d}"
            _save_visuals(
                out_dir=out_dir,
                stem=stem,
                recon_bgr=recon_bgr,
                sum_bgr=sum_bgr,
                motion_debug=motion_debug,
                cmap_id=cmap_id,
                score_vmax=float(args.score_vmax),
                score_percentile=float(args.score_percentile),
                fast_window=int(args.stea_fast_window),
                slow_window=int(args.stea_kernel_size or args.stea_slow_window),
                temporal_window=int(args.stea_temporal_window),
                fast_tau=float(args.stea_fast_tau),
                motion_sharpness=float(args.stea_motion_sharpness),
                motion_threshold=float(args.stea_motion_threshold),
                blend_const=float(args.stea_blend_const),
                temporal_slices=int(args.temporal_slices),
            )
            frame_idx += 1

        if frame_idx == 0:
            print(
                f"Warning: no frames saved for {sample_name}/video{video_idx:05d} "
                f"(n_bins={n_bins}; check chunk_size={args.chunk_size})"
            )

    print(f"Saved STEA visualizations under {save_root / sample_name}")


if __name__ == "__main__":
    main()
