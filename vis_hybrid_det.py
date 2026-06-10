# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Visualize STEA routing evidence and reconstructions.

Writes per-chunk outputs under ``{save_dir}/{sample}/videoXXXXX/``:

- ``{stem}_hyb_stats.txt`` — percentile summary of STEA evidence tensors
- ``{stem}_hyb_compare.png`` — sum vs hyb reconstruction (side-by-side)
- ``{stem}_hyb_scores.png`` — heatmaps of KL evidence, routing weights, and temporal bases

Example
-------
python ultralytics/vis_hybrid_det.py \\
  --in_path /path/to/sample \\
  --save_dir /path/to/hyb_vis \\
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

from ultralytics.data.spad_packed import infer_packed_nch, is_packed_spad, packed_frames_to_raw_bayer
from ultralytics.quanta_hybrid_networks.integrator import SpatioTemporalEvidenceAccumulation

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
        raw = packed_frames_to_raw_bayer(packed, ch_order=packed_ch_order)
        return raw[:, :, :, None]
    if source.layout == "thwc1":
        return np.ascontiguousarray(source.array[t0:t1].astype(np.uint8, copy=False))
    if source.layout == "thw":
        return np.ascontiguousarray(source.array[t0:t1, :, :, None].astype(np.uint8, copy=False))
    if source.layout == "hwt":
        return np.ascontiguousarray(
            np.transpose(source.array[:, :, t0:t1], (2, 0, 1))[:, :, :, None].astype(np.uint8, copy=False)
        )
    raise ValueError(f"Unsupported layout: {source.layout}")


def _raw_hwt_to_rgb_float(raw_hwt: torch.Tensor, *, packed_nch: int) -> torch.Tensor:
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
    sharpness: float,
    bias: float,
) -> None:
    percentiles = (1.0, 5.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)
    display_hw = recon_bgr.shape[:2]

    maps = {
        "k_smoothed": motion_debug["k_smoothed_last"].detach().float().cpu().numpy(),
        "route_weight": motion_debug["route_weight_last"].detach().float().cpu().numpy(),
        "y_fast": motion_debug["y_scales_last"][..., 0].detach().float().cpu().numpy(),
        "y_slow": motion_debug["y_scales_last"][..., 1].detach().float().cpu().numpy(),
        "fused": motion_debug["fused_last"].detach().float().cpu().numpy(),
    }
    k_vmax_scale = _resolve_vmax(maps["k_smoothed"], fixed_vmax=score_vmax, percentile=score_percentile)

    score_panels = []
    score_labels = []
    for label, score_map in maps.items():
        disp = _resize_map_to_display(score_map, display_hw)
        if label == "k_smoothed":
            panel = _value_to_heatmap(disp, cmap_id, vmax=k_vmax_scale)
        else:
            panel = _value_to_gray_bgr(disp, vmin=0.0, vmax=1.0)
        score_panels.append(panel)
        score_labels.append(label)
    cv2.imwrite(str(out_dir / f"{stem}_hyb_scores.png"), _stitch_panels(score_panels, score_labels))

    cv2.imwrite(
        str(out_dir / f"{stem}_hyb_compare.png"),
        _stitch_panels([sum_bgr, recon_bgr], ["sum", "hyb"]),
    )

    route_from_k = 1.0 / (1.0 + np.exp(-float(sharpness) * (maps["k_smoothed"] - float(bias))))
    route_abs_err = np.abs(route_from_k - maps["route_weight"])
    stats_lines = [
        f"stem={stem}",
        f"fast_window={fast_window} slow_window={slow_window} temporal_window={temporal_window}",
        f"fast_tau={fast_tau} sharpness={sharpness} bias={bias}",
        f"k_smoothed_vmax={k_vmax_scale:.6f} (fixed={score_vmax:g}, percentile={score_percentile:g})",
        "k_smoothed is causal-smoothed Bernoulli variance-normalized evidence, not raw KL.",
        "route_weight visualization is fixed grayscale [0, 1] so brightness is monotonic.",
        "route_weight = sigmoid(sharpness * (k_smoothed - bias))",
        f"route_from_k_abs_err_max={float(route_abs_err.max()):.8f} mean={float(route_abs_err.mean()):.8f}",
        "",
    ]
    for label, values in maps.items():
        stats_lines.extend(_percentile_summary(values, name=label, percentiles=percentiles))
        stats_lines.append("")
    (out_dir / f"{stem}_hyb_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize hybrid integrator scores and reconstructions")
    ap.add_argument("--in_path", type=str, required=True, help="SPAD sample directory or .npy path")
    ap.add_argument("--save_dir", type=str, required=True, help="Output root directory")
    ap.add_argument("--chunk_size", type=int, default=320)
    ap.add_argument("--chunk_stride", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--hyb_fast_window", type=int, default=16)
    ap.add_argument("--hyb_slow_window", type=int, default=128)
    ap.add_argument("--hyb_temporal_window", type=int, default=5)
    ap.add_argument("--hyb_fast_tau", type=float, default=4.0)
    ap.add_argument("--hyb_sharpness", type=float, default=1.0)
    ap.add_argument("--hyb_bias", type=float, default=3.0)
    ap.add_argument("--hyb_eps", type=float, default=1e-5)
    ap.add_argument("--hyb_kernel_size", type=int, default=None, help="Deprecated alias for --hyb_slow_window")
    ap.add_argument("--hyb_prior_strength", type=float, default=1.0, help="Deprecated; ignored by STEA")
    ap.add_argument("--hyb_gating_tau", type=float, default=0.1, help="Deprecated; ignored by STEA")
    ap.add_argument("--hyb_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hyb_quantile", type=float, default=1.0)
    ap.add_argument("--hyb_max_filter_size", type=int, default=3, help="Deprecated; ignored by STEA")
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
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    cmap_id = COLORMAPS[args.colormap]

    hyb = SpatioTemporalEvidenceAccumulation(
        chunk_size=int(args.chunk_size),
        fast_window=int(args.hyb_fast_window),
        slow_window=int(args.hyb_kernel_size or args.hyb_slow_window),
        temporal_window=int(args.hyb_temporal_window),
        fast_tau=float(args.hyb_fast_tau),
        sharpness=float(args.hyb_sharpness),
        bias=float(args.hyb_bias),
        eps=float(args.hyb_eps),
        subsampling=int(args.chunk_size),
        normalize=bool(args.hyb_normalize),
        quantile=float(args.hyb_quantile),
    ).to(device)

    sample_name = in_path.name if in_path.is_dir() else in_path.stem
    stride = int(args.chunk_stride) if int(args.chunk_stride) > 0 else int(args.chunk_size)
    sources = list(_iter_sources(in_path))

    for video_idx, source in enumerate(tqdm(sources, desc=f"hyb vis [{sample_name}]")):
        n_bins = _num_bins(source)
        out_dir = _output_dir(save_root, sample_name, video_idx)
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        for cube_idx, t0 in enumerate(range(0, n_bins, stride)):
            t1 = min(t0 + int(args.chunk_size), n_bins)
            raw_chunk = _slice_raw(source, t0, t1, packed_ch_order=args.packed_ch_order)
            if raw_chunk.shape[0] == 0:
                continue

            raw = torch.from_numpy(raw_chunk[:, :, :, 0]).to(device).permute(1, 2, 0).bool()
            chunk_t = int(raw.shape[-1])
            recons, motion_debug = hyb.process_photon_cube_with_motion(raw, clear_states=cube_idx == 0)
            if int(recons.shape[-1]) == 0:
                tqdm.write(f"Skip cube {cube_idx} (no temporal blocks): t{t0:06d}_{t1:06d}")
                continue

            vis_kw = dict(
                vis_mode=args.vis_mode,
                percentile=float(args.vis_percentile),
                gamma=float(args.vis_gamma),
            )
            recon_bgr = _rgb_tensor_to_bgr_u8(
                _raw_hwt_to_rgb_float(recons, packed_nch=source.packed_nch), **vis_kw
            )[0]
            sum_bgr = _rgb_tensor_to_bgr_u8(
                _raw_hwt_to_rgb_float(raw.float().mean(dim=-1, keepdim=True), packed_nch=source.packed_nch),
                **vis_kw,
            )[0]

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
                fast_window=int(args.hyb_fast_window),
                slow_window=int(args.hyb_kernel_size or args.hyb_slow_window),
                temporal_window=int(args.hyb_temporal_window),
                fast_tau=float(args.hyb_fast_tau),
                sharpness=float(args.hyb_sharpness),
                bias=float(args.hyb_bias),
            )
            frame_idx += 1

        if frame_idx == 0:
            print(
                f"Warning: no frames saved for {sample_name}/video{video_idx:05d} "
                f"(n_bins={n_bins}; check chunk_size={args.chunk_size})"
            )

    print(f"Saved hybrid visualizations under {save_root / sample_name}")


if __name__ == "__main__":
    main()
