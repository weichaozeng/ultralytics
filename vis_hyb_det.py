#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Visualize hybrid STEA+velocity routing, reconstructions, and tracker feedback."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from det_spad import (
    _draw_results,
    _iter_sources,
    _num_bins,
    _output_dir,
    _reset_tracker,
    _resolve_device,
    _rgb_tensor_to_bgr_u8,
    _run_detector_on_frames,
    _set_tracker_field_on_integrator,
    _slice_raw,
)
from ultralytics import YOLO
from ultralytics.data.spad_packed import raw_hwt_to_rgb_float, raw_plane_to_photon_cube, sum_raw_chunk_to_rgb
from ultralytics.quanta_hyb_networks.integrator import HybridSpatioTemporalEvidenceAccumulation
from ultralytics.quanta_neural_networks.ops.image import nearest_neighbor_inpaint

COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "hot": cv2.COLORMAP_HOT,
}


def _resize_map_to_display(score_hw: np.ndarray, display_hw: tuple[int, int]) -> np.ndarray:
    disp_h, disp_w = display_hw
    return cv2.resize(score_hw.astype(np.float32), (disp_w, disp_h), interpolation=cv2.INTER_AREA)


def _value_to_heatmap(values: np.ndarray, cmap_id: int, *, vmax: float) -> np.ndarray:
    vmax = max(float(vmax), 1e-8)
    u8 = (np.clip(values / vmax, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return cv2.applyColorMap(u8, cmap_id)


def _value_to_gray_bgr(values: np.ndarray, *, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
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


def _flow_to_bgr(flow_hw2: np.ndarray, *, max_speed: float) -> np.ndarray:
    flow = np.asarray(flow_hw2, dtype=np.float32)
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(ang / 2.0, 180.0).astype(np.uint8)
    hsv[..., 1] = 255
    vmax = max(float(max_speed), 1e-6)
    hsv[..., 2] = np.clip((mag / vmax) * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _overlay_flow_arrows(
    base_bgr: np.ndarray,
    flow_hw2: np.ndarray,
    *,
    stride: int,
    arrow_scale: float,
    min_speed: float,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    out = np.ascontiguousarray(base_bgr.copy())
    h, w = out.shape[:2]
    flow = np.asarray(flow_hw2, dtype=np.float32)
    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            vx, vy = float(flow[y, x, 0]), float(flow[y, x, 1])
            speed = float(np.hypot(vx, vy))
            if speed < min_speed:
                continue
            p0 = (int(x), int(y))
            p1 = (int(round(x + arrow_scale * vx)), int(round(y + arrow_scale * vy)))
            cv2.arrowedLine(out, p0, p1, color, 1, cv2.LINE_AA, tipLength=0.25)
    return out


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


def _process_chunk_with_full_debug(
    hyb: HybridSpatioTemporalEvidenceAccumulation,
    raw: torch.Tensor,
    *,
    clear_states: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if clear_states:
        hyb.reset()
    hyb.set_cube(raw)
    fused, motion_debug = hyb._integrate_last_with_debug(raw)
    recons = hyb._subsample_reconstruction(fused)
    motion_debug["recons_prenorm"] = hyb._subsample_reconstruction(motion_debug["recons_prenorm"])
    if hyb.hot_pixel_mask is not None:
        recons = nearest_neighbor_inpaint(recons, hyb.hot_pixel_mask)
    recons = hyb.clamp_recons(recons)
    hyb.t_absolute += int(raw.shape[-1])
    return recons, motion_debug


def _save_visuals(
    *,
    out_dir: Path,
    stem: str,
    recon_bgr: np.ndarray,
    sum_bgr: np.ndarray,
    tracked_bgr: np.ndarray,
    flow_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    motion_debug: dict[str, torch.Tensor],
    flow_hw2: np.ndarray,
    tracker_motion: list[dict[str, float]],
    cmap_id: int,
    score_vmax: float,
    score_percentile: float,
    max_speed: float,
    motion_sharpness: float,
    motion_threshold: float,
) -> None:
    percentiles = (1.0, 5.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)
    display_hw = recon_bgr.shape[:2]

    route_weight_hwt = motion_debug["route_weight"].detach().float().cpu().numpy()
    k_smoothed_hwt = motion_debug["k_smoothed"].detach().float().cpu().numpy()
    maps = {
        "k_smoothed": motion_debug["k_smoothed_last"].detach().float().cpu().numpy(),
        "k_smoothed_peak": k_smoothed_hwt.max(axis=-1),
        "route_weight_raw": motion_debug["route_weight_raw_last"].detach().float().cpu().numpy(),
        "future_motion": motion_debug["future_motion_last"].detach().float().cpu().numpy(),
        "route_weight_peak": route_weight_hwt.max(axis=-1),
        "valid_weight_sum": motion_debug["stable_support"].detach().float().cpu().numpy() / max(
            float(motion_debug["valid_weight"].shape[-1]), 1.0
        ),
        "y_fast": motion_debug["y_scales_last"][..., 0].detach().float().cpu().numpy(),
        "y_slow": motion_debug["y_scales_last"][..., 1].detach().float().cpu().numpy(),
        "mean_stable": motion_debug["mean_stable"].detach().float().cpu().numpy(),
        "w_mean": motion_debug["w_mean"].detach().float().cpu().numpy(),
        "fused": motion_debug["fused_last"].detach().float().cpu().numpy(),
    }
    k_vmax_scale = _resolve_vmax(maps["k_smoothed"], fixed_vmax=score_vmax, percentile=score_percentile)

    score_panels = []
    score_labels = []
    for label, score_map in maps.items():
        disp = _resize_map_to_display(score_map, display_hw)
        if label in {"k_smoothed", "k_smoothed_peak"}:
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
    cv2.imwrite(
        str(out_dir / f"{stem}_hyb_overlay.png"),
        _stitch_panels([tracked_bgr, overlay_bgr, flow_bgr], ["tracked", "flow_arrows", "flow"]),
    )

    speed = np.linalg.norm(flow_hw2, axis=2)
    route_from_k = 1.0 / (
        1.0 + np.exp(-float(motion_sharpness) * (maps["k_smoothed"] - float(motion_threshold)))
    )
    route_abs_err = np.abs(route_from_k - maps["route_weight_raw"])
    stats_lines = [
        f"stem={stem}",
        f"k_smoothed_vmax={k_vmax_scale:.6f} (fixed={score_vmax:g}, percentile={score_percentile:g})",
        f"flow_vmax={max_speed:.6f}",
        "k_smoothed is hybrid causal-smoothed Bernoulli KL evidence.",
        "route_weight visualization is fixed grayscale [0, 1] so brightness is monotonic.",
        "y_slow is computed from velocity-compensated slow-branch photon history.",
        f"route_raw_from_k_abs_err_max={float(route_abs_err.max()):.8f} mean={float(route_abs_err.mean()):.8f}",
        "",
    ]
    for label, values in maps.items():
        stats_lines.extend(_percentile_summary(values, name=label, percentiles=percentiles))
        stats_lines.append("")
    stats_lines.extend(_percentile_summary(flow_hw2[..., 0], name="flow_vx", percentiles=percentiles))
    stats_lines.append("")
    stats_lines.extend(_percentile_summary(flow_hw2[..., 1], name="flow_vy", percentiles=percentiles))
    stats_lines.append("")
    stats_lines.extend(_percentile_summary(speed, name="flow_speed", percentiles=percentiles))
    stats_lines.append("")
    stats_lines.append("[tracks]")
    if not tracker_motion:
        stats_lines.append("  none")
    else:
        for item in tracker_motion:
            stats_lines.append(
                "  "
                f"id={int(item['track_id'])} "
                f"center=({item['cx']:.2f},{item['cy']:.2f}) "
                f"size=({item['w']:.2f},{item['h']:.2f}) "
                f"pred_v=({item['predicted_vx']:.3f},{item['predicted_vy']:.3f}) "
                f"meas_v=({item.get('measured_vx', 0.0):.3f},{item.get('measured_vy', 0.0):.3f}) "
                f"score={item['score']:.3f} len={int(item['tracklet_len'])}"
            )
    (out_dir / f"{stem}_hyb_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize hybrid STEA+velocity scores, reconstructions, and flow feedback")
    ap.add_argument("--in_path", type=str, required=True, help="SPAD sample directory or .npy path")
    ap.add_argument("--save_dir", type=str, required=True, help="Output root directory")
    ap.add_argument("--ckpt", type=str, required=True, help="YOLO checkpoint for tracking")
    ap.add_argument("--chunk_size", type=int, default=320)
    ap.add_argument("--chunk_stride", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--tracker", type=str, default="spad_tracker", choices=["spad_tracker"])
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--hyb_fast_window", type=int, default=64)
    ap.add_argument("--hyb_slow_window", type=int, default=128)
    ap.add_argument("--hyb_temporal_window", type=int, default=5)
    ap.add_argument("--hyb_fast_tau", type=float, default=6.0)
    ap.add_argument("--hyb_motion_sharpness", type=float, default=60.0)
    ap.add_argument("--hyb_motion_threshold", type=float, default=0.05)
    ap.add_argument("--hyb_eps", type=float, default=1e-5)
    ap.add_argument("--hyb_blend_const", type=float, default=16.0)
    ap.add_argument("--hyb_kernel_size", type=int, default=None, help="Deprecated alias for --hyb_slow_window")
    ap.add_argument("--hyb_prior_strength", type=float, default=1.0, help="Deprecated; ignored by hybrid STEA")
    ap.add_argument("--hyb_gating_tau", type=float, default=0.1, help="Deprecated; ignored by hybrid STEA")
    ap.add_argument("--hyb_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hyb_quantile", type=float, default=1.0)
    ap.add_argument("--hyb_max_filter_size", type=int, default=3, help="Deprecated; ignored by hybrid STEA")
    ap.add_argument("--hyb_warp_block_size", type=int, default=16)
    ap.add_argument("--vis_mode", type=str, default="linear", choices=["linear", "gamma", "percentile", "percentile_gamma"])
    ap.add_argument("--vis_percentile", type=float, default=99.5)
    ap.add_argument("--vis_gamma", type=float, default=2.2)
    ap.add_argument("--colormap", type=str, default="turbo", choices=sorted(COLORMAPS))
    ap.add_argument("--score_vmax", type=float, default=0.0, help="Fixed max for score heatmaps (0 = use --score_percentile)")
    ap.add_argument("--score_percentile", type=float, default=99.5, help="Percentile vmax for score heatmaps when --score_vmax=0")
    ap.add_argument("--flow_max_speed", type=float, default=0.0, help="Fixed max speed for flow color value (0 = auto p99)")
    ap.add_argument("--arrow_stride", type=int, default=16)
    ap.add_argument("--arrow_scale", type=float, default=3.0)
    ap.add_argument("--arrow_min_speed", type=float, default=0.25)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    cmap_id = COLORMAPS[args.colormap]
    tracker_cfg = f"{args.tracker}.yaml"
    vis_kw = dict(vis_mode=args.vis_mode, percentile=float(args.vis_percentile), gamma=float(args.vis_gamma))

    model = YOLO(args.ckpt)
    hyb = HybridSpatioTemporalEvidenceAccumulation(
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

    sample_name = in_path.name if in_path.is_dir() else in_path.stem
    stride = int(args.chunk_stride) if int(args.chunk_stride) > 0 else int(args.chunk_size)
    sources = list(_iter_sources(in_path))

    _reset_tracker(model)
    for video_idx, source in enumerate(tqdm(sources, desc=f"hyb vis [{sample_name}]")):
        n_bins = _num_bins(source)
        out_dir = _output_dir(save_root, sample_name, video_idx)
        out_dir.mkdir(parents=True, exist_ok=True)

        hyb.reset()
        _reset_tracker(model)

        frame_idx = 0
        for cube_idx, t0 in enumerate(range(0, n_bins, stride)):
            t1 = min(t0 + int(args.chunk_size), n_bins)
            raw_chunk = _slice_raw(source, t0, t1, packed_ch_order=args.packed_ch_order)
            if raw_chunk.shape[0] == 0:
                continue

            raw = raw_plane_to_photon_cube(raw_chunk[..., 0], device=device, as_bool=True)
            recons, motion_debug = _process_chunk_with_full_debug(hyb, raw, clear_states=cube_idx == 0)
            if int(recons.shape[-1]) == 0:
                tqdm.write(f"Skip cube {cube_idx} (no temporal blocks): t{t0:06d}_{t1:06d}")
                continue

            recon_rgb = raw_hwt_to_rgb_float(recons.float(), packed_nch=source.packed_nch)
            sum_rgb = sum_raw_chunk_to_rgb(raw_chunk, packed_nch=source.packed_nch, device=device)
            recon_bgr = _rgb_tensor_to_bgr_u8(recon_rgb, **vis_kw)[0]
            sum_bgr = _rgb_tensor_to_bgr_u8(sum_rgb, **vis_kw)[0]

            results = _run_detector_on_frames(
                model,
                [recon_bgr],
                conf=float(args.det_thresh),
                tracker_cfg=tracker_cfg,
                device=args.device,
            )
            result = results[0]
            tracked_bgr = _draw_results(recon_bgr, result)
            _set_tracker_field_on_integrator(model, hyb)
            tracker = getattr(model.predictor, "trackers", [None])[0]

            flow_t = getattr(tracker, "last_velocity_field", None) if tracker is not None else None
            if flow_t is None:
                flow_hw2 = np.zeros((recon_bgr.shape[0], recon_bgr.shape[1], 2), dtype=np.float32)
                tracker_motion = []
            else:
                flow_hw2 = flow_t.detach().cpu().numpy().astype(np.float32)
                tracker_motion = list(getattr(tracker, "last_track_motion", []))

            speed = np.linalg.norm(flow_hw2, axis=2)
            if float(args.flow_max_speed) > 0.0:
                max_speed = float(args.flow_max_speed)
            elif speed.size == 0:
                max_speed = 1.0
            else:
                max_speed = max(float(np.percentile(speed.reshape(-1), 99.0)), 1e-6)
            flow_bgr = _flow_to_bgr(flow_hw2, max_speed=max_speed)
            overlay_bgr = _overlay_flow_arrows(
                tracked_bgr,
                flow_hw2,
                stride=int(args.arrow_stride),
                arrow_scale=float(args.arrow_scale),
                min_speed=float(args.arrow_min_speed),
            )

            stem = f"cube{cube_idx:05d}_t{t0:06d}_{t1:06d}_frame{frame_idx:07d}"
            _save_visuals(
                out_dir=out_dir,
                stem=stem,
                recon_bgr=recon_bgr,
                sum_bgr=sum_bgr,
                tracked_bgr=tracked_bgr,
                flow_bgr=flow_bgr,
                overlay_bgr=overlay_bgr,
                motion_debug=motion_debug,
                flow_hw2=flow_hw2,
                tracker_motion=tracker_motion,
                cmap_id=cmap_id,
                score_vmax=float(args.score_vmax),
                score_percentile=float(args.score_percentile),
                max_speed=max_speed,
                motion_sharpness=float(args.hyb_motion_sharpness),
                motion_threshold=float(args.hyb_motion_threshold),
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
