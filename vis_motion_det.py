#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Visualize SPADTracker dense motion fields and velocity-guided reconstructions.

Writes per-chunk outputs under ``{save_dir}/{sample}/videoXXXXX/``:

- ``{stem}_motion_compare.png`` - sum vs vel reconstruction side-by-side
- ``{stem}_motion_flow.png`` - dense velocity field rendered as a color wheel
- ``{stem}_motion_overlay.png`` - velocity arrows over the tracked vel reconstruction
- ``{stem}_motion_stats.txt`` - percentile summary and per-track motion values
"""

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
    _preprocess_vel,
    _reset_tracker,
    _resolve_device,
    _rgb_tensor_to_bgr_u8,
    _run_detector_on_frames,
    _set_vel_field_from_tracker,
    _slice_raw,
)
from ultralytics import YOLO
from ultralytics.data.spad_packed import sum_raw_chunk_to_rgb
from ultralytics.quanta_motion_networks.integrator import VelIntegrator


def _resize_map_to_display(score_hw: np.ndarray, display_hw: tuple[int, int]) -> np.ndarray:
    disp_h, disp_w = display_hw
    return cv2.resize(score_hw.astype(np.float32), (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)


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


def _percentile_summary(values: np.ndarray, *, name: str, percentiles: tuple[float, ...]) -> list[str]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return [f"[{name}] n=0 (empty)"]
    lines = [f"[{name}] n={flat.size} min={flat.min():.6f} max={flat.max():.6f} mean={flat.mean():.6f}"]
    for p in percentiles:
        lines.append(f"  p{p:g} = {float(np.percentile(flat, p)):.6f}")
    return lines


def _save_visuals(
    *,
    out_dir: Path,
    stem: str,
    sum_bgr: np.ndarray,
    recon_bgr: np.ndarray,
    tracked_bgr: np.ndarray,
    flow_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    flow_hw2: np.ndarray,
    tracker_motion: list[dict[str, float]],
    tracker_name: str,
    arrow_stride: int,
    arrow_scale: float,
    arrow_min_speed: float,
) -> None:
    cv2.imwrite(
        str(out_dir / f"{stem}_motion_compare.png"),
        _stitch_panels([sum_bgr, recon_bgr], ["sum", "vel"]),
    )
    cv2.imwrite(str(out_dir / f"{stem}_motion_flow.png"), flow_bgr)
    cv2.imwrite(
        str(out_dir / f"{stem}_motion_overlay.png"),
        _stitch_panels([tracked_bgr, overlay_bgr], ["tracked", "flow_arrows"]),
    )

    percentiles = (1.0, 5.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)
    fx = flow_hw2[..., 0]
    fy = flow_hw2[..., 1]
    speed = np.linalg.norm(flow_hw2, axis=2)
    lines = [
        f"stem={stem}",
        f"tracker={tracker_name}",
        f"arrow_stride={arrow_stride} arrow_scale={arrow_scale} arrow_min_speed={arrow_min_speed}",
        f"n_tracks={len(tracker_motion)}",
        "",
    ]
    lines.extend(_percentile_summary(fx, name="vx", percentiles=percentiles))
    lines.append("")
    lines.extend(_percentile_summary(fy, name="vy", percentiles=percentiles))
    lines.append("")
    lines.extend(_percentile_summary(speed, name="speed", percentiles=percentiles))
    lines.append("")
    lines.append("[tracks]")
    if not tracker_motion:
        lines.append("  none")
    else:
        for item in tracker_motion:
            lines.append(
                "  "
                f"id={int(item['track_id'])} "
                f"center=({item['cx']:.2f},{item['cy']:.2f}) "
                f"size=({item['w']:.2f},{item['h']:.2f}) "
                f"pred_v=({item['predicted_vx']:.3f},{item['predicted_vy']:.3f}) "
                f"meas_v=({item.get('measured_vx', 0.0):.3f},{item.get('measured_vy', 0.0):.3f}) "
                f"score={item['score']:.3f} len={int(item['tracklet_len'])}"
            )
    (out_dir / f"{stem}_motion_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize SPADTracker motion fields and vel reconstructions")
    ap.add_argument("--in_path", type=str, required=True, help="SPAD sample directory or .npy path")
    ap.add_argument("--save_dir", type=str, required=True, help="Output root directory")
    ap.add_argument("--ckpt", type=str, required=True, help="YOLO checkpoint for tracking")
    ap.add_argument("--chunk_size", type=int, default=320)
    ap.add_argument("--chunk_stride", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--tracker", type=str, default="spad_tracker", choices=["spad_tracker"])
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--vel_max_shift", type=int, default=16)
    ap.add_argument("--vel_compensate", type=str, default="rgb", choices=["rgb", "raw"])
    ap.add_argument("--vel_quantile", type=float, default=1.0)
    ap.add_argument("--vel_normalize", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vis_mode", type=str, default="linear", choices=["linear", "gamma", "percentile", "percentile_gamma"])
    ap.add_argument("--vis_percentile", type=float, default=99.5)
    ap.add_argument("--vis_gamma", type=float, default=2.2)
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
    tracker_cfg = f"{args.tracker}.yaml"
    vis_kw = dict(
        vis_mode=args.vis_mode,
        percentile=float(args.vis_percentile),
        gamma=float(args.vis_gamma),
    )

    model = YOLO(args.ckpt)
    vel = VelIntegrator(
        chunk_size=int(args.chunk_size),
        max_shift=int(args.vel_max_shift),
        patch_size=0,
        compensate_space=str(args.vel_compensate),
        normalize=bool(args.vel_normalize),
        quantile=float(args.vel_quantile),
    ).to(device)

    sample_name = in_path.name if in_path.is_dir() else in_path.stem
    stride = int(args.chunk_stride) if int(args.chunk_stride) > 0 else int(args.chunk_size)
    sources = list(_iter_sources(in_path))

    _reset_tracker(model)
    for video_idx, source in enumerate(tqdm(sources, desc=f"motion vis [{sample_name}]")):
        n_bins = _num_bins(source)
        out_dir = _output_dir(save_root, sample_name, video_idx)
        out_dir.mkdir(parents=True, exist_ok=True)

        vel.reset()
        _reset_tracker(model)

        frame_idx = 0
        for cube_idx, t0 in enumerate(range(0, n_bins, stride)):
            t1 = min(t0 + int(args.chunk_size), n_bins)
            raw_chunk = _slice_raw(source, t0, t1, packed_ch_order=args.packed_ch_order)
            if raw_chunk.shape[0] == 0:
                continue

            vel_rgb = _preprocess_vel(
                raw_chunk,
                packed_nch=source.packed_nch,
                device=device,
                integrator=vel,
                clear_states=cube_idx == 0,
            )
            sum_rgb = sum_raw_chunk_to_rgb(raw_chunk, packed_nch=source.packed_nch, device=device)
            recon_bgr = _rgb_tensor_to_bgr_u8(vel_rgb, **vis_kw)[0]
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
            _set_vel_field_from_tracker(model, vel)
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
                sum_bgr=sum_bgr,
                recon_bgr=recon_bgr,
                tracked_bgr=tracked_bgr,
                flow_bgr=flow_bgr,
                overlay_bgr=overlay_bgr,
                flow_hw2=flow_hw2,
                tracker_motion=tracker_motion,
                tracker_name=args.tracker,
                arrow_stride=int(args.arrow_stride),
                arrow_scale=float(args.arrow_scale),
                arrow_min_speed=float(args.arrow_min_speed),
            )
            frame_idx += 1

        if frame_idx == 0:
            print(
                f"Warning: no frames saved for {sample_name}/video{video_idx:05d} "
                f"(n_bins={n_bins}; check chunk_size={args.chunk_size})"
            )

    print(f"Saved motion visualizations under {save_root / sample_name}")


if __name__ == "__main__":
    main()
