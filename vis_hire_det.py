# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Visualize HIRE (I^f / I^s + hard reset) intermediates.

Writes per-chunk outputs under ``{save_dir}/{sample}/videoXXXXX/``:

- ``{stem}_hire_compare.png`` — sum / hire / ages / reset footprint
- ``{stem}_hire_scores.png`` — last-bin maps (n_slow, w_slow, S, …)
- ``{stem}_hire_temporal.png`` — time slices of key volumes
- ``{stem}_hire_stats.txt`` — optional percentile summary (``--write_stats``)

Example
-------
python ultralytics/vis_hire_det.py \\
  --in_path /path/to/frames.npy \\
  --save_dir /tmp/hire_vis \\
  --chunk_size 80 --bin_rate_hz 2000
  # defaults: W_f/s/S=24/160/4, hold=80, θ_on/off=0.08/0.02,
  # confirm=4, gate_pool=max, reset_open/grow=15/6, vis_mode=gamma
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
from ultralytics.quanta_hire_networks.integrator import HIRE

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


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _save_visuals(
    *,
    out_dir: Path,
    stem: str,
    recon_bgr: np.ndarray,
    sum_bgr: np.ndarray,
    debug: dict[str, torch.Tensor],
    hire: HIRE,
    cmap_id: int,
    score_vmax: float,
    score_percentile: float,
    temporal_slices: int,
    write_stats: bool,
) -> None:
    percentiles = (1.0, 5.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)
    display_hw = recon_bgr.shape[:2]

    s_tilde_hwt = _np(debug["s_tilde_hwt"])
    w_slow_hwt = _np(debug["w_slow_hwt"])
    s_raw_hwt = _np(debug["s_raw_hwt"])
    i_fast_hwt = _np(debug["i_fast_hwt"])
    i_slow_hwt = _np(debug["i_slow_hwt"])
    i_out_hwt = _np(debug["i_out_hwt"])
    n_slow_hwt = _np(debug["n_slow_hwt"])
    cooldown_hwt = _np(debug["cooldown_hwt"])
    did_reset_hwt = _np(debug["did_reset_hwt"])

    n_last = _np(debug["n_slow_last"])
    n_min = _np(debug["n_slow_min"])
    reset_any = _np(debug["reset_any"])
    w_f = float(hire.fast_bins)

    maps: dict[str, np.ndarray] = {
        # reconstructions
        "i_out": _np(debug["i_out_last"]),
        "i_fast": _np(debug["i_fast_last"]),
        "i_slow": _np(debug["i_slow_last"]),
        # evidence (absolute scale shared)
        "s_tilde": _np(debug["s_tilde_last"]),
        "s_raw": _np(debug["s_raw_last"]),
        # age / mix (most informative post-reset signals)
        "n_slow": n_last,
        "n_slow_min": n_min,
        "w_slow": _np(debug["w_slow_last"]),
        "g_fast": _np(debug["g_fast_last"]),
        "t_mix": _np(debug["t_mix_last"]),
        # reset footprint over this chunk (not last-bin latch)
        "reset_any": reset_any,
        "cooldown": _np(debug["cooldown_last"]),
        "young": (n_last <= w_f + 0.5).astype(np.float32),
        "holding": (_np(debug["t_mix_last"]) < float(hire.effective_mix_hold_bins())).astype(np.float32),
    }

    s_vmax = _resolve_vmax(s_tilde_hwt, fixed_vmax=score_vmax, percentile=score_percentile)
    # Fixed counter scales: n saturates at W_s, t_mix at hold H (values above clip hot).
    n_vmax = max(float(hire.slow_bins), 1.0)
    t_mix_vmax = max(float(hire.effective_mix_hold_bins()), 1.0)
    cd_vmax = max(float(hire.cooldown_bins), 1.0)

    score_panels = []
    score_labels = []
    for label, score_map in maps.items():
        if label.startswith("s_"):
            panel = _panel_from_map(
                score_map, display_hw=display_hw, cmap_id=cmap_id, mode="heatmap", vmax=s_vmax
            )
        elif label in {"n_slow", "n_slow_min", "t_mix"}:
            vmax = n_vmax if label.startswith("n_") else t_mix_vmax
            panel = _panel_from_map(
                score_map, display_hw=display_hw, cmap_id=cmap_id, mode="heatmap", vmax=vmax
            )
        elif label == "cooldown":
            panel = _panel_from_map(
                score_map, display_hw=display_hw, cmap_id=cmap_id, mode="heatmap", vmax=cd_vmax
            )
        else:
            panel = _panel_from_map(
                score_map, display_hw=display_hw, cmap_id=cmap_id, mode="gray", vmax=1.0
            )
        score_panels.append(panel)
        score_labels.append(label)
    cv2.imwrite(str(out_dir / f"{stem}_hire_scores.png"), _stitch_panels(score_panels, score_labels))

    temporal_rows = [
        _temporal_strip(
            s_tilde_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="s_tilde",
            mode="heatmap",
            vmax=s_vmax,
            max_slices=temporal_slices,
        ),
        _temporal_strip(
            n_slow_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="n_slow",
            mode="heatmap",
            vmax=n_vmax,
            max_slices=temporal_slices,
        ),
        _temporal_strip(
            did_reset_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="did_reset",
            mode="gray",
            vmax=1.0,
            max_slices=temporal_slices,
        ),
        _temporal_strip(
            cooldown_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="cooldown",
            mode="heatmap",
            vmax=cd_vmax,
            max_slices=temporal_slices,
        ),
        _temporal_strip(
            w_slow_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="w_slow",
            mode="gray",
            vmax=1.0,
            max_slices=temporal_slices,
        ),
        _temporal_strip(
            _np(debug["g_fast_hwt"]),
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="g_fast",
            mode="gray",
            vmax=1.0,
            max_slices=temporal_slices,
        ),
        _temporal_strip(
            _np(debug["t_mix_hwt"]),
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="t_mix",
            mode="heatmap",
            vmax=t_mix_vmax,
            max_slices=temporal_slices,
        ),
        _temporal_strip(
            i_out_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="i_out",
            mode="gray",
            vmax=1.0,
            max_slices=temporal_slices,
        ),
        _temporal_strip(
            i_slow_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="i_slow",
            mode="gray",
            vmax=1.0,
            max_slices=temporal_slices,
        ),
        _temporal_strip(
            i_fast_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="i_fast",
            mode="gray",
            vmax=1.0,
            max_slices=temporal_slices,
        ),
        _temporal_strip(
            s_raw_hwt,
            display_hw=display_hw,
            cmap_id=cmap_id,
            label_prefix="s_raw",
            mode="heatmap",
            vmax=s_vmax,
            max_slices=temporal_slices,
        ),
    ]
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
        cv2.imwrite(str(out_dir / f"{stem}_hire_temporal.png"), temporal_canvas)

    cv2.imwrite(
        str(out_dir / f"{stem}_hire_compare.png"),
        _stitch_panels(
            [
                sum_bgr,
                recon_bgr,
                _panel_from_map(maps["i_fast"], display_hw=display_hw, cmap_id=cmap_id, mode="gray", vmax=1.0),
                _panel_from_map(maps["i_slow"], display_hw=display_hw, cmap_id=cmap_id, mode="gray", vmax=1.0),
                _panel_from_map(maps["n_slow"], display_hw=display_hw, cmap_id=cmap_id, mode="heatmap", vmax=n_vmax),
                _panel_from_map(maps["g_fast"], display_hw=display_hw, cmap_id=cmap_id, mode="gray", vmax=1.0),
                _panel_from_map(maps["holding"], display_hw=display_hw, cmap_id=cmap_id, mode="gray", vmax=1.0),
                _panel_from_map(maps["reset_any"], display_hw=display_hw, cmap_id=cmap_id, mode="gray", vmax=1.0),
                _panel_from_map(maps["young"], display_hw=display_hw, cmap_id=cmap_id, mode="gray", vmax=1.0),
            ],
            ["sum", "hire", "i_fast", "i_slow", "n_slow", "g_fast", "holding", "reset_any", "young"],
        ),
    )

    if not write_stats:
        return

    hold_h = float(hire.effective_mix_hold_bins())
    tau_m = float(hire.mix_bins)
    t_m = maps["t_mix"]
    # g_reset from hold+exp; g_fast = max(g_reset, g_soft), so the gap is the soft-gate lift.
    g_reset = np.where(t_m < hold_h, 1.0, np.exp(-np.maximum(t_m - hold_h, 0.0) / tau_m))
    g_soft_lift = np.clip(maps["g_fast"].astype(np.float64) - g_reset.astype(np.float64), 0.0, None)

    stats_lines = [
        f"stem={stem}",
        (
            f"fs={hire.sample_rate_hz:g} "
            f"bins={hire.fast_bins}/{hire.slow_bins}/{hire.surprise_bins} "
            f"mix_hold={hire.effective_mix_hold_bins()} mix_τ={hire.mix_bins:g} "
            f"mix_θ/floor={hire.mix_theta:g}/{hire.effective_mix_floor():g} "
            f"theta_on/off/grow={hire.theta_on:g}/{hire.theta_off:g}/{hire.effective_theta_grow():g} "
            f"confirm={hire.confirm_bins} cooldown={hire.cooldown_bins} "
            f"spatial_kernel={hire.spatial_kernel} gate_pool={hire.gate_pool} "
            f"reset_open/grow={hire.reset_open}/{hire.reset_grow} "
            f"normalize={hire.normalize} quantile={hire.quantile:g}"
        ),
        (
            f"alpha_fast={hire.alpha_fast:.6f} alpha_slow={hire.alpha_slow:.6f} "
            f"alpha_surprise={hire.alpha_surprise:.6f}"
        ),
        f"s_tilde_vmax={s_vmax:.6f} (fixed={score_vmax:g}, percentile={score_percentile:g})",
        "Pipeline: I^f/I^s → BernKL → S EMA → pool → geodesic reset → g=max(hold+exp, deadzone-soft) → I_out",
        f"g_soft_lift_over_reset_max={float(g_soft_lift.max()):.6f} mean={float(g_soft_lift.mean()):.6f}",
        (
            f"reset_any_frac={float(reset_any.mean()):.6f} holding_frac={float(maps['holding'].mean()):.6f} "
            f"n_slow_mean={float(n_last.mean()):.3f} g_fast_mean={float(maps['g_fast'].mean()):.4f}"
        ),
        "",
    ]
    for label, values in maps.items():
        stats_lines.extend(_percentile_summary(values, name=label, percentiles=percentiles))
        stats_lines.append("")
    for label, values in {
        "s_raw_hwt": s_raw_hwt,
        "s_tilde_hwt": s_tilde_hwt,
        "w_slow_hwt": w_slow_hwt,
        "n_slow_hwt": n_slow_hwt,
        "did_reset_hwt": did_reset_hwt,
        "cooldown_hwt": cooldown_hwt,
        "i_fast_hwt": i_fast_hwt,
        "i_slow_hwt": i_slow_hwt,
        "i_out_hwt": i_out_hwt,
    }.items():
        stats_lines.extend(_percentile_summary(values, name=label, percentiles=percentiles))
        stats_lines.append("")
    (out_dir / f"{stem}_hire_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize HIRE v0.3 I^f/I^s change-point intermediates")
    ap.add_argument("--in_path", type=str, required=True, help="SPAD sample directory or .npy path")
    ap.add_argument("--save_dir", type=str, required=True, help="Output root directory")
    ap.add_argument("--chunk_size", type=int, default=80)
    ap.add_argument("--chunk_stride", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    # HIRE (2 kHz / chunk=80; α=exp(-1/W) from *_bins)
    ap.add_argument("--bin_rate_hz", type=float, default=2000.0, help="SPAD bin rate f_s (logging / tau override)")
    ap.add_argument("--hire_fast_bins", type=int, default=24, help="W_f: α_f=exp(-1/W_f)")
    ap.add_argument("--hire_slow_bins", type=int, default=160, help="W_s: α_s=exp(-1/W_s), n_s cap")
    ap.add_argument("--hire_surprise_bins", type=int, default=4, help="W_S: α_S=exp(-1/W_S)")
    ap.add_argument(
        "--hire_mix_hold_bins",
        type=int,
        default=80,
        help="Hold full I^f for H bins after reset; <0 => chunk_size (legacy); 0 => no hold",
    )
    ap.add_argument(
        "--hire_mix_bins",
        type=float,
        default=12.0,
        help="τ after hold: g=exp(-(t-H)/τ) toward I^s",
    )
    ap.add_argument(
        "--hire_mix_theta",
        type=float,
        default=0.06,
        help="Soft gate on relu(S̄-θ_floor); <=0 disables",
    )
    ap.add_argument(
        "--hire_mix_floor",
        type=float,
        default=-1.0,
        help="Soft deadzone; <0 => use theta_off (suppress background I^f bleed)",
    )
    ap.add_argument("--hire_theta_on", type=float, default=0.08)
    ap.add_argument("--hire_theta_off", type=float, default=0.02)
    ap.add_argument(
        "--hire_theta_grow",
        type=float,
        default=-1.0,
        help="Geodesic support S̄>θ_grow; <0 => use theta_off",
    )
    ap.add_argument("--hire_confirm_bins", type=int, default=4)
    ap.add_argument(
        "--hire_cooldown_bins",
        type=int,
        default=0,
        help="Unused (kept for CLI compat); n_f/n_s←W_f after reset replaces cooldown",
    )
    ap.add_argument("--hire_spatial_kernel", type=int, default=5)
    ap.add_argument(
        "--hire_gate_pool",
        type=str,
        default="max",
        choices=["max", "avg"],
        help="Spatial pool on S: avg=denoise spikes, max=connect blobs",
    )
    ap.add_argument(
        "--hire_reset_open",
        type=int,
        default=15,
        help="Odd morph open on seeds (1=off). >1 kills thin edge seeds",
    )
    ap.add_argument(
        "--hire_reset_grow",
        type=int,
        default=6,
        help="Geodesic dilate steps of confirmed seeds inside {S̄>θ_grow}",
    )
    ap.add_argument(
        "--hire_reset_dilate",
        type=int,
        default=-1,
        help="Legacy: odd blind-dilate kernel; >0 maps to reset_grow≈k//2",
    )
    ap.add_argument("--hire_eps", type=float, default=1e-5)
    ap.add_argument("--hire_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hire_quantile", type=float, default=1.0)
    # Vis
    ap.add_argument("--vis_mode", type=str, default="gamma", choices=["linear", "gamma", "percentile", "percentile_gamma"])
    ap.add_argument("--vis_percentile", type=float, default=99.5)
    ap.add_argument("--vis_gamma", type=float, default=2.2)
    ap.add_argument("--colormap", type=str, default="turbo", choices=sorted(COLORMAPS))
    ap.add_argument("--score_vmax", type=float, default=0.0, help="Fixed vmax for surprise heatmaps (0=percentile)")
    ap.add_argument("--score_percentile", type=float, default=99.5)
    ap.add_argument("--temporal_slices", type=int, default=6)
    ap.add_argument(
        "--write_stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write *_hire_stats.txt with full percentile summaries (slow on large maps). Default: off.",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    cmap_id = COLORMAPS[args.colormap]
