# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Visualize GatedMultiScaleEMA motion maps alongside hybrid reconstructions.

Writes per-chunk PNGs under ``{save_dir}/{sample}/videoXXXXX/``:

- ``{stem}_hyb_recon.png`` — reconstruction (same tonemap as ``det_spad``)
- ``{stem}_hyb_motion_peak.png`` — ``motion_peak`` heatmap (post peak min-pool)
- ``{stem}_hyb_motion_blend.png`` — chunk blend weight toward last block
- ``{stem}_hyb_motion_overlay.png`` — recon + ``motion_peak`` overlay
- ``{stem}_hyb_motion_blocks.png`` — per-block ``motion_score`` strip (if B > 1)
- ``{stem}_hyb_motion_mosaic.png`` — recon | peak | blend | overlay

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
from ultralytics.quanta_hybrid_networks.integrator import GatedMultiScaleEMA

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


def _score_to_heatmap(score01: np.ndarray, cmap_id: int) -> np.ndarray:
    u8 = (np.clip(score01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return cv2.applyColorMap(u8, cmap_id)


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


def _overlay_heatmap(base_bgr: np.ndarray, heat_bgr: np.ndarray, alpha: float) -> np.ndarray:
    if base_bgr.shape[:2] != heat_bgr.shape[:2]:
        heat_bgr = cv2.resize(heat_bgr, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heat_bgr, alpha, 0.0)


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


def _motion_block_strip(
    motion_blocks: np.ndarray,
    display_hw: tuple[int, int],
    cmap_id: int,
) -> np.ndarray | None:
    """``motion_blocks`` shape ``[H, W, B]`` → labelled horizontal strip."""
    if motion_blocks.ndim != 3 or motion_blocks.shape[2] <= 1:
        return None
    panels = []
    labels = []
    for b in range(motion_blocks.shape[2]):
        disp = _resize_map_to_display(motion_blocks[:, :, b], display_hw)
        panels.append(_score_to_heatmap(disp, cmap_id))
        labels.append(f"block{b}")
    return _stitch_panels(panels, labels)


def _save_motion_visuals(
    *,
    out_dir: Path,
    stem: str,
    recon_bgr: np.ndarray,
    motion_debug: dict[str, torch.Tensor],
    cmap_id: int,
    overlay_alpha: float,
    save_blocks: bool,
) -> None:
    display_hw = recon_bgr.shape[:2]

    def _tensor_hw(t: torch.Tensor) -> np.ndarray:
        return t.detach().float().cpu().numpy()

    motion_peak = _tensor_hw(motion_debug["motion_peak"])
    motion_blend = _tensor_hw(motion_debug["motion_blend"])
    motion_blocks = _tensor_hw(motion_debug["motion_blocks"])

    peak_disp = _resize_map_to_display(motion_peak, display_hw)
    blend_disp = _resize_map_to_display(motion_blend, display_hw)
    peak_heat = _score_to_heatmap(peak_disp, cmap_id)
    blend_heat = _score_to_heatmap(blend_disp, cmap_id)
    overlay = _overlay_heatmap(recon_bgr, peak_heat, overlay_alpha)

    cv2.imwrite(str(out_dir / f"{stem}_hyb_recon.png"), recon_bgr)
    cv2.imwrite(str(out_dir / f"{stem}_hyb_motion_peak.png"), peak_heat)
    cv2.imwrite(str(out_dir / f"{stem}_hyb_motion_blend.png"), blend_heat)
    cv2.imwrite(str(out_dir / f"{stem}_hyb_motion_overlay.png"), overlay)

    mosaic = _stitch_panels(
        [recon_bgr, peak_heat, blend_heat, overlay],
        ["recon", "motion_peak", "motion_blend", "overlay"],
    )
    cv2.imwrite(str(out_dir / f"{stem}_hyb_motion_mosaic.png"), mosaic)

    if save_blocks:
        block_strip = _motion_block_strip(motion_blocks, display_hw, cmap_id)
        if block_strip is not None:
            cv2.imwrite(str(out_dir / f"{stem}_hyb_motion_blocks.png"), block_strip)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize hybrid integrator motion_score maps")
    ap.add_argument("--in_path", type=str, required=True, help="SPAD sample directory or .npy path")
    ap.add_argument("--save_dir", type=str, required=True, help="Output root directory")
    ap.add_argument("--chunk_size", type=int, default=320)
    ap.add_argument("--chunk_stride", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--hyb_kernel_size", type=int, default=64)
    ap.add_argument("--hyb_v_threshold", type=float, default=0.1)
    ap.add_argument("--hyb_gating_sharpness", type=float, default=20.0)
    ap.add_argument("--hyb_gating_tau", type=float, default=0.1)
    ap.add_argument("--hyb_normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hyb_quantile", type=float, default=1.0)
    ap.add_argument("--hyb_min_filter_size", type=int, default=7)
    ap.add_argument("--hyb_peak_min_filter_size", type=int, default=7)
    ap.add_argument("--vis_mode", type=str, default="linear", choices=["linear", "gamma", "percentile", "percentile_gamma"])
    ap.add_argument("--vis_percentile", type=float, default=99.5)
    ap.add_argument("--vis_gamma", type=float, default=2.2)
    ap.add_argument("--colormap", type=str, default="turbo", choices=sorted(COLORMAPS))
    ap.add_argument("--overlay_alpha", type=float, default=0.45, help="Heatmap alpha on recon overlay")
    ap.add_argument("--no_blocks", action="store_true", help="Skip per-block motion strip PNG")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    cmap_id = COLORMAPS[args.colormap]

    hyb = GatedMultiScaleEMA(
        chunk_size=int(args.chunk_size),
        kernel_size=int(args.hyb_kernel_size),
        subsampling=int(args.chunk_size),
        v_threshold=float(args.hyb_v_threshold),
        gating_sharpness=float(args.hyb_gating_sharpness),
        gating_tau=float(args.hyb_gating_tau),
        normalize=bool(args.hyb_normalize),
        quantile=float(args.hyb_quantile),
        min_filter_size=int(args.hyb_min_filter_size),
        peak_min_filter_size=int(args.hyb_peak_min_filter_size),
    ).to(device)

    sample_name = in_path.name if in_path.is_dir() else in_path.stem
    stride = int(args.chunk_stride) if int(args.chunk_stride) > 0 else int(args.chunk_size)
    sources = list(_iter_sources(in_path))

    for video_idx, source in enumerate(tqdm(sources, desc=f"hyb motion vis [{sample_name}]")):
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
            recons, motion_debug = hyb.process_photon_cube_with_motion(raw, clear_states=cube_idx == 0)
            frames = _raw_hwt_to_rgb_float(recons, packed_nch=source.packed_nch)
            frames_bgr = _rgb_tensor_to_bgr_u8(
                frames,
                vis_mode=args.vis_mode,
                percentile=float(args.vis_percentile),
                gamma=float(args.vis_gamma),
            )
            if not frames_bgr:
                continue

            stem = f"cube{cube_idx:05d}_t{t0:06d}_{t1:06d}_frame{frame_idx:07d}"
            _save_motion_visuals(
                out_dir=out_dir,
                stem=stem,
                recon_bgr=frames_bgr[0],
                motion_debug=motion_debug,
                cmap_id=cmap_id,
                overlay_alpha=float(args.overlay_alpha),
                save_blocks=not args.no_blocks,
            )
            frame_idx += 1

    print(f"Saved hybrid motion visualizations under {save_root / sample_name}")


if __name__ == "__main__":
    main()
