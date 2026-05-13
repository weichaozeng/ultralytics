# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""det_qnns.py

Run pose tracking on SPAD photon-cube sequences using a custom predictor (`SPADPosePredictor`).

This script intentionally does NOT modify `det.py` (RGB-image workflow). Instead it provides a
parallel entrypoint for SPAD/QNN experiments where the model input is a photon cube HxWxT.

Expected input formats
----------------------
- A directory containing per-frame cubes as `.npy` files, each shaped (H, W, T)
  (sorted lexicographically).
- OR a single `.npy` file shaped:
    * (H, W, T)          (single cube)
    * (N, H, W, T)       (sequence of cubes)

The predictor expands each cube into a list of reconstructed frames (T' images) and runs YOLO
on each reconstructed frame.

Example
-------
python ultralytics/det_qnns.py \
  --in_path /path/to/acq00002 \
  --ckpt weights/detector.pt \
  --save_dir /path/to/save \
  --det_thresh 0.4 \
  --tracker botsort \
  --subsampling 64

Notes
-----
- This script relies on your `SPADPosePredictor` to do the cube->frames reconstruction.
- For now we keep visualization identical to `det.py`: draw bbox + 21-keypoint hand skeleton.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.models.yolo.pose.spad_predict import SPADPosePredictor



def _np_load(path: Path) -> np.ndarray:
    """Load .npy with memory mapping (keeps most data on disk)."""
    return np.load(path, mmap_mode='r')


# ----------------------------
# Visualization (copied from det.py)
# ----------------------------
BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Mid
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]
FINGER_COLORS = [
    (0, 0, 255),  # Thumb - Red
    (255, 0, 0),  # Index - Blue
    (0, 255, 0),  # Mid - Green
    (0, 255, 255),  # Ring - Yellow
    (255, 0, 255),  # Pinky - magenta
]
COLOR_KEYPOINT = (255, 255, 255)  # Joint - White
COLOR_WRIST = (255, 165, 0)  # Wrist - Orange


def _get_finger_color(bone_index: int) -> tuple[int, int, int]:
    if bone_index < 4:
        return FINGER_COLORS[0]
    if bone_index < 8:
        return FINGER_COLORS[1]
    if bone_index < 12:
        return FINGER_COLORS[2]
    if bone_index < 16:
        return FINGER_COLORS[3]
    return FINGER_COLORS[4]


def draw_bbox(img_bgr: np.ndarray, track_id: int, box_xyxyc: np.ndarray, handedness: float) -> np.ndarray:
    x1, y1, x2, y2, conf = box_xyxyc
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    color = (0, 0, 255) if handedness > 0 else (255, 0, 0)
    text = f"ID: {int(track_id)}"

    font_scale = 0.8
    thickness = 2
    text_color = (255, 255, 255)

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pt1 = (x1, y1)
    pt2 = (x1 + tw, y1 + th + baseline)
    text_org = (x1, y1 + baseline + baseline)

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(img_bgr, pt1, pt2, color, -1)
    cv2.putText(img_bgr, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    return img_bgr


def draw_pose(img_bgr: np.ndarray, pose_kpts: np.ndarray, thresh: float = 0.5, k: int = 21) -> np.ndarray:
    # pose_kpts: (K,3) with (x,y,conf)
    if pose_kpts.shape != (k, 3):
        raise ValueError(f"Pose shape must be ({k}, 3), but got {pose_kpts.shape}")

    for i, (s, e) in enumerate(BONE_CONNECTIONS):
        ks = pose_kpts[s]
        ke = pose_kpts[e]
        if ks[2] > thresh and ke[2] > thresh:
            cv2.line(
                img_bgr,
                (int(ks[0]), int(ks[1])),
                (int(ke[0]), int(ke[1])),
                _get_finger_color(i),
                3,
            )

    for i in range(k):
        kk = pose_kpts[i]
        if kk[2] > thresh:
            center = (int(kk[0]), int(kk[1]))
            if i == 0:
                color, radius = COLOR_WRIST, 6
            else:
                color, radius = COLOR_KEYPOINT, 4
            cv2.circle(img_bgr, center, radius, color, -1)

    return img_bgr


# ----------------------------
# Data loading utilities
# ----------------------------

def _packed_frames_to_cube(frames_packed: np.ndarray, *, reduce: str = "any", expected_w: int = 512) -> np.ndarray:
    """Convert VisionSIM packed frames (N,H,Wpacked,C) -> cube (H,W,T) bool.

    VisionSIM often packs width bits: Wpacked = W/8. For example (N,512,64,3) packs a 512-wide image.

    Args:
        frames_packed: ndarray shaped (N,H,Wpacked,C) with uint8 bit-packed values.
        reduce: how to reduce channels to a single photon observation per pixel.
            - 'any': photon = OR over channels
            - 'sum': photon = sum(ch)>0
        expected_w: crop/unpack to this width.

    Returns:
        cube bool array shaped (H,W,T)
    """
    if frames_packed.ndim != 4:
        raise ValueError(f"Expected packed frames (N,H,Wpacked,C), got shape={frames_packed.shape}")

    # Unpack bits along the packed-width axis; result: (N,H,Wbits,C)
    unpacked = np.unpackbits(frames_packed, axis=2)
    if unpacked.shape[2] > expected_w:
        unpacked = unpacked[:, :, :expected_w, :]

    if reduce == "any":
        ph_nhw = unpacked.any(axis=3)
    elif reduce == "sum":
        ph_nhw = unpacked.sum(axis=3) > 0
    elif reduce == "rggb_raw":
        # Treat channels as (R,G,B) responses (or BGR if specified) and sample into a Bayer mosaic RAW plane.
        ph_nhw = np.zeros(unpacked.shape[:3], dtype=bool)

        ch_order = getattr(_packed_frames_to_cube, "_packed_ch_order", "RGB")
        if ch_order == "BGR":
            r_ch, g_ch, b_ch = 2, 1, 0
        else:
            r_ch, g_ch, b_ch = 0, 1, 2

        ph_nhw[:, 0::2, 0::2] = unpacked[:, 0::2, 0::2, r_ch].astype(bool, copy=False)  # R
        ph_nhw[:, 0::2, 1::2] = unpacked[:, 0::2, 1::2, g_ch].astype(bool, copy=False)  # G
        ph_nhw[:, 1::2, 0::2] = unpacked[:, 1::2, 0::2, g_ch].astype(bool, copy=False)  # G
        ph_nhw[:, 1::2, 1::2] = unpacked[:, 1::2, 1::2, b_ch].astype(bool, copy=False)  # B
    elif reduce == "rggb_expand":
        # Expand 3-channel packed info into a wider (H, 2W) single-channel raw plane before PPB.
        # Intended workflow: unpacked (N,H,W,3) -> raw (N,H,2W) -> cube (H,2W,T) -> PPB -> demosaic/pack back.
        ch_order = getattr(_packed_frames_to_cube, "_packed_ch_order", "RGB")
        if ch_order == "BGR":
            r_ch, g_ch, b_ch = 2, 1, 0
        else:
            r_ch, g_ch, b_ch = 0, 1, 2

        n, h, w, _ = unpacked.shape
        raw = np.zeros((n, h, w * 2), dtype=bool)

        # Pack Bayer sites horizontally: raw[..., 2*x] holds R/B sites, raw[..., 2*x+1] holds G sites.
        # R at (even row, even col)
        raw[:, 0::2, 0::4] = unpacked[:, 0::2, 0::2, r_ch].astype(bool, copy=False)
        # B at (odd row, odd col)
        raw[:, 1::2, 2::4] = unpacked[:, 1::2, 1::2, b_ch].astype(bool, copy=False)
        # G at (even row, odd col)
        raw[:, 0::2, 3::4] = unpacked[:, 0::2, 1::2, g_ch].astype(bool, copy=False)
        # G at (odd row, even col)
        raw[:, 1::2, 1::4] = unpacked[:, 1::2, 0::2, g_ch].astype(bool, copy=False)

        ph_nhw = raw
    else:
        raise ValueError(f"Unsupported reduce mode: {reduce}")

    # (N,H,W) -> (H,W,N)
    return np.transpose(ph_nhw, (1, 2, 0)).astype(bool, copy=False)


def _iter_cubes_from_path(in_path: Path, *, packed_reduce: str = "any") -> Iterable[np.ndarray]:
    """Yield photon cubes from a path.

    Accepts:
      - directory of *.npy, each (H,W,T)
      - single *.npy containing (H,W,T) or (N,H,W,T)
      - VisionSIM packed frames: (N,H,Wpacked,3) e.g. (17921,512,64,3)
    """

    def _accept_or_convert(arr: np.ndarray, src: Path | None = None):
        if arr.ndim == 3:
            return arr
        if arr.ndim == 4 and arr.shape[-1] == 3 and arr.shape[2] in (64, 128):
            return _packed_frames_to_cube(arr, reduce=packed_reduce)
        if src is not None:
            raise ValueError(f"Unsupported array shape in {src}: {arr.shape}")
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    if in_path.is_dir():
        files = sorted([p for p in in_path.iterdir() if p.suffix.lower() == ".npy"])
        if not files:
            raise FileNotFoundError(f"No .npy files found in directory: {in_path}")
        for p in files:
            yield _accept_or_convert(_np_load(p), src=p)
        return

    if in_path.suffix.lower() == ".npy":
        arr = _np_load(in_path)
        if arr.ndim == 3:
            yield arr
            return
        if arr.ndim == 4 and not (arr.shape[-1] == 3 and arr.shape[2] in (64, 128)):
            # (N,H,W,T) sequence of cubes
            for i in range(arr.shape[0]):
                yield arr[i]
            return
        yield _accept_or_convert(arr, src=in_path)
        return

    raise ValueError(f"Unsupported input path: {in_path} (expect directory or .npy)")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="SPAD cube -> reconstructed frames -> YOLO pose tracking")
    ap.add_argument("--in_path", type=str, required=True, help="Directory of .npy cubes or a .npy file")
    ap.add_argument("--in_glob", type=str, default=None,
                    help="Optional glob (e.g. '*') when --in_path is a root folder containing many dataset subfolders")
    ap.add_argument("--save_dir", type=str, required=True, help="Directory to save visualized frames")
    ap.add_argument("--ckpt", type=str, default="weights/detector.pt")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--tracker", type=str, default="botsort", choices=["bytetrack", "botsort"])

    # PerPixelBayesian / SPAD options passed into predictor
    ap.add_argument("--subsampling", type=int, default=64, help="PerPixelBayesian subsampling (T -> T')")
    ap.add_argument("--bocpd_gamma", type=float, default=5e-4)
    ap.add_argument("--quantile", type=float, default=1.0)
    ap.add_argument("--min_filter_size", type=int, default=7)

    # VisionSIM bit-packed video 输入 (N,H,Wpacked,3) 的解包设置
    ap.add_argument("--packed_reduce", type=str, default="any", choices=["any", "sum", "rggb_raw", "rggb_expand"],
                    help="bit-packed (N,H,Wpacked,3) 输入时，将 3 通道归约为单通道 photon：any=OR；sum=SUM>0")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"],
                    help="packed_reduce=rggb_raw 时，packed 的 3 通道顺序（用于从三通道采样 Bayer raw）。")
    ap.add_argument("--bayer_pattern", type=str, default="RGGB", choices=["RGGB", "BGGR", "GRBG", "GBRG"],
                    help="packed_reduce=rggb_raw 时，单通道 raw 的 Bayer 排列（用于 demosaic）。")


    # Cube chunking (temporal)
    ap.add_argument("--cube_chunk_t", type=int, default=0, help="If >0, split each cube into chunks of this many time bins and call track() per chunk")
    ap.add_argument("--cube_chunk_stride", type=int, default=0, help="Stride for chunking; default uses cube_chunk_t (no overlap)")
    ap.add_argument("--vis_bg", type=str, default="sum", choices=["sum", "recon"], help="可视化背景：sum=时间维求和；recon=PerPixelBayesian 重建帧")

    args = ap.parse_args()

    # det_qnns 始终传给 predictor 单通道 cube；若该单通道来自 rggb_raw 采样，则 predictor 必须按 Bayer RGGB demosaic 成 RGB。
    spad_rgb_mode = "rggb_demosaic" if args.packed_reduce in ("rggb_raw", "rggb_expand") else "gray"

    _packed_frames_to_cube._packed_ch_order = args.packed_ch_order

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Support root folder with multiple datasets: iterate subfolders when --in_glob is set.
    if in_path.is_dir() and args.in_glob:
        dataset_paths = [p for p in sorted(in_path.glob(args.in_glob)) if p.is_dir()]
        if not dataset_paths:
            raise FileNotFoundError(f"No dataset subfolders matched: {in_path}/{args.in_glob}")
    else:
        dataset_paths = [in_path]

    model = YOLO(args.ckpt)

    tracker_cfg = f"{args.tracker}.yaml" if args.tracker in ("bytetrack", "botsort") else "botsort.yaml"

    # Predictor kwargs
    # Important: SPADPosePredictor is expected to accept (H,W,T) cube and expand into a list of frames.
    spad_bayes_kwargs = {
        "subsampling": args.subsampling,
        "bocpd_gamma": args.bocpd_gamma,
        # IMPORTANT: disable integrator's internal normalization to avoid non-causal global statistics.
        "normalize": False,
        "quantile": args.quantile,
        "min_filter_size": args.min_filter_size,
    }
    global_frame_idx = 0

    for dataset_path in dataset_paths:
        dataset_name = dataset_path.name if dataset_path.is_dir() else dataset_path.stem
        out_dir = save_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        cube_iter = _iter_cubes_from_path(dataset_path, packed_reduce=args.packed_reduce)

        # Process each cube (each cube yields T' reconstructed frames, each then yields a YOLO result).
        for cube_idx, cube in enumerate(tqdm(cube_iter, desc=f"Processing cubes [{dataset_name}]")):

            # Reset SPAD state at the start of each big cube/sequence.
            first_chunk = True

            # Split big cube into temporal chunks (optional) and call track() sequentially.
            T = cube.shape[2]
            chunk_t = int(args.cube_chunk_t) if int(args.cube_chunk_t) > 0 else T
            stride = int(args.cube_chunk_stride) if int(args.cube_chunk_stride) > 0 else chunk_t
            for t0 in range(0, T, stride):
                t1 = min(T, t0 + chunk_t)
                cube_chunk = cube[:, :, t0:t1]
                if cube_chunk.shape[2] == 0:
                    continue

                results = model.track(
                    cube_chunk,
                    conf=args.det_thresh,
                    persist=True,
                    tracker=tracker_cfg,
                    verbose=False,
                    predictor=SPADPosePredictor,
                    spad=True,
                    spad_pre="bayes",
                    spad_clear_states=first_chunk,
                    spad_bayes_kwargs=spad_bayes_kwargs,
                    spad_collapse="frames",
                    spad_rgb_mode=spad_rgb_mode,
                    spad_packed_reduce=args.packed_reduce,
                    spad_bayer_pattern=args.bayer_pattern,
                    spad_packed_ch_order=args.packed_ch_order,
                )
                first_chunk = False

                # 可视化背景：sum 或 recon
                recon_frames = None
                if args.vis_bg == "recon":
                    try:
                        recon_frames = getattr(getattr(model, "predictor", None), "last_recon_frames_u8", None)
                    except Exception:
                        recon_frames = None

                bg_bgr = None
                if args.vis_bg == "sum" or recon_frames is None:
                    bg = cube_chunk.astype(np.float32).sum(axis=2)
                    bg = bg / (bg.max() + 1e-6)
                    bg_u8 = (bg * 255.0).round().astype(np.uint8)
                    bg_bgr = np.repeat(bg_u8[:, :, None], 3, axis=2)

                for i, r in enumerate(results):
                    if args.vis_bg == "recon" and recon_frames is not None and i < len(recon_frames):
                        vis = recon_frames[i].copy()
                    else:
                        vis = bg_bgr.copy() if bg_bgr is not None else np.zeros((cube_chunk.shape[0], cube_chunk.shape[1], 3), dtype=np.uint8)

                    if r.boxes is not None and r.boxes.id is not None:
                        track_id = r.boxes.id.cpu().numpy()
                        boxes = r.boxes.xyxy.cpu().numpy()
                        box_confs = r.boxes.conf.cpu().numpy()
                        handedness = r.boxes.cls.cpu().numpy()

                        if hasattr(r, "keypoints") and r.keypoints is not None:
                            poses = r.keypoints.xy.cpu().numpy()  # (n,K,2)
                            pose_confs = r.keypoints.conf.cpu().numpy()  # (n,K)
                            poses = np.concatenate([poses, pose_confs[..., None]], axis=2)  # (n,K,3)
                        else:
                            poses = None

                        for j, tid in enumerate(track_id):
                            box_xyxyc = np.concatenate([boxes[j], [box_confs[j]]], axis=0)
                            vis = draw_bbox(vis, int(tid), box_xyxyc, float(handedness[j]))
                            if poses is not None:
                                vis = draw_pose(vis, poses[j])

                    out_path = out_dir / f"cube{cube_idx:05d}_t{t0:06d}_{t1:06d}_frame{global_frame_idx:07d}.png"
                    cv2.imwrite(str(out_path), vis)
                    global_frame_idx += 1


if __name__ == "__main__":
    main()
