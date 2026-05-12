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
            yield _accept_or_convert(np.load(p), src=p)
        return

    if in_path.suffix.lower() == ".npy":
        arr = np.load(in_path)
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
    ap.add_argument("--packed_reduce", type=str, default="any", choices=["any", "sum"],
                    help="bit-packed (N,H,Wpacked,3) 输入时，将 3 通道归约为单通道 photon：any=OR；sum=SUM>0")

    # Cube chunking (temporal)
    ap.add_argument("--cube_chunk_t", type=int, default=0, help="If >0, split each cube into chunks of this many time bins and call track() per chunk")
    ap.add_argument("--cube_chunk_stride", type=int, default=0, help="Stride for chunking; default uses cube_chunk_t (no overlap)")
    ap.add_argument("--vis_bg", type=str, default="sum", choices=["sum", "recon"], help="可视化背景：sum=时间维求和；recon=PerPixelBayesian 重建帧")

    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    cube_iter = list(_iter_cubes_from_path(in_path, packed_reduce=args.packed_reduce))

    # Process each cube (each cube yields T' reconstructed frames, each then yields a YOLO result).
    # We store results and also render per reconstructed frame.
    global_frame_idx = 0

    for cube_idx, cube in enumerate(tqdm(cube_iter, desc="Processing cubes")):
        # Reset SPAD state at the start of each big cube/sequence.
        # We do this by setting spad_clear_states=True only for the first chunk call.
        first_chunk = True

        # Split big cube into temporal chunks (optional) and call track() sequentially.
        T = cube.shape[2]
        chunk_t = int(args.cube_chunk_t) if int(args.cube_chunk_t) > 0 else T
        stride = int(args.cube_chunk_stride) if int(args.cube_chunk_stride) > 0 else chunk_t

        for t0 in range(0, T, stride):
            t1 = min(T, t0 + chunk_t)
            cube_chunk = cube[:, :, t0:t1]

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
            )
            _result_i = 0  # reset per chunk
            first_chunk = False

            # 可视化背景：
            # - sum: 对当前 chunk 在时间维求和（快，但信息粗）
            # - recon: 使用 SPADPosePredictor 内部缓存的 PerPixelBayesian 重建帧（更直观）
            recon_frames = None
            if args.vis_bg == "recon":
                try:
                    # 注意：predictor 是由 Ultralytics 内部实例化的；这里通过 model.predictor 取最近一次调用的实例
                    recon_frames = getattr(getattr(model, "predictor", None), "last_recon_frames_u8", None)
                except Exception:
                    recon_frames = None

            if args.vis_bg == "sum" or recon_frames is None:
                bg = cube_chunk.astype(np.float32).sum(axis=2)
                bg = bg / (bg.max() + 1e-6)
                bg_u8 = (bg * 255.0).round().astype(np.uint8)
                bg_bgr = np.repeat(bg_u8[:, :, None], 3, axis=2)
            # recon_frames 已经是 HxWx3 uint8 list，与 results 一一对应

            for r in results:
                # 逐帧选择背景：recon 模式下用对应的重建帧；否则用 bg_bgr(summed)
                # 这里用 enumerate 取索引更稳妥，但不重排结构：用一个计数器
                if args.vis_bg == "recon" and recon_frames is not None and _result_i < len(recon_frames):
                    vis = recon_frames[_result_i].copy()
                else:
                    vis = bg_bgr.copy()
                _result_i += 1

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

                out_path = save_dir / f"cube{cube_idx:05d}_t{t0:06d}_{t1:06d}_frame{global_frame_idx:07d}.png"
                cv2.imwrite(str(out_path), vis)
                global_frame_idx += 1

if __name__ == "__main__":
    main()
