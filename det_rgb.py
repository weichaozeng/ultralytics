# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""det_rgb.py

Run YOLO pose tracking on pre-rendered RGB frames stored in a numpy array.

This is a simplified sibling of `det_qnns.py`:
- Input is an RGB frames numpy file (default: frames.npy) shaped (N,H,W,3).
- No SPAD preprocessing; runs standard Ultralytics predictor.
- Visualization matches `det_qnns.py`/`det.py`: bbox + 21-keypoint skeleton.

Example
-------
python ultralytics/det_rgb.py \
  --in_path /path/to/dir_or_frames.npy \
  --ckpt weights/detector.pt \
  --save_dir /path/to/save/rgb_25fps \
  --det_thresh 0.4 \
  --tracker botsort
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO


# ----------------------------
# Visualization (kept consistent with det_qnns.py)
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


def _load_rgb_frames(in_path: Path) -> np.ndarray:
    """Load RGB frames array (N,H,W,3) from a path.

    Accepts:
    - a directory containing frames.npy
    - a direct path to a .npy file
    """
    if in_path.is_dir():
        npy = in_path / "frames.npy"
        if not npy.exists():
            raise FileNotFoundError(f"Directory input requires frames.npy, not found: {npy}")
        arr = np.load(npy)
        src = npy
    else:
        if in_path.suffix.lower() != ".npy":
            raise ValueError(f"Unsupported input: {in_path} (expect directory or .npy)")
        arr = np.load(in_path)
        src = in_path

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames shaped (N,H,W,3) in {src}, got {arr.shape}")

    # Ensure uint8 RGB
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            # assume [0,1]
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def main():
    ap = argparse.ArgumentParser(description="RGB frames (.npy) -> YOLO pose tracking")
    ap.add_argument("--in_path", type=str, required=True, help="Directory containing frames.npy or a frames.npy path")
    ap.add_argument("--save_dir", type=str, default="rgb_25fps", help="Directory to save visualized frames")
    ap.add_argument("--ckpt", type=str, default="weights/detector.pt")
    ap.add_argument("--det_thresh", type=float, default=0.4)
    ap.add_argument("--tracker", type=str, default="botsort", choices=["bytetrack", "botsort"])

    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    save_dir = Path(args.save_dir)
    # As requested, default save folder name is rgb_25fps
    save_dir.mkdir(parents=True, exist_ok=True)

    frames_rgb = _load_rgb_frames(in_path)  # (N,H,W,3) RGB

    model = YOLO(args.ckpt)
    tracker_cfg = f"{args.tracker}.yaml" if args.tracker in ("bytetrack", "botsort") else "botsort.yaml"

    # Convert RGB->BGR for OpenCV drawing; Ultralytics accepts np arrays, but we keep a BGR copy for rendering.
    frames_bgr = frames_rgb[..., ::-1].copy()

    global_frame_idx = 0
    for i in tqdm(range(frames_bgr.shape[0]), desc="Processing RGB frames"):
        frame_bgr = frames_bgr[i]

        # Track on a single frame. Use persist=True so IDs continue across frames.
        results = model.track(
            frame_bgr,
            conf=args.det_thresh,
            persist=True,
            tracker=tracker_cfg,
            verbose=False,
        )

        # Ultralytics returns a list (len==1 for single-frame input)
        r = results[0]
        vis = frame_bgr.copy()

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

        out_path = save_dir / f"frame{global_frame_idx:07d}.png"
        cv2.imwrite(str(out_path), vis)
        global_frame_idx += 1


if __name__ == "__main__":
    main()
