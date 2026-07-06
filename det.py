from ultralytics import YOLO
import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Mid
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]
FINGER_COLORS = [
    (0, 0, 255),    # Thumb - Red
    (255, 0, 0),    # Index - Blue
    (0, 255, 0),    # Mid - Green
    (0, 255, 255),  # Ring - Yellow
    (255, 0, 255)   # Pinky - magenta
]
COLOR_KEYPOINT = (255, 255, 255) # Joint - White
COLOR_WRIST = (255, 165, 0)      # Wrist - Orange

TRACKER_CHOICES = ("bytetrack", "botsort", "spad_tracker", "posetrack", "spad_posetrack")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _tracker_yaml(tracker: str) -> str:
    """Resolve tracker config yaml path for ``model.track(tracker=...)``."""
    if tracker not in TRACKER_CHOICES:
        raise ValueError(f"Unsupported tracker {tracker!r}; choose from {TRACKER_CHOICES}")
    return f"{tracker}.yaml"


def _load_video_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")
    return frames


def detect_track(args, model, frames):
    """按视频序列逐帧进行 Tracking"""
    # 每次进入新的视频序列前，重置 Tracker，防止和上一个序列的 ID 串联
    if hasattr(model, 'predictor') and model.predictor is not None:
        if hasattr(model.predictor, 'trackers') and model.predictor.trackers:
            for tracker in model.predictor.trackers:
                tracker.reset()

    tracker_cfg = _tracker_yaml(args.tracker)
    results = []
    for frame_cv2 in frames:
        with torch.no_grad():
            with autocast():
                # 使用 track 代替 predict，开启 persist=True 保持帧间追踪
                result = model.track(frame_cv2, conf=args.det_thresh, persist=True, tracker=tracker_cfg, verbose=False)[0]
                
                # 必须存在检测框，并且 tracker 成功分配了 id
                if result.boxes is not None and result.boxes.id is not None:
                    track_id = result.boxes.id.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    box_confs = result.boxes.conf.cpu().numpy()
                    handedness = result.boxes.cls.cpu().numpy()
                    
                    if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints):
                        poses = result.keypoints.data.cpu().numpy()[..., :2]
                        if result.keypoints.data.shape[-1] >= 3:
                            pose_confs = result.keypoints.data.cpu().numpy()[..., 2]
                        else:
                            pose_confs = result.keypoints.conf.cpu().numpy()
                    else:
                        num_hands = len(boxes)
                        poses = np.zeros((num_hands, 21, 2))
                        pose_confs = np.zeros((num_hands, 21))

                    out = {
                        'has_det': True,
                        'track_id': track_id,
                        'boxes': np.hstack([boxes, box_confs[:, None]]),
                        'poses': np.concatenate([poses, pose_confs[..., None]], axis=2),
                        'handedness': handedness,
                    }
                else:
                    out = {
                        'has_det': False,
                    }
        results.append(out)
    return results


def draw_bbox(img_cv2, id, box, is_right):
    x1, y1, x2, y2, conf = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    color = (0, 0, 255) if is_right > 0 else (255, 0, 0)
    text = f'ID: {int(id)}'  # 恢复显示为 Tracking ID
    FONT_SCALE = 0.8
    THICKNESS_TEXT = 2
    TEXT_COLOR = (255, 255, 255)
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS_TEXT)
    pt1 = (x1, y1)
    pt2 = (x1 + text_w, y1 + text_h + baseline)
    text_org = (x1, y1 + baseline + baseline)
    cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(img_cv2, pt1, pt2, color, -1)
    cv2.putText(img_cv2, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS_TEXT)
    return img_cv2


def get_finger_color(bone_index: int) -> tuple:
    if bone_index < 4: return FINGER_COLORS[0]
    elif bone_index < 8: return FINGER_COLORS[1]
    elif bone_index < 12: return FINGER_COLORS[2]
    elif bone_index < 16: return FINGER_COLORS[3]
    else: return FINGER_COLORS[4]


def draw_pose(img_cv2, pose, thresh=0.5, K=21):
    if pose.shape != (K, 3):
        raise ValueError(f"Pose shape must be ({K}, 3), but got {pose.shape}")
    if isinstance(pose, torch.Tensor):
        keypoints = pose.cpu().numpy()
    else:
        keypoints = pose
    for i, (start_idx, end_idx) in enumerate(BONE_CONNECTIONS):
        kp_start = keypoints[start_idx]
        kp_end = keypoints[end_idx]
        if kp_start[2] > thresh and kp_end[2] > thresh:
            pt1 = (int(kp_start[0]), int(kp_start[1]))
            pt2 = (int(kp_end[0]), int(kp_end[1]))
            color = get_finger_color(i)
            cv2.line(img_cv2, pt1, pt2, color, 3)
    for i in range(K):
        kp = keypoints[i]
        if kp[2] > thresh:
            center = (int(kp[0]), int(kp[1]))
            if i == 0:
                color = COLOR_WRIST 
                radius = 6
            else:
                color = COLOR_KEYPOINT
                radius = 4
            cv2.circle(img_cv2, center, radius, color, -1) 
    return img_cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image sequence or RGB video tracking")
    parser.add_argument("--in_dir", type=str, default="/home/zvc/Project/SPADHand/Vis/spad_sum_100",
                        help="Image folder, parent of sequences, or RGB video file (.mp4, ...)")
    parser.add_argument("--save_base_dir", type=str, default="/home/zvc/Project/SPADHand/Pred")
    parser.add_argument("--det_thresh", type=float, default=0.4)
    parser.add_argument("--ckpt", type=str, default="weights/detector.pt")
    parser.add_argument("--tracker", type=str, default="posetrack", choices=list(TRACKER_CHOICES))
    
    args = parser.parse_args()
    model = YOLO(args.ckpt)

    if not os.path.exists(args.in_dir):
        raise FileNotFoundError(f"Input path not found: {args.in_dir}")

    video_mode = False
    # 单个 RGB 视频文件
    if os.path.isfile(args.in_dir) and args.in_dir.lower().endswith(VIDEO_EXT):
        video_name = os.path.splitext(os.path.basename(args.in_dir))[0]
        frames = _load_video_frames(args.in_dir)
        seq_dict = {video_name: None}
        in_dir_name = video_name
        video_mode = True
    else:
        items = sorted(os.listdir(args.in_dir))
        contains_images = any(f.lower().endswith(IMG_EXT) for f in items)
        seq_dict = {}
        in_dir_name = os.path.basename(os.path.normpath(args.in_dir))

        if contains_images:
            seq_dict[""] = args.in_dir
        else:
            for item in items:
                item_path = os.path.join(args.in_dir, item)
                if os.path.isdir(item_path):
                    seq_dict[item] = item_path

    if not seq_dict:
        print(f"No valid image sequences found in {args.in_dir}")
        exit()
        
    print(f"Found {len(seq_dict)} sequence(s) to process. tracker={args.tracker}")

    for seq_rel_name, seq_dir in tqdm(seq_dict.items(), desc="Processing Sequences"):
        if video_mode:
            seq_frames = frames
            img_files = [f"frame_{i:06d}.jpg" for i in range(len(seq_frames))]
        else:
            img_files = [f for f in sorted(os.listdir(seq_dir)) if f.lower().endswith(IMG_EXT)]
            if not img_files:
                continue

            seq_frames = []
            for img_name in img_files:
                img_path = os.path.join(seq_dir, img_name)
                frame = cv2.imread(img_path)
                if frame is not None:
                    seq_frames.append(frame)

        if not seq_frames:
            continue

        try:
            results = detect_track(args, model, seq_frames)

            if video_mode:
                seq_save_dir = os.path.join(args.save_base_dir, in_dir_name)
            elif seq_rel_name == "":
                seq_save_dir = os.path.join(args.save_base_dir, in_dir_name)
            else:
                seq_save_dir = os.path.join(args.save_base_dir, in_dir_name, seq_rel_name)

            os.makedirs(seq_save_dir, exist_ok=True)

            for i, frame in enumerate(seq_frames):
                vis_frame = frame.copy()
                if results[i]['has_det']:
                    for j, track_id in enumerate(results[i]['track_id']):
                        vis_frame = draw_bbox(vis_frame, track_id, results[i]['boxes'][j], results[i]['handedness'][j])
                        vis_frame = draw_pose(vis_frame, results[i]['poses'][j])

                save_path = os.path.join(seq_save_dir, img_files[i])
                cv2.imwrite(save_path, vis_frame)

        except Exception as e:
            seq_label = seq_rel_name if video_mode else seq_dir
            print(f"Error when processing sequence {seq_label}: {e}")