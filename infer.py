from ultralytics import YOLO
import argparse
import os
import cv2
import numpy as np
from typing import List
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


def get_frames(args, name) -> List[np.ndarray]:
    out_frames = []
    out_names = []
    if args.in_type == "video":
        assert name.lower().endswith('.mp4'), print(f"Conflit with file type {name} and args.intype {args.in_type}.")
        video_path = os.path.join(args.in_dir, name)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Error opening video stream: {video_path}")
        idx_name = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            out_frames.append(frame)
            out_names.append(str(idx_name).zfill(6) + '.jpg')
            idx_name += 1
        cap.release()
        if not out_frames:
            raise FileNotFoundError(f"No valid frames found for: {name}")
        return out_frames, out_names
    else:
        if args.in_type == "":
            seq_dir = os.path.join(args.in_dir, name)
        else:
            seq_dir = os.path.join(args.in_dir, name, args.in_type)
        if not os.path.isdir(seq_dir):
            raise FileNotFoundError(f"Image sequence directory not found: {seq_dir}")
        img_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        img_paths = []
        for filename in sorted(os.listdir(seq_dir)):
            if filename.lower().endswith(img_ext):
                img_paths.append(os.path.join(seq_dir, filename))
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read image: {img_path}. Skipping.")
                continue
            out_frames.append(img)
            out_names.append(os.path.basename(img_path))
        if not out_frames:
            raise FileNotFoundError(f"No valid images found for: {name}")
        return out_frames, out_names


def detect_track(args, model, frames):
    results = []
    for frame_cv2 in frames:
            with torch.no_grad():
                with autocast():
                    result = model.track(frame_cv2, conf=args.det_thresh, persist=True, verbose=False)
                    if not result[0].boxes.id is None:
                        track_id = result[0].boxes.id.cpu().numpy()
                        boxes = result[0].boxes.xyxy.cpu().numpy()
                        box_confs = result[0].boxes.conf.cpu().numpy()
                        handedness = result[0].boxes.cls.cpu().numpy()
                        poses = result[0].keypoints.xy.cpu().numpy()
                        pose_confs = result[0].keypoints.conf.cpu().numpy()
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


def save_results(args, frames, results, seq_name, frame_names):
    seq_name = seq_name.split('.')[0]
    assert len(frames) == len(results), print(f"Lenght missmatch between frames {len(frames)} and results {len(results)}.")
    if args.save_type == "video":
        img_h, img_w = frames[0].shape[:2]
        video_output_path = os.path.join(args.save_dir, f'{seq_name}.mp4')
        video_writer = cv2.VideoWriter(video_output_path,
                                       cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_w, img_h)) 
    else:
        os.makedirs(os.path.join(args.save_dir, seq_name), exist_ok=True)

    for i, frame in enumerate(frames):
        vis_frame = frame.copy()
        if results[i]['has_det']:
            for j, track_id in enumerate(results[i]['track_id']):
                vis_frame = draw_bbox(vis_frame, track_id, results[i]['boxes'][j],results[i]['handedness'][j])
                vis_frame = draw_pose(vis_frame, results[i]['poses'][j])
        if args.save_type == "video":
            video_writer.write(vis_frame)
        else:
            cv2.imwrite(os.path.join(args.save_dir, seq_name, frame_names[i]), vis_frame)


def draw_bbox(img_cv2, id, box, is_right):
    x1, y1, x2, y2, conf = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    color = (0, 0, 255) if is_right > 0 else (255, 0, 0)
    text = f'ID: {int(id)}'
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
    # Thumb (0-3) -> 0
    if bone_index < 4:
        return FINGER_COLORS[0]
    # Index (4-7) -> 1
    elif bone_index < 8:
        return FINGER_COLORS[1]
    # Mid (8-11) -> 2
    elif bone_index < 12:
        return FINGER_COLORS[2]
    # Ring (12-15) -> 3
    elif bone_index < 16:
        return FINGER_COLORS[3]
    # Pinky (16-19) -> 4
    else:
        return FINGER_COLORS[4]


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
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--in_dir", type=str, default="example_data/FPHA/")
    parser.add_argument("--in_type", type=str, choices=["video", "rgb", "color", ""], default="color")
    parser.add_argument("--save_dir", type=str, default="example_out/pa_nms/FPHA/")
    parser.add_argument("--save_type", type=str, default="img")
    parser.add_argument("--det_thresh", type=float, default=0.1)
    parser.add_argument("--ckpt", type=str, default="weights/detector.pt")

    args = parser.parse_args()

    model = YOLO(args.ckpt)

    for seq_name in tqdm(os.listdir(args.in_dir)):
        frames, frame_names = get_frames(args, seq_name)
        if args.save_type == "video":
            first_frame = frames[0]
            img_h, img_w = first_frame.shape[:2]
            video_output_path = os.path.join(args.save_dir, f'{seq_name}.mp4')
            video_writer = cv2.VideoWriter(video_output_path,
                                        cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_w, img_h)) 
        else:
            os.makedirs(args.save_dir, exist_ok=True)
        
        results = detect_track(args, model, frames)
        save_results(args, frames, results, seq_name, frame_names)




        

