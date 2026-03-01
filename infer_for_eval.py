import os
import cv2
import numpy as np
import torch
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import glob


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

def interpolate_track(track_dict):
    frames = sorted(list(track_dict.keys()))
    if len(frames) <= 1:
        return track_dict
    
    first_frame, last_frame = frames[0], frames[-1]
    known_frames = np.array(frames)
    known_data = np.array([track_dict[f] for f in known_frames])
    
    interpolated_dict = {}
    for f in range(first_frame, last_frame + 1):
        if f in track_dict:
            interpolated_dict[f] = track_dict[f]
        else:
            interp_val = np.zeros(known_data.shape[1])
            for dim in range(known_data.shape[1]):
                interp_val[dim] = np.interp(f, known_frames, known_data[:, dim])
            interpolated_dict[f] = interp_val
            
    return interpolated_dict

def main(args):
    model = YOLO(args.ckpt)
    
    yolo_to_fpha_idx = [
        0,                  
        1, 5, 9, 13, 17,  
        2, 3, 4,            
        6, 7, 8,          
        10, 11, 12,        
        14, 15, 16,         
        18, 19, 20          
    ]

    print("Scanning dataset directories...")
    seq_tasks = []
    person_list = sorted([p for p in os.listdir(args.in_dir) if os.path.isdir(os.path.join(args.in_dir, p))])
    
    for person_id in person_list:
        person_path = os.path.join(args.in_dir, person_id)
        action_list = sorted(os.listdir(person_path))
        
        for action_id in action_list:
            action_path = os.path.join(person_path, action_id)
            rep_list = sorted(os.listdir(action_path))
            
            for rep in rep_list:
                color_dir = os.path.join(action_path, rep, 'color')
                if not os.path.exists(color_dir):
                    continue
                
                image_files = sorted(glob.glob(os.path.join(color_dir, '*.jpeg')))
                if len(image_files) == 0:
                    continue
                
                
                seq_tasks.append({
                    'person_id': person_id,
                    'action_id': action_id,
                    'rep': rep,
                    'image_files': image_files,
                    'num_frames': len(image_files)
                })

    print(f"Found {len(seq_tasks)} video sequences to process.")

   
    for task in tqdm(seq_tasks, desc="Tracking Sequences"):
        
        person_id = task['person_id']
        action_id = task['action_id']
        rep = task['rep']
        image_files = task['image_files']
        num_frames = task['num_frames']
        tqdm.write(f"Tracking: {person_id}/{action_id}/{rep}")
   
        seq_save_dir = os.path.join(args.save_dir, person_id, action_id, rep)
        os.makedirs(seq_save_dir, exist_ok=True)
                
        
        if hasattr(model, 'predictor') and model.predictor is not None:
            if hasattr(model.predictor, 'trackers') and model.predictor.trackers:
                model.predictor.trackers[0].reset()

        tracks_bbox_data = {}
        tracks_kp_data = {}
        

        for frame_idx, img_path in enumerate(image_files):
            frame_cv2 = cv2.imread(img_path)

            with torch.no_grad():
                with autocast():
                    if args.tracker == "posetrack":
                        result = model.track(frame_cv2, conf=args.det_thresh, persist=True, verbose=False, tracker="./ultralytics/custom/posetrack.yaml")
                    elif args.tracker == "bytetrack":
                        result = model.track(frame_cv2, conf=args.det_thresh, persist=True, verbose=False, tracker="./ultralytics/cfg/trackers/bytetrack.yaml")
                    elif args.tracker == "botsort":
                        result = model.track(frame_cv2, conf=args.det_thresh, persist=True, verbose=False, tracker="./ultralytics/cfg/trackers/botsort.yaml")
                    else:
                        raise ValueError(f"Unsupported tracker type: {args.tracker}")
                    
                    if not result[0].boxes.id is None:
                        track_ids = result[0].boxes.id.cpu().numpy().astype(int)
                        boxes_xywh = result[0].boxes.xywh.cpu().numpy()
                        poses = result[0].keypoints.data.cpu().numpy()
                        
                        for i, tid in enumerate(track_ids):
                            if tid not in tracks_bbox_data:
                                tracks_bbox_data[tid] = {}
                                tracks_kp_data[tid] = {}
                            

                            cx, cy, w, h = boxes_xywh[i]
                            tracks_bbox_data[tid][frame_idx] = np.array([cx, cy, w, h])
                            
                            # 21 * 2
                            kps = poses[i]
                            kps_reordered = kps[yolo_to_fpha_idx]
                            kps_xy = kps_reordered[:, :2] 
                            tracks_kp_data[tid][frame_idx] = kps_xy.flatten()


        for tid in tracks_bbox_data.keys():
            interp_bbox = interpolate_track(tracks_bbox_data[tid])
            interp_kp = interpolate_track(tracks_kp_data[tid])
            
            bbox_txt_path = os.path.join(seq_save_dir, f"track_{tid}_bbox.txt")
            kp_txt_path = os.path.join(seq_save_dir, f"track_{tid}_kp.txt")
            

            zero_bbox = " ".join(["0.00"] * 4)
            zero_kp = " ".join(["0.00"] * 42)
            
            with open(bbox_txt_path, 'w') as f_bbox:
                for frame_idx in range(num_frames):
                    display_frame_id = frame_idx + 1 
                    if frame_idx in interp_bbox:
                        row_str = " ".join([f"{val:.2f}" for val in interp_bbox[frame_idx]])
                    else:
                        row_str = zero_bbox 
                    f_bbox.write(f"{display_frame_id} {row_str}\n")
                    
            with open(kp_txt_path, 'w') as f_kp:
                for frame_idx in range(num_frames):
                    display_frame_id = frame_idx + 1 
                    if frame_idx in interp_kp:
                        row_str = " ".join([f"{val:.2f}" for val in interp_kp[frame_idx]])
                    else:
                        row_str = zero_kp 
                    f_kp.write(f"{display_frame_id} {row_str}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPHA Tracking Inference")
    parser.add_argument("--in_dir", type=str, default="/home/zvc/Data/FPHA/Video_files")
    parser.add_argument("--save_dir", type=str, default="/home/zvc/Project/FPHA/Tracking_Results/ByteTrack_0.5")
    parser.add_argument("--det_thresh", type=float, default=0.5)
    parser.add_argument("--ckpt", type=str, default="weights/detector.pt")
    parser.add_argument("--tracker", type=str, default="bytetrack", choices=["posetrack", "bytetrack", "botsort"])

    args = parser.parse_args()
    main(args)