import numpy as np
import cv2
import os
import glob

person_id = 'Subject_6'
action_id = 'receive_coin'
rep = '2'
root = '/home/zvc/Data/FPHA/'

folder_frame = os.path.join(root, 'Video_files', person_id, action_id, rep)
folder_pose = os.path.join(root, 'Hand_pose_annotation_v1', person_id, action_id, rep)


file_kp2d = os.path.join(folder_pose, '2d_keypoints.txt')
file_bbox2d = os.path.join(folder_pose, '2d_bbox.txt')

rgb_folder = os.path.join(folder_frame, 'color')
output_video_path = f'{person_id}_{action_id}_{rep}_2d_anno_demo.mp4'


fingers_indices = [
    [0, 1, 6, 7, 8],     
    [0, 2, 9, 10, 11],   
    [0, 3, 12, 13, 14],  
    [0, 4, 15, 16, 17],  
    [0, 5, 18, 19, 20]   
]

colors_bgr = [
    (255, 0, 255),   # Magenta 
    (255, 0, 0),     # Blue 
    (0, 255, 0),     # Green 
    (0, 255, 255),   # Yellow 
    (0, 0, 255)      # Red 
]


kp_data = np.loadtxt(file_kp2d, dtype=str)
bbox_data = np.loadtxt(file_bbox2d, dtype=str)

image_files = sorted(glob.glob(os.path.join(rgb_folder, '*.jpeg')))

if not image_files:
    print(f"Error: cannot find .jpeg file in {rgb_folder}.")
    exit()

first_frame = cv2.imread(image_files[0])
height, width, layers = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

print(f"Start annotating, all {len(image_files)} frames...")


for i, img_path in enumerate(image_files):

    if i >= len(kp_data) or i >= len(bbox_data):
        break
        

    kp_row = kp_data[i, 1:].astype(float) 
    u = np.int32(np.round(kp_row[0::2])) 
    v = np.int32(np.round(kp_row[1::2])) 


    cx, cy, w, h = bbox_data[i, 1:].astype(float)
    

    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))

    img = cv2.imread(img_path)


    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=2)


    cv2.circle(img, (u[0], v[0]), radius=5, color=(255, 255, 255), thickness=-1)

    for finger_idx, color in zip(fingers_indices, colors_bgr):
        for j in range(len(finger_idx) - 1):
            pt1 = (u[finger_idx[j]], v[finger_idx[j]])
            pt2 = (u[finger_idx[j+1]], v[finger_idx[j+1]])
            
            cv2.line(img, pt1, pt2, color, thickness=2)
            if j > 0 or j == 0: 
                cv2.circle(img, pt2, radius=4, color=color, thickness=-1)

    video_writer.write(img)

    if (i + 1) % 50 == 0:
        print(f"Already {i + 1} / {len(image_files)}")

video_writer.release()
print(f"Done! Save to: {output_video_path}")