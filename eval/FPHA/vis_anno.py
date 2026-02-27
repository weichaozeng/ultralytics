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

file_pose = os.path.join(folder_pose, 'skeleton.txt')
rgb_folder = os.path.join(folder_frame, 'color')


output_video_path = f'{person_id}_{action_id}_{rep}_tracking_anno.mp4'


fx_c, fy_c = 1395.749023, 1395.749268
u0_c, v0_c = 935.732544, 540.681030


M = np.array([
    [ 0.999988496304,   -0.00468848412856,  0.000982563360594, 25.7  ],
    [ 0.00469115935266,  0.999985218048,   -0.00273845880292,  1.22  ],
    [-0.000969709653873, 0.00274303671904,  0.99999576807,     3.902 ],
    [ 0.0,               0.0,               0.0,               1.0   ]
])


fingers_indices = [
    [0, 1, 6, 7, 8],     
    [0, 2, 9, 10, 11],   
    [0, 3, 12, 13, 14],  
    [0, 4, 15, 16, 17],  
    [0, 5, 18, 19, 20]   
]


colors_bgr = [
    (255, 0, 255),   # Magenta -> BGR
    (255, 0, 0),     # Blue -> BGR
    (0, 255, 0),     # Green -> BGR
    (0, 255, 255),   # Yellow -> BGR
    (0, 0, 255)      # Red -> BGR
]


all_poses = np.loadtxt(file_pose)

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
    if i >= len(all_poses):
        break
        
    pose_data = all_poses[i, 1:] 
    jointLocations = pose_data.reshape(21, 3)


    joints_homogeneous = np.hstack((jointLocations, np.ones((21, 1))))
    joints_tf = (M @ joints_homogeneous.T).T
    
    u_c = u0_c + fx_c * joints_tf[:, 0] / joints_tf[:, 2]
    v_c = v0_c + fy_c * joints_tf[:, 1] / joints_tf[:, 2]


    u = np.int32(np.round(u_c))
    v = np.int32(np.round(v_c))


    img = cv2.imread(img_path)


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
        print(f"Alread {i + 1} / {len(image_files)}")

video_writer.release()
print(f"Done! Save to: {output_video_path}")