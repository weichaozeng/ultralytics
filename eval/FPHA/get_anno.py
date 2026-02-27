import numpy as np
import os
from tqdm import tqdm

root = '/home/zvc/Data/FPHA/Hand_pose_annotation_v1'  
anno_files = []
save_kp_files = []
save_bbox_files = []

fx_c, fy_c = 1395.749023, 1395.749268
u0_c, v0_c = 935.732544, 540.681030


M = np.array([
    [ 0.999988496304,   -0.00468848412856,  0.000982563360594, 25.7  ],
    [ 0.00469115935266,  0.999985218048,   -0.00273845880292,  1.22  ],
    [-0.000969709653873, 0.00274303671904,  0.99999576807,     3.902 ],
    [ 0.0,               0.0,               0.0,               1.0   ]
])

for person_id in os.listdir(root):
    for action_id in os.listdir(os.path.join(root, person_id)):
        for rep in os.listdir(os.path.join(root, person_id, action_id)):
            anno_files.append(os.path.join(root, person_id, action_id, rep, 'skeleton.txt'))
            save_kp_files.append(os.path.join(root, person_id, action_id, rep, '2d_keypoints.txt'))
            save_bbox_files.append(os.path.join(root, person_id, action_id, rep, '2d_bbox.txt'))
            

for i in tqdm(range(len(anno_files))):
    file_skeleton = anno_files[i]

    data = np.loadtxt(file_skeleton, dtype=str)

    keypoints_results = []
    bbox_results = []
   
    for row in data:
        frame_id = row[0]  
        pose_3d = row[1:].astype(float)
        jointLocations = pose_3d.reshape((21, 3))
        

        joints_homogeneous = np.hstack((jointLocations, np.ones((21, 1))))
        joints_tf = (M @ joints_homogeneous.T).T
        u_c = u0_c + fx_c * joints_tf[:, 0] / joints_tf[:, 2]
        v_c = v0_c + fy_c * joints_tf[:, 1] / joints_tf[:, 2]
        

        kp_2d_flat = np.empty(42, dtype=float)
        kp_2d_flat[0::2] = u_c
        kp_2d_flat[1::2] = v_c
        
        kp_str = " ".join([f"{val:.2f}" for val in kp_2d_flat])
        keypoints_results.append(f"{frame_id} {kp_str}\n")
        

        u_min, u_max = np.min(u_c), np.max(u_c)
        v_min, v_max = np.min(v_c), np.max(v_c)
        
        w = u_max - u_min
        h = v_max - v_min
        cx = u_min + w / 2.0
        cy = v_min + h / 2.0
        
        bbox_results.append(f"{frame_id} {cx:.2f} {cy:.2f} {w:.2f} {h:.2f}\n")

    with open(save_kp_files[i], 'w') as f_kp:
        f_kp.writelines(keypoints_results)

    with open(save_bbox_files[i], 'w') as f_bbox:
        f_bbox.writelines(bbox_results)