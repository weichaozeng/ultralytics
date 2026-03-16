import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 定义常量与骨架结构
# ==========================================
BONE_CONNECTIONS = torch.tensor([
    [0, 1], [1, 2], [2, 3], [3, 4],        # Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],        # Index
    [0, 9], [9, 10], [10, 11], [11, 12],   # Middle
    [0, 13], [13, 14], [14, 15], [15, 16], # Ring
    [0, 17], [17, 18], [18, 19], [19, 20]  # Pinky
], dtype=torch.int64)

parent_idx = BONE_CONNECTIONS[:, 0]
child_idx = BONE_CONNECTIONS[:, 1]

FINGER_COLORS = ['m', 'r', 'b', 'g', 'k'] # 依次为: 拇指(洋红), 食指(红), 中指(蓝), 无名指(绿), 小指(黑)

# ==========================================
# 2. 生成模拟的 3D 手部姿态
# ==========================================
def generate_synthetic_poses(num_poses=5):
    poses = []
    
    # 基础骨骼向量 (近似平展状态)
    base_vecs = torch.zeros((20, 3))
    base_vecs[0:4]   = torch.tensor([[2, 2, 0], [1.5, 1.5, 0], [1, 1, 0], [1, 1, 0]])     # Thumb
    base_vecs[4:8]   = torch.tensor([[1, 4, 0], [0, 3, 0], [0, 2, 0], [0, 1.5, 0]])       # Index
    base_vecs[8:12]  = torch.tensor([[0, 4.5, 0], [0, 3.5, 0], [0, 2.5, 0], [0, 1.5, 0]]) # Middle
    base_vecs[12:16] = torch.tensor([[-1, 4, 0], [0, 3, 0], [0, 2, 0], [0, 1.5, 0]])      # Ring
    base_vecs[16:20] = torch.tensor([[-2, 3.5, 0], [0, 2.5, 0], [0, 1.5, 0], [0, 1.5, 0]])# Pinky
    
    for i in range(num_poses):
        bend_factor = i * (np.pi / 6) # 每次弯曲 30 度
        
        current_vecs = base_vecs.clone()
        for j in range(20):
            # 仅对指间关节进行显著弯曲 (绕 X 轴旋转，模拟握拳)
            if j not in [0, 4, 8, 12, 16]: # 排除掌骨，主要弯曲指骨
                # 越靠指尖，累计弯曲角度越大
                mult = 1.0 if j in [1, 5, 9, 13, 17] else (2.0 if j in [2, 6, 10, 14, 18] else 3.0)
                angle = bend_factor * mult
                c, s = np.cos(angle), np.sin(angle)
                
                # 绕局部 X 轴旋转矩阵
                Rot_x = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
                current_vecs[j] = current_vecs[j] @ Rot_x.T

        # 利用前向运动学，从向量重建 21 个 3D 关键点坐标
        kps = torch.zeros((21, 3))
        for b in range(20):
            p, c_idx = BONE_CONNECTIONS[b, 0], BONE_CONNECTIONS[b, 1]
            kps[c_idx] = kps[p] + current_vecs[b]
        
        poses.append(kps)
        
    return torch.stack(poses) # Shape: (5, 21, 3)

# ==========================================
# 3. 核心计算：每只手仅与初始手 (第0帧) 进行比较
# ==========================================
def calculate_dissim(kps):
    B = kps.shape[0]
    box_scales = torch.ones((B, 1, 1)) 
    
    # 1. 提取相对关节向量并归一化
    rel_vecs = (kps[:, child_idx, :] - kps[:, parent_idx, :]) / box_scales.clamp(min=1e-6)
    bone_vecs_norm = F.normalize(rel_vecs, p=2, dim=2) # (B, 20, 3)
    
    # 2. 获取初始手 (Anchor Pose) 的特征，即第 0 帧
    anchor_vecs_norm = bone_vecs_norm[0:1] # Shape: (1, 20, 3)
    
    # 3. 计算 Cosine Similarity (利用广播机制，B只手全部与这 1 只初始手相乘)
    cos_sim = torch.sum(anchor_vecs_norm * bone_vecs_norm, dim=2) # Shape: (B, 20)
    
    # 4. 计算 Dissimilarity
    bone_dissim = 1.0 - (cos_sim + 1.0) / 2.0 # (B, 20)
    pose_dissim = torch.mean(bone_dissim, dim=1) # 对每只手的20根骨头取平均 -> (B,)
    
    return pose_dissim, bone_dissim

# ==========================================
# 4. 3D 可视化绘制 (加入初始手残影对比)
# ==========================================
def plot_poses_vertically(kps_tensor, dissim_scores):
    B = kps_tensor.shape[0]
    fig = plt.figure(figsize=(4, 3 * B))
    
    # 提取初始手的坐标，用于画“背景残影”
    anchor_pose = kps_tensor[0].numpy() 
    
    for i in range(B):
        ax = fig.add_subplot(B, 1, i + 1, projection='3d')
        current_pose = kps_tensor[i].numpy()
        
        # 遍历每根骨头进行绘制
        for bone_idx, (p_idx, c_idx) in enumerate(BONE_CONNECTIONS.numpy()):
            # --- 新增：画出初始手(第0帧)的灰色半透明残影作为参考底图 ---
            ax.plot([anchor_pose[p_idx, 0], anchor_pose[c_idx, 0]],
                    [anchor_pose[p_idx, 1], anchor_pose[c_idx, 1]],
                    [anchor_pose[p_idx, 2], anchor_pose[c_idx, 2]],
                    c='gray', alpha=0.3, linewidth=1.5, linestyle='--')
            
            # --- 画出当前帧的手部姿态 ---
            finger_group = bone_idx // 4
            color = FINGER_COLORS[finger_group]
            ax.plot([current_pose[p_idx, 0], current_pose[c_idx, 0]],
                    [current_pose[p_idx, 1], current_pose[c_idx, 1]],
                    [current_pose[p_idx, 2], current_pose[c_idx, 2]],
                    c=color, linewidth=2.5)
            
        # 统一坐标轴范围以方便对比
        ax.set_xlim([-10, 10])
        ax.set_ylim([-5, 15])
        ax.set_zlim([-10, 10])
        ax.view_init(elev=20, azim=-60)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        title_color = 'green' if i == 0 else 'red'
        ax.set_title(f"Pose {i} | Dis_sim vs Initial: {dissim_scores[i]:.4f}", color=title_color, fontsize=12)

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    kps = generate_synthetic_poses(5)
    pose_dissim, _ = calculate_dissim(kps)
    
    print("各姿态与初始手 (Pose 0) 的结构区分度:")
    for i, score in enumerate(pose_dissim):
        print(f"Pose {i} vs Pose 0: {score.item():.4f}")
        
    plot_poses_vertically(kps, pose_dissim)