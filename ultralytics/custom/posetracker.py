
import numpy as np
from typing import Any
from __future__ import annotations


from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT, BOTrack

from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
from ultralytics.custom.kalman_filter_pose import KalmanFilterPose

from ultralytics.trackers.utils import matching



class PTrack(BOTrack):
    
    shared_kalman = KalmanFilterXYWH()
    shared_kalman_pose = KalmanFilterPose()

    def __init__(self, xywh: np.ndarray, score: float, cls: int, pxy: np.ndarray, pscore: np.ndarray, feat: np.ndarray | None = None, feat_history: int = 50
    ):
        super().__init__(xywh, score, cls, feat, feat_history)
    
        M = self.shared_kalman_pose.ndim 
        assert pxy.size == M, f"Expected {M} dimensions for pxy but got {pxy.size}"
        assert pscore.size == M // 2, f"Expected {M // 2} scores but got {pscore.size}"

        self.kps_pos = pxy.flatten()
        self.kps_score = pscore.flatten()

        self.pose_kalman_filter = None
        self.pose_mean, self.pose_covariance = None, None

    def predict(self):
        super().predict()

        if self.pose_mean is not None and self.pose_kalman_filter is not None:
            pose_mean_state = self.pose_mean.copy()
            self.pose_mean, self.pose_covariance = self.pose_kalman_filter.predict(pose_mean_state, self.pose_covariance)

    @staticmethod
    def multi_predict(stracks: list[PTrack]):
        if len(stracks) <= 0:
            return
        # box
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = PTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

        # pose
        multi_pose_mean = np.asarray([st.pose_mean.copy() for st in stracks])
        multi_pose_covariance = np.asarray([st.pose_covariance for st in stracks])

        multi_pose_mean, multi_pose_covariance = PTrack.shared_kalman_pose.multi_predict(multi_pose_mean, multi_pose_covariance)
        
        # 
        for i, (mean, cov, pose_mean, pose_cov) in enumerate(zip(multi_mean, multi_covariance, multi_pose_mean, multi_pose_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov
            stracks[i].pose_mean = pose_mean
            stracks[i].pose_covariance = pose_cov
        
    def activate(self, kalman_filter, pose_kalman_filter, frame_id):
        super().activate(kalman_filter, frame_id)
        self.pose_kalman_filter = pose_kalman_filter
        self.pose_mean, self.pose_covariance = self.pose_kalman_filter.initiate(self.kps_pos)

    def re_activate(self, new_track, frame_id, new_id = False):
        super().re_activate(new_track, frame_id, new_id)
        self.pose_mean, self.pose_covariance = self.pose_kalman_filter.update(
            self.pose_mean, self.pose_covariance, new_track.kps_pos
        )
        self.kps_pos = new_track.kps_pos
        self.kps_score = new_track.kps_score

    def update(self, new_track, frame_id):
        super().update(new_track, frame_id)
        self.pose_mean, self.pose_covariance = self.pose_kalman_filter.update(
            self.pose_mean, self.pose_covariance, new_track.kps_pos
        )
        self.kps_pos = new_track.kps_pos
        self.kps_score = new_track.kps_score
            
    @property
    def predicted_kps_pos(self):
        if self.pose_mean is None:
             return self.kps_pos
        M = self.shared_kalman_pose.ndim
        return self.pose_mean[:M].copy()



class PoseTracker(BOTSORT):
    def __init__(self, args, frame_rate = 30):
        super().__init__(args, frame_rate)
        self.pose_weight = getattr(args, 'pose_weight', 0.5)
        self.pose_kalman_filter = KalmanFilterPose()

        self.bone_thresh = args.bone_thresh
        self.kp_thresh = args.kp_thresh

    def init_track(self, dets, poses, img = None):
        if len(dets) == 0:
            return []
        assert len(dets) == len(poses), f"Length mismatch with det {len(dets)} and pose {len(poses)}"

        bboxes = dets.xywhr if hasattr(dets, "xywhr") else dets.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)

        features_keep = []
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder(img, bboxes)
        
        detections = []
        for i, (xywh, score, cls) in enumerate(zip(bboxes, dets.conf, dets.cls)):
            kps_pos = poses.xy[i].flatten()
            kps_score = poses.conf[i].flatten()
            feat = features_keep[i] if features_keep else None
            track = PTrack(xywh, score, cls, kps_pos, kps_score, feat)
            track.pose_kalman_filter = self.pose_kalman_filter
            track.kalman_filter = self.kalman_filter
            detections.append(track)
        return detections
    
    def get_dists(self, tracks, detections):
        # iou
        dists_iou = matching.iou_distance(tracks, detections)
        dists_iou_mask = dists_iou > (1 - self.proximity_thresh)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists_iou, detections)
        else:
            dists = dists_iou

        # reid
        if self.args.with_reid and self.encoder is not None:
            dists_emb = matching.embedding_distance(tracks, detections) / 2.0
            dists_emb[dists_emb > (1 - self.appearance_thresh)] = 1.0
            dists_emb[dists_iou_mask] = 1.0
            dists = np.minimum(dists, dists_emb)
        
        # bone
        dists_bone = matching.bone_distance(tracks, detections)
        dists_bone[dists_bone < (1 - self.bone_thresh)] = 1.0
        dists_bone[dists_iou_mask] = 1.0
        dists = np.minimum(dists, dists_bone)

        # kp
        dists_kp = matching.kp_distance(tracks, detections)
        dists_kp[dists_kp > (1 - self.kp_thresh)] = 1.0
        dists_kp[dists_iou_mask] = 1.0
        dists = np.minimum(dists, dists_kp)

        return dists
    