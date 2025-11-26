
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
            
    @staticmethod
    def multi_gmc(tracks, H: np.ndarray = np.eye(2, 3)):
        if tracks:
            multi_bbox_mean = np.asarray([t.mean.copy() for t in tracks])
            multi_bbox_cov = np.asarray([t.covariance for t in tracks])
            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_bbox_mean, multi_bbox_cov)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())      
                tracks[i].mean = mean
                tracks[i].covariance = cov

            kf_pose = tracks[0].pose_kalman_filter 
            M = kf_pose.ndim # 42
            D = 2 * M
            pose_tracks = [t for t in tracks if t.pose_mean is not None]
            if not pose_tracks:
                return

            pose_means = np.asarray([t.pose_mean.copy() for t in pose_tracks])
            pose_covs = np.asarray([t.pose_covariance for t in pose_tracks])

            R_kps = np.kron(np.eye(M // 2, dtype=float), R)
            R_pose_block = np.kron(np.eye(2, dtype=float), R_kps)
            for i, (mean, cov) in enumerate(zip(pose_means, pose_covs)):
                mean = R_pose_block.dot(mean)
                t_pose = np.tile(t, M // 2) # (42,)
                mean[:M] += t_pose
                cov = R_pose_block.dot(cov).dot(R_pose_block.transpose())
                pose_tracks[i].pose_mean = mean
                pose_tracks[i].pose_covariance = cov
    @property
    def pxyxy(self):
        if self.pose_mean is None:
             return self.kps_pos
        M = self.shared_kalman_pose.ndim
        return self.pose_mean[:M].copy()
    
    @property
    def result(self):
        coords = self.xyxy if self.angle is None else self.xywha
        kps = self.pxyxy.tolist()
        kps_score = self.kps_score.tolist()
        output_list = [
            *coords.tolist(),
            self.track_id,
            self.score,
            self.cls,
            self.idx,
            *kps,
            *kps_score
        ]
        return output_list



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
    
    def update(self, dets, poses, img, feats):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = dets.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_high & inds_low

        dets_main = dets[remain_inds]
        poses_main = poses[remain_inds]
        feats_main = feats[remain_inds] if feats is not None else None

        dets_second = dets[inds_second]
        poses_second = poses[inds_second]
        feats_second = feats[inds_second] if feats is not None else None

        detections = self.init_track(dets_main, poses_main, img, feats_main)
        detections_second = self.init_track(dets_second, poses_second, img, feats_second)

        unconfirmed = []
        tracked_stracks = [] 
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

        self.multi_predict(strack_pool)

        # GMC
        if hasattr(self, "gmc") and img is not None:
            try:
                warp = self.gmc.apply(img, dets_main.xyxy) 
            except Exception:
                warp = np.eye(2, 3)
            PTrack.multi_gmc(strack_pool, warp)
            PTrack.multi_gmc(unconfirmed, warp)

        # First Association
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id) 
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False) 
                refind_stracks.append(track)
        
        # Second Association
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        # Unconfirmed
        detections_remaining = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections_remaining)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_remaining[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
            
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Init new stracks
        for inew in u_detection:
            track = detections_remaining[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.pose_kalman_filter, self.frame_id)
            activated_stracks.append(track)


        # Update state lists
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]
        
        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    
    def multi_predict(self, tracks: list[PTrack]):
        PTrack.multi_predict(tracks)