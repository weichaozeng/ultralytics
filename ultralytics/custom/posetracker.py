from __future__ import annotations

import numpy as np

from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT, BOTrack

from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
from ultralytics.custom.kalman_filter_pose import KalmanFilterPose, KalmanFilterPose_Polar

from ultralytics.trackers.utils import matching

BONE_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20]
]

class PTrack(BOTrack):
    
    shared_kalman = KalmanFilterXYWH()
    shared_kalman_pose = KalmanFilterPose()
    # shared_kalman_pose = KalmanFilterPose_Polar()
    BONE_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20]
]

    def __init__(self, xywh: np.ndarray, score: float, cls: int, pxy: np.ndarray, pscore: np.ndarray, feat: np.ndarray | None = None, feat_history: int = 50
    ):
        self.obs_scale = np.sqrt(xywh[2]**2 + xywh[3]**2) + 1e-6
        self.wrist_rel_to_box = (pxy[0] - xywh[:2]) / self.obs_scale

        _rel_pose, _rel_pose_score = self._encode_to_relative(pxy, pscore, self.obs_scale)
        super().__init__(xywh, score, cls, feat, feat_history)
        self.pose = _rel_pose            
        self.pose_score = _rel_pose_score 
        self.raw_pixel_kps = pxy.copy()
        self.raw_pixel_kps_score = pscore.copy()

        # handedness 0: left, 1: right
        self.handedness = float(cls)
        self.handedness_weight_sum = 0.5 * score + 0.5 * np.mean(pscore)

        # Inertial Path 
        self.pose_kalman_filter = None
        self.pose_mean, self.pose_covariance = None, None

        # Static Path 
        self.static_mean = None
        self.static_covariance = None
        self.static_pose_mean = None
        self.static_pose_covariance = None
        self.static_wrist_rel_to_box = None

    def _encode_to_relative(self, pxy, pscore, scale):
        rel_pos = []
        rel_score = []
        for p, c in PTrack.BONE_CONNECTIONS:
            rel_pos.append((pxy[c] - pxy[p]) / scale)
            rel_score.append(min(pscore[p], pscore[c]))
        return np.array(rel_pos).flatten(), np.array(rel_score).flatten()

    def mark_lost(self):
        super().mark_lost()
        if self.mean is not None:
            self.static_mean = self.mean.copy()
            self.static_mean[4:] = 0
            self.static_covariance = self.covariance.copy()
        if self.pose_mean is not None:
            self.static_pose_mean = self.pose_mean.copy()
            self.static_pose_mean[20:] = 0
            self.static_pose_covariance = self.pose_covariance.copy()
            self.static_wrist_rel_to_box = self.wrist_rel_to_box.copy()

    def _update_handedness(self, score, pose_score, cls):
        new_w = 0.5 * score + 0.5 * np.mean(pose_score)
        total_w = self.handedness_weight_sum + new_w
        self.handedness = (self.handedness * self.handedness_weight_sum + float(cls) * new_w) / total_w
        self.handedness_weight_sum = total_w
        self.cls = int(self.handedness + 0.5)

    def activate(self, kalman_filter, pose_kalman_filter, frame_id):
        super().activate(kalman_filter, frame_id)
        self.pose_kalman_filter = pose_kalman_filter
        self.pose_mean, self.pose_covariance = self.pose_kalman_filter.initiate(self.pose)

    def re_activate(self, new_det, frame_id, new_id = False):
        new_scale = np.sqrt(new_det.xywh[2]**2 + new_det.xywh[3]**2) + 1e-6
        rel_pose, rel_score = self._encode_to_relative(new_det.raw_pixel_kps, new_det.raw_pixel_kps_score, new_scale)
        
        super().re_activate(new_det, frame_id, new_id)
        self._update_handedness(new_det.score, new_det.raw_pixel_kps_score, new_det.cls)
        
        self.pose_mean, self.pose_covariance = self.pose_kalman_filter.update(
            self.pose_mean, self.pose_covariance, rel_pose, rel_score
        )
        self.pose, self.pose_score = rel_pose, rel_score
        self.raw_pixel_kps = new_det.raw_pixel_kps.copy()
        self.raw_pixel_kps_score = new_det.raw_pixel_kps_score.copy()
        self.wrist_rel_to_box = new_det.wrist_rel_to_box.copy()

        self.static_mean = self.static_pose_mean = self.static_wrist_rel_to_box = None
        self.static_covariance = self.static_pose_covariance = None


    def update(self, new_det, frame_id):
        self.obs_scale = np.sqrt(new_det.xywh[2]**2 + new_det.xywh[3]**2) + 1e-6
        self.wrist_rel_to_box = (new_det.raw_pixel_kps[0] - new_det.xywh[:2]) / self.obs_scale
        
        rel_pose, rel_score = self._encode_to_relative(
            new_det.raw_pixel_kps, 
            new_det.raw_pixel_kps_score,
            self.obs_scale
        )
        
        super().update(new_det, frame_id)
        self._update_handedness(new_det.score, new_det.raw_pixel_kps_score, new_det.cls)
        
        self.pose_mean, self.pose_covariance = self.pose_kalman_filter.update(
            self.pose_mean, self.pose_covariance, rel_pose, rel_score
        )

        self.pose, self.pose_score = rel_pose, rel_score
        self.raw_pixel_kps = new_det.raw_pixel_kps.copy()
        self.raw_pixel_kps_score = new_det.raw_pixel_kps_score.copy()

    @staticmethod
    def multi_predict(stracks: list[PTrack]):
        if len(stracks) <= 0:
            return
        # box
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0 # v_w
                multi_mean[i][7] = 0 # v_h
        multi_mean, multi_covariance = PTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, st in enumerate(stracks):
            stracks[i].mean = multi_mean[i]
            stracks[i].covariance = multi_covariance[i]

        # pose
        pose_tracks = [st for st in stracks if st.pose_mean is not None]
        if len(pose_tracks) > 0:
            multi_pose_mean = np.asarray([st.pose_mean.copy() for st in pose_tracks])
            multi_pose_covariance = np.asarray([st.pose_covariance for st in pose_tracks])

            multi_pose_mean, multi_pose_covariance = PTrack.shared_kalman_pose.multi_predict(multi_pose_mean, multi_pose_covariance)
            
            for i, st in enumerate(pose_tracks):
                    st.pose_mean = multi_pose_mean[i]
                    st.pose_covariance = multi_pose_covariance[i]
                
    @staticmethod
    def multi_gmc(tracks, H: np.ndarray = np.eye(2, 3)):
        if not tracks:
            return
    
        R = H[:2, :2] 
        t = H[:2, 2]  
        # Bbox
        R8x8 = np.kron(np.eye(4, dtype=float), R)
        for st in tracks:
            st.mean = R8x8.dot(st.mean)
            st.mean[:2] += t
            st.covariance = R8x8.dot(st.covariance).dot(R8x8.T)

        # Pose
        pose_tracks = [t for t in tracks if t.pose_mean is not None]
        if pose_tracks:
            M = PTrack.shared_kalman_pose.ndim_obs
            R_rel = np.kron(np.eye(M // 2, dtype=float), R)
            R_pose_total = np.kron(np.eye(2, dtype=float), R_rel)
            for st in pose_tracks:
                st.pose_mean = R_pose_total.dot(st.pose_mean)
                st.pose_covariance = R_pose_total.dot(st.pose_covariance).dot(R_pose_total.T)

    @property
    def pxyxy(self):
        if self.pose_mean is None:
             return self.raw_pixel_kps.flatten()
        curr_xy = self.mean[:2]
        curr_wh = self.mean[2:4]
        curr_scale = np.sqrt(curr_wh[0]**2 + curr_wh[1]**2) + 1e-6
        M = self.shared_kalman_pose.ndim_obs
        rel_vecs = self.pose_mean[:M].reshape(20, 2)
        pixel_kps = np.zeros((21, 2), dtype=np.float32)
        anchor_off = self.static_wrist_rel_to_box if self.state == TrackState.Lost else self.wrist_rel_to_box
        pixel_kps[0] = curr_xy + anchor_off * curr_scale
        for i, (p, c) in enumerate(PTrack.BONE_CONNECTIONS):
            pixel_kps[c] = pixel_kps[p] + rel_vecs[i] * curr_scale
        return pixel_kps.flatten()

    @property
    def result(self):
        coords = self.xyxy if self.angle is None else self.xywha
        kps = self.pxyxy.tolist()
        kps_score = self.raw_pixel_kps_score.tolist()
        output_list = [
            *coords.tolist(),
            int(self.track_id),
            float(self.score),
            int(self.cls),
            float(self.idx),
            *kps,
            *kps_score
        ]
        return output_list


class PoseTracker(BOTSORT):
    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)
        self.pose_kalman_filter = KalmanFilterPose()
        self.box_gate_thresh = getattr(args, 'box_gate_thresh', 9.488) 
        self.first_match_thresh = getattr(args, 'first_match_thresh', 0.62)
        self.second_match_thresh = getattr(args, 'second_match_thresh', 0.62)
        self.unconf_match_thresh = getattr(args, 'unconf_match_thresh', 0.62)

        self.WO_POSE = 0.2
        self.WO_REID = 0.05
        self.WO_IOU = 0.75
        # with interacting in get_dists
        self.W_POSE = 0.6
        self.W_REID = 0.1
        self.W_IOU  = 0.3
    
    def init_track(self, bboxes, scores, clses, poses_xy, poses_conf, img):

        if len(bboxes) == 0:
            return []

        detections = []
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        features_keep = []
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder(img, bboxes)
        
        for i in range(len(bboxes)):   
            # PTrack(xywh, score, cls, pxy, pscore, feat)
            track = PTrack(
                xywh=bboxes[i],
                score=scores[i],
                cls=int(clses[i]),
                pxy=poses_xy[i],
                pscore=poses_conf[i],
                feat=features_keep[i] if features_keep else None
            )
            track.pose_kalman_filter = self.pose_kalman_filter
            track.kalman_filter = self.kalman_filter
            
            detections.append(track)
            
        return detections
    
    def update(self, dets, poses, img, feats):
        self.frame_id += 1
        print(self.frame_id)
        print(f"dets: {len(dets)}")
        if self.frame_id in [450, 451]:
            print("here")
    
        activated_stracks = []  
        refind_stracks = []    
        lost_stracks = []      
        removed_stracks = [] 

        pose_scores = np.mean(poses.conf, axis=1)
        combined_scores = 0.5 * dets.conf + 0.5 * pose_scores

        mask_high = combined_scores >= self.args.track_high_thresh
        mask_second = (combined_scores < self.args.track_high_thresh) & (combined_scores > self.args.track_low_thresh)

        feats_high = feats[mask_high] if feats is not None and len(feats) else img
        feats_second = feats[mask_second] if feats is not None and len(feats) else img
        detections = self.init_track(
            dets.xywh[mask_high], dets.conf[mask_high], dets.cls[mask_high],
            poses.xy[mask_high], poses.conf[mask_high], feats_high
        )
        detections_second = self.init_track(
            dets.xywh[mask_second], dets.conf[mask_second], dets.cls[mask_second],
            poses.xy[mask_second], poses.conf[mask_second], feats_second
        )
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        PTrack.multi_predict(strack_pool)
        if hasattr(self, "gmc") and img is not None:
            warp = self.gmc.apply(img, dets.xyxy[mask_high])
            PTrack.multi_gmc(strack_pool, warp)
            PTrack.multi_gmc(unconfirmed, warp)

        # First Association
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_det = matching.linear_assignment(dists, self.first_match_thresh)
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
        # r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        r_strack_pool = [strack_pool[i] for i in u_track]
        dists_second = self.get_iou_dists(r_strack_pool, detections_second)
        matches_second, u_track_second, u_det_second = matching.linear_assignment(dists_second, self.second_match_thresh)

        for itracked, idet in matches_second:
            track = r_strack_pool[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track_second:
            track = r_strack_pool[it]
            if track.state == TrackState.Tracked:
                track.mark_lost()
                lost_stracks.append(track)

        # Third for unconfirmed tracks and remaining detections
        detections_remaining = [detections[i] for i in u_det]
        dists_unconfirmed = self.get_iou_dists(unconfirmed, detections_remaining)
        matches_unconf, u_unconf, u_det_unconf = matching.linear_assignment(dists_unconfirmed, self.unconf_match_thresh)

        for itracked, idet in matches_unconf:
            unconfirmed[itracked].update(detections_remaining[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconf:
            unconfirmed[it].mark_removed()
            removed_stracks.append(unconfirmed[it])

        for inew in u_det_unconf:
            track = detections_remaining[inew]
            # no need for pose score since new tracks
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.pose_kalman_filter, self.frame_id)
            activated_stracks.append(track)
        
        # Fourth for time out lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, removed_stracks)

        # self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)
        
        # # if visualize
        # all_tracks = self.tracked_stracks + self.lost_stracks
        # res = np.asarray(
        #     [x.result for x in all_tracks if x.is_activated or x.state == TrackState.Lost], 
        #     dtype=np.float32
        # )
        # if self.frame_id == 7:
        #     print(res)
        # print(f"frame id {self.frame_id} has {len(res)} vis tracks.")
        # return res

    def get_iou_dists(self, tracks, detections):
        M, N = len(tracks), len(detections)
        dists = np.ones((M, N), dtype=np.float32)
        
        if M == 0 or N == 0:
            return dists
        
        det_xyxys = np.array([d.xyxy for d in detections], dtype=np.float32)
        for i, track in enumerate(tracks):
            iou_inertial = matching.iou_distance([track], detections)[0]
            if track.state == TrackState.Lost and track.static_mean is not None:
                dt = self.frame_id - track.end_frame
                growth = min(1.0 + 0.02 * dt, 1.4)
                s_xywh = track.static_mean[:4].copy()
                s_xywh[2:4] *= growth
                s_xyxy = self.xywh2xyxy(s_xywh)
                iou_static = matching.iou_distance(s_xyxy, det_xyxys)[0]
                dists[i] = np.minimum(iou_inertial, iou_static)
            else:
                dists[i] = iou_inertial
        return dists


    def get_dists(self, tracks, detections):
        M, N = len(tracks), len(detections)
        dists = np.ones((M, N), dtype=np.float32)
        if M == 0 or N == 0: 
            return dists

        det_xyxys = np.array([d.xyxy for d in detections], dtype=np.float32)        # (N, 4) [cx, cy, w, h]
        det_xywhs = np.array([d.xywh for d in detections], dtype=np.float32)
        det_poses = np.array([d.pose for d in detections], dtype=np.float32)       # (N, 40) 
        det_pose_scores = np.array([d.pose_score for d in detections], dtype=np.float32) # (N, 20)

        iou_matrix = matching.iou_distance(tracks, detections)
        pose_disim_matrix = np.ones((M, N), dtype=np.float32)
        if self.args.with_reid:
            reid_matrix = matching.embedding_distance(tracks, detections) / 2.0
            reid_matrix[reid_matrix > (1 - self.appearance_thresh)] = 1.0
        else:
            reid_matrix = np.ones((M, N), dtype=np.float32)
        for i, track in enumerate(tracks):
            bbox_maha_dists = self.kalman_filter.gating_distance(
                track.mean, track.covariance, det_xywhs, metric='maha'
            )
            pose_sim = self.batch_cosine_similarity(track.pose_mean[:40], det_poses, det_pose_scores)
            pixel_dists = np.linalg.norm(det_xywhs[:, :2] - track.mean[:2], axis=1)
            dead_lines = np.minimum(track.mean[3], det_xywhs[:, 3]) * 3.0

            if track.state == TrackState.Lost and track.static_mean is not None:
                dt = self.frame_id - track.end_frame
                growth = min(1.0 + 0.02 * dt, 1.4)
                s_xywh = track.static_mean[:4].copy()
                s_xywh[2:4] *= growth
                s_xyxy = self.xywh2xyxy(s_xywh)
                static_iou_dists = matching.iou_distance(s_xyxy, det_xyxys)[0]
                iou_dists_refined = np.minimum(iou_matrix[i], static_iou_dists)
                static_pose_sim = self.batch_cosine_similarity(track.static_pose_mean[:40], det_poses, det_pose_scores)
                pose_sim = np.maximum(pose_sim, static_pose_sim)
                static_pixel_dists = np.linalg.norm(det_xywhs[:, :2] - track.static_mean[:2], axis=1)
                static_dead_lines = np.minimum(track.static_mean[3], det_xywhs[:, 3]) * 3.0
            else:
                iou_dists_refined = iou_matrix[i]
                static_pixel_dists = np.ones(det_xywhs[:, 3].shape)
                static_dead_lines = np.zeros(det_xywhs[:, 3].shape)
                
           
            pose_disim = (1.0 - pose_sim) / 2.0
            pose_disim_matrix[i, :] = pose_disim
            
            for j in range(N):
                if pixel_dists[j] > dead_lines[j] and static_pixel_dists[j] > static_dead_lines[j]:
                    continue
                has_iou = iou_dists_refined[j] < 1.0
                in_gate = bbox_maha_dists[j] < self.box_gate_thresh
                pose_reliable = pose_disim[j] < 0.25
                if has_iou or in_gate or pose_reliable:
                    box_score = min(iou_dists_refined[j], max(bbox_maha_dists[j] / self.box_gate_thresh, 0.1))
                    # if track.state == TrackState.Tracked:
                    #     dists[i, j] = box_score * self.WO_IOU + pose_disim[j] * self.WO_POSE + reid_matrix[i, j] * self.WO_REID
                    # else:
                    #     dists[i, j] = box_score * self.W_IOU + pose_disim[j] * self.W_POSE + reid_matrix[i, j] * self.W_REID
                    # modify
                    dist_WO = box_score * self.WO_IOU + pose_disim[j] * self.WO_POSE + reid_matrix[i, j] * self.WO_REID
                    dist_W = box_score * self.W_IOU + pose_disim[j] * self.W_POSE + reid_matrix[i, j] * self.W_REID
                    dists[i, j] = min(dist_WO, dist_W)
            
        for j in range(N):
            potential_matches = np.where(dists[:, j] < 0.62)[0]
            if len(potential_matches) > 1:
                # for idx in potential_matches:
                #     if pose_disim_matrix[idx, j] < 0.15:
                #         dists[idx, j] = pose_disim_matrix[idx, j]
                #     elif pose_disim_matrix[idx, j] > 0.5:
                #         dists[idx, j] = 1.0
                # modify
                current_pose_disims = pose_disim_matrix[potential_matches, j]
                min_pose_val = np.min(current_pose_disims)
                for k, idx in enumerate(potential_matches):
                    this_pose_val = current_pose_disims[k]
                    is_track_own_best = (j == np.argmin(pose_disim_matrix[idx, :]))
                    if this_pose_val == min_pose_val and this_pose_val < 0.2:
                        dists[idx, j] = this_pose_val
                    elif this_pose_val > 0.8:
                        if this_pose_val > min_pose_val and not is_track_own_best:
                            dists[idx, j] = 1.0
                        else:
                            pass
        # if self.frame_id == 54: 
        #     print(dists)
        return dists
    
    def batch_cosine_similarity(self, track_pose, det_poses, det_pose_scores):
        N = det_poses.shape[0]
        if N == 0:
            return np.array([], dtype=np.float32)
        # bone importancy
        finger_decay = np.array([1.0, 0.5, 0.3, 0.1], dtype=np.float32)
        pos_weights = np.tile(finger_decay, 5)

        # length-based gating
        t_v = track_pose.reshape(20, 2)
        d_vs = det_poses.reshape(-1, 20, 2)

        t_v_unit = t_v / (np.linalg.norm(t_v, axis=1, keepdims=True) + 1e-6)
        d_vs_unit = d_vs / (np.linalg.norm(d_vs, axis=2, keepdims=True) + 1e-6)

        d_lens = np.linalg.norm(d_vs, axis=2)  # (N, 20)
        max_lens = np.max(d_lens, axis=1, keepdims=True) + 1e-6 # (N, 1)
        len_weights = np.clip(d_lens / (max_lens * 0.5), 0.1, 1.0)

        cos_matrix = np.einsum('jk,ijk->ij', t_v_unit, d_vs_unit)
        combined_weights = det_pose_scores * pos_weights * len_weights
        weights_sum = np.sum(combined_weights, axis=1, keepdims=True) + 1e-6

        weighted_cos_sim = np.sum(cos_matrix * combined_weights, axis=1, keepdims=True) / weights_sum

        return np.clip(weighted_cos_sim.flatten(), -1.0, 1.0)
    
    @staticmethod
    def xywh2xyxy(xywh):
        xywh = np.asarray(xywh)
        if xywh.ndim == 1:
            xywh = xywh.reshape(1, 4)
            
        xyxy = np.zeros_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2
        return xyxy


# class PoseTracker(BOTSORT):
#     def __init__(self, args, frame_rate = 30):
#         super().__init__(args, frame_rate)
#         self.pose_weight = getattr(args, 'pose_weight', 0.5)
#         self.pose_kalman_filter = KalmanFilterPose()

#         self.bone_thresh = args.bone_thresh
#         self.kp_thresh = args.kp_thresh

#     def init_track(self, dets, poses, img = None):
#         if len(dets) == 0:
#             return []
#         assert len(dets) == len(poses), f"Length mismatch with det {len(dets)} and pose {len(poses)}"

#         bboxes = dets.xywhr if hasattr(dets, "xywhr") else dets.xywh
#         bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)

#         features_keep = []
#         if self.args.with_reid and self.encoder is not None:
#             features_keep = self.encoder(img, bboxes)
        
#         detections = []
#         for i, (xywh, score, cls) in enumerate(zip(bboxes, dets.conf, dets.cls)):
#             kps_pos = poses.xy[i].flatten()
#             kps_score = poses.conf[i].flatten()
#             feat = features_keep[i] if features_keep else None
#             track = PTrack(xywh, score, cls, kps_pos, kps_score, feat)
#             track.pose_kalman_filter = self.pose_kalman_filter
#             track.kalman_filter = self.kalman_filter
#             detections.append(track)
#         return detections
    
#     def get_dists(self, tracks, detections):
#         # iou
#         dists_iou = matching.iou_distance(tracks, detections)
#         dists_iou_mask = dists_iou > (1 - self.proximity_thresh)
#         if self.args.fuse_score:
#             dists = matching.fuse_score(dists_iou, detections)
#         else:
#             dists = dists_iou

#         # reid
#         if self.args.with_reid and self.encoder is not None:
#             dists_emb = matching.embedding_distance(tracks, detections) / 2.0
#             dists_emb[dists_emb > (1 - self.appearance_thresh)] = 1.0
#             dists_emb[dists_iou_mask] = 1.0
#             dists = np.minimum(dists, dists_emb)
        
#         # bone
#         dists_bone = matching.bone_distance(tracks, detections)
#         dists_bone[dists_bone > (1 - self.bone_thresh)] = 1.0
#         dists_bone[dists_iou_mask] = 1.0
#         dists = np.minimum(dists, dists_bone)

#         # kp
#         dists_kp = matching.kp_distance(tracks, detections)
#         dists_kp[dists_kp > (1 - self.kp_thresh)] = 1.0
#         dists_kp[dists_iou_mask] = 1.0
#         dists = np.minimum(dists, dists_kp)

#         return dists
    
#     def update(self, dets, poses, img, feats):
#         self.frame_id += 1
#         activated_stracks = []
#         refind_stracks = []
#         lost_stracks = []
#         removed_stracks = []

#         scores = dets.conf
#         remain_inds = scores >= self.args.track_high_thresh
#         inds_low = scores > self.args.track_low_thresh
#         inds_high = scores < self.args.track_high_thresh

#         inds_second = inds_high & inds_low

#         dets_main = dets[remain_inds]
#         poses_main = poses[remain_inds]
#         feats_main = feats[remain_inds] if feats is not None and len(feats) else img

#         dets_second = dets[inds_second]
#         poses_second = poses[inds_second]
#         feats_second = feats[inds_second] if feats is not None and len(feats) else img

#         detections = self.init_track(dets_main, poses_main, feats_main)
#         detections_second = self.init_track(dets_second, poses_second, feats_second)

#         unconfirmed = []
#         tracked_stracks = [] 
#         for track in self.tracked_stracks:
#             if not track.is_activated:
#                 unconfirmed.append(track)
#             else:
#                 tracked_stracks.append(track)
                
#         strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)

#         self.multi_predict(strack_pool)

#         # GMC
#         if hasattr(self, "gmc") and img is not None:
#             try:
#                 warp = self.gmc.apply(img, dets_main.xyxy) 
#             except Exception:
#                 warp = np.eye(2, 3)
#             PTrack.multi_gmc(strack_pool, warp)
#             PTrack.multi_gmc(unconfirmed, warp)

#         # First Association
#         dists = self.get_dists(strack_pool, detections)
#         matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

#         for itracked, idet in matches:
#             track = strack_pool[itracked]
#             det = detections[idet]
#             if track.state == TrackState.Tracked:
#                 track.update(det, self.frame_id) 
#                 activated_stracks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False) 
#                 refind_stracks.append(track)
        
#         # Second Association
#         r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
#         dists = matching.iou_distance(r_tracked_stracks, detections_second)
#         matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)

#         for itracked, idet in matches:
#             track = r_tracked_stracks[itracked]
#             det = detections_second[idet]
#             if track.state == TrackState.Tracked:
#                 track.update(det, self.frame_id)
#                 activated_stracks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False)
#                 refind_stracks.append(track)

#         for it in u_track:
#             track = r_tracked_stracks[it]
#             if track.state != TrackState.Lost:
#                 track.mark_lost()
#                 lost_stracks.append(track)
        
#         # Unconfirmed
#         detections_remaining = [detections[i] for i in u_detection]
#         dists = self.get_dists(unconfirmed, detections_remaining)
#         matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

#         for itracked, idet in matches:
#             unconfirmed[itracked].update(detections_remaining[idet], self.frame_id)
#             activated_stracks.append(unconfirmed[itracked])
            
#         for it in u_unconfirmed:
#             track = unconfirmed[it]
#             track.mark_removed()
#             removed_stracks.append(track)

#         # Init new stracks
#         for inew in u_detection:
#             track = detections_remaining[inew]
#             if track.score < self.args.new_track_thresh:
#                 continue
#             track.activate(self.kalman_filter, self.pose_kalman_filter, self.frame_id)
#             activated_stracks.append(track)


#         # Update state lists
#         for track in self.lost_stracks:
#             if self.frame_id - track.end_frame > self.max_time_lost:
#                 track.mark_removed()
#                 removed_stracks.append(track)

#         self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
#         self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
#         self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
#         self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
#         self.lost_stracks.extend(lost_stracks)
#         self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
#         # self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
#         self.removed_stracks.extend(removed_stracks)
#         if len(self.removed_stracks) > 1000:
#             self.removed_stracks = self.removed_stracks[-999:]
        
#         return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    
#     def multi_predict(self, tracks: list[PTrack]):
#         PTrack.multi_predict(tracks)