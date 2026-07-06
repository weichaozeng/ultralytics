# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Pose-aware ByteTrack-style tracker with parent-relative pose Kalman filtering."""

from __future__ import annotations

from typing import Any

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYWH
from .utils.pose_kalman_filter import HAND_MATCH_WEIGHTS, KalmanFilterPoseChain
from .utils.result_layout import pose_track_result_dim


def _handedness_enabled(args: Any, class_names: dict | list | None = None) -> bool:
    if not bool(getattr(args, "handedness_filter", False)):
        return False
    names = class_names
    if names is None:
        names = getattr(args, "class_names", None)
    if isinstance(names, dict):
        values = {str(v).lower() for v in names.values()}
    elif isinstance(names, (list, tuple)):
        values = {str(v).lower() for v in names}
    else:
        return int(getattr(args, "nc", 0) or 0) == 2
    return {"left_hand", "right_hand"}.issubset(values) or int(getattr(args, "nc", 0) or 0) == 2


class PoseSTrack(STrack):
    """Single track with box and parent-relative pose Kalman states."""

    shared_pose_kalman: KalmanFilterPoseChain | None = None

    def __init__(self, xywh: list[float], score: float, cls: Any, keypoints: np.ndarray):
        super().__init__(xywh, score, cls)
        self.keypoints = np.asarray(keypoints, dtype=np.float32)
        self.n_keypoints = int(self.keypoints.shape[0])
        self.pose_kalman_filter: KalmanFilterPoseChain | None = None
        self.pose_mean: np.ndarray | None = None
        self.pose_covariance: np.ndarray | None = None
        self._anchor = self.keypoints[0, :2].astype(np.float32).copy()
        self.cls_log_odds = 0.0
        self.enable_handedness = False
        self.kpt_conf_ema: np.ndarray | None = None
        self._kpt_conf_thresh = 0.3
        self._cls_learning_rate = 0.3

    @classmethod
    def _shared_pose_kalman(cls, n_keypoints: int) -> KalmanFilterPoseChain:
        if cls.shared_pose_kalman is None or cls.shared_pose_kalman.n_keypoints != n_keypoints:
            cls.shared_pose_kalman = KalmanFilterPoseChain(n_keypoints=n_keypoints)
        return cls.shared_pose_kalman

    @property
    def anchor(self) -> np.ndarray:
        """Wrist anchor used for pose forward kinematics."""
        return self._anchor.astype(np.float32)

    @property
    def pred_keypoints_xy(self) -> np.ndarray:
        if self.pose_mean is None or self.pose_kalman_filter is None:
            return self.keypoints[:, :2].astype(np.float32)
        return self.pose_kalman_filter.rel_to_abs(self.pose_mean, self.anchor)

    @property
    def output_keypoints(self) -> np.ndarray:
        xy = self.pred_keypoints_xy
        if self.kpt_conf_ema is None:
            conf = self.keypoints[:, 2] if self.keypoints.shape[1] > 2 else np.ones(len(xy), dtype=np.float32)
        else:
            conf = self.kpt_conf_ema
        return np.concatenate([xy, conf[:, None]], axis=1).astype(np.float32)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

        if self.pose_mean is not None and self.pose_kalman_filter is not None:
            pose_mean = self.pose_mean.copy()
            if self.state != TrackState.Tracked:
                for i in range(self.pose_kalman_filter.n_rel):
                    pose_mean[4 * i + 2] = 0.0
                    pose_mean[4 * i + 3] = 0.0
            self.pose_mean, self.pose_covariance = self.pose_kalman_filter.predict(pose_mean, self.pose_covariance)

    @staticmethod
    def multi_predict(stracks: list[PoseSTrack]):
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
            elif st.pose_mean is not None and st.mean is not None:
                st._anchor = st._anchor + np.array([st.mean[4], st.mean[5]], dtype=np.float32)
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

        pose_tracks = [st for st in stracks if st.pose_mean is not None and st.pose_kalman_filter is not None]
        if not pose_tracks:
            return
        pose_kf = pose_tracks[0].pose_kalman_filter
        pose_mean = np.asarray([st.pose_mean.copy() for st in pose_tracks])
        pose_cov = np.asarray([st.pose_covariance for st in pose_tracks])
        for i, st in enumerate(pose_tracks):
            if st.state != TrackState.Tracked:
                for j in range(pose_kf.n_rel):
                    pose_mean[i][4 * j + 2] = 0.0
                    pose_mean[i][4 * j + 3] = 0.0
        pose_mean, pose_cov = pose_kf.multi_predict(pose_mean, pose_cov)
        for st, mean, cov in zip(pose_tracks, pose_mean, pose_cov):
            st.pose_mean = mean
            st.pose_covariance = cov

    def activate(
        self,
        kalman_filter: KalmanFilterXYWH,
        pose_kalman_filter: KalmanFilterPoseChain,
        frame_id: int,
        *,
        enable_handedness: bool = False,
        cls_init_strength: float = 2.0,
        kpt_conf_thresh: float = 0.3,
    ):
        self.kalman_filter = kalman_filter
        self.pose_kalman_filter = pose_kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))
        self._anchor = self.keypoints[0, :2].astype(np.float32).copy()
        self.pose_mean, self.pose_covariance = pose_kalman_filter.initiate(self.keypoints, self._anchor)
        self.enable_handedness = enable_handedness
        self._kpt_conf_thresh = float(kpt_conf_thresh)
        self._init_cls_belief(cls_init_strength=cls_init_strength, kpt_conf_thresh=kpt_conf_thresh)
        self._update_kpt_conf_ema(alpha=1.0, kpt_conf_thresh=kpt_conf_thresh)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: PoseSTrack, frame_id: int, new_id: bool = False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self._update_pose(new_track)
        self._update_cls_belief_from_det(new_track)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.idx = new_track.idx
        self.keypoints = new_track.keypoints.copy()

    def update(self, new_track: PoseSTrack, frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self._update_pose(new_track)
        self._update_cls_belief_from_det(new_track)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.idx = new_track.idx
        self.keypoints = new_track.keypoints.copy()

    def _visible_mask(self, kpt_conf_thresh: float) -> np.ndarray:
        if self.keypoints.shape[1] > 2:
            return self.keypoints[:, 2] > kpt_conf_thresh
        return np.ones(self.n_keypoints, dtype=bool)

    def _update_pose(self, new_track: PoseSTrack, kpt_conf_thresh: float | None = None):
        thresh = self._kpt_conf_thresh if kpt_conf_thresh is None else kpt_conf_thresh
        self._anchor = new_track.keypoints[0, :2].astype(np.float32).copy()
        if self.pose_kalman_filter is None:
            return
        visible = new_track._visible_mask(thresh)
        self.pose_mean, self.pose_covariance = self.pose_kalman_filter.update(
            self.pose_mean,
            self.pose_covariance,
            new_track.keypoints,
            self._anchor,
            visible_mask=visible,
        )
        self._update_kpt_conf_ema(alpha=0.8, kpt_conf_thresh=thresh, source=new_track.keypoints)

    def _update_kpt_conf_ema(
        self,
        alpha: float,
        kpt_conf_thresh: float,
        source: np.ndarray | None = None,
    ):
        source = self.keypoints if source is None else source
        if source.shape[1] < 3:
            return
        conf = source[:, 2].astype(np.float32)
        if self.kpt_conf_ema is None:
            self.kpt_conf_ema = conf.copy()
        else:
            self.kpt_conf_ema = alpha * self.kpt_conf_ema + (1.0 - alpha) * conf
        self.kpt_conf_ema = np.where(conf > kpt_conf_thresh, self.kpt_conf_ema, self.kpt_conf_ema * alpha)

    def _init_cls_belief(self, cls_init_strength: float, kpt_conf_thresh: float):
        if not self.enable_handedness:
            return
        w = self._frame_reliability(kpt_conf_thresh)
        obs_sign = 1.0 if int(self.cls) == 1 else -1.0
        self.cls_log_odds = float(cls_init_strength * w * obs_sign)
        self.cls = 1 if self.cls_log_odds > 0 else 0

    def _update_cls_belief_from_det(self, det: PoseSTrack, kpt_conf_thresh: float | None = None, lr: float | None = None):
        if not self.enable_handedness:
            self.cls = det.cls
            return
        thresh = self._kpt_conf_thresh if kpt_conf_thresh is None else kpt_conf_thresh
        learning_rate = self._cls_learning_rate if lr is None else lr
        w = det._frame_reliability(thresh)
        obs_sign = 1.0 if int(det.cls) == 1 else -1.0
        self.cls_log_odds += float(learning_rate * w * obs_sign)
        self.cls = 1 if self.cls_log_odds > 0 else 0

    def _frame_reliability(self, kpt_conf_thresh: float) -> float:
        w_box = float(self.score)
        visible = self._visible_mask(kpt_conf_thresh)
        if visible.any() and self.keypoints.shape[1] > 2:
            w_kpt = float(self.keypoints[visible, 2].mean())
        else:
            w_kpt = 0.0
        return w_box * w_kpt

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def result(self) -> list[float]:
        kpts = self.output_keypoints.reshape(-1)
        return [
            *self.xyxy.tolist(),
            float(self.track_id),
            float(self.score),
            float(self.cls),
            float(self.idx),
            *kpts.tolist(),
        ]


class PoseTrack(BYTETracker):
    """ByteTrack-style pose tracker with parent-relative OKS association."""

    def __init__(self, args: Any, frame_rate: int = 30, class_names: dict | list | None = None):
        self.n_keypoints = int(getattr(args, "n_keypoints", 21))
        self.kpt_dims = int(getattr(args, "kpt_dims", 3))
        PoseSTrack._shared_pose_kalman(self.n_keypoints)
        STrack.shared_kalman = KalmanFilterXYWH()
        super().__init__(args, frame_rate)
        self.pose_kalman_filter = self.get_pose_kalmanfilter()
        self.enable_handedness = _handedness_enabled(args, class_names)
        self._match_weights = self._resolve_match_weights(args)
        self._oks_sigma = getattr(args, "oks_sigma", 0.05)
        self._kpt_conf_thresh = float(getattr(args, "kpt_conf_thresh", 0.3))
        self._cls_learning_rate = float(getattr(args, "cls_learning_rate", 0.3))
        self._cls_init_strength = float(getattr(args, "cls_init_strength", 2.0))

    def _decorate_detections(self, detections: list[PoseSTrack]) -> list[PoseSTrack]:
        for det in detections:
            det._kpt_conf_thresh = self._kpt_conf_thresh
            det._cls_learning_rate = self._cls_learning_rate
        return detections

    def get_kalmanfilter(self) -> KalmanFilterXYWH:
        return KalmanFilterXYWH()

    def get_pose_kalmanfilter(self) -> KalmanFilterPoseChain:
        return KalmanFilterPoseChain(n_keypoints=self.n_keypoints)

    @staticmethod
    def _resolve_match_weights(args: Any) -> np.ndarray:
        weights = getattr(args, "match_weight_depth", None)
        if weights is None:
            return HAND_MATCH_WEIGHTS
        depth_to_weight = {i: float(w) for i, w in enumerate(weights)}
        from .utils.pose_kalman_filter import HAND_CHAIN_DEPTH

        return np.array([depth_to_weight[int(HAND_CHAIN_DEPTH[k])] for k in range(len(HAND_CHAIN_DEPTH))], dtype=np.float32)

    def init_track(self, results, img: np.ndarray | None = None, keypoints: np.ndarray | None = None) -> list[PoseSTrack]:
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        if keypoints is None:
            keypoints = np.zeros((len(bboxes), self.n_keypoints, self.kpt_dims), dtype=np.float32)
        return self._decorate_detections(
            [
                PoseSTrack(xywh, s, c, keypoints[i])
                for i, (xywh, s, c) in enumerate(zip(bboxes, results.conf, results.cls))
            ]
        )

    def get_dists(self, tracks: list[PoseSTrack], detections: list[PoseSTrack], stage: int = 1) -> np.ndarray:
        alpha = float(getattr(self.args, "box_weight", 0.4))
        beta = float(getattr(self.args, "pose_weight_second" if stage == 2 else "pose_weight", 0.6))
        total = max(alpha + beta, 1e-6)
        alpha, beta = alpha / total, beta / total

        d_box = matching.iou_distance(tracks, detections)
        proximity = float(getattr(self.args, "proximity_thresh", 0.5))
        d_box_mask = d_box > (1.0 - proximity)
        d_box[d_box_mask] = 1.0

        d_pose = matching.pose_oks_distance(
            tracks,
            detections,
            sigmas=self._oks_sigma,
            match_weights=self._match_weights,
            use_pred_pose=True,
            conf_thresh=self._kpt_conf_thresh,
            pose_match_thresh=float(getattr(self.args, "pose_match_thresh", 0.0)),
        )
        d_pose[d_box_mask] = 1.0

        dists = alpha * d_box + beta * d_pose
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        if self.enable_handedness:
            penalty = float(getattr(self.args, "cls_soft_match_penalty", 0.0))
            dists = np.minimum(1.0, dists + matching.cls_soft_match_penalty(tracks, detections, penalty=penalty))
        return dists

    def multi_predict(self, tracks: list[PoseSTrack]):
        PoseSTrack.multi_predict(tracks)

    def update(
        self,
        results,
        img: np.ndarray | None = None,
        feats: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update tracker with detections and optional pose keypoints shaped (N, K, C)."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        inds_second = inds_low & inds_high

        results_second = results[inds_second]
        results = results[remain_inds]
        keypoints_keep = keypoints[remain_inds] if keypoints is not None and len(keypoints) else None
        keypoints_second = keypoints[inds_second] if keypoints is not None and len(keypoints) else None

        detections = self.init_track(results, keypoints=keypoints_keep)
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)

        dists = self.get_dists(strack_pool, detections, stage=1)
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

        detections_second = self.init_track(results_second, keypoints=keypoints_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = self.get_dists(r_tracked_stracks, detections_second, stage=2)
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

        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections, stage=1)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            unconfirmed[it].mark_removed()
            removed_stracks.append(unconfirmed[it])

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(
                self.kalman_filter,
                self.pose_kalman_filter,
                self.frame_id,
                enable_handedness=self.enable_handedness,
                cls_init_strength=self._cls_init_strength,
                kpt_conf_thresh=self._kpt_conf_thresh,
            )
            activated_stracks.append(track)

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

        expected_dim = pose_track_result_dim(self.n_keypoints, self.kpt_dims)
        rows = [x.result for x in self.tracked_stracks if x.is_activated]
        if not rows:
            return np.zeros((0, expected_dim), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def reset(self):
        super().reset()
        self.pose_kalman_filter = self.get_pose_kalmanfilter()
