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
from ultralytics.utils.metrics import bbox_ioa

POSE_OUTPUT_DETECTOR = "detector"
POSE_OUTPUT_FILTERED = "filtered"
POSE_OUTPUT_MODES = frozenset({POSE_OUTPUT_DETECTOR, POSE_OUTPUT_FILTERED})


def _normalize_pose_output(mode: Any) -> str:
    value = str(mode or POSE_OUTPUT_DETECTOR).strip().lower()
    if value not in POSE_OUTPUT_MODES:
        raise ValueError(f"pose_output must be one of {sorted(POSE_OUTPUT_MODES)}, got {mode!r}")
    return value


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
        self.wrist_rel_to_box = self._encode_wrist_rel(xywh[:4], self.keypoints)
        self.cls_log_odds = 0.0
        self.enable_handedness = False
        self.kpt_conf_ema: np.ndarray | None = None
        self._kpt_conf_thresh = 0.3
        self._cls_learning_rate = 0.3
        self._pose_output = POSE_OUTPUT_DETECTOR
        self._vel_history: list[np.ndarray] = []

    @staticmethod
    def _encode_wrist_rel(xywh: np.ndarray | list[float], keypoints: np.ndarray) -> np.ndarray:
        xywh = np.asarray(xywh[:4], dtype=np.float32)
        scale = float(np.sqrt(xywh[2] ** 2 + xywh[3] ** 2) + 1e-6)
        return (keypoints[0, :2].astype(np.float32) - xywh[:2]) / scale

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
    def detector_keypoints(self) -> np.ndarray:
        """Return the latest matched detection keypoints for export."""
        kpts = np.asarray(self.keypoints, dtype=np.float32)
        if kpts.shape[1] >= 3:
            return kpts[:, :3].copy()
        conf = np.ones(kpts.shape[0], dtype=np.float32)
        return np.concatenate([kpts[:, :2], conf[:, None]], axis=1).astype(np.float32)

    @property
    def filtered_keypoints(self) -> np.ndarray:
        """Return pose-Kalman keypoints for export."""
        xy = self.pred_keypoints_xy
        if self.kpt_conf_ema is None:
            conf = self.keypoints[:, 2] if self.keypoints.shape[1] > 2 else np.ones(len(xy), dtype=np.float32)
        else:
            conf = self.kpt_conf_ema
        return np.concatenate([xy, conf[:, None]], axis=1).astype(np.float32)

    @property
    def output_keypoints(self) -> np.ndarray:
        if getattr(self, "_pose_output", POSE_OUTPUT_DETECTOR) == POSE_OUTPUT_FILTERED:
            return self.filtered_keypoints
        return self.detector_keypoints

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
        self._push_velocity_history()

    def _push_velocity_history(self, max_len: int | None = None):
        if self.mean is None:
            return
        if max_len is None:
            max_len = 4
        self._vel_history.append(self.mean[4:6].astype(np.float32).copy())
        if len(self._vel_history) > max_len:
            self._vel_history = self._vel_history[-max_len:]

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
        self.wrist_rel_to_box = new_track.wrist_rel_to_box.copy()
        self._push_velocity_history()

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
        self.wrist_rel_to_box = new_track.wrist_rel_to_box.copy()
        self._push_velocity_history()

    def mark_lost(self):
        super().mark_lost()

    def _visible_mask(self, kpt_conf_thresh: float) -> np.ndarray:
        if self.keypoints.shape[1] > 2:
            return self.keypoints[:, 2] > kpt_conf_thresh
        return np.ones(self.n_keypoints, dtype=bool)

    def _update_pose(self, new_track: PoseSTrack, kpt_conf_thresh: float | None = None):
        thresh = self._kpt_conf_thresh if kpt_conf_thresh is None else kpt_conf_thresh
        self._anchor = new_track.keypoints[0, :2].astype(np.float32).copy()
        self.wrist_rel_to_box = new_track.wrist_rel_to_box.copy()
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
        self.lost_track_max_frames = int(getattr(args, "lost_track_max_frames", 30))
        self.max_time_lost = self.lost_track_max_frames
        self.enable_handedness = _handedness_enabled(args, class_names)
        self._match_weights = self._resolve_match_weights(args)
        self._oks_sigma = getattr(args, "oks_sigma", 0.05)
        self._kpt_conf_thresh = float(getattr(args, "kpt_conf_thresh", 0.3))
        self._cls_learning_rate = float(getattr(args, "cls_learning_rate", 0.3))
        self._cls_init_strength = float(getattr(args, "cls_init_strength", 2.0))
        self._pose_output = _normalize_pose_output(getattr(args, "pose_output", POSE_OUTPUT_DETECTOR))

    def _apply_track_settings(self, tracks: list[PoseSTrack]) -> None:
        for track in tracks:
            track._kpt_conf_thresh = self._kpt_conf_thresh
            track._cls_learning_rate = self._cls_learning_rate
            track._pose_output = self._pose_output

    def _decorate_detections(self, detections: list[PoseSTrack]) -> list[PoseSTrack]:
        self._apply_track_settings(detections)
        return detections

    def _lost_track_age(self, track: PoseSTrack) -> int:
        return max(int(self.frame_id) - int(track.end_frame), 0)

    def _is_young_lost(self, track: PoseSTrack) -> bool:
        if track.state != TrackState.Lost:
            return False
        recall_frames = int(getattr(self.args, "lost_pose_recall_frames", 3))
        return self._lost_track_age(track) <= recall_frames

    def _motion_confidence(self, track: PoseSTrack) -> float:
        """Higher when speed and velocity direction are both stable."""
        if track.mean is None:
            return 0.0
        speed = float(np.linalg.norm(track.mean[4:6]))
        speed_ref = float(getattr(self.args, "motion_speed_ref", 8.0))
        min_speed = float(getattr(self.args, "motion_speed_min", 1.0))
        if speed < min_speed:
            speed_score = 0.5 * speed / max(min_speed, 1e-6)
        else:
            speed_score = min(speed / max(speed_ref, 1e-6), 1.0)

        dir_score = 0.5
        history = getattr(track, "_vel_history", [])
        if len(history) >= 2:
            v1 = history[-1]
            v2 = history[-2]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 > min_speed and n2 > min_speed:
                cos = float(np.dot(v1, v2) / (n1 * n2 + 1e-6))
                dir_score = (cos + 1.0) * 0.5

        dir_weight = float(getattr(self.args, "motion_dir_weight", 0.5))
        return float(np.clip(speed_score * ((1.0 - dir_weight) + dir_weight * dir_score), 0.0, 1.0))

    def _association_weights(self, track: PoseSTrack, stage: int) -> tuple[float, float]:
        """Interpolate box/pose weights from low-motion to high-motion settings."""
        motion_conf = self._motion_confidence(track)
        if stage == 2:
            box_low = float(getattr(self.args, "box_weight_second", 0.35))
        else:
            box_low = float(getattr(self.args, "box_weight", 0.25))
        box_high = float(getattr(self.args, "motion_box_weight", 0.75))
        box_w = box_low + (box_high - box_low) * motion_conf
        pose_w = 1.0 - box_w
        return box_w, pose_w

    @staticmethod
    def _velocity_shifted_xywh(mean: np.ndarray, steps: int, *, velocity: np.ndarray | None = None) -> np.ndarray:
        xywh = mean[:4].astype(np.float32).copy()
        vel = mean[4:6] if velocity is None else velocity
        xywh[:2] += vel.astype(np.float32) * float(max(steps, 0))
        return xywh

    def _is_lost_expired(self, track: PoseSTrack) -> bool:
        if track.state != TrackState.Lost:
            return False
        return self._lost_track_age(track) > self.lost_track_max_frames

    def _purge_expired_lost_tracks(self) -> list[PoseSTrack]:
        """Remove lost tracks that exceeded the short recall window."""
        kept: list[PoseSTrack] = []
        removed: list[PoseSTrack] = []
        for track in self.lost_stracks:
            if self._is_lost_expired(track):
                track.mark_removed()
                removed.append(track)
            else:
                kept.append(track)
        self.lost_stracks = kept
        return removed

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

    def init_track(
        self,
        results,
        img: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
        det_indices: np.ndarray | None = None,
    ) -> list[PoseSTrack]:
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        if det_indices is None:
            det_indices = np.arange(len(bboxes))
        det_indices = np.asarray(det_indices).reshape(-1)
        bboxes = np.concatenate([bboxes, det_indices.reshape(-1, 1).astype(np.float32)], axis=-1)
        if keypoints is None:
            keypoints = np.zeros((len(bboxes), self.n_keypoints, self.kpt_dims), dtype=np.float32)
        return self._decorate_detections(
            [
                PoseSTrack(xywh, s, c, keypoints[i])
                for i, (xywh, s, c) in enumerate(zip(bboxes, results.conf, results.cls))
            ]
        )

    def _byte_split_scores(self, results) -> np.ndarray:
        """BYTE high/low pools use detector box confidence only (same as BoTSORT)."""
        return np.asarray(results.conf, dtype=np.float32)

    @staticmethod
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        xywh = np.asarray(xywh, dtype=np.float32)
        if xywh.ndim == 1:
            xywh = xywh.reshape(1, 4)
        xyxy = np.zeros_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        return xyxy

    def _pose_dissimilarity(self, tracks: list[PoseSTrack], detections: list[PoseSTrack]) -> np.ndarray:
        d_oks = matching.pose_oks_distance(
            tracks,
            detections,
            sigmas=self._oks_sigma,
            match_weights=self._match_weights,
            use_pred_pose=True,
            conf_thresh=self._kpt_conf_thresh,
            pose_match_thresh=float(getattr(self.args, "pose_match_thresh", 0.0)),
        )
        d_bone = matching.bone_cosine_distance(
            tracks,
            detections,
            use_pred_pose=True,
            conf_thresh=self._kpt_conf_thresh,
        )
        metric = str(getattr(self.args, "pose_metric", "hybrid")).lower()
        if metric == "oks":
            return d_oks
        if metric == "bone":
            return d_bone
        return np.minimum(d_oks, d_bone)

    def _is_redundant_new_detection(self, det: PoseSTrack, active_tracks: list[PoseSTrack]) -> bool:
        """Return True when an unmatched high-score det duplicates an active track.

        Redundancy criteria (either is enough):
        1. Det box is largely contained in an active track box (IoA of det area).
        2. Pose is highly similar to an active track (OKS/bone hybrid dissimilarity).
        """
        if not active_tracks:
            return False
        ioa_thresh = float(getattr(self.args, "new_track_ioa_thresh", 0.65))
        pose_thresh = float(getattr(self.args, "new_track_pose_dissim_thresh", 0.25))

        if ioa_thresh > 0:
            track_xyxy = np.asarray([t.xyxy for t in active_tracks], dtype=np.float32)
            det_xyxy = np.asarray(det.xyxy, dtype=np.float32).reshape(1, 4)
            # bbox_ioa(box1, box2) = inter / box2_area → containment of det inside tracks
            ioa = bbox_ioa(track_xyxy, det_xyxy)
            if float(np.max(ioa)) >= ioa_thresh:
                return True

        if pose_thresh > 0:
            pose_disim = self._pose_dissimilarity(active_tracks, [det]).reshape(-1)
            if pose_disim.size and float(np.min(pose_disim)) < pose_thresh:
                return True
        return False

    def _refine_iou_for_track(self, track: PoseSTrack, iou_row: np.ndarray, det_xyxys: np.ndarray) -> np.ndarray:
        if track.state != TrackState.Lost or track.mean is None:
            return iou_row
        dt = min(self._lost_track_age(track), self.lost_track_max_frames)
        velocity = track.mean[4:6]
        shifted = self._velocity_shifted_xywh(track.mean, dt, velocity=velocity)
        shifted_iou = matching.iou_distance(self._xywh_to_xyxy(shifted), det_xyxys)[0]
        return np.minimum(iou_row, shifted_iou)

    def _resolve_pose_collisions(
        self,
        dists: np.ndarray,
        pose_disim: np.ndarray,
        tracks: list[PoseSTrack],
        *,
        second_thresh: float,
    ) -> np.ndarray:
        if dists.size == 0:
            return dists
        out = dists.copy()
        n_det = dists.shape[1]
        for j in range(n_det):
            candidates = np.where(out[:, j] < second_thresh)[0]
            if len(candidates) <= 1:
                continue
            pose_vals = pose_disim[candidates, j]
            best = float(np.min(pose_vals))
            for k, idx in enumerate(candidates):
                val = float(pose_vals[k])
                own_best = j == int(np.argmin(pose_disim[idx, :]))
                if val == best and val < 0.2:
                    out[idx, j] = max(0.1, out[idx, j] - 0.2)
                elif val > 0.4 and (val > best or not own_best):
                    out[idx, j] = min(0.9, out[idx, j] + 0.2)
        return out

    def _apply_velocity_bonus(
        self,
        dists: np.ndarray,
        tracks: list[PoseSTrack],
        detections: list[PoseSTrack],
        pose_disim: np.ndarray,
        *,
        first_thresh: float,
        second_thresh: float,
    ) -> np.ndarray:
        if dists.size == 0:
            return dists
        out = dists.copy()
        det_xywhs = np.array([d.xywh for d in detections], dtype=np.float32)
        for j in range(out.shape[1]):
            candidates = np.where(out[:, j] < second_thresh)[0]
            if len(candidates) != 1:
                continue
            idx = int(candidates[0])
            track = tracks[idx]
            if track.mean is None:
                continue
            if first_thresh > out[idx, j] or out[idx, j] >= second_thresh:
                continue
            velocity = track.mean[4:6]
            speed = float(np.linalg.norm(velocity))
            min_speed = float(getattr(self.args, "velocity_bonus_min_speed", 2.0))
            if speed <= min_speed:
                continue
            innovation = det_xywhs[j, :2] - track.mean[:2]
            cos_sim = float(np.dot(innovation, velocity) / (np.linalg.norm(innovation) * speed + 1e-6))
            if self._motion_confidence(track) < 0.45:
                continue
            pose_limit = 0.85 if self._is_young_lost(track) else 0.75
            if cos_sim > 0.5 and pose_disim[idx, j] < pose_limit:
                out[idx, j] = out[idx, j] - 0.35 * cos_sim
        return out

    def get_dists(self, tracks: list[PoseSTrack], detections: list[PoseSTrack], stage: int = 1) -> np.ndarray:
        m, n = len(tracks), len(detections)
        dists = np.ones((m, n), dtype=np.float32)
        if m == 0 or n == 0:
            return dists

        det_xywhs = np.array([d.xywh for d in detections], dtype=np.float32)
        det_xyxys = np.array([d.xyxy for d in detections], dtype=np.float32)
        iou_matrix = matching.iou_distance(tracks, detections)
        pose_disim = self._pose_dissimilarity(tracks, detections)

        box_gate = float(getattr(self.args, "box_gate_thresh", 9.488))
        dead_line_scale = float(getattr(self.args, "dead_line_scale", 3.0))
        dead_line_velocity_scale = float(getattr(self.args, "dead_line_velocity_scale", 0.5))
        pose_reliable_thresh = float(getattr(self.args, "pose_reliable_thresh", 0.25))
        motion_pose_recall_max = float(getattr(self.args, "motion_pose_recall_max", 0.45))
        use_gating = bool(getattr(self.args, "use_maha_gating", True))

        for i, track in enumerate(tracks):
            if track.mean is None:
                continue
            alpha, beta = self._association_weights(track, stage)
            iou_row = self._refine_iou_for_track(track, iou_matrix[i], det_xyxys)
            pose_row = pose_disim[i]
            young_lost = self._is_young_lost(track)
            motion_conf = self._motion_confidence(track)
            miss_age = self._lost_track_age(track) if track.state == TrackState.Lost else 0
            speed = float(np.linalg.norm(track.mean[4:6]))
            velocity_pad = dead_line_velocity_scale * speed * max(miss_age, 1)

            if use_gating and track.kalman_filter is not None:
                bbox_maha = track.kalman_filter.gating_distance(
                    track.mean, track.covariance, det_xywhs, metric="maha"
                )
            else:
                bbox_maha = np.full(n, np.inf, dtype=np.float32)

            pixel_dists = np.linalg.norm(det_xywhs[:, :2] - track.mean[:2], axis=1)
            dead_lines = np.minimum(track.mean[3], det_xywhs[:, 3]) * dead_line_scale + velocity_pad

            for j in range(n):
                if pixel_dists[j] > dead_lines[j]:
                    continue
                has_iou = iou_row[j] < 1.0
                in_gate = bbox_maha[j] < box_gate
                pose_reliable = pose_row[j] < pose_reliable_thresh
                if not (has_iou or in_gate or pose_reliable):
                    continue
                allow_pose_only = young_lost and pose_reliable and motion_conf < motion_pose_recall_max
                if track.state == TrackState.Lost and not has_iou and not allow_pose_only:
                    continue
                if track.state == TrackState.Lost:
                    box_score = min(iou_row[j], max(bbox_maha[j] / box_gate, pose_row[j]))
                else:
                    box_score = iou_row[j]
                dists[i, j] = alpha * box_score + beta * pose_row[j]

        proximity = float(getattr(self.args, "proximity_thresh", 0.5))
        far_mask = iou_matrix > (1.0 - proximity)
        still_unmatched = dists >= 1.0 - 1e-5
        dists[np.logical_and(far_mask, still_unmatched)] = 1.0

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        if self.enable_handedness:
            penalty = float(getattr(self.args, "cls_soft_match_penalty", 0.0))
            dists = np.minimum(1.0, dists + matching.cls_soft_match_penalty(tracks, detections, penalty=penalty))

        if stage == 1:
            first_thresh = float(getattr(self.args, "match_thresh", 0.8))
            second_thresh = float(getattr(self.args, "second_match_thresh", 0.9))
            dists = self._resolve_pose_collisions(dists, pose_disim, tracks, second_thresh=second_thresh)
            dists = self._apply_velocity_bonus(
                dists, tracks, detections, pose_disim, first_thresh=first_thresh, second_thresh=second_thresh
            )
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
        self._apply_track_settings(self.tracked_stracks)
        removed_stracks = list(self._purge_expired_lost_tracks())
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []

        scores = self._byte_split_scores(results)
        n_dets = len(scores)
        global_inds = np.arange(n_dets)
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        inds_second = inds_low & inds_high

        results_second = results[inds_second]
        results = results[remain_inds]
        keypoints_keep = keypoints[remain_inds] if keypoints is not None and len(keypoints) else None
        keypoints_second = keypoints[inds_second] if keypoints is not None and len(keypoints) else None

        detections = self.init_track(results, keypoints=keypoints_keep, det_indices=global_inds[remain_inds])
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

        detections_second = self.init_track(
            results_second, keypoints=keypoints_second, det_indices=global_inds[inds_second]
        )
        # Stage-2: tracked + lost, pose+box association (same metric as stage-1, stage-2 weights).
        # Box-only BYTE split is kept; stage-2 stays pose-aware so low-score dets can re-id lost tracks.
        r_strack_pool = [strack_pool[i] for i in u_track]
        dists = self.get_dists(r_strack_pool, detections_second, stage=2)
        second_thresh = float(getattr(self.args, "second_match_thresh", 0.8))
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=second_thresh)
        for itracked, idet in matches:
            track = r_strack_pool[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_strack_pool[it]
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
            # Prefer rejecting duplicate / contained high-score dets over spawning a new ID.
            live_tracks = self.joint_stracks(activated_stracks, refind_stracks)
            if self._is_redundant_new_detection(track, live_tracks):
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
            if self._is_lost_expired(track):
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        if bool(getattr(self.args, "remove_duplicate_stracks", False)):
            self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
                self.tracked_stracks, self.lost_stracks
            )
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

        self._apply_track_settings(self.tracked_stracks)
        expected_dim = pose_track_result_dim(self.n_keypoints, self.kpt_dims)
        rows = [x.result for x in self.tracked_stracks if x.is_activated]
        if not rows:
            return np.zeros((0, expected_dim), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def reset(self):
        super().reset()
        self.pose_kalman_filter = self.get_pose_kalmanfilter()
