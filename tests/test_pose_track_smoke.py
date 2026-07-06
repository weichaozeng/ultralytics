"""Smoke tests for PoseTrack / SPADPoseTrack (run on server with ultralytics env)."""

from __future__ import annotations

import numpy as np
import pytest

from ultralytics.utils import IterableSimpleNamespace
from ultralytics.trackers.pose_track import PoseTrack
from ultralytics.trackers.spad_pose_track import SPADPoseTrack
from ultralytics.trackers.utils.pose_kalman_filter import KalmanFilterPoseChain, fk_abs, abs_to_rel_meas, HAND_PARENT
from ultralytics.trackers.utils.matching import weighted_oks, pose_oks_distance
from ultralytics.trackers.utils.result_layout import pose_track_result_dim, parse_track_keypoints, apply_pose_tracks_to_result


def _tracker_args():
    return IterableSimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        n_keypoints=21,
        kpt_dims=3,
        box_weight=0.4,
        pose_weight=0.6,
        pose_weight_second=0.3,
        proximity_thresh=0.5,
        pose_match_thresh=0.0,
        kpt_conf_thresh=0.3,
        oks_sigma=0.05,
        handedness_filter=True,
        cls_learning_rate=0.3,
        cls_init_strength=2.0,
        cls_soft_match_penalty=0.1,
        nc=2,
    )


def _hand_skeleton_keypoints(cx: float, cy: float, scale: float = 40.0) -> np.ndarray:
    """Build a crude 21-point hand layout around (cx, cy)."""
    kpts = np.zeros((21, 3), dtype=np.float32)
    kpts[0, :2] = (cx, cy)
    # five fingers, four segments each (indices follow HAND_PARENT chain)
    tips = [4, 8, 12, 16, 20]
    bases = [1, 5, 9, 13, 17]
    for b, t, ang in zip(bases, tips, [0.3, 0.8, 1.5, 2.1, 2.7]):
        for j, idx in enumerate(range(b, t + 1)):
            kpts[idx, 0] = cx + scale * (j + 1) * 0.25 * np.cos(ang)
            kpts[idx, 1] = cy + scale * (j + 1) * 0.25 * np.sin(ang)
    kpts[:, 2] = 0.9
    return kpts


class _FakeBoxes:
    """Minimal Boxes-like wrapper for tracker smoke tests."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32)
        if self.xywh.ndim == 1:
            self.xywh = self.xywh[None, :]
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _FakeBoxes(self.xywh[idx], self.conf[idx], self.cls[idx])


def test_pose_kalman_chain_roundtrip():
    kf = KalmanFilterPoseChain(n_keypoints=21)
    kpts = _hand_skeleton_keypoints(256, 256)
    anchor = kpts[0, :2].copy()
    mean, cov = kf.initiate(kpts, anchor)
    mean, cov = kf.predict(mean, cov)
    noisy = kpts.copy()
    noisy[:, :2] += np.random.default_rng(0).normal(0, 1.5, noisy[:, :2].shape)
    mean, cov = kf.update(mean, cov, noisy, anchor, visible_mask=noisy[:, 2] > 0.3)
    abs_xy = kf.rel_to_abs(mean, anchor)
    assert abs_xy.shape == (21, 2)
    rel = abs_to_rel_meas(abs_xy, HAND_PARENT)
    recon = fk_abs(anchor, rel, HAND_PARENT)
    assert np.allclose(recon, abs_xy, atol=1e-4)


def test_pose_track_two_frames_stable_ids():
    args = _tracker_args()
    tracker = PoseTrack(args, frame_rate=25, class_names={0: "left_hand", 1: "right_hand"})
    img = np.zeros((512, 512, 3), np.uint8)

    boxes = _FakeBoxes([[256, 256, 120, 120], [340, 280, 110, 115]], [0.9, 0.85], [1, 0])
    kpts = np.stack([_hand_skeleton_keypoints(256, 256), _hand_skeleton_keypoints(340, 280)])

    t1 = tracker.update(boxes, img, keypoints=kpts)
    assert t1.shape == (2, pose_track_result_dim(21, 3))
    ids1 = t1[:, 4].astype(int)

    boxes2 = _FakeBoxes([[258, 258, 120, 120], [342, 282, 110, 115]], [0.9, 0.85], [0, 1])
    kpts2 = np.stack([_hand_skeleton_keypoints(258, 258), _hand_skeleton_keypoints(342, 282)])
    t2 = tracker.update(boxes2, img, keypoints=kpts2)
    ids2 = t2[:, 4].astype(int)
    assert len(ids1) == len(ids2) == 2
    # IDs should remain stable when motion is small and pose layout is distinct.
    assert set(ids1) == set(ids2)


def test_handedness_belief_resists_single_flip():
    args = _tracker_args()
    tracker = PoseTrack(args, frame_rate=25, class_names={0: "left_hand", 1: "right_hand"})
    img = np.zeros((512, 512, 3), np.uint8)

    # establish right-hand track
    for _ in range(8):
        boxes = _FakeBoxes([[256, 256, 120, 120]], [0.95], [1])
        kpts = _hand_skeleton_keypoints(256, 256)[None]
        tracks = tracker.update(boxes, img, keypoints=kpts)
    assert tracks[0, 6] == 1  # filtered cls stays right

    # one noisy left classification should not flip immediately
    boxes = _FakeBoxes([[258, 258, 120, 120]], [0.95], [0])
    kpts = _hand_skeleton_keypoints(258, 258)[None]
    tracks = tracker.update(boxes, img, keypoints=kpts)
    assert tracks[0, 6] == 1


def test_spad_posetrack_velocity_field():
    args = _tracker_args()
    args.field_value_source = "predicted"
    tracker = SPADPoseTrack(args, frame_rate=25, class_names={0: "left_hand", 1: "right_hand"})
    img = np.zeros((512, 512, 3), np.uint8)
    boxes = _FakeBoxes([[256, 256, 120, 120]], [0.9], [1])
    kpts = _hand_skeleton_keypoints(256, 256)[None]
    tracker.update(boxes, img, keypoints=kpts)
    tracker.update(boxes, img, keypoints=kpts)
    assert tracker.last_velocity_field is not None
    assert tracker.last_velocity_field.shape == (512, 512, 2)


def test_pose_track_uses_global_detection_indices():
    """Track idx must refer to the full detection list, not the high-score subset."""
    args = _tracker_args()
    args.track_high_thresh = 0.5
    tracker = PoseTrack(args, frame_rate=25, class_names={0: "left_hand", 1: "right_hand"})
    img = np.zeros((512, 512, 3), np.uint8)

    # det 0: low score (second stage only), det 1-2: high score
    boxes = _FakeBoxes(
        [[200, 200, 100, 100], [256, 256, 120, 120], [340, 280, 110, 115]],
        [0.2, 0.9, 0.88],
        [1, 1, 0],
    )
    kpts = np.stack(
        [
            _hand_skeleton_keypoints(200, 200),
            _hand_skeleton_keypoints(256, 256),
            _hand_skeleton_keypoints(340, 280),
        ]
    )
    tracks = tracker.update(boxes, img, keypoints=kpts)
    idx = tracks[:, 7].astype(int)
    assert np.all(idx >= 0)
    assert np.max(idx) < 3


def test_result_layout_keypoints_roundtrip():
    args = _tracker_args()
    tracker = PoseTrack(args, frame_rate=25, class_names={0: "left_hand", 1: "right_hand"})
    img = np.zeros((512, 512, 3), np.uint8)
    boxes = _FakeBoxes([[256, 256, 120, 120]], [0.9], [1])
    kpts = _hand_skeleton_keypoints(256, 256)[None]
    tracks = tracker.update(boxes, img, keypoints=kpts)
    parsed = parse_track_keypoints(tracks, n_keypoints=21, kpt_dims=3)
    assert parsed.shape == (1, 21, 3)
    assert parsed[0, 0, 2] > 0
