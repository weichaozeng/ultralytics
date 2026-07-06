# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Column layout for tracker result arrays."""

from __future__ import annotations

import numpy as np

TRACK_BOX_DIM = 4
TRACK_META_DIM = 4
TRACK_META_START = TRACK_BOX_DIM
TRACK_IDX_COL = 7
TRACK_KPT_START = 8
TRACK_RESULT_DIM_BASE = TRACK_BOX_DIM + TRACK_META_DIM


def pose_track_result_dim(n_keypoints: int = 21, kpt_dims: int = 3) -> int:
    """Return total columns for a pose-track result row."""
    return TRACK_RESULT_DIM_BASE + int(n_keypoints) * int(kpt_dims)


def is_pose_track_result(tracks: np.ndarray, *, n_keypoints: int = 21, kpt_dims: int = 3) -> bool:
    """Return True when ``tracks`` includes flattened keypoint columns."""
    if tracks.ndim != 2 or tracks.shape[0] == 0:
        return False
    return tracks.shape[1] >= pose_track_result_dim(n_keypoints, kpt_dims)


def parse_track_idx(tracks: np.ndarray) -> np.ndarray:
    """Return detection-index column used to reorder Results."""
    return tracks[:, TRACK_IDX_COL].astype(int)


def parse_track_boxes(tracks: np.ndarray) -> np.ndarray:
    """Return Ultralytics track box tensor columns: xyxy, track_id, conf, cls."""
    return tracks[:, : TRACK_KPT_START - 1]


def parse_track_keypoints(tracks: np.ndarray, *, n_keypoints: int = 21, kpt_dims: int = 3) -> np.ndarray:
    """Return keypoints array with shape (N, K, C)."""
    width = int(n_keypoints) * int(kpt_dims)
    return tracks[:, TRACK_KPT_START : TRACK_KPT_START + width].reshape(-1, int(n_keypoints), int(kpt_dims))


def apply_pose_tracks_to_result(result, tracks, *, n_keypoints: int = 21, kpt_dims: int = 3):
    """Reorder and update a Results object from pose-track rows."""
    import torch

    if len(tracks) == 0:
        return result
    if is_pose_track_result(tracks, n_keypoints=n_keypoints, kpt_dims=kpt_dims):
        idx = parse_track_idx(tracks)
        tracked = result[idx]
        device = result.boxes.data.device
        tracked.update(
            boxes=torch.as_tensor(parse_track_boxes(tracks), device=device),
            keypoints=torch.as_tensor(parse_track_keypoints(tracks, n_keypoints=n_keypoints, kpt_dims=kpt_dims), device=device),
        )
        return tracked

    idx = tracks[:, TRACK_IDX_COL if tracks.shape[1] > TRACK_IDX_COL else -1].astype(int)
    tracked = result[idx]
    box_cols = TRACK_KPT_START - 1 if tracks.shape[1] >= TRACK_KPT_START else tracks.shape[1] - 1
    tracked.update(boxes=torch.as_tensor(tracks[:, :box_cols], device=result.boxes.data.device))
    return tracked
