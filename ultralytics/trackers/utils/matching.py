# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lap>=0.5.12")  # https://github.com/gatagat/lap
    import lap


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True):
    """Perform linear assignment using either the scipy or lap.lapjv method.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        thresh (float): Threshold for considering an assignment valid.
        use_lap (bool): Use lap.lapjv for the assignment. If False, scipy.optimize.linear_sum_assignment is used.

    Returns:
        matched_indices (np.ndarray): Array of matched indices of shape (K, 2), where K is the number of matches.
        unmatched_a (np.ndarray): Array of unmatched indices from the first set, with shape (L,).
        unmatched_b (np.ndarray): Array of unmatched indices from the second set, with shape (M,).

    Examples:
        >>> cost_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> thresh = 5.0
        >>> matched_indices, unmatched_a, unmatched_b = linear_assignment(cost_matrix, thresh, use_lap=True)
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # Use lap.lapjv
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # Use scipy.optimize.linear_sum_assignment
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU with shape (len(atracks), len(btracks)).

    Examples:
        Compute IoU distance between two sets of tracks
        >>> atracks = [np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])]
        >>> btracks = [np.array([5, 5, 15, 15]), np.array([25, 25, 35, 35])]
        >>> cost_matrix = iou_distance(atracks, btracks)
    """
    if (atracks and isinstance(atracks[0], np.ndarray)) or (btracks and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
            ).numpy()
        else:
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou=True,
            )
    return 1 - ious  # cost matrix


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:
    """Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (list[STrack]): List of tracks, where each track contains embedding features.
        detections (list[BaseTrack]): List of detections, where each detection contains embedding features.
        metric (str): Metric for distance computation. Supported metrics include 'cosine', 'euclidean', etc.

    Returns:
        (np.ndarray): Cost matrix computed based on embeddings with shape (N, M), where N is the number of tracks and M
            is the number of detections.

    Examples:
        Compute the embedding distance between tracks and detections using cosine metric
        >>> tracks = [STrack(...), STrack(...)]  # List of track objects with embedding features
        >>> detections = [BaseTrack(...), BaseTrack(...)]  # List of detection objects with embedding features
        >>> cost_matrix = embedding_distance(tracks, detections, metric="cosine")
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def _track_keypoints_xy(track, *, use_pred_pose: bool) -> np.ndarray:
    """Return absolute xy keypoints for a track or detection object."""
    if use_pred_pose and hasattr(track, "pred_keypoints_xy"):
        return np.asarray(track.pred_keypoints_xy, dtype=np.float32)
    if hasattr(track, "keypoints"):
        return np.asarray(track.keypoints[:, :2], dtype=np.float32)
    raise AttributeError("Track object does not expose keypoints for pose matching.")


def _track_keypoints_conf(track) -> np.ndarray:
    if hasattr(track, "keypoints") and track.keypoints.shape[1] > 2:
        return np.asarray(track.keypoints[:, 2], dtype=np.float32)
    return np.ones((_track_keypoints_xy(track, use_pred_pose=False).shape[0],), dtype=np.float32)


def weighted_oks(
    pred_xy: np.ndarray,
    det_xy: np.ndarray,
    det_conf: np.ndarray,
    bbox_xyxy: np.ndarray,
    *,
    sigmas: float | np.ndarray = 0.05,
    match_weights: np.ndarray | None = None,
    conf_thresh: float = 0.0,
) -> float:
    """Compute depth-weighted Object Keypoint Similarity between two poses."""
    n = pred_xy.shape[0]
    if match_weights is None:
        match_weights = np.ones(n, dtype=np.float32)
    else:
        match_weights = np.asarray(match_weights, dtype=np.float32)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if sigmas.ndim == 0:
        sigmas = np.full(n, float(sigmas), dtype=np.float64)
    variances = (sigmas * 2.0) ** 2
    area = max(float((bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])), 1.0)

    visible = det_conf > conf_thresh
    if not np.any(visible):
        return 0.0

    delta = pred_xy - det_xy
    squared_dist = np.sum(delta * delta, axis=1)
    oks_per_keypoint = np.exp(-squared_dist / variances / area / 2.0)
    oks_per_keypoint = oks_per_keypoint * visible

    weights = match_weights * visible.astype(np.float32)
    denom = float(weights.sum())
    if denom <= 0:
        return 0.0
    return float((oks_per_keypoint * weights).sum() / denom)


def pose_oks_distance(
    atracks: list,
    btracks: list,
    *,
    sigmas: float | np.ndarray = 0.05,
    match_weights: np.ndarray | None = None,
    use_pred_pose: bool = True,
    conf_thresh: float = 0.0,
    pose_match_thresh: float = 0.0,
) -> np.ndarray:
    """Compute cost matrix ``1 - weighted OKS`` between track and detection poses."""
    cost_matrix = np.ones((len(atracks), len(btracks)), dtype=np.float32)
    if not atracks or not btracks:
        return cost_matrix

    for i, track in enumerate(atracks):
        pred_xy = _track_keypoints_xy(track, use_pred_pose=use_pred_pose)
        for j, det in enumerate(btracks):
            det_xy = _track_keypoints_xy(det, use_pred_pose=False)
            det_conf = _track_keypoints_conf(det)
            det_box = det.xyxy if hasattr(det, "xyxy") else det[:4]
            oks = weighted_oks(
                pred_xy,
                det_xy,
                det_conf,
                np.asarray(det_box, dtype=np.float32),
                sigmas=sigmas,
                match_weights=match_weights,
                conf_thresh=conf_thresh,
            )
            if pose_match_thresh > 0 and oks < pose_match_thresh:
                cost_matrix[i, j] = 1.0
            else:
                cost_matrix[i, j] = 1.0 - oks
    return cost_matrix


def _rel_bones_from_keypoints(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return parent-relative bone vectors (20, 2) and pair confidences (20,)."""
    from .pose_kalman_filter import HAND_PARENT, abs_to_rel_meas

    kpts = np.asarray(keypoints, dtype=np.float32)
    rel = abs_to_rel_meas(kpts[:, :2], HAND_PARENT)
    pair_conf = np.ones(rel.shape[0], dtype=np.float32)
    if kpts.shape[1] > 2:
        for i, child in enumerate(range(1, kpts.shape[0])):
            parent = int(HAND_PARENT[child])
            pair_conf[i] = float(min(kpts[parent, 2], kpts[child, 2]))
    return rel, pair_conf


def _rel_bones_from_track(
    track,
    *,
    use_pred_pose: bool = True,
    use_static: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract parent-relative bones for association from a track or detection."""
    from .basetrack import TrackState

    if (
        use_static
        and getattr(track, "state", None) == TrackState.Lost
        and getattr(track, "static_pose_mean", None) is not None
        and getattr(track, "pose_kalman_filter", None) is not None
    ):
        rel = track.pose_kalman_filter.get_rel_positions(track.static_pose_mean)
        conf = getattr(track, "static_pose_pair_conf", None)
        if conf is None:
            conf = np.ones(rel.shape[0], dtype=np.float32)
        return rel.astype(np.float32), np.asarray(conf, dtype=np.float32)

    if use_pred_pose and getattr(track, "pose_mean", None) is not None and getattr(track, "pose_kalman_filter", None) is not None:
        rel = track.pose_kalman_filter.get_rel_positions(track.pose_mean)
        conf = np.ones(rel.shape[0], dtype=np.float32)
        if getattr(track, "kpt_conf_ema", None) is not None:
            from .pose_kalman_filter import HAND_PARENT

            for i, child in enumerate(range(1, track.n_keypoints)):
                parent = int(HAND_PARENT[child])
                conf[i] = float(min(track.kpt_conf_ema[parent], track.kpt_conf_ema[child]))
        return rel.astype(np.float32), conf

    if hasattr(track, "keypoints"):
        return _rel_bones_from_keypoints(track.keypoints)
    raise AttributeError("Track object does not expose pose data for bone matching.")


def weighted_bone_cosine_similarity(
    track_rel: np.ndarray,
    det_rel: np.ndarray,
    det_pair_conf: np.ndarray,
    *,
    bone_weights: np.ndarray | None = None,
    conf_thresh: float = 0.0,
) -> float:
    """Cosine similarity between parent-relative bone vectors with finger-depth weights."""
    from .pose_kalman_filter import BONE_MATCH_WEIGHTS

    track_rel = np.asarray(track_rel, dtype=np.float32).reshape(-1, 2)
    det_rel = np.asarray(det_rel, dtype=np.float32).reshape(-1, 2)
    det_pair_conf = np.asarray(det_pair_conf, dtype=np.float32).reshape(-1)
    if bone_weights is None:
        bone_weights = BONE_MATCH_WEIGHTS
    else:
        bone_weights = np.asarray(bone_weights, dtype=np.float32).reshape(-1)

    n = min(track_rel.shape[0], det_rel.shape[0], det_pair_conf.shape[0], bone_weights.shape[0])
    if n == 0:
        return 0.0

    track_rel = track_rel[:n]
    det_rel = det_rel[:n]
    det_pair_conf = det_pair_conf[:n]
    bone_weights = bone_weights[:n]

    visible = det_pair_conf > conf_thresh
    if not np.any(visible):
        return 0.0

    t_unit = track_rel / (np.linalg.norm(track_rel, axis=1, keepdims=True) + 1e-6)
    d_unit = det_rel / (np.linalg.norm(det_rel, axis=1, keepdims=True) + 1e-6)
    d_lens = np.linalg.norm(det_rel, axis=1)
    len_weights = np.clip(d_lens / (np.max(d_lens) * 0.5 + 1e-6), 0.1, 1.0)

    cos = np.sum(t_unit * d_unit, axis=1)
    weights = bone_weights * det_pair_conf * len_weights * visible.astype(np.float32)
    denom = float(weights.sum())
    if denom <= 0:
        return 0.0
    return float(np.clip((cos * weights).sum() / denom, -1.0, 1.0))


def bone_cosine_distance(
    atracks: list,
    btracks: list,
    *,
    use_pred_pose: bool = True,
    use_static: bool = False,
    conf_thresh: float = 0.0,
    bone_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute cost matrix ``(1 - cosine_sim) / 2`` on parent-relative bones."""
    cost_matrix = np.ones((len(atracks), len(btracks)), dtype=np.float32)
    if not atracks or not btracks:
        return cost_matrix

    for i, track in enumerate(atracks):
        track_rel, _ = _rel_bones_from_track(track, use_pred_pose=use_pred_pose, use_static=False)
        static_rel = static_conf = None
        if use_static and getattr(track, "static_pose_mean", None) is not None:
            static_rel, static_conf = _rel_bones_from_track(track, use_pred_pose=True, use_static=True)

        for j, det in enumerate(btracks):
            det_rel, det_conf = _rel_bones_from_keypoints(det.keypoints) if hasattr(det, "keypoints") else _rel_bones_from_track(det, use_pred_pose=False)
            sim = weighted_bone_cosine_similarity(
                track_rel,
                det_rel,
                det_conf,
                bone_weights=bone_weights,
                conf_thresh=conf_thresh,
            )
            if static_rel is not None:
                sim_static = weighted_bone_cosine_similarity(
                    static_rel,
                    det_rel,
                    static_conf if static_conf is not None else det_conf,
                    bone_weights=bone_weights,
                    conf_thresh=conf_thresh,
                )
                sim = max(sim, sim_static)
            cost_matrix[i, j] = (1.0 - sim) / 2.0
    return cost_matrix


def cls_soft_match_penalty(
    tracks: list,
    detections: list,
    *,
    penalty: float,
) -> np.ndarray:
    """Return soft penalty matrix when detection class disagrees with track belief."""
    cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if penalty <= 0 or not tracks or not detections:
        return cost
    for i, track in enumerate(tracks):
        track_cls = int(round(float(getattr(track, "cls", 0))))
        for j, det in enumerate(detections):
            det_cls = int(round(float(getattr(det, "cls", 0))))
            if det_cls != track_cls:
                cost[i, j] = float(penalty)
    return cost


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:
    """Fuse cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments, with shape (N, M).
        detections (list[BaseTrack]): List of detections, each containing a score attribute.

    Returns:
        (np.ndarray): Fused similarity matrix with shape (N, M).

    Examples:
        Fuse a cost matrix with detection scores
        >>> cost_matrix = np.random.rand(5, 10)  # 5 tracks and 10 detections
        >>> detections = [BaseTrack(score=np.random.rand()) for _ in range(10)]
        >>> fused_matrix = fuse_score(cost_matrix, detections)
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost
