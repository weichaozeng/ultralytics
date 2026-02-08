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
    def get_coords(tracks):
        if len(tracks) == 0:
            return []
        if isinstance(tracks, np.ndarray):
            return tracks
        if isinstance(tracks[0], np.ndarray):
            return np.asarray(tracks)
        return [t.xywha if t.angle is not None else t.xyxy for t in tracks]
    atlbrs = get_coords(atracks)
    btlbrs = get_coords(btracks)
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
    print("33333")
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


HAND_BONE_CONNECTIONS = np.array([
    # Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    # Index
    [0, 5], [5, 6], [6, 7], [7, 8],
    # Middle
    [0, 9], [9, 10], [10, 11], [11, 12],
    # Ring
    [0, 13], [13, 14], [14, 15], [15, 16],
    # Pinky
    [0, 17], [17, 18], [18, 19], [19, 20]
], dtype=np.int32)

def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    norms[norms < 1e-6] = 1.0 
    return vecs / norms

def bone_distance(atracks: list, btracks: list) -> np.ndarray: 
    Nt = len(atracks)
    Nd = len(btracks)
    if Nt == 0 or Nd == 0:
        return np.zeros((Nt, Nd), dtype=np.float32)
    kps_a = np.array([track.pxyxy.reshape(-1, 2) for track in atracks], dtype=np.float32)
    kps_b = np.array([track.pxyxy.reshape(-1, 2) for track in btracks], dtype=np.float32)
    confs_a = np.array([track.kps_score for track in atracks], dtype=np.float32)
    confs_b = np.array([track.kps_score for track in btracks], dtype=np.float32) 
    
    conn = HAND_BONE_CONNECTIONS
    vecs_a = kps_a[:, conn[:, 1], :] - kps_a[:, conn[:, 0], :]
    vecs_b = kps_b[:, conn[:, 1], :] - kps_b[:, conn[:, 0], :]
    vecs_norm_a = normalize_vectors(vecs_a)
    vecs_norm_b = normalize_vectors(vecs_b)

    confs_start_a = confs_a[:, conn[:, 0]]
    confs_end_a = confs_a[:, conn[:, 1]]
    bone_weight_a = np.minimum(confs_start_a, confs_end_a)

    confs_start_b = confs_b[:, conn[:, 0]]
    confs_end_b = confs_b[:, conn[:, 1]]
    bone_weight_b = np.minimum(confs_start_b, confs_end_b)

    vecs_norm_a_exp = np.expand_dims(vecs_norm_a, 1)
    vecs_norm_b_exp = np.expand_dims(vecs_norm_b, 0)

    cos_sim = np.sum(vecs_norm_a_exp * vecs_norm_b_exp, axis=-1) # (Nt, Nd, 20)
    dissim = (1 - cos_sim) / 2.0

    weights_a_exp = np.expand_dims(bone_weight_a, 1)
    weights_b_exp = np.expand_dims(bone_weight_b, 0)
    final_weights = 1 - np.minimum(weights_a_exp, weights_b_exp) # (Nt, Nd, 20)

    weighted_dissim = np.sum(dissim * final_weights, axis=-1) # (Nt, Nd)

    total_weight = np.sum(final_weights, axis=-1)
    total_weight[total_weight < 1e-6] = 1e-6

    return weighted_dissim / total_weight


def kp_distance(atracks: list, btracks: list, scale_factor: float = 50.0) -> np.ndarray:
    Nt = len(atracks)
    Nd = len(btracks)
    if Nt == 0 or Nd == 0:
        return np.zeros((Nt, Nd), dtype=np.float32)
    
    kps_a = np.array([track.pxyxy for track in atracks], dtype=np.float32) # (Nt, 42)
    kps_b = np.array([track.pxyxy for track in btracks], dtype=np.float32) # (Nd, 42)
    
    scores_a_21 = np.array([track.kps_score for track in atracks], dtype=np.float32) # (Nt, 21)
    scores_b_21 = np.array([track.kps_score for track in btracks], dtype=np.float32) # (Nd, 21)

    scores_a_42 = np.repeat(scores_a_21, 2, axis=1) 
    scores_b_42 = np.repeat(scores_b_21, 2, axis=1)

    kps_a_exp = np.expand_dims(kps_a, 1)
    kps_b_exp = np.expand_dims(kps_b, 0)
    squared_diff = np.square(kps_a_exp - kps_b_exp)

    confs_a_42 = 1.0 - scores_a_42
    confs_b_42 = 1.0 - scores_b_42
    confs_a_exp = np.expand_dims(confs_a_42, 1)
    confs_b_exp = np.expand_dims(confs_b_42, 0)

    final_confs = np.maximum(confs_a_exp, confs_b_exp) # (Nt, Nd, 42)
    weighted_distance_sum = np.sum(squared_diff * (1.0 + final_confs), axis=-1) # (Nt, Nd)

    M = scores_a_42.shape[1] # 42
    normalization_factor = M * (scale_factor ** 2)

    kp_dissimilarity_matrix = np.sqrt(weighted_distance_sum / normalization_factor)

    kp_dissimilarity_matrix[kp_dissimilarity_matrix > 1.0] = 1.0

    return kp_dissimilarity_matrix
