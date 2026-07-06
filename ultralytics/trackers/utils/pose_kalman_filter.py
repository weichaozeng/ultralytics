# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Parent-relative hand pose Kalman filter."""

from __future__ import annotations

import numpy as np
import scipy.linalg

# Parent index per keypoint (wrist=0 has parent -1).
HAND_PARENT = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], dtype=np.int32)

# Depth from wrist used for process/observation noise scaling.
HAND_CHAIN_DEPTH = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32)

# Association weights: proximal joints are more stable.
MATCH_WEIGHT_BY_DEPTH = (1.0, 1.0, 0.85, 0.70, 0.50)
HAND_MATCH_WEIGHTS = np.array([MATCH_WEIGHT_BY_DEPTH[d] for d in HAND_CHAIN_DEPTH], dtype=np.float32)


def build_parent_idx(n_keypoints: int, parent_idx: np.ndarray | None = None) -> np.ndarray:
    """Return parent index array for ``n_keypoints`` joints."""
    base = HAND_PARENT if parent_idx is None else np.asarray(parent_idx, dtype=np.int32)
    if base.shape[0] != n_keypoints:
        raise ValueError(f"parent_idx length must be {n_keypoints}, got {base.shape[0]}")
    return base


def abs_to_rel_meas(abs_xy: np.ndarray, parent_idx: np.ndarray) -> np.ndarray:
    """Convert absolute xy positions to parent-relative measurements."""
    rel = np.zeros((parent_idx.shape[0] - 1, 2), dtype=np.float32)
    for k in range(1, parent_idx.shape[0]):
        rel[k - 1] = abs_xy[k] - abs_xy[parent_idx[k]]
    return rel


def fk_abs(anchor: np.ndarray, rel_pos: np.ndarray, parent_idx: np.ndarray) -> np.ndarray:
    """Forward kinematics from wrist anchor and parent-relative positions."""
    n_keypoints = rel_pos.shape[0] + 1
    abs_xy = np.zeros((n_keypoints, 2), dtype=np.float32)
    abs_xy[0] = anchor
    for k in range(1, n_keypoints):
        abs_xy[k] = abs_xy[parent_idx[k]] + rel_pos[k - 1]
    return abs_xy


class KalmanFilterPoseChain:
    """Kalman filter over parent-relative hand keypoint bone vectors.

    State layout for joints k=1..K-1 (80 dims when K=21):
        [dx, dy, vdx, vdy] per relative joint, concatenated in keypoint order.
  Wrist (k=0) is provided externally as an anchor.
    """

    def __init__(
        self,
        n_keypoints: int = 21,
        parent_idx: np.ndarray | None = None,
        dt: float = 1.0,
        std_weight_position: float = 1.0 / 20,
        std_weight_velocity: float = 1.0 / 160,
        depth_noise_scale: tuple[float, ...] = (0.5, 0.8, 1.0, 1.3, 1.6),
    ):
        self.parent_idx = build_parent_idx(n_keypoints, parent_idx)
        self.n_keypoints = int(n_keypoints)
        self.n_rel = self.n_keypoints - 1
        self.ndim = 4 * self.n_rel
        self.dt = float(dt)
        self._std_weight_position = float(std_weight_position)
        self._std_weight_velocity = float(std_weight_velocity)
        self._depth_noise_scale = tuple(float(x) for x in depth_noise_scale)

        block = np.eye(4, dtype=np.float64)
        block[0, 2] = self.dt
        block[1, 3] = self.dt
        self._block_f = block
        self._block_h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

        self._motion_mat = np.zeros((self.ndim, self.ndim), dtype=np.float64)
        self._update_mat = np.zeros((2 * self.n_rel, self.ndim), dtype=np.float64)
        for i in range(self.n_rel):
            s = 4 * i
            self._motion_mat[s : s + 4, s : s + 4] = self._block_f
            self._update_mat[2 * i : 2 * i + 2, s : s + 4] = self._block_h

        self._rel_depth = np.array([HAND_CHAIN_DEPTH[k] for k in range(1, self.n_keypoints)], dtype=np.int32)

    def _depth_scale(self, depth: int) -> float:
        depth = int(np.clip(depth, 0, len(self._depth_noise_scale) - 1))
        return self._depth_noise_scale[depth]

    def _position_std(self, scale: float, depth: int) -> float:
        return 2.0 * self._std_weight_position * scale * self._depth_scale(depth)

    def _velocity_std(self, scale: float, depth: int) -> float:
        return 10.0 * self._std_weight_velocity * scale * self._depth_scale(depth)

    def get_rel_positions(self, mean: np.ndarray) -> np.ndarray:
        """Return parent-relative xy positions with shape (n_rel, 2)."""
        rel = np.zeros((self.n_rel, 2), dtype=np.float32)
        for i in range(self.n_rel):
            rel[i, 0] = mean[4 * i]
            rel[i, 1] = mean[4 * i + 1]
        return rel

    def rel_to_abs(self, mean: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """Return absolute xy positions with shape (n_keypoints, 2)."""
        return fk_abs(np.asarray(anchor, dtype=np.float32), self.get_rel_positions(mean), self.parent_idx)

    def initiate(self, keypoints: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a new pose track state from keypoints shaped (K, 2+) and wrist anchor."""
        abs_xy = np.asarray(keypoints[:, :2], dtype=np.float64)
        anchor = np.asarray(anchor, dtype=np.float64)
        rel = abs_to_rel_meas(abs_xy, self.parent_idx).astype(np.float64)
        scale = float(max(np.linalg.norm(abs_xy.max(0) - abs_xy.min(0)), 1.0))

        mean = np.zeros(self.ndim, dtype=np.float64)
        covariance = np.zeros((self.ndim, self.ndim), dtype=np.float64)
        for i in range(self.n_rel):
            depth = int(self._rel_depth[i])
            mean[4 * i : 4 * i + 2] = rel[i]
            pos_std = self._position_std(scale, depth)
            vel_std = self._velocity_std(scale, depth)
            idx = slice(4 * i, 4 * i + 4)
            covariance[idx, idx] = np.diag(np.square([pos_std, pos_std, vel_std, vel_std]))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step."""
        mean = np.asarray(mean, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)
        scale = float(max(np.linalg.norm(self.get_rel_positions(mean), axis=1).mean(), 1.0))

        motion_cov = np.zeros((self.ndim, self.ndim), dtype=np.float64)
        for i in range(self.n_rel):
            depth = int(self._rel_depth[i])
            pos_std = self._std_weight_position * scale * self._depth_scale(depth)
            vel_std = self._std_weight_velocity * scale * self._depth_scale(depth)
            s = 4 * i
            motion_cov[s : s + 4, s : s + 4] = np.diag(np.square([pos_std, pos_std, vel_std, vel_std]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized prediction for N tracks."""
        if mean.ndim == 1:
            return self.predict(mean, covariance)
        n = mean.shape[0]
        new_mean = np.zeros_like(mean)
        new_cov = np.zeros_like(covariance)
        for i in range(n):
            new_mean[i], new_cov[i] = self.predict(mean[i], covariance[i])
        return new_mean, new_cov

    def _project_block(self, mean_block: np.ndarray, cov_block: np.ndarray, depth: int, scale: float):
        innovation_cov = np.diag(
            np.square(
                [
                    self._std_weight_position * scale * self._depth_scale(depth),
                    self._std_weight_position * scale * self._depth_scale(depth),
                ]
            )
        )
        mean = self._block_h @ mean_block
        covariance = self._block_h @ cov_block @ self._block_h.T + innovation_cov
        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        keypoints: np.ndarray,
        anchor: np.ndarray,
        visible_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman correction with parent-relative measurements."""
        mean = np.asarray(mean, dtype=np.float64).copy()
        covariance = np.asarray(covariance, dtype=np.float64).copy()
        abs_xy = np.asarray(keypoints[:, :2], dtype=np.float64)
        anchor = np.asarray(anchor, dtype=np.float64)
        rel_meas = abs_to_rel_meas(abs_xy, self.parent_idx)
        scale = float(max(np.linalg.norm(abs_xy.max(0) - abs_xy.min(0)), 1.0))

        if visible_mask is None:
            if keypoints.shape[1] > 2:
                visible_mask = keypoints[:, 2] > 0
            else:
                visible_mask = np.ones(self.n_keypoints, dtype=bool)
        visible_mask = np.asarray(visible_mask, dtype=bool)

        for k in range(1, self.n_keypoints):
            if not visible_mask[k]:
                continue
            i = k - 1
            depth = int(self._rel_depth[i])
            s = 4 * i
            mean_block = mean[s : s + 4]
            cov_block = covariance[s : s + 4, s : s + 4]
            projected_mean, projected_cov = self._project_block(mean_block, cov_block, depth, scale)
            measurement = rel_meas[i]

            chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower),
                (cov_block @ self._block_h.T).T,
                check_finite=False,
            ).T
            innovation = measurement - projected_mean
            mean[s : s + 4] = mean_block + kalman_gain @ innovation
            covariance[s : s + 4, s : s + 4] = cov_block - kalman_gain @ projected_cov @ kalman_gain.T
        return mean, covariance
