import numpy as np
import scipy.linalg

from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH


class KalmanFilterPose:
    def __init__(self):
        ndim = 42 
        dt = 1.0
        # F = [[I, dt*I], [0, I]]
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt