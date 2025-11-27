import numpy as np
import scipy.linalg

from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH


class KalmanFilterPose:
    def __init__(self):
        self.ndim = 42 
        self.dt = 1.0
        self.scale_factor = 5.0
        # F = [[I, dt*I], [0, I]]
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        
        # H = [[I, 0]]
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

        # 
        self._std_weight_position = 1.0 / 5.0 # 1.0 / 20.0 
        self._std_weight_velocity = 1.0 / 40.0 # 1.0 / 160.0

    def initiate(self, measurement: np.ndarray):
        """
        Args:
            measurement (np.ndarray): 42 dim pose keypoints.
        
        Returns:
            mean (np.ndarray): 84 dim mean vector.
            covariance (np.ndarray): 84 * 84 dim cov matrix.
        """


        mean_pos = measurement.flatten()
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std_pos_kps = 2 * self._std_weight_position * self.scale_factor
        std_vel_kps = 10 * self._std_weight_velocity * self.scale_factor

        std_pos_array = np.full(self.ndim, std_pos_kps)
        std_vel_array = np.full(self.ndim, std_vel_kps)

        std = np.r_[std_pos_array, std_vel_array]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        std_pos_kps = self._std_weight_position * self.scale_factor
        std_vel_kps = self._std_weight_velocity * self.scale_factor

        std_pos = np.full(self.ndim, std_pos_kps)
        std_vel = np.full(self.ndim, std_vel_kps)

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        # X_{t} = F * X_{t-1}
        mean = np.dot(mean, self._motion_mat.T)
        # P_{t} = F * P_{t-1} * F.T + Q
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance
    
    def project(self, mean: np.ndarray, covariance: np.ndarray):

        std_kps = self._std_weight_position * self.scale_factor
        std = np.full(self.ndim, std_kps)
        innovation_cov = np.diag(np.square(std))

        # Z_{t} = H & X_{t}
        mean = np.dot(self._update_mat, mean)
        # S = H * P * H.T + R
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        
        return mean, covariance + innovation_cov
    
    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """
        Args:
            mean (np.ndarray): 84 dim.
            covariance (np.ndarray): 84 * 84 dim.
            measurement (np.ndarray): 42 dim.

        Returns:
            new_mean (no.ndarray): Measurement-corrected state mean.
            new_covariance (np.ndarray): Measurement-corrected state covariance.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # K = P_bar * H.T * S^-1
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        # P_bar * H.T
        P_bar_HT = np.dot(covariance, self._update_mat.T).T
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), P_bar_HT, check_finite=False
        ).T

        innovation = measurement - projected_mean

        # X_new = X_bar + K * innovation
        new_mean = mean + np.dot(innovation, kalman_gain.T)

        # P_new = P_bar - K * S * K.T 
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance
    
    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """
        Args:
            mean (np.ndarray): Nx84 dim.
            covariance (np.ndarray): Nx84x84.

        Returns:
            mean (np.ndarray): (N, 84).
            covariance (np.ndarray): (N, 84, 84).
        """
        if mean.shape[0] == 0:
            return mean, covariance
        
        N = mean.shape[0]
        std_pos_kps = self._std_weight_position * self.scale_factor
        std_vel_kps = self._std_weight_velocity * self.scale_factor

        std_pos = np.full((N, self.ndim), std_pos_kps)
        std_vel = np.full((N, self.ndim), std_vel_kps)

        sqr = np.square(np.concatenate((std_pos, std_vel), axis=1))
        motion_cov = np.array([np.diag(sqr[i]) for i in range(N)])

        # X_t = X_{t-1} * F.T
        mean = np.dot(mean, self._motion_mat.T) # Shape (N, 84)
        
        # P_t = F * P_{t-1} * F.T + Q
        # F * P_{t-1}
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2)) # Shape (N, 84, 84)
        
        # (F * P_{t-1}) * F.T + Q
        covariance = np.dot(left, self._motion_mat.T) + motion_cov # Shape (N, 84, 84)

        return mean, covariance


    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray, only_position: bool = False, metric: str = "maha"):
        projected_mean, projected_cov = self.project(mean, covariance)
        if only_position:
            pass
        d = measurements - projected_mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        
        elif metric == "maha":
            # d^2 = y.T * S^-1 * y
            cholesky_factor = np.linalg.cholesky(projected_cov)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)
        
        else:
            raise ValueError("Invalid distance metric")
