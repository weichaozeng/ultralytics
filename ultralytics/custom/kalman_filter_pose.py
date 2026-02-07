import numpy as np
import scipy.linalg

from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH


# class KalmanFilterPose:
#     def __init__(self):
#         self.ndim = 42 
#         self.dt = 1.0
#         self.scale_factor = 10.0
#         # F = [[I, dt*I], [0, I]]
#         self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
#         for i in range(self.ndim):
#             self._motion_mat[i, self.ndim + i] = self.dt
        
#         # H = [[I, 0]]
#         self._update_mat = np.eye(self.ndim, 2 * self.ndim)

#         # noise in measurement
#         self._std_weight_measurement = 1.0 / 5.0

#         # noise in estimation
#         self._std_weight_motion_pos = 1.0 / 100.0 
#         self._std_weight_motion_vel = 1.0 / 20.0

#     def initiate(self, measurement: np.ndarray):
#         """
#         Args:
#             measurement (np.ndarray): 42 dim pose keypoints.
        
#         Returns:
#             mean (np.ndarray): 84 dim mean vector.
#             covariance (np.ndarray): 84 * 84 dim cov matrix.
#         """


#         mean_pos = measurement.flatten()
#         mean_vel = np.zeros_like(mean_pos)
#         mean = np.r_[mean_pos, mean_vel]

#         std_pos_kps = 2 * self._std_weight_measurement * self.scale_factor
#         std_vel_kps = 10 * self._std_weight_motion_vel * self.scale_factor

#         std_pos_array = np.full(self.ndim, std_pos_kps)
#         std_vel_array = np.full(self.ndim, std_vel_kps)

#         std = np.r_[std_pos_array, std_vel_array]
#         covariance = np.diag(np.square(std))
#         return mean, covariance
    
#     def predict(self, mean: np.ndarray, covariance: np.ndarray):
#         std_pos_kps = self._std_weight_motion_pos * self.scale_factor
#         std_vel_kps = self._std_weight_motion_vel * self.scale_factor

#         std_pos = np.full(self.ndim, std_pos_kps)
#         std_vel = np.full(self.ndim, std_vel_kps)

#         motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
#         # X_{t} = F * X_{t-1}
#         mean = np.dot(mean, self._motion_mat.T)
#         # P_{t} = F * P_{t-1} * F.T + Q
#         covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

#         return mean, covariance
    
#     def project(self, mean: np.ndarray, covariance: np.ndarray):

#         std_kps = self._std_weight_measurement * self.scale_factor
#         std = np.full(self.ndim, std_kps)
#         innovation_cov = np.diag(np.square(std))

#         # Z_{t} = H & X_{t}
#         mean = np.dot(self._update_mat, mean)
#         # S = H * P * H.T + R
#         covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        
#         return mean, covariance + innovation_cov
    
#     def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
#         """
#         Args:
#             mean (np.ndarray): 84 dim.
#             covariance (np.ndarray): 84 * 84 dim.
#             measurement (np.ndarray): 42 dim.

#         Returns:
#             new_mean (no.ndarray): Measurement-corrected state mean.
#             new_covariance (np.ndarray): Measurement-corrected state covariance.
#         """
#         projected_mean, projected_cov = self.project(mean, covariance)

#         # K = P_bar * H.T * S^-1
#         chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
#         # P_bar * H.T
#         P_bar_HT = np.dot(covariance, self._update_mat.T).T
#         kalman_gain = scipy.linalg.cho_solve(
#             (chol_factor, lower), P_bar_HT, check_finite=False
#         ).T

#         innovation = measurement - projected_mean

#         # X_new = X_bar + K * innovation
#         new_mean = mean + np.dot(innovation, kalman_gain.T)

#         # P_new = P_bar - K * S * K.T 
#         new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

#         return new_mean, new_covariance
    
#     def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
#         """
#         Args:
#             mean (np.ndarray): Nx84 dim.
#             covariance (np.ndarray): Nx84x84.

#         Returns:
#             mean (np.ndarray): (N, 84).
#             covariance (np.ndarray): (N, 84, 84).
#         """
#         if mean.shape[0] == 0:
#             return mean, covariance
        
#         N = mean.shape[0]
#         std_pos_kps = self._std_weight_motion_pos * self.scale_factor
#         std_vel_kps = self._std_weight_motion_vel * self.scale_factor

#         std_pos = np.full((N, self.ndim), std_pos_kps)
#         std_vel = np.full((N, self.ndim), std_vel_kps)

#         sqr = np.square(np.concatenate((std_pos, std_vel), axis=1))
#         motion_cov = np.array([np.diag(sqr[i]) for i in range(N)])

#         # X_t = X_{t-1} * F.T
#         mean = np.dot(mean, self._motion_mat.T) # Shape (N, 84)
        
#         # P_t = F * P_{t-1} * F.T + Q
#         # F * P_{t-1}
#         left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2)) # Shape (N, 84, 84)
        
#         # (F * P_{t-1}) * F.T + Q
#         covariance = np.dot(left, self._motion_mat.T) + motion_cov # Shape (N, 84, 84)

#         return mean, covariance


#     def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray, only_position: bool = False, metric: str = "maha"):
#         projected_mean, projected_cov = self.project(mean, covariance)
#         if only_position:
#             pass
#         d = measurements - projected_mean
#         if metric == "gaussian":
#             return np.sum(d * d, axis=1)
        
#         elif metric == "maha":
#             # d^2 = y.T * S^-1 * y
#             cholesky_factor = np.linalg.cholesky(projected_cov)
#             z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
#             return np.sum(z * z, axis=0)
        
#         else:
#             raise ValueError("Invalid distance metric")


################################################################

class KalmanFilterPose:
    def __init__(self):
        # 20: (P_child - P_parent), 4: (x, y, vx, vy)
        self.ndim_state = 80
        self.ndim_obs = 40
        self.dt = 1.0

        # motion matrix
        self._motion_mat = np.eye(self.ndim_state)
        for i in range(20):
            self._motion_mat[i*2, 40 + i*2] = self.dt     # x + vx*dt
            self._motion_mat[i*2+1, 40 + i*2+1] = self.dt # y + vy*dt

        # observation matrix
        self._update_mat = np.zeros((self.ndim_obs, self.ndim_state))
        for i in range(20):
            self._update_mat[i*2, i*2] = 1
            self._update_mat[i*2+1, i*2+1] = 1

        self._std_weight_measurement = 0.02 
        self._std_weight_motion_pos = 0.002 
        self._std_weight_motion_vel = 0.005
        self.scale = 10.0

    def initiate(self, rel_measurements):
        mean = np.zeros(self.ndim_state)
        mean[:40] = rel_measurements.flatten()
        std = np.r_[
            np.full(40, self._std_weight_measurement),
            np.full(40, self._std_weight_motion_vel * self.scale)
        ]
        covariance = np.diag(np.square(std))
        
        return mean, covariance

    def predict(self, mean, covariance):
        std = np.r_[
            np.full(40, self._std_weight_motion_pos),
            np.full(40, self._std_weight_motion_vel)
        ]
        motion_cov = np.diag(np.square(std))

        # X = F * X_prev
        mean = np.dot(self._motion_mat, mean)

        # P = F * P_prev * F.T + Q 
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance
    
    def project(self, mean, covariance, confidence=None):
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        
        if confidence is not None:
            assert len(confidence) == 20
            conf_expanded = np.repeat(confidence, 2)
            R = np.diag(np.square(self._std_weight_measurement / np.clip(conf_expanded, 1e-4, 1.0)))
            return projected_mean, projected_cov + R
        
        return projected_mean, projected_cov
    
    def update(self, mean, covariance, measurement, confidences):
        projected_mean, projected_cov = self.project(mean, covariance, confidences)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        P_HT = np.dot(covariance, self._update_mat.T)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), P_HT.T).T
        innovation = measurement.flatten() - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance
    
    def multi_predict(self, mean, covariance):
        """
        mean: (N, 80)
        covariance: (N, 80, 80)
        """
        if mean.shape[0] == 0:
            return mean, covariance
        
        N = mean.shape[0]
        std = np.r_[
            np.full(40, self._std_weight_motion_pos),
            np.full(40, self._std_weight_motion_vel)
        ]
        q_single = np.diag(np.square(std))
        motion_cov = np.stack([q_single] * N)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.matmul(self._motion_mat, covariance)
        covariance = np.matmul(left, self._motion_mat.T) + motion_cov

        return mean, covariance
    
    def gating_distance(self, mean, covariance, measurements,  metric="maha"):
        """
        measurements: (M, 40)
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidences=None)
        r_std = np.full(40, self._std_weight_measurement) 
        R_static = np.diag(np.square(r_std))
        S = projected_cov + R_static
        
        # d = z - z_hat
        d = measurements - projected_mean # (M, 40)
        
        if metric == "maha":
            try:
                cholesky_factor = np.linalg.cholesky(S)
                z = scipy.linalg.solve_triangular(
                    cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
                return np.sum(z * z, axis=0) # (M,)
            except np.linalg.LinAlgError:
                return np.sum(d**2, axis=1)
        else:
            return np.sum(d**2, axis=1)
################################################################

class KalmanFilterPose_Polar:
    def __init__(self):
        # 20: (P_child - P_parent), 4: (rho, theta, v_rho, v_theta)
        self.ndim_state = 80
        self.ndim_obs = 40
        self.dt = 1.0

        # rho_t = rho + v_rho * dt
        # theta_t = theta + v_theta * dt
        self._motion_mat = np.eye(self.ndim_state)
        for i in range(20):
            self._motion_mat[i*2, 40 + i*2] = self.dt     # rho + v_rho*dt
            self._motion_mat[i*2+1, 40 + i*2+1] = self.dt # theta + v_theta*dt

        self._std_weight_measurement = 0.02
        self._std_weight_motion_rho = 0.0001 # small
        self._std_weight_motion_theta = 0.01  
        self._std_weight_v_rho = 0.0001      
        self._std_weight_v_theta = 0.02

        self.scale = 10.0

    def initiate(self, rel_measurements):
        mean = np.zeros(self.ndim_state)
        dx = rel_measurements[:, 0]
        dy = rel_measurements[:, 1]

        rho = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)  # (-pi, pi)
        for i in range(20):
            mean[i*2] = rho[i]
            mean[i*2 + 1] = theta[i]

        std_rho = self._std_weight_motion_rho * self.scale
        std_theta = self._std_weight_motion_theta * self.scale
        std_v_rho = self._std_weight_v_rho * self.scale
        std_v_theta = self._std_weight_v_theta * self.scale

        std_pos = np.array([[std_rho, std_theta] for _ in range(20)]).flatten()
        std_vel = np.array([[std_v_rho, std_v_theta] for _ in range(20)]).flatten()

        std = np.r_[std_pos, std_vel]
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(self, mean, covariance):

        std_rho = self._std_weight_motion_rho
        std_theta = self._std_weight_motion_theta
        std_v_rho = self._std_weight_v_rho
        std_v_theta = self._std_weight_v_theta

        std_pos = np.array([[std_rho, std_theta] for _ in range(20)]).flatten()
        std_vel = np.array([[std_v_rho, std_v_theta] for _ in range(20)]).flatten()

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # rho_new = rho + v_rho * dt; theta_new = theta + v_theta * dt
        mean = np.dot(self._motion_mat, mean)

        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance
    
    def project(self, mean, covariance, confidences):
        projected_mean = np.zeros(self.ndim_obs)
        for i in range(20):
            rho = mean[i*2]
            theta = mean[i*2 + 1]
            projected_mean[i*2] = rho * np.cos(theta)
            projected_mean[i*2 + 1] = rho * np.sin(theta)

        H = np.zeros((self.ndim_obs, self.ndim_state))
        for i in range(20):
            rho = mean[i*2]
            theta = mean[i*2 + 1]
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            H[i*2, i*2]     = cos_t          # dx/d_rho
            H[i*2, i*2 + 1] = -rho * sin_t   # dx/d_theta
            H[i*2 + 1, i*2] = sin_t          # dy/d_rho
            H[i*2 + 1, i*2+1] = rho * cos_t  # dy/d_theta
        
        conf_expanded = np.repeat(confidences, 2)
        r_std = self._std_weight_measurement / (conf_expanded + 1e-6)
        R = np.diag(np.square(r_std))

        # S = H * P * H.T + R
        projected_cov = np.linalg.multi_dot((H, covariance, H.T)) + R
        
        return projected_mean, projected_cov, H
    
    def update(self, mean, covariance, measurement, confidences):
        projected_mean, projected_cov, H = self.project(mean, covariance, confidences)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        P_HT = np.dot(covariance, H.T)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), P_HT.T).T
        innovation = measurement.flatten() - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)

        for i in range(20):
            theta_idx = i * 2 + 1
            new_mean[theta_idx] = (new_mean[theta_idx] + np.pi) % (2 * np.pi) - np.pi
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance
    
    def multi_predict(self, mean, covariance):
        """
        mean: (N, 80)
        covariance: (N, 80, 80)
        """
        if mean.shape[0] == 0:
            return mean, covariance

        N = mean.shape[0]

        std_pos = np.array([[self._std_weight_motion_rho, self._std_weight_motion_theta] for _ in range(20)]).flatten()
        std_vel = np.array([[self._std_weight_v_rho, self._std_weight_v_theta] for _ in range(20)]).flatten()
        
        q_single = np.diag(np.square(np.r_[std_pos, std_vel]))
        motion_cov = np.stack([q_single] * N)

        mean = np.dot(mean, self._motion_mat.T)
        for i in range(20):
            theta_idx = i * 2 + 1
            mean[:, theta_idx] = (mean[:, theta_idx] + np.pi) % (2 * np.pi) - np.pi
        left = np.matmul(self._motion_mat, covariance)
        covariance = np.matmul(left, self._motion_mat.T) + motion_cov

        return mean, covariance
    
    def gating_distance(self, mean, covariance, measurements, confidences, metric="maha"):
        """
        measurements: (M, 40)
        confidences: (20,) 
        """
        projected_mean, projected_cov, _ = self.project(mean, covariance, confidences)
        
        d = measurements - projected_mean # (M, 40)
        
        if metric == "maha":
 
            cholesky_factor = np.linalg.cholesky(projected_cov)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)
        
        return np.sum(d**2, axis=1)