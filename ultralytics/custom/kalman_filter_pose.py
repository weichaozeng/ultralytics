import numpy as np
import scipy.linalg

from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH


class KalmanFilterPose(KalmanFilterXYAH):