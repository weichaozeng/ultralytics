
import numpy as np


from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT, BOTrack





class PTrack(BOTrack):
    def __init__(self, xywh: np.ndarray, score: float, cls: int, pxy: np.ndarray, pscore: np.ndarray, feat: np.ndarray | None = None, feat_history: int = 50
    ):
        """
        Args:
            xywh (np.ndarray): Bounding box coordinates in xywh format (center x, center y, width, height).
            score (float): Confidence score of the detection.
            cls (int): Class ID of the detected object.
            pxyxy (np.ndarray): Pose coordinates in xy format.
            pscore (float): Confidence score of the pose.
            feat (np.ndarray, optional): Feature vector associated with the detection.
            feat_history (int): Maximum length of the feature history deque.
        """
        super().__init__(xywh, score, cls, feat, feat_history)
    
    



class PoseTracker(BOTSORT):
    