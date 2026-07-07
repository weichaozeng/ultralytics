# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, nms, ops
from ultralytics.utils.pose_nms import is_pose_track_tracker, pose_aware_non_max_suppression


class PosePredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a pose model.

    This class specializes in pose estimation, handling keypoints detection alongside standard object detection
    capabilities inherited from DetectionPredictor.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO pose model with keypoint detection capabilities.

    Methods:
        construct_result: Construct the result object from the prediction, including keypoints.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.pose import PosePredictor
        >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
        >>> predictor = PosePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize PosePredictor for pose estimation tasks.

        Sets up a PosePredictor instance, configuring it for pose detection tasks and handling device-specific warnings
        for Apple MPS.

        Args:
            cfg (Any): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.

        Examples:
            >>> from ultralytics.utils import ASSETS
            >>> from ultralytics.models.yolo.pose import PosePredictor
            >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
            >>> predictor = PosePredictor(overrides=args)
            >>> predictor.predict_cli()
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def _use_pose_nms(self) -> bool:
        return getattr(self.args, "mode", None) == "track" and is_pose_track_tracker(getattr(self.args, "tracker", None))

    def _pose_nms_thresholds(self) -> tuple[float, float, float]:
        tracker = getattr(self.args, "tracker", None)
        if not tracker:
            return 0.25, 0.25, 0.65
        try:
            from ultralytics.utils import YAML
            from ultralytics.utils.checks import check_yaml

            cfg = YAML.load(check_yaml(tracker))
            return (
                float(cfg.get("point_thres", 0.25)),
                float(cfg.get("bone_thres", 0.25)),
                float(cfg.get("ioa_thres", 0.65)),
            )
        except Exception:
            return 0.25, 0.25, 0.65

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process predictions, using pose-aware NMS when PoseTrack is active."""
        save_feats = getattr(self, "_feats", None) is not None
        if self._use_pose_nms():
            point_thres, bone_thres, ioa_thres = self._pose_nms_thresholds()
            preds = pose_aware_non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                self.args.classes,
                self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=len(self.model.names),
                end2end=getattr(self.model, "end2end", False),
                rotated=False,
                return_idxs=save_feats,
                point_thres=point_thres,
                bone_thres=bone_thres,
                ioa_thres=ioa_thres,
            )
        else:
            preds = nms.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                self.args.classes,
                self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=len(self.model.names),
                end2end=getattr(self.model, "end2end", False),
                rotated=False,
                return_idxs=save_feats,
            )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f

        return results

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct the result object from the prediction, including keypoints.

        Extends the parent class implementation by extracting keypoint data from predictions and adding them to the
        result object.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and keypoints with shape (N, 6+K*D) where N is
                the number of detections, K is the number of keypoints, and D is the keypoint dimension.
            img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
            orig_img (np.ndarray): The original unprocessed image as a numpy array.
            img_path (str): The path to the original image file.

        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes, and
                keypoints.
        """
        result = super().construct_result(pred, img, orig_img, img_path)
        # Extract keypoints from prediction and reshape according to model's keypoint shape
        pred_kpts = pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)
        # Scale keypoints coordinates to match the original image dimensions
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        return result
