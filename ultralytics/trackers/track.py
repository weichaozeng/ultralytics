# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from functools import partial
from pathlib import Path

import torch

from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from .utils.result_layout import apply_pose_tracks_to_result
from .byte_tracker import BYTETracker
from .spad_tracker import SPADTracker
from .spad_pose_track import SPADPoseTrack
from .pose_track import PoseTrack
from .bot_sort import BOTSORT

# A mapping of tracker types to corresponding tracker classes
TRACKER_MAP = {
    "bytetrack": BYTETracker,
    "botsort": BOTSORT,
    "spad_tracker": SPADTracker,
    "posetrack": PoseTrack,
    "spad_posetrack": SPADPoseTrack,
}


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """Initialize trackers for object tracking during prediction.

    Args:
        predictor (ultralytics.engine.predictor.BasePredictor): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist.

    Examples:
        Initialize trackers for a predictor object
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if predictor.args.task == "classify":
        raise ValueError("❌ Classification doesn't support 'mode=track'")

    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**YAML.load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort", "spad_tracker", "posetrack", "spad_posetrack"}:
        raise AssertionError(
            f"Only 'bytetrack', 'botsort', 'spad_tracker', 'posetrack', and 'spad_posetrack' are supported "
            f"for now, but got '{cfg.tracker_type}'"
        )

    predictor._feats = None  # reset in case used earlier
    if hasattr(predictor, "_hook"):
        predictor._hook.remove()
    if cfg.tracker_type == "botsort" and cfg.with_reid and cfg.model == "auto":
        from ultralytics.nn.modules.head import Detect

        if not (
            isinstance(predictor.model.model, torch.nn.Module)
            and isinstance(predictor.model.model.model[-1], Detect)
            and not predictor.model.model.model[-1].end2end
        ):
            cfg.model = "yolo11n-cls.pt"
        else:
            # Register hook to extract input of Detect layer
            def pre_hook(module, input):
                predictor._feats = list(input[0])  # unroll to new list to avoid mutation in forward

            predictor._hook = predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)

    trackers = []
    class_names = getattr(getattr(predictor, "model", None), "names", None)
    for _ in range(predictor.dataset.bs):
        tracker_cls = TRACKER_MAP[cfg.tracker_type]
        if cfg.tracker_type in {"posetrack", "spad_posetrack"}:
            trackers.append(tracker_cls(args=cfg, frame_rate=30, class_names=class_names))
        else:
            trackers.append(tracker_cls(args=cfg, frame_rate=30))
        if predictor.dataset.mode != "stream":  # only need one tracker for other modes
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # for determining when to reset tracker on new video


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool, optional): Whether to persist the trackers if they already exist.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    for i, result in enumerate(predictor.results):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(result.path).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (result.obb if is_obb else result.boxes).cpu().numpy()
        keypoints = None
        if predictor.args.task == "pose" and getattr(result, "keypoints", None) is not None and len(result.keypoints):
            keypoints = result.keypoints.data.cpu().numpy()

        if keypoints is not None and hasattr(tracker, "n_keypoints"):
            tracks = tracker.update(det, result.orig_img, getattr(result, "feats", None), keypoints=keypoints)
        else:
            tracks = tracker.update(det, result.orig_img, getattr(result, "feats", None))
        if len(tracks) == 0:
            continue

        if predictor.args.task == "pose" and tracks.shape[1] > 8:
            n_keypoints = int(getattr(tracker, "n_keypoints", 21))
            kpt_dims = int(getattr(tracker, "kpt_dims", 3))
            idx = tracks[:, 7].astype(int)
            predictor.results[i] = result[idx]
            predictor.results[i] = apply_pose_tracks_to_result(
                predictor.results[i], tracks, n_keypoints=n_keypoints, kpt_dims=kpt_dims
            )
        else:
            idx = tracks[:, -1].astype(int)
            predictor.results[i] = result[idx]
            update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
            predictor.results[i].update(**update_args)


def register_tracker(model: object, persist: bool) -> None:
    """Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
