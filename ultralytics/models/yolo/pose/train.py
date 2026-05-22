# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data.qnn_spad_dataset import QNNSpadPoseDataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel, QNNPoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER


class _QNNNoOpValidator:
    """Placeholder validator; QNN evaluation is handled by train_qnn_pose.py callbacks."""

    def __init__(self, args):
        self.args = copy(args)
        self.metrics = type("QNNNoOpMetrics", (), {"keys": []})()

    def __call__(self, *args, **kwargs):
        return {"fitness": 0.0}


class PoseTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training YOLO pose estimation models.

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization
    of pose keypoints alongside bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (PoseModel): The pose estimation model being trained.
        data (dict): Dataset configuration including keypoint shape information.
        loss_names (tuple): Names of the loss components used in training.

    Methods:
        get_model: Retrieve a pose estimation model with specified configuration.
        set_model_attributes: Set keypoints shape attribute on the model.
        get_validator: Create a validator instance for model evaluation.
        plot_training_samples: Visualize training samples with keypoints.
        get_dataset: Retrieve the dataset and ensure it contains required kpt_shape key.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize a PoseTrainer object for training YOLO pose estimation models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Notes:
            This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
            A warning is issued when using Apple MPS device due to known bugs with pose models.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PoseModel:
        """Get pose estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (PoseModel): Initialized pose estimation model.
        """
        model = PoseModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Set keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        kpt_names = self.data.get("kpt_names")
        if not kpt_names:
            names = list(map(str, range(self.model.kpt_shape[0])))
            kpt_names = {i: names for i in range(self.model.nc)}
        self.model.kpt_names = kpt_names

    def get_validator(self):
        """Return an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """Retrieve the dataset and ensure it contains the required `kpt_shape` key.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.

        Raises:
            KeyError: If the `kpt_shape` key is not present in the dataset.
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose/")
        return data


class QNNPoseTrainer(PoseTrainer):
    """Pose trainer that builds QNNPoseModel and freezes the pretrained YOLO detector by default."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize QNN trainer while allowing custom qnn_* args through Ultralytics cfg validation."""
        overrides = overrides or {}
        if any(str(k).startswith("qnn_") for k in overrides):
            cfg_dict = dict(vars(cfg)) if hasattr(cfg, "__dict__") else dict(cfg)
            cfg_dict.update({k: v for k, v in overrides.items() if str(k).startswith("qnn_")})
            cfg = cfg_dict
        super().__init__(cfg, overrides, _callbacks)

    def get_dataset(self) -> dict[str, Any]:
        """Return a minimal dataset dictionary for QNN SPAD training."""
        qnn_gt_root = getattr(self.args, "qnn_gt_root", None)
        qnn_spad_root = getattr(self.args, "qnn_spad_root", None)
        if qnn_gt_root and qnn_spad_root:
            return {
                "train": qnn_gt_root,
                "val": qnn_gt_root,
                "nc": 2,
                "names": {0: "left_hand", 1: "right_hand"},
                "channels": 3,
                "kpt_shape": [21, 3],
                "qnn_gt_root": qnn_gt_root,
                "qnn_spad_root": qnn_spad_root,
            }
        return super().get_dataset()

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build QNN SPAD pose dataset for train/val."""
        gt_root = getattr(self.args, "qnn_gt_root", None) or self.data.get("qnn_gt_root") or img_path
        spad_root = getattr(self.args, "qnn_spad_root", None) or self.data.get("qnn_spad_root")
        if not spad_root:
            raise ValueError("QNNPoseTrainer requires `qnn_spad_root` in args or data yaml.")

        return QNNSpadPoseDataset(
            gt_root=gt_root,
            spad_root=spad_root,
            split="train" if mode == "train" else "val",
            test_keywords=getattr(self.args, "qnn_test_keywords", self.data.get("qnn_test_keywords", None)),
            test_fraction=float(getattr(self.args, "qnn_test_fraction", self.data.get("qnn_test_fraction", 0.2))),
            split_seed=int(getattr(self.args, "qnn_split_seed", self.data.get("qnn_split_seed", 0))),
            output_frames=int(getattr(self.args, "qnn_output_frames", self.data.get("qnn_output_frames", 4))),
            spad_per_gt=int(getattr(self.args, "qnn_spad_per_gt", self.data.get("qnn_spad_per_gt", 64))),
            spad_step=int(getattr(self.args, "qnn_subsampling", self.data.get("qnn_subsampling", 64))),
            stride_frames=int(getattr(self.args, "qnn_stride_frames", self.data.get("qnn_stride_frames", 0))) or None,
            image_size=int(getattr(self.args, "qnn_image_size", self.data.get("qnn_image_size", 512))),
            packed_ch_order=getattr(self.args, "qnn_packed_ch_order", self.data.get("qnn_packed_ch_order", "RGB")),
        )

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> QNNPoseModel:
        """Get QNN-augmented pose model with optional pretrained detector weights."""
        qnn_integrator_kwargs = {
            "subsampling": int(getattr(self.args, "qnn_subsampling", getattr(self.args, "subsampling", 64))),
            "bocpd_gamma": float(getattr(self.args, "qnn_bocpd_gamma", getattr(self.args, "bocpd_gamma", 5e-4))),
            "normalize": bool(getattr(self.args, "qnn_normalize", True)),
            "quantile": float(getattr(self.args, "qnn_quantile", getattr(self.args, "quantile", 1.0))),
            "min_filter_size": int(getattr(self.args, "qnn_min_filter_size", getattr(self.args, "min_filter_size", 7))),
        }
        qnn_ssd_after_layers = getattr(self.args, "qnn_ssd_after_layers", None)
        if isinstance(qnn_ssd_after_layers, str):
            qnn_ssd_after_layers = [int(x) for x in qnn_ssd_after_layers.split(",") if x.strip()]

        model = QNNPoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose,
            qnn_enabled=bool(getattr(self.args, "qnn_enabled", True)),
            qnn_integrator_kwargs=qnn_integrator_kwargs,
            qnn_ssd_after_layers=qnn_ssd_after_layers,
            qnn_ssd_state_dim=int(getattr(self.args, "qnn_ssd_state_dim", 8)),
            qnn_ssd_head_divisor=int(getattr(self.args, "qnn_ssd_head_divisor", 4)),
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Set pose attributes and freeze the detector graph unless the user overrides qnn_freeze_detector."""
        super().set_model_attributes()
        if bool(getattr(self.args, "qnn_freeze_detector", True)):
            self.args.freeze = list(range(len(self.model.model)))
            LOGGER.info("QNNPoseTrainer: freezing pretrained YOLO detector layers; QNN modules remain trainable.")

    def preprocess_batch(self, batch: dict) -> dict:
        """Move QNN video batches to device without applying image-style normalization."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def get_validator(self):
        """Return a no-op validator because standard PoseValidator expects image batches."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return _QNNNoOpValidator(self.args)

    def validate(self):
        """Skip built-in validation; QNN val loss/visualization runs from the training callback."""
        fitness = -float(self.loss.detach().cpu()) if hasattr(self, "loss") else 0.0
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return {}, fitness
