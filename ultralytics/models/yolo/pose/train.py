# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ultralytics.data.qnn_spad_dataset import QNNSpadPoseDataset, load_visionsim_split_json
from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel, QNNPoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, nms


QNN_BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


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
        self.add_callback("on_train_start", self._qnn_on_train_start)
        self.add_callback("on_fit_epoch_end", self._qnn_on_fit_epoch_end)

    def get_dataset(self) -> dict[str, Any]:
        """Return a minimal dataset dictionary for QNN SPAD training."""
        qnn_train_json = getattr(self.args, "qnn_train_json", None)
        qnn_test_json = getattr(self.args, "qnn_test_json", None)
        if qnn_train_json and qnn_test_json:
            return {
                "train": qnn_train_json,
                "val": qnn_test_json,
                "nc": 2,
                "names": {0: "left_hand", 1: "right_hand"},
                "channels": 3,
                "kpt_shape": [21, 3],
                "qnn_train_json": qnn_train_json,
                "qnn_test_json": qnn_test_json,
            }
        return super().get_dataset()

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build QNN SPAD pose dataset for train/val from VisionSIM split JSON."""
        json_path = getattr(self.args, "qnn_train_json", None) or self.data.get("qnn_train_json")
        if mode != "train":
            json_path = getattr(self.args, "qnn_test_json", None) or self.data.get("qnn_test_json")
        if not json_path:
            raise ValueError("QNNPoseTrainer requires `qnn_train_json` and `qnn_test_json` in args or data yaml.")

        samples = load_visionsim_split_json(json_path)
        LOGGER.info(f"Loaded {len(samples)} samples from {json_path} for mode={mode!r}")

        return QNNSpadPoseDataset(
            samples=samples,
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
        if "packed_nch" in batch:
            packed_nch = int(batch["packed_nch"])
            self.model.qnn_packed_nch = packed_nch
            if getattr(self, "ema", None) is not None and getattr(self.ema, "ema", None) is not None:
                self.ema.ema.qnn_packed_nch = packed_nch
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

    def _qnn_on_train_start(self, trainer):
        """Save baseline checkpoint and visualizations before the first training epoch."""
        if RANK in {-1, 0}:
            initial_path = Path(self.save_dir) / "weights" / "initial.pt"
            initial_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": -1, "model": self.model, "train_args": vars(self.args)}, initial_path)
        self._qnn_eval_visualize(epoch_idx=0)

    def _qnn_on_fit_epoch_end(self, trainer):
        """Run QNN test split evaluation and visualization after selected epochs."""
        period = int(getattr(self.args, "qnn_viz_period", 1))
        if period <= 0 or (self.epoch + 1) % period != 0:
            return
        self._qnn_eval_visualize(epoch_idx=self.epoch + 1)

    def _qnn_eval_visualize(self, *, epoch_idx: int):
        """Evaluate random test windows and save recon/overlay visualizations on rank 0."""
        if RANK not in {-1, 0}:
            return
        model = self.ema.ema if getattr(self, "ema", None) is not None else self.model
        was_training = model.training
        model.eval()

        dataset = self.test_loader.dataset
        max_batches = int(getattr(self.args, "qnn_eval_max_batches", -1))
        eval_count = len(dataset) if max_batches < 0 else min(max_batches, len(dataset))
        eval_count = max(eval_count, 1)
        generator = torch.Generator().manual_seed(int(getattr(self.args, "qnn_eval_seed", 0)) + int(epoch_idx))
        indices = torch.randperm(len(dataset), generator=generator)[:eval_count].tolist()
        loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)

        save_dir = Path(self.save_dir) / "qnn_viz" / f"epoch{epoch_idx:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        loss_sum_total = 0.0
        loss_items_total = None
        seen_batches = 0
        written = []
        last_processed_count = 0
        last_target_images = 0
        viz_batches = int(getattr(self.args, "qnn_viz_batches", 1))
        viz_frames = int(getattr(self.args, "qnn_viz_frames", 4))

        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                batch = self.preprocess_batch(batch)
                preds = model(batch["img"])
                loss, loss_items = model.loss(batch, preds)
                processed = self._qnn_postprocess_pose(preds)

                loss_sum_total += float(loss.sum().detach().cpu())
                loss_items_cpu = loss_items.detach().cpu()
                loss_items_total = loss_items_cpu if loss_items_total is None else loss_items_total + loss_items_cpu
                seen_batches += 1
                last_processed_count = len(processed)
                last_target_images = int(batch["batch_idx"].max().item() + 1) if batch["batch_idx"].numel() else 0

                if batch_i < viz_batches:
                    num_images = min(viz_frames, max(len(processed), last_target_images))
                    for si in range(num_images):
                        canvas = self._qnn_recon_canvas(model, si)
                        cv2.imwrite(str(save_dir / f"batch{batch_i:03d}_sample{si:03d}_recon.png"), canvas)
                        self._qnn_draw_labels(canvas, batch, si)
                        if si < len(processed):
                            self._qnn_draw_predictions(canvas, processed[si])
                        out_path = save_dir / f"batch{batch_i:03d}_sample{si:03d}_overlay.png"
                        ok = cv2.imwrite(str(out_path), canvas)
                        written.append(f"{out_path.name}: {'ok' if ok else 'failed'}")

        if seen_batches:
            self.metrics["qnn_val/loss_sum"] = loss_sum_total / seen_batches
            mean_loss_items = loss_items_total / seen_batches
            for i, value in enumerate(mean_loss_items.tolist()):
                self.metrics[f"qnn_val/loss_{i}"] = float(value)
        else:
            mean_loss_items = torch.zeros(5)

        with (save_dir / "summary.txt").open("w", encoding="utf-8") as f:
            f.write(f"epoch_idx: {epoch_idx}\n")
            f.write(f"seen_batches: {seen_batches}\n")
            f.write(f"eval_max_batches: {max_batches}\n")
            f.write(f"random_indices: {indices}\n")
            f.write(f"mean_loss_sum: {self.metrics.get('qnn_val/loss_sum', 0.0)}\n")
            f.write(f"mean_loss_items: {mean_loss_items.tolist()}\n")
            f.write(f"last_processed_predictions: {last_processed_count}\n")
            f.write(f"last_target_images: {last_target_images}\n")
            f.write("\n".join(written))
            f.write("\n")

        if was_training:
            model.train()

    def _qnn_postprocess_pose(self, preds):
        raw = preds[0] if isinstance(preds, (list, tuple)) and torch.is_tensor(preds[0]) else preds
        outputs = nms.non_max_suppression(
            raw,
            float(getattr(self.args, "qnn_viz_conf", 0.25)),
            float(getattr(self.args, "qnn_viz_iou", 0.7)),
            nc=self.data["nc"],
            multi_label=True,
            max_det=int(getattr(self.args, "qnn_viz_max_det", 20)),
        )
        return [
            {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "keypoints": x[:, 6:].view(-1, 21, 3)}
            for x in outputs
        ]

    def _qnn_recon_canvas(self, model, si: int) -> np.ndarray:
        image_size = int(getattr(self.args, "qnn_image_size", 512))
        frames = getattr(model, "qnn_last_recon_frames", None)
        if frames is None or si >= frames.shape[0]:
            return np.zeros((image_size, image_size, 3), dtype=np.uint8)
        rgb = frames[si, 0].detach().float().cpu().permute(1, 2, 0).numpy()
        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb_u8[:, :, ::-1])

    def _qnn_draw_pose(self, img: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int]):
        img = np.ascontiguousarray(img)
        for s, e in QNN_BONE_CONNECTIONS:
            if keypoints[s, 2] > 0 and keypoints[e, 2] > 0:
                cv2.line(img, tuple(keypoints[s, :2].astype(int)), tuple(keypoints[e, :2].astype(int)), color, 2)
        for x, y, v in keypoints:
            if v > 0:
                cv2.circle(img, (int(x), int(y)), 3, color, -1)

    def _qnn_draw_labels(self, img: np.ndarray, batch: dict[str, Any], si: int):
        image_size = int(getattr(self.args, "qnn_image_size", 512))
        idx = batch["batch_idx"].view(-1).cpu() == si
        boxes = batch["bboxes"][idx].cpu().numpy()
        cls = batch["cls"][idx].view(-1).cpu().numpy()
        kpts = batch["keypoints"][idx].cpu().numpy()
        for box, cls_id, pose in zip(boxes, cls, kpts):
            cx, cy, w, h = box * image_size
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            color = (0, 0, 255) if int(cls_id) == 0 else (255, 0, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            self._qnn_draw_pose(img, pose * np.array([image_size, image_size, 1.0]), color)

    def _qnn_draw_predictions(self, img: np.ndarray, pred: dict[str, torch.Tensor]):
        boxes = pred["bboxes"].detach().cpu().numpy()
        scores = pred["conf"].detach().cpu().numpy()
        cls = pred["cls"].detach().cpu().numpy()
        keypoints = pred["keypoints"].detach().cpu().numpy()
        conf = float(getattr(self.args, "qnn_viz_conf", 0.25))
        for box, score, cls_id, pose in zip(boxes, scores, cls, keypoints):
            if score < conf:
                continue
            color = (0, 255, 255) if int(cls_id) == 0 else (255, 255, 0)
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv2.putText(img, f"{int(cls_id)} {score:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            self._qnn_draw_pose(img, pose, color)
