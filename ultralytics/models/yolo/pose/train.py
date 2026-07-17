# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ultralytics.data.spad_pose_dataset import (
    SpadPoseDataset,
    SpadPoseFrameDataset,
    SpadPoseRenderedFrameDataset,
    SpadPoseRenderedSequenceDataset,
    SpadPoseSequenceDataset,
    load_visionsim_split_json,
)
from ultralytics.data.spad_render_cache import build_render_config
from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel, SpadPoseFrameModel, SpadPoseModel, SpadPoseSequenceModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, nms


SPAD_BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


class _SpadNoOpValidator:
    """Placeholder validator; SPAD evaluation is handled by the training callbacks."""

    def __init__(self, args):
        self.args = copy(args)
        self.metrics = type("SpadNoOpMetrics", (), {"keys": []})()

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


class SpadPoseSequenceTrainer(PoseTrainer):
    """Pose trainer that builds SpadPoseModel and freezes the pretrained YOLO detector by default."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize SPAD trainer while allowing custom SPAD args through Ultralytics cfg validation."""
        overrides = overrides or {}
        custom_prefixes = ("spad_", "ppb_", "stea_", "hire_", "ssd_", "attn_")
        if any(str(k).startswith(custom_prefixes) for k in overrides):
            cfg_dict = dict(vars(cfg)) if hasattr(cfg, "__dict__") else dict(cfg)
            cfg_dict.update({k: v for k, v in overrides.items() if str(k).startswith(custom_prefixes)})
            cfg = cfg_dict
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_start", self._spad_on_train_start)
        self.add_callback("on_fit_epoch_end", self._spad_on_fit_epoch_end)

    def get_dataset(self) -> dict[str, Any]:
        """Return a minimal dataset dictionary for SPAD pose training."""
        spad_train_json = getattr(self.args, "spad_train_json", None)
        spad_test_json = getattr(self.args, "spad_test_json", None)
        if spad_train_json and spad_test_json:
            return {
                "train": spad_train_json,
                "val": spad_test_json,
                "nc": 2,
                "names": {0: "left_hand", 1: "right_hand"},
                "channels": 3,
                "kpt_shape": [21, 3],
                "spad_train_json": spad_train_json,
                "spad_test_json": spad_test_json,
            }
        return super().get_dataset()

    @staticmethod
    def _samples_have_explicit_render(samples: list[dict[str, str]], preprocessor_name: str) -> bool:
        key = str(preprocessor_name).strip().lower()
        if not samples:
            return False
        return all(
            bool(
                sample.get(key)
                or sample.get(f"render_{key}")
                or sample.get(f"render_{key}_frames")
            )
            for sample in samples
        )

    def _resolve_spad_cache_mode(self, samples: list[dict[str, str]], preprocessor_name: str) -> str:
        requested = str(getattr(self.args, "spad_cache_mode", "auto")).strip().lower()
        if requested == "raw":
            return "raw"
        if requested == "rendered":
            return "rendered"
        if requested == "auto":
            return "rendered" if self._samples_have_explicit_render(samples, preprocessor_name) else "raw"
        raise ValueError(f"Unsupported spad_cache_mode={requested!r}; expected raw, rendered, or auto.")

    def _build_preprocessor_kwargs(self, preprocessor_name: str, spad_subsampling: int) -> dict[str, Any]:
        preprocessor_name = str(preprocessor_name).strip().lower()
        if preprocessor_name == "ppb":
            return {
                "subsampling": spad_subsampling,
                "bocpd_gamma": float(getattr(self.args, "ppb_bocpd_gamma", getattr(self.args, "bocpd_gamma", 1e-3))),
                "memory_size": int(getattr(self.args, "ppb_memory_size", getattr(self.args, "memory_size", 10))),
                "normalize": bool(getattr(self.args, "ppb_normalize", True)),
                "quantile": float(getattr(self.args, "ppb_quantile", getattr(self.args, "quantile", 1.0))),
                "min_filter_size": int(getattr(self.args, "ppb_min_filter_size", getattr(self.args, "min_filter_size", 5))),
            }
        if preprocessor_name == "stea":
            return {
                "subsampling": spad_subsampling,
                "fast_window": int(getattr(self.args, "stea_fast_window", 16)),
                "slow_window": int(getattr(self.args, "stea_slow_window", 128)),
                "temporal_window": int(getattr(self.args, "stea_temporal_window", 5)),
                "fast_tau": None if getattr(self.args, "stea_fast_tau", None) in {None, 0} else float(getattr(self.args, "stea_fast_tau")),
                "motion_sharpness": float(getattr(self.args, "stea_motion_sharpness", 60.0)),
                "motion_threshold": float(getattr(self.args, "stea_motion_threshold", 0.05)),
                "stable_prior": float(getattr(self.args, "stea_stable_prior", 16.0)),
                "normalize": bool(getattr(self.args, "stea_normalize", True)),
                "quantile": float(getattr(self.args, "stea_quantile", 1.0)),
            }
        if preprocessor_name == "sum":
            return {
                "subsampling": spad_subsampling,
                "normalize": bool(getattr(self.args, "sum_normalize", True)),
                "quantile": float(getattr(self.args, "sum_quantile", 1.0)),
            }
        if preprocessor_name == "ema":
            return {
                "subsampling": spad_subsampling,
                "ema_alpha": float(getattr(self.args, "ema_alpha", 0.0)),
                "normalize": bool(getattr(self.args, "ema_normalize", True)),
                "quantile": float(getattr(self.args, "ema_quantile", 1.0)),
            }
        if preprocessor_name == "hire":
            def _tau_or_none(key: str) -> float | None:
                val = getattr(self.args, key, 0.0)
                if val in {None, 0, 0.0}:
                    return None
                return float(val)

            return {
                "subsampling": spad_subsampling,
                "sample_rate_hz": float(getattr(self.args, "spad_bin_rate_hz", 2000.0)),
                "ref_rate_hz": float(getattr(self.args, "hire_ref_rate_hz", 2000.0)),
                "fast_bins": int(getattr(self.args, "hire_fast_bins", 24)),
                "slow_bins": int(getattr(self.args, "hire_slow_bins", 160)),
                "surprise_bins": int(getattr(self.args, "hire_surprise_bins", 4)),
                "tau_fast": _tau_or_none("hire_tau_fast"),
                "tau_slow": _tau_or_none("hire_tau_slow"),
                "tau_surprise": _tau_or_none("hire_tau_surprise"),
                "mix_hold_bins": int(getattr(self.args, "hire_mix_hold_bins", 80)),
                "mix_bins": float(
                    getattr(
                        self.args,
                        "hire_mix_bins",
                        getattr(self.args, "hire_mix_kappa", getattr(self.args, "hire_gate_theta", 12.0)),
                    )
                ),
                "mix_theta": float(getattr(self.args, "hire_mix_theta", 0.06)),
                "mix_floor": float(getattr(self.args, "hire_mix_floor", -1.0)),
                "theta_on": float(getattr(self.args, "hire_theta_on", 0.08)),
                "theta_off": float(getattr(self.args, "hire_theta_off", 0.02)),
                "theta_grow": float(getattr(self.args, "hire_theta_grow", -1.0)),
                "confirm_bins": int(getattr(self.args, "hire_confirm_bins", 4)),
                "cooldown_bins": int(getattr(self.args, "hire_cooldown_bins", 0)),
                "spatial_kernel": int(getattr(self.args, "hire_spatial_kernel", 5)),
                "gate_pool": str(getattr(self.args, "hire_gate_pool", "max")),
                "reset_open": int(getattr(self.args, "hire_reset_open", 15)),
                "reset_grow": int(getattr(self.args, "hire_reset_grow", 6)),
                "normalize": bool(getattr(self.args, "hire_normalize", True)),
                "quantile": float(getattr(self.args, "hire_quantile", 1.0)),
            }
        raise ValueError(f"Unsupported training preprocessor: {preprocessor_name!r}")

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build SPAD pose dataset for train/val from VisionSIM split JSON."""
        json_path = getattr(self.args, "spad_train_json", None) or self.data.get("spad_train_json")
        if mode != "train":
            json_path = getattr(self.args, "spad_test_json", None) or self.data.get("spad_test_json")
        if not json_path:
            raise ValueError("SpadPoseTrainer requires `spad_train_json` and `spad_test_json` in args or data yaml.")

        samples = load_visionsim_split_json(json_path)
        LOGGER.info(f"Loaded {len(samples)} samples from {json_path} for mode={mode!r}")

        output_frames = int(getattr(self.args, "spad_output_frames", self.data.get("spad_output_frames", 4)))
        spad_bins_per_gt = int(getattr(self.args, "spad_bins_per_gt", self.data.get("spad_bins_per_gt", 64)))
        spad_subsampling = int(getattr(self.args, "spad_subsampling", self.data.get("spad_subsampling", 64)))
        stride_frames = int(getattr(self.args, "spad_stride_frames", self.data.get("spad_stride_frames", 0))) or None
        preprocessor_name = str(getattr(self.args, "spad_preprocessor", "ppb")).strip().lower()
        cache_mode = self._resolve_spad_cache_mode(samples, preprocessor_name)
        LOGGER.info(f"SpadPoseSequenceTrainer cache_mode={cache_mode!r} preprocessor={preprocessor_name!r}")

        if cache_mode == "rendered":
            chunk_size = int(getattr(self.args, "spad_chunk_size", 0)) or (5 * spad_subsampling)
            input_gamma = float(getattr(self.args, "spad_input_gamma", getattr(self.args, "input_gamma", 1.0)))
            render_root = getattr(self.args, "spad_render_root", None)
            stride_bins = chunk_size if stride_frames is None else (int(stride_frames) * spad_bins_per_gt)
            has_explicit_render = self._samples_have_explicit_render(samples, preprocessor_name)
            expected_config = None if has_explicit_render else build_render_config(
                preprocessor=preprocessor_name,
                chunk_size=chunk_size,
                stride_bins=stride_bins,
                spad_bins_per_gt=spad_bins_per_gt,
                packed_ch_order=getattr(self.args, "spad_packed_ch_order", self.data.get("spad_packed_ch_order", "RGB")),
                input_gamma=input_gamma,
                extra_kwargs=self._build_preprocessor_kwargs(preprocessor_name, spad_subsampling),
            )
            return SpadPoseRenderedSequenceDataset(
                samples=samples,
                render_root=render_root,
                preprocessor=preprocessor_name,
                output_frames=output_frames,
                spad_bins_per_gt=spad_bins_per_gt,
                stride_frames=stride_frames,
                image_size=int(getattr(self.args, "spad_image_size", self.data.get("spad_image_size", 512))),
                render_contains_confidence=bool(getattr(self.args, "spad_render_contains_confidence", True)),
                expected_render_config=expected_config,
                source_render_dirname=str(getattr(self.args, "spad_source_render_dirname", "renders-spc8kHz")),
            )

        return SpadPoseDataset(
            samples=samples,
            output_frames=output_frames,
            spad_bins_per_gt=spad_bins_per_gt,
            spad_step=spad_subsampling,
            stride_frames=stride_frames,
            image_size=int(getattr(self.args, "spad_image_size", self.data.get("spad_image_size", 512))),
            packed_ch_order=getattr(self.args, "spad_packed_ch_order", self.data.get("spad_packed_ch_order", "RGB")),
        )

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> SpadPoseSequenceModel:
        """Get SPAD pose model with optional pretrained detector weights."""
        preprocessor_name = str(getattr(self.args, "spad_preprocessor", "ppb")).strip().lower()
        spad_subsampling = int(getattr(self.args, "spad_subsampling", getattr(self.args, "subsampling", 64)))
        spad_bin_rate_hz = float(getattr(self.args, "spad_bin_rate_hz", 8000.0))
        train_json = getattr(self.args, "spad_train_json", None) or self.data.get("spad_train_json")
        samples = load_visionsim_split_json(train_json) if train_json else []
        resolved_cache_mode = self._resolve_spad_cache_mode(samples, preprocessor_name)
        preprocessor_kwargs = self._build_preprocessor_kwargs(preprocessor_name, spad_subsampling)

        plugin_layers = getattr(self.args, "spad_plugin_scales", None)
        if isinstance(plugin_layers, str):
            stripped = plugin_layers.strip().lower()
            if stripped == "backbone":
                plugin_layers = "backbone"
            else:
                plugin_layers = [int(x) for x in plugin_layers.split(",") if x.strip()]

        model = SpadPoseSequenceModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose,
            spad_enabled=bool(getattr(self.args, "spad_enabled", True)),
            preprocessor=preprocessor_name,
            preprocessor_kwargs=preprocessor_kwargs,
            plugin=str(getattr(self.args, "spad_plugin", "temporal_ssd")),
            plugin_layers=plugin_layers,
            temporal_core=str(getattr(self.args, "spad_temporal_core", "ssd")),
            ssd_state_dim=int(getattr(self.args, "ssd_state_dim", 20)),
            ssd_head_divisor=int(getattr(self.args, "ssd_head_divisor", 4)),
            attn_state_dim=int(getattr(self.args, "attn_state_dim", 8)),
            attn_head_divisor=int(getattr(self.args, "attn_head_divisor", 4)),
            attn_dropout=float(getattr(self.args, "attn_dropout", 0.0)),
            attn_sr_ratio=int(getattr(self.args, "attn_sr_ratio", 2)),
            attn_window_size=tuple(getattr(self.args, "attn_window_size", (3, 7, 7))),
            attn_max_history=int(getattr(self.args, "attn_max_history", 20)),
            spatial_reduce_ratio=int(getattr(self.args, "spad_spatial_reduce_ratio", 2)),
            spatial_kernel_size=int(getattr(self.args, "spad_spatial_kernel_size", 3)),
            plugin_alpha_init=float(getattr(self.args, "spad_plugin_alpha_init", 0.0)),
            spad_input_gamma=float(getattr(self.args, "spad_input_gamma", getattr(self.args, "input_gamma", 1.0))),
            spad_bin_rate_hz=spad_bin_rate_hz,
            spad_cache_mode=resolved_cache_mode,
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Set pose attributes and freeze the detector graph unless the user overrides spad_freeze_detector."""
        super().set_model_attributes()
        if bool(getattr(self.args, "spad_freeze_detector", True)):
            self.args.freeze = list(range(len(self.model.model)))
            LOGGER.info("SpadPoseSequenceTrainer: freezing pretrained YOLO detector layers; SPAD modules remain trainable.")

    def preprocess_batch(self, batch: dict) -> dict:
        """Move SPAD video batches to device without applying image-style normalization."""
        if "packed_nch" in batch:
            packed_nch = int(batch["packed_nch"])
            self.model.spad_packed_nch = packed_nch
            if getattr(self, "ema", None) is not None and getattr(self.ema, "ema", None) is not None:
                self.ema.ema.spad_packed_nch = packed_nch
        t_index_ll = batch.get("t_index_ll")
        self.model.spad_pending_t_index_ll = t_index_ll
        if getattr(self, "ema", None) is not None and getattr(self.ema, "ema", None) is not None:
            self.ema.ema.spad_pending_t_index_ll = t_index_ll
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        cached_confidence = batch.get("confidence")
        self.model.spad_cached_confidence_batch = cached_confidence
        if getattr(self, "ema", None) is not None and getattr(self.ema, "ema", None) is not None:
            self.ema.ema.spad_cached_confidence_batch = cached_confidence
        return batch

    def get_validator(self):
        """Return a no-op validator because standard PoseValidator expects image batches."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return _SpadNoOpValidator(self.args)

    def validate(self):
        """Skip built-in validation; SPAD val loss/visualization runs from the training callback."""
        fitness = -float(self.loss.detach().cpu()) if hasattr(self, "loss") else 0.0
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return {}, fitness

    def _spad_on_train_start(self, trainer):
        """Save baseline checkpoint and visualizations before the first training epoch."""
        if RANK in {-1, 0}:
            initial_path = Path(self.save_dir) / "weights" / "initial.pt"
            initial_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": -1, "model": self.model, "train_args": vars(self.args)}, initial_path)
        self._spad_eval_visualize(epoch_idx=0)

    def _spad_on_fit_epoch_end(self, trainer):
        """Run SPAD test split evaluation and visualization after selected epochs."""
        period = int(getattr(self.args, "spad_viz_period", 1))
        if period <= 0 or (self.epoch + 1) % period != 0:
            return
        self._spad_eval_visualize(epoch_idx=self.epoch + 1)

    def _spad_eval_visualize(self, *, epoch_idx: int):
        """Evaluate random test windows and save recon/overlay visualizations on rank 0."""
        if RANK not in {-1, 0}:
            return
        model = self.ema.ema if getattr(self, "ema", None) is not None else self.model
        was_training = model.training
        model.eval()

        dataset = self.test_loader.dataset
        max_batches = int(getattr(self.args, "spad_eval_max_batches", -1))
        eval_count = len(dataset) if max_batches < 0 else min(max_batches, len(dataset))
        eval_count = max(eval_count, 1)
        generator = torch.Generator().manual_seed(int(getattr(self.args, "spad_eval_seed", 0)) + int(epoch_idx))
        indices = torch.randperm(len(dataset), generator=generator)[:eval_count].tolist()
        loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)

        save_dir = Path(self.save_dir) / "spad_viz" / f"epoch{epoch_idx:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        loss_sum_total = 0.0
        loss_items_total = None
        seen_batches = 0
        written = []
        last_processed_count = 0
        last_target_images = 0
        viz_batches = int(getattr(self.args, "spad_viz_batches", 1))
        viz_frames = int(getattr(self.args, "spad_viz_frames", 4))

        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                batch = self.preprocess_batch(batch)
                preds = model(batch["img"])
                loss, loss_items = model.loss(batch, preds)
                processed = self._spad_postprocess_pose(preds)

                loss_sum_total += float(loss.sum().detach().cpu())
                loss_items_cpu = loss_items.detach().cpu()
                loss_items_total = loss_items_cpu if loss_items_total is None else loss_items_total + loss_items_cpu
                seen_batches += 1
                last_processed_count = len(processed)
                last_target_images = int(batch["batch_idx"].max().item() + 1) if batch["batch_idx"].numel() else 0

                if batch_i < viz_batches:
                    num_images = min(viz_frames, max(len(processed), last_target_images))
                    for si in range(num_images):
                        canvas = self._spad_recon_canvas(model, si)
                        cv2.imwrite(str(save_dir / f"batch{batch_i:03d}_sample{si:03d}_recon.png"), canvas)
                        self._spad_draw_labels(canvas, batch, si)
                        if si < len(processed):
                            self._spad_draw_predictions(canvas, processed[si])
                        out_path = save_dir / f"batch{batch_i:03d}_sample{si:03d}_overlay.png"
                        ok = cv2.imwrite(str(out_path), canvas)
                        written.append(f"{out_path.name}: {'ok' if ok else 'failed'}")

        if seen_batches:
            self.metrics["spad_val/loss_sum"] = loss_sum_total / seen_batches
            mean_loss_items = loss_items_total / seen_batches
            for i, value in enumerate(mean_loss_items.tolist()):
                self.metrics[f"spad_val/loss_{i}"] = float(value)
        else:
            mean_loss_items = torch.zeros(5)

        with (save_dir / "summary.txt").open("w", encoding="utf-8") as f:
            f.write(f"epoch_idx: {epoch_idx}\n")
            f.write(f"seen_batches: {seen_batches}\n")
            f.write(f"eval_max_batches: {max_batches}\n")
            f.write(f"random_indices: {indices}\n")
            f.write(f"mean_loss_sum: {self.metrics.get('spad_val/loss_sum', 0.0)}\n")
            f.write(f"mean_loss_items: {mean_loss_items.tolist()}\n")
            f.write(f"last_processed_predictions: {last_processed_count}\n")
            f.write(f"last_target_images: {last_target_images}\n")
            f.write("\n".join(written))
            f.write("\n")

        if was_training:
            model.train()

    def _spad_postprocess_pose(self, preds):
        raw = preds[0] if isinstance(preds, (list, tuple)) and torch.is_tensor(preds[0]) else preds
        outputs = nms.non_max_suppression(
            raw,
            float(getattr(self.args, "spad_viz_conf", 0.25)),
            float(getattr(self.args, "spad_viz_iou", 0.7)),
            nc=self.data["nc"],
            multi_label=True,
            max_det=int(getattr(self.args, "spad_viz_max_det", 20)),
        )
        return [
            {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "keypoints": x[:, 6:].view(-1, 21, 3)}
            for x in outputs
        ]

    def _spad_recon_canvas(self, model, si: int) -> np.ndarray:
        image_size = int(getattr(self.args, "spad_image_size", 512))
        frames = getattr(model, "spad_last_recon_frames", None)
        if frames is None or si >= frames.shape[0]:
            return np.zeros((image_size, image_size, 3), dtype=np.uint8)
        rgb = frames[si, 0].detach().float().cpu().permute(1, 2, 0).numpy()
        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb_u8[:, :, ::-1])

    def _spad_draw_pose(self, img: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int]):
        img = np.ascontiguousarray(img)
        for s, e in SPAD_BONE_CONNECTIONS:
            if keypoints[s, 2] > 0 and keypoints[e, 2] > 0:
                cv2.line(img, tuple(keypoints[s, :2].astype(int)), tuple(keypoints[e, :2].astype(int)), color, 2)
        for x, y, v in keypoints:
            if v > 0:
                cv2.circle(img, (int(x), int(y)), 3, color, -1)

    def _spad_draw_labels(self, img: np.ndarray, batch: dict[str, Any], si: int):
        image_size = int(getattr(self.args, "spad_image_size", 512))
        idx = batch["batch_idx"].view(-1).cpu() == si
        boxes = batch["bboxes"][idx].cpu().numpy()
        cls = batch["cls"][idx].view(-1).cpu().numpy()
        kpts = batch["keypoints"][idx].cpu().numpy()
        for box, cls_id, pose in zip(boxes, cls, kpts):
            cx, cy, w, h = box * image_size
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            color = (0, 0, 255) if int(cls_id) == 0 else (255, 0, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            self._spad_draw_pose(img, pose * np.array([image_size, image_size, 1.0]), color)

    def _spad_draw_predictions(self, img: np.ndarray, pred: dict[str, torch.Tensor]):
        boxes = pred["bboxes"].detach().cpu().numpy()
        scores = pred["conf"].detach().cpu().numpy()
        cls = pred["cls"].detach().cpu().numpy()
        keypoints = pred["keypoints"].detach().cpu().numpy()
        conf = float(getattr(self.args, "spad_viz_conf", 0.25))
        for box, score, cls_id, pose in zip(boxes, scores, cls, keypoints):
            if score < conf:
                continue
            color = (0, 255, 255) if int(cls_id) == 0 else (255, 255, 0)
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv2.putText(img, f"{int(cls_id)} {score:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            self._spad_draw_pose(img, pose, color)


class SpadPoseFrameTrainer(SpadPoseSequenceTrainer):
    """Pose trainer that collapses each raw SPAD chunk into one end-of-chunk detector frame."""

    def _resolve_frame_cache_mode(self, samples: list[dict[str, str]], preprocessor_name: str) -> str:
        return self._resolve_spad_cache_mode(samples, preprocessor_name)

    def _build_frame_preprocessor_kwargs(self, preprocessor_name: str, spad_subsampling: int) -> dict[str, Any]:
        return self._build_preprocessor_kwargs(preprocessor_name, spad_subsampling)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build frame-mode SPAD pose dataset for train/val from VisionSIM split JSON."""
        json_path = getattr(self.args, "spad_train_json", None) or self.data.get("spad_train_json")
        if mode != "train":
            json_path = getattr(self.args, "spad_test_json", None) or self.data.get("spad_test_json")
        if not json_path:
            raise ValueError("SpadPoseFrameTrainer requires `spad_train_json` and `spad_test_json` in args or data yaml.")

        samples = load_visionsim_split_json(json_path)
        LOGGER.info(f"Loaded {len(samples)} samples from {json_path} for mode={mode!r}")

        subsampling = int(getattr(self.args, "spad_subsampling", self.data.get("spad_subsampling", 64)))
        legacy_output_frames = int(getattr(self.args, "spad_output_frames", self.data.get("spad_output_frames", 4)))
        chunk_size = int(getattr(self.args, "spad_chunk_size", 0)) or (legacy_output_frames * subsampling)
        preprocessor_name = str(getattr(self.args, "spad_preprocessor", "stea")).strip().lower()
        cache_mode = self._resolve_frame_cache_mode(samples, preprocessor_name)
        input_gamma = float(getattr(self.args, "spad_input_gamma", getattr(self.args, "input_gamma", 1.0)))
        if cache_mode == "rendered":
            render_root = getattr(self.args, "spad_render_root", None)
            stride_frames = int(getattr(self.args, "spad_stride_frames", self.data.get("spad_stride_frames", 0))) or None
            stride_bins = chunk_size if stride_frames is None else (int(stride_frames) * int(getattr(self.args, "spad_bins_per_gt", self.data.get("spad_bins_per_gt", 64))))
            has_explicit_render = self._samples_have_explicit_render(samples, preprocessor_name)
            expected_config = None if has_explicit_render else build_render_config(
                preprocessor=preprocessor_name,
                chunk_size=chunk_size,
                stride_bins=stride_bins,
                spad_bins_per_gt=int(getattr(self.args, "spad_bins_per_gt", self.data.get("spad_bins_per_gt", 64))),
                packed_ch_order=getattr(self.args, "spad_packed_ch_order", self.data.get("spad_packed_ch_order", "RGB")),
                input_gamma=input_gamma,
                extra_kwargs=self._build_frame_preprocessor_kwargs(preprocessor_name, subsampling),
            )
            return SpadPoseRenderedFrameDataset(
                samples=samples,
                render_root=render_root,
                preprocessor=preprocessor_name,
                image_size=int(getattr(self.args, "spad_image_size", self.data.get("spad_image_size", 512))),
                render_contains_confidence=bool(getattr(self.args, "spad_render_contains_confidence", True)),
                expected_render_config=expected_config,
                source_render_dirname=str(getattr(self.args, "spad_source_render_dirname", "renders-spc8kHz")),
            )

        return SpadPoseFrameDataset(
            samples=samples,
            chunk_size=chunk_size,
            spad_bins_per_gt=int(getattr(self.args, "spad_bins_per_gt", self.data.get("spad_bins_per_gt", 64))),
            stride_frames=int(getattr(self.args, "spad_stride_frames", self.data.get("spad_stride_frames", 0))) or None,
            image_size=int(getattr(self.args, "spad_image_size", self.data.get("spad_image_size", 512))),
            packed_ch_order=getattr(self.args, "spad_packed_ch_order", self.data.get("spad_packed_ch_order", "RGB")),
        )

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> SpadPoseFrameModel:
        """Get frame-mode SPAD pose model with optional UA adapter."""
        preprocessor_name = str(getattr(self.args, "spad_preprocessor", "stea")).strip().lower()
        train_json = getattr(self.args, "spad_train_json", None) or self.data.get("spad_train_json")
        samples = load_visionsim_split_json(train_json) if train_json else []
        resolved_cache_mode = self._resolve_frame_cache_mode(samples, preprocessor_name)

        spad_subsampling = int(getattr(self.args, "spad_subsampling", getattr(self.args, "subsampling", 64)))
        spad_bin_rate_hz = float(getattr(self.args, "spad_bin_rate_hz", 8000.0))
        legacy_output_frames = int(getattr(self.args, "spad_output_frames", 4))
        chunk_size = int(getattr(self.args, "spad_chunk_size", 0)) or (legacy_output_frames * spad_subsampling)
        preprocessor_kwargs = self._build_frame_preprocessor_kwargs(preprocessor_name, spad_subsampling)

        model = SpadPoseFrameModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose,
            spad_enabled=bool(getattr(self.args, "spad_enabled", True)),
            preprocessor=preprocessor_name,
            preprocessor_kwargs=preprocessor_kwargs,
            frame_adapter=str(getattr(self.args, "spad_frame_adapter", "ua")),
            frame_adapter_kernel_size=int(getattr(self.args, "spad_spatial_kernel_size", 3)),
            frame_adapter_alpha_init=float(
                getattr(self.args, "spad_frame_adapter_alpha_init", getattr(self.args, "spad_plugin_alpha_init", 0.0))
            ),
            spad_chunk_size=chunk_size,
            spad_cache_mode=resolved_cache_mode,
            spad_input_gamma=float(getattr(self.args, "spad_input_gamma", getattr(self.args, "input_gamma", 1.0))),
            spad_bin_rate_hz=spad_bin_rate_hz,
        )
        if weights:
            model.load(weights)
        return model

    def _spad_recon_canvas(self, model, si: int) -> np.ndarray:
        """Visualize the sole reconstructed frame from each frame-mode batch item."""
        image_size = int(getattr(self.args, "spad_image_size", 512))
        frames = getattr(model, "spad_last_recon_frames", None)
        if frames is None or frames.ndim != 5 or frames.shape[1] == 0:
            return np.zeros((image_size, image_size, 3), dtype=np.uint8)
        if si >= frames.shape[1]:
            return np.zeros((image_size, image_size, 3), dtype=np.uint8)
        rgb = frames[0, si].detach().float().cpu().permute(1, 2, 0).numpy()
        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb_u8[:, :, ::-1])


SpadPoseTrainer = SpadPoseSequenceTrainer
