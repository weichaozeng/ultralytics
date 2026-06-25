"""Train a SPAD pose model on VisionSIM SPAD windows.

Start small. A raw SPAD window is large after unpacking, so debug with
`--spad-output-frames 1` or `4` and `--batch 1` first.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ultralytics.models.yolo.pose import SpadPoseTrainer
from ultralytics.utils import DEFAULT_CFG_DICT


DEFAULT_TRAIN_JSON = "/home/zvc/Data/visionsim/outputs/train.json"
DEFAULT_TEST_JSON = "/home/zvc/Data/visionsim/outputs/test.json"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train SpadPoseModel on VisionSIM SPAD data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", type=str, required=True, help="Path to pretrained YOLO pose checkpoint (.pt)")
    ap.add_argument(
        "--train-json",
        type=str,
        default=DEFAULT_TRAIN_JSON,
        help="VisionSIM train split JSON from build_visionsim_split.py",
    )
    ap.add_argument(
        "--test-json",
        type=str,
        default=DEFAULT_TEST_JSON,
        help="VisionSIM test split JSON from build_visionsim_split.py",
    )
    ap.add_argument("--project", type=str, default="Runs/spad_pose", help="Ultralytics project directory for runs")
    ap.add_argument("--name", type=str, default="debug", help="Run name under --project")
    ap.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device id(s), e.g. 0 or 0,1; multi-GPU uses DDP and requires --batch >= number of GPUs",
    )
    ap.add_argument("--epochs", type=int, default=30, help="Training epochs")
    ap.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Global batch size (split across GPUs under DDP); use >=2 with --device 0,1",
    )
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    ap.add_argument("--imgsz", type=int, default=512, help="Square image size for normalized bbox/keypoint labels")
    ap.add_argument("--lr0", type=float, default=1e-3, help="Initial learning rate")
    ap.add_argument("--weight-decay", type=float, default=5e-4, help="Optimizer weight decay")
    ap.add_argument("--amp", action="store_true", help="Enable automatic mixed precision training")
    ap.add_argument(
        "--val",
        action="store_true",
        help="Also run the built-in Ultralytics validator (experimental for SPAD batches)",
    )
    ap.add_argument(
        "--preprocessor",
        type=str,
        default="ppb",
        choices=["ppb", "sum", "stea"],
        help="SPAD preprocessor used to reconstruct frame sequences before the detector.",
    )
    ap.add_argument(
        "--spad-output-frames",
        type=int,
        default=10,
        help=(
            "Reconstructed output frames (and pose labels) per training window. "
            "Each frame needs raw SPAD bins; use 1 or 4 for smoke tests before scaling up."
        ),
    )
    ap.add_argument(
        "--spad-stride-frames",
        type=int,
        default=64,
        help=(
            "Sliding-window stride in GT frame units when enumerating training windows. "
            "0 falls back to --spad-output-frames."
        ),
    )
    ap.add_argument(
        "--spad-bins-per-gt",
        type=int,
        default=64,
        help=(
            "Raw SPAD time bins per one GT annotation frame. "
            "Controls GT/SPAD temporal alignment (spad_start = gt_start * this value)."
            "8,000 / 125 = 64"
        ),
    )
    ap.add_argument(
        "--spad-packed-ch-order",
        type=str,
        default="RGB",
        choices=["RGB", "BGR"],
        help="Channel order used when unpacking packed SPAD frames into Bayer raw video.",
    )
    ap.add_argument(
        "--spad-subsampling",
        type=int,
        default=320,
        help=(
            "Shared temporal downsampling factor: (1) preprocessor raw-bin aggregation per "
            "reconstructed frame; (2) SPAD bins between consecutive output-frame labels "
            "(spad_len = output_frames * subsampling)."
        ),
    )
    ap.add_argument(
        "--ppb-bocpd-gamma",
        type=float,
        default=5e-4,
        help="PerPixelBayesian BOCPD hazard rate; higher values adapt faster to photon-rate changes",
    )
    ap.add_argument(
        "--ppb-quantile",
        type=float,
        default=1.0,
        help="PPB normalization upper quantile (1.0 = use max; lower values suppress hot pixels)",
    )
    ap.add_argument(
        "--ppb-min-filter-size",
        type=int,
        default=7,
        help="Odd kernel size for min-filter smoothing of per-pixel runlength estimates in PPB",
    )
    ap.add_argument(
        "--stea-fast-window",
        type=int,
        default=16,
        help="Fast Gamma temporal basis length for STEA.",
    )
    ap.add_argument(
        "--stea-slow-window",
        type=int,
        default=128,
        help="Slow boxcar temporal basis length for STEA.",
    )
    ap.add_argument(
        "--stea-temporal-window",
        type=int,
        default=5,
        help="Causal evidence time blur window for STEA.",
    )
    ap.add_argument(
        "--stea-fast-tau",
        type=float,
        default=6.0,
        help="Gamma kernel tau for the fast STEA basis.",
    )
    ap.add_argument(
        "--stea-motion-sharpness",
        type=float,
        default=60.0,
        help="Sigmoid sharpness used by STEA's motion routing probability.",
    )
    ap.add_argument(
        "--stea-motion-threshold",
        type=float,
        default=0.05,
        help="KL threshold used by STEA's motion routing probability.",
    )
    ap.add_argument(
        "--stea-stable-prior",
        type=float,
        default=16.0,
        help="Bayesian prior strength on the stable branch in STEA.",
    )
    ap.add_argument(
        "--plugin",
        type=str,
        default="temporal_ssd",
        choices=["none", "temporal_ssd", "spatial_only", "spatial_temporal"],
        help="Detector-side feature plugin applied on the selected YOLO scales.",
    )
    ap.add_argument(
        "--plugin-scales",
        type=str,
        default="16,19,22",
        help="Comma-separated YOLO layer indices after which detector plugins are inserted (default: P3/P4/P5).",
    )
    ap.add_argument(
        "--temporal-core",
        type=str,
        default="ssd",
        choices=["ssd"],
        help="Temporal core used by detector plugins that include a temporal branch.",
    )
    ap.add_argument(
        "--ssd-state-dim",
        type=int,
        default=8,
        help="Hidden state dimension of each SSD (semi-separable dynamics) temporal block",
    )
    ap.add_argument(
        "--ssd-head-divisor",
        type=int,
        default=4,
        help="SSD head_dim = YOLO feature channels // this divisor (must divide channels evenly)",
    )
    ap.add_argument(
        "--spatial-reduce-ratio",
        type=int,
        default=2,
        help="Channel bottleneck ratio for spatial detector plugins.",
    )
    ap.add_argument(
        "--spatial-kernel-size",
        type=int,
        default=3,
        help="Depthwise spatial kernel size for spatial detector plugins.",
    )
    ap.add_argument(
        "--plugin-alpha-init",
        type=float,
        default=0.0,
        help="Initial residual gate value for spatial detector plugins.",
    )
    ap.add_argument(
        "--freeze-detector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze pretrained YOLO detector layers; only SPAD modules are trained",
    )
    ap.add_argument(
        "--viz-period",
        type=int,
        default=5,
        help="Run SPAD eval visualization every N epochs (0 disables periodic viz; epoch 0 baseline always runs)",
    )
    ap.add_argument("--eval-max-batches", type=int, default=-1, help="-1 evaluates the full test split")
    ap.add_argument("--eval-seed", type=int, default=0, help="Seed for random eval batch selection")
    ap.add_argument("--viz-batches", type=int, default=1, help="Number of eval batches to visualize")
    ap.add_argument(
        "--viz-frames",
        type=int,
        default=10,
        help="Max reconstructed frames to save per visualized batch (recon + overlay PNGs)",
    )
    ap.add_argument("--viz-conf", type=float, default=0.4, help="Confidence threshold for viz NMS/predictions")
    ap.add_argument("--viz-iou", type=float, default=0.7, help="IoU threshold for viz NMS")
    ap.add_argument("--viz-max-det", type=int, default=20, help="Max detections per frame in viz NMS")
    args = ap.parse_args()
    _validate_args(ap, args)
    return args


def _validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    train_json = Path(args.train_json)
    test_json = Path(args.test_json)
    if not train_json.is_file():
        ap.error(f"--train-json not found: {train_json}")
    if not test_json.is_file():
        ap.error(f"--test-json not found: {test_json}")

    device = str(args.device).strip().lower()
    if device not in {"cpu", "mps"} and "," in device:
        n_gpus = len([x for x in device.split(",") if x.strip()])
        if args.batch < n_gpus:
            ap.error(
                f"--batch {args.batch} is too small for {n_gpus} GPUs. "
                f"Ultralytics DDP splits global batch across GPUs (per_gpu = batch // n_gpus). "
                f"Use --batch >= {n_gpus}, e.g. --batch {n_gpus} for one sample per GPU."
            )
    if args.ppb_min_filter_size % 2 == 0:
        ap.error(f"--ppb-min-filter-size must be odd, got {args.ppb_min_filter_size}")
    if args.spatial_kernel_size % 2 == 0:
        ap.error(f"--spatial-kernel-size must be odd, got {args.spatial_kernel_size}")


def _ensure_ddp_pythonpath() -> None:
    """Make the local ultralytics repo importable in Ultralytics DDP subprocesses."""
    repo_root = str(_REPO_ROOT)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if repo_root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_root, *parts]) if parts else repo_root


def main():
    args = parse_args()
    _ensure_ddp_pythonpath()
    cfg = dict(DEFAULT_CFG_DICT)
    cfg.update(
        {
            "spad_train_json": args.train_json,
            "spad_test_json": args.test_json,
            "spad_output_frames": args.spad_output_frames,
            "spad_stride_frames": args.spad_stride_frames,
            "spad_bins_per_gt": args.spad_bins_per_gt,
            "spad_image_size": args.imgsz,
            "spad_packed_ch_order": args.spad_packed_ch_order,
            "spad_subsampling": args.spad_subsampling,
            "spad_preprocessor": args.preprocessor,
            "ppb_bocpd_gamma": args.ppb_bocpd_gamma,
            "ppb_quantile": args.ppb_quantile,
            "ppb_normalize": True,
            "ppb_min_filter_size": args.ppb_min_filter_size,
            "stea_fast_window": args.stea_fast_window,
            "stea_slow_window": args.stea_slow_window,
            "stea_temporal_window": args.stea_temporal_window,
            "stea_fast_tau": args.stea_fast_tau,
            "stea_motion_sharpness": args.stea_motion_sharpness,
            "stea_motion_threshold": args.stea_motion_threshold,
            "stea_stable_prior": args.stea_stable_prior,
            "stea_normalize": True,
            "stea_quantile": 1.0,
            "spad_plugin": args.plugin,
            "spad_plugin_scales": args.plugin_scales,
            "spad_temporal_core": args.temporal_core,
            "ssd_state_dim": args.ssd_state_dim,
            "ssd_head_divisor": args.ssd_head_divisor,
            "spad_spatial_reduce_ratio": args.spatial_reduce_ratio,
            "spad_spatial_kernel_size": args.spatial_kernel_size,
            "spad_plugin_alpha_init": args.plugin_alpha_init,
            "spad_freeze_detector": args.freeze_detector,
            "spad_eval_max_batches": args.eval_max_batches,
            "spad_eval_seed": args.eval_seed,
            "spad_viz_batches": args.viz_batches,
            "spad_viz_frames": args.viz_frames,
            "spad_viz_period": args.viz_period,
            "spad_viz_conf": args.viz_conf,
            "spad_viz_iou": args.viz_iou,
            "spad_viz_max_det": args.viz_max_det,
        }
    )
    overrides = {
        "model": args.ckpt,
        "data": "spad_pose",
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
        "val": args.val,
        "plots": False,
        "multi_scale": False,
        "task": "pose",
    }

    trainer = SpadPoseTrainer(cfg=cfg, overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
