"""Train QNN-augmented YOLO pose model on VisionSIM SPAD windows.

Start small. A raw SPAD window is large after unpacking, so debug with
`--qnn-output-frames 1` or `4` and `--batch 1` first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from ultralytics.models.yolo.pose import QNNPoseTrainer
from ultralytics.utils import DEFAULT_CFG_DICT, RANK, nms


BONE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _draw_pose(img: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int]):
    for s, e in BONE_CONNECTIONS:
        if keypoints[s, 2] > 0 and keypoints[e, 2] > 0:
            cv2.line(img, tuple(keypoints[s, :2].astype(int)), tuple(keypoints[e, :2].astype(int)), color, 2)
    for x, y, v in keypoints:
        if v > 0:
            cv2.circle(img, (int(x), int(y)), 3, color, -1)


def _draw_labels(img: np.ndarray, batch: dict[str, Any], si: int, image_size: int):
    idx = batch["batch_idx"].view(-1).cpu() == si
    boxes = batch["bboxes"][idx].cpu().numpy()
    cls = batch["cls"][idx].view(-1).cpu().numpy()
    kpts = batch["keypoints"][idx].cpu().numpy()
    for box, cls_id, pose in zip(boxes, cls, kpts):
        cx, cy, w, h = box * image_size
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        color = (0, 0, 255) if int(cls_id) == 0 else (255, 0, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        _draw_pose(img, pose * np.array([image_size, image_size, 1.0]), color)


def _draw_predictions(img: np.ndarray, pred: dict[str, torch.Tensor], conf: float):
    boxes = pred["bboxes"].detach().cpu().numpy()
    scores = pred["conf"].detach().cpu().numpy()
    cls = pred["cls"].detach().cpu().numpy()
    keypoints = pred["keypoints"].detach().cpu().numpy()
    for box, score, cls_id, pose in zip(boxes, scores, cls, keypoints):
        if score < conf:
            continue
        color = (0, 255, 255) if int(cls_id) == 0 else (255, 255, 0)
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        cv2.putText(img, f"{int(cls_id)} {score:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        _draw_pose(img, pose, color)


def _postprocess_pose(preds, *, nc: int, conf: float, iou: float, max_det: int):
    raw = preds[0] if isinstance(preds, (list, tuple)) and torch.is_tensor(preds[0]) else preds
    outputs = nms.non_max_suppression(raw, conf, iou, nc=nc, multi_label=True, max_det=max_det)
    processed = []
    for x in outputs:
        extra = x[:, 6:]
        processed.append(
            {
                "bboxes": x[:, :4],
                "conf": x[:, 4],
                "cls": x[:, 5],
                "keypoints": extra.view(-1, 21, 3),
            }
        )
    return processed


def _recon_canvas(model, si: int, image_size: int) -> np.ndarray:
    frames = getattr(model, "qnn_last_recon_frames", None)
    if frames is None:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    t = si
    b = 0
    if t >= frames.shape[0]:
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)
    img = frames[t, b].detach().float().cpu().permute(1, 2, 0).numpy()
    rgb_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb_u8[:, :, ::-1])


def _run_qnn_eval_visualization(trainer, args, *, epoch_idx: int):
    if RANK not in {-1, 0}:
        return

    model = trainer.ema.ema if getattr(trainer, "ema", None) is not None else trainer.model
    was_training = model.training
    model.eval()
    dataset = trainer.test_loader.dataset
    max_batches = int(args.eval_max_batches)
    viz_batches = int(args.viz_batches)
    eval_count = len(dataset) if max_batches < 0 else min(max_batches, len(dataset))
    if eval_count <= 0:
        eval_count = min(1, len(dataset))
    generator = torch.Generator().manual_seed(int(args.eval_seed) + int(epoch_idx))
    indices = torch.randperm(len(dataset), generator=generator)[:eval_count].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
    save_dir = Path(trainer.save_dir) / "qnn_viz" / f"epoch{epoch_idx:03d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    loss_sum_total = 0.0
    loss_items_total = None
    seen_batches = 0
    written = []
    last_processed_count = 0
    last_target_images = 0

    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            batch = trainer.preprocess_batch(batch)
            if "packed_nch" in batch:
                model.qnn_packed_nch = int(batch["packed_nch"])
            preds = model(batch["img"])
            loss, loss_items = model.loss(batch, preds)
            processed = _postprocess_pose(
                preds, nc=trainer.data["nc"], conf=args.viz_conf, iou=args.viz_iou, max_det=args.viz_max_det
            )

            loss_sum_total += float(loss.sum().detach().cpu())
            loss_items_cpu = loss_items.detach().cpu()
            loss_items_total = loss_items_cpu if loss_items_total is None else loss_items_total + loss_items_cpu
            seen_batches += 1
            last_processed_count = len(processed)
            last_target_images = int(batch["batch_idx"].max().item() + 1) if batch["batch_idx"].numel() else 0

            if batch_i < viz_batches:
                image_size = int(args.imgsz)
                num_images = min(int(args.viz_frames), max(len(processed), last_target_images))
                for si in range(num_images):
                    canvas = _recon_canvas(model, si, image_size)
                    cv2.imwrite(str(save_dir / f"batch{batch_i:03d}_sample{si:03d}_recon.png"), canvas)
                    _draw_labels(canvas, batch, si, image_size)
                    if si < len(processed):
                        _draw_predictions(canvas, processed[si], conf=args.viz_conf)
                    out_path = save_dir / f"batch{batch_i:03d}_sample{si:03d}_overlay.png"
                    ok = cv2.imwrite(str(out_path), canvas)
                    written.append(f"{out_path.name}: {'ok' if ok else 'failed'}")

    if seen_batches:
        trainer.metrics["qnn_val/loss_sum"] = loss_sum_total / seen_batches
        mean_loss_items = loss_items_total / seen_batches
        for i, value in enumerate(mean_loss_items.tolist()):
            trainer.metrics[f"qnn_val/loss_{i}"] = float(value)
    else:
        mean_loss_items = torch.zeros(5)

    with (save_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"epoch_idx: {epoch_idx}\n")
        f.write(f"seen_batches: {seen_batches}\n")
        f.write(f"eval_max_batches: {max_batches}\n")
        f.write(f"random_indices: {indices}\n")
        f.write(f"mean_loss_sum: {trainer.metrics.get('qnn_val/loss_sum', 0.0)}\n")
        f.write(f"mean_loss_items: {mean_loss_items.tolist()}\n")
        f.write(f"last_processed_predictions: {last_processed_count}\n")
        f.write(f"last_target_images: {last_target_images}\n")
        f.write("\n".join(written))
        f.write("\n")

    if was_training:
        model.train()


def _make_eval_callback(args, *, baseline: bool = False):
    def callback(trainer):
        if baseline:
            _run_qnn_eval_visualization(trainer, args, epoch_idx=0)
            if RANK in {-1, 0}:
                initial_path = Path(trainer.save_dir) / "weights" / "initial.pt"
                initial_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": -1,
                        "model": trainer.model,
                        "train_args": vars(trainer.args),
                    },
                    initial_path,
                )
            return

        period = int(args.viz_period)
        if period <= 0 or (trainer.epoch + 1) % period != 0:
            return
        _run_qnn_eval_visualization(trainer, args, epoch_idx=trainer.epoch + 1)

    return callback


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train QNNPoseModel on VisionSIM SPAD data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", type=str, required=True, help="Path to pretrained YOLO pose checkpoint (.pt)")
    ap.add_argument(
        "--gt-root",
        type=str,
        required=True,
        help="VisionSIM GT root; each video subdir must contain hand_ann.json",
    )
    ap.add_argument(
        "--spad-root",
        type=str,
        required=True,
        help="VisionSIM packed SPAD root; each video subdir must contain frames.npy",
    )
    ap.add_argument("--project", type=str, default="runs/qnn_pose", help="Ultralytics project directory for runs")
    ap.add_argument("--name", type=str, default="debug", help="Run name under --project")
    ap.add_argument("--device", type=str, default="0", help="CUDA device id(s), e.g. 0 or 0,1; use cpu for CPU")
    ap.add_argument("--epochs", type=int, default=30, help="Training epochs")
    ap.add_argument("--batch", type=int, default=1, help="Batch size; keep at 1 while debugging large SPAD windows")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    ap.add_argument("--imgsz", type=int, default=512, help="Square image size for normalized bbox/keypoint labels")
    ap.add_argument("--lr0", type=float, default=1e-3, help="Initial learning rate")
    ap.add_argument("--weight-decay", type=float, default=5e-4, help="Optimizer weight decay")
    ap.add_argument("--amp", action="store_true", help="Enable automatic mixed precision training")
    ap.add_argument(
        "--val",
        action="store_true",
        help="Also run the built-in Ultralytics validator (experimental for QNN batches)",
    )
    ap.add_argument(
        "--test-keywords",
        type=str,
        default="white-room",
        help="Comma-separated substrings; matching video names go to val/test split",
    )
    ap.add_argument(
        "--qnn-output-frames",
        type=int,
        default=1,
        help=(
            "Reconstructed output frames (and pose labels) per training window. "
            "Each frame needs raw SPAD bins; use 1 or 4 for smoke tests before scaling up."
        ),
    )
    ap.add_argument(
        "--qnn-stride-frames",
        type=int,
        default=64,
        help=(
            "Sliding-window stride in GT frame units when enumerating training windows. "
            "0 falls back to --qnn-output-frames."
        ),
    )
    ap.add_argument(
        "--qnn-spad-per-gt",
        type=int,
        default=64,
        help=(
            "Raw SPAD time bins per one GT annotation frame. "
            "Controls GT/SPAD temporal alignment (spad_start = gt_start * this value)."
            "8,000 / 125 = 64"
        ),
    )
    ap.add_argument(
        "--qnn-subsampling",
        type=int,
        default=320,
        help=(
            "Shared temporal downsampling factor: (1) PerPixelBayesian raw-bin aggregation per "
            "reconstructed frame; (2) SPAD bins between consecutive output-frame labels "
            "(spad_len = output_frames * subsampling + 1)."
        ),
    )
    ap.add_argument(
        "--qnn-bocpd-gamma",
        type=float,
        default=5e-4,
        help="PerPixelBayesian BOCPD hazard rate; higher values adapt faster to photon-rate changes",
    )
    ap.add_argument(
        "--qnn-quantile",
        type=float,
        default=1.0,
        help="PPB normalization upper quantile (1.0 = use max; lower values suppress hot pixels)",
    )
    ap.add_argument(
        "--qnn-min-filter-size",
        type=int,
        default=8,
        help="Odd kernel size for min-filter smoothing of per-pixel runlength estimates in PPB",
    )
    ap.add_argument(
        "--qnn-ssd-after-layers",
        type=str,
        default=None,
        help="Comma-separated YOLO layer indices after which SSD blocks are inserted (default: 0,2,4,6)",
    )
    ap.add_argument(
        "--qnn-ssd-state-dim",
        type=int,
        default=8,
        help="Hidden state dimension of each SSD (semi-separable dynamics) temporal block",
    )
    ap.add_argument(
        "--qnn-ssd-head-divisor",
        type=int,
        default=4,
        help="SSD head_dim = YOLO feature channels // this divisor (must divide channels evenly)",
    )
    ap.add_argument(
        "--qnn-freeze-detector",
        action="store_true",
        default=True,
        help="Freeze pretrained YOLO detector layers; only QNN modules (PPB + SSD) are trained",
    )
    ap.add_argument(
        "--viz-period",
        type=int,
        default=5,
        help="Run QNN eval visualization every N epochs (0 disables periodic viz; epoch 0 baseline always runs)",
    )
    ap.add_argument("--eval-max-batches", type=int, default=-1, help="-1 evaluates the full test split")
    ap.add_argument("--eval-seed", type=int, default=0, help="Seed for random eval batch selection")
    ap.add_argument("--viz-batches", type=int, default=1, help="Number of eval batches to visualize")
    ap.add_argument(
        "--viz-frames",
        type=int,
        default=4,
        help="Max reconstructed frames to save per visualized batch (recon + overlay PNGs)",
    )
    ap.add_argument("--viz-conf", type=float, default=0.4, help="Confidence threshold for viz NMS/predictions")
    ap.add_argument("--viz-iou", type=float, default=0.7, help="IoU threshold for viz NMS")
    ap.add_argument("--viz-max-det", type=int, default=20, help="Max detections per frame in viz NMS")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = dict(DEFAULT_CFG_DICT)
    cfg.update(
        {
            "qnn_gt_root": args.gt_root,
            "qnn_spad_root": args.spad_root,
            "qnn_test_keywords": args.test_keywords,
            "qnn_output_frames": args.qnn_output_frames,
            "qnn_stride_frames": args.qnn_stride_frames,
            "qnn_spad_per_gt": args.qnn_spad_per_gt,
            "qnn_image_size": args.imgsz,
            "qnn_subsampling": args.qnn_subsampling,
            "qnn_bocpd_gamma": args.qnn_bocpd_gamma,
            "qnn_quantile": args.qnn_quantile,
            "qnn_min_filter_size": args.qnn_min_filter_size,
            "qnn_ssd_after_layers": args.qnn_ssd_after_layers,
            "qnn_ssd_state_dim": args.qnn_ssd_state_dim,
            "qnn_ssd_head_divisor": args.qnn_ssd_head_divisor,
            "qnn_freeze_detector": args.qnn_freeze_detector,
            "qnn_eval_max_batches": args.eval_max_batches,
            "qnn_eval_seed": args.eval_seed,
            "qnn_viz_batches": args.viz_batches,
            "qnn_viz_frames": args.viz_frames,
            "qnn_viz_period": args.viz_period,
            "qnn_viz_conf": args.viz_conf,
            "qnn_viz_iou": args.viz_iou,
            "qnn_viz_max_det": args.viz_max_det,
        }
    )
    overrides = {
        "model": args.ckpt,
        "data": "qnn_spad",
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

    trainer = QNNPoseTrainer(cfg=cfg, overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
