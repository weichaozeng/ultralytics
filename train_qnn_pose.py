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

from ultralytics.models.yolo.pose import QNNPoseTrainer
from ultralytics.utils import DEFAULT_CFG_DICT, nms


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


def _make_eval_callback(args):
    def on_fit_epoch_end(trainer):
        period = int(args.viz_period)
        if period <= 0 or (trainer.epoch + 1) % period != 0:
            return

        model = trainer.ema.ema if getattr(trainer, "ema", None) is not None else trainer.model
        model.eval()
        dataset = trainer.test_loader.dataset
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
        batch = next(iter(loader))
        batch = trainer.preprocess_batch(batch)

        save_dir = Path(trainer.save_dir) / "qnn_viz" / f"epoch{trainer.epoch + 1:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            preds = model(batch["img"])
            loss, loss_items = model.loss(batch, preds)
            processed = _postprocess_pose(preds, nc=trainer.data["nc"], conf=args.viz_conf, iou=args.viz_iou, max_det=args.viz_max_det)

        trainer.metrics["qnn_val/loss_sum"] = float(loss.sum().detach().cpu())
        for i, value in enumerate(loss_items.detach().cpu().tolist()):
            trainer.metrics[f"qnn_val/loss_{i}"] = float(value)

        image_size = int(args.qnn_image_size)
        num_images = min(int(args.viz_frames), len(processed))
        for si in range(num_images):
            canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            _draw_labels(canvas, batch, si, image_size)
            _draw_predictions(canvas, processed[si], conf=args.viz_conf)
            cv2.imwrite(str(save_dir / f"sample{si:03d}.png"), canvas)

        model.train()

    return on_fit_epoch_end


def parse_args():
    ap = argparse.ArgumentParser(description="Train QNNPoseModel on VisionSIM SPAD data")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--gt-root", type=str, required=True)
    ap.add_argument("--spad-root", type=str, required=True)
    ap.add_argument("--project", type=str, default="runs/qnn_pose")
    ap.add_argument("--name", type=str, default="debug")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--val", action="store_true", help="Also run the built-in Ultralytics validator (experimental for QNN batches)")
    ap.add_argument("--test-keywords", type=str, default="attic")
    ap.add_argument("--qnn-output-frames", type=int, default=1)
    ap.add_argument("--qnn-stride-frames", type=int, default=0)
    ap.add_argument("--qnn-spad-per-gt", type=int, default=64)
    ap.add_argument("--qnn-subsampling", type=int, default=64)
    ap.add_argument("--qnn-bocpd-gamma", type=float, default=5e-4)
    ap.add_argument("--qnn-quantile", type=float, default=1.0)
    ap.add_argument("--qnn-min-filter-size", type=int, default=7)
    ap.add_argument("--qnn-ssd-after-layers", type=str, default=None)
    ap.add_argument("--qnn-ssd-state-dim", type=int, default=8)
    ap.add_argument("--qnn-ssd-head-divisor", type=int, default=4)
    ap.add_argument("--qnn-freeze-detector", action="store_true", default=True)
    ap.add_argument("--viz-period", type=int, default=1)
    ap.add_argument("--viz-frames", type=int, default=4)
    ap.add_argument("--viz-conf", type=float, default=0.25)
    ap.add_argument("--viz-iou", type=float, default=0.7)
    ap.add_argument("--viz-max-det", type=int, default=20)
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
    trainer.add_callback("on_fit_epoch_end", _make_eval_callback(args))
    trainer.train()


if __name__ == "__main__":
    main()
