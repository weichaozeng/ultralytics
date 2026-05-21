"""Smoke test QNNPoseModel forward and pose loss on one QNN SPAD batch.

Use very small windows first. Even `output_frames=4` expands to hundreds of MB
before PPB/YOLO activations.

Example:
    python ultralytics/inspect_qnn_forward.py \
      --ckpt /home/zvc/Project/SPADHand/ultralytics/weights/detector.pt \
      --gt-root /home/zvc/Data/visionsim/outputs/hamnosys_v2/renders \
      --spad-root /home/zvc/Data/visionsim/outputs/hamnosys_v2/renders-spc8kHz \
      --out Print/qnn_forward_inspect.txt \
      --output-frames 1 \
      --batch 1 \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ultralytics.cfg import get_cfg
from ultralytics.data.qnn_spad_dataset import QNNSpadPoseDataset
from ultralytics.nn.tasks import QNNPoseModel, load_checkpoint


def _shape_summary(x: Any):
    if torch.is_tensor(x):
        return {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "requires_grad": bool(x.requires_grad),
        }
    if isinstance(x, (list, tuple)):
        return [_shape_summary(v) for v in x]
    if isinstance(x, dict):
        return {k: _shape_summary(v) for k, v in x.items()}
    return str(type(x))


def _tensor_summary(x: torch.Tensor):
    out = _shape_summary(x)
    if x.numel():
        xf = x.detach().float()
        out.update({"min": float(xf.min()), "max": float(xf.max()), "mean": float(xf.mean())})
    return out


def _move_batch(batch: dict[str, Any], device: torch.device):
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(v) else v
    return moved


def _make_dataset(args):
    return QNNSpadPoseDataset(
        gt_root=args.gt_root,
        spad_root=args.spad_root,
        split=args.split,
        test_keywords=args.test_keywords,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        output_frames=args.output_frames,
        spad_per_gt=args.spad_per_gt,
        stride_frames=args.stride_frames,
        image_size=args.image_size,
        packed_ch_order=args.packed_ch_order,
    )


def _make_model(args, device: torch.device):
    weights, _ = load_checkpoint(args.ckpt, device="cpu", fuse=False)
    qnn_integrator_kwargs = {
        "subsampling": args.qnn_subsampling,
        "bocpd_gamma": args.qnn_bocpd_gamma,
        "normalize": True,
        "quantile": args.qnn_quantile,
        "min_filter_size": args.qnn_min_filter_size,
    }
    qnn_ssd_after_layers = None
    if args.qnn_ssd_after_layers:
        qnn_ssd_after_layers = [int(x) for x in args.qnn_ssd_after_layers.split(",") if x.strip()]

    model = QNNPoseModel(
        weights.yaml,
        nc=getattr(weights, "nc", 2),
        ch=3,
        data_kpt_shape=getattr(weights, "kpt_shape", (21, 3)),
        verbose=False,
        qnn_enabled=True,
        qnn_integrator_kwargs=qnn_integrator_kwargs,
        qnn_ssd_after_layers=qnn_ssd_after_layers,
        qnn_ssd_state_dim=args.qnn_ssd_state_dim,
        qnn_ssd_head_divisor=args.qnn_ssd_head_divisor,
    )
    model.load(weights)
    model.names = getattr(weights, "names", {0: "left_hand", 1: "right_hand"})
    model.args = get_cfg(overrides={})
    model.to(device)
    return model


def _main_impl(args):
    device = torch.device(args.device)
    print("QNN forward smoke test")
    print(f"device={device}")
    print(f"ckpt={args.ckpt}")
    print(f"output_frames={args.output_frames}, batch={args.batch}")

    ds = _make_dataset(args)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=ds.collate_fn)
    batch = next(iter(loader))
    print("\nDataset/batch summary")
    print(f"dataset videos={len(ds.video_names)}, windows={len(ds)}")
    for key in ("img", "cls", "bboxes", "keypoints", "batch_idx"):
        print(f"{key}: {_tensor_summary(batch[key])}")
    print(f"im_file={batch['im_file']}")

    model = _make_model(args, device)
    print("\nModel summary")
    print(f"type={type(model).__name__}")
    print(f"qnn_ssd_layer_info={model.qnn_ssd_layer_info}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total={total_params}, trainable={trainable_params}")

    batch = _move_batch(batch, device)
    model.train() if args.train_mode else model.eval()
    print(f"model.training={model.training}")

    with torch.set_grad_enabled(args.with_grad):
        print("\nRunning forward...")
        preds = model(batch["img"])
        print("Forward output summary:")
        print(json.dumps(_shape_summary(preds), indent=2))

        if args.loss:
            print("\nRunning loss...")
            loss, loss_items = model.loss(batch, preds)
            print(f"loss: {_tensor_summary(loss)}")
            print(f"loss_items: {_tensor_summary(loss_items)}")
            print(f"loss scalar sum: {float(loss.sum().detach())}")

    print("\nDone.")


def main():
    ap = argparse.ArgumentParser(description="Smoke test QNNPoseModel forward/loss")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--gt-root", type=str, required=True)
    ap.add_argument("--spad-root", type=str, required=True)
    ap.add_argument("--out", type=str, default="Print/qnn_forward_inspect.txt")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--test-keywords", type=str, default=None)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--output-frames", type=int, default=1)
    ap.add_argument("--spad-per-gt", type=int, default=64)
    ap.add_argument("--stride-frames", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--packed-ch-order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--qnn-subsampling", type=int, default=64)
    ap.add_argument("--qnn-bocpd-gamma", type=float, default=5e-4)
    ap.add_argument("--qnn-quantile", type=float, default=1.0)
    ap.add_argument("--qnn-min-filter-size", type=int, default=7)
    ap.add_argument("--qnn-ssd-after-layers", type=str, default=None)
    ap.add_argument("--qnn-ssd-state-dim", type=int, default=8)
    ap.add_argument("--qnn-ssd-head-divisor", type=int, default=4)
    ap.add_argument("--loss", action="store_true", help="Also run pose loss")
    ap.add_argument("--with-grad", action="store_true", help="Enable autograd during forward/loss")
    ap.add_argument("--train-mode", action="store_true", help="Run model.train() instead of eval()")
    args = ap.parse_args()

    stride_frames = int(args.stride_frames)
    args.stride_frames = stride_frames if stride_frames > 0 else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f, redirect_stdout(f):
        _main_impl(args)
    print(f"Wrote QNN forward inspection report to: {out_path}")


if __name__ == "__main__":
    main()
