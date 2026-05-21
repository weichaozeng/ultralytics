"""Inspect one QNN SPAD pose dataset batch.

Example:
    python ultralytics/inspect_qnn_batch.py \
      --gt-root /home/zvc/Data/visionsim/outputs/hamnosys_v2/renders \
      --spad-root /home/zvc/Data/visionsim/outputs/hamnosys_v2/renders-spc8kHz \
      --out Print/qnn_batch_inspect.txt \
      --output-frames 4 \
      --batch 2 \
      --test-keywords attic
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ultralytics.data.qnn_spad_dataset import QNNSpadPoseDataset


def _tensor_summary(x: torch.Tensor) -> dict[str, Any]:
    out = {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "numel": int(x.numel()),
    }
    if x.numel():
        x_float = x.float()
        out.update(
            {
                "min": float(x_float.min()),
                "max": float(x_float.max()),
                "mean": float(x_float.mean()),
                "sum": float(x_float.sum()),
            }
        )
    return out


def _print_sample(sample: dict[str, Any], *, title: str):
    print("\n" + "=" * 100)
    print(title)
    for key in ("im_file", "output_frames", "ori_shape", "resized_shape"):
        print(f"{key}: {sample.get(key)}")
    for key in ("img", "cls", "bboxes", "keypoints", "batch_idx"):
        value = sample[key]
        print(f"{key}: {_tensor_summary(value)}")
        if torch.is_tensor(value) and value.numel():
            flat = value.reshape(-1)
            print(f"{key} first values: {flat[:20].tolist()}")


def _main_impl(args):
    print("QNN batch inspection")
    print(f"gt_root={args.gt_root}")
    print(f"spad_root={args.spad_root}")
    print(f"split={args.split}")
    print(f"test_keywords={args.test_keywords}")
    print(f"output_frames={args.output_frames}")
    print(f"stride_frames={args.stride_frames}")
    print(f"batch={args.batch}")

    ds = QNNSpadPoseDataset(
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

    print("\n" + "=" * 100)
    print("Dataset summary")
    print(f"video_count={len(ds.video_names)}")
    print(f"window_count={len(ds)}")
    print(f"video_names_sample={ds.video_names[:5]}{' ...' if len(ds.video_names) > 5 else ''}")
    print(f"first_windows={ds.windows[: min(5, len(ds.windows))]}")

    sample = ds[args.index]
    _print_sample(sample, title=f"Single sample index={args.index}")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=ds.collate_fn)
    batch = next(iter(loader))
    _print_sample(batch, title="Collated batch")

    if batch["batch_idx"].numel():
        print("\nBatch index diagnostics")
        print(f"batch_idx unique: {torch.unique(batch['batch_idx']).tolist()}")
        print(f"batch_idx min/max: {float(batch['batch_idx'].min())} / {float(batch['batch_idx'].max())}")
        expected_pred_batch = args.batch * args.output_frames
        print(f"expected flattened prediction batch size: {expected_pred_batch}")

    print("\nDone.")


def main():
    ap = argparse.ArgumentParser(description="Inspect QNNSpadPoseDataset samples and collated batches")
    ap.add_argument("--gt-root", type=str, required=True)
    ap.add_argument("--spad-root", type=str, required=True)
    ap.add_argument("--out", type=str, default="Print/qnn_batch_inspect.txt")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--test-keywords", type=str, default=None)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--output-frames", type=int, default=4)
    ap.add_argument("--spad-per-gt", type=int, default=64)
    ap.add_argument("--stride-frames", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--packed-ch-order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    stride_frames = int(args.stride_frames)
    args.stride_frames = stride_frames if stride_frames > 0 else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f, redirect_stdout(f):
        _main_impl(args)
    print(f"Wrote QNN batch inspection report to: {out_path}")


if __name__ == "__main__":
    main()
