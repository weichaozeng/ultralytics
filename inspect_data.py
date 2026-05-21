"""Inspect VisionSIM hand GT and SPAD npy data layout.

This script is intentionally read-only. It prints enough structure to design a
training dataset without dumping large arrays or full annotations.

Example:
    python ultralytics/inspect_data.py \
      --gt-root /home/zvc/Data/visionsim/outputs/hamnosys_v2/renders \
      --spad-root /home/zvc/Data/visionsim/outputs/hamnosys_v2/renders-spc8kHz \
      --out Print/data_inspect.txt
"""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _shape_of_nested_list(x: Any) -> tuple[int, ...] | None:
    shape = []
    cur = x
    while isinstance(cur, list):
        shape.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return tuple(shape) if shape else None


def _fmt_path(path: Path) -> str:
    return str(path)


def _sample_items(seq, n: int):
    seq = list(seq)
    if len(seq) <= n:
        return seq
    return seq[: n // 2] + ["..."] + seq[-(n - n // 2) :]


def _parse_frame_index(frame_name: str) -> int | None:
    stem = Path(frame_name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def _inspect_annotation(ann: dict[str, Any], *, sample_frames: int):
    frame_names = sorted(ann.keys())
    print(f"GT frame count: {len(frame_names)}")
    print(f"GT frame name samples: {_sample_items(frame_names, sample_frames)}")

    frame_indices = [_parse_frame_index(name) for name in frame_names]
    parsed = [idx for idx in frame_indices if idx is not None]
    if parsed:
        diffs = np.diff(parsed)
        unique_diffs = sorted(set(int(x) for x in diffs.tolist())) if len(diffs) else []
        print(f"GT parsed frame index range: {parsed[0]} -> {parsed[-1]}")
        print(f"GT parsed frame index unique diffs: {_sample_items(unique_diffs, 20)}")

    hand_counts = {"left_hand": 0, "right_hand": 0}
    missing_keys = []
    bbox_examples = {}
    kpt_examples = {}
    for frame_name in frame_names:
        frame_ann = ann.get(frame_name, {})
        if not isinstance(frame_ann, dict):
            missing_keys.append((frame_name, "frame_not_dict"))
            continue
        for hand_name in ("left_hand", "right_hand"):
            hand = frame_ann.get(hand_name)
            if not hand:
                continue
            hand_counts[hand_name] += 1
            bbox = hand.get("bbox")
            kpts = hand.get("keypoints_2d")
            if bbox is not None and hand_name not in bbox_examples:
                bbox_examples[hand_name] = bbox
            if kpts is not None and hand_name not in kpt_examples:
                kpt_examples[hand_name] = {
                    "shape": _shape_of_nested_list(kpts),
                    "first": kpts[:2] if isinstance(kpts, list) else None,
                }
            if bbox is None or kpts is None:
                missing_keys.append((frame_name, hand_name))

    print(f"GT hand presence counts: {hand_counts}")
    print(f"GT missing bbox/keypoints examples: {_sample_items(missing_keys, 10)}")
    print(f"GT bbox examples: {bbox_examples}")
    print(f"GT keypoint examples: {kpt_examples}")


def _inspect_spad_npy(path: Path, *, sample_values: bool):
    if not path.exists():
        print(f"SPAD frames.npy missing: {_fmt_path(path)}")
        return None

    arr = np.load(path, mmap_mode="r")
    print(f"SPAD npy path: {_fmt_path(path)}")
    print(f"SPAD shape: {arr.shape}")
    print(f"SPAD dtype: {arr.dtype}")
    print(f"SPAD ndim: {arr.ndim}")
    print(f"SPAD itemsize: {arr.dtype.itemsize}")
    print(f"SPAD estimated bytes: {int(np.prod(arr.shape)) * arr.dtype.itemsize}")
    print(f"SPAD C-contiguous metadata: {getattr(arr, 'flags', {}).c_contiguous if hasattr(arr, 'flags') else 'unknown'}")

    if arr.ndim >= 1:
        print(f"SPAD first dimension length: {arr.shape[0]}")
    if arr.ndim == 4:
        print(f"SPAD possible layout hints:")
        print(f"  THWC if shape is T,H,W,C: T={arr.shape[0]}, H={arr.shape[1]}, W={arr.shape[2]}, C={arr.shape[3]}")
        print(f"  NHWpackedC if packed: N={arr.shape[0]}, H={arr.shape[1]}, Wpacked={arr.shape[2]}, C={arr.shape[3]}")

    if sample_values:
        sample = np.asarray(arr[0])
        print(f"SPAD first frame min/max/sum: {sample.min()} / {sample.max()} / {sample.sum()}")
        flat = sample.reshape(-1)
        print(f"SPAD first frame first values: {flat[:20].tolist()}")

    return arr.shape


def _inspect_transforms(path: Path):
    if not path.exists():
        print(f"transforms.json missing: {_fmt_path(path)}")
        return
    data = _load_json(path)
    if isinstance(data, dict):
        print(f"transforms.json keys: {sorted(data.keys())}")
        for key in ("camera_angle_x", "camera_angle_y", "fl_x", "fl_y", "cx", "cy", "w", "h", "frames"):
            if key in data:
                value = data[key]
                if key == "frames" and isinstance(value, list):
                    print(f"transforms.frames count: {len(value)}")
                    if value:
                        print(f"transforms.frames[0] keys: {sorted(value[0].keys()) if isinstance(value[0], dict) else type(value[0])}")
                else:
                    print(f"transforms.{key}: {value}")
    else:
        print(f"transforms.json type: {type(data).__name__}")


def _inspect_one_video(gt_dir: Path, spad_dir: Path, *, sample_frames: int, sample_values: bool):
    print("\n" + "=" * 100)
    print(f"VIDEO: {gt_dir.name}")
    print(f"GT dir: {_fmt_path(gt_dir)}")
    print(f"SPAD dir: {_fmt_path(spad_dir)}")

    ann_path = gt_dir / "hand_ann.json"
    transforms_path = gt_dir / "transforms.json"
    spad_path = spad_dir / "frames.npy"

    if ann_path.exists():
        ann = _load_json(ann_path)
        if isinstance(ann, dict):
            _inspect_annotation(ann, sample_frames=sample_frames)
        else:
            print(f"hand_ann.json type is not dict: {type(ann).__name__}")
    else:
        print(f"hand_ann.json missing: {_fmt_path(ann_path)}")

    _inspect_transforms(transforms_path)
    spad_shape = _inspect_spad_npy(spad_path, sample_values=sample_values)

    if ann_path.exists() and spad_shape is not None:
        ann = _load_json(ann_path)
        gt_frames = len(ann) if isinstance(ann, dict) else None
        if gt_frames:
            spad_t = spad_shape[0]
            print("Alignment hints:")
            print(f"  spad_t / gt_frames = {spad_t / gt_frames:.6f}")
            print(f"  expected ratio for 8000fps/125fps = 64.000000")
            print(f"  if chunk_size=64, expected output frames = {spad_t // 64} remainder={spad_t % 64}")


def _main_impl(args):
    gt_root = Path(args.gt_root)
    spad_root = Path(args.spad_root)
    print(f"GT root: {_fmt_path(gt_root)}")
    print(f"SPAD root: {_fmt_path(spad_root)}")
    print(f"GT root exists: {gt_root.exists()}")
    print(f"SPAD root exists: {spad_root.exists()}")

    gt_videos = sorted([p for p in gt_root.iterdir() if p.is_dir()]) if gt_root.exists() else []
    spad_videos = sorted([p for p in spad_root.iterdir() if p.is_dir()]) if spad_root.exists() else []
    gt_names = {p.name for p in gt_videos}
    spad_names = {p.name for p in spad_videos}
    common_names = sorted(gt_names & spad_names)

    print("\n" + "=" * 100)
    print("Dataset-level summary")
    print(f"GT video dirs: {len(gt_videos)}")
    print(f"SPAD video dirs: {len(spad_videos)}")
    print(f"Matched video dirs: {len(common_names)}")
    print(f"GT-only samples: {_sample_items(sorted(gt_names - spad_names), 20)}")
    print(f"SPAD-only samples: {_sample_items(sorted(spad_names - gt_names), 20)}")
    print(f"Matched sample names: {_sample_items(common_names, 20)}")

    selected = common_names
    if args.video:
        selected = [name for name in common_names if name in set(args.video)]
    selected = selected[: args.max_videos]

    for name in selected:
        _inspect_one_video(
            gt_root / name,
            spad_root / name,
            sample_frames=args.sample_frames,
            sample_values=args.sample_values,
        )


def main():
    ap = argparse.ArgumentParser(description="Inspect VisionSIM hand GT and SPAD npy dataset")
    ap.add_argument("--gt-root", type=str, required=True, help="Root containing video folders with hand_ann.json")
    ap.add_argument("--spad-root", type=str, required=True, help="Root containing video folders with frames.npy")
    ap.add_argument("--out", type=str, default="Print/data_inspect.txt", help="Report path")
    ap.add_argument("--max-videos", type=int, default=5, help="Number of matched videos to inspect")
    ap.add_argument("--video", type=str, nargs="*", default=None, help="Optional specific video names to inspect")
    ap.add_argument("--sample-frames", type=int, default=8, help="Number of frame names/examples to print")
    ap.add_argument("--sample-values", action="store_true", help="Print min/max/sum and first values from first SPAD frame")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f, redirect_stdout(f):
        _main_impl(args)
    print(f"Wrote data inspection report to: {out_path}")


if __name__ == "__main__":
    main()
