"""QNN SPAD pose dataset for VisionSIM hand annotations."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.spad_packed import packed_frames_to_raw_bayer


@dataclass(frozen=True)
class QNNWindow:
    """A fixed temporal training window inside one video."""

    name: str
    gt_dir: Path
    spad_path: Path
    gt_start: int
    output_frames: int
    spad_step: int


class QNNSpadPoseDataset(Dataset):
    """Load fixed windows from packed SPAD videos and timestamp-aligned hand pose labels."""

    HAND_TO_CLASS = {"left_hand": 0, "right_hand": 1}

    def __init__(
        self,
        gt_root: str | Path,
        spad_root: str | Path,
        *,
        split: str = "train",
        test_keywords: str | list[str] | tuple[str, ...] | None = None,
        test_fraction: float = 0.2,
        split_seed: int = 0,
        output_frames: int = 4,
        spad_per_gt: int = 64,
        spad_step: int | None = None,
        stride_frames: int | None = None,
        image_size: int = 512,
        packed_ch_order: str = "RGB",
    ):
        self.gt_root = Path(gt_root)
        self.spad_root = Path(spad_root)
        self.split = split
        self.test_keywords = self._parse_keywords(test_keywords)
        self.test_fraction = float(test_fraction)
        self.split_seed = int(split_seed)
        self.output_frames = int(output_frames)
        self.spad_per_gt = int(spad_per_gt)
        self.spad_step = int(spad_step or spad_per_gt)
        self.stride_frames = int(stride_frames or output_frames)
        self.image_size = int(image_size)
        self.packed_ch_order = packed_ch_order.upper()

        if self.output_frames <= 0:
            raise ValueError(f"output_frames must be > 0, got {self.output_frames}")
        if self.spad_per_gt <= 0:
            raise ValueError(f"spad_per_gt must be > 0, got {self.spad_per_gt}")
        if self.spad_step <= 0:
            raise ValueError(f"spad_step must be > 0, got {self.spad_step}")

        self.video_names = self._select_video_names()
        self.annotations = {name: self._load_annotation(name) for name in self.video_names}
        self.windows = self._build_windows()
        if not self.windows:
            raise RuntimeError(f"No QNN SPAD windows found for split={split!r}")

    @staticmethod
    def _parse_keywords(keywords) -> tuple[str, ...]:
        if keywords is None:
            return ()
        if isinstance(keywords, str):
            return tuple(x.strip() for x in keywords.split(",") if x.strip())
        return tuple(str(x).strip() for x in keywords if str(x).strip())

    def _select_video_names(self) -> list[str]:
        gt_names = {p.name for p in self.gt_root.iterdir() if p.is_dir()}
        spad_names = {p.name for p in self.spad_root.iterdir() if p.is_dir() and (p / "frames.npy").exists()}
        names = sorted(gt_names & spad_names)

        if self.test_keywords:
            test_names = [n for n in names if any(k in n for k in self.test_keywords)]
        else:
            rng = random.Random(self.split_seed)
            shuffled = names[:]
            rng.shuffle(shuffled)
            n_test = max(1, int(round(len(shuffled) * self.test_fraction))) if shuffled else 0
            test_names = sorted(shuffled[:n_test])

        test_set = set(test_names)
        if self.split in {"val", "test"}:
            return sorted(test_set)
        if self.split == "train":
            return [n for n in names if n not in test_set]
        raise ValueError(f"Unsupported split: {self.split}")

    def _load_annotation(self, name: str) -> dict[str, Any]:
        path = self.gt_root / name / "hand_ann.json"
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_windows(self) -> list[QNNWindow]:
        windows: list[QNNWindow] = []
        for name in self.video_names:
            ann = self.annotations[name]
            n_gt = len(ann)
            last_gt_offset = self.output_frames * self.spad_step / self.spad_per_gt
            max_start = int(np.floor((n_gt - 1) - last_gt_offset))
            if max_start < 0:
                continue
            for gt_start in range(0, max_start + 1, self.stride_frames):
                windows.append(
                    QNNWindow(
                        name=name,
                        gt_dir=self.gt_root / name,
                        spad_path=self.spad_root / name / "frames.npy",
                        gt_start=gt_start,
                        output_frames=self.output_frames,
                        spad_step=self.spad_step,
                    )
                )
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        window = self.windows[index]
        img = self._load_raw_window(window)
        cls, bboxes, keypoints, batch_idx = self._labels_for_window(window)
        return {
            "img": torch.from_numpy(img),
            "cls": cls,
            "bboxes": bboxes,
            "keypoints": keypoints,
            "batch_idx": batch_idx,
            "im_file": f"{window.name}:{window.gt_start}",
            "output_frames": window.output_frames,
            "ori_shape": (self.image_size, self.image_size),
            "resized_shape": (self.image_size, self.image_size),
        }

    def _load_raw_window(self, window: QNNWindow) -> np.ndarray:
        spad_start = window.gt_start * self.spad_per_gt
        spad_len = window.output_frames * window.spad_step + 1
        spad_end = spad_start + spad_len

        arr = np.load(window.spad_path, mmap_mode="r")
        if spad_end > arr.shape[0]:
            raise IndexError(f"SPAD slice [{spad_start}:{spad_end}] exceeds {window.spad_path} shape {arr.shape}")

        packed = np.asarray(arr[spad_start:spad_end])
        raw = packed_frames_to_raw_bayer(packed, ch_order=self.packed_ch_order)
        return raw[:, :, :, None].astype(np.uint8, copy=False)

    def _labels_for_window(self, window: QNNWindow):
        ann = self.annotations[window.name]
        cls_ll, bbox_ll, kpt_ll, batch_idx_ll = [], [], [], []

        for out_i in range(window.output_frames):
            gt_time = window.gt_start + ((out_i + 1) * window.spad_step / self.spad_per_gt)
            for hand_name, cls_id in self.HAND_TO_CLASS.items():
                hand = self._interpolate_hand_annotation(ann, gt_time, hand_name)
                if not hand:
                    continue
                cls_ll.append([float(cls_id)])
                bbox_ll.append(self._xyxy_to_normalized_xywh(hand["bbox"]))
                kpt_ll.append(self._keypoints_to_normalized_xyv(hand["keypoints_2d"]))
                batch_idx_ll.append([float(out_i)])

        if cls_ll:
            cls = torch.tensor(cls_ll, dtype=torch.float32)
            bboxes = torch.tensor(bbox_ll, dtype=torch.float32)
            keypoints = torch.tensor(kpt_ll, dtype=torch.float32)
            batch_idx = torch.tensor(batch_idx_ll, dtype=torch.float32)
        else:
            cls = torch.zeros((0, 1), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            keypoints = torch.zeros((0, 21, 3), dtype=torch.float32)
            batch_idx = torch.zeros((0, 1), dtype=torch.float32)
        return cls, bboxes, keypoints, batch_idx

    def _interpolate_hand_annotation(self, ann: dict[str, Any], gt_time: float, hand_name: str):
        lo = int(np.floor(gt_time))
        hi = int(np.ceil(gt_time))
        alpha = float(gt_time - lo)
        lo_hand = ann.get(f"frame_{lo:06d}.png", {}).get(hand_name)
        hi_hand = ann.get(f"frame_{hi:06d}.png", {}).get(hand_name)
        if lo_hand is None and hi_hand is None:
            return None
        if hi_hand is None or alpha == 0.0:
            return lo_hand
        if lo_hand is None:
            return hi_hand

        bbox = (1.0 - alpha) * np.asarray(lo_hand["bbox"], dtype=np.float32) + alpha * np.asarray(
            hi_hand["bbox"], dtype=np.float32
        )
        keypoints = (1.0 - alpha) * np.asarray(lo_hand["keypoints_2d"], dtype=np.float32) + alpha * np.asarray(
            hi_hand["keypoints_2d"], dtype=np.float32
        )
        return {"bbox": bbox.tolist(), "keypoints_2d": keypoints.tolist()}

    def _xyxy_to_normalized_xywh(self, bbox) -> list[float]:
        x1, y1, x2, y2 = [float(x) for x in bbox]
        w = max(x2 - x1, 0.0)
        h = max(y2 - y1, 0.0)
        return [
            (x1 + w / 2.0) / self.image_size,
            (y1 + h / 2.0) / self.image_size,
            w / self.image_size,
            h / self.image_size,
        ]

    def _keypoints_to_normalized_xyv(self, keypoints) -> list[list[float]]:
        out = []
        for x, y in keypoints:
            out.append([float(x) / self.image_size, float(y) / self.image_size, 1.0])
        return out

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        new_batch = {}
        new_batch["img"] = torch.stack([b["img"] for b in batch], 0)
        new_batch["cls"] = torch.cat([b["cls"] for b in batch], 0)
        new_batch["bboxes"] = torch.cat([b["bboxes"] for b in batch], 0)
        new_batch["keypoints"] = torch.cat([b["keypoints"] for b in batch], 0)

        batch_idx = []
        t_offset = 0
        for b in batch:
            idx = b["batch_idx"].clone()
            if idx.numel():
                idx += t_offset
            batch_idx.append(idx)
            t_offset += int(b["output_frames"])
        new_batch["batch_idx"] = torch.cat(batch_idx, 0)

        new_batch["im_file"] = [b["im_file"] for b in batch]
        new_batch["output_frames"] = [b["output_frames"] for b in batch]
        new_batch["ori_shape"] = [b["ori_shape"] for b in batch]
        new_batch["resized_shape"] = [b["resized_shape"] for b in batch]
        return new_batch
