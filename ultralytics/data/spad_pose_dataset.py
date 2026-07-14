"""SPAD pose dataset for VisionSIM hand annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.spad_packed import infer_packed_nch, packed_frames_to_raw_video
from ultralytics.data.spad_render_cache import (
    CACHE_META_VERSION,
    render_config_fingerprint,
    sample_render_dir,
    sibling_sample_render_dir,
)


@dataclass(frozen=True)
class SpadPoseSequenceWindow:
    """A fixed temporal training window inside one video."""

    name: str
    gt_ann_path: Path
    spad_path: Path
    gt_start: int
    output_frames: int
    spad_step: int


def load_visionsim_split_json(path: str | Path) -> list[dict[str, str]]:
    """Load VisionSIM train/test split JSON produced by build_visionsim_split.py."""
    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"Split JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, dict) or not raw_samples:
        raise ValueError(f"No samples found in split JSON: {json_path}")

    def _abs_path(value: str) -> Path:
        # Prefer absolute() so symlink-based renders-spc8kHz paths stay intact.
        # exists() still follows the symlink to validate the target.
        path_value = Path(value)
        return path_value if path_value.is_absolute() else (json_path.parent / path_value).absolute()

    records: list[dict[str, str]] = []
    for sample_id, entry in sorted(raw_samples.items()):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid sample entry for {sample_id!r} in {json_path}")
        gt = entry.get("gt")
        spad = entry.get("spad")
        if not gt or not spad:
            raise ValueError(f"Sample {sample_id!r} must include gt and spad paths in {json_path}")

        gt_path = _abs_path(gt)
        spad_path = _abs_path(spad)
        if not gt_path.exists():
            raise FileNotFoundError(f"GT annotation not found for {sample_id!r}: {gt_path}")
        if not spad_path.exists():
            raise FileNotFoundError(f"SPAD frames not found for {sample_id!r}: {spad_path}")

        record = {
            "id": str(sample_id),
            "gt": str(gt_path),
            "spad": str(spad_path),
            "version": str(entry.get("version", "")),
            "name": str(entry.get("name", Path(sample_id).name)),
        }
        rgb = entry.get("rgb")
        if rgb:
            record["rgb"] = str(_abs_path(rgb))
        for key, value in entry.items():
            if not isinstance(key, str) or not value:
                continue
            if key in {"stea", "sum", "ema", "ppb", "hire"} or key.startswith("render_") or key.endswith("_confidence") or key.endswith("_meta"):
                if isinstance(value, str):
                    record[key] = str(_abs_path(value))
        records.append(record)

    return records


class SpadPoseSequenceDataset(Dataset):
    """Load fixed windows from packed SPAD videos and timestamp-aligned hand pose labels."""

    HAND_TO_CLASS = {"left_hand": 0, "right_hand": 1}

    def __init__(
        self,
        samples: list[dict[str, str]],
        *,
        output_frames: int = 4,
        spad_bins_per_gt: int = 64,
        spad_step: int | None = None,
        stride_frames: int | None = None,
        image_size: int = 512,
        packed_ch_order: str = "RGB",
    ):
        if not samples:
            raise ValueError("samples must be a non-empty list")

        self.output_frames = int(output_frames)
        self.spad_bins_per_gt = int(spad_bins_per_gt)
        self.spad_step = int(spad_step or spad_bins_per_gt)
        self.stride_frames = int(stride_frames or output_frames)
        self.image_size = int(image_size)
        self.packed_ch_order = packed_ch_order.upper()

        if self.output_frames <= 0:
            raise ValueError(f"output_frames must be > 0, got {self.output_frames}")
        if self.spad_bins_per_gt <= 0:
            raise ValueError(f"spad_bins_per_gt must be > 0, got {self.spad_bins_per_gt}")
        if self.spad_step <= 0:
            raise ValueError(f"spad_step must be > 0, got {self.spad_step}")

        self.sample_records = {rec["id"]: rec for rec in samples}
        self.video_names = sorted(self.sample_records)
        self.annotations = {name: self._load_annotation(name) for name in self.video_names}
        self.windows = self._build_windows()
        if not self.windows:
            raise RuntimeError(f"No SPAD pose windows found for {len(self.video_names)} samples")
        self.labels = self._build_ultralytics_labels()
        self.im_files = [str(lb["im_file"]) for lb in self.labels]
        self.ni = len(self.labels)

    def _load_annotation(self, name: str) -> dict[str, Any]:
        path = Path(self.sample_records[name]["gt"])
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_windows(self) -> list[SpadPoseSequenceWindow]:
        windows: list[SpadPoseSequenceWindow] = []
        for name in self.video_names:
            ann = self.annotations[name]
            n_gt = len(ann)
            last_gt_offset = self.output_frames * self.spad_step / self.spad_bins_per_gt
            max_start = int(np.floor((n_gt - 1) - last_gt_offset))
            if max_start < 0:
                continue
            spad_path = Path(self.sample_records[name]["spad"])
            gt_ann_path = Path(self.sample_records[name]["gt"])
            for gt_start in range(0, max_start + 1, self.stride_frames):
                windows.append(
                    SpadPoseSequenceWindow(
                        name=name,
                        gt_ann_path=gt_ann_path,
                        spad_path=spad_path,
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
        img, packed_nch = self._load_raw_window(window)
        cls, bboxes, keypoints, batch_idx = self._labels_for_window(window)
        return {
            "img": torch.from_numpy(img),
            "packed_nch": int(packed_nch),
            "cls": cls,
            "bboxes": bboxes,
            "keypoints": keypoints,
            "batch_idx": batch_idx,
            "im_file": f"{window.name}:{window.gt_start}",
            "output_frames": window.output_frames,
            "ori_shape": (self.image_size, self.image_size),
            "resized_shape": (self.image_size, self.image_size),
        }

    def _load_raw_window(self, window: SpadPoseSequenceWindow) -> np.ndarray:
        spad_start = window.gt_start * self.spad_bins_per_gt
        spad_len = window.output_frames * window.spad_step
        spad_end = spad_start + spad_len

        arr = np.load(window.spad_path, mmap_mode="r")
        if spad_end > arr.shape[0]:
            raise IndexError(f"SPAD slice [{spad_start}:{spad_end}] exceeds {window.spad_path} shape {arr.shape}")

        packed = np.asarray(arr[spad_start:spad_end])
        packed_nch = infer_packed_nch(arr)
        raw = packed_frames_to_raw_video(packed, ch_order=self.packed_ch_order)
        return raw.astype(np.uint8, copy=False), packed_nch

    def _labels_for_window(self, window: SpadPoseSequenceWindow):
        ann = self.annotations[window.name]
        cls_ll, bbox_ll, kpt_ll, batch_idx_ll = [], [], [], []

        for out_i in range(window.output_frames):
            gt_time = window.gt_start + ((out_i + 1) * window.spad_step / self.spad_bins_per_gt)
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

    def _build_ultralytics_labels(self) -> list[dict[str, Any]]:
        labels = []
        for window in self.windows:
            cls, bboxes, keypoints, _ = self._labels_for_window(window)
            labels.append(
                {
                    "im_file": f"{window.name}:{window.gt_start}",
                    "shape": (self.image_size, self.image_size),
                    "cls": cls.detach().cpu().numpy(),
                    "bboxes": bboxes.detach().cpu().numpy(),
                    "segments": [],
                    "keypoints": keypoints.detach().cpu().numpy(),
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        return labels

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
        new_batch["packed_nch"] = int(batch[0]["packed_nch"])
        new_batch["ori_shape"] = [b["ori_shape"] for b in batch]
        new_batch["resized_shape"] = [b["resized_shape"] for b in batch]
        return new_batch


@dataclass(frozen=True)
class SpadPoseFrameWindow:
    """One raw SPAD chunk supervised only at the chunk end time."""

    name: str
    gt_ann_path: Path
    spad_path: Path
    gt_start: int
    chunk_size: int


@dataclass(frozen=True)
class SpadPoseRenderedFrameWindow:
    """One cached rendered frame aligned to one end-of-chunk supervision target."""

    name: str
    gt_ann_path: Path
    render_dir: Path
    frame_index: int
    gt_start: int
    target_gt_time: float
    spad_start_bin: int
    spad_end_bin: int
    chunk_size: int
    packed_nch: int


class SpadPoseFrameDataset(Dataset):
    """Load fixed raw chunks and supervise only the end-of-chunk pose annotation."""

    HAND_TO_CLASS = SpadPoseSequenceDataset.HAND_TO_CLASS

    def __init__(
        self,
        samples: list[dict[str, str]],
        *,
        chunk_size: int,
        spad_bins_per_gt: int = 64,
        stride_frames: int | None = None,
        image_size: int = 512,
        packed_ch_order: str = "RGB",
    ):
        if not samples:
            raise ValueError("samples must be a non-empty list")

        self.chunk_size = int(chunk_size)
        self.spad_bins_per_gt = int(spad_bins_per_gt)
        self.stride_frames = int(stride_frames or 1)
        self.image_size = int(image_size)
        self.packed_ch_order = packed_ch_order.upper()

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")
        if self.spad_bins_per_gt <= 0:
            raise ValueError(f"spad_bins_per_gt must be > 0, got {self.spad_bins_per_gt}")
        if self.stride_frames <= 0:
            raise ValueError(f"stride_frames must be > 0, got {self.stride_frames}")

        self.sample_records = {rec["id"]: rec for rec in samples}
        self.video_names = sorted(self.sample_records)
        self.annotations = {name: self._load_annotation(name) for name in self.video_names}
        self.windows = self._build_windows()
        if not self.windows:
            raise RuntimeError(f"No SPAD pose frame windows found for {len(self.video_names)} samples")
        self.labels = self._build_ultralytics_labels()
        self.im_files = [str(lb["im_file"]) for lb in self.labels]
        self.ni = len(self.labels)

    def _load_annotation(self, name: str) -> dict[str, Any]:
        path = Path(self.sample_records[name]["gt"])
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_windows(self) -> list[SpadPoseFrameWindow]:
        windows: list[SpadPoseFrameWindow] = []
        gt_offset = self.chunk_size / self.spad_bins_per_gt
        for name in self.video_names:
            ann = self.annotations[name]
            n_gt = len(ann)
            max_start = int(np.floor((n_gt - 1) - gt_offset))
            if max_start < 0:
                continue
            spad_path = Path(self.sample_records[name]["spad"])
            gt_ann_path = Path(self.sample_records[name]["gt"])
            for gt_start in range(0, max_start + 1, self.stride_frames):
                windows.append(
                    SpadPoseFrameWindow(
                        name=name,
                        gt_ann_path=gt_ann_path,
                        spad_path=spad_path,
                        gt_start=gt_start,
                        chunk_size=self.chunk_size,
                    )
                )
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | int]:
        window = self.windows[index]
        img, packed_nch, spad_start, spad_end = self._load_raw_window(window)
        cls, bboxes, keypoints, batch_idx, target_gt_time = self._labels_for_window(window)
        return {
            "img": torch.from_numpy(img),
            "packed_nch": int(packed_nch),
            "cls": cls,
            "bboxes": bboxes,
            "keypoints": keypoints,
            "batch_idx": batch_idx,
            "im_file": f"{window.name}:{spad_start}:{spad_end}",
            "ori_shape": (self.image_size, self.image_size),
            "resized_shape": (self.image_size, self.image_size),
            "sample_name": window.name,
            "target_gt_time": float(target_gt_time),
            "spad_start_bin": int(spad_start),
            "spad_end_bin": int(spad_end),
            "chunk_size": int(window.chunk_size),
        }

    def _load_raw_window(self, window: SpadPoseFrameWindow) -> tuple[np.ndarray, int, int, int]:
        spad_start = window.gt_start * self.spad_bins_per_gt
        spad_end = spad_start + window.chunk_size

        arr = np.load(window.spad_path, mmap_mode="r")
        if spad_end > arr.shape[0]:
            raise IndexError(f"SPAD slice [{spad_start}:{spad_end}] exceeds {window.spad_path} shape {arr.shape}")

        packed = np.asarray(arr[spad_start:spad_end])
        packed_nch = infer_packed_nch(arr)
        raw = packed_frames_to_raw_video(packed, ch_order=self.packed_ch_order)
        return raw.astype(np.uint8, copy=False), packed_nch, spad_start, spad_end

    def _labels_for_window(self, window: SpadPoseFrameWindow):
        ann = self.annotations[window.name]
        target_gt_time = window.gt_start + (window.chunk_size / self.spad_bins_per_gt)
        cls_ll, bbox_ll, kpt_ll = [], [], []

        for hand_name, cls_id in self.HAND_TO_CLASS.items():
            hand = self._interpolate_hand_annotation(ann, target_gt_time, hand_name)
            if not hand:
                continue
            cls_ll.append([float(cls_id)])
            bbox_ll.append(self._xyxy_to_normalized_xywh(hand["bbox"]))
            kpt_ll.append(self._keypoints_to_normalized_xyv(hand["keypoints_2d"]))

        if cls_ll:
            cls = torch.tensor(cls_ll, dtype=torch.float32)
            bboxes = torch.tensor(bbox_ll, dtype=torch.float32)
            keypoints = torch.tensor(kpt_ll, dtype=torch.float32)
            batch_idx = torch.zeros((len(cls_ll), 1), dtype=torch.float32)
        else:
            cls = torch.zeros((0, 1), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            keypoints = torch.zeros((0, 21, 3), dtype=torch.float32)
            batch_idx = torch.zeros((0, 1), dtype=torch.float32)
        return cls, bboxes, keypoints, batch_idx, target_gt_time

    def _interpolate_hand_annotation(self, ann: dict[str, Any], gt_time: float, hand_name: str):
        return SpadPoseSequenceDataset._interpolate_hand_annotation(self, ann, gt_time, hand_name)

    def _xyxy_to_normalized_xywh(self, bbox) -> list[float]:
        return SpadPoseSequenceDataset._xyxy_to_normalized_xywh(self, bbox)

    def _keypoints_to_normalized_xyv(self, keypoints) -> list[list[float]]:
        return SpadPoseSequenceDataset._keypoints_to_normalized_xyv(self, keypoints)

    def _build_ultralytics_labels(self) -> list[dict[str, Any]]:
        labels = []
        for window in self.windows:
            cls, bboxes, keypoints, _, _ = self._labels_for_window(window)
            spad_start = window.gt_start * self.spad_bins_per_gt
            spad_end = spad_start + window.chunk_size
            labels.append(
                {
                    "im_file": f"{window.name}:{spad_start}:{spad_end}",
                    "shape": (self.image_size, self.image_size),
                    "cls": cls.detach().cpu().numpy(),
                    "bboxes": bboxes.detach().cpu().numpy(),
                    "segments": [],
                    "keypoints": keypoints.detach().cpu().numpy(),
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        return labels

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        new_batch = {}
        new_batch["img"] = torch.stack([b["img"] for b in batch], 0)
        new_batch["cls"] = torch.cat([b["cls"] for b in batch], 0)
        new_batch["bboxes"] = torch.cat([b["bboxes"] for b in batch], 0)
        new_batch["keypoints"] = torch.cat([b["keypoints"] for b in batch], 0)

        batch_idx = []
        for sample_i, b in enumerate(batch):
            idx = b["batch_idx"].clone()
            if idx.numel():
                idx += float(sample_i)
            batch_idx.append(idx)
        new_batch["batch_idx"] = torch.cat(batch_idx, 0) if batch_idx else torch.zeros((0, 1), dtype=torch.float32)

        new_batch["im_file"] = [b["im_file"] for b in batch]
        new_batch["packed_nch"] = int(batch[0]["packed_nch"])
        new_batch["ori_shape"] = [b["ori_shape"] for b in batch]
        new_batch["resized_shape"] = [b["resized_shape"] for b in batch]
        new_batch["sample_name"] = [b["sample_name"] for b in batch]
        new_batch["target_gt_time"] = torch.tensor([b["target_gt_time"] for b in batch], dtype=torch.float32)
        new_batch["spad_start_bin"] = torch.tensor([b["spad_start_bin"] for b in batch], dtype=torch.long)
        new_batch["spad_end_bin"] = torch.tensor([b["spad_end_bin"] for b in batch], dtype=torch.long)
        new_batch["chunk_size"] = torch.tensor([b["chunk_size"] for b in batch], dtype=torch.long)
        return new_batch


class SpadPoseRenderedFrameDataset(Dataset):
    """Load cached rendered frame chunks plus optional confidence maps for frame-mode training."""

    HAND_TO_CLASS = SpadPoseSequenceDataset.HAND_TO_CLASS

    def __init__(
        self,
        samples: list[dict[str, str]],
        *,
        render_root: str | Path | None,
        preprocessor: str,
        image_size: int = 512,
        render_contains_confidence: bool = True,
        expected_render_config: dict[str, Any] | None = None,
        source_render_dirname: str = "renders-spc8kHz",
    ):
        if not samples:
            raise ValueError("samples must be a non-empty list")

        self.render_root = None if render_root in {None, ""} else Path(render_root)
        self.preprocessor = str(preprocessor).strip().lower()
        self.image_size = int(image_size)
        self.render_contains_confidence = bool(render_contains_confidence)
        self.expected_render_config = dict(expected_render_config or {})
        self.source_render_dirname = str(source_render_dirname).strip()
        self.expected_render_fingerprint = (
            render_config_fingerprint(self.expected_render_config) if self.expected_render_config else None
        )

        self.sample_records = {rec["id"]: rec for rec in samples}
        self.video_names = sorted(self.sample_records)
        self.annotations = {name: self._load_annotation(name) for name in self.video_names}
        self.windows = self._build_windows()
        if not self.windows:
            render_hint = self.render_root if self.render_root is not None else f"<sibling:{self.source_render_dirname}>"
            raise RuntimeError(f"No cached rendered SPAD frame windows found under {render_hint}")
        self.labels = self._build_ultralytics_labels()
        self.im_files = [str(lb["im_file"]) for lb in self.labels]
        self.ni = len(self.labels)
        self._frames_cache: dict[str, np.ndarray] = {}
        self._confidence_cache: dict[str, np.ndarray | None] = {}
        self._frames_path_cache: dict[str, Path] = {}
        self._confidence_path_cache: dict[str, Path | None] = {}

    def _load_annotation(self, name: str) -> dict[str, Any]:
        path = Path(self.sample_records[name]["gt"])
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _sample_meta_paths(self, name: str) -> tuple[Path, Path, Path]:
        sample_record = self.sample_records[name]
        sample_name = sample_record["name"]
        explicit_frames = sample_record.get(self.preprocessor) or sample_record.get(f"render_{self.preprocessor}_frames")
        explicit_render_dir = sample_record.get(f"render_{self.preprocessor}")
        explicit_meta = sample_record.get(f"{self.preprocessor}_meta") or sample_record.get(f"render_{self.preprocessor}_meta")

        if explicit_frames or explicit_render_dir:
            if explicit_frames:
                frames_path = Path(explicit_frames)
                render_dir = frames_path.parent
            else:
                render_dir = Path(explicit_render_dir)
                frames_path = render_dir / "frames.npy"
            meta_path = Path(explicit_meta) if explicit_meta else render_dir / "meta.json"
            return render_dir, frames_path, meta_path

        if self.render_root is not None:
            render_dir = sample_render_dir(self.render_root, sample_name)
        else:
            render_dir = sibling_sample_render_dir(
                sample_record["spad"],
                preprocessor=self.preprocessor,
                sample_name=sample_name,
                source_render_dirname=self.source_render_dirname,
            )
        return render_dir, render_dir / "frames.npy", render_dir / "meta.json"

    def _build_windows(self) -> list[SpadPoseRenderedFrameWindow]:
        windows: list[SpadPoseRenderedFrameWindow] = []
        for name in self.video_names:
            render_dir, frames_path, meta_path = self._sample_meta_paths(name)
            if not meta_path.is_file():
                raise FileNotFoundError(f"Cached render metadata not found for {name!r}: {meta_path}")
            if not frames_path.is_file():
                raise FileNotFoundError(f"Cached frames not found for {name!r}: {frames_path}")

            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            version = int(meta.get("version", -1))
            if version != CACHE_META_VERSION:
                raise ValueError(f"Unsupported cache meta version for {meta_path}: {version}")

            if self.expected_render_fingerprint is not None:
                cached = str(meta.get("config_fingerprint", "")).strip()
                if cached != self.expected_render_fingerprint:
                    raise ValueError(
                        f"Cached render fingerprint mismatch for {name!r}: expected "
                        f"{self.expected_render_fingerprint}, got {cached or '<missing>'}"
                    )
            cached_preprocessor = str(meta.get("preprocessor", "")).strip().lower()
            if cached_preprocessor and cached_preprocessor != self.preprocessor:
                raise ValueError(
                    f"Cached render preprocessor mismatch for {name!r}: expected {self.preprocessor!r}, "
                    f"got {cached_preprocessor!r}"
                )
            source_gt = meta.get("source_gt")
            if source_gt and Path(source_gt).resolve() != Path(self.sample_records[name]["gt"]).resolve():
                raise ValueError(
                    f"Cached render GT mismatch for {name!r}: meta points to {source_gt}, "
                    f"sample uses {self.sample_records[name]['gt']}"
                )
            source_spad = meta.get("source_spad")
            if source_spad and Path(source_spad).resolve() != Path(self.sample_records[name]["spad"]).resolve():
                raise ValueError(
                    f"Cached render SPAD mismatch for {name!r}: meta points to {source_spad}, "
                    f"sample uses {self.sample_records[name]['spad']}"
                )

            chunks = meta.get("chunks")
            if not isinstance(chunks, list) or not chunks:
                raise ValueError(f"Cached render metadata must include non-empty chunks list: {meta_path}")

            for chunk in chunks:
                windows.append(
                    SpadPoseRenderedFrameWindow(
                        name=name,
                        gt_ann_path=Path(self.sample_records[name]["gt"]),
                        render_dir=render_dir,
                        frame_index=int(chunk["chunk_index"]),
                        gt_start=int(chunk["gt_start"]),
                        target_gt_time=float(chunk["target_gt_time"]),
                        spad_start_bin=int(chunk["spad_start_bin"]),
                        spad_end_bin=int(chunk["spad_end_bin"]),
                        chunk_size=int(chunk["chunk_size"]),
                        packed_nch=int(chunk.get("packed_nch", 3)),
                    )
                )
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def _frames_array(self, window: SpadPoseRenderedFrameWindow) -> np.ndarray:
        key = str(window.render_dir)
        frames = self._frames_cache.get(key)
        if frames is None:
            frames_path = self._frames_path_cache.get(key)
            if frames_path is None:
                frames_path = self._sample_meta_paths(window.name)[1]
                self._frames_path_cache[key] = frames_path
            frames = np.load(frames_path, mmap_mode="r")
            self._frames_cache[key] = frames
        return frames

    def _confidence_array(self, window: SpadPoseRenderedFrameWindow) -> np.ndarray | None:
        key = str(window.render_dir)
        if key in self._confidence_cache:
            return self._confidence_cache[key]
        path = self._confidence_path_cache.get(key)
        if path is None:
            sample_record = self.sample_records[window.name]
            path_str = sample_record.get(f"{self.preprocessor}_confidence") or sample_record.get(
                f"render_{self.preprocessor}_confidence"
            )
            path = Path(path_str) if path_str else (window.render_dir / "confidence.npy")
            self._confidence_path_cache[key] = path
        if self.render_contains_confidence and not path.is_file():
            raise FileNotFoundError(f"Cached confidence required but not found: {path}")
        conf = np.load(path, mmap_mode="r") if path.is_file() else None
        self._confidence_cache[key] = conf
        return conf

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | int]:
        window = self.windows[index]
        frame = np.array(self._frames_array(window)[window.frame_index], copy=True)
        conf_arr = self._confidence_array(window)
        confidence = None if conf_arr is None else np.array(conf_arr[window.frame_index], copy=True)
        cls, bboxes, keypoints, batch_idx = self._labels_for_window(window)

        img = torch.from_numpy(frame)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        else:
            img = img.float()

        if confidence is None:
            conf_tensor = torch.zeros((1, img.shape[-2], img.shape[-1]), dtype=torch.float32)
        else:
            conf_tensor = torch.from_numpy(confidence).float()
            if conf_tensor.ndim == 2:
                conf_tensor = conf_tensor.unsqueeze(0)

        return {
            "img": img,
            "confidence": conf_tensor,
            "packed_nch": int(window.packed_nch),
            "cls": cls,
            "bboxes": bboxes,
            "keypoints": keypoints,
            "batch_idx": batch_idx,
            "im_file": f"{window.name}:{window.spad_start_bin}:{window.spad_end_bin}",
            "ori_shape": (self.image_size, self.image_size),
            "resized_shape": (self.image_size, self.image_size),
            "sample_name": window.name,
            "target_gt_time": float(window.target_gt_time),
            "spad_start_bin": int(window.spad_start_bin),
            "spad_end_bin": int(window.spad_end_bin),
            "chunk_size": int(window.chunk_size),
        }

    def _labels_for_window(self, window: SpadPoseRenderedFrameWindow):
        ann = self.annotations[window.name]
        cls_ll, bbox_ll, kpt_ll = [], [], []
        for hand_name, cls_id in self.HAND_TO_CLASS.items():
            hand = self._interpolate_hand_annotation(ann, window.target_gt_time, hand_name)
            if not hand:
                continue
            cls_ll.append([float(cls_id)])
            bbox_ll.append(self._xyxy_to_normalized_xywh(hand["bbox"]))
            kpt_ll.append(self._keypoints_to_normalized_xyv(hand["keypoints_2d"]))

        if cls_ll:
            cls = torch.tensor(cls_ll, dtype=torch.float32)
            bboxes = torch.tensor(bbox_ll, dtype=torch.float32)
            keypoints = torch.tensor(kpt_ll, dtype=torch.float32)
            batch_idx = torch.zeros((len(cls_ll), 1), dtype=torch.float32)
        else:
            cls = torch.zeros((0, 1), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            keypoints = torch.zeros((0, 21, 3), dtype=torch.float32)
            batch_idx = torch.zeros((0, 1), dtype=torch.float32)
        return cls, bboxes, keypoints, batch_idx

    def _interpolate_hand_annotation(self, ann: dict[str, Any], gt_time: float, hand_name: str):
        return SpadPoseSequenceDataset._interpolate_hand_annotation(self, ann, gt_time, hand_name)

    def _xyxy_to_normalized_xywh(self, bbox) -> list[float]:
        return SpadPoseSequenceDataset._xyxy_to_normalized_xywh(self, bbox)

    def _keypoints_to_normalized_xyv(self, keypoints) -> list[list[float]]:
        return SpadPoseSequenceDataset._keypoints_to_normalized_xyv(self, keypoints)

    def _build_ultralytics_labels(self) -> list[dict[str, Any]]:
        labels = []
        for window in self.windows:
            cls, bboxes, keypoints, _ = self._labels_for_window(window)
            labels.append(
                {
                    "im_file": f"{window.name}:{window.spad_start_bin}:{window.spad_end_bin}",
                    "shape": (self.image_size, self.image_size),
                    "cls": cls.detach().cpu().numpy(),
                    "bboxes": bboxes.detach().cpu().numpy(),
                    "segments": [],
                    "keypoints": keypoints.detach().cpu().numpy(),
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        return labels

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        new_batch = {}
        new_batch["img"] = torch.stack([b["img"] for b in batch], 0)
        new_batch["confidence"] = torch.stack([b["confidence"] for b in batch], 0)
        new_batch["cls"] = torch.cat([b["cls"] for b in batch], 0)
        new_batch["bboxes"] = torch.cat([b["bboxes"] for b in batch], 0)
        new_batch["keypoints"] = torch.cat([b["keypoints"] for b in batch], 0)

        batch_idx = []
        for sample_i, b in enumerate(batch):
            idx = b["batch_idx"].clone()
            if idx.numel():
                idx += float(sample_i)
            batch_idx.append(idx)
        new_batch["batch_idx"] = torch.cat(batch_idx, 0) if batch_idx else torch.zeros((0, 1), dtype=torch.float32)

        new_batch["im_file"] = [b["im_file"] for b in batch]
        new_batch["packed_nch"] = int(batch[0]["packed_nch"])
        new_batch["ori_shape"] = [b["ori_shape"] for b in batch]
        new_batch["resized_shape"] = [b["resized_shape"] for b in batch]
        new_batch["sample_name"] = [b["sample_name"] for b in batch]
        new_batch["target_gt_time"] = torch.tensor([b["target_gt_time"] for b in batch], dtype=torch.float32)
        new_batch["spad_start_bin"] = torch.tensor([b["spad_start_bin"] for b in batch], dtype=torch.long)
        new_batch["spad_end_bin"] = torch.tensor([b["spad_end_bin"] for b in batch], dtype=torch.long)
        new_batch["chunk_size"] = torch.tensor([b["chunk_size"] for b in batch], dtype=torch.long)
        return new_batch


@dataclass(frozen=True)
class SpadPoseRenderedSequenceWindow:
    """A fixed-length sequence of consecutive cached rendered frames."""

    name: str
    gt_ann_path: Path
    render_dir: Path
    chunks: tuple[dict[str, Any], ...]
    packed_nch: int


class SpadPoseRenderedSequenceDataset(Dataset):
    """Load consecutive cached rendered frames for sequence-mode SSD training."""

    HAND_TO_CLASS = SpadPoseSequenceDataset.HAND_TO_CLASS

    def __init__(
        self,
        samples: list[dict[str, str]],
        *,
        render_root: str | Path | None,
        preprocessor: str,
        output_frames: int = 10,
        spad_bins_per_gt: int = 64,
        stride_frames: int | None = None,
        image_size: int = 512,
        render_contains_confidence: bool = True,
        expected_render_config: dict[str, Any] | None = None,
        source_render_dirname: str = "renders-spc8kHz",
    ):
        if not samples:
            raise ValueError("samples must be a non-empty list")

        self.render_root = None if render_root in {None, ""} else Path(render_root)
        self.preprocessor = str(preprocessor).strip().lower()
        self.output_frames = int(output_frames)
        self.spad_bins_per_gt = int(spad_bins_per_gt)
        self.stride_frames = int(stride_frames or 5)
        self.image_size = int(image_size)
        self.render_contains_confidence = bool(render_contains_confidence)
        self.expected_render_config = dict(expected_render_config or {})
        self.source_render_dirname = str(source_render_dirname).strip()
        self.expected_render_fingerprint = (
            render_config_fingerprint(self.expected_render_config) if self.expected_render_config else None
        )

        if self.output_frames <= 0:
            raise ValueError(f"output_frames must be > 0, got {self.output_frames}")
        if self.spad_bins_per_gt <= 0:
            raise ValueError(f"spad_bins_per_gt must be > 0, got {self.spad_bins_per_gt}")
        if self.stride_frames <= 0:
            raise ValueError(f"stride_frames must be > 0, got {self.stride_frames}")

        self.sample_records = {rec["id"]: rec for rec in samples}
        self.video_names = sorted(self.sample_records)
        self.annotations = {name: self._load_annotation(name) for name in self.video_names}
        self._sample_chunks: dict[str, list[dict[str, Any]]] = {}
        self._sample_render_dirs: dict[str, Path] = {}
        self.windows = self._build_windows()
        if not self.windows:
            render_hint = self.render_root if self.render_root is not None else f"<sibling:{self.source_render_dirname}>"
            raise RuntimeError(f"No cached rendered SPAD sequence windows found under {render_hint}")
        self.labels = self._build_ultralytics_labels()
        self.im_files = [str(lb["im_file"]) for lb in self.labels]
        self.ni = len(self.labels)
        self._frames_cache: dict[str, np.ndarray] = {}

    def _load_annotation(self, name: str) -> dict[str, Any]:
        path = Path(self.sample_records[name]["gt"])
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _sample_meta_paths(self, name: str) -> tuple[Path, Path, Path]:
        sample_record = self.sample_records[name]
        sample_name = sample_record["name"]
        explicit_frames = sample_record.get(self.preprocessor) or sample_record.get(f"render_{self.preprocessor}_frames")
        explicit_render_dir = sample_record.get(f"render_{self.preprocessor}")
        explicit_meta = sample_record.get(f"{self.preprocessor}_meta") or sample_record.get(f"render_{self.preprocessor}_meta")

        if explicit_frames or explicit_render_dir:
            if explicit_frames:
                frames_path = Path(explicit_frames)
                render_dir = frames_path.parent
            else:
                render_dir = Path(explicit_render_dir)
                frames_path = render_dir / "frames.npy"
            meta_path = Path(explicit_meta) if explicit_meta else render_dir / "meta.json"
            return render_dir, frames_path, meta_path

        if self.render_root is not None:
            render_dir = sample_render_dir(self.render_root, sample_name)
        else:
            render_dir = sibling_sample_render_dir(
                sample_record["spad"],
                preprocessor=self.preprocessor,
                sample_name=sample_name,
                source_render_dirname=self.source_render_dirname,
            )
        return render_dir, render_dir / "frames.npy", render_dir / "meta.json"

    def _load_sample_chunks(self, name: str) -> tuple[Path, list[dict[str, Any]]]:
        if name in self._sample_chunks:
            return self._sample_render_dirs[name], self._sample_chunks[name]

        render_dir, frames_path, meta_path = self._sample_meta_paths(name)
        if not meta_path.is_file():
            raise FileNotFoundError(f"Cached render metadata not found for {name!r}: {meta_path}")
        if not frames_path.is_file():
            raise FileNotFoundError(f"Cached frames not found for {name!r}: {frames_path}")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        version = int(meta.get("version", -1))
        if version != CACHE_META_VERSION:
            raise ValueError(f"Unsupported cache meta version for {meta_path}: {version}")

        if self.expected_render_fingerprint is not None:
            cached = str(meta.get("config_fingerprint", "")).strip()
            if cached != self.expected_render_fingerprint:
                raise ValueError(
                    f"Cached render fingerprint mismatch for {name!r}: expected "
                    f"{self.expected_render_fingerprint}, got {cached or '<missing>'}"
                )
        cached_preprocessor = str(meta.get("preprocessor", "")).strip().lower()
        if cached_preprocessor and cached_preprocessor != self.preprocessor:
            raise ValueError(
                f"Cached render preprocessor mismatch for {name!r}: expected {self.preprocessor!r}, "
                f"got {cached_preprocessor!r}"
            )

        chunks = meta.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"Cached render metadata must include non-empty chunks list: {meta_path}")

        chunks = sorted(chunks, key=lambda c: int(c["chunk_index"]))
        self._sample_render_dirs[name] = render_dir
        self._sample_chunks[name] = chunks
        return render_dir, chunks

    def _build_windows(self) -> list[SpadPoseRenderedSequenceWindow]:
        windows: list[SpadPoseRenderedSequenceWindow] = []
        last_gt_offset = self.output_frames * self.spad_bins_per_gt / float(self.spad_bins_per_gt)

        for name in self.video_names:
            ann = self.annotations[name]
            n_gt = len(ann)
            max_start = int(np.floor((n_gt - 1) - last_gt_offset))
            if max_start < 0:
                continue

            render_dir, chunks = self._load_sample_chunks(name)
            gt_ann_path = Path(self.sample_records[name]["gt"])
            for gt_start in range(0, max_start + 1, self.stride_frames):
                ci = gt_start // self.stride_frames
                if ci + self.output_frames > len(chunks):
                    break
                window_chunks = chunks[ci : ci + self.output_frames]
                if len(window_chunks) != self.output_frames:
                    continue
                chunk_indices = [int(chunk["chunk_index"]) for chunk in window_chunks]
                if chunk_indices != list(range(chunk_indices[0], chunk_indices[0] + self.output_frames)):
                    continue
                if int(window_chunks[0].get("gt_start", gt_start)) != int(gt_start):
                    continue
                windows.append(
                    SpadPoseRenderedSequenceWindow(
                        name=name,
                        gt_ann_path=gt_ann_path,
                        render_dir=render_dir,
                        chunks=tuple(window_chunks),
                        packed_nch=int(window_chunks[0].get("packed_nch", 3)),
                    )
                )
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def _frames_array(self, window: SpadPoseRenderedSequenceWindow) -> np.ndarray:
        key = str(window.render_dir)
        frames = self._frames_cache.get(key)
        if frames is None:
            frames_path = self._sample_meta_paths(window.name)[1]
            frames = np.load(frames_path, mmap_mode="r")
            self._frames_cache[key] = frames
        return frames

    def _t_index_ll_for_window(self, window: SpadPoseRenderedSequenceWindow) -> list[int]:
        return [int(chunk["spad_end_bin"]) for chunk in window.chunks]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | int | list[int]]:
        window = self.windows[index]
        frames_arr = self._frames_array(window)
        frame_ll = []
        for chunk in window.chunks:
            frame = np.array(frames_arr[int(chunk["chunk_index"])], copy=True)
            img = torch.from_numpy(frame)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            else:
                img = img.float()
            frame_ll.append(img)
        img = torch.stack(frame_ll, dim=0)

        cls, bboxes, keypoints, batch_idx = self._labels_for_window(window)
        first_chunk = window.chunks[0]
        last_chunk = window.chunks[-1]
        return {
            "img": img,
            "packed_nch": int(window.packed_nch),
            "cls": cls,
            "bboxes": bboxes,
            "keypoints": keypoints,
            "batch_idx": batch_idx,
            "output_frames": int(self.output_frames),
            "t_index_ll": self._t_index_ll_for_window(window),
            "im_file": (
                f"{window.name}:{int(first_chunk['spad_start_bin'])}:"
                f"{int(last_chunk['spad_end_bin'])}"
            ),
            "ori_shape": (self.image_size, self.image_size),
            "resized_shape": (self.image_size, self.image_size),
            "sample_name": window.name,
            "spad_start_bin": int(first_chunk["spad_start_bin"]),
            "spad_end_bin": int(last_chunk["spad_end_bin"]),
            "chunk_size": int(first_chunk.get("chunk_size", self.output_frames * self.spad_bins_per_gt)),
        }

    def _labels_for_window(self, window: SpadPoseRenderedSequenceWindow):
        ann = self.annotations[window.name]
        cls_ll, bbox_ll, kpt_ll, batch_idx_ll = [], [], [], []

        for out_i, chunk in enumerate(window.chunks):
            gt_time = float(chunk["target_gt_time"])
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
        return SpadPoseSequenceDataset._interpolate_hand_annotation(self, ann, gt_time, hand_name)

    def _xyxy_to_normalized_xywh(self, bbox) -> list[float]:
        return SpadPoseSequenceDataset._xyxy_to_normalized_xywh(self, bbox)

    def _keypoints_to_normalized_xyv(self, keypoints) -> list[list[float]]:
        return SpadPoseSequenceDataset._keypoints_to_normalized_xyv(self, keypoints)

    def _build_ultralytics_labels(self) -> list[dict[str, Any]]:
        labels = []
        for window in self.windows:
            cls, bboxes, keypoints, _ = self._labels_for_window(window)
            first_chunk = window.chunks[0]
            last_chunk = window.chunks[-1]
            labels.append(
                {
                    "im_file": (
                        f"{window.name}:{int(first_chunk['spad_start_bin'])}:"
                        f"{int(last_chunk['spad_end_bin'])}"
                    ),
                    "shape": (self.image_size, self.image_size),
                    "cls": cls.detach().cpu().numpy(),
                    "bboxes": bboxes.detach().cpu().numpy(),
                    "segments": [],
                    "keypoints": keypoints.detach().cpu().numpy(),
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        return labels

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
        new_batch["batch_idx"] = torch.cat(batch_idx, 0) if batch_idx else torch.zeros((0, 1), dtype=torch.float32)

        new_batch["im_file"] = [b["im_file"] for b in batch]
        new_batch["output_frames"] = [b["output_frames"] for b in batch]
        new_batch["packed_nch"] = int(batch[0]["packed_nch"])
        new_batch["ori_shape"] = [b["ori_shape"] for b in batch]
        new_batch["resized_shape"] = [b["resized_shape"] for b in batch]
        new_batch["sample_name"] = [b["sample_name"] for b in batch]
        new_batch["spad_start_bin"] = torch.tensor([b["spad_start_bin"] for b in batch], dtype=torch.long)
        new_batch["spad_end_bin"] = torch.tensor([b["spad_end_bin"] for b in batch], dtype=torch.long)
        new_batch["chunk_size"] = torch.tensor([b["chunk_size"] for b in batch], dtype=torch.long)
        new_batch["t_index_ll"] = batch[0]["t_index_ll"]
        return new_batch


# Backward-compatible aliases while the codebase transitions to explicit Sequence/Frame naming.
SpadWindow = SpadPoseSequenceWindow
SpadPoseDataset = SpadPoseSequenceDataset
