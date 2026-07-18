from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.models.yolo.pose.train import RgbPoseTrainer
from ultralytics.utils import YAML


def _parse_value(value: str) -> Any:
    text = str(value).strip()
    lowered = text.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if "," in text and not text.startswith("[") and not text.startswith("{"):
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) > 1:
            return [_parse_value(p) for p in parts]
    try:
        return YAML.load(text)
    except Exception:
        return text


def _load_cfg(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data = YAML.load(cfg_path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML in {cfg_path}, got {type(data).__name__}")
    return data


def _apply_kv_overrides(cfg: dict[str, Any], items: list[str]) -> dict[str, Any]:
    updated = dict(cfg)
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        updated[key.strip()] = _parse_value(value)
    return updated


def build_train_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg.pop("spad_train_mode", None)  # unused; RGB trainer is always frame-aligned
    model = cfg.pop("model", "weights/detector.pt")
    # RGB finetune always opens the detector unless freeze= is set explicitly.
    cfg.setdefault("spad_freeze_detector", False)
    return {"model": model, **cfg}


def run_train(cfg_path: str | Path, *, overrides: list[str] | None = None):
    cfg = _load_cfg(cfg_path)
    if overrides:
        cfg = _apply_kv_overrides(cfg, overrides)
    train_kwargs = build_train_kwargs(cfg)
    model_path = train_kwargs.pop("model")
    yolo = YOLO(model_path)
    return yolo.train(trainer=RgbPoseTrainer, **train_kwargs)


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Finetune original YOLO pose detector.pt on VisionSIM RGB frames.npy "
            "(aligned with 2 kHz SPAD train/test JSON / GT timing)."
        )
    )
    ap.add_argument("--cfg", type=str, required=True, help="Path to YAML config under train_cfg/rgb/")
    ap.add_argument(
        "overrides",
        nargs="*",
        help="Optional key=value overrides, e.g. device=0 batch=40 epochs=60",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    run_train(args.cfg, overrides=list(args.overrides))


if __name__ == "__main__":
    main()
