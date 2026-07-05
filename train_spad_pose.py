from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.models.yolo.pose.train import SpadPoseFrameTrainer, SpadPoseSequenceTrainer
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


def _select_trainer(mode: str):
    mode = str(mode).strip().lower()
    if mode == "sequence":
        return SpadPoseSequenceTrainer
    if mode == "frame":
        return SpadPoseFrameTrainer
    raise ValueError(f"Unsupported spad_train_mode={mode!r}; expected 'sequence' or 'frame'.")


def build_train_kwargs(cfg: dict[str, Any], *, forced_mode: str | None = None) -> tuple[type, dict[str, Any]]:
    cfg = dict(cfg)
    mode = forced_mode or cfg.pop("spad_train_mode", None) or "sequence"
    trainer = _select_trainer(mode)
    model = cfg.pop("model", "ultralytics/cfg/models/11/yolo11-pose.yaml")
    train_kwargs = {"model": model, **cfg}
    return trainer, train_kwargs


def run_train(cfg_path: str | Path, *, forced_mode: str | None = None, overrides: list[str] | None = None):
    cfg = _load_cfg(cfg_path)
    if overrides:
        cfg = _apply_kv_overrides(cfg, overrides)
    trainer, train_kwargs = build_train_kwargs(cfg, forced_mode=forced_mode)
    model_path = train_kwargs.pop("model")
    yolo = YOLO(model_path)
    return yolo.train(trainer=trainer, **train_kwargs)


def parse_args():
    ap = argparse.ArgumentParser(description="Train SPAD pose models from YAML configs.")
    ap.add_argument("--cfg", type=str, required=True, help="Path to YAML config under train_cfg/")
    ap.add_argument(
        "overrides",
        nargs="*",
        help="Optional key=value overrides, e.g. device=0 batch=2 spad_chunk_size=320",
    )
    return ap.parse_args()


def main(*, forced_mode: str | None = None):
    args = parse_args()
    run_train(args.cfg, forced_mode=forced_mode, overrides=list(args.overrides))


if __name__ == "__main__":
    main()
