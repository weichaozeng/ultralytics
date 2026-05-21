"""inspect_model.py

Utility to print *structure only* (no weights) for an Ultralytics YOLO *.pt model.

This script helps verify:
- task type (detect/pose/etc.)
- model class and important attributes (kpt_shape, nc, stride)
- YAML cfg reference / model.yaml content if available
- a readable layer/module summary (safe, textual)
- output tensor shapes for a dummy forward pass (optional)
- YOLO sequential layer indices/channels for QNN/SSD insertion planning

Usage (zsh examples):
  python ultralytics/inspect_model.py --ckpt ultralytics/weights/detector.pt
  python ultralytics/inspect_model.py --ckpt weights/detector.pt --imgsz 640 --device cpu

Notes:
- This script avoids printing parameters/weights tensors.
- Dummy forward is best-effort; some exported backends may not support it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import ultralytics. Run this script from the repo root or ensure the ultralytics package is importable."
    ) from e


def _to_jsonable(x: Any):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, Path):
        return str(x)
    if torch.is_tensor(x):
        return {"tensor": True, "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
    return str(x)


def _safe_getattr(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _conv_summary(module: Any) -> list[str]:
    """Return compact Conv2d summaries found on common Ultralytics module attributes."""
    rows = []
    for attr in ("conv", "cv1", "cv2", "cv3", "cv4", "cv5"):
        sub = getattr(module, attr, None)
        conv = getattr(sub, "conv", None)
        if isinstance(conv, torch.nn.Conv2d):
            rows.append(
                f"{attr}.conv in={conv.in_channels} out={conv.out_channels} "
                f"k={tuple(conv.kernel_size)} s={tuple(conv.stride)}"
            )
        elif isinstance(sub, torch.nn.Conv2d):
            rows.append(
                f"{attr} in={sub.in_channels} out={sub.out_channels} "
                f"k={tuple(sub.kernel_size)} s={tuple(sub.stride)}"
            )
    return rows


def _infer_layer_out_channels(module: Any) -> int | None:
    """Best-effort output channel inference matching the QNNPoseModel helper."""
    for attr in ("cv2", "cv1", "conv"):
        sub = getattr(module, attr, None)
        conv = getattr(sub, "conv", None)
        if isinstance(conv, torch.nn.Conv2d):
            return int(conv.out_channels)
        if isinstance(sub, torch.nn.Conv2d):
            return int(sub.out_channels)
    if isinstance(getattr(module, "out_channels", None), int):
        return int(module.out_channels)
    return None


def _print_yolo_layers(m: Any):
    """Print the top-level Ultralytics Sequential layer list."""
    seq = getattr(m, "model", None)
    if not isinstance(seq, torch.nn.Sequential):
        print("No `model.model` Sequential found.")
        return

    print(f"Top-level layers: {len(seq)}")
    print(f"{'idx':>3s} {'from':>12s} {'class':<22s} {'out_ch':>8s} {'params':>12s}  details")
    for i, layer in enumerate(seq):
        n_params = sum(p.numel() for p in layer.parameters())
        out_ch = _infer_layer_out_channels(layer)
        convs = "; ".join(_conv_summary(layer))
        print(
            f"{i:3d} {str(getattr(layer, 'f', None)):>12s} {layer.__class__.__name__:<22s} "
            f"{str(out_ch):>8s} {n_params:12d}  {convs}"
        )

    print("\nSSD insertion candidates (layers with inferred output channels):")
    for i, layer in enumerate(seq):
        out_ch = _infer_layer_out_channels(layer)
        if out_ch is None:
            continue
        head_dim = max(out_ch // 4, 1)
        while out_ch % head_dim != 0 and head_dim > 1:
            head_dim -= 1
        print(f"  after layer {i:2d}: in_dim={out_ch:4d}, suggested_head_dim={head_dim:4d}")


def _print_state_dict_keys(m: Any, limit: int):
    """Print state dict keys and shapes to confirm checkpoint naming."""
    print(f"First {limit} state_dict entries:")
    for i, (k, v) in enumerate(m.state_dict().items()):
        if i >= limit:
            break
        shape = tuple(v.shape) if hasattr(v, "shape") else "-"
        print(f"  {i:04d} {k:60s} {shape}")


def main():
    ap = argparse.ArgumentParser(description="Inspect Ultralytics YOLO .pt (print structure only)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    ap.add_argument("--imgsz", type=int, default=None, help="Dummy forward image size (required with --dummy-forward)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | 0 | 0,1 ...")
    ap.add_argument("--half", action="store_true", help="Use fp16 for dummy forward (if supported)")
    ap.add_argument("--dummy-forward", action="store_true", help="Run a dummy forward pass to print output shapes")
    ap.add_argument("--state-limit", type=int, default=120, help="Number of state_dict entries to print")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("=" * 88)
    print("[1/5] Load YOLO model")
    print(f"ckpt = {ckpt_path}")

    yolo = YOLO(str(ckpt_path))

    # Ultralytics Model wrapper
    print("\n=" * 2)
    print("[2/5] High-level attributes")
    info = {
        "yolo_type": type(yolo).__name__,
        "task": _safe_getattr(yolo, "task"),
        "model_type": type(_safe_getattr(yolo, "model")).__name__,
        "ckpt_path": str(ckpt_path),
    }

    m = yolo.model
    # Common model fields
    info.update(
        {
            "nc": _safe_getattr(m, "nc"),
            "names_type": type(_safe_getattr(m, "names")).__name__,
            "kpt_shape": _safe_getattr(m, "kpt_shape"),
            "stride": _safe_getattr(m, "stride"),
            "yaml_type": type(_safe_getattr(m, "yaml")).__name__,
        }
    )

    # Try to include model yaml dict if present (structure only)
    yaml_dict = _safe_getattr(m, "yaml")
    if isinstance(yaml_dict, dict):
        # Keep it small: only meta keys that help debugging
        meta_keys = [
            "task",
            "nc",
            "kpt_shape",
            "ch",
            "backbone",
            "head",
            "scale",
            "depth_multiple",
            "width_multiple",
        ]
        info["yaml_meta"] = {k: _to_jsonable(yaml_dict.get(k)) for k in meta_keys if k in yaml_dict}

    print(json.dumps(_to_jsonable(info), indent=2, ensure_ascii=False))

    print("\n=" * 2)
    print("[3/5] Text summary of modules (repr)")
    # This prints a readable module tree; does not include raw weights.
    try:
        print(m)
    except Exception as e:
        print(f"<failed to print model repr: {e}>")

    print("\n=" * 2)
    print("[4/6] Layer list (name -> class, trainable params count)")
    trainable = 0
    total = 0
    for name, module in m.named_modules():
        # Skip the root repeated line
        if name == "":
            continue
        # Count parameters directly owned by this module (not children)
        own_params = list(module.parameters(recurse=False))
        if not own_params:
            continue
        n_total = sum(p.numel() for p in own_params)
        n_train = sum(p.numel() for p in own_params if p.requires_grad)
        total += n_total
        trainable += n_train
        print(f"{name:45s}  {type(module).__name__:25s}  params={n_total:10d}  trainable={n_train:10d}")

    print("\nTotals:")
    print(f"  total_params     = {total}")
    print(f"  trainable_params = {trainable}")

    print("\n=" * 2)
    print("[5/6] YOLO top-level layers and checkpoint keys")
    _print_yolo_layers(m)
    print()
    _print_state_dict_keys(m, args.state_limit)

    print("\n=" * 2)
    print("[6/6] Dummy forward (optional)")
    if not args.dummy_forward:
        print("Skipping dummy forward. Pass --dummy-forward to run it.")
        print("=" * 88)
        return

    if args.imgsz is None:
        print("Dummy forward requested but --imgsz was not provided.")
        print("Example: python ultralytics/inspect_model.py --ckpt weights/detector.pt --dummy-forward --imgsz 640")
        print("=" * 88)
        return

    if args.imgsz is None:
        print("Dummy forward requested but --imgsz was not provided.")
        print("Example: python ultralytics/inspect_model.py --ckpt weights/detector.pt --dummy-forward --imgsz 640")
        print("=" * 88)
        return

    device = args.device
    try:
        yolo.to(device)
    except Exception:
        # Some ultralytics wrappers may not implement .to in the wrapper; fall back to underlying model.
        try:
            m.to(device)
        except Exception as e:
            print(f"Failed to move model to device {device}: {e}")

    m.eval()

    # Determine input channels: default 3 if unknown
    ch = 3
    if isinstance(yaml_dict, dict) and isinstance(yaml_dict.get("ch"), int):
        ch = int(yaml_dict["ch"])

    try:
        param_device = next(m.parameters()).device
    except StopIteration:
        param_device = torch.device("cpu")

    x = torch.zeros(1, ch, args.imgsz, args.imgsz, device=param_device)
    if args.half:
        x = x.half()

    with torch.no_grad():
        try:
            out = m(x)
        except Exception as e:
            print(f"Dummy forward failed: {e}")
            print("Tip: if this is an exported backend or expects different input, skip dummy forward.")
            print("=" * 88)
            return

    def _shape(o: Any):
        if torch.is_tensor(o):
            return list(o.shape)
        if isinstance(o, (list, tuple)):
            return [_shape(v) for v in o]
        if isinstance(o, dict):
            return {k: _shape(v) for k, v in o.items()}
        return str(type(o))

    print("Output shape summary:")
    print(json.dumps(_to_jsonable(_shape(out)), indent=2, ensure_ascii=False))
    print("=" * 88)


if __name__ == "__main__":
    main()
