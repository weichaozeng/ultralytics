# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""inspect_binary_npy.py

Load ``binary.npy`` and print its array structure (shape, dtype, inferred layout, stats).

Accepts:
- a directory containing ``binary.npy``
- a direct path to ``binary.npy`` (or any ``.npy`` file)

Example
-------
python ultralytics/inspect_binary_npy.py --in_path /path/to/sample_dir
python ultralytics/inspect_binary_npy.py --in_path /path/to/binary.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PACKED_WIDTHS = (64, 128)


def _is_packed_spad(shape: tuple[int, ...]) -> bool:
    return len(shape) == 4 and shape[-1] in (3, 4) and shape[2] in PACKED_WIDTHS


def _infer_packed_nch(shape: tuple[int, ...]) -> int:
    return int(shape[-1]) if _is_packed_spad(shape) else 4


def _resolve_npy(in_path: Path) -> Path:
    if in_path.is_dir():
        npy = in_path / "binary.npy"
        if not npy.exists():
            raise FileNotFoundError(f"Directory input requires binary.npy, not found: {npy}")
        return npy
    if in_path.suffix.lower() != ".npy":
        raise ValueError(f"Unsupported input: {in_path} (expect directory or .npy)")
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")
    return in_path


def _looks_like_hwt(shape: tuple[int, ...]) -> bool:
    if len(shape) != 3:
        return False
    h, w, t = shape
    return h == w and t != h


def _infer_layout(shape: tuple[int, ...]) -> str:
    if len(shape) == 4 and shape[-1] in (3, 4) and shape[2] in (64, 128):
        return f"packed (T,H,Wpacked,C) C={shape[-1]}"
    if len(shape) == 4 and shape[-1] == 1:
        return "thwc1 (T,H,W,1)"
    if len(shape) == 3:
        if _looks_like_hwt(shape):
            return "hwt (H,W,T)"
        return "thw (T,H,W)"
    if len(shape) == 4:
        return f"batch (N,{shape[1]},{shape[2]},{shape[3]})"
    if len(shape) == 2:
        return "hw (H,W)"
    if len(shape) == 1:
        return "vector (N,)"
    return "unknown"


def _axis_labels(layout: str, shape: tuple[int, ...]) -> dict[str, int]:
    if layout.startswith("packed"):
        return {"T": shape[0], "H": shape[1], "Wpacked": shape[2], "C": shape[3]}
    if layout.startswith("thwc1"):
        return {"T": shape[0], "H": shape[1], "W": shape[2], "C": shape[3]}
    if layout.startswith("hwt"):
        return {"H": shape[0], "W": shape[1], "T": shape[2]}
    if layout.startswith("thw"):
        return {"T": shape[0], "H": shape[1], "W": shape[2]}
    if layout.startswith("hw"):
        return {"H": shape[0], "W": shape[1]}
    return {f"dim{i}": v for i, v in enumerate(shape)}


def _sample_stats(arr: np.ndarray) -> dict:
    flat = arr.reshape(-1)
    stats: dict = {
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "mean": float(flat.mean()) if flat.size else None,
    }
    if arr.dtype == bool or np.issubdtype(arr.dtype, np.bool_):
        stats["occupancy"] = stats["mean"]
    uniq = np.unique(flat[: min(flat.size, 1_000_000)])
    if uniq.size <= 16:
        stats["unique_values"] = uniq.tolist()
    else:
        stats["num_unique_sampled"] = int(uniq.size)
    return stats


def inspect_binary_npy(in_path: Path, *, mmap: bool = True) -> dict:
    npy = _resolve_npy(in_path)
    arr = np.load(npy, mmap_mode="r" if mmap else None, allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected ndarray in {npy}, got {type(arr)}")

    layout = _infer_layout(arr.shape)
    info = {
        "path": str(npy.resolve()),
        "file_size_bytes": npy.stat().st_size,
        "ndim": int(arr.ndim),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "layout": layout,
        "axes": _axis_labels(layout, arr.shape),
        "itemsize_bytes": int(arr.dtype.itemsize),
        "nbytes": int(arr.nbytes),
        "stats": _sample_stats(np.asarray(arr)),
    }
    if _is_packed_spad(arr.shape):
        info["packed_nch"] = _infer_packed_nch(arr.shape)
    return info


def _print_report(info: dict) -> None:
    print(f"path:      {info['path']}")
    print(f"file_size: {info['file_size_bytes']:,} bytes")
    print(f"ndim:      {info['ndim']}")
    print(f"shape:     {tuple(info['shape'])}")
    print(f"dtype:     {info['dtype']}")
    print(f"layout:    {info['layout']}")
    print(f"axes:      {info['axes']}")
    if "packed_nch" in info:
        print(f"packed_nch: {info['packed_nch']}")
    print(f"nbytes:    {info['nbytes']:,}")
    stats = info["stats"]
    print(
        f"stats:     min={stats['min']} max={stats['max']} mean={stats['mean']:.6f}"
        + (f" occupancy={stats['occupancy']:.6f}" if "occupancy" in stats else "")
    )
    if "unique_values" in stats:
        print(f"unique:    {stats['unique_values']}")
    elif "num_unique_sampled" in stats:
        print(f"unique:    >= {stats['num_unique_sampled']} distinct values (sampled)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect binary.npy array structure")
    parser.add_argument("--in_path", type=Path, required=True, help="Directory containing binary.npy or direct .npy path")
    parser.add_argument("--json", action="store_true", help="Print JSON report to stdout")
    parser.add_argument("--no_mmap", action="store_true", help="Load array into memory instead of mmap")
    args = parser.parse_args()

    info = inspect_binary_npy(args.in_path, mmap=not args.no_mmap)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        _print_report(info)


if __name__ == "__main__":
    main()
