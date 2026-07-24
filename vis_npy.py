#!/usr/bin/env python3
"""Quick CLI visualizer for SPAD/HIRE/PPB ``.npy`` dumps.

Supports common layouts from ``vis_pre`` / render caches:

- ``(H, W)``            — e.g. ``*_hire_n_slow.npy``, ``*_hire_s_raw.npy``, ``*_ppb_run_length.npy``
- ``(T, H, W)``         — temporal stack of maps / gray frames
- ``(T, C, H, W)``      — RGB/float frames (C=1 or 3)
- ``(H, W, C)``         — HWC image

Examples
--------
# HIRE n_slow map from vis_pre
python ultralytics/vis_npy.py \\
  /path/to/cube00000_..._hire_n_slow.npy \\
  --save /tmp/n_slow.png

# RGB stack, show frame 40
python ultralytics/vis_npy.py /path/to/frames.npy --index 40 --rgb

# Temporal map stack
python ultralytics/vis_npy.py /path/to/maps.npy --index 10 --cmap magma --vmin 0 --vmax 160
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Quick .npy image / map visualizer")
    ap.add_argument("path", type=Path, help="Path to .npy file")
    ap.add_argument(
        "--index",
        "-i",
        type=int,
        default=0,
        help="Temporal index for (T,...) arrays (default: 0)",
    )
    ap.add_argument(
        "--rgb",
        action="store_true",
        help="Force RGB display for (C,H,W)/(T,C,H,W) with C in {1,3}",
    )
    ap.add_argument("--cmap", type=str, default="turbo", help="Colormap for scalar maps")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument(
        "--percentile",
        type=float,
        default=0.0,
        help="If >0, set vmax to this percentile of the selected slice (e.g. 99)",
    )
    ap.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title; omit for heatmap + colorbar only",
    )
    ap.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save figure to this path instead of interactive show",
    )
    ap.add_argument("--figsize", type=float, nargs=2, default=(6.0, 6.0))
    ap.add_argument("--dpi", type=int, default=140)
    return ap.parse_args()


def _as_display_image(arr: np.ndarray, *, index: int, force_rgb: bool) -> tuple[np.ndarray, str]:
    """Return ``(img, kind)`` where kind is ``rgb`` or ``scalar``."""
    x = np.asarray(arr)
    if x.ndim == 2:
        return x.astype(np.float32, copy=False), "scalar"

    if x.ndim == 3:
        # (H,W,C)
        if x.shape[-1] in (1, 3, 4) and x.shape[0] != x.shape[-1]:
            img = x[..., :3].astype(np.float32, copy=False)
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            return img, "rgb"
        # (C,H,W)
        if x.shape[0] in (1, 3, 4) and (force_rgb or x.shape[0] <= 4):
            img = np.transpose(x[:3], (1, 2, 0)).astype(np.float32, copy=False)
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            return img, "rgb"
        # (T,H,W)
        t = int(np.clip(index, 0, x.shape[0] - 1))
        return x[t].astype(np.float32, copy=False), "scalar"

    if x.ndim == 4:
        # (T,C,H,W)
        t = int(np.clip(index, 0, x.shape[0] - 1))
        frame = x[t]
        if frame.shape[0] in (1, 3, 4) or force_rgb:
            img = np.transpose(frame[:3], (1, 2, 0)).astype(np.float32, copy=False)
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            return img, "rgb"
        raise ValueError(f"Unsupported 4D layout {x.shape}; expected (T,C,H,W) with C in {{1,3,4}}")

    raise ValueError(f"Unsupported array ndim={x.ndim}, shape={x.shape}")


def main() -> None:
    args = _parse_args()
    path = args.path
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, mmap_mode="r")
    print(f"loaded {path}")
    print(f"shape={data.shape} dtype={data.dtype}")

    img, kind = _as_display_image(data, index=int(args.index), force_rgb=bool(args.rgb))

    import matplotlib

    if args.save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    if kind == "rgb":
        # float RGB in [0,1] or uint-like
        if np.issubdtype(img.dtype, np.floating):
            show = np.clip(img, 0.0, 1.0)
            if float(np.nanmax(show)) > 1.5:
                show = np.clip(show / 255.0, 0.0, 1.0)
        else:
            show = img
        ax.imshow(show)
    else:
        vmin = args.vmin
        vmax = args.vmax
        if vmax is None and float(args.percentile) > 0:
            vmax = float(np.nanpercentile(img, float(args.percentile)))
        im = ax.imshow(img, cmap=args.cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if args.title:
        ax.set_title(args.title)
    ax.axis("off")
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=int(args.dpi), bbox_inches="tight")
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
