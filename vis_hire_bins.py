#!/usr/bin/env python3
"""Export per-bin HIRE maps (i_out, i_fast, i_slow, s_raw, n_slow) from ``frames.npy``.

Runs HIRE causally with ``subsampling=1`` (one update per SPAD bin) and writes
``--num`` bins starting at ``--start`` (optional ``--stride``).

Layout
------
``{save_dir}/out/bin_XXXXXXX.npy``
``{save_dir}/i_fast/bin_XXXXXXX.npy``
``{save_dir}/i_slow/bin_XXXXXXX.npy``
``{save_dir}/s_raw/bin_XXXXXXX.npy``
``{save_dir}/n_slow/bin_XXXXXXX.npy``

Each file is ``(H, W)`` float32 (Bayer photon-plane resolution).

Examples
--------
python ultralytics/vis_hire_bins.py \\
  --in_path /path/to/frames.npy \\
  --save_dir /tmp/hire_bins \\
  --num 64

# Cold start at bin 1000 (no warmup 0..999)
python ultralytics/vis_hire_bins.py \\
  --in_path /path/to/frames.npy \\
  --save_dir /tmp/hire_bins \\
  --start 1000 --num 32 --stride 2 --no_warmup
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

from ultralytics.data.spad_packed import (
    is_packed_spad,
    packed_frames_to_raw_video,
    raw_plane_to_photon_cube,
)
from ultralytics.quanta_hire_networks.integrator import HIRE


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Save per-bin HIRE out / i_fast / i_slow / s_raw / n_slow maps")
    ap.add_argument("--in_path", type=Path, required=True, help="frames.npy or dir containing it")
    ap.add_argument("--save_dir", type=Path, required=True)
    ap.add_argument("--num", type=int, required=True, help="Number of bins to save")
    ap.add_argument("--start", type=int, default=0, help="First bin index to save")
    ap.add_argument("--stride", type=int, default=1, help="Step between saved bins")
    ap.add_argument(
        "--no_warmup",
        action="store_true",
        help="Do not process bins before --start (cold start at --start)",
    )
    ap.add_argument("--expected_w", type=int, default=0, help="Crop unpacked width; 0=full")
    ap.add_argument("--packed_ch_order", type=str, default="RGB", choices=["RGB", "BGR"])
    ap.add_argument("--flip_x", action="store_true")
    ap.add_argument("--flip_y", action="store_true")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument(
        "--png",
        action="store_true",
        help="Also write PNG previews (out/i_fast/i_slow: gray; s_raw/n_slow: turbo)",
    )
    ap.add_argument("--prefix", type=str, default="bin")

    # HIRE defaults = sequence_hire_attn_wst_8kHz / vis_pre
    ap.add_argument("--bin_rate_hz", type=float, default=8000.0)
    ap.add_argument("--hire_fast_bins", type=int, default=24)
    ap.add_argument("--hire_slow_bins", type=int, default=160)
    ap.add_argument("--hire_surprise_bins", type=int, default=4)
    ap.add_argument("--hire_mix_hold_bins", type=int, default=80)
    ap.add_argument("--hire_mix_bins", type=float, default=12.0)
    ap.add_argument("--hire_mix_theta", type=float, default=0.06)
    ap.add_argument("--hire_mix_floor", type=float, default=-1.0)
    ap.add_argument("--hire_theta_on", type=float, default=0.08)
    ap.add_argument("--hire_theta_off", type=float, default=0.02)
    ap.add_argument("--hire_theta_grow", type=float, default=-1.0)
    ap.add_argument("--hire_confirm_bins", type=int, default=4)
    ap.add_argument("--hire_cooldown_bins", type=int, default=0)
    ap.add_argument("--hire_spatial_kernel", type=int, default=5)
    ap.add_argument("--hire_gate_pool", type=str, default="max", choices=["max", "avg"])
    ap.add_argument("--hire_reset_open", type=int, default=15)
    ap.add_argument("--hire_reset_grow", type=int, default=6)
    ap.add_argument("--hire_eps", type=float, default=1e-5)
    ap.add_argument("--hire_normalize", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--hire_quantile", type=float, default=1.0)
    return ap.parse_args()


def _resolve_npy(in_path: Path) -> Path:
    if in_path.is_dir():
        for name in ("frames.npy", "binary.npy"):
            cand = in_path / name
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No frames.npy/binary.npy under {in_path}")
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if in_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected .npy, got {in_path}")
    return in_path


def _resolve_device(device: str) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_raw_plane(
    path: Path,
    *,
    t0: int,
    t1: int,
    expected_w: int,
    ch_order: str,
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    """Return Bayer plane ``(T, H, W)`` for bins [t0, t1)."""
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if t0 < 0 or t1 > int(arr.shape[0]) or t0 >= t1:
        raise IndexError(f"Invalid range [{t0}:{t1}) for T={arr.shape[0]}")
    slab = np.asarray(arr[t0:t1])

    if is_packed_spad(slab) or (
        slab.ndim == 4 and slab.shape[-1] in (3, 4) and np.issubdtype(slab.dtype, np.integer)
    ):
        raw = packed_frames_to_raw_video(
            slab,
            expected_w=expected_w if expected_w > 0 else None,
            ch_order=ch_order,
        )
        plane = np.ascontiguousarray(raw[..., 0])
    elif slab.ndim == 3:
        plane = np.ascontiguousarray(slab)
    elif slab.ndim == 4 and slab.shape[-1] == 1:
        plane = np.ascontiguousarray(slab[..., 0])
    else:
        raise ValueError(f"Unsupported frames.npy shape {slab.shape} dtype={slab.dtype}")

    if flip_x:
        plane = np.flip(plane, axis=2)
    if flip_y:
        plane = np.flip(plane, axis=1)
    return np.ascontiguousarray(plane)


def _build_hire(args: argparse.Namespace, device: torch.device) -> HIRE:
    return HIRE(
        subsampling=1,
        sample_rate_hz=float(args.bin_rate_hz),
        fast_bins=int(args.hire_fast_bins),
        slow_bins=int(args.hire_slow_bins),
        surprise_bins=int(args.hire_surprise_bins),
        mix_hold_bins=int(args.hire_mix_hold_bins),
        mix_bins=float(args.hire_mix_bins),
        mix_theta=float(args.hire_mix_theta),
        mix_floor=float(args.hire_mix_floor),
        theta_on=float(args.hire_theta_on),
        theta_off=float(args.hire_theta_off),
        theta_grow=float(args.hire_theta_grow),
        confirm_bins=int(args.hire_confirm_bins),
        cooldown_bins=int(args.hire_cooldown_bins),
        spatial_kernel=int(args.hire_spatial_kernel),
        gate_pool=str(args.hire_gate_pool),
        reset_open=int(args.hire_reset_open),
        reset_grow=int(args.hire_reset_grow),
        eps=float(args.hire_eps),
        normalize=bool(args.hire_normalize),
        quantile=float(args.hire_quantile),
    ).to(device)


def _to_npy(x: Tensor) -> np.ndarray:
    return np.ascontiguousarray(x.detach().float().cpu().numpy())


def _save_map(
    out_dir: Path,
    stem: str,
    arr: np.ndarray,
    *,
    write_png: bool,
    kind: str,
    n_slow_cap: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}.npy", arr.astype(np.float32, copy=False))
    if not write_png:
        return
    if kind in {"out", "i_fast", "i_slow"}:
        vis = np.clip(arr, 0.0, 1.0)
        peak = float(np.nanmax(vis)) if vis.size else 0.0
        if peak > 1.5:
            vis = np.clip(arr / max(float(np.nanpercentile(arr, 99.5)), 1e-6), 0.0, 1.0)
        u8 = (vis * 255.0).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{stem}.png"), u8)
    elif kind == "n_slow":
        vis = np.clip(arr / max(n_slow_cap, 1e-6), 0.0, 1.0)
        u8 = (vis * 255.0).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{stem}.png"), cv2.applyColorMap(u8, cv2.COLORMAP_TURBO))
    else:
        vmax = float(np.nanpercentile(arr, 99.5)) if arr.size else 1.0
        vis = np.clip(arr / max(vmax, 1e-6), 0.0, 1.0)
        u8 = (vis * 255.0).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{stem}.png"), cv2.applyColorMap(u8, cv2.COLORMAP_TURBO))


@torch.inference_mode()
def _run_bins(
    hire: HIRE,
    plane_thw: np.ndarray,
    *,
    device: torch.device,
    abs_t0: int,
    save_abs: set[int],
    save_dir: Path,
    prefix: str,
    write_png: bool,
) -> int:
    cube = raw_plane_to_photon_cube(plane_thw, device=device, as_bool=True)
    raw = cube.float()
    t_raw = int(raw.shape[-1])

    i_fast = hire.i_fast
    i_slow = hire.i_slow
    n_fast = hire.n_fast
    n_slow = hire.n_slow
    s_tilde = hire.s_tilde
    in_change = hire.in_change
    confirm_count = hire.confirm_count
    t_mix = hire.t_mix

    beta_f_floor = raw.new_tensor(hire.beta_fast_floor)
    beta_s_floor = raw.new_tensor(hire.beta_slow_floor)
    n_f_max_t = raw.new_tensor(float(hire.fast_bins))
    n_s_max_t = raw.new_tensor(float(hire.slow_bins))
    hold_h = float(hire.effective_mix_hold_bins())
    tau_mix = float(hire.mix_bins)
    mix_floor = float(hire.effective_mix_floor())
    theta_grow = float(hire.effective_theta_grow())
    t_mix_init = float(max(hold_h, 1) + 10.0 * tau_mix)
    ones_hw, zeros_hw = hire._ensure_scratch(raw[..., 0])
    step_fn = hire._get_infer_step()

    dirs = {
        "out": save_dir / "out",
        "i_fast": save_dir / "i_fast",
        "i_slow": save_dir / "i_slow",
        "s_raw": save_dir / "s_raw",
        "n_slow": save_dir / "n_slow",
    }
    n_saved = 0
    i_out: Tensor | None = None
    s_raw: Tensor | None = None

    for t in range(t_raw):
        abs_t = abs_t0 + t
        xt = raw[..., t]
        if i_fast is None or i_slow is None or s_tilde is None:
            i_fast = xt.clone()
            i_slow = xt.clone()
            n_fast = xt.new_ones(xt.shape)
            n_slow = xt.new_ones(xt.shape)
            s_tilde = xt.new_zeros(xt.shape)
            in_change = xt.new_zeros(xt.shape)
            confirm_count = xt.new_zeros(xt.shape)
            t_mix = xt.new_full(xt.shape, t_mix_init)
            i_out = i_slow
            s_raw = xt.new_zeros(xt.shape)
        else:
            (
                i_fast,
                i_slow,
                n_fast,
                n_slow,
                s_tilde,
                in_change,
                confirm_count,
                t_mix,
                i_out,
                s_raw,
            ) = step_fn(
                xt,
                i_fast,
                i_slow,
                n_fast,
                n_slow,
                s_tilde,
                in_change,
                confirm_count,
                t_mix,
                ones_hw=ones_hw,
                zeros_hw=zeros_hw,
                beta_f_floor=beta_f_floor,
                beta_s_floor=beta_s_floor,
                n_f_max_t=n_f_max_t,
                n_s_max_t=n_s_max_t,
                hold_h=hold_h,
                tau_mix=tau_mix,
                mix_floor=mix_floor,
                theta_grow=theta_grow,
                record_debug=False,
            )

        if abs_t in save_abs:
            assert (
                i_out is not None
                and i_fast is not None
                and i_slow is not None
                and s_raw is not None
                and n_slow is not None
            )
            stem = f"{prefix}_{abs_t:07d}"
            cap = float(hire.slow_bins)
            _save_map(dirs["out"], stem, _to_npy(i_out), write_png=write_png, kind="out", n_slow_cap=cap)
            _save_map(dirs["i_fast"], stem, _to_npy(i_fast), write_png=write_png, kind="i_fast", n_slow_cap=cap)
            _save_map(dirs["i_slow"], stem, _to_npy(i_slow), write_png=write_png, kind="i_slow", n_slow_cap=cap)
            _save_map(dirs["s_raw"], stem, _to_npy(s_raw), write_png=write_png, kind="s_raw", n_slow_cap=cap)
            _save_map(dirs["n_slow"], stem, _to_npy(n_slow), write_png=write_png, kind="n_slow", n_slow_cap=cap)
            n_saved += 1

    hire._store_states(
        i_fast=i_fast,
        i_slow=i_slow,
        n_fast=n_fast,
        n_slow=n_slow,
        s_tilde=s_tilde,
        in_change=in_change,
        confirm_count=confirm_count,
        t_mix=t_mix,
        s_raw=s_raw,
    )
    return n_saved


def main() -> None:
    args = _parse_args()
    if int(args.num) <= 0:
        raise ValueError("--num must be > 0")
    if int(args.stride) <= 0:
        raise ValueError("--stride must be > 0")
    if int(args.start) < 0:
        raise ValueError("--start must be >= 0")

    path = _resolve_npy(args.in_path)
    packed = np.load(path, mmap_mode="r", allow_pickle=False)
    t_total = int(packed.shape[0])
    print(f"loaded {path}")
    print(f"shape={packed.shape} dtype={packed.dtype}")

    save_indices = [int(args.start) + i * int(args.stride) for i in range(int(args.num))]
    if save_indices[-1] >= t_total:
        raise IndexError(
            f"Requested last index {save_indices[-1]} >= T={t_total} "
            f"(start={args.start}, num={args.num}, stride={args.stride})"
        )
    save_abs = set(save_indices)

    process_t0 = 0 if not args.no_warmup else int(args.start)
    process_t1 = save_indices[-1] + 1
    print(
        f"process bins [{process_t0}:{process_t1}) | save {len(save_indices)} bins "
        f"{save_indices[0]}…{save_indices[-1]} stride={args.stride}"
    )

    device = _resolve_device(args.device)
    hire = _build_hire(args, device)
    hire.clear_states()
    print(f"device={device} | {hire}", flush=True)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    chunk = 256
    n_saved = 0
    t = process_t0
    while t < process_t1:
        t1 = min(t + chunk, process_t1)
        plane = _load_raw_plane(
            path,
            t0=t,
            t1=t1,
            expected_w=int(args.expected_w),
            ch_order=str(args.packed_ch_order),
            flip_x=bool(args.flip_x),
            flip_y=bool(args.flip_y),
        )
        n_saved += _run_bins(
            hire,
            plane,
            device=device,
            abs_t0=t,
            save_abs=save_abs,
            save_dir=args.save_dir,
            prefix=str(args.prefix),
            write_png=bool(args.png),
        )
        t = t1
        print(f"  … processed through bin {t1 - 1} (saved {n_saved}/{len(save_indices)})", flush=True)

    print(f"Done. Wrote {n_saved} × {{out,i_fast,i_slow,s_raw,n_slow}} → {args.save_dir}")


if __name__ == "__main__":
    main()
