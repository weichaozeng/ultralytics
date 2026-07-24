#!/usr/bin/env python3
"""Fast HIRE did_reset spatial diagnostic on packed SPAD frames.npy."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from ultralytics.data.spad_packed import unpack_packed_frames
from ultralytics.quanta_hire_networks.integrator import HIRE

PATH = Path(
    "/Users/zvc/Code/Proj/SPADHand/Data/2khz/"
    "female_diner_17124_skin_f_asian_05_ALB_rp_corey_posed_026_texture_06/frames.npy"
)
SUBSAMPLING = 80  # 2 kHz: 80 bins per emit
N_EMIT = 6
T0 = 0  # start from beginning
DS = 1  # full resolution (no spatial downsample)
KEEP_FULL_KERNELS = True
TRACK_GATES = True


def main() -> None:
    t_wall = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}", flush=True)

    packed = np.load(PATH, mmap_mode="r")
    print(f"packed={packed.shape}", flush=True)
    t1 = T0 + N_EMIT * SUBSAMPLING
    chunk = np.asarray(packed[T0:t1])
    print(f"bins=[{T0}:{t1}] chunk={chunk.shape}", flush=True)

    hits = unpack_packed_frames(chunk, expected_w=512).any(axis=-1)  # T,H,W
    hits = hits[:, ::DS, ::DS]
    print(f"hits_ds={hits.shape} hit_rate={hits.mean():.4f}", flush=True)

    rates = np.stack(
        [hits[e * SUBSAMPLING : (e + 1) * SUBSAMPLING].mean(0) for e in range(N_EMIT)],
        0,
    )
    motion = rates.std(0)
    motion_mask = motion >= np.quantile(motion, 0.7)
    static_mask = ~motion_mask
    print(f"motion_frac={motion_mask.mean():.3f}", flush=True)

    cube = torch.from_numpy(np.ascontiguousarray(np.transpose(hits, (1, 2, 0)))).to(
        device=device, dtype=torch.float32
    )
    h, w, t_raw = cube.shape

    # Scale odd morph kernels with downsample (keep odd, >=1).
    def odd_scale(k: int) -> int:
        if KEEP_FULL_KERNELS:
            return k if k % 2 == 1 else k + 1
        v = max(1, int(round(k / DS)))
        return v if v % 2 == 1 else v + 1

    hire = HIRE(
        subsampling=SUBSAMPLING,
        sample_rate_hz=2000.0,
        fast_bins=24,
        slow_bins=160,
        surprise_bins=4,
        mix_hold_bins=80,
        mix_bins=12.0,
        mix_theta=0.06,
        theta_on=0.08,
        theta_off=0.02,
        confirm_bins=4,
        spatial_kernel=odd_scale(5),
        gate_pool="max",
        reset_open=odd_scale(15),
        reset_grow=6 if KEEP_FULL_KERNELS else max(1, 6 // DS),
        normalize=False,
    )
    print(
        f"kernels spatial={hire.spatial_kernel} open={hire.reset_open} grow={hire.reset_grow}",
        flush=True,
    )
    hire.clear_states()

    beta_f_floor = cube.new_tensor(hire.beta_fast_floor)
    beta_s_floor = cube.new_tensor(hire.beta_slow_floor)
    n_f_max_t = cube.new_tensor(float(hire.fast_bins))
    n_s_max_t = cube.new_tensor(float(hire.slow_bins))
    hold_h = float(hire.effective_mix_hold_bins())
    tau_mix = float(hire.mix_bins)
    mix_floor = float(hire.effective_mix_floor())
    theta_grow = float(hire.effective_theta_grow())
    t_mix_init = float(max(hold_h, 1) + 10.0 * tau_mix)
    ones_hw, zeros_hw = hire._ensure_scratch(cube[..., 0])

    i_fast = None
    reset_sum = torch.zeros(h, w, device=device)
    s_sum = torch.zeros(h, w, device=device)
    g_sum = torch.zeros(h, w, device=device)
    chg_sum = torch.zeros(h, w, device=device)
    n_acc = 0

    print("HIRE loop...", flush=True)
    with torch.inference_mode():
        for t in range(t_raw):
            xt = cube[..., t]
            if i_fast is None:
                i_fast = xt.clone()
                i_slow = xt.clone()
                n_fast = xt.new_ones(xt.shape)
                n_slow = xt.new_ones(xt.shape)
                s_tilde = xt.new_zeros(xt.shape)
                in_change = xt.new_zeros(xt.shape)
                confirm_count = xt.new_zeros(xt.shape)
                t_mix = xt.new_full(xt.shape, t_mix_init)
                continue
            (
                i_fast,
                i_slow,
                n_fast,
                n_slow,
                s_tilde,
                in_change,
                confirm_count,
                t_mix,
                _i_out,
                _s_raw,
                _w_slow,
                g_fast,
                did_reset,
            ) = hire._step(
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
                record_debug=True,
            )
            if t >= SUBSAMPLING:
                reset_sum += did_reset
                s_sum += s_tilde
                g_sum += g_fast
                chg_sum += in_change
                n_acc += 1
            if t % SUBSAMPLING == 0:
                print(f"  t={t}/{t_raw} ({time.time() - t_wall:.1f}s)", flush=True)

    reset_rate = (reset_sum / max(n_acc, 1)).detach().cpu().numpy()
    s_mean = (s_sum / max(n_acc, 1)).detach().cpu().numpy()
    g_mean = (g_sum / max(n_acc, 1)).detach().cpu().numpy()
    chg_mean = (chg_sum / max(n_acc, 1)).detach().cpu().numpy()

    print("\n=== rates by row band (top→bot) ===", flush=True)
    for i in range(8):
        a, b = h * i // 8, h * (i + 1) // 8
        print(
            f"band{i}[{a:3d}:{b:3d}] reset={reset_rate[a:b].mean():.5f} "
            f"in_chg={chg_mean[a:b].mean():.5f} g={g_mean[a:b].mean():.5f} "
            f"s={s_mean[a:b].mean():.5f} motion={motion_mask[a:b].mean():.3f}",
            flush=True,
        )

    def block(name: str, sl: slice) -> None:
        rr, mm, sm = reset_rate[sl], motion_mask[sl], static_mask[sl]
        print(
            f"{name}: reset all={rr.mean():.5f} |motion={rr[mm].mean():.5f} |static={rr[sm].mean():.5f} "
            f"g|static={g_mean[sl][sm].mean():.5f} chg|static={chg_mean[sl][sm].mean():.5f}",
            flush=True,
        )

    print("\n=== TOP/MID/BOT ===", flush=True)
    block("TOP1/4", slice(0, h // 4))
    block("MID   ", slice(h // 4, 3 * h // 4))
    block("BOT1/4", slice(3 * h // 4, h))
    print(
        f"global static reset={reset_rate[static_mask].mean():.5f} "
        f"motion reset={reset_rate[motion_mask].mean():.5f} "
        f"static g={g_mean[static_mask].mean():.5f} motion g={g_mean[motion_mask].mean():.5f}",
        flush=True,
    )
    row = reset_rate.mean(1)
    topk = np.argsort(-row)[:15]
    print("highest reset rows", topk.tolist(), np.round(row[topk], 5).tolist(), flush=True)
    grow = g_mean.mean(1)
    gtopk = np.argsort(-grow)[:15]
    print("highest g_fast rows", gtopk.tolist(), np.round(grow[gtopk], 5).tolist(), flush=True)

    def to_u8(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)
        return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    panels = [to_u8(rates.mean(0)), to_u8(motion), to_u8(g_mean), to_u8(reset_rate)]
    for p in panels:
        p[h // 4 - 1 : h // 4 + 1, :] = 255
    strip = np.concatenate(panels, axis=1)
    out = PATH.parent / "hire_reset_diag.png"
    if cv2 is not None:
        cv2.imwrite(str(out), strip)
    else:
        np.save(out.with_suffix(".npy"), strip)
        out = out.with_suffix(".npy")
    np.savez_compressed(
        PATH.parent / "hire_reset_diag_stats.npz",
        reset_rate=reset_rate,
        s_mean=s_mean,
        g_mean=g_mean,
        chg_mean=chg_mean,
        motion=motion,
        rates_mean=rates.mean(0),
        motion_mask=motion_mask,
    )
    print(f"saved {out} total={time.time() - t_wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
