#!/usr/bin/env python3
"""Lightweight CPU verification for STEA/HYB streaming integrators."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

# Minimal stubs so integrator deps import without opencv/scipy/numba in CI.
if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.INPAINT_TELEA = 0
    cv2_stub.inpaint = lambda *a, **k: a[0]
    sys.modules["cv2"] = cv2_stub

_PKG = Path(__file__).resolve().parent / "ultralytics"
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load(name: str, rel: str):
    path = _PKG / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("ultralytics.quanta_neural_networks.ops.array_ops", "quanta_neural_networks/ops/array_ops.py")
_load("ultralytics.quanta_neural_networks.ops.image", "quanta_neural_networks/ops/image.py")
stea_mod = _load("ultralytics.quanta_stea_networks.integrator", "quanta_stea_networks/integrator.py")
_load("ultralytics.data.spad_packed", "data/spad_packed.py")
hyb_mod = _load("ultralytics.quanta_hyb_networks.integrator", "quanta_hyb_networks/integrator.py")

import torch

SpatioTemporalEvidenceAccumulation = stea_mod.SpatioTemporalEvidenceAccumulation
HybridSpatioTemporalEvidenceAccumulation = hyb_mod.HybridSpatioTemporalEvidenceAccumulation


def _random_cube(h: int, w: int, t: int, *, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand((h, w, t), generator=gen) > 0.92


def _fused_last_from_debug(integrator, cube: torch.Tensor):
    integrator._clear_histories()
    fused, debug = integrator._integrate_last_with_debug(cube)
    return debug["fused_last"], fused


def _fused_last_from_streaming(integrator, cube: torch.Tensor):
    integrator._clear_histories()
    fused = integrator._integrate_last(cube)
    return fused[..., -1] if int(fused.shape[-1]) > 0 else fused.squeeze(-1)


def _assert_close(a: torch.Tensor, b: torch.Tensor, *, label: str) -> None:
    diff = (a - b).abs().max().item()
    if diff >= 1e-5:
        raise AssertionError(f"{label}: max diff {diff:.3e} >= 1e-5")
    print(f"  ok {label} (max diff {diff:.2e})")


def main() -> None:
    print("STEA streaming vs debug")
    for h, w, t in [(32, 32, 64), (16, 24, 37), (8, 8, 1)]:
        cube = _random_cube(h, w, t, seed=h + w + t)
        stea = SpatioTemporalEvidenceAccumulation(
            fast_window=8,
            slow_window=16,
            temporal_window=5,
            motion_sharpness=40.0,
            motion_threshold=0.05,
            stable_prior=8.0,
        )
        debug_last, _ = _fused_last_from_debug(stea, cube)
        stream_last = _fused_last_from_streaming(stea, cube)
        _assert_close(debug_last, stream_last, label=f"stea hwt=({h},{w},{t})")

    print("STEA history continuity (streaming vs debug after partial chunk)")
    cube = _random_cube(24, 24, 100, seed=7)
    mid = 50
    partial = cube[..., :mid]
    stream = SpatioTemporalEvidenceAccumulation(fast_window=6, slow_window=12)
    stream._clear_histories()
    stream._integrate_last(partial)
    debug = SpatioTemporalEvidenceAccumulation(fast_window=6, slow_window=12)
    debug._clear_histories()
    debug._integrate_last_with_debug(partial)
    _assert_close(stream.photon_history, debug.photon_history, label="photon_history after partial chunk")
    _assert_close(stream.kl_history, debug.kl_history, label="kl_history after partial chunk")

    print("HYB no velocity")
    kwargs = dict(fast_window=8, slow_window=16, temporal_window=5)
    cube = _random_cube(20, 20, 48, seed=11)
    stea = SpatioTemporalEvidenceAccumulation(**kwargs)
    hyb = HybridSpatioTemporalEvidenceAccumulation(**kwargs)
    hyb.set_velocity_field(None)
    _assert_close(
        _fused_last_from_streaming(stea, cube),
        _fused_last_from_streaming(hyb, cube),
        label="hyb no-vel vs stea",
    )
    debug_last, _ = _fused_last_from_debug(hyb, cube)
    _assert_close(debug_last, _fused_last_from_streaming(hyb, cube), label="hyb no-vel debug")

    print("HYB with velocity")
    h, w, t = 32, 32, 40
    cube = _random_cube(h, w, t, seed=17)
    hyb = HybridSpatioTemporalEvidenceAccumulation(
        fast_window=8,
        slow_window=16,
        temporal_window=5,
        warp_block_size=8,
        chunk_size=32,
    )
    gen = torch.Generator().manual_seed(19)
    flow = torch.randn((h // 2, w // 2, 2), generator=gen) * 0.5
    hyb.set_velocity_field(flow, source_space="rgb")
    debug_last, _ = _fused_last_from_debug(hyb, cube)
    _assert_close(debug_last, _fused_last_from_streaming(hyb, cube), label="hyb with velocity")

    print("All checks passed.")


if __name__ == "__main__":
    main()
