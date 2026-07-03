"""Numerical regression tests for STEA/HYB streaming integrators."""

from __future__ import annotations

import pytest
import torch

from ultralytics.quanta_hyb_networks.integrator import HybridSpatioTemporalEvidenceAccumulation
from ultralytics.quanta_stea_networks.integrator import SpatioTemporalEvidenceAccumulation


def _random_cube(h: int, w: int, t: int, *, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand((h, w, t), generator=gen) > 0.92


def _fused_last_from_debug(integrator, cube: torch.Tensor) -> torch.Tensor:
    integrator._clear_histories()
    fused, debug = integrator._integrate_last_with_debug(cube)
    return debug["fused_last"], fused


def _fused_last_from_streaming(integrator, cube: torch.Tensor) -> torch.Tensor:
    integrator._clear_histories()
    fused = integrator._integrate_last(cube)
    return fused[..., -1] if int(fused.shape[-1]) > 0 else fused.squeeze(-1)


@pytest.mark.parametrize("h,w,t", [(32, 32, 64), (16, 24, 37), (8, 8, 1)])
def test_stea_streaming_matches_debug(h: int, w: int, t: int) -> None:
    cube = _random_cube(h, w, t, seed=h + w + t)
    stea = SpatioTemporalEvidenceAccumulation(
        fast_window=8,
        slow_window=16,
        temporal_window=5,
        motion_sharpness=40.0,
        motion_threshold=0.05,
        stable_prior=8.0,
    )
    debug_last, debug_fused = _fused_last_from_debug(stea, cube)
    stream_last = _fused_last_from_streaming(stea, cube)
    assert torch.allclose(debug_last, stream_last, atol=1e-5, rtol=1e-5)
    assert debug_fused.shape == stream_last.unsqueeze(-1).shape


def test_stea_history_continuity_matches_debug() -> None:
    cube = _random_cube(24, 24, 100, seed=7)
    partial = cube[..., :50]
    stream = SpatioTemporalEvidenceAccumulation(fast_window=6, slow_window=12)
    stream._clear_histories()
    stream._integrate_last(partial)
    debug = SpatioTemporalEvidenceAccumulation(fast_window=6, slow_window=12)
    debug._clear_histories()
    debug._integrate_last_with_debug(partial)
    assert torch.allclose(stream.photon_history, debug.photon_history, atol=1e-5, rtol=1e-5)
    assert torch.allclose(stream.kl_history, debug.kl_history, atol=1e-5, rtol=1e-5)


def test_hyb_no_velocity_matches_stea_streaming() -> None:
    cube = _random_cube(20, 20, 48, seed=11)
    kwargs = dict(fast_window=8, slow_window=16, temporal_window=5)
    stea = SpatioTemporalEvidenceAccumulation(**kwargs)
    hyb = HybridSpatioTemporalEvidenceAccumulation(**kwargs)
    hyb.set_velocity_field(None)

    stea_out = _fused_last_from_streaming(stea, cube)
    hyb_out = _fused_last_from_streaming(hyb, cube)
    assert torch.allclose(stea_out, hyb_out, atol=1e-5, rtol=1e-5)


def test_hyb_streaming_matches_debug_without_velocity() -> None:
    cube = _random_cube(24, 24, 55, seed=13)
    hyb = HybridSpatioTemporalEvidenceAccumulation(fast_window=8, slow_window=20, temporal_window=5)
    hyb.set_velocity_field(None)
    debug_last, _ = _fused_last_from_debug(hyb, cube)
    stream_last = _fused_last_from_streaming(hyb, cube)
    assert torch.allclose(debug_last, stream_last, atol=1e-5, rtol=1e-5)


def test_hyb_streaming_matches_debug_with_velocity() -> None:
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
    stream_last = _fused_last_from_streaming(hyb, cube)
    assert torch.allclose(debug_last, stream_last, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for peak memory check")
def test_stea_streaming_reduces_peak_memory() -> None:
    h, w, t = 128, 128, 96
    cube = _random_cube(h, w, t, seed=23).cuda()

    def _peak_bytes(fn) -> int:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated() - base)

    stea = SpatioTemporalEvidenceAccumulation(fast_window=12, slow_window=32, temporal_window=5).cuda()

    debug_peak = _peak_bytes(lambda: _fused_last_from_debug(stea, cube))
    stream_peak = _peak_bytes(lambda: _fused_last_from_streaming(stea, cube))
    assert stream_peak < debug_peak * 0.6
