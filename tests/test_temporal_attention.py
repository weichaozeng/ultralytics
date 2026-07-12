"""Unit tests for spatio-temporal attention cores and plugins."""

from __future__ import annotations

import copy

import pytest
import torch

from ultralytics.models.yolo.pose.spad_plugins import (
    SRAttentionPlugin,
    TemporalAttentionPlugin,
    TemporalSSDPlugin,
    WindowSTAttentionPlugin,
    build_spad_plugin,
)
from ultralytics.quanta_neural_networks.ssd import SSD
from ultralytics.quanta_stea_networks.attn import (
    AbsolutePositionalEncoding3D,
    SRAttention,
    TemporalAttention,
    WindowSTAttention,
    build_causal_mask,
    count_parameters,
    flatten_tbchw,
    unflatten_tbchw,
)


@pytest.mark.parametrize("channels,state_dim,head_divisor", [(128, 8, 4), (64, 8, 4), (256, 8, 4)])
def test_temporal_attention_shape(channels: int, state_dim: int, head_divisor: int):
    head_dim = channels // head_divisor
    while channels % head_dim != 0 and head_dim > 1:
        head_dim -= 1
    t, b, h, w = 6, 2, 8, 8
    x = torch.randn(t, b * h * w, channels)
    model = TemporalAttention(in_dim=channels, state_dim=state_dim, head_dim=head_dim)
    out, out_t = model(x, list(range(t)))
    assert out.shape == x.shape
    assert out_t == list(range(t))


def test_temporal_attention_plugin_shape():
    t, b, c, h, w = 5, 2, 64, 8, 8
    head_dim = 16
    x = torch.randn(t, b, c, h, w)
    plugin = TemporalAttentionPlugin(in_dim=c, state_dim=8, head_dim=head_dim)
    out, out_t = plugin(x, list(range(t)))
    assert out.shape == x.shape
    assert out_t == list(range(t))


def test_absolute_positional_encoding_tbchw():
    t, b, c, h, w = 4, 2, 30, 5, 6
    x = torch.zeros(t, b, c, h, w)
    pe = AbsolutePositionalEncoding3D(c)
    y = pe(x, layout="TBCHW")
    assert y.shape == x.shape
    assert not torch.allclose(y, x)


def test_flatten_unflatten_roundtrip():
    t, b, c, h, w = 3, 2, 16, 4, 5
    x = torch.randn(t, b, c, h, w)
    flat, meta = flatten_tbchw(x)
    assert flat.shape == (t, b * h * w, c)
    restored = unflatten_tbchw(flat, meta)
    assert torch.allclose(restored, x)


def test_causal_mask_blocks_future():
    t = 5
    mask = build_causal_mask(t, torch.device("cpu"))
    assert torch.isinf(mask[0, 1])
    assert mask[1, 0] == 0
    assert mask[0, 0] == 0


def test_temporal_attention_causality():
    torch.manual_seed(0)
    t, batch, c = 8, 16, 64
    head_dim = 16
    model = TemporalAttention(in_dim=c, state_dim=8, head_dim=head_dim)
    x = torch.randn(t, batch, c)
    out_full, _ = model(x, list(range(t)))

    for ti in range(t):
        x_perturbed = x.clone()
        x_perturbed[ti + 1 :] = torch.randn_like(x_perturbed[ti + 1 :])
        out_perturbed, _ = model(x_perturbed, list(range(t)))
        assert torch.allclose(out_full[: ti + 1], out_perturbed[: ti + 1], atol=1e-5, rtol=1e-5)


def test_temporal_attention_online_matches_batch():
    torch.manual_seed(1)
    t, batch, c = 6, 12, 64
    head_dim = 16
    model = TemporalAttention(in_dim=c, state_dim=8, head_dim=head_dim)
    x = torch.randn(t, batch, c)
    out_batch, _ = model(x, list(range(t)))

    model.clear_hidden_state()
    online_out = []
    for ti in range(t):
        online_out.append(model.forward_online(x[ti], time_instant=float(ti)))
    out_online = torch.stack(online_out, dim=0)
    assert torch.allclose(out_batch, out_online, atol=1e-5, rtol=1e-5)


def test_temporal_attention_plugin_online():
    torch.manual_seed(2)
    b, c, h, w = 2, 64, 4, 4
    t = 5
    plugin = TemporalAttentionPlugin(in_dim=c, state_dim=8, head_dim=16)
    x = torch.randn(t, b, c, h, w)

    out_batch, _ = plugin(x, list(range(t)))
    plugin.set_online_mode(True)
    plugin.clear_temporal_state()
    online = []
    for ti in range(t):
        frame = x[ti : ti + 1]
        step, _ = plugin(frame, [ti])
        online.append(step[0])
    out_online = torch.stack(online, dim=0)
    assert torch.allclose(out_batch, out_online, atol=1e-5, rtol=1e-5)


def test_parameter_count_near_ssd():
    channels = 128
    state_dim = 8
    head_dim = 32
    ssd = SSD(in_dim=channels, state_dim=state_dim, head_dim=head_dim)
    attn = TemporalAttention(in_dim=channels, state_dim=state_dim, head_dim=head_dim)
    ssd_params = count_parameters(ssd)
    attn_params = count_parameters(attn)
    ratio = attn_params / ssd_params
    assert 0.5 <= ratio <= 3.0, f"attn/ssd param ratio out of range: {ratio:.2f}"


def test_sr_attention_shape():
    t, b, c, h, w = 4, 2, 64, 8, 8
    head_dim = 16
    x = torch.randn(t, b, c, h, w)
    model = SRAttention(in_dim=c, state_dim=8, head_dim=head_dim, sr_ratio=2)
    out, out_t = model(x, list(range(t)))
    flat, _ = flatten_tbchw(x)
    assert out.shape == flat.shape
    assert out_t == list(range(t))


def test_sr_attention_plugin_shape():
    t, b, c, h, w = 3, 2, 64, 8, 8
    plugin = SRAttentionPlugin(in_dim=c, state_dim=8, head_dim=16, attn_kwargs={"sr_ratio": 2})
    x = torch.randn(t, b, c, h, w)
    out, out_t = plugin(x, list(range(t)))
    assert out.shape == x.shape
    assert out_t == list(range(t))


def test_window_st_attention_shape():
    t, b, c, h, w = 4, 2, 64, 14, 14
    head_dim = 16
    x = torch.randn(t, b, c, h, w)
    model = WindowSTAttention(in_dim=c, state_dim=8, head_dim=head_dim, window_size=(3, 7, 7))
    out, out_t = model(x, list(range(t)))
    flat, _ = flatten_tbchw(x)
    assert out.shape == flat.shape
    assert out_t == list(range(t))


def test_window_st_attention_plugin_shape():
    t, b, c, h, w = 3, 2, 64, 14, 14
    plugin = WindowSTAttentionPlugin(
        in_dim=c,
        state_dim=8,
        head_dim=16,
        attn_kwargs={"window_size": (3, 7, 7)},
    )
    x = torch.randn(t, b, c, h, w)
    out, out_t = plugin(x, list(range(t)))
    assert out.shape == x.shape
    assert out_t == list(range(t))


def test_window_st_attention_causality():
    torch.manual_seed(0)
    t, b, c, h, w = 6, 2, 64, 14, 14
    model = WindowSTAttention(in_dim=c, state_dim=8, head_dim=16, window_size=(3, 7, 7))
    model.eval()
    x = torch.randn(t, b, c, h, w)
    out_full, _ = model(x, list(range(t)))
    for ti in range(t):
        x_perturbed = x.clone()
        x_perturbed[ti + 1 :] = torch.randn_like(x_perturbed[ti + 1 :])
        out_perturbed, _ = model(x_perturbed, list(range(t)))
        assert torch.allclose(out_full[: ti + 1], out_perturbed[: ti + 1], atol=1e-5, rtol=1e-5)


def test_window_st_attention_online_matches_batch():
    torch.manual_seed(1)
    t, b, c, h, w = 7, 2, 64, 14, 14
    model = WindowSTAttention(in_dim=c, state_dim=8, head_dim=16, window_size=(3, 7, 7))
    model.eval()
    x = torch.randn(t, b, c, h, w)
    with torch.no_grad():
        out_batch, _ = model(x, list(range(t)))
        model.clear_hidden_state()
        online = []
        for ti in range(t):
            online.append(model.forward_online(x[ti], time_instant=float(ti)))
        out_online = torch.stack(online, dim=0)
        out_online_flat, _ = flatten_tbchw(out_online)
    assert torch.allclose(out_batch, out_online_flat, atol=1e-5, rtol=1e-5)


def test_window_st_attention_rel_pos_used_for_long_t():
    """rel_pos must affect outputs even when T != window_t."""
    torch.manual_seed(2)
    t, b, c, h, w = 5, 1, 64, 14, 14
    model = WindowSTAttention(in_dim=c, state_dim=8, head_dim=16, window_size=(3, 7, 7))
    model.eval()
    x = torch.randn(t, b, c, h, w)
    with torch.no_grad():
        out0, _ = model(x, list(range(t)))
        model.rel_pos_bias.bias_table.data.uniform_(-1.0, 1.0)
        out1, _ = model(x, list(range(t)))
    assert not torch.allclose(out0, out1, atol=1e-6)


def test_window_st_attention_plugin_online():
    torch.manual_seed(3)
    t, b, c, h, w = 5, 2, 64, 14, 14
    plugin = WindowSTAttentionPlugin(
        in_dim=c,
        state_dim=8,
        head_dim=16,
        attn_kwargs={"window_size": (3, 7, 7)},
    )
    plugin.eval()
    x = torch.randn(t, b, c, h, w)
    with torch.no_grad():
        out_batch, _ = plugin(x, list(range(t)))
        plugin.set_online_mode(True)
        plugin.clear_temporal_state()
        online = []
        for ti in range(t):
            step, _ = plugin(x[ti : ti + 1], [ti])
            online.append(step[0])
        out_online = torch.stack(online, dim=0)
    assert torch.allclose(out_batch, out_online, atol=1e-5, rtol=1e-5)

def test_build_spad_plugin_variants():
    for name in ("temporal_attn", "sr_attn", "window_st_attn"):
        plugin = build_spad_plugin(name, in_dim=64, state_dim=8, head_dim=16)
        assert plugin is not None


def test_spatial_temporal_plugin_with_temporal_attn_core():
    plugin = build_spad_plugin(
        "spatial_temporal",
        in_dim=64,
        state_dim=8,
        head_dim=16,
        temporal_core="temporal_attn",
    )
    t, b, c, h, w = 4, 2, 64, 8, 8
    x = torch.randn(t, b, c, h, w)
    out, out_t = plugin(x, list(range(t)))
    assert out.shape[0] == t
    assert out.shape[1:] == x.shape[1:]


def test_temporal_ssd_plugin_still_works():
    plugin = TemporalSSDPlugin(in_dim=64, state_dim=8, head_dim=16)
    t, b, c, h, w = 3, 2, 64, 4, 4
    x = torch.randn(t, b, c, h, w)
    out, out_t = plugin(x, list(range(t)))
    assert out.shape == x.shape
