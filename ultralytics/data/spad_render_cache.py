"""Shared helpers for offline SPAD render caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CACHE_META_VERSION = 1


def sanitize_render_sample_name(name: str) -> str:
    """Make sample names filesystem-safe while preserving readability."""
    text = str(name).strip()
    if not text:
        raise ValueError("Sample name must be non-empty.")
    return text.replace("\\", "_").replace("/", "_")


def sample_render_dir(root: str | Path, sample_name: str) -> Path:
    """Return the per-sample cache directory under one render root."""
    return Path(root) / sanitize_render_sample_name(sample_name)


def build_render_config(
    *,
    preprocessor: str,
    chunk_size: int,
    stride_bins: int,
    spad_bins_per_gt: int,
    packed_ch_order: str,
    input_gamma: float,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical config payload for cache metadata and fingerprinting."""
    return {
        "preprocessor": str(preprocessor).strip().lower(),
        "chunk_size": int(chunk_size),
        "stride_bins": int(stride_bins),
        "spad_bins_per_gt": int(spad_bins_per_gt),
        "packed_ch_order": str(packed_ch_order).strip().upper(),
        "input_gamma": float(input_gamma),
        "extra_kwargs": dict(extra_kwargs or {}),
    }


def render_config_fingerprint(config: dict[str, Any]) -> str:
    """Create a stable short fingerprint for one render-cache configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]
