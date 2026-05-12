# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""SPAD-specific pose predictor.

This module provides a drop-in replacement for Ultralytics' `PosePredictor` that lets you
insert SPAD preprocessing (e.g., PerPixelBayesian reconstruction and/or QNN-based preprocessing)
*before* the standard YOLO preprocessing (letterbox, normalization, etc.).

Design goals:
- Keep upstream behavior unchanged when SPAD preprocessing is disabled.
- Allow stateful preprocessors (e.g., PerPixelBayesian) to be reset between sequences.

How to use (example):
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose.spad_predict import SPADPosePredictor

    model = YOLO('weights/detector.pt')
    results = model.track(source, predictor=SPADPosePredictor, spad=True)

Notes:
- `ultralytics/det_qnns.py` drives chunking by calling `model.track()` per chunk. The predictor
  is responsible for causal SPAD reconstruction and per-frame normalization.
"""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.utils import DEFAULT_CFG


class SPADPosePredictor(PosePredictor):
    """PosePredictor with an optional SPAD preprocessing stage."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # Ultralytics BasePredictor expects `cfg` to be a dict/path; passing None will crash in get_cfg().
        super().__init__(cfg=DEFAULT_CFG if cfg is None else cfg, overrides=overrides, _callbacks=_callbacks)

        # Toggle: enabled when `spad=True` is passed via overrides/kwargs (see Model.predict())
        self.spad_enabled = bool(getattr(self.args, "spad", False))

        # Runtime options (passed via overrides/kwargs)
        self.spad_pre = getattr(self.args, "spad_pre", "bayes")
        self.spad_clear_states = bool(getattr(self.args, "spad_clear_states", False))
        self.spad_bayes_kwargs = getattr(self.args, "spad_bayes_kwargs", None) or {}

        # If a full cube is passed in, we can optionally chunk internally.
        self.spad_chunk_t = int(getattr(self.args, "spad_chunk_t", 0) or 0)

        # Output collapse behavior for cube inputs.
        # - 'frames': expand a cube to list of reconstructed frames and run model per frame.
        # - 'sum'   : collapse to a single image (debug)
        self.spad_collapse = getattr(self.args, "spad_collapse", "frames")

        # Stateful preprocessor
        self.perpixel_bayes = None

        if self.spad_enabled and self.spad_pre == "bayes":
            from ultralytics.quanta_neural_networks.integrator import PerPixelBayesian

            # IMPORTANT: disable integrator's internal normalization to keep preprocessing causal.
            # We do per-frame normalization on the output recon frames instead.
            bayes_kwargs = dict(self.spad_bayes_kwargs)
            bayes_kwargs["normalize"] = False
            self.perpixel_bayes = PerPixelBayesian(**bayes_kwargs)

        # Cache for debug/visualization. Populated on each cube preprocess.
        # last_recon_frames_u8: list of HxWx3 uint8 frames (after per-frame normalization)
        self.last_recon_frames_u8: list[np.ndarray] | None = None
        # Absolute time offset of the current cube chunk in the original big cube (set by caller/script)
        self.spad_t_offset: int = int(getattr(self.args, "spad_t_offset", 0) or 0)
        # Cached (t0, t1) for the most recent preprocess cube input, in absolute big-cube indices
        self.last_recon_t01: tuple[int, int] | None = None

    # --------------------------
    # Public helpers
    # --------------------------
    def reset_spad_state(self):
        """Reset any stateful SPAD preprocessors (call this at the start of each new sequence)."""
        if self.perpixel_bayes is None:
            return

        # Reset BOCPD arrays and time index.
        self.perpixel_bayes.t_absolute = 0
        if hasattr(self.perpixel_bayes, "init_bocpd_arrays"):
            try:
                self.perpixel_bayes.init_bocpd_arrays()
            except Exception:
                # Safe fallback: next set_cube() call will re-init when t_absolute==0.
                pass

    # --------------------------
    # Overrides
    # --------------------------
    def preprocess(self, im):
        """Apply optional SPAD preprocessing, then run the default Ultralytics preprocess pipeline."""

        if not self.spad_enabled:
            return super().preprocess(im)

        # If the caller passes a photon cube (H,W,T) directly.
        if isinstance(im, np.ndarray) and im.ndim == 3:
            if self.spad_clear_states:
                self.reset_spad_state()
                # Ensure only cleared once per *big* cube if the script chunks externally.
                self.spad_clear_states = False

            frames = self._spad_cube_to_frames(im)
            return super().preprocess(frames)

        # If caller passes list-like, allow either cube per item or already-image frames.
        if not isinstance(im, torch.Tensor):
            out = []
            for x in im:
                if isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[-1] != 3:
                    out.extend(self._spad_cube_to_frames(x))
                else:
                    out.append(self._spad_preprocess_frame(x))
            return super().preprocess(out)

        # Tensor path is assumed already preprocessed.
        return super().preprocess(im)

    # --------------------------
    # SPAD preprocessing
    # --------------------------
    @staticmethod
    def _normalize_frame_per_frame(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Causal per-frame min-max normalization to [0,1]."""
        x = x.astype(np.float32, copy=False)
        mn = float(x.min())
        mx = float(x.max())
        return (x - mn) / (mx - mn + eps)

    def _spad_cube_to_frames(self, cube_hwt: np.ndarray) -> list[np.ndarray]:
        """Convert a photon cube (H,W,T) to a list of BGR uint8 frames (HxWx3)."""
        # Clear previous cache
        self.last_recon_frames_u8 = None
        self.last_recon_t01 = None

        if self.perpixel_bayes is None:
            raise RuntimeError("PerPixelBayesian is not initialized (spad_pre != 'bayes'?)")

        if cube_hwt.ndim != 3:
            raise ValueError(f"Expected cube (H,W,T), got shape={cube_hwt.shape}")

        # Move integrator to same device as the model.
        device = getattr(self, "device", torch.device("cpu"))
        self.perpixel_bayes.to(device)

        cube_bool = cube_hwt.astype(bool, copy=False)
        _, _, t = cube_bool.shape

        chunk_t = self.spad_chunk_t if self.spad_chunk_t > 0 else t

        frames_out: list[np.ndarray] = []
        for t0 in range(0, t, chunk_t):
            t1 = min(t, t0 + chunk_t)
            photon = torch.from_numpy(cube_bool[:, :, t0:t1]).to(device)

            recons = self.perpixel_bayes.process_photon_cube(
                photon,
                clear_states=False,
                normalize=False,
                **{k: v for k, v in self.spad_bayes_kwargs.items() if k != "normalize"},
            )
            recons_np = recons.detach().float().cpu().numpy()  # (H,W,T')

            # If subsampling > (t1-t0), integrator may produce T'==0. Avoid returning an empty list.
            if recons_np.shape[2] == 0:
                # Best-effort causal fallback: use current EMA as a single frame if available.
                try:
                    ema = getattr(self.perpixel_bayes, 'ema', None)
                    if ema is not None:
                        recons_np = ema.detach().float().cpu().numpy()[:, :, None]
                    else:
                        raise AttributeError
                except Exception:
                    # Last resort: sum the photon chunk (still causal) to produce one frame.
                    img = photon.detach().float().cpu().numpy().sum(axis=2)
                    recons_np = img[:, :, None]


            if self.spad_collapse == "sum":
                img = recons_np.sum(axis=2)
                img01 = self._normalize_frame_per_frame(img)
                img_u8 = (img01 * 255.0).round().astype(np.uint8)
                frames_out.append(np.repeat(img_u8[:, :, None], 3, axis=2))
                continue

            for i in range(recons_np.shape[2]):
                f01 = self._normalize_frame_per_frame(recons_np[:, :, i])
                f_u8 = (f01 * 255.0).round().astype(np.uint8)
                frames_out.append(np.repeat(f_u8[:, :, None], 3, axis=2))

        # Save cache for visualization/debug (used by det_qnns.py)
        self.last_recon_frames_u8 = frames_out
        self.last_recon_t01 = (int(self.spad_t_offset), int(self.spad_t_offset) + int(t))
        return frames_out

    def _spad_preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Pass-through when already provided with image-like frames."""
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(frame_bgr)}")

        if frame_bgr.ndim == 2:
            u8 = frame_bgr.astype(np.uint8, copy=False)
            return np.repeat(u8[:, :, None], 3, axis=2)

        if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
            return frame_bgr

        return frame_bgr
