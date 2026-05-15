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

        # How to convert a single-channel reconstruction to a 3-channel image for YOLO/visualization.
        # - 'gray': repeat grayscale to 3 channels (default)
        # - 'rggb_demosaic': treat as Bayer RGGB RAW and demosaic to BGR
        self.spad_rgb_mode = getattr(self.args, "spad_rgb_mode", "gray")

        # Optional hint from caller about how the single-channel cube was produced.
        # det_qnns.py can set this to sanity-check mode combinations.
        self.spad_packed_reduce = getattr(self.args, "spad_packed_reduce", None)

        # Additional hints for interpreting raw mosaic.
        self.spad_bayer_pattern = getattr(self.args, "spad_bayer_pattern", "RGGB")
        self.spad_packed_ch_order = getattr(self.args, "spad_packed_ch_order", "RGB")

        # If demosaic output has R/B swapped due to unknown sensor/channel conventions, allow a post-swap.
        self.spad_swap_rb = bool(getattr(self.args, "spad_swap_rb", False))

        # Consistency check / auto-correction.
        if self.spad_packed_reduce == "rggb_raw" and self.spad_rgb_mode != "rggb_demosaic":
            self.spad_rgb_mode = "rggb_demosaic"
        elif self.spad_packed_reduce in ("any", "sum") and self.spad_rgb_mode == "rggb_demosaic":
            self.spad_rgb_mode = "gray"

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

    @staticmethod
    def _demosaic_bayer_to_bgr_u8(raw_u8: np.ndarray, *, pattern: str = "RGGB") -> np.ndarray:
        """Demosaic a Bayer uint8 image to BGR uint8 using OpenCV."""
        import cv2

        if raw_u8.ndim != 2:
            raise ValueError(f"Expected raw (H,W) uint8, got shape={raw_u8.shape}")
        if raw_u8.dtype != np.uint8:
            raw_u8 = raw_u8.astype(np.uint8, copy=False)
        code_map = {
            "RGGB": cv2.COLOR_BayerRG2BGR,
            "BGGR": cv2.COLOR_BayerBG2BGR,
            "GRBG": cv2.COLOR_BayerGR2BGR,
            "GBRG": cv2.COLOR_BayerGB2BGR,
        }
        code = code_map.get(pattern, cv2.COLOR_BayerRG2BGR)
        return cv2.cvtColor(raw_u8, code)

    @staticmethod
    def _unexpand_to_bgr_u8(raw2w_u8: np.ndarray) -> np.ndarray:
        """
        Directly un-expand the (2H, 2W) PPB reconstruction back to (H, W, 3) BGR.
        This preserves 100% of the pixels and avoids cv2 demosaic artifacts.
        """
        if raw2w_u8.ndim != 2:
            raise ValueError(f"Expected (2H,2W) raw, got shape={raw2w_u8.shape}")
        h2, w2 = raw2w_u8.shape
        if (h2 % 2) != 0 or (w2 % 2) != 0:
            raise ValueError(f"Expected even (2H,2W), got shape={raw2w_u8.shape}")
        
        h, w = h2 // 2, w2 // 2
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # 从展开的 2H x 2W 阵列中提取各个通道
        r  = raw2w_u8[0::2, 0::2]
        g1 = raw2w_u8[0::2, 1::2]
        g2 = raw2w_u8[1::2, 0::2]
        b  = raw2w_u8[1::2, 1::2]

        # 因为 PPB 平滑后两个 G 像素可能存在微小差异，取平均以防高频噪点
        # 转换为 uint16 防止加法溢出
        g = ((g1.astype(np.uint16) + g2.astype(np.uint16)) // 2).astype(np.uint8)

        # 装填为 OpenCV 标准的 BGR 格式
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return rgb

    def _to_3ch_u8(self, img01: np.ndarray) -> np.ndarray:
        """Convert float [0,1] (H,W) to uint8 (H,W,3) according to spad_rgb_mode."""
        u8 = np.clip((img01 * 255.0).round(), 0, 255).astype(np.uint8)
        
        if self.spad_rgb_mode == "rggb_demosaic":
            if self.spad_packed_reduce == "rggb_expand":
                # 无损逆向还原，跳过 cv2.cvtColor
                bgr = self._unexpand_to_bgr_u8(u8)
            else:
                # 只有真实的单传感 Bayer Raw 才需要走 OpenCV demosaic
                bgr = self._demosaic_bayer_to_bgr_u8(u8, pattern=self.spad_bayer_pattern)
                
            if self.spad_swap_rb and bgr.ndim == 3 and bgr.shape[2] == 3:
                bgr = bgr[:, :, ::-1]  # swap R/B
            return bgr
            
        return np.repeat(u8[:, :, None], 3, axis=2)

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
                frames_out.append(self._to_3ch_u8(img01))
                continue

            for i in range(recons_np.shape[2]):
                f01 = self._normalize_frame_per_frame(recons_np[:, :, i])
                frames_out.append(self._to_3ch_u8(f01))

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

