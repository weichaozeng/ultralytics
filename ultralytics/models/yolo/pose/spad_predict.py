# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""SPAD-specific pose predictor.

This module provides a drop-in replacement for Ultralytics' `PosePredictor` that lets you
insert SPAD preprocessing (e.g., PerPixelBayesian reconstruction and/or QNN-based preprocessing)
*before* the standard YOLO preprocessing (letterbox, normalization, etc.).
"""

from __future__ import annotations

import numpy as np
import torch

from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.utils import DEFAULT_CFG


class SPADPosePredictor(PosePredictor):
    """PosePredictor with an optional SPAD preprocessing stage."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg=DEFAULT_CFG if cfg is None else cfg, overrides=overrides, _callbacks=_callbacks)

        self.spad_enabled = bool(getattr(self.args, "spad", False))
        self.spad_pre = getattr(self.args, "spad_pre", "bayes")
        self.spad_bayes_kwargs = getattr(self.args, "spad_bayes_kwargs", None) or {}
        self.spad_chunk_t = int(getattr(self.args, "spad_chunk_t", 0) or 0)
        self.spad_collapse = getattr(self.args, "spad_collapse", "frames")
        self.spad_rgb_mode = getattr(self.args, "spad_rgb_mode", "gray")
        self.spad_packed_reduce = getattr(self.args, "spad_packed_reduce", None)
        self.spad_bayer_pattern = getattr(self.args, "spad_bayer_pattern", "RGGB")
        self.spad_packed_ch_order = getattr(self.args, "spad_packed_ch_order", "RGB")
        self.spad_swap_rb = bool(getattr(self.args, "spad_swap_rb", False))

        if self.spad_packed_reduce == "rggb_raw" and self.spad_rgb_mode != "rggb_demosaic":
            self.spad_rgb_mode = "rggb_demosaic"
        elif self.spad_packed_reduce in ("any", "sum") and self.spad_rgb_mode == "rggb_demosaic":
            self.spad_rgb_mode = "gray"

        self.perpixel_bayes = None

        if self.spad_enabled and self.spad_pre == "bayes":
            from ultralytics.quanta_neural_networks.integrator import PerPixelBayesian
            bayes_kwargs = dict(self.spad_bayes_kwargs)
            bayes_kwargs["normalize"] = False
            self.perpixel_bayes = PerPixelBayesian(**bayes_kwargs)

        self.last_recon_frames_u8: list[np.ndarray] | None = None
        self.spad_t_offset: int = int(getattr(self.args, "spad_t_offset", 0) or 0)
        self.last_recon_t01: tuple[int, int] | None = None

    # --------------------------
    # Overrides
    # --------------------------
    def preprocess(self, im):
        """Apply optional SPAD preprocessing, then run the default Ultralytics preprocess pipeline."""
        if not self.spad_enabled:
            return super().preprocess(im)

        # 🚀 极致纯粹：直接传，不要有任何多余的骚操作
        if isinstance(im, np.ndarray) and im.ndim == 3:
            frames = self._spad_cube_to_frames(im)
            return super().preprocess(frames)

        if not isinstance(im, torch.Tensor):
            out = []
            for x in im:
                if isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[-1] != 3:
                    out.extend(self._spad_cube_to_frames(x))
                else:
                    out.append(self._spad_preprocess_frame(x))
            return super().preprocess(out)

        return super().preprocess(im)

    def postprocess(self, preds, img, orig_imgs):
        """Override postprocess to fix orig_imgs mismatch when expanding cubes."""
        if self.spad_enabled and self.last_recon_frames_u8 is not None:
            orig_imgs = self.last_recon_frames_u8
        return super().postprocess(preds, img, orig_imgs)

    # --------------------------
    # SPAD preprocessing
    # --------------------------
    @staticmethod
    def _normalize_frame_per_frame(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        mn = float(x.min())
        mx = float(x.max())
        return (x - mn) / (mx - mn + eps)

    @staticmethod
    def _demosaic_bayer_to_bgr_u8(raw_u8: np.ndarray, *, pattern: str = "RGGB") -> np.ndarray:
        import cv2
        if raw_u8.ndim != 2: raise ValueError(f"Expected raw (H,W) uint8, got shape={raw_u8.shape}")
        if raw_u8.dtype != np.uint8: raw_u8 = raw_u8.astype(np.uint8, copy=False)
        code_map = {
            "RGGB": cv2.COLOR_BayerRG2BGR,
            "BGGR": cv2.COLOR_BayerBG2BGR,
            "GRBG": cv2.COLOR_BayerGR2BGR,
            "GBRG": cv2.COLOR_BayerGB2BGR,
        }
        return cv2.cvtColor(raw_u8, code_map.get(pattern, cv2.COLOR_BayerRG2BGR))

    @staticmethod
    def _unexpand_to_bgr_u8(raw2w_u8: np.ndarray) -> np.ndarray:
        if raw2w_u8.ndim != 2: raise ValueError(f"Expected (2H,2W) raw, got {raw2w_u8.shape}")
        h2, w2 = raw2w_u8.shape
        h, w = h2 // 2, w2 // 2
        bgr = np.zeros((h, w, 3), dtype=np.uint8)

        r  = raw2w_u8[0::2, 0::2]
        g1 = raw2w_u8[0::2, 1::2]
        g2 = raw2w_u8[1::2, 0::2]
        b  = raw2w_u8[1::2, 1::2]

        g = ((g1.astype(np.uint16) + g2.astype(np.uint16)) // 2).astype(np.uint8)
        bgr[:, :, 0] = b
        bgr[:, :, 1] = g
        bgr[:, :, 2] = r
        return bgr

    def _to_3ch_u8(self, img01: np.ndarray) -> np.ndarray:
        u8 = np.clip((img01 * 255.0).round(), 0, 255).astype(np.uint8)
        if self.spad_rgb_mode == "rggb_demosaic":
            if self.spad_packed_reduce == "rggb_expand":
                bgr = self._unexpand_to_bgr_u8(u8)
            else:
                bgr = self._demosaic_bayer_to_bgr_u8(u8, pattern=self.spad_bayer_pattern)
            if self.spad_swap_rb and bgr.ndim == 3 and bgr.shape[2] == 3:
                bgr = bgr[:, :, ::-1]
            return bgr
        return np.repeat(u8[:, :, None], 3, axis=2)

    def _spad_cube_to_frames(self, cube_hwt: np.ndarray) -> list[np.ndarray]:
        self.last_recon_frames_u8 = None
        self.last_recon_t01 = None

        if self.perpixel_bayes is None:
            raise RuntimeError("PerPixelBayesian is not initialized (spad_pre != 'bayes'?)")

        device = getattr(self, "device", torch.device("cpu"))
        self.perpixel_bayes.to(device)

        cube_bool = cube_hwt.astype(bool, copy=False)
        _, _, t = cube_bool.shape

        chunk_t = self.spad_chunk_t if self.spad_chunk_t > 0 else t

        # 🚀 在循环外只读取一次：你外面传进来的完美信号 (True 还是 False)
        current_clear = bool(getattr(self.args, "spad_clear_states", False))

        frames_out: list[np.ndarray] = []
        for t0 in range(0, t, chunk_t):
            t1 = min(t, t0 + chunk_t)
            photon = torch.from_numpy(cube_bool[:, :, t0:t1]).to(device)

            recons = self.perpixel_bayes.process_photon_cube(
                photon,
                clear_states=current_clear, # 🚀 原封不动传给底层，触发完美的 ema.zero_()
                normalize=False,
                **{k: v for k, v in self.spad_bayes_kwargs.items() if k != "normalize"},
            )
            
            # 🚀 以防万一有人没用 det_qnns，而是传了个几万帧的整张大图进来，
            # 强制保证循环的第二圈不重置
            current_clear = False 

            recons_np = recons.detach().float().cpu().numpy()

            if recons_np.shape[2] == 0:
                try:
                    ema = getattr(self.perpixel_bayes, 'ema', None)
                    if ema is not None:
                        recons_np = ema.detach().float().cpu().numpy()[:, :, None]
                    else:
                        raise AttributeError
                except Exception:
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

        self.last_recon_frames_u8 = frames_out
        self.last_recon_t01 = (int(self.spad_t_offset), int(self.spad_t_offset) + int(t))
        return frames_out

    def _spad_preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not isinstance(frame_bgr, np.ndarray): raise TypeError(f"Expected np.ndarray, got {type(frame_bgr)}")
        if frame_bgr.ndim == 2: return np.repeat(frame_bgr.astype(np.uint8, copy=False)[:, :, None], 3, axis=2)
        return frame_bgr