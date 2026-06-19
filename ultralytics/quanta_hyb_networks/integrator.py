"""Hybrid STEA+velocity SPAD integrator."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ultralytics.data.spad_packed import rgb_tchw_to_raw_hwt
from ultralytics.quanta_neural_networks.ops.image import nearest_neighbor_inpaint
from ultralytics.quanta_stea_networks.integrator import SpatioTemporalEvidenceAccumulation
from ultralytics.quanta_neural_networks.ops.array_ops import torch_quantile


class HybridSpatioTemporalEvidenceAccumulation(SpatioTemporalEvidenceAccumulation):
    """STEA variant whose slow branch uses velocity-compensated photon history."""

    def __init__(
        self,
        *args,
        warp_block_size: int = 16,
        source_space: str = "rgb",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.warp_block_size = max(int(warp_block_size), 1)
        self.velocity_field: Tensor | None = None
        self.velocity_field_space: str = str(source_space).lower()
        if self.velocity_field_space not in {"rgb", "raw"}:
            raise ValueError(f"source_space must be 'rgb' or 'raw', got {source_space!r}")

    def reset(self) -> None:
        self.t_absolute = 0
        self._clear_histories()
        self.velocity_field = None

    def set_velocity_field(self, field: Tensor | np.ndarray | None, *, source_space: str = "rgb") -> None:
        source_space = str(source_space).lower()
        if source_space not in {"rgb", "raw"}:
            raise ValueError(f"source_space must be 'rgb' or 'raw', got {source_space!r}")
        self.velocity_field_space = source_space
        if field is None:
            self.velocity_field = None
            return
        field_t = torch.as_tensor(field, dtype=torch.float32)
        if field_t.ndim != 3 or field_t.shape[-1] != 2:
            raise ValueError(f"Velocity field must be shaped (H,W,2), got {tuple(field_t.shape)}")
        self.velocity_field = field_t.detach().cpu().contiguous()

    def update_hyperparams(self, **kwargs) -> None:
        source_space = kwargs.pop("source_space", None)
        warp_block_size = kwargs.pop("warp_block_size", None)
        if source_space is not None:
            source_space = str(source_space).lower()
            if source_space not in {"rgb", "raw"}:
                raise ValueError(f"source_space must be 'rgb' or 'raw', got {source_space!r}")
            self.velocity_field_space = source_space
        if warp_block_size is not None:
            self.warp_block_size = max(int(warp_block_size), 1)
        super().update_hyperparams(**kwargs)

    def clamp_recons(self, recons: Tensor) -> Tensor:
        if recons.numel() == 0:
            return recons.float()
        recons = recons.float()
        max_value = 1.0
        if self.normalize:
            max_value = torch_quantile(recons, self.quantile).clamp(min=1e-6)
        return (recons / max_value).clamp(0, 1)

    @staticmethod
    def _base_grid(h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((xx, yy), dim=-1)

    @staticmethod
    def _normalize_flow(flow_hw2: Tensor) -> Tensor:
        h, w = int(flow_hw2.shape[0]), int(flow_hw2.shape[1])
        scale_x = 0.0 if w <= 1 else 2.0 / float(w - 1)
        scale_y = 0.0 if h <= 1 else 2.0 / float(h - 1)
        flow = flow_hw2.clone()
        flow[..., 0] *= scale_x
        flow[..., 1] *= scale_y
        return flow

    def _resized_velocity_field(self, target_hw: tuple[int, int], *, device: torch.device, dtype: torch.dtype) -> Tensor:
        h, w = map(int, target_hw)
        if self.velocity_field is None:
            return torch.zeros((h, w, 2), device=device, dtype=dtype)
        field = self.velocity_field.to(device=device, dtype=dtype)
        src_h, src_w = int(field.shape[0]), int(field.shape[1])
        field_chw = field.permute(2, 0, 1).unsqueeze(0)
        if (src_h, src_w) != (h, w):
            field_chw = F.interpolate(field_chw, size=(h, w), mode="bilinear", align_corners=True)
        field = field_chw.squeeze(0).permute(1, 2, 0).contiguous()
        if self.velocity_field_space != "raw":
            field[..., 0] *= float(w) / max(float(src_w), 1.0)
            field[..., 1] *= float(h) / max(float(src_h), 1.0)
        elif (src_h, src_w) != (h, w):
            field[..., 0] *= float(w) / max(float(src_w), 1.0)
            field[..., 1] *= float(h) / max(float(src_h), 1.0)
        return field

    def _warp_frames(self, frames_tchw: Tensor, flow_hw2: Tensor, gaps: Tensor, denom: float) -> Tensor:
        if frames_tchw.ndim != 4:
            raise ValueError(f"Expected frames_tchw (T,C,H,W), got shape={tuple(frames_tchw.shape)}")
        _, _, h, w = frames_tchw.shape
        base_grid = self._base_grid(h, w, device=frames_tchw.device, dtype=frames_tchw.dtype)
        flow_norm = self._normalize_flow(flow_hw2.to(device=frames_tchw.device, dtype=frames_tchw.dtype))
        alpha = (gaps.to(device=frames_tchw.device, dtype=frames_tchw.dtype) / max(float(denom), 1.0)).view(-1, 1, 1, 1)
        grids = base_grid.unsqueeze(0) - alpha * flow_norm.unsqueeze(0)
        return F.grid_sample(
            frames_tchw,
            grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    @staticmethod
    def _raw_hwt_to_rgb_tchw(raw_hwt: Tensor) -> Tensor:
        if raw_hwt.ndim != 3:
            raise ValueError(f"Expected raw_hwt (H,W,T), got shape={tuple(raw_hwt.shape)}")
        h, w, _ = map(int, raw_hwt.shape)
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Expected even Bayer dimensions, got {(h, w)}")
        r = raw_hwt[0::2, 0::2, :]
        g = 0.5 * (raw_hwt[0::2, 1::2, :] + raw_hwt[1::2, 0::2, :])
        b = raw_hwt[1::2, 1::2, :]
        return torch.stack((r, g, b), dim=0).permute(3, 0, 1, 2).contiguous()

    def _warp_raw_frames_in_rgb(self, raw_hwt: Tensor, flow_hw2: Tensor, gaps: Tensor, denom: float) -> Tensor:
        rgb_tchw = self._raw_hwt_to_rgb_tchw(raw_hwt)
        warped_rgb = self._warp_frames(rgb_tchw, flow_hw2, gaps, denom)
        return rgb_tchw_to_raw_hwt(warped_rgb)

    def _fast_temporal_basis(self, photon_cube: Tensor) -> Tensor:
        h, w, t = map(int, photon_cube.shape)
        x = photon_cube.float()
        x_flat = x.reshape(h * w, 1, t)
        fast_hist = self._history_or_zeros(self.photon_history, h, w, self.fast_window - 1, x)
        y_fast = self._causal_conv1d(x_flat, self.fast_kernel, fast_hist)
        return y_fast.clamp(self.eps, 1.0 - self.eps)

    def _slow_temporal_basis(self, photon_cube: Tensor) -> Tensor:
        h, w, t = map(int, photon_cube.shape)
        x = photon_cube.float()
        slow_hist = self._history_or_zeros(self.photon_history, h, w, self.slow_window - 1, x)

        if self.velocity_field is None:
            support = torch.cat([slow_hist, x], dim=-1)
            support_flat = support.reshape(h * w, 1, support.shape[-1])
            y_slow = F.conv1d(support_flat, self.slow_kernel)
            return y_slow.clamp(self.eps, 1.0 - self.eps)

        flow = self._resized_velocity_field((h // 2, w // 2), device=x.device, dtype=x.dtype)
        if not bool(torch.any(flow.abs() > 1e-6)):
            support = torch.cat([slow_hist, x], dim=-1)
            support_flat = support.reshape(h * w, 1, support.shape[-1])
            y_slow = F.conv1d(support_flat, self.slow_kernel)
            return y_slow.clamp(self.eps, 1.0 - self.eps)

        support = torch.cat([slow_hist, x], dim=-1)
        support_len = int(support.shape[-1])
        prefix_len = support_len - t
        diff = x.new_zeros(h, w, t + 1)
        denom = max(int(self.chunk_size) - 1, 1)

        for start in range(0, support_len, self.warp_block_size):
            end = min(start + self.warp_block_size, support_len)
            frames = support[..., start:end]
            gaps = (support_len - 1) - torch.arange(start, end, device=x.device)
            warped = self._warp_raw_frames_in_rgb(frames, flow, gaps, denom)
            for local_idx, support_idx in enumerate(range(start, end)):
                t_start = max(0, support_idx - self.slow_window + 1)
                t_end = min(support_idx, t - 1)
                if t_start > t_end:
                    continue
                frame = warped[..., local_idx]
                diff[..., t_start] += frame
                if t_end + 1 < t:
                    diff[..., t_end + 1] -= frame

        y_slow = torch.cumsum(diff[..., :t], dim=-1) / float(self.slow_window)
        return y_slow.reshape(h * w, 1, t).clamp(self.eps, 1.0 - self.eps)

    def _mean_stable_tail_aligned(self, photon_cube: Tensor, valid_weight: Tensor) -> Tensor:
        h, w, t = map(int, photon_cube.shape)
        x = photon_cube.float()
        if self.velocity_field is None:
            stable_support = valid_weight.sum(dim=-1)
            return (valid_weight * x).sum(dim=-1) / stable_support.clamp(min=self.eps)

        flow = self._resized_velocity_field((h // 2, w // 2), device=x.device, dtype=x.dtype)
        if not bool(torch.any(flow.abs() > 1e-6)):
            stable_support = valid_weight.sum(dim=-1)
            return (valid_weight * x).sum(dim=-1) / stable_support.clamp(min=self.eps)

        accum = x.new_zeros(h, w)
        denom = max(int(self.chunk_size) - 1, 1)
        for start in range(0, t, self.warp_block_size):
            end = min(start + self.warp_block_size, t)
            frames = x[..., start:end]
            gaps = (t - 1) - torch.arange(start, end, device=x.device)
            warped = self._warp_raw_frames_in_rgb(frames, flow, gaps, denom)
            accum += (warped * valid_weight[..., start:end]).sum(dim=-1)
        stable_support = valid_weight.sum(dim=-1)
        return accum / stable_support.clamp(min=self.eps)

    def _integrate_last(self, photon_cube: Tensor) -> Tensor:
        h, w, t = map(int, photon_cube.shape)
        if t == 0:
            return photon_cube.new_zeros(h, w, 0, dtype=torch.float32)

        y_fast = self._fast_temporal_basis(photon_cube)
        y_slow = self._slow_temporal_basis(photon_cube)
        y_fast_last = y_fast[..., -1].reshape(h, w)

        k_raw_flat = y_fast * torch.log(y_fast / y_slow) + (1.0 - y_fast) * torch.log((1.0 - y_fast) / (1.0 - y_slow))
        k_raw_hwt = k_raw_flat.reshape(h, w, t)
        k_smoothed_hwt = self._smooth_kl(k_raw_hwt)

        p_motion = torch.sigmoid(self.motion_sharpness * (k_smoothed_hwt - self.motion_threshold))
        future_motion = torch.flip(torch.flip(p_motion, dims=(-1,)).cummax(dim=-1).values, dims=(-1,))
        valid_weight = 1.0 - future_motion
        stable_support = valid_weight.sum(dim=-1)
        mean_stable = self._mean_stable_tail_aligned(photon_cube, valid_weight)
        w_mean = stable_support / (stable_support + max(self.stable_prior, self.eps))
        fused_last = w_mean * mean_stable + (1.0 - w_mean) * y_fast_last
        fused = fused_last.unsqueeze(-1)

        max_photon_hist = max(self.fast_window, self.slow_window) - 1
        max_kl_hist = self.temporal_window - 1
        self.photon_history = photon_cube.float()[..., -max_photon_hist:].detach() if max_photon_hist > 0 else None
        self.kl_history = k_raw_hwt[..., -max_kl_hist:].detach() if max_kl_hist > 0 else None
        return fused

    @torch.no_grad()
    def _integrate_last_with_debug(self, photon_cube: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        h, w, t = map(int, photon_cube.shape)
        if t == 0:
            empty = photon_cube.new_zeros(h, w, 0, dtype=torch.float32)
            debug = {
                "y_fast": empty,
                "y_slow": empty,
                "k_raw": empty,
                "k_smoothed": empty,
                "route_weight": empty,
                "route_weight_raw": empty,
                "p_motion": empty,
                "p_motion_raw": empty,
                "future_motion": empty,
                "valid_weight": empty,
                "fused": empty,
                "fused_blocks": empty,
                "motion_blocks": empty,
                "doe_blocks": empty,
                "weight_gamma_blocks": empty,
                "fused_mean": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "fused_last": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "motion_peak": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "motion_blend": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "y_scales_last": photon_cube.new_zeros(h, w, 2, dtype=torch.float32),
                "scores_last": photon_cube.new_zeros(h, w, 2, dtype=torch.float32),
                "k_smoothed_last": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "route_weight_last": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "route_weight_raw_last": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "p_motion_last": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "future_motion_last": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "valid_weight_last": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "mean_stable": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "stable_support": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "w_mean": photon_cube.new_zeros(h, w, dtype=torch.float32),
                "recons_prenorm": empty,
            }
            return empty, debug

        y_fast = self._fast_temporal_basis(photon_cube)
        y_slow = self._slow_temporal_basis(photon_cube)
        y_fast_last = y_fast[..., -1].reshape(h, w)
        y_slow_last = y_slow[..., -1].reshape(h, w)

        k_raw_flat = y_fast * torch.log(y_fast / y_slow) + (1.0 - y_fast) * torch.log((1.0 - y_fast) / (1.0 - y_slow))
        k_raw_hwt = k_raw_flat.reshape(h, w, t)
        k_smoothed_hwt = self._smooth_kl(k_raw_hwt)

        p_motion = torch.sigmoid(self.motion_sharpness * (k_smoothed_hwt - self.motion_threshold))
        p_motion_raw = p_motion
        future_motion = torch.flip(torch.flip(p_motion, dims=(-1,)).cummax(dim=-1).values, dims=(-1,))
        valid_weight = 1.0 - future_motion
        stable_support = valid_weight.sum(dim=-1)
        mean_stable = self._mean_stable_tail_aligned(photon_cube, valid_weight)
        w_mean = stable_support / (stable_support + max(self.stable_prior, self.eps))
        fused_last = w_mean * mean_stable + (1.0 - w_mean) * y_fast_last
        fused = fused_last.unsqueeze(-1)

        max_photon_hist = max(self.fast_window, self.slow_window) - 1
        max_kl_hist = self.temporal_window - 1
        self.photon_history = photon_cube.float()[..., -max_photon_hist:].detach() if max_photon_hist > 0 else None
        self.kl_history = k_raw_hwt[..., -max_kl_hist:].detach() if max_kl_hist > 0 else None

        debug = {
            "y_fast": y_fast_last.unsqueeze(-1),
            "y_slow": y_slow_last.unsqueeze(-1),
            "k_raw": k_raw_hwt[..., -1:],
            "k_smoothed": k_smoothed_hwt,
            "route_weight": future_motion,
            "route_weight_raw": p_motion_raw,
            "p_motion": p_motion,
            "p_motion_raw": p_motion_raw,
            "future_motion": future_motion,
            "valid_weight": valid_weight,
            "fused": fused,
            "fused_blocks": fused,
            "motion_blocks": future_motion,
            "doe_blocks": k_smoothed_hwt,
            "weight_gamma_blocks": future_motion,
            "fused_mean": fused_last,
            "fused_last": fused_last,
            "motion_peak": future_motion.max(dim=-1).values,
            "motion_blend": future_motion[..., -1],
            "y_scales_last": torch.stack([y_fast_last, y_slow_last], dim=-1),
            "scores_last": torch.stack([k_smoothed_hwt[..., -1], future_motion[..., -1]], dim=-1),
            "k_smoothed_last": k_smoothed_hwt[..., -1],
            "route_weight_last": future_motion[..., -1],
            "route_weight_raw_last": p_motion_raw[..., -1],
            "p_motion_last": p_motion[..., -1],
            "future_motion_last": future_motion[..., -1],
            "valid_weight_last": valid_weight[..., -1],
            "mean_stable": mean_stable,
            "stable_support": stable_support,
            "w_mean": w_mean,
            "recons_prenorm": fused,
        }
        return fused, debug

    @torch.no_grad()
    def process_photon_cube(
        self,
        photon_cube: Tensor,
        subsampling: int | None = None,
        hot_pixel_mask: np.ndarray | None = None,
        quantile: float | None = None,
        normalize: bool | None = None,
        clear_states: bool = True,
        chunk_size: int | None = None,
        fast_window: int | None = None,
        slow_window: int | None = None,
        temporal_window: int | None = None,
        fast_tau: float | None = None,
        motion_sharpness: float | None = None,
        motion_threshold: float | None = None,
        eps: float | None = None,
        stable_prior: float | None = None,
        source_space: str | None = None,
        warp_block_size: int | None = None,
        **kwargs,
    ) -> Tensor:
        if clear_states:
            self.reset()

        self.update_hyperparams(
            subsampling=subsampling,
            hot_pixel_mask=hot_pixel_mask,
            normalize=normalize,
            quantile=quantile,
            chunk_size=chunk_size,
            fast_window=fast_window,
            slow_window=slow_window,
            temporal_window=temporal_window,
            fast_tau=fast_tau,
            motion_sharpness=motion_sharpness,
            motion_threshold=motion_threshold,
            eps=eps,
            stable_prior=stable_prior,
            source_space=source_space,
            warp_block_size=warp_block_size,
            **kwargs,
        )

        self.set_cube(photon_cube)
        fused = self._integrate_last(photon_cube)
        recons = self._subsample_reconstruction(fused)
        if self.hot_pixel_mask is not None:
            recons = nearest_neighbor_inpaint(recons, self.hot_pixel_mask)
        recons = self.clamp_recons(recons)
        self.t_absolute += self._t
        return recons
