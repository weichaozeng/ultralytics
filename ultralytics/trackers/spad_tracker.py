from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack


class SPADTracker(BYTETracker):
    """ByteTrack-style tracker that also renders a dense Gaussian velocity field."""

    def __init__(self, args: Any, frame_rate: int = 30):
        super().__init__(args=args, frame_rate=frame_rate)
        self.field_value_source = str(getattr(args, "field_value_source", "predicted")).lower()
        self.field_sigma_scale_x = float(getattr(args, "field_sigma_scale_x", 0.35))
        self.field_sigma_scale_y = float(getattr(args, "field_sigma_scale_y", 0.35))
        self.field_min_sigma = float(getattr(args, "field_min_sigma", 2.0))
        self.field_min_weight = float(getattr(args, "field_min_weight", 1e-4))
        self.field_max_speed = float(getattr(args, "field_max_speed", 32.0))
        self.field_extent = float(getattr(args, "field_extent", 3.0))
        self.prediction_horizon = float(getattr(args, "prediction_horizon", 1.0))

        self.last_velocity_field: torch.Tensor | None = None
        self.last_velocity_field_vx: torch.Tensor | None = None
        self.last_velocity_field_vy: torch.Tensor | None = None
        self.last_track_motion: list[dict[str, float]] = []
        self._last_frame_hw: tuple[int, int] | None = None
        self._prev_centers_by_track: dict[int, np.ndarray] = {}

    def reset(self):
        super().reset()
        self.last_velocity_field = None
        self.last_velocity_field_vx = None
        self.last_velocity_field_vy = None
        self.last_track_motion = []
        self._last_frame_hw = None
        self._prev_centers_by_track = {}

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        tracks = super().update(results, img=img, feats=feats)
        frame_hw = tuple(img.shape[:2]) if img is not None else None
        self._last_frame_hw = frame_hw
        motion = self._collect_track_motion(frame_hw=frame_hw)
        self.last_track_motion = motion
        if frame_hw is None:
            self.last_velocity_field = None
            self.last_velocity_field_vx = None
            self.last_velocity_field_vy = None
        else:
            field = self._render_velocity_field(frame_hw=frame_hw, motion=motion)
            self.last_velocity_field = field
            self.last_velocity_field_vx = field[..., 0]
            self.last_velocity_field_vy = field[..., 1]
        return tracks

    def _collect_track_motion(self, frame_hw: tuple[int, int] | None) -> list[dict[str, float]]:
        motion: list[dict[str, float]] = []
        next_prev_centers: dict[int, np.ndarray] = {}
        if frame_hw is None:
            self._prev_centers_by_track = {}
            return motion

        h, w = map(int, frame_hw)
        for track in self.tracked_stracks:
            if not track.is_activated or track.state != TrackState.Tracked:
                continue
            state = self._track_state_to_motion(track=track, frame_hw=(h, w))
            if state is None:
                continue
            tid = int(track.track_id)
            prev_center = self._prev_centers_by_track.get(tid)
            center = np.array([state["cx"], state["cy"]], dtype=np.float32)
            if prev_center is None:
                measured_v = np.zeros(2, dtype=np.float32)
            else:
                measured_v = center - prev_center
            next_prev_centers[tid] = center
            state["measured_vx"] = float(measured_v[0])
            state["measured_vy"] = float(measured_v[1])
            motion.append(state)

        self._prev_centers_by_track = next_prev_centers
        return motion

    def _track_state_to_motion(self, track: STrack, frame_hw: tuple[int, int]) -> dict[str, float] | None:
        if track.mean is None:
            xywh = track.xywh.astype(np.float32)
            cx, cy, bw, bh = map(float, xywh)
            vx = vy = 0.0
        else:
            cx = float(track.mean[0])
            cy = float(track.mean[1])
            bh = max(float(track.mean[3]), 1.0)
            bw = max(float(track.mean[2] * track.mean[3]), 1.0)
            vx = float(track.mean[4])
            vy = float(track.mean[5])
        vx = float(np.clip(vx, -self.field_max_speed, self.field_max_speed))
        vy = float(np.clip(vy, -self.field_max_speed, self.field_max_speed))
        h, w = frame_hw
        if w <= 0 or h <= 0:
            return None
        return {
            "track_id": float(track.track_id),
            "cx": float(np.clip(cx, 0.0, max(w - 1, 0))),
            "cy": float(np.clip(cy, 0.0, max(h - 1, 0))),
            "w": float(np.clip(bw, 1.0, w)),
            "h": float(np.clip(bh, 1.0, h)),
            "predicted_vx": vx,
            "predicted_vy": vy,
            "score": float(track.score),
            "tracklet_len": float(track.tracklet_len),
        }

    def _render_velocity_field(self, frame_hw: tuple[int, int], motion: list[dict[str, float]]) -> torch.Tensor:
        h, w = map(int, frame_hw)
        if h <= 0 or w <= 0 or not motion:
            return torch.zeros((h, w, 2), dtype=torch.float32)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        numer = np.zeros((h, w, 2), dtype=np.float32)
        denom = np.zeros((h, w), dtype=np.float32)

        for item in motion:
            if self.field_value_source == "measured" and item["tracklet_len"] > 0:
                vx = item["measured_vx"]
                vy = item["measured_vy"]
                center_x = item["cx"]
                center_y = item["cy"]
            else:
                vx = item["predicted_vx"]
                vy = item["predicted_vy"]
                center_x = item["cx"] + self.prediction_horizon * vx
                center_y = item["cy"] + self.prediction_horizon * vy

            sigma_x = max(item["w"] * self.field_sigma_scale_x, self.field_min_sigma)
            sigma_y = max(item["h"] * self.field_sigma_scale_y, self.field_min_sigma)
            radius_x = max(int(np.ceil(self.field_extent * sigma_x)), 1)
            radius_y = max(int(np.ceil(self.field_extent * sigma_y)), 1)

            x0 = max(int(np.floor(center_x)) - radius_x, 0)
            x1 = min(int(np.ceil(center_x)) + radius_x + 1, w)
            y0 = max(int(np.floor(center_y)) - radius_y, 0)
            y1 = min(int(np.ceil(center_y)) + radius_y + 1, h)
            if x0 >= x1 or y0 >= y1:
                continue

            local_x = xx[y0:y1, x0:x1] - float(center_x)
            local_y = yy[y0:y1, x0:x1] - float(center_y)
            gaussian = np.exp(-0.5 * ((local_x / sigma_x) ** 2 + (local_y / sigma_y) ** 2)).astype(np.float32)
            weight = gaussian * max(float(item["score"]), 1e-3)
            numer[y0:y1, x0:x1, 0] += weight * float(vx)
            numer[y0:y1, x0:x1, 1] += weight * float(vy)
            denom[y0:y1, x0:x1] += weight

        field = np.zeros((h, w, 2), dtype=np.float32)
        valid = denom > self.field_min_weight
        if np.any(valid):
            field_x = field[..., 0]
            field_y = field[..., 1]
            numer_x = numer[..., 0]
            numer_y = numer[..., 1]
            field_x[valid] = numer_x[valid] / denom[valid]
            field_y[valid] = numer_y[valid] / denom[valid]
        return torch.from_numpy(field)
