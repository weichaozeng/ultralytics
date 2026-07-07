# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Pose-aware non-maximum suppression for hand pose detection."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import xywh2xyxy

BONE_CONNECTIONS = torch.tensor(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ],
    dtype=torch.int64,
)

POSE_TRACKERS = frozenset({"posetrack", "spad_posetrack"})


def is_pose_track_tracker(tracker: str | None) -> bool:
    """Return True when tracker config should enable pose-aware NMS."""
    if not tracker:
        return False
    name = Path(str(tracker)).stem.lower()
    return name in POSE_TRACKERS or any(token in name for token in POSE_TRACKERS)


def pose_aware_non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
    point_thres: float = 0.25,
    bone_thres: float = 0.25,
    ioa_thres: float = 0.65,
):
    """NMS with pose similarity for overlapping hand detections."""
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    extra = prediction.shape[1] - nc - 4
    if extra < 3:
        from ultralytics.utils import nms

        return nms.non_max_suppression(
            prediction,
            conf_thres,
            iou_thres,
            classes,
            agnostic,
            multi_label,
            labels,
            max_det,
            nc,
            max_time_img,
            max_nms,
            max_wh,
            rotated,
            end2end,
            return_idxs,
        )

    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]

    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    prediction = prediction.transpose(-1, -2)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):
        filt = xc[xi]
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        box, cls, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]

        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            filt = x[:, 4].argsort(descending=True)[:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = x[:, :4]
            pose_data = x[:, 6 : 6 + extra].reshape(x.shape[0], -1, 3)
            poses = pose_data[..., :2].reshape(x.shape[0], -1) + c
            pose_scores = pose_data[..., 2].reshape(x.shape[0], -1)
            det_classes = x[:, 5]
            i = PoseNMS.soft_pa_nms(
                boxes,
                scores,
                iou_thres,
                poses,
                pose_scores,
                point_thres,
                bone_thres,
                classes=det_classes,
                ioa_threshold=ioa_thres,
            )
        i = i[:max_det]

        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break

    return (output, keepi) if return_idxs else output


class PoseNMS:
    """Pose-aware NMS helpers."""

    @staticmethod
    def soft_pa_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        poses: torch.Tensor,
        pose_scores: torch.Tensor,
        point_threshold: float,
        bone_threshold: float,
        *,
        classes: torch.Tensor | None = None,
        ioa_threshold: float = 0.65,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        bone_idx = BONE_CONNECTIONS.to(device=boxes.device)
        x1, y1, x2, y2 = boxes.unbind(1)
        widths = (x2 - x1).clamp(min=1e-6)
        heights = (y2 - y1).clamp(min=1e-6)
        box_scales = torch.sqrt(widths**2 + heights**2).view(-1, 1, 1)
        areas = widths * heights

        n = poses.shape[0]
        kps = poses.reshape(n, -1, 2)
        parent_idx = bone_idx[:, 0]
        child_idx = bone_idx[:, 1]
        node_valid = pose_scores > 0.5
        valid_nums = node_valid.sum(dim=1)

        rel_vecs = (kps[:, child_idx, :] - kps[:, parent_idx, :]) / box_scales.clamp(min=1e-6)
        bone_vecs_norm = F.normalize(rel_vecs, p=2, dim=2)

        pair_weights = torch.minimum(pose_scores[:, parent_idx], pose_scores[:, child_idx])
        pair_weights = pair_weights.clone()
        pair_weights[pair_weights < 0.3] = 0.0

        order = scores.argsort(0, descending=True)
        keep = torch.zeros(n, dtype=torch.int64, device=boxes.device)
        keep_idx = 0

        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1
            if order.numel() == 1:
                break

            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)
            inter_over_min = inter / torch.minimum(areas[i], areas[rest]).clamp(min=1e-6)
            contain_mask = inter_over_min > ioa_threshold

            candidate_mask = (iou > iou_threshold) | contain_mask
            if not candidate_mask.any():
                order = rest
                continue

            if classes is not None:
                same_cls = classes[i] == classes[rest]
            else:
                same_cls = torch.ones_like(candidate_mask, dtype=torch.bool, device=boxes.device)

            pose_reliable_mask = (valid_nums[i] >= 10) & (valid_nums[rest] >= 10)
            final_mask = torch.zeros_like(candidate_mask)

            active_pose_mask = candidate_mask & pose_reliable_mask
            if active_pose_mask.any():
                target_indices = rest[active_pose_mask]
                diff_rel = rel_vecs[i : i + 1] - rel_vecs[target_indices]
                point_norm = torch.norm(diff_rel, dim=2)
                point_dissim = 1.0 - torch.exp(-5.0 * (point_norm**2))

                cos_sim = torch.sum(bone_vecs_norm[i : i + 1] * bone_vecs_norm[target_indices], dim=2)
                bone_dissim = 1.0 - (cos_sim + 1.0) / 2.0

                w_i = pair_weights[i : i + 1]
                w_target = pair_weights[target_indices]
                combined_weights = torch.minimum(w_i, w_target).clamp(min=1e-6)

                m_point_d = torch.sum(point_dissim * combined_weights, dim=1) / torch.sum(combined_weights, dim=1)
                m_bone_d = torch.sum(bone_dissim * combined_weights, dim=1) / torch.sum(combined_weights, dim=1)

                suppress = (m_point_d < point_threshold) & (m_bone_d < bone_threshold)
                final_mask[active_pose_mask] = suppress

            # Nested duplicate dets: same class, pose unreliable or containment without pose check.
            fallback_mask = candidate_mask & (~pose_reliable_mask) & same_cls
            dup_by_contain = fallback_mask & contain_mask
            dup_by_iou = fallback_mask & (~contain_mask) & (iou > iou_threshold)
            if dup_by_contain.any():
                final_mask[dup_by_contain] = True
            if dup_by_iou.any():
                final_mask[dup_by_iou] = True

            order = rest[~final_mask]

        return keep[:keep_idx]
