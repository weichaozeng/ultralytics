# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
import time

import torch
import torch.nn.functional as F
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou, box_iou
from ultralytics.utils.ops import xywh2xyxy

BONE_CONNECTIONS = torch.tensor([
    [0, 1], [1, 2], [2, 3], [3, 4],    # Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],    # Index
    [0, 9], [9, 10], [10, 11], [11, 12], # Middle
    [0, 13], [13, 14], [14, 15], [15, 16], # Ring
    [0, 17], [17, 18], [18, 19], [19, 20]  # Pinky
], dtype=torch.int64)


point_thres = 0.25
bone_thres = 0.25

def pose_aware_non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """Perform non-maximum suppression (NMS) on prediction results.

    Applies NMS to filter overlapping bounding boxes based on confidence and IoU thresholds. Supports multiple detection
    formats including standard boxes, rotated boxes, and masks.

    Args:
        prediction (torch.Tensor): Predictions with shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing boxes, classes, and optional masks.
        conf_thres (float): Confidence threshold for filtering detections. Valid values are between 0.0 and 1.0.
        iou_thres (float): IoU threshold for NMS filtering. Valid values are between 0.0 and 1.0.
        classes (list[int], optional): List of class indices to consider. If None, all classes are considered.
        agnostic (bool): Whether to perform class-agnostic NMS.
        multi_label (bool): Whether each box can have multiple labels.
        labels (list[list[Union[int, float, torch.Tensor]]]): A priori labels for each image.
        max_det (int): Maximum number of detections to keep per image.
        nc (int): Number of classes. Indices after this are considered masks.
        max_time_img (float): Maximum time in seconds for processing one image.
        max_nms (int): Maximum number of boxes for NMS.
        max_wh (int): Maximum box width and height in pixels.
        rotated (bool): Whether to handle Oriented Bounding Boxes (OBB).
        end2end (bool): Whether the model is end-to-end and doesn't require NMS.
        return_idxs (bool): Whether to return the indices of kept detections.

    Returns:
        output (list[torch.Tensor]): List of detections per image with shape (num_boxes, 6 + num_masks) containing (x1,
            y1, x2, y2, confidence, class, mask1, mask2, ...).
        keepi (list[torch.Tensor]): Indices of kept detections if return_idxs=True.
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]  # to track idxs

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        filt = xc[xi]  # confidence
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]

        # Filter by class
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            # Speed strategy: torchvision for val or already loaded (faster), TorchNMS for predict (lower latency)
            # if "torchvision" in sys.modules:
            #     import torchvision  # scope as slow import

            #     i = torchvision.ops.nms(boxes, scores, iou_thres)
            # else:
            #     i = TorchNMS.nms(boxes, scores, iou_thres)
            
            # modify
            pose_data = x[:, 6:69].reshape(x.shape[0], -1, 3)
            poses = pose_data[..., :2].reshape(x.shape[0], -1) + c
            pose_scores = pose_data[..., 2].reshape(x.shape[0], -1)
            i = TorchNMS.soft_pa_nms(boxes, scores, iou_thres, poses, pose_scores, point_thres, bone_thres)
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return (output, keepi) if return_idxs else output


class TorchNMS:
    """Ultralytics custom NMS implementation optimized for YOLO.

    This class provides static methods for performing non-maximum suppression (NMS) operations on bounding boxes,
    including both standard NMS and batched NMS for multi-class scenarios.

    Methods:
        nms: Optimized NMS with early termination that matches torchvision behavior exactly.
        batched_nms: Batched NMS for class-aware suppression.

    Examples:
        Perform standard NMS on boxes and scores
        >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> scores = torch.tensor([0.9, 0.8])
        >>> keep = TorchNMS.nms(boxes, scores, 0.5)
    """

    @staticmethod
    def fast_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        use_triu: bool = True,
        iou_func=box_iou,
        exit_early: bool = True,
    ) -> torch.Tensor:
        """Fast-NMS implementation from https://arxiv.org/pdf/1904.02689 using upper triangular matrix operations.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            iou_threshold (float): IoU threshold for suppression.
            use_triu (bool): Whether to use torch.triu operator for upper triangular matrix operations.
            iou_func (callable): Function to compute IoU between boxes.
            exit_early (bool): Whether to exit early if there are no boxes.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply NMS to a set of boxes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> keep = TorchNMS.nms(boxes, scores, 0.5)
        """
        if boxes.numel() == 0 and exit_early:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = iou_func(boxes, boxes)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            # NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
            pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        else:
            n = boxes.shape[0]
            row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            # Zeroing these scores ensures the additional indices would not affect the final results
            scores_ = scores[sorted_idx]
            scores_[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            scores[sorted_idx] = scores_  # update original tensor for NMSModel
            # NOTE: return indices with fixed length to avoid TFLite reshape error
            pick = torch.topk(scores_, scores_.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Optimized NMS with early termination that matches torchvision behavior exactly.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            iou_threshold (float): IoU threshold for suppression.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply NMS to a set of boxes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> keep = TorchNMS.nms(boxes, scores, 0.5)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # Pre-allocate and extract coordinates once
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)

        # Sort by scores descending
        order = scores.argsort(0, descending=True)

        # Pre-allocate keep list with maximum possible size
        keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0
        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1

            if order.numel() == 1:
                break
            # Vectorized IoU calculation for remaining boxes
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            # Fast intersection and IoU
            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            # Early exit: skip IoU calculation if no intersection
            if inter.sum() == 0:
                # No overlaps with current box, keep all remaining boxes
                order = rest
                continue
            iou = inter / (areas[i] + areas[rest] - inter)
            # Keep boxes with IoU <= threshold
            order = rest[iou <= iou_threshold]

        return keep[:keep_idx]

    @staticmethod
    def batched_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        idxs: torch.Tensor,
        iou_threshold: float,
        use_fast_nms: bool = False,
    ) -> torch.Tensor:
        """Batched NMS for class-aware suppression.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            idxs (torch.Tensor): Class indices with shape (N,).
            iou_threshold (float): IoU threshold for suppression.
            use_fast_nms (bool): Whether to use the Fast-NMS implementation.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.

        Examples:
            Apply batched NMS across multiple classes
            >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
            >>> scores = torch.tensor([0.9, 0.8])
            >>> idxs = torch.tensor([0, 1])
            >>> keep = TorchNMS.batched_nms(boxes, scores, idxs, 0.5)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # Strategy: offset boxes by class index to prevent cross-class suppression
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        return (
            TorchNMS.fast_nms(boxes_for_nms, scores, iou_threshold)
            if use_fast_nms
            else TorchNMS.nms(boxes_for_nms, scores, iou_threshold)
        )

    @staticmethod
    def pa_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, poses: torch.Tensor, pose_scores: torch.Tensor, point_threshold: float, bone_threshold: float) -> torch.Tensor:
        """Optimized NMS with early termination that matches torchvision behavior exactly.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            iou_threshold (float): IoU threshold for suppression.
            poses (torch.Tensor): Keppoints with shape (N, 42) in xy format.
            pose_scores (torch.Tensor): Confidence scores with shape (N, 21)
            point_threshold (float): Threshold for point similarity for suppression.
            bone_threshold (float): Threshold for bone similarity for suppression.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.
        """

        
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)

        N = poses.shape[0]
        kps = poses.reshape(N, -1, 2)
        widths = (x2 - x1).clamp(min=1e-6).view(N, 1, 1)
        heights = (y2 - y1).clamp(min=1e-6).view(N, 1, 1)
        
        min_coords = boxes[:, :2].view(N, 1, 2)
        kps_tr = kps - min_coords
        scales = torch.cat([widths, heights], dim=2)
        kps_norm = kps_tr / scales # (N, K, 2)

        pose_scores[pose_scores < 0.5] = 0.0
        # point dist
        point_diff = kps_norm.unsqueeze(0) - kps_norm.unsqueeze(1) # (N. N, K, 2)
        conf_A = pose_scores.unsqueeze(1) # (N, 1, K)
        conf_B = pose_scores.unsqueeze(0) # (1, N, K)
        weights_min = torch.minimum(conf_A, conf_B).unsqueeze(-1) # (N, N, K, 1)
        point_dist_sq_weighted = torch.sum(point_diff ** 2 * weights_min, dim=3) # (N, N, K)
        total_weight = torch.sum(weights_min.squeeze(-1), dim=2).clamp(min=1e-6) # (N, N)
        point_distance_matrix = torch.sum(torch.sqrt(point_dist_sq_weighted), dim=2) / total_weight # (N, N)

        # bone dissim
        bone_starts = kps_norm[:, BONE_CONNECTIONS[:, 0], :] # (N, 20, 2)
        bone_ends = kps_norm[:, BONE_CONNECTIONS[:, 1], :]   # (N, 20, 2)
        bone_vecs = bone_ends - bone_starts     # (N, 20, 2)

        bone_vecs_norm = F.normalize(bone_vecs, p=2, dim=2) # (N, 20, 2)
        conf_start = pose_scores[:, BONE_CONNECTIONS[:, 0]] # (N, 20)
        conf_end = pose_scores[:, BONE_CONNECTIONS[:, 1]]   # (N, 20)
        bone_weights = torch.minimum(conf_start, conf_end) # (N, 20)

        bone_A = bone_vecs_norm.unsqueeze(1)  # (N, 1, 20, 2)
        bone_B = bone_vecs_norm.unsqueeze(0)  # (1, N, 20, 2)
        cos_sim = (torch.sum(bone_A * bone_B, dim=3) + 1.0) / 2.0  # (N, N, 20)
        weights_A = bone_weights.unsqueeze(1) # (N, 1, 20)
        weights_B = bone_weights.unsqueeze(0) # (1, N, 20)
        final_weights = torch.minimum(weights_A, weights_B) # (N, N, 20)
        weighted_dissim_sum = torch.sum((1 - cos_sim) * final_weights, dim=2) # (N, N)
        total_weight = torch.sum(final_weights, dim=2).clamp(min=1e-6) # (N, N)
        bone_dissimilarity_matrix = weighted_dissim_sum / total_weight # (N, N)


        order = scores.argsort(0, descending=True)
        keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0        
        while order.numel()>0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1

            if order.numel() == 1:
                break
            
            # area iou
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            if inter.sum() == 0:
                order = rest
                continue
            iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)  # （rN，）
            point_d = point_distance_matrix[i, rest]   #  (rN,)
            bone_d = bone_dissimilarity_matrix[i, rest]
            suppress_mask = (iou > iou_threshold) & (point_d < point_threshold) & (bone_d < bone_threshold)

            order = rest[~suppress_mask]

        return keep[:keep_idx]
    
    @staticmethod
    def soft_pa_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, poses: torch.Tensor, pose_scores: torch.Tensor, point_threshold: float, bone_threshold: float) -> torch.Tensor:
        """Optimized NMS with early termination that matches torchvision behavior exactly.

        Args:
            boxes (torch.Tensor): Bounding boxes with shape (N, 4) in xyxy format.
            scores (torch.Tensor): Confidence scores with shape (N,).
            iou_threshold (float): IoU threshold for suppression.
            poses (torch.Tensor): Keppoints with shape (N, 42) in xy format.
            pose_scores (torch.Tensor): Confidence scores with shape (N, 21)
            point_threshold (float): Threshold for point similarity for suppression.
            bone_threshold (float): Threshold for bone similarity for suppression.

        Returns:
            (torch.Tensor): Indices of boxes to keep after NMS.
        """

        
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        
        # bbox
        x1, y1, x2, y2 = boxes.unbind(1)
        widths = (x2 - x1).clamp(min=1e-6)
        heights = (y2 - y1).clamp(min=1e-6)
        box_scales = torch.sqrt(widths**2 + heights**2).view(-1, 1, 1) # (N, 1, 1)
        areas = (x2 - x1) * (y2 - y1)

        # pose
        N = poses.shape[0]
        kps = poses.reshape(N, 21, 2)  # (N, 21, 2)
        parent_idx = BONE_CONNECTIONS[:, 0] # (20,)
        child_idx = BONE_CONNECTIONS[:, 1]  # (20,)

        # P_child - P_parent
        # (N, 20, 2)
        rel_vecs = (kps[:, child_idx, :] - kps[:, parent_idx, :]) / box_scales.clamp(min=1e-6)
        bone_vecs_norm = F.normalize(rel_vecs, p=2, dim=2)
        
        pair_weights = torch.minimum(pose_scores[:, parent_idx], pose_scores[:, child_idx])
        pair_weights[pair_weights < 0.3] = 0.0    


        # loop        
        order = scores.argsort(0, descending=True)
        keep = torch.zeros(N, dtype=torch.int64, device=boxes.device)
        keep_idx = 0

        while order.numel()>0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1
            if order.numel() == 1:
                break

            rest = order[1:]

            # iou
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)

            # only pose compare for iou > threshold
            candidate_mask = iou > iou_threshold
            if not candidate_mask.any():
                order = rest
                continue
            
            target_indices = rest[candidate_mask]

            # point similarity
            # (M, 20, 2) -> (M, 20)
            diff_rel = rel_vecs[i:i+1] - rel_vecs[target_indices] 
            point_norm = torch.norm(diff_rel, dim=2)
            point_dissim = 1.0 - torch.exp(-5.0 * (point_norm**2))

            # bone consine similarity
            # (1, 20, 2) * (M, 20, 2) -> (M, 20)
            cos_sim = torch.sum(bone_vecs_norm[i:i+1] * bone_vecs_norm[target_indices], dim=2)
            bone_dissim = 1.0 - (cos_sim + 1.0) / 2.0

            # weighted
            w_i = pair_weights[i:i+1] # (1, 20)
            w_target = pair_weights[target_indices] # (M, 20)
            combined_weights = torch.minimum(w_i, w_target).clamp(min=1e-6) # (M, 20)

            m_point_d = torch.sum(point_dissim * combined_weights, dim=1) / torch.sum(combined_weights, dim=1)
            m_bone_d = torch.sum(bone_dissim * combined_weights, dim=1) / torch.sum(combined_weights, dim=1)

            suppress = (m_point_d < point_threshold) & (m_bone_d < bone_threshold)
            final_mask = candidate_mask.clone()
            final_mask[candidate_mask] = suppress

            order = rest[~final_mask]

        return keep[:keep_idx]




