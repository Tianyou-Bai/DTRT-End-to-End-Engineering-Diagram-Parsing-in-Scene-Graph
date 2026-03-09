"""Bounding-box utility operators used across RelTR models."""

from typing import Tuple

import torch
import torchvision
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = torchvision.ops.box_area(boxes1)
    area2 = torchvision.ops.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    return torchvision.ops.generalized_box_iou(boxes1, boxes2)


def masks_to_boxes(masks: Tensor) -> Tensor:
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=masks.dtype)

    h, w = masks.shape[-2:]
    y = torch.arange(h, device=masks.device)
    x = torch.arange(w, device=masks.device)

    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * x_grid
    y_mask = masks * y_grid

    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~masks.bool(), 1e8).flatten(1).min(-1)[0]
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~masks.bool(), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def elementwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = torchvision.ops.box_area(boxes1)
    area2 = torchvision.ops.box_area(boxes2)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def elementwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = elementwise_box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, 0] * wh[:, 1]
    return iou - (area - union) / area.clamp(min=1e-6)


def check_point_inside_box(points: Tensor, boxes: Tensor, eps: float = 1e-9) -> Tensor:
    x, y = [p.unsqueeze(-1) for p in points.unbind(-1)]
    x1, y1, x2, y2 = [coord.unsqueeze(0) for coord in boxes.unbind(-1)]
    l = x - x1
    t = y - y1
    r = x2 - x
    b = y2 - y
    ltrb = torch.stack([l, t, r, b], dim=-1)
    return ltrb.min(dim=-1).values > eps


def point_box_distance(points: Tensor, boxes: Tensor) -> Tensor:
    x1y1, x2y2 = torch.split(boxes, 2, dim=-1)
    lt = points - x1y1
    rb = x2y2 - points
    return torch.cat([lt, rb], dim=-1)


def point_distance_box(points: Tensor, distances: Tensor) -> Tensor:
    lt, rb = torch.split(distances, 2, dim=-1)
    x1y1 = points - lt
    x2y2 = points + rb
    return torch.cat([x1y1, x2y2], dim=-1)


def get_union_box(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)
    lt = torch.min(boxes1_xyxy[:, :2], boxes2_xyxy[:, :2])
    rb = torch.max(boxes1_xyxy[:, 2:], boxes2_xyxy[:, 2:])
    union_xyxy = torch.cat((lt, rb), dim=-1)
    return box_xyxy_to_cxcywh(union_xyxy)


def rescale_bboxes(out_bbox: Tensor, size: Tensor) -> Tensor:
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=b.dtype, device=b.device)
    return b * scale
