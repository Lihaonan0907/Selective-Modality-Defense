"""Utilities for four-channel proposal preprocessing and mask creation."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def clip_box_xyxy(box: Iterable[float], width: int, height: int) -> list[int] | None:
    """Clip an xyxy box to image bounds."""
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(1, min(width, round(x2))))
    y2 = int(max(1, min(height, round(y2))))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return [x1, y1, x2, y2]


def boxes_to_mask(
    shape_hw: tuple[int, int],
    boxes: Iterable[Iterable[float]],
    dilation_kernel: int = 0,
    min_area: int = 0,
) -> np.ndarray:
    """Convert proposal boxes to a binary uint8 mask."""
    height, width = shape_hw
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        clipped = clip_box_xyxy(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        mask[y1:y2, x1:x2] = 255

    if dilation_kernel and dilation_kernel > 1 and cv2.countNonZero(mask) > 0:
        kernel = np.ones((int(dilation_kernel), int(dilation_kernel)), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    if min_area > 0:
        mask = remove_small_components(mask, min_area)
    return mask


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than `min_area` pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == label_id] = 255
    return out

