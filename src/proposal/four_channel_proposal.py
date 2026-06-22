"""Four-channel anomaly proposal.

This module mirrors the proposal stage used in the original end-to-end script:
RGB visible image + CLAHE-enhanced infrared image are concatenated into a
four-channel tensor, letterboxed to the detector input size, and decoded with
YOLO NMS. The returned boxes are padded before mask generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:  # pragma: no cover - compatibility with older ultralytics
    from ultralytics.utils.ops import non_max_suppression

from src.proposal.proposal_utils import boxes_to_mask
from src.utils.common import require_path


class FourChannelProposal:
    """Four-channel anomaly proposal detector."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        self.input_size = int(cfg.get("input_size", 960))
        self.conf_thres = float(cfg.get("conf_thres", 0.01))
        self.iou_thres = float(cfg.get("iou_thres", 0.45))
        self.max_det = int(cfg.get("max_det", 30))
        self.padding = int(cfg.get("padding", 25))
        self.clahe_clip_limit = float(cfg.get("clahe_clip_limit", 2.0))
        self.clahe_tile_grid_size = tuple(cfg.get("clahe_tile_grid_size", [8, 8]))

        ckpt = require_path(cfg.get("checkpoint"), "proposal checkpoint")
        self.model = YOLO(str(ckpt))
        self.model.model.to(self.device).eval()

    def build_input(self, vis_img: np.ndarray, ir_img: np.ndarray) -> np.ndarray:
        """Build visible RGB + CLAHE infrared four-channel input."""
        if ir_img.ndim == 3:
            ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
        im_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )
        im_ir = clahe.apply(ir_img)
        return np.concatenate([im_rgb, im_ir[..., None]], axis=-1)

    def _letterbox(self, im_4ch: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        height, width = im_4ch.shape[:2]
        size = self.input_size
        scale = min(size / height, size / width)
        new_unpad = int(round(width * scale)), int(round(height * scale))
        dw, dh = (size - new_unpad[0]) / 2, (size - new_unpad[1]) / 2
        if (width, height) != new_unpad:
            resized = cv2.resize(im_4ch, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = im_4ch
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114, 0),
        )
        return padded, (height, width)

    @torch.no_grad()
    def predict(self, vis_img: np.ndarray, ir_img: np.ndarray) -> list[dict[str, Any]]:
        """Return proposal boxes and scores for a visible-infrared pair."""
        im_4ch = self.build_input(vis_img, ir_img)
        padded, original_shape = self._letterbox(im_4ch)
        tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        preds = non_max_suppression(
            self.model.model(tensor),
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=self.max_det,
        )

        height, width = original_shape
        proposals: list[dict[str, Any]] = []
        if len(preds[0]) == 0:
            return proposals

        boxes = scale_boxes((self.input_size, self.input_size), preds[0][:, :4], (height, width)).round().cpu().numpy()
        scores = preds[0][:, 4].detach().cpu().numpy()
        for box, score in zip(boxes.astype(int), scores):
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(width, x2 + self.padding)
            y2 = min(height, y2 + self.padding)
            proposals.append({"box": [x1, y1, x2, y2], "score": float(score)})
        return proposals

    def proposal_mask(self, proposals: list[dict[str, Any]], image_shape: tuple[int, int]) -> np.ndarray:
        """Build a binary proposal mask from predicted boxes."""
        return boxes_to_mask(
            image_shape,
            [item["box"] for item in proposals],
            dilation_kernel=int(self.cfg.get("dilation_kernel", 0)),
            min_area=int(self.cfg.get("min_area", 0)),
        )

