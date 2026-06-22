"""Single-branch anomaly proposal detectors."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:  # pragma: no cover
    from ultralytics.utils.ops import non_max_suppression

from src.proposal.proposal_utils import boxes_to_mask
from src.utils.common import require_path


class SingleBranchProposal:
    """YOLO-based proposal detector for visible-only or infrared-only input."""

    def __init__(self, cfg: dict[str, Any], modality: str):
        if modality not in {"vis", "ir"}:
            raise ValueError("modality must be 'vis' or 'ir'")
        self.cfg = cfg
        self.modality = modality
        self.device = torch.device(cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        self.input_size = int(cfg.get("input_size", 960))
        self.conf_thres = float(cfg.get("conf_thres", 0.01))
        self.iou_thres = float(cfg.get("iou_thres", 0.45))
        self.max_det = int(cfg.get("max_det", 30))
        self.padding = int(cfg.get("padding", 25))
        self.input_channels = int(cfg.get("input_channels", 3))
        self.clahe_clip_limit = float(cfg.get("clahe_clip_limit", 2.0))
        self.clahe_tile_grid_size = tuple(cfg.get("clahe_tile_grid_size", [8, 8]))

        ckpt = require_path(cfg.get("checkpoint"), f"{modality} single-branch proposal checkpoint")
        self.model = YOLO(str(ckpt))
        self.model.model.to(self.device).eval()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Build the detector input for the configured single modality."""
        if self.modality == "vis":
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )
        enhanced = clahe.apply(gray)
        if self.input_channels == 1:
            return enhanced
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        height, width = image.shape[:2]
        size = self.input_size
        scale = min(size / height, size / width)
        new_unpad = int(round(width * scale)), int(round(height * scale))
        dw, dh = (size - new_unpad[0]) / 2, (size - new_unpad[1]) / 2
        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR) if (width, height) != new_unpad else image
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        border_value = 114 if resized.ndim == 2 else (114, 114, 114)
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value)
        return padded, (height, width)

    @torch.no_grad()
    def _predict_one_channel(self, image: np.ndarray) -> list[dict[str, Any]]:
        padded, original_shape = self._letterbox(image)
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0
        preds = non_max_suppression(
            self.model.model(tensor),
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=self.max_det,
        )
        return self._decode_predictions(preds, original_shape)

    def _decode_predictions(self, preds: list[torch.Tensor], original_shape: tuple[int, int]) -> list[dict[str, Any]]:
        height, width = original_shape
        proposals: list[dict[str, Any]] = []
        if not preds or len(preds[0]) == 0:
            return proposals
        boxes = scale_boxes((self.input_size, self.input_size), preds[0][:, :4], (height, width)).round().cpu().numpy()
        scores = preds[0][:, 4].detach().cpu().numpy()
        for box, score in zip(boxes.astype(int), scores):
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(width, x2 + self.padding)
            y2 = min(height, y2 + self.padding)
            proposals.append({"box": [x1, y1, x2, y2], "score": float(score), "modality": self.modality})
        return proposals

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Return proposal boxes and scores for a single image branch."""
        prepared = self.preprocess(image)
        if self.input_channels == 1:
            if prepared.ndim == 3:
                prepared = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
            return self._predict_one_channel(prepared)

        results = self.model.predict(
            prepared,
            imgsz=self.input_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            verbose=False,
            device=str(self.device),
        )
        if not results or results[0].boxes is None:
            return []
        h, w = prepared.shape[:2]
        proposals: list[dict[str, Any]] = []
        boxes = results[0].boxes.xyxy.detach().cpu().numpy()
        scores = results[0].boxes.conf.detach().cpu().numpy()
        for box, score in zip(boxes.astype(int), scores):
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(w, x2 + self.padding)
            y2 = min(h, y2 + self.padding)
            proposals.append({"box": [x1, y1, x2, y2], "score": float(score), "modality": self.modality})
        return proposals

    def proposal_mask(self, proposals: list[dict[str, Any]], image_shape: tuple[int, int]) -> np.ndarray:
        """Build a binary proposal mask from predicted boxes."""
        return boxes_to_mask(
            image_shape,
            [item["box"] for item in proposals],
            dilation_kernel=int(self.cfg.get("dilation_kernel", 0)),
            min_area=int(self.cfg.get("min_area", 0)),
        )
