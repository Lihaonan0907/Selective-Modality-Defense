"""Ultralytics YOLO detector wrapper."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.detection.detector_api import Detection, iou_xyxy
from src.utils.common import require_path


class YOLODetector:
    """YOLO detector with standardized output."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.checkpoint = require_path(cfg.get("checkpoint"), "detector checkpoint")
        self.model = YOLO(str(self.checkpoint))
        self.input_size = int(cfg.get("input_size", 640))
        self.conf_thres = float(cfg.get("conf_thres", 0.7))
        self.iou_thres = float(cfg.get("iou_thres", 0.5))
        self.target_class_id = int(cfg.get("target_class_id", 0))
        self.device = cfg.get("device")

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Return standardized detection results."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        results = self.model.predict(
            image,
            imgsz=self.input_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            classes=[self.target_class_id],
            verbose=False,
            device=self.device,
        )
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes.xyxy.detach().cpu().numpy()
        scores = results[0].boxes.conf.detach().cpu().numpy()
        classes = results[0].boxes.cls.detach().cpu().numpy()
        return [
            Detection([float(v) for v in box.tolist()], float(score), int(cls))
            for box, score, cls in zip(boxes, scores, classes)
        ]

    def score_in_box(self, image: np.ndarray, box: list[float]) -> float:
        """Return local pedestrian confidence around a proposal box."""
        detections = self.predict(image)
        if not detections:
            return 0.0
        return max((det.score for det in detections if iou_xyxy(det.box, box) > 0.05), default=0.0)

