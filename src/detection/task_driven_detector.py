"""Task-driven single-modal robust detector."""

from __future__ import annotations

from typing import Any

import numpy as np

from .detector_api import Detection
from .yolo_detector import YOLODetector


class TaskDrivenDetector:
    """Visible or infrared detector trained on clean/restored mixed domains."""

    def __init__(self, cfg: dict[str, Any], modality: str):
        if modality not in {"vis", "ir"}:
            raise ValueError("modality must be 'vis' or 'ir'")
        self.modality = modality
        self.detector = YOLODetector(cfg)

    @property
    def model(self):
        """Expose the underlying YOLO object for legacy adapters."""
        return self.detector.model

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Return standardized detection results."""
        return self.detector.predict(image)

    def score_in_box(self, image: np.ndarray, box: list[float]) -> float:
        """Return local pedestrian confidence around a proposal box."""
        return self.detector.score_in_box(image, box)

