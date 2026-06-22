"""Image IO utilities with explicit visible/infrared conventions."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import ensure_dir


def read_visible_bgr(path: str | Path) -> np.ndarray:
    """Read a visible image in OpenCV BGR format."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read visible image: {path}")
    return img


def read_infrared_gray(path: str | Path) -> np.ndarray:
    """Read an infrared image as a single-channel grayscale array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read infrared image: {path}")
    return img


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an image, creating parent directories if needed."""
    path = Path(path)
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")

