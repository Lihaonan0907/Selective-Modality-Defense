"""Inference helpers used by scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.pipeline import SelectiveModalityDefense
from src.utils.common import ensure_dir
from src.utils.image_io import read_infrared_gray, read_visible_bgr, save_image


def _serialize_detections(detections: list[Any]) -> list[dict[str, Any]]:
    return [
        {"box": det.box, "score": det.score, "class_id": det.class_id}
        for det in detections
    ]


def run_pair(cfg: dict[str, Any], visible_path: str | Path, infrared_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run paired visible-infrared inference and save outputs."""
    output_dir = ensure_dir(output_dir)
    pipeline = SelectiveModalityDefense(cfg)
    vis = read_visible_bgr(visible_path)
    ir = read_infrared_gray(infrared_path)
    result = pipeline.defend_pair(vis, ir)

    save_image(output_dir / "restored_visible.png", result["restored_visible"])
    save_image(output_dir / "restored_infrared.png", result["restored_infrared"])
    save_image(output_dir / "mask_visible.png", result["visible_mask"])
    save_image(output_dir / "mask_infrared.png", result["infrared_mask"])
    return {
        "proposals": result["proposals"],
        "attribution": result["attribution"],
        "visible_detections": _serialize_detections(result["visible_detections"]),
        "infrared_detections": _serialize_detections(result["infrared_detections"]),
    }


def run_single(cfg: dict[str, Any], image_path: str | Path, modality: str, output_dir: str | Path) -> dict[str, Any]:
    """Run single-branch inference and save outputs."""
    from src.pipeline.single_branch_variant import SingleBranchDefense

    output_dir = ensure_dir(output_dir)
    image = read_infrared_gray(image_path) if modality == "ir" else read_visible_bgr(image_path)
    pipeline = SingleBranchDefense.from_config(cfg, modality)
    result = pipeline.defend(image)
    save_image(output_dir / f"restored_{modality}.png", result["restored"])
    save_image(output_dir / f"mask_{modality}.png", result["mask"])
    return {
        "proposals": result["proposals"],
        "attribution": result["attribution"],
        "detections": _serialize_detections(result["detections"]),
    }
