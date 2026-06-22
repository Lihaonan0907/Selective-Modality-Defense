"""Attribution-guided asymmetric mask generation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.proposal.proposal_utils import boxes_to_mask, remove_small_components


def decide_attribution(
    p_vis: float,
    p_ir: float,
    tau_vis: float = 0.90,
    tau_ir: float = 0.65,
    dominance_margin: float = 0.10,
) -> str:
    """Return Clean, VIS, IR, or Both after dominance correction."""
    pv, pi = float(p_vis), float(p_ir)
    if pv - pi > dominance_margin:
        pi = 0.0
    elif pi - pv > dominance_margin:
        pv = 0.0
    if pv < tau_vis and pi < tau_ir:
        return "Clean"
    if pv >= tau_vis and pi < tau_ir:
        return "VIS"
    if pv < tau_vis and pi >= tau_ir:
        return "IR"
    return "Both"


class AsymmetricMaskGenerator:
    """Generate modality-specific restoration masks from proposal attribution."""

    def __init__(self, cfg: dict[str, Any]):
        self.tau_vis = float(cfg.get("tau_vis", cfg.get("attribution", {}).get("tau_vis", 0.9)))
        self.tau_ir = float(cfg.get("tau_ir", cfg.get("attribution", {}).get("tau_ir", 0.65)))
        self.dominance_margin = float(cfg.get("dominance_margin", 0.10))
        self.dilation_kernel = int(cfg.get("dilation_kernel", cfg.get("mask_dilate", 5)))
        self.min_area = int(cfg.get("min_area", cfg.get("min_mask_area", 80)))

    def _apply_dominance_correction(self, p_vis: float, p_ir: float) -> tuple[bool, bool]:
        """Suppress the weaker branch when one modality is clearly dominant."""
        pred = decide_attribution(p_vis, p_ir, self.tau_vis, self.tau_ir, self.dominance_margin)
        use_vis = pred in {"VIS", "Both"}
        use_ir = pred in {"IR", "Both"}
        return use_vis, use_ir

    def generate(
        self,
        proposals: list[dict[str, Any]],
        p_vis: list[float] | np.ndarray,
        p_ir: list[float] | np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate `M_vis` and `M_ir` from proposal-level probabilities."""
        vis_boxes: list[list[float]] = []
        ir_boxes: list[list[float]] = []
        for proposal, pv, pi in zip(proposals, p_vis, p_ir):
            box = proposal.get("box")
            if box is None:
                continue
            select_vis, select_ir = self._apply_dominance_correction(float(pv), float(pi))
            if select_vis:
                vis_boxes.append(box)
            if select_ir:
                ir_boxes.append(box)

        vis_mask = boxes_to_mask(image_shape, vis_boxes, self.dilation_kernel, self.min_area)
        ir_mask = boxes_to_mask(image_shape, ir_boxes, self.dilation_kernel, self.min_area)
        return vis_mask, ir_mask

    def postprocess(self, mask: np.ndarray) -> np.ndarray:
        """Apply dilation and small component removal to an existing mask."""
        if mask is None or cv2.countNonZero(mask) == 0:
            return np.zeros_like(mask, dtype=np.uint8)
        out = (mask > 0).astype(np.uint8) * 255
        if self.dilation_kernel > 1:
            out = cv2.dilate(out, np.ones((self.dilation_kernel, self.dilation_kernel), np.uint8), iterations=1)
        if self.min_area > 0:
            out = remove_small_components(out, self.min_area)
        return out
