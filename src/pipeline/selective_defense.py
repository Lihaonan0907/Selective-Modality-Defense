"""Full selective modality defense pipeline."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from src.attribution import DRFMA, AsymmetricMaskGenerator
from src.detection import TaskDrivenDetector
from src.proposal import FourChannelProposal
from src.restoration import InfraredRestorer, VisibleRestorer


def _mask_nonempty(mask: np.ndarray | None) -> bool:
    return mask is not None and cv2.countNonZero(mask) > 0


class SelectiveModalityDefense:
    """Full visible-infrared selective defense pipeline."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        device = cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
        paths = cfg.get("paths", {})

        proposal_cfg = dict(cfg.get("proposal", {}))
        proposal_cfg.update({"checkpoint": paths.get("checkpoints", {}).get("proposal"), "device": device})
        self.proposal = FourChannelProposal(proposal_cfg)

        attribution_cfg = dict(cfg.get("attribution", {}))
        attribution_cfg.update(
            {
                "legacy_code_root": paths.get("legacy_code_root"),
                "prototype_path": paths.get("checkpoints", {}).get("prototypes"),
                "learned_fusion_ckpt": paths.get("checkpoints", {}).get("drf_ma"),
                "device": device,
            }
        )
        self.attributor = DRFMA(attribution_cfg)
        attr_device = torch.device(device if torch.cuda.is_available() or str(device).startswith("cpu") else "cpu")
        self.attributor.to(attr_device).eval()

        mask_cfg = dict(cfg.get("mask", {}))
        mask_cfg.update(cfg.get("attribution", {}))
        self.mask_generator = AsymmetricMaskGenerator(mask_cfg)

        restoration_cfg = dict(cfg.get("restoration", {}))
        restoration_cfg.update({"stable_diffusion_inpaint": paths.get("models", {}).get("stable_diffusion_inpaint"), "device": device})
        vis_restoration_cfg = dict(restoration_cfg)
        vis_restoration_cfg["checkpoint"] = paths.get("checkpoints", {}).get("vis_restorer")
        ir_restoration_cfg = dict(restoration_cfg)
        ir_restoration_cfg["checkpoint"] = paths.get("checkpoints", {}).get("ir_restorer")
        self.vis_restorer = VisibleRestorer(vis_restoration_cfg)
        self.ir_restorer = InfraredRestorer(ir_restoration_cfg)

        detector_base = dict(cfg.get("detector", {}))
        detector_base["device"] = device
        vis_detector_cfg = dict(detector_base)
        vis_detector_cfg["checkpoint"] = paths.get("checkpoints", {}).get("vis_detector")
        ir_detector_cfg = dict(detector_base)
        ir_detector_cfg["checkpoint"] = paths.get("checkpoints", {}).get("ir_detector")
        self.vis_detector = TaskDrivenDetector(vis_detector_cfg, "vis")
        self.ir_detector = TaskDrivenDetector(ir_detector_cfg, "ir")

    def defend_pair(self, vis_img: np.ndarray, ir_img: np.ndarray) -> dict[str, Any]:
        """Run the full paired visible-infrared defense."""
        if ir_img.ndim == 3:
            ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
        else:
            ir_gray = ir_img

        h_vis, w_vis = vis_img.shape[:2]
        if ir_gray.shape[:2] != (h_vis, w_vis):
            ir_for_proposal = cv2.resize(ir_gray, (w_vis, h_vis), interpolation=cv2.INTER_LINEAR)
        else:
            ir_for_proposal = ir_gray

        proposals = self.proposal.predict(vis_img, ir_for_proposal)
        if proposals:
            det_vis = self.vis_detector.model if self.attributor.legacy_adapter is not None else self.vis_detector
            det_ir = self.ir_detector.model if self.attributor.legacy_adapter is not None else self.ir_detector
            vis_mask, ir_mask, attribution_records = self.attributor.build_masks(
                proposals=proposals,
                vis_img=vis_img,
                ir_img=ir_for_proposal,
                det_vis=det_vis,
                det_ir=det_ir,
            )
        else:
            vis_mask = np.zeros((h_vis, w_vis), dtype=np.uint8)
            ir_mask = np.zeros((h_vis, w_vis), dtype=np.uint8)
            attribution_records = []

        vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        restored_vis_rgb = self.vis_restorer.restore(vis_rgb, vis_mask) if _mask_nonempty(vis_mask) else vis_rgb.copy()
        restored_vis_bgr = cv2.cvtColor(restored_vis_rgb, cv2.COLOR_RGB2BGR)

        ir_rgb = cv2.cvtColor(ir_for_proposal, cv2.COLOR_GRAY2RGB)
        restored_ir_rgb = self.ir_restorer.restore(ir_rgb, ir_mask) if _mask_nonempty(ir_mask) else ir_rgb.copy()
        restored_ir_gray = cv2.cvtColor(restored_ir_rgb, cv2.COLOR_RGB2GRAY)
        if restored_ir_gray.shape[:2] != ir_gray.shape[:2]:
            restored_ir_gray = cv2.resize(restored_ir_gray, (ir_gray.shape[1], ir_gray.shape[0]), interpolation=cv2.INTER_LINEAR)

        vis_detections = self.vis_detector.predict(restored_vis_bgr)
        ir_detections = self.ir_detector.predict(restored_ir_gray)

        return {
            "restored_visible": restored_vis_bgr,
            "restored_infrared": restored_ir_gray,
            "visible_mask": vis_mask,
            "infrared_mask": ir_mask,
            "proposals": proposals,
            "attribution": attribution_records,
            "visible_detections": vis_detections,
            "infrared_detections": ir_detections,
        }

    def defend_single(self, image: np.ndarray, modality: str) -> dict[str, Any]:
        """Run the single-branch variant for visible-only or infrared-only input."""
        from src.pipeline.single_branch_variant import SingleBranchDefense

        return SingleBranchDefense.from_config(self.cfg, modality).defend(image)
