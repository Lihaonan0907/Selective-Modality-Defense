"""Single-branch defense variant."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from src.attribution import SingleBranchAttribution, SingleBranchEvidenceExtractor, generate_single_branch_mask
from src.detection import TaskDrivenDetector
from src.proposal import SingleBranchProposal
from src.restoration import InfraredRestorer, VisibleRestorer


def _mask_nonempty(mask: np.ndarray | None) -> bool:
    return mask is not None and cv2.countNonZero(mask) > 0


class SingleBranchDefense:
    """Visible-only or infrared-only defense with cross-modal signals disabled."""

    def __init__(
        self,
        proposal: SingleBranchProposal,
        attribution: SingleBranchAttribution,
        restorer: Any,
        detector: TaskDrivenDetector,
        modality: str,
        cfg: dict[str, Any],
    ):
        if modality not in {"vis", "ir"}:
            raise ValueError("modality must be 'vis' or 'ir'")
        self.proposal = proposal
        self.attribution = attribution
        self.restorer = restorer
        self.detector = detector
        self.modality = modality
        self.cfg = cfg
        self.device = torch.device(cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.attribution.to(self.device).eval()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], modality: str) -> "SingleBranchDefense":
        """Create a single-branch pipeline without initializing paired modules."""
        if modality not in {"vis", "ir"}:
            raise ValueError("modality must be 'vis' or 'ir'")
        device = cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
        paths = cfg.get("paths", {})
        checkpoints = paths.get("checkpoints", {})
        single_ckpts = checkpoints.get("single_branch", {})
        branch_cfg = dict(cfg.get("single_branch", {}))

        proposal_cfg = dict(cfg.get("proposal", {}))
        proposal_cfg.update(branch_cfg.get("proposal", {}))
        proposal_cfg["device"] = device
        proposal_cfg["checkpoint"] = (
            single_ckpts.get(f"{modality}_proposal")
            or checkpoints.get(f"{modality}_proposal")
            or branch_cfg.get("proposal", {}).get("checkpoint")
        )
        if modality == "ir":
            proposal_cfg.setdefault("input_channels", 3)
        else:
            proposal_cfg["input_channels"] = 3
        proposal = SingleBranchProposal(proposal_cfg, modality)

        attr_cfg = dict(branch_cfg.get("attribution", {}))
        attr_cfg.setdefault("tau_vis", cfg.get("attribution", {}).get("tau_vis", 0.90))
        attr_cfg.setdefault("tau_ir", cfg.get("attribution", {}).get("tau_ir", 0.65))
        attribution = SingleBranchAttribution(attr_cfg, modality=modality)
        attr_ckpt = (
            single_ckpts.get(f"{modality}_attribution")
            or checkpoints.get(f"{modality}_attribution")
            or branch_cfg.get("attribution", {}).get("checkpoint")
        )
        if attr_ckpt:
            attribution.load_checkpoint(attr_ckpt, map_location="cpu")
        else:
            raise ValueError(
                f"Missing {modality} single-branch attribution checkpoint. "
                "Set paths.checkpoints.single_branch.<mode>_attribution in configs/paths.yaml."
            )

        restoration_cfg = dict(cfg.get("restoration", {}))
        restoration_cfg.update({"stable_diffusion_inpaint": paths.get("models", {}).get("stable_diffusion_inpaint"), "device": device})
        if modality == "vis":
            restoration_cfg["checkpoint"] = checkpoints.get("vis_restorer")
            restorer = VisibleRestorer(restoration_cfg)
        else:
            restoration_cfg["checkpoint"] = checkpoints.get("ir_restorer")
            restorer = InfraredRestorer(restoration_cfg)

        detector_cfg = dict(cfg.get("detector", {}))
        detector_cfg["device"] = device
        detector_cfg["checkpoint"] = checkpoints.get("vis_detector") if modality == "vis" else checkpoints.get("ir_detector")
        detector = TaskDrivenDetector(detector_cfg, modality)
        return cls(proposal, attribution, restorer, detector, modality, cfg)

    def _prepare_restoration_input(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.modality == "ir":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def defend(self, image: np.ndarray) -> dict[str, Any]:
        """Defend a single visible or infrared image."""
        original_for_detection, restoration_input = self._prepare_restoration_input(image)
        proposals = self.proposal.predict(original_for_detection)
        extractor = SingleBranchEvidenceExtractor(self.cfg.get("single_branch", {}).get("evidence", {}), detector=self.detector)
        records: list[dict[str, Any]] = []
        for idx, proposal in enumerate(proposals):
            evidence = extractor.extract(original_for_detection, proposal["box"])
            records.append(
                {
                    "proposal_index": idx,
                    "proposal_box": proposal["box"],
                    "proposal_score": float(proposal.get("score", 1.0)),
                    **evidence,
                }
            )

        probabilities = self.attribution.predict_records(records, self.device) if records else []
        mask_cfg = self.cfg.get("single_branch", {}).get("mask", self.cfg.get("mask", {}))
        tau = self.attribution.threshold
        mask = generate_single_branch_mask(
            proposals,
            probabilities,
            original_for_detection.shape[:2],
            tau=tau,
            dilation_kernel=int(mask_cfg.get("dilation_kernel", mask_cfg.get("mask_dilate", 5))),
            min_component_area=int(mask_cfg.get("min_component_area", mask_cfg.get("min_area", 80))),
        )

        if _mask_nonempty(mask):
            restored_rgb = self.restorer.restore(restoration_input, mask)
        else:
            restored_rgb = restoration_input.copy()

        if self.modality == "ir":
            restored = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2GRAY)
        else:
            restored = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        detections = self.detector.predict(restored)

        attribution_records = []
        for record, prob in zip(records, probabilities):
            attribution_records.append({**record, "probability": float(prob), "selected": bool(float(prob) >= tau), "tau": float(tau)})

        return {
            "restored": restored,
            "mask": mask,
            "proposals": proposals,
            "probabilities": probabilities,
            "attribution": attribution_records,
            "detections": detections,
        }
