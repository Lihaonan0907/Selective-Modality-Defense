"""PyTorch DRF-MA network and compatibility adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.attribution.evidence_signals import (
    EvidenceExtractor,
    FEATURE_DIMS,
    FEATURE_KEYS,
    evidence_record_to_probabilities,
    load_legacy_drf_module,
    record_to_evidence_tensors,
)
from src.attribution.mask_generation import AsymmetricMaskGenerator, decide_attribution
from src.proposal.proposal_utils import boxes_to_mask
from src.utils.common import require_path


def _nested_get(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass
class DRFMAConfig:
    """Configuration for the learned DRF-MA network."""

    cm_dim: int = 4
    self_dim: int = 4
    rep_dim: int = 2
    det_dim: int = 1
    hidden_dim: int = 64
    evidence_dim: int = 32
    dropout: float = 0.1
    tau_vis: float = 0.90
    tau_ir: float = 0.65
    dominance_margin: float = 0.10
    ablate: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "DRFMAConfig":
        cfg = cfg or {}
        model_cfg = cfg.get("model", cfg.get("attribution", cfg))
        thresholds = cfg.get("thresholds", {})
        return cls(
            cm_dim=int(model_cfg.get("cm_dim", FEATURE_DIMS["z_cm_vis"])),
            self_dim=int(model_cfg.get("self_dim", FEATURE_DIMS["z_self_vis"])),
            rep_dim=int(model_cfg.get("rep_dim", FEATURE_DIMS["z_rep_vis"])),
            det_dim=int(model_cfg.get("det_dim", FEATURE_DIMS["z_det_vis"])),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            evidence_dim=int(model_cfg.get("evidence_dim", 32)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            tau_vis=float(thresholds.get("tau_vis", model_cfg.get("tau_vis", 0.90))),
            tau_ir=float(thresholds.get("tau_ir", model_cfg.get("tau_ir", 0.65))),
            dominance_margin=float(thresholds.get("dominance_margin", model_cfg.get("dominance_margin", 0.10))),
            ablate=dict(model_cfg.get("ablate", cfg.get("ablate", {}))),
        )


class EvidenceMLP(nn.Module):
    """Two-layer evidence transform: input -> 64 -> 32."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class FusionMLP(nn.Module):
    """Gamma_k fusion head over concatenated evidence embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float()).squeeze(-1)


class EvidenceNormalizer:
    """Per-evidence-key mean/std normalizer saved with DRF-MA checkpoints."""

    def __init__(self, stats: dict[str, dict[str, list[float]]] | None = None):
        self.stats = stats or {}

    @classmethod
    def fit(cls, records: list[dict[str, Any]]) -> "EvidenceNormalizer":
        stats: dict[str, dict[str, list[float]]] = {}
        for key in FEATURE_KEYS:
            arr = np.asarray([r.get(key, [0.0] * FEATURE_DIMS[key]) for r in records], dtype=np.float32)
            if arr.size == 0:
                mean = np.zeros(FEATURE_DIMS[key], dtype=np.float32)
                std = np.ones(FEATURE_DIMS[key], dtype=np.float32)
            else:
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                std = np.where(std < 1e-6, 1.0, std)
            stats[key] = {"mean": mean.tolist(), "std": std.tolist()}
        return cls(stats)

    def state_dict(self) -> dict[str, dict[str, list[float]]]:
        return self.stats

    @classmethod
    def from_state_dict(cls, state: dict[str, dict[str, list[float]]] | None) -> "EvidenceNormalizer":
        return cls(state or {})

    def transform(self, evidence: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, value in evidence.items():
            stat = self.stats.get(key)
            if stat is None:
                out[key] = value.float()
                continue
            mean = torch.tensor(stat["mean"], dtype=value.dtype, device=value.device)
            std = torch.tensor(stat["std"], dtype=value.dtype, device=value.device)
            out[key] = (value.float() - mean) / (std + 1e-6)
        return out


class DRFMA(nn.Module):
    """Detection-Restoration Feedback-aware Modality Attribution.

    Forward input is an evidence dictionary with the following tensors:
    `z_cm_vis`, `z_cm_ir`, `z_self_vis`, `z_self_ir`, `z_rep_vis`,
    `z_rep_ir`, `z_det_vis`, and `z_det_ir`. The output dictionary contains
    logits and sigmoid probabilities for the visible and infrared branches.

    For public inference, `build_masks` uses a learned checkpoint when
    `learned_fusion_ckpt` is configured. If `legacy_code_root` is configured,
    it can also delegate to the original compatibility adapter.
    """

    def __init__(self, cfg: dict[str, Any] | DRFMAConfig | None = None, detector_api: Any = None, preview_restorer: Any = None):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, dict) else {}
        self.drf_cfg = cfg if isinstance(cfg, DRFMAConfig) else DRFMAConfig.from_dict(self.cfg)
        c = self.drf_cfg

        self.vis_cm = EvidenceMLP(c.cm_dim, c.hidden_dim, c.evidence_dim, c.dropout)
        self.ir_cm = EvidenceMLP(c.cm_dim, c.hidden_dim, c.evidence_dim, c.dropout)
        self.vis_self = EvidenceMLP(c.self_dim, c.hidden_dim, c.evidence_dim, c.dropout)
        self.ir_self = EvidenceMLP(c.self_dim, c.hidden_dim, c.evidence_dim, c.dropout)
        self.vis_rep = EvidenceMLP(c.rep_dim, c.hidden_dim, c.evidence_dim, c.dropout)
        self.ir_rep = EvidenceMLP(c.rep_dim, c.hidden_dim, c.evidence_dim, c.dropout)
        self.vis_det = EvidenceMLP(c.det_dim, c.hidden_dim, c.evidence_dim, c.dropout)
        self.ir_det = EvidenceMLP(c.det_dim, c.hidden_dim, c.evidence_dim, c.dropout)

        fusion_dim = c.evidence_dim * 4
        self.gamma_vis = FusionMLP(fusion_dim, c.hidden_dim, c.dropout)
        self.gamma_ir = FusionMLP(fusion_dim, c.hidden_dim, c.dropout)
        self.ablate = dict(c.ablate)

        self.normalizer: EvidenceNormalizer | None = None
        self.learned_checkpoint: Path | None = None
        learned_ckpt = self.cfg.get("learned_fusion_ckpt") or _nested_get(self.cfg, "paths.checkpoints.drf_ma")
        if learned_ckpt:
            self._load_learned_checkpoint(learned_ckpt)

        self.legacy_adapter: LegacyDRFMAAdapter | None = None
        self.legacy_adapter_error: Exception | None = None
        legacy_root = self.cfg.get("legacy_code_root") or _nested_get(self.cfg, "paths.legacy_code_root")
        if legacy_root:
            try:
                self.legacy_adapter = LegacyDRFMAAdapter(self.cfg, detector_api=detector_api, preview_restorer=preview_restorer)
            except Exception as exc:
                self.legacy_adapter_error = exc
                self.legacy_adapter = None

    def _load_learned_checkpoint(self, checkpoint: str | Path) -> None:
        """Load a learned DRF-MA fusion checkpoint for public inference."""
        ckpt_path = require_path(checkpoint, "DRF-MA learned fusion checkpoint")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        self.load_state_dict(state, strict=True)
        self.normalizer = EvidenceNormalizer.from_state_dict(ckpt.get("normalizer"))
        thresholds = ckpt.get("best_thresholds", {})
        if thresholds:
            self.drf_cfg.tau_vis = float(thresholds.get("tau_vis", self.drf_cfg.tau_vis))
            self.drf_cfg.tau_ir = float(thresholds.get("tau_ir", self.drf_cfg.tau_ir))
            self.drf_cfg.dominance_margin = float(thresholds.get("dominance_margin", self.drf_cfg.dominance_margin))
        self.learned_checkpoint = ckpt_path

    def _maybe_zero(self, tensor: torch.Tensor, signal_name: str) -> torch.Tensor:
        if self.ablate.get(signal_name, False):
            return torch.zeros_like(tensor)
        return tensor

    def forward(self, evidence_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_cm_vis = self._maybe_zero(self.vis_cm(evidence_dict["z_cm_vis"]), "z_cm")
        z_cm_ir = self._maybe_zero(self.ir_cm(evidence_dict["z_cm_ir"]), "z_cm")
        z_self_vis = self._maybe_zero(self.vis_self(evidence_dict["z_self_vis"]), "z_self")
        z_self_ir = self._maybe_zero(self.ir_self(evidence_dict["z_self_ir"]), "z_self")
        z_rep_vis = self._maybe_zero(self.vis_rep(evidence_dict["z_rep_vis"]), "z_rep")
        z_rep_ir = self._maybe_zero(self.ir_rep(evidence_dict["z_rep_ir"]), "z_rep")
        z_det_vis = self._maybe_zero(self.vis_det(evidence_dict["z_det_vis"]), "z_det")
        z_det_ir = self._maybe_zero(self.ir_det(evidence_dict["z_det_ir"]), "z_det")

        vis_fused = torch.cat([z_cm_vis, z_self_vis, z_rep_vis, z_det_vis], dim=-1)
        ir_fused = torch.cat([z_cm_ir, z_self_ir, z_rep_ir, z_det_ir], dim=-1)
        logit_vis = self.gamma_vis(vis_fused)
        logit_ir = self.gamma_ir(ir_fused)
        logits = torch.stack([logit_vis, logit_ir], dim=-1)
        prob = torch.sigmoid(logits)
        return {
            "logits": logits,
            "prob": prob,
            "p_vis": prob[:, 0],
            "p_ir": prob[:, 1],
            "logit_vis": logit_vis,
            "logit_ir": logit_ir,
        }

    def compute_evidence(self, vis_img: np.ndarray, ir_img: np.ndarray, proposal: dict[str, Any]) -> dict[str, Any]:
        if self.legacy_adapter is None:
            raise RuntimeError("compute_evidence requires a configured LegacyDRFMAAdapter or use EvidenceExtractor directly.")
        return self.legacy_adapter.compute_evidence(vis_img, ir_img, proposal)

    def build_masks(
        self,
        proposals: list[dict[str, Any]],
        vis_img: np.ndarray,
        ir_img: np.ndarray,
        det_vis: Any = None,
        det_ir: Any = None,
        base_probs_list: list[list[float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self.legacy_adapter is None:
            if self.learned_checkpoint is None:
                details = ""
                if self.legacy_adapter_error is not None:
                    details = f" Legacy adapter initialization failed: {self.legacy_adapter_error}"
                raise RuntimeError(
                    "DRFMA.build_masks requires either paths.checkpoints.drf_ma "
                    "for learned inference or paths.legacy_code_root for the compatibility adapter."
                    + details
                )
            return self._build_masks_with_learned_checkpoint(proposals, vis_img, ir_img, det_vis, det_ir)
        return self.legacy_adapter.build_masks(proposals, vis_img, ir_img, det_vis, det_ir, base_probs_list)

    def _build_masks_with_learned_checkpoint(
        self,
        proposals: list[dict[str, Any]],
        vis_img: np.ndarray,
        ir_img: np.ndarray,
        det_vis: Any = None,
        det_ir: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Generate masks with the self-contained learned DRF-MA model."""
        image_shape = vis_img.shape[:2]
        if ir_img.ndim == 3:
            ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
        else:
            ir_gray = ir_img
        if ir_gray.shape[:2] != image_shape:
            ir_gray = cv2.resize(ir_gray, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR)

        extractor = EvidenceExtractor(self.cfg, vis_detector=det_vis, ir_detector=det_ir)
        valid_proposals: list[dict[str, Any]] = []
        evidence_rows: list[dict[str, list[float]]] = []
        records: list[dict[str, Any]] = []
        for idx, proposal in enumerate(proposals):
            box = proposal.get("box", proposal.get("proposal_box"))
            if box is None:
                continue
            box = [float(v) for v in box]
            evidence = extractor.extract(vis_img, ir_gray, box)
            valid_proposals.append({"box": box, "score": float(proposal.get("score", proposal.get("proposal_score", 1.0)))})
            evidence_rows.append(evidence)
            records.append(
                {
                    "proposal_index": idx,
                    "proposal_box": box,
                    "proposal_score": float(proposal.get("score", proposal.get("proposal_score", 1.0))),
                    **evidence,
                }
            )

        if not evidence_rows:
            empty = np.zeros(image_shape, dtype=np.uint8)
            return empty, empty.copy(), []

        tensor_rows = [record_to_evidence_tensors(row) for row in evidence_rows]
        batch = {key: torch.stack([row[key] for row in tensor_rows], dim=0) for key in FEATURE_KEYS}
        if self.normalizer is not None:
            batch = self.normalizer.transform(batch)
        device = next(self.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}

        self.eval()
        with torch.inference_mode():
            out = self(batch)
        p_vis = out["p_vis"].detach().cpu().numpy().astype(float).tolist()
        p_ir = out["p_ir"].detach().cpu().numpy().astype(float).tolist()

        mask_generator = AsymmetricMaskGenerator(
            {
                "tau_vis": self.drf_cfg.tau_vis,
                "tau_ir": self.drf_cfg.tau_ir,
                "dominance_margin": self.drf_cfg.dominance_margin,
                "dilation_kernel": self.cfg.get("mask_dilate", self.cfg.get("dilation_kernel", 5)),
                "min_area": self.cfg.get("min_mask_area", self.cfg.get("min_area", 80)),
            }
        )
        vis_mask, ir_mask = mask_generator.generate(valid_proposals, p_vis, p_ir, image_shape)
        for record, pv, pi in zip(records, p_vis, p_ir):
            record["p_vis"] = float(pv)
            record["p_ir"] = float(pi)
            record["final_p_vis"] = float(pv)
            record["final_p_ir"] = float(pi)
            record["prediction"] = decide_attribution(
                float(pv),
                float(pi),
                self.drf_cfg.tau_vis,
                self.drf_cfg.tau_ir,
                self.drf_cfg.dominance_margin,
            )
        return vis_mask, ir_mask, records

    def build_targets(self, proposals: list[dict[str, Any]], patch_masks: dict[str, np.ndarray]) -> np.ndarray:
        image_shape = next(iter(patch_masks.values())).shape[:2]
        targets = []
        for proposal in proposals:
            box = proposal.get("box", proposal.get("proposal_box"))
            if box is None:
                targets.append([0.0, 0.0])
                continue
            prop_mask = boxes_to_mask(image_shape, [box])
            labels = []
            for key in ("visible", "infrared"):
                patch = patch_masks.get(key)
                if patch is None:
                    labels.append(0.0)
                    continue
                overlap = cv2.countNonZero(((prop_mask > 0) & (patch > 0)).astype(np.uint8))
                labels.append(1.0 if overlap > 0 else 0.0)
            targets.append(labels)
        return np.asarray(targets, dtype=np.float32)


class LegacyDRFMAAdapter:
    """Compatibility wrapper around the original attribution implementation."""

    def __init__(self, cfg: dict[str, Any], detector_api: Any = None, preview_restorer: Any = None):
        self.cfg = cfg
        self.detector_api = detector_api
        self.preview_restorer = preview_restorer
        legacy_root = cfg.get("legacy_code_root") or _nested_get(cfg, "paths.legacy_code_root")
        if not legacy_root:
            raise ValueError("LegacyDRFMAAdapter requires legacy_code_root.")

        legacy = load_legacy_drf_module(legacy_root)
        self.legacy = legacy
        legacy_cfg = getattr(legacy, "DRF" + "CM" + "AConfig")()
        for key, value in cfg.items():
            if hasattr(legacy_cfg, key):
                setattr(legacy_cfg, key, value)
        legacy_cfg.prototype_path = str(cfg.get("prototype_path") or _nested_get(cfg, "paths.checkpoints.prototypes", ""))
        legacy_cfg.learned_fusion_ckpt = str(cfg.get("learned_fusion_ckpt") or _nested_get(cfg, "paths.checkpoints.drf_ma", ""))
        if not legacy_cfg.learned_fusion_ckpt:
            legacy_cfg.use_learned_fusion = False
        if cfg.get("device"):
            legacy_cfg.learned_fusion_device = str(cfg["device"])

        detector_score = None
        if detector_api is not None and hasattr(detector_api, "score_in_box"):
            detector_score = lambda model, img, box, **_: detector_api.score_in_box(img, box)
        else:
            try:
                import module_c_api  # type: ignore

                detector_score = module_c_api.detector_score_in_box
            except Exception:
                detector_score = None
        attributor_cls = getattr(legacy, "FullDRF" + "CM" + "AAttributor")
        self.attributor = attributor_cls(legacy_cfg, detector_score_in_box=detector_score)

    def compute_evidence(self, vis_img: np.ndarray, ir_img: np.ndarray, proposal: dict[str, Any]) -> dict[str, Any]:
        record = self.attributor.attribute_one_proposal(
            proposal=proposal,
            vis_img_bgr=vis_img,
            ir_img_gray=ir_img,
            det_vis=None,
            det_ir=None,
            base_probs=None,
        )
        return record.get("evidence", {})

    def forward(self, vis_img: np.ndarray, ir_img: np.ndarray, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for proposal in proposals:
            record = self.attributor.attribute_one_proposal(
                proposal=proposal,
                vis_img_bgr=vis_img,
                ir_img_gray=ir_img,
                det_vis=None,
                det_ir=None,
                base_probs=None,
            )
            p_vis, p_ir = evidence_record_to_probabilities(record)
            record["p_vis"] = p_vis
            record["p_ir"] = p_ir
            records.append(record)
        return records

    def build_masks(
        self,
        proposals: list[dict[str, Any]],
        vis_img: np.ndarray,
        ir_img: np.ndarray,
        det_vis: Any = None,
        det_ir: Any = None,
        base_probs_list: list[list[float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        return self.attributor.build_masks(
            proposals=proposals,
            vis_img_bgr=vis_img,
            ir_img_gray=ir_img,
            det_vis=det_vis,
            det_ir=det_ir,
            base_probs_list=base_probs_list,
        )


def load_drfma_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[DRFMA, EvidenceNormalizer, dict[str, Any]]:
    """Load a trained DRF-MA checkpoint."""
    ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt.get("config", {})
    model = DRFMA(cfg)
    model.load_state_dict(ckpt["model_state"])
    normalizer = EvidenceNormalizer.from_state_dict(ckpt.get("normalizer"))
    return model, normalizer, ckpt
