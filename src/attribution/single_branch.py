"""Single-branch attribution for visible-only or infrared-only defense."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.attribution.evidence_signals import (
    contrast_deviation,
    edge_density,
    high_frequency_energy,
    local_detector_score,
    preview_inpaint,
    read_patch_annotation,
    safe_imread,
)
from src.detection.detector_api import iou_xyxy
from src.proposal.proposal_utils import boxes_to_mask, clip_box_xyxy
from src.utils.common import ensure_dir, require_path


SINGLE_FEATURE_KEYS = ("z_self", "z_rep", "z_det")
SINGLE_FEATURE_DIMS = {"z_self": 4, "z_rep": 2, "z_det": 1}


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _crop_box(image: np.ndarray, box: Iterable[float]) -> np.ndarray:
    h, w = image.shape[:2]
    clipped = clip_box_xyxy(box, w, h)
    if clipped is None:
        return image[:0, :0]
    x1, y1, x2, y2 = clipped
    return image[y1:y2, x1:x2]


def _restoration_residual(original: np.ndarray, restored: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or cv2.countNonZero(mask) == 0:
        return 0.0
    ori = original.astype(np.float32)
    rep = restored.astype(np.float32)
    if ori.max() > 1.5:
        ori /= 255.0
    if rep.max() > 1.5:
        rep /= 255.0
    if rep.shape[:2] != ori.shape[:2]:
        rep = cv2.resize(rep, (ori.shape[1], ori.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = np.abs(rep - ori)
    mask_bool = mask > 0
    return float(diff[mask_bool].mean()) if np.any(mask_bool) else 0.0


def _safe_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    return Path(path)


class SingleBranchEvidenceExtractor:
    """Extract single-modality evidence without cross-modal signals."""

    def __init__(self, cfg: dict[str, Any] | None = None, detector: Any = None):
        cfg = cfg or {}
        self.inpaint_radius = int(cfg.get("inpaint_radius", 5))
        self.preview_dilate = int(cfg.get("preview_dilate", 7))
        self.det_overlap_threshold = float(cfg.get("det_overlap_threshold", 0.3))
        self.detector = detector

    def extract(self, image: np.ndarray, proposal_box: Iterable[float]) -> dict[str, list[float]]:
        h, w = image.shape[:2]
        box = clip_box_xyxy(proposal_box, w, h)
        if box is None:
            return {key: [0.0] * SINGLE_FEATURE_DIMS[key] for key in SINGLE_FEATURE_KEYS}

        mask = boxes_to_mask((h, w), [box])
        restored = preview_inpaint(image, mask, radius=self.inpaint_radius, dilate=self.preview_dilate)
        crop = _crop_box(image, box)
        restored_crop = _crop_box(restored, box)

        residual = _restoration_residual(image, restored, mask)
        edge_before = edge_density(crop)
        edge_after = edge_density(restored_crop)
        high_freq_before = high_frequency_energy(crop)
        high_freq_after = high_frequency_energy(restored_crop)
        score_before = local_detector_score(self.detector, image, box, self.det_overlap_threshold)
        score_after = local_detector_score(self.detector, restored, box, self.det_overlap_threshold)

        return {
            "z_self": [
                float(edge_before),
                float(high_freq_before),
                float(contrast_deviation(image, box)),
                float(residual),
            ],
            "z_rep": [
                float(residual),
                float(max(0.0, high_freq_before - high_freq_after) + max(0.0, edge_before - edge_after)),
            ],
            "z_det": [float(score_after - score_before)],
        }


class SingleBranchEvidenceNormalizer:
    """Per-feature mean/std normalizer for single-branch attribution."""

    def __init__(self, stats: dict[str, dict[str, list[float]]] | None = None):
        self.stats = stats or {}

    @classmethod
    def fit(cls, records: list[dict[str, Any]]) -> "SingleBranchEvidenceNormalizer":
        stats: dict[str, dict[str, list[float]]] = {}
        for key in SINGLE_FEATURE_KEYS:
            arr = np.asarray([r.get(key, [0.0] * SINGLE_FEATURE_DIMS[key]) for r in records], dtype=np.float32)
            mean = arr.mean(axis=0) if arr.size else np.zeros(SINGLE_FEATURE_DIMS[key], dtype=np.float32)
            std = arr.std(axis=0) if arr.size else np.ones(SINGLE_FEATURE_DIMS[key], dtype=np.float32)
            std = np.where(std < 1e-6, 1.0, std)
            stats[key] = {"mean": mean.tolist(), "std": std.tolist()}
        return cls(stats)

    def state_dict(self) -> dict[str, dict[str, list[float]]]:
        return self.stats

    @classmethod
    def from_state_dict(cls, state: dict[str, dict[str, list[float]]] | None) -> "SingleBranchEvidenceNormalizer":
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


@dataclass
class SingleBranchAttributionConfig:
    self_dim: int = 4
    rep_dim: int = 2
    det_dim: int = 1
    hidden_dim: int = 64
    evidence_dim: int = 32
    dropout: float = 0.10
    tau_vis: float = 0.90
    tau_ir: float = 0.65

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "SingleBranchAttributionConfig":
        cfg = cfg or {}
        model_cfg = cfg.get("model", cfg.get("single_branch", cfg))
        thresholds = cfg.get("thresholds", cfg)
        return cls(
            self_dim=int(model_cfg.get("self_dim", SINGLE_FEATURE_DIMS["z_self"])),
            rep_dim=int(model_cfg.get("rep_dim", SINGLE_FEATURE_DIMS["z_rep"])),
            det_dim=int(model_cfg.get("det_dim", SINGLE_FEATURE_DIMS["z_det"])),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            evidence_dim=int(model_cfg.get("evidence_dim", 32)),
            dropout=float(model_cfg.get("dropout", 0.10)),
            tau_vis=float(thresholds.get("tau_vis", model_cfg.get("tau_vis", 0.90))),
            tau_ir=float(thresholds.get("tau_ir", model_cfg.get("tau_ir", 0.65))),
        )


class _EvidenceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
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


class SingleBranchAttribution(nn.Module):
    """Single-branch attribution head P_i^k = sigmoid(Gamma_k(Z_self,Z_rep,Z_det))."""

    def __init__(self, cfg: dict[str, Any] | SingleBranchAttributionConfig | None = None, modality: str = "vis"):
        super().__init__()
        if modality not in {"vis", "ir"}:
            raise ValueError("modality must be 'vis' or 'ir'")
        self.modality = modality
        self.cfg = cfg if isinstance(cfg, dict) else {}
        c = cfg if isinstance(cfg, SingleBranchAttributionConfig) else SingleBranchAttributionConfig.from_dict(self.cfg)
        self.attr_cfg = c
        self.self_mlp = _EvidenceMLP(c.self_dim, c.hidden_dim, c.evidence_dim, c.dropout)
        self.rep_mlp = _EvidenceMLP(c.rep_dim, c.hidden_dim, c.evidence_dim, c.dropout)
        self.det_mlp = _EvidenceMLP(c.det_dim, c.hidden_dim, c.evidence_dim, c.dropout)
        self.gamma = nn.Sequential(
            nn.Linear(c.evidence_dim * 3, c.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(c.hidden_dim),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, 1),
        )
        self.normalizer: SingleBranchEvidenceNormalizer | None = None

    @property
    def threshold(self) -> float:
        return self.attr_cfg.tau_vis if self.modality == "vis" else self.attr_cfg.tau_ir

    def forward(self, evidence: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_self = self.self_mlp(evidence["z_self"])
        z_rep = self.rep_mlp(evidence["z_rep"])
        z_det = self.det_mlp(evidence["z_det"])
        logit = self.gamma(torch.cat([z_self, z_rep, z_det], dim=-1)).squeeze(-1)
        prob = torch.sigmoid(logit)
        return {"logit": logit, "prob": prob}

    def load_checkpoint(self, checkpoint: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        ckpt_path = require_path(checkpoint, f"{self.modality} single-branch attribution checkpoint")
        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.load_state_dict(ckpt["model_state"])
        self.normalizer = SingleBranchEvidenceNormalizer.from_state_dict(ckpt.get("normalizer"))
        thresholds = ckpt.get("thresholds", {})
        if self.modality == "vis":
            self.attr_cfg.tau_vis = float(thresholds.get("tau", thresholds.get("tau_vis", self.attr_cfg.tau_vis)))
        else:
            self.attr_cfg.tau_ir = float(thresholds.get("tau", thresholds.get("tau_ir", self.attr_cfg.tau_ir)))
        return ckpt

    def predict_records(self, records: list[dict[str, Any]], device: torch.device) -> list[float]:
        if not records:
            return []
        evidence = records_to_single_branch_tensors(records)
        if self.normalizer is not None:
            evidence = self.normalizer.transform(evidence)
        evidence = {key: value.to(device) for key, value in evidence.items()}
        self.to(device).eval()
        with torch.inference_mode():
            out = self(evidence)
        return out["prob"].detach().cpu().numpy().astype(float).tolist()


def records_to_single_branch_tensors(records: list[dict[str, Any]] | dict[str, Any]) -> dict[str, torch.Tensor]:
    if isinstance(records, dict):
        records = [records]
    return {
        key: torch.tensor([r.get(key, [0.0] * SINGLE_FEATURE_DIMS[key]) for r in records], dtype=torch.float32)
        for key in SINGLE_FEATURE_KEYS
    }


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def patch_support_to_mask(label_path: str | Path | None, image_shape_hw: tuple[int, int]) -> np.ndarray:
    annotation = read_patch_annotation(label_path, image_shape_hw)
    mask = annotation.mask if annotation.mask is not None else np.zeros(image_shape_hw, dtype=np.uint8)
    for box in annotation.boxes:
        mask = cv2.bitwise_or(mask, boxes_to_mask(image_shape_hw, [box]))
    return (mask > 0).astype(np.uint8) * 255


def label_from_patch_support(proposal_box: Iterable[float], patch_mask: np.ndarray, iou_threshold: float = 0.50) -> int:
    if patch_mask is None or cv2.countNonZero(patch_mask) == 0:
        return 0
    prop_mask = boxes_to_mask(patch_mask.shape[:2], [proposal_box])
    inter = cv2.countNonZero(((prop_mask > 0) & (patch_mask > 0)).astype(np.uint8))
    union = cv2.countNonZero(((prop_mask > 0) | (patch_mask > 0)).astype(np.uint8))
    return int((inter / (union + 1e-6)) >= float(iou_threshold))


def _find_by_stem(directory: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".png", ".jpeg", ".bmp", ".txt"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def discover_single_branch_samples(data_root: str | Path, modality: str, split: str = "val") -> list[dict[str, Any]]:
    """Discover images from supported single-branch data layouts."""
    root = Path(data_root)
    branch_names = ["vis", "visible"] if modality == "vis" else ["ir", "infrared"]
    image_dirs = [root / "images", *(root / name / "images" for name in branch_names), *(root / name for name in branch_names)]
    image_dir = next((p for p in image_dirs if p.exists()), None)
    if image_dir is None:
        return []
    split_dir = image_dir / split
    if split_dir.exists():
        image_dir = split_dir

    label_roots = [
        root / "patch_labels",
        root / "patch_label",
        *(root / name / "patch_labels" for name in branch_names),
        *(root / name / "patch_label" for name in branch_names),
    ]
    label_root = next((p for p in label_roots if p.exists()), None)
    if label_root is not None and (label_root / split).exists():
        label_root = label_root / split

    samples: list[dict[str, Any]] = []
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".png", ".jpeg", ".bmp"}:
            continue
        label_path = _find_by_stem(label_root, path.stem) if label_root is not None else None
        samples.append({"image": str(path), "patch_label": str(label_path) if label_path else None, "stem": path.stem})
    return samples


class SingleBranchAttributionDataset(Dataset):
    """Dataset over proposal-level single-branch evidence records."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        normalizer: SingleBranchEvidenceNormalizer | None = None,
    ):
        self.records = records
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        evidence = records_to_single_branch_tensors(record)
        evidence = {key: value.squeeze(0) for key, value in evidence.items()}
        if self.normalizer is not None:
            evidence = self.normalizer.transform(evidence)
        return {
            "evidence": evidence,
            "target": torch.tensor(float(record.get("label", 0)), dtype=torch.float32),
        }


def build_single_branch_records(
    samples: list[dict[str, Any]],
    modality: str,
    proposal_detector: Any,
    extractor: SingleBranchEvidenceExtractor,
    split: str,
    iou_threshold: float = 0.50,
    use_patch_labels: bool = False,
) -> list[dict[str, Any]]:
    """Materialize evidence records. Patch labels are used only when explicitly enabled."""
    records: list[dict[str, Any]] = []
    for sample in samples:
        image = safe_imread(sample["image"], cv2.IMREAD_GRAYSCALE if modality == "ir" else cv2.IMREAD_COLOR)
        image_shape = image.shape[:2]
        proposals = proposal_detector.predict(image)
        patch_mask = None
        if use_patch_labels:
            patch_mask = patch_support_to_mask(_safe_path(sample.get("patch_label")), image_shape)
        for idx, proposal in enumerate(proposals):
            box = proposal.get("box")
            if box is None:
                continue
            label = label_from_patch_support(box, patch_mask, iou_threshold) if patch_mask is not None else 0
            evidence = extractor.extract(image, box)
            records.append(
                {
                    "split": split,
                    "modality": modality,
                    "image": sample["image"],
                    "stem": sample.get("stem"),
                    "proposal_index": idx,
                    "proposal_box": [float(v) for v in box],
                    "proposal_score": float(proposal.get("score", 1.0)),
                    "label": int(label),
                    **evidence,
                }
            )
    return records


def generate_single_branch_mask(
    proposals: list[dict[str, Any]],
    probabilities: list[float] | np.ndarray,
    image_shape: tuple[int, int],
    tau: float,
    dilation_kernel: int = 5,
    min_component_area: int = 80,
) -> np.ndarray:
    """Generate M_k = Post(union boxes with P_i^k >= tau_k)."""
    selected = [proposal["box"] for proposal, prob in zip(proposals, probabilities) if float(prob) >= float(tau)]
    return boxes_to_mask(image_shape, selected, dilation_kernel=dilation_kernel, min_area=min_component_area)
