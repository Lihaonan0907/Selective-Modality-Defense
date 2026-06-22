"""Evaluation metrics and threshold search for DRF-MA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.attribution.evidence_signals import CLASS_TO_ID, ID_TO_CLASS, class_from_labels


DISPLAY_NAMES = ["Clean", "VIS", "IR", "Both"]


@dataclass
class ThresholdConfig:
    tau_vis: float = 0.90
    tau_ir: float = 0.65
    dominance_margin: float = 0.10

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "ThresholdConfig":
        cfg = cfg or {}
        thresholds = cfg.get("thresholds", cfg.get("attribution", cfg))
        return cls(
            tau_vis=float(thresholds.get("tau_vis", 0.90)),
            tau_ir=float(thresholds.get("tau_ir", 0.65)),
            dominance_margin=float(thresholds.get("dominance_margin", 0.10)),
        )


def labels_to_class_ids(y_true: np.ndarray) -> np.ndarray:
    """Convert [y_vis, y_ir] multilabel targets to four-class ids."""
    return np.asarray([CLASS_TO_ID[class_from_labels(y[0], y[1])] for y in y_true], dtype=np.int64)


def apply_dominance_correction(
    p_vis: np.ndarray,
    p_ir: np.ndarray,
    dominance_margin: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Suppress the weaker probability when one branch clearly dominates."""
    pv = np.asarray(p_vis, dtype=np.float32).copy()
    pi = np.asarray(p_ir, dtype=np.float32).copy()
    vis_dominant = pv - pi > dominance_margin
    ir_dominant = pi - pv > dominance_margin
    pi[vis_dominant] = 0.0
    pv[ir_dominant] = 0.0
    return pv, pi


def predict_classes(
    probabilities: np.ndarray,
    tau_vis: float = 0.90,
    tau_ir: float = 0.65,
    dominance_margin: float = 0.10,
) -> np.ndarray:
    """Predict Clean/VIS/IR/Both class ids from p_vis and p_ir."""
    probs = np.asarray(probabilities, dtype=np.float32)
    pv, pi = apply_dominance_correction(probs[:, 0], probs[:, 1], dominance_margin)
    pred = np.zeros(len(probs), dtype=np.int64)
    pred[(pv >= tau_vis) & (pi < tau_ir)] = CLASS_TO_ID["vis_only"]
    pred[(pv < tau_vis) & (pi >= tau_ir)] = CLASS_TO_ID["ir_only"]
    pred[(pv >= tau_vis) & (pi >= tau_ir)] = CLASS_TO_ID["both"]
    return pred


def confusion_matrix_4x4(y_true_cls: np.ndarray, y_pred_cls: np.ndarray) -> np.ndarray:
    """Return a 4x4 confusion matrix in Clean, VIS, IR, Both order."""
    matrix = np.zeros((4, 4), dtype=np.int64)
    for t, p in zip(y_true_cls.astype(int), y_pred_cls.astype(int)):
        matrix[t, p] += 1
    return matrix


def _class_accuracy(matrix: np.ndarray, class_id: int) -> float:
    denom = matrix[class_id].sum()
    if denom == 0:
        return 0.0
    return float(matrix[class_id, class_id] / denom)


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    tau_vis: float = 0.90,
    tau_ir: float = 0.65,
    dominance_margin: float = 0.10,
) -> dict[str, Any]:
    """Compute DRF-MA attribution metrics."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_true_cls = labels_to_class_ids(y_true)
    pred_cls = predict_classes(probabilities, tau_vis, tau_ir, dominance_margin)
    cm = confusion_matrix_4x4(y_true_cls, pred_cls)
    acc = float((pred_cls == y_true_cls).mean()) if len(y_true_cls) else 0.0

    both_id = CLASS_TO_ID["both"]
    both_total = max(1, int((y_true_cls == both_id).sum()))
    both_recall = float(((pred_cls == both_id) & (y_true_cls == both_id)).sum() / both_total)

    vis_id = CLASS_TO_ID["vis_only"]
    ir_id = CLASS_TO_ID["ir_only"]
    single = (y_true_cls == vis_id) | (y_true_cls == ir_id)
    if single.any():
        unnecessary = ((y_true_cls == vis_id) & ((pred_cls == ir_id) | (pred_cls == both_id))) | (
            (y_true_cls == ir_id) & ((pred_cls == vis_id) | (pred_cls == both_id))
        )
        urr = float((unnecessary & single).sum() / single.sum())
    else:
        urr = 0.0

    return {
        "attribution_accuracy": acc,
        "both_modality_recall": both_recall,
        "unnecessary_restoration_ratio_URR": urr,
        "clean_accuracy": _class_accuracy(cm, CLASS_TO_ID["clean"]),
        "vis_only_accuracy": _class_accuracy(cm, vis_id),
        "ir_only_accuracy": _class_accuracy(cm, ir_id),
        "both_accuracy": _class_accuracy(cm, both_id),
        "confusion_matrix_4x4": cm,
        "y_pred_cls": pred_cls,
        "tau_vis": float(tau_vis),
        "tau_ir": float(tau_ir),
        "dominance_margin": float(dominance_margin),
    }


def threshold_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    tau_vis_values: list[float] | tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 1.00),
    tau_ir_values: list[float] | tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75),
    dominance_margin: float = 0.10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sweep tau_vis/tau_ir and return rows plus the best row."""
    rows: list[dict[str, Any]] = []
    for tau_vis in tau_vis_values:
        for tau_ir in tau_ir_values:
            metrics = compute_metrics(y_true, probabilities, tau_vis, tau_ir, dominance_margin)
            row = {
                "tau_vis": float(tau_vis),
                "tau_ir": float(tau_ir),
                "attr_acc": metrics["attribution_accuracy"],
                "both_recall": metrics["both_modality_recall"],
                "urr": metrics["unnecessary_restoration_ratio_URR"],
                "clean_acc": metrics["clean_accuracy"],
                "vis_acc": metrics["vis_only_accuracy"],
                "ir_acc": metrics["ir_only_accuracy"],
                "both_acc": metrics["both_accuracy"],
            }
            rows.append(row)
    best = max(rows, key=lambda r: (r["attr_acc"], r["both_recall"], -r["urr"]))
    return rows, best


def matrix_to_rows(matrix: np.ndarray) -> list[dict[str, Any]]:
    """Convert a confusion matrix to CSV-friendly rows."""
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(DISPLAY_NAMES):
        row = {"true": name}
        for j, pred_name in enumerate(DISPLAY_NAMES):
            row[pred_name] = int(matrix[i, j])
        rows.append(row)
    return rows
