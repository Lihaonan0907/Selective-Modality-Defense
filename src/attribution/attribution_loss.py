"""Losses for DRF-MA learned attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weighted_mean(raw: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return raw.mean()
    weights = weights.float().view(-1)
    return (raw.view(-1) * weights).sum() / (weights.sum() + 1e-6)


def multilabel_attribution_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary multi-label BCE for visible/infrared corruption labels."""
    raw = F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight, reduction="none").mean(dim=1)
    return _weighted_mean(raw, sample_weight)


def ranking_loss(logits: torch.Tensor, targets: torch.Tensor, margin: float = 0.2, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Rank the attacked branch above the clean branch for single-modal attacks."""
    lv = logits[:, 0]
    li = logits[:, 1]
    yv = targets[:, 0] > 0.5
    yi = targets[:, 1] > 0.5
    vis_only = yv & ~yi
    ir_only = ~yv & yi

    raw = torch.zeros_like(lv)
    raw = torch.where(vis_only, F.relu(margin - (lv - li)), raw)
    raw = torch.where(ir_only, F.relu(margin - (li - lv)), raw)
    active = vis_only | ir_only
    if not active.any():
        return logits.new_tensor(0.0)
    weights = sample_weight[active] if sample_weight is not None else None
    return _weighted_mean(raw[active], weights)


def both_modality_protection_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    tau_b: float = 0.7,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep both branches above a floor for cross-modal attacks."""
    both = (targets[:, 0] > 0.5) & (targets[:, 1] > 0.5)
    if not both.any():
        return probabilities.new_tensor(0.0)
    raw = F.relu(tau_b - probabilities[both, 0]) + F.relu(tau_b - probabilities[both, 1])
    weights = sample_weight[both] if sample_weight is not None else None
    return _weighted_mean(raw, weights)


def clean_suppression_loss(probabilities: torch.Tensor, targets: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Suppress both probabilities for clean proposals with squared penalty."""
    clean = (targets[:, 0] < 0.5) & (targets[:, 1] < 0.5)
    if not clean.any():
        return probabilities.new_tensor(0.0)
    raw = probabilities[clean, 0].pow(2) + probabilities[clean, 1].pow(2)
    weights = sample_weight[clean] if sample_weight is not None else None
    return _weighted_mean(raw, weights)


@dataclass
class DRFMALossConfig:
    lambda_rank: float = 0.5
    lambda_both: float = 1.0
    lambda_clean: float = 0.3
    gamma: float = 0.2
    tau_b: float = 0.7

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "DRFMALossConfig":
        cfg = cfg or {}
        loss_cfg = cfg.get("loss", cfg)
        return cls(
            lambda_rank=float(loss_cfg.get("lambda_rank", 0.5)),
            lambda_both=float(loss_cfg.get("lambda_both", 1.0)),
            lambda_clean=float(loss_cfg.get("lambda_clean", 0.3)),
            gamma=float(loss_cfg.get("gamma", 0.2)),
            tau_b=float(loss_cfg.get("tau_b", 0.7)),
        )


class DRFMALoss(nn.Module):
    """Composite DRF-MA loss."""

    def __init__(self, cfg: dict[str, Any] | DRFMALossConfig | None = None, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, DRFMALossConfig) else DRFMALossConfig.from_dict(cfg)
        self.register_buffer("pos_weight", pos_weight.float() if pos_weight is not None else None)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = outputs["logits"]
        probabilities = outputs["prob"]
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = multilabel_attribution_loss(logits, targets, pos_weight=pos_weight, sample_weight=sample_weight)
        rank = ranking_loss(logits, targets, margin=self.cfg.gamma, sample_weight=sample_weight)
        both = both_modality_protection_loss(probabilities, targets, tau_b=self.cfg.tau_b, sample_weight=sample_weight)
        clean = clean_suppression_loss(probabilities, targets, sample_weight=sample_weight)
        total = bce + self.cfg.lambda_rank * rank + self.cfg.lambda_both * both + self.cfg.lambda_clean * clean
        return total, {"bce": bce.detach(), "rank": rank.detach(), "both": both.detach(), "clean": clean.detach()}
