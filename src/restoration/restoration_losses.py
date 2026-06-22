"""Training losses for modality-specific restoration."""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgeAwareTVLoss(nn.Module):
    """Total variation regularizer weighted by image edges."""

    def forward(self, image: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        dx = torch.abs(image[..., :, 1:] - image[..., :, :-1])
        dy = torch.abs(image[..., 1:, :] - image[..., :-1, :])
        if edge_weight is not None:
            dx = dx * edge_weight[..., :, 1:]
            dy = dy * edge_weight[..., 1:, :]
        return dx.mean() + dy.mean()


class FrequencyLoss(nn.Module):
    """Frequency-domain reconstruction loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")
        return torch.mean(torch.abs(pred_fft - target_fft))

