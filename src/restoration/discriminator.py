"""Mask-conditioned discriminator blocks used for restoration fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedPatchGAN(nn.Module):
    """A compact PatchGAN discriminator conditioned on an image mask."""

    def __init__(self, in_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, 3, 1, 1),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        return self.net(torch.cat([image, mask], dim=1))

