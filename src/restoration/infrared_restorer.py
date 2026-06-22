"""Infrared image restoration expert."""

from __future__ import annotations

from typing import Any

from .diffusion_restorer import ModalityRestorer


class InfraredRestorer(ModalityRestorer):
    """Infrared branch restoration expert."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg, modality="ir")

