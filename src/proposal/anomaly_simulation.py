"""Compatibility entry point for proposal anomaly-simulation training."""

from __future__ import annotations

from pathlib import Path

from src.utils.common import add_legacy_code_root


def load_legacy_training_function(legacy_code_root: str | Path):
    """Load the original proposal-training function without modifying it."""
    add_legacy_code_root(legacy_code_root)
    from train_patch_detector import train  # type: ignore

    return train
