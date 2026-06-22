"""Corrupted-branch attribution and asymmetric mask generation."""

from .drf_ma import DRFMA, EvidenceNormalizer, LegacyDRFMAAdapter, load_drfma_checkpoint
from .evidence_signals import DRFMADataset, EvidenceExtractor, EvidenceRecordDataset
from .mask_generation import AsymmetricMaskGenerator, decide_attribution
from .single_branch import (
    SingleBranchAttribution,
    SingleBranchAttributionDataset,
    SingleBranchEvidenceExtractor,
    SingleBranchEvidenceNormalizer,
    generate_single_branch_mask,
)

__all__ = [
    "DRFMA",
    "EvidenceNormalizer",
    "LegacyDRFMAAdapter",
    "load_drfma_checkpoint",
    "DRFMADataset",
    "EvidenceExtractor",
    "EvidenceRecordDataset",
    "AsymmetricMaskGenerator",
    "decide_attribution",
    "SingleBranchAttribution",
    "SingleBranchAttributionDataset",
    "SingleBranchEvidenceExtractor",
    "SingleBranchEvidenceNormalizer",
    "generate_single_branch_mask",
]
