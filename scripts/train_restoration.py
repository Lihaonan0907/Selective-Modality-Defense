#!/usr/bin/env python3
"""Train modality-specific restoration experts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.restoration import RestorationFineTuneDataset, RestorationFineTuner
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train restoration expert")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--paths", default=None)
    parser.add_argument("--modality", choices=["vis", "ir"], required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.paths)
    restoration_cfg = dict(cfg.get("restoration", {}))
    train_cfg = dict(restoration_cfg.get("train", {}))
    paths = cfg.get("paths", {})
    restoration_cfg.update(
        {
            "stable_diffusion_inpaint": paths.get("models", {}).get("stable_diffusion_inpaint"),
            "device": cfg.get("device", "cuda:0"),
            **train_cfg,
        }
    )
    data_root = args.data_root or train_cfg.get("data_root") or paths.get("data_root")
    if not data_root:
        raise ValueError("Missing restoration training data root. Use --data-root or restoration.train.data_root.")
    resolution = int(args.resolution or train_cfg.get("resolution", 512))
    dataset = RestorationFineTuneDataset(data_root, args.modality, split=args.split, resolution=resolution)
    if len(dataset) == 0:
        raise RuntimeError("No restoration fine-tuning samples were found. Check data_root and dataset layout.")
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    output_dir = args.output or train_cfg.get("output_dir") or f"outputs/restoration_{args.modality}"
    tuner = RestorationFineTuner(restoration_cfg, args.modality)
    tuner.fit(loader, epochs=int(args.epochs or train_cfg.get("epochs", 30)), output_dir=output_dir)


if __name__ == "__main__":
    main()
