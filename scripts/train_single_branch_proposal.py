#!/usr/bin/env python3
"""Train a single-branch anomaly proposal detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import ensure_dir, require_path
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train visible-only or infrared-only proposal detector")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--mode", choices=["vis", "ir"], required=True)
    parser.add_argument("--data-yaml", required=True, help="Ultralytics dataset YAML for the current modality")
    parser.add_argument("--weights", default="yolov8l.pt", help="Initial YOLO weights or model YAML")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.paths)
    proposal_cfg = dict(cfg.get("proposal", {}))
    proposal_cfg.update(cfg.get("single_branch", {}).get("proposal", {}))
    train_cfg = cfg.get("single_branch", {}).get("proposal_train", {})

    output_dir = ensure_dir(args.output or cfg.get("single_branch", {}).get("output_dir", f"outputs/single_branch_{args.mode}") + "/proposal")
    model = YOLO(str(args.weights))
    results = model.train(
        data=str(require_path(args.data_yaml, "single-branch proposal data YAML")),
        epochs=int(args.epochs or train_cfg.get("epochs", 100)),
        imgsz=int(args.imgsz or proposal_cfg.get("input_size", 960)),
        batch=int(args.batch or train_cfg.get("batch", 4)),
        device=str(cfg.get("device", "cuda:0")).replace("cuda:", ""),
        workers=int(train_cfg.get("workers", 0)),
        project=str(output_dir),
        name=f"{args.mode}_proposal",
        save=True,
        val=True,
        single_cls=True,
    )
    print(json.dumps({"output_dir": str(output_dir), "mode": args.mode, "results": str(results)}, indent=2))


if __name__ == "__main__":
    main()
