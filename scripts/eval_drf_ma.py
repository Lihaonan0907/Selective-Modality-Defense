#!/usr/bin/env python3
"""Evaluate a trained DRF-MA checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_drf_ma import materialize_records, write_csv
from src.attribution.drf_ma import load_drfma_checkpoint
from src.attribution.evidence_signals import EvidenceExtractor, EvidenceRecordDataset
from src.attribution.metrics import compute_metrics, matrix_to_rows, threshold_sweep
from src.utils.common import ensure_dir
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DRF-MA attribution")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "drf_ma.yaml"))
    parser.add_argument("--paths", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.paths)
    checkpoint = Path(args.checkpoint or Path(cfg.get("output_dir", "outputs/drf_ma")) / "best.pt")
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    output_dir = Path(args.output or checkpoint.parent / f"eval_{args.split}")
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    ensure_dir(output_dir)

    model, normalizer, ckpt = load_drfma_checkpoint(checkpoint, map_location="cpu")
    device_name = cfg.get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() or str(device_name).startswith("cpu") else "cpu")
    model.to(device).eval()

    if args.rebuild_cache:
        cfg.setdefault("cache", {})["rebuild"] = True
    records = materialize_records(cfg, args.split, EvidenceExtractor(cfg.get("evidence", {})), max_samples=args.max_samples)
    dataset = EvidenceRecordDataset(records, normalizer=normalizer)
    loader = DataLoader(dataset, batch_size=int(cfg.get("train", {}).get("batch_size", 256)), shuffle=False, num_workers=0)

    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            evidence = {k: v.to(device) for k, v in batch["evidence"].items()}
            out = model(evidence)
            probs.append(out["prob"].cpu().numpy())
            targets.append(batch["target"].cpu().numpy())
    probabilities = np.concatenate(probs, axis=0) if probs else np.zeros((0, 2), dtype=np.float32)
    y_true = np.concatenate(targets, axis=0) if targets else np.zeros((0, 2), dtype=np.float32)

    thresholds = ckpt.get("best_thresholds", {})
    tau_vis = float(thresholds.get("tau_vis", cfg.get("thresholds", {}).get("tau_vis", 0.90)))
    tau_ir = float(thresholds.get("tau_ir", cfg.get("thresholds", {}).get("tau_ir", 0.65)))
    dominance_margin = float(cfg.get("thresholds", {}).get("dominance_margin", 0.10))
    metrics = compute_metrics(y_true, probabilities, tau_vis, tau_ir, dominance_margin)
    sweep_rows, best = threshold_sweep(
        y_true,
        probabilities,
        cfg.get("thresholds", {}).get("tau_vis_values", [0.80, 0.85, 0.90, 0.95, 1.00]),
        cfg.get("thresholds", {}).get("tau_ir_values", [0.55, 0.60, 0.65, 0.70, 0.75]),
        dominance_margin,
    )

    write_csv(sweep_rows, output_dir / "threshold_sweep.csv")
    write_csv(matrix_to_rows(metrics["confusion_matrix_4x4"]), output_dir / "confusion_matrix.csv")
    pred_rows: list[dict[str, Any]] = []
    for record, prob, pred in zip(records, probabilities, metrics["y_pred_cls"]):
        pred_rows.append(
            {
                "attack_type": record.get("attack_type"),
                "stem": record.get("stem"),
                "proposal_index": record.get("proposal_index"),
                "label_vis": record.get("label_vis"),
                "label_ir": record.get("label_ir"),
                "p_vis": float(prob[0]),
                "p_ir": float(prob[1]),
                "pred_class_id": int(pred),
            }
        )
    write_csv(pred_rows, output_dir / "predictions.csv")

    summary = {k: v for k, v in metrics.items() if k not in {"confusion_matrix_4x4", "y_pred_cls"}}
    summary["best_sweep_thresholds"] = best
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
