#!/usr/bin/env python3
"""Train single-branch attribution head."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attribution.single_branch import (
    SingleBranchAttribution,
    SingleBranchAttributionDataset,
    SingleBranchEvidenceExtractor,
    SingleBranchEvidenceNormalizer,
    build_single_branch_records,
    discover_single_branch_samples,
    load_jsonl,
    save_jsonl,
)
from src.detection import TaskDrivenDetector
from src.proposal import SingleBranchProposal
from src.utils.common import ensure_dir
from src.utils.config import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _ckpt_paths(cfg: dict[str, Any], mode: str) -> tuple[str | None, str | None]:
    checkpoints = cfg.get("paths", {}).get("checkpoints", {})
    single_ckpts = checkpoints.get("single_branch", {})
    proposal = single_ckpts.get(f"{mode}_proposal") or checkpoints.get(f"{mode}_proposal")
    detector = checkpoints.get("vis_detector") if mode == "vis" else checkpoints.get("ir_detector")
    return proposal, detector


def _build_records(
    cfg: dict[str, Any],
    mode: str,
    split: str,
    rebuild_cache: bool,
) -> list[dict[str, Any]]:
    single_cfg = cfg.get("single_branch", {})
    data_root = single_cfg.get("data_root") or cfg.get("paths", {}).get("data_root")
    if not data_root:
        raise ValueError("Missing single_branch.data_root or paths.data_root")
    output_dir = Path(single_cfg.get("output_dir", f"outputs/single_branch_{mode}"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    cache_path = output_dir / "evidence_cache" / f"{split}.jsonl"
    if cache_path.exists() and not rebuild_cache:
        return load_jsonl(cache_path)

    proposal_ckpt, detector_ckpt = _ckpt_paths(cfg, mode)
    proposal_cfg = dict(cfg.get("proposal", {}))
    proposal_cfg.update(single_cfg.get("proposal", {}))
    proposal_cfg["device"] = cfg.get("device", "cuda:0")
    proposal_cfg["checkpoint"] = proposal_ckpt
    proposal = SingleBranchProposal(proposal_cfg, mode)

    detector_cfg = dict(cfg.get("detector", {}))
    detector_cfg["device"] = cfg.get("device", "cuda:0")
    detector_cfg["checkpoint"] = detector_ckpt
    detector = TaskDrivenDetector(detector_cfg, mode)

    samples = discover_single_branch_samples(data_root, mode, split=split)
    extractor = SingleBranchEvidenceExtractor(single_cfg.get("evidence", {}), detector=detector)
    records = build_single_branch_records(
        samples,
        mode,
        proposal,
        extractor,
        split,
        iou_threshold=float(single_cfg.get("labels", {}).get("attr_iou_threshold", 0.50)),
        use_patch_labels=True,
    )
    save_jsonl(records, cache_path)
    return records


def _move_evidence(evidence: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in evidence.items()}


def evaluate(model: SingleBranchAttribution, loader: DataLoader, device: torch.device, tau: float) -> dict[str, float]:
    model.eval()
    probs: list[float] = []
    targets: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            evidence = _move_evidence(batch["evidence"], device)
            target = batch["target"].to(device)
            out = model(evidence)
            loss = F.binary_cross_entropy_with_logits(out["logit"], target)
            losses.append(float(loss.detach().cpu()))
            probs.extend(out["prob"].detach().cpu().numpy().astype(float).tolist())
            targets.extend(target.detach().cpu().numpy().astype(float).tolist())
    probs_np = np.asarray(probs, dtype=np.float32)
    targets_np = np.asarray(targets, dtype=np.float32)
    pred = probs_np >= float(tau)
    truth = targets_np >= 0.5
    tp = float((pred & truth).sum())
    fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    acc = float((pred == truth).mean()) if len(truth) else 0.0
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {"loss": float(np.mean(losses)) if losses else 0.0, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-branch attribution")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--mode", choices=["vis", "ir"], required=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.paths)
    set_seed(int(cfg.get("seed", 42)))
    device_name = cfg.get("device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() or str(device_name).startswith("cpu") else "cpu")
    single_cfg = cfg.get("single_branch", {})
    output_dir = Path(single_cfg.get("output_dir", f"outputs/single_branch_{args.mode}")) / "attribution"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    ensure_dir(output_dir)

    train_records = _build_records(cfg, args.mode, "train", rebuild_cache=args.rebuild_cache)
    val_records = _build_records(cfg, args.mode, "val", rebuild_cache=args.rebuild_cache)
    if not train_records or not val_records:
        raise RuntimeError("No single-branch attribution records were built. Check data_root, proposal checkpoint, and patch_labels.")

    normalizer = SingleBranchEvidenceNormalizer.fit(train_records)
    train_dataset = SingleBranchAttributionDataset(train_records, normalizer=normalizer)
    val_dataset = SingleBranchAttributionDataset(val_records, normalizer=normalizer)
    train_cfg = single_cfg.get("train", {})
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 256))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SingleBranchAttribution(single_cfg.get("attribution", {}), modality=args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    epochs = int(args.epochs or train_cfg.get("epochs", 50))
    tau = model.threshold
    best_f1 = -1.0
    rows: list[dict[str, Any]] = []
    patience = int(train_cfg.get("early_stop_patience", 10))
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running: list[float] = []
        for batch in train_loader:
            evidence = _move_evidence(batch["evidence"], device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(evidence)
            loss = F.binary_cross_entropy_with_logits(out["logit"], target)
            loss.backward()
            optimizer.step()
            running.append(float(loss.detach().cpu()))

        metrics = evaluate(model, val_loader, device, tau)
        row = {"epoch": epoch, "train_loss": float(np.mean(running)) if running else 0.0, **metrics}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "config": single_cfg.get("attribution", {}),
                    "modality": args.mode,
                    "thresholds": {"tau": tau},
                    "metrics": metrics,
                },
                output_dir / "best.pt",
            )
        else:
            bad_epochs += 1
        write_csv(rows, output_dir / "training_log.csv")
        if bad_epochs >= patience:
            break

    print(json.dumps({"output_dir": str(output_dir), "best_f1": best_f1}, indent=2))


if __name__ == "__main__":
    main()
