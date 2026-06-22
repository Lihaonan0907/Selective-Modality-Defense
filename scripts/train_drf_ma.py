#!/usr/bin/env python3
"""Train DRF-MA proposal-level modality attribution."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attribution.attribution_loss import DRFMALoss
from src.attribution.drf_ma import DRFMA, EvidenceNormalizer
from src.attribution.evidence_signals import (
    DRFMADataset,
    EvidenceExtractor,
    EvidenceRecordDataset,
    load_jsonl,
    records_to_arrays,
    save_jsonl,
    summarize_records,
)
from src.attribution.metrics import compute_metrics, matrix_to_rows, threshold_sweep
from src.utils.common import ensure_dir
from src.utils.config import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cache_path(cfg: dict[str, Any], split: str, run_name: str = "full") -> Path:
    cache_dir = Path(cfg.get("cache", {}).get("evidence_dir", cfg.get("output_dir", "outputs/drf_ma") + "/evidence_cache"))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    suffix = cfg.get("cache", {}).get("suffix", "")
    name = f"{split}{('_' + suffix) if suffix else ''}.jsonl"
    return cache_dir / name


def materialize_records(
    cfg: dict[str, Any],
    split: str,
    extractor: EvidenceExtractor,
    rebuild_cache: bool = False,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    cache_path = _cache_path(cfg, split)
    debug_limited = max_samples is not None
    if cache_path.exists() and not rebuild_cache:
        rows = load_jsonl(cache_path)
        return rows[:max_samples] if max_samples is not None else rows

    max_images = cfg.get("dataset", {}).get("max_images_per_attack")
    if max_samples is not None and max_images is None:
        max_images = 1
    print(f"[DRF-MA] indexing {split} dataset ...", flush=True)
    dataset = DRFMADataset(cfg, split=split, evidence_extractor=extractor, max_images=max_images)
    records: list[dict[str, Any]] = []
    limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"[DRF-MA] building {split} evidence: {limit} proposals", flush=True)
    partial_path = cache_path.with_suffix(cache_path.suffix + ".partial")
    writer = None
    if not debug_limited:
        ensure_dir(partial_path.parent)
        writer = partial_path.open("w", encoding="utf-8")
    try:
        for idx, record in enumerate(dataset.iter_records(max_samples=limit), 1):
            records.append(record)
            if writer is not None:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                if idx % 100 == 0:
                    writer.flush()
            if idx == 1 or idx % 100 == 0 or idx == limit:
                print(f"[DRF-MA] {split} evidence {idx}/{limit}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if not debug_limited:
        partial_path.replace(cache_path)
        stats_path = cache_path.with_suffix(".stats.json")
        stats_path.write_text(json.dumps(summarize_records(records), indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def _move_evidence_to_device(evidence: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in evidence.items()}


def _sample_weight_from_batch(batch: dict[str, Any], train_cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    weights_cfg = train_cfg.get("case_weights")
    if not weights_cfg:
        return None
    class_ids = batch["class_id"].to(device)
    weights = torch.ones_like(class_ids, dtype=torch.float32)
    mapping = {
        0: float(weights_cfg.get("clean", 1.0)),
        1: float(weights_cfg.get("vis_only", 1.0)),
        2: float(weights_cfg.get("ir_only", 1.0)),
        3: float(weights_cfg.get("both", 1.0)),
    }
    for class_id, weight in mapping.items():
        weights = torch.where(class_ids == class_id, torch.full_like(weights, weight), weights)
    return weights


def evaluate(
    model: DRFMA,
    loader: DataLoader,
    criterion: DRFMALoss,
    device: torch.device,
    tau_vis_values: list[float],
    tau_ir_values: list[float],
    dominance_margin: float,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            evidence = _move_evidence_to_device(batch["evidence"], device)
            target = batch["target"].to(device)
            outputs = model(evidence)
            loss, _ = criterion(outputs, target)
            losses.append(float(loss.detach().cpu()))
            probs.append(outputs["prob"].detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
    prob_arr = np.concatenate(probs, axis=0) if probs else np.zeros((0, 2), dtype=np.float32)
    target_arr = np.concatenate(targets, axis=0) if targets else np.zeros((0, 2), dtype=np.float32)
    sweep_rows, best = threshold_sweep(target_arr, prob_arr, tau_vis_values, tau_ir_values, dominance_margin)
    metrics = compute_metrics(target_arr, prob_arr, best["tau_vis"], best["tau_ir"], dominance_margin)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "probabilities": prob_arr,
        "targets": target_arr,
        "sweep_rows": sweep_rows,
        "best_thresholds": best,
        "metrics": metrics,
    }


def train_from_config(
    cfg: dict[str, Any],
    run_name: str = "full",
    ablation: dict[str, bool] | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if ablation:
        cfg.setdefault("model", {})
        cfg["model"]["ablate"] = dict(ablation)

    seed = int(cfg.get("seed", cfg.get("train", {}).get("seed", 42)))
    set_seed(seed)
    requested_device = cfg.get("device", cfg.get("train", {}).get("device", "cuda:0"))
    device = torch.device(requested_device if torch.cuda.is_available() or str(requested_device).startswith("cpu") else "cpu")

    output_dir = Path(cfg.get("output_dir", "outputs/drf_ma"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir / run_name if run_name != "full" else output_dir
    ensure_dir(output_dir)

    extractor = EvidenceExtractor(cfg.get("evidence", {}))
    rebuild_cache = bool(cfg.get("cache", {}).get("rebuild", False))
    train_records = materialize_records(cfg, "train", extractor, rebuild_cache=rebuild_cache, max_samples=max_samples)
    val_records = materialize_records(cfg, "val", extractor, rebuild_cache=rebuild_cache, max_samples=max_samples)
    if not train_records or not val_records:
        raise RuntimeError("No DRF-MA train/val records were built. Check dataset.root, labels, and proposal fallback/cache.")

    normalizer = EvidenceNormalizer.fit(train_records)
    train_dataset = EvidenceRecordDataset(train_records, normalizer=normalizer)
    val_dataset = EvidenceRecordDataset(val_records, normalizer=normalizer)

    train_cfg = cfg.get("train", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("batch_size", 256)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )

    y_train, _ = records_to_arrays(train_records)
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    pos_weight = torch.tensor(np.clip(neg / (pos + 1e-6), 0.5, 20.0), dtype=torch.float32)

    model = DRFMA(cfg).to(device)
    criterion = DRFMALoss(cfg, pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 80))
    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        if scheduler_name == "cosine"
        else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    )

    thresholds_cfg = cfg.get("thresholds", {})
    tau_vis_values = [float(v) for v in thresholds_cfg.get("tau_vis_values", [0.80, 0.85, 0.90, 0.95, 1.00])]
    tau_ir_values = [float(v) for v in thresholds_cfg.get("tau_ir_values", [0.55, 0.60, 0.65, 0.70, 0.75])]
    dominance_margin = float(thresholds_cfg.get("dominance_margin", 0.10))

    best_score = -1.0
    best_payload: dict[str, Any] | None = None
    patience = int(train_cfg.get("early_stop_patience", 15))
    bad_epochs = 0
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        for batch in train_loader:
            evidence = _move_evidence_to_device(batch["evidence"], device)
            target = batch["target"].to(device)
            sample_weight = _sample_weight_from_batch(batch, train_cfg, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(evidence)
            loss, _ = criterion(outputs, target, sample_weight=sample_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 5.0)))
            optimizer.step()
            running.append(float(loss.detach().cpu()))

        val = evaluate(model, val_loader, criterion, device, tau_vis_values, tau_ir_values, dominance_margin)
        attr_acc = float(val["metrics"]["attribution_accuracy"])
        both_recall = float(val["metrics"]["both_modality_recall"])
        urr = float(val["metrics"]["unnecessary_restoration_ratio_URR"])
        if scheduler_name == "cosine":
            scheduler.step()
        else:
            scheduler.step(attr_acc)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running)) if running else 0.0,
            "val_loss": val["loss"],
            "attr_acc": attr_acc,
            "both_recall": both_recall,
            "urr": urr,
            "best_tau_vis": val["best_thresholds"]["tau_vis"],
            "best_tau_ir": val["best_thresholds"]["tau_ir"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_rows.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"attr_acc={attr_acc:.4f} both_recall={both_recall:.4f} urr={urr:.4f} "
            f"tau=({row['best_tau_vis']:.2f},{row['best_tau_ir']:.2f})"
        )

        score = attr_acc
        if score > best_score:
            best_score = score
            bad_epochs = 0
            best_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "normalizer": normalizer.state_dict(),
                "config": cfg,
                "best_thresholds": val["best_thresholds"],
                "metrics": {k: v for k, v in val["metrics"].items() if k not in {"confusion_matrix_4x4", "y_pred_cls"}},
                "confusion_matrix": val["metrics"]["confusion_matrix_4x4"].tolist(),
                "run_name": run_name,
            }
            torch.save(best_payload, output_dir / "best.pt")
            write_csv(val["sweep_rows"], output_dir / "threshold_sweep.csv")
            write_csv(matrix_to_rows(val["metrics"]["confusion_matrix_4x4"]), output_dir / "confusion_matrix.csv")
        else:
            bad_epochs += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "normalizer": normalizer.state_dict(),
                "config": cfg,
                "last_thresholds": val["best_thresholds"],
                "run_name": run_name,
            },
            output_dir / "last.pt",
        )
        write_csv(log_rows, output_dir / "training_log.csv")
        if bad_epochs >= patience:
            print(f"[DRF-MA] early stopping at epoch {epoch} (patience={patience})")
            break

    if best_payload is None:
        raise RuntimeError("Training finished without a best checkpoint.")
    (output_dir / "run_summary.json").write_text(json.dumps(best_payload["metrics"], indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "best_score": best_score,
        "best_epoch": best_payload["epoch"],
        "best_thresholds": best_payload["best_thresholds"],
        "metrics": best_payload["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DRF-MA attribution")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "drf_ma.yaml"))
    parser.add_argument("--paths", default=None)
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug limit over proposal records.")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override train.num_workers.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.paths)
    if args.rebuild_cache:
        cfg.setdefault("cache", {})["rebuild"] = True
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg.setdefault("train", {})["num_workers"] = args.num_workers
    result = train_from_config(cfg, run_name=args.run_name, max_samples=args.max_samples)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
