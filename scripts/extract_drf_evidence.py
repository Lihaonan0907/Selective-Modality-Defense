#!/usr/bin/env python3
"""Extract DRF-MA evidence records and optional debug visualizations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_drf_ma import materialize_records, write_csv
from src.attribution.drf_ma import load_drfma_checkpoint
from src.attribution.evidence_signals import (
    EvidenceExtractor,
    box_to_mask,
    read_patch_annotation,
    record_to_evidence_tensors,
    safe_imread,
    save_jsonl,
)
from src.attribution.mask_generation import decide_attribution
from src.utils.common import ensure_dir
from src.utils.config import load_config


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    if mask is None or cv2.countNonZero(mask) == 0:
        return out
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    color_layer = np.zeros_like(out)
    color_layer[:, :] = color
    out[mask > 0] = cv2.addWeighted(out, 0.55, color_layer, 0.45, 0)[mask > 0]
    return out


def _load_model(config: dict[str, Any], checkpoint_arg: str | None):
    checkpoint = checkpoint_arg
    if checkpoint is None:
        candidate = Path(config.get("output_dir", "outputs/drf_ma")) / "best.pt"
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        checkpoint = str(candidate) if candidate.exists() else None
    if checkpoint is None:
        return None, None, None
    model, normalizer, ckpt = load_drfma_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    return model, normalizer, ckpt


def _predict_record(model, normalizer, record: dict[str, Any]) -> tuple[float | None, float | None]:
    if model is None or normalizer is None:
        return None, None
    evidence = record_to_evidence_tensors(record)
    evidence = normalizer.transform(evidence)
    evidence = {k: v.unsqueeze(0) for k, v in evidence.items()}
    with torch.no_grad():
        out = model(evidence)
    return float(out["p_vis"].item()), float(out["p_ir"].item())


def _draw_debug(record: dict[str, Any], out_path: Path, p_vis: float | None, p_ir: float | None, cfg: dict[str, Any]) -> None:
    vis = safe_imread(record["visible_image"], cv2.IMREAD_COLOR)
    ir = safe_imread(record["infrared_image"], cv2.IMREAD_GRAYSCALE)
    if ir.shape[:2] != vis.shape[:2]:
        ir = cv2.resize(ir, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_LINEAR)
    h, w = ir.shape[:2]
    box = [int(round(v)) for v in record["proposal_box"]]
    proposal_mask = box_to_mask((h, w), box)

    if record.get("is_clean_pair"):
        vis_patch = np.zeros((h, w), dtype=np.uint8)
        ir_patch = np.zeros((h, w), dtype=np.uint8)
    else:
        vis_patch_ann = read_patch_annotation(record.get("visible_patch_label"), (h, w))
        ir_patch_ann = read_patch_annotation(record.get("infrared_patch_label"), (h, w))
        vis_patch = vis_patch_ann.mask if vis_patch_ann.mask is not None else np.zeros((h, w), dtype=np.uint8)
        ir_patch = ir_patch_ann.mask if ir_patch_ann.mask is not None else np.zeros((h, w), dtype=np.uint8)
        for patch_box in vis_patch_ann.boxes:
            vis_patch = cv2.bitwise_or(vis_patch, box_to_mask((h, w), patch_box))
        for patch_box in ir_patch_ann.boxes:
            ir_patch = cv2.bitwise_or(ir_patch, box_to_mask((h, w), patch_box))

    vis_panel = _overlay_mask(vis, vis_patch, (0, 0, 255))
    vis_panel = _overlay_mask(vis_panel, proposal_mask, (0, 255, 255))
    ir_panel = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    ir_panel = _overlay_mask(ir_panel, ir_patch, (255, 0, 0))
    ir_panel = _overlay_mask(ir_panel, proposal_mask, (0, 255, 255))
    for panel in (vis_panel, ir_panel):
        cv2.rectangle(panel, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)

    label = f"GT=({record['label_vis']},{record['label_ir']}) {record['case_type']}"
    if p_vis is None or p_ir is None:
        pred_text = "P_vis=n/a P_ir=n/a"
    else:
        pred = decide_attribution(
            p_vis,
            p_ir,
            float(cfg.get("thresholds", {}).get("tau_vis", 0.90)),
            float(cfg.get("thresholds", {}).get("tau_ir", 0.65)),
            float(cfg.get("thresholds", {}).get("dominance_margin", 0.10)),
        )
        pred_text = f"P_vis={p_vis:.3f} P_ir={p_ir:.3f} pred={pred}"
    cv2.putText(vis_panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_panel, pred_text, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ir_panel, "IR", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    canvas = np.concatenate([vis_panel, ir_panel], axis=1)
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DRF-MA evidence")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "drf_ma.yaml"))
    parser.add_argument("--paths", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.paths)
    if args.rebuild_cache:
        cfg.setdefault("cache", {})["rebuild"] = True
    output_dir = Path(args.output or Path(cfg.get("output_dir", "outputs/drf_ma")) / "debug_evidence")
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    ensure_dir(output_dir)

    records = materialize_records(
        cfg,
        args.split,
        EvidenceExtractor(cfg.get("evidence", {})),
        rebuild_cache=bool(args.rebuild_cache),
        max_samples=args.max_samples,
    )
    model, normalizer, _ = _load_model(cfg, args.checkpoint)
    debug_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        p_vis, p_ir = _predict_record(model, normalizer, record)
        row = {
            "idx": idx,
            "attack_type": record.get("attack_type"),
            "stem": record.get("stem"),
            "proposal_index": record.get("proposal_index"),
            "proposal_box": record.get("proposal_box"),
            "label_vis": record.get("label_vis"),
            "label_ir": record.get("label_ir"),
            "case_type": record.get("case_type"),
            "p_vis": p_vis,
            "p_ir": p_ir,
        }
        for key in ("z_cm_vis", "z_cm_ir", "z_self_vis", "z_self_ir", "z_rep_vis", "z_rep_ir", "z_det_vis", "z_det_ir"):
            row[key] = record.get(key)
        debug_rows.append(row)
        if args.save_debug:
            _draw_debug(record, output_dir / "visualizations" / f"{idx:04d}_{record.get('attack_type')}_{record.get('stem')}.jpg", p_vis, p_ir, cfg)

    save_jsonl(debug_rows, output_dir / f"{args.split}_evidence_debug.jsonl")
    write_csv(debug_rows, output_dir / f"{args.split}_evidence_debug.csv")
    print(json.dumps({"records": len(records), "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
