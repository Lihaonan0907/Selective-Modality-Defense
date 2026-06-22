#!/usr/bin/env python3
"""Run single-branch proposal inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.proposal import SingleBranchProposal
from src.proposal.proposal_utils import boxes_to_mask
from src.utils.common import ensure_dir
from src.utils.config import load_config
from src.utils.image_io import read_infrared_gray, read_visible_bgr, save_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-branch proposal inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default=None)
    parser.add_argument("--mode", choices=["vis", "ir"], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config, args.paths)
    paths = cfg.get("paths", {})
    single_ckpts = paths.get("checkpoints", {}).get("single_branch", {})
    proposal_cfg = dict(cfg.get("proposal", {}))
    proposal_cfg.update(cfg.get("single_branch", {}).get("proposal", {}))
    proposal_cfg["device"] = cfg.get("device", "cuda:0")
    proposal_cfg["checkpoint"] = single_ckpts.get(f"{args.mode}_proposal") or paths.get("checkpoints", {}).get(f"{args.mode}_proposal")

    detector = SingleBranchProposal(proposal_cfg, args.mode)
    image = read_infrared_gray(args.image) if args.mode == "ir" else read_visible_bgr(args.image)
    proposals = detector.predict(image)
    mask = boxes_to_mask(
        image.shape[:2],
        [item["box"] for item in proposals],
        dilation_kernel=int(cfg.get("single_branch", {}).get("mask", {}).get("dilation_kernel", 5)),
        min_area=int(cfg.get("single_branch", {}).get("mask", {}).get("min_component_area", 80)),
    )

    output_dir = ensure_dir(args.output)
    (output_dir / "proposals.json").write_text(json.dumps(proposals, indent=2), encoding="utf-8")
    save_image(output_dir / "proposal_mask.png", mask)
    print(json.dumps({"num_proposals": len(proposals), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
