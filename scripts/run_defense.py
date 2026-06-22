#!/usr/bin/env python3
"""Run selective modality defense inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.common import ensure_dir
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selective Modality Defense inference")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--paths", default=None, help="Optional local paths YAML")
    parser.add_argument("--visible", default=None, help="Visible image for paired inference")
    parser.add_argument("--infrared", default=None, help="Infrared image for paired inference")
    parser.add_argument("--single", action="store_true", help="Run single-branch inference")
    parser.add_argument("--mode", choices=["vis", "ir"], default=None, help="Single-branch mode; implies --single")
    parser.add_argument("--modality", choices=["vis", "ir"], default=None)
    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.paths)
    set_seed(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(args.output)

    single_mode = args.mode or args.modality
    if args.single or args.mode is not None:
        if not args.image or not single_mode:
            raise ValueError("single-branch inference requires --image and --mode {vis,ir}")
        from src.pipeline.inference import run_single

        summary = run_single(cfg, args.image, single_mode, output_dir)
    else:
        if not args.visible or not args.infrared:
            raise ValueError("paired inference requires --visible and --infrared")
        from src.pipeline.inference import run_pair

        summary = run_pair(cfg, args.visible, args.infrared, output_dir)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
