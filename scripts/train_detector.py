#!/usr/bin/env python3
"""Train task-driven single-modal detectors."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import add_legacy_code_root, require_legacy_code_root
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train task-driven detector")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--paths", default=None)
    args, legacy_args = parser.parse_known_args()
    cfg = load_config(args.config, args.paths)
    add_legacy_code_root(require_legacy_code_root(cfg))
    sys.argv = [sys.argv[0], *legacy_args]
    runpy.run_module("train_module_c_clean_frozen", run_name="__main__")


if __name__ == "__main__":
    main()
