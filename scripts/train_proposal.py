#!/usr/bin/env python3
"""Train the four-channel anomaly proposal model through the legacy wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.proposal.anomaly_simulation import load_legacy_training_function
from src.utils.common import require_legacy_code_root
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train four-channel proposal detector")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--paths", default=None)
    args, legacy_args = parser.parse_known_args()
    cfg = load_config(args.config, args.paths)
    train = load_legacy_training_function(require_legacy_code_root(cfg))
    sys.argv = [sys.argv[0], *legacy_args]
    train()


if __name__ == "__main__":
    main()
