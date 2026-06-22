"""Small shared helpers."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a `Path`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_legacy_code_root(path: str | Path | None) -> Path | None:
    """Add the original research implementation directory to `sys.path`."""
    if path is None or str(path) == "":
        return None
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"legacy_code_root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def require_legacy_code_root(cfg: dict) -> str:
    """Return the configured legacy code root with a release-friendly error."""
    legacy_root = cfg.get("paths", {}).get("legacy_code_root")
    if not legacy_root:
        raise ValueError(
            "Missing paths.legacy_code_root. Copy configs/paths.yaml.example to "
            "configs/paths.yaml and fill this value before using legacy training wrappers."
        )
    return str(legacy_root)


def require_path(path: str | Path | None, name: str) -> Path:
    """Validate a required runtime path."""
    if path is None or str(path) == "":
        raise ValueError(f"Missing required path: {name}")
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path
