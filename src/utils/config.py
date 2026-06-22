"""Configuration loading helpers.

The public code keeps private machine paths outside Python files. Runtime paths
are read from YAML, usually `configs/default.yaml` plus a local
`configs/paths.yaml` copied from `configs/paths.yaml.example`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableMapping

import yaml


ConfigDict = dict[str, Any]


def _deep_update(base: MutableMapping[str, Any], update: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, MutableMapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def read_yaml(path: str | Path) -> ConfigDict:
    """Read a YAML file as a dictionary."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_config(config_path: str | Path, paths_path: str | Path | None = None) -> ConfigDict:
    """Load the main config and optionally merge a local paths config."""
    config_path = Path(config_path)
    cfg = read_yaml(config_path)
    project_root = config_path.parent.parent

    include = cfg.get("paths", {}).get("include")
    resolved_paths_path = Path(paths_path) if paths_path else None
    if resolved_paths_path is None and include:
        candidate = Path(include)
        resolved_paths_path = candidate if candidate.is_absolute() else project_root / candidate

    if resolved_paths_path is not None and resolved_paths_path.exists():
        paths_cfg = read_yaml(resolved_paths_path)
        cfg.setdefault("paths", {})
        cfg["paths"].pop("include", None)
        _deep_update(cfg["paths"], paths_cfg)

    return cfg


def cfg_get(cfg: ConfigDict, dotted_key: str, default: Any = None) -> Any:
    """Read a nested config value with a dotted key."""
    cur: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

