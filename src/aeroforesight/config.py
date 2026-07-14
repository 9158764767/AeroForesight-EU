"""Configuration loading + small shared helpers.

Keeping this dependency-light (only pyyaml) means every other module can import
paths and settings without pulling in heavy ML libraries.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repository root = three levels up from this file (src/aeroforesight/config.py).
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the YAML config once and cache it."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def data_dir() -> Path:
    d = ROOT / os.environ.get("AEROFORESIGHT_DATA_DIR", "data")
    d.mkdir(parents=True, exist_ok=True)
    return d


def artifact_dir() -> Path:
    d = ROOT / os.environ.get("AEROFORESIGHT_ARTIFACT_DIR", "artifacts")
    d.mkdir(parents=True, exist_ok=True)
    return d
