"""A tiny file-based model registry.

Deliberately dependency-free (JSON + local files) so the project runs anywhere.
In production this is where you'd swap in MLflow / SageMaker Model Registry —
the interface (``register`` / ``latest`` / ``list_versions``) stays the same.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import artifact_dir


class ModelRegistry:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else artifact_dir() / "registry"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"models": {}}, indent=2))

    def _index(self) -> dict:
        return json.loads(self.index_path.read_text())

    def register(self, name: str, artifact_path: str, metrics: dict, stage: str = "staging") -> dict:
        idx = self._index()
        versions = idx["models"].setdefault(name, [])
        entry = {
            "version": len(versions) + 1,
            "artifact_path": str(artifact_path),
            "metrics": metrics,
            "stage": stage,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        versions.append(entry)
        self.index_path.write_text(json.dumps(idx, indent=2))
        return entry

    def latest(self, name: str) -> dict | None:
        versions = self._index()["models"].get(name, [])
        return versions[-1] if versions else None

    def list_versions(self, name: str) -> list[dict]:
        return self._index()["models"].get(name, [])

    def promote(self, name: str, version: int, stage: str = "production") -> dict | None:
        idx = self._index()
        for v in idx["models"].get(name, []):
            if v["version"] == version:
                v["stage"] = stage
                self.index_path.write_text(json.dumps(idx, indent=2))
                return v
        return None
