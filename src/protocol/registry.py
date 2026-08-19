"""Atomic experiment registry for canonical runs."""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


REGISTRY_FIELDS = [
    "run_id", "phase", "scenario", "model", "fold", "feature_set",
    "pretraining", "protocol_hash", "semantic_config_hash",
    "implementation_commit", "environment_hash", "pos_weight",
    "best_epoch", "best_validation_auc", "status", "artifact_path",
]


def read_registry(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def upsert_registry(path: Path, row: Mapping[str, object]) -> None:
    unknown = set(row) - set(REGISTRY_FIELDS)
    if unknown:
        raise ValueError(f"Unknown registry fields: {sorted(unknown)}")
    run_id = str(row.get("run_id", ""))
    if not run_id:
        raise ValueError("Registry row requires run_id")
    existing = read_registry(path)
    normalized = {field: str(row.get(field, "")) for field in REGISTRY_FIELDS}
    rows = [item for item in existing if item.get("run_id") != run_id]
    rows.append(normalized)
    rows.sort(key=lambda item: item["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
