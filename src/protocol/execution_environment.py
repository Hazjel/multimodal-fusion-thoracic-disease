"""Per-stage environment locks for fair canonical comparisons."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from src.protocol.contracts import read_json
from src.protocol.registry import read_registry


class EnvironmentConsistencyError(RuntimeError):
    pass


_STAGE_DIRECTORIES = {
    "C1": ("screening", "tabular"),
    "C2": ("screening", "image"),
    "C4": ("main",),
    "C5": ("ablation",),
}


def environment_lock_path(protocol_dir: Path, stage: str) -> Path:
    if stage not in _STAGE_DIRECTORIES:
        raise EnvironmentConsistencyError(f"No environment-lock policy for {stage}")
    return Path(protocol_dir).joinpath(*_STAGE_DIRECTORIES[stage], "environment_lock.json")


def ensure_stage_environment(
    *,
    protocol_dir: Path,
    stage: str,
    protocol_hash: str,
    environment_hash: str,
    environment: Dict[str, Any],
    implementation_commit: str,
) -> Dict[str, Any]:
    """Create the first-run lock or reject a different environment hash."""
    path = environment_lock_path(protocol_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "LOCKED",
        "stage": stage,
        "protocol_hash": protocol_hash,
        "environment_hash": environment_hash,
        "environment": environment,
        "first_implementation_commit": implementation_commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not path.exists():
        # Exclusive creation prevents two candidates from silently choosing
        # different first environments. fsync makes a successful lock durable.
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            pass
    locked = read_json(path)
    if locked.get("status") != "LOCKED" or locked.get("stage") != stage:
        raise EnvironmentConsistencyError(f"Invalid environment lock: {path}")
    if locked.get("protocol_hash") != protocol_hash:
        raise EnvironmentConsistencyError("Environment lock belongs to another protocol")
    if locked.get("environment_hash") != environment_hash:
        raise EnvironmentConsistencyError(
            f"{stage} environment mismatch: locked={locked.get('environment_hash')}, "
            f"current={environment_hash}. Repeat the run in the locked environment."
        )
    return locked


def assert_registered_runs_match_environment(
    *,
    protocol_dir: Path,
    stage: str,
    run_ids: Iterable[str],
) -> str:
    """C3 guard: every C2 fold must match the single stage environment."""
    lock = read_json(environment_lock_path(protocol_dir, stage))
    expected = str(lock.get("environment_hash", ""))
    registry = {
        row["run_id"]: row
        for row in read_registry(Path(protocol_dir) / "experiment_registry.csv")
    }
    missing = []
    mismatched = []
    for run_id in sorted(set(map(str, run_ids))):
        row = registry.get(run_id)
        if row is None:
            missing.append(run_id)
        elif row.get("phase") != stage or row.get("environment_hash") != expected:
            mismatched.append(run_id)
    if missing or mismatched:
        raise EnvironmentConsistencyError(
            "C3 rejected inconsistent C2 evidence; "
            f"missing_registry={missing}, mismatched_environment={mismatched}"
        )
    return expected
