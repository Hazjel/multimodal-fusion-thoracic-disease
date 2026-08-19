"""Promote a passing C0 implementation to a frozen protocol artifact."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from configs.config import cfg
from src.data.dataset import build_image_index, load_and_prepare_metadata
from src.protocol.contracts import (
    atomic_write_json,
    file_sha256,
    git_commit,
    git_is_dirty,
    protocol_hash,
    read_json,
)
from src.protocol.environment import collect_environment, environment_hash, pip_freeze
from src.protocol.manifests import generate_manifests
from src.protocol.registry import REGISTRY_FIELDS
from src.protocol.spec import build_scientific_spec


def _write_registry_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(REGISTRY_FIELDS)


def freeze_protocol(c0_report_path: Path) -> Path:
    c0_report = read_json(c0_report_path)
    if c0_report.get("status") != "PASS":
        raise RuntimeError("Protocol freeze requires a PASS C0 acceptance report")
    if git_is_dirty(cfg.paths.project_root):
        raise RuntimeError(
            "Freeze requires a clean Git worktree so implementation_commit is unambiguous"
        )
    implementation_commit = git_commit(cfg.paths.project_root)
    if c0_report.get("implementation_commit") != implementation_commit:
        raise RuntimeError("C0 report was not produced from the current implementation commit")

    environment = collect_environment(implementation_commit)
    if environment["packages"].get("pytabkit") == "NOT_INSTALLED":
        raise RuntimeError("C0 cannot freeze until pytabkit is installed and smoke-tested")
    environment_hash_value = environment_hash(environment)

    staging_root = cfg.paths.canonical_dir / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="protocol-v1.0.0-", dir=staging_root))
    try:
        image_index = build_image_index(cfg.paths.image_dirs)
        metadata = load_and_prepare_metadata(cfg.paths.csv_path, image_index)
        manifests = generate_manifests(metadata, staging)
        scientific_spec = build_scientific_spec(
            fold_manifest_hash=manifests["fold_manifest_hash"],
            deployment_split_hash=manifests["deployment_split_hash"],
        )
        protocol_hash_value = protocol_hash(scientific_spec)
        protocol = {
            "status": "FROZEN",
            "protocol_hash": protocol_hash_value,
            "scientific_spec": scientific_spec,
            "provenance": {
                "freeze_candidate": cfg.protocol_candidate,
                "implementation_commit": implementation_commit,
                "environment_hash": environment_hash_value,
                "c0_report_hash": file_sha256(c0_report_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        atomic_write_json(staging / "protocol.json", protocol)
        atomic_write_json(staging / "environment.json", environment)
        atomic_write_json(staging / "data_audit.json", manifests["audit"])
        (staging / "pip-freeze.txt").write_text(pip_freeze(), encoding="utf-8", newline="\n")
        _write_registry_header(staging / "experiment_registry.csv")
        for directory in (
            "screening", "main", "ablation", "xai", "statistics",
            "secondary_holdout", "deployment", "runs", "oof",
        ):
            (staging / directory).mkdir(parents=True, exist_ok=True)

        checksum_targets = [
            "protocol.json", "environment.json", "data_audit.json", "pip-freeze.txt",
            "folds.csv", "deployment_split.csv", "experiment_registry.csv",
        ]
        checksums = {name: file_sha256(staging / name) for name in checksum_targets}
        atomic_write_json(staging / "checksums.json", checksums)
        (staging / "_SUCCESS").write_text("C0 PASS; protocol frozen\n", encoding="utf-8")

        target = cfg.paths.canonical_dir / protocol_hash_value
        if target.exists():
            raise FileExistsError(f"Frozen protocol directory already exists: {target}")
        os.replace(staging, target)
        return target
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c0-report",
        type=Path,
        default=cfg.paths.results_dir / "c0" / "c0_acceptance.json",
    )
    args = parser.parse_args()
    target = freeze_protocol(args.c0_report)
    print(target)


if __name__ == "__main__":
    main()
