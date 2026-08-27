"""Stable hashing and atomic artifact contracts for canonical runs."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported canonical JSON type: {type(value)!r}")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize without operational whitespace or key-order ambiguity."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def protocol_hash(scientific_spec: Mapping[str, Any]) -> str:
    """Hash scientific decisions only; never include code/environment."""
    return sha256_bytes(canonical_json_bytes(scientific_spec))


def semantic_config_hash(
    *,
    protocol_hash_value: str,
    selected_architecture: str,
    weight_checksum: str,
    fold: int,
    feature_set: str,
    resolved_runtime_config: Mapping[str, Any],
    environment_hash: str,
    implementation_commit: str,
) -> str:
    return sha256_bytes(canonical_json_bytes({
        "protocol_hash": protocol_hash_value,
        "selected_architecture": selected_architecture,
        "weight_checksum": weight_checksum,
        "fold": int(fold),
        "feature_set": feature_set,
        "resolved_runtime_config": resolved_runtime_config,
        "environment_hash": environment_hash,
        "implementation_commit": implementation_commit,
    }))


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(module: torch.nn.Module) -> str:
    """Hash parameters and persistent buffers in deterministic key order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def git_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_is_dirty(project_root: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def git_paths_are_dirty(project_root: Path, paths: list[str]) -> bool:
    """Check implementation paths while ignoring unrelated documents/results."""
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_runtime_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    expected_bytes = canonical_json_bytes(expected)
    actual_bytes = canonical_json_bytes(actual)
    if expected_bytes != actual_bytes:
        raise RuntimeError(
            "Runtime configuration differs from the canonical scientific specification. "
            f"expected={sha256_bytes(expected_bytes)}, actual={sha256_bytes(actual_bytes)}"
        )
