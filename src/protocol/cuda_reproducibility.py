"""CUDA environment requirements for canonical deterministic execution.

This module intentionally imports no torch modules.  The cuBLAS workspace
configuration must be present before the first CUDA/cuBLAS operation.
"""
from __future__ import annotations

import os


CUBLAS_WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"
CANONICAL_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_cublas_workspace() -> str:
    """Set the canonical value, rejecting a conflicting caller setting."""
    current = os.environ.get(CUBLAS_WORKSPACE_ENV)
    if current not in {None, CANONICAL_CUBLAS_WORKSPACE_CONFIG}:
        raise RuntimeError(
            f"{CUBLAS_WORKSPACE_ENV} must be "
            f"{CANONICAL_CUBLAS_WORKSPACE_CONFIG!r} for canonical execution; "
            f"received {current!r}"
        )
    os.environ[CUBLAS_WORKSPACE_ENV] = CANONICAL_CUBLAS_WORKSPACE_CONFIG
    return CANONICAL_CUBLAS_WORKSPACE_CONFIG


def require_cublas_workspace(device: object) -> None:
    """Hard-fail CUDA execution if the canonical cuBLAS setting is absent."""
    if not str(device).lower().startswith("cuda"):
        return
    actual = os.environ.get(CUBLAS_WORKSPACE_ENV)
    if actual != CANONICAL_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "Canonical CUDA execution is blocked because "
            f"{CUBLAS_WORKSPACE_ENV}={actual!r}; expected "
            f"{CANONICAL_CUBLAS_WORKSPACE_CONFIG!r}. Launch through "
            "run_experiment.py or configure the environment before CUDA use."
        )
