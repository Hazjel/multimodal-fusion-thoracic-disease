"""Environment provenance separate from the scientific protocol identity."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

from src.protocol.contracts import canonical_json_bytes, sha256_bytes
from src.protocol.cuda_reproducibility import CUBLAS_WORKSPACE_ENV


PACKAGES = (
    "torch", "torchvision", "numpy", "pandas", "scikit-learn", "scipy",
    "shap", "pytabkit", "Pillow", "opencv-python", "matplotlib",
)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def collect_environment(implementation_commit: str) -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cudnn_version = torch.backends.cudnn.version() if cuda_available else None
    gpu = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "implementation_commit": implementation_commit,
        "os": platform.platform(),
        "python": sys.version,
        "packages": {name: _version(name) for name in PACKAGES},
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn": cudnn_version,
        "gpu": gpu,
        "determinism": {
            "base_seed": 42,
            "cublas_workspace_config": os.environ.get(CUBLAS_WORKSPACE_ENV),
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "deterministic_algorithms": True,
            "warn_only": True,
        },
    }


def environment_hash(environment: Dict[str, Any]) -> str:
    scientific_environment = dict(environment)
    scientific_environment.pop("implementation_commit", None)
    return sha256_bytes(canonical_json_bytes(scientific_environment))


def pip_freeze() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())
    return "\n".join(lines) + "\n"
