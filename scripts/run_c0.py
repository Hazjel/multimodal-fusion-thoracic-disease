"""Run non-performance C0 acceptance tests and emit a provenance report."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import cfg
from src.protocol.contracts import atomic_write_json, git_commit


def main() -> int:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    report = {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "candidate": cfg.protocol_candidate,
        "implementation_commit": git_commit(PROJECT_ROOT),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "note": "C0 validates implementation contracts only; smoke-test AUC is not used for design decisions.",
    }
    path = cfg.paths.results_dir / "c0" / "c0_acceptance.json"
    atomic_write_json(path, report)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    print(f"C0 {report['status']}: {path}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
