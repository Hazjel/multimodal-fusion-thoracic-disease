"""Safe command-line entry point for the canonical execution protocol."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from configs.config import cfg
from src.protocol.freeze import freeze_protocol
from src.training.cv import run_cross_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NIH multimodal canonical protocol")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("c0", help="Run non-performance C0 acceptance tests")
    freeze = subcommands.add_parser("freeze", help="Generate frozen protocol artifacts after C0 PASS")
    freeze.add_argument(
        "--c0-report",
        type=Path,
        default=cfg.paths.results_dir / "c0" / "c0_acceptance.json",
    )
    cv = subcommands.add_parser("cv", help="Run a canonical manifest-driven CV phase")
    cv.add_argument("--protocol-dir", type=Path, required=True)
    cv.add_argument("--scenario", choices=["S1", "S2", "S3"], required=True)
    cv.add_argument("--backbone", choices=list(cfg.model.image_candidates), default="densenet121")
    cv.add_argument("--pretraining", choices=["imagenet", "chexnet"], default="imagenet")
    cv.add_argument("--feature-set", choices=["A", "B", "C", "D"], default="D")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "c0":
        return subprocess.call([sys.executable, "scripts/run_c0.py"], cwd=cfg.paths.project_root)
    if args.command == "freeze":
        target = freeze_protocol(args.c0_report)
        print(f"Frozen protocol artifacts: {target}")
        return 0
    summary = run_cross_validation(
        args.scenario,
        protocol_dir=args.protocol_dir,
        backbone_name=args.backbone,
        pretraining=args.pretraining,
        feature_set=args.feature_set,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
