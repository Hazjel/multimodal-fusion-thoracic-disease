"""Stage-gated command-line entry point for the canonical protocol."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# This must run before imports that load torch and before any CUDA operation.
from src.protocol.cuda_reproducibility import configure_cublas_workspace

configure_cublas_workspace()

from configs.config import cfg
from src.protocol.chexnet import write_chexnet_provenance_audit
from src.protocol.freeze import freeze_protocol
from src.protocol.stages import (
    finalize_cv_stage_if_complete,
    load_model_lock,
    oof_path_for,
    stage_status,
)
from src.training.cv import run_cross_validation
from src.training.deployment import (
    OFFICIAL_TEST_CONFIRMATION,
    run_deployment_refit,
    run_secondary_holdout,
)
from src.training.model_selection import create_model_lock
from src.training.tabular_benchmark import TABULAR_MODELS, run_tabular_benchmark


def _default_protocol_dir() -> Path:
    candidates = sorted(
        path.parent
        for path in cfg.paths.canonical_dir.glob("*/protocol.json")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Pass --protocol-dir explicitly when zero or multiple frozen protocols exist"
        )
    return candidates[0]


def _add_protocol_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol-dir", type=Path, default=None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NIH multimodal canonical protocol")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("c0", help="Run non-performance C0 acceptance tests")
    freeze = subcommands.add_parser("freeze", help="Generate frozen artifacts after C0 PASS")
    freeze.add_argument(
        "--c0-report",
        type=Path,
        default=cfg.paths.results_dir / "c0" / "c0_acceptance.json",
    )

    status = subcommands.add_parser("status", help="Show canonical stage readiness")
    _add_protocol_dir(status)

    tabular = subcommands.add_parser("benchmark-tabular", help="Run C1 tabular benchmark")
    _add_protocol_dir(tabular)
    tabular.add_argument("--model", choices=["all", *TABULAR_MODELS], default="all")
    tabular.add_argument("--device", choices=["cpu", "cuda"], default=None)

    screen = subcommands.add_parser("screen-image", help="Run guarded C2 CNN screening")
    _add_protocol_dir(screen)
    screen.add_argument(
        "--backbone", choices=["all", *cfg.model.image_candidates], default="all"
    )
    screen.add_argument("--pretraining", choices=["imagenet", "chexnet"], default="imagenet")

    select = subcommands.add_parser("select", help="Apply frozen C3 model-selection rule")
    _add_protocol_dir(select)

    audit_chexnet = subcommands.add_parser(
        "audit-chexnet", help="Audit conditional CheXNet checkpoint provenance"
    )
    _add_protocol_dir(audit_chexnet)
    audit_chexnet.add_argument("--declaration", type=Path, default=None)

    main = subcommands.add_parser("main", help="Run C4 locked S1/S2/S3 experiments")
    _add_protocol_dir(main)
    main.add_argument("--scenario", choices=["all", "S1", "S2", "S3"], default="all")

    ablate = subcommands.add_parser("ablate", help="Run C5 metadata ablation A/B/C")
    _add_protocol_dir(ablate)
    ablate.add_argument("--scenario", choices=["both", "S1", "S3"], default="both")
    ablate.add_argument("--feature-set", choices=["all", "A", "B", "C"], default="all")

    c6 = subcommands.add_parser(
        "c6", help="Run canonical C6 OOF statistics, SHAP, and Grad-CAM"
    )
    _add_protocol_dir(c6)
    c6.add_argument(
        "--component", choices=["all", "statistics", "shap", "gradcam"], default="all"
    )
    c6.add_argument("--device", choices=["cpu", "cuda"], default=None)
    c6.add_argument("--shap-nsamples", type=int, default=128)

    c7 = subcommands.add_parser(
        "c7", help="Run deployment refit or explicitly open the secondary holdout"
    )
    _add_protocol_dir(c7)
    c7.add_argument("--phase", choices=["refit", "evaluate"], default="refit")
    c7.add_argument("--scenario", choices=["all", "S1", "S2", "S3"], default="all")
    c7.add_argument("--device", choices=["cpu", "cuda"], default=None)
    c7.add_argument(
        "--confirm-official-test-access",
        default=None,
        help=f"Required only for evaluate; exact value: {OFFICIAL_TEST_CONFIRMATION}",
    )
    return parser


def _protocol_dir(args: argparse.Namespace) -> Path:
    return Path(args.protocol_dir) if args.protocol_dir is not None else _default_protocol_dir()


def _run_screen(protocol_dir: Path, backbone: str, pretraining: str) -> None:
    run_cross_validation(
        "S2",
        stage="C2",
        protocol_dir=protocol_dir,
        backbone_name=backbone,
        pretraining=pretraining,
        feature_set="D",
    )


def main() -> int:
    args = _parser().parse_args()
    if args.command == "c0":
        return subprocess.call([sys.executable, "scripts/run_c0.py"], cwd=cfg.paths.project_root)
    if args.command == "freeze":
        target = freeze_protocol(args.c0_report)
        print(f"Frozen protocol artifacts: {target}")
        return 0

    protocol_dir = _protocol_dir(args)
    if args.command == "status":
        print(json.dumps(stage_status(protocol_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "audit-chexnet":
        audit = write_chexnet_provenance_audit(
            protocol_dir=protocol_dir,
            declaration_path=args.declaration,
        )
        print(json.dumps(audit, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "benchmark-tabular":
        models = TABULAR_MODELS if args.model == "all" else (args.model,)
        result = run_tabular_benchmark(
            protocol_dir=protocol_dir, models=models, device=args.device
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "screen-image":
        if args.pretraining == "chexnet" and args.backbone != "densenet121":
            raise RuntimeError("Conditional CheXNet screening only allows DenseNet-121")
        backbones = cfg.model.image_candidates if args.backbone == "all" else (args.backbone,)
        if args.pretraining == "chexnet" and len(backbones) != 1:
            raise RuntimeError("CheXNet is not an all-backbone screening option")
        for backbone in backbones:
            _run_screen(protocol_dir, backbone, args.pretraining)
        if args.pretraining == "imagenet" and all(
            oof_path_for(
                protocol_dir,
                stage="C2",
                scenario="S2",
                model=name,
                pretraining="imagenet",
                feature_set="D",
            ).exists()
            for name in cfg.model.image_candidates
        ):
            marker = protocol_dir / "screening" / "image" / "_SUCCESS"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("C2 ImageNet screening complete\n", encoding="utf-8")
        return 0
    if args.command == "select":
        print(json.dumps(create_model_lock(protocol_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "main":
        lock = load_model_lock(protocol_dir)
        scenarios = ("S1", "S2", "S3") if args.scenario == "all" else (args.scenario,)
        for scenario in scenarios:
            if scenario == "S1":
                backbone, pretraining = "canonical_mlp", "not_applicable"
            else:
                backbone = lock["selected_backbone"]
                pretraining = lock["selected_pretraining"]
            run_cross_validation(
                scenario,
                stage="C4",
                protocol_dir=protocol_dir,
                backbone_name=backbone,
                pretraining=pretraining,
                feature_set="D",
            )
        finalize_cv_stage_if_complete(
            protocol_dir,
            stage="C4",
            backbone=lock["selected_backbone"],
            pretraining=lock["selected_pretraining"],
        )
        return 0
    if args.command == "ablate":
        lock = load_model_lock(protocol_dir)
        scenarios = ("S1", "S3") if args.scenario == "both" else (args.scenario,)
        feature_sets = ("A", "B", "C") if args.feature_set == "all" else (args.feature_set,)
        for scenario in scenarios:
            for feature_set in feature_sets:
                if scenario == "S1":
                    backbone, pretraining = "canonical_mlp", "not_applicable"
                else:
                    backbone = lock["selected_backbone"]
                    pretraining = lock["selected_pretraining"]
                run_cross_validation(
                    scenario,
                    stage="C5",
                    protocol_dir=protocol_dir,
                    backbone_name=backbone,
                    pretraining=pretraining,
                    feature_set=feature_set,
                )
        finalize_cv_stage_if_complete(
            protocol_dir,
            stage="C5",
            backbone=lock["selected_backbone"],
            pretraining=lock["selected_pretraining"],
        )
        return 0
    if args.command == "c6":
        import torch
        from src.reporting.c6 import run_c6

        device = None if args.device is None else torch.device(args.device)
        result = run_c6(
            protocol_dir,
            component=args.component,
            device=device,
            shap_nsamples=args.shap_nsamples,
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "c7":
        device = None if args.device is None else __import__("torch").device(args.device)
        if args.phase == "refit":
            scenarios = ("S1", "S2", "S3") if args.scenario == "all" else (args.scenario,)
            result = run_deployment_refit(
                protocol_dir,
                scenarios=scenarios,
                device=device,
            )
        else:
            if args.scenario != "all":
                raise RuntimeError("C7 official holdout must evaluate frozen S1/S2/S3 together")
            result = run_secondary_holdout(
                protocol_dir,
                confirmation=args.confirm_official_test_access or "",
                device=device,
            )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
