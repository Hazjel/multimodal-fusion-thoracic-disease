"""Canonical C7 deployment refit and guarded secondary-holdout evaluation."""
from __future__ import annotations

import json
import os
import pickle
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from configs.config import cfg
from src.data.dataset import (
    TABULAR_FEATURE_SETS,
    build_image_index,
    load_and_prepare_metadata,
    load_official_partitions,
    load_official_training_pool,
    make_fold_dataloaders,
    make_inference_dataloader,
)
from src.evaluation import (
    calibration_table,
    collect_prediction_frame,
    compute_metrics,
    write_prediction_frame,
)
from src.models.architectures import build_model, build_s2_s3_pair, image_initial_hashes
from src.protocol.contracts import (
    atomic_write_json,
    file_sha256,
    git_commit,
    git_paths_are_dirty,
    semantic_config_hash,
)
from src.protocol.cuda_reproducibility import (
    configure_cublas_workspace,
    require_cublas_workspace,
)
from src.protocol.environment import collect_environment, environment_hash
from src.protocol.execution_environment import ensure_stage_environment
from src.protocol.guards import OfficialTestAccessError, assert_official_test_access
from src.protocol.registry import upsert_registry
from src.protocol.stages import StageGateError, validate_c7_prerequisites
from src.training import save_scaler, train


SCENARIOS = ("S1", "S2", "S3")
OFFICIAL_TEST_CONFIRMATION = "OPEN-OFFICIAL-NIH-TEST"
OFFICIAL_TEST_COLUMNS = ["image_index", "patient_id", "true_label"]
SECONDARY_HOLDOUT_OPERATIONAL_FILES = frozenset({
    "c7_holdout_stdout.log",
    "c7_holdout_stderr.log",
})


def _modalities(scenario: str) -> Sequence[str]:
    return {
        "S1": ("tabular",),
        "S2": ("image",),
        "S3": ("image", "tabular"),
    }[scenario]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _relative(path: Path, protocol_dir: Path) -> str:
    return str(path.resolve().relative_to(Path(protocol_dir).resolve())).replace("\\", "/")


def _path_from_entry(protocol_dir: Path, value: str) -> Path:
    root = Path(protocol_dir).resolve()
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StageGateError(f"Recorded artifact escapes protocol directory: {value}") from exc
    return resolved


def _execution_context(
    protocol_dir: Path,
    device: torch.device,
) -> tuple[Dict[str, Any], str, str, Dict[str, Any]]:
    configure_cublas_workspace()
    require_cublas_workspace(device)
    state = validate_c7_prerequisites(protocol_dir, require_refit=False)
    if git_paths_are_dirty(
        cfg.paths.project_root,
        ["run_experiment.py", "configs", "src"],
    ):
        raise StageGateError(
            "C7 implementation paths are uncommitted; commit the reviewed runner before execution"
        )
    implementation = git_commit(cfg.paths.project_root)
    environment = collect_environment(implementation)
    environment["execution_device"] = str(device)
    environment_hash_value = environment_hash(environment)
    ensure_stage_environment(
        protocol_dir=protocol_dir,
        stage="C7",
        protocol_hash=state["protocol"]["protocol_hash"],
        environment_hash=environment_hash_value,
        environment=environment,
        implementation_commit=implementation,
    )
    return state, implementation, environment_hash_value, environment


def _load_deployment_pool(
    protocol_dir: Path,
) -> tuple[pd.DataFrame, Dict[str, Path], pd.DataFrame]:
    image_index = build_image_index(cfg.paths.image_dirs)
    metadata = load_and_prepare_metadata(cfg.paths.csv_path, image_index)
    training_pool = load_official_training_pool(metadata, cfg.paths.train_list_path)
    manifest = pd.read_csv(Path(protocol_dir) / "deployment_split.csv")
    expected_columns = ["image_index", "patient_id", "true_label", "split"]
    if list(manifest.columns) != expected_columns:
        raise StageGateError(f"Deployment manifest schema mismatch: {list(manifest.columns)}")
    if manifest["image_index"].duplicated().any():
        raise StageGateError("Deployment manifest contains duplicate image_index values")
    if set(manifest["split"]) != {"train", "validation"}:
        raise StageGateError("Deployment manifest must contain train and validation")
    if int(manifest.groupby("patient_id")["split"].nunique().max()) != 1:
        raise StageGateError("Patient leakage detected in deployment manifest")
    lookup = manifest.set_index("image_index")
    if set(training_pool["Image Index"].astype(str)) != set(lookup.index.astype(str)):
        raise StageGateError("Deployment manifest coverage differs from official training pool")
    training_pool = training_pool.copy()
    training_pool["split"] = training_pool["Image Index"].map(lookup["split"])
    expected_labels = training_pool["Image Index"].map(lookup["true_label"]).astype(np.int64)
    expected_patients = training_pool["Image Index"].map(lookup["patient_id"]).astype(np.int64)
    if not np.array_equal(expected_labels.to_numpy(), training_pool["binary_label"].to_numpy()):
        raise StageGateError("Deployment manifest labels differ from canonical metadata")
    if not np.array_equal(expected_patients.to_numpy(), training_pool["Patient ID"].to_numpy()):
        raise StageGateError("Deployment manifest patients differ from canonical metadata")
    return training_pool, image_index, manifest


def _build_deployment_model(
    scenario: str,
    *,
    backbone: str,
    pretraining: str,
    input_dim: int,
) -> tuple[torch.nn.Module, Optional[tuple[str, str]], str]:
    if scenario == "S1":
        return (
            build_model("S1", fold=0, tabular_input_dim=input_dim),
            None,
            "not_applicable",
        )
    s2, s3 = build_s2_s3_pair(
        backbone_name=backbone,
        pretraining=pretraining,
        fold=0,
        tabular_input_dim=input_dim,
    )
    model = s2 if scenario == "S2" else s3
    unused = s3 if scenario == "S2" else s2
    hashes = image_initial_hashes(model)
    if hashes != image_initial_hashes(unused):
        raise AssertionError("Deployment S2/S3 image initialization contract failed")
    branch = model.branch if scenario == "S2" else model.image_branch
    weight_checksum = branch.pretrained_state_checksum
    del unused
    return model, hashes, weight_checksum


def _run_refit_scenario(
    protocol_dir: Path,
    scenario: str,
    *,
    state: Mapping[str, Any],
    implementation: str,
    environment_hash_value: str,
    device: torch.device,
    training_pool: pd.DataFrame,
    image_index: Dict[str, Path],
) -> Dict[str, Any]:
    protocol = state["protocol"]
    lock = state["model_lock"]
    train_frame = training_pool[training_pool["split"] == "train"].reset_index(drop=True)
    validation_frame = training_pool[training_pool["split"] == "validation"].reset_index(drop=True)
    loaders = make_fold_dataloaders(
        train_frame,
        validation_frame,
        image_index,
        modalities=_modalities(scenario),
        feature_set="D",
        seed=cfg.train.seed,
    )
    train_loader, validation_loader, scaler, pos_weight = loaders
    backbone = "canonical_mlp" if scenario == "S1" else lock["selected_backbone"]
    pretraining = "not_applicable" if scenario == "S1" else lock["selected_pretraining"]
    model, initial_hashes, weight_checksum = _build_deployment_model(
        scenario,
        backbone=lock["selected_backbone"],
        pretraining=lock["selected_pretraining"],
        input_dim=len(TABULAR_FEATURE_SETS["D"]),
    )
    resolved = {
        "stage": "C7_DEPLOYMENT_REFIT",
        "scenario": scenario,
        "backbone": backbone,
        "pretraining": pretraining,
        "feature_set": "D",
        "deployment_split_hash": file_sha256(Path(protocol_dir) / "deployment_split.csv"),
        "validation_fold": cfg.data.deployment_validation_fold,
        "runtime": cfg.scientific_runtime_values(),
        "no_new_tuning": True,
    }
    semantic_hash = semantic_config_hash(
        protocol_hash_value=protocol["protocol_hash"],
        selected_architecture=backbone,
        weight_checksum=weight_checksum,
        fold=cfg.data.deployment_validation_fold,
        feature_set="D",
        resolved_runtime_config=resolved,
        environment_hash=environment_hash_value,
        implementation_commit=implementation,
    )
    run_id = f"C7-refit-{scenario}-{backbone}-{pretraining}-D-{semantic_hash[:12]}"
    run_dir = Path(protocol_dir) / "deployment" / "runs" / run_id
    summary_path = Path(protocol_dir) / "deployment" / f"refit_{scenario}.json"
    success_path = run_dir / "_SUCCESS"
    if success_path.is_file() and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    model = model.to(device)
    run_metadata = {
        "run_id": run_id,
        "protocol_hash": protocol["protocol_hash"],
        "semantic_config_hash": semantic_hash,
        "implementation_commit": implementation,
        "environment_hash": environment_hash_value,
        "stage": "C7_DEPLOYMENT_REFIT",
        "scenario": scenario,
        "fold": cfg.data.deployment_validation_fold,
        "feature_set": "D",
        "backbone": backbone,
        "pretraining": pretraining,
        "deployment_split_hash": resolved["deployment_split_hash"],
        "initial_image_hashes": initial_hashes,
        "no_new_tuning": True,
    }
    model = train(
        model,
        train_loader,
        validation_loader,
        pos_weight,
        scenario,
        device,
        run_dir=run_dir,
        run_metadata=run_metadata,
        resume=True,
    )
    if scaler is not None:
        save_scaler(scaler, run_dir / "scaler.pkl")
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    predictions = collect_prediction_frame(
        model,
        validation_loader,
        device,
        fold=cfg.data.deployment_validation_fold,
        split="deployment_validation",
        model_name=backbone,
        scenario=scenario,
        feature_set="D",
        pretraining=pretraining,
        checkpoint_epoch=int(checkpoint["best_epoch"]),
        protocol_hash=protocol["protocol_hash"],
        semantic_config_hash=semantic_hash,
        run_id=run_id,
    )
    expected_images = set(validation_frame["Image Index"].astype(str))
    if set(predictions["image_index"].astype(str)) != expected_images or len(predictions) != len(expected_images):
        raise RuntimeError(f"Deployment validation coverage mismatch for {scenario}")
    prediction_path = run_dir / "validation_predictions.csv"
    metrics_path = run_dir / "validation_metrics.json"
    calibration_path = run_dir / "validation_calibration.csv"
    write_prediction_frame(predictions, prediction_path)
    metrics = compute_metrics(predictions["probability"], predictions["true_label"])
    atomic_write_json(metrics_path, metrics)
    _atomic_csv(
        calibration_table(
            predictions["probability"].to_numpy(),
            predictions["true_label"].to_numpy(),
            n_bins=cfg.evaluation.calibration_bins,
        ),
        calibration_path,
    )
    entry: Dict[str, Any] = {
        "status": "COMPLETE",
        "scenario": scenario,
        "run_id": run_id,
        "model": backbone,
        "pretraining": pretraining,
        "feature_set": "D",
        "semantic_config_hash": semantic_hash,
        "implementation_commit": implementation,
        "environment_hash": environment_hash_value,
        "checkpoint_path": _relative(checkpoint_path, protocol_dir),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["best_epoch"]),
        "best_validation_auc": float(checkpoint["best_validation_auc"]),
        "pos_weight": float(pos_weight.item()),
        "validation_predictions_path": _relative(prediction_path, protocol_dir),
        "validation_predictions_sha256": file_sha256(prediction_path),
        "validation_metrics_path": _relative(metrics_path, protocol_dir),
        "initial_image_hashes": list(initial_hashes) if initial_hashes else None,
        "scaler_path": None,
        "scaler_sha256": None,
    }
    if scaler is not None:
        scaler_path = run_dir / "scaler.pkl"
        entry["scaler_path"] = _relative(scaler_path, protocol_dir)
        entry["scaler_sha256"] = file_sha256(scaler_path)
    atomic_write_json(summary_path, entry)
    upsert_registry(Path(protocol_dir) / "experiment_registry.csv", {
        "run_id": run_id,
        "phase": "C7_REFIT",
        "scenario": scenario,
        "model": backbone,
        "fold": cfg.data.deployment_validation_fold,
        "feature_set": "D",
        "pretraining": pretraining,
        "protocol_hash": protocol["protocol_hash"],
        "semantic_config_hash": semantic_hash,
        "implementation_commit": implementation,
        "environment_hash": environment_hash_value,
        "pos_weight": float(pos_weight.item()),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_validation_auc": float(checkpoint["best_validation_auc"]),
        "status": "done",
        "artifact_path": str(run_dir),
    })
    success_path.write_text("C7 deployment refit complete\n", encoding="utf-8")
    return entry


def _finalize_refits(
    protocol_dir: Path,
    *,
    state: Mapping[str, Any],
    implementation: str,
    environment_hash_value: str,
    training_pool: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    root = Path(protocol_dir)
    deployment_dir = root / "deployment"
    entries: Dict[str, Any] = {}
    for scenario in SCENARIOS:
        path = deployment_dir / f"refit_{scenario}.json"
        if not path.is_file():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("status") != "COMPLETE" or entry.get("scenario") != scenario:
            raise RuntimeError(f"Invalid deployment refit summary for {scenario}")
        checkpoint = _path_from_entry(root, entry["checkpoint_path"])
        if file_sha256(checkpoint) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"Deployment checkpoint checksum mismatch for {scenario}")
        entries[scenario] = entry
    entry_commits = {entry["implementation_commit"] for entry in entries.values()}
    entry_environments = {entry["environment_hash"] for entry in entries.values()}
    if entry_commits != {implementation}:
        raise RuntimeError(
            "All deployment refits must use the current single implementation commit"
        )
    if entry_environments != {environment_hash_value}:
        raise RuntimeError("All deployment refits must use the locked C7 environment")
    if entries["S2"]["initial_image_hashes"] != entries["S3"]["initial_image_hashes"]:
        raise RuntimeError("Deployment S2/S3 initial image hashes differ")

    lock = state["model_lock"]
    protocol = state["protocol"]
    index = {
        "status": "READY_FOR_SECONDARY_HOLDOUT",
        "protocol_hash": protocol["protocol_hash"],
        "model_lock_hash": file_sha256(root / "model_lock.json"),
        "deployment_split_hash": file_sha256(root / "deployment_split.csv"),
        "implementation_commit": implementation,
        "environment_hash": environment_hash_value,
        "selected_backbone": lock["selected_backbone"],
        "selected_pretraining": lock["selected_pretraining"],
        "scenarios": entries,
        "no_new_tuning": True,
    }
    index_path = deployment_dir / "refit_index.json"
    atomic_write_json(index_path, index)

    s3 = entries["S3"]
    scaler_path = _path_from_entry(root, s3["scaler_path"])
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)
    train_frame = training_pool[training_pool["split"] == "train"].reset_index(drop=True)
    rng = np.random.RandomState(cfg.train.seed)
    positions = rng.choice(
        len(train_frame),
        size=min(cfg.evaluation.shap_background_size, len(train_frame)),
        replace=False,
    )
    raw = train_frame.iloc[positions][list(TABULAR_FEATURE_SETS["D"])].to_numpy(np.float32)
    background = scaler.transform(raw).astype(np.float32)
    background_path = deployment_dir / "shap_background_scaled.npy"
    background_cases_path = deployment_dir / "shap_background_cases.csv"
    _atomic_npy(background, background_path)
    _atomic_csv(
        train_frame.iloc[positions][["Image Index", "Patient ID", "binary_label"]]
        .rename(columns={"Image Index": "image_index", "Patient ID": "patient_id", "binary_label": "true_label"}),
        background_cases_path,
    )
    checkpoint_path = _path_from_entry(root, s3["checkpoint_path"])
    manifest = {
        "status": "READY",
        "scenario": "S3",
        "protocol_hash": protocol["protocol_hash"],
        "protocol_path": "../protocol.json",
        "refit_index_path": "refit_index.json",
        "refit_index_checksum": file_sha256(index_path),
        "backbone": lock["selected_backbone"],
        "pretraining": lock["selected_pretraining"],
        "checkpoint_path": str(checkpoint_path.relative_to(deployment_dir)).replace("\\", "/"),
        "checkpoint_checksum": file_sha256(checkpoint_path),
        "scaler_path": str(scaler_path.relative_to(deployment_dir)).replace("\\", "/"),
        "scaler_checksum": file_sha256(scaler_path),
        "shap_background_path": background_path.name,
        "shap_background_checksum": file_sha256(background_path),
        "shap_background_cases_path": background_cases_path.name,
        "shap_background_cases_checksum": file_sha256(background_cases_path),
        "threshold": cfg.evaluation.decision_threshold,
        "display": "Skor model untuk kelas Abnormal; bukan probabilitas klinis terkalibrasi",
    }
    atomic_write_json(deployment_dir / "deployment_manifest.json", manifest)
    (deployment_dir / "_REFIT_SUCCESS").write_text(
        "C7 S1/S2/S3 deployment refits complete\n", encoding="utf-8"
    )
    return index


def run_deployment_refit(
    protocol_dir: Path,
    *,
    scenarios: Iterable[str] = SCENARIOS,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Refit S1/S2/S3 using only the immutable 90/10 training-pool split."""
    protocol_dir = Path(protocol_dir)
    requested = tuple(dict.fromkeys(scenarios))
    if not requested or not set(requested).issubset(SCENARIOS):
        raise ValueError(f"Unsupported C7 refit scenarios: {requested}")
    if (protocol_dir / "secondary_holdout" / "official_test_access_receipt.json").exists():
        raise StageGateError("Deployment refits are immutable after official-test access is claimed")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, implementation, env_hash, _ = _execution_context(protocol_dir, device)
    training_pool, image_index, _ = _load_deployment_pool(protocol_dir)
    results: Dict[str, Any] = {}
    for scenario in requested:
        print(f"[C7 refit] scenario={scenario}", flush=True)
        results[scenario] = _run_refit_scenario(
            protocol_dir,
            scenario,
            state=state,
            implementation=implementation,
            environment_hash_value=env_hash,
            device=device,
            training_pool=training_pool,
            image_index=image_index,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    index = _finalize_refits(
        protocol_dir,
        state=state,
        implementation=implementation,
        environment_hash_value=env_hash,
        training_pool=training_pool,
    )
    return {
        "status": "READY_FOR_SECONDARY_HOLDOUT" if index else "REFIT_PARTIAL",
        "completed": sorted(results),
        "refit_index": index,
    }


def claim_official_test_access(
    protocol_dir: Path,
    *,
    confirmation: str,
    implementation_commit: str,
    environment_hash_value: str,
) -> Dict[str, Any]:
    """Atomically claim the sole resumable official-test evaluation event."""
    if confirmation != OFFICIAL_TEST_CONFIRMATION:
        raise OfficialTestAccessError(
            f"Explicit confirmation must equal {OFFICIAL_TEST_CONFIRMATION!r}"
        )
    root = Path(protocol_dir)
    state = validate_c7_prerequisites(root, require_refit=True)
    output = root / "secondary_holdout"
    if (output / "_SUCCESS").is_file():
        raise OfficialTestAccessError("Secondary holdout is complete; a new access event is forbidden")
    receipt_path = output / "official_test_access_receipt.json"
    expected_refit_hash = file_sha256(state["refit_index_path"])
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "CLAIMED"
            or receipt.get("protocol_hash") != state["protocol"]["protocol_hash"]
            or receipt.get("refit_index_hash") != expected_refit_hash
            or receipt.get("environment_hash") != environment_hash_value
            or receipt.get("implementation_commit") != implementation_commit
        ):
            raise OfficialTestAccessError("Existing official-test receipt is incompatible")
        return receipt
    receipt = {
        "status": "CLAIMED",
        "access_event_id": str(uuid.uuid4()),
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": state["protocol"]["protocol_hash"],
        "refit_index_hash": expected_refit_hash,
        "implementation_commit": implementation_commit,
        "environment_hash": environment_hash_value,
        "scope": "single resumable secondary-holdout evaluation of frozen S1/S2/S3",
        "prior_exposure_disclosure": True,
    }
    try:
        _exclusive_json(receipt_path, receipt)
    except FileExistsError:
        return claim_official_test_access(
            root,
            confirmation=confirmation,
            implementation_commit=implementation_commit,
            environment_hash_value=environment_hash_value,
        )
    return receipt


def _official_test_frame(
    protocol_dir: Path,
    *,
    access_event_id: str,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Materialize the test manifest once; resumes reuse it without reopening test_list."""
    root = Path(protocol_dir)
    assert_official_test_access(
        stage="C7",
        protocol_path=root / "protocol.json",
        access_event_id=access_event_id,
    )
    output = root / "secondary_holdout"
    manifest_path = output / "official_test_manifest.csv"
    metadata_path = output / "official_test_manifest.json"
    if manifest_path.is_file() and metadata_path.is_file():
        manifest_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if manifest_meta.get("access_event_id") != access_event_id:
            raise OfficialTestAccessError("Official-test manifest belongs to another access event")
        if manifest_meta.get("manifest_sha256") != file_sha256(manifest_path):
            raise OfficialTestAccessError("Official-test manifest checksum mismatch")
        manifest = pd.read_csv(manifest_path)
    else:
        _, test = load_official_partitions(
            metadata, cfg.paths.train_list_path, cfg.paths.test_list_path
        )
        manifest = test[["Image Index", "Patient ID", "binary_label"]].rename(
            columns={"Image Index": "image_index", "Patient ID": "patient_id", "binary_label": "true_label"}
        ).sort_values("image_index", kind="mergesort").reset_index(drop=True)
        _atomic_csv(manifest, manifest_path)
        receipt = json.loads(
            (output / "official_test_access_receipt.json").read_text(encoding="utf-8")
        )
        manifest_meta = {
            "status": "IMMUTABLE",
            "access_event_id": access_event_id,
            "protocol_hash": receipt["protocol_hash"],
            "test_list_sha256": file_sha256(cfg.paths.test_list_path),
            "manifest_sha256": file_sha256(manifest_path),
            "images": int(len(manifest)),
            "patients": int(manifest["patient_id"].nunique()),
            "abnormal_prevalence": float(manifest["true_label"].mean()),
        }
        atomic_write_json(metadata_path, manifest_meta)
    if list(manifest.columns) != OFFICIAL_TEST_COLUMNS:
        raise OfficialTestAccessError("Official-test manifest schema mismatch")
    if manifest["image_index"].duplicated().any() or not manifest["true_label"].isin([0, 1]).all():
        raise OfficialTestAccessError("Official-test manifest contains invalid rows")
    lookup = metadata.set_index("Image Index")
    missing = set(manifest["image_index"].astype(str)) - set(lookup.index.astype(str))
    if missing:
        raise FileNotFoundError(f"Official-test manifest references missing metadata: {sorted(missing)[0]}")
    test_frame = lookup.loc[manifest["image_index"].astype(str)].reset_index()
    if not np.array_equal(test_frame["Patient ID"].to_numpy(), manifest["patient_id"].to_numpy()):
        raise OfficialTestAccessError("Official-test Patient ID differs from immutable manifest")
    if not np.array_equal(test_frame["binary_label"].to_numpy(), manifest["true_label"].to_numpy()):
        raise OfficialTestAccessError("Official-test label differs from immutable manifest")
    return test_frame, manifest_meta


def _load_refit_model(
    protocol_dir: Path,
    entry: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, Optional[Any]]:
    scenario = str(entry["scenario"])
    checkpoint_path = _path_from_entry(protocol_dir, str(entry["checkpoint_path"]))
    if file_sha256(checkpoint_path) != entry["checkpoint_sha256"]:
        raise RuntimeError(f"Refit checkpoint checksum mismatch for {scenario}")
    model = build_model(
        scenario,
        backbone_name=str(entry["model"]) if scenario != "S1" else "densenet121",
        pretraining="none" if scenario != "S1" else "imagenet",
        fold=cfg.data.deployment_validation_fold,
        tabular_input_dim=len(TABULAR_FEATURE_SETS["D"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    scaler = None
    if scenario in {"S1", "S3"}:
        scaler_path = _path_from_entry(protocol_dir, str(entry["scaler_path"]))
        if file_sha256(scaler_path) != entry["scaler_sha256"]:
            raise RuntimeError(f"Refit scaler checksum mismatch for {scenario}")
        with scaler_path.open("rb") as handle:
            scaler = pickle.load(handle)
    return model.to(device).eval(), scaler


def _evaluate_holdout_scenario(
    protocol_dir: Path,
    *,
    scenario: str,
    entry: Mapping[str, Any],
    test_frame: pd.DataFrame,
    image_index: Dict[str, Path],
    official_manifest_hash: str,
    access_event_id: str,
    state: Mapping[str, Any],
    implementation: str,
    environment_hash_value: str,
    device: torch.device,
) -> Dict[str, Any]:
    assert_official_test_access(
        stage="C7",
        protocol_path=Path(protocol_dir) / "protocol.json",
        access_event_id=access_event_id,
    )
    output = Path(protocol_dir) / "secondary_holdout" / scenario
    summary_path = output / "summary.json"
    if (output / "_SUCCESS").is_file() and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("access_event_id") != access_event_id
            or existing.get("implementation_commit") != implementation
            or existing.get("environment_hash") != environment_hash_value
            or existing.get("checkpoint_sha256") != entry["checkpoint_sha256"]
        ):
            raise RuntimeError(f"Completed {scenario} holdout artifact is incompatible")
        return existing
    model, scaler = _load_refit_model(protocol_dir, entry, device)
    loader = make_inference_dataloader(
        test_frame,
        image_index,
        modalities=_modalities(scenario),
        scaler=scaler,
        feature_set="D",
        seed=cfg.train.seed,
    )
    resolved = {
        "stage": "C7_SECONDARY_HOLDOUT",
        "scenario": scenario,
        "official_test_manifest_hash": official_manifest_hash,
        "refit_semantic_config_hash": entry["semantic_config_hash"],
        "access_event_id": access_event_id,
        "runtime": cfg.scientific_runtime_values(),
        "prior_exposure_disclosure": True,
    }
    semantic_hash = semantic_config_hash(
        protocol_hash_value=state["protocol"]["protocol_hash"],
        selected_architecture=str(entry["model"]),
        weight_checksum=str(entry["checkpoint_sha256"]),
        fold=cfg.data.deployment_validation_fold,
        feature_set="D",
        resolved_runtime_config=resolved,
        environment_hash=environment_hash_value,
        implementation_commit=implementation,
    )
    run_id = f"C7-holdout-{scenario}-{entry['model']}-{semantic_hash[:12]}"
    predictions = collect_prediction_frame(
        model,
        loader,
        device,
        fold=cfg.data.deployment_validation_fold,
        split="official_test_secondary_prior_exposure",
        model_name=str(entry["model"]),
        scenario=scenario,
        feature_set="D",
        pretraining=str(entry["pretraining"]),
        checkpoint_epoch=int(entry["checkpoint_epoch"]),
        protocol_hash=state["protocol"]["protocol_hash"],
        semantic_config_hash=semantic_hash,
        run_id=run_id,
    )
    expected = set(test_frame["Image Index"].astype(str))
    if set(predictions["image_index"].astype(str)) != expected or len(predictions) != len(expected):
        raise RuntimeError(f"Official-test prediction coverage mismatch for {scenario}")
    prediction_path = output / "predictions.csv"
    metrics_path = output / "metrics.json"
    calibration_path = output / "calibration.csv"
    write_prediction_frame(predictions, prediction_path)
    metrics = compute_metrics(predictions["probability"], predictions["true_label"])
    atomic_write_json(metrics_path, metrics)
    _atomic_csv(
        calibration_table(
            predictions["probability"].to_numpy(),
            predictions["true_label"].to_numpy(),
            n_bins=cfg.evaluation.calibration_bins,
        ),
        calibration_path,
    )
    summary = {
        "status": "COMPLETE",
        "scenario": scenario,
        "run_id": run_id,
        "semantic_config_hash": semantic_hash,
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "access_event_id": access_event_id,
        "implementation_commit": implementation,
        "environment_hash": environment_hash_value,
        "predictions_path": _relative(prediction_path, protocol_dir),
        "predictions_sha256": file_sha256(prediction_path),
        "metrics": metrics,
        "prior_exposure_disclosure": True,
    }
    atomic_write_json(summary_path, summary)
    upsert_registry(Path(protocol_dir) / "experiment_registry.csv", {
        "run_id": run_id,
        "phase": "C7_HOLDOUT",
        "scenario": scenario,
        "model": entry["model"],
        "fold": cfg.data.deployment_validation_fold,
        "feature_set": "D",
        "pretraining": entry["pretraining"],
        "protocol_hash": state["protocol"]["protocol_hash"],
        "semantic_config_hash": semantic_hash,
        "implementation_commit": implementation,
        "environment_hash": environment_hash_value,
        "pos_weight": entry["pos_weight"],
        "best_epoch": entry["checkpoint_epoch"],
        "best_validation_auc": entry["best_validation_auc"],
        "status": "done",
        "artifact_path": str(output),
    })
    (output / "_SUCCESS").write_text("C7 secondary holdout scenario complete\n", encoding="utf-8")
    del model
    return summary


def _finalize_secondary_holdout(
    protocol_dir: Path,
    *,
    access_event_id: str,
    implementation: str,
    environment_hash_value: str,
) -> Dict[str, Any]:
    root = Path(protocol_dir)
    output = root / "secondary_holdout"
    scenarios = {
        scenario: json.loads((output / scenario / "summary.json").read_text(encoding="utf-8"))
        for scenario in SCENARIOS
    }
    disclosure = {
        "status": "DISCLOSED",
        "official_partition_role": "secondary holdout with prior exposure",
        "primary_evidence": "patient-level five-fold OOF evaluation on official training pool",
        "claim": "not presented as a pristine independent generalization estimate",
        "access_event_id": access_event_id,
    }
    atomic_write_json(output / "prior_exposure_disclosure.json", disclosure)
    summary = {
        "status": "COMPLETE",
        "protocol_hash": root.name,
        "access_event_id": access_event_id,
        "implementation_commit": implementation,
        "environment_hash": environment_hash_value,
        "scenarios": scenarios,
        "interpretation": "secondary holdout with prior exposure; primary evidence remains pooled OOF",
    }
    atomic_write_json(output / "summary.json", summary)
    artifacts = _secondary_holdout_artifacts(root, output)
    manifest = {
        "status": "COMPLETE",
        "protocol_hash": root.name,
        "access_event_id": access_event_id,
        "implementation_commit": implementation,
        "environment_hash": environment_hash_value,
        "artifacts": artifacts,
    }
    atomic_write_json(output / "artifact_manifest.json", manifest)
    (output / "_SUCCESS").write_text(
        "C7 secondary holdout complete with prior-exposure disclosure\n",
        encoding="utf-8",
    )
    return summary


def _secondary_holdout_artifacts(root: Path, output: Path) -> Dict[str, Dict[str, Any]]:
    """Hash immutable C7 evidence while excluding process-owned operational logs."""
    excluded = {"artifact_manifest.json", "_SUCCESS"} | set(
        SECONDARY_HOLDOUT_OPERATIONAL_FILES
    )
    artifacts: Dict[str, Dict[str, Any]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in excluded:
            artifacts[str(path.relative_to(root)).replace("\\", "/")] = {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
    return artifacts


def rebuild_secondary_holdout_manifest(protocol_dir: Path) -> Dict[str, Any]:
    """Rebuild C7 checksums from completed artifacts without reopening the test set."""
    root = Path(protocol_dir)
    output = root / "secondary_holdout"
    required = [
        output / "_SUCCESS",
        output / "summary.json",
        output / "prior_exposure_disclosure.json",
        output / "official_test_access_receipt.json",
        *(output / scenario / "_SUCCESS" for scenario in SCENARIOS),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise StageGateError(
            "Cannot rebuild an incomplete secondary-holdout manifest: " + ", ".join(missing)
        )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    receipt = json.loads(
        (output / "official_test_access_receipt.json").read_text(encoding="utf-8")
    )
    if summary.get("status") != "COMPLETE":
        raise StageGateError("Secondary-holdout summary is not COMPLETE")
    if summary.get("access_event_id") != receipt.get("access_event_id"):
        raise StageGateError("Secondary-holdout summary and receipt event IDs differ")
    if summary.get("protocol_hash") != receipt.get("protocol_hash"):
        raise StageGateError("Secondary-holdout summary and receipt protocol hashes differ")

    manifest = {
        "status": "COMPLETE",
        "protocol_hash": summary["protocol_hash"],
        "access_event_id": summary["access_event_id"],
        "implementation_commit": summary["implementation_commit"],
        "environment_hash": summary["environment_hash"],
        "artifacts": _secondary_holdout_artifacts(root, output),
    }
    atomic_write_json(output / "artifact_manifest.json", manifest)
    return manifest


def run_secondary_holdout(
    protocol_dir: Path,
    *,
    confirmation: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate frozen refits under one explicit, resumable official-test event."""
    protocol_dir = Path(protocol_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, implementation, env_hash, _ = _execution_context(protocol_dir, device)
    state = validate_c7_prerequisites(protocol_dir, require_refit=True)
    receipt = claim_official_test_access(
        protocol_dir,
        confirmation=confirmation,
        implementation_commit=implementation,
        environment_hash_value=env_hash,
    )
    event_id = str(receipt["access_event_id"])
    image_index = build_image_index(cfg.paths.image_dirs)
    metadata = load_and_prepare_metadata(cfg.paths.csv_path, image_index)
    test_frame, official_meta = _official_test_frame(
        protocol_dir,
        access_event_id=event_id,
        metadata=metadata,
    )
    results = {}
    for scenario in SCENARIOS:
        print(f"[C7 secondary holdout] scenario={scenario}", flush=True)
        results[scenario] = _evaluate_holdout_scenario(
            protocol_dir,
            scenario=scenario,
            entry=state["refit_index"]["scenarios"][scenario],
            test_frame=test_frame,
            image_index=image_index,
            official_manifest_hash=official_meta["manifest_sha256"],
            access_event_id=event_id,
            state=state,
            implementation=implementation,
            environment_hash_value=env_hash,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = _finalize_secondary_holdout(
        protocol_dir,
        access_event_id=event_id,
        implementation=implementation,
        environment_hash_value=env_hash,
    )
    return {"status": "COMPLETE", "scenarios": results, "summary": summary}
