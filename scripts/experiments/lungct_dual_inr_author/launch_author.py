#!/usr/bin/env python3
"""Dry-run and launch the locked DUAL-INR-DIR A0/A1 cohort.

The launcher deliberately invokes one ``run_author`` process per case.  A
fixed worker owns each GPU and pulls from a shared queue, so a GPU never hosts
more than one optimizer while faster GPUs can take additional cases.  The A0
phase is a cohort barrier: every A0 record must validate before any dependent
A1 replay is started.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from scripts.experiments.lungct_dual_inr_author.external_api import (  # type: ignore
        AUTHOR_EXTERNAL_COMMIT,
        external_commit,
        external_repository,
        git_directory,
        repository_root,
    )
else:
    from .external_api import (
        AUTHOR_EXTERNAL_COMMIT,
        external_commit,
        external_repository,
        git_directory,
        repository_root,
    )


RUNNER_MODULE = "scripts.experiments.lungct_dual_inr_author.run_author"
LOCKED_SEED = 42
LOCKED_TOTAL_STEPS = 3000
LOCKED_BLUR_SIGMAS = [4.0, 2.0, 0.0]
IDENTITY_TRE_MM = {
    1: 3.892406,
    2: 4.337842,
    3: 6.942993,
    4: 9.830147,
    5: 7.476853,
    6: 10.891024,
    7: 11.026220,
    8: 14.994681,
    9: 7.918264,
    10: 7.301382,
}
IDENTITY_TRE_ABS_TOLERANCE_MM = 1.0e-5
SMOKE_STEPS = 3
GPU_POOL = ("0", "1", "2", "3")
CASES_BY_MODE: dict[str, tuple[int, ...]] = {
    "smoke": (1,),
    "pilot": (1, 8),
    "production": tuple(range(1, 11)),
}
VARIANTS = ("A0", "A1")

REQUIRED_CASE_FILES = (
    "DONE",
    "RUN_STATE.json",
    "summary.json",
    "resolved_config.yaml",
    "resolved_config.json",
    "provenance.json",
    "history.csv",
    "stage_transitions.json",
    "stage_trace.csv",
    "tre_scores.txt",
    "dirinr_dvf.nii.gz",
    "dense_dvf_crop_voxel_xyz.npy",
    "inr_dvf.pth",
    "inr_dvf_config.yaml",
    "landmark_metrics.json",
    "landmark_details.npz",
    "five_lobe_dice.json",
    "stdout.log",
    "stderr.log",
)
NONEMPTY_CASE_FILES = tuple(
    name for name in REQUIRED_CASE_FILES if name not in {"DONE", "stderr.log"}
)
HISTORY_NUMERIC_FIELDS = (
    "zero_step",
    "step",
    "stage_index",
    "blur_sigma",
    "loss_total",
    "loss_similarity",
    "loss_spatial_weighted",
    "loss_mask_raw",
    "loss_mask_weighted",
    "coarse_jacobian_raw",
    "coarse_jacobian_weighted",
    "coarse_laplacian_raw",
    "coarse_laplacian_weighted",
    "fine_jacobian_raw",
    "fine_jacobian_weighted",
    "fine_laplacian_raw",
    "fine_laplacian_weighted",
    "learning_rate",
    "coarse_scale",
    "fine_scale",
)


class LaunchError(RuntimeError):
    """Raised when a dry-run, dependency, or completed-record gate fails."""


@dataclass(frozen=True)
class JobSpec:
    variant: str
    case: int
    seed: int
    mode: str
    smoke_steps: int | None
    expected_steps: int
    data_root: Path
    output_root: Path
    execution_root: Path
    trace_root: Path

    @property
    def job_id(self) -> str:
        return f"{self.variant.lower()}_case_{self.case:02d}_seed_{self.seed}"

    @property
    def case_dir(self) -> Path:
        return self.execution_root / self.variant.lower() / f"case_{self.case:02d}"

    @property
    def a0_case_dir(self) -> Path:
        return self.execution_root / "a0" / f"case_{self.case:02d}"

    def command(self, *, python_executable: str, device: str) -> list[str]:
        command = [
            python_executable,
            "-m",
            RUNNER_MODULE,
            "--variant",
            self.variant,
            "--case",
            str(self.case),
            "--device",
            str(device),
            "--data-root",
            str(self.data_root),
            "--output-root",
            str(self.output_root),
            "--seed",
            str(self.seed),
        ]
        if self.smoke_steps is not None:
            command.extend(["--smoke-steps", str(self.smoke_steps)])
        if self.variant == "A1":
            command.extend(["--replay-a0-transitions", str(self.trace_root)])
        return command


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    summary: dict[str, Any] | None = None


@dataclass
class ProcessResult:
    returncode: int
    elapsed_seconds: float
    stdout_path: Path
    stderr_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(dict(payload)), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LaunchError(f"Expected a JSON object in {path}")
    return payload


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_roots(data_root: Path, output_root: Path) -> tuple[Path, Path]:
    if not data_root.is_absolute() or not output_root.is_absolute():
        raise LaunchError("--data-root and --output-root must both be absolute")
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    if not data_root.is_dir():
        raise LaunchError(f"Converted data root does not exist: {data_root}")
    if _is_below(output_root, repository_root()):
        raise LaunchError("Large run outputs must remain outside the Git repository")
    if _is_below(output_root, data_root):
        raise LaunchError("Run outputs must remain outside the prepared data root")
    return data_root, output_root


def execution_root(output_root: Path, mode: str) -> Path:
    return output_root / f"smoke_{SMOKE_STEPS}steps" if mode == "smoke" else output_root


def code_sha256_binding() -> dict[str, str]:
    root = repository_root()
    relative_paths = (
        "scripts/experiments/lungct_dual_inr_author/run_author.py",
        "scripts/experiments/lungct_dual_inr_author/adapter.py",
        "scripts/experiments/lungct_dual_inr_author/external_api.py",
        "scripts/experiments/lungct_dual_inr_author/launch_author.py",
        "scripts/experiments/lungct_dual_inr_author/aggregate_author.py",
        "scripts/experiments/lungct_dual_inr_author/plot_author_review.py",
        "scripts/experiments/lungct_dual_inr_author/posthoc_author_regularity.py",
        "scripts/experiments/lungct_no_sweep_repro/common.py",
    )
    return {relative: file_sha256(root / relative) for relative in relative_paths}


def adapter_bundle_sha256() -> str:
    """Match ``run_author.adapter_source_bundle()['bundle_sha256']`` exactly."""

    root = repository_root().resolve()
    paths = [
        root / "scripts/experiments/lungct_dual_inr_author/run_author.py",
        root / "scripts/experiments/lungct_dual_inr_author/adapter.py",
        root / "scripts/experiments/lungct_dual_inr_author/external_api.py",
    ]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def verify_code_snapshot(snapshot_root: Path, binding: Mapping[str, str]) -> None:
    checksum_path = snapshot_root / "checksums.sha256"
    if not checksum_path.is_file():
        raise LaunchError(f"Code snapshot checksum manifest is missing: {checksum_path}")
    recorded: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or not expected or not relative:
            raise LaunchError(f"Malformed code snapshot checksum line: {line!r}")
        recorded[relative] = expected
    if recorded != dict(binding):
        raise LaunchError("Code snapshot checksum manifest differs from the immutable plan")
    for relative, expected in binding.items():
        path = snapshot_root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise LaunchError(f"Code snapshot file is missing or changed: {relative}")


def write_code_snapshot(run_root: Path, binding: Mapping[str, str]) -> Path:
    """Create an immutable small source snapshot next to external run outputs."""

    snapshot_root = run_root / "code_snapshot"
    if snapshot_root.exists():
        verify_code_snapshot(snapshot_root, binding)
        return snapshot_root
    run_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".code_snapshot_", dir=run_root) as temporary:
        staging = Path(temporary)
        for relative, expected in binding.items():
            source = repository_root() / relative
            if file_sha256(source) != expected:
                raise LaunchError(f"Source changed while taking code snapshot: {relative}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        checksum_lines = [f"{digest}  {relative}" for relative, digest in sorted(binding.items())]
        (staging / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        verify_code_snapshot(staging, binding)
        staging.replace(snapshot_root)
    return snapshot_root


def audit_prepared_data(data_root: Path) -> dict[str, Any]:
    """Run the canonical read-only adapter/checksum validation for all ten cases."""

    required = (
        data_root / "dataset_manifest.json",
        data_root / "preprocess_contract.json",
        data_root / "checksums.sha256",
    )
    missing = [path.name for path in required if not path.is_file()]
    binding = {
        "dataset_manifest_sha256": file_sha256(required[0]) if required[0].is_file() else None,
        "preprocess_contract_sha256": file_sha256(required[1]) if required[1].is_file() else None,
        "checksums_manifest_sha256": file_sha256(required[2]) if required[2].is_file() else None,
    }
    if missing:
        return {
            "status": "fail",
            "errors": [f"missing required prepared-data artifact: {name}" for name in missing],
            "binding": binding,
        }

    try:
        manifest = _load_json(required[0])
        contract = _load_json(required[1])
        from scripts.data.preprocessing.lungct.prepare_dual_author_4dct import (
            ADAPTER_ID,
            validate_existing,
        )

        errors: list[str] = []
        expected_cases = list(range(1, 11))
        if manifest.get("adapter_id") != ADAPTER_ID:
            errors.append(f"dataset_manifest adapter_id is not {ADAPTER_ID!r}")
        if manifest.get("status") != "pass":
            errors.append("dataset_manifest status is not 'pass'")
        if manifest.get("case_ids") != expected_cases or manifest.get("case_count") != 10:
            errors.append("dataset_manifest must contain exactly case_ids 1..10")
        if contract.get("adapter_id") != ADAPTER_ID:
            errors.append(f"preprocess_contract adapter_id is not {ADAPTER_ID!r}")
        if contract.get("case_ids") != expected_cases:
            errors.append("preprocess_contract must contain exactly case_ids 1..10")
        validation = validate_existing(data_root, "1-10")
        if validation.get("status") != "pass":
            errors.append("canonical prepared-data structural/checksum audit failed")
        if validation.get("case_ids") != expected_cases:
            errors.append("canonical validator did not audit exactly case_ids 1..10")
        return {
            "status": "pass" if not errors else "fail",
            "errors": errors,
            "adapter_id": ADAPTER_ID,
            "case_ids": expected_cases,
            "checksum_status": validation.get("checksum_audit", {}).get("status"),
            "checksum_checked_files": validation.get("checksum_audit", {}).get("checked_files"),
            "binding": binding,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "errors": [f"{type(exc).__name__}: {exc}"],
            "binding": binding,
        }


def parse_gpu_pool(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(text).split(",") if part.strip())
    if not values or len(values) != len(set(values)):
        raise LaunchError(f"invalid GPU pool: {text!r}")
    try:
        numeric = [int(value) for value in values]
    except ValueError as exc:
        raise LaunchError(f"GPU pool must contain integer device indices: {text!r}") from exc
    if any(value < 0 for value in numeric):
        raise LaunchError(f"GPU pool must contain non-negative indices: {text!r}")
    return values


def audit_gpu_pool(gpus: Sequence[str] = GPU_POOL) -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        names = [torch.cuda.get_device_name(index) for index in range(count)]
        required = [int(value) for value in gpus]
        passed = available and all(index < count for index in required)
        return {
            "status": "pass" if passed else "fail",
            "cuda_available": available,
            "device_count": count,
            "device_names": names,
            "required_indices": required,
            "max_optimizers_per_gpu": 1,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "required_indices": [int(value) for value in gpus],
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_jobs(
    *, mode: str, data_root: Path, output_root: Path, seed: int = LOCKED_SEED
) -> list[JobSpec]:
    if mode not in CASES_BY_MODE:
        raise LaunchError(f"Unknown mode {mode!r}")
    if seed != LOCKED_SEED:
        raise LaunchError(f"The locked author cohort requires seed {LOCKED_SEED}")
    run_root = execution_root(output_root, mode)
    smoke_steps = SMOKE_STEPS if mode == "smoke" else None
    expected_steps = smoke_steps or LOCKED_TOTAL_STEPS
    jobs = [
        JobSpec(
            variant=variant,
            case=case,
            seed=seed,
            mode=mode,
            smoke_steps=smoke_steps,
            expected_steps=expected_steps,
            data_root=data_root,
            output_root=output_root,
            execution_root=run_root,
            trace_root=run_root,
        )
        for variant in VARIANTS
        for case in CASES_BY_MODE[mode]
    ]
    expected = 2 * len(CASES_BY_MODE[mode])
    if len(jobs) != expected or len({job.job_id for job in jobs}) != expected:
        raise LaunchError("Internal job-matrix uniqueness check failed")
    return jobs


def _float(value: Any, *, label: str, errors: list[str]) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} is not numeric: {value!r}")
        return None
    if not math.isfinite(result):
        errors.append(f"{label} is non-finite: {value!r}")
        return None
    return result


def _validate_history(
    path: Path,
    expected_steps: int,
    errors: list[str],
    transition: Mapping[str, Any] | None,
) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = (set(HISTORY_NUMERIC_FIELDS) | {"fine_active"}) - set(
                reader.fieldnames or ()
            )
            if missing:
                errors.append(f"history.csv lacks columns: {sorted(missing)}")
                return
            count = 0
            for count, row in enumerate(reader, start=1):
                for field in HISTORY_NUMERIC_FIELDS:
                    _float(row.get(field), label=f"history row {count} {field}", errors=errors)
                try:
                    if int(row["step"]) != count or int(row["zero_step"]) != count - 1:
                        errors.append(f"history row {count} has a non-sequential step index")
                    if transition is not None:
                        starts = [int(value) for value in transition["stage_start_steps"]]
                        expected_stage = max(
                            index for index, start in enumerate(starts) if start <= count - 1
                        )
                        observed_stage = int(row["stage_index"])
                        if observed_stage != expected_stage:
                            errors.append(
                                f"history row {count} stage {observed_stage} does not match trace stage {expected_stage}"
                            )
                        observed_sigma = float(row["blur_sigma"])
                        expected_sigma = float(transition["blur_sigmas"][expected_stage])
                        if observed_sigma != expected_sigma:
                            errors.append(
                                f"history row {count} sigma {observed_sigma} does not match {expected_sigma}"
                            )
                        fine_active = str(row["fine_active"]).strip().lower()
                        if fine_active not in {"true", "false"}:
                            errors.append(f"history row {count} has invalid fine_active")
                        elif (fine_active == "true") != (expected_stage >= 1):
                            errors.append(
                                f"history row {count} fine branch state does not match stage {expected_stage}"
                            )
                except (TypeError, ValueError):
                    pass
    except OSError as exc:
        errors.append(f"cannot read history.csv: {exc}")
        return
    if count != expected_steps:
        errors.append(f"history.csv has {count} rows; expected {expected_steps}")


def _validate_transition(
    path: Path,
    *,
    variant: str,
    case: int,
    seed: int,
    expected_steps: int,
    require_all_stages: bool,
    errors: list[str],
) -> dict[str, Any] | None:
    try:
        trace = _load_json(path)
    except LaunchError as exc:
        errors.append(str(exc))
        return None
    expected_policy = "author_dynamic_convergence" if variant == "A0" else "replay_a0"
    checks = {
        "schema_version": 1,
        "source_variant": variant,
        "case": case,
        "seed": seed,
        "total_steps": expected_steps,
        "policy": expected_policy,
    }
    for key, expected in checks.items():
        if trace.get(key) != expected:
            errors.append(f"stage_transitions.{key}={trace.get(key)!r}; expected {expected!r}")
    if trace.get("blur_sigmas") != LOCKED_BLUR_SIGMAS:
        errors.append("stage_transitions blur sigmas do not match [4,2,0]")
    starts = trace.get("stage_start_steps")
    if not isinstance(starts, list) or not starts or starts[0] != 0:
        errors.append("stage_start_steps must be a nonempty list beginning at zero")
    else:
        try:
            starts_int = [int(value) for value in starts]
            if starts_int != sorted(set(starts_int)):
                errors.append("stage_start_steps are not strictly increasing")
            if any(value < 0 or value >= expected_steps for value in starts_int):
                errors.append("stage_start_steps fall outside the run")
            if require_all_stages and len(starts_int) != len(LOCKED_BLUR_SIGMAS):
                errors.append("full-length record did not visit every blur stage")
        except (TypeError, ValueError):
            errors.append("stage_start_steps contain non-integers")
    return trace


def validate_done_summary(
    case_dir: Path,
    *,
    variant: str,
    case: int,
    seed: int,
    expected_steps: int,
    a0_case_dir: Path | None = None,
    require_cuda_memory: bool = True,
    expected_adapter_bundle_sha256: str | None = None,
    identity_tre_mm: Mapping[int, float] | None = IDENTITY_TRE_MM,
) -> ValidationResult:
    """Validate a record strongly enough for safe launcher skip/reuse."""

    errors: list[str] = []
    if not case_dir.is_dir():
        return ValidationResult(False, [f"case directory is absent: {case_dir}"])
    for name in REQUIRED_CASE_FILES:
        if not (case_dir / name).is_file():
            errors.append(f"missing required artifact: {name}")
    for name in NONEMPTY_CASE_FILES:
        path = case_dir / name
        if path.is_file() and path.stat().st_size == 0:
            errors.append(f"required artifact is empty: {name}")
    if errors:
        return ValidationResult(False, errors)

    try:
        summary = _load_json(case_dir / "summary.json")
        state = _load_json(case_dir / "RUN_STATE.json")
        config = _load_json(case_dir / "resolved_config.json")
        provenance = _load_json(case_dir / "provenance.json")
    except LaunchError as exc:
        return ValidationResult(False, [str(exc)])

    expected_run_mode = "short_smoke" if expected_steps != LOCKED_TOTAL_STEPS else "full_length"
    smoke_steps = expected_steps if expected_run_mode == "short_smoke" else None
    for name, payload in (("summary", summary), ("RUN_STATE", state)):
        expected_values = {
            "status": "complete",
            "variant": variant,
            "case": case,
            "seed": seed,
            "run_mode": expected_run_mode,
            "smoke_steps": smoke_steps,
        }
        for key, expected in expected_values.items():
            if payload.get(key) != expected:
                errors.append(f"{name}.{key}={payload.get(key)!r}; expected {expected!r}")
    if summary.get("fixed_phase") != 0 or summary.get("moving_phase") != 5:
        errors.append("summary phase contract must be fixed T00=0 and moving T50=5")

    expected_weight = 0.1 if variant == "A0" else 0.0
    if config.get("loss", {}).get("mask_weight") != expected_weight:
        errors.append(f"resolved mask weight is not {expected_weight}")
    if config.get("training", {}).get("total_steps") != expected_steps:
        errors.append("resolved total_steps does not match this cohort")
    export = config.get("export", {})
    for key, expected in {"case": case, "seed": seed, "run_mode": expected_run_mode}.items():
        if export.get(key) != expected:
            errors.append(f"resolved_config.export.{key} mismatch")

    if provenance.get("external_commit") != AUTHOR_EXTERNAL_COMMIT:
        errors.append("provenance external commit does not match the locked commit")
    if provenance.get("external_git_status_porcelain") != "":
        errors.append("provenance records a dirty external checkout")
    expected_bundle = expected_adapter_bundle_sha256 or adapter_bundle_sha256()
    if provenance.get("adapter_bundle_sha256") != expected_bundle:
        errors.append("provenance adapter source bundle differs from the approved plan")
    if provenance.get("adapter_source_bundle", {}).get("bundle_sha256") != expected_bundle:
        errors.append("nested provenance adapter source bundle differs from the approved plan")
    for key, expected in {"variant": variant, "case": case, "seed": seed}.items():
        if provenance.get(key) != expected:
            errors.append(f"provenance.{key} mismatch")

    transition = _validate_transition(
        case_dir / "stage_transitions.json",
        variant=variant,
        case=case,
        seed=seed,
        expected_steps=expected_steps,
        require_all_stages=expected_steps == LOCKED_TOTAL_STEPS,
        errors=errors,
    )
    if transition is not None and summary.get("stage_start_steps") != transition.get("stage_start_steps"):
        errors.append("summary and transition manifest stage boundaries disagree")

    try:
        with (case_dir / "stage_trace.csv").open("r", encoding="utf-8", newline="") as stream:
            stage_rows = list(csv.DictReader(stream))
        starts_csv = [int(row["start_zero_step"]) for row in stage_rows]
        if transition is not None and starts_csv != transition.get("stage_start_steps"):
            errors.append("stage_trace.csv and transition manifest boundaries disagree")
        if transition is not None and starts_csv == transition.get("stage_start_steps"):
            for index, row in enumerate(stage_rows):
                expected_end = (
                    starts_csv[index + 1] - 1
                    if index + 1 < len(starts_csv)
                    else expected_steps - 1
                )
                if int(row["stage_index"]) != index:
                    errors.append(f"stage_trace row {index} has wrong stage_index")
                if float(row["blur_sigma"]) != float(transition["blur_sigmas"][index]):
                    errors.append(f"stage_trace row {index} has wrong blur_sigma")
                if int(row["end_zero_step_inclusive"]) != expected_end:
                    errors.append(f"stage_trace row {index} has wrong end step")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid stage_trace.csv: {exc}")

    _validate_history(case_dir / "history.csv", expected_steps, errors, transition)

    landmark_metrics = summary.get("landmark_metrics", {})
    for family in ("initial", "direct_inr", "dense_dvf"):
        for mode in ("snap", "subvoxel"):
            metric = landmark_metrics.get(family, {}).get(mode, {})
            if metric.get("n") != 300:
                errors.append(f"{family}.{mode} does not contain 300 landmarks")
            _float(metric.get("mean_mm"), label=f"{family}.{mode}.mean_mm", errors=errors)
    initial_snap = _float(
        landmark_metrics.get("initial", {}).get("snap", {}).get("mean_mm"),
        label="initial.snap.mean_mm",
        errors=errors,
    )
    if (
        identity_tre_mm is not None
        and initial_snap is not None
        and abs(initial_snap - float(identity_tre_mm[case])) > IDENTITY_TRE_ABS_TOLERANCE_MM
    ):
        errors.append(
            f"identity snap TRE {initial_snap} differs from locked case{case:02d} reference"
        )
    agreement = landmark_metrics.get("direct_dense_agreement", {})
    snap_agreement = _float(
        agreement.get("snap_mean_tre_absolute_difference_mm"),
        label="direct_dense_agreement.snap",
        errors=errors,
    )
    _float(
        agreement.get("subvoxel_mean_tre_absolute_difference_mm"),
        label="direct_dense_agreement.subvoxel",
        errors=errors,
    )
    if snap_agreement is not None and snap_agreement > 0.05:
        errors.append("direct versus dense snap TRE disagreement exceeds 0.05 mm")
    for key in ("training_wall_seconds", "total_wall_seconds_through_evaluation"):
        value = _float(summary.get("runtime", {}).get(key), label=f"runtime.{key}", errors=errors)
        if value is not None and value <= 0:
            errors.append(f"runtime.{key} must be positive")
    if require_cuda_memory:
        for key in ("training_allocated", "training_reserved", "total_allocated", "total_reserved"):
            value = _float(
                summary.get("cuda_peak_memory_bytes", {}).get(key),
                label=f"cuda_peak_memory_bytes.{key}",
                errors=errors,
            )
            if value is not None and value <= 0:
                errors.append(f"cuda_peak_memory_bytes.{key} must be positive")

    if variant == "A1":
        if a0_case_dir is None:
            errors.append("A1 validation requires its paired A0 case directory")
        else:
            try:
                a0_config = _load_json(a0_case_dir / "resolved_config.json")
                comparable_a0 = json.loads(json.dumps(a0_config))
                comparable_a1 = json.loads(json.dumps(config))
                allowed_paths = (
                    ("loss", "mask_weight"),
                    ("adapter", "variant"),
                    ("adapter", "stage_policy"),
                    ("adapter", "loss_scope"),
                    ("export", "device_requested"),
                    ("export", "device_resolved"),
                    ("export", "output_dir"),
                    ("export", "a0_transition_trace"),
                )
                for first, second in allowed_paths:
                    comparable_a0.get(first, {}).pop(second, None)
                    comparable_a1.get(first, {}).pop(second, None)
                if comparable_a0 != comparable_a1:
                    errors.append("paired A0/A1 resolved configs differ outside allowed control fields")
            except (LaunchError, OSError, TypeError, ValueError) as exc:
                errors.append(f"cannot audit paired A0/A1 resolved configs: {exc}")
            a0_errors: list[str] = []
            a0_trace = _validate_transition(
                a0_case_dir / "stage_transitions.json",
                variant="A0",
                case=case,
                seed=seed,
                expected_steps=expected_steps,
                require_all_stages=expected_steps == LOCKED_TOTAL_STEPS,
                errors=a0_errors,
            )
            if a0_errors:
                errors.extend(f"paired A0: {error}" for error in a0_errors)
            if transition is not None and a0_trace is not None:
                if transition.get("stage_start_steps") != a0_trace.get("stage_start_steps"):
                    errors.append("A1 realized stage boundaries differ from paired A0")
                expected_trace = (a0_case_dir / "stage_transitions.json").resolve()
                for label, value in (
                    ("summary.a0_transition_trace", summary.get("a0_transition_trace")),
                    ("transition.replayed_manifest_path", transition.get("replayed_manifest_path")),
                ):
                    try:
                        if Path(str(value)).resolve() != expected_trace:
                            errors.append(f"{label} does not resolve to paired A0 trace")
                    except (OSError, TypeError, ValueError):
                        errors.append(f"{label} is invalid")

    return ValidationResult(not errors, errors, summary)


def inspect_jobs(
    jobs: Sequence[JobSpec], *, expected_adapter_bundle_sha256: str
) -> dict[str, dict[str, Any]]:
    inspections: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if not job.case_dir.exists():
            inspections[job.job_id] = {"status": "pending", "errors": []}
            continue
        result = validate_done_summary(
            job.case_dir,
            variant=job.variant,
            case=job.case,
            seed=job.seed,
            expected_steps=job.expected_steps,
            a0_case_dir=job.a0_case_dir if job.variant == "A1" else None,
            expected_adapter_bundle_sha256=expected_adapter_bundle_sha256,
        )
        inspections[job.job_id] = {
            "status": "validated_done" if result.valid else "invalid_existing",
            "errors": result.errors,
        }
    return inspections


def _git_status(repo: Path) -> str:
    repo = repo.resolve()
    result = subprocess.run(
        [
            "git",
            f"--git-dir={git_directory(repo)}",
            f"--work-tree={repo}",
            "status",
            "--porcelain",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def immutable_plan_payload(
    jobs: Sequence[JobSpec],
    *,
    python_executable: str,
    dataset_binding: Mapping[str, Any],
    code_binding: Mapping[str, str],
    gpus: Sequence[str] = GPU_POOL,
    technical_max_attempts: int = 2,
) -> dict[str, Any]:
    if not jobs:
        raise LaunchError("Cannot build an empty launch plan")
    first = jobs[0]
    return {
        "schema_version": 1,
        "experiment_id": "dual_inr_dirlab_author_repro_v1",
        "mode": first.mode,
        "seed": first.seed,
        "cases": list(CASES_BY_MODE[first.mode]),
        "variants": list(VARIANTS),
        "phase_order": ["A0", "A1"],
        "a0_cohort_barrier_before_a1": True,
        "gpu_pool": list(gpus),
        "max_optimizers_per_gpu": 1,
        "scheduler": "fixed_gpu_workers_shared_queue_work_stealing",
        "technical_max_attempts": int(technical_max_attempts),
        "data_root": str(first.data_root),
        "output_root": str(first.output_root),
        "execution_root": str(first.execution_root),
        "a1_trace_root": str(first.trace_root),
        "external_commit": AUTHOR_EXTERNAL_COMMIT,
        "adapter_source_bundle_sha256": adapter_bundle_sha256(),
        "prepared_data_sha256": dict(dataset_binding),
        "orchestration_code_sha256": dict(code_binding),
        "code_snapshot_root": str(first.execution_root / "code_snapshot"),
        "python_executable": python_executable,
        "jobs": [
            {
                "job_id": job.job_id,
                "variant": job.variant,
                "case": job.case,
                "dependency": f"a0_case_{job.case:02d}_seed_{job.seed}" if job.variant == "A1" else None,
                "case_dir": str(job.case_dir),
                "expected_steps": job.expected_steps,
                "command_template": job.command(
                    python_executable=python_executable, device="<GPU_WORKER>"
                ),
            }
            for job in jobs
        ],
    }


def plan_sha256(plan: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(dict(plan)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def default_manifest_path(run_root: Path, mode: str) -> Path:
    return run_root / f"launcher_manifest_{mode}.json"


def default_state_path(run_root: Path, mode: str) -> Path:
    return run_root / f"launcher_state_{mode}.json"


def build_dry_run_manifest(
    *,
    plan: Mapping[str, Any],
    inspections: Mapping[str, Mapping[str, Any]],
    source_clean: bool,
    source_commit_matches: bool,
    data_audit: Mapping[str, Any],
    gpu_audit: Mapping[str, Any],
) -> dict[str, Any]:
    invalid = [key for key, value in inspections.items() if value["status"] == "invalid_existing"]
    dry_run_passed = (
        source_clean
        and source_commit_matches
        and data_audit.get("status") == "pass"
        and gpu_audit.get("status") == "pass"
        and not invalid
    )
    return {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "dry_run_only": True,
        "dry_run_passed": dry_run_passed,
        "plan_sha256": plan_sha256(plan),
        "plan": dict(plan),
        "preflight": {
            "source_clean": source_clean,
            "source_commit_matches": source_commit_matches,
            "prepared_data_audit": dict(data_audit),
            "gpu_audit": dict(gpu_audit),
            "invalid_existing_jobs": invalid,
            "jobs": dict(inspections),
        },
        "approval_note": "Inspect this manifest before invoking --execute.",
    }


class LauncherState:
    def __init__(self, path: Path, payload: dict[str, Any]):
        self.path = path
        self.payload = payload
        self._lock = threading.Lock()
        atomic_write_json(self.path, self.payload)

    def update_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            self.payload["jobs"][job_id].update(updates)
            self.payload["updated_utc"] = utc_now()
            atomic_write_json(self.path, self.payload)

    def set_status(self, status: str, **updates: Any) -> None:
        with self._lock:
            self.payload["status"] = status
            self.payload.update(updates)
            self.payload["updated_utc"] = utc_now()
            atomic_write_json(self.path, self.payload)


def initial_state_payload(
    *, mode: str, plan_hash: str, jobs: Sequence[JobSpec], inspections: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": "dual_inr_dirlab_author_repro_v1",
        "mode": mode,
        "plan_sha256": plan_hash,
        "status": "initialized",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "jobs": {
            job.job_id: {
                "variant": job.variant,
                "case": job.case,
                "seed": job.seed,
                "case_dir": str(job.case_dir),
                "status": inspections[job.job_id]["status"],
                "validation_errors": list(inspections[job.job_id]["errors"]),
                "attempts_this_launch": 0,
            }
            for job in jobs
        },
    }


def run_subprocess_job(
    job: JobSpec,
    *,
    gpu: str,
    python_executable: str,
    log_root: Path,
) -> ProcessResult:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{job.job_id}.stdout.log"
    stderr_path = log_root / f"{job.job_id}.stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        retry = 2
        while True:
            candidate_stdout = log_root / f"{job.job_id}.attempt{retry:02d}.stdout.log"
            candidate_stderr = log_root / f"{job.job_id}.attempt{retry:02d}.stderr.log"
            if not candidate_stdout.exists() and not candidate_stderr.exists():
                stdout_path, stderr_path = candidate_stdout, candidate_stderr
                break
            retry += 1
    command = job.command(python_executable=python_executable, device=gpu)
    started = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
        completed = subprocess.run(
            command,
            cwd=repository_root(),
            env=environment,
            stdout=stdout_stream,
            stderr=stderr_stream,
            check=False,
        )
    return ProcessResult(
        returncode=int(completed.returncode),
        elapsed_seconds=time.perf_counter() - started,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


RunCallable = Callable[[JobSpec, str], ProcessResult]
ValidateCallable = Callable[[JobSpec], ValidationResult]


def execute_phase(
    jobs: Sequence[JobSpec],
    *,
    state: LauncherState,
    run_job: RunCallable,
    gpus: Sequence[str] = GPU_POOL,
    python_executable: str = sys.executable,
    expected_adapter_bundle_sha256: str | None = None,
    validate_job: ValidateCallable | None = None,
    max_attempts: int = 2,
) -> bool:
    """Execute pending jobs using one fixed worker per GPU and a shared queue.

    A process or artifact-validation failure is a technical failure, not a
    scientific decision.  Each record is therefore retried unchanged up to
    ``max_attempts`` while the remaining independent records continue.  The
    caller retains the A0-to-A1 cohort barrier after this phase returns.
    """

    if max_attempts < 1:
        raise LaunchError("max_attempts must be at least one")

    pending = [job for job in jobs if state.payload["jobs"][job.job_id]["status"] == "pending"]
    work: queue.Queue[tuple[JobSpec, int]] = queue.Queue()
    for job in pending:
        work.put((job, 1))

    def worker(gpu: str) -> None:
        while True:
            try:
                job, attempt = work.get_nowait()
            except queue.Empty:
                return
            state.update_job(
                job.job_id,
                status="running",
                gpu=gpu,
                started_utc=utc_now(),
                attempts_this_launch=attempt,
                command=job.command(python_executable=python_executable, device=gpu),
            )
            try:
                result = run_job(job, gpu)
                if result.returncode == 0:
                    validation = (
                        validate_job(job)
                        if validate_job is not None
                        else validate_done_summary(
                            job.case_dir,
                            variant=job.variant,
                            case=job.case,
                            seed=job.seed,
                            expected_steps=job.expected_steps,
                            a0_case_dir=job.a0_case_dir if job.variant == "A1" else None,
                            expected_adapter_bundle_sha256=expected_adapter_bundle_sha256,
                        )
                    )
                else:
                    validation = ValidationResult(
                        False, [f"runner exited {result.returncode}"]
                    )
                if result.returncode == 0 and validation.valid:
                    state.update_job(
                        job.job_id,
                        status="validated_done",
                        finished_utc=utc_now(),
                        elapsed_seconds=result.elapsed_seconds,
                        returncode=result.returncode,
                        stdout_path=str(result.stdout_path),
                        stderr_path=str(result.stderr_path),
                        validation_errors=[],
                    )
                else:
                    state.update_job(
                        job.job_id,
                        status="failed",
                        finished_utc=utc_now(),
                        elapsed_seconds=result.elapsed_seconds,
                        returncode=result.returncode,
                        stdout_path=str(result.stdout_path),
                        stderr_path=str(result.stderr_path),
                        validation_errors=validation.errors,
                    )
                    if attempt < max_attempts:
                        work.put((job, attempt + 1))
            except Exception as exc:  # retain state if orchestration itself fails
                state.update_job(
                    job.job_id,
                    status="failed",
                    finished_utc=utc_now(),
                    orchestration_error=f"{type(exc).__name__}: {exc}",
                )
                if attempt < max_attempts:
                    work.put((job, attempt + 1))
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return all(
        state.payload["jobs"][job.job_id]["status"] == "validated_done"
        for job in jobs
    )


def write_cohort_index(path: Path, jobs: Sequence[JobSpec], state: LauncherState) -> None:
    records = []
    for job in jobs:
        item = state.payload["jobs"][job.job_id]
        records.append(
            {
                "job_id": job.job_id,
                "variant": job.variant,
                "case": job.case,
                "seed": job.seed,
                "status": item["status"],
                "case_dir": str(job.case_dir),
                "summary": str(job.case_dir / "summary.json"),
            }
        )
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "mode": jobs[0].mode,
            "records": records,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(CASES_BY_MODE), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=LOCKED_SEED)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpus",
        default=",".join(GPU_POOL),
        help="comma-separated physical GPU indices; one optimizer is assigned to each",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="technical attempts per record; reruns preserve the locked record configuration",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--state-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_attempts < 1:
        raise LaunchError("--max-attempts must be at least one")
    gpus = parse_gpu_pool(args.gpus)
    data_root, output_root = validate_roots(args.data_root, args.output_root)
    jobs = build_jobs(mode=args.mode, data_root=data_root, output_root=output_root, seed=args.seed)
    run_root = jobs[0].execution_root
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (args.manifest_path or default_manifest_path(run_root, args.mode)).resolve()
    state_path = (args.state_path or default_state_path(run_root, args.mode)).resolve()
    for label, path in (("manifest", manifest_path), ("state", state_path)):
        if not _is_below(path, run_root):
            raise LaunchError(f"Custom {label} path must remain below the external execution root")

    data_audit = audit_prepared_data(data_root)
    gpu_audit = audit_gpu_pool(gpus)
    plan = immutable_plan_payload(
        jobs,
        python_executable=args.python,
        dataset_binding=data_audit.get("binding", {}),
        code_binding=code_sha256_binding(),
        gpus=gpus,
        technical_max_attempts=args.max_attempts,
    )
    plan_hash = plan_sha256(plan)
    approved_adapter_bundle = adapter_bundle_sha256()
    inspections = inspect_jobs(
        jobs, expected_adapter_bundle_sha256=approved_adapter_bundle
    )
    source_repo = external_repository()
    source_clean = _git_status(source_repo) == ""
    source_commit_matches = external_commit(source_repo) == AUTHOR_EXTERNAL_COMMIT
    manifest = build_dry_run_manifest(
        plan=plan,
        inspections=inspections,
        source_clean=source_clean,
        source_commit_matches=source_commit_matches,
        data_audit=data_audit,
        gpu_audit=gpu_audit,
    )

    if args.dry_run:
        if manifest["dry_run_passed"]:
            write_code_snapshot(run_root, plan["orchestration_code_sha256"])
        atomic_write_json(manifest_path, manifest)
        state = LauncherState(
            state_path,
            initial_state_payload(
                mode=args.mode, plan_hash=plan_hash, jobs=jobs, inspections=inspections
            ),
        )
        state.set_status("dry_run_passed" if manifest["dry_run_passed"] else "dry_run_blocked")
        print(f"Dry-run manifest: {manifest_path}")
        return 0 if manifest["dry_run_passed"] else 2

    if not manifest_path.is_file():
        raise LaunchError(f"Execution requires a prior dry-run manifest: {manifest_path}")
    approved = _load_json(manifest_path)
    if not approved.get("dry_run_passed"):
        raise LaunchError("The prior dry-run manifest did not pass")
    if approved.get("plan_sha256") != plan_hash:
        raise LaunchError("The current immutable launch plan differs from the dry-run manifest")
    verify_code_snapshot(run_root / "code_snapshot", plan["orchestration_code_sha256"])
    if not manifest["dry_run_passed"]:
        raise LaunchError("Current preflight no longer passes; regenerate the dry-run manifest")

    state = LauncherState(
        state_path,
        initial_state_payload(mode=args.mode, plan_hash=plan_hash, jobs=jobs, inspections=inspections),
    )
    state.set_status("running_a0")
    log_root = run_root / "launcher_logs" / args.mode

    def run_job(job: JobSpec, gpu: str) -> ProcessResult:
        return run_subprocess_job(
            job,
            gpu=gpu,
            python_executable=args.python,
            log_root=log_root,
        )

    a0_jobs = [job for job in jobs if job.variant == "A0"]
    if not execute_phase(
        a0_jobs,
        state=state,
        run_job=run_job,
        python_executable=args.python,
        expected_adapter_bundle_sha256=approved_adapter_bundle,
        max_attempts=args.max_attempts,
        gpus=gpus,
    ):
        for job in jobs:
            if job.variant == "A1" and state.payload["jobs"][job.job_id]["status"] == "pending":
                state.update_job(job.job_id, status="blocked_by_a0_barrier")
        state.set_status("failed_a0_phase")
        write_cohort_index(run_root / f"cohort_index_{args.mode}.json", jobs, state)
        return 1

    # Revalidate the complete A0 cohort before releasing any A1 work.
    for job in a0_jobs:
        validation = validate_done_summary(
            job.case_dir,
            variant="A0",
            case=job.case,
            seed=job.seed,
            expected_steps=job.expected_steps,
            expected_adapter_bundle_sha256=approved_adapter_bundle,
        )
        if not validation.valid:
            state.update_job(job.job_id, status="failed_barrier_revalidation", validation_errors=validation.errors)
            state.set_status("failed_a0_barrier")
            write_cohort_index(run_root / f"cohort_index_{args.mode}.json", jobs, state)
            return 1

    state.set_status("running_a1")
    a1_jobs = [job for job in jobs if job.variant == "A1"]
    if not execute_phase(
        a1_jobs,
        state=state,
        run_job=run_job,
        python_executable=args.python,
        expected_adapter_bundle_sha256=approved_adapter_bundle,
        max_attempts=args.max_attempts,
        gpus=gpus,
    ):
        state.set_status("failed_a1_phase")
        write_cohort_index(run_root / f"cohort_index_{args.mode}.json", jobs, state)
        return 1

    state.set_status("complete", finished_utc=utc_now())
    write_cohort_index(run_root / f"cohort_index_{args.mode}.json", jobs, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
