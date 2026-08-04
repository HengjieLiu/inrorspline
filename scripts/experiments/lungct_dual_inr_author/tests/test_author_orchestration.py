from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import pytest

from scripts.experiments.lungct_dual_inr_author.aggregate_author import (
    AggregationError,
    OUTPUT_FILENAMES,
    aggregate_cohort,
)
from scripts.experiments.lungct_dual_inr_author.external_api import (
    AUTHOR_EXTERNAL_COMMIT,
)
from scripts.experiments.lungct_dual_inr_author.posthoc_author_regularity import (
    COMMON_SOURCE,
    POSTHOC_SOURCE,
    REPORT_METRICS,
)
from scripts.experiments.lungct_dual_inr_author.launch_author import (
    GPU_POOL,
    HISTORY_NUMERIC_FIELDS,
    IDENTITY_TRE_MM,
    LOCKED_SEED,
    REQUIRED_CASE_FILES,
    LaunchError,
    LauncherState,
    ProcessResult,
    ValidationResult,
    adapter_bundle_sha256,
    audit_prepared_data,
    atomic_write_json,
    build_jobs,
    code_sha256_binding,
    execute_phase,
    file_sha256,
    initial_state_payload,
    inspect_jobs,
    immutable_plan_payload,
    validate_done_summary,
    verify_code_snapshot,
    write_code_snapshot,
)


def _stage_starts(steps: int) -> list[int]:
    return [0] if steps < 3_000 else [0, 1_000, 2_000]


def _write_synthetic_case(
    run_root: Path,
    *,
    variant: str,
    case: int,
    steps: int,
    dense_snap_tre: float,
    starts: list[int] | None = None,
) -> Path:
    case_dir = run_root / variant.lower() / f"case_{case:02d}"
    case_dir.mkdir(parents=True)
    starts = list(starts or _stage_starts(steps))
    run_mode = "short_smoke" if steps != 3_000 else "full_length"
    smoke_steps = steps if run_mode == "short_smoke" else None
    policy = "author_dynamic_convergence" if variant == "A0" else "replay_a0"
    a0_trace = run_root / "a0" / f"case_{case:02d}" / "stage_transitions.json"
    bundle = adapter_bundle_sha256()

    metrics = {
        "initial": {
            "snap": {"n": 300, "mean_mm": IDENTITY_TRE_MM[case]},
            "subvoxel": {"n": 300, "mean_mm": IDENTITY_TRE_MM[case]},
        },
        "direct_inr": {
            "snap": {"n": 300, "mean_mm": dense_snap_tre + 0.01},
            "subvoxel": {"n": 300, "mean_mm": dense_snap_tre + 0.03},
        },
        "dense_dvf": {
            "snap": {"n": 300, "mean_mm": dense_snap_tre},
            "subvoxel": {"n": 300, "mean_mm": dense_snap_tre + 0.02},
        },
        "direct_dense_agreement": {
            "snap_mean_tre_absolute_difference_mm": 0.01,
            "subvoxel_mean_tre_absolute_difference_mm": 0.01,
        },
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "variant": variant,
        "case": case,
        "seed": LOCKED_SEED,
        "run_mode": run_mode,
        "smoke_steps": smoke_steps,
        "fixed_phase": 0,
        "moving_phase": 5,
        "stage_start_steps": starts,
        "stage_policy": policy,
        "a0_transition_trace": str(a0_trace.resolve()) if variant == "A1" else None,
        "landmark_metrics": metrics,
        "five_lobe_dice": {"mean_dice": 0.9 if variant == "A0" else 0.89},
        "runtime": {
            "training_wall_seconds": 10.0 + case,
            "total_wall_seconds_through_evaluation": 12.0 + case,
        },
        "cuda_peak_memory_bytes": {
            "training_allocated": 1_000,
            "training_reserved": 2_000,
            "total_allocated": 1_500,
            "total_reserved": 2_500,
        },
    }
    atomic_write_json(case_dir / "summary.json", summary)
    atomic_write_json(
        case_dir / "RUN_STATE.json",
        {
            "status": "complete",
            "variant": variant,
            "case": case,
            "seed": LOCKED_SEED,
            "run_mode": run_mode,
            "smoke_steps": smoke_steps,
        },
    )
    atomic_write_json(
        case_dir / "resolved_config.json",
        {
            "loss": {"mask_weight": 0.1 if variant == "A0" else 0.0},
            "training": {"total_steps": steps},
            "export": {
                "case": case,
                "seed": LOCKED_SEED,
                "run_mode": run_mode,
                "adapter_bundle_sha256": bundle,
            },
        },
    )
    atomic_write_json(
        case_dir / "provenance.json",
        {
            "external_commit": AUTHOR_EXTERNAL_COMMIT,
            "external_git_status_porcelain": "",
            "adapter_bundle_sha256": bundle,
            "adapter_source_bundle": {"bundle_sha256": bundle},
            "variant": variant,
            "case": case,
            "seed": LOCKED_SEED,
        },
    )
    atomic_write_json(
        case_dir / "stage_transitions.json",
        {
            "schema_version": 1,
            "source_variant": variant,
            "case": case,
            "seed": LOCKED_SEED,
            "total_steps": steps,
            "blur_sigmas": [4.0, 2.0, 0.0],
            "stage_start_steps": starts,
            "events": [],
            "policy": policy,
            "replayed_manifest_path": str(a0_trace.resolve()) if variant == "A1" else None,
        },
    )

    with (case_dir / "stage_trace.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "stage_index",
                "blur_sigma",
                "start_zero_step",
                "end_zero_step_inclusive",
                "trigger",
            ),
        )
        writer.writeheader()
        for index, start in enumerate(starts):
            end = starts[index + 1] - 1 if index + 1 < len(starts) else steps - 1
            writer.writerow(
                {
                    "stage_index": index,
                    "blur_sigma": [4.0, 2.0, 0.0][index],
                    "start_zero_step": start,
                    "end_zero_step_inclusive": end,
                    "trigger": "initial" if index == 0 else policy,
                }
            )

    with (case_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=(*HISTORY_NUMERIC_FIELDS, "fine_active")
        )
        writer.writeheader()
        for zero_step in range(steps):
            stage = max(index for index, start in enumerate(starts) if start <= zero_step)
            row = {field: 0.1 for field in HISTORY_NUMERIC_FIELDS}
            row.update(
                {
                    "zero_step": zero_step,
                    "step": zero_step + 1,
                    "stage_index": stage,
                    "blur_sigma": [4.0, 2.0, 0.0][stage],
                    "loss_total": 1.0 + 0.001 * zero_step,
                    "loss_similarity": 0.7,
                    "loss_spatial_weighted": 0.2,
                    "loss_mask_raw": 1.0,
                    "loss_mask_weighted": 0.1 if variant == "A0" else 0.0,
                    "fine_active": stage >= 1,
                }
            )
            writer.writerow(row)

    atomic_write_json(case_dir / "landmark_metrics.json", metrics)
    atomic_write_json(case_dir / "five_lobe_dice.json", summary["five_lobe_dice"])
    for name in REQUIRED_CASE_FILES:
        path = case_dir / name
        if path.exists():
            continue
        if name == "DONE":
            path.touch()
        elif name == "stderr.log":
            path.touch()
        else:
            path.write_text("synthetic\n", encoding="utf-8")
    return case_dir


def _write_launcher_binding(run_root: Path, mode: str, data_root: Path) -> None:
    binding = code_sha256_binding()
    write_code_snapshot(run_root, binding)
    atomic_write_json(
        run_root / f"launcher_manifest_{mode}.json",
        {
            "dry_run_passed": True,
            "plan_sha256": "synthetic-plan",
            "plan": {
                "mode": mode,
                "data_root": str(data_root.resolve()),
                "adapter_source_bundle_sha256": adapter_bundle_sha256(),
                "orchestration_code_sha256": binding,
            },
        },
    )


def _write_posthoc_gate(
    run_root: Path,
    data_root: Path,
    posthoc_root: Path,
    *,
    mode: str,
    cases: tuple[int, ...],
) -> None:
    posthoc_root.mkdir(parents=True)
    common_hash = file_sha256(COMMON_SOURCE)
    posthoc_source = {
        "path": str(POSTHOC_SOURCE),
        "repository_relative_path": POSTHOC_SOURCE.relative_to(
            POSTHOC_SOURCE.parents[3]
        ).as_posix(),
        "size_bytes": POSTHOC_SOURCE.stat().st_size,
        "sha256": file_sha256(POSTHOC_SOURCE),
    }
    outputs = []
    for case in cases:
        a0_metrics = {metric: float(index + case) for index, metric in enumerate(REPORT_METRICS)}
        a1_metrics = {metric: value + 0.25 for metric, value in a0_metrics.items()}
        payload = {
            "schema_version": 1,
            "case": case,
            "seed": LOCKED_SEED,
            "run_mode": "short_smoke" if mode == "smoke" else "full_length",
            "posthoc_source": posthoc_source,
            "a0": {"metrics": a0_metrics},
            "a1": {"metrics": a1_metrics},
            "a1_minus_a0": {
                metric: a1_metrics[metric] - a0_metrics[metric]
                for metric in REPORT_METRICS
            },
            "metric_source": {"sha256": common_hash},
            "inputs": {
                "a0_dense_dvf": str(
                    (run_root / "a0" / f"case_{case:02d}" / "dense_dvf_crop_voxel_xyz.npy").resolve()
                ),
                "a1_dense_dvf": str(
                    (run_root / "a1" / f"case_{case:02d}" / "dense_dvf_crop_voxel_xyz.npy").resolve()
                ),
                "fixed_segmentation": str(
                    (data_root / f"case_{case:02d}" / "segmentations_00.nii").resolve()
                ),
            },
        }
        json_path = posthoc_root / f"case_{case:02d}_paired_regularity.json"
        csv_path = posthoc_root / f"case_{case:02d}_paired_regularity.csv"
        markdown_path = posthoc_root / f"case_{case:02d}_paired_regularity.md"
        atomic_write_json(json_path, payload)
        csv_path.write_text("case,variant\nsynthetic,A0\n", encoding="utf-8")
        markdown_path.write_text("# Synthetic posthoc gate\n", encoding="utf-8")
        outputs.append(
            {
                "case": case,
                "csv": str(csv_path.resolve()),
                "json": str(json_path.resolve()),
                "markdown": str(markdown_path.resolve()),
            }
        )
    atomic_write_json(
        posthoc_root / "posthoc_author_regularity_manifest.json",
        {
            "schema_version": 1,
            "run_root": str(run_root.resolve()),
            "data_root": str(data_root.resolve()),
            "review_root": str(posthoc_root.resolve()),
            "cases": list(cases),
            "posthoc_source": posthoc_source,
            "common_source_sha256": common_hash,
            "conversion": "(X,Y,Z,xyz) -> (Z,Y,X,zyx)",
            "outputs": outputs,
        },
    )


def test_job_matrix_locks_smoke_nesting_and_production_commands(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "runs"
    smoke = build_jobs(mode="smoke", data_root=data_root, output_root=output_root)
    assert len(smoke) == 2
    assert all(job.execution_root == output_root / "smoke_3steps" for job in smoke)
    for job in smoke:
        command = job.command(python_executable="python", device="2")
        assert command[command.index("--smoke-steps") + 1] == "3"
    smoke_a1 = next(job for job in smoke if job.variant == "A1")
    smoke_command = smoke_a1.command(python_executable="python", device="2")
    assert smoke_command[smoke_command.index("--replay-a0-transitions") + 1] == str(
        output_root / "smoke_3steps"
    )

    production = build_jobs(mode="production", data_root=data_root, output_root=output_root)
    assert len(production) == 20
    assert all("--smoke-steps" not in job.command(python_executable="python", device="0") for job in production)
    assert all(job.trace_root == output_root for job in production if job.variant == "A1")

    code_binding = code_sha256_binding()
    dataset_binding = {
        "dataset_manifest_sha256": "a" * 64,
        "preprocess_contract_sha256": "b" * 64,
        "checksums_manifest_sha256": "c" * 64,
    }
    plan = immutable_plan_payload(
        production,
        python_executable="python",
        dataset_binding=dataset_binding,
        code_binding=code_binding,
    )
    assert plan["prepared_data_sha256"] == dataset_binding
    assert plan["orchestration_code_sha256"] == code_binding
    assert plan["adapter_source_bundle_sha256"] == adapter_bundle_sha256()


def test_preflight_requires_data_manifests_and_code_snapshot_is_tamper_evident(
    tmp_path: Path,
) -> None:
    empty_data = tmp_path / "empty_data"
    empty_data.mkdir()
    audit = audit_prepared_data(empty_data)
    assert audit["status"] == "fail"
    error_text = " ".join(audit["errors"])
    assert all(
        name in error_text
        for name in ("dataset_manifest.json", "preprocess_contract.json", "checksums.sha256")
    )

    binding = code_sha256_binding()
    snapshot = write_code_snapshot(tmp_path / "run", binding)
    verify_code_snapshot(snapshot, binding)
    relative = next(iter(binding))
    (snapshot / relative).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(LaunchError, match="missing or changed"):
        verify_code_snapshot(snapshot, binding)


def test_execute_phase_has_one_worker_per_gpu_with_shared_work_queue(tmp_path: Path) -> None:
    jobs = [
        job
        for job in build_jobs(
            mode="production", data_root=tmp_path / "data", output_root=tmp_path / "runs"
        )
        if job.variant == "A0"
    ]
    inspections = {job.job_id: {"status": "pending", "errors": []} for job in jobs}
    state = LauncherState(
        tmp_path / "state.json",
        initial_state_payload(
            mode="production", plan_hash="test", jobs=jobs, inspections=inspections
        ),
    )
    active = {gpu: 0 for gpu in GPU_POOL}
    assignments: list[str] = []
    lock = threading.Lock()

    def fake_run(job, gpu):
        with lock:
            assert active[gpu] == 0
            active[gpu] += 1
            assignments.append(gpu)
        time.sleep(0.005)
        with lock:
            active[gpu] -= 1
        return ProcessResult(0, 0.005, tmp_path / "out.log", tmp_path / "err.log")

    assert execute_phase(
        jobs,
        state=state,
        run_job=fake_run,
        validate_job=lambda job: ValidationResult(True, [], {}),
    )
    assert set(assignments) == set(GPU_POOL)
    assert all(state.payload["jobs"][job.job_id]["status"] == "validated_done" for job in jobs)


def test_execute_phase_retries_technical_failure_without_blocking_other_jobs(tmp_path: Path) -> None:
    jobs = [
        job
        for job in build_jobs(
            mode="pilot", data_root=tmp_path / "data", output_root=tmp_path / "runs"
        )
        if job.variant == "A0"
    ]
    inspections = {job.job_id: {"status": "pending", "errors": []} for job in jobs}
    state = LauncherState(
        tmp_path / "state_retry.json",
        initial_state_payload(mode="pilot", plan_hash="test", jobs=jobs, inspections=inspections),
    )
    calls: dict[str, int] = {}

    def fake_run(job, gpu):
        calls[job.job_id] = calls.get(job.job_id, 0) + 1
        return ProcessResult(
            1 if job.case == 1 and calls[job.job_id] == 1 else 0,
            0.001,
            tmp_path / "out.log",
            tmp_path / "err.log",
        )

    assert execute_phase(
        jobs,
        state=state,
        run_job=fake_run,
        validate_job=lambda job: ValidationResult(True, [], {}),
        max_attempts=2,
    )
    assert calls == {job.job_id: (2 if job.case == 1 else 1) for job in jobs}
    assert all(state.payload["jobs"][job.job_id]["status"] == "validated_done" for job in jobs)


def test_skip_validation_requires_every_done_artifact_and_rejects_empty_case_dir(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs" / "smoke_3steps"
    case_dir = _write_synthetic_case(
        run_root, variant="A0", case=1, steps=3, dense_snap_tre=1.0
    )
    result = validate_done_summary(
        case_dir,
        variant="A0",
        case=1,
        seed=LOCKED_SEED,
        expected_steps=3,
        expected_adapter_bundle_sha256=adapter_bundle_sha256(),
    )
    assert result.valid, result.errors
    (case_dir / "DONE").unlink()
    result = validate_done_summary(
        case_dir,
        variant="A0",
        case=1,
        seed=LOCKED_SEED,
        expected_steps=3,
        expected_adapter_bundle_sha256=adapter_bundle_sha256(),
    )
    assert not result.valid

    empty_root = tmp_path / "empty"
    job = build_jobs(mode="smoke", data_root=tmp_path / "data", output_root=empty_root)[0]
    job.case_dir.mkdir(parents=True)
    inspected = inspect_jobs(
        [job], expected_adapter_bundle_sha256=adapter_bundle_sha256()
    )
    assert inspected[job.job_id]["status"] == "invalid_existing"


def test_aggregate_requires_complete_pairs_and_writes_review_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "pilot_runs"
    for case in (1, 8):
        _write_synthetic_case(
            run_root,
            variant="A0",
            case=case,
            steps=3_000,
            dense_snap_tre=1.0 + case / 100,
        )
        _write_synthetic_case(
            run_root,
            variant="A1",
            case=case,
            steps=3_000,
            dense_snap_tre=1.1 + case / 100,
        )
    data_root = tmp_path / "data"
    for case in (1, 8):
        case_root = data_root / f"case_{case:02d}"
        case_root.mkdir(parents=True)
        (case_root / "segmentations_00.nii").write_text("synthetic\n", encoding="utf-8")
    _write_launcher_binding(run_root, "pilot", data_root)
    posthoc_root = tmp_path / "posthoc"
    _write_posthoc_gate(
        run_root, data_root, posthoc_root, mode="pilot", cases=(1, 8)
    )
    review_root = tmp_path / "review"
    payload = aggregate_cohort(
        run_root=run_root,
        posthoc_root=posthoc_root,
        review_root=review_root,
        mode="pilot",
    )
    assert payload["record_count"] == 4
    assert payload["paired_case_count"] == 2
    assert payload["all_realized_boundaries_identical"] is True
    assert payload["common_posthoc_regularity_gate"]["status"] == "pass"
    assert all((review_root / name).is_file() for name in OUTPUT_FILENAMES)
    assert "Common posthoc regularity" in (review_root / "dual_author_report.md").read_text()
    deltas = payload["paired_deltas_a1_minus_a0"]
    assert all(row["delta_dense_snap_tre_mm"] == pytest.approx(0.1) for row in deltas)

    # A report generated by different posthoc source bytes cannot enter the aggregate.
    posthoc_manifest_path = posthoc_root / "posthoc_author_regularity_manifest.json"
    posthoc_manifest = json.loads(posthoc_manifest_path.read_text(encoding="utf-8"))
    accepted_posthoc_source = dict(posthoc_manifest["posthoc_source"])
    posthoc_manifest["posthoc_source"]["sha256"] = "0" * 64
    atomic_write_json(posthoc_manifest_path, posthoc_manifest)
    with pytest.raises(AggregationError, match="generator provenance"):
        aggregate_cohort(
            run_root=run_root,
            posthoc_root=posthoc_root,
            review_root=tmp_path / "stale_posthoc_review",
            mode="pilot",
        )
    posthoc_manifest["posthoc_source"] = accepted_posthoc_source
    atomic_write_json(posthoc_manifest_path, posthoc_manifest)

    # A missing accepted artifact blocks aggregation before any partial report.
    (run_root / "a1" / "case_08" / "DONE").unlink()
    with pytest.raises(AggregationError, match="Complete paired-cohort gate failed"):
        aggregate_cohort(
            run_root=run_root,
            posthoc_root=posthoc_root,
            review_root=tmp_path / "blocked_review",
            mode="pilot",
        )
