from __future__ import annotations

import ast
import csv
import inspect
import json
from pathlib import Path

import pytest

from scripts.experiments.lungct_dual_inr_controlled import launch_all


def _result(
    job: launch_all.Job,
    *,
    gpu: int,
    attempt: int,
    status: str,
) -> launch_all.JobResult:
    return launch_all.JobResult(
        variant=job.variant,
        case_id=job.case_id,
        run_mode=job.run_mode,
        gpu=gpu,
        attempt=attempt,
        status=status,
        returncode=0 if status in {"complete", "reused_complete"} else 1,
        elapsed_seconds=0.0,
        output_dir=f"/tmp/{job.key}",
        log_path=f"/tmp/{job.key}.log",
        started_utc="2026-08-02T00:00:00+00:00",
        finished_utc="2026-08-02T00:00:01+00:00",
    )


def test_production_and_smoke_job_matrices_are_complete_and_unique() -> None:
    full = launch_all.jobs_for("full")
    smoke = launch_all.jobs_for("smoke")

    assert len(full) == 40
    assert len({job.key for job in full}) == 40
    assert {(job.variant, job.case_id) for job in full} == {
        (variant, case_id)
        for variant in ("N0", "N1", "P0", "P1")
        for case_id in range(1, 11)
    }
    assert all(job.run_mode == "full" for job in full)

    assert len(smoke) == 4
    assert len({job.key for job in smoke}) == 4
    assert [(job.variant, job.case_id) for job in smoke] == [
        ("N0", 1),
        ("N1", 1),
        ("P0", 1),
        ("P1", 1),
    ]
    assert all(job.run_mode == "smoke" for job in smoke)


def test_output_root_containment_guards(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    native = tmp_path / "native"
    masks = tmp_path / "masks"
    ptv = tmp_path / "ptv"
    valid_output = tmp_path / "scratch" / "controlled"
    for root in (repo, native, masks, ptv):
        root.mkdir()

    assert launch_all.path_is_below(repo / "outputs", repo)
    assert launch_all.path_is_below(repo, repo)
    assert not launch_all.path_is_below(valid_output, repo)

    launch_all.validate_roots(
        repo_root=repo,
        native_raw_root=native,
        native_mask_root=masks,
        ptv_root=ptv,
        output_root=valid_output,
    )
    with pytest.raises(ValueError, match="outside the repository"):
        launch_all.validate_roots(
            repo_root=repo,
            native_raw_root=native,
            native_mask_root=masks,
            ptv_root=ptv,
            output_root=repo / "large_outputs",
        )
    with pytest.raises(ValueError, match="must not be below input data"):
        launch_all.validate_roots(
            repo_root=repo,
            native_raw_root=native,
            native_mask_root=masks,
            ptv_root=ptv,
            output_root=ptv / "large_outputs",
        )
    with pytest.raises(ValueError, match="must be absolute"):
        launch_all.validate_roots(
            repo_root=repo,
            native_raw_root=native,
            native_mask_root=masks,
            ptv_root=ptv,
            output_root=Path("relative-output"),
        )


def test_queue_continues_other_jobs_and_retries_only_to_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs = [
        launch_all.Job("N0", 1, "full"),
        launch_all.Job("N1", 1, "full"),
        launch_all.Job("P0", 1, "full"),
    ]
    calls: list[tuple[str, int]] = []

    def fake_run_job(
        *,
        job: launch_all.Job,
        gpu: int,
        attempt: int,
        **_kwargs: object,
    ) -> launch_all.JobResult:
        calls.append((job.variant, attempt))
        if job.variant == "N0" and attempt == 1:
            status = "failed"
        elif job.variant == "N1":
            status = "failed"
        else:
            status = "complete"
        return _result(job, gpu=gpu, attempt=attempt, status=status)

    monkeypatch.setattr(launch_all, "run_job", fake_run_job)
    results = launch_all.execute_queue(
        jobs=jobs,
        gpus=[0],
        max_attempts=2,
        kwargs={},
        tracker_path=tmp_path / "tracker.csv",
    )

    assert calls == [
        ("N0", 1),
        ("N1", 1),
        ("P0", 1),
        ("N0", 2),
        ("N1", 2),
    ]
    assert ("P0", 1) in launch_all.accepted_keys(results)
    assert ("N0", 1) in launch_all.accepted_keys(results)
    assert ("N1", 1) not in launch_all.accepted_keys(results)
    assert (tmp_path / "tracker.csv").is_file()
    assert (tmp_path / "tracker.json").is_file()


def test_scheduler_has_no_tre_threshold_and_accepts_poor_completed_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs = [
        launch_all.Job("N0", 1, "full"),
        launch_all.Job("P1", 10, "full"),
    ]
    attempted: list[str] = []

    def scientifically_poor_but_complete(
        *,
        job: launch_all.Job,
        gpu: int,
        attempt: int,
        **_kwargs: object,
    ) -> launch_all.JobResult:
        attempted.append(job.key)
        result = _result(job, gpu=gpu, attempt=attempt, status="complete")
        # The queue contract intentionally ignores scientific outcomes.
        result.tre_mm = float("inf")
        return result

    monkeypatch.setattr(launch_all, "run_job", scientifically_poor_but_complete)
    results = launch_all.execute_queue(
        jobs=jobs,
        gpus=[0],
        max_attempts=2,
        kwargs={},
        tracker_path=tmp_path / "no_tre_gate.csv",
    )

    assert attempted == [job.key for job in jobs]
    assert launch_all.accepted_keys(results) == {("N0", 1), ("P1", 10)}
    scheduler_tree = ast.parse(inspect.getsource(launch_all.execute_queue))
    scheduler_names = {
        value.lower()
        for node in ast.walk(scheduler_tree)
        for value in (
            [node.id]
            if isinstance(node, ast.Name)
            else [node.attr]
            if isinstance(node, ast.Attribute)
            else []
        )
    }
    assert not {
        name for name in scheduler_names if name == "tre" or name.startswith("tre_")
    }


def test_tracker_serializes_run_paths_portably(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "scratch" / "controlled"
    tracker = tmp_path / "review" / "tracker.csv"
    job = launch_all.Job("N0", 1, "full")

    def fake_run_job(**kwargs: object) -> launch_all.JobResult:
        return launch_all.JobResult(
            variant="N0",
            case_id=1,
            run_mode="full",
            gpu=int(kwargs["gpu"]),
            attempt=int(kwargs["attempt"]),
            status="complete",
            returncode=0,
            elapsed_seconds=1.0,
            output_dir=str(run_root / "n0" / "case_01"),
            log_path=str(run_root / "launcher_logs" / "n0.log"),
            started_utc="2026-08-03T00:00:00+00:00",
            finished_utc="2026-08-03T00:00:01+00:00",
        )

    monkeypatch.setattr(launch_all, "run_job", fake_run_job)
    launch_all.execute_queue(
        jobs=[job],
        gpus=[0],
        max_attempts=1,
        kwargs={},
        tracker_path=tracker,
        portable_roots=((run_root, "${RUN_ROOT}"),),
    )

    with tracker.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    payload = json.loads(tracker.with_suffix(".json").read_text(encoding="utf-8"))
    assert row["output_dir"] == "${RUN_ROOT}/n0/case_01"
    assert row["log_path"] == "${RUN_ROOT}/launcher_logs/n0.log"
    assert payload["attempt_records"][0]["output_dir"] == row["output_dir"]
