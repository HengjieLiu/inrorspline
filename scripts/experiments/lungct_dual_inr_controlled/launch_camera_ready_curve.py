#!/usr/bin/env python3
"""Launch the compact camera-ready DUAL-INR TRE/stdlogJ curve sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments.lungct_dual_inr_author.portable_paths import (
    portable_review_value,
)
from scripts.experiments.lungct_dual_inr_controlled.launch_all import (
    JobResult,
    archive_incomplete,
    parse_gpus,
    path_is_below,
    resolve_root,
)


VARIANTS = ("PL1D", "PL1B", "PL1A", "PL1C")
CASES = tuple(range(1, 11))
NONDEFAULT_NATIVE_MULTIPLIERS = (0.01, 0.1, 10.0, 100.0)
NONDEFAULT_PTV_WEIGHTS = (0.0055, 0.017, 0.17, 0.55)
RUNNER_MODULE = "scripts.experiments.lungct_dual_inr_controlled.run_one"
AGGREGATE_MODULE = (
    "scripts.experiments.lungct_dual_inr_controlled.aggregate_camera_ready_curve"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        if fieldnames:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def format_token(value: float) -> str:
    text = f"{float(value):.8g}".replace("-", "m").replace(".", "p")
    return text.replace("+", "")


@dataclass(frozen=True)
class SweepJob:
    variant: str
    case_id: int
    run_tag: str
    weight_label: str
    regularization_multiplier: float
    ptv_isotv_weight: float | None

    @property
    def key(self) -> str:
        return f"{self.variant.lower()}_{self.weight_label}_case{self.case_id:02d}"


def sweep_jobs() -> list[SweepJob]:
    jobs: list[SweepJob] = []
    for variant in ("PL1D", "PL1B", "PL1A"):
        for multiplier in NONDEFAULT_NATIVE_MULTIPLIERS:
            label = f"m{format_token(multiplier)}"
            tag = f"{variant.lower()}_{label}"
            for case_id in CASES:
                jobs.append(
                    SweepJob(
                        variant=variant,
                        case_id=case_id,
                        run_tag=tag,
                        weight_label=label,
                        regularization_multiplier=float(multiplier),
                        ptv_isotv_weight=None,
                    )
                )
    for weight in NONDEFAULT_PTV_WEIGHTS:
        label = f"w{format_token(weight)}"
        tag = f"pl1c_{label}"
        for case_id in CASES:
            jobs.append(
                SweepJob(
                    variant="PL1C",
                    case_id=case_id,
                    run_tag=tag,
                    weight_label=label,
                    regularization_multiplier=1.0,
                    ptv_isotv_weight=float(weight),
                )
            )
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate Stage 2 sweep job keys")
    return jobs


def case_output(root: Path, job: SweepJob) -> Path:
    return root / job.run_tag / job.variant.lower() / f"case_{job.case_id:02d}"


def validate_roots(
    *,
    repo_root: Path,
    native_raw_root: Path,
    native_mask_root: Path,
    ptv_root: Path,
    output_root: Path,
) -> None:
    for label, path in (
        ("repository", repo_root),
        ("native raw", native_raw_root),
        ("native mask", native_mask_root),
        ("pTV", ptv_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} root is missing: {path}")
    if not output_root.is_absolute():
        raise ValueError("output root must be absolute")
    if path_is_below(output_root, repo_root):
        raise ValueError(f"large output root must remain outside the repository: {output_root}")
    for data_root in (native_raw_root, native_mask_root, ptv_root):
        if path_is_below(output_root, data_root):
            raise ValueError(f"output root must not be below input data: {output_root}")


def run_one_job(
    *,
    job: SweepJob,
    gpu: int,
    attempt: int,
    repo_root: Path,
    native_raw_root: Path,
    native_mask_root: Path,
    ptv_root: Path,
    run_root: Path,
    seed: int,
    eval_chunk_size: int,
) -> JobResult:
    output_dir = case_output(run_root, job)
    if (output_dir / "DONE").is_file():
        now = utc_now()
        return JobResult(
            variant=job.variant,
            case_id=job.case_id,
            run_mode=job.weight_label,
            gpu=gpu,
            attempt=attempt,
            status="reused_complete",
            returncode=0,
            elapsed_seconds=0.0,
            output_dir=str(output_dir),
            log_path="",
            started_utc=now,
            finished_utc=now,
        )
    archived = archive_incomplete(output_dir)
    log_dir = run_root / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.key}_attempt{attempt:02d}_gpu{gpu}.log"
    command = [
        sys.executable,
        "-m",
        RUNNER_MODULE,
        "--variant",
        job.variant,
        "--case-id",
        str(job.case_id),
        "--device",
        "cuda:0",
        "--seed",
        str(seed),
        "--native-raw-root",
        str(native_raw_root),
        "--native-mask-root",
        str(native_mask_root),
        "--ptv-root",
        str(ptv_root),
        "--output-root",
        str(run_root),
        "--run-tag",
        job.run_tag,
        "--regularization-multiplier",
        str(job.regularization_multiplier),
        "--eval-chunk-size",
        str(eval_chunk_size),
    ]
    if job.ptv_isotv_weight is not None:
        command.extend(["--ptv-isotv-weight", str(job.ptv_isotv_weight)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    started_utc = utc_now()
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"started_utc={started_utc}\n")
        stream.write(f"gpu_physical={gpu}\n")
        stream.write(f"attempt={attempt}\n")
        stream.write(f"run_tag={job.run_tag}\n")
        stream.write(f"weight_label={job.weight_label}\n")
        stream.write(f"archived_incomplete={archived}\n")
        stream.write("command=" + " ".join(command) + "\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    success = completed.returncode == 0 and (output_dir / "DONE").is_file()
    return JobResult(
        variant=job.variant,
        case_id=job.case_id,
        run_mode=job.weight_label,
        gpu=gpu,
        attempt=attempt,
        status="complete" if success else "failed",
        returncode=int(completed.returncode),
        elapsed_seconds=elapsed,
        output_dir=str(output_dir),
        log_path=str(log_path),
        started_utc=started_utc,
        finished_utc=utc_now(),
    )


def execute_variant_queue(
    *,
    variant: str,
    gpu: int,
    jobs: Sequence[SweepJob],
    max_attempts: int,
    kwargs: dict[str, Any],
    tracker_path: Path,
    portable_roots: Sequence[tuple[Path, str]],
    lock: threading.Lock,
    results: list[JobResult],
) -> None:
    variant_jobs = [job for job in jobs if job.variant == variant]
    for job in variant_jobs:
        result: JobResult | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = run_one_job(job=job, gpu=gpu, attempt=attempt, **kwargs)
            except BaseException as exc:
                now = utc_now()
                emergency_dir = Path(kwargs["run_root"]) / "launcher_logs"
                emergency_dir.mkdir(parents=True, exist_ok=True)
                emergency_path = emergency_dir / (
                    f"{job.key}_attempt{attempt:02d}_gpu{gpu}_launcher_exception.log"
                )
                emergency_path.write_text(
                    f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
                result = JobResult(
                    variant=job.variant,
                    case_id=job.case_id,
                    run_mode=job.weight_label,
                    gpu=gpu,
                    attempt=attempt,
                    status="failed",
                    returncode=-1,
                    elapsed_seconds=0.0,
                    output_dir=str(case_output(Path(kwargs["run_root"]), job)),
                    log_path=str(emergency_path),
                    started_utc=now,
                    finished_utc=now,
                )
            with lock:
                results.append(result)
                rows = [asdict(row) for row in results]
                write_csv(tracker_path, portable_review_value(rows, portable_roots))
                write_json(
                    tracker_path.with_suffix(".json"),
                    {
                        "updated_utc": utc_now(),
                        "attempt_records": portable_review_value(rows, portable_roots),
                    },
                )
                print(
                    f"[{result.finished_utc}] {job.variant} {job.weight_label} "
                    f"case{job.case_id:02d} attempt={attempt} gpu={gpu} "
                    f"status={result.status} elapsed={result.elapsed_seconds:.1f}s",
                    flush=True,
                )
            if result.status in {"complete", "reused_complete"}:
                break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dry-run", "full"), default="dry-run")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-chunk-size", type=int, default=262144)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--native-raw-root", type=Path, default=None)
    parser.add_argument("--native-mask-root", type=Path, default=None)
    parser.add_argument("--ptv-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--skip-aggregate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be >=1")
    repo_root = resolve_root(args.repo_root, "INRORSPLINE_REPO_ROOT")
    native_raw_root = resolve_root(
        args.native_raw_root, "INRORSPLINE_DIRLAB_NATIVE_4DCT_ROOT"
    )
    native_mask_root = resolve_root(
        args.native_mask_root, "INRORSPLINE_DIRLAB_NATIVE_MASK_ROOT"
    )
    ptv_root = resolve_root(args.ptv_root, "INRORSPLINE_DIRLAB_PTVREG_ISO1MM_ROOT")
    output_root = resolve_root(
        args.output_root, "INRORSPLINE_DUAL_CAMERA_READY_STAGE2_RUN_ROOT"
    )
    review_root = resolve_root(
        args.review_root, "INRORSPLINE_DUAL_CAMERA_READY_STAGE2_REVIEW_ROOT"
    )
    validate_roots(
        repo_root=repo_root,
        native_raw_root=native_raw_root,
        native_mask_root=native_mask_root,
        ptv_root=ptv_root,
        output_root=output_root,
    )
    gpus = parse_gpus(args.gpus)
    if len(gpus) < len(VARIANTS):
        raise ValueError(f"Stage 2 needs at least {len(VARIANTS)} GPUs, got {gpus}")
    gpu_by_variant = dict(zip(VARIANTS, gpus[: len(VARIANTS)]))
    jobs = sweep_jobs()
    review_root.mkdir(parents=True, exist_ok=True)
    portable_roots = (
        (native_raw_root, "${NATIVE_RAW_ROOT}"),
        (native_mask_root, "${NATIVE_MASK_ROOT}"),
        (ptv_root, "${PTV_ROOT}"),
        (output_root, "${RUN_ROOT}"),
        (review_root, "${REVIEW_ROOT}"),
    )
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": args.stage,
        "variants": list(VARIANTS),
        "cases": list(CASES),
        "new_job_count": len(jobs),
        "default_points_reused": {
            "PL1D": "multiplier 1 from P27",
            "PL1B": "multiplier 1 from P27",
            "PL1A": "multiplier 1 from P28",
            "PL1C": "weight 0.055 from P29 Stage 1",
        },
        "gpu_by_variant": gpu_by_variant,
        "one_variant_per_gpu": True,
        "seed": int(args.seed),
        "technical_retry_limit": int(args.max_attempts),
        "scientific_stop_rule": "none; every nondefault independent job is attempted",
        "external_paths": {
            "native_raw_root": str(native_raw_root),
            "native_mask_root": str(native_mask_root),
            "ptv_root": str(ptv_root),
            "output_root": str(output_root),
        },
        "jobs": [asdict(job) for job in jobs],
    }
    write_json(
        review_root / "stage2_compact_sweep_resolved_job_manifest.json",
        portable_review_value(manifest, portable_roots),
    )
    print(json.dumps(manifest, indent=2), flush=True)
    if args.stage == "dry-run":
        return

    common_kwargs = {
        "repo_root": repo_root,
        "native_raw_root": native_raw_root,
        "native_mask_root": native_mask_root,
        "ptv_root": ptv_root,
        "run_root": output_root,
        "seed": int(args.seed),
        "eval_chunk_size": int(args.eval_chunk_size),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[JobResult] = []
    lock = threading.Lock()
    tracker_path = review_root / "stage2_compact_sweep_tracker.csv"
    threads = [
        threading.Thread(
            target=execute_variant_queue,
            kwargs={
                "variant": variant,
                "gpu": gpu_by_variant[variant],
                "jobs": jobs,
                "max_attempts": int(args.max_attempts),
                "kwargs": common_kwargs,
                "tracker_path": tracker_path,
                "portable_roots": portable_roots,
                "lock": lock,
                "results": results,
            },
            daemon=False,
        )
        for variant in VARIANTS
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    complete = {
        (row.variant, row.run_mode, int(row.case_id))
        for row in results
        if row.status in {"complete", "reused_complete"}
    }
    expected = {(job.variant, job.weight_label, job.case_id) for job in jobs}
    write_json(
        review_root / "stage2_compact_sweep_completion.json",
        {
            "updated_utc": utc_now(),
            "all_complete": complete == expected,
            "accepted_count": len(complete),
            "expected_count": len(expected),
            "missing": sorted(
                f"{variant}_{weight}_case{case:02d}"
                for variant, weight, case in expected - complete
            ),
            "attempt_count": len(results),
            "one_variant_per_gpu": True,
            "gpu_by_variant": gpu_by_variant,
        },
    )
    if not args.skip_aggregate:
        command = [
            sys.executable,
            "-m",
            AGGREGATE_MODULE,
            "--stage2-run-root",
            str(output_root),
            "--review-root",
            str(review_root),
        ]
        subprocess.run(command, cwd=repo_root, check=True)
    if complete != expected:
        raise SystemExit(
            f"all jobs were attempted, but {len(expected - complete)} remain failed"
        )


if __name__ == "__main__":
    main()
