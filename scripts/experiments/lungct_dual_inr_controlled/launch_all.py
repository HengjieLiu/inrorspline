#!/usr/bin/env python3
"""Validate, smoke, and run all controlled DUAL DIR-Lab jobs on a GPU pool.

Every independent full job is attempted even when another job fails.  TRE or
other scientific outcomes are never used for scheduling or early stopping.
Only process-level technical failures are retried.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import shutil
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


VARIANTS = ("N0", "N1", "P0", "P1")
CASES = tuple(range(1, 11))
RUNNER_MODULE = "scripts.experiments.lungct_dual_inr_controlled.run_one"
AGGREGATE_MODULE = "scripts.experiments.lungct_dual_inr_controlled.aggregate"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def path_is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_root(value: Path | None, env_name: str) -> Path:
    candidate = value or (Path(os.environ[env_name]) if os.environ.get(env_name) else None)
    if candidate is None:
        raise ValueError(f"provide the path argument or set {env_name}")
    return candidate.resolve()


def parse_gpus(text: str) -> list[int]:
    text = str(text).strip().lower()
    if text == "auto":
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        values = [int(line.strip()) for line in probe.stdout.splitlines() if line.strip()]
    else:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise ValueError(f"invalid GPU list: {text!r}")
    return values


@dataclass(frozen=True)
class Job:
    variant: str
    case_id: int
    run_mode: str

    @property
    def key(self) -> str:
        return f"{self.variant.lower()}_case{self.case_id:02d}_{self.run_mode}"


@dataclass
class JobResult:
    variant: str
    case_id: int
    run_mode: str
    gpu: int
    attempt: int
    status: str
    returncode: int
    elapsed_seconds: float
    output_dir: str
    log_path: str
    started_utc: str
    finished_utc: str


def case_output(root: Path, job: Job) -> Path:
    return root / job.variant.lower() / f"case_{job.case_id:02d}"


def archive_incomplete(path: Path) -> Path | None:
    if not path.exists():
        return None
    if (path / "DONE").is_file():
        return path
    index = 1
    while True:
        target = path.with_name(f"{path.name}.failed_attempt_{index:02d}")
        if not target.exists():
            path.rename(target)
            return target
        index += 1


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


def jobs_for(run_mode: str) -> list[Job]:
    cases = (1,) if run_mode == "smoke" else CASES
    # Case-major ordering starts one N0/N1/P0/P1 block together on four GPUs,
    # so every arm progresses overnight and paired records finish near each other.
    jobs = [Job(variant, case_id, run_mode) for case_id in cases for variant in VARIANTS]
    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("job matrix contains duplicate keys")
    return jobs


def run_job(
    *,
    job: Job,
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
            run_mode=job.run_mode,
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
    if log_path.exists():
        rerun_index = 1
        while True:
            candidate = log_path.with_name(
                f"{log_path.stem}_rerun{rerun_index:02d}{log_path.suffix}"
            )
            if not candidate.exists():
                log_path = candidate
                break
            rerun_index += 1
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
        "--eval-chunk-size",
        str(eval_chunk_size),
    ]
    if job.run_mode == "smoke":
        command.append("--smoke")
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
        run_mode=job.run_mode,
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


def execute_queue(
    *,
    jobs: Sequence[Job],
    gpus: Sequence[int],
    max_attempts: int,
    kwargs: dict[str, Any],
    tracker_path: Path,
    portable_roots: Sequence[tuple[Path, str]] = (),
) -> list[JobResult]:
    pending: queue.Queue[tuple[Job, int]] = queue.Queue()
    for job in jobs:
        pending.put((job, 1))
    results: list[JobResult] = []
    lock = threading.Lock()

    def persist() -> None:
        rows = [asdict(result) for result in results]
        review_rows = portable_review_value(rows, portable_roots)
        write_json(
            tracker_path.with_suffix(".json"),
            {
                "updated_utc": utc_now(),
                "planned_jobs": len(jobs),
                "attempt_records": review_rows,
                "terminal_complete_keys": sorted(
                    {
                        f"{row['variant']}_{int(row['case_id']):02d}"
                        for row in rows
                        if row["status"] in {"complete", "reused_complete"}
                    }
                ),
            },
        )
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(JobResult.__dataclass_fields__)
        with tracker_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_rows)

    def worker(gpu: int) -> None:
        while True:
            try:
                job, attempt = pending.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_job(job=job, gpu=gpu, attempt=attempt, **kwargs)
            except BaseException as exc:
                now = utc_now()
                emergency_dir = Path(kwargs["run_root"]) / "launcher_logs"
                emergency_dir.mkdir(parents=True, exist_ok=True)
                emergency_path = emergency_dir / (
                    f"{job.key}_attempt{attempt:02d}_gpu{gpu}_launcher_exception.log"
                )
                if emergency_path.exists():
                    emergency_path = emergency_path.with_name(
                        f"{emergency_path.stem}_{time.time_ns()}{emergency_path.suffix}"
                    )
                emergency_path.write_text(
                    f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
                result = JobResult(
                    variant=job.variant,
                    case_id=job.case_id,
                    run_mode=job.run_mode,
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
                persist()
                print(
                    f"[{result.finished_utc}] {job.variant} case{job.case_id:02d} "
                    f"attempt={attempt} gpu={gpu} status={result.status} "
                    f"elapsed={result.elapsed_seconds:.1f}s",
                    flush=True,
                )
            if result.status == "failed" and attempt < max_attempts:
                pending.put((job, attempt + 1))
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def accepted_keys(results: Sequence[JobResult]) -> set[tuple[str, int]]:
    return {
        (result.variant, int(result.case_id))
        for result in results
        if result.status in {"complete", "reused_complete"}
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dry-run", "smoke", "full", "all"), default="all")
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
        args.output_root, "INRORSPLINE_DUAL_CONTROLLED_RUN_ROOT"
    )
    review_root = resolve_root(
        args.review_root, "INRORSPLINE_DUAL_CONTROLLED_REVIEW_ROOT"
    )
    validate_roots(
        repo_root=repo_root,
        native_raw_root=native_raw_root,
        native_mask_root=native_mask_root,
        ptv_root=ptv_root,
        output_root=output_root,
    )
    gpus = parse_gpus(args.gpus)
    dry_manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "variants": list(VARIANTS),
        "cases": list(CASES),
        "seed": int(args.seed),
        "full_job_count": 40,
        "smoke_job_count": 4,
        "gpus": gpus,
        "external_paths": {
            "native_raw_root": str(native_raw_root),
            "native_mask_root": str(native_mask_root),
            "ptv_root": str(ptv_root),
            "output_root": str(output_root),
        },
        "scientific_stop_rule": "none; every independent job is attempted",
        "technical_retry_limit": int(args.max_attempts),
    }
    review_root.mkdir(parents=True, exist_ok=True)
    review_manifest = portable_review_value(
        dry_manifest,
        (
            (native_raw_root, "${NATIVE_RAW_ROOT}"),
            (native_mask_root, "${NATIVE_MASK_ROOT}"),
            (ptv_root, "${PTV_ROOT}"),
            (output_root, "${RUN_ROOT}"),
        ),
    )
    write_json(review_root / "resolved_job_manifest.json", review_manifest)
    print(json.dumps(dry_manifest, indent=2), flush=True)
    if args.stage == "dry-run":
        return

    common_kwargs = {
        "repo_root": repo_root,
        "native_raw_root": native_raw_root,
        "native_mask_root": native_mask_root,
        "ptv_root": ptv_root,
        "seed": int(args.seed),
        "eval_chunk_size": int(args.eval_chunk_size),
    }
    if args.stage in {"smoke", "all"}:
        smoke_root = output_root.with_name(output_root.name + "_smoke")
        smoke_results = execute_queue(
            jobs=jobs_for("smoke"),
            gpus=gpus,
            max_attempts=int(args.max_attempts),
            kwargs={**common_kwargs, "run_root": smoke_root},
            tracker_path=review_root / "smoke_tracker.csv",
            portable_roots=((smoke_root, "${RUN_ROOT}"),),
        )
        smoke_ok = accepted_keys(smoke_results)
        expected_smoke = {(variant, 1) for variant in VARIANTS}
        if smoke_ok != expected_smoke:
            raise RuntimeError(
                f"technical smoke gate failed: accepted {sorted(smoke_ok)}, "
                f"expected {sorted(expected_smoke)}"
            )
        if args.stage == "smoke":
            return

    if args.stage in {"full", "all"}:
        output_root.mkdir(parents=True, exist_ok=True)
        full_results = execute_queue(
            jobs=jobs_for("full"),
            gpus=gpus,
            max_attempts=int(args.max_attempts),
            kwargs={**common_kwargs, "run_root": output_root},
            tracker_path=review_root / "full_tracker.csv",
            portable_roots=((output_root, "${RUN_ROOT}"),),
        )
        completed = accepted_keys(full_results)
        expected = {(variant, case_id) for variant in VARIANTS for case_id in CASES}
        all_complete = completed == expected
        write_json(
            review_root / "overnight_completion.json",
            {
                "updated_utc": utc_now(),
                "all_40_complete": all_complete,
                "accepted_count": len(completed),
                "missing": sorted(f"{v}_case{c:02d}" for v, c in expected - completed),
                "attempt_count": len(full_results),
            },
        )
        if not args.skip_aggregate:
            aggregate_command = [
                sys.executable,
                "-m",
                AGGREGATE_MODULE,
                "--run-root",
                str(output_root),
                "--review-root",
                str(review_root),
            ]
            if not all_complete:
                aggregate_command.append("--allow-partial")
            subprocess.run(aggregate_command, cwd=repo_root, check=False)
        if not all_complete:
            raise SystemExit(
                f"all jobs were attempted, but {len(expected - completed)} remain technically failed"
            )


if __name__ == "__main__":
    main()
