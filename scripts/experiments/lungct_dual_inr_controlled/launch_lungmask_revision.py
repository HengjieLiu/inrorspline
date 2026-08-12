#!/usr/bin/env python3
"""Launch the pTV 1 mm fixed-lung-mask DUAL revision pilot.

This launcher is intentionally separate from ``launch_all.py`` so the completed
N0/N1/P0/P1 controlled cohort remains reproducible.  The default action is a
dry run; use ``--stage smoke`` for one-step technical checks and ``--stage
pilot`` for the full-length cases 01 and 08 pilot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
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
from scripts.experiments.lungct_dual_inr_controlled.adapter import (
    LUNGMASK_REVISION_VARIANTS,
)
from scripts.experiments.lungct_dual_inr_controlled.launch_all import (
    JobResult,
    archive_incomplete,
    parse_gpus,
    path_is_below,
    resolve_root,
)


PILOT_CASES = (1, 8)
FULL_CASES = tuple(range(1, 11))
RUNNER_MODULE = "scripts.experiments.lungct_dual_inr_controlled.run_one"
AGGREGATE_MODULE = (
    "scripts.experiments.lungct_dual_inr_controlled.aggregate_lungmask_revision"
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
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def parse_cases(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.replace(" ", "").split(",") if part)
    if not values or any(value not in FULL_CASES for value in values):
        raise argparse.ArgumentTypeError("cases must be comma-separated values in 1..10")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("cases must be unique")
    return values


def parse_variants(text: str) -> tuple[str, ...]:
    values = tuple(part.strip().upper() for part in text.split(",") if part.strip())
    if not values or any(value not in LUNGMASK_REVISION_VARIANTS for value in values):
        raise argparse.ArgumentTypeError(
            f"variants must be from {','.join(LUNGMASK_REVISION_VARIANTS)}"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("variants must be unique")
    return values


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


@dataclass(frozen=True)
class Job:
    variant: str
    case_id: int
    run_mode: str

    @property
    def key(self) -> str:
        return f"{self.variant.lower()}_case{self.case_id:02d}_{self.run_mode}"


def case_output(root: Path, job: Job) -> Path:
    return root / job.variant.lower() / f"case_{job.case_id:02d}"


def jobs_for(run_mode: str, *, variants: Sequence[str], cases: Sequence[int]) -> list[Job]:
    selected_cases = (int(cases[0]),) if run_mode == "smoke" else tuple(int(c) for c in cases)
    jobs = [Job(variant, case_id, run_mode) for case_id in selected_cases for variant in variants]
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
    portable_roots: Sequence[tuple[Path, str]],
) -> list[JobResult]:
    pending: queue.Queue[tuple[Job, int]] = queue.Queue()
    for job in jobs:
        pending.put((job, 1))
    results: list[JobResult] = []
    lock = threading.Lock()

    def persist() -> None:
        rows = [asdict(result) for result in results]
        review_rows = portable_review_value(rows, portable_roots)
        write_csv(tracker_path, review_rows)
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
    parser.add_argument("--stage", choices=("dry-run", "smoke", "pilot", "full", "all"), default="dry-run")
    parser.add_argument("--cases", type=parse_cases, default=PILOT_CASES)
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=LUNGMASK_REVISION_VARIANTS,
    )
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
        args.output_root, "INRORSPLINE_DUAL_LUNGMASK_REVISION_RUN_ROOT"
    )
    review_root = resolve_root(
        args.review_root, "INRORSPLINE_DUAL_LUNGMASK_REVISION_REVIEW_ROOT"
    )
    validate_roots(
        repo_root=repo_root,
        native_raw_root=native_raw_root,
        native_mask_root=native_mask_root,
        ptv_root=ptv_root,
        output_root=output_root,
    )
    selected_cases = FULL_CASES if args.stage == "full" else tuple(args.cases)
    gpus = parse_gpus(args.gpus)
    dry_manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": args.stage,
        "variants": list(args.variants),
        "cases": list(selected_cases),
        "pilot_cases": list(PILOT_CASES),
        "seed": int(args.seed),
        "planned_job_count": len(args.variants) * len(selected_cases),
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
    portable_roots = (
        (native_raw_root, "${NATIVE_RAW_ROOT}"),
        (native_mask_root, "${NATIVE_MASK_ROOT}"),
        (ptv_root, "${PTV_ROOT}"),
        (output_root, "${RUN_ROOT}"),
        (review_root, "${REVIEW_ROOT}"),
    )
    write_json(
        review_root / "resolved_job_manifest.json",
        portable_review_value(dry_manifest, portable_roots),
    )
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
        smoke_jobs = jobs_for("smoke", variants=args.variants, cases=selected_cases)
        smoke_results = execute_queue(
            jobs=smoke_jobs,
            gpus=gpus,
            max_attempts=int(args.max_attempts),
            kwargs={**common_kwargs, "run_root": smoke_root},
            tracker_path=review_root / "smoke_tracker.csv",
            portable_roots=((smoke_root, "${RUN_ROOT}"), (review_root, "${REVIEW_ROOT}")),
        )
        expected_smoke = {(variant, int(selected_cases[0])) for variant in args.variants}
        if accepted_keys(smoke_results) != expected_smoke:
            raise RuntimeError(
                f"technical smoke gate failed: accepted {sorted(accepted_keys(smoke_results))}, "
                f"expected {sorted(expected_smoke)}"
            )
        if args.stage == "smoke":
            return

    if args.stage in {"pilot", "full", "all"}:
        output_root.mkdir(parents=True, exist_ok=True)
        run_cases = selected_cases
        run_mode = "pilot" if args.stage in {"pilot", "all"} else "full"
        full_results = execute_queue(
            jobs=jobs_for(run_mode, variants=args.variants, cases=run_cases),
            gpus=gpus,
            max_attempts=int(args.max_attempts),
            kwargs={**common_kwargs, "run_root": output_root},
            tracker_path=review_root / f"{run_mode}_tracker.csv",
            portable_roots=((output_root, "${RUN_ROOT}"), (review_root, "${REVIEW_ROOT}")),
        )
        completed = accepted_keys(full_results)
        expected = {(variant, case_id) for variant in args.variants for case_id in run_cases}
        all_complete = completed == expected
        write_json(
            review_root / f"{run_mode}_completion.json",
            {
                "updated_utc": utc_now(),
                "all_complete": all_complete,
                "accepted_count": len(completed),
                "expected_count": len(expected),
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
                "--cases",
                ",".join(str(case) for case in run_cases),
                "--variants",
                ",".join(args.variants),
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
