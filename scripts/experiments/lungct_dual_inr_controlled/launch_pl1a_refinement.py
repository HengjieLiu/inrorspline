#!/usr/bin/env python3
"""Launch PL1A refinement multipliers around the Stage 2 best point."""

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
    archive_incomplete,
    parse_gpus,
    path_is_below,
    resolve_root,
)


VARIANT = "PL1A"
CASES = tuple(range(1, 11))
REFINEMENT_MULTIPLIERS = (1.0 / 3.0, 3.0)
RUNNER_MODULE = "scripts.experiments.lungct_dual_inr_controlled.run_one"
AGGREGATE_MODULE = "scripts.experiments.lungct_dual_inr_controlled.aggregate_pl1a_refinement"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_token(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p").replace("+", "")


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


@dataclass(frozen=True)
class Job:
    case_id: int
    multiplier: float

    @property
    def weight_label(self) -> str:
        return f"m{format_token(self.multiplier)}"

    @property
    def run_tag(self) -> str:
        return f"pl1a_{self.weight_label}"

    @property
    def key(self) -> str:
        return f"{self.run_tag}_case{self.case_id:02d}"


@dataclass
class Result:
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


def jobs() -> list[Job]:
    return [
        Job(case_id=case_id, multiplier=float(multiplier))
        for multiplier in REFINEMENT_MULTIPLIERS
        for case_id in CASES
    ]


def case_output(root: Path, job: Job) -> Path:
    return root / job.run_tag / VARIANT.lower() / f"case_{job.case_id:02d}"


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


def run_job(
    *,
    job: Job,
    gpu: int,
    attempt: int,
    repo_root: Path,
    native_raw_root: Path,
    native_mask_root: Path,
    ptv_root: Path,
    output_root: Path,
    seed: int,
    eval_chunk_size: int,
) -> Result:
    output_dir = case_output(output_root, job)
    if (output_dir / "DONE").is_file():
        now = utc_now()
        return Result(
            variant=VARIANT,
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
    archive_incomplete(output_dir)
    log_dir = output_root / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.key}_attempt{attempt:02d}_gpu{gpu}.log"
    command = [
        sys.executable,
        "-m",
        RUNNER_MODULE,
        "--variant",
        VARIANT,
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
        str(output_root),
        "--run-tag",
        job.run_tag,
        "--regularization-multiplier",
        str(job.multiplier),
        "--eval-chunk-size",
        str(eval_chunk_size),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    started = utc_now()
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"started_utc={started}\n")
        stream.write(f"gpu={gpu}\n")
        stream.write(f"command={' '.join(command)}\n\n")
        stream.flush()
        proc = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - start
    status = "complete" if proc.returncode == 0 and (output_dir / "DONE").is_file() else "failed"
    return Result(
        variant=VARIANT,
        case_id=job.case_id,
        run_mode=job.weight_label,
        gpu=gpu,
        attempt=attempt,
        status=status,
        returncode=int(proc.returncode),
        elapsed_seconds=float(elapsed),
        output_dir=str(output_dir),
        log_path=str(log_path),
        started_utc=started,
        finished_utc=utc_now(),
    )


def launch(args: argparse.Namespace) -> None:
    repo_root = resolve_root(args.repo_root, "INRORSPLINE_REPO_ROOT")
    native_raw_root = resolve_root(
        args.native_raw_root,
        "INRORSPLINE_DIRLAB_NATIVE_RAW_ROOT",
    )
    native_mask_root = resolve_root(
        args.native_mask_root,
        "INRORSPLINE_DIRLAB_NATIVE_MASK_ROOT",
    )
    ptv_root = resolve_root(args.ptv_root, "INRORSPLINE_DIRLAB_PTVREG_ISO1MM_ROOT")
    output_root = resolve_root(
        args.output_root,
        "INRORSPLINE_STAGE3_PL1A_REFINEMENT_ROOT",
    )
    review_root = resolve_root(
        args.review_root,
        "INRORSPLINE_STAGE3_PL1A_REFINEMENT_REVIEW_ROOT",
    )
    validate_roots(
        repo_root=repo_root,
        native_raw_root=native_raw_root,
        native_mask_root=native_mask_root,
        ptv_root=ptv_root,
        output_root=output_root,
    )
    selected_jobs = jobs()
    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "variant": VARIANT,
        "multipliers": list(REFINEMENT_MULTIPLIERS),
        "cases": list(CASES),
        "job_count": len(selected_jobs),
        "output_root": str(output_root),
        "review_root": str(review_root),
        "jobs": [asdict(job) | {"run_tag": job.run_tag, "weight_label": job.weight_label} for job in selected_jobs],
    }
    review_root.mkdir(parents=True, exist_ok=True)
    write_json(
        review_root / "stage3_pl1a_refinement_resolved_job_manifest.json",
        portable_review_value(
            manifest,
            (
                (output_root, "${STAGE3_RUN_ROOT}"),
                (review_root, "${REVIEW_ROOT}"),
                (repo_root, "${REPO_ROOT}"),
            ),
        ),
    )
    if args.stage == "dry-run":
        print(f"Dry-run manifest wrote {len(selected_jobs)} jobs to {review_root}", flush=True)
        return

    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise ValueError("at least one GPU is required")
    work: queue.Queue[Job] = queue.Queue()
    for job in selected_jobs:
        work.put(job)
    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()
    tracker_csv = review_root / "stage3_pl1a_refinement_tracker.csv"
    tracker_json = review_root / "stage3_pl1a_refinement_tracker.json"

    def persist() -> None:
        portable_rows = portable_review_value(
            rows,
            (
                (output_root, "${STAGE3_RUN_ROOT}"),
                (review_root, "${REVIEW_ROOT}"),
                (repo_root, "${REPO_ROOT}"),
            ),
        )
        write_csv(tracker_csv, portable_rows)
        write_json(tracker_json, {"schema_version": 1, "rows": portable_rows})

    def worker(gpu: int) -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            result: Result | None = None
            for attempt in range(1, int(args.max_attempts) + 1):
                result = run_job(
                    job=job,
                    gpu=gpu,
                    attempt=attempt,
                    repo_root=repo_root,
                    native_raw_root=native_raw_root,
                    native_mask_root=native_mask_root,
                    ptv_root=ptv_root,
                    output_root=output_root,
                    seed=int(args.seed),
                    eval_chunk_size=int(args.eval_chunk_size),
                )
                print(
                    f"[{result.finished_utc}] {result.variant} {result.run_mode} "
                    f"case{result.case_id:02d} attempt={attempt} gpu={gpu} "
                    f"status={result.status} elapsed={result.elapsed_seconds:.1f}s",
                    flush=True,
                )
                if result.status in {"complete", "reused_complete"}:
                    break
            assert result is not None
            with rows_lock:
                rows.append(asdict(result))
                persist()
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    missing = [
        {
            "run_tag": job.run_tag,
            "case_id": job.case_id,
            "output_dir": str(case_output(output_root, job)),
        }
        for job in selected_jobs
        if not (case_output(output_root, job) / "DONE").is_file()
    ]
    completion = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "expected_count": len(selected_jobs),
        "accepted_count": len(selected_jobs) - len(missing),
        "attempt_count": len(rows),
        "all_complete": not missing,
        "missing": missing,
        "gpus": list(gpus),
    }
    write_json(
        review_root / "stage3_pl1a_refinement_completion.json",
        portable_review_value(
            completion,
            (
                (output_root, "${STAGE3_RUN_ROOT}"),
                (review_root, "${REVIEW_ROOT}"),
            ),
        ),
    )
    if missing:
        raise RuntimeError(f"Stage 3 PL1A refinement incomplete: {missing}")
    aggregate_cmd = [
        sys.executable,
        "-m",
        AGGREGATE_MODULE,
        "--stage3-run-root",
        str(output_root),
        "--review-root",
        str(review_root),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(aggregate_cmd, cwd=repo_root, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dry-run", "full"), default="dry-run")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--native-raw-root", type=Path, required=True)
    parser.add_argument("--native-mask-root", type=Path, required=True)
    parser.add_argument("--ptv-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-chunk-size", type=int, default=262144)
    return parser


def main() -> None:
    launch(build_parser().parse_args())


if __name__ == "__main__":
    main()
