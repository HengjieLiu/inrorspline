#!/usr/bin/env python
"""Second-phase case08 standard-pyramid probe for mechanism exploration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in [THIS_DIR, *THIS_DIR.parents] if (p / "docs").is_dir() and (p / "scripts").is_dir())
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402


EXPERIMENT_KEY = "2026-06-29_sinr_cp_kernel_followup"
RUN_KEY = "sinr_cp_mechanism_explore_20260629_v1"
RUN_ONE = REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro/run_one.py"
DEFAULT_PTVPREP_ROOT = Path(
    "/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm"
)


@dataclass(frozen=True)
class ProbeJob:
    job_index: int
    group: str
    variant: str
    case_id: int
    method: str
    run_name: str
    gpu: str
    stage_downsample: str
    stage_epochs: str
    lr: float
    alpha: Optional[float]
    ptv_weight: float
    command: List[str]
    run_dir: str
    log_path: str


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str) + "\n")


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fieldnames and not isinstance(value, (list, dict)):
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_gpus(text: str) -> List[str]:
    gpus = [token.strip() for token in str(text).split(",") if token.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required.")
    return gpus


def format_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m").replace("+", "")


def thread_env(cpu_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, int(cpu_threads)))
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ]:
        env[key] = threads
    return env


def log_round(path: Path, *, role: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %z")
    with path.open("a") as f:
        f.write(f"\n## {stamp} - {role}\n\n{content.strip()}\n")


def phase_root(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / str(args.phase_name)


def build_job(
    args: argparse.Namespace,
    *,
    job_index: int,
    group: str,
    variant: str,
    method: str,
    gpu: str,
    lr: float,
    stage_epochs: str,
    stage_downsample: str = "8,4,2,1",
    alpha: Optional[float] = None,
    ptv_weight: float = 0.11,
) -> ProbeJob:
    root = phase_root(args)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    epoch_slug = stage_epochs.replace(",", "-")
    downsample_slug = stage_downsample.replace(",", "-")
    suffix = f"case08_{variant}_ds{downsample_slug}_e{epoch_slug}_lr{format_float(lr)}"
    suffix += f"_a{format_float(alpha)}" if alpha is not None else f"_w{format_float(ptv_weight)}"
    run_name = f"stdprobe_{suffix}"
    run_dir = root / method / run_name
    cmd = [
        sys.executable,
        str(RUN_ONE),
        "--method",
        method,
        "--case-id",
        "8",
        "--data-root",
        str(args.ptvprep_root),
        "--out-root",
        str(root),
        "--run-name",
        run_name,
        "--gpu",
        gpu,
        "--seed",
        str(args.seed),
        "--epochs",
        "2500",
        "--lr",
        f"{lr:g}",
        "--eval-interval",
        "250",
        "--ptv-isotv-weight",
        f"{ptv_weight:g}",
        "--alpha",
        f"{float(alpha if alpha is not None else args.alpha):g}",
        "--stage-full-cps",
        "32,16,8,4",
        "--stage-downsample",
        stage_downsample,
        "--stage-epochs",
        stage_epochs,
        "--stage-lr",
        f"{lr:g}",
        "--sample-checkpoints",
        "--checkpoint-sample-count",
        str(args.checkpoint_sample_count),
        "--checkpoint-sample-seed",
        str(args.checkpoint_sample_seed),
        "--checkpoint-epochs",
        "0,1,10,25,50,100,250,500,1000,1500,2000,2400",
    ]
    if args.overwrite_jobs:
        cmd.append("--overwrite")
    return ProbeJob(
        job_index=job_index,
        group=group,
        variant=variant,
        case_id=8,
        method=method,
        run_name=run_name,
        gpu=gpu,
        stage_downsample=stage_downsample,
        stage_epochs=stage_epochs,
        lr=lr,
        alpha=alpha,
        ptv_weight=ptv_weight,
        command=cmd,
        run_dir=str(run_dir),
        log_path=str(log_dir / f"{run_name}_gpu{gpu}.log"),
    )


def standard_specs() -> List[Dict[str, object]]:
    return [
        {
            "group": "standard_pyramid_short_ptv",
            "variant": "ptv_standard_short",
            "method": "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "lr": 1e-3,
            "stage_epochs": "25,25,25,25",
            "stage_downsample": "8,4,2,1",
        },
        {
            "group": "standard_pyramid_short_fdbe",
            "variant": "fdbe_standard_short",
            "method": "direct_cp_32_16_8_4_global_ncc_fd_be",
            "lr": 1e-3,
            "stage_epochs": "25,25,25,25",
            "stage_downsample": "8,4,2,1",
            "alpha": 10000.0,
        },
        {
            "group": "standard_pyramid_lr_probe_ptv",
            "variant": "ptv_standard_short_lr0p01",
            "method": "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "lr": 1e-2,
            "stage_epochs": "25,25,25,25",
            "stage_downsample": "8,4,2,1",
        },
        {
            "group": "standard_pyramid_longer_ptv",
            "variant": "ptv_standard_e50",
            "method": "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "lr": 1e-3,
            "stage_epochs": "50,50,50,50",
            "stage_downsample": "8,4,2,1",
        },
    ]


def refinement_specs() -> List[Dict[str, object]]:
    return [
        {
            "group": "standard_pyramid_runtime_probe_ptv",
            "variant": "ptv_standard_lr0p01_e100",
            "method": "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "lr": 1e-2,
            "stage_epochs": "100,100,100,100",
            "stage_downsample": "8,4,2,1",
        },
        {
            "group": "standard_pyramid_runtime_probe_ptv",
            "variant": "ptv_standard_lr0p01_e200",
            "method": "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "lr": 1e-2,
            "stage_epochs": "200,200,200,200",
            "stage_downsample": "8,4,2,1",
        },
        {
            "group": "partial_pyramid_lr_runtime_probe_ptv",
            "variant": "ptv_partial_lr0p01_e100",
            "method": "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "lr": 1e-2,
            "stage_epochs": "100,100,100,100",
            "stage_downsample": "4,2,1,1",
        },
        {
            "group": "no_pyramid_lr_runtime_probe_ptv",
            "variant": "ptv_nopyr_lr0p01_e100",
            "method": "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "lr": 1e-2,
            "stage_epochs": "100,100,100,100",
            "stage_downsample": "1,1,1,1",
        },
    ]


def build_jobs(args: argparse.Namespace) -> List[ProbeJob]:
    gpus = parse_gpus(args.gpus)
    specs = refinement_specs() if args.probe_set == "refinement" else standard_specs()
    jobs: List[ProbeJob] = []
    for idx, spec in enumerate(specs):
        jobs.append(build_job(args, job_index=idx, gpu=gpus[idx % len(gpus)], **spec))
    return jobs


def write_manifest(args: argparse.Namespace, jobs: Sequence[ProbeJob]) -> None:
    root = phase_root(args)
    rows = [asdict(job) for job in jobs]
    write_csv(root / "job_manifest.csv", rows)
    write_json(
        root / "job_manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "purpose": (
                "case08 LR/runtime refinement after standard pyramid recovery"
                if args.probe_set == "refinement"
                else "case08 standard 8,4,2,1 pyramid probe after no/partial-pyramid failure"
            ),
            "probe_set": args.probe_set,
            "job_count": len(jobs),
            "gpus": args.gpus,
            "out_root": str(root),
            "review_root": str(args.review_root),
            "jobs": rows,
        },
    )


def launch_jobs(args: argparse.Namespace, jobs: Sequence[ProbeJob]) -> Dict[str, object]:
    if args.dry_run:
        for job in jobs:
            print(json.dumps({"job": asdict(job), "command": job.command}), flush=True)
        return {"status": "dry_run", "job_count": len(jobs)}
    pending = list(jobs)
    running: List[Dict[str, object]] = []
    completed: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    env = thread_env(args.cpu_threads)
    max_concurrent = min(len(parse_gpus(args.gpus)), int(args.max_concurrent), len(pending))
    status_path = phase_root(args) / "phase_status.json"
    started = time.time()
    while pending or running:
        while pending and len(running) < max_concurrent:
            job = pending.pop(0)
            log_path = Path(job.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w")
            print(json.dumps({"launching": job.run_name, "gpu": job.gpu, "log": str(log_path)}), flush=True)
            proc = subprocess.Popen(job.command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
            running.append({"job": job, "proc": proc, "log_handle": log_handle, "started": time.time()})
            time.sleep(float(args.launch_stagger_sec))
        still_running: List[Dict[str, object]] = []
        for item in running:
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            job = item["job"]
            assert isinstance(job, ProbeJob)
            code = proc.poll()
            if code is None:
                still_running.append(item)
                continue
            item["log_handle"].close()
            row = {
                "job": job.run_name,
                "group": job.group,
                "variant": job.variant,
                "case_id": job.case_id,
                "method": job.method,
                "returncode": code,
                "elapsed_sec": time.time() - float(item["started"]),
            }
            if code == 0:
                completed.append(row)
            else:
                failures.append(row)
            print(json.dumps({"finished": row}), flush=True)
        running = still_running
        write_json(
            status_path,
            {
                "status": "running" if pending or running else "finished",
                "pending": len(pending),
                "running": len(running),
                "completed": len(completed),
                "failures": len(failures),
                "elapsed_sec": time.time() - started,
                "running_jobs": [
                    {
                        "job": item["job"].run_name,
                        "gpu": item["job"].gpu,
                        "started_sec_ago": time.time() - float(item["started"]),
                        "log": item["job"].log_path,
                    }
                    for item in running
                ],
                "completed_jobs": completed,
                "failed_jobs": failures,
            },
        )
        if failures and not args.allow_partial:
            for item in running:
                proc = item["proc"]
                assert isinstance(proc, subprocess.Popen)
                proc.terminate()
                item["log_handle"].close()
            raise RuntimeError(f"Stopping after failures: {failures}")
        if pending or running:
            time.sleep(float(args.poll_sec))
    return {
        "status": "completed" if not failures else "partial_failed",
        "completed": completed,
        "failures": failures,
        "elapsed_sec": time.time() - started,
    }


def collect_rows(args: argparse.Namespace, jobs: Sequence[ProbeJob]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for job in jobs:
        summary_path = Path(job.run_dir) / "summary.json"
        if not summary_path.exists():
            rows.append({**asdict(job), "status": "missing_summary", "summary_path": str(summary_path)})
            continue
        summary = read_json(summary_path)
        if not isinstance(summary, dict):
            continue
        training = summary.get("training", {})
        if not isinstance(training, dict):
            training = {}
        snap = summary.get("tre_snap_to_voxel", {})
        sub = summary.get("tre_subvoxel", {})
        rows.append(
            {
                "group": job.group,
                "variant": job.variant,
                "case_id": job.case_id,
                "method": job.method,
                "run_name": job.run_name,
                "status": summary.get("status"),
                "stage_downsample": ",".join(str(v) for v in training.get("stage_downsample", [])),
                "stage_epochs": ",".join(str(v) for v in training.get("stage_epochs", [])),
                "stage_lr": ",".join(str(v) for v in training.get("stage_lr", [])),
                "loss": training.get("loss"),
                "alpha": training.get("alpha"),
                "ptv_isotv_weight": training.get("ptv_isotv_weight"),
                "reg1mm_snap_tre_mean_mm": snap.get("tre_mean_mm") if isinstance(snap, dict) else float("nan"),
                "reg1mm_subvoxel_tre_mean_mm": sub.get("tre_mean_mm") if isinstance(sub, dict) else float("nan"),
                "elapsed_sec": summary.get("elapsed_sec"),
                "peak_mem_gb": summary.get("peak_mem_gb"),
                "summary_path": str(summary_path),
            }
        )
    return rows


def write_report(args: argparse.Namespace, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    review = Path(args.review_root) / "exploration"
    tables = review / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    stem = str(args.phase_name)
    write_csv(tables / f"{stem}_rows.csv", rows)
    completed = [row for row in rows if row.get("status") == "completed"]
    best = None
    for row in completed:
        try:
            tre = float(row.get("reg1mm_snap_tre_mean_mm", float("nan")))
        except Exception:
            continue
        if tre == tre and (best is None or tre < float(best["reg1mm_snap_tre_mean_mm"])):
            best = {**row, "reg1mm_snap_tre_mean_mm": tre}
    lines = [
        f"# {stem.replace('_', ' ').title()}",
        "",
        (
            "Purpose: after standard `8,4,2,1` LR `0.01` substantially recovered case08, test whether longer runtime and partial/no-pyramid LR controls close the remaining gap."
            if args.probe_set == "refinement"
            else "Purpose: after no-pyramid and partial `4,2,1,1` probes failed to rescue case08, test whether the full standard `8,4,2,1` image pyramid, LR, or a modestly longer schedule changes the conclusion."
        ),
        "",
    ]
    if best:
        lines.append(
            f"Best completed standard-pyramid probe: `{best.get('variant')}` with reg1mm snap TRE `{float(best['reg1mm_snap_tre_mean_mm']):.3f}` mm."
        )
    else:
        lines.append("No completed standard-pyramid probe rows yet.")
    lines.extend(["", "Rows:", ""])
    for row in rows:
        metric = row.get("reg1mm_snap_tre_mean_mm", "")
        lines.append(f"- `{row.get('variant')}`: `{row.get('status')}`, reg1mm snap TRE `{metric}`")
    report_path = review / f"{stem}_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    summary = {
        "status": "completed",
        "rows": len(rows),
        "completed_rows": len(completed),
        "best_completed": best,
        "rows_csv": str(tables / f"{stem}_rows.csv"),
        "report": str(report_path),
    }
    write_json(review / f"{stem}_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--ptvprep-root", type=Path, default=DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--phase-name", type=str, default="standard_pyramid_probe")
    parser.add_argument("--probe-set", choices=["standard", "refinement"], default="standard")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=10000.0)
    parser.add_argument("--checkpoint-sample-count", type=int, default=100000)
    parser.add_argument("--checkpoint-sample-seed", type=int, default=1729)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite-jobs", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve generated P6 run and review roots outside the Git worktree."""

    args.out_root = artifact_paths.resolve_run_root(
        RUN_KEY,
        args.out_root,
        repo_root=REPO_ROOT,
    )
    args.review_root = artifact_paths.resolve_review_root(
        EXPERIMENT_KEY,
        args.review_root,
        repo_root=REPO_ROOT,
    )
    return args


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    jobs = build_jobs(args)
    write_manifest(args, jobs)
    log_path = Path(args.review_root) / "scientist_reviewer_log.md"
    if args.aggregate_only:
        summary = write_report(args, collect_rows(args, jobs))
        print(json.dumps({"aggregation": summary}, indent=2, default=str), flush=True)
        return
    if args.probe_set == "refinement":
        reviewer_content = "Refinement instruction: launch case08 LR/runtime controls after standard-pyramid recovery."
        scientist_label = "Capture-refinement probe"
    else:
        reviewer_content = (
            "Second-phase instruction: because no-pyramid and partial `4,2,1,1` did not rescue case08, "
            "launch case08 standard `8,4,2,1` probes to separate image-pyramid capture range, LR, objective, and modest schedule length."
        )
        scientist_label = "Standard-pyramid probe"
    log_round(
        log_path,
        role="Reviewer",
        content=reviewer_content,
    )
    launch_result = launch_jobs(args, jobs)
    write_json(phase_root(args) / "launch_result.json", launch_result)
    print(json.dumps({"launch_result": launch_result}, indent=2, allow_nan=True), flush=True)
    log_round(log_path, role="Scientist", content=f"{scientist_label} launch result: `{launch_result}`.")
    summary = write_report(args, collect_rows(args, jobs))
    log_round(log_path, role="Scientist", content=f"{scientist_label} aggregation refreshed: `{summary}`.")
    print(json.dumps({"aggregation": summary}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
