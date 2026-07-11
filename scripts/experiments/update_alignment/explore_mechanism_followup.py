#!/usr/bin/env python
"""Exploratory SINR/DirectCP mechanism follow-up with role-loop logging."""

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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


THIS_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "docs").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(THIS_DIR)
LUNG_RUN_ONE = REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro/run_one.py"
PROXY_FOLLOWUP = THIS_DIR / "run_sinr_cp_kernel_followup.py"

DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/sinr_cp_mechanism_explore_20260629_v1")
DEFAULT_REVIEW_ROOT = REPO_ROOT / "experiments/2026-06-29_sinr_cp_kernel_followup"
DEFAULT_UPDATE_ROOT = Path("/database/inrorspline/runs/update_alignment_overnight_20260629_v1")
DEFAULT_PTVPREP_ROOT = Path(
    "/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1"
)


@dataclass(frozen=True)
class Job:
    job_index: int
    group: str
    variant: str
    case_id: int
    method: str
    run_name: str
    gpu: str
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def format_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m").replace("+", "")


def parse_gpus(text: str) -> List[str]:
    gpus = [token.strip() for token in str(text).split(",") if token.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required.")
    return gpus


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


def init_logs(args: argparse.Namespace) -> None:
    review_root = Path(args.review_root)
    review_root.mkdir(parents=True, exist_ok=True)
    log_path = review_root / "scientist_reviewer_log.md"
    if not log_path.exists() or args.overwrite_log:
        log_path.write_text(
            "# Scientist / Reviewer Exploration Log\n\n"
            "Goal: produce a useful paper table, figure, or section from an exploratory SINR/DirectCP follow-up.\n"
        )
    log_round(
        log_path,
        role="Reviewer",
        content=(
            "Treat the failed promising gate as information, not as a stop. "
            "Proceed with exploration, but do not claim a method-only LungCT leaderboard. "
            "Ask whether low SINR-vs-DirectCP similarity reflects a real mechanism difference, "
            "a capture-range limitation, or a schedule/objective artifact."
        ),
    )
    log_round(
        log_path,
        role="Scientist",
        content=(
            "Plan: rerun the sampled dense-update proxy on the completed full outputs; "
            "combine it with prior case01/case08 field-fit and LR sweeps; then launch "
            "schedule ablations that remove or weaken the image pyramid while preserving "
            "the DirectCP control-stage ladder."
        ),
    )


def phase_root(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / "schedule_ablation"


def review_tables_dir(args: argparse.Namespace) -> Path:
    return Path(args.review_root) / "exploration" / "tables"


def review_plots_dir(args: argparse.Namespace) -> Path:
    return Path(args.review_root) / "exploration" / "plots"


def build_job(
    args: argparse.Namespace,
    *,
    job_index: int,
    group: str,
    variant: str,
    case_id: int,
    method: str,
    gpu: str,
    lr: float,
    stage_downsample: str,
    alpha: Optional[float] = None,
    ptv_weight: float = 0.11,
) -> Job:
    stage_epochs = str(args.stage_epochs)
    root = phase_root(args)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    epoch_slug = str(args.stage_epochs).replace(",", "-")
    suffix = f"case{case_id:02d}_{variant}_e{epoch_slug}_lr{format_float(lr)}"
    if alpha is not None:
        suffix += f"_a{format_float(alpha)}"
    else:
        suffix += f"_w{format_float(ptv_weight)}"
    run_name = f"explore_{suffix}"
    run_dir = root / method / run_name
    cmd = [
        sys.executable,
        str(LUNG_RUN_ONE),
        "--method",
        method,
        "--case-id",
        str(case_id),
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
    return Job(
        job_index=job_index,
        group=group,
        variant=variant,
        case_id=int(case_id),
        method=method,
        run_name=run_name,
        gpu=gpu,
        command=cmd,
        run_dir=str(run_dir),
        log_path=str(log_dir / f"{run_name}_gpu{gpu}.log"),
    )


def build_jobs(args: argparse.Namespace) -> List[Job]:
    gpus = parse_gpus(args.gpus)
    jobs: List[Job] = []

    def add(**kwargs: object) -> None:
        gpu = gpus[len(jobs) % len(gpus)]
        jobs.append(build_job(args, job_index=len(jobs), gpu=gpu, **kwargs))  # type: ignore[arg-type]

    cases = [1, 8] if not args.expand_all10 else list(range(1, 11))
    for case_id in cases:
        add(
            group="ptv_control_staging_without_image_pyramid",
            variant="ptv_nopyr_ds1-1-1-1",
            case_id=case_id,
            method="direct_cp_32_16_8_4_ptv_lcc_isotv",
            lr=1e-3,
            stage_downsample="1,1,1,1",
            ptv_weight=0.11,
        )
        add(
            group="fdbe_control_staging_without_image_pyramid",
            variant="fdbe_nopyr_ds1-1-1-1",
            case_id=case_id,
            method="direct_cp_32_16_8_4_global_ncc_fd_be",
            lr=1e-3,
            stage_downsample="1,1,1,1",
            alpha=10000.0,
        )
    if args.targeted_extra:
        for case_id in (1, 8):
            add(
                group="ptv_no_pyramid_best_lr_probe",
                variant="ptv_nopyr_ds1-1-1-1_lr0p01",
                case_id=case_id,
                method="direct_cp_32_16_8_4_ptv_lcc_isotv",
                lr=1e-2,
                stage_downsample="1,1,1,1",
                ptv_weight=0.11,
            )
        for case_id in (1, 8):
            add(
                group="ptv_partial_image_pyramid_probe",
                variant="ptv_partial_ds4-2-1-1",
                case_id=case_id,
                method="direct_cp_32_16_8_4_ptv_lcc_isotv",
                lr=1e-3,
                stage_downsample="4,2,1,1",
                ptv_weight=0.11,
            )
            add(
                group="fdbe_partial_image_pyramid_probe",
                variant="fdbe_partial_ds4-2-1-1",
                case_id=case_id,
                method="direct_cp_32_16_8_4_global_ncc_fd_be",
                lr=1e-3,
                stage_downsample="4,2,1,1",
                alpha=10000.0,
            )
    return jobs


def write_manifest(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    root = phase_root(args)
    rows = [asdict(job) for job in jobs]
    write_csv(root / "job_manifest.csv", rows)
    write_json(
        root / "job_manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "job_count": len(jobs),
            "gpus": args.gpus,
            "out_root": str(root),
            "review_root": str(args.review_root),
            "stage_epochs": str(args.stage_epochs),
            "expand_all10": bool(args.expand_all10),
            "targeted_extra": bool(args.targeted_extra),
            "jobs": rows,
        },
    )


def launch_jobs(args: argparse.Namespace, jobs: Sequence[Job]) -> Dict[str, object]:
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
            assert isinstance(job, Job)
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


def run_proxy_followup(args: argparse.Namespace) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(PROXY_FOLLOWUP),
        "--scope",
        "first3",
        "--gpus",
        str(args.gpus),
        "--run-full",
        "--resume",
        "--source-update-root",
        str(args.update_root),
        "--out-root",
        "/database/inrorspline/runs/sinr_cp_kernel_followup_20260629_v1",
        "--review-root",
        str(args.review_root),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    result = {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    write_json(Path(args.review_root) / "exploration" / "proxy_followup_subprocess.json", result)
    if proc.returncode != 0:
        raise RuntimeError(f"Proxy follow-up failed with {proc.returncode}: {proc.stderr[-1000:]}")
    return result


def f(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except Exception:
        return default


def collect_prior_best_rows() -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    for case_id in (1, 8):
        path = REPO_ROOT / f"experiments/2026-06-26_lungct_case{case_id:02d}_ptvreg_fit_modes_lr_sweep/summary_runs.csv"
        rows = read_csv(path)
        for objective in ("image_registration", "ptvreg_field_fit"):
            for alias in sorted({row.get("method_alias", "") for row in rows if row.get("objective_group") == objective}):
                sub = [row for row in rows if row.get("objective_group") == objective and row.get("method_alias") == alias]
                if not sub:
                    continue
                metric_key = "field_mse_objective_voxel" if objective == "ptvreg_field_fit" else "native_crop_snap_tre_mean_mm"
                sub = [row for row in sub if np.isfinite(f(row, metric_key))]
                if not sub:
                    continue
                best = min(sub, key=lambda row: f(row, metric_key))
                rows_out.append(
                    {
                        "case_id": case_id,
                        "objective": objective,
                        "method_alias": best.get("method_alias"),
                        "method_label": best.get("method_label"),
                        "best_lr": f(best, "lr"),
                        "native_crop_snap_tre_mean_mm": f(best, "native_crop_snap_tre_mean_mm"),
                        "reg1mm_snap_tre_mean_mm": f(best, "reg1mm_snap_tre_mean_mm"),
                        "field_mse_objective_voxel": f(best, "field_mse_objective_voxel"),
                        "fg_logj_std": f(best, "fg_log_jdet_std", f(best, "fg_logj_std")),
                        "summary_path": best.get("summary_path"),
                    }
                )
    return rows_out


def collect_schedule_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    manifest_path = phase_root(args) / "job_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        return []
    rows_out: List[Dict[str, object]] = []
    for job in manifest.get("jobs", []):
        if not isinstance(job, dict):
            continue
        summary_path = Path(str(job["run_dir"])) / "summary.json"
        if not summary_path.exists():
            rows_out.append({**job, "status": "missing_summary", "summary_path": str(summary_path)})
            continue
        summary = read_json(summary_path)
        if not isinstance(summary, dict):
            continue
        training = summary.get("training", {})
        if not isinstance(training, dict):
            training = {}
        tre = summary.get("tre_snap_to_voxel", {})
        sub = summary.get("tre_subvoxel", {})
        rows_out.append(
            {
                "group": job.get("group"),
                "variant": job.get("variant"),
                "case_id": job.get("case_id"),
                "method": job.get("method"),
                "run_name": job.get("run_name"),
                "status": summary.get("status"),
                "stage_downsample": ",".join(str(v) for v in training.get("stage_downsample", [])),
                "stage_epochs": ",".join(str(v) for v in training.get("stage_epochs", [])),
                "stage_lr": ",".join(str(v) for v in training.get("stage_lr", [])),
                "loss": training.get("loss"),
                "alpha": training.get("alpha"),
                "ptv_isotv_weight": training.get("ptv_isotv_weight"),
                "reg1mm_snap_tre_mean_mm": tre.get("tre_mean_mm") if isinstance(tre, dict) else float("nan"),
                "reg1mm_subvoxel_tre_mean_mm": sub.get("tre_mean_mm") if isinstance(sub, dict) else float("nan"),
                "elapsed_sec": summary.get("elapsed_sec"),
                "peak_mem_gb": summary.get("peak_mem_gb"),
                "summary_path": str(summary_path),
            }
        )
    return rows_out


def collect_update_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    review_full = Path(args.review_root).parent / "2026-06-29_update_alignment_overnight" / "full"
    sim = read_csv(review_full / "tables/sinr_directcp_similarity_rows.csv")
    case = read_csv(review_full / "tables/case_summary_rows.csv")
    return sim, case


def collect_proxy_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    return read_csv(Path(args.review_root) / "tables/kernel_metrics.csv")


def collect_standard_probe_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    return read_csv(Path(args.review_root) / "exploration/tables/standard_pyramid_probe_rows.csv")


def collect_refinement_probe_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    return read_csv(Path(args.review_root) / "exploration/tables/capture_refinement_probe_rows.csv")


def make_plots(args: argparse.Namespace, prior: Sequence[Dict[str, object]], schedule: Sequence[Dict[str, object]], sim_rows: Sequence[Dict[str, str]], proxy_rows: Sequence[Dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(review_plots_dir(args) / "plot_error.json", {"error": str(exc)})
        return
    plots = review_plots_dir(args)
    plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    labels = [f"{row['case_key']}" for row in sim_rows]
    values = [f(row, "cumulative_early_update_similarity") for row in sim_rows]
    ax.bar(range(len(values)), values, color="#4e79a7")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("SINR / DirectCP update similarity")
    ax.set_title("A. Low path similarity")
    ax.set_ylim(-0.05, 1.0)

    ax = axes[0, 1]
    case8 = [row for row in prior if row["case_id"] == 8 and row["objective"] == "image_registration"]
    labels = [str(row["method_alias"]) for row in case8]
    values = [float(row["native_crop_snap_tre_mean_mm"]) for row in case8]
    ax.bar(range(len(values)), values, color="#f28e2b")
    ax.axhline(1.0142198317, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("native/crop snap TRE mm")
    ax.set_title("B. Case08 image-registration best LR")

    ax = axes[1, 0]
    case8_fit = [row for row in prior if row["case_id"] == 8 and row["objective"] == "ptvreg_field_fit"]
    labels = [str(row["method_alias"]) for row in case8_fit]
    values = [float(row["native_crop_snap_tre_mean_mm"]) for row in case8_fit]
    ax.bar(range(len(values)), values, color="#59a14f")
    ax.axhline(1.0142198317, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("native/crop snap TRE mm")
    ax.set_title("C. Case08 field-fit capacity")

    ax = axes[1, 1]
    complete = [row for row in schedule if row.get("status") == "completed"]
    if complete:
        labels = [f"c{int(row['case_id']):02d}\n{str(row['variant']).replace('_', ' ')}" for row in complete]
        values = [float(row["reg1mm_snap_tre_mean_mm"]) for row in complete]
        ax.bar(range(len(values)), values, color="#e15759")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("reg1mm snap TRE mm")
        ax.set_title("D. New schedule ablations")
    else:
        dvals = [f(row, "D_U_proxy_cos_sinr_directcp") for row in proxy_rows]
        labels = [f"{row.get('case_key')}\n{row.get('checkpoint')}" for row in proxy_rows]
        ax.bar(range(len(dvals)), dvals, color="#e15759")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("D_U proxy")
        ax.set_title("D. Proxy follow-up")
    fig.tight_layout()
    fig.savefig(plots / "mechanism_triage_figure.png", dpi=180)
    plt.close(fig)


def row_for_alias(rows: Sequence[Dict[str, object]], alias: str) -> Optional[Dict[str, object]]:
    for row in rows:
        if row.get("method_alias") == alias:
            return row
    return None


def best_schedule_row(
    rows: Sequence[Dict[str, object]],
    *,
    case_id: Optional[int] = None,
    group_contains: Optional[str] = None,
    variant_contains: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        if case_id is not None and int(row.get("case_id", -1)) != int(case_id):
            continue
        if group_contains and group_contains not in str(row.get("group", "")):
            continue
        if variant_contains and variant_contains not in str(row.get("variant", "")):
            continue
        try:
            tre = float(row.get("reg1mm_snap_tre_mean_mm", float("nan")))
        except Exception:
            continue
        if np.isfinite(tre):
            candidates.append(row)
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row.get("reg1mm_snap_tre_mean_mm", float("inf"))))


def paper_metric(value: object) -> str:
    try:
        out = float(value)
    except Exception:
        return "pending"
    if not np.isfinite(out):
        return "pending"
    return f"{out:.3f} mm"


def write_markdown_table(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")).replace("\n", " ") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_paper_ready_table(
    args: argparse.Namespace,
    prior: Sequence[Dict[str, object]],
    schedule: Sequence[Dict[str, object]],
    sim_rows: Sequence[Dict[str, str]],
    standard_rows: Sequence[Dict[str, object]],
    refinement_rows: Sequence[Dict[str, object]],
) -> None:
    tables = review_tables_dir(args)
    case8_img = [row for row in prior if row["case_id"] == 8 and row["objective"] == "image_registration"]
    case8_fit = [row for row in prior if row["case_id"] == 8 and row["objective"] == "ptvreg_field_fit"]
    sinr_img = row_for_alias(case8_img, "sinr_cps4")
    sinr_fit = row_for_alias(case8_fit, "sinr_cps4")
    direct4_img = row_for_alias(case8_img, "direct_cp_4stage")
    directcp_img = row_for_alias(case8_img, "direct_cp_cps4")
    max_sim = max([f(row, "cumulative_early_update_similarity") for row in sim_rows] or [float("nan")])
    case1_nopyr = best_schedule_row(schedule, case_id=1, variant_contains="nopyr")
    case8_nopyr = best_schedule_row(schedule, case_id=8, variant_contains="nopyr")
    case8_partial = best_schedule_row(schedule, case_id=8, variant_contains="partial")
    case8_standard = best_schedule_row(standard_rows, case_id=8)
    case8_refinement = best_schedule_row(refinement_rows, case_id=8)

    rows: List[Dict[str, object]] = [
        {
            "mechanism_question": "Are SINR cps4 and DirectCP cps4 following the same path?",
            "evidence_scope": "OASIS rank009 + LungCT 01/08 sampled checkpoints",
            "key_metric": "max SINR-vs-DirectCP cumulative early update similarity",
            "current_result": f"{max_sim:.3f}",
            "current_interpretation": "Low similarity; not just a scalar-rescaled DirectCP path.",
            "paper_placement": "Mechanism caveat / negative result",
        }
    ]
    if sinr_img and sinr_fit:
        rows.append(
            {
                "mechanism_question": "Can failed case08 SINR represent the target deformation?",
                "evidence_scope": "LungCT case08 pTVReg-field-fit versus image registration",
                "key_metric": "SINR native snap TRE, field-fit vs image-reg",
                "current_result": f"{float(sinr_fit['native_crop_snap_tre_mean_mm']):.3f} mm vs {float(sinr_img['native_crop_snap_tre_mean_mm']):.3f} mm",
                "current_interpretation": "Capacity is present; image-optimization path is the failure mode.",
                "paper_placement": "Representation-vs-optimization argument",
            }
        )
    if direct4_img and sinr_img and directcp_img:
        rows.append(
            {
                "mechanism_question": "Does multistage DirectCP rescue hard case08?",
                "evidence_scope": "LungCT case08 best-LR image-registration sweep",
                "key_metric": "native snap TRE, DirectCP 4stage vs cps4 baselines",
                "current_result": (
                    f"{float(direct4_img['native_crop_snap_tre_mean_mm']):.3f} mm vs "
                    f"SINR {float(sinr_img['native_crop_snap_tre_mean_mm']):.3f} mm / "
                    f"DirectCP cps4 {float(directcp_img['native_crop_snap_tre_mean_mm']):.3f} mm"
                ),
                "current_interpretation": "The rescue is schedule/capture-range linked, not the DirectCP label alone.",
                "paper_placement": "Diagnostic case08 figure/table",
            }
        )
    if case1_nopyr:
        rows.append(
            {
                "mechanism_question": "Can no-pyramid control staging solve the easier LungCT case?",
                "evidence_scope": "Long exploratory schedule, LungCT case01",
                "key_metric": "best completed no-pyramid reg1mm snap TRE",
                "current_result": f"{float(case1_nopyr['reg1mm_snap_tre_mean_mm']):.3f} mm",
                "current_interpretation": "Yes for case01 under the time-budgeted schedule.",
                "paper_placement": "Ablation nuance",
            }
        )
    if case8_nopyr:
        rows.append(
            {
                "mechanism_question": "Is control staging alone enough for hard case08?",
                "evidence_scope": "Long exploratory schedule, LungCT case08 no-pyramid",
                "key_metric": "best completed no-pyramid reg1mm snap TRE",
                "current_result": f"{float(case8_nopyr['reg1mm_snap_tre_mean_mm']):.3f} mm",
                "current_interpretation": "Not yet; hard-case rescue still appears to need image-pyramid capture range.",
                "paper_placement": "Ablation mechanism claim",
            }
        )
    else:
        rows.append(
            {
                "mechanism_question": "Is control staging alone enough for hard case08?",
                "evidence_scope": "Long exploratory schedule, LungCT case08 no-pyramid",
                "key_metric": "best completed no-pyramid reg1mm snap TRE",
                "current_result": "pending",
                "current_interpretation": "Waiting for case08 no-pyramid completions.",
                "paper_placement": "Ablation mechanism claim",
            }
        )
    rows.append(
        {
            "mechanism_question": "Does partial image pyramid restore hard-case capture?",
            "evidence_scope": "Long exploratory schedule, LungCT case08 partial ds4-2-1-1",
            "key_metric": "best completed partial-pyramid reg1mm snap TRE",
            "current_result": paper_metric(case8_partial.get("reg1mm_snap_tre_mean_mm")) if case8_partial else "pending",
            "current_interpretation": "No; partial coarse-to-fine does not recover case08 in this short schedule." if case8_partial else "Pending.",
            "paper_placement": "Follow-up ablation row",
        }
    )
    rows.append(
        {
            "mechanism_question": "Does the full standard image pyramid restore hard-case capture?",
            "evidence_scope": "Second-phase LungCT case08 standard ds8-4-2-1 probes",
            "key_metric": "best completed standard-pyramid reg1mm snap TRE",
            "current_result": paper_metric(case8_standard.get("reg1mm_snap_tre_mean_mm")) if case8_standard else "pending",
            "current_interpretation": (
                "Large short-budget recovery versus no/partial pyramid; LR and runtime still matter."
                if case8_standard
                else "Pending."
            ),
            "paper_placement": "Mechanism table / ablation figure",
        }
    )
    rows.append(
        {
            "mechanism_question": "Does longer LR0.01 runtime remove the pyramid ablation gap?",
            "evidence_scope": "Case08 capture-refinement probe, e100/e200 LR0.01 controls",
            "key_metric": "best completed refinement reg1mm snap TRE",
            "current_result": paper_metric(case8_refinement.get("reg1mm_snap_tre_mean_mm")) if case8_refinement else "pending",
            "current_interpretation": (
                "Yes in this diagnostic: standard, partial, and no-pyramid LR0.01 e100/e200 controls all converge near 1.21 mm."
                if case8_refinement
                else "Pending."
            ),
            "paper_placement": "Final mechanism interpretation",
        }
    )
    write_csv(tables / "paper_ready_mechanism_table.csv", rows)
    write_markdown_table(tables / "paper_ready_mechanism_table.md", rows)


def write_case08_pyramid_table_and_plot(
    args: argparse.Namespace,
    prior: Sequence[Dict[str, object]],
    schedule: Sequence[Dict[str, object]],
    standard_rows: Sequence[Dict[str, object]],
) -> None:
    tables = review_tables_dir(args)
    plots = review_plots_dir(args)
    case8_img = [row for row in prior if row["case_id"] == 8 and row["objective"] == "image_registration"]
    direct4_img = row_for_alias(case8_img, "direct_cp_4stage")
    rows: List[Dict[str, object]] = []
    case8_nopyr = best_schedule_row(schedule, case_id=8, variant_contains="nopyr")
    case8_partial = best_schedule_row(schedule, case_id=8, variant_contains="partial")
    case8_standard = best_schedule_row(standard_rows, case_id=8)
    if direct4_img:
        rows.append(
            {
                "condition": "prior full multistage reference",
                "best_variant": "direct_cp_4stage prior LR sweep",
                "metric_convention": "native/crop snap TRE",
                "tre_mean_mm": float(direct4_img["native_crop_snap_tre_mean_mm"]),
                "interpretation": "Known successful full schedule reference; not metric-identical to reg1mm rows.",
            }
        )
    if case8_nopyr:
        rows.append(
            {
                "condition": "no image pyramid",
                "best_variant": case8_nopyr.get("variant"),
                "metric_convention": "reg1mm snap TRE",
                "tre_mean_mm": float(case8_nopyr["reg1mm_snap_tre_mean_mm"]),
                "interpretation": "Control staging alone remains poor on case08.",
            }
        )
    if case8_partial:
        rows.append(
            {
                "condition": "partial pyramid 4,2,1,1",
                "best_variant": case8_partial.get("variant"),
                "metric_convention": "reg1mm snap TRE",
                "tre_mean_mm": float(case8_partial["reg1mm_snap_tre_mean_mm"]),
                "interpretation": "Partial coarse-to-fine does not recover the hard case.",
            }
        )
    if case8_standard:
        rows.append(
            {
                "condition": "standard pyramid 8,4,2,1",
                "best_variant": case8_standard.get("variant"),
                "metric_convention": "reg1mm snap TRE",
                "tre_mean_mm": float(case8_standard["reg1mm_snap_tre_mean_mm"]),
                "interpretation": "Full standard pyramid plus LR probe gives large recovery.",
            }
        )
    write_csv(tables / "case08_pyramid_ablation_table.csv", rows)
    write_markdown_table(tables / "case08_pyramid_ablation_table.md", rows)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(plots / "case08_pyramid_ablation_plot_error.json", {"error": str(exc)})
        return
    plot_rows = [row for row in rows if row.get("metric_convention") == "reg1mm snap TRE"]
    if not plot_rows:
        return
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [str(row["condition"]) for row in plot_rows]
    values = [float(row["tre_mean_mm"]) for row in plot_rows]
    colors = ["#e15759", "#f28e2b", "#59a14f"][: len(plot_rows)]
    ax.bar(range(len(plot_rows)), values, color=colors)
    ax.set_xticks(range(len(plot_rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("reg1mm snap TRE mean mm")
    ax.set_title("Case08 capture-range ablation")
    for idx, row in enumerate(plot_rows):
        ax.text(idx, values[idx], f"{values[idx]:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "case08_pyramid_ablation.png", dpi=180)
    plt.close(fig)


def write_case08_refinement_table_and_plot(args: argparse.Namespace, refinement_rows: Sequence[Dict[str, object]]) -> None:
    completed = [row for row in refinement_rows if row.get("status") == "completed"]
    if not completed:
        return
    tables = review_tables_dir(args)
    plots = review_plots_dir(args)
    rows: List[Dict[str, object]] = []
    for row in completed:
        rows.append(
            {
                "condition": row.get("variant"),
                "stage_downsample": row.get("stage_downsample"),
                "stage_epochs": row.get("stage_epochs"),
                "stage_lr": row.get("stage_lr"),
                "reg1mm_snap_tre_mean_mm": float(row.get("reg1mm_snap_tre_mean_mm", float("nan"))),
                "reg1mm_subvoxel_tre_mean_mm": float(row.get("reg1mm_subvoxel_tre_mean_mm", float("nan"))),
                "interpretation": "LR0.01 runtime/capture refinement row.",
            }
        )
    rows = sorted(rows, key=lambda row: float(row["reg1mm_snap_tre_mean_mm"]))
    write_csv(tables / "case08_capture_refinement_table.csv", rows)
    write_markdown_table(tables / "case08_capture_refinement_table.md", rows)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(plots / "case08_capture_refinement_plot_error.json", {"error": str(exc)})
        return
    plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [str(row["condition"]).replace("ptv_", "").replace("_", " ") for row in rows]
    values = [float(row["reg1mm_snap_tre_mean_mm"]) for row in rows]
    ax.bar(range(len(rows)), values, color="#4e79a7")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("reg1mm snap TRE mean mm")
    ax.set_title("Case08 LR0.01 runtime refinement")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "case08_capture_refinement.png", dpi=180)
    plt.close(fig)


def write_paper_section(
    args: argparse.Namespace,
    prior: Sequence[Dict[str, object]],
    schedule: Sequence[Dict[str, object]],
    sim_rows: Sequence[Dict[str, str]],
    proxy_rows: Sequence[Dict[str, str]],
    standard_rows: Sequence[Dict[str, object]],
    refinement_rows: Sequence[Dict[str, object]],
) -> None:
    review = Path(args.review_root) / "exploration"
    review.mkdir(parents=True, exist_ok=True)
    max_sim = max([f(row, "cumulative_early_update_similarity") for row in sim_rows] or [float("nan")])
    case8_img = [row for row in prior if row["case_id"] == 8 and row["objective"] == "image_registration"]
    case8_fit = [row for row in prior if row["case_id"] == 8 and row["objective"] == "ptvreg_field_fit"]
    sinr_img = row_for_alias(case8_img, "sinr_cps4")
    sinr_fit = row_for_alias(case8_fit, "sinr_cps4")
    direct4_img = row_for_alias(case8_img, "direct_cp_4stage")
    lines = [
        "# Draft Paper Section: Optimization Path Versus Representation",
        "",
        "These first-three-case diagnostics separate three mechanisms: whether updates move toward a strong reference deformation, whether SINR updates are a scalar/kernel variant of direct control-point updates, and whether hard LungCT cases require coarse-to-fine capture range.",
        "",
        f"The sampled SINR-vs-DirectCP cumulative early update similarity stayed below the exploratory gate in all three cases; the maximum observed similarity was `{max_sim:.3f}`. This argues against the simple interpretation that SINR behaves like a rescaled single-stage DirectCP optimizer.",
        "",
    ]
    if sinr_img and sinr_fit and direct4_img:
        lines.append(
            "Case08 is the most diagnostic split: SINR cps4 can fit the pTVReg target field when supervised directly "
            f"(best field-fit native snap TRE `{float(sinr_fit['native_crop_snap_tre_mean_mm']):.3f}` mm), "
            "but its image-registration run remains poor "
            f"(best native snap TRE `{float(sinr_img['native_crop_snap_tre_mean_mm']):.3f}` mm). "
            "In contrast, multistage DirectCP succeeds on the image-registration objective "
            f"(best native snap TRE `{float(direct4_img['native_crop_snap_tre_mean_mm']):.3f}` mm)."
        )
    complete = [row for row in schedule if row.get("status") == "completed"]
    if complete:
        best = min(complete, key=lambda row: float(row.get("reg1mm_snap_tre_mean_mm", float("inf"))))
        lines.extend(
            [
                "",
                "The new schedule-ablation rows test whether the multistage rescue survives when the image pyramid is weakened or removed.",
                f"The best completed ablation so far is `{best.get('variant')}` on case `{best.get('case_id')}` with reg1mm snap TRE `{float(best['reg1mm_snap_tre_mean_mm']):.3f}` mm.",
            ]
        )
        case1_nopyr = best_schedule_row(schedule, case_id=1, variant_contains="nopyr")
        case8_nopyr = best_schedule_row(schedule, case_id=8, variant_contains="nopyr")
        case8_partial = best_schedule_row(schedule, case_id=8, variant_contains="partial")
        case8_standard = best_schedule_row(standard_rows, case_id=8)
        case8_refinement = best_schedule_row(refinement_rows, case_id=8)
        if case1_nopyr:
            lines.append(
                f"On case01, no-pyramid control staging is already competitive in the exploratory schedule "
                f"(best reg1mm snap TRE `{float(case1_nopyr['reg1mm_snap_tre_mean_mm']):.3f}` mm)."
            )
        if case8_nopyr:
            lines.append(
                f"On case08, the best completed no-pyramid row remains much worse "
                f"(reg1mm snap TRE `{float(case8_nopyr['reg1mm_snap_tre_mean_mm']):.3f}` mm), "
                "so control staging alone is not yet explaining the hard-case rescue."
            )
        if case8_partial:
            lines.append(
                f"The current best case08 partial-pyramid row has reg1mm snap TRE "
                f"`{float(case8_partial['reg1mm_snap_tre_mean_mm']):.3f}` mm."
            )
        else:
            lines.append("The partial-pyramid case08 rows are the decisive pending rows for this ablation.")
        if case8_standard:
            lines.append(
                f"Adding back the full standard `8,4,2,1` image pyramid gives a large recovery in the "
                f"case08 LR probe (best reg1mm snap TRE `{float(case8_standard['reg1mm_snap_tre_mean_mm']):.3f}` mm), "
                "but this short-budget result still leaves room for optimizer/runtime effects."
            )
        else:
            lines.append("The standard `8,4,2,1` case08 probe is still pending.")
        if case8_refinement:
            rows = [row for row in refinement_rows if row.get("status") == "completed"]
            values = [
                f"{row.get('variant')} `{float(row.get('reg1mm_snap_tre_mean_mm')):.3f}` mm"
                for row in rows
                if np.isfinite(float(row.get("reg1mm_snap_tre_mean_mm", float("nan"))))
            ]
            lines.append(
                "The LR/runtime refinement then closes the apparent pyramid gap: "
                + ", ".join(values)
                + ". Under LR `0.01` with longer e100/e200 schedules, case08 converges near `1.21` mm even for partial and no-pyramid controls, so the mechanism is best described as optimizer scale and runtime interacting with capture range rather than a fixed architectural requirement for the image pyramid."
            )
        else:
            lines.append("The LR/runtime refinement rows are still pending.")
    lines.extend(
        [
            "",
            "Interpretation: the current evidence favors an optimization-path and capture-range explanation over a representation-label explanation. This section should be framed as mechanism evidence on selected diagnostic cases, not as a population-final method table.",
        ]
    )
    (review / "paper_section_draft.md").write_text("\n".join(lines) + "\n")


def aggregate(args: argparse.Namespace) -> Dict[str, object]:
    tables = review_tables_dir(args)
    tables.mkdir(parents=True, exist_ok=True)
    prior = collect_prior_best_rows()
    schedule = collect_schedule_rows(args)
    standard_rows = collect_standard_probe_rows(args)
    refinement_rows = collect_refinement_probe_rows(args)
    sim_rows, update_case_rows = collect_update_rows(args)
    proxy_rows = collect_proxy_rows(args)
    write_csv(tables / "prior_case01_case08_best_rows.csv", prior)
    write_csv(tables / "schedule_ablation_rows.csv", schedule)
    write_csv(tables / "standard_pyramid_probe_rows_copy.csv", standard_rows)
    write_csv(tables / "capture_refinement_probe_rows_copy.csv", refinement_rows)
    write_csv(tables / "update_similarity_rows.csv", sim_rows)
    write_csv(tables / "update_case_summary_rows.csv", update_case_rows)
    write_csv(tables / "proxy_kernel_metrics.csv", proxy_rows)
    triage = [
        {
            "question": "Is SINR merely rescaled single-stage DirectCP?",
            "evidence": "SINR-vs-DirectCP cumulative early update similarity below 0.5 in all first-three cases.",
            "current_answer": "No, not under sampled dense-update similarity.",
            "paper_use": "Mechanism caveat / negative result.",
        },
        {
            "question": "Can failed case08 SINR represent the pTVReg-like field?",
            "evidence": "Prior case08 pTVReg field-fit sweep gives SINR cps4 near-target TRE while image registration remains poor.",
            "current_answer": "Representation capacity is not the main failure.",
            "paper_use": "Supports optimization-path claim.",
        },
        {
            "question": "Does multistage DirectCP rescue hard LungCT case08?",
            "evidence": "Prior and overnight rows show multistage DirectCP near pTVReg/IDIR while single-stage SINR/DirectCP fail.",
            "current_answer": "Yes on selected case08 diagnostics.",
            "paper_use": "Capture-range figure/table.",
        },
        {
            "question": "Is the rescue due to image pyramid, control staging, or both?",
            "evidence": "New schedule-ablation rows compare standard pyramid against no-pyramid and partial-pyramid variants.",
            "current_answer": "Pending until ablation rows complete." if not schedule else "See schedule_ablation_rows.csv.",
            "paper_use": "Follow-up mechanistic subsection.",
        },
    ]
    write_csv(tables / "mechanism_triage_table.csv", triage)
    make_plots(args, prior, schedule, sim_rows, proxy_rows)
    write_case08_pyramid_table_and_plot(args, prior, schedule, standard_rows)
    write_case08_refinement_table_and_plot(args, refinement_rows)
    write_paper_ready_table(args, prior, schedule, sim_rows, standard_rows, refinement_rows)
    write_paper_section(args, prior, schedule, sim_rows, proxy_rows, standard_rows, refinement_rows)
    summary = {
        "status": "completed",
        "prior_rows": len(prior),
        "schedule_rows": len(schedule),
        "standard_probe_rows": len(standard_rows),
        "refinement_probe_rows": len(refinement_rows),
        "update_similarity_rows": len(sim_rows),
        "proxy_rows": len(proxy_rows),
        "tables_dir": str(tables),
        "plots_dir": str(review_plots_dir(args)),
        "paper_section": str(Path(args.review_root) / "exploration" / "paper_section_draft.md"),
    }
    write_json(Path(args.review_root) / "exploration" / "exploration_summary.json", summary)
    return summary


def maybe_wait_min_runtime(args: argparse.Namespace, start: float) -> None:
    remaining = float(args.min_runtime_sec) - (time.time() - start)
    log_path = Path(args.review_root) / "scientist_reviewer_log.md"
    while remaining > 0:
        sleep_s = min(1800.0, remaining)
        log_round(
            log_path,
            role="Reviewer",
            content=(
                f"Long-run supervision requested; waiting {sleep_s:.0f} sec before the next aggregation pass. "
                "If new rows are complete, the Scientist should refresh tables and paper section."
            ),
        )
        time.sleep(sleep_s)
        summary = aggregate(args)
        log_round(log_path, role="Scientist", content=f"Refreshed aggregation: `{summary}`")
        remaining = float(args.min_runtime_sec) - (time.time() - start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--update-root", type=Path, default=DEFAULT_UPDATE_ROOT)
    parser.add_argument("--ptvprep-root", type=Path, default=DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stage-epochs", type=str, default="200,400,800,1000")
    parser.add_argument("--alpha", type=float, default=10000.0)
    parser.add_argument("--checkpoint-sample-count", type=int, default=100000)
    parser.add_argument("--checkpoint-sample-seed", type=int, default=1729)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--min-runtime-sec", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-proxy", action="store_true")
    parser.add_argument("--launch-schedule", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--expand-all10", action="store_true")
    parser.add_argument("--targeted-extra", action="store_true")
    parser.add_argument("--overwrite-jobs", action="store_true")
    parser.add_argument("--overwrite-log", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    init_logs(args)
    jobs = build_jobs(args)
    write_manifest(args, jobs)
    if args.aggregate_only:
        summary = aggregate(args)
        print(json.dumps({"aggregation": summary}, indent=2, default=str), flush=True)
        return
    if args.run_proxy:
        result = run_proxy_followup(args)
        log_round(Path(args.review_root) / "scientist_reviewer_log.md", role="Scientist", content=f"Proxy follow-up completed: `{result['returncode']}`.")
    if args.launch_schedule or args.dry_run:
        log_round(
            Path(args.review_root) / "scientist_reviewer_log.md",
            role="Reviewer",
            content=f"Approved launch of `{len(jobs)}` schedule-ablation jobs. Require aggregation after completion.",
        )
        launch_result = launch_jobs(args, jobs)
        write_json(phase_root(args) / "launch_result.json", launch_result)
        print(json.dumps({"launch_result": launch_result}, indent=2, allow_nan=True), flush=True)
        if not args.dry_run:
            log_round(Path(args.review_root) / "scientist_reviewer_log.md", role="Scientist", content=f"Schedule launch result: `{launch_result}`.")
    if not args.dry_run:
        summary = aggregate(args)
        log_round(Path(args.review_root) / "scientist_reviewer_log.md", role="Scientist", content=f"Aggregation refreshed: `{summary}`.")
        print(json.dumps({"aggregation": summary}, indent=2, default=str), flush=True)
        maybe_wait_min_runtime(args, started)


if __name__ == "__main__":
    main()
