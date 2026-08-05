#!/usr/bin/env python
"""Run serial one-GPU DirectCP 8-4-2 DVF-only timing jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


METHOD = "direct_cp_8_4_2_be_bspline_analytic_multistage"
VARIANT = "oasis_seed3_directcp842_dvf_only_timing_alpha1000_v1"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_directcp842_dvf_only_timing_alpha1000_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-04_oasis_directcp842_dvf_only_timing"
ALPHA = 1000.0
STAGE_FULL_CPS = "8,4,2"
STAGE_DOWNSAMPLE = "4,2,1"


@dataclass(frozen=True)
class Candidate:
    name: str
    lr: float
    stage_epochs: str
    stage_lr: str = ""
    early_stop_mode: str = "none"
    stage_min_epochs: str = ""
    stage_patience: str = "100"
    stage_check_interval: str = "25"
    stage_rel_delta_total: str = "5e-5"
    stage_rel_delta_data: str = "2e-5"
    description: str = ""
    method: str = METHOD
    epochs: int = 2500


@dataclass(frozen=True)
class Job:
    job_index: int
    candidate: str
    method: str
    phase: str
    case_rank: int
    original_case_index: int
    fixed_id: str
    moving_id: str
    alpha: float
    lr: float
    epochs: int
    stage_epochs: str
    stage_lr: str
    early_stop_mode: str
    stage_min_epochs: str
    stage_patience: str
    stage_check_interval: str
    stage_rel_delta_total: str
    stage_rel_delta_data: str
    prolongation_preserve_check: bool
    save_intermediate_dense: bool


CANDIDATES: Sequence[Candidate] = [
    Candidate(
        "original_mrdbscp",
        1e-3,
        "400,800,1300",
        early_stop_mode="none",
        description="Original MR-D-BSCP / DirectCP 8-4-2 fixed 2500-epoch schedule.",
    ),
    Candidate(
        "A_fast_decay",
        1e-2,
        "400,800,1300",
        stage_lr="3e-2,1e-2,3e-3",
        early_stop_mode="adaptive",
        stage_min_epochs="75,150,250",
        stage_patience="50,75,100",
        stage_rel_delta_total="1e-4",
        stage_rel_delta_data="5e-5",
        description="Fast adaptive plateau with decayed stage learning rates.",
    ),
    Candidate(
        "sinr_cps4",
        1e-4,
        "2500",
        early_stop_mode="none",
        description="SINR / INR-BSCP cps4 fixed 2500-epoch schedule.",
        method="sinr_cps4_be_bspline_analytic",
        epochs=2500,
    ),
    Candidate(
        "sinr_cps2",
        1e-4,
        "2500",
        early_stop_mode="none",
        description="SINR / INR-BSCP cps2 fixed 2500-epoch schedule.",
        method="sinr_cps2_be_bspline_analytic",
        epochs=2500,
    ),
]


def candidate_map() -> Dict[str, Candidate]:
    return {candidate.name: candidate for candidate in CANDIDATES}


def parse_candidate_names(text: str) -> List[str]:
    requested = str(text).strip()
    if requested in {"", "all"}:
        return [candidate.name for candidate in CANDIDATES]
    names = [token.strip() for token in requested.split(",") if token.strip()]
    known = candidate_map()
    missing = [name for name in names if name not in known]
    if missing:
        raise ValueError(f"Unknown candidates {missing}; known={sorted(known)}")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "rank0"], default="rank0")
    parser.add_argument("--candidates", type=str, default="all")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--timing-detail", choices=["coarse", "deep"], default="deep")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--stage-full-cps", type=str, default=STAGE_FULL_CPS)
    parser.add_argument("--stage-downsample", type=str, default=STAGE_DOWNSAMPLE)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--prolongation-preserve-check", action="store_true")
    parser.add_argument("--save-intermediate-dense", action="store_true")
    parser.add_argument(
        "--previous-review-root",
        type=Path,
        default=c.REPO_ROOT / "experiments/2026-08-04_oasis_directcp842_dvf_only_timing",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--allow-incomplete-analyze", action="store_true")
    return parser.parse_args()


def phase_candidate(candidate: Candidate, phase: str) -> Candidate:
    if str(phase) != "smoke":
        return candidate
    return Candidate(
        candidate.name,
        candidate.lr,
        "1,1,1" if candidate.method == METHOD else "1",
        stage_lr=candidate.stage_lr,
        early_stop_mode=candidate.early_stop_mode,
        stage_min_epochs="1,1,1" if candidate.early_stop_mode == "adaptive" else "",
        stage_patience="1,1,1" if candidate.early_stop_mode == "adaptive" else candidate.stage_patience,
        stage_check_interval="1,1,1",
        stage_rel_delta_total=candidate.stage_rel_delta_total,
        stage_rel_delta_data=candidate.stage_rel_delta_data,
        description=f"Tiny smoke version of {candidate.name}.",
        method=candidate.method,
        epochs=3 if candidate.method == METHOD else 1,
    )


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


def stage_epoch_total(stage_epochs: str) -> int:
    return int(sum(c.parse_int_list(str(stage_epochs))))


def run_name(job: Job) -> str:
    suffix = f"dvfonly_{job.candidate}_rank{job.case_rank:03d}_a{c.format_float(job.alpha)}"
    if job.prolongation_preserve_check:
        suffix += "_preservecheck"
    if job.save_intermediate_dense:
        suffix += "_saveinterdense"
    if str(job.phase) == "smoke":
        suffix += "_smoke_e1-1-1"
    return f"{job.method}_{suffix}"


def run_dir(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.out_root) / str(job.method) / run_name(job)


def build_jobs(args: argparse.Namespace) -> List[Job]:
    case = c.resolve_case(rank=0, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
    jobs: List[Job] = []
    for base_candidate in [candidate_map()[name] for name in parse_candidate_names(str(args.candidates))]:
        candidate = phase_candidate(base_candidate, str(args.phase))
        jobs.append(
            Job(
                job_index=len(jobs),
                candidate=candidate.name,
                method=str(candidate.method),
                phase=str(args.phase),
                case_rank=int(case.rank),
                original_case_index=int(case.original_case_index),
                fixed_id=str(case.fixed_id),
                moving_id=str(case.moving_id),
                alpha=float(args.alpha),
                lr=float(candidate.lr),
                epochs=int(candidate.epochs),
                stage_epochs=str(candidate.stage_epochs),
                stage_lr=str(candidate.stage_lr),
                early_stop_mode=str(candidate.early_stop_mode),
                stage_min_epochs=str(candidate.stage_min_epochs),
                stage_patience=str(candidate.stage_patience),
                stage_check_interval=str(candidate.stage_check_interval),
                stage_rel_delta_total=str(candidate.stage_rel_delta_total),
                stage_rel_delta_data=str(candidate.stage_rel_delta_data),
                prolongation_preserve_check=bool(args.prolongation_preserve_check),
                save_intermediate_dense=bool(args.save_intermediate_dense),
            )
        )
    return jobs


def build_cmd(args: argparse.Namespace, job: Job) -> List[str]:
    is_directcp_multistage = str(job.method) == METHOD
    cmd = [
        sys.executable,
        str(THIS_DIR / "run_one.py"),
        "--method",
        str(job.method),
        "--output-mode",
        "dvf_only",
        "--alpha",
        f"{job.alpha:g}",
        "--lr",
        f"{job.lr:g}",
        "--gpu",
        str(args.gpu),
        "--case-rank",
        str(job.case_rank),
        "--epochs",
        str(stage_epoch_total(job.stage_epochs) if is_directcp_multistage else int(job.epochs)),
        "--eval-interval",
        str(1 if str(job.phase) == "smoke" else args.eval_interval),
        "--seed",
        str(args.seed),
        "--runtime-candidate",
        str(job.candidate),
        "--timing-detail",
        str(args.timing_detail),
        "--variant",
        str(args.variant),
        "--out-root",
        str(args.out_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--run-name",
        run_name(job),
        "--expected-original-case-index",
        str(job.original_case_index),
        "--expected-fixed-id",
        str(job.fixed_id),
        "--expected-moving-id",
        str(job.moving_id),
    ]
    if is_directcp_multistage:
        cmd.extend(
            [
                "--stage-full-cps",
                str(args.stage_full_cps),
                "--stage-downsample",
                str(args.stage_downsample),
                "--stage-epochs",
                str(job.stage_epochs),
                "--stage-early-stop-mode",
                str(job.early_stop_mode),
                "--stage-patience",
                str(job.stage_patience),
                "--stage-check-interval",
                str(job.stage_check_interval),
                "--stage-rel-delta-total",
                str(job.stage_rel_delta_total),
                "--stage-rel-delta-data",
                str(job.stage_rel_delta_data),
            ]
        )
    if is_directcp_multistage and job.stage_lr:
        cmd.extend(["--stage-lr", str(job.stage_lr)])
    if is_directcp_multistage and job.stage_min_epochs:
        cmd.extend(["--stage-min-epochs", str(job.stage_min_epochs)])
    if is_directcp_multistage and job.prolongation_preserve_check:
        cmd.append("--prolongation-preserve-check")
    if is_directcp_multistage and job.save_intermediate_dense:
        cmd.append("--save-intermediate-dense")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def capture_gpu_state() -> Dict[str, object]:
    state: Dict[str, object] = {"captured_at": c.now_text(), "gpus": [], "compute_apps": [], "errors": []}
    try:
        gpu_raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = []
        for line in gpu_raw.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                rows.append(
                    {
                        "index": parts[0],
                        "uuid": parts[1],
                        "name": parts[2],
                        "memory_used_mb": parts[3],
                        "utilization_gpu_percent": parts[4],
                    }
                )
        state["gpus"] = rows
    except (OSError, subprocess.CalledProcessError) as exc:
        state["errors"].append(f"gpu_query_failed:{exc}")
    try:
        app_raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        apps = []
        for line in app_raw.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4 and parts[0]:
                apps.append(
                    {
                        "gpu_uuid": parts[0],
                        "pid": parts[1],
                        "process_name": parts[2],
                        "used_memory_mb": parts[3],
                    }
                )
        state["compute_apps"] = apps
    except (OSError, subprocess.CalledProcessError) as exc:
        state["errors"].append(f"compute_app_query_failed:{exc}")
    return state


def non_target_gpu_apps(state: Dict[str, object], target_gpu: str) -> List[Dict[str, object]]:
    gpu_rows = state.get("gpus", [])
    apps = state.get("compute_apps", [])
    if not isinstance(gpu_rows, list) or not isinstance(apps, list):
        return []
    uuid_to_index = {
        str(row.get("uuid")): str(row.get("index"))
        for row in gpu_rows
        if isinstance(row, dict) and row.get("uuid") is not None
    }
    output = []
    for app in apps:
        if not isinstance(app, dict):
            continue
        index = uuid_to_index.get(str(app.get("gpu_uuid")), "")
        if index and index != str(target_gpu):
            output.append({**app, "gpu_index": index})
    return output


def command_timing_paths(args: argparse.Namespace) -> List[Path]:
    filename = f"command_timing_dvf_only_{args.phase}.csv"
    return [Path(args.out_root) / filename, Path(args.review_root) / filename]


def write_command_rows(args: argparse.Namespace, rows: Sequence[Dict[str, object]]) -> None:
    for path in command_timing_paths(args):
        c.write_csv(path, rows)


def write_manifest(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.review_root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(job) for job in jobs]
    selected = parse_candidate_names(str(args.candidates))
    manifest = {
        "created_at": c.now_text(),
        "phase": str(args.phase),
        "job_count": len(jobs),
        "case_ranks": [0],
        "candidates": selected,
        "candidate_definitions": [asdict(candidate_map()[name]) for name in selected],
        "methods": sorted({str(job.method) for job in jobs}),
        "variant": str(args.variant),
        "alpha": float(args.alpha),
        "stage_full_cps": str(args.stage_full_cps),
        "stage_downsample": str(args.stage_downsample),
        "timing_detail": str(args.timing_detail),
        "output_mode": "dvf_only",
        "prolongation_preserve_check": bool(args.prolongation_preserve_check),
        "save_intermediate_dense": bool(args.save_intermediate_dense),
        "gpu": str(args.gpu),
        "serial_one_gpu": True,
        "out_root": str(args.out_root),
        "review_root": str(args.review_root),
        "jobs": rows,
    }
    for root in [args.out_root, args.review_root]:
        c.write_csv(root / f"job_manifest_dvf_only_{args.phase}.csv", rows)
        c.write_json(root / f"job_manifest_dvf_only_{args.phase}.json", manifest)


def run_jobs(args: argparse.Namespace, jobs: Sequence[Job]) -> List[Dict[str, object]]:
    log_dir = Path(args.out_root) / "logs" / f"dvf_only_{args.phase}"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = thread_env(int(args.cpu_threads))
    command_rows: List[Dict[str, object]] = []
    gpu_state_rows: List[Dict[str, object]] = []

    if args.dry_run:
        for job in jobs:
            print(json.dumps({"cmd": build_cmd(args, job), "run_dir": str(run_dir(args, job))}), flush=True)
        return []

    for job in jobs:
        target_dir = run_dir(args, job)
        log_path = log_dir / f"{run_name(job)}_gpu{args.gpu}.log"
        pre_state = capture_gpu_state()
        non_target_pre = non_target_gpu_apps(pre_state, str(args.gpu))
        if non_target_pre:
            print(json.dumps({"warning": "non_target_gpu_compute_apps_pre", "apps": non_target_pre}), flush=True)
        cmd = build_cmd(args, job)
        print(json.dumps({"launch_serial": cmd, "log": str(log_path)}), flush=True)
        start_wall = time.perf_counter()
        started_at = c.now_text()
        with log_path.open("w") as log_f:
            proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, check=False)
        external_wall_sec = time.perf_counter() - start_wall
        completed_at = c.now_text()
        post_state = capture_gpu_state()
        non_target_post = non_target_gpu_apps(post_state, str(args.gpu))
        dense_path = target_dir / f"disp_norm_{job.fixed_id}_{job.moving_id}.npy"
        row = {
            **asdict(job),
            "started_at": started_at,
            "completed_at": completed_at,
            "gpu": str(args.gpu),
            "returncode": int(proc.returncode),
            "external_wall_sec": float(external_wall_sec),
            "run_dir": str(target_dir),
            "log_path": str(log_path),
            "done_exists": (target_dir / "DONE").exists(),
            "summary_exists": (target_dir / "summary.json").exists(),
            "timing_exists": (target_dir / "timing.json").exists(),
            "dense_dvf_exists": dense_path.exists(),
            "dense_dvf_path": str(dense_path),
            "cmd_json": json.dumps(cmd),
            "non_target_gpu_apps_pre_count": len(non_target_pre),
            "non_target_gpu_apps_post_count": len(non_target_post),
        }
        command_rows.append(row)
        gpu_state_rows.extend(
            [
                {
                    "phase": str(args.phase),
                    "job_index": int(job.job_index),
                    "candidate": str(job.candidate),
                    "case_rank": int(job.case_rank),
                    "snapshot": "pre",
                    "state_json": json.dumps(pre_state, allow_nan=True),
                },
                {
                    "phase": str(args.phase),
                    "job_index": int(job.job_index),
                    "candidate": str(job.candidate),
                    "case_rank": int(job.case_rank),
                    "snapshot": "post",
                    "state_json": json.dumps(post_state, allow_nan=True),
                },
            ]
        )
        write_command_rows(args, command_rows)
        for root in [args.out_root, args.review_root]:
            c.write_csv(root / f"gpu_state_dvf_only_{args.phase}.csv", gpu_state_rows)
        if proc.returncode != 0:
            raise RuntimeError(f"DVF-only timing job failed with returncode {proc.returncode}; see {log_path}")
    return command_rows


def analyze(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "analyze_directcp842_dvf_only_timing.py"),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--phase",
        str(args.phase),
        "--expected-case-ranks",
        "0",
        "--expected-candidates",
        ",".join(parse_candidate_names(str(args.candidates))),
        "--previous-review-root",
        str(args.previous_review_root),
    ]
    if args.allow_incomplete_analyze:
        cmd.append("--allow-incomplete")
    print(json.dumps({"analyze_cmd": cmd}), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    write_manifest(args, jobs)
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "phase": str(args.phase),
                "job_count": len(jobs),
                "case_ranks": [0],
                "candidates": parse_candidate_names(str(args.candidates)),
                "gpu": str(args.gpu),
                "serial_one_gpu": True,
                "timing_detail": str(args.timing_detail),
                "output_mode": "dvf_only",
                "prolongation_preserve_check": bool(args.prolongation_preserve_check),
                "save_intermediate_dense": bool(args.save_intermediate_dense),
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        ),
        flush=True,
    )
    rows = run_jobs(args, jobs)
    summary = {
        "completed_at": c.now_text(),
        "phase": str(args.phase),
        "job_count": len(jobs),
        "completed_count": sum(1 for row in rows if int(row.get("returncode", 1)) == 0),
        "done_count": sum(1 for job in jobs if (run_dir(args, job) / "DONE").exists()),
        "timing_json_count": sum(1 for job in jobs if (run_dir(args, job) / "timing.json").exists()),
        "dense_dvf_count": sum(
            1 for job in jobs if (run_dir(args, job) / f"disp_norm_{job.fixed_id}_{job.moving_id}.npy").exists()
        ),
        "serial_one_gpu": True,
        "gpu": str(args.gpu),
    }
    for root in [args.out_root, args.review_root]:
        c.write_json(root / f"launch_summary_dvf_only_{args.phase}.json", summary)
    if not args.dry_run and not args.skip_analyze:
        analyze(args)


if __name__ == "__main__":
    main()
