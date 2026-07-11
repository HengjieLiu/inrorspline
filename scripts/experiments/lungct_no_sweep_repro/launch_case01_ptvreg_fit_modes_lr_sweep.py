#!/usr/bin/env python
"""Launch a LungCT registration vs pTVReg-field fitting LR sweep."""

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


DEFAULT_CASE_ID = 1
LRS = [1e-2, 1e-3, 1e-4]


def variant_for_case(case_id: int) -> str:
    return f"lungct_case{int(case_id):02d}_ptvreg_fit_modes_lr_sweep_v1"


def default_run_root(case_id: int) -> Path:
    return Path(f"/database/inrorspline/runs/{variant_for_case(case_id)}")


def default_review_root(case_id: int) -> Path:
    return c.REPO_ROOT / f"experiments/2026-06-26_lungct_case{int(case_id):02d}_ptvreg_fit_modes_lr_sweep"


def default_ptvreg_mat(case_id: int) -> Path:
    return Path(f"/database/inrorspline/ptvreg/4DCT/case{int(case_id):02d}_T50_to_T00_deformation.mat")


@dataclass(frozen=True)
class MethodSpec:
    alias: str
    label: str
    image_method: str
    field_method: str
    default_gpu: str


@dataclass(frozen=True)
class Job:
    job_index: int
    objective: str
    method_alias: str
    method_label: str
    image_method: str
    field_method: str
    lr: float
    gpu: str
    case_id: int


METHOD_SPECS = [
    MethodSpec("idir", "IDIR", "idir_be_default_lungmask", "idir", "0"),
    MethodSpec("sinr_cps4", "SINR cps4", "sinr_cps4_ptv_lcc_isotv", "sinr_cps4", "1"),
    MethodSpec("direct_cp_cps4", "DirectCP cps4", "direct_cp_cps4_ptv_lcc_isotv", "direct_cp_cps4", "2"),
    MethodSpec("direct_cp_4stage", "DirectCP 4stage cps4", "direct_cp_32_16_8_4_ptv_lcc_isotv", "direct_cp_4stage", "3"),
]
OBJECTIVES = ["image_registration", "ptvreg_field_fit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-mode", choices=("method_batches", "method_pinned"), default="method_batches")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--method-gpu-map", type=str, default="idir:0,sinr_cps4:1,direct_cp_cps4:2,direct_cp_4stage:3")
    parser.add_argument("--case-id", type=int, default=DEFAULT_CASE_ID)
    parser.add_argument("--data-root", type=Path, default=c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--ptvreg-mat", type=Path, default=None)
    parser.add_argument("--expected-tre-phys", type=float, default=None)
    parser.add_argument("--expected-tre-std", type=float, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--stage-epochs", type=str, default="400,400,800,900")
    parser.add_argument("--stage-full-cps", type=str, default="32,16,8,4")
    parser.add_argument("--stage-downsample", type=str, default="8,4,2,1")
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--coords-batch-size", type=int, default=10000)
    parser.add_argument("--eval-chunk-size", type=int, default=200000)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
    parser.add_argument("--alpha-bending", type=float, default=10.0)
    parser.add_argument("--ptv-isotv-weight", type=float, default=0.11)
    parser.add_argument("--border-mask-voxels", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()
    if args.ptvreg_mat is None:
        args.ptvreg_mat = default_ptvreg_mat(int(args.case_id))
    if args.out_root is None:
        args.out_root = default_run_root(int(args.case_id))
    if args.review_root is None:
        args.review_root = default_review_root(int(args.case_id))
    return args


def parse_method_gpu_map(text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid method GPU token {token!r}; expected method:gpu")
        method, gpu = token.split(":", 1)
        mapping[method.strip()] = gpu.strip()
    aliases = {spec.alias for spec in METHOD_SPECS}
    missing = sorted(aliases - set(mapping))
    extra = sorted(set(mapping) - aliases)
    if missing or extra:
        raise ValueError(f"Invalid method GPU map. missing={missing}, extra={extra}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"Each method must use a distinct GPU: {mapping}")
    return mapping


def parse_gpus(text: str) -> List[str]:
    gpus = [token.strip() for token in str(text).split(",") if token.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required.")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"Duplicate GPUs are not allowed: {gpus}")
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


def build_jobs(args: argparse.Namespace) -> List[Job]:
    gpu_map = parse_method_gpu_map(args.method_gpu_map)
    jobs: List[Job] = []
    for spec in METHOD_SPECS:
        for objective in OBJECTIVES:
            for lr in LRS:
                jobs.append(
                    Job(
                        job_index=len(jobs),
                        objective=objective,
                        method_alias=spec.alias,
                        method_label=spec.label,
                        image_method=spec.image_method,
                        field_method=spec.field_method,
                        lr=float(lr),
                        gpu=str(gpu_map[spec.alias]),
                        case_id=int(args.case_id),
                    )
                )
    return jobs


def run_name_for_job(args: argparse.Namespace, job: Job) -> str:
    lr_text = c.format_float(job.lr)
    if job.method_alias == "direct_cp_4stage":
        stage_epochs = str(args.stage_epochs).replace(",", "-")
        schedule = "sched32-16-8-4"
        if job.objective == "image_registration":
            return f"image_reg_{job.method_alias}_case{job.case_id:02d}_{schedule}_e{stage_epochs}_lr{lr_text}"
        return f"ptvreg_field_fit_{job.method_alias}_case{job.case_id:02d}_{schedule}_e{stage_epochs}_lr{lr_text}"
    if job.objective == "image_registration":
        return f"image_reg_{job.method_alias}_case{job.case_id:02d}_e{args.epochs}_lr{lr_text}"
    return f"ptvreg_field_fit_{job.method_alias}_case{job.case_id:02d}_e{args.epochs}_lr{lr_text}"


def field_method_key(alias: str) -> str:
    return {
        "idir": "idir_ptvreg_field_mse",
        "sinr_cps4": "sinr_cps4_ptvreg_field_mse",
        "direct_cp_cps4": "direct_cp_cps4_ptvreg_field_mse",
        "direct_cp_4stage": "direct_cp_4stage_ptvreg_field_mse",
    }[alias]


def run_dir_for_job(args: argparse.Namespace, job: Job) -> Path:
    if job.objective == "image_registration":
        return Path(args.out_root) / job.image_method / run_name_for_job(args, job)
    return Path(args.out_root) / field_method_key(job.method_alias) / run_name_for_job(args, job)


def is_completed(args: argparse.Namespace, job: Job) -> bool:
    run_dir = run_dir_for_job(args, job)
    if job.objective == "ptvreg_field_fit":
        return (run_dir / "DONE").exists() and (run_dir / "summary.json").exists()
    summary = run_dir / "summary.json"
    if not summary.exists():
        return False
    try:
        payload = c.read_json(summary)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("status") == "completed"


def build_cmd(args: argparse.Namespace, job: Job, gpu: str | None = None) -> List[str]:
    assigned_gpu = str(job.gpu if gpu is None else gpu)
    if job.objective == "image_registration":
        cmd = [
            sys.executable,
            str(THIS_DIR / "run_one.py"),
            "--method",
            job.image_method,
            "--case-id",
            str(job.case_id),
            "--data-root",
            str(args.data_root),
            "--out-root",
            str(args.out_root),
            "--run-name",
            run_name_for_job(args, job),
            "--gpu",
            assigned_gpu,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--lr",
            f"{job.lr:g}",
            "--eval-interval",
            str(args.eval_interval),
            "--eval-chunk-size",
            str(args.eval_chunk_size),
            "--batch-size",
            str(args.coords_batch_size),
            "--alpha-bending",
            f"{args.alpha_bending:g}",
            "--ptv-isotv-weight",
            f"{args.ptv_isotv_weight:g}",
            "--border-mask-voxels",
            str(args.border_mask_voxels),
        ]
        if job.method_alias == "direct_cp_4stage":
            cmd.extend(
                [
                    "--stage-full-cps",
                    str(args.stage_full_cps),
                    "--stage-downsample",
                    str(args.stage_downsample),
                    "--stage-epochs",
                    str(args.stage_epochs),
                    "--stage-lr",
                    f"{job.lr:g}",
                ]
            )
    else:
        cmd = [
            sys.executable,
            str(THIS_DIR / "run_case01_ptvreg_field_fit_one.py"),
            "--method",
            job.field_method,
            "--case-id",
            str(job.case_id),
            "--data-root",
            str(args.data_root),
            "--ptvreg-mat",
            str(args.ptvreg_mat),
            "--out-root",
            str(args.out_root),
            "--run-name",
            run_name_for_job(args, job),
            "--gpu",
            assigned_gpu,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--lr",
            f"{job.lr:g}",
            "--eval-interval",
            str(args.eval_interval),
            "--eval-chunk-size",
            str(args.eval_chunk_size),
            "--coords-batch-size",
            str(args.coords_batch_size),
            "--stage-full-cps",
            str(args.stage_full_cps),
            "--stage-downsample",
            str(args.stage_downsample),
            "--stage-epochs",
            str(args.stage_epochs),
            "--stage-lr",
            f"{job.lr:g}",
        ]
        if args.expected_tre_phys is not None:
            cmd.extend(["--expected-tre-phys", f"{float(args.expected_tre_phys):.17g}"])
        if args.expected_tre_std is not None:
            cmd.extend(["--expected-tre-std", f"{float(args.expected_tre_std):.17g}"])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def log_name(job: Job, gpu: str | None = None) -> str:
    assigned_gpu = str(job.gpu if gpu is None else gpu)
    return (
        f"{job.objective}_{job.method_alias}_case{job.case_id:02d}"
        f"_lr{c.format_float(job.lr)}_gpu{assigned_gpu}.log"
    )


def write_manifest(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(job) | {"run_name": run_name_for_job(args, job), "run_dir": str(run_dir_for_job(args, job))} for job in jobs]
    c.write_csv(args.out_root / "job_manifest.csv", rows)
    c.write_json(
        args.out_root / "job_manifest.json",
        {
            "created_at": c.now_text(),
            "variant": variant_for_case(int(args.case_id)),
            "case_id": int(args.case_id),
            "job_count": len(jobs),
            "objectives": OBJECTIVES,
            "learning_rates": LRS,
            "schedule_mode": str(args.schedule_mode),
            "gpu_pool": parse_gpus(args.gpus),
            "method_gpu_map": parse_method_gpu_map(args.method_gpu_map),
            "epochs": int(args.epochs),
            "stage_epochs": str(args.stage_epochs),
            "stage_full_cps": str(args.stage_full_cps),
            "stage_downsample": str(args.stage_downsample),
            "seed": int(args.seed),
            "coords_batch_size": int(args.coords_batch_size),
            "alpha_bending": float(args.alpha_bending),
            "ptv_isotv_weight": float(args.ptv_isotv_weight),
            "field_fit_domain": "fixed_t00_lung_mask_reg1mm",
            "out_root": str(args.out_root),
            "review_root": str(args.review_root),
            "data_root": str(args.data_root),
            "ptvreg_mat": str(args.ptvreg_mat),
            "expected_tre_phys": args.expected_tre_phys,
            "expected_tre_std": args.expected_tre_std,
            "method_specs": [asdict(spec) for spec in METHOD_SPECS],
            "jobs": rows,
        },
    )


def launch_method_pinned(args: argparse.Namespace, jobs: Sequence[Job]) -> List[Dict[str, object]]:
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Job]] = {spec.alias: [] for spec in METHOD_SPECS}
    for job in jobs:
        by_method[job.method_alias].append(job)
    running: Dict[str, Dict[str, object]] = {}
    failures: List[Dict[str, object]] = []
    env = thread_env(args.cpu_threads)
    started = time.time()

    if args.dry_run:
        for job in jobs:
            cmd = build_cmd(args, job, job.gpu)
            print(
                json.dumps(
                    {
                        "schedule_mode": "method_pinned",
                        "launch": cmd,
                        "log": str(log_dir / log_name(job, job.gpu)),
                        "run_dir": str(run_dir_for_job(args, job)),
                    }
                ),
                flush=True,
            )
        return []

    while any(by_method.values()) or running:
        for method, queue in by_method.items():
            if method in running or not queue:
                continue
            job = queue.pop(0)
            if is_completed(args, job) and not args.overwrite:
                print(json.dumps({"skip_existing_completed": asdict(job), "run_dir": str(run_dir_for_job(args, job))}), flush=True)
                continue
            cmd = build_cmd(args, job, job.gpu)
            log_path = log_dir / log_name(job, job.gpu)
            print(json.dumps({"launch": cmd, "log": str(log_path), "run_dir": str(run_dir_for_job(args, job))}), flush=True)
            log_f = log_path.open("w")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            running[method] = {"job": job, "proc": proc, "log_file": log_f, "log_path": log_path, "cmd": cmd}
            time.sleep(float(args.launch_stagger_sec))

        completed_methods: List[str] = []
        for method, item in running.items():
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            code = proc.poll()
            if code is None:
                continue
            log_f = item["log_file"]
            log_f.close()
            job = item["job"]
            assert isinstance(job, Job)
            if code != 0:
                failures.append(
                    {
                        "job": asdict(job),
                        "returncode": code,
                        "log_path": str(item["log_path"]),
                        "cmd": item["cmd"],
                    }
                )
            else:
                print(json.dumps({"completed": asdict(job), "run_dir": str(run_dir_for_job(args, job))}), flush=True)
            completed_methods.append(method)
        for method in completed_methods:
            running.pop(method, None)
        if running or any(by_method.values()):
            time.sleep(5.0)

    return failures


def launch_method_batches(args: argparse.Namespace, jobs: Sequence[Job]) -> List[Dict[str, object]]:
    log_dir = args.out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = parse_gpus(args.gpus)
    failures: List[Dict[str, object]] = []
    env = thread_env(args.cpu_threads)

    for spec in METHOD_SPECS:
        queue = [job for job in jobs if job.method_alias == spec.alias]
        running: List[Dict[str, object]] = []
        print(
            json.dumps(
                {
                    "method_batch_start": spec.alias,
                    "method_label": spec.label,
                    "jobs": len(queue),
                    "gpus": gpus,
                }
            ),
            flush=True,
        )

        if args.dry_run:
            for index, job in enumerate(queue):
                gpu = gpus[index % len(gpus)]
                cmd = build_cmd(args, job, gpu)
                print(
                    json.dumps(
                        {
                            "schedule_mode": "method_batches",
                            "method_batch": spec.alias,
                            "launch": cmd,
                            "log": str(log_dir / log_name(job, gpu)),
                            "run_dir": str(run_dir_for_job(args, job)),
                        }
                    ),
                    flush=True,
                )
            continue

        while queue or running:
            while queue and len(running) < len(gpus):
                busy_gpus = {str(item["gpu"]) for item in running}
                free_gpus = [gpu for gpu in gpus if gpu not in busy_gpus]
                if not free_gpus:
                    break
                job = queue.pop(0)
                gpu = free_gpus[0]
                if is_completed(args, job) and not args.overwrite:
                    print(json.dumps({"skip_existing_completed": asdict(job), "run_dir": str(run_dir_for_job(args, job))}), flush=True)
                    continue
                cmd = build_cmd(args, job, gpu)
                log_path = log_dir / log_name(job, gpu)
                print(
                    json.dumps(
                        {
                            "schedule_mode": "method_batches",
                            "method_batch": spec.alias,
                            "launch": cmd,
                            "log": str(log_path),
                            "run_dir": str(run_dir_for_job(args, job)),
                        }
                    ),
                    flush=True,
                )
                log_f = log_path.open("w")
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
                running.append({"job": job, "gpu": gpu, "proc": proc, "log_file": log_f, "log_path": log_path, "cmd": cmd})
                time.sleep(float(args.launch_stagger_sec))

            still_running: List[Dict[str, object]] = []
            for item in running:
                proc = item["proc"]
                assert isinstance(proc, subprocess.Popen)
                code = proc.poll()
                if code is None:
                    still_running.append(item)
                    continue
                log_f = item["log_file"]
                log_f.close()
                job = item["job"]
                assert isinstance(job, Job)
                if code != 0:
                    failures.append(
                        {
                            "job": asdict(job),
                            "gpu": item["gpu"],
                            "returncode": code,
                            "log_path": str(item["log_path"]),
                            "cmd": item["cmd"],
                        }
                    )
                else:
                    print(json.dumps({"completed": asdict(job), "gpu": item["gpu"], "run_dir": str(run_dir_for_job(args, job))}), flush=True)
            running = still_running
            if running or queue:
                time.sleep(5.0)

        print(json.dumps({"method_batch_completed": spec.alias, "failures_so_far": len(failures)}), flush=True)
    return failures


def launch_jobs(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    started = time.time()
    if args.schedule_mode == "method_pinned":
        failures = launch_method_pinned(args, jobs)
    else:
        failures = launch_method_batches(args, jobs)
    if args.dry_run:
        return
    c.write_json(
        args.out_root / "launch_summary.json",
        {
            "completed_at": c.now_text(),
            "variant": variant_for_case(int(args.case_id)),
            "schedule_mode": str(args.schedule_mode),
            "job_count": len(jobs),
            "failures": failures,
            "elapsed_sec": time.time() - started,
        },
    )
    if failures:
        raise RuntimeError(
            f"One or more LungCT case{int(args.case_id):02d} fit-mode jobs failed: "
            + json.dumps(failures, indent=2, default=str)
        )


def aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_case01_ptvreg_fit_modes_lr_sweep.py"),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--case-id",
        str(args.case_id),
        "--expected-epochs",
        str(args.epochs),
        "--expected-stage-epochs",
        str(args.stage_epochs),
        "--expected-seed",
        str(args.seed),
    ]
    if args.expected_tre_phys is not None:
        cmd.extend(["--expected-source-tre", f"{float(args.expected_tre_phys):.17g}"])
    print(json.dumps({"aggregate_cmd": cmd}), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    write_manifest(args, jobs)
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "mode": "dry_run" if args.dry_run else "launch",
                "variant": variant_for_case(int(args.case_id)),
                "job_count": len(jobs),
                "schedule_mode": str(args.schedule_mode),
                "gpus": parse_gpus(args.gpus),
                "method_gpu_map": parse_method_gpu_map(args.method_gpu_map),
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
            },
            indent=2,
        ),
        flush=True,
    )
    launch_jobs(args, jobs)
    if not args.dry_run and not args.skip_aggregate:
        aggregate(args)


if __name__ == "__main__":
    main()
