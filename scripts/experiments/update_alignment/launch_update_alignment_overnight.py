#!/usr/bin/env python
"""Launch and aggregate the first-three-case update-alignment diagnostic."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


THIS_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "docs").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(THIS_DIR)
OASIS_DIR = REPO_ROOT / "scripts/experiments/oasis_alpha_sweep"
LUNG_DIR = REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro"
OASIS_RUN_ONE = OASIS_DIR / "run_one.py"
LUNG_RUN_ONE = LUNG_DIR / "run_one.py"
FOLLOWUP_SCRIPT = THIS_DIR / "run_sinr_cp_kernel_followup.py"

DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/update_alignment_overnight_20260629_v1")
DEFAULT_REVIEW_ROOT = REPO_ROOT / "experiments/2026-06-29_update_alignment_overnight"
DEFAULT_OASIS_ROOT = Path("/database/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1")
DEFAULT_PAIR_FILE = REPO_ROOT / "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"
DEFAULT_PTVPREP_ROOT = Path(
    "/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm"
)
DEFAULT_PTVREG_ROOT = Path("/database/inrorspline/ptvreg/4DCT")

CHECKPOINT_EPOCHS_FULL = "0,1,10,25,50,100,250,500,1000,1500,2000,2500"
CHECKPOINT_EPOCHS_SMOKE = "0,1,2"
PROMISING_SIMILARITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class Job:
    job_index: int
    dataset: str
    case_key: str
    method: str
    method_key: str
    run_name: str
    gpu: str
    command: List[str]
    run_dir: str
    log_path: str
    teacher: str


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
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def format_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m").replace("+", "")


def parse_gpus(text: str) -> List[str]:
    gpus = [token.strip() for token in str(text).split(",") if token.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required.")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"Duplicate GPUs are not allowed: {gpus}")
    return gpus


def phase_root(args: argparse.Namespace) -> Path:
    phase = "smoke" if args.smoke else "full"
    return Path(args.out_root) / phase


def review_phase_root(args: argparse.Namespace) -> Path:
    phase = "smoke" if args.smoke else "full"
    return Path(args.review_root) / phase


def oasis_method_key(method: str) -> str:
    if method == "idir_original":
        return "idir_original_dense_siren_be_autograd"
    return method


def build_oasis_jobs(args: argparse.Namespace, root: Path, log_dir: Path, gpus: Sequence[str], smoke: bool) -> List[Job]:
    epochs = 2 if smoke else int(args.epochs)
    checkpoint_epochs = CHECKPOINT_EPOCHS_SMOKE if smoke else CHECKPOINT_EPOCHS_FULL
    sample_count = int(args.smoke_sample_count if smoke else args.checkpoint_sample_count)
    specs = [
        ("idir_original", 1.0, 1e-4),
        ("sinr_cps4_be_bspline_analytic", 1000.0, 1e-4),
        ("direct_cp_cps4_be_bspline_analytic", 1000.0, 1e-3),
    ]
    jobs: List[Job] = []
    for method, alpha, lr in specs:
        method_key = oasis_method_key(method)
        run_name = f"update_align_oasis_rank009_{method_key}_e{epochs}_lr{format_float(lr)}"
        gpu = gpus[len(jobs) % len(gpus)]
        run_dir = root / method_key / run_name
        log_path = log_dir / f"oasis_rank009_{method_key}_gpu{gpu}.log"
        cmd = [
            sys.executable,
            str(OASIS_RUN_ONE),
            "--method",
            method,
            "--alpha",
            f"{alpha:g}",
            "--lr",
            f"{lr:g}",
            "--gpu",
            gpu,
            "--case-rank",
            "9",
            "--epochs",
            str(epochs),
            "--eval-interval",
            "1" if smoke else "250",
            "--seed",
            str(args.seed),
            "--variant",
            "update_alignment_overnight_20260629_v1",
            "--out-root",
            str(root),
            "--oasis-root",
            str(args.oasis_root),
            "--pair-file",
            str(args.pair_file),
            "--expected-original-case-index",
            "88",
            "--expected-fixed-id",
            "0173",
            "--expected-moving-id",
            "0136",
            "--run-name",
            run_name,
            "--sample-checkpoints",
            "--checkpoint-sample-count",
            str(sample_count),
            "--checkpoint-sample-seed",
            str(args.checkpoint_sample_seed),
            "--checkpoint-epochs",
            checkpoint_epochs,
        ]
        if args.overwrite or smoke:
            cmd.append("--overwrite")
        jobs.append(
            Job(
                job_index=len(jobs),
                dataset="oasis",
                case_key="oasis_rank009",
                method=method,
                method_key=method_key,
                run_name=run_name,
                gpu=gpu,
                command=cmd,
                run_dir=str(run_dir),
                log_path=str(log_path),
                teacher="sitreg_bs4",
            )
        )
    return jobs


def build_lung_jobs(args: argparse.Namespace, root: Path, log_dir: Path, gpus: Sequence[str], smoke: bool) -> List[Job]:
    epochs = 2 if smoke else int(args.epochs)
    stage_epochs = "1,1,1,1" if smoke else str(args.stage_epochs)
    checkpoint_epochs = CHECKPOINT_EPOCHS_SMOKE if smoke else CHECKPOINT_EPOCHS_FULL
    sample_count = int(args.smoke_sample_count if smoke else args.checkpoint_sample_count)
    specs = [
        ("idir_be_default_lungmask", 1e-4),
        ("sinr_cps4_ptv_lcc_isotv", 1e-4),
        ("direct_cp_cps4_ptv_lcc_isotv", 1e-3),
        ("direct_cp_32_16_8_4_ptv_lcc_isotv", 1e-3),
    ]
    jobs: List[Job] = []
    for case_id in (1, 8):
        for method, lr in specs:
            gpu = gpus[(3 + len(jobs)) % len(gpus)]
            method_suffix = method.replace("direct_cp_32_16_8_4", "directcp321684")
            run_name = f"update_align_lungct_case{case_id:02d}_{method_suffix}_e{epochs}_lr{format_float(lr)}"
            run_dir = root / method / run_name
            log_path = log_dir / f"lungct_case{case_id:02d}_{method}_gpu{gpu}.log"
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
                str(epochs),
                "--lr",
                f"{lr:g}",
                "--eval-interval",
                "1" if smoke else "250",
                "--ptv-isotv-weight",
                "0.11",
                "--alpha-bending",
                "10",
                "--stage-epochs",
                stage_epochs,
                "--stage-full-cps",
                "32,16,8,4",
                "--stage-downsample",
                "8,4,2,1",
                "--stage-lr",
                "1e-3",
                "--sample-checkpoints",
                "--checkpoint-sample-count",
                str(sample_count),
                "--checkpoint-sample-seed",
                str(args.checkpoint_sample_seed),
                "--checkpoint-epochs",
                checkpoint_epochs,
            ]
            if args.overwrite or smoke:
                cmd.append("--overwrite")
            jobs.append(
                Job(
                    job_index=3 + len(jobs),
                    dataset="lungct",
                    case_key=f"lungct_case{case_id:02d}",
                    method=method,
                    method_key=method,
                    run_name=run_name,
                    gpu=gpu,
                    command=cmd,
                    run_dir=str(run_dir),
                    log_path=str(log_path),
                    teacher="ptvreg",
                )
            )
    return jobs


def build_jobs(args: argparse.Namespace) -> List[Job]:
    gpus = parse_gpus(args.gpus)
    root = phase_root(args)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_oasis_jobs(args, root, log_dir, gpus, args.smoke)
    jobs.extend(build_lung_jobs(args, root, log_dir, gpus, args.smoke))
    jobs = [Job(**{**asdict(job), "job_index": idx}) for idx, job in enumerate(jobs)]
    return jobs


def write_manifest(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    root = phase_root(args)
    rows = [asdict(job) for job in jobs]
    write_csv(root / "job_manifest.csv", rows)
    write_json(
        root / "job_manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "phase": "smoke" if args.smoke else "full",
            "job_count": len(jobs),
            "scope": args.scope,
            "gpus": args.gpus,
            "out_root": str(root),
            "review_root": str(review_phase_root(args)),
            "checkpoint_sample_count": int(args.smoke_sample_count if args.smoke else args.checkpoint_sample_count),
            "checkpoint_sample_seed": int(args.checkpoint_sample_seed),
            "jobs": rows,
        },
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


def launch_jobs(args: argparse.Namespace, jobs: Sequence[Job]) -> Dict[str, object]:
    if args.dry_run:
        for job in jobs:
            print(json.dumps({"job": asdict(job), "command": job.command, "log": job.log_path}), flush=True)
        return {"status": "dry_run", "job_count": len(jobs)}

    pending = list(jobs)
    running: List[Dict[str, object]] = []
    completed: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    env = thread_env(args.cpu_threads)
    max_concurrent = min(len(parse_gpus(args.gpus)), int(args.max_concurrent), len(pending))
    root = phase_root(args)
    status_path = root / "phase_status.json"
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
            handle = item["log_handle"]
            handle.close()
            elapsed = time.time() - float(item["started"])
            row = {"job": job.run_name, "dataset": job.dataset, "case_key": job.case_key, "method": job.method, "returncode": code, "elapsed_sec": elapsed}
            if code == 0:
                completed.append(row)
            else:
                failures.append(row)
            print(json.dumps({"finished": row}), flush=True)
        running = still_running
        write_json(
            status_path,
            {
                "status": "running" if running or pending else "finished",
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
            raise RuntimeError(f"Stopping after job failures: {failures}")
        if pending or running:
            time.sleep(float(args.poll_sec))
    return {
        "status": "completed" if not failures else "partial_failed",
        "completed": completed,
        "failures": failures,
        "elapsed_sec": time.time() - started,
    }


def load_module(module_name: str, path: Path, search_dir: Path):
    sys.modules.pop("common", None)
    sys.modules.pop("run_one", None)
    if str(search_dir) not in sys.path:
        sys.path.insert(0, str(search_dir))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scale_norm_to_vox_xyz(values: np.ndarray, shape_zyx: Sequence[int]) -> np.ndarray:
    d, h, w = [int(v) for v in shape_zyx]
    scale = np.asarray([(w - 1) / 2.0, (h - 1) / 2.0, (d - 1) / 2.0], dtype=np.float32)
    return values.astype(np.float32, copy=False) * scale.reshape(1, 3)


def load_oasis_teacher(args: argparse.Namespace, flat_indices: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    sitreg_mod = load_module("update_alignment_oasis_sitreg", OASIS_DIR / "run_rank009_sitreg_field_fit_one.py", OASIS_DIR)
    oasis_common = sitreg_mod.c
    device = sitreg_mod.torch.device("cpu")
    batch = oasis_common.load_oasis_case(rank=9, device=device, oasis_root=Path(args.oasis_root), pair_file=Path(args.pair_file))
    load_args = SimpleNamespace(sitreg_root=sitreg_mod.DEFAULT_SITREG_ROOT)
    target, audit = sitreg_mod.load_sitreg_target(
        load_args,
        fixed_id=str(batch["fixed_id"]),
        moving_id=str(batch["moving_id"]),
        device=device,
    )
    target_np = target.cpu().numpy().reshape(-1, 3)[flat_indices]
    return scale_norm_to_vox_xyz(target_np, target.shape[:3]), audit


def load_lung_teacher(args: argparse.Namespace, case_id: int, flat_indices: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    lung_mod = load_module("update_alignment_lung_ptvreg", LUNG_DIR / "run_case01_ptvreg_field_fit_one.py", LUNG_DIR)
    lung_common = lung_mod.c
    case = lung_common.load_case(Path(args.ptvprep_root), int(case_id))
    expected = {
        1: (0.7671410824, 0.8917488134),
        8: (1.0142198317, 1.2922227355),
    }
    exp_tre, exp_std = expected[int(case_id)]
    load_args = SimpleNamespace(
        ptvreg_mat=Path(args.ptvreg_root) / f"case{int(case_id):02d}_T50_to_T00_deformation.mat",
        expected_tre_phys=exp_tre,
        expected_tre_std=exp_std,
        case_id=int(case_id),
    )
    target_np, audit = lung_mod.load_ptvreg_target(load_args, case)
    sampled = target_np.reshape(-1, 3)[flat_indices]
    return scale_norm_to_vox_xyz(sampled, target_np.shape[:3]), audit


def cosine_flat(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    av = a.reshape(-1).astype(np.float64, copy=False)
    bv = b.reshape(-1).astype(np.float64, copy=False)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= eps:
        return float("nan")
    return float(np.dot(av, bv) / (denom + eps))


def rmse_field(a: np.ndarray) -> float:
    return float(math.sqrt(np.mean(np.square(a.astype(np.float64, copy=False)))))


def nested_get(payload: Dict[str, object], path: Sequence[str], default: object = None) -> object:
    cur: object = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def aggregate(args: argparse.Namespace, *, smoke: bool = False) -> Dict[str, object]:
    root = Path(args.out_root) / ("smoke" if smoke else "full")
    review_root = Path(args.review_root) / ("smoke" if smoke else "full")
    tables_dir = review_root / "tables"
    plots_dir = review_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root / "job_manifest.json")
    jobs = [Job(**row) for row in manifest["jobs"]]  # type: ignore[index]
    alignment_rows: List[Dict[str, object]] = []
    case_rows: List[Dict[str, object]] = []
    teacher_audits: Dict[str, object] = {}
    cumulative_updates: Dict[Tuple[str, str, str], np.ndarray] = {}
    all_complete = True
    errors: List[str] = []
    for job in jobs:
        run_dir = Path(job.run_dir)
        sample_path = run_dir / "checkpoint_samples.npz"
        summary_path = run_dir / "summary.json"
        if not sample_path.exists() or not summary_path.exists():
            all_complete = False
            errors.append(f"missing outputs for {job.run_name}")
            continue
        samples = np.load(sample_path)
        flat_indices = samples["flat_indices"].astype(np.int64)
        epochs = samples["epochs"].astype(np.int64)
        disp = samples["disp_vox_xyz"].astype(np.float32)
        if job.dataset == "oasis":
            teacher, audit = load_oasis_teacher(args, flat_indices)
        else:
            case_id = int(job.case_key.split("case")[1])
            teacher, audit = load_lung_teacher(args, case_id, flat_indices)
        teacher_audits[job.case_key] = audit
        summary = read_json(summary_path)
        if not isinstance(summary, dict):
            errors.append(f"summary is not a dict for {job.run_name}")
            continue
        early_indices: List[int] = []
        for i in range(len(epochs) - 1):
            cur = disp[i]
            nxt = disp[i + 1]
            delta = nxt - cur
            residual = teacher - cur
            before = rmse_field(residual)
            after = rmse_field(teacher - nxt)
            row = {
                "dataset": job.dataset,
                "case_key": job.case_key,
                "method": job.method,
                "method_key": job.method_key,
                "run_name": job.run_name,
                "epoch": int(epochs[i]),
                "next_epoch": int(epochs[i + 1]),
                "update_alignment": cosine_flat(delta, residual),
                "update_norm": float(np.linalg.norm(delta.reshape(-1))),
                "teacher_residual_norm": float(np.linalg.norm(residual.reshape(-1))),
                "teacher_distance_before_rmse_vox": before,
                "teacher_distance_after_rmse_vox": after,
                "teacher_distance_delta_rmse_vox": after - before,
            }
            alignment_rows.append(row)
            if int(epochs[i + 1]) <= 250:
                early_indices.append(i + 1)
        if early_indices:
            early_end = max(early_indices)
        else:
            early_end = min(len(epochs) - 1, 1)
        cumulative = disp[early_end] - disp[0]
        initial_residual = teacher - disp[0]
        cumulative_updates[(job.dataset, job.case_key, job.method)] = cumulative
        method_alignments = [
            float(row["update_alignment"])
            for row in alignment_rows
            if row["run_name"] == job.run_name and int(row["next_epoch"]) <= 250 and np.isfinite(float(row["update_alignment"]))
        ]
        if job.dataset == "oasis":
            final_metric_name = "dice_mean"
            final_metric_value = float(summary.get("dice_mean", float("nan")))
            final_metric_direction = "higher_is_better"
        else:
            final_metric_name = "reg1mm_snap_tre_mean_mm"
            final_metric_value = float(nested_get(summary, ["tre_snap_to_voxel", "tre_mean_mm"], float("nan")))
            final_metric_direction = "lower_is_better"
        case_rows.append(
            {
                "dataset": job.dataset,
                "case_key": job.case_key,
                "method": job.method,
                "method_key": job.method_key,
                "run_name": job.run_name,
                "num_checkpoints": int(len(epochs)),
                "early_until_epoch": int(epochs[early_end]),
                "early_alignment_mean": float(np.mean(method_alignments)) if method_alignments else float("nan"),
                "early_alignment_median": float(np.median(method_alignments)) if method_alignments else float("nan"),
                "cumulative_early_alignment": cosine_flat(cumulative, initial_residual),
                "final_teacher_distance_rmse_vox": rmse_field(teacher - disp[-1]),
                "final_registration_metric_name": final_metric_name,
                "final_registration_metric_value": final_metric_value,
                "final_registration_metric_direction": final_metric_direction,
                "summary_path": str(summary_path),
                "sample_path": str(sample_path),
            }
        )
    sim_rows: List[Dict[str, object]] = []
    for dataset, case_key in sorted({(row["dataset"], row["case_key"]) for row in case_rows}):
        if dataset == "oasis":
            sinr_method = "sinr_cps4_be_bspline_analytic"
            cp_method = "direct_cp_cps4_be_bspline_analytic"
        else:
            sinr_method = "sinr_cps4_ptv_lcc_isotv"
            cp_method = "direct_cp_cps4_ptv_lcc_isotv"
        key_s = (str(dataset), str(case_key), sinr_method)
        key_c = (str(dataset), str(case_key), cp_method)
        if key_s in cumulative_updates and key_c in cumulative_updates:
            sim_rows.append(
                {
                    "dataset": dataset,
                    "case_key": case_key,
                    "sinr_method": sinr_method,
                    "directcp_method": cp_method,
                    "cumulative_early_update_similarity": cosine_flat(cumulative_updates[key_s], cumulative_updates[key_c]),
                }
            )
    finite_sims = [
        float(row["cumulative_early_update_similarity"])
        for row in sim_rows
        if np.isfinite(float(row["cumulative_early_update_similarity"]))
    ]
    promising_count = sum(v >= PROMISING_SIMILARITY_THRESHOLD for v in finite_sims)
    followup_recommended = bool(all_complete and not errors and promising_count >= 2)
    write_csv(tables_dir / "checkpoint_alignment_rows.csv", alignment_rows)
    write_csv(tables_dir / "case_summary_rows.csv", case_rows)
    write_csv(tables_dir / "sinr_directcp_similarity_rows.csv", sim_rows)
    write_json(tables_dir / "teacher_audits.json", teacher_audits)
    result = {
        "status": "completed" if all_complete and not errors else "incomplete",
        "all_complete": all_complete,
        "errors": errors,
        "job_count": len(jobs),
        "alignment_rows": len(alignment_rows),
        "case_rows": len(case_rows),
        "similarity_rows": len(sim_rows),
        "promising_similarity_threshold": PROMISING_SIMILARITY_THRESHOLD,
        "promising_similarity_count": promising_count,
        "followup_recommended": followup_recommended,
        "tables_dir": str(tables_dir),
        "plots_dir": str(plots_dir),
    }
    write_json(review_root / "aggregation_summary.json", result)
    write_report(review_root, result, case_rows, sim_rows)
    make_plots(plots_dir, alignment_rows, case_rows, sim_rows)
    return result


def write_report(
    review_root: Path,
    result: Dict[str, object],
    case_rows: Sequence[Dict[str, object]],
    sim_rows: Sequence[Dict[str, object]],
) -> None:
    lines = [
        "# Update-Alignment Overnight Report",
        "",
        f"Status: `{result['status']}`",
        f"Follow-up recommended: `{result['followup_recommended']}`",
        "",
        "## Case Summary",
        "",
        "| Dataset | Case | Method | Early alignment | Cumulative early alignment | Final metric |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in case_rows:
        lines.append(
            "| {dataset} | {case_key} | {method} | {early:.4f} | {cum:.4f} | {metric:.4f} {metric_name} |".format(
                dataset=row["dataset"],
                case_key=row["case_key"],
                method=row["method"],
                early=float(row["early_alignment_mean"]),
                cum=float(row["cumulative_early_alignment"]),
                metric=float(row["final_registration_metric_value"]),
                metric_name=row["final_registration_metric_name"],
            )
        )
    lines.extend(["", "## SINR vs DirectCP Similarity", "", "| Dataset | Case | Similarity |", "|---|---|---:|"])
    for row in sim_rows:
        lines.append(
            f"| {row['dataset']} | {row['case_key']} | {float(row['cumulative_early_update_similarity']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- OASIS alignment uses fixed raw image-positive sampled foreground.",
            "- LungCT alignment uses fixed T00 lung-mask samples on the canonical pTV 1mm crop.",
            "- LungCT final registration metrics from the current runner are reg1mm crop TRE diagnostics; original-native pTVReg snap TRE is validated for the teacher field.",
            "- These selected-case diagnostics are not a full population claim.",
        ]
    )
    (review_root / "update_alignment_report.md").write_text("\n".join(lines) + "\n")


def make_plots(
    plots_dir: Path,
    alignment_rows: Sequence[Dict[str, object]],
    case_rows: Sequence[Dict[str, object]],
    sim_rows: Sequence[Dict[str, object]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(plots_dir / "plot_error.json", {"error": str(exc)})
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    by_case: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in alignment_rows:
        by_case.setdefault((str(row["dataset"]), str(row["case_key"])), []).append(row)
    fig, axes = plt.subplots(1, max(1, len(by_case)), figsize=(5 * max(1, len(by_case)), 4), squeeze=False)
    for ax, ((dataset, case_key), rows) in zip(axes[0], sorted(by_case.items())):
        methods = sorted({str(row["method"]) for row in rows})
        for method in methods:
            sub = [row for row in rows if str(row["method"]) == method]
            sub = sorted(sub, key=lambda row: int(row["next_epoch"]))
            ax.plot([int(row["next_epoch"]) for row in sub], [float(row["update_alignment"]) for row in sub], marker="o", label=method)
        ax.set_title(f"{dataset} {case_key}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("update alignment")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "brain_lung_alignment_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f"{row['case_key']}\n{row['method']}" for row in case_rows]
    values = [float(row["early_alignment_mean"]) for row in case_rows]
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("early alignment mean")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "early_alignment_summary.png", dpi=180)
    plt.close(fig)

    if sim_rows:
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = [str(row["case_key"]) for row in sim_rows]
        values = [float(row["cumulative_early_update_similarity"]) for row in sim_rows]
        ax.bar(range(len(values)), values)
        ax.axhline(PROMISING_SIMILARITY_THRESHOLD, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("SINR/DirectCP cumulative similarity")
        ax.set_ylim(-1.05, 1.05)
        fig.tight_layout()
        fig.savefig(plots_dir / "sinr_directcp_update_similarity.png", dpi=180)
        plt.close(fig)


def maybe_start_followup(args: argparse.Namespace, aggregation: Dict[str, object]) -> Optional[Dict[str, object]]:
    if not args.auto_followup:
        return None
    if not bool(aggregation.get("followup_recommended")):
        return {"started": False, "reason": "promising gate not met"}
    if not FOLLOWUP_SCRIPT.exists():
        return {"started": False, "reason": f"follow-up script missing: {FOLLOWUP_SCRIPT}"}
    cmd = [
        sys.executable,
        str(FOLLOWUP_SCRIPT),
        "--scope",
        "first3",
        "--gpus",
        str(args.gpus),
        "--run-full",
        "--resume",
        "--out-root",
        "/database/inrorspline/runs/sinr_cp_kernel_followup_20260629_v1",
        "--review-root",
        str(REPO_ROOT / "experiments/2026-06-29_sinr_cp_kernel_followup"),
    ]
    log_path = Path(args.out_root) / "followup_launch.log"
    with log_path.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return {"started": True, "pid": proc.pid, "command": cmd, "log_path": str(log_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("first3",), default="first3")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=DEFAULT_PAIR_FILE)
    parser.add_argument("--ptvprep-root", type=Path, default=DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--ptvreg-root", type=Path, default=DEFAULT_PTVREG_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--stage-epochs", type=str, default="200,400,800,1000")
    parser.add_argument("--checkpoint-sample-count", type=int, default=100000)
    parser.add_argument("--smoke-sample-count", type=int, default=2000)
    parser.add_argument("--checkpoint-sample-seed", type=int, default=1729)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--auto-followup", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.dry_run or args.smoke or args.run_full or args.aggregate_only):
        raise SystemExit("Choose one of --dry-run, --smoke, --run-full, or --aggregate-only.")
    if args.run_full:
        args.smoke = False
    jobs = build_jobs(args)
    write_manifest(args, jobs)
    if args.aggregate_only:
        aggregation = aggregate(args, smoke=bool(args.smoke))
        print(json.dumps({"aggregation": aggregation}, indent=2, allow_nan=True), flush=True)
        followup = maybe_start_followup(args, aggregation)
        if followup is not None:
            print(json.dumps({"followup": followup}, indent=2, default=str), flush=True)
        return
    launch_result = launch_jobs(args, jobs)
    print(json.dumps({"launch_result": launch_result}, indent=2, allow_nan=True), flush=True)
    if args.dry_run:
        return
    aggregation = aggregate(args, smoke=bool(args.smoke))
    print(json.dumps({"aggregation": aggregation}, indent=2, allow_nan=True), flush=True)
    followup = maybe_start_followup(args, aggregation)
    if followup is not None:
        write_json(phase_root(args) / "followup_status.json", followup)
        print(json.dumps({"followup": followup}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
