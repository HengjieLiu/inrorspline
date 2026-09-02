#!/usr/bin/env python
"""Run the six MR-D-BSCP schedules on the canonical 1-mm LungCT cohorts."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import common as c
import run_one


EXPERIMENT_ID = "2026-09-01_lungct_mrdbscp_acceleration_best1mm"
METHOD = run_one.METHOD_DIRECT_CP_32_PTV
VARIANT_IDS = (
    "original_corrected",
    "frontloaded_fixed",
    "oasis_bal_decay_cumulative",
    "oasis_fast_decay_cumulative",
    "oasis_bal_decay_ordinal",
    "oasis_fast_decay_ordinal",
)
DATASETS = ("4dct", "copd")
CASES = tuple(range(1, 11))
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/2026-09-01_lungct_mrdbscp_acceleration_best1mm_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-09-01_lungct_mrdbscp_acceleration_best1mm"


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    stage_full_cps: tuple[int, ...]
    stage_downsample: tuple[int, ...]
    stage_epochs: tuple[int, ...]
    stage_lr: tuple[float, ...]
    early_stop_mode: str
    min_epochs: tuple[int, ...]
    patience: tuple[int, ...]
    total_thresholds: tuple[float, ...]
    data_thresholds: tuple[float, ...]
    projection_rule: str
    description: str


def _five_stage_specs() -> Dict[str, Dict[str, object]]:
    balanced_min = (120, 90, 60, 60, 60)
    balanced_patience = (50, 38, 25, 25, 25)
    fast_min = (80, 60, 40, 40, 40)
    fast_patience = (40, 26, 25, 25, 25)
    balanced_thresholds = (5e-5,) * 5
    balanced_data = (2e-5,) * 5
    fast_thresholds = (1e-4,) * 5
    fast_data = (5e-5,) * 5
    return {
        "original_corrected": {
            "stage_full_cps": (64, 32, 16, 8, 4),
            "stage_downsample": (16, 8, 4, 2, 1),
            "stage_epochs": (400, 400, 400, 400, 900),
            "stage_lr": (1e-3,) * 5,
            "early_stop_mode": "disabled",
            "min_epochs": (0,) * 5,
            "patience": (0,) * 5,
            "total_thresholds": (0.0,) * 5,
            "data_thresholds": (0.0,) * 5,
            "description": "Original corrected 2500-epoch schedule.",
        },
        "frontloaded_fixed": {
            "stage_full_cps": (64, 32, 16, 8, 4),
            "stage_downsample": (16, 8, 4, 2, 1),
            "stage_epochs": (800, 600, 400, 400, 400),
            "stage_lr": (1e-3,) * 5,
            "early_stop_mode": "disabled",
            "min_epochs": (0,) * 5,
            "patience": (0,) * 5,
            "total_thresholds": (0.0,) * 5,
            "data_thresholds": (0.0,) * 5,
            "description": "Front-loaded fixed 2600-cap schedule.",
        },
        "oasis_bal_decay_cumulative": {
            "stage_full_cps": (64, 32, 16, 8, 4),
            "stage_downsample": (16, 8, 4, 2, 1),
            "stage_epochs": (800, 600, 400, 400, 400),
            "stage_lr": (0.03, 0.01, 0.003, 0.003, 0.003),
            "early_stop_mode": "adaptive",
            "min_epochs": balanced_min,
            "patience": balanced_patience,
            "total_thresholds": balanced_thresholds,
            "data_thresholds": balanced_data,
            "description": "Balanced dual-median adaptive stopping with cumulative-work LRs.",
        },
        "oasis_fast_decay_cumulative": {
            "stage_full_cps": (64, 32, 16, 8, 4),
            "stage_downsample": (16, 8, 4, 2, 1),
            "stage_epochs": (800, 600, 400, 400, 400),
            "stage_lr": (0.03, 0.01, 0.003, 0.003, 0.003),
            "early_stop_mode": "adaptive",
            "min_epochs": fast_min,
            "patience": fast_patience,
            "total_thresholds": fast_thresholds,
            "data_thresholds": fast_data,
            "description": "Fast dual-median adaptive stopping with cumulative-work LRs.",
        },
        "oasis_bal_decay_ordinal": {
            "stage_full_cps": (64, 32, 16, 8, 4),
            "stage_downsample": (16, 8, 4, 2, 1),
            "stage_epochs": (800, 600, 400, 400, 400),
            "stage_lr": (0.03, 0.03, 0.01, 0.01, 0.003),
            "early_stop_mode": "adaptive",
            "min_epochs": balanced_min,
            "patience": balanced_patience,
            "total_thresholds": balanced_thresholds,
            "data_thresholds": balanced_data,
            "description": "Balanced dual-median adaptive stopping with ordinal-band LRs.",
        },
        "oasis_fast_decay_ordinal": {
            "stage_full_cps": (64, 32, 16, 8, 4),
            "stage_downsample": (16, 8, 4, 2, 1),
            "stage_epochs": (800, 600, 400, 400, 400),
            "stage_lr": (0.03, 0.03, 0.01, 0.01, 0.003),
            "early_stop_mode": "adaptive",
            "min_epochs": fast_min,
            "patience": fast_patience,
            "total_thresholds": fast_thresholds,
            "data_thresholds": fast_data,
            "description": "Fast dual-median adaptive stopping with ordinal-band LRs.",
        },
    }


def variant_spec(dataset: str, variant_id: str) -> VariantSpec:
    dataset = str(dataset).lower()
    variant_id = str(variant_id)
    base = _five_stage_specs()[variant_id]
    if dataset == "copd":
        return VariantSpec(
            variant_id=variant_id,
            stage_full_cps=tuple(base["stage_full_cps"]),
            stage_downsample=tuple(base["stage_downsample"]),
            stage_epochs=tuple(base["stage_epochs"]),
            stage_lr=tuple(base["stage_lr"]),
            early_stop_mode=str(base["early_stop_mode"]),
            min_epochs=tuple(base["min_epochs"]),
            patience=tuple(base["patience"]),
            total_thresholds=tuple(base["total_thresholds"]),
            data_thresholds=tuple(base["data_thresholds"]),
            projection_rule="copd_exact_l2b_five_stage_schedule",
            description=str(base["description"]),
        )
    if dataset != "4dct":
        raise ValueError(f"Unknown dataset {dataset!r}")
    if variant_id == "original_corrected":
        return VariantSpec(
            variant_id=variant_id,
            stage_full_cps=(32, 16, 8, 4),
            stage_downsample=(8, 4, 2, 1),
            stage_epochs=(400, 400, 800, 900),
            stage_lr=(1e-3,) * 4,
            early_stop_mode="disabled",
            min_epochs=(0,) * 4,
            patience=(0,) * 4,
            total_thresholds=(0.0,) * 4,
            data_thresholds=(0.0,) * 4,
            projection_rule="exact_existing_best_4dct_2500_anchor",
            description="Fresh rerun of the existing best 4DCT 2500-epoch anchor.",
        )
    # The 64- and 32-control stages collapse into the existing 4DCT 32-control
    # stage. Epoch/minimum work is additive; the finer (32-control) LR is kept;
    # patience is conservative via max().
    return VariantSpec(
        variant_id=variant_id,
        stage_full_cps=(32, 16, 8, 4),
        stage_downsample=(8, 4, 2, 1),
        stage_epochs=(int(base["stage_epochs"][0]) + int(base["stage_epochs"][1]), *tuple(base["stage_epochs"][2:])),
        stage_lr=(float(base["stage_lr"][1]), *tuple(base["stage_lr"][2:])),
        early_stop_mode=str(base["early_stop_mode"]),
        min_epochs=(int(base["min_epochs"][0]) + int(base["min_epochs"][1]), *tuple(base["min_epochs"][2:])),
        patience=(max(int(base["patience"][0]), int(base["patience"][1])), *tuple(base["patience"][2:])),
        total_thresholds=(float(base["total_thresholds"][1]), *tuple(base["total_thresholds"][2:])),
        data_thresholds=(float(base["data_thresholds"][1]), *tuple(base["data_thresholds"][2:])),
        projection_rule="collapse_l2b_64_and_32_into_4dct_32;sum_caps_and_minimums;max_patience;retain_32_lr",
        description=f"4DCT four-stage projection of {base['description']}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("4dct", "copd", "both"), default="both")
    parser.add_argument("--cases", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--data-root", type=Path, default=c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ptv-lcc-sigma-mm", type=float, default=2.1)
    parser.add_argument("--ptv-isotv-weight", type=float, default=0.055)
    parser.add_argument("--ptv-csqrt", type=float, default=5e-3)
    parser.add_argument("--border-mask-voxels", type=int, default=5)
    parser.add_argument("--phase", choices=("full", "smoke"), default="full")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-dataset", choices=DATASETS, default="")
    parser.add_argument("--worker-variant", choices=VARIANT_IDS, default="")
    parser.add_argument("--worker-gpu", type=str, default="")
    return parser.parse_args()


def selected_datasets(value: str) -> Sequence[str]:
    return DATASETS if str(value) == "both" else (str(value),)


def csv_list(values: Iterable[object]) -> str:
    return ",".join(str(value) for value in values)


def worker_argv(
    args: argparse.Namespace,
    dataset: str,
    variant: VariantSpec,
    case_id: int,
    sequence_index: int,
    gpu: str,
) -> List[str]:
    smoke = str(args.phase) == "smoke"
    stage_epochs = (1,) * len(variant.stage_epochs) if smoke else variant.stage_epochs
    early_stop_mode = "disabled" if smoke else variant.early_stop_mode
    condition = "cold" if int(sequence_index) == 0 else "warm"
    run_name = f"{EXPERIMENT_ID}_{dataset}_{variant.variant_id}_case{int(case_id):02d}"
    argv = [
        "run_one.py",
        "--dataset", dataset,
        "--method", METHOD,
        "--case-id", str(int(case_id)),
        "--data-root", str(args.data_root),
        "--out-root", str(args.out_root),
        "--run-name", run_name,
        "--gpu", str(gpu),
        "--seed", str(int(args.seed)),
        "--eval-interval", "1" if smoke else "250",
        "--lr", "1e-3",
        "--stage-full-cps", csv_list(variant.stage_full_cps),
        "--stage-downsample", csv_list(variant.stage_downsample),
        "--stage-epochs", csv_list(stage_epochs),
        "--stage-lr", csv_list(variant.stage_lr),
        "--ptv-lcc-sigma-mm", str(float(args.ptv_lcc_sigma_mm)),
        "--ptv-isotv-weight", str(float(args.ptv_isotv_weight)),
        "--ptv-csqrt", str(float(args.ptv_csqrt)),
        "--border-mask-voxels", str(int(args.border_mask_voxels)),
        "--timing-mode", "input_to_dvf",
        "--timing-condition", condition,
        "--timing-sequence-index", str(int(sequence_index)),
        "--timing-worker-id", f"{dataset}_{variant.variant_id}_gpu{gpu}",
        "--variant-id", variant.variant_id,
        "--no-prolongation-diagnostics",
        "--save-dense-dvf",
    ]
    if early_stop_mode == "adaptive":
        argv.extend(
            [
                "--stage-early-stop-mode", "adaptive",
                "--stage-min-epochs", csv_list(variant.min_epochs),
                "--stage-patience", csv_list(variant.patience),
                "--stage-check-interval", "25",
                "--stage-window", "25",
                "--stage-total-thresholds", csv_list(variant.total_thresholds),
                "--stage-data-thresholds", csv_list(variant.data_thresholds),
            ]
        )
    if args.force:
        argv.append("--overwrite")
    return argv


def expected_jobs(args: argparse.Namespace) -> List[Dict[str, object]]:
    cases = c.parse_int_list(args.cases)
    gpus = [str(value) for value in c.parse_int_list(args.gpus)]
    rows: List[Dict[str, object]] = []
    cohort_index = 0
    for dataset in selected_datasets(args.dataset):
        for variant_id in VARIANT_IDS:
            variant = variant_spec(dataset, variant_id)
            assigned_gpu = gpus[cohort_index % len(gpus)]
            for sequence_index, case_id in enumerate(cases):
                rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "dataset": dataset,
                        "variant_id": variant_id,
                        "case_id": int(case_id),
                        "case_sequence_index": int(sequence_index),
                        "timing_condition": "cold" if sequence_index == 0 else "warm",
                        "assigned_gpu": assigned_gpu,
                        "cohort_index": cohort_index,
                        "stage_full_cps": csv_list(variant.stage_full_cps),
                        "stage_downsample": csv_list(variant.stage_downsample),
                        "stage_epochs": csv_list(variant.stage_epochs),
                        "stage_lr": csv_list(variant.stage_lr),
                        "early_stop_mode": variant.early_stop_mode,
                        "projection_rule": variant.projection_rule,
                    }
                )
            cohort_index += 1
    return rows


def write_manifest(args: argparse.Namespace, rows: Sequence[Dict[str, object]]) -> None:
    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "job_manifest.csv", rows)
    c.write_json(
        args.review_root / "job_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "immutable": True,
            "requested_case_rows": len(rows),
            "datasets": list(selected_datasets(args.dataset)),
            "cases": c.parse_int_list(args.cases),
            "variants": list(VARIANT_IDS),
            "gpus": [str(value) for value in c.parse_int_list(args.gpus)],
            "timing_endpoint": "loaded input images to saved dense DVF",
            "materialization_policy": "final_stage_only",
            "rows": list(rows),
        },
    )


def run_worker(args: argparse.Namespace) -> int:
    dataset = str(args.worker_dataset)
    variant_id = str(args.worker_variant)
    gpu = str(args.worker_gpu)
    variant = variant_spec(dataset, variant_id)
    cases = c.parse_int_list(args.cases)
    worker_log = args.out_root / "logs" / dataset / f"{variant_id}_gpu{gpu}.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.review_root / f"worker_progress_{dataset}_{variant_id}.csv"
    progress: List[Dict[str, object]] = []
    old_argv = list(sys.argv)
    returncode = 0
    with worker_log.open("w") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
        print(json.dumps({"worker_start": time.strftime("%Y-%m-%d %H:%M:%S %z"), "dataset": dataset, "variant_id": variant_id, "gpu": gpu}), flush=True)
        for sequence_index, case_id in enumerate(cases):
            argv = worker_argv(args, dataset, variant, int(case_id), int(sequence_index), gpu)
            started = time.perf_counter()
            case_result: Dict[str, object] = {
                "dataset": dataset,
                "variant_id": variant_id,
                "case_id": int(case_id),
                "case_sequence_index": int(sequence_index),
                "timing_condition": "cold" if sequence_index == 0 else "warm",
                "gpu": gpu,
                "worker_pid": os.getpid(),
                "worker_log": str(worker_log),
                "command": json.dumps(argv),
            }
            try:
                sys.argv = argv
                run_one.main()
                run_dir = args.out_root / METHOD / f"{EXPERIMENT_ID}_{dataset}_{variant_id}_case{int(case_id):02d}"
                summary_path = run_dir / "summary.json"
                summary = c.read_json(summary_path)
                if not isinstance(summary, dict):
                    raise RuntimeError(f"Invalid summary payload: {summary_path}")
                timing = summary.get("timing", {})
                if not isinstance(timing, dict):
                    raise RuntimeError(f"Missing timing payload: {summary_path}")
                case_result.update(
                    {
                        "status": str(summary.get("status", "completed")),
                        "returncode": 0,
                        "run_dir": str(run_dir),
                        "summary_path": str(summary_path),
                        "timing_path": str(run_dir / "timing.json"),
                        "worker_case_elapsed_sec": float(time.perf_counter() - started),
                        "loaded_images_to_dvf_file_sec": timing.get("loaded_images_to_dvf_file_sec"),
                        "loaded_images_to_dvf_in_memory_sec": timing.get("loaded_images_to_dvf_in_memory_sec"),
                        "full_script_to_dvf_file_sec": timing.get("full_script_to_dvf_file_sec"),
                        "optimizer_elapsed_sec": timing.get("optimizer_elapsed_sec"),
                        "full_case_elapsed_sec": timing.get("full_case_elapsed_sec"),
                        "native_crop_tre_snap_mean_mm": summary.get("native_crop_tre_snap_to_voxel", {}).get("tre_mean_mm"),
                        "native_crop_tre_subvoxel_mean_mm": summary.get("native_crop_tre_subvoxel", {}).get("tre_mean_mm"),
                        "full_grid_materialization_trace": summary.get("full_grid_materialization_trace", summary.get("training", {}).get("full_grid_materialization_trace")),
                    }
                )
            except BaseException as exc:
                returncode = 1
                case_result.update(
                    {
                        "status": "failed",
                        "returncode": 1,
                        "worker_case_elapsed_sec": float(time.perf_counter() - started),
                        "error": repr(exc),
                    }
                )
                traceback.print_exc()
            finally:
                sys.argv = old_argv
            progress.append(case_result)
            c.write_csv(progress_path, progress)
            print(json.dumps(case_result, allow_nan=True), flush=True)
            if returncode != 0:
                break
    c.write_json(
        args.review_root / f"worker_summary_{dataset}_{variant_id}.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "dataset": dataset,
            "variant_id": variant_id,
            "gpu": gpu,
            "pid": os.getpid(),
            "case_count": len(progress),
            "completed_count": sum(int(row.get("returncode", 1)) == 0 for row in progress),
            "returncode": returncode,
            "progress_path": str(progress_path),
        },
    )
    return returncode


def launch_workers(args: argparse.Namespace, rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    gpus = [str(value) for value in c.parse_int_list(args.gpus)]
    cohorts: List[Dict[str, object]] = []
    for dataset in selected_datasets(args.dataset):
        for variant_id in VARIANT_IDS:
            cohort_rows = [row for row in rows if row["dataset"] == dataset and row["variant_id"] == variant_id]
            cohorts.append({"dataset": dataset, "variant_id": variant_id, "rows": cohort_rows})
    pending = list(cohorts)
    active: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    while pending or active:
        used_gpus = {str(item["gpu"]) for item in active}
        while pending and len(active) < len(gpus):
            free_gpus = [gpu_id for gpu_id in gpus if gpu_id not in used_gpus]
            if not free_gpus:
                break
            gpu = free_gpus[0]
            used_gpus.add(gpu)
            cohort = pending.pop(0)
            dataset = str(cohort["dataset"])
            variant_id = str(cohort["variant_id"])
            log_path = args.out_root / "logs" / "controller" / f"{dataset}_{variant_id}_gpu{gpu}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dataset", dataset,
                "--cases", str(args.cases),
                "--gpus", str(args.gpus),
                "--data-root", str(args.data_root),
                "--out-root", str(args.out_root),
                "--review-root", str(args.review_root),
                "--seed", str(int(args.seed)),
                "--ptv-lcc-sigma-mm", str(float(args.ptv_lcc_sigma_mm)),
                "--ptv-isotv-weight", str(float(args.ptv_isotv_weight)),
                "--ptv-csqrt", str(float(args.ptv_csqrt)),
                "--border-mask-voxels", str(int(args.border_mask_voxels)),
                "--phase", str(args.phase),
                "--worker-dataset", dataset,
                "--worker-variant", variant_id,
                "--worker-gpu", gpu,
            ]
            if args.force:
                cmd.append("--force")
            log_f = log_path.open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                cmd,
                cwd=c.REPO_ROOT,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append({"proc": proc, "log_f": log_f, "gpu": gpu, "dataset": dataset, "variant_id": variant_id, "log_path": str(log_path), "started": time.perf_counter()})
            print(json.dumps({"worker_launched": True, "dataset": dataset, "variant_id": variant_id, "gpu": gpu, "pid": proc.pid, "log": str(log_path)}), flush=True)
        time.sleep(5)
        remaining: List[Dict[str, object]] = []
        for item in active:
            proc = item["proc"]
            code = proc.poll()
            if code is None:
                remaining.append(item)
                continue
            item["log_f"].close()
            result = {
                "dataset": item["dataset"],
                "variant_id": item["variant_id"],
                "gpu": item["gpu"],
                "returncode": int(code),
                "worker_elapsed_sec": float(time.perf_counter() - float(item["started"])),
                "log": item["log_path"],
            }
            print(json.dumps({"worker_finished": result}, allow_nan=True), flush=True)
            if int(code) != 0:
                failures.append(result)
        active = remaining
    c.write_json(args.review_root / "controller_summary.json", {"experiment_id": EXPERIMENT_ID, "failures": failures, "status": "completed" if not failures else "failed"})
    return failures


def main() -> None:
    args = parse_args()
    if args.worker_dataset and args.worker_variant:
        raise SystemExit(run_worker(args))
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = expected_jobs(args)
    write_manifest(args, rows)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "requested_case_rows": len(rows),
        "requested_cohorts": len(selected_datasets(args.dataset)) * len(VARIANT_IDS),
        "datasets": list(selected_datasets(args.dataset)),
        "cases": c.parse_int_list(args.cases),
        "variants": list(VARIANT_IDS),
        "gpus": [str(value) for value in c.parse_int_list(args.gpus)],
        "out_root": str(args.out_root),
        "review_root": str(args.review_root),
        "phase": str(args.phase),
        "timing_endpoint": "loaded input images to saved dense DVF",
        "materialization_policy": "final_stage_only",
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        for row in rows:
            print(json.dumps({"dry_run_job": row}), flush=True)
        return
    failures = launch_workers(args, rows)
    if failures:
        raise SystemExit(json.dumps({"failed_workers": failures}, indent=2))
    if str(args.phase) == "full":
        aggregator = Path(__file__).with_name("aggregate_mrdbscp_acceleration_timing.py")
        result = subprocess.run(
            [
                sys.executable,
                str(aggregator),
                "--run-root", str(args.out_root),
                "--review-root", str(args.review_root),
                "--data-root", str(args.data_root),
            ],
            cwd=c.REPO_ROOT,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
