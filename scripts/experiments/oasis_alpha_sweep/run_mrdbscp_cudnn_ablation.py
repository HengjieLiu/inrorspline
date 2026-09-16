#!/usr/bin/env python
"""Run and aggregate the focused MR-D-BSCP cuDNN cold/warm ablation."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import artifact_paths  # noqa: E402
import common as c  # noqa: E402
import launch_oasis_unified_acceleration_sweep as unified  # noqa: E402


VARIANT = "oasis_seed3_first5_mrdbscp_cudnn_ablation_v1"
CANDIDATE_NAME = "MR_A_fast_decay"
METHOD = unified.MR_METHOD
EXPERIMENT_KEY = "2026-09-01_oasis_mrdbscp_cudnn_ablation"
RUN_KEY = "oasis_mrdbscp_cudnn_ablation_20260901_v1"


@dataclass(frozen=True)
class Config:
    slug: str
    label: str
    benchmark: bool
    deterministic: bool


CONFIGS: Sequence[Config] = (
    Config("benchmark_true_deterministic_false", "benchmark=True, deterministic=False", True, False),
    Config("benchmark_false_deterministic_false", "benchmark=False, deterministic=False", False, False),
    Config("benchmark_true_deterministic_true", "benchmark=True, deterministic=True", True, True),
    Config("benchmark_false_deterministic_true", "benchmark=False, deterministic=True", False, True),
)
CONFIG_BY_SLUG = {config.slug: config for config in CONFIGS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", choices=sorted(CONFIG_BY_SLUG), default="")
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--case-ranks", type=str, default="0-4")
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--timing-detail", choices=["deep"], default="deep")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dice", action="store_true")
    return parser.parse_args()


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
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


def parse_case_ranks(text: str) -> List[int]:
    ranks: List[int] = []
    for token in str(text).replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            ranks.extend(range(int(start), int(end) + 1))
        else:
            ranks.append(int(token))
    if not ranks or ranks != sorted(set(ranks)):
        raise ValueError(f"Invalid case ranks: {text!r}")
    return ranks


def config_out_root(args: argparse.Namespace, config: Config) -> Path:
    return Path(args.out_root) / config.slug


def config_review_root(args: argparse.Namespace, config: Config) -> Path:
    return Path(args.review_root) / config.slug


def runner_namespace(args: argparse.Namespace, config: Config) -> argparse.Namespace:
    return SimpleNamespace(
        alpha=float(args.alpha),
        seed=int(args.seed),
        phase="primary",
        timing_detail="deep",
        variant=str(args.variant),
        out_root=config_out_root(args, config),
        oasis_root=Path(args.oasis_root),
        pair_file=Path(args.pair_file),
        eval_interval=int(args.eval_interval),
        epoch_override=0,
        overwrite=bool(args.overwrite),
    )


def build_run_one_argv(args: argparse.Namespace, config: Config, rank: int) -> List[str]:
    candidate = unified.candidate_map()[CANDIDATE_NAME]
    argv = unified.run_one_argv(runner_namespace(args, config), candidate, int(rank), str(args.gpu))
    argv.extend(
        [
            "--output-mode",
            "dvf_only",
            "--timing-detail",
            "deep",
            "--cudnn-benchmark",
            str(config.benchmark).lower(),
            "--cudnn-deterministic",
            str(config.deterministic).lower(),
        ]
    )
    return argv


def run_dir(args: argparse.Namespace, config: Config, rank: int) -> Path:
    candidate = unified.candidate_map()[CANDIDATE_NAME]
    return config_out_root(args, config) / METHOD / unified.run_name(candidate, int(rank), float(args.alpha), "primary")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str) + "\n")


def event_total(timing: Mapping[str, object], name: str) -> float:
    events = timing.get("events", [])
    if not isinstance(events, list):
        return float("nan")
    values = [
        float(event.get("elapsed_sec", 0.0))
        for event in events
        if isinstance(event, dict) and event.get("name") == name
    ]
    return float(sum(values)) if values else 0.0


def phase_total(timing: Mapping[str, object], name: str) -> float:
    summaries = timing.get("stage_phase_summaries", [])
    if not isinstance(summaries, list):
        return float("nan")
    return float(
        sum(
            float(stage.get("phase_totals_sec", {}).get(name, 0.0))
            for stage in summaries
            if isinstance(stage, dict) and isinstance(stage.get("phase_totals_sec", {}), dict)
        )
    )


def mark_offset(timing: Mapping[str, object], name: str) -> float:
    marks = timing.get("marks", [])
    if not isinstance(marks, list):
        return float("nan")
    for mark in marks:
        if isinstance(mark, dict) and mark.get("name") == name:
            return float(mark["offset_sec"])
    return float("nan")


def finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def summarize_case(args: argparse.Namespace, config: Config, rank: int, sequence_index: int) -> Dict[str, object]:
    target = run_dir(args, config, rank)
    summary_path = target / "summary.json"
    timing_path = target / "timing.json"
    row: Dict[str, object] = {
        "config": config.slug,
        "config_label": config.label,
        "cudnn_benchmark": config.benchmark,
        "cudnn_deterministic": config.deterministic,
        "gpu": str(args.gpu),
        "candidate": CANDIDATE_NAME,
        "method": METHOD,
        "case_rank": int(rank),
        "case_sequence_index": int(sequence_index),
        "timing_condition": "cold" if sequence_index == 0 else "warm",
        "run_dir": str(target),
        "summary_path": str(summary_path),
        "timing_path": str(timing_path),
        "status": "missing",
        "done": False,
    }
    if not summary_path.exists() or not timing_path.exists():
        return row
    try:
        summary = json.loads(summary_path.read_text())
        timing = json.loads(timing_path.read_text())
        if not isinstance(summary, dict) or not isinstance(timing, dict):
            raise ValueError("summary.json or timing.json is not an object")
        load_offset = mark_offset(timing, "case_load_complete")
        ready_offset = mark_offset(timing, "dvf_file_ready")
        endpoint = ready_offset - load_offset
        phase_names = ["control_grid", "expand", "generator_forward", "regularizer", "warp_ncc", "backward_step"]
        phase_sum = sum(phase_total(timing, name) for name in phase_names)
        initial_grid = event_total(timing, "training.initial_dense_grid")
        loss_setup = sum(event_total(timing, f"stage_{idx}.loss_volume_setup") for idx in range(3))
        training = event_total(timing, "training")
        dvf_output = event_total(timing, "method_execution.dvf_file_prep") + event_total(
            timing, "method_execution.dvf_file_write"
        )
        row.update(
            {
                "status": "completed" if (target / "DONE").exists() else "incomplete",
                "done": bool((target / "DONE").exists()),
                "fixed_id": summary.get("fixed_id"),
                "moving_id": summary.get("moving_id"),
                "original_case_index": summary.get("original_case_index"),
                "dvf_file_path": summary.get("dvf_file_path"),
                "loaded_images_to_dvf_file_sec": endpoint,
                "full_script_to_dvf_file_sec": ready_offset,
                "setup_device_sec": event_total(timing, "setup.device"),
                "setup_case_load_sec": event_total(timing, "setup.case_load"),
                "training_sec": training,
                "initial_dense_grid_sec": initial_grid,
                "loss_volume_setup_sec": loss_setup,
                "training_phase_sum_sec": phase_sum,
                "training_phase_unaccounted_sec": training - initial_grid - loss_setup - phase_sum,
                "dvf_output_sec": dvf_output,
                "actual_epochs": summary.get("actual_epochs"),
                "training_peak_mem_gb": summary.get("training_peak_mem_gb"),
                "cuda_backend": timing.get("cuda_backend", {}),
            }
        )
        for name in [
            "control_grid",
            "expand",
            "generator_forward",
            "regularizer",
            "warp_ncc",
            "backward_step",
            "logging_early_stop_wall_sec",
        ]:
            row[f"phase_{name}_sec"] = phase_total(timing, name)
    except Exception as exc:
        row["status"] = "invalid"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def thread_env(cpu_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, int(cpu_threads)))
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]:
        env[key] = threads
    return env


def worker(args: argparse.Namespace) -> int:
    config = CONFIG_BY_SLUG[str(args.worker_config)]
    ranks = parse_case_ranks(str(args.case_ranks))
    progress: List[Dict[str, object]] = []
    log_root = config_review_root(args, config) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    worker_log = log_root / "worker.log"
    failure = False
    with worker_log.open("w") as log_handle, contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
        print(json.dumps({"worker_start": asdict(config), "gpu": str(args.gpu), "ranks": ranks}), flush=True)
        for sequence_index, rank in enumerate(ranks):
            argv = build_run_one_argv(args, config, rank)
            start = time.perf_counter()
            error = ""
            status = "completed"
            old_argv = sys.argv[:]
            try:
                sys.argv = argv
                import run_one  # noqa: PLC0415

                run_one.main()
            except Exception:
                status = "failed"
                error = traceback.format_exc()
                print(error, flush=True)
            finally:
                sys.argv = old_argv
            elapsed = time.perf_counter() - start
            row = summarize_case(args, config, rank, sequence_index)
            row.update({"worker_status": status, "worker_wall_sec": elapsed, "error": error[:4000]})
            progress.append(row)
            write_csv(config_review_root(args, config) / "worker_progress.csv", progress)
            print(json.dumps({"case_complete": row}, allow_nan=True, default=str), flush=True)
            gc.collect()
            if status != "completed" or not bool(row.get("done")):
                failure = True
                break
        write_json(
            config_review_root(args, config) / "worker_summary.json",
            {
                "config": asdict(config),
                "gpu": str(args.gpu),
                "case_ranks": ranks,
                "completed_count": sum(bool(row.get("done")) for row in progress),
                "failed": failure,
                "rows": progress,
            },
        )
        print(json.dumps({"worker_done": not failure, "completed_count": len(progress)}), flush=True)
    return 1 if failure else 0


def git_metadata() -> Dict[str, object]:
    def command(argv: Sequence[str]) -> str:
        try:
            return subprocess.check_output(list(argv), cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    status = command(["git", "status", "--short", "--untracked-files=normal"])
    return {
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_branch": command(["git", "branch", "--show-current"]),
        "git_dirty": bool(status),
        "git_status_short": status,
    }


def nvidia_smi(argv: Sequence[str]) -> str:
    return subprocess.check_output(["nvidia-smi", *argv], text=True, stderr=subprocess.STDOUT).strip()


def environment_preflight(args: argparse.Namespace) -> Dict[str, object]:
    gpu_query = nvidia_smi(["--id=2", "--query-gpu=index,name,driver_version,memory.used,memory.total", "--format=csv,noheader,nounits"])
    compute_apps = nvidia_smi(["--id=2", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    if compute_apps and "No running processes found" not in compute_apps:
        raise RuntimeError(f"GPU 2 is not idle; refusing to start the ablation:\n{compute_apps}")
    import torch

    return {
        **git_metadata(),
        "created_at": c.now_text(),
        "gpu_requested": str(args.gpu),
        "gpu_query": gpu_query,
        "compute_apps_gpu2": compute_apps,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "container": "py310sitreg_v1_test1",
        "image": "hengjieliu96/dl_dir:py310sitreg_v1",
    }


def worker_command(args: argparse.Namespace, config: Config) -> List[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-config",
        config.slug,
        "--gpu",
        str(args.gpu),
        "--case-ranks",
        str(args.case_ranks),
        "--alpha",
        f"{float(args.alpha):g}",
        "--seed",
        str(int(args.seed)),
        "--eval-interval",
        str(int(args.eval_interval)),
        "--cpu-threads",
        str(int(args.cpu_threads)),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--variant",
        str(args.variant),
    ] + (["--overwrite"] if args.overwrite else [])


def write_manifest(args: argparse.Namespace, environment: Mapping[str, object]) -> None:
    ranks = parse_case_ranks(str(args.case_ranks))
    payload = {
        "created_at": c.now_text(),
        "variant": str(args.variant),
        "candidate": CANDIDATE_NAME,
        "method": METHOD,
        "gpu": str(args.gpu),
        "case_ranks": ranks,
        "case_sequence": "rank 0 cold; ranks 1-4 warm in one persistent worker",
        "configuration_count": len(CONFIGS),
        "expected_timing_runs": len(CONFIGS) * len(ranks),
        "configs": [asdict(config) for config in CONFIGS],
        "cudnn_settings": {
            "requested_matrix": [asdict(config) for config in CONFIGS],
            "effective_values_recorded_per_run": True,
            "effective_value_sources": ["timing.json.cuda_backend", "summary.json.cuda_backend"],
        },
        "environment": dict(environment),
        "worker_commands": {config.slug: worker_command(args, config) for config in CONFIGS},
    }
    write_json(Path(args.review_root) / "execution_manifest.json", payload)
    write_json(Path(args.out_root) / "execution_manifest.json", payload)


def finite_values(rows: Iterable[Mapping[str, object]], key: str) -> List[float]:
    values = [finite(row.get(key)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def aggregate_timing(args: argparse.Namespace) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    ranks = parse_case_ranks(str(args.case_ranks))
    case_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    phase_names = ["expand", "backward_step", "regularizer", "warp_ncc", "logging_early_stop_wall_sec"]
    for config in CONFIGS:
        rows = [summarize_case(args, config, rank, sequence_index) for sequence_index, rank in enumerate(ranks)]
        case_rows.extend(rows)
        cold = [row for row in rows if row["timing_condition"] == "cold" and row.get("status") == "completed"]
        warm = [row for row in rows if row["timing_condition"] == "warm" and row.get("status") == "completed"]
        cold_endpoint = finite_values(cold, "loaded_images_to_dvf_file_sec")
        warm_endpoint = finite_values(warm, "loaded_images_to_dvf_file_sec")
        row: Dict[str, object] = {
            "config": config.slug,
            "config_label": config.label,
            "cudnn_benchmark": config.benchmark,
            "cudnn_deterministic": config.deterministic,
            "gpu": str(args.gpu),
            "cold_count": len(cold),
            "warm_count": len(warm),
            "cold_endpoint_sec": cold_endpoint[0] if cold_endpoint else float("nan"),
            "warm_endpoint_mean_sec": mean(warm_endpoint) if warm_endpoint else float("nan"),
            "warm_endpoint_sd_sec": stdev(warm_endpoint) if len(warm_endpoint) > 1 else float("nan"),
        }
        row["cold_minus_warm_gap_sec"] = finite(row["cold_endpoint_sec"]) - finite(row["warm_endpoint_mean_sec"])
        aggregate_keys = [
            "setup_device_sec",
            "training_sec",
            "initial_dense_grid_sec",
            "loss_volume_setup_sec",
            "training_phase_sum_sec",
            "training_phase_unaccounted_sec",
            "dvf_output_sec",
            "actual_epochs",
            "training_peak_mem_gb",
        ] + [f"phase_{name}_sec" for name in phase_names]
        for key in aggregate_keys:
            cold_values = finite_values(cold, key)
            warm_values = finite_values(warm, key)
            row[f"cold_{key}"] = cold_values[0] if cold_values else float("nan")
            row[f"warm_{key}_mean"] = mean(warm_values) if warm_values else float("nan")
            row[f"gap_{key}"] = finite(row[f"cold_{key}"]) - finite(row[f"warm_{key}_mean"])
        summary_rows.append(row)

    case_fields = list(case_rows[0].keys()) if case_rows else []
    write_csv(Path(args.review_root) / "timing_case_rows.csv", case_rows, case_fields)
    write_csv(Path(args.review_root) / "timing_summary.csv", summary_rows)
    write_json(Path(args.review_root) / "timing_summary.json", summary_rows)

    phase_rows: List[Dict[str, object]] = []
    for row in summary_rows:
        for name in phase_names:
            phase_rows.append(
                {
                    "config": row["config"],
                    "config_label": row["config_label"],
                    "phase": name,
                    "cold_sec": row.get(f"cold_phase_{name}_sec"),
                    "warm_mean_sec": row.get(f"warm_phase_{name}_sec_mean"),
                    "cold_minus_warm_sec": row.get(f"gap_phase_{name}_sec"),
                }
            )
    write_csv(Path(args.review_root) / "timing_phase_breakdown.csv", phase_rows)
    return case_rows, summary_rows


def load_dice_summary(review_root: Path) -> List[Dict[str, object]]:
    path = Path(review_root) / "dice_summary.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def run_workers(args: argparse.Namespace) -> Dict[str, object]:
    results: List[Dict[str, object]] = []
    env = thread_env(int(args.cpu_threads))
    for config in CONFIGS:
        log_path = Path(args.review_root) / config.slug / "logs" / "controller.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = worker_command(args, config)
        started = time.time()
        with log_path.open("w") as log_handle:
            completed = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env, check=False)
        results.append(
            {
                "config": config.slug,
                "config_label": config.label,
                "returncode": int(completed.returncode),
                "elapsed_sec": time.time() - started,
                "log_path": str(log_path),
            }
        )
    payload = {"completed_at": c.now_text(), "sequential": True, "results": results}
    write_json(Path(args.review_root) / "timing_launch_summary.json", payload)
    write_json(Path(args.out_root) / "timing_launch_summary.json", payload)
    return payload


def write_report(
    args: argparse.Namespace,
    environment: Mapping[str, object],
    summary_rows: Sequence[Mapping[str, object]],
    dice_summary: Sequence[Mapping[str, object]] | None,
) -> None:
    lines = [
        "# MR-D-BSCP cuDNN Cold/Warm Ablation",
        "",
        f"- Candidate: {CANDIDATE_NAME}; protocol: seed-3 shuffled OASIS ranks {args.case_ranks}.",
        f"- GPU: physical GPU {args.gpu}; four configurations were run sequentially.",
        "- Timing endpoint: loaded images resident on the selected device through saved dense DVF .npy; evaluation is excluded.",
        "- Cold/warm convention: rank 0 is cold; ranks 1-4 are warm cases in the same worker.",
        "",
        "## Timing summary",
        "",
        "| Configuration | Cold (s) | Warm mean (s) | Warm SD (s) | Gap (s) | setup.device cold/warm (s) | Expand gap (s) | Backward gap (s) | Dice mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    dice_by_config = {str(row.get("config")): row for row in (dice_summary or [])}
    for row in summary_rows:
        dice = dice_by_config.get(str(row.get("config")), {})
        dice_value = finite(dice.get("dice_mean"))
        dice_text = f"{dice_value:.4f}" if math.isfinite(dice_value) else "n/a"
        lines.append(
            "| {label} | {cold:.2f} | {warm:.2f} | {sd:.2f} | {gap:.2f} | {setup_cold:.2f}/{setup_warm:.2f} | {expand:.2f} | {backward:.2f} | {dice} |".format(
                label=row.get("config_label"),
                cold=finite(row.get("cold_endpoint_sec")),
                warm=finite(row.get("warm_endpoint_mean_sec")),
                sd=finite(row.get("warm_endpoint_sd_sec")),
                gap=finite(row.get("cold_minus_warm_gap_sec")),
                setup_cold=finite(row.get("cold_setup_device_sec")),
                setup_warm=finite(row.get("warm_setup_device_sec_mean")),
                expand=finite(row.get("gap_phase_expand_sec")),
                backward=finite(row.get("gap_phase_backward_step_sec")),
                dice=dice_text,
            )
        )
    lines.extend(
        [
            "",
            "## Detailed timing breakdown",
            "",
            "| Configuration | Loss-volume setup gap (s) | Regularizer gap (s) | Warp/NCC gap (s) | Logging gap (s) | Training residual gap (s) | Optimizer steps cold/warm | Peak memory cold/warm (GB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {label} | {loss:.2f} | {regularizer:.2f} | {warp:.2f} | {logging:.2f} | {residual:.2f} | {steps_cold:.1f}/{steps_warm:.1f} | {mem_cold:.2f}/{mem_warm:.2f} |".format(
                label=row.get("config_label"),
                loss=finite(row.get("gap_loss_volume_setup_sec")),
                regularizer=finite(row.get("gap_phase_regularizer_sec")),
                warp=finite(row.get("gap_phase_warp_ncc_sec")),
                logging=finite(row.get("gap_phase_logging_early_stop_wall_sec_sec")),
                residual=finite(row.get("gap_training_phase_unaccounted_sec")),
                steps_cold=finite(row.get("cold_actual_epochs")),
                steps_warm=finite(row.get("warm_actual_epochs_mean")),
                mem_cold=finite(row.get("cold_training_peak_mem_gb")),
                mem_warm=finite(row.get("warm_training_peak_mem_gb_mean")),
            )
        )
    by_config = {str(row.get("config")): row for row in summary_rows}
    comparison_lines: List[str] = []
    ref = by_config.get("benchmark_true_deterministic_false")
    no_benchmark = by_config.get("benchmark_false_deterministic_false")
    if ref is not None and no_benchmark is not None:
        ref_gap = finite(ref.get("cold_minus_warm_gap_sec"))
        no_benchmark_gap = finite(no_benchmark.get("cold_minus_warm_gap_sec"))
        if math.isfinite(ref_gap) and math.isfinite(no_benchmark_gap) and ref_gap != 0.0:
            reduction = 100.0 * (ref_gap - no_benchmark_gap) / ref_gap
            comparison_lines.append(
                f"- Benchmark effect with deterministic=False: the cold/warm gap changed from {ref_gap:.2f} s to {no_benchmark_gap:.2f} s ({reduction:.1f}% reduction when benchmark was disabled)."
            )
    det_benchmark = by_config.get("benchmark_true_deterministic_true")
    det_no_benchmark = by_config.get("benchmark_false_deterministic_true")
    if ref is not None and det_benchmark is not None:
        delta = finite(det_benchmark.get("cold_minus_warm_gap_sec")) - finite(ref.get("cold_minus_warm_gap_sec"))
        if math.isfinite(delta):
            comparison_lines.append(
                f"- Deterministic effect with benchmark=True: changing deterministic=False to True changed the gap by {delta:+.2f} s."
            )
    if no_benchmark is not None and det_no_benchmark is not None:
        delta = finite(det_no_benchmark.get("cold_minus_warm_gap_sec")) - finite(no_benchmark.get("cold_minus_warm_gap_sec"))
        if math.isfinite(delta):
            comparison_lines.append(
                f"- Deterministic effect with benchmark=False: changing deterministic=False to True changed the gap by {delta:+.2f} s."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "All gap values are cold minus the mean of warm cases. The expand phase is dense B-spline expansion; backward_step includes zeroing gradients, autograd backward, and the optimizer update.",
            "The setup.device block is reported separately because it occurs before the loaded-images timing endpoint. DVF output preparation and writing are also reported separately.",
            *comparison_lines,
            "",
            "## Environment",
            "",
            f"- PyTorch: {environment.get('torch_version')}; CUDA runtime: {environment.get('cuda_runtime_version')}; cuDNN: {environment.get('cudnn_version')}.",
            f"- GPU query: {environment.get('gpu_query')}.",
            f"- Git commit: {environment.get('git_commit')}; dirty tree at launch: {environment.get('git_dirty')}.",
        ]
    )
    if dice_summary:
        lines.extend(["", "## Dice evaluation", "", "Dice was evaluated after timing aggregation from the 20 saved DVFs. Evaluation time is not included in the timing table."])
    report_path = Path(args.review_root) / "mrdbscp_cudnn_ablation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = resolve_artifact_paths(parse_args())
    ranks = parse_case_ranks(str(args.case_ranks))
    if str(args.gpu) != "2":
        raise ValueError("This focused ablation is fixed to physical GPU 2.")
    if ranks != [0, 1, 2, 3, 4]:
        raise ValueError(f"This focused ablation requires case ranks 0-4, got {ranks}")
    if args.worker_config:
        return worker(args)

    environment = environment_preflight(args) if not args.dry_run else git_metadata()
    write_manifest(args, environment)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "configs": [asdict(config) for config in CONFIGS],
                    "expected_timing_runs": 20,
                },
                indent=2,
            ),
            flush=True,
        )
        for config in CONFIGS:
            print(json.dumps({"config": config.slug, "command": worker_command(args, config)}), flush=True)
        return 0

    launch = run_workers(args)
    _, summary_rows = aggregate_timing(args)
    all_timing_complete = all(
        int(row.get("cold_count", 0)) == 1 and int(row.get("warm_count", 0)) == 4 for row in summary_rows
    ) and all(int(item.get("returncode", 1)) == 0 for item in launch.get("results", []))
    if not all_timing_complete:
        write_report(args, environment, summary_rows, None)
        raise RuntimeError("Timing ablation incomplete; Dice evaluation was not started.")

    dice_summary = None
    if not args.skip_dice:
        evaluator = [
            sys.executable,
            str(THIS_DIR / "evaluate_mrdbscp_cudnn_ablation.py"),
            "--gpu",
            str(args.gpu),
            "--case-ranks",
            str(args.case_ranks),
            "--alpha",
            f"{float(args.alpha):g}",
            "--out-root",
            str(args.out_root),
            "--review-root",
            str(args.review_root),
            "--oasis-root",
            str(args.oasis_root),
            "--pair-file",
            str(args.pair_file),
            "--variant",
            str(args.variant),
        ]
        eval_log = Path(args.review_root) / "logs" / "dice_evaluation.log"
        eval_log.parent.mkdir(parents=True, exist_ok=True)
        with eval_log.open("w") as log_handle:
            completed = subprocess.run(
                evaluator,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=thread_env(int(args.cpu_threads)),
            )
        if completed.returncode != 0:
            write_report(args, environment, summary_rows, None)
            raise RuntimeError(f"Posthoc Dice evaluation failed; see {eval_log}")
        dice_summary = load_dice_summary(Path(args.review_root))
    write_report(args, environment, summary_rows, dice_summary)
    print(
        json.dumps(
            {"completed": True, "timing_runs": 20, "dice_evaluated": not args.skip_dice},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
