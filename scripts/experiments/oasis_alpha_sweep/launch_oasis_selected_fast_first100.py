#!/usr/bin/env python
"""Launch selected fast OASIS100 schedules with first10 reuse."""

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
from types import SimpleNamespace
from typing import Dict, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
from launch_first10_be_2500e import parse_case_ranks  # noqa: E402
import launch_oasis_unified_acceleration_sweep as unified  # noqa: E402
import run_one  # noqa: E402


VARIANT = "oasis_selected_fast_first100_alpha1000_v1"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_selected_fast_first100_alpha1000_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-13_oasis_selected_fast_first100"
FIRST10_SOURCE_ROOT = Path("/database/inrorspline/runs/oasis_unified_acceleration_sweep_alpha1000_v1")
SELECTED_CANDIDATES = [
    "MR_A_fast_decay",
    "D2_A_bal_lr10x",
    "D4_A_bal_lr10x",
    "S2_A_bal_lr10x",
    "S4_A_bal_lr10x",
]


@dataclass(frozen=True)
class WorkItem:
    name: str
    candidate_name: str
    ranks: str
    gpu: str
    phase_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "mr", "models", "all", "aggregate"], default="all")
    parser.add_argument("--worker-candidate", type=str, default="")
    parser.add_argument("--worker-ranks", type=str, default="")
    parser.add_argument("--worker-name", type=str, default="")
    parser.add_argument("--worker-gpu", type=str, default="")
    parser.add_argument("--phase-name", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--first10-source-root", type=Path, default=FIRST10_SOURCE_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--timing-detail", choices=["off", "coarse", "deep"], default="coarse")
    parser.add_argument("--epoch-override", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--allow-incomplete-aggregate", action="store_true")
    return parser.parse_args()


def parse_gpus(text: str) -> List[str]:
    return [token.strip() for token in str(text).split(",") if token.strip()]


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


def selected_candidate(name: str) -> unified.Candidate:
    known = unified.candidate_map()
    if name not in known:
        raise ValueError(f"Unknown candidate {name!r}; known={sorted(known)}")
    return known[name]


def work_items_for_stage(stage: str, gpus: Sequence[str]) -> List[WorkItem]:
    if len(gpus) < 4:
        raise ValueError("This selected-first100 plan requires four GPU identifiers.")
    if stage == "smoke":
        return [
            WorkItem(f"smoke_{name}", name, "10", gpus[idx % len(gpus)], "selected_first100_smoke")
            for idx, name in enumerate(SELECTED_CANDIDATES)
        ]
    mr_items = [
        WorkItem("mr_chunk0_rank010_032", "MR_A_fast_decay", "10-32", gpus[0], "selected_first100"),
        WorkItem("mr_chunk1_rank033_055", "MR_A_fast_decay", "33-55", gpus[1], "selected_first100"),
        WorkItem("mr_chunk2_rank056_077", "MR_A_fast_decay", "56-77", gpus[2], "selected_first100"),
        WorkItem("mr_chunk3_rank078_099", "MR_A_fast_decay", "78-99", gpus[3], "selected_first100"),
    ]
    model_items = [
        WorkItem("model_d2_rank010_099", "D2_A_bal_lr10x", "10-99", gpus[0], "selected_first100"),
        WorkItem("model_d4_rank010_099", "D4_A_bal_lr10x", "10-99", gpus[1], "selected_first100"),
        WorkItem("model_s2_rank010_099", "S2_A_bal_lr10x", "10-99", gpus[2], "selected_first100"),
        WorkItem("model_s4_rank010_099", "S4_A_bal_lr10x", "10-99", gpus[3], "selected_first100"),
    ]
    if stage == "mr":
        return mr_items
    if stage == "models":
        return model_items
    if stage == "all":
        return mr_items + model_items
    if stage == "aggregate":
        return []
    raise ValueError(stage)


def unified_args(args: argparse.Namespace, item: WorkItem, *, gpu: str) -> argparse.Namespace:
    return SimpleNamespace(
        phase=item.phase_name,
        case_ranks=item.ranks,
        gpus=str(gpu),
        alpha=float(args.alpha),
        variant=str(args.variant),
        eval_interval=int(args.eval_interval),
        seed=int(args.seed),
        out_root=Path(args.out_root),
        review_root=Path(args.review_root),
        oasis_root=Path(args.oasis_root),
        pair_file=Path(args.pair_file),
        cpu_threads=int(args.cpu_threads),
        timing_detail=str(args.timing_detail),
        epoch_override=int(args.epoch_override),
        overwrite=bool(args.overwrite),
    )


def run_dir(args: argparse.Namespace, item: WorkItem, rank: int) -> Path:
    candidate = unified.phase_candidate(selected_candidate(item.candidate_name), unified_args(args, item, gpu=item.gpu))
    return Path(args.out_root) / candidate.method / unified.run_name(candidate, int(rank), float(args.alpha), item.phase_name)


def write_manifest(args: argparse.Namespace, items: Sequence[WorkItem]) -> None:
    rows: List[Dict[str, object]] = []
    job_rows: List[Dict[str, object]] = []
    for item in items:
        candidate = selected_candidate(item.candidate_name)
        ranks = parse_case_ranks(item.ranks)
        rows.append(
            {
                "stage": args.stage,
                "worker_name": item.name,
                "candidate": item.candidate_name,
                "method": candidate.method,
                "method_label": candidate.method_label,
                "family": candidate.family,
                "ranks": item.ranks,
                "rank_count": len(ranks),
                "gpu": item.gpu,
                "phase_name": item.phase_name,
            }
        )
        for rank in ranks:
            case = c.resolve_case(rank=int(rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
            job_rows.append(
                {
                    "stage": args.stage,
                    "worker_name": item.name,
                    "candidate": item.candidate_name,
                    "method": candidate.method,
                    "method_label": candidate.method_label,
                    "case_rank": int(rank),
                    "original_case_index": int(case.original_case_index),
                    "fixed_id": case.fixed_id,
                    "moving_id": case.moving_id,
                    "gpu": item.gpu,
                    "phase_name": item.phase_name,
                }
            )
    for root in [Path(args.review_root), Path(args.out_root)]:
        c.write_csv(root / f"selected_first100_worker_manifest_{args.stage}.csv", rows)
        c.write_json(root / f"selected_first100_worker_manifest_{args.stage}.json", rows)
        c.write_csv(root / f"selected_first100_job_manifest_{args.stage}.csv", job_rows)
        c.write_json(
            root / f"selected_first100_job_manifest_{args.stage}.json",
            {
                "created_at": c.now_text(),
                "stage": str(args.stage),
                "worker_count": len(rows),
                "registration_count": len(job_rows),
                "first10_reuse_root": str(args.first10_source_root),
                "new_output_root": str(args.out_root),
                "jobs": job_rows,
            },
        )


def run_worker(args: argparse.Namespace) -> None:
    item = WorkItem(
        str(args.worker_name),
        str(args.worker_candidate),
        str(args.worker_ranks),
        str(args.worker_gpu),
        str(args.phase_name),
    )
    candidate = selected_candidate(item.candidate_name)
    ranks = parse_case_ranks(item.ranks)
    log_dir = Path(args.out_root) / "logs" / "selected_first100_worker" / item.name
    log_dir.mkdir(parents=True, exist_ok=True)
    command_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    ns = unified_args(args, item, gpu=item.gpu)
    for sequence_index, rank in enumerate(ranks):
        effective = unified.phase_candidate(candidate, ns)
        target_dir = Path(args.out_root) / effective.method / unified.run_name(effective, int(rank), float(args.alpha), item.phase_name)
        case = c.resolve_case(rank=int(rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        log_path = log_dir / f"{target_dir.name}.log"
        start = time.perf_counter()
        status = "completed"
        error = ""
        if (target_dir / "DONE").exists() and not args.overwrite:
            status = "skipped_existing"
        else:
            argv = unified.run_one_argv(ns, candidate, int(rank), item.gpu)
            with log_path.open("w") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                old_argv = sys.argv[:]
                try:
                    sys.argv = argv
                    run_one.main()
                except Exception:
                    status = "failed"
                    error = traceback.format_exc()
                    print(error, flush=True)
                finally:
                    sys.argv = old_argv
        elapsed = time.perf_counter() - start
        done = (target_dir / "DONE").exists()
        row = {
            "worker_name": item.name,
            "candidate": candidate.name,
            "method": candidate.method,
            "method_label": candidate.method_label,
            "family": candidate.family,
            "gpu": str(item.gpu),
            "case_rank": int(rank),
            "case_sequence_index": int(sequence_index),
            "warm_status": "cold_worker_first_case" if sequence_index == 0 else "warm_worker_later_case",
            "original_case_index": int(case.original_case_index),
            "fixed_id": case.fixed_id,
            "moving_id": case.moving_id,
            "status": status,
            "done": bool(done),
            "external_case_wall_sec": float(elapsed),
            "run_dir": str(target_dir),
            "log_path": str(log_path),
            "error": error[:2000],
        }
        command_rows.append(row)
        if status == "failed" or not done:
            failure_rows.append(row)
        c.write_csv(Path(args.out_root) / f"command_timing_selected_first100_{item.name}.csv", command_rows)
    summary = {
        "completed_at": c.now_text(),
        "worker_name": item.name,
        "candidate": candidate.name,
        "rank_count": len(ranks),
        "command_rows": command_rows,
        "failures": failure_rows,
    }
    c.write_json(Path(args.out_root) / f"worker_summary_selected_first100_{item.name}.json", summary)
    if failure_rows:
        raise RuntimeError(f"Selected-first100 worker {item.name} had {len(failure_rows)} failures")


def launch_worker(args: argparse.Namespace, item: WorkItem) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-candidate",
        item.candidate_name,
        "--worker-ranks",
        item.ranks,
        "--worker-name",
        item.name,
        "--worker-gpu",
        item.gpu,
        "--phase-name",
        item.phase_name,
        "--alpha",
        f"{float(args.alpha):g}",
        "--variant",
        str(args.variant),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(args.seed),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--first10-source-root",
        str(args.first10_source_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--cpu-threads",
        str(args.cpu_threads),
        "--timing-detail",
        str(args.timing_detail),
        "--epoch-override",
        str(args.epoch_override),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    log_dir = Path(args.out_root) / "logs" / "selected_first100_launcher"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_{item.name}_gpu{item.gpu}.log"
    return {"item": item, "cmd": cmd, "log_path": log_path}


def run_items(args: argparse.Namespace, items: Sequence[WorkItem], *, label: str) -> Dict[str, object]:
    max_concurrent = min(max(1, int(args.max_concurrent)), len(parse_gpus(args.gpus)), len(items) if items else 1)
    pending = list(items)
    running: List[Dict[str, object]] = []
    completed: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    env = thread_env(int(args.cpu_threads))
    started = time.time()
    while pending or running:
        while pending and len(running) < max_concurrent:
            worker = launch_worker(args, pending.pop(0))
            item = worker["item"]
            assert isinstance(item, WorkItem)
            print(json.dumps({"launch_worker": item.__dict__, "cmd": worker["cmd"], "log": str(worker["log_path"])}), flush=True)
            log_f = Path(worker["log_path"]).open("w")
            proc = subprocess.Popen(worker["cmd"], stdout=log_f, stderr=subprocess.STDOUT, env=env)
            worker.update({"proc": proc, "log_file": log_f, "started_epoch": time.time()})
            running.append(worker)
        still: List[Dict[str, object]] = []
        for worker in running:
            proc = worker["proc"]
            assert isinstance(proc, subprocess.Popen)
            code = proc.poll()
            if code is None:
                still.append(worker)
                continue
            worker["log_file"].close()
            item = worker["item"]
            assert isinstance(item, WorkItem)
            row = {
                "worker_name": item.name,
                "candidate": item.candidate_name,
                "gpu": item.gpu,
                "ranks": item.ranks,
                "returncode": int(code),
                "log_path": str(worker["log_path"]),
                "elapsed_sec": time.time() - float(worker["started_epoch"]),
            }
            if code == 0:
                completed.append(row)
            else:
                failures.append(row)
            print(json.dumps({"worker_done": row}), flush=True)
        running = still
        if pending or running:
            time.sleep(5.0)
    payload = {
        "completed_at": c.now_text(),
        "label": label,
        "worker_count": len(items),
        "completed_workers": completed,
        "failed_workers": failures,
        "elapsed_sec": time.time() - started,
    }
    for root in [Path(args.out_root), Path(args.review_root)]:
        c.write_json(root / f"launch_summary_selected_first100_{label}.json", payload)
    if failures:
        raise RuntimeError("Selected-first100 workers failed: " + json.dumps(failures, indent=2))
    return payload


def aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_oasis_selected_fast_first100.py"),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--first10-source-root",
        str(args.first10_source_root),
        "--variant",
        str(args.variant),
    ]
    if args.allow_incomplete_aggregate:
        cmd.append("--allow-incomplete")
    print(json.dumps({"aggregate_cmd": cmd}), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    if args.worker_candidate:
        run_worker(args)
        return
    gpus = parse_gpus(str(args.gpus))
    if args.stage == "aggregate":
        aggregate(args)
        return
    if args.stage == "all":
        mr_items = work_items_for_stage("mr", gpus)
        model_items = work_items_for_stage("models", gpus)
        items = mr_items + model_items
    else:
        items = work_items_for_stage(str(args.stage), gpus)
    write_manifest(args, items)
    registration_count = sum(len(parse_case_ranks(item.ranks)) for item in items)
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "stage": str(args.stage),
                "worker_count": len(items),
                "registration_count": registration_count,
                "selected_candidates": SELECTED_CANDIDATES,
                "gpus": gpus,
                "max_concurrent": int(args.max_concurrent),
                "dry_run": bool(args.dry_run),
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
                "first10_source_root": str(args.first10_source_root),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        for item in items:
            print(json.dumps({"work_item": item.__dict__}), flush=True)
        return
    if args.stage == "all":
        run_items(args, mr_items, label="mr")
        run_items(args, model_items, label="models")
    else:
        run_items(args, items, label=str(args.stage))
    if not args.skip_aggregate and args.stage in {"all", "models"}:
        aggregate(args)


if __name__ == "__main__":
    main()
