#!/usr/bin/env python
"""Run DirectCP 8-4-2 DVF-only timing with ranks 0-9 in persistent workers."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import run_directcp842_dvf_only_timing as base_runner  # noqa: E402
import run_one  # noqa: E402


DIRECTCP_METHOD = base_runner.METHOD
VARIANT = "oasis_seed3_directcp842_persistent_dvf_timing_alpha1000_v1"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_directcp842_persistent_dvf_timing_alpha1000_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-04_oasis_directcp842_persistent_process_dvf_timing"


def parse_rank_list(text: str, *, phase: str) -> List[int]:
    requested = str(text).strip()
    if not requested:
        requested = "0,1" if str(phase) == "smoke" else "0-9"
    ranks: List[int] = []
    for token in requested.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            ranks.extend(range(int(left), int(right) + 1))
        else:
            ranks.append(int(token))
    return sorted(dict.fromkeys(ranks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-candidate", type=str, default="")
    parser.add_argument("--phase", choices=["smoke", "first10"], default="first10")
    parser.add_argument("--ranks", type=str, default="")
    parser.add_argument("--candidates", type=str, default="all")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--timing-detail", choices=["coarse", "deep"], default="deep")
    parser.add_argument("--alpha", type=float, default=base_runner.ALPHA)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--stage-full-cps", type=str, default=base_runner.STAGE_FULL_CPS)
    parser.add_argument("--stage-downsample", type=str, default=base_runner.STAGE_DOWNSAMPLE)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--allow-incomplete-analyze", action="store_true")
    return parser.parse_args()


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


def candidate_names(text: str) -> List[str]:
    return base_runner.parse_candidate_names(str(text))


def candidate_for_phase(name: str, phase: str) -> base_runner.Candidate:
    return base_runner.phase_candidate(base_runner.candidate_map()[name], "smoke" if phase == "smoke" else "rank0")


def stage_epoch_total(stage_epochs: str) -> int:
    return int(sum(c.parse_int_list(str(stage_epochs))))


def candidate_method(candidate: str) -> str:
    return str(base_runner.candidate_map()[str(candidate)].method)


def persistent_run_name(candidate: str, rank: int, alpha: float, phase: str) -> str:
    suffix = f"persistent_dvfonly_{candidate}_rank{rank:03d}_a{c.format_float(alpha)}"
    if str(phase) == "smoke":
        suffix += "_smoke_e1-1-1"
    return f"{candidate_method(str(candidate))}_{suffix}"


def run_dir(args: argparse.Namespace, candidate: str, rank: int) -> Path:
    return Path(args.out_root) / candidate_method(str(candidate)) / persistent_run_name(candidate, rank, float(args.alpha), str(args.phase))


def run_one_argv(args: argparse.Namespace, candidate: base_runner.Candidate, rank: int) -> List[str]:
    case = c.resolve_case(rank=int(rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
    argv = [
        "run_one.py",
        "--method",
        str(candidate.method),
        "--output-mode",
        "dvf_only",
        "--alpha",
        f"{float(args.alpha):g}",
        "--lr",
        f"{float(candidate.lr):g}",
        "--gpu",
        str(args.gpu),
        "--case-rank",
        str(int(rank)),
        "--epochs",
        str(stage_epoch_total(candidate.stage_epochs) if str(candidate.method) == DIRECTCP_METHOD else int(candidate.epochs)),
        "--eval-interval",
        str(1 if str(args.phase) == "smoke" else int(args.eval_interval)),
        "--seed",
        str(int(args.seed)),
        "--runtime-candidate",
        str(candidate.name),
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
        persistent_run_name(str(candidate.name), int(rank), float(args.alpha), str(args.phase)),
        "--expected-original-case-index",
        str(case.original_case_index),
        "--expected-fixed-id",
        str(case.fixed_id),
        "--expected-moving-id",
        str(case.moving_id),
    ]
    if str(candidate.method) == DIRECTCP_METHOD:
        argv.extend(
            [
                "--stage-full-cps",
                str(args.stage_full_cps),
                "--stage-downsample",
                str(args.stage_downsample),
                "--stage-epochs",
                str(candidate.stage_epochs),
                "--stage-early-stop-mode",
                str(candidate.early_stop_mode),
                "--stage-patience",
                str(candidate.stage_patience),
                "--stage-check-interval",
                str(candidate.stage_check_interval),
                "--stage-rel-delta-total",
                str(candidate.stage_rel_delta_total),
                "--stage-rel-delta-data",
                str(candidate.stage_rel_delta_data),
            ]
        )
    if str(candidate.method) == DIRECTCP_METHOD and str(candidate.stage_lr):
        argv.extend(["--stage-lr", str(candidate.stage_lr)])
    if str(candidate.method) == DIRECTCP_METHOD and str(candidate.stage_min_epochs):
        argv.extend(["--stage-min-epochs", str(candidate.stage_min_epochs)])
    if args.overwrite:
        argv.append("--overwrite")
    return argv


def write_command_rows(args: argparse.Namespace, rows: Sequence[Dict[str, object]], *, suffix: str = "") -> None:
    stem = f"command_timing_dvf_only_{args.phase}{suffix}.csv"
    for root in [Path(args.out_root), Path(args.review_root)]:
        c.write_csv(root / stem, rows)


def write_worker_rows(args: argparse.Namespace, rows: Sequence[Dict[str, object]], candidate: str) -> None:
    write_command_rows(args, rows, suffix=f"_{candidate}")


def capture_gpu_rows(
    args: argparse.Namespace,
    rows: List[Dict[str, object]],
    *,
    candidate: str,
    rank: int,
    sequence_index: int,
    snapshot: str,
    state: Dict[str, object],
) -> None:
    rows.append(
        {
            "phase": str(args.phase),
            "candidate": str(candidate),
            "case_rank": int(rank),
            "case_sequence_index": int(sequence_index),
            "snapshot": str(snapshot),
            "state_json": json.dumps(state, allow_nan=True),
        }
    )


def run_worker(args: argparse.Namespace) -> None:
    candidate_name = str(args.worker_candidate)
    candidate = candidate_for_phase(candidate_name, str(args.phase))
    ranks = parse_rank_list(str(args.ranks), phase=str(args.phase))
    log_dir = Path(args.out_root) / "logs" / f"persistent_dvf_only_{args.phase}" / candidate_name
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.review_root).mkdir(parents=True, exist_ok=True)

    command_rows: List[Dict[str, object]] = []
    gpu_state_rows: List[Dict[str, object]] = []
    print(
        json.dumps(
            {
                "persistent_worker_start": c.now_text(),
                "candidate": candidate_name,
                "ranks": ranks,
                "gpu": str(args.gpu),
                "pid": os.getpid(),
                "phase": str(args.phase),
            }
        ),
        flush=True,
    )
    for sequence_index, rank in enumerate(ranks):
        case = c.resolve_case(rank=int(rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        target_dir = run_dir(args, candidate_name, int(rank))
        log_path = log_dir / f"{persistent_run_name(candidate_name, int(rank), float(args.alpha), str(args.phase))}_gpu{args.gpu}.log"
        pre_state = base_runner.capture_gpu_state()
        capture_gpu_rows(
            args,
            gpu_state_rows,
            candidate=candidate_name,
            rank=int(rank),
            sequence_index=int(sequence_index),
            snapshot="pre",
            state=pre_state,
        )
        non_target_pre = base_runner.non_target_gpu_apps(pre_state, str(args.gpu))
        argv = run_one_argv(args, candidate, int(rank))
        print(
            json.dumps(
                {
                    "persistent_case_start": c.now_text(),
                    "candidate": candidate_name,
                    "rank": int(rank),
                    "sequence_index": int(sequence_index),
                    "warm_status": "cold_rank0" if sequence_index == 0 else "warm_persistent",
                    "log": str(log_path),
                }
            ),
            flush=True,
        )
        started_at = c.now_text()
        start = time.perf_counter()
        returncode = 0
        old_argv = list(sys.argv)
        with log_path.open("w") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
            try:
                sys.argv = argv
                run_one.main()
            except Exception:
                returncode = 1
                traceback.print_exc()
            finally:
                sys.argv = old_argv
        case_wall_sec = time.perf_counter() - start
        completed_at = c.now_text()
        post_state = base_runner.capture_gpu_state()
        capture_gpu_rows(
            args,
            gpu_state_rows,
            candidate=candidate_name,
            rank=int(rank),
            sequence_index=int(sequence_index),
            snapshot="post",
            state=post_state,
        )
        non_target_post = base_runner.non_target_gpu_apps(post_state, str(args.gpu))
        dense_path = target_dir / f"disp_norm_{case.fixed_id}_{case.moving_id}.npy"
        row = {
            "job_index": int(sequence_index),
            "candidate": candidate_name,
            "method": str(candidate.method),
            "phase": str(args.phase),
            "case_rank": int(case.rank),
            "case_sequence_index": int(sequence_index),
            "warm_status": "cold_rank0" if sequence_index == 0 else "warm_persistent",
            "original_case_index": int(case.original_case_index),
            "fixed_id": str(case.fixed_id),
            "moving_id": str(case.moving_id),
            "alpha": float(args.alpha),
            "lr": float(candidate.lr),
            "epochs": int(stage_epoch_total(candidate.stage_epochs) if str(candidate.method) == DIRECTCP_METHOD else int(candidate.epochs)),
            "stage_epochs": str(candidate.stage_epochs),
            "stage_lr": str(candidate.stage_lr),
            "early_stop_mode": str(candidate.early_stop_mode),
            "stage_min_epochs": str(candidate.stage_min_epochs),
            "stage_patience": str(candidate.stage_patience),
            "stage_check_interval": str(candidate.stage_check_interval),
            "stage_rel_delta_total": str(candidate.stage_rel_delta_total),
            "stage_rel_delta_data": str(candidate.stage_rel_delta_data),
            "persistent_process": True,
            "persistent_worker_pid": os.getpid(),
            "started_at": started_at,
            "completed_at": completed_at,
            "gpu": str(args.gpu),
            "returncode": int(returncode),
            "external_wall_sec": float(case_wall_sec),
            "persistent_case_wall_sec": float(case_wall_sec),
            "run_dir": str(target_dir),
            "log_path": str(log_path),
            "done_exists": (target_dir / "DONE").exists(),
            "summary_exists": (target_dir / "summary.json").exists(),
            "timing_exists": (target_dir / "timing.json").exists(),
            "dense_dvf_exists": dense_path.exists(),
            "dense_dvf_path": str(dense_path),
            "cmd_json": json.dumps(argv),
            "non_target_gpu_apps_pre_count": len(non_target_pre),
            "non_target_gpu_apps_post_count": len(non_target_post),
        }
        command_rows.append(row)
        write_worker_rows(args, command_rows, candidate_name)
        for root in [Path(args.out_root), Path(args.review_root)]:
            c.write_csv(root / f"gpu_state_persistent_dvf_only_{args.phase}_{candidate_name}.csv", gpu_state_rows)
        print(
            json.dumps(
                {
                    "persistent_case_done": c.now_text(),
                    "candidate": candidate_name,
                    "rank": int(rank),
                    "returncode": int(returncode),
                    "case_wall_sec": float(case_wall_sec),
                    "done_exists": row["done_exists"],
                    "timing_exists": row["timing_exists"],
                    "dense_dvf_exists": row["dense_dvf_exists"],
                }
            ),
            flush=True,
        )
        if returncode != 0:
            raise RuntimeError(f"Persistent worker case failed; see {log_path}")

    summary = {
        "completed_at": c.now_text(),
        "candidate": candidate_name,
        "phase": str(args.phase),
        "rank_count": len(ranks),
        "ranks": ranks,
        "completed_count": sum(1 for row in command_rows if int(row.get("returncode", 1)) == 0),
        "done_count": sum(1 for row in command_rows if bool(row.get("done_exists"))),
        "timing_json_count": sum(1 for row in command_rows if bool(row.get("timing_exists"))),
        "dense_dvf_count": sum(1 for row in command_rows if bool(row.get("dense_dvf_exists"))),
        "persistent_worker_pid": os.getpid(),
    }
    for root in [Path(args.out_root), Path(args.review_root)]:
        c.write_json(root / f"persistent_worker_summary_{args.phase}_{candidate_name}.json", summary)


def merge_worker_rows(args: argparse.Namespace, candidates: Sequence[str]) -> List[Dict[str, object]]:
    import csv

    launched_names = {str(candidate) for candidate in candidates}
    preserved: List[Dict[str, object]] = []
    combined_name = f"command_timing_dvf_only_{args.phase}.csv"
    for path in [Path(args.review_root) / combined_name, Path(args.out_root) / combined_name]:
        if path.exists():
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    if str(row.get("candidate")) not in launched_names:
                        preserved.append(dict(row))
            break

    launched: List[Dict[str, object]] = []
    for candidate in candidates:
        path = Path(args.out_root) / f"command_timing_dvf_only_{args.phase}_{candidate}.csv"
        if not path.exists():
            path = Path(args.review_root) / f"command_timing_dvf_only_{args.phase}_{candidate}.csv"
        if path.exists():
            with path.open(newline="") as f:
                launched.extend(list(csv.DictReader(f)))

    order = {name: idx for idx, name in enumerate(base_runner.parse_candidate_names("all"))}
    merged = preserved + launched
    return sorted(
        merged,
        key=lambda row: (
            order.get(str(row.get("candidate")), 999),
            int(float(row.get("case_rank", 0) or 0)),
        ),
    )


def candidate_names_from_rows(rows: Sequence[Dict[str, object]]) -> List[str]:
    order = {name: idx for idx, name in enumerate(base_runner.parse_candidate_names("all"))}
    names = sorted({str(row.get("candidate")) for row in rows if row.get("candidate")}, key=lambda name: order.get(name, 999))
    return names


def launch_worker(args: argparse.Namespace, candidate: str) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-candidate",
        str(candidate),
        "--phase",
        str(args.phase),
        "--ranks",
        ",".join(str(v) for v in parse_rank_list(str(args.ranks), phase=str(args.phase))),
        "--gpu",
        str(args.gpu),
        "--timing-detail",
        str(args.timing_detail),
        "--alpha",
        f"{float(args.alpha):g}",
        "--variant",
        str(args.variant),
        "--stage-full-cps",
        str(args.stage_full_cps),
        "--stage-downsample",
        str(args.stage_downsample),
        "--eval-interval",
        str(int(args.eval_interval)),
        "--seed",
        str(int(args.seed)),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--oasis-root",
        str(args.oasis_root),
        "--pair-file",
        str(args.pair_file),
        "--cpu-threads",
        str(int(args.cpu_threads)),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        return {"candidate": candidate, "cmd_json": json.dumps(cmd), "dry_run": True}
    log_dir = Path(args.out_root) / "logs" / f"persistent_worker_{args.phase}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_{candidate}_gpu{args.gpu}.log"
    print(json.dumps({"launch_persistent_worker": cmd, "log": str(log_path)}), flush=True)
    start = time.perf_counter()
    started_at = c.now_text()
    with log_path.open("w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=thread_env(int(args.cpu_threads)), check=False)
    elapsed = time.perf_counter() - start
    row = {
        "phase": str(args.phase),
        "candidate": str(candidate),
        "started_at": started_at,
        "completed_at": c.now_text(),
        "gpu": str(args.gpu),
        "returncode": int(proc.returncode),
        "worker_wall_sec": float(elapsed),
        "log_path": str(log_path),
        "cmd_json": json.dumps(cmd),
    }
    if proc.returncode != 0:
        raise RuntimeError(f"Persistent worker failed for {candidate}; see {log_path}")
    return row


def write_manifest(args: argparse.Namespace, candidates: Sequence[str], ranks: Sequence[int]) -> None:
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.review_root).mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": c.now_text(),
        "phase": str(args.phase),
        "ranks": list(int(v) for v in ranks),
        "candidates": list(candidates),
        "methods": sorted({candidate_method(candidate) for candidate in candidates}),
        "candidate_definitions": [
            base_runner.asdict(base_runner.candidate_map()[candidate])
            if hasattr(base_runner, "asdict")
            else str(base_runner.candidate_map()[candidate])
            for candidate in candidates
        ],
        "variant": str(args.variant),
        "alpha": float(args.alpha),
        "timing_detail": str(args.timing_detail),
        "output_mode": "dvf_only",
        "process_model": "one persistent Python worker subprocess per candidate; ranks are sequential inside that worker",
        "cold_case_definition": "sequence_index=0, rank0 in each candidate worker",
        "warm_case_definition": "sequence_index>0, ranks1-9 in the same candidate worker",
        "gpu": str(args.gpu),
        "out_root": str(args.out_root),
        "review_root": str(args.review_root),
    }
    for root in [Path(args.out_root), Path(args.review_root)]:
        c.write_json(root / f"job_manifest_persistent_dvf_only_{args.phase}.json", manifest)


def analyze(args: argparse.Namespace, candidates: Sequence[str], ranks: Sequence[int]) -> None:
    base_cmd = [
        sys.executable,
        str(THIS_DIR / "analyze_directcp842_dvf_only_timing.py"),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--phase",
        str(args.phase),
        "--expected-case-ranks",
        ",".join(str(v) for v in ranks),
        "--expected-candidates",
        ",".join(candidates),
        "--previous-review-root",
        str(DEFAULT_REVIEW_ROOT),
    ]
    if args.allow_incomplete_analyze:
        base_cmd.append("--allow-incomplete")
    print(json.dumps({"analyze_cmd": base_cmd}), flush=True)
    subprocess.check_call(base_cmd)
    persistent_cmd = [
        sys.executable,
        str(THIS_DIR / "analyze_directcp842_persistent_dvf_timing.py"),
        "--review-root",
        str(args.review_root),
        "--phase",
        str(args.phase),
    ]
    print(json.dumps({"persistent_analyze_cmd": persistent_cmd}), flush=True)
    subprocess.check_call(persistent_cmd)


def main() -> None:
    args = parse_args()
    if str(args.worker_candidate):
        run_worker(args)
        return
    candidates = candidate_names(str(args.candidates))
    ranks = parse_rank_list(str(args.ranks), phase=str(args.phase))
    write_manifest(args, candidates, ranks)
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "phase": str(args.phase),
                "ranks": ranks,
                "candidates": candidates,
                "gpu": str(args.gpu),
                "process_model": "one persistent worker subprocess per candidate",
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        ),
        flush=True,
    )
    worker_rows: List[Dict[str, object]] = []
    for candidate in candidates:
        worker_rows.append(launch_worker(args, candidate))
    if not args.dry_run:
        merged_rows = merge_worker_rows(args, candidates)
        combined_candidates = candidate_names_from_rows(merged_rows)
        write_command_rows(args, merged_rows)
        for root in [Path(args.out_root), Path(args.review_root)]:
            c.write_csv(root / f"persistent_worker_timing_{args.phase}.csv", worker_rows)
            c.write_json(
                root / f"launch_summary_persistent_dvf_only_{args.phase}.json",
                {
                    "completed_at": c.now_text(),
                    "phase": str(args.phase),
                    "candidate_count": len(candidates),
                    "combined_candidate_count": len(combined_candidates),
                    "rank_count": len(ranks),
                    "expected_case_count": len(candidates) * len(ranks),
                    "combined_case_count": len(merged_rows),
                    "combined_candidates": combined_candidates,
                    "command_row_count": len(merged_rows),
                    "worker_rows": worker_rows,
                },
            )
    if not args.dry_run and not args.skip_analyze:
        analyze(args, combined_candidates, ranks)


if __name__ == "__main__":
    main()
