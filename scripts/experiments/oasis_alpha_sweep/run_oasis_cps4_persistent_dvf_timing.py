#!/usr/bin/env python
"""Run persistent-process DVF-only timing for OASIS cps4 sweep candidates."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import launch_oasis_cps4_runtime_sweep as sweep  # noqa: E402
import run_directcp842_dvf_only_timing as dvf_base  # noqa: E402
import run_one  # noqa: E402


VARIANT = "oasis_seed3_first10_cps4_single_stage_persistent_dvf_timing_alpha1000_v1"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_cps4_single_stage_persistent_dvf_timing_alpha1000_v1")
DEFAULT_REVIEW_ROOT = sweep.DEFAULT_REVIEW_ROOT
DEFAULT_PROMISING = ["D_A_lr1e-2", "D_F600_lr1e-2", "D_F1000_lr3e-3", "S_F1000_lr1e-3", "S_A_lr1e-3"]
BASELINES = ["D_baseline_lr1e-3_e2500", "S_baseline_lr1e-4_e2500"]


def parse_rank_list(text: str, *, phase: str) -> List[int]:
    requested = str(text).strip() or ("0,1" if str(phase) == "smoke" else "0-9")
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
    parser.add_argument("--candidates", type=str, default="promising_plus_baselines")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--timing-detail", choices=["coarse", "deep"], default="deep")
    parser.add_argument("--alpha", type=float, default=sweep.ALPHA)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--screen-review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
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


def read_promising_candidates(review_root: Path) -> List[str]:
    txt = Path(review_root) / "promising_candidates.txt"
    if txt.exists():
        tokens: List[str] = []
        for raw in txt.read_text().replace("\n", ",").split(","):
            token = raw.strip()
            if token:
                tokens.append(token)
        if tokens:
            return tokens
    csv_path = Path(review_root) / "promising_candidates.csv"
    if csv_path.exists() and csv_path.read_text().strip():
        with csv_path.open(newline="") as f:
            return [str(row.get("runtime_candidate", "")).strip() for row in csv.DictReader(f) if row.get("runtime_candidate")]
    return list(DEFAULT_PROMISING)


def ordered_names(names: Sequence[str]) -> List[str]:
    order = {candidate.name: idx for idx, candidate in enumerate(sweep.CANDIDATES)}
    deduped = list(dict.fromkeys(str(name) for name in names if str(name)))
    missing = [name for name in deduped if name not in sweep.candidate_map()]
    if missing:
        raise ValueError(f"Unknown cps4 timing candidates {missing}; known={sorted(sweep.candidate_map())}")
    return sorted(deduped, key=lambda name: order.get(name, 999))


def include_family_baselines(names: Sequence[str]) -> List[str]:
    known = sweep.candidate_map()
    families = {known[name].family for name in names if name in known}
    out: List[str] = []
    if "D-BSCP" in families:
        out.append("D_baseline_lr1e-3_e2500")
    if "SINR" in families:
        out.append("S_baseline_lr1e-4_e2500")
    out.extend(names)
    return ordered_names(out)


def candidate_names(args: argparse.Namespace) -> List[str]:
    requested = str(args.candidates).strip()
    if requested in {"", "promising_plus_baselines", "accepted_plus_baselines"}:
        return include_family_baselines(read_promising_candidates(Path(args.screen_review_root)))
    if requested in {"promising", "accepted"}:
        return ordered_names(read_promising_candidates(Path(args.screen_review_root)))
    if requested == "baselines":
        return ordered_names(BASELINES)
    if requested == "all":
        return [candidate.name for candidate in sweep.CANDIDATES]
    return ordered_names([token.strip() for token in requested.split(",") if token.strip()])


def phase_candidate(candidate: sweep.Candidate, phase: str) -> sweep.Candidate:
    if str(phase) != "smoke":
        return candidate
    adaptive = str(candidate.early_stop_mode) == "adaptive"
    return sweep.Candidate(
        name=candidate.name,
        method=candidate.method,
        method_label=candidate.method_label,
        family=candidate.family,
        lr=float(candidate.lr),
        epochs=1,
        early_stop_mode=candidate.early_stop_mode,
        min_epochs=1 if adaptive else 0,
        patience=1 if adaptive else 0,
        min_delta=float(candidate.min_delta),
        description=f"Tiny persistent DVF smoke for {candidate.name}",
        is_baseline=bool(candidate.is_baseline),
    )


def persistent_run_name(candidate: str, rank: int, alpha: float, phase: str) -> str:
    method = sweep.candidate_map()[str(candidate)].method
    suffix = f"persistent_cps4_dvfonly_{candidate}_rank{rank:03d}_a{c.format_float(alpha)}"
    if str(phase) == "smoke":
        suffix += "_smoke_e1"
    return f"{method}_{suffix}"


def run_dir(args: argparse.Namespace, candidate: str, rank: int) -> Path:
    method = sweep.candidate_map()[str(candidate)].method
    return Path(args.out_root) / method / persistent_run_name(candidate, rank, float(args.alpha), str(args.phase))


def run_one_argv(args: argparse.Namespace, candidate: sweep.Candidate, rank: int) -> List[str]:
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
        str(int(candidate.epochs)),
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
        "--stage-early-stop-mode",
        str(candidate.early_stop_mode),
    ]
    if str(candidate.early_stop_mode) == "adaptive":
        argv.extend(
            [
                "--stage-min-epochs",
                str(int(candidate.min_epochs)),
                "--stage-patience",
                str(int(candidate.patience)),
                "--stage-check-interval",
                "1",
                "--stage-min-delta",
                f"{float(candidate.min_delta):g}",
            ]
        )
    if args.overwrite:
        argv.append("--overwrite")
    return argv


def write_csv_roots(args: argparse.Namespace, filename: str, rows: Sequence[Dict[str, object]]) -> None:
    for root in [Path(args.out_root), Path(args.review_root)]:
        c.write_csv(root / filename, rows)


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
    candidate = phase_candidate(sweep.candidate_map()[candidate_name], str(args.phase))
    ranks = parse_rank_list(str(args.ranks), phase=str(args.phase))
    log_dir = Path(args.out_root) / "logs" / f"persistent_cps4_dvf_only_{args.phase}" / candidate_name
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
        run_name = persistent_run_name(candidate_name, int(rank), float(args.alpha), str(args.phase))
        log_path = log_dir / f"{run_name}_gpu{args.gpu}.log"
        pre_state = dvf_base.capture_gpu_state()
        capture_gpu_rows(
            args,
            gpu_state_rows,
            candidate=candidate_name,
            rank=int(rank),
            sequence_index=int(sequence_index),
            snapshot="pre",
            state=pre_state,
        )
        non_target_pre = dvf_base.non_target_gpu_apps(pre_state, str(args.gpu))
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
        post_state = dvf_base.capture_gpu_state()
        capture_gpu_rows(
            args,
            gpu_state_rows,
            candidate=candidate_name,
            rank=int(rank),
            sequence_index=int(sequence_index),
            snapshot="post",
            state=post_state,
        )
        non_target_post = dvf_base.non_target_gpu_apps(post_state, str(args.gpu))
        dense_path = target_dir / f"disp_norm_{case.fixed_id}_{case.moving_id}.npy"
        row = {
            "job_index": int(sequence_index),
            "candidate": candidate_name,
            "method": str(candidate.method),
            "method_label": str(candidate.method_label),
            "family": str(candidate.family),
            "phase": str(args.phase),
            "case_rank": int(case.rank),
            "case_sequence_index": int(sequence_index),
            "warm_status": "cold_rank0" if sequence_index == 0 else "warm_persistent",
            "original_case_index": int(case.original_case_index),
            "fixed_id": str(case.fixed_id),
            "moving_id": str(case.moving_id),
            "alpha": float(args.alpha),
            "lr": float(candidate.lr),
            "epochs": int(candidate.epochs),
            "early_stop_mode": str(candidate.early_stop_mode),
            "min_epochs": int(candidate.min_epochs),
            "patience": int(candidate.patience),
            "min_delta": float(candidate.min_delta),
            "is_baseline": bool(candidate.is_baseline),
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
        write_csv_roots(args, f"command_timing_dvf_only_{args.phase}_{candidate_name}.csv", command_rows)
        write_csv_roots(args, f"gpu_state_persistent_cps4_dvf_only_{args.phase}_{candidate_name}.csv", gpu_state_rows)
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
        c.write_json(root / f"persistent_worker_summary_cps4_dvf_only_{args.phase}_{candidate_name}.json", summary)


def merge_worker_rows(args: argparse.Namespace, candidates: Sequence[str]) -> List[Dict[str, object]]:
    launched_names = {str(candidate) for candidate in candidates}
    combined_name = f"command_timing_dvf_only_{args.phase}.csv"
    preserved: List[Dict[str, object]] = []
    for path in [Path(args.review_root) / combined_name, Path(args.out_root) / combined_name]:
        if path.exists():
            with path.open(newline="") as f:
                preserved.extend(dict(row) for row in csv.DictReader(f) if str(row.get("candidate")) not in launched_names)
            break

    launched: List[Dict[str, object]] = []
    for candidate in candidates:
        path = Path(args.out_root) / f"command_timing_dvf_only_{args.phase}_{candidate}.csv"
        if not path.exists():
            path = Path(args.review_root) / f"command_timing_dvf_only_{args.phase}_{candidate}.csv"
        if path.exists():
            with path.open(newline="") as f:
                launched.extend(list(csv.DictReader(f)))

    order = {candidate.name: idx for idx, candidate in enumerate(sweep.CANDIDATES)}
    merged = preserved + launched
    return sorted(
        merged,
        key=lambda row: (
            order.get(str(row.get("candidate")), 999),
            int(float(row.get("case_rank", 0) or 0)),
        ),
    )


def candidate_names_from_rows(rows: Sequence[Dict[str, object]]) -> List[str]:
    return ordered_names(sorted({str(row.get("candidate")) for row in rows if row.get("candidate")}))


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
        "--eval-interval",
        str(int(args.eval_interval)),
        "--seed",
        str(int(args.seed)),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--screen-review-root",
        str(args.screen_review_root),
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
    log_dir = Path(args.out_root) / "logs" / f"persistent_cps4_worker_{args.phase}"
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
        "methods": sorted({sweep.candidate_map()[candidate].method for candidate in candidates}),
        "candidate_definitions": [asdict(sweep.candidate_map()[candidate]) for candidate in candidates],
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
        "screen_review_root": str(args.screen_review_root),
    }
    for root in [Path(args.out_root), Path(args.review_root)]:
        c.write_json(root / f"job_manifest_cps4_persistent_dvf_only_{args.phase}.json", manifest)


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
        str(args.review_root),
    ]
    if args.allow_incomplete_analyze:
        base_cmd.append("--allow-incomplete")
    print(json.dumps({"analyze_cmd": base_cmd}), flush=True)
    subprocess.check_call(base_cmd)
    cps4_cmd = [
        sys.executable,
        str(THIS_DIR / "analyze_oasis_cps4_persistent_dvf_timing.py"),
        "--review-root",
        str(args.review_root),
        "--screen-review-root",
        str(args.screen_review_root),
        "--phase",
        str(args.phase),
    ]
    print(json.dumps({"cps4_persistent_analyze_cmd": cps4_cmd}), flush=True)
    subprocess.check_call(cps4_cmd)


def main() -> None:
    args = parse_args()
    if str(args.worker_candidate):
        run_worker(args)
        return
    candidates = candidate_names(args)
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
        write_csv_roots(args, f"command_timing_dvf_only_{args.phase}.csv", merged_rows)
        for root in [Path(args.out_root), Path(args.review_root)]:
            c.write_csv(root / f"persistent_worker_timing_cps4_{args.phase}.csv", worker_rows)
            c.write_json(
                root / f"launch_summary_cps4_persistent_dvf_only_{args.phase}.json",
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
