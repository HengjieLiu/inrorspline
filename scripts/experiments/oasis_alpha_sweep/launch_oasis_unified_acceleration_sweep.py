#!/usr/bin/env python
"""Launch the unified OASIS acceleration screen with persistent candidate workers."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
from launch_first10_be_2500e import parse_case_ranks  # noqa: E402
import run_one  # noqa: E402

EXPERIMENTS_DIR = c.REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402


VARIANT = "oasis_seed3_first10_unified_acceleration_sweep_alpha1000_v1"
RUN_KEY = "oasis_unified_acceleration_sweep_alpha1000_v1"
EXPERIMENT_KEY = "2026-08-13_oasis_unified_acceleration_sweep"
DEFAULT_OUT_ROOT = None
DEFAULT_REVIEW_ROOT = None
ALPHA = 1000.0

MR_METHOD = "direct_cp_8_4_2_be_bspline_analytic_multistage"
D2_METHOD = "direct_cp_cps2_be_bspline_analytic"
S2_METHOD = "sinr_cps2_be_bspline_analytic"
D4_METHOD = "direct_cp_cps4_be_bspline_analytic"
S4_METHOD = "sinr_cps4_be_bspline_analytic"


@dataclass(frozen=True)
class Candidate:
    name: str
    phase: str
    method: str
    method_label: str
    family: str
    baseline_name: str
    lr: float
    epochs: int
    early_stop_mode: str = "none"
    min_epochs: int = 0
    patience: int = 0
    min_delta: float = 0.0
    stage_full_cps: str = ""
    stage_downsample: str = ""
    stage_epochs: str = ""
    stage_lr: str = ""
    stage_min_epochs: str = ""
    stage_patience: str = ""
    stage_check_interval: str = ""
    stage_rel_delta_total: str = ""
    stage_rel_delta_data: str = ""
    lr_policy: str = "single"
    template: str = ""
    description: str = ""
    is_baseline: bool = False


def _single_stage(prefix: str, method: str, label: str, family: str, base_lr: float, phase: str) -> List[Candidate]:
    baseline = f"{prefix}_baseline_lr1x_full"
    return [
        Candidate(baseline, phase, method, label, family, baseline, base_lr, 2500, template="baseline_lr1x_full", description="baseline full 2500 epochs", is_baseline=True),
        Candidate(f"{prefix}_F60_lr3x", phase, method, label, family, baseline, base_lr * 3.0, 1500, template="F60_lr3x", description="fixed 60% epoch budget with 3x LR"),
        Candidate(f"{prefix}_F40_lr10x", phase, method, label, family, baseline, base_lr * 10.0, 1000, template="F40_lr10x", description="fixed 40% epoch budget with 10x LR"),
        Candidate(f"{prefix}_F25_lr30x", phase, method, label, family, baseline, base_lr * 30.0, 600, template="F25_lr30x", description="fixed 25% epoch budget with 30x LR"),
        Candidate(
            f"{prefix}_A_bal_lr10x",
            phase,
            method,
            label,
            family,
            baseline,
            base_lr * 10.0,
            2500,
            early_stop_mode="adaptive",
            min_epochs=375,
            patience=150,
            min_delta=1e-4,
            template="A_bal_lr10x",
            description="adaptive balanced; 10x LR; min 15%, patience 6%",
        ),
        Candidate(
            f"{prefix}_A_fast_lr30x",
            phase,
            method,
            label,
            family,
            baseline,
            base_lr * 30.0,
            2500,
            early_stop_mode="adaptive",
            min_epochs=250,
            patience=100,
            min_delta=1e-4,
            template="A_fast_lr30x",
            description="adaptive fast; 30x LR; min 10%, patience 4%",
        ),
    ]


def _mr_candidate(
    name: str,
    *,
    template: str,
    lr: float,
    stage_epochs: str,
    stage_lr: str,
    early_stop_mode: str = "none",
    stage_min_epochs: str = "",
    stage_patience: str = "0,0,0",
    stage_rel_delta_total: str = "0,0,0",
    stage_rel_delta_data: str = "0,0,0",
    lr_policy: str,
    description: str,
    is_baseline: bool = False,
) -> Candidate:
    return Candidate(
        name=name,
        phase="primary",
        method=MR_METHOD,
        method_label="MR-D-BSCP 8-4-2",
        family="MR-D-BSCP",
        baseline_name="MR_baseline_lr1x_full",
        lr=lr,
        epochs=sum(c.parse_int_list(stage_epochs)),
        early_stop_mode=early_stop_mode,
        stage_full_cps="8,4,2",
        stage_downsample="4,2,1",
        stage_epochs=stage_epochs,
        stage_lr=stage_lr,
        stage_min_epochs=stage_min_epochs,
        stage_patience=stage_patience,
        stage_check_interval="25,25,25",
        stage_rel_delta_total=stage_rel_delta_total,
        stage_rel_delta_data=stage_rel_delta_data,
        lr_policy=lr_policy,
        template=template,
        description=description,
        is_baseline=is_baseline,
    )


def candidates_all() -> List[Candidate]:
    mr = [
        _mr_candidate("MR_baseline_lr1x_full", template="baseline_lr1x_full", lr=1e-3, stage_epochs="400,800,1300", stage_lr="1e-3,1e-3,1e-3", lr_policy="same_lr", description="MR baseline full 2500 epochs", is_baseline=True),
        _mr_candidate("MR_F60_lr3x", template="F60_lr3x", lr=3e-3, stage_epochs="240,480,780", stage_lr="3e-3,3e-3,3e-3", lr_policy="same_lr", description="strict unified fixed 60% epoch budget with 3x LR"),
        _mr_candidate("MR_F40_lr10x", template="F40_lr10x", lr=1e-2, stage_epochs="160,320,520", stage_lr="1e-2,1e-2,1e-2", lr_policy="same_lr", description="strict unified fixed 40% epoch budget with 10x LR"),
        _mr_candidate("MR_F25_lr30x", template="F25_lr30x", lr=3e-2, stage_epochs="100,200,300", stage_lr="3e-2,3e-2,3e-2", lr_policy="same_lr", description="strict unified fixed 25% epoch budget with 30x LR"),
        _mr_candidate("MR_A_bal_lr10x", template="A_bal_lr10x", lr=1e-2, stage_epochs="400,800,1300", stage_lr="1e-2,1e-2,1e-2", early_stop_mode="adaptive", stage_min_epochs="60,120,195", stage_patience="25,50,80", stage_rel_delta_total="5e-5,5e-5,5e-5", stage_rel_delta_data="2e-5,2e-5,2e-5", lr_policy="same_lr", description="strict unified adaptive balanced with 10x LR"),
        _mr_candidate("MR_A_fast_lr30x", template="A_fast_lr30x", lr=3e-2, stage_epochs="400,800,1300", stage_lr="3e-2,3e-2,3e-2", early_stop_mode="adaptive", stage_min_epochs="40,80,130", stage_patience="20,35,50", stage_rel_delta_total="1e-4,1e-4,1e-4", stage_rel_delta_data="5e-5,5e-5,5e-5", lr_policy="same_lr", description="strict unified adaptive fast with 30x LR"),
        _mr_candidate("MR_F60_decay", template="F60_lr3x_decay", lr=1e-2, stage_epochs="240,480,780", stage_lr="3e-2,1e-2,3e-3", lr_policy="decayed_stage_lr", description="MR decayed-stage fixed 60% epoch budget"),
        _mr_candidate("MR_F40_decay", template="F40_lr10x_decay", lr=1e-2, stage_epochs="160,320,520", stage_lr="3e-2,1e-2,3e-3", lr_policy="decayed_stage_lr", description="MR decayed-stage fixed 40% epoch budget"),
        _mr_candidate("MR_F25_decay", template="F25_lr30x_decay", lr=1e-2, stage_epochs="100,200,300", stage_lr="3e-2,1e-2,3e-3", lr_policy="decayed_stage_lr", description="MR decayed-stage fixed 25% epoch budget"),
        _mr_candidate("MR_A_bal_decay", template="A_bal_lr10x_decay", lr=1e-2, stage_epochs="400,800,1300", stage_lr="3e-2,1e-2,3e-3", early_stop_mode="adaptive", stage_min_epochs="60,120,195", stage_patience="25,50,80", stage_rel_delta_total="5e-5,5e-5,5e-5", stage_rel_delta_data="2e-5,2e-5,2e-5", lr_policy="decayed_stage_lr", description="MR decayed-stage adaptive balanced"),
        _mr_candidate("MR_A_fast_decay", template="A_fast_lr30x_decay", lr=1e-2, stage_epochs="400,800,1300", stage_lr="3e-2,1e-2,3e-3", early_stop_mode="adaptive", stage_min_epochs="40,80,130", stage_patience="20,35,50", stage_rel_delta_total="1e-4,1e-4,1e-4", stage_rel_delta_data="5e-5,5e-5,5e-5", lr_policy="decayed_stage_lr", description="MR decayed-stage adaptive fast"),
        _mr_candidate("MR_A_fast_decay_legacy", template="A_fast_decay_legacy", lr=1e-2, stage_epochs="400,800,1300", stage_lr="3e-2,1e-2,3e-3", early_stop_mode="adaptive", stage_min_epochs="75,150,250", stage_patience="50,75,100", stage_rel_delta_total="1e-4,1e-4,1e-4", stage_rel_delta_data="5e-5,5e-5,5e-5", lr_policy="decayed_stage_lr", description="legacy previous best A_fast_decay settings"),
    ]
    return (
        mr
        + _single_stage("D2", D2_METHOD, "D-BSCP cps2", "D-BSCP", 1e-3, "primary")
        + _single_stage("S2", S2_METHOD, "SINR cps2", "SINR", 1e-4, "primary")
        + _single_stage("D4", D4_METHOD, "D-BSCP cps4", "D-BSCP", 1e-3, "cps4")
        + _single_stage("S4", S4_METHOD, "SINR cps4", "SINR", 1e-4, "cps4")
    )


def candidate_map() -> Dict[str, Candidate]:
    return {candidate.name: candidate for candidate in candidates_all()}


def phase_candidates(phase: str) -> List[Candidate]:
    if phase == "all":
        return candidates_all()
    wanted = "primary" if phase == "primary" else "cps4"
    return [candidate for candidate in candidates_all() if candidate.phase == wanted]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-candidate", type=str, default="")
    parser.add_argument("--phase", choices=["primary", "cps4", "all", "smoke"], default="primary")
    parser.add_argument("--smoke-target-phase", choices=["primary", "cps4", "all"], default="all")
    parser.add_argument("--case-ranks", type=str, default="0-9")
    parser.add_argument("--candidates", type=str, default="phase")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
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


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve unified-screen outputs under the external artifact store."""

    args.out_root = artifact_paths.resolve_run_root(
        RUN_KEY,
        args.out_root,
        repo_root=c.REPO_ROOT,
    )
    args.review_root = artifact_paths.resolve_review_root(
        EXPERIMENT_KEY,
        args.review_root,
        repo_root=c.REPO_ROOT,
    )
    return args


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


def parse_gpus(text: str) -> List[str]:
    return [token.strip() for token in str(text).split(",") if token.strip()]


def selected_candidates(args: argparse.Namespace) -> List[Candidate]:
    if str(args.candidates).strip() not in {"", "phase", "all"}:
        names = [token.strip() for token in str(args.candidates).split(",") if token.strip()]
        known = candidate_map()
        missing = [name for name in names if name not in known]
        if missing:
            raise ValueError(f"Unknown candidates {missing}; known={sorted(known)}")
        return [known[name] for name in names]
    if str(args.phase) == "smoke":
        return phase_candidates(str(args.smoke_target_phase))
    return phase_candidates(str(args.phase))


def phase_candidate(candidate: Candidate, args: argparse.Namespace) -> Candidate:
    if int(args.epoch_override) <= 0:
        return candidate
    if candidate.method == MR_METHOD:
        n_stages = len(c.parse_int_list(candidate.stage_epochs))
        stage_epochs = ",".join([str(int(args.epoch_override))] * n_stages)
        return replace(
            candidate,
            epochs=int(args.epoch_override) * n_stages,
            stage_epochs=stage_epochs,
            stage_min_epochs=stage_epochs if candidate.early_stop_mode == "adaptive" else candidate.stage_min_epochs,
            stage_patience=stage_epochs if candidate.early_stop_mode == "adaptive" else candidate.stage_patience,
            stage_check_interval="1,1,1",
        )
    return replace(
        candidate,
        epochs=int(args.epoch_override),
        min_epochs=min(int(candidate.min_epochs), int(args.epoch_override)) if candidate.early_stop_mode == "adaptive" else 0,
        patience=1 if candidate.early_stop_mode == "adaptive" else 0,
    )


def run_name(candidate: Candidate, rank: int, alpha: float, phase: str) -> str:
    suffix = f"unified_{phase}_{candidate.name}_rank{rank:03d}_a{c.format_float(alpha)}"
    if phase == "smoke":
        suffix += "_smoke"
    return f"{candidate.method}_{suffix}"


def run_dir(args: argparse.Namespace, candidate: Candidate, rank: int) -> Path:
    return Path(args.out_root) / candidate.method / run_name(candidate, rank, float(args.alpha), str(args.phase))


def run_one_argv(args: argparse.Namespace, candidate: Candidate, rank: int, gpu: str) -> List[str]:
    case = c.resolve_case(rank=int(rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
    effective = phase_candidate(candidate, args)
    argv = [
        "run_one.py",
        "--method",
        effective.method,
        "--alpha",
        f"{float(args.alpha):g}",
        "--lr",
        f"{float(effective.lr):g}",
        "--gpu",
        str(gpu),
        "--case-rank",
        str(int(rank)),
        "--epochs",
        str(int(effective.epochs)),
        "--eval-interval",
        str(1 if str(args.phase) == "smoke" else int(args.eval_interval)),
        "--seed",
        str(int(args.seed)),
        "--runtime-candidate",
        effective.name,
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
        run_name(effective, int(rank), float(args.alpha), str(args.phase)),
        "--expected-original-case-index",
        str(case.original_case_index),
        "--expected-fixed-id",
        str(case.fixed_id),
        "--expected-moving-id",
        str(case.moving_id),
        "--stage-early-stop-mode",
        effective.early_stop_mode,
    ]
    if effective.method == MR_METHOD:
        argv.extend(
            [
                "--stage-full-cps",
                effective.stage_full_cps,
                "--stage-downsample",
                effective.stage_downsample,
                "--stage-epochs",
                effective.stage_epochs,
                "--stage-lr",
                effective.stage_lr,
                "--stage-patience",
                effective.stage_patience,
                "--stage-check-interval",
                effective.stage_check_interval,
                "--stage-rel-delta-total",
                effective.stage_rel_delta_total,
                "--stage-rel-delta-data",
                effective.stage_rel_delta_data,
            ]
        )
        if effective.stage_min_epochs:
            argv.extend(["--stage-min-epochs", effective.stage_min_epochs])
    elif effective.early_stop_mode == "adaptive":
        argv.extend(
            [
                "--stage-min-epochs",
                str(int(effective.min_epochs)),
                "--stage-patience",
                str(int(effective.patience)),
                "--stage-check-interval",
                "1",
                "--stage-min-delta",
                f"{float(effective.min_delta):g}",
            ]
        )
    if args.overwrite:
        argv.append("--overwrite")
    return argv


def write_manifests(args: argparse.Namespace, candidates: Sequence[Candidate], ranks: Sequence[int], gpus: Sequence[str]) -> None:
    Path(args.review_root).mkdir(parents=True, exist_ok=True)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    candidate_rows = [asdict(candidate) for candidate in candidates]
    job_rows = []
    for candidate in candidates:
        for rank in ranks:
            case = c.resolve_case(rank=int(rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
            job_rows.append(
                {
                    "candidate": candidate.name,
                    "phase": candidate.phase,
                    "method": candidate.method,
                    "method_label": candidate.method_label,
                    "family": candidate.family,
                    "case_rank": int(rank),
                    "original_case_index": int(case.original_case_index),
                    "fixed_id": case.fixed_id,
                    "moving_id": case.moving_id,
                    "worker_process_model": "persistent_candidate_worker_sequential_cases",
                }
            )
    phase_name = str(args.phase if args.phase != "smoke" else f"smoke_{args.smoke_target_phase}")
    for root in [Path(args.review_root), Path(args.out_root)]:
        c.write_csv(root / f"candidate_manifest_{phase_name}.csv", candidate_rows)
        c.write_json(root / f"candidate_manifest_{phase_name}.json", candidate_rows)
        c.write_csv(root / f"job_manifest_{phase_name}.csv", job_rows)
        c.write_json(
            root / f"job_manifest_{phase_name}.json",
            {
                "created_at": c.now_text(),
                "phase": str(args.phase),
                "smoke_target_phase": str(args.smoke_target_phase),
                "candidate_count": len(candidates),
                "rank_count": len(ranks),
                "registration_count": len(job_rows),
                "gpus": list(gpus),
                "max_concurrent": int(args.max_concurrent),
                "process_model": "one worker per candidate; ranks sequential inside worker",
                "candidates": candidate_rows,
                "jobs": job_rows,
            },
        )


def run_worker(args: argparse.Namespace) -> None:
    known = candidate_map()
    base_candidate = known[str(args.worker_candidate)]
    candidate = phase_candidate(base_candidate, args)
    ranks = parse_case_ranks(str(args.case_ranks))
    log_dir = Path(args.out_root) / "logs" / f"unified_worker_{args.phase}" / candidate.name
    log_dir.mkdir(parents=True, exist_ok=True)
    command_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    for sequence_index, rank in enumerate(ranks):
        target_dir = run_dir(args, candidate, int(rank))
        case = c.resolve_case(rank=int(rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        argv = run_one_argv(args, base_candidate, int(rank), str(parse_gpus(args.gpus)[0]))
        log_path = log_dir / f"{run_name(candidate, int(rank), float(args.alpha), str(args.phase))}.log"
        start = time.perf_counter()
        status = "completed"
        error = ""
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
            "phase": str(args.phase),
            "candidate": candidate.name,
            "method": candidate.method,
            "method_label": candidate.method_label,
            "family": candidate.family,
            "gpu": str(parse_gpus(args.gpus)[0]),
            "case_rank": int(rank),
            "case_sequence_index": int(sequence_index),
            "warm_status": "cold_rank0" if sequence_index == 0 else "warm_persistent",
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
        if status != "completed" or not done:
            failure_rows.append(row)
        c.write_csv(Path(args.out_root) / f"command_timing_unified_{args.phase}_{candidate.name}.csv", command_rows)
    summary = {
        "completed_at": c.now_text(),
        "candidate": candidate.name,
        "phase": str(args.phase),
        "rank_count": len(ranks),
        "command_rows": command_rows,
        "failures": failure_rows,
    }
    c.write_json(Path(args.out_root) / f"worker_summary_unified_{args.phase}_{candidate.name}.json", summary)
    if failure_rows:
        raise RuntimeError(f"Unified worker {candidate.name} had {len(failure_rows)} failures")


def launch_worker(args: argparse.Namespace, candidate: Candidate, gpu: str) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-candidate",
        candidate.name,
        "--phase",
        str(args.phase),
        "--case-ranks",
        str(args.case_ranks),
        "--gpus",
        str(gpu),
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
    log_dir = Path(args.out_root) / "logs" / f"unified_launcher_{args.phase}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"worker_{candidate.name}_gpu{gpu}.log"
    return {"candidate": candidate, "gpu": str(gpu), "cmd": cmd, "log_path": log_path}


def aggregate(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_oasis_unified_acceleration_sweep.py"),
        "--out-root",
        str(args.out_root),
        "--review-root",
        str(args.review_root),
        "--phase",
        str(args.phase),
        "--case-ranks",
        str(args.case_ranks),
        "--candidates",
        ",".join(candidate.name for candidate in selected_candidates(args)),
        "--variant",
        str(args.variant),
    ]
    if args.allow_incomplete_aggregate:
        cmd.append("--allow-incomplete")
    print(json.dumps({"aggregate_cmd": cmd}), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    if args.worker_candidate:
        run_worker(args)
        return
    candidates = selected_candidates(args)
    ranks = parse_case_ranks(str(args.case_ranks))
    gpus = parse_gpus(str(args.gpus))
    max_concurrent = min(max(1, int(args.max_concurrent)), len(gpus))
    write_manifests(args, candidates, ranks, gpus)
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "phase": str(args.phase),
                "candidate_count": len(candidates),
                "rank_count": len(ranks),
                "registration_count": len(candidates) * len(ranks),
                "gpus": gpus,
                "max_concurrent": max_concurrent,
                "process_model": "one persistent worker per candidate; cases sequential inside worker",
                "dry_run": bool(args.dry_run),
                "out_root": str(args.out_root),
                "review_root": str(args.review_root),
            },
            indent=2,
        ),
        flush=True,
    )
    workers = [launch_worker(args, candidate, gpus[idx % len(gpus)]) for idx, candidate in enumerate(candidates)]
    if args.dry_run:
        for worker in workers:
            print(json.dumps({"launch_worker": worker["cmd"], "log": str(worker["log_path"])}), flush=True)
        return
    pending = list(workers)
    running: List[Dict[str, object]] = []
    completed: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    env = thread_env(int(args.cpu_threads))
    started = time.time()
    while pending or running:
        while pending and len(running) < max_concurrent:
            busy = {item["gpu"] for item in running}
            free_gpus = [gpu for gpu in gpus if gpu not in busy]
            if not free_gpus:
                break
            worker = pending.pop(0)
            worker["gpu"] = free_gpus[0]
            worker = launch_worker(args, worker["candidate"], worker["gpu"])  # rebuild cmd with selected GPU
            print(json.dumps({"launch_worker": worker["cmd"], "log": str(worker["log_path"])}), flush=True)
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
            row = {
                "candidate": worker["candidate"].name,
                "gpu": worker["gpu"],
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
    for root in [Path(args.out_root), Path(args.review_root)]:
        c.write_json(
            root / f"launch_summary_unified_{args.phase}.json",
            {
                "completed_at": c.now_text(),
                "phase": str(args.phase),
                "candidate_count": len(candidates),
                "rank_count": len(ranks),
                "registration_count": len(candidates) * len(ranks),
                "completed_workers": completed,
                "failed_workers": failures,
                "elapsed_sec": time.time() - started,
            },
        )
    if failures and not args.allow_incomplete_aggregate:
        raise RuntimeError("Unified acceleration sweep workers failed: " + json.dumps(failures, indent=2))
    if not args.skip_aggregate:
        aggregate(args)


if __name__ == "__main__":
    main()
