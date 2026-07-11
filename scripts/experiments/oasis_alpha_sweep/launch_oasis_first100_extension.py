#!/usr/bin/env python
"""Top-level launcher for the OASIS seed-3 first100 extension."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import oasis_first100_extension_common as f100  # noqa: E402


ORDERED_STAGES = ["saved_baselines", "directcp842", "single_stage", "aggregate"]
LAUNCH_STAGES = {"saved_baselines", "directcp842", "single_stage"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Comma-separated stages: all, saved_baselines, directcp842, single_stage, aggregate, monitor.",
    )
    parser.add_argument("--gpus", type=str, default="auto")
    parser.add_argument("--case-ranks", type=str, default="20-99")
    parser.add_argument("--max-concurrent", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--launch-stagger-sec", type=float, default=2.0)
    parser.add_argument("--baseline-methods", type=str, default=",".join(f100.BASELINE_METHODS))
    parser.add_argument(
        "--single-stage-methods",
        type=str,
        default=",".join(spec.method_key for spec in f100.SINGLE_STAGE_METHOD_SPECS),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-incomplete-aggregate", action="store_true")
    parser.add_argument("--update-tracker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-monitor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monitor-interval-sec", type=float, default=300.0)
    parser.add_argument("--monitor-watch", action="store_true", help="Only meaningful with --stage monitor.")
    return parser.parse_args()


def parse_stages(text: str) -> List[str]:
    requested = [token.strip() for token in str(text).split(",") if token.strip()]
    if not requested:
        raise ValueError("No stages requested.")
    if "all" in requested:
        return list(ORDERED_STAGES)
    out: List[str] = []
    allowed = set(ORDERED_STAGES) | {"monitor"}
    for stage in requested:
        if stage not in allowed:
            raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(allowed | {'all'})}")
        if stage not in out:
            out.append(stage)
    return out


def append_common_launch_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--gpus",
            str(args.gpus),
            "--case-ranks",
            str(args.case_ranks),
            "--max-concurrent",
            str(args.max_concurrent),
            "--cpu-threads",
            str(args.cpu_threads),
            "--launch-stagger-sec",
            str(args.launch_stagger_sec),
        ]
    )
    if args.dry_run:
        cmd.append("--dry-run")
    if args.overwrite:
        cmd.append("--overwrite")


def saved_baseline_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(THIS_DIR / "launch_oasis_first100_saved_baselines.py"),
        "--methods",
        str(args.baseline_methods),
        "--variant",
        f100.BASELINE_EXTENSION_VARIANT,
        "--out-root",
        str(f100.BASELINE_EXTENSION_ROOT),
        "--review-root",
        str(f100.REVIEW_ROOT / "saved_baselines"),
    ]
    append_common_launch_args(cmd, args)
    return cmd


def directcp842_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(THIS_DIR / "launch_directcp842_lr1e3_alpha_sweep.py"),
        "--alphas",
        ",".join(f"{alpha:g}" for alpha in f100.SPLINE_ALPHAS),
        "--lr",
        f"{float(f100.DCP842_LR):g}",
        "--variant",
        f100.DCP842_EXTENSION_VARIANT,
        "--out-root",
        str(f100.DCP842_EXTENSION_ROOT),
        "--review-root",
        str(f100.REVIEW_ROOT / "directcp842"),
        "--stage-full-cps",
        f100.DCP842_STAGE_FULL_CPS,
        "--stage-downsample",
        f100.DCP842_STAGE_DOWNSAMPLE,
        "--stage-epochs",
        f100.DCP842_STAGE_EPOCHS,
        "--epochs",
        str(sum(c.parse_int_list(f100.DCP842_STAGE_EPOCHS))),
        "--eval-interval",
        "250",
        "--seed",
        "1",
        "--skip-aggregate",
    ]
    append_common_launch_args(cmd, args)
    return cmd


def single_stage_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(THIS_DIR / "launch_oasis_first100_single_stage_extension.py"),
        "--methods",
        str(args.single_stage_methods),
        "--variant",
        f100.SINGLE_STAGE_EXTENSION_VARIANT,
        "--out-root",
        str(f100.SINGLE_STAGE_EXTENSION_ROOT),
        "--review-root",
        str(f100.REVIEW_ROOT / "single_stage"),
        "--epochs",
        "2500",
        "--eval-interval",
        "250",
        "--seed",
        "1",
    ]
    append_common_launch_args(cmd, args)
    return cmd


def aggregate_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_oasis_first100_extension.py"),
        "--review-root",
        str(f100.FIRST100_ARTIFACT_ROOT),
    ]
    if args.allow_incomplete_aggregate:
        cmd.append("--allow-incomplete")
    return cmd


def monitor_cmd(args: argparse.Namespace, *, watch: bool = False) -> List[str]:
    cmd = [
        sys.executable,
        str(THIS_DIR / "monitor_oasis_first100_extension.py"),
        "--review-root",
        str(f100.REVIEW_ROOT),
        "--progress-log",
        str(f100.PROGRESS_LOG),
        "--tracker-root",
        str(f100.TRACKER_ROOT),
        "--tracker-figure",
        str(f100.TRACKER_FIGURE),
        "--interval-sec",
        str(args.monitor_interval_sec),
    ]
    if watch:
        cmd.append("--watch")
    return cmd


def stage_command(stage: str, args: argparse.Namespace) -> List[str]:
    if stage == "saved_baselines":
        return saved_baseline_cmd(args)
    if stage == "directcp842":
        return directcp842_cmd(args)
    if stage == "single_stage":
        return single_stage_cmd(args)
    if stage == "aggregate":
        return aggregate_cmd(args)
    if stage == "monitor":
        return monitor_cmd(args, watch=bool(args.monitor_watch))
    raise ValueError(f"Unknown stage {stage}")


def print_command(stage: str, cmd: Sequence[str], *, dry_run_only: bool = False) -> None:
    payload = {"stage": stage, "cmd": list(cmd)}
    if dry_run_only:
        payload["dry_run_only"] = True
    print(json.dumps(payload, indent=2), flush=True)


def run_command(stage: str, cmd: Sequence[str], *, dry_run_only: bool = False) -> None:
    print_command(stage, cmd, dry_run_only=dry_run_only)
    if dry_run_only:
        return
    subprocess.check_call(list(cmd))


def maybe_update_tracker(args: argparse.Namespace) -> None:
    if not args.update_tracker:
        return
    run_command("monitor", monitor_cmd(args, watch=False))


def start_live_monitor(args: argparse.Namespace, stages: Iterable[str]) -> subprocess.Popen | None:
    if args.dry_run or not args.live_monitor or not any(stage in LAUNCH_STAGES for stage in stages):
        return None
    f100.TRACKER_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = f100.TRACKER_ROOT / "monitor_watch.log"
    log_f = log_path.open("a")
    cmd = monitor_cmd(args, watch=True)
    print(json.dumps({"stage": "monitor_watch", "cmd": cmd, "log": str(log_path)}, indent=2), flush=True)
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    proc._codex_log_file = log_f  # type: ignore[attr-defined]
    return proc


def stop_live_monitor(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
    log_f = getattr(proc, "_codex_log_file", None)
    if log_f is not None:
        log_f.close()


def main() -> None:
    args = parse_args()
    stages = parse_stages(str(args.stage))
    f100.REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "started_at": c.now_text(),
                "stages": stages,
                "case_ranks": str(args.case_ranks),
                "gpus": str(args.gpus),
                "max_concurrent": int(args.max_concurrent),
                "dry_run": bool(args.dry_run),
                "review_root": str(f100.REVIEW_ROOT),
                "progress_log": str(f100.PROGRESS_LOG),
                "tracker_figure": str(f100.TRACKER_FIGURE),
            },
            indent=2,
        ),
        flush=True,
    )
    if stages == ["monitor"]:
        run_command("monitor", monitor_cmd(args, watch=bool(args.monitor_watch)))
        return

    monitor_proc = start_live_monitor(args, stages)
    try:
        maybe_update_tracker(args)
        for stage in stages:
            cmd = stage_command(stage, args)
            run_command(stage, cmd, dry_run_only=bool(args.dry_run and stage == "aggregate"))
            maybe_update_tracker(args)
    finally:
        stop_live_monitor(monitor_proc)
        maybe_update_tracker(args)


if __name__ == "__main__":
    main()
