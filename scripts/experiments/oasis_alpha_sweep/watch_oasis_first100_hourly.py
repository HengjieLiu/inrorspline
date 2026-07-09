#!/usr/bin/env python
"""Hourly watchdog for the OASIS seed-3 first100 extension launch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import oasis_first100_extension_common as f100  # noqa: E402


WATCHDOG_LOG = f100.REVIEW_ROOT / "hourly_watchdog.md"
WATCHDOG_JSONL = f100.REVIEW_ROOT / "hourly_watchdog_snapshots.jsonl"
WATCHDOG_STDOUT = f100.REVIEW_ROOT / "hourly_watchdog.stdout.log"
PROCESS_SNAPSHOT = f100.REVIEW_ROOT / "production_processes_latest.txt"
RESTART_STATE = f100.REVIEW_ROOT / "hourly_watchdog_restart_state.json"
TOTAL_NEW_JOBS = 3280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=3600.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--restart-cooldown-sec", type=float, default=7200.0)
    parser.add_argument("--no-auto-restart", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: object) -> None:
    c.write_json(path, payload)


def run_monitor_once() -> Dict[str, Any]:
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
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def process_lines() -> List[str]:
    proc = subprocess.run(["ps", "-eo", "pid,ppid,stat,etime,cmd"], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return []
    wanted = [
        "launch_oasis_first100_extension.py",
        "launch_oasis_first100_saved_baselines.py",
        "launch_directcp842_lr1e3_alpha_sweep.py",
        "launch_oasis_first100_single_stage_extension.py",
        "evaluate_saved_baselines_first10.py",
        "run_one.py",
        "monitor_oasis_first100_extension.py",
    ]
    lines = [line for line in proc.stdout.splitlines() if any(token in line for token in wanted)]
    lines = [line for line in lines if "watch_oasis_first100_hourly.py" not in line]
    PROCESS_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    PROCESS_SNAPSHOT.write_text("\n".join(lines) + ("\n" if lines else ""))
    return lines


def top_level_running(lines: Sequence[str]) -> bool:
    return any("launch_oasis_first100_extension.py" in line and "--stage all" in line for line in lines)


def launcher_state_path_pairs() -> List[List[Path]]:
    return [
        [
            f100.BASELINE_EXTENSION_ROOT / "launcher_state.json",
            f100.REVIEW_ROOT / "saved_baselines/launcher_state.json",
        ],
        [
            f100.DCP842_EXTENSION_ROOT / "launcher_state.json",
            f100.REVIEW_ROOT / "directcp842/launcher_state.json",
        ],
        [
            f100.SINGLE_STAGE_EXTENSION_ROOT / "launcher_state.json",
            f100.REVIEW_ROOT / "single_stage/launcher_state.json",
        ],
    ]


def collect_launcher_summary() -> Dict[str, Any]:
    states = []
    for paths in launcher_state_path_pairs():
        candidates = [path for path in paths if path.exists()]
        if not candidates:
            continue
        path = max(candidates, key=lambda item: item.stat().st_mtime)
        state = read_json(path)
        if not state:
            continue
        states.append(
            {
                "path": str(path),
                "job_count": state.get("job_count"),
                "done_count": state.get("done_count"),
                "running_count": state.get("running_count"),
                "pending_count": state.get("pending_count"),
                "failure_count": state.get("failure_count"),
                "updated_at": state.get("updated_at"),
            }
        )
    return {
        "states": states,
        "failure_count": sum(int(item.get("failure_count") or 0) for item in states),
        "running_count": sum(int(item.get("running_count") or 0) for item in states),
    }


def progress_summary() -> Dict[str, Any]:
    snapshot = read_json(f100.TRACKER_ROOT / "progress_snapshot.json")
    summary = snapshot.get("summary")
    return summary if isinstance(summary, dict) else {}


def restart_state() -> Dict[str, Any]:
    return read_json(RESTART_STATE)


def should_restart(
    *,
    args: argparse.Namespace,
    summary: Mapping[str, Any],
    running: bool,
    launcher_summary: Mapping[str, Any],
) -> tuple[bool, str]:
    if args.no_auto_restart:
        return False, "auto restart disabled"
    if running:
        return False, "top-level launcher running"
    done = int(summary.get("new_done_jobs") or 0)
    failures = int(summary.get("failed_jobs") or 0) + int(launcher_summary.get("failure_count") or 0)
    active = int(summary.get("running_jobs") or 0) + int(launcher_summary.get("running_count") or 0)
    if done >= TOTAL_NEW_JOBS and failures == 0:
        return False, "all new jobs complete"
    if active > 0:
        return False, "child jobs still active"
    state = restart_state()
    restart_count = int(state.get("restart_count") or 0)
    last_restart_epoch = float(state.get("last_restart_epoch") or 0.0)
    if restart_count >= int(args.max_restarts):
        return False, f"restart limit reached ({restart_count})"
    if last_restart_epoch and time.time() - last_restart_epoch < float(args.restart_cooldown_sec):
        return False, "restart cooldown active"
    return True, "top-level launcher absent before completion"


def restart_launcher(args: argparse.Namespace, reason: str) -> Dict[str, Any]:
    restart_log = f100.REVIEW_ROOT / f"watchdog_restart_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        sys.executable,
        str(THIS_DIR / "launch_oasis_first100_extension.py"),
        "--stage",
        "all",
        "--gpus",
        "auto",
        "--max-concurrent",
        "0",
    ]
    log_f = restart_log.open("a")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    state = restart_state()
    payload = {
        "restart_count": int(state.get("restart_count") or 0) + 1,
        "last_restart_epoch": time.time(),
        "last_restart_at": c.now_text(),
        "last_restart_reason": reason,
        "last_restart_log": str(restart_log),
        "last_restart_pid": proc.pid,
        "last_restart_cmd": cmd,
    }
    write_json(RESTART_STATE, payload)
    return payload


def write_watchdog_markdown(item: Mapping[str, Any]) -> None:
    summary = item.get("progress_summary") if isinstance(item.get("progress_summary"), Mapping) else {}
    launcher = item.get("launcher_summary") if isinstance(item.get("launcher_summary"), Mapping) else {}
    lines = [
        "# OASIS First100 Hourly Watchdog",
        "",
        f"- Updated: `{item.get('checked_at')}`",
        f"- Top-level launcher running: `{item.get('top_level_running')}`",
        f"- Restart action: `{item.get('restart_action')}`",
        f"- Restart reason: `{item.get('restart_reason')}`",
        f"- Progress log: `{f100.PROGRESS_LOG}`",
        f"- Tracker figure: `{f100.TRACKER_FIGURE}`",
        f"- Process snapshot: `{PROCESS_SNAPSHOT}`",
        "",
        "## Progress",
        "",
        f"- New jobs done: `{summary.get('new_done_jobs', 'unknown')}/{summary.get('new_expected_jobs', TOTAL_NEW_JOBS)}`",
        f"- Running jobs: `{summary.get('running_jobs', 'unknown')}`",
        f"- Failed jobs: `{summary.get('failed_jobs', 'unknown')}`",
        f"- Pending new jobs: `{summary.get('new_pending_jobs', 'unknown')}`",
        f"- Complete first100 groups: `{summary.get('complete_first100_groups', 'unknown')}/{summary.get('group_count', 'unknown')}`",
        "",
        "## Launcher States",
        "",
    ]
    states = launcher.get("states") if isinstance(launcher, Mapping) else []
    if states:
        lines.extend(
            [
                "| Path | Done | Running | Pending | Failed | Updated |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for state in states:
            if not isinstance(state, Mapping):
                continue
            lines.append(
                f"| `{state.get('path')}` | {state.get('done_count')} | {state.get('running_count')} | "
                f"{state.get('pending_count')} | {state.get('failure_count')} | {state.get('updated_at')} |"
            )
    else:
        lines.append("- no launcher state files yet")
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    WATCHDOG_LOG.write_text("\n".join(lines) + "\n")


def snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    monitor = run_monitor_once()
    lines = process_lines()
    running = top_level_running(lines)
    launcher = collect_launcher_summary()
    summary = progress_summary()
    do_restart, reason = should_restart(args=args, summary=summary, running=running, launcher_summary=launcher)
    restart_payload: Dict[str, Any] = {}
    if do_restart:
        restart_payload = restart_launcher(args, reason)
    item = {
        "checked_at": c.now_text(),
        "checked_epoch": time.time(),
        "monitor": monitor,
        "top_level_running": running,
        "process_count": len(lines),
        "process_lines": lines,
        "launcher_summary": launcher,
        "progress_summary": summary,
        "restart_action": "started" if do_restart else "none",
        "restart_reason": reason,
        "restart": restart_payload,
    }
    WATCHDOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with WATCHDOG_JSONL.open("a") as f:
        f.write(json.dumps(item, sort_keys=True, allow_nan=True, default=str) + "\n")
    write_watchdog_markdown(item)
    return item


def main() -> None:
    args = parse_args()
    while True:
        item = snapshot(args)
        print(json.dumps({key: item[key] for key in ["checked_at", "top_level_running", "restart_action", "restart_reason", "progress_summary"]}, indent=2), flush=True)
        summary = item.get("progress_summary")
        done = int(summary.get("new_done_jobs") or 0) if isinstance(summary, Mapping) else 0
        failed = int(summary.get("failed_jobs") or 0) if isinstance(summary, Mapping) else 0
        if not args.watch or (done >= TOTAL_NEW_JOBS and failed == 0):
            break
        time.sleep(float(args.interval_sec))


if __name__ == "__main__":
    main()
