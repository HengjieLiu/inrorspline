#!/usr/bin/env python
"""Hourly completion watcher that generates the final OASIS20/OASIS100 figure once."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import oasis_first100_extension_common as f100  # noqa: E402


TOTAL_NEW_JOBS = 3280
TOTAL_FIRST100_GROUPS = 41
WATCH_LOG = f100.REVIEW_ROOT / "completion_figure_watch.md"
WATCH_JSONL = f100.REVIEW_ROOT / "completion_figure_watch_snapshots.jsonl"
WATCH_STATE = f100.REVIEW_ROOT / "completion_figure_watch_state.json"
TRIGGER_LOG = f100.REVIEW_ROOT / "completion_figure_trigger.log"
FINAL_ARTIFACT_DIR = (
    c.REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v002_candidate/oasis20_vs_oasis100_complete_all_methods"
)
FINAL_POINTER_DIR = f100.REVIEW_ROOT / "final_completion_figures/oasis20_vs_oasis100_complete_all_methods"
FINAL_MANIFEST = FINAL_ARTIFACT_DIR / "oasis20_vs_oasis100_complete_pareto_2x2_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=3600.0)
    parser.add_argument("--force-trigger", action="store_true")
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


def progress_summary() -> Dict[str, Any]:
    snapshot = read_json(f100.TRACKER_ROOT / "progress_snapshot.json")
    summary = snapshot.get("summary")
    return summary if isinstance(summary, dict) else {}


def completion_status(summary: Mapping[str, Any]) -> Dict[str, Any]:
    new_expected = int(summary.get("new_expected_jobs") or TOTAL_NEW_JOBS)
    new_done = int(summary.get("new_done_jobs") or 0)
    groups = int(summary.get("group_count") or TOTAL_FIRST100_GROUPS)
    complete_groups = int(summary.get("complete_first100_groups") or 0)
    running = int(summary.get("running_jobs") or 0)
    failed = int(summary.get("failed_jobs") or 0)
    pending = int(summary.get("new_pending_jobs") or 0)
    complete = (
        new_expected == TOTAL_NEW_JOBS
        and new_done >= TOTAL_NEW_JOBS
        and groups == TOTAL_FIRST100_GROUPS
        and complete_groups >= TOTAL_FIRST100_GROUPS
        and running == 0
        and failed == 0
        and pending == 0
    )
    return {
        "complete": complete,
        "new_expected_jobs": new_expected,
        "new_done_jobs": new_done,
        "group_count": groups,
        "complete_first100_groups": complete_groups,
        "running_jobs": running,
        "failed_jobs": failed,
        "new_pending_jobs": pending,
    }


def append_trigger_log(text: str) -> None:
    TRIGGER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TRIGGER_LOG.open("a") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def run_logged(cmd: list[str], *, label: str) -> Dict[str, Any]:
    append_trigger_log(f"\n## {label} {c.now_text()}\n")
    append_trigger_log(json.dumps({"cmd": cmd}, indent=2) + "\n")
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    append_trigger_log(f"returncode: {proc.returncode}\n")
    if proc.stdout:
        append_trigger_log("stdout:\n" + proc.stdout[-12000:] + "\n")
    if proc.stderr:
        append_trigger_log("stderr:\n" + proc.stderr[-12000:] + "\n")
    return {
        "label": label,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def trigger_final_figure(*, force: bool) -> Dict[str, Any]:
    state = read_json(WATCH_STATE)
    if state.get("trigger_attempted") and not force:
        return {
            "action": "already_attempted",
            "state": state,
        }
    started = c.now_text()
    trigger_results = []
    aggregate_cmd = [
        sys.executable,
        str(THIS_DIR / "aggregate_oasis_first100_extension.py"),
        "--review-root",
        str(f100.FIRST100_ARTIFACT_ROOT),
    ]
    figure_cmd = [
        sys.executable,
        str(THIS_DIR / "plot_oasis20_vs_oasis100_after_dbscp.py"),
        "--method-set",
        "complete-all",
    ]
    trigger_results.append(run_logged(aggregate_cmd, label="strict_first100_aggregate"))
    if int(trigger_results[-1]["returncode"]) == 0:
        trigger_results.append(run_logged(figure_cmd, label="complete_all_2x2_figure"))
    success = all(int(result.get("returncode", 1)) == 0 for result in trigger_results) and FINAL_MANIFEST.exists()
    payload = {
        "trigger_attempted": True,
        "trigger_started_at": started,
        "trigger_finished_at": c.now_text(),
        "success": success,
        "trigger_log": str(TRIGGER_LOG),
        "final_artifact_dir": str(FINAL_ARTIFACT_DIR),
        "final_pointer_dir": str(FINAL_POINTER_DIR),
        "final_manifest": str(FINAL_MANIFEST),
        "results": trigger_results,
    }
    write_json(WATCH_STATE, payload)
    return {
        "action": "succeeded" if success else "failed",
        "state": payload,
    }


def write_watch_markdown(item: Mapping[str, Any]) -> None:
    status = item.get("completion") if isinstance(item.get("completion"), Mapping) else {}
    trigger = item.get("trigger") if isinstance(item.get("trigger"), Mapping) else {}
    state = trigger.get("state") if isinstance(trigger.get("state"), Mapping) else read_json(WATCH_STATE)
    lines = [
        "# OASIS First100 Completion Figure Watcher",
        "",
        f"- Updated: `{item.get('checked_at')}`",
        f"- Complete: `{status.get('complete')}`",
        f"- Trigger action: `{trigger.get('action', 'none')}`",
        f"- Trigger success: `{state.get('success', False)}`",
        f"- Trigger log: `{TRIGGER_LOG}`",
        f"- Final artifact dir: `{FINAL_ARTIFACT_DIR}`",
        f"- Final pointer dir: `{FINAL_POINTER_DIR}`",
        "",
        "## Completion Criteria",
        "",
        f"- New jobs done: `{status.get('new_done_jobs')}/{status.get('new_expected_jobs')}`",
        f"- First100 groups complete: `{status.get('complete_first100_groups')}/{status.get('group_count')}`",
        f"- Running jobs: `{status.get('running_jobs')}`",
        f"- Failed jobs: `{status.get('failed_jobs')}`",
        f"- Pending new jobs: `{status.get('new_pending_jobs')}`",
        "",
    ]
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    WATCH_LOG.write_text("\n".join(lines) + "\n")


def snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    monitor = run_monitor_once()
    summary = progress_summary()
    completion = completion_status(summary)
    trigger: Dict[str, Any] = {"action": "none"}
    if bool(completion["complete"]):
        trigger = trigger_final_figure(force=bool(args.force_trigger))
    item = {
        "checked_at": c.now_text(),
        "checked_epoch": time.time(),
        "monitor": monitor,
        "progress_summary": summary,
        "completion": completion,
        "trigger": trigger,
    }
    WATCH_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with WATCH_JSONL.open("a") as f:
        f.write(json.dumps(item, sort_keys=True, allow_nan=True, default=str) + "\n")
    write_watch_markdown(item)
    return item


def main() -> None:
    args = parse_args()
    while True:
        item = snapshot(args)
        print(
            json.dumps(
                {
                    "checked_at": item["checked_at"],
                    "completion": item["completion"],
                    "trigger": {"action": item["trigger"].get("action")},
                },
                indent=2,
                allow_nan=True,
                default=str,
            ),
            flush=True,
        )
        trigger_action = str(item["trigger"].get("action"))
        if not args.watch:
            break
        if bool(item["completion"].get("complete")) and trigger_action in {"succeeded", "failed", "already_attempted"}:
            break
        time.sleep(float(args.interval_sec))


if __name__ == "__main__":
    main()
