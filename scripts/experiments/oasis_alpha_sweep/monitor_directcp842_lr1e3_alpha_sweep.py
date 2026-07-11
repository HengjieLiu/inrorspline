#!/usr/bin/env python
"""Write progress snapshots for the OASIS DirectCP 8-4-2 lr=1e-3 sweep."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


METHOD = "direct_cp_8_4_2_be_bspline_analytic_multistage"
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_v1")
DEFAULT_REVIEW_ROOT = Path(
    "/homebase/code_sync/inrorspline/experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--expected-jobs", type=int, default=0)
    parser.add_argument("--active-log-window-sec", type=float, default=900.0)
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def utc_text(epoch: float | None = None) -> str:
    if epoch is None:
        epoch = time.time()
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="seconds")


def fmt_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def load_expected_jobs(out_root: Path, review_root: Path, explicit: int) -> int:
    if explicit > 0:
        return int(explicit)
    for path in [out_root / "job_manifest.json", review_root / "job_manifest.json"]:
        payload = read_json(path)
        if payload.get("job_count"):
            return int(payload["job_count"])
    return 0


def newest_log(log_dir: Path) -> tuple[str, float | None]:
    if not log_dir.exists():
        return "", None
    logs = [path for path in log_dir.glob("*.log") if path.is_file()]
    if not logs:
        return "", None
    latest = max(logs, key=lambda path: path.stat().st_mtime)
    return str(latest), float(latest.stat().st_mtime)


def active_logs(log_dir: Path, done_dirs: set[str], window_sec: float) -> list[str]:
    if not log_dir.exists():
        return []
    cutoff = time.time() - float(window_sec)
    out: list[str] = []
    for path in sorted(log_dir.glob("*.log")):
        if path.stat().st_mtime < cutoff:
            continue
        text = path.name
        if any(done_name in text for done_name in done_dirs):
            continue
        out.append(str(path))
    return out


def snapshot(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    review_root = Path(args.review_root)
    state = read_json(review_root / "launcher_state.json") or read_json(out_root / "launcher_state.json")
    summary = read_json(review_root / "launch_summary.json") or read_json(out_root / "launch_summary.json")
    expected = load_expected_jobs(out_root, review_root, int(args.expected_jobs))
    method_root = out_root / METHOD
    done_paths = sorted(method_root.glob("*/DONE")) if method_root.exists() else []
    done_count = len(done_paths)
    done_dirs = {path.parent.name for path in done_paths}
    failures = state.get("failures") or summary.get("failures") or []
    running_from_state = state.get("running") or []
    log_path, log_epoch = newest_log(out_root / "logs")
    active_log_paths = active_logs(out_root / "logs", done_dirs, float(args.active_log_window_sec))
    if state.get("running_count") is not None:
        running_count = int(state["running_count"])
    elif running_from_state:
        running_count = len(running_from_state)
    else:
        running_count = len(active_log_paths)
    failed_count = len(failures)
    remaining = max(expected - done_count - failed_count, 0) if expected else 0
    started_epoch = state.get("started_epoch")
    if started_epoch is None:
        started_epoch = summary.get("started_epoch")
    now = time.time()
    elapsed = now - float(started_epoch) if started_epoch else float("nan")
    jobs_per_hour = done_count / (elapsed / 3600.0) if elapsed and math.isfinite(elapsed) and elapsed > 0 else 0.0
    eta_sec = remaining / jobs_per_hour * 3600.0 if jobs_per_hour > 0 and remaining > 0 else 0.0
    return {
        "snapshot_utc": utc_text(now),
        "out_root": str(out_root),
        "review_root": str(review_root),
        "expected_jobs": expected,
        "completed_jobs": done_count,
        "failed_jobs": failed_count,
        "running_jobs": running_count,
        "remaining_jobs": remaining,
        "elapsed_sec": elapsed if math.isfinite(elapsed) else None,
        "elapsed_text": fmt_duration(elapsed),
        "jobs_per_hour": jobs_per_hour,
        "eta_sec": eta_sec,
        "eta_text": fmt_duration(eta_sec) if remaining else "complete",
        "newest_log_path": log_path,
        "newest_log_update_utc": utc_text(log_epoch) if log_epoch else "",
        "active_log_paths": active_log_paths,
        "failure_log_paths": [str(item.get("log_path", "")) for item in failures if isinstance(item, Mapping)],
        "launcher_pending_count": state.get("pending_count"),
        "launcher_running_count": state.get("running_count"),
        "launcher_failure_count": state.get("failure_count"),
    }


def write_markdown(path: Path, item: Mapping[str, Any]) -> None:
    lines = [
        "# OASIS DirectCP 8-4-2 lr=1e-3 Sweep Monitor",
        "",
        f"- Snapshot UTC: `{item['snapshot_utc']}`",
        f"- Completed: `{item['completed_jobs']}/{item['expected_jobs']}`",
        f"- Running: `{item['running_jobs']}`",
        f"- Failed: `{item['failed_jobs']}`",
        f"- Remaining: `{item['remaining_jobs']}`",
        f"- Elapsed: `{item['elapsed_text']}`",
        f"- Throughput: `{float(item['jobs_per_hour']):.3f}` jobs/hour",
        f"- ETA: `{item['eta_text']}`",
        f"- Newest log: `{item['newest_log_path']}`",
        f"- Newest log update UTC: `{item['newest_log_update_utc']}`",
        "",
        "## Failure Logs",
        "",
    ]
    failure_logs = list(item.get("failure_log_paths") or [])
    if failure_logs:
        lines.extend(f"- `{path}`" for path in failure_logs)
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    item = snapshot(args)
    args.review_root.mkdir(parents=True, exist_ok=True)
    if args.append:
        with (args.review_root / "monitor_snapshots.jsonl").open("a") as f:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    else:
        (args.review_root / "monitor_snapshots.jsonl").write_text(json.dumps(item, sort_keys=True) + "\n")
    write_markdown(args.review_root / "monitor_latest.md", item)
    print(json.dumps(item, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
