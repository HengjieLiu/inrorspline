#!/usr/bin/env python3
"""Stop the ad-hoc PL1B backfill worker before it collides with the main queue."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import signal
import subprocess
import time
from pathlib import Path


ORDER = [
    (tag, case)
    for tag in ("pl1b_m0p01", "pl1b_m0p1", "pl1b_m10", "pl1b_m100")
    for case in range(1, 11)
]
ORDER_INDEX = {item: index for index, item in enumerate(ORDER)}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{now()}] {message}\n")


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_pl1b_job(cmd: str) -> tuple[str, int] | None:
    tag = re.search(r"--run-tag\s+(pl1b_[^\s]+)", cmd)
    case = re.search(r"--case-id\s+(\d+)", cmd)
    if not tag or not case:
        return None
    return tag.group(1), int(case.group(1))


def active_pl1b_jobs(wrapper_pid: int) -> tuple[tuple[str, int] | None, tuple[str, int] | None, int | None]:
    output = subprocess.check_output(
        ["ps", "-eo", "pid,ppid,cmd"],
        text=True,
        errors="replace",
    )
    main: tuple[str, int] | None = None
    backfill: tuple[str, int] | None = None
    backfill_pid: int | None = None
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid_text, ppid_text, cmd = parts
        if "--variant PL1B" not in cmd or "--run-tag pl1b_" not in cmd:
            continue
        job = parse_pl1b_job(cmd)
        if job is None:
            continue
        try:
            pid = int(pid_text)
            ppid = int(ppid_text)
        except ValueError:
            continue
        if ppid == wrapper_pid:
            backfill = job
            backfill_pid = pid
        else:
            main = job
    return main, backfill, backfill_pid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    args = parser.parse_args()

    append_log(
        args.log,
        (
            "guard started "
            f"wrapper={args.wrapper_pid} threshold={args.threshold} "
            f"interval={args.interval_seconds}s"
        ),
    )
    while process_exists(args.wrapper_pid):
        main_job, backfill_job, backfill_pid = active_pl1b_jobs(args.wrapper_pid)
        if main_job and backfill_job:
            gap = ORDER_INDEX[backfill_job] - ORDER_INDEX[main_job]
            append_log(args.log, f"main={main_job} backfill={backfill_job} gap={gap}")
            if gap <= args.threshold:
                append_log(
                    args.log,
                    (
                        f"gap <= {args.threshold}; stopping backfill "
                        f"wrapper={args.wrapper_pid} child={backfill_pid}"
                    ),
                )
                if backfill_pid is not None:
                    os.kill(backfill_pid, signal.SIGTERM)
                os.kill(args.wrapper_pid, signal.SIGTERM)
                return
        else:
            append_log(
                args.log,
                f"waiting for both active jobs main={main_job} backfill={backfill_job}",
            )
        time.sleep(args.interval_seconds)
    append_log(args.log, "guard exiting; backfill wrapper is gone")


if __name__ == "__main__":
    main()
