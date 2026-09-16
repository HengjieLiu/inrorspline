#!/usr/bin/env python
"""Sidecar monitor for the exploratory SINR/DirectCP mechanism run.

This intentionally does not launch or stop jobs. It watches the active
schedule-ablation manifest/status, refreshes partial review artifacts after
completed rows appear, and appends short Scientist/Reviewer rounds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import explore_mechanism_followup as explore


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return {"read_error": str(exc), "path": str(path)}
    return payload if isinstance(payload, dict) else {"payload": payload}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str) + "\n")


def finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def aggregation_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        out_root=Path(args.out_root),
        review_root=Path(args.review_root),
        update_root=Path(args.update_root),
        gpus=args.gpus,
    )


def summarize_completed(rows: List[Dict[str, object]]) -> Dict[str, object]:
    completed = [row for row in rows if row.get("status") == "completed"]
    by_case: Dict[str, int] = {}
    for row in completed:
        key = str(row.get("case_id", "unknown"))
        by_case[key] = by_case.get(key, 0) + 1
    best = None
    for row in completed:
        tre = finite_float(row.get("reg1mm_snap_tre_mean_mm"))
        if tre is None:
            continue
        if best is None or tre < float(best["reg1mm_snap_tre_mean_mm"]):
            best = {
                "case_id": row.get("case_id"),
                "variant": row.get("variant"),
                "run_name": row.get("run_name"),
                "reg1mm_snap_tre_mean_mm": tre,
                "reg1mm_subvoxel_tre_mean_mm": finite_float(row.get("reg1mm_subvoxel_tre_mean_mm")),
            }
    return {"completed_rows": len(completed), "completed_by_case": by_case, "best_completed": best}


def status_payload(args: argparse.Namespace) -> Dict[str, object]:
    agg_args = aggregation_args(args)
    status_path = Path(args.out_root) / "schedule_ablation" / "phase_status.json"
    manifest_path = Path(args.out_root) / "schedule_ablation" / "job_manifest.json"
    schedule_rows = explore.collect_schedule_rows(agg_args)
    completed_summary = summarize_completed(schedule_rows)
    status = read_json(status_path)
    manifest = read_json(manifest_path)
    return {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status_path": str(status_path),
        "manifest_job_count": manifest.get("job_count"),
        "phase_status": status,
        "schedule_row_count": len(schedule_rows),
        **completed_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--update-root", type=Path, default=explore.DEFAULT_UPDATE_ROOT)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--poll-sec", type=float, default=900.0)
    parser.add_argument("--max-runtime-sec", type=float, default=14400.0)
    parser.add_argument("--aggregate-empty", action="store_true")
    parser.add_argument("--exit-when-finished", action="store_true")
    args = explore.resolve_artifact_paths(parser.parse_args())

    review_root = Path(args.review_root)
    log_path = review_root / "scientist_reviewer_log.md"
    live_status_path = review_root / "exploration" / "live_monitor_status.json"
    agg_args = aggregation_args(args)

    explore.log_round(
        log_path,
        role="Reviewer",
        content=(
            "Sidecar monitor started. It must not stop or relaunch jobs; it should "
            "refresh partial artifacts only when completed schedule rows exist."
        ),
    )
    started = time.time()
    last_completed = -1
    last_failures = -1
    last_status = ""

    while True:
        payload = status_payload(args)
        write_json(live_status_path, payload)
        phase = payload.get("phase_status", {})
        if not isinstance(phase, dict):
            phase = {}
        completed = int(phase.get("completed", 0) or 0)
        failures = int(phase.get("failures", 0) or 0)
        status = str(phase.get("status", "unknown"))

        if completed > 0 or args.aggregate_empty:
            try:
                summary = explore.aggregate(agg_args)
                payload["aggregation"] = summary
                write_json(live_status_path, payload)
            except Exception as exc:
                payload["aggregation_error"] = str(exc)
                write_json(live_status_path, payload)
                explore.log_round(log_path, role="Scientist", content=f"Partial aggregation failed: `{exc}`.")

        if completed != last_completed or failures != last_failures or status != last_status:
            best = payload.get("best_completed")
            best_text = f" Best completed row so far: `{best}`." if best else ""
            explore.log_round(
                log_path,
                role="Scientist",
                content=(
                    f"Monitor status: `{status}`, completed `{completed}`, failures `{failures}`, "
                    f"running `{phase.get('running')}`, pending `{phase.get('pending')}`.{best_text}"
                ),
            )
            if failures > last_failures and failures > 0:
                explore.log_round(
                    log_path,
                    role="Reviewer",
                    content=(
                        "A schedule-ablation failure appeared. Continue collecting partial rows "
                        "because the run was launched with `--allow-partial`; interpret failures "
                        "as operational evidence unless they reveal a systematic protocol bug."
                    ),
                )
            last_completed = completed
            last_failures = failures
            last_status = status

        if args.exit_when_finished and status == "finished":
            break
        if args.max_runtime_sec > 0 and time.time() - started >= args.max_runtime_sec:
            break
        time.sleep(max(30.0, float(args.poll_sec)))

    explore.log_round(log_path, role="Reviewer", content="Sidecar monitor exiting after its supervision window.")


if __name__ == "__main__":
    main()
