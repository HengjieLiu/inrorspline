#!/usr/bin/env python
"""Update the OASIS seed-3 first100 progress log and tracker figure."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import oasis_first100_extension_common as f100  # noqa: E402
import plot_first10_first20_alpha5_saved_baselines as p  # noqa: E402
import plot_first10_first20_alpha5_saved_baselines_plus_directcp842_lr1e3 as plus  # noqa: E402


GroupKey = Tuple[str, str]
STATUS_COLORS = {
    "done": "#2E7D32",
    "running": "#1976D2",
    "failed": "#C62828",
    "pending": "#757575",
}


@dataclass(frozen=True)
class ExpectedGroup:
    stage: str
    method_key: str
    label: str
    alpha: float | None
    lr: float | None
    expected_cases: int = 100
    expected_new_jobs: int = 80


@dataclass
class GroupStatus:
    stage: str
    method_key: str
    label: str
    alpha: float | None
    lr: float | None
    expected_cases: int
    done_cases: int
    new_done_cases: int
    running_jobs: int
    failed_jobs: int
    pending_cases: int
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=f100.REVIEW_ROOT)
    parser.add_argument("--progress-log", type=Path, default=f100.PROGRESS_LOG)
    parser.add_argument("--tracker-root", type=Path, default=f100.TRACKER_ROOT)
    parser.add_argument("--tracker-figure", type=Path, default=f100.TRACKER_FIGURE)
    parser.add_argument("--baseline-first20-root", type=Path, default=f100.FIRST20_BASELINE_ROOT)
    parser.add_argument("--baseline-extension-root", type=Path, default=f100.BASELINE_EXTENSION_ROOT)
    parser.add_argument("--single-stage-extension-root", type=Path, default=f100.SINGLE_STAGE_EXTENSION_ROOT)
    parser.add_argument("--directcp842-first20-root", type=Path, default=f100.FIRST20_DCP842_ROOT)
    parser.add_argument("--directcp842-extension-root", type=Path, default=f100.DCP842_EXTENSION_ROOT)
    parser.add_argument("--min-partial-cases", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=300.0)
    return parser.parse_args()


def alpha_text(alpha: float | None) -> str:
    return "baseline" if alpha is None else f"{float(alpha):g}"


def alpha_key(alpha: object, *, baseline: bool = False) -> str:
    return "baseline" if baseline else p.alpha_id(alpha)


def group_key(method_key: str, alpha: float | None) -> GroupKey:
    return str(method_key), alpha_key(alpha, baseline=alpha is None)


def row_group_key(row: Mapping[str, object]) -> GroupKey:
    method = str(row.get("method_key"))
    if method in p.BASELINE_METHODS:
        return group_key(method, None)
    return group_key(method, p.as_float(row.get("alpha")))


def job_group_key(job: Mapping[str, object]) -> GroupKey:
    method = str(job.get("method_key") or job.get("method"))
    if method in p.BASELINE_METHODS:
        return group_key(method, None)
    return group_key(method, p.as_float(job.get("alpha")))


def expected_groups() -> List[ExpectedGroup]:
    groups: List[ExpectedGroup] = []
    baseline_labels = {
        "bs1_vxm_v0_2": "VoxelMorph",
        "bs2_tm_v0": "TransMorph",
        "bs3_vfa_v0": "VFA",
        "bs4_sitreg_v0": "SITReg",
        "greedy": "Greedy",
    }
    for method in f100.BASELINE_METHODS:
        groups.append(
            ExpectedGroup(
                stage="reference eval",
                method_key=method,
                label=baseline_labels.get(method, method),
                alpha=None,
                lr=None,
            )
        )
    for alpha in f100.SPLINE_ALPHAS:
        groups.append(
            ExpectedGroup(
                stage="MR-D-BSCP 8-4-2",
                method_key=f100.DCP842_METHOD,
                label=f100.DCP842_LABEL,
                alpha=float(alpha),
                lr=f100.DCP842_LR,
            )
        )
    for spec in f100.SINGLE_STAGE_METHOD_SPECS:
        for alpha in spec.alphas:
            groups.append(
                ExpectedGroup(
                    stage=spec.display_name,
                    method_key=spec.method_key,
                    label=spec.display_name,
                    alpha=float(alpha),
                    lr=float(spec.lr),
                )
            )
    return groups


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def newest_json(paths: Iterable[Path]) -> Dict[str, Any]:
    candidates = [Path(path) for path in paths if Path(path).exists()]
    if not candidates:
        return {}
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return read_json(newest)


def collect_summary_rows_safe(root: Path, ranks: Sequence[int] | None = None) -> List[Dict[str, object]]:
    root = Path(root)
    if not root.exists():
        return []
    allowed = None if ranks is None else {int(rank) for rank in ranks}
    rows: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/*/summary.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload["summary_path"] = str(path)
        payload["run_dir_actual"] = str(path.parent)
        payload.setdefault("run_dir", str(path.parent))
        payload.setdefault("run_name", path.parent.name)
        payload["method_key"] = str(payload.get("method_key") or payload.get("method") or path.parent.parent.name)
        try:
            rank = int(p.as_float(payload.get("case_rank")))
        except Exception:
            continue
        if allowed is not None and rank not in allowed:
            continue
        rows.append(payload)
    return rows


def is_done_row(row: Mapping[str, object]) -> bool:
    run_dir = Path(str(row.get("run_dir_actual") or row.get("run_dir") or ""))
    return (run_dir / "DONE").exists()


def dedupe_completed_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for row in rows:
        if not is_done_row(row):
            continue
        try:
            rank = int(p.as_float(row.get("case_rank")))
        except Exception:
            continue
        method, alpha = row_group_key(row)
        key = method, alpha, rank
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = row
            continue
        prev_path = Path(str(previous.get("summary_path", "")))
        this_path = Path(str(row.get("summary_path", "")))
        prev_mtime = prev_path.stat().st_mtime if prev_path.exists() else 0.0
        this_mtime = this_path.stat().st_mtime if this_path.exists() else 0.0
        if this_mtime >= prev_mtime:
            by_key[key] = row
    return list(by_key.values())


def collect_completed_rows(args: argparse.Namespace) -> Dict[str, List[Dict[str, object]]]:
    first20_sweeps: List[Dict[str, object]] = []
    for root in [f100.FIRST10_MAIN_ROOT, f100.FIRST10_CPS2_PRELIM_ROOT, f100.FIRST10_CPS2_COMPLETION_ROOT]:
        first20_sweeps.extend(collect_summary_rows_safe(root, f100.RANKS_FIRST20[:10]))
    first20_sweeps.extend(collect_summary_rows_safe(f100.NEXT10_ROOT, list(range(10, 20))))
    extension_sweeps = collect_summary_rows_safe(Path(args.single_stage_extension_root), f100.RANKS_EXTENSION)
    first20_dcp842 = [
        row
        for row in collect_summary_rows_safe(Path(args.directcp842_first20_root), f100.RANKS_FIRST20)
        if str(row.get("method_key")) == f100.DCP842_METHOD
    ]
    extension_dcp842 = [
        row
        for row in collect_summary_rows_safe(Path(args.directcp842_extension_root), f100.RANKS_EXTENSION)
        if str(row.get("method_key")) == f100.DCP842_METHOD
    ]
    first20_baselines = collect_summary_rows_safe(Path(args.baseline_first20_root), f100.RANKS_FIRST20)
    extension_baselines = collect_summary_rows_safe(Path(args.baseline_extension_root), f100.RANKS_EXTENSION)
    all_rows = dedupe_completed_rows(
        first20_sweeps
        + extension_sweeps
        + first20_dcp842
        + extension_dcp842
        + first20_baselines
        + extension_baselines
    )
    new_rows = dedupe_completed_rows(extension_sweeps + extension_dcp842 + extension_baselines)
    return {
        "all": all_rows,
        "new": new_rows,
        "extension_sweeps": dedupe_completed_rows(extension_sweeps),
        "extension_dcp842": dedupe_completed_rows(extension_dcp842),
        "extension_baselines": dedupe_completed_rows(extension_baselines),
    }


def collect_launcher_counts(args: argparse.Namespace) -> Tuple[Counter[GroupKey], Counter[GroupKey], List[str]]:
    running: Counter[GroupKey] = Counter()
    failed: Counter[GroupKey] = Counter()
    failure_logs: List[str] = []
    state_specs = [
        (
            Path(args.baseline_extension_root) / "launcher_state.json",
            Path(args.review_root) / "saved_baselines/launcher_state.json",
        ),
        (
            Path(args.directcp842_extension_root) / "launcher_state.json",
            Path(args.review_root) / "directcp842/launcher_state.json",
        ),
        (
            Path(args.single_stage_extension_root) / "launcher_state.json",
            Path(args.review_root) / "single_stage/launcher_state.json",
        ),
    ]
    for paths in state_specs:
        state = newest_json(paths)
        for item in state.get("running") or []:
            if not isinstance(item, Mapping):
                continue
            job = item.get("job")
            if isinstance(job, Mapping):
                running[job_group_key(job)] += 1
        for item in state.get("failures") or []:
            if not isinstance(item, Mapping):
                continue
            job = item.get("job")
            if isinstance(job, Mapping):
                failed[job_group_key(job)] += 1
            log_path = str(item.get("log_path") or "")
            if log_path:
                failure_logs.append(log_path)
    return running, failed, failure_logs


def build_statuses(args: argparse.Namespace, rows: Mapping[str, List[Dict[str, object]]]) -> List[GroupStatus]:
    done_cases: Dict[GroupKey, set[int]] = defaultdict(set)
    new_done_cases: Dict[GroupKey, set[int]] = defaultdict(set)
    for row in rows["all"]:
        done_cases[row_group_key(row)].add(int(p.as_float(row.get("case_rank"))))
    for row in rows["new"]:
        new_done_cases[row_group_key(row)].add(int(p.as_float(row.get("case_rank"))))

    running, failed, _failure_logs = collect_launcher_counts(args)
    out: List[GroupStatus] = []
    for group in expected_groups():
        key = group_key(group.method_key, group.alpha)
        done_count = len(done_cases.get(key, set()))
        running_count = int(running.get(key, 0))
        failed_count = int(failed.get(key, 0))
        pending_count = max(int(group.expected_cases) - done_count - running_count - failed_count, 0)
        if failed_count:
            status = "failed"
        elif done_count >= int(group.expected_cases):
            status = "done"
        elif running_count:
            status = "running"
        else:
            status = "pending"
        out.append(
            GroupStatus(
                stage=group.stage,
                method_key=group.method_key,
                label=group.label,
                alpha=group.alpha,
                lr=group.lr,
                expected_cases=group.expected_cases,
                done_cases=done_count,
                new_done_cases=len(new_done_cases.get(key, set())),
                running_jobs=running_count,
                failed_jobs=failed_count,
                pending_cases=pending_count,
                status=status,
            )
        )
    return out


def status_summary(statuses: Sequence[GroupStatus]) -> Dict[str, object]:
    new_expected = sum(80 for _status in statuses)
    new_done = sum(status.new_done_cases for status in statuses)
    running = sum(status.running_jobs for status in statuses)
    failed = sum(status.failed_jobs for status in statuses)
    return {
        "group_count": len(statuses),
        "complete_first100_groups": sum(1 for status in statuses if status.done_cases >= status.expected_cases),
        "new_expected_jobs": int(new_expected),
        "new_done_jobs": int(new_done),
        "running_jobs": int(running),
        "failed_jobs": int(failed),
        "new_pending_jobs": max(int(new_expected) - int(new_done) - int(running) - int(failed), 0),
    }


def format_lr(value: float | None) -> str:
    return "" if value is None else f"{float(value):g}"


def write_progress_log(
    args: argparse.Namespace,
    statuses: Sequence[GroupStatus],
    rows: Mapping[str, List[Dict[str, object]]],
    failure_logs: Sequence[str],
) -> None:
    summary = status_summary(statuses)
    lines = [
        "# OASIS Seed-3 First100 Extension Progress Log",
        "",
        f"- Updated: `{c.now_text()}`",
        f"- Progress log: `{Path(args.progress_log)}`",
        f"- Tracker figure: `{Path(args.tracker_figure)}`",
        f"- Final artifact root: `{f100.FIRST100_ARTIFACT_ROOT}`",
        f"- Baseline extension root: `{Path(args.baseline_extension_root)}`",
        f"- Single-stage extension root: `{Path(args.single_stage_extension_root)}`",
        f"- MR-D-BSCP extension root: `{Path(args.directcp842_extension_root)}`",
        "- Metric convention: absent fixed, moving, or warped label voxels produce `NaN`; population means use finite-only `nanmean`.",
        "",
        "## Overall",
        "",
        f"- First100-complete method/hyperparameter rows: `{summary['complete_first100_groups']}/{summary['group_count']}`",
        f"- New rank20-99 jobs done: `{summary['new_done_jobs']}/{summary['new_expected_jobs']}`",
        f"- Running jobs: `{summary['running_jobs']}`",
        f"- Failed jobs: `{summary['failed_jobs']}`",
        f"- Pending new jobs: `{summary['new_pending_jobs']}`",
        "",
        "## Run Matrix Status",
        "",
        "| Status | Stage | Method | Alpha | LR | Cases done | New done | Running | Failed | Pending |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for status in statuses:
        lines.append(
            "| "
            f"{status.status} | "
            f"{status.stage} | "
            f"`{status.method_key}` | "
            f"{alpha_text(status.alpha)} | "
            f"{format_lr(status.lr)} | "
            f"{status.done_cases}/{status.expected_cases} | "
            f"{status.new_done_cases}/80 | "
            f"{status.running_jobs} | "
            f"{status.failed_jobs} | "
            f"{status.pending_cases} |"
        )
    lines.extend(
        [
            "",
            "## Completed Extension Rows",
            "",
            f"- Saved baselines rank20-99 summaries: `{len(rows['extension_baselines'])}/400`",
            f"- MR-D-BSCP 8-4-2 rank20-99 summaries: `{len(rows['extension_dcp842'])}/480`",
            f"- Single-stage rank20-99 summaries: `{len(rows['extension_sweeps'])}/2400`",
            "",
            "## Failure Logs",
            "",
        ]
    )
    if failure_logs:
        lines.extend(f"- `{path}`" for path in sorted(set(failure_logs))[:50])
    else:
        lines.append("- none")
    Path(args.progress_log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.progress_log).write_text("\n".join(lines) + "\n")


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def partial_population_rows(rows: Sequence[Dict[str, object]], min_cases: int) -> List[Dict[str, object]]:
    if not rows:
        return []
    pop_rows = p.population_summary(rows)
    return [row for row in pop_rows if int(p.as_float(row.get("n_cases"))) >= int(min_cases)]


def draw_partial_curve_panel(
    ax: plt.Axes,
    pop_rows: Sequence[Dict[str, object]],
    *,
    y_key: str,
    y_label: str,
) -> None:
    x_key = "log_jdet_std_fg_mean"
    y_mean = f"{y_key}_mean"
    plotted = False
    for method in p.SWEEP_METHODS:
        rows = [
            row
            for row in pop_rows
            if str(row.get("method_key")) == method and finite(row.get(x_key)) and finite(row.get(y_mean))
        ]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: p.as_float(row.get("alpha")))
        color = str(p.METHOD_SPECS.get(method, {}).get("color", "#333333"))
        label = str(p.METHOD_SPECS.get(method, {}).get("dot_label", method))
        ax.plot(
            [float(row[x_key]) for row in rows],
            [float(row[y_mean]) for row in rows],
            marker="o",
            linewidth=1.5,
            markersize=3.5,
            color=color,
            label=label,
        )
        plotted = True
    for method in p.BASELINE_METHODS:
        rows = [
            row
            for row in pop_rows
            if str(row.get("method_key")) == method and finite(row.get(x_key)) and finite(row.get(y_mean))
        ]
        for row in rows:
            color = str(p.METHOD_SPECS.get(method, {}).get("color", "#333333"))
            label = str(p.METHOD_SPECS.get(method, {}).get("dot_label", method))
            ax.scatter(float(row[x_key]), float(row[y_mean]), s=28, color=color, label=label, edgecolor="white", linewidth=0.4)
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No completed metrics yet", ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_xlabel("Foreground std(log J)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    dedup: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), fontsize=6.5, ncol=3, frameon=False, loc="best")


def draw_tracker_figure(
    args: argparse.Namespace,
    statuses: Sequence[GroupStatus],
    completed_rows: Sequence[Dict[str, object]],
) -> None:
    plus.configure_plot_module(float(f100.DCP842_LR))
    height = max(12.0, 0.28 * len(statuses) + 2.5)
    fig = plt.figure(figsize=(18.0, height), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.0, 1.0])
    ax_status = fig.add_subplot(gs[:, 0])
    ax_dice = fig.add_subplot(gs[0, 1])
    ax_hd95 = fig.add_subplot(gs[1, 1])

    y_positions = list(range(len(statuses)))
    colors = [STATUS_COLORS.get(status.status, "#757575") for status in statuses]
    done_values = [status.done_cases for status in statuses]
    ax_status.barh(y_positions, done_values, color=colors, alpha=0.9, height=0.72)
    ax_status.set_xlim(0, 100)
    ax_status.set_xlabel("Completed cases out of 100")
    ax_status.set_yticks(y_positions)
    labels = [
        f"{status.label}  a={alpha_text(status.alpha)}  {status.done_cases}/{status.expected_cases}"
        for status in statuses
    ]
    ax_status.set_yticklabels(labels, fontsize=6.7)
    ax_status.invert_yaxis()
    ax_status.grid(axis="x", alpha=0.25, linewidth=0.6)
    for idx, status in enumerate(statuses):
        ax_status.text(
            min(status.done_cases + 1.5, 98.0),
            idx,
            status.status,
            va="center",
            ha="left" if status.done_cases < 90 else "right",
            fontsize=6.5,
            color="#222222",
        )
    summary = status_summary(statuses)
    ax_status.set_title(
        "OASIS seed-3 first100 extension: "
        f"{summary['new_done_jobs']}/{summary['new_expected_jobs']} new jobs done, "
        f"{summary['running_jobs']} running, {summary['failed_jobs']} failed",
        loc="left",
        fontsize=11,
    )

    pop_rows = partial_population_rows(completed_rows, int(args.min_partial_cases))
    draw_partial_curve_panel(ax_dice, pop_rows, y_key="dice_mean", y_label="Mean Dice")
    draw_partial_curve_panel(ax_hd95, pop_rows, y_key="hd95_mean", y_label="Mean HD95")
    ax_dice.set_title(f"Partial Dice curve, n >= {int(args.min_partial_cases)} completed cases", fontsize=10)
    ax_hd95.set_title(f"Partial HD95 curve, n >= {int(args.min_partial_cases)} completed cases", fontsize=10)
    fig.suptitle(f"Updated {c.now_text()}", fontsize=12)
    Path(args.tracker_figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(args.tracker_figure), dpi=int(args.dpi))
    plt.close(fig)


def snapshot(args: argparse.Namespace) -> Dict[str, object]:
    plus.configure_plot_module(float(f100.DCP842_LR))
    rows = collect_completed_rows(args)
    statuses = build_statuses(args, rows)
    _running, _failed, failure_logs = collect_launcher_counts(args)
    write_progress_log(args, statuses, rows, failure_logs)
    draw_tracker_figure(args, statuses, rows["all"])
    item = {
        "updated_at": c.now_text(),
        "progress_log": str(args.progress_log),
        "tracker_figure": str(args.tracker_figure),
        "summary": status_summary(statuses),
        "statuses": [asdict(status) for status in statuses],
        "failure_logs": sorted(set(failure_logs)),
    }
    Path(args.tracker_root).mkdir(parents=True, exist_ok=True)
    c.write_json(Path(args.tracker_root) / "progress_snapshot.json", item)
    with (Path(args.tracker_root) / "progress_snapshots.jsonl").open("a") as f:
        f.write(json.dumps(item, sort_keys=True, allow_nan=True) + "\n")
    return item


def main() -> None:
    args = parse_args()
    while True:
        item = snapshot(args)
        print(json.dumps(item["summary"], indent=2, sort_keys=True), flush=True)
        if not args.watch:
            break
        time.sleep(float(args.interval_sec))


if __name__ == "__main__":
    main()
