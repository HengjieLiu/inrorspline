#!/usr/bin/env python
"""Analyze LungCT IDIR and DirectCP 4-stage training curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as c
from launch_reg_sweep_idir_directcp32 import (
    BASE_DIRECTCP_ISOTV,
    BASE_IDIR_ALPHA,
    DEFAULT_STAGE_EPOCHS,
    DEFAULT_SWEEP_ROOT,
    METHOD_DIRECTCP32,
    METHOD_IDIR,
    SWEEP_METHODS,
)


DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32"
FACTORS = [0.01, 0.1, 1.0, 10.0, 100.0]
IDIR_CANDIDATES = [250, 500, 750, 1000, 1250, 1500, 2000, 2500]
DIRECT_STAGE_CANDIDATES = {
    0: [50, 100, 150, 200],
    1: [100, 200, 300, 400],
    2: [200, 400, 600, 800],
    3: [250, 500, 750, 1000],
}
THRESHOLDS = [0.99, 0.995, 0.999]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=c.DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    return parser.parse_args()


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def method_label(method: str) -> str:
    if method == METHOD_IDIR:
        return "IDIR BE default"
    if method == METHOD_DIRECTCP32:
        return "DirectCP 32-16-8-4 pTV LCC + isoTV"
    return method


def reg_factor(method: str, payload: Dict[str, object]) -> float:
    training = payload.get("training", {})
    if not isinstance(training, dict):
        training = {}
    if method == METHOD_IDIR:
        return float(training.get("alpha_bending", BASE_IDIR_ALPHA)) / BASE_IDIR_ALPHA
    if method == METHOD_DIRECTCP32:
        return float(training.get("ptv_isotv_weight", BASE_DIRECTCP_ISOTV)) / BASE_DIRECTCP_ISOTV
    raise ValueError(f"Unsupported method: {method}")


def collect_runs(root: Path, method: str, source: str) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for summary_path in sorted((root / method).glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        if payload.get("status") != "completed":
            continue
        factor = reg_factor(method, payload)
        history_name = "loss_history.csv" if method == METHOD_IDIR else "stage_history.csv"
        history_path = summary_path.parent / history_name
        if not history_path.exists():
            continue
        training = payload.get("training", {})
        if not isinstance(training, dict):
            training = {}
        runs.append(
            {
                "method": method,
                "method_label": method_label(method),
                "case_id": int(payload["case_id"]),
                "reg_factor": float(factor),
                "reg_value": float(
                    training.get(
                        "alpha_bending" if method == METHOD_IDIR else "ptv_isotv_weight",
                        BASE_IDIR_ALPHA if method == METHOD_IDIR else BASE_DIRECTCP_ISOTV,
                    )
                ),
                "summary_path": summary_path,
                "history_path": history_path,
                "summary": payload,
                "source": source,
            }
        )
    return runs


def collect_all_runs(sweep_root: Path, baseline_root: Path) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for method in SWEEP_METHODS:
        runs.extend(collect_runs(sweep_root, method, "sweep"))
        runs.extend(collect_runs(baseline_root, method, "baseline_1e0"))
    # The sweep root intentionally excludes factor 1, but guard against future duplicates.
    unique: Dict[tuple[str, int, float], Dict[str, object]] = {}
    for run in runs:
        if not any(abs(float(run["reg_factor"]) - factor) < 1e-10 for factor in FACTORS):
            continue
        key = (str(run["method"]), int(run["case_id"]), round(float(run["reg_factor"]), 12))
        if key not in unique or run["source"] == "baseline_1e0":
            unique[key] = run
    return sorted(unique.values(), key=lambda r: (str(r["method"]), float(r["reg_factor"]), int(r["case_id"])))


def rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(arr) == 0:
        return arr
    out = np.empty_like(arr)
    half = int(window) // 2
    for idx in range(len(arr)):
        lo = max(0, idx - half)
        hi = min(len(arr), idx + half + 1)
        out[idx] = np.nanmean(arr[lo:hi])
    return out


def progress_curve(values: Sequence[float], window: int, final_window: int = 50) -> np.ndarray:
    smooth = rolling_mean(values, window)
    if len(smooth) == 0:
        return smooth
    final = float(np.nanmean(smooth[-min(final_window, len(smooth)) :]))
    start = float(smooth[0])
    denom = start - final
    if abs(denom) <= 1e-12:
        return np.full_like(smooth, np.nan)
    return (start - smooth) / denom


def first_epoch_for_threshold(epochs: Sequence[int], values: Sequence[float], threshold: float, window: int) -> float:
    progress = progress_curve(values, window)
    finite = np.isfinite(progress)
    if not bool(np.any(finite)):
        return float("nan")
    idx = np.where(finite & (progress >= threshold))[0]
    if len(idx) == 0:
        return float("nan")
    return float(epochs[int(idx[0])])


def progress_at_candidates(
    epochs: Sequence[int], values: Sequence[float], candidates: Iterable[int], window: int
) -> Dict[int, float]:
    progress = progress_curve(values, window)
    out: Dict[int, float] = {}
    epoch_arr = np.asarray(epochs, dtype=np.int64)
    for candidate in candidates:
        idx = np.where(epoch_arr <= int(candidate))[0]
        if len(idx) == 0:
            out[int(candidate)] = float("nan")
        else:
            out[int(candidate)] = float(progress[int(idx[-1])])
    return out


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def fmt(value: object, digits: int = 3) -> str:
    value_f = safe_float(value)
    if not math.isfinite(value_f):
        return "NA"
    return f"{value_f:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def lr_metadata_rows(runs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run in runs:
        training = run["summary"].get("training", {})
        if not isinstance(training, dict):
            training = {}
        rows.append(
            {
                "method": run["method"],
                "case_id": run["case_id"],
                "reg_factor": run["reg_factor"],
                "reg_value": run["reg_value"],
                "epochs": training.get("epochs"),
                "stage_epochs": json.dumps(training.get("stage_epochs")),
                "lr": training.get("lr"),
                "stage_lr": json.dumps(training.get("stage_lr")),
                "history_path": run["history_path"],
            }
        )
    return rows


def idir_progress_rows(runs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for factor in FACTORS:
        factor_runs = [run for run in runs if run["method"] == METHOD_IDIR and abs(float(run["reg_factor"]) - factor) < 1e-10]
        for candidate in IDIR_CANDIDATES:
            values = []
            for run in factor_runs:
                hist = read_csv(Path(run["history_path"]))
                epochs = [int(safe_float(row["epoch"])) for row in hist]
                data_loss = [safe_float(row["data_loss"]) for row in hist]
                values.append(progress_at_candidates(epochs, data_loss, [candidate], 51)[candidate])
            mean, std = mean_std(values)
            rows.append(
                {
                    "method": METHOD_IDIR,
                    "reg_factor": factor,
                    "candidate_epoch": candidate,
                    "mean_data_loss_progress": mean,
                    "std_data_loss_progress": std,
                    "n_cases": len(values),
                }
            )
    return rows


def idir_threshold_rows(runs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for factor in FACTORS:
        factor_runs = [run for run in runs if run["method"] == METHOD_IDIR and abs(float(run["reg_factor"]) - factor) < 1e-10]
        for threshold in THRESHOLDS:
            values = []
            for run in factor_runs:
                hist = read_csv(Path(run["history_path"]))
                epochs = [int(safe_float(row["epoch"])) for row in hist]
                data_loss = [safe_float(row["data_loss"]) for row in hist]
                values.append(first_epoch_for_threshold(epochs, data_loss, threshold, 51))
            finite = [value for value in values if math.isfinite(value)]
            rows.append(
                {
                    "method": METHOD_IDIR,
                    "reg_factor": factor,
                    "threshold": threshold,
                    "mean_epoch": float(np.mean(finite)) if finite else float("nan"),
                    "median_epoch": float(np.median(finite)) if finite else float("nan"),
                    "max_epoch": float(np.max(finite)) if finite else float("nan"),
                    "n_cases": len(finite),
                }
            )
    return rows


def direct_threshold_rows(runs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for factor in FACTORS:
        factor_runs = [
            run for run in runs if run["method"] == METHOD_DIRECTCP32 and abs(float(run["reg_factor"]) - factor) < 1e-10
        ]
        for stage in range(4):
            for threshold in THRESHOLDS:
                values = []
                for run in factor_runs:
                    hist = [
                        row
                        for row in read_csv(Path(run["history_path"]))
                        if int(safe_float(row["stage_idx"])) == int(stage)
                    ]
                    epochs = [int(safe_float(row["epoch"])) for row in hist]
                    data_loss = [safe_float(row["data_loss"]) for row in hist]
                    values.append(first_epoch_for_threshold(epochs, data_loss, threshold, 11))
                finite = [value for value in values if math.isfinite(value)]
                rows.append(
                    {
                        "method": METHOD_DIRECTCP32,
                        "reg_factor": factor,
                        "stage_idx": stage,
                        "threshold": threshold,
                        "mean_epoch": float(np.mean(finite)) if finite else float("nan"),
                        "median_epoch": float(np.median(finite)) if finite else float("nan"),
                        "max_epoch": float(np.max(finite)) if finite else float("nan"),
                        "n_cases": len(finite),
                    }
                )
    return rows


def direct_progress_rows(runs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for factor in FACTORS:
        factor_runs = [
            run for run in runs if run["method"] == METHOD_DIRECTCP32 and abs(float(run["reg_factor"]) - factor) < 1e-10
        ]
        for stage, candidates in DIRECT_STAGE_CANDIDATES.items():
            for candidate in candidates:
                values = []
                for run in factor_runs:
                    hist = [
                        row
                        for row in read_csv(Path(run["history_path"]))
                        if int(safe_float(row["stage_idx"])) == int(stage)
                    ]
                    epochs = [int(safe_float(row["epoch"])) for row in hist]
                    data_loss = [safe_float(row["data_loss"]) for row in hist]
                    values.append(progress_at_candidates(epochs, data_loss, [candidate], 11)[candidate])
                mean, std = mean_std(values)
                rows.append(
                    {
                        "method": METHOD_DIRECTCP32,
                        "reg_factor": factor,
                        "stage_idx": stage,
                        "candidate_epoch": candidate,
                        "mean_data_loss_progress": mean,
                        "std_data_loss_progress": std,
                        "n_cases": len(values),
                    }
                )
    return rows


def stage_time_rows(runs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    direct_default = [
        run for run in runs if run["method"] == METHOD_DIRECTCP32 and abs(float(run["reg_factor"]) - 1.0) < 1e-10
    ]
    for run in direct_default:
        summaries = run["summary"].get("stage_summaries", [])
        if not isinstance(summaries, list):
            continue
        total_stage_sec = sum(safe_float(stage.get("elapsed_sec")) for stage in summaries if isinstance(stage, dict))
        for stage in summaries:
            if not isinstance(stage, dict):
                continue
            elapsed = safe_float(stage.get("elapsed_sec"))
            epochs = safe_float(stage.get("epochs"))
            rows.append(
                {
                    "case_id": run["case_id"],
                    "stage_idx": int(safe_float(stage.get("stage_idx"))),
                    "epochs": int(epochs),
                    "elapsed_sec": elapsed,
                    "sec_per_epoch": elapsed / epochs if epochs > 0 else float("nan"),
                    "stage_time_fraction": elapsed / total_stage_sec if total_stage_sec > 0 else float("nan"),
                }
            )
    return rows


def plot_idir_curves(runs: Sequence[Dict[str, object]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(7.0, 12.0), dpi=160, sharex=True)
    for ax, factor in zip(axes, FACTORS):
        factor_runs = [run for run in runs if run["method"] == METHOD_IDIR and abs(float(run["reg_factor"]) - factor) < 1e-10]
        for run in factor_runs:
            hist = read_csv(Path(run["history_path"]))
            epochs = [safe_float(row["epoch"]) for row in hist]
            data_loss = rolling_mean([safe_float(row["data_loss"]) for row in hist], 51)
            ax.plot(epochs, data_loss, linewidth=0.8, alpha=0.75)
        ax.set_title(f"IDIR factor {factor:g}x")
        ax.set_ylabel("data loss")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("epoch")
    fig.suptitle("IDIR data-loss curves, all cases")
    fig.tight_layout()
    fig.savefig(plot_dir / "idir_data_loss_all_cases_by_factor.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    for factor in FACTORS:
        factor_runs = [run for run in runs if run["method"] == METHOD_IDIR and abs(float(run["reg_factor"]) - factor) < 1e-10]
        progress_all = []
        epochs_ref: List[int] = []
        for run in factor_runs:
            hist = read_csv(Path(run["history_path"]))
            epochs = [int(safe_float(row["epoch"])) for row in hist]
            data_loss = [safe_float(row["data_loss"]) for row in hist]
            progress_all.append(progress_curve(data_loss, 51))
            epochs_ref = epochs
        if progress_all:
            arr = np.vstack(progress_all)
            ax.plot(epochs_ref, np.nanmean(arr, axis=0), linewidth=1.2, label=f"{factor:g}x")
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction of final data-loss improvement")
    ax.set_ylim(0.8, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(title="factor", fontsize=8)
    ax.set_title("IDIR mean convergence progress by factor")
    fig.tight_layout()
    fig.savefig(plot_dir / "idir_mean_progress_by_factor.png")
    plt.close(fig)


def plot_direct_curves(runs: Sequence[Dict[str, object]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    for factor in FACTORS:
        factor_runs = [
            run for run in runs if run["method"] == METHOD_DIRECTCP32 and abs(float(run["reg_factor"]) - factor) < 1e-10
        ]
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), dpi=160, sharey=True)
        for stage, ax in enumerate(axes.reshape(-1)):
            for run in factor_runs:
                hist = [
                    row
                    for row in read_csv(Path(run["history_path"]))
                    if int(safe_float(row["stage_idx"])) == int(stage)
                ]
                epochs = [int(safe_float(row["epoch"])) for row in hist]
                data_loss = [safe_float(row["data_loss"]) for row in hist]
                ax.plot(epochs, progress_curve(data_loss, 11), linewidth=0.8, alpha=0.75)
            ax.set_title(f"stage {stage}")
            ax.set_xlabel("stage epoch")
            ax.grid(True, alpha=0.25)
        axes[0, 0].set_ylabel("fraction of final data-loss improvement")
        axes[1, 0].set_ylabel("fraction of final data-loss improvement")
        fig.suptitle(f"DirectCP 4-stage data-loss progress, factor {factor:g}x")
        fig.tight_layout()
        fig.savefig(plot_dir / f"directcp_progress_factor_{str(factor).replace('.', 'p')}.png")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), dpi=160, sharey=True)
    for stage, ax in enumerate(axes.reshape(-1)):
        for factor in FACTORS:
            factor_runs = [
                run
                for run in runs
                if run["method"] == METHOD_DIRECTCP32 and abs(float(run["reg_factor"]) - factor) < 1e-10
            ]
            progress_all = []
            epochs_ref: List[int] = []
            for run in factor_runs:
                hist = [
                    row
                    for row in read_csv(Path(run["history_path"]))
                    if int(safe_float(row["stage_idx"])) == int(stage)
                ]
                epochs = [int(safe_float(row["epoch"])) for row in hist]
                data_loss = [safe_float(row["data_loss"]) for row in hist]
                progress_all.append(progress_curve(data_loss, 11))
                epochs_ref = epochs
            if progress_all:
                arr = np.vstack(progress_all)
                ax.plot(epochs_ref, np.nanmean(arr, axis=0), linewidth=1.2, label=f"{factor:g}x")
        ax.set_title(f"stage {stage}")
        ax.set_xlabel("stage epoch")
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("fraction of final data-loss improvement")
    axes[1, 0].set_ylabel("fraction of final data-loss improvement")
    axes[0, 1].legend(title="factor", fontsize=8)
    fig.suptitle("DirectCP 4-stage mean convergence progress by factor")
    fig.tight_layout()
    fig.savefig(plot_dir / "directcp_mean_progress_by_factor_and_stage.png")
    plt.close(fig)


def rows_for_markdown(rows: Sequence[Dict[str, object]], keys: Sequence[str]) -> List[List[object]]:
    out: List[List[object]] = []
    for row in rows:
        rendered = []
        for key in keys:
            value = row.get(key)
            if isinstance(value, float):
                rendered.append(fmt(value, 3))
            else:
                rendered.append(value)
        out.append(rendered)
    return out


def write_report(
    report_path: Path,
    runs: Sequence[Dict[str, object]],
    idir_progress: Sequence[Dict[str, object]],
    idir_thresholds: Sequence[Dict[str, object]],
    direct_progress: Sequence[Dict[str, object]],
    direct_thresholds: Sequence[Dict[str, object]],
    stage_times: Sequence[Dict[str, object]],
) -> None:
    idir_lrs = sorted({safe_float(run["summary"].get("training", {}).get("lr")) for run in runs if run["method"] == METHOD_IDIR})
    direct_stage_lrs = sorted(
        {
            tuple(run["summary"].get("training", {}).get("stage_lr", []))
            for run in runs
            if run["method"] == METHOD_DIRECTCP32
        }
    )
    direct_stage_epochs = sorted(
        {
            tuple(run["summary"].get("training", {}).get("stage_epochs", []))
            for run in runs
            if run["method"] == METHOD_DIRECTCP32
        }
    )

    idir_progress_focus = [
        row
        for row in idir_progress
        if int(row["candidate_epoch"]) in [1000, 1500, 2000, 2500]
    ]
    idir_threshold_focus = [row for row in idir_thresholds if float(row["reg_factor"]) in [0.1, 1.0, 10.0]]
    direct_default_thresholds = [
        row for row in direct_thresholds if abs(float(row["reg_factor"]) - 1.0) < 1e-10
    ]
    direct_default_progress = [
        row
        for row in direct_progress
        if abs(float(row["reg_factor"]) - 1.0) < 1e-10
        and (
            (int(row["stage_idx"]) == 0 and int(row["candidate_epoch"]) in [150, 200])
            or (int(row["stage_idx"]) == 1 and int(row["candidate_epoch"]) in [200, 300, 400])
            or (int(row["stage_idx"]) == 2 and int(row["candidate_epoch"]) in [200, 400, 800])
            or (int(row["stage_idx"]) == 3 and int(row["candidate_epoch"]) in [250, 500, 750, 1000])
        )
    ]
    stage_summary_rows = []
    for stage in range(4):
        stage_values = [row for row in stage_times if int(row["stage_idx"]) == stage]
        elapsed_mean, elapsed_std = mean_std([safe_float(row["elapsed_sec"]) for row in stage_values])
        frac_mean, frac_std = mean_std([safe_float(row["stage_time_fraction"]) for row in stage_values])
        sec_epoch_mean, sec_epoch_std = mean_std([safe_float(row["sec_per_epoch"]) for row in stage_values])
        stage_summary_rows.append(
            [
                stage,
                fmt(elapsed_mean, 2) + " +/- " + fmt(elapsed_std, 2),
                fmt(sec_epoch_mean, 4) + " +/- " + fmt(sec_epoch_std, 4),
                fmt(frac_mean, 3) + " +/- " + fmt(frac_std, 3),
            ]
        )

    text = "\n".join(
        [
            "# LungCT Regularization Sweep Training-Curve Analysis",
            "",
            "This analysis combines the 40 nonbaseline sweep runs with the reused 10-case `1x` baseline for each method, giving 50 IDIR histories and 50 DirectCP 4-stage histories.",
            "",
            "Progress is computed from smoothed `data_loss` curves. For each run or stage, progress is `(start - value_at_epoch) / (start - final_window_mean)`, so `1.0` means the curve has reached the final smoothed data-loss level observed in the completed run. This is an optimization-objective diagnostic, not a substitute for rerunning TRE at shorter schedules.",
            "",
            "## Learning Rate Confirmation",
            "",
            f"- IDIR histories: `{sum(1 for run in runs if run['method'] == METHOD_IDIR)}`; unique recorded LR values: `{idir_lrs}`.",
            f"- DirectCP histories: `{sum(1 for run in runs if run['method'] == METHOD_DIRECTCP32)}`; unique recorded stage LR values: `{direct_stage_lrs}`.",
            f"- DirectCP unique recorded stage epochs: `{direct_stage_epochs}`.",
            "",
            "## Current Epoch Configuration",
            "",
            markdown_table(
                ["Method", "Epoch setting", "LR", "Default regularization"],
                [
                    [method_label(METHOD_IDIR), "2500 epochs", "1e-4", f"alpha_bending={BASE_IDIR_ALPHA:g}"],
                    [
                        method_label(METHOD_DIRECTCP32),
                        f"stage epochs {DEFAULT_STAGE_EPOCHS}; total {sum(c.parse_stage_ints(DEFAULT_STAGE_EPOCHS, 4))}",
                        "1e-3 each stage",
                        f"ptv_isotv_weight={BASE_DIRECTCP_ISOTV:g}",
                    ],
                ],
            ),
            "",
            "## IDIR Observations",
            "",
            "- IDIR curves are stochastic because each epoch samples 10,000 fixed-lung-mask coordinates.",
            "- At the default `1x` factor, the mean smoothed data-loss progress is already about 99.1% at 1000 epochs and 99.6% at 1500 epochs.",
            "- Low-regularization factors `0.01x` and `0.1x` converge more slowly; if those are being compared as final Pareto points, cutting too aggressively can bias them.",
            "",
            "### IDIR Data-Loss Progress",
            "",
            markdown_table(
                ["Factor", "Candidate epoch", "Mean progress", "SD"],
                rows_for_markdown(
                    idir_progress_focus,
                    ["reg_factor", "candidate_epoch", "mean_data_loss_progress", "std_data_loss_progress"],
                ),
            ),
            "",
            "### IDIR Epoch To Reach Progress Threshold",
            "",
            markdown_table(
                ["Factor", "Threshold", "Mean epoch", "Median epoch", "Max epoch", "Cases"],
                rows_for_markdown(
                    idir_threshold_focus,
                    ["reg_factor", "threshold", "mean_epoch", "median_epoch", "max_epoch", "n_cases"],
                ),
            ),
            "",
            "## DirectCP 4-Stage Observations",
            "",
            "- Stage 0 should mostly stay at 200 epochs; it usually needs roughly 160-175 epochs to reach 99.9% of its final data-loss progress.",
            "- Stage 1 is cheap, around 9% of stage optimization time at default `1x`; cutting it has limited wall-time benefit.",
            "- At default `1x`, stages 2 and 3 are the main overtraining candidates. Stage 2 reaches 99.9% at about 89 epochs on average and at 218 epochs in the slowest case. Stage 3 reaches 99.9% at about 270 epochs on average and at 628 epochs in the slowest case.",
            "- Extreme regularization factors have different behavior, especially stages 2-3. A shortened schedule should be labeled and validated if used for Pareto comparison.",
            "",
            "### DirectCP Default-Factor Data-Loss Progress",
            "",
            markdown_table(
                ["Stage", "Candidate epoch", "Mean progress", "SD"],
                rows_for_markdown(
                    direct_default_progress,
                    ["stage_idx", "candidate_epoch", "mean_data_loss_progress", "std_data_loss_progress"],
                ),
            ),
            "",
            "### DirectCP Default-Factor Epoch To Reach Progress Threshold",
            "",
            markdown_table(
                ["Stage", "Threshold", "Mean epoch", "Median epoch", "Max epoch", "Cases"],
                rows_for_markdown(
                    direct_default_thresholds,
                    ["stage_idx", "threshold", "mean_epoch", "median_epoch", "max_epoch", "n_cases"],
                ),
            ),
            "",
            "### DirectCP Default-Factor Stage Timing",
            "",
            markdown_table(["Stage", "Elapsed sec", "Sec/epoch", "Stage-time fraction"], stage_summary_rows),
            "",
            "## Epoch-Reduction Recommendation",
            "",
            "For final paper-facing reproduction, do not change epochs silently. Any shortened schedule should be a new protocol tag and should be validated by rerunning TRE/regularity for at least the default `1x` point and the neighboring Pareto-relevant factors.",
            "",
            "Recommended next validation runs:",
            "",
            "- IDIR default/near-default validation: compare `epochs=1500`, `2000`, and current `2500` for factors `0.1,1,10` on all 10 cases. The objective curves support `1500` as a likely screening setting and `2000` as a safer final-lite setting.",
            "- DirectCP default validation: compare current `200,400,800,1000` against `200,400,300,650` and fast-screen `200,200,200,400` at factor `1x` on all 10 cases. The curve evidence favors cutting stages 2-3 first, not stage 0.",
            "- If the shortened DirectCP default point matches original-snap TRE within an acceptable tolerance, extend only to the Pareto-relevant neighboring factors `0.1x` and `10x`. Avoid using the shortened schedule for extreme `0.01x`/`100x` conclusions without separate validation.",
            "",
            "Practical default recommendation before validation:",
            "",
            "- IDIR screen: `1500` epochs; IDIR safer final-lite: `2000` epochs.",
            "- DirectCP screen: `200,200,200,400`; DirectCP safer final-lite: `200,400,300,650`.",
            "- Keep the completed `2500` and `200,400,800,1000` rows as the current canonical reproduction until shorter schedules are empirically checked with TRE.",
            "",
            "## Plots",
            "",
            "- `plots/training_curves/idir_data_loss_all_cases_by_factor.png`",
            "- `plots/training_curves/idir_mean_progress_by_factor.png`",
            "- `plots/training_curves/directcp_progress_factor_*.png`",
            "- `plots/training_curves/directcp_mean_progress_by_factor_and_stage.png`",
        ]
    )
    report_path.write_text(text + "\n")


def main() -> None:
    args = parse_args()
    analysis_root = args.review_root / "training_curve_analysis"
    plot_dir = args.review_root / "plots/training_curves"
    analysis_root.mkdir(parents=True, exist_ok=True)
    runs = collect_all_runs(args.sweep_root, args.baseline_root)
    metadata = lr_metadata_rows(runs)
    idir_progress = idir_progress_rows(runs)
    idir_thresholds = idir_threshold_rows(runs)
    direct_progress = direct_progress_rows(runs)
    direct_thresholds = direct_threshold_rows(runs)
    stage_times = stage_time_rows(runs)

    write_csv(analysis_root / "training_curve_metadata.csv", metadata)
    write_csv(analysis_root / "idir_progress_by_factor.csv", idir_progress)
    write_csv(analysis_root / "idir_threshold_epochs.csv", idir_thresholds)
    write_csv(analysis_root / "directcp_progress_by_factor_stage.csv", direct_progress)
    write_csv(analysis_root / "directcp_threshold_epochs_by_stage.csv", direct_thresholds)
    write_csv(analysis_root / "directcp_stage_timing_default_factor.csv", stage_times)
    plot_idir_curves(runs, plot_dir)
    plot_direct_curves(runs, plot_dir)
    write_report(
        analysis_root / "training_curve_epoch_analysis.md",
        runs,
        idir_progress,
        idir_thresholds,
        direct_progress,
        direct_thresholds,
        stage_times,
    )

    print(
        json.dumps(
            {
                "runs": len(runs),
                "idir_runs": sum(1 for run in runs if run["method"] == METHOD_IDIR),
                "directcp_runs": sum(1 for run in runs if run["method"] == METHOD_DIRECTCP32),
                "analysis_root": str(analysis_root),
                "plot_dir": str(plot_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
