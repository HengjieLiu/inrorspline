#!/usr/bin/env python
"""Aggregate and plot the OASIS seed-3 case-0 pre-run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


EXPECTED = {
    "idir_original_dense_siren_be_autograd": [0.01, 0.1, 1.0],
    "sinr_cps4_be_bspline_analytic": [100.0, 1000.0, 10000.0],
    "sinr_cps2_be_bspline_analytic": [100.0, 1000.0, 10000.0],
    "direct_cp_cps4_be_bspline_analytic": [100.0, 1000.0, 10000.0],
}

METHOD_LABELS = {
    "idir_original_dense_siren_be_autograd": "IDIR",
    "sinr_cps4_be_bspline_analytic": "SINR cps4",
    "sinr_cps2_be_bspline_analytic": "SINR cps2",
    "direct_cp_cps4_be_bspline_analytic": "DirectCP cps4 lr1e-3",
}

METHOD_COLORS = {
    "idir_original_dense_siren_be_autograd": "#1f77b4",
    "sinr_cps4_be_bspline_analytic": "#2ca02c",
    "sinr_cps2_be_bspline_analytic": "#9467bd",
    "direct_cp_cps4_be_bspline_analytic": "#d62728",
}

ACCURACY_METRICS: List[Tuple[str, str, bool]] = [
    ("dice_mean", "Mean Dice", True),
    ("hd95_mean", "Mean HD95", False),
    ("asd_warped_to_fixed_mean", "ASD warped-to-fixed", False),
    ("asd_symmetric_mean", "ASD symmetric", False),
]

REGULARITY_METRICS: List[Tuple[str, str]] = [
    ("be_fd_forward_fixed_fg_voxel", "FG BE forward FD"),
    ("be_native_analytic_fixed_fg_voxel", "FG BE native analytic"),
    ("diffusion_l2_fg", "FG diffusion L2"),
    ("diffusion_fd_forward_fixed_fg_voxel", "FG diffusion forward FD"),
    ("diffusion_native_analytic_fixed_fg_voxel", "FG diffusion native analytic"),
    ("jdet_le0_pct_fg", "FG Jdet <= 0 (%)"),
    ("log_jdet_std_fg", "FG std(log Jdet)"),
    ("ndv_frac_fg", "FG NDV fraction"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=c.DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=c.DEFAULT_REVIEW_ROOT)
    parser.add_argument("--expected-original-case-index", type=int, default=158)
    parser.add_argument("--expected-fixed-id", type=str, default="0333")
    parser.add_argument("--expected-moving-id", type=str, default="0115")
    parser.add_argument("--case-label", type=str, default="OASIS seed-3 rank 0")
    parser.add_argument("--figure-title", type=str, default="OASIS seed-3 rank 0 pre-run")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(value: object) -> bool:
    return math.isfinite(as_float(value))


def collect_summaries(out_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(out_root.glob("*/*/summary.json")):
        row = json.loads(path.read_text())
        row["summary_path"] = str(path)
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("method_key")), as_float(r.get("alpha"))))
    return rows


def check_expected(
    rows: Sequence[Dict[str, object]],
    *,
    expected_original_case_index: int,
    expected_fixed_id: str,
    expected_moving_id: str,
) -> List[str]:
    errors: List[str] = []
    by_key: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_key.setdefault(str(row.get("method_key")), []).append(row)
        if int(as_float(row.get("case_rank"))) != 0:
            errors.append(f"Unexpected case_rank for {row.get('run_name')}: {row.get('case_rank')}")
        if int(as_float(row.get("original_case_index"))) != int(expected_original_case_index):
            errors.append(f"Unexpected original_case_index for {row.get('run_name')}: {row.get('original_case_index')}")
        if str(row.get("fixed_id")) != str(expected_fixed_id) or str(row.get("moving_id")) != str(expected_moving_id):
            errors.append(f"Unexpected fixed/moving IDs for {row.get('run_name')}: {row.get('fixed_id')}/{row.get('moving_id')}")
    for key, alphas in EXPECTED.items():
        got = sorted(as_float(row.get("alpha")) for row in by_key.get(key, []))
        want = sorted(alphas)
        if len(got) != len(want) or any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(got, want)):
            errors.append(f"Missing or extra alpha rows for {key}: got {got}, expected {want}")
    return errors


def collect_history(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        summary_path = Path(str(row["summary_path"]))
        history_path = summary_path.parent / "history.csv"
        if not history_path.exists():
            continue
        for hist in c.read_csv(history_path):
            out.append(
                {
                    "run_name": row.get("run_name"),
                    "method_key": row.get("method_key"),
                    "case_rank": row.get("case_rank"),
                    "fixed_id": row.get("fixed_id"),
                    "moving_id": row.get("moving_id"),
                    "alpha": row.get("alpha"),
                    "lr": row.get("lr"),
                    "epoch": hist.get("epoch"),
                    "loss": hist.get("loss"),
                    "data_loss": hist.get("data_loss"),
                    "reg_loss": hist.get("reg_loss"),
                    "history_path": str(history_path),
                }
            )
    return out


def pareto_front(rows: Sequence[Dict[str, object]], acc_key: str, reg_key: str, higher: bool) -> List[Dict[str, object]]:
    usable = [row for row in rows if is_finite(row.get(acc_key)) and is_finite(row.get(reg_key))]
    usable.sort(key=lambda r: as_float(r.get(reg_key)))
    front: List[Dict[str, object]] = []
    best = -math.inf if higher else math.inf
    for row in usable:
        score = as_float(row.get(acc_key))
        if (higher and score >= best) or ((not higher) and score <= best):
            front.append(row)
            best = score
    return front


def plot_contact_sheet(rows: Sequence[Dict[str, object]], path: Path, title: str) -> None:
    n_rows = len(ACCURACY_METRICS)
    n_cols = len(REGULARITY_METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), squeeze=False)
    for row_idx, (acc_key, acc_label, higher) in enumerate(ACCURACY_METRICS):
        for col_idx, (reg_key, reg_label) in enumerate(REGULARITY_METRICS):
            ax = axes[row_idx][col_idx]
            for method_key in EXPECTED:
                group = [
                    row
                    for row in rows
                    if str(row.get("method_key")) == method_key
                    and is_finite(row.get(acc_key))
                    and is_finite(row.get(reg_key))
                ]
                group.sort(key=lambda r: as_float(r.get("alpha")))
                if not group:
                    continue
                xs = [as_float(row.get(reg_key)) for row in group]
                ys = [as_float(row.get(acc_key)) for row in group]
                ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=4.5, color=METHOD_COLORS[method_key], label=METHOD_LABELS[method_key])
                for row, x, y in zip(group, xs, ys):
                    alpha = as_float(row.get("alpha"))
                    ax.annotate(f"{alpha:g}", (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
            front = pareto_front(rows, acc_key, reg_key, higher)
            if len(front) >= 2:
                front = sorted(front, key=lambda r: as_float(r.get(reg_key)))
                ax.plot(
                    [as_float(row.get(reg_key)) for row in front],
                    [as_float(row.get(acc_key)) for row in front],
                    color="black",
                    linewidth=1.0,
                    alpha=0.5,
                    linestyle="--",
                    label="global Pareto",
                )
            ax.set_xlabel(reg_label)
            ax.set_ylabel(acc_label)
            ax.grid(True, alpha=0.25)
            if col_idx == 0 and row_idx == 0:
                ax.legend(fontsize=7)
            if "jdet" not in reg_key and "ndv" not in reg_key and "log_jdet" not in reg_key:
                ax.set_xscale("symlog", linthresh=1e-8)
    fig.suptitle(f"{title}: accuracy vs foreground regularity", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_loss_curves(history_rows: Sequence[Dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    metrics = [("data_loss", "Data loss"), ("reg_loss", "Reg loss"), ("loss", "Total loss"), ("loss", "Total loss, zoom")]
    for ax, (key, ylabel) in zip(axes, metrics):
        for method_key in EXPECTED:
            for alpha in EXPECTED[method_key]:
                rows = [
                    row
                    for row in history_rows
                    if str(row.get("method_key")) == method_key
                    and math.isclose(as_float(row.get("alpha")), alpha, rel_tol=1e-9, abs_tol=1e-12)
                ]
                rows.sort(key=lambda r: as_float(r.get("epoch")))
                if not rows:
                    continue
                label = f"{METHOD_LABELS[method_key]} a={alpha:g}"
                ax.plot(
                    [as_float(row.get("epoch")) for row in rows],
                    [as_float(row.get(key)) for row in rows],
                    linewidth=1.0,
                    color=METHOD_COLORS[method_key],
                    alpha=0.45 + 0.15 * EXPECTED[method_key].index(alpha),
                    label=label,
                )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Epoch")
    axes[-1].set_ylim(bottom=None, top=0.0)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def best_alpha_lines(rows: Sequence[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for method_key in EXPECTED:
        group = [row for row in rows if str(row.get("method_key")) == method_key]
        if not group:
            continue
        label = METHOD_LABELS[method_key]
        best_dice = max(group, key=lambda row: as_float(row.get("dice_mean")))
        best_hd95 = min(group, key=lambda row: as_float(row.get("hd95_mean")))
        best_asd = min(group, key=lambda row: as_float(row.get("asd_symmetric_mean")))
        lines.append(
            f"- {label}: best Dice alpha `{as_float(best_dice.get('alpha')):g}`, "
            f"best HD95 alpha `{as_float(best_hd95.get('alpha')):g}`, "
            f"best symmetric ASD alpha `{as_float(best_asd.get('alpha')):g}`."
        )
    return lines


def write_report(
    review_root: Path,
    out_root: Path,
    rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
    *,
    case_label: str,
    expected_original_case_index: int,
    expected_fixed_id: str,
    expected_moving_id: str,
) -> None:
    lines = [
        "# OASIS Alpha Sweep Pre-Run Review",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Large output root: `{out_root}`",
        f"- Review root: `{review_root}`",
        f"- Expected runs: `12`",
        f"- Completed summaries: `{len(rows)}`",
        f"- Case: {case_label}, original sorted index `{expected_original_case_index}`, fixed/moving `({expected_fixed_id}, {expected_moving_id})`",
        f"- Foreground rule: fixed raw image `> 0`",
        "- Native analytic regularity: IDIR uses coordinate-autograd BE/diffusion; spline methods use B-spline analytic BE/diffusion.",
        "- Contact sheet plots native analytic BE/diffusion; spline-only analytic columns remain in CSV/JSON for audit.",
        "",
        "## Acceptance",
        "",
    ]
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors)
    else:
        lines.append("- PASS: all expected runs are present with the expected case IDs.")
    lines.extend(["", "## Best Alpha By Method", ""])
    lines.extend(best_alpha_lines(rows))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary_runs.csv`",
            "- `summary_runs.json`",
            "- `loss_history_all.csv`",
            "- `pareto_contact_sheet.pdf`",
            "- `loss_curves.pdf`",
            "",
        ]
    )
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = collect_summaries(args.out_root)
    errors = check_expected(
        rows,
        expected_original_case_index=int(args.expected_original_case_index),
        expected_fixed_id=str(args.expected_fixed_id),
        expected_moving_id=str(args.expected_moving_id),
    )
    if errors and not args.allow_incomplete:
        c.write_json(args.review_root / "aggregation_errors.json", errors)
        raise RuntimeError("Aggregation failed acceptance checks: " + "; ".join(errors))
    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "summary_runs.csv", rows)
    c.write_json(args.review_root / "summary_runs.json", rows)
    history_rows = collect_history(rows)
    c.write_csv(args.review_root / "loss_history_all.csv", history_rows)
    plot_contact_sheet(rows, args.review_root / "pareto_contact_sheet.pdf", str(args.figure_title))
    plot_loss_curves(history_rows, args.review_root / "loss_curves.pdf")
    write_report(
        args.review_root,
        args.out_root,
        rows,
        errors,
        case_label=str(args.case_label),
        expected_original_case_index=int(args.expected_original_case_index),
        expected_fixed_id=str(args.expected_fixed_id),
        expected_moving_id=str(args.expected_moving_id),
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "history_rows": len(history_rows),
                "errors": errors,
                "review_root": str(args.review_root),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
