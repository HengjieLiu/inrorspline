#!/usr/bin/env python
"""Aggregate the OASIS seed-3 first10 alpha sweep into population ARC curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


DEFAULT_RUN_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep"
IDIR_ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
SPLINE_ALPHAS = [0.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

METHOD_SPECS = {
    "idir_original_dense_siren_be_autograd": {
        "label": "IDIR",
        "method": "idir_original",
        "lr": 1e-4,
        "alphas": IDIR_ALPHAS,
        "color": "#1f77b4",
    },
    "sinr_cps4_be_bspline_analytic": {
        "label": "SINR cps4",
        "method": "sinr_cps4_be_bspline_analytic",
        "lr": 1e-4,
        "alphas": SPLINE_ALPHAS,
        "color": "#2ca02c",
    },
    "sinr_cps2_be_bspline_analytic": {
        "label": "SINR cps2",
        "method": "sinr_cps2_be_bspline_analytic",
        "lr": 1e-4,
        "alphas": SPLINE_ALPHAS,
        "color": "#9467bd",
    },
    "direct_cp_cps4_be_bspline_analytic": {
        "label": "DirectCP cps4 lr1e-3",
        "method": "direct_cp_cps4_be_bspline_analytic",
        "lr": 1e-3,
        "alphas": SPLINE_ALPHAS,
        "color": "#d62728",
    },
}
DIRECT_CP_CPS2_METHOD_SPEC = {
    "label": "DirectCP cps2 lr1e-3",
    "method": "direct_cp_cps2_be_bspline_analytic",
    "lr": 1e-3,
    "alphas": SPLINE_ALPHAS,
    "color": "#ff7f0e",
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

PRIMARY_ARC_PANELS: List[Tuple[str, str, bool]] = [
    ("dice_mean", "jdet_le0_pct_fg", True),
    ("dice_mean", "log_jdet_std_fg", True),
    ("dice_mean", "be_native_analytic_fixed_fg_voxel", True),
    ("dice_mean", "diffusion_native_analytic_fixed_fg_voxel", True),
]

SUMMARY_METRICS = [
    "dense_image_loss",
    "final_loss",
    "final_data_loss",
    "final_reg_loss",
    "dice_mean",
    "hd95_mean",
    "asd_warped_to_fixed_mean",
    "asd_fixed_to_warped_mean",
    "asd_symmetric_mean",
    "diffusion_l2_full",
    "diffusion_l2_fg",
    "be_fd_forward_full_voxel",
    "be_fd_forward_fixed_fg_voxel",
    "be_native_analytic_full_voxel",
    "be_native_analytic_fixed_fg_voxel",
    "diffusion_fd_forward_full_voxel",
    "diffusion_fd_forward_fixed_fg_voxel",
    "diffusion_native_analytic_full_voxel",
    "diffusion_native_analytic_fixed_fg_voxel",
    "jdet_le0_pct_full",
    "jdet_le0_pct_fg",
    "log_jdet_std_full",
    "log_jdet_std_fg",
    "log_jdet_pos_frac_full",
    "log_jdet_pos_frac_fg",
    "jdet_mean_full",
    "jdet_mean_fg",
    "ndv_full",
    "ndv_fg",
    "ndv_frac_full",
    "ndv_frac_fg",
    "training_elapsed_sec",
    "total_elapsed_sec",
    "training_peak_mem_gb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--case-ranks", type=str, default="0-9")
    parser.add_argument("--expected-epochs", type=int, default=2500)
    parser.add_argument("--expected-eval-interval", type=int, default=250)
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument("--expected-variant", type=str, default="oasis_seed3_first10_alpha_sweep_be_2500e_v1")
    parser.add_argument("--include-direct-cp-cps2", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-history", action="store_true")
    return parser.parse_args()


def configure_method_specs(args: argparse.Namespace) -> None:
    global METHOD_SPECS
    if args.include_direct_cp_cps2:
        specs = dict(METHOD_SPECS)
        specs["direct_cp_cps2_be_bspline_analytic"] = dict(DIRECT_CP_CPS2_METHOD_SPEC)
        METHOD_SPECS = specs


def parse_case_ranks(text: str) -> List[int]:
    ranks: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start = int(start_text)
            stop = int(stop_text)
            step = 1 if stop >= start else -1
            ranks.extend(range(start, stop + step, step))
        else:
            ranks.append(int(token))
    out: List[int] = []
    for rank in ranks:
        if rank not in out:
            out.append(rank)
    if not out:
        raise ValueError(f"No case ranks parsed from {text!r}")
    return out


def case_ranks_short_label(case_ranks: Sequence[int]) -> str:
    ranks = [int(rank) for rank in case_ranks]
    if ranks == list(range(10)):
        return "first10"
    if ranks == list(range(10, 20)):
        return "next10"
    if ranks and ranks == list(range(ranks[0], ranks[-1] + 1)):
        return f"preliminary ranks {ranks[0]:03d}-{ranks[-1]:03d}"
    return "preliminary ranks " + ",".join(f"{rank:03d}" for rank in ranks)


def case_ranks_report_label(case_ranks: Sequence[int]) -> str:
    ranks = [int(rank) for rank in case_ranks]
    if ranks == list(range(10)):
        return "First10"
    if ranks == list(range(10, 20)):
        return "Next10"
    if ranks and ranks == list(range(ranks[0], ranks[-1] + 1)):
        return f"Preliminary Ranks {ranks[0]:03d}-{ranks[-1]:03d}"
    return "Preliminary Ranks " + ",".join(f"{rank:03d}" for rank in ranks)


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(value: object) -> bool:
    return math.isfinite(as_float(value))


def alpha_id(value: object) -> str:
    return f"{as_float(value):.12g}"


def combo_key(row: Mapping[str, object]) -> Tuple[int, str, str, str]:
    return (
        int(as_float(row.get("case_rank"))),
        str(row.get("method_key")),
        alpha_id(row.get("alpha")),
        alpha_id(row.get("lr")),
    )


def group_key(row: Mapping[str, object]) -> Tuple[str, str, str]:
    return (str(row.get("method_key")), alpha_id(row.get("alpha")), alpha_id(row.get("lr")))


def expected_cases(args: argparse.Namespace) -> Dict[int, c.CaseInfo]:
    return {
        rank: c.resolve_case(rank=rank, pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for rank in parse_case_ranks(str(args.case_ranks))
    }


def expected_combos(case_ranks: Sequence[int]) -> Dict[Tuple[int, str, str, str], Dict[str, object]]:
    expected: Dict[Tuple[int, str, str, str], Dict[str, object]] = {}
    for rank in case_ranks:
        for method_key, spec in METHOD_SPECS.items():
            for alpha in spec["alphas"]:  # type: ignore[index]
                key = (int(rank), method_key, alpha_id(alpha), alpha_id(spec["lr"]))
                expected[key] = {
                    "case_rank": rank,
                    "method_key": method_key,
                    "alpha": float(alpha),
                    "lr": float(spec["lr"]),
                }
    return expected


def expected_run_count(case_ranks: Sequence[int]) -> int:
    return int(len(case_ranks) * sum(len(spec["alphas"]) for spec in METHOD_SPECS.values()))  # type: ignore[arg-type]


def collect_summaries(out_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(out_root.glob("*/*/summary.json")):
        row = json.loads(path.read_text())
        row["summary_path"] = str(path)
        row["run_dir_actual"] = str(path.parent)
        rows.append(row)
    rows.sort(key=lambda r: (int(as_float(r.get("case_rank"))), str(r.get("method_key")), as_float(r.get("alpha"))))
    return rows


def filter_summaries_by_case_rank(
    rows: Sequence[Dict[str, object]],
    case_ranks: Sequence[int],
) -> List[Dict[str, object]]:
    allowed = {int(rank) for rank in case_ranks}
    return [row for row in rows if int(as_float(row.get("case_rank"))) in allowed]


def required_artifact_errors(row: Mapping[str, object]) -> List[str]:
    run_dir = Path(str(row["run_dir_actual"]))
    fixed_id = str(row.get("fixed_id"))
    moving_id = str(row.get("moving_id"))
    required = [
        "DONE",
        "summary.json",
        "history.csv",
        "case_metrics.csv",
        "regularity_metrics.csv",
        "be_metrics.csv",
        "diffusion_metrics.csv",
        "model_state.pt",
        "mask_audit.json",
        f"disp_norm_{fixed_id}_{moving_id}.npy",
    ]
    if str(row.get("method_key")) != "idir_original_dense_siren_be_autograd":
        required.append("control_output.pt")
    return [f"Missing {run_dir / name}" for name in required if not (run_dir / name).exists()]


def check_expected(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    cases = expected_cases(args)
    expected = expected_combos(list(cases.keys()))
    observed: Dict[Tuple[int, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        observed[combo_key(row)].append(row)

    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    for key in missing[:50]:
        errors.append(f"Missing run combo: case_rank={key[0]}, method={key[1]}, alpha={key[2]}, lr={key[3]}")
    if len(missing) > 50:
        errors.append(f"... {len(missing) - 50} additional missing run combos")
    for key in extra[:50]:
        errors.append(f"Unexpected run combo: case_rank={key[0]}, method={key[1]}, alpha={key[2]}, lr={key[3]}")
    if len(extra) > 50:
        errors.append(f"... {len(extra) - 50} additional unexpected run combos")
    for key, group in sorted(observed.items()):
        if len(group) > 1:
            errors.append(f"Duplicate run combo {key}: {len(group)} summaries")

    for row in rows:
        run_name = str(row.get("run_name"))
        rank = int(as_float(row.get("case_rank")))
        case = cases.get(rank)
        if case is None:
            continue
        if int(as_float(row.get("original_case_index"))) != int(case.original_case_index):
            errors.append(f"{run_name}: original_case_index {row.get('original_case_index')} != {case.original_case_index}")
        if str(row.get("fixed_id")) != case.fixed_id or str(row.get("moving_id")) != case.moving_id:
            errors.append(f"{run_name}: fixed/moving {row.get('fixed_id')}/{row.get('moving_id')} != {case.fixed_id}/{case.moving_id}")
        if int(as_float(row.get("epochs"))) != int(args.expected_epochs):
            errors.append(f"{run_name}: epochs {row.get('epochs')} != {args.expected_epochs}")
        if int(as_float(row.get("eval_interval"))) != int(args.expected_eval_interval):
            errors.append(f"{run_name}: eval_interval {row.get('eval_interval')} != {args.expected_eval_interval}")
        if int(as_float(row.get("seed"))) != int(args.expected_seed):
            errors.append(f"{run_name}: seed {row.get('seed')} != {args.expected_seed}")
        if str(row.get("variant")) != str(args.expected_variant):
            errors.append(f"{run_name}: variant {row.get('variant')} != {args.expected_variant}")
        if str(row.get("grid_order")) != "pytorch":
            errors.append(f"{run_name}: grid_order {row.get('grid_order')} != pytorch")
        if str(row.get("foreground_metric_mask")) != "fixed_image_positive":
            errors.append(f"{run_name}: foreground_metric_mask {row.get('foreground_metric_mask')} != fixed_image_positive")
        if str(row.get("segmentation_mask_role")) != "dice_hd95_asd_only":
            errors.append(f"{run_name}: segmentation_mask_role {row.get('segmentation_mask_role')} != dice_hd95_asd_only")
        if str(row.get("foreground_rule")) != "fixed raw image > 0":
            errors.append(f"{run_name}: foreground_rule {row.get('foreground_rule')} != fixed raw image > 0")
        method_key = str(row.get("method_key"))
        if method_key == "idir_original_dense_siren_be_autograd":
            if str(row.get("sample_domain")) != "fixed_image_positive":
                errors.append(f"{run_name}: IDIR sample_domain {row.get('sample_domain')} != fixed_image_positive")
            if str(row.get("native_analytic_be_formula")) != "idir_coordinate_autograd_hessian":
                errors.append(f"{run_name}: IDIR native BE formula is {row.get('native_analytic_be_formula')}")
            if str(row.get("native_analytic_diffusion_formula")) != "idir_coordinate_autograd_jacobian":
                errors.append(f"{run_name}: IDIR native diffusion formula is {row.get('native_analytic_diffusion_formula')}")
        else:
            if str(row.get("sample_domain")) != "none_full_image_ncc":
                errors.append(f"{run_name}: spline sample_domain {row.get('sample_domain')} != none_full_image_ncc")
            if str(row.get("native_analytic_be_formula")) != "bspline_analytic":
                errors.append(f"{run_name}: spline native BE formula is {row.get('native_analytic_be_formula')}")
            if str(row.get("native_analytic_diffusion_formula")) != "bspline_analytic":
                errors.append(f"{run_name}: spline native diffusion formula is {row.get('native_analytic_diffusion_formula')}")
            if str(row.get("bspline_mode")) != "disp":
                errors.append(f"{run_name}: bspline_mode {row.get('bspline_mode')} != disp")
        for metric in ["dense_image_loss", "dice_mean", "hd95_mean", "asd_symmetric_mean", "be_native_analytic_fixed_fg_voxel", "diffusion_native_analytic_fixed_fg_voxel"]:
            if not is_finite(row.get(metric)):
                errors.append(f"{run_name}: missing/non-finite required metric {metric}")
        errors.extend(f"{run_name}: {error}" for error in required_artifact_errors(row))
    return errors


def context_from_summary(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "run_name": row.get("run_name"),
        "method_key": row.get("method_key"),
        "case_rank": row.get("case_rank"),
        "original_case_index": row.get("original_case_index"),
        "fixed_id": row.get("fixed_id"),
        "moving_id": row.get("moving_id"),
        "alpha": row.get("alpha"),
        "lr": row.get("lr"),
        "summary_path": row.get("summary_path"),
    }


def collect_metric_file(rows: Sequence[Dict[str, object]], filename: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        path = Path(str(row["run_dir_actual"])) / filename
        if not path.exists():
            continue
        context = context_from_summary(row)
        for metric_row in c.read_csv(path):
            merged = dict(context)
            merged.update(metric_row)
            merged[f"{filename}_path"] = str(path)
            out.append(merged)
    return out


def collect_history(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        path = Path(str(row["run_dir_actual"])) / "history.csv"
        if not path.exists():
            continue
        context = context_from_summary(row)
        for hist in c.read_csv(path):
            merged = dict(context)
            merged.update(hist)
            merged["history_path"] = str(path)
            out.append(merged)
    return out


def metric_keys(rows: Sequence[Mapping[str, object]]) -> List[str]:
    keys = list(SUMMARY_METRICS)
    for label in c.OASIS_LABELS:
        keys.extend(
            [
                f"dice_label_{label:02d}",
                f"hd95_label_{label:02d}",
                f"asd_warped_to_fixed_label_{label:02d}",
                f"asd_fixed_to_warped_label_{label:02d}",
                f"asd_symmetric_label_{label:02d}",
            ]
        )
    out: List[str] = []
    for key in keys:
        if key in out:
            continue
        if any(is_finite(row.get(key)) for row in rows):
            out.append(key)
    return out


def stats(values: Iterable[object]) -> Dict[str, float | int]:
    arr = np.asarray([as_float(value) for value in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def population_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    keys = metric_keys(rows)
    out: List[Dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: (list(METHOD_SPECS).index(item[0][0]), as_float(item[0][1]))):
        method_key, alpha_text, lr_text = key
        case_ranks = sorted(int(as_float(row.get("case_rank"))) for row in group)
        row: Dict[str, object] = {
            "method_key": method_key,
            "method_label": METHOD_SPECS[method_key]["label"],
            "alpha": as_float(alpha_text),
            "lr": as_float(lr_text),
            "n_cases": len(group),
            "case_ranks_text": ",".join(str(rank) for rank in case_ranks),
        }
        for metric in keys:
            metric_stats = stats(group_row.get(metric) for group_row in group)
            for stat_name, value in metric_stats.items():
                row[f"{metric}_{stat_name}"] = value
        out.append(row)
    return out


def population_label_metrics(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    out: List[Dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: (list(METHOD_SPECS).index(item[0][0]), as_float(item[0][1]))):
        method_key, alpha_text, lr_text = key
        for label in c.OASIS_LABELS:
            row: Dict[str, object] = {
                "method_key": method_key,
                "method_label": METHOD_SPECS[method_key]["label"],
                "alpha": as_float(alpha_text),
                "lr": as_float(lr_text),
                "label": label,
                "n_cases": len(group),
            }
            for metric in [
                f"dice_label_{label:02d}",
                f"hd95_label_{label:02d}",
                f"asd_warped_to_fixed_label_{label:02d}",
                f"asd_fixed_to_warped_label_{label:02d}",
                f"asd_symmetric_label_{label:02d}",
            ]:
                metric_stats = stats(group_row.get(metric) for group_row in group)
                metric_name = metric.rsplit("_label_", 1)[0]
                for stat_name, value in metric_stats.items():
                    row[f"{metric_name}_{stat_name}"] = value
            out.append(row)
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


def err_value(row: Mapping[str, object], metric: str) -> float:
    value = as_float(row.get(f"{metric}_std"))
    return value if math.isfinite(value) else 0.0


def plot_population_grid(pop_rows: Sequence[Dict[str, object]], path: Path, title: str) -> None:
    n_rows = len(ACCURACY_METRICS)
    n_cols = len(REGULARITY_METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 3.3 * n_rows), squeeze=False)
    for row_idx, (acc_key, acc_label, higher) in enumerate(ACCURACY_METRICS):
        acc_mean = f"{acc_key}_mean"
        for col_idx, (reg_key, reg_label) in enumerate(REGULARITY_METRICS):
            reg_mean = f"{reg_key}_mean"
            ax = axes[row_idx][col_idx]
            for method_key, spec in METHOD_SPECS.items():
                group = [
                    row
                    for row in pop_rows
                    if str(row.get("method_key")) == method_key
                    and is_finite(row.get(acc_mean))
                    and is_finite(row.get(reg_mean))
                ]
                group.sort(key=lambda r: as_float(r.get("alpha")))
                if not group:
                    continue
                xs = [as_float(row.get(reg_mean)) for row in group]
                ys = [as_float(row.get(acc_mean)) for row in group]
                xerr = [err_value(row, reg_key) for row in group]
                yerr = [err_value(row, acc_key) for row in group]
                ax.errorbar(
                    xs,
                    ys,
                    xerr=xerr,
                    yerr=yerr,
                    marker="o",
                    linewidth=1.2,
                    markersize=4.5,
                    capsize=2,
                    color=str(spec["color"]),
                    label=str(spec["label"]),
                )
                for row, x, y in zip(group, xs, ys):
                    ax.annotate(f"{as_float(row.get('alpha')):g}", (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
            front = pareto_front(pop_rows, acc_mean, reg_mean, higher)
            if len(front) >= 2:
                front = sorted(front, key=lambda r: as_float(r.get(reg_mean)))
                ax.plot(
                    [as_float(row.get(reg_mean)) for row in front],
                    [as_float(row.get(acc_mean)) for row in front],
                    color="black",
                    linewidth=1.0,
                    alpha=0.5,
                    linestyle="--",
                    label="population Pareto",
                )
            ax.set_xlabel(reg_label)
            ax.set_ylabel(acc_label)
            ax.grid(True, alpha=0.25)
            if col_idx == 0 and row_idx == 0:
                ax.legend(fontsize=7)
            if "jdet" not in reg_key and "ndv" not in reg_key and "log_jdet" not in reg_key:
                ax.set_xscale("symlog", linthresh=1e-8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_primary_arc(pop_rows: Sequence[Dict[str, object]], path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), squeeze=False)
    label_by_acc = {key: label for key, label, _higher in ACCURACY_METRICS}
    label_by_reg = {key: label for key, label in REGULARITY_METRICS}
    for ax, (acc_key, reg_key, higher) in zip(axes.ravel(), PRIMARY_ARC_PANELS):
        acc_mean = f"{acc_key}_mean"
        reg_mean = f"{reg_key}_mean"
        for method_key, spec in METHOD_SPECS.items():
            group = [
                row
                for row in pop_rows
                if str(row.get("method_key")) == method_key
                and is_finite(row.get(acc_mean))
                and is_finite(row.get(reg_mean))
            ]
            group.sort(key=lambda r: as_float(r.get("alpha")))
            if not group:
                continue
            xs = [as_float(row.get(reg_mean)) for row in group]
            ys = [as_float(row.get(acc_mean)) for row in group]
            ax.plot(xs, ys, marker="o", linewidth=1.5, color=str(spec["color"]), label=str(spec["label"]))
            for row, x, y in zip(group, xs, ys):
                ax.annotate(f"{as_float(row.get('alpha')):g}", (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
        front = pareto_front(pop_rows, acc_mean, reg_mean, higher)
        if len(front) >= 2:
            front = sorted(front, key=lambda r: as_float(r.get(reg_mean)))
            ax.plot(
                [as_float(row.get(reg_mean)) for row in front],
                [as_float(row.get(acc_mean)) for row in front],
                color="black",
                linewidth=1.2,
                alpha=0.6,
                linestyle="--",
                label="population Pareto",
            )
        ax.set_xlabel(label_by_reg[reg_key])
        ax.set_ylabel(label_by_acc[acc_key])
        ax.grid(True, alpha=0.25)
        if "jdet" not in reg_key and "ndv" not in reg_key and "log_jdet" not in reg_key:
            ax.set_xscale("symlog", linthresh=1e-8)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate_loss_history(history_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in history_rows:
        key = (
            str(row.get("method_key")),
            alpha_id(row.get("alpha")),
            alpha_id(row.get("lr")),
            int(as_float(row.get("epoch"))),
        )
        groups[key].append(row)
    out: List[Dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: (list(METHOD_SPECS).index(item[0][0]), as_float(item[0][1]), item[0][3])):
        method_key, alpha_text, lr_text, epoch = key
        row: Dict[str, object] = {
            "method_key": method_key,
            "method_label": METHOD_SPECS[method_key]["label"],
            "alpha": as_float(alpha_text),
            "lr": as_float(lr_text),
            "epoch": epoch,
            "n_cases": len(group),
        }
        for metric in ["loss", "data_loss", "reg_loss"]:
            metric_stats = stats(item.get(metric) for item in group)
            for stat_name, value in metric_stats.items():
                row[f"{metric}_{stat_name}"] = value
        out.append(row)
    return out


def plot_loss_curves(loss_rows: Sequence[Dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for ax, metric, label in zip(axes, ["data_loss", "reg_loss", "loss"], ["Data loss", "Reg loss", "Total loss"]):
        mean_key = f"{metric}_mean"
        for method_key, spec in METHOD_SPECS.items():
            alphas = sorted({as_float(row.get("alpha")) for row in loss_rows if str(row.get("method_key")) == method_key})
            for alpha in alphas:
                group = [
                    row
                    for row in loss_rows
                    if str(row.get("method_key")) == method_key
                    and math.isclose(as_float(row.get("alpha")), alpha, rel_tol=1e-9, abs_tol=1e-12)
                    and is_finite(row.get(mean_key))
                ]
                group.sort(key=lambda r: as_float(r.get("epoch")))
                if not group:
                    continue
                ax.plot(
                    [as_float(row.get("epoch")) for row in group],
                    [as_float(row.get(mean_key)) for row in group],
                    linewidth=0.9,
                    color=str(spec["color"]),
                    alpha=0.35 + 0.1 * min(5, len(alphas)),
                    label=f"{spec['label']} a={alpha:g}",
                )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Epoch")
    axes[0].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def best_alpha_lines(pop_rows: Sequence[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for method_key, spec in METHOD_SPECS.items():
        group = [row for row in pop_rows if str(row.get("method_key")) == method_key]
        if not group:
            continue
        best_dice = max(group, key=lambda row: as_float(row.get("dice_mean_mean")))
        best_hd95 = min(group, key=lambda row: as_float(row.get("hd95_mean_mean")))
        best_asd = min(group, key=lambda row: as_float(row.get("asd_symmetric_mean_mean")))
        lines.append(
            f"- {spec['label']}: best mean Dice alpha `{as_float(best_dice.get('alpha')):g}`, "
            f"best mean HD95 alpha `{as_float(best_hd95.get('alpha')):g}`, "
            f"best mean symmetric ASD alpha `{as_float(best_asd.get('alpha')):g}`."
        )
    return lines


def write_report(
    review_root: Path,
    out_root: Path,
    summary_rows: Sequence[Dict[str, object]],
    pop_rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
    *,
    case_ranks: Sequence[int],
    skip_history: bool,
) -> None:
    scope_label = case_ranks_report_label(case_ranks)
    lines = [
        f"# OASIS Seed-3 {scope_label} Alpha Sweep Review",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Large output root: `{out_root}`",
        f"- Review root: `{review_root}`",
        f"- Expected runs: `{expected_run_count(case_ranks)}`",
        f"- Completed summaries: `{len(summary_rows)}`",
        f"- Case ranks: `{','.join(str(rank) for rank in case_ranks)}`",
        "- Split: seed-3 shuffled `pair_test_200`; selected ranks follow the shuffled protocol order.",
        "- Population ARC convention: aggregate each method/alpha across cases first, then draw the Pareto/front curves.",
        "- Native analytic regularity: IDIR uses coordinate-autograd BE/diffusion; spline methods use B-spline analytic BE/diffusion.",
        "- Contact sheets plot native analytic BE/diffusion; spline-only analytic columns remain in CSV/JSON for audit.",
        "",
        "## Acceptance",
        "",
    ]
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors[:80])
        if len(errors) > 80:
            lines.append(f"- FAIL: ... {len(errors) - 80} additional errors")
    else:
        lines.append("- PASS: all expected runs are present with expected cases, alphas, metadata, and artifacts.")
    lines.extend(["", "## Best Alpha By Population Mean", ""])
    lines.extend(best_alpha_lines(pop_rows))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary_runs.csv/json`: one row per run, including per-label metrics and regularity columns.",
            "- `population_summary.csv/json`: one row per method/alpha, with across-case mean/std/median/min/max.",
            "- `population_label_metrics.csv`: per-label population aggregates.",
            "- `case_metrics_all.csv`: per-run Dice/HD95/ASD with per-label columns.",
            "- `regularity_metrics_all.csv`: per-run full-field and foreground regularity metrics.",
            "- `be_metrics_all.csv` and `diffusion_metrics_all.csv`: long-form BE/diffusion audit rows.",
            "- `population_arc_primary.pdf`: focused ARC-style Dice-vs-regularity curves.",
            "- `population_pareto_contact_sheet.pdf`: all accuracy-regularity population curves.",
            "- `loss_curves_population.pdf`: across-case mean loss curves." if not skip_history else "- Loss history aggregation skipped by request.",
            "",
        ]
    )
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def write_csv_stream(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fieldnames and not isinstance(value, (list, dict)):
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    configure_method_specs(args)
    case_ranks = parse_case_ranks(str(args.case_ranks))
    scope_label = case_ranks_short_label(case_ranks)
    summary_rows = filter_summaries_by_case_rank(collect_summaries(Path(args.out_root)), case_ranks)
    errors = check_expected(summary_rows, args)
    if errors and not args.allow_incomplete:
        c.write_json(Path(args.review_root) / "aggregation_errors.json", errors)
        raise RuntimeError("Aggregation failed acceptance checks: " + "; ".join(errors[:20]))

    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "summary_runs.csv", summary_rows)
    c.write_json(args.review_root / "summary_runs.json", summary_rows)
    case_metric_rows = collect_metric_file(summary_rows, "case_metrics.csv")
    regularity_rows = collect_metric_file(summary_rows, "regularity_metrics.csv")
    be_rows = collect_metric_file(summary_rows, "be_metrics.csv")
    diffusion_rows = collect_metric_file(summary_rows, "diffusion_metrics.csv")
    c.write_csv(args.review_root / "case_metrics_all.csv", case_metric_rows)
    c.write_csv(args.review_root / "regularity_metrics_all.csv", regularity_rows)
    c.write_csv(args.review_root / "be_metrics_all.csv", be_rows)
    c.write_csv(args.review_root / "diffusion_metrics_all.csv", diffusion_rows)

    pop_rows = population_summary(summary_rows)
    label_rows = population_label_metrics(summary_rows)
    c.write_csv(args.review_root / "population_summary.csv", pop_rows)
    c.write_json(args.review_root / "population_summary.json", pop_rows)
    c.write_csv(args.review_root / "population_label_metrics.csv", label_rows)
    plot_population_grid(
        pop_rows,
        args.review_root / "population_pareto_contact_sheet.pdf",
        f"OASIS seed-3 {scope_label}: population accuracy vs foreground regularity",
    )
    plot_primary_arc(
        pop_rows,
        args.review_root / "population_arc_primary.pdf",
        f"OASIS seed-3 {scope_label}: primary population ARC curves",
    )

    if not args.skip_history:
        history_rows = collect_history(summary_rows)
        loss_rows = aggregate_loss_history(history_rows)
        write_csv_stream(args.review_root / "loss_history_all.csv", history_rows)
        c.write_csv(args.review_root / "loss_history_population.csv", loss_rows)
        plot_loss_curves(loss_rows, args.review_root / "loss_curves_population.pdf")
    else:
        history_rows = []

    write_report(
        args.review_root,
        args.out_root,
        summary_rows,
        pop_rows,
        errors,
        case_ranks=case_ranks,
        skip_history=bool(args.skip_history),
    )
    print(
        json.dumps(
            {
                "summary_rows": len(summary_rows),
                "population_rows": len(pop_rows),
                "case_metric_rows": len(case_metric_rows),
                "regularity_rows": len(regularity_rows),
                "be_rows": len(be_rows),
                "diffusion_rows": len(diffusion_rows),
                "history_rows": len(history_rows),
                "errors": errors,
                "review_root": str(args.review_root),
            },
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
