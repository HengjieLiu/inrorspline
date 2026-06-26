#!/usr/bin/env python
"""Generate OASIS first10/first20 five-sweep Pareto plots with baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
import numpy as np  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import aggregate_first10_population as base  # noqa: E402
import common as c  # noqa: E402


DEFAULT_REVIEW_ROOT = (
    c.REPO_ROOT
    / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/"
    "pareto_first10_first20_alpha5_saved_baselines_paper_v1"
)
DEFAULT_FIRST10_MAIN_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1")
DEFAULT_FIRST10_CPS2_PRELIM_ROOT = Path(
    "/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1"
)
DEFAULT_FIRST10_CPS2_COMPLETION_ROOT = Path(
    "/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_completion_a0_10_100000_v1"
)
DEFAULT_NEXT10_ROOT = Path("/database/inrorspline/runs/oasis_seed3_next10_alpha_sweep_be_2500e_5models_v1")
DEFAULT_BASELINE_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_saved_baselines_v1")

IDIR_ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
SPLINE_ALPHAS = [0.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

SWEEP_METHODS = [
    "idir_original_dense_siren_be_autograd",
    "sinr_cps4_be_bspline_analytic",
    "direct_cp_cps4_be_bspline_analytic",
    "sinr_cps2_be_bspline_analytic",
    "direct_cp_cps2_be_bspline_analytic",
]
BASELINE_METHODS = [
    "bs1_vxm_v0_2",
    "bs2_tm_v0",
    "bs3_vfa_v0",
    "bs4_sitreg_v0",
    "greedy",
]

METHOD_SPECS: Dict[str, Dict[str, object]] = {
    "idir_original_dense_siren_be_autograd": {
        "label": "IDIR",
        "dot_label": "IDIR",
        "lr": 1e-4,
        "alphas": IDIR_ALPHAS,
        "color": "#005AB5",
    },
    "sinr_cps4_be_bspline_analytic": {
        "label": "SINR cps4",
        "dot_label": "SINR4",
        "lr": 1e-4,
        "alphas": SPLINE_ALPHAS,
        "color": "#009E73",
    },
    "direct_cp_cps4_be_bspline_analytic": {
        "label": "DirectCP cps4",
        "dot_label": "DCP4",
        "lr": 1e-3,
        "alphas": SPLINE_ALPHAS,
        "color": "#66CDAA",
    },
    "sinr_cps2_be_bspline_analytic": {
        "label": "SINR cps2",
        "dot_label": "SINR2",
        "lr": 1e-4,
        "alphas": SPLINE_ALPHAS,
        "color": "#AA3377",
    },
    "direct_cp_cps2_be_bspline_analytic": {
        "label": "DirectCP cps2",
        "dot_label": "DCP2",
        "lr": 1e-3,
        "alphas": SPLINE_ALPHAS,
        "color": "#EE77BB",
    },
    "bs1_vxm_v0_2": {
        "label": "BS1 VoxelMorph",
        "dot_label": "BS1",
        "color": "#F4A3A3",
    },
    "bs2_tm_v0": {
        "label": "BS2 TransMorph",
        "dot_label": "BS2",
        "color": "#E06666",
    },
    "bs3_vfa_v0": {
        "label": "BS3 VFA",
        "dot_label": "BS3",
        "color": "#B22222",
    },
    "bs4_sitreg_v0": {
        "label": "BS4 SITReg",
        "dot_label": "BS4",
        "color": "#8B0000",
    },
    "greedy": {
        "label": "Greedy",
        "dot_label": "Greedy",
        "color": "#D55E00",
    },
}

ACCURACY_METRICS: List[Tuple[str, str, bool]] = [
    ("dice_mean", "Mean Dice", True),
    ("hd95_mean", "Mean HD95", False),
    ("asd_warped_to_fixed_mean", "ASD warped-to-fixed", False),
    ("asd_symmetric_mean", "ASD symmetric", False),
]
REGULARITY_METRICS: List[Tuple[str, str]] = [
    ("be_fd_forward_fixed_fg_voxel", "FG BE forward FD"),
    ("diffusion_fd_forward_fixed_fg_voxel", "FG diffusion forward FD"),
    ("diffusion_l2_fg", "FG diffusion L2"),
    ("jdet_le0_pct_fg", "FG Jdet <= 0 (%)"),
    ("log_jdet_std_fg", "FG std(log Jdet)"),
    ("ndv_frac_fg", "FG NDV fraction"),
]
SPOTLIGHT_PANELS: List[Tuple[str, str]] = [
    ("dice_mean", "log_jdet_std_fg"),
    ("hd95_mean", "log_jdet_std_fg"),
]
STYLES = ["line_segments", "line_segments_errorbars", "ellipses"]
SUMMARY_METRICS = list(base.SUMMARY_METRICS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--first10-main-root", type=Path, default=DEFAULT_FIRST10_MAIN_ROOT)
    parser.add_argument("--first10-cps2-prelim-root", type=Path, default=DEFAULT_FIRST10_CPS2_PRELIM_ROOT)
    parser.add_argument("--first10-cps2-completion-root", type=Path, default=DEFAULT_FIRST10_CPS2_COMPLETION_ROOT)
    parser.add_argument("--next10-root", type=Path, default=DEFAULT_NEXT10_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--expected-epochs", type=int, default=2500)
    parser.add_argument("--expected-eval-interval", type=int, default=250)
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(value: object) -> bool:
    return math.isfinite(as_float(value))


def alpha_id(value: object) -> str:
    return f"{as_float(value):.12g}"


def parse_case_ranks(text: str) -> List[int]:
    return base.parse_case_ranks(text)


def metric_mean(metric: str) -> str:
    return f"{metric}_mean"


def metric_std(metric: str) -> str:
    return f"{metric}_std"


def slug(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return out or "metric"


def label_for_metric(metric: str, metrics: Sequence[Tuple[str, str]]) -> str:
    return dict(metrics)[metric]


def accuracy_label(metric: str) -> str:
    return {key: label for key, label, _higher in ACCURACY_METRICS}[metric]


def accuracy_higher(metric: str) -> bool:
    return {key: higher for key, _label, higher in ACCURACY_METRICS}[metric]


def should_symlog(reg_key: str) -> bool:
    return "jdet" not in reg_key and "ndv" not in reg_key and "log_jdet" not in reg_key


def format_alpha(value: object) -> str:
    value_float = as_float(value)
    if not math.isfinite(value_float):
        return ""
    return f"{value_float:g}"


def method_order(method_key: str) -> int:
    order = SWEEP_METHODS + BASELINE_METHODS
    return order.index(method_key) if method_key in order else len(order)


def read_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return payload


def collect_summary_rows(root: Path) -> List[Dict[str, object]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Missing run root: {root}")
    rows: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/*/summary.json")):
        row = read_json(path)
        row["summary_path"] = str(path)
        row["run_dir_actual"] = str(path.parent)
        row.setdefault("run_dir", str(path.parent))
        row.setdefault("run_name", path.parent.name)
        row["method_key"] = str(row.get("method_key") or row.get("method") or path.parent.parent.name)
        rows.append(row)
    return rows


def filter_ranks(rows: Sequence[Dict[str, object]], ranks: Sequence[int]) -> List[Dict[str, object]]:
    allowed = {int(rank) for rank in ranks}
    return [row for row in rows if int(as_float(row.get("case_rank"))) in allowed]


def combo_key(row: Mapping[str, object]) -> Tuple[str, int, str]:
    return (
        str(row.get("method_key")),
        int(as_float(row.get("case_rank"))),
        alpha_id(row.get("alpha")),
    )


def baseline_key(row: Mapping[str, object]) -> Tuple[str, int]:
    return str(row.get("method_key")), int(as_float(row.get("case_rank")))


def expected_sweep_combos(ranks: Sequence[int]) -> set[Tuple[str, int, str]]:
    expected: set[Tuple[str, int, str]] = set()
    for method in SWEEP_METHODS:
        for rank in ranks:
            for alpha in METHOD_SPECS[method]["alphas"]:  # type: ignore[index]
                expected.add((method, int(rank), alpha_id(alpha)))
    return expected


def expected_baseline_combos(ranks: Sequence[int]) -> set[Tuple[str, int]]:
    return {(method, int(rank)) for method in BASELINE_METHODS for rank in ranks}


def required_sweep_artifact_errors(row: Mapping[str, object]) -> List[str]:
    run_dir = Path(str(row.get("run_dir_actual")))
    required = [
        "DONE",
        "summary.json",
        "history.csv",
        "case_metrics.csv",
        "regularity_metrics.csv",
        "be_metrics.csv",
        "diffusion_metrics.csv",
        "mask_audit.json",
    ]
    errors = [f"Missing {run_dir / name}" for name in required if not (run_dir / name).exists()]
    if not list(run_dir.glob("disp_norm_*.npy")):
        errors.append(f"Missing dense displacement in {run_dir}")
    return errors


def required_baseline_artifact_errors(row: Mapping[str, object]) -> List[str]:
    run_dir = Path(str(row.get("run_dir_actual")))
    required = [
        "DONE",
        "summary.json",
        "history.csv",
        "case_metrics.csv",
        "regularity_metrics.csv",
        "be_metrics.csv",
        "diffusion_metrics.csv",
        "mask_audit.json",
    ]
    return [f"Missing {run_dir / name}" for name in required if not (run_dir / name).exists()]


def metadata_errors(
    row: Mapping[str, object],
    cases: Mapping[int, c.CaseInfo],
    args: argparse.Namespace,
    *,
    baseline: bool,
) -> List[str]:
    errors: List[str] = []
    run_name = str(row.get("run_name"))
    rank = int(as_float(row.get("case_rank")))
    case = cases.get(rank)
    if case is None:
        return [f"{run_name}: unexpected case_rank {rank}"]
    if int(as_float(row.get("original_case_index"))) != int(case.original_case_index):
        errors.append(f"{run_name}: original_case_index {row.get('original_case_index')} != {case.original_case_index}")
    if str(row.get("fixed_id")) != case.fixed_id or str(row.get("moving_id")) != case.moving_id:
        errors.append(f"{run_name}: fixed/moving {row.get('fixed_id')}/{row.get('moving_id')} != {case.fixed_id}/{case.moving_id}")
    if str(row.get("foreground_metric_mask")) != "fixed_image_positive":
        errors.append(f"{run_name}: foreground_metric_mask {row.get('foreground_metric_mask')} != fixed_image_positive")
    if str(row.get("segmentation_mask_role")) != "dice_hd95_asd_only":
        errors.append(f"{run_name}: segmentation_mask_role {row.get('segmentation_mask_role')} != dice_hd95_asd_only")
    if str(row.get("foreground_rule")) != "fixed raw image > 0":
        errors.append(f"{run_name}: foreground_rule {row.get('foreground_rule')} != fixed raw image > 0")
    if baseline:
        return errors
    method = str(row.get("method_key"))
    spec = METHOD_SPECS.get(method, {})
    if int(as_float(row.get("epochs"))) != int(args.expected_epochs):
        errors.append(f"{run_name}: epochs {row.get('epochs')} != {args.expected_epochs}")
    if int(as_float(row.get("eval_interval"))) != int(args.expected_eval_interval):
        errors.append(f"{run_name}: eval_interval {row.get('eval_interval')} != {args.expected_eval_interval}")
    if int(as_float(row.get("seed"))) != int(args.expected_seed):
        errors.append(f"{run_name}: seed {row.get('seed')} != {args.expected_seed}")
    expected_lr = as_float(spec.get("lr"))
    if math.isfinite(expected_lr) and not math.isclose(as_float(row.get("lr")), expected_lr, rel_tol=1e-9, abs_tol=1e-12):
        errors.append(f"{run_name}: lr {row.get('lr')} != {expected_lr:g}")
    if method == "idir_original_dense_siren_be_autograd":
        if str(row.get("sample_domain")) != "fixed_image_positive":
            errors.append(f"{run_name}: IDIR sample_domain {row.get('sample_domain')} != fixed_image_positive")
        if str(row.get("native_analytic_be_formula")) != "idir_coordinate_autograd_hessian":
            errors.append(f"{run_name}: IDIR native BE formula {row.get('native_analytic_be_formula')}")
    else:
        if str(row.get("sample_domain")) != "none_full_image_ncc":
            errors.append(f"{run_name}: spline sample_domain {row.get('sample_domain')} != none_full_image_ncc")
        if str(row.get("native_analytic_be_formula")) != "bspline_analytic":
            errors.append(f"{run_name}: spline native BE formula {row.get('native_analytic_be_formula')}")
        if str(row.get("bspline_mode")) != "disp":
            errors.append(f"{run_name}: bspline_mode {row.get('bspline_mode')} != disp")
    return errors


def validate_sweep_rows(
    rows: Sequence[Dict[str, object]],
    ranks: Sequence[int],
    args: argparse.Namespace,
) -> List[str]:
    errors: List[str] = []
    expected = expected_sweep_combos(ranks)
    observed_counts = Counter(combo_key(row) for row in rows)
    observed = set(observed_counts)
    for method, rank, alpha in sorted(expected - observed, key=lambda item: (method_order(item[0]), item[1], as_float(item[2]))):
        errors.append(f"Missing sweep run: method={method}, case_rank={rank}, alpha={alpha}")
    for method, rank, alpha in sorted(observed - expected, key=lambda item: (method_order(item[0]), item[1], as_float(item[2]))):
        errors.append(f"Unexpected sweep run: method={method}, case_rank={rank}, alpha={alpha}")
    for key, count in sorted(observed_counts.items(), key=lambda item: (method_order(item[0][0]), item[0][1], as_float(item[0][2]))):
        if count != 1:
            errors.append(f"Duplicate sweep run {key}: {count}")

    cases = {
        rank: c.resolve_case(rank=int(rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for rank in ranks
    }
    for row in rows:
        run_name = str(row.get("run_name"))
        method = str(row.get("method_key"))
        for metric in ["dice_mean", "hd95_mean", "asd_symmetric_mean", "log_jdet_std_fg"]:
            if not is_finite(row.get(metric)):
                errors.append(f"{run_name}: missing/non-finite metric {metric}")
        if method not in SWEEP_METHODS:
            errors.append(f"{run_name}: unexpected method {method}")
        errors.extend(metadata_errors(row, cases, args, baseline=False))
        errors.extend(f"{run_name}: {error}" for error in required_sweep_artifact_errors(row))
    return errors


def validate_baseline_rows(
    rows: Sequence[Dict[str, object]],
    ranks: Sequence[int],
    args: argparse.Namespace,
) -> List[str]:
    errors: List[str] = []
    expected = expected_baseline_combos(ranks)
    observed_counts = Counter(baseline_key(row) for row in rows)
    observed = set(observed_counts)
    for method, rank in sorted(expected - observed, key=lambda item: (method_order(item[0]), item[1])):
        errors.append(f"Missing baseline row: method={method}, case_rank={rank}")
    for method, rank in sorted(observed - expected, key=lambda item: (method_order(item[0]), item[1])):
        errors.append(f"Unexpected baseline row: method={method}, case_rank={rank}")
    for key, count in sorted(observed_counts.items(), key=lambda item: (method_order(item[0][0]), item[0][1])):
        if count != 1:
            errors.append(f"Duplicate baseline row {key}: {count}")

    cases = {
        rank: c.resolve_case(rank=int(rank), pair_file=Path(args.pair_file), oasis_root=Path(args.oasis_root))
        for rank in ranks
    }
    for row in rows:
        run_name = str(row.get("run_name"))
        method = str(row.get("method_key"))
        for metric in ["dice_mean", "hd95_mean", "asd_symmetric_mean", "log_jdet_std_fg"]:
            if not is_finite(row.get(metric)):
                errors.append(f"{run_name}: missing/non-finite metric {metric}")
        if method not in BASELINE_METHODS:
            errors.append(f"{run_name}: unexpected baseline method {method}")
        errors.extend(metadata_errors(row, cases, args, baseline=True))
        errors.extend(f"{run_name}: {error}" for error in required_baseline_artifact_errors(row))
    return errors


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
        if key not in out and any(is_finite(row.get(key)) for row in rows):
            out.append(key)
    return out


def stats(values: Iterable[object]) -> Dict[str, float | int]:
    arr = np.asarray([as_float(value) for value in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def population_group_key(row: Mapping[str, object]) -> Tuple[str, str, str]:
    method = str(row.get("method_key"))
    if method in BASELINE_METHODS:
        return method, "baseline", "baseline"
    return method, alpha_id(row.get("alpha")), alpha_id(row.get("lr"))


def population_sort_key(row: Mapping[str, object]) -> Tuple[int, float]:
    method = str(row.get("method_key"))
    alpha = as_float(row.get("alpha"))
    return method_order(method), alpha if math.isfinite(alpha) else math.inf


def population_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[population_group_key(row)].append(row)
    keys = metric_keys(rows)
    out: List[Dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: (method_order(item[0][0]), as_float(item[0][1]))):
        method_key, alpha_text, lr_text = key
        case_ranks = sorted(int(as_float(row.get("case_rank"))) for row in group)
        alpha = "" if method_key in BASELINE_METHODS else as_float(alpha_text)
        lr = "" if method_key in BASELINE_METHODS else as_float(lr_text)
        row_out: Dict[str, object] = {
            "method_key": method_key,
            "method_label": METHOD_SPECS[method_key]["label"],
            "alpha": alpha,
            "lr": lr,
            "n_cases": len(group),
            "case_ranks_text": ",".join(str(rank) for rank in case_ranks),
        }
        for metric in keys:
            metric_stats = stats(item.get(metric) for item in group)
            for stat_name, value in metric_stats.items():
                row_out[f"{metric}_{stat_name}"] = value
        out.append(row_out)
    return sorted(out, key=population_sort_key)


def population_label_metrics(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[population_group_key(row)].append(row)
    out: List[Dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: (method_order(item[0][0]), as_float(item[0][1]))):
        method_key, alpha_text, lr_text = key
        alpha = "" if method_key in BASELINE_METHODS else as_float(alpha_text)
        lr = "" if method_key in BASELINE_METHODS else as_float(lr_text)
        for label in c.OASIS_LABELS:
            row_out: Dict[str, object] = {
                "method_key": method_key,
                "method_label": METHOD_SPECS[method_key]["label"],
                "alpha": alpha,
                "lr": lr,
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
                metric_stats = stats(item.get(metric) for item in group)
                metric_name = metric.rsplit("_label_", 1)[0]
                for stat_name, value in metric_stats.items():
                    row_out[f"{metric_name}_{stat_name}"] = value
            out.append(row_out)
    return out


def validate_population_rows(rows: Sequence[Dict[str, object]], cohort: str, n_cases: int) -> List[str]:
    errors: List[str] = []
    expected_counts = {method: 6 for method in SWEEP_METHODS}
    expected_counts.update({method: 1 for method in BASELINE_METHODS})
    observed = Counter(str(row.get("method_key")) for row in rows)
    if len(rows) != 35:
        errors.append(f"{cohort}: expected 35 population rows, found {len(rows)}")
    for method, expected in expected_counts.items():
        if observed.get(method, 0) != expected:
            errors.append(f"{cohort}: method {method} expected {expected} rows, found {observed.get(method, 0)}")
    for row in rows:
        method = str(row.get("method_key"))
        if int(as_float(row.get("n_cases"))) != int(n_cases):
            errors.append(f"{cohort}: {method} n_cases {row.get('n_cases')} != {n_cases}")
    for acc_key, _acc_label, _higher in ACCURACY_METRICS:
        for reg_key, _reg_label in REGULARITY_METRICS:
            for key in [metric_mean(acc_key), metric_std(acc_key), metric_mean(reg_key), metric_std(reg_key)]:
                bad = [str(row.get("method_key")) for row in rows if not is_finite(row.get(key))]
                if bad:
                    errors.append(f"{cohort}: column {key} has non-finite values for {', '.join(bad[:8])}")
    return errors


def add_ellipse(ax: plt.Axes, row: Mapping[str, object], x: float, y: float, reg_key: str, acc_key: str, color: str) -> None:
    x_std = max(as_float(row.get(metric_std(reg_key))), 0.0)
    y_std = max(as_float(row.get(metric_std(acc_key))), 0.0)
    if not math.isfinite(x_std) or not math.isfinite(y_std) or (x_std == 0.0 and y_std == 0.0):
        return
    ax.add_patch(
        Ellipse(
            (x, y),
            width=2.0 * x_std,
            height=2.0 * y_std,
            facecolor=color,
            edgecolor=color,
            linewidth=0.9,
            alpha=0.14,
            zorder=1,
        )
    )


def collect_extents(
    rows: Sequence[Mapping[str, object]],
    reg_key: str,
    acc_key: str,
    *,
    include_std: bool,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        x = as_float(row.get(metric_mean(reg_key)))
        y = as_float(row.get(metric_mean(acc_key)))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        xs.append(x)
        ys.append(y)
        if include_std:
            x_std = max(as_float(row.get(metric_std(reg_key))), 0.0)
            y_std = max(as_float(row.get(metric_std(acc_key))), 0.0)
            if math.isfinite(x_std):
                xs.extend([x - x_std, x + x_std])
            if math.isfinite(y_std):
                ys.extend([y - y_std, y + y_std])
    return xs, ys


def set_limits(ax: plt.Axes, rows: Sequence[Mapping[str, object]], reg_key: str, acc_key: str, *, include_std: bool) -> None:
    xs, ys = collect_extents(rows, reg_key, acc_key, include_std=include_std)
    if not xs or not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = 0.06 * max(x_max - x_min, 1e-8)
    y_pad = 0.06 * max(y_max - y_min, 1e-8)
    ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)


def draw_panel(
    ax: plt.Axes,
    rows: Sequence[Dict[str, object]],
    *,
    acc_key: str,
    reg_key: str,
    style: str,
    show_legend: bool,
    annotate: bool,
) -> None:
    acc_mean = metric_mean(acc_key)
    reg_mean = metric_mean(reg_key)
    include_std = style in {"line_segments_errorbars", "ellipses"}

    for method in SWEEP_METHODS:
        group = [
            row
            for row in rows
            if str(row.get("method_key")) == method
            and is_finite(row.get(acc_mean))
            and is_finite(row.get(reg_mean))
        ]
        group.sort(key=lambda row: as_float(row.get("alpha")))
        if not group:
            continue
        color = str(METHOD_SPECS[method]["color"])
        label = str(METHOD_SPECS[method]["label"])
        xs = [as_float(row.get(reg_mean)) for row in group]
        ys = [as_float(row.get(acc_mean)) for row in group]
        if style == "ellipses":
            for row, x, y in zip(group, xs, ys):
                add_ellipse(ax, row, x, y, reg_key, acc_key, color)
        if style == "line_segments_errorbars":
            ax.errorbar(
                xs,
                ys,
                xerr=[max(as_float(row.get(metric_std(reg_key))), 0.0) for row in group],
                yerr=[max(as_float(row.get(metric_std(acc_key))), 0.0) for row in group],
                marker="o",
                linestyle="-",
                linewidth=1.55,
                markersize=4.6,
                capsize=2.0,
                color=color,
                label=label,
                zorder=3,
            )
        else:
            ax.plot(xs, ys, marker="o", linestyle="-", linewidth=1.55, markersize=4.8, color=color, label=label, zorder=3)
        if annotate:
            for row, x, y in zip(group, xs, ys):
                ax.annotate(format_alpha(row.get("alpha")), (x, y), fontsize=5.6, xytext=(2, 2), textcoords="offset points")

    for method in BASELINE_METHODS:
        group = [
            row
            for row in rows
            if str(row.get("method_key")) == method
            and is_finite(row.get(acc_mean))
            and is_finite(row.get(reg_mean))
        ]
        if not group:
            continue
        row = group[0]
        color = str(METHOD_SPECS[method]["color"])
        label = str(METHOD_SPECS[method]["label"])
        x = as_float(row.get(reg_mean))
        y = as_float(row.get(acc_mean))
        marker_zorder = 12
        label_zorder = 13
        if style == "ellipses":
            add_ellipse(ax, row, x, y, reg_key, acc_key, color)
        if style == "line_segments_errorbars":
            ax.errorbar(
                [x],
                [y],
                xerr=[max(as_float(row.get(metric_std(reg_key))), 0.0)],
                yerr=[max(as_float(row.get(metric_std(acc_key))), 0.0)],
                marker="D",
                linestyle="none",
                markersize=5.2,
                capsize=2.0,
                color=color,
                markeredgecolor="#202020",
                markeredgewidth=0.45,
                label=label,
                zorder=marker_zorder,
            )
        else:
            ax.scatter(
                [x],
                [y],
                marker="D",
                s=42,
                color=color,
                edgecolors="#202020",
                linewidths=0.45,
                label=label,
                zorder=marker_zorder,
            )
        if annotate:
            ax.annotate(
                str(METHOD_SPECS[method]["dot_label"]),
                (x, y),
                fontsize=5.8,
                xytext=(4, 3),
                textcoords="offset points",
                zorder=label_zorder,
            )

    set_limits(ax, rows, reg_key, acc_key, include_std=include_std)
    if should_symlog(reg_key):
        ax.set_xscale("symlog", linthresh=1e-8)
    ax.set_xlabel(label_for_metric(reg_key, REGULARITY_METRICS))
    ax.set_ylabel(accuracy_label(acc_key))
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(fontsize=6.4, ncol=2)


def save_figure(fig: plt.Figure, path_base: Path, dpi: int) -> List[Path]:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [path_base.with_suffix(".pdf"), path_base.with_suffix(".png")]
    fig.savefig(outputs[0], dpi=dpi)
    fig.savefig(outputs[1], dpi=dpi)
    plt.close(fig)
    return outputs


def plot_small_panel(
    rows: Sequence[Dict[str, object]],
    out_dir: Path,
    *,
    cohort: str,
    style: str,
    acc_key: str,
    reg_key: str,
    dpi: int,
) -> List[Path]:
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.4))
    draw_panel(ax, rows, acc_key=acc_key, reg_key=reg_key, style=style, show_legend=True, annotate=True)
    ax.set_title(f"{cohort}: {accuracy_label(acc_key)} vs {label_for_metric(reg_key, REGULARITY_METRICS)}", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, out_dir / f"{style}__{acc_key}_vs_{reg_key}", dpi)


def plot_contact_sheet(
    rows: Sequence[Dict[str, object]],
    out_dir: Path,
    *,
    cohort: str,
    style: str,
    dpi: int,
) -> List[Path]:
    n_rows = len(ACCURACY_METRICS)
    n_cols = len(REGULARITY_METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.55 * n_rows), squeeze=False)
    for row_idx, (acc_key, acc_label, _higher) in enumerate(ACCURACY_METRICS):
        for col_idx, (reg_key, reg_label) in enumerate(REGULARITY_METRICS):
            ax = axes[row_idx][col_idx]
            draw_panel(
                ax,
                rows,
                acc_key=acc_key,
                reg_key=reg_key,
                style=style,
                show_legend=(row_idx == 0 and col_idx == 0),
                annotate=True,
            )
            ax.set_title(f"{acc_label} vs {reg_label}", fontsize=8.5)
    fig.suptitle(f"OASIS seed-3 {cohort}: five alpha sweeps with DL/Greedy references ({style})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return save_figure(fig, out_dir / f"contact_sheet_{style}", dpi)


def plot_spotlight(
    rows: Sequence[Dict[str, object]],
    out_dir: Path,
    *,
    cohort: str,
    style: str,
    acc_key: str,
    reg_key: str,
    dpi: int,
) -> List[Path]:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
    draw_panel(ax, rows, acc_key=acc_key, reg_key=reg_key, style=style, show_legend=True, annotate=True)
    ax.set_title(f"OASIS seed-3 {cohort}: {accuracy_label(acc_key)} vs {label_for_metric(reg_key, REGULARITY_METRICS)}", fontsize=11)
    fig.tight_layout()
    return save_figure(fig, out_dir / f"spotlight_{style}__{acc_key}_vs_{reg_key}", dpi)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    c.write_csv(path, rows)


def write_json(path: Path, payload: object) -> None:
    c.write_json(path, payload)


def best_lines(pop_rows: Sequence[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for method in SWEEP_METHODS:
        group = [row for row in pop_rows if str(row.get("method_key")) == method]
        if not group:
            continue
        best_dice = max(group, key=lambda row: as_float(row.get("dice_mean_mean")))
        best_hd95 = min(group, key=lambda row: as_float(row.get("hd95_mean_mean")))
        lines.append(
            f"- {METHOD_SPECS[method]['label']}: best Dice alpha `{format_alpha(best_dice.get('alpha'))}`, "
            f"best HD95 alpha `{format_alpha(best_hd95.get('alpha'))}`."
        )
    for method in BASELINE_METHODS:
        group = [row for row in pop_rows if str(row.get("method_key")) == method]
        if not group:
            continue
        row = group[0]
        lines.append(
            f"- {METHOD_SPECS[method]['label']}: Dice `{as_float(row.get('dice_mean_mean')):.4f}`, "
            f"HD95 `{as_float(row.get('hd95_mean_mean')):.4f}`, "
            f"symmetric ASD `{as_float(row.get('asd_symmetric_mean_mean')):.4f}`."
        )
    return lines


def write_cohort_report(
    cohort_dir: Path,
    *,
    cohort: str,
    sweep_rows: Sequence[Dict[str, object]],
    baseline_rows: Sequence[Dict[str, object]],
    pop_rows: Sequence[Dict[str, object]],
    output_paths: Sequence[Path],
) -> None:
    lines = [
        f"# OASIS Seed-3 {cohort} Five-Sweep Pareto Plot Package",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Sweep rows: `{len(sweep_rows)}`",
        f"- Baseline rows: `{len(baseline_rows)}`",
        f"- Population rows: `{len(pop_rows)}`",
        "- Styles: line segments, line segments with x/y 1-SD bars, and axis-aligned 1-SD ellipses.",
        "- DirectCP cps2 uses a solid line, matching the other sweep methods.",
        "- Foreground log-J metric: `log_jdet_std_fg`, the foreground standard deviation of log Jacobian determinant.",
        "",
        "## Best/Reference Points",
        "",
    ]
    lines.extend(best_lines(pop_rows))
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `population_summary_{cohort}_with_saved_baselines.csv/json`",
            f"- `population_label_metrics_{cohort}_with_saved_baselines.csv`",
            "- `small_plots/`: individual accuracy-regularity panels.",
            "- `big_plots/`: high-DPI contact sheets and spotlight plots.",
            f"- Generated figure files: `{len(output_paths)}`",
            "",
        ]
    )
    (cohort_dir / "review_report.md").write_text("\n".join(lines) + "\n")


def generate_plots_for_cohort(
    cohort_dir: Path,
    *,
    cohort: str,
    pop_rows: Sequence[Dict[str, object]],
    dpi: int,
) -> List[Path]:
    small_dir = cohort_dir / "small_plots"
    big_dir = cohort_dir / "big_plots"
    output_paths: List[Path] = []
    for style in STYLES:
        for acc_key, _acc_label, _higher in ACCURACY_METRICS:
            for reg_key, _reg_label in REGULARITY_METRICS:
                output_paths.extend(
                    plot_small_panel(
                        pop_rows,
                        small_dir,
                        cohort=cohort,
                        style=style,
                        acc_key=acc_key,
                        reg_key=reg_key,
                        dpi=dpi,
                    )
                )
        output_paths.extend(plot_contact_sheet(pop_rows, big_dir, cohort=cohort, style=style, dpi=dpi))
        for acc_key, reg_key in SPOTLIGHT_PANELS:
            output_paths.extend(
                plot_spotlight(
                    pop_rows,
                    big_dir,
                    cohort=cohort,
                    style=style,
                    acc_key=acc_key,
                    reg_key=reg_key,
                    dpi=dpi,
                )
            )
    return output_paths


def assert_outputs(paths: Sequence[Path]) -> List[str]:
    return [str(path) for path in paths if not path.exists() or path.stat().st_size <= 0]


def build_input_rows(args: argparse.Namespace) -> Dict[str, List[Dict[str, object]]]:
    first10_main = collect_summary_rows(Path(args.first10_main_root))
    first10_cps2_prelim = collect_summary_rows(Path(args.first10_cps2_prelim_root))
    first10_cps2_completion = collect_summary_rows(Path(args.first10_cps2_completion_root))
    next10 = collect_summary_rows(Path(args.next10_root))
    baselines = collect_summary_rows(Path(args.baseline_root))

    first10_ranks = list(range(10))
    next10_ranks = list(range(10, 20))
    first20_ranks = list(range(20))
    first10_sweeps = (
        filter_ranks(first10_main, first10_ranks)
        + filter_ranks(first10_cps2_prelim, first10_ranks)
        + filter_ranks(first10_cps2_completion, first10_ranks)
    )
    next10_sweeps = filter_ranks(next10, next10_ranks)
    first20_sweeps = first10_sweeps + next10_sweeps
    return {
        "first10_sweeps": first10_sweeps,
        "next10_sweeps": next10_sweeps,
        "first20_sweeps": first20_sweeps,
        "first10_baselines": filter_ranks(baselines, first10_ranks),
        "first20_baselines": filter_ranks(baselines, first20_ranks),
    }


def process_cohort(
    args: argparse.Namespace,
    *,
    cohort: str,
    ranks: Sequence[int],
    sweep_rows: Sequence[Dict[str, object]],
    baseline_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    cohort_dir = Path(args.review_root) / cohort
    errors = validate_sweep_rows(sweep_rows, ranks, args)
    errors.extend(validate_baseline_rows(baseline_rows, ranks, args))
    if errors and not args.allow_incomplete:
        write_json(cohort_dir / "validation_errors.json", errors)
        raise RuntimeError(f"{cohort} validation failed:\n" + "\n".join(errors[:40]))

    pop_rows = population_summary(list(sweep_rows) + list(baseline_rows))
    label_rows = population_label_metrics(list(sweep_rows) + list(baseline_rows))
    pop_errors = validate_population_rows(pop_rows, cohort, len(ranks))
    if pop_errors and not args.allow_incomplete:
        write_json(cohort_dir / "validation_errors.json", errors + pop_errors)
        raise RuntimeError(f"{cohort} population validation failed:\n" + "\n".join(pop_errors[:40]))

    cohort_dir.mkdir(parents=True, exist_ok=True)
    write_csv(cohort_dir / f"population_summary_{cohort}_with_saved_baselines.csv", pop_rows)
    write_json(cohort_dir / f"population_summary_{cohort}_with_saved_baselines.json", pop_rows)
    write_csv(cohort_dir / f"population_label_metrics_{cohort}_with_saved_baselines.csv", label_rows)
    output_paths = generate_plots_for_cohort(cohort_dir, cohort=cohort, pop_rows=pop_rows, dpi=int(args.dpi))
    missing_outputs = assert_outputs(output_paths)
    if missing_outputs:
        raise RuntimeError(f"{cohort} missing/empty outputs:\n" + "\n".join(missing_outputs[:40]))

    manifest = {
        "generated_at": c.now_text(),
        "cohort": cohort,
        "case_ranks": list(ranks),
        "sweep_rows": len(sweep_rows),
        "baseline_rows": len(baseline_rows),
        "population_rows": len(pop_rows),
        "styles": STYLES,
        "accuracy_metrics": ACCURACY_METRICS,
        "regularity_metrics": REGULARITY_METRICS,
        "spotlight_panels": SPOTLIGHT_PANELS,
        "method_specs": METHOD_SPECS,
        "output_files": [str(path) for path in output_paths],
        "validation_errors": errors + pop_errors,
    }
    write_json(cohort_dir / "plot_manifest.json", manifest)
    write_cohort_report(
        cohort_dir,
        cohort=cohort,
        sweep_rows=sweep_rows,
        baseline_rows=baseline_rows,
        pop_rows=pop_rows,
        output_paths=output_paths,
    )
    return {
        "cohort": cohort,
        "cohort_dir": str(cohort_dir),
        "sweep_rows": len(sweep_rows),
        "baseline_rows": len(baseline_rows),
        "population_rows": len(pop_rows),
        "label_rows": len(label_rows),
        "output_files": len(output_paths),
        "validation_errors": errors + pop_errors,
    }


def write_root_report(review_root: Path, results: Sequence[Mapping[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# OASIS First10/First20 Five-Sweep Pareto Plot Package",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Review root: `{review_root}`",
        f"- First10 main root: `{args.first10_main_root}`",
        f"- First10 DirectCP cps2 prelim root: `{args.first10_cps2_prelim_root}`",
        f"- First10 DirectCP cps2 completion root: `{args.first10_cps2_completion_root}`",
        f"- Next10 five-model root: `{args.next10_root}`",
        f"- Saved-baseline root: `{args.baseline_root}`",
        "- DirectCP cps2 is solid-line light magenta, not dashed.",
        "",
        "## Cohorts",
        "",
    ]
    for result in results:
        lines.append(
            f"- {result['cohort']}: sweep rows `{result['sweep_rows']}`, baseline rows `{result['baseline_rows']}`, "
            f"population rows `{result['population_rows']}`, figure files `{result['output_files']}`."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `first10/` and `first20/`: cohort-specific summaries, manifests, reports, small plots, and big plots.",
            "- Existing first10, next10, and baseline aggregate folders were not overwritten.",
            "",
        ]
    )
    (review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    inputs = build_input_rows(args)
    review_root = Path(args.review_root)
    review_root.mkdir(parents=True, exist_ok=True)

    first10_result = process_cohort(
        args,
        cohort="first10",
        ranks=list(range(10)),
        sweep_rows=inputs["first10_sweeps"],
        baseline_rows=inputs["first10_baselines"],
    )
    first20_result = process_cohort(
        args,
        cohort="first20",
        ranks=list(range(20)),
        sweep_rows=inputs["first20_sweeps"],
        baseline_rows=inputs["first20_baselines"],
    )
    results = [first10_result, first20_result]
    root_manifest = {
        "generated_at": c.now_text(),
        "review_root": str(review_root),
        "results": results,
        "input_roots": {
            "first10_main_root": str(args.first10_main_root),
            "first10_cps2_prelim_root": str(args.first10_cps2_prelim_root),
            "first10_cps2_completion_root": str(args.first10_cps2_completion_root),
            "next10_root": str(args.next10_root),
            "baseline_root": str(args.baseline_root),
        },
    }
    write_json(review_root / "plot_manifest.json", root_manifest)
    write_root_report(review_root, results, args)
    print(json.dumps(root_manifest, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
