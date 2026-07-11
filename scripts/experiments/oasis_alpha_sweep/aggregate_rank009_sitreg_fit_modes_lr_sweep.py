#!/usr/bin/env python
"""Aggregate the OASIS rank-9 SITReg-field fit LR sweep."""

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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import aggregate_rank009_fit_modes_lr_sweep as greedy_ag  # noqa: E402
import common as c  # noqa: E402


VARIANT = "oasis_rank009_sitreg_fit_modes_lr_sweep_v1"
IMAGE_REG_VARIANT = "oasis_rank009_fit_modes_lr_sweep_v1"
DEFAULT_FIELD_ROOT = Path("/database/inrorspline/runs/oasis_rank009_sitreg_fit_modes_lr_sweep_v1")
DEFAULT_IMAGE_ROOT = Path("/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1")
DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-06-26_oasis_rank009_sitreg_fit_modes_lr_sweep"
CASE_RANK = 9
EXPECTED_ORIGINAL_INDEX = 88
EXPECTED_FIXED_ID = "0173"
EXPECTED_MOVING_ID = "0136"
LRS = [1e-2, 1e-3, 1e-4]
IMAGE_METHODS = greedy_ag.IMAGE_METHODS
FIELD_METHODS = greedy_ag.FIELD_METHODS
LR_STYLES = greedy_ag.LR_STYLES
FIELD_OBJECTIVE = "sitreg_field_fit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--field-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None, help="Alias for --field-root.")
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--expected-field-epochs", type=int, default=None)
    parser.add_argument("--expected-field-eval-interval", type=int, default=None)
    parser.add_argument("--expected-image-epochs", type=int, default=2500)
    parser.add_argument("--expected-image-eval-interval", type=int, default=250)
    parser.add_argument("--expected-epochs", type=int, default=None, help="Alias for --expected-field-epochs.")
    parser.add_argument("--expected-eval-interval", type=int, default=None, help="Alias for --expected-field-eval-interval.")
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.field_root is None:
        args.field_root = args.out_root if args.out_root is not None else DEFAULT_FIELD_ROOT
    if args.expected_field_epochs is None:
        args.expected_field_epochs = args.expected_epochs if args.expected_epochs is not None else 2500
    if args.expected_field_eval_interval is None:
        args.expected_field_eval_interval = (
            args.expected_eval_interval if args.expected_eval_interval is not None else 250
        )
    return args


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(value: object) -> bool:
    return math.isfinite(as_float(value))


def alpha_id(value: object) -> str:
    return f"{as_float(value):.12g}"


def lr_id(value: object) -> str:
    return f"{as_float(value):.12g}"


def objective_for_row(row: Mapping[str, object]) -> str:
    objective_group = str(row.get("objective_group", "")).strip()
    if objective_group in {"image_registration", FIELD_OBJECTIVE, "greedy_field_fit"}:
        return objective_group
    objective = str(row.get("objective", "")).strip()
    method_key = str(row.get("method_key"))
    if objective == "sitreg_field_fit_mse":
        return FIELD_OBJECTIVE
    if method_key in FIELD_METHODS and str(row.get("source_method_key")) == "bs4_sitreg_v0":
        return FIELD_OBJECTIVE
    if objective == "greedy_field_fit_mse":
        return "greedy_field_fit"
    if method_key in FIELD_METHODS:
        return "greedy_field_fit"
    return "image_registration"


def method_label(row: Mapping[str, object]) -> str:
    objective = objective_for_row(row)
    method_key = str(row.get("method_key"))
    spec = (FIELD_METHODS if objective == FIELD_OBJECTIVE else IMAGE_METHODS).get(method_key, {})
    return str(spec.get("label", row.get("method_label", method_key)))


def method_color(row: Mapping[str, object]) -> str:
    objective = objective_for_row(row)
    method_key = str(row.get("method_key"))
    spec = (FIELD_METHODS if objective == FIELD_OBJECTIVE else IMAGE_METHODS).get(method_key, {})
    return str(spec.get("color", "#444444"))


def method_sort_index(objective: str, method_key: str) -> int:
    methods = list(FIELD_METHODS if objective == FIELD_OBJECTIVE else IMAGE_METHODS)
    return methods.index(method_key) if method_key in methods else len(methods)


def collect_root_summaries(root: Path, keep_objective: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(root.glob("*/*/summary.json")):
        row = json.loads(path.read_text())
        row["summary_path"] = str(path)
        row["run_dir_actual"] = str(path.parent)
        row["summary_source_root"] = str(root)
        row["objective_group"] = objective_for_row(row)
        if str(row["objective_group"]) != keep_objective:
            continue
        row["method_label"] = method_label(row)
        rows.append(row)
    return rows


def collect_combined_summaries(image_root: Path, field_root: Path) -> List[Dict[str, object]]:
    rows = collect_root_summaries(image_root, "image_registration")
    rows.extend(collect_root_summaries(field_root, FIELD_OBJECTIVE))
    rows.sort(
        key=lambda row: (
            str(row.get("objective_group")),
            method_sort_index(str(row.get("objective_group")), str(row.get("method_key"))),
            as_float(row.get("lr")),
        )
    )
    return rows


def expected_combos() -> Dict[Tuple[str, str, str, str], Dict[str, object]]:
    expected: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for objective, specs in [("image_registration", IMAGE_METHODS), (FIELD_OBJECTIVE, FIELD_METHODS)]:
        for method_key, spec in specs.items():
            for lr in LRS:
                key = (objective, method_key, alpha_id(spec["alpha"]), lr_id(lr))
                expected[key] = {
                    "objective": objective,
                    "method_key": method_key,
                    "alpha": float(spec["alpha"]),
                    "lr": float(lr),
                }
    return expected


def combo_key(row: Mapping[str, object]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("objective_group")),
        str(row.get("method_key")),
        alpha_id(row.get("alpha")),
        lr_id(row.get("lr")),
    )


def required_artifacts(row: Mapping[str, object]) -> List[str]:
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
    objective = str(row.get("objective_group"))
    method_key = str(row.get("method_key"))
    if objective == FIELD_OBJECTIVE:
        required.append("target_audit.json")
    if "cps4" in method_key:
        required.append("control_output.pt")
    return [f"Missing {run_dir / name}" for name in required if not (run_dir / name).exists()]


def expected_epochs_for(row: Mapping[str, object], args: argparse.Namespace) -> int:
    if str(row.get("objective_group")) == "image_registration":
        return int(args.expected_image_epochs)
    return int(args.expected_field_epochs)


def expected_eval_interval_for(row: Mapping[str, object], args: argparse.Namespace) -> int:
    if str(row.get("objective_group")) == "image_registration":
        return int(args.expected_image_eval_interval)
    return int(args.expected_field_eval_interval)


def check_expected(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    expected = expected_combos()
    observed: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        observed[combo_key(row)].append(row)

    for key in sorted(set(expected) - set(observed)):
        errors.append(f"Missing run combo: objective={key[0]}, method={key[1]}, alpha={key[2]}, lr={key[3]}")
    for key in sorted(set(observed) - set(expected)):
        errors.append(f"Unexpected run combo: objective={key[0]}, method={key[1]}, alpha={key[2]}, lr={key[3]}")
    for key, group in sorted(observed.items()):
        if len(group) > 1:
            errors.append(f"Duplicate run combo {key}: {len(group)} summaries")

    for row in rows:
        run_name = str(row.get("run_name"))
        objective = str(row.get("objective_group"))
        expected_variant = IMAGE_REG_VARIANT if objective == "image_registration" else VARIANT
        if str(row.get("variant")) != expected_variant:
            errors.append(f"{run_name}: variant {row.get('variant')} != {expected_variant}")
        if int(as_float(row.get("case_rank"))) != CASE_RANK:
            errors.append(f"{run_name}: case_rank {row.get('case_rank')} != {CASE_RANK}")
        if int(as_float(row.get("original_case_index"))) != EXPECTED_ORIGINAL_INDEX:
            errors.append(f"{run_name}: original_case_index {row.get('original_case_index')} != {EXPECTED_ORIGINAL_INDEX}")
        if str(row.get("fixed_id")) != EXPECTED_FIXED_ID or str(row.get("moving_id")) != EXPECTED_MOVING_ID:
            errors.append(f"{run_name}: fixed/moving {row.get('fixed_id')}/{row.get('moving_id')} != {EXPECTED_FIXED_ID}/{EXPECTED_MOVING_ID}")
        if int(as_float(row.get("epochs"))) != expected_epochs_for(row, args):
            errors.append(f"{run_name}: epochs {row.get('epochs')} != {expected_epochs_for(row, args)}")
        if int(as_float(row.get("eval_interval"))) != expected_eval_interval_for(row, args):
            errors.append(f"{run_name}: eval_interval {row.get('eval_interval')} != {expected_eval_interval_for(row, args)}")
        if int(as_float(row.get("seed"))) != int(args.expected_seed):
            errors.append(f"{run_name}: seed {row.get('seed')} != {args.expected_seed}")
        if str(row.get("foreground_metric_mask")) != "fixed_image_positive":
            errors.append(f"{run_name}: foreground_metric_mask {row.get('foreground_metric_mask')} != fixed_image_positive")
        if str(row.get("segmentation_mask_role")) != "dice_hd95_asd_only":
            errors.append(f"{run_name}: segmentation_mask_role {row.get('segmentation_mask_role')} != dice_hd95_asd_only")
        if str(row.get("foreground_rule")) != "fixed raw image > 0":
            errors.append(f"{run_name}: foreground_rule {row.get('foreground_rule')} != fixed raw image > 0")

        if objective == "image_registration":
            if str(row.get("method_key")) == "idir_original_dense_siren_be_autograd":
                if str(row.get("sample_domain")) != "fixed_image_positive":
                    errors.append(f"{run_name}: IDIR sample_domain {row.get('sample_domain')} != fixed_image_positive")
            elif str(row.get("sample_domain")) != "none_full_image_ncc":
                errors.append(f"{run_name}: spline sample_domain {row.get('sample_domain')} != none_full_image_ncc")
        elif objective == FIELD_OBJECTIVE:
            if str(row.get("objective")) != "sitreg_field_fit_mse":
                errors.append(f"{run_name}: objective {row.get('objective')} != sitreg_field_fit_mse")
            if str(row.get("regularizer")) != "none_mse_only":
                errors.append(f"{run_name}: field-fit regularizer {row.get('regularizer')} != none_mse_only")
            if str(row.get("field_fit_domain")) != "fixed_image_positive":
                errors.append(f"{run_name}: field_fit_domain {row.get('field_fit_domain')} != fixed_image_positive")
            if str(row.get("field_fit_mask_definition")) != "fixed raw image > 0 after standard OASIS reorientation":
                errors.append(f"{run_name}: field_fit_mask_definition mismatch: {row.get('field_fit_mask_definition')}")
            if str(row.get("field_fit_target_units")) != "normalized_grid":
                errors.append(f"{run_name}: field target units {row.get('field_fit_target_units')} != normalized_grid")
            if str(row.get("field_fit_target_shape")) != "160,224,192,3":
                errors.append(f"{run_name}: field target shape {row.get('field_fit_target_shape')} != 160,224,192,3")
            if str(row.get("source_kind")) != "dl_pretrained":
                errors.append(f"{run_name}: source_kind {row.get('source_kind')} != dl_pretrained")
            if str(row.get("source_family")) != "bs":
                errors.append(f"{run_name}: source_family {row.get('source_family')} != bs")
            if str(row.get("source_method_key")) != "bs4_sitreg_v0":
                errors.append(f"{run_name}: source_method_key {row.get('source_method_key')} != bs4_sitreg_v0")
            if str(row.get("source_selector")) != "bestdice":
                errors.append(f"{run_name}: source_selector {row.get('source_selector')} != bestdice")
            if str(row.get("source_disp_raw_shape")) != "3,160,224,192":
                errors.append(f"{run_name}: source raw shape {row.get('source_disp_raw_shape')} != 3,160,224,192")
            if str(row.get("source_disp_shape")) != "1,3,160,224,192":
                errors.append(f"{run_name}: source eval shape {row.get('source_disp_shape')} != 1,3,160,224,192")
            if str(row.get("source_disp_storage_layout")) != "channel_first_zyx_repo_ras":
                errors.append(f"{run_name}: SITReg storage layout mismatch: {row.get('source_disp_storage_layout')}")
            if str(row.get("source_disp_reorientation")) != "none":
                errors.append(f"{run_name}: SITReg reorientation mismatch: {row.get('source_disp_reorientation')}")
            for metric in ["field_mse_full_normalized_grid", "field_mse_full_voxel", "field_mse_objective_normalized_grid", "field_mse_objective_voxel"]:
                if not is_finite(row.get(metric)):
                    errors.append(f"{run_name}: missing/non-finite field metric {metric}")

        for metric in ["dense_image_loss", "dice_mean", "hd95_mean", "asd_symmetric_mean", "jdet_le0_pct_fg", "log_jdet_std_fg"]:
            if not is_finite(row.get(metric)):
                errors.append(f"{run_name}: missing/non-finite required metric {metric}")
        errors.extend(f"{run_name}: {error}" for error in required_artifacts(row))
    return errors


def read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv_stream(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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


def context_from_summary(row: Mapping[str, object]) -> Dict[str, object]:
    return {
        "objective_group": row.get("objective_group"),
        "run_name": row.get("run_name"),
        "method_key": row.get("method_key"),
        "method_label": row.get("method_label"),
        "case_rank": row.get("case_rank"),
        "original_case_index": row.get("original_case_index"),
        "fixed_id": row.get("fixed_id"),
        "moving_id": row.get("moving_id"),
        "alpha": row.get("alpha"),
        "lr": row.get("lr"),
        "coords_batch_size": row.get("coords_batch_size"),
        "field_chunk_size": row.get("field_chunk_size"),
        "summary_path": row.get("summary_path"),
    }


def collect_metric_file(rows: Sequence[Dict[str, object]], filename: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        path = Path(str(row["run_dir_actual"])) / filename
        if not path.exists():
            continue
        context = context_from_summary(row)
        for metric_row in read_csv(path):
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
        for hist in read_csv(path):
            merged = dict(context)
            merged.update(hist)
            merged["history_path"] = str(path)
            out.append(merged)
    return out


def augment_history(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    augmented: List[Dict[str, object]] = []
    first_values: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("objective_group")),
            str(row.get("method_key")),
            lr_id(row.get("lr")),
            as_float(row.get("epoch")),
        ),
    )
    for row in sorted_rows:
        out = dict(row)
        alpha = as_float(out.get("alpha"))
        reg_loss = as_float(out.get("reg_loss"))
        data_loss = as_float(out.get("data_loss"))
        loss = as_float(out.get("loss"))
        weighted_reg = alpha * reg_loss if math.isfinite(alpha) and math.isfinite(reg_loss) else float("nan")
        out["weighted_reg_loss"] = weighted_reg
        out["reg_loss_display"] = reg_loss
        out["weighted_reg_loss_display"] = weighted_reg
        out["loss_display"] = loss
        if (
            str(out.get("objective_group")) == "image_registration"
            and str(out.get("method_key")) == "idir_original_dense_siren_be_autograd"
            and is_finite(out.get("coords_batch_size"))
        ):
            scale = as_float(out.get("coords_batch_size"))
            out["reg_loss_display"] = reg_loss * scale
            out["weighted_reg_loss_display"] = weighted_reg * scale
            out["loss_display"] = data_loss + weighted_reg * scale
            out["display_scale_note"] = "IDIR original BE multiplied by coords_batch_size for plot-only pre-divide diagnostic."
        elif str(out.get("objective_group")) == "image_registration":
            out["display_scale_note"] = "Raw method-native BE scale."
        else:
            out["display_scale_note"] = "MSE-only SITReg-field objective; no regularization."

        key = (
            str(out.get("objective_group")),
            str(out.get("method_key")),
            alpha_id(out.get("alpha")),
            lr_id(out.get("lr")),
        )
        if key not in first_values:
            first_values[key] = {"loss": loss, "data_loss": data_loss}
        first = first_values[key]
        eps = 1e-12
        out["loss_relative_change_from_epoch1"] = (
            (loss - first["loss"]) / (abs(first["loss"]) + eps) if math.isfinite(loss) and math.isfinite(first["loss"]) else float("nan")
        )
        out["data_loss_relative_change_from_epoch1"] = (
            (data_loss - first["data_loss"]) / (abs(first["data_loss"]) + eps)
            if math.isfinite(data_loss) and math.isfinite(first["data_loss"])
            else float("nan")
        )
        if str(out.get("objective_group")) == FIELD_OBJECTIVE:
            first_loss = first["loss"]
            out["field_mse_ratio_to_epoch1"] = loss / first_loss if math.isfinite(loss) and first_loss > 0 else float("nan")
        augmented.append(out)
    return augmented


def final_metric_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return greedy_ag.final_metric_summary(rows)


def group_history(rows: Sequence[Dict[str, object]], objective: str, metric: str) -> Dict[Tuple[str, float], List[Dict[str, object]]]:
    groups: Dict[Tuple[str, float], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("objective_group")) != objective:
            continue
        if not is_finite(row.get(metric)):
            continue
        groups[(str(row.get("method_key")), as_float(row.get("lr")))].append(row)
    for group in groups.values():
        group.sort(key=lambda row: as_float(row.get("epoch")))
    return groups


def rows_objective(rows: Sequence[Mapping[str, object]]) -> str:
    return str(rows[0].get("objective_group")) if rows else ""


def plot_lines(
    ax: plt.Axes,
    groups: Mapping[Tuple[str, float], Sequence[Dict[str, object]]],
    metric: str,
    *,
    ylabel: str,
    log_y: bool = False,
) -> None:
    for (method_key, lr), rows in sorted(
        groups.items(),
        key=lambda item: (method_sort_index(rows_objective(item[1]), item[0][0]), item[0][1]),
    ):
        if not rows:
            continue
        sample = rows[0]
        label = f"{method_label(sample)} lr={lr:g}"
        ax.plot(
            [as_float(row.get("epoch")) for row in rows],
            [as_float(row.get(metric)) for row in rows],
            color=method_color(sample),
            linestyle=LR_STYLES.get(lr, "-"),
            linewidth=1.2,
            label=label,
        )
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if log_y:
        ax.set_yscale("log")


def plot_image_registration_losses(history_rows: Sequence[Dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    metrics = [
        ("data_loss", "Data loss"),
        ("weighted_reg_loss", "alpha * reg loss, raw"),
        ("weighted_reg_loss_display", "alpha * reg loss, display scale"),
        ("data_loss_relative_change_from_epoch1", "Data-loss relative change"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        plot_lines(ax, group_history(history_rows, "image_registration", metric), metric, ylabel=ylabel)
    axes[-1].set_xlabel("Epoch")
    axes[0].legend(fontsize=7, ncol=3)
    fig.suptitle("OASIS rank 9 reused image registration: loss curves by LR", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_field_mse_losses(history_rows: Sequence[Dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    metrics = [
        ("field_mse_objective_normalized_grid", "Brain-mask objective MSE, normalized-grid"),
        ("field_mse_ratio_to_epoch1", "MSE ratio to epoch 1"),
        ("loss_relative_change_from_epoch1", "Loss relative change"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        plot_lines(
            ax,
            group_history(history_rows, FIELD_OBJECTIVE, metric),
            metric,
            ylabel=ylabel,
            log_y=metric != "loss_relative_change_from_epoch1",
        )
    axes[-1].set_xlabel("Epoch")
    axes[0].legend(fontsize=7, ncol=3)
    fig.suptitle("OASIS rank 9 SITReg-field fitting: MSE curves by LR", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_final_metric_contact(summary_rows: Sequence[Dict[str, object]], path: Path) -> None:
    panels = [
        ("image_registration", "dice_mean", "Image registration: Dice", False),
        ("image_registration", "dense_image_loss", "Image registration: dense NCC loss", False),
        (FIELD_OBJECTIVE, "field_mse_objective_normalized_grid", "SITReg-field fit: brain-mask final MSE", True),
        (FIELD_OBJECTIVE, "dice_mean", "SITReg-field fit: posthoc Dice", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for ax, (objective, metric, title, log_y) in zip(axes.ravel(), panels):
        specs = FIELD_METHODS if objective == FIELD_OBJECTIVE else IMAGE_METHODS
        for method_key, spec in specs.items():
            rows = [
                row
                for row in summary_rows
                if str(row.get("objective_group")) == objective
                and str(row.get("method_key")) == method_key
                and is_finite(row.get(metric))
            ]
            rows.sort(key=lambda row: as_float(row.get("lr")))
            if not rows:
                continue
            ax.plot(
                [as_float(row.get("lr")) for row in rows],
                [as_float(row.get(metric)) for row in rows],
                marker="o",
                linewidth=1.4,
                color=str(spec["color"]),
                label=str(spec["label"]),
            )
        ax.set_xscale("log")
        ax.invert_xaxis()
        if log_y:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Learning rate")
        ax.grid(True, alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def best_lines(rows: Sequence[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    image_rows = [row for row in rows if str(row.get("objective_group")) == "image_registration"]
    field_rows = [row for row in rows if str(row.get("objective_group")) == FIELD_OBJECTIVE]
    if image_rows:
        best_dice = max(image_rows, key=lambda row: as_float(row.get("dice_mean")))
        lines.append(
            f"- Best image-registration Dice: {method_label(best_dice)} lr `{as_float(best_dice.get('lr')):g}` "
            f"Dice `{as_float(best_dice.get('dice_mean')):.4f}`."
        )
    if field_rows:
        best_mse = min(field_rows, key=lambda row: as_float(row.get("field_mse_objective_normalized_grid")))
        lines.append(
            f"- Best SITReg-field brain-mask MSE: {method_label(best_mse)} lr `{as_float(best_mse.get('lr')):g}` "
            f"MSE `{as_float(best_mse.get('field_mse_objective_normalized_grid')):.6g}`."
        )
    return lines


def write_report(
    args: argparse.Namespace,
    summary_rows: Sequence[Dict[str, object]],
    errors: Sequence[str],
) -> None:
    lines = [
        "# OASIS Rank-9 Registration vs SITReg-Field Fit LR Sweep",
        "",
        f"- Generated: `{c.now_text()}`",
        f"- Field output root: `{args.field_root}`",
        f"- Reused image-registration root: `{args.image_root}`",
        f"- Review root: `{args.review_root}`",
        f"- Completed summaries: `{len(summary_rows)}` / `18`",
        f"- Case: rank `9`, original sorted index `88`, fixed/moving `({EXPECTED_FIXED_ID}, {EXPECTED_MOVING_ID})`",
        "- Objectives: reused image registration with BE regularization; fixed-image-positive brain-mask SITReg-field MSE with no regularizer.",
        "- SITReg target: `bs4_sitreg_v0` best-Dice saved dense displacement, channel-first `zyx` repo/RAS, no reorientation.",
        "- Brain mask: raw fixed OASIS image > 0 after standard reorientation; segmentations are used only for label metrics.",
        "- IDIR display-scale regularization curves are diagnostic only; IDIR image-registration training preserves original BE scale.",
        "",
        "## Acceptance",
        "",
    ]
    if errors:
        lines.extend(f"- FAIL: {error}" for error in errors[:80])
        if len(errors) > 80:
            lines.append(f"- FAIL: ... {len(errors) - 80} additional errors")
    else:
        lines.append("- PASS: all expected jobs are present with expected case, objective, metadata, and artifacts.")
    lines.extend(["", "## Best Final Points", ""])
    lines.extend(best_lines(summary_rows))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary_runs.csv/json`: one row per completed run.",
            "- `loss_history_all.csv`: raw per-epoch history with run context.",
            "- `loss_history_augmented.csv`: history with weighted regularization and IDIR display-scale diagnostics.",
            "- `final_metric_summary.csv`: final metric table for quick plotting.",
            "- `image_registration_loss_curves.pdf`: reused Exp1 loss curves.",
            "- `sitreg_field_mse_curves.pdf`: SITReg-field MSE curves.",
            "- `final_metric_contact_sheet.pdf`: final Dice/NCC/MSE panels by learning rate.",
            "",
        ]
    )
    args.review_root.mkdir(parents=True, exist_ok=True)
    (args.review_root / "review_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    summary_rows = collect_combined_summaries(Path(args.image_root), Path(args.field_root))
    errors = check_expected(summary_rows, args)
    if errors and not args.allow_incomplete:
        args.review_root.mkdir(parents=True, exist_ok=True)
        c.write_json(args.review_root / "aggregation_errors.json", errors)
        raise RuntimeError("Rank-9 SITReg fit-mode aggregation failed: " + "; ".join(errors[:20]))

    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(args.review_root / "summary_runs.csv", summary_rows)
    c.write_json(args.review_root / "summary_runs.json", summary_rows)
    c.write_json(args.review_root / "aggregation_errors.json", errors)
    c.write_csv(args.review_root / "case_metrics_all.csv", collect_metric_file(summary_rows, "case_metrics.csv"))
    c.write_csv(args.review_root / "regularity_metrics_all.csv", collect_metric_file(summary_rows, "regularity_metrics.csv"))
    c.write_csv(args.review_root / "be_metrics_all.csv", collect_metric_file(summary_rows, "be_metrics.csv"))
    c.write_csv(args.review_root / "diffusion_metrics_all.csv", collect_metric_file(summary_rows, "diffusion_metrics.csv"))

    history_rows = collect_history(summary_rows)
    augmented_history = augment_history(history_rows)
    write_csv_stream(args.review_root / "loss_history_all.csv", history_rows)
    write_csv_stream(args.review_root / "loss_history_augmented.csv", augmented_history)
    c.write_csv(args.review_root / "final_metric_summary.csv", final_metric_summary(summary_rows))
    plot_image_registration_losses(augmented_history, args.review_root / "image_registration_loss_curves.pdf")
    plot_field_mse_losses(augmented_history, args.review_root / "sitreg_field_mse_curves.pdf")
    plot_final_metric_contact(summary_rows, args.review_root / "final_metric_contact_sheet.pdf")
    write_report(args, summary_rows, errors)
    print(
        json.dumps(
            {
                "summary_rows": len(summary_rows),
                "history_rows": len(history_rows),
                "errors": errors,
                "image_root": str(args.image_root),
                "field_root": str(args.field_root),
                "review_root": str(args.review_root),
            },
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
