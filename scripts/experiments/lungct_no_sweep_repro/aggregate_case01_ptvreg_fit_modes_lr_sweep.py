#!/usr/bin/env python
"""Aggregate a LungCT registration vs pTVReg-field fitting LR sweep."""

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

import aggregate_report as base_report  # noqa: E402
import common as c  # noqa: E402


DEFAULT_CASE_ID = 1
KNOWN_SOURCE_TRE = {
    1: 0.7671410824122017,
    8: 1.014219831733267,
}
LRS = [1e-2, 1e-3, 1e-4]


def variant_for_case(case_id: int) -> str:
    return f"lungct_case{int(case_id):02d}_ptvreg_fit_modes_lr_sweep_v1"


def default_run_root(case_id: int) -> Path:
    return Path(f"/database/inrorspline/runs/{variant_for_case(case_id)}")


def default_review_root(case_id: int) -> Path:
    return c.REPO_ROOT / f"experiments/2026-06-26_lungct_case{int(case_id):02d}_ptvreg_fit_modes_lr_sweep"

IMAGE_METHODS = {
    "idir_be_default_lungmask": {"alias": "idir", "label": "IDIR", "color": "#1f77b4"},
    "sinr_cps4_ptv_lcc_isotv": {"alias": "sinr_cps4", "label": "SINR cps4", "color": "#2ca02c"},
    "direct_cp_cps4_ptv_lcc_isotv": {"alias": "direct_cp_cps4", "label": "DirectCP cps4", "color": "#d62728"},
    "direct_cp_32_16_8_4_ptv_lcc_isotv": {"alias": "direct_cp_4stage", "label": "DirectCP 4stage cps4", "color": "#9467bd"},
}
FIELD_METHODS = {
    "idir_ptvreg_field_mse": {"alias": "idir", "label": "IDIR", "color": "#1f77b4"},
    "sinr_cps4_ptvreg_field_mse": {"alias": "sinr_cps4", "label": "SINR cps4", "color": "#2ca02c"},
    "direct_cp_cps4_ptvreg_field_mse": {"alias": "direct_cp_cps4", "label": "DirectCP cps4", "color": "#d62728"},
    "direct_cp_4stage_ptvreg_field_mse": {"alias": "direct_cp_4stage", "label": "DirectCP 4stage cps4", "color": "#9467bd"},
}
METHOD_ORDER = ["idir", "sinr_cps4", "direct_cp_cps4", "direct_cp_4stage"]
LR_STYLES = {1e-2: "-", 1e-3: "--", 1e-4: ":"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--case-id", type=int, default=DEFAULT_CASE_ID)
    parser.add_argument("--expected-source-tre", type=float, default=None)
    parser.add_argument("--expected-epochs", type=int, default=2500)
    parser.add_argument("--expected-stage-epochs", type=str, default="400,400,800,900")
    parser.add_argument("--expected-seed", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.out_root is None:
        args.out_root = default_run_root(int(args.case_id))
    if args.review_root is None:
        args.review_root = default_review_root(int(args.case_id))
    if args.expected_source_tre is None:
        args.expected_source_tre = KNOWN_SOURCE_TRE.get(int(args.case_id))
    return args


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(value: object) -> bool:
    return math.isfinite(as_float(value))


def fmt(value: object, digits: int = 3) -> str:
    f = as_float(value)
    if not math.isfinite(f):
        return "n/a"
    return f"{f:.{digits}f}"


def lr_id(value: object) -> str:
    return f"{as_float(value):.12g}"


def parse_int_list(value: object) -> List[int]:
    if isinstance(value, list):
        return [int(as_float(v)) for v in value if is_finite(v)]
    if value is None:
        return []
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(as_float(part)))
    return out


def objective_for_row(row: Mapping[str, object]) -> str:
    objective = str(row.get("objective", "")).strip()
    if objective == "ptvreg_field_fit_mse":
        return "ptvreg_field_fit"
    return "image_registration"


def method_alias(row: Mapping[str, object]) -> str:
    objective = objective_for_row(row)
    if objective == "ptvreg_field_fit":
        return str(FIELD_METHODS.get(str(row.get("method_key")), {}).get("alias", row.get("method")))
    return str(IMAGE_METHODS.get(str(row.get("method")), {}).get("alias", row.get("method")))


def method_label(row: Mapping[str, object]) -> str:
    alias = method_alias(row)
    for spec in list(IMAGE_METHODS.values()) + list(FIELD_METHODS.values()):
        if spec["alias"] == alias:
            return str(spec["label"])
    return alias


def method_color(row: Mapping[str, object]) -> str:
    alias = method_alias(row)
    for spec in list(IMAGE_METHODS.values()) + list(FIELD_METHODS.values()):
        if spec["alias"] == alias:
            return str(spec["color"])
    return "#444444"


def method_sort(alias: str) -> int:
    return METHOD_ORDER.index(alias) if alias in METHOD_ORDER else len(METHOD_ORDER)


def collect_summaries(out_root: Path, case_id: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(out_root.glob("*/*/summary.json")):
        payload = json.loads(path.read_text())
        if int(as_float(payload.get("case_id"))) != int(case_id):
            continue
        native = base_report.native_crop_tre_from_summary(path, payload)
        snap_native = native.get("snap", {}) if isinstance(native, dict) else {}
        sub_native = native.get("subvoxel", {}) if isinstance(native, dict) else {}
        identity_native = native.get("identity", {}) if isinstance(native, dict) else {}
        snap_1mm = payload.get("tre_snap_to_voxel", {})
        sub_1mm = payload.get("tre_subvoxel", {})
        identity_1mm = payload.get("identity_tre", {})
        reg = payload.get("regularity_fg_registration_crop", {})
        training = payload.get("training", {})
        if not isinstance(snap_1mm, dict):
            snap_1mm = {}
        if not isinstance(sub_1mm, dict):
            sub_1mm = {}
        if not isinstance(identity_1mm, dict):
            identity_1mm = {}
        if not isinstance(reg, dict):
            reg = {}
        if not isinstance(training, dict):
            training = {}
        lr = payload.get("lr", training.get("lr"))
        if not is_finite(lr) and isinstance(training.get("stage_lr"), list) and training.get("stage_lr"):
            lr = training["stage_lr"][0]
        row: Dict[str, object] = {
            "summary_path": str(path),
            "run_dir": str(path.parent),
            "run_name": payload.get("run_name", path.parent.name),
            "status": payload.get("status"),
            "objective_group": objective_for_row(payload),
            "method_alias": method_alias(payload),
            "method_label": method_label(payload),
            "method": payload.get("method"),
            "method_key": payload.get("method_key", payload.get("method")),
            "case_id": payload.get("case_id"),
            "lr": lr,
            "seed": payload.get("seed"),
            "epochs": payload.get("epochs", training.get("epochs")),
            "stage_epochs": ",".join(str(v) for v in training.get("stage_epochs", [])) if isinstance(training.get("stage_epochs"), list) else "",
            "native_crop_identity_tre_mean_mm": identity_native.get("tre_mean_mm"),
            "native_crop_snap_tre_mean_mm": snap_native.get("tre_mean_mm"),
            "native_crop_snap_tre_std_mm": snap_native.get("tre_std_mm"),
            "native_crop_sub_tre_mean_mm": sub_native.get("tre_mean_mm"),
            "reg1mm_identity_tre_mean_mm": identity_1mm.get("tre_mean_mm"),
            "reg1mm_snap_tre_mean_mm": snap_1mm.get("tre_mean_mm"),
            "reg1mm_sub_tre_mean_mm": sub_1mm.get("tre_mean_mm"),
            "fg_log_jdet_std": reg.get("log_jdet_std_fg"),
            "fg_jdet_le0_pct": reg.get("jdet_le0_pct_fg"),
            "fg_bending_energy": reg.get("bending_energy_fg"),
            "fg_diffusion_l2": reg.get("diffusion_l2_fg"),
            "elapsed_sec": payload.get("elapsed_sec"),
            "peak_mem_gb": payload.get("peak_mem_gb"),
            "final_loss": payload.get("final_loss"),
            "field_mse_objective_normalized_grid": payload.get("field_mse_objective_normalized_grid"),
            "field_mse_full_normalized_grid": payload.get("field_mse_full_normalized_grid"),
            "field_mse_objective_voxel": payload.get("field_mse_objective_voxel"),
            "field_mse_full_voxel": payload.get("field_mse_full_voxel"),
            "source_recomputed_snap_tre_mean_mm": payload.get("source_recomputed_snap_tre_mean_mm"),
            "target_reg1mm_snap_tre_mean_mm": payload.get("target_reg1mm_snap_tre_mean_mm"),
        }
        rows.append(row)
    rows.sort(key=lambda r: (str(r["objective_group"]), method_sort(str(r["method_alias"])), as_float(r["lr"])))
    return rows


def expected_keys() -> Dict[Tuple[str, str, str], Dict[str, object]]:
    expected: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for objective in ["image_registration", "ptvreg_field_fit"]:
        for alias in METHOD_ORDER:
            for lr in LRS:
                expected[(objective, alias, lr_id(lr))] = {"objective": objective, "method_alias": alias, "lr": float(lr)}
    return expected


def combo_key(row: Mapping[str, object]) -> Tuple[str, str, str]:
    return (str(row.get("objective_group")), str(row.get("method_alias")), lr_id(row.get("lr")))


def required_artifact_errors(row: Mapping[str, object]) -> List[str]:
    run_dir = Path(str(row["run_dir"]))
    errors = []
    if not (run_dir / "summary.json").exists():
        errors.append(f"Missing {run_dir / 'summary.json'}")
    if not (run_dir / "disp_vox_reg1mm_zyx.npy").exists():
        errors.append(f"Missing {run_dir / 'disp_vox_reg1mm_zyx.npy'}")
    if str(row.get("objective_group")) == "ptvreg_field_fit":
        for name in ["DONE", "target_audit.json", "model_state.pt"]:
            if not (run_dir / name).exists():
                errors.append(f"Missing {run_dir / name}")
        history_name = "stage_history.csv" if str(row.get("method_alias")) == "direct_cp_4stage" else "history.csv"
        if not (run_dir / history_name).exists():
            errors.append(f"Missing {run_dir / history_name}")
    else:
        history_name = "stage_history.csv" if str(row.get("method_alias")) == "direct_cp_4stage" else "loss_history.csv"
        if not (run_dir / history_name).exists():
            errors.append(f"Missing {run_dir / history_name}")
    return errors


def check_expected(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    expected = expected_keys()
    observed: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        observed[combo_key(row)].append(row)
    for key in sorted(set(expected) - set(observed)):
        errors.append(f"Missing run combo: objective={key[0]}, method={key[1]}, lr={key[2]}")
    for key in sorted(set(observed) - set(expected)):
        errors.append(f"Unexpected run combo: objective={key[0]}, method={key[1]}, lr={key[2]}")
    for key, group in sorted(observed.items()):
        if len(group) > 1:
            errors.append(f"Duplicate run combo {key}: {len(group)} summaries")
    for row in rows:
        run_name = str(row.get("run_name"))
        if str(row.get("status")) != "completed":
            errors.append(f"{run_name}: status {row.get('status')} != completed")
        if int(as_float(row.get("case_id"))) != int(args.case_id):
            errors.append(f"{run_name}: case_id {row.get('case_id')} != {args.case_id}")
        if int(as_float(row.get("seed"))) != int(args.expected_seed):
            errors.append(f"{run_name}: seed {row.get('seed')} != {args.expected_seed}")
        if str(row.get("method_alias")) == "direct_cp_4stage":
            if str(row.get("stage_epochs")) != str(args.expected_stage_epochs):
                errors.append(f"{run_name}: stage_epochs {row.get('stage_epochs')} != {args.expected_stage_epochs}")
        elif int(as_float(row.get("epochs"))) != int(args.expected_epochs):
            errors.append(f"{run_name}: epochs {row.get('epochs')} != {args.expected_epochs}")
        for metric in ["native_crop_snap_tre_mean_mm", "reg1mm_snap_tre_mean_mm", "fg_log_jdet_std", "fg_jdet_le0_pct"]:
            if not is_finite(row.get(metric)):
                errors.append(f"{run_name}: missing/non-finite metric {metric}")
        if str(row.get("objective_group")) == "ptvreg_field_fit":
            for metric in [
                "field_mse_objective_normalized_grid",
                "field_mse_full_normalized_grid",
                "field_mse_objective_voxel",
                "source_recomputed_snap_tre_mean_mm",
            ]:
                if not is_finite(row.get(metric)):
                    errors.append(f"{run_name}: missing/non-finite field metric {metric}")
            if args.expected_source_tre is not None:
                observed_tre = as_float(row.get("source_recomputed_snap_tre_mean_mm"))
                if abs(observed_tre - float(args.expected_source_tre)) > 1e-8:
                    errors.append(
                        f"{run_name}: pTVReg target TRE validation mismatch "
                        f"{observed_tre} != {args.expected_source_tre}"
                    )
        errors.extend(f"{run_name}: {err}" for err in required_artifact_errors(row))
    return errors


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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


def read_history(row: Mapping[str, object]) -> List[Dict[str, object]]:
    run_dir = Path(str(row["run_dir"]))
    if str(row.get("method_alias")) == "direct_cp_4stage":
        path = run_dir / "stage_history.csv"
    elif str(row.get("objective_group")) == "image_registration":
        path = run_dir / "loss_history.csv"
    else:
        path = run_dir / "history.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [dict(v) for v in csv.DictReader(f)]


def history_x(row: Mapping[str, object], hist_row: Mapping[str, object]) -> float:
    if str(row.get("method_alias")) == "direct_cp_4stage":
        if is_finite(hist_row.get("global_epoch")):
            return as_float(hist_row.get("global_epoch"))
        stage_epochs = parse_int_list(row.get("stage_epochs"))
        stage_idx = int(as_float(hist_row.get("stage_idx"))) if is_finite(hist_row.get("stage_idx")) else 0
        if stage_epochs and 0 <= stage_idx < len(stage_epochs):
            return float(sum(stage_epochs[:stage_idx]) + as_float(hist_row.get("epoch")))
        return as_float(hist_row.get("epoch"))
    return as_float(hist_row.get("epoch"))


def history_y(row: Mapping[str, object], hist_row: Mapping[str, object]) -> float:
    if str(row.get("objective_group")) == "ptvreg_field_fit":
        return as_float(
            hist_row.get(
                "field_mse_masked_normalized_grid",
                hist_row.get("field_mse_objective_normalized_grid", hist_row.get("loss")),
            )
        )
    return as_float(hist_row.get("loss"))


def best_row_for_objective(rows: Sequence[Dict[str, object]], objective: str, alias: str) -> Dict[str, object] | None:
    subset = [row for row in rows if row.get("objective_group") == objective and row.get("method_alias") == alias]
    if not subset:
        return None
    key = "native_crop_snap_tre_mean_mm" if objective == "image_registration" else "field_mse_objective_voxel"
    finite = [row for row in subset if is_finite(row.get(key))]
    if not finite:
        return None
    return min(finite, key=lambda r: as_float(r.get(key)))


def history_pairs(row: Mapping[str, object]) -> List[Tuple[float, float]]:
    hist = read_history(row)
    pairs = []
    for h in hist:
        x = history_x(row, h)
        y = history_y(row, h)
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    return pairs


def best_curve_label(row: Mapping[str, object], objective: str) -> str:
    lr = as_float(row.get("lr"))
    if objective == "image_registration":
        metric = fmt(row.get("native_crop_snap_tre_mean_mm"), 4)
        return f"{row.get('method_label')} lr={lr:g} (TRE={metric} mm)"
    metric = fmt(row.get("field_mse_objective_voxel"), 8)
    return f"{row.get('method_label')} lr={lr:g} (MSE={metric} vox)"


def plot_histories(rows: Sequence[Dict[str, object]], objective: str, path: Path) -> None:
    subset = [row for row in rows if row.get("objective_group") == objective]
    if not subset:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for ax, alias in zip(axes.reshape(-1), METHOD_ORDER):
        method_rows = [row for row in subset if row.get("method_alias") == alias]
        for row in sorted(method_rows, key=lambda r: as_float(r.get("lr"))):
            hist = read_history(row)
            xs = [history_x(row, h) for h in hist]
            ys = [history_y(row, h) for h in hist]
            pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
            if not pairs:
                continue
            xs, ys = zip(*pairs)
            lr = as_float(row.get("lr"))
            ax.plot(xs, ys, LR_STYLES.get(lr, "-"), color=method_color(row), label=f"lr={lr:g}")
        ax.set_title(str(method_rows[0]["method_label"]) if method_rows else alias)
        ax.set_xlabel("epoch")
        ax.set_ylabel("masked MSE" if objective == "ptvreg_field_fit" else "training loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_final_metric(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for objective, ax in zip(["image_registration", "ptvreg_field_fit"], axes):
        subset = [row for row in rows if row.get("objective_group") == objective]
        for alias in METHOD_ORDER:
            method_rows = sorted([row for row in subset if row.get("method_alias") == alias], key=lambda r: as_float(r.get("lr")))
            if not method_rows:
                continue
            xs = [as_float(row.get("lr")) for row in method_rows]
            y_key = "native_crop_snap_tre_mean_mm" if objective == "image_registration" else "field_mse_objective_voxel"
            ys = [as_float(row.get(y_key)) for row in method_rows]
            ax.plot(xs, ys, marker="o", color=method_color(method_rows[0]), label=str(method_rows[0]["method_label"]))
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("learning rate")
        ax.set_ylabel("original snap TRE (mm)" if objective == "image_registration" else "masked field MSE (voxel)")
        ax.set_title("Image registration" if objective == "image_registration" else "pTVReg field fit")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_best_field_mse(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2))
    for alias in METHOD_ORDER:
        row = best_row_for_objective(rows, "ptvreg_field_fit", alias)
        if row is None:
            continue
        pairs = [(x, y) for x, y in history_pairs(row) if y > 0]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, color=method_color(row), linewidth=1.8, label=best_curve_label(row, "ptvreg_field_fit"))
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE (normalized-grid units)")
    ax.set_title("Best-LR pTVReg Field-Fit MSE")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_best_registration_losses(rows: Sequence[Dict[str, object]], idir_path: Path, spline_path: Path) -> None:
    idir_row = best_row_for_objective(rows, "image_registration", "idir")
    if idir_row is not None:
        pairs = history_pairs(idir_row)
        if pairs:
            xs, ys = zip(*pairs)
            fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
            ax.plot(xs, ys, color=method_color(idir_row), linewidth=1.8, label=best_curve_label(idir_row, "image_registration"))
            ax.set_xlabel("epoch")
            ax.set_ylabel("training loss")
            ax.set_title("Best-LR Registration Loss: IDIR")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            idir_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(idir_path, dpi=180)
            plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2))
    for alias in ["sinr_cps4", "direct_cp_cps4", "direct_cp_4stage"]:
        row = best_row_for_objective(rows, "image_registration", alias)
        if row is None:
            continue
        pairs = history_pairs(row)
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, color=method_color(row), linewidth=1.8, label=best_curve_label(row, "image_registration"))
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Best-LR Registration Loss: Spline Methods")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    spline_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(spline_path, dpi=180)
    plt.close(fig)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def best_rows(rows: Sequence[Dict[str, object]]) -> List[Sequence[object]]:
    out: List[Sequence[object]] = []
    for objective in ["image_registration", "ptvreg_field_fit"]:
        for alias in METHOD_ORDER:
            subset = [row for row in rows if row.get("objective_group") == objective and row.get("method_alias") == alias]
            if not subset:
                continue
            key = "native_crop_snap_tre_mean_mm" if objective == "image_registration" else "field_mse_objective_voxel"
            finite = [row for row in subset if is_finite(row.get(key))]
            if not finite:
                continue
            best = min(finite, key=lambda r: as_float(r.get(key)))
            out.append(
                [
                    objective,
                    best.get("method_label"),
                    f"{as_float(best.get('lr')):g}",
                    fmt(best.get("native_crop_snap_tre_mean_mm"), 4),
                    fmt(best.get("field_mse_objective_voxel"), 8),
                    fmt(best.get("fg_log_jdet_std"), 6),
                    fmt(best.get("elapsed_sec"), 1),
                ]
            )
    return out


def main() -> None:
    args = parse_args()
    rows = collect_summaries(Path(args.out_root), int(args.case_id))
    errors = check_expected(rows, args)
    if errors and not args.allow_incomplete:
        args.review_root.mkdir(parents=True, exist_ok=True)
        c.write_json(args.review_root / "aggregation_errors.json", errors)
        raise RuntimeError("Aggregation validation failed:\n" + "\n".join(errors[:50]))

    args.review_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.review_root / "summary_runs.csv", rows)
    c.write_json(args.review_root / "summary_runs.json", rows)
    c.write_json(
        args.review_root / "aggregation_status.json",
        {
            "completed_at": c.now_text(),
            "variant": variant_for_case(int(args.case_id)),
            "case_id": int(args.case_id),
            "rows": len(rows),
            "errors": errors,
            "pass": not errors,
        },
    )
    plot_dir = args.review_root / "plots"
    plot_histories(rows, "image_registration", plot_dir / "image_registration_loss_curves.png")
    plot_histories(rows, "ptvreg_field_fit", plot_dir / "ptvreg_field_mse_curves.png")
    plot_final_metric(rows, plot_dir / "final_metric_vs_lr.png")
    plot_best_field_mse(rows, plot_dir / "best_lr_masked_mse_curves.png")
    plot_best_registration_losses(
        rows,
        plot_dir / "best_lr_registration_loss_idir.png",
        plot_dir / "best_lr_registration_loss_spline_methods.png",
    )

    report = "\n".join(
        [
            f"# LungCT Case{int(args.case_id):02d} pTVReg Fit-Modes LR Sweep",
            "",
            f"Completed rows: `{len(rows)}/24`.",
            f"Validation: `{'PASS' if not errors else 'INCOMPLETE/FAIL'}`.",
            "",
            "## Best Rows",
            "",
            markdown_table(
                ["Objective", "Method", "LR", "Original Snap TRE", "Masked Field MSE vox", "FG logJ std", "Time sec"],
                best_rows(rows),
            ),
            "",
            "## Artifacts",
            "",
            "- `summary_runs.csv` and `summary_runs.json` contain one row per completed job.",
            "- `plots/image_registration_loss_curves.png` shows registration training losses.",
            "- `plots/ptvreg_field_mse_curves.png` shows field-fit MSE curves.",
            "- `plots/final_metric_vs_lr.png` summarizes final LR sensitivity.",
            "- `plots/best_lr_masked_mse_curves.png` overlays the best-LR field-fit MSE curve for each method with log-scale y-axis.",
            "- `plots/best_lr_registration_loss_idir.png` shows the best-LR IDIR registration loss separately because its objective scale differs.",
            "- `plots/best_lr_registration_loss_spline_methods.png` overlays the best-LR spline registration losses, with 4-stage DirectCP plotted on a continuous 1-2500 epoch axis.",
            "",
            "## Protocol Notes",
            "",
            "- Field-fit MSE is optimized only inside the fixed T00 lung mask in the 1mm crop.",
            "- Full-crop field MSE is diagnostic only.",
            "- Original Snap TRE means native/crop-space snap-to-voxel TRE after resizing the saved 1mm displacement back to the pTVReg native landmark crop.",
            "- The pTVReg target loader validates the source orientation by reproducing "
            + (
                f"`TRE_phys={float(args.expected_source_tre):.10f}`."
                if args.expected_source_tre is not None
                else "the saved `TRE_phys` value."
            ),
        ]
    )
    if errors:
        report += "\n\n## Validation Errors\n\n" + "\n".join(f"- {err}" for err in errors[:100]) + "\n"
    (args.review_root / "review_report.md").write_text(report + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "pass": not errors,
                "review_root": str(args.review_root),
                "summary_csv": str(args.review_root / "summary_runs.csv"),
                "report": str(args.review_root / "review_report.md"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
