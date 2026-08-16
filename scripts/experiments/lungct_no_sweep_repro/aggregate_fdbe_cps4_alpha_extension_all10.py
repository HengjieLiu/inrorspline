#!/usr/bin/env python
"""Aggregate the LungCT unmasked FD-BE cps4 alpha-extension runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


DEFAULT_RUN_ROOT = Path("/database/inrorspline/runs/lungct_fdbe_cps4_alpha1e5_1e6_all10_v1")
DEFAULT_REVIEW_DIR = REPO_ROOT / "experiments/2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10"
DEFAULT_CASES = tuple(range(1, 11))
EXPECTED_DIAGNOSTIC_EPOCHS = [1, *range(10, 2501, 10)]


@dataclass(frozen=True)
class LaneSpec:
    lane_slug: str
    method_slug: str
    display_method: str
    method_family: str
    implementation_id: str
    lr: float
    alpha: float


LANES = (
    LaneSpec("inr_bscp_a100000", "inr_bscp", "INR-BSCP cps4 (FD-BE)", "sinr", "sinr_cps4_global_ncc_fd_be", 1.0e-4, 100000.0),
    LaneSpec("inr_bscp_a1000000", "inr_bscp", "INR-BSCP cps4 (FD-BE)", "sinr", "sinr_cps4_global_ncc_fd_be", 1.0e-4, 1000000.0),
    LaneSpec("d_bscp_a100000", "d_bscp", "D-BSCP cps4 (FD-BE)", "directcp", "direct_cp_cps4_global_ncc_fd_be", 1.0e-3, 100000.0),
    LaneSpec("d_bscp_a1000000", "d_bscp", "D-BSCP cps4 (FD-BE)", "directcp", "direct_cp_cps4_global_ncc_fd_be", 1.0e-3, 1000000.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--cases", default=",".join(str(case_id) for case_id in DEFAULT_CASES))
    parser.add_argument("--expected-epochs", type=int, default=2500)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def parse_cases(text: str) -> tuple[int, ...]:
    cases: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            case_id = int(token)
            if case_id < 1 or case_id > 10:
                raise ValueError(f"DIR-Lab case id out of range: {case_id}")
            cases.append(case_id)
    result = tuple(dict.fromkeys(cases))
    if not result:
        raise ValueError("--cases cannot be empty")
    return result


def run_name(case_id: int, lane: LaneSpec, epochs: int) -> str:
    return (
        f"{lane.implementation_id}_case{case_id:02d}_e{int(epochs)}_"
        f"lr{c.format_float(lane.lr)}_a{c.format_float(lane.alpha)}_seed1"
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def close(value: Any, target: float, tol: float = 1.0e-8) -> bool:
    value_f = finite(value)
    return value_f is not None and abs(value_f - float(target)) <= tol * max(1.0, abs(float(target)))


def last_history_value(summary: dict[str, Any], field: str) -> float | None:
    tail = summary.get("history_tail")
    if not isinstance(tail, list) or not tail:
        return None
    last = tail[-1]
    if not isinstance(last, dict):
        return None
    return finite(last.get(field))


def read_curve(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def curve_final_fields(path: Path, errors: list[str], case_id: int, lane: LaneSpec) -> dict[str, float | str | None]:
    rows = read_curve(path)
    if not rows:
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:curve_missing_or_empty")
        return {}
    if len(rows) != len(EXPECTED_DIAGNOSTIC_EPOCHS):
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:curve_rows={len(rows)}")
    epochs: list[int] = []
    for row in rows:
        try:
            epochs.append(int(float(row.get("global_epoch", row.get("epoch", "nan")))))
        except ValueError:
            epochs.append(-1)
    if epochs != EXPECTED_DIAGNOSTIC_EPOCHS:
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:diagnostic_epoch_set")
    final = rows[-1]
    if str(final.get("eval_domain", "")) != "full_1mm_crop":
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:curve_eval_domain")
    if str(final.get("lung_mask_affects_training", "")).lower() != "false":
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:lung_mask_affects_training")
    global_ncc = finite(final.get("global_ncc_value"))
    lung_one_minus = finite(final.get("lung_mask_one_minus_ncc"))
    return {
        "diagnostic_rows": len(rows),
        "full_crop_one_minus_ncc": None if global_ncc is None else 1.0 - global_ncc,
        "fixed_t00_lung_mask_one_minus_ncc": lung_one_minus,
        "final_global_ncc_value": global_ncc,
        "final_lung_mask_ncc_value": finite(final.get("lung_mask_ncc_value")),
    }


def summary_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    regularity = summary.get("regularity_fg_registration_crop")
    if not isinstance(regularity, dict):
        regularity = {}
    return {
        "original_snap_tre_mean_mm": finite(nested(summary, "native_crop_tre_snap_to_voxel", "tre_mean_mm")),
        "original_snap_tre_std_mm": finite(nested(summary, "native_crop_tre_snap_to_voxel", "tre_std_mm")),
        "original_subvoxel_tre_mean_mm": finite(nested(summary, "native_crop_tre_subvoxel", "tre_mean_mm")),
        "original_subvoxel_tre_std_mm": finite(nested(summary, "native_crop_tre_subvoxel", "tre_std_mm")),
        "reg1mm_snap_tre_mean_mm": finite(nested(summary, "tre_snap_to_voxel", "tre_mean_mm")),
        "reg1mm_snap_tre_std_mm": finite(nested(summary, "tre_snap_to_voxel", "tre_std_mm")),
        "reg1mm_subvoxel_tre_mean_mm": finite(nested(summary, "tre_subvoxel", "tre_mean_mm")),
        "reg1mm_subvoxel_tre_std_mm": finite(nested(summary, "tre_subvoxel", "tre_std_mm")),
        "fg_logj_std": finite(regularity.get("log_jdet_std_fg")),
        "fg_jdet_le0_pct": finite(regularity.get("jdet_le0_pct_fg")),
        "fg_bending_energy": finite(regularity.get("bending_energy_fg")),
        "fg_diffusion_l2": finite(regularity.get("diffusion_l2_fg")),
        "elapsed_sec": finite(summary.get("elapsed_sec")),
        "peak_mem_gb": finite(summary.get("peak_mem_gb")),
        "final_loss": last_history_value(summary, "loss"),
        "final_data_loss": last_history_value(summary, "data_loss"),
        "final_reg_loss": last_history_value(summary, "reg_loss"),
    }


def validate_summary(summary: dict[str, Any], case_id: int, lane: LaneSpec, expected_epochs: int) -> list[str]:
    errors: list[str] = []
    if summary.get("status") != "completed":
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:status_not_completed")
    if summary.get("method") != lane.implementation_id:
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:method")
    if "masked" in str(summary.get("method", "")).lower():
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:masked_method")
    if summary.get("dataset") != "4dct":
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:dataset")
    if int(summary.get("case_id", -1)) != int(case_id):
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:case_id")
    if str(summary.get("direction", "")).replace(" ", "") != "T50->T00":
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:direction")
    if summary.get("image_domain") != "pTVReg landmark crop":
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:image_domain")
    if summary.get("spacing") != "isotropic_1mm":
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:spacing")
    if summary.get("mask_rule") != "fixed_t00_lung_mask_metric_only":
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:mask_rule")
    if summary.get("loss_domain") != "full_crop_global_ncc":
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:loss_domain")
    if "masked" in str(summary.get("protocol_tag", "")).lower():
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:masked_protocol_tag")
    training = summary.get("training")
    if not isinstance(training, dict):
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:training_missing")
        return errors
    checks = {
        "epochs": int(training.get("epochs", -1)) == int(expected_epochs),
        "lr": close(training.get("lr"), lane.lr),
        "alpha": close(training.get("alpha"), lane.alpha),
        "cps": int(training.get("cps", -1)) == 4,
        "seed": int(training.get("seed", -1)) == 1,
        "image_loss": training.get("image_loss") == "full_crop_global_ncc",
        "regularizer": training.get("regularizer") == "dense_forward_difference_be",
        "global_ncc_diagnostic_interval": int(training.get("global_ncc_diagnostic_interval", -1)) == 10,
        "global_diagnostic_name": training.get("global_ncc_diagnostic_name") == "full_crop_global_ncc",
        "global_diagnostic_nontraining": training.get("global_ncc_diagnostic_affects_training") is False,
        "lung_mask_diagnostic_nontraining": training.get("lung_mask_ncc_diagnostic_affects_training") is False,
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(f"case{case_id:02d}:{lane.lane_slug}:{name}")
    return errors


def portable(path: Path, *, run_root: Path, review_dir: Path) -> str:
    text = str(path)
    for root, token in [(run_root, "${RUN_ROOT}"), (review_dir, "${REVIEW_DIR}"), (REPO_ROOT, "${REPO_ROOT}")]:
        root_text = str(root)
        if text == root_text or text.startswith(root_text + "/"):
            return text.replace(root_text, token, 1)
    return text


def case_row(run_root: Path, review_dir: Path, case_id: int, lane: LaneSpec, expected_epochs: int) -> tuple[dict[str, Any], list[str]]:
    run_dir = run_root / lane.implementation_id / run_name(case_id, lane, expected_epochs)
    summary_path = run_dir / "summary.json"
    curve_path = run_dir / "global_ncc_diagnostic.csv"
    summary = load_json(summary_path)
    errors = []
    if not summary_path.is_file():
        errors.append(f"case{case_id:02d}:{lane.lane_slug}:summary_missing")
    errors.extend(validate_summary(summary, case_id, lane, expected_epochs))
    curve_fields = curve_final_fields(curve_path, errors, case_id, lane)
    metrics = summary_metrics(summary)
    for name, value in metrics.items():
        if name not in {"final_loss", "final_data_loss", "final_reg_loss"} and value is None:
            errors.append(f"case{case_id:02d}:{lane.lane_slug}:{name}_nonfinite")
    row = {
        "dataset": "lungct10",
        "case_key": f"case{case_id:02d}",
        "case_id": int(case_id),
        "lane_slug": lane.lane_slug,
        "method_slug": lane.method_slug,
        "display_method": lane.display_method,
        "method_id": lane.implementation_id,
        "method_family": lane.method_family,
        "summary_path": portable(summary_path, run_root=run_root, review_dir=review_dir),
        "curve_path": portable(curve_path, run_root=run_root, review_dir=review_dir),
        "source_summary_path": portable(summary_path, run_root=run_root, review_dir=review_dir),
        "summary_exists": summary_path.is_file(),
        "tre_source": str(summary.get("headline_tre_metric", "native_crop_tre_snap_to_voxel")),
        "lr": float(lane.lr),
        "alpha": float(lane.alpha),
        "cps": "4",
        "epochs": int(expected_epochs),
        "seed": 1,
        "loss_domain": summary.get("loss_domain", ""),
        "mask_rule": summary.get("mask_rule", ""),
        "regularizer": nested(summary, "training", "regularizer"),
        "masked_loss_used": False,
        "status": "completed" if not errors else "invalid",
        "validation_errors": ";".join(errors),
        **metrics,
        **curve_fields,
    }
    return row, errors


def mean_std(values: Iterable[Any]) -> tuple[float | None, float | None]:
    finite_values = [float(value) for value in values if finite(value) is not None]
    if not finite_values:
        return None, None
    if len(finite_values) == 1:
        return finite_values[0], 0.0
    return statistics.mean(finite_values), statistics.pstdev(finite_values)


def aggregate_mean_rows(case_rows: Sequence[dict[str, Any]], source_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in LANES:
        group = [row for row in case_rows if row["lane_slug"] == lane.lane_slug]
        if not group:
            continue
        row: dict[str, Any] = {
            "method": lane.implementation_id,
            "method_id": lane.implementation_id,
            "method_label": lane.display_method,
            "display_method": lane.display_method,
            "method_family": lane.method_family,
            "reg_value": float(lane.alpha),
            "reg_label": f"a={lane.alpha:g}",
            "ratio_to_best_tre_weight": float(lane.alpha) / 10000.0,
            "cases": len({item["case_key"] for item in group}),
            "dataset": "lungct10",
            "selected_lr": float(lane.lr),
            "selected_lr_label": f"{lane.lr:g}",
            "source_path": source_path,
            "source_group": "lungct_fdbe_cps4_alpha1e5_1e6_all10",
            "schedule": "2500",
            "loss_domain": "full_crop_global_ncc",
            "regularizer": "dense_forward_difference_be",
            "panel3_include": False,
            "panel4_include": True,
            "masked_loss_used": False,
            "legacy_context": False,
        }
        for source_col, output_stem in [
            ("original_snap_tre_mean_mm", "original_snap_tre_mean_mm"),
            ("original_subvoxel_tre_mean_mm", "original_subvoxel_tre_mean_mm"),
            ("reg1mm_snap_tre_mean_mm", "reg1mm_snap_tre_mean_mm"),
            ("reg1mm_subvoxel_tre_mean_mm", "reg1mm_subvoxel_tre_mean_mm"),
            ("fg_logj_std", "fg_logj_std"),
            ("fg_jdet_le0_pct", "fg_jdet_le0_pct"),
            ("fg_bending_energy", "fg_bending_energy"),
            ("fg_diffusion_l2", "fg_diffusion_l2"),
            ("full_crop_one_minus_ncc", "full_crop_one_minus_ncc"),
            ("fixed_t00_lung_mask_one_minus_ncc", "fixed_t00_lung_mask_one_minus_ncc"),
            ("final_data_loss", "final_data_loss"),
            ("final_reg_loss", "final_reg_loss"),
            ("elapsed_sec", "elapsed_sec"),
            ("peak_mem_gb", "peak_mem_gb"),
        ]:
            mean, sd = mean_std(item.get(source_col) for item in group)
            row[f"{output_stem}_mean"] = mean
            row[f"{output_stem}_std"] = sd
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_report(path: Path, manifest: dict[str, Any], mean_rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# LungCT FD-BE cps4 alpha-extension aggregation",
        "",
        f"- Updated: `{manifest['updated_at_utc']}`",
        "- Scope: unmasked full-crop global NCC + dense FD-BE, LungCT cases 01-10.",
        "- Fixed T00 lung mask is metric-only in these rows.",
        f"- Acceptance: `{manifest['accepted']}` (`{manifest['completed_valid_rows']}/{manifest['expected_rows']}` valid rows).",
        "",
        "| Method | alpha | Cases | TRE | SDlogJ | Folding % | Full 1-NCC | Fixed-mask 1-NCC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mean_rows:
        lines.append(
            "| {method} | {alpha:g} | {cases} | {tre:.3f} | {logj:.4f} | {fold:.4f} | {full:.5f} | {mask:.5f} |".format(
                method=row["display_method"],
                alpha=float(row["reg_value"]),
                cases=int(row["cases"]),
                tre=float(row["original_snap_tre_mean_mm_mean"]),
                logj=float(row["fg_logj_std_mean"]),
                fold=float(row["fg_jdet_le0_pct_mean"]),
                full=float(row["full_crop_one_minus_ncc_mean"]),
                mask=float(row["fixed_t00_lung_mask_one_minus_ncc_mean"]),
            )
        )
    if manifest["validation_errors"]:
        lines.extend(["", "## Validation Errors", ""])
        for error in manifest["validation_errors"][:200]:
            lines.append(f"- `{error}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    review_dir = Path(args.review_dir)
    run_root = Path(args.run_root)
    review_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for case_id in parse_cases(args.cases):
        for lane in LANES:
            row, row_errors = case_row(run_root, review_dir, int(case_id), lane, int(args.expected_epochs))
            case_rows.append(row)
            errors.extend(row_errors)
    case_rows_path = review_dir / "case_rows.csv"
    mean_rows_path = review_dir / "mean_rows.csv"
    write_csv(case_rows_path, case_rows)
    source_path = str(case_rows_path.relative_to(REPO_ROOT)) if case_rows_path.is_relative_to(REPO_ROOT) else str(case_rows_path)
    mean_rows = aggregate_mean_rows(case_rows, source_path)
    write_csv(mean_rows_path, mean_rows)
    completed_valid = sum(row.get("status") == "completed" for row in case_rows)
    expected_rows = len(parse_cases(args.cases)) * len(LANES)
    accepted = completed_valid == expected_rows and not errors
    manifest = {
        "updated_at_utc": utc_text(),
        "run_root": str(run_root),
        "review_dir": str(review_dir),
        "case_rows": str(case_rows_path),
        "mean_rows": str(mean_rows_path),
        "cases": list(parse_cases(args.cases)),
        "expected_rows": expected_rows,
        "completed_valid_rows": completed_valid,
        "accepted": accepted,
        "allow_incomplete": bool(args.allow_incomplete),
        "validation_errors": errors,
    }
    (review_dir / "acceptance_report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(review_dir / "acceptance_report.md", manifest, mean_rows)
    print(json.dumps(manifest, indent=2))
    if not accepted and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
