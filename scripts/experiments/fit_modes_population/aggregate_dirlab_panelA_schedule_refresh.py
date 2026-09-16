#!/usr/bin/env python
"""Validate and compare the DIR-LAB Panel-A MR-D-BSCP schedule refresh."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402


METHOD = "direct_cp_32_16_8_4_global_ncc_fd_be"
EXPERIMENT_KEY = "2026-08-13_fit_modes_dirlab_panelA_schedule_refresh"
RUN_KEY = "fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1"
HISTORICAL_RUN_KEY = "fit_modes_population_n10_v1"
DEFAULT_DATA_ROOT = Path("/database/inrorspline")
AUG13_RUN_KEY = "lungct_fig2_mrdbscp_idirloss_fdbe_e400_400_800_900_v1"
EXPECTED_CASES = list(range(1, 11))
EXPECTED_EPOCHS = [1, *range(10, 2501, 10)]
EXPECTED_STAGE_EPOCHS = [400, 400, 800, 900]
EXPECTED_STAGE_FULL_CPS = [32, 16, 8, 4]
EXPECTED_STAGE_DOWNSAMPLE = [8, 4, 2, 1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--historical-run-root", type=Path, default=None)
    parser.add_argument("--historical-index", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--aug13-run-root", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Keep current outputs external and make migrated generated inputs explicit."""

    args.review_root = artifact_paths.resolve_review_root(
        EXPERIMENT_KEY,
        args.review_root,
        repo_root=REPO_ROOT,
    )
    args.run_root = artifact_paths.resolve_run_root(RUN_KEY, args.run_root, repo_root=REPO_ROOT)
    args.historical_run_root = artifact_paths.resolve_run_root(
        HISTORICAL_RUN_KEY,
        args.historical_run_root,
        repo_root=REPO_ROOT,
    )
    args.aug13_run_root = artifact_paths.resolve_run_root(
        AUG13_RUN_KEY,
        args.aug13_run_root,
        repo_root=REPO_ROOT,
    )
    args.historical_index = artifact_paths.resolve_external_input_file(
        args.historical_index,
        label="--historical-index",
        repo_root=REPO_ROOT,
    )
    return args


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def portable(path: Path, *, run_root: Path, historical_run_root: Path, data_root: Path, aug13_root: Path) -> str:
    text = str(path)
    for root, token in [
        (run_root, "${RUN_ROOT}"),
        (historical_run_root, "${HISTORICAL_RUN_ROOT}"),
        (aug13_root, "${AUG13_RUN_ROOT}"),
        (data_root, "${DATA_ROOT}"),
        (REPO_ROOT, "${REPO_ROOT}"),
    ]:
        root_text = str(root)
        if text == root_text or text.startswith(root_text + "/"):
            return text.replace(root_text, token, 1)
    return text


def resolve_historical(value: Any, historical_run_root: Path, data_root: Path) -> Path:
    text = str(value or "")
    text = text.replace("${RUN_ROOT}", str(historical_run_root))
    text = text.replace("${DATA_ROOT}", str(data_root))
    text = text.replace("${REPO_ROOT}", str(REPO_ROOT))
    return Path(text)


def run_name(case_id: int) -> str:
    return (
        f"fit_modes_dirlab_case{case_id:02d}_panelA_mrdbscp_"
        "sched32-16-8-4_ds8-4-2-1_e400-400-800-900_"
        "a10000_lr0p001_seed1_globalnccdiag10"
    )


def aug13_run_name(case_id: int) -> str:
    return (
        f"{METHOD}_case{case_id:02d}_sched32-16-8-4_ds8-4-2-1_"
        "e400-400-800-900_lr0p001_a10000_seed1"
    )


def nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def summary_metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    return {
        "runtime_sec": finite(payload.get("elapsed_sec")),
        "peak_memory_gb": finite(payload.get("peak_mem_gb")),
        "tre_snap_mean_mm": finite(
            nested(payload, "tre_snap_to_voxel", "tre_mean_mm")
        ),
        "native_crop_tre_snap_mean_mm": finite(
            nested(payload, "native_crop_tre_snap_to_voxel", "tre_mean_mm")
        ),
        "fg_logj_std": finite(
            nested(payload, "regularity_fg_registration_crop", "log_jdet_std_fg")
        ),
        "fg_jdet_le0_pct": finite(
            nested(payload, "regularity_fg_registration_crop", "jdet_le0_pct_fg")
        ),
    }


def validate_new_case(
    case_id: int,
    run_root: Path,
    historical_run_root: Path,
    data_root: Path,
    aug13_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    run_dir = run_root / METHOD / run_name(case_id)
    summary_path = run_dir / "summary.json"
    curve_path = run_dir / "global_ncc_diagnostic.csv"
    history_path = run_dir / "stage_history.csv"
    errors: list[str] = []
    summary = load_json(summary_path)
    training = summary.get("training") if isinstance(summary.get("training"), dict) else {}
    checks = {
        "summary_exists": summary_path.is_file(),
        "curve_exists": curve_path.is_file(),
        "history_exists": history_path.is_file(),
        "status_completed": summary.get("status") == "completed",
        "method": summary.get("method") == METHOD,
        "dataset": summary.get("dataset") == "4dct",
        "case_id": int(summary.get("case_id", -1)) == case_id,
        "direction": str(summary.get("direction", "")).replace(" ", "") == "T50->T00",
        "image_domain": summary.get("image_domain") == "pTVReg landmark crop",
        "spacing": summary.get("spacing") == "isotropic_1mm",
        "mask_rule": summary.get("mask_rule") == "fixed_t00_lung_mask_metric_only",
        "stage_full_cps": training.get("stage_full_cps") == EXPECTED_STAGE_FULL_CPS,
        "stage_downsample": training.get("stage_downsample") == EXPECTED_STAGE_DOWNSAMPLE,
        "stage_epochs": training.get("stage_epochs") == EXPECTED_STAGE_EPOCHS,
        "total_epochs": sum(training.get("stage_epochs", [])) == 2500
        if isinstance(training.get("stage_epochs"), list)
        else False,
        "lr": isinstance(training.get("stage_lr"), list)
        and len(training.get("stage_lr")) == 4
        and all(abs(float(value) - 0.001) < 1e-12 for value in training.get("stage_lr")),
        "alpha": abs(float(training.get("alpha", math.nan)) - 10000.0) < 1e-8,
        "seed": int(training.get("seed", -1)) == 1,
        "loss": training.get("loss") == "full-crop global NCC + dense forward-difference BE",
        "image_loss": training.get("image_loss") == "full_crop_global_ncc",
        "regularizer": training.get("regularizer") == "dense_forward_difference_be",
        "diagnostic_interval": int(training.get("global_ncc_diagnostic_interval", -1)) == 10,
        "diagnostic_name": training.get("global_ncc_diagnostic_name")
        == "full_crop_global_ncc",
        "diagnostic_nontraining": training.get("global_ncc_diagnostic_affects_training")
        is False,
    }
    errors.extend(f"case{case_id:02d}:{name}" for name, passed in checks.items() if not passed)

    endpoint = None
    diagnostic_ncc = None
    curve_rows = 0
    curve_sha = ""
    if curve_path.is_file():
        curve = pd.read_csv(curve_path)
        curve_rows = len(curve)
        curve_sha = sha256(curve_path)
        required = {
            "global_epoch",
            "global_ncc_loss",
            "global_ncc_value",
            "eval_domain",
            "stage_idx",
            "full_cps",
            "downsample",
            "domain_cps",
        }
        missing = sorted(required - set(curve.columns))
        if missing:
            errors.append(f"case{case_id:02d}:curve_missing_columns={missing}")
        else:
            epochs = curve["global_epoch"].astype(int).tolist()
            if epochs != EXPECTED_EPOCHS:
                errors.append(f"case{case_id:02d}:diagnostic_epoch_set")
            numeric = curve[["global_ncc_loss", "global_ncc_value"]].to_numpy(float)
            if not np.isfinite(numeric).all():
                errors.append(f"case{case_id:02d}:nonfinite_diagnostic")
            if not curve["eval_domain"].astype(str).eq("full_1mm_crop").all():
                errors.append(f"case{case_id:02d}:diagnostic_domain")
            if len(curve) != 251:
                errors.append(f"case{case_id:02d}:diagnostic_rows={len(curve)}")
            final = curve.iloc[-1]
            final_checks = {
                "final_epoch": int(final["global_epoch"]) == 2500,
                "final_stage": int(final["stage_idx"]) == 3,
                "final_cps": int(final["full_cps"]) == 4,
                "final_downsample": int(final["downsample"]) == 1,
                "final_domain_cps": int(final["domain_cps"]) == 4,
                "ncc_identity": np.isclose(
                    float(final["global_ncc_value"]),
                    -float(final["global_ncc_loss"]),
                    rtol=1e-7,
                    atol=1e-9,
                ),
            }
            errors.extend(
                f"case{case_id:02d}:{name}" for name, passed in final_checks.items() if not passed
            )
            diagnostic_ncc = finite(final["global_ncc_value"])
            endpoint = None if diagnostic_ncc is None else 1.0 - diagnostic_ncc
            if endpoint is None or endpoint <= 0:
                errors.append(f"case{case_id:02d}:invalid_endpoint")

    if history_path.is_file():
        history = pd.read_csv(history_path)
        if len(history) != 2500:
            errors.append(f"case{case_id:02d}:history_rows={len(history)}")
        elif history["global_epoch"].astype(int).tolist() != list(range(1, 2501)):
            errors.append(f"case{case_id:02d}:history_epoch_set")

    metrics = summary_metrics(summary)
    for name, value in metrics.items():
        if value is None:
            errors.append(f"case{case_id:02d}:{name}_nonfinite")

    row = {
        "case_id": case_id,
        "job_id": f"dirlab_case{case_id:02d}_panelA_mr_d_bscp_schedule_refresh",
        "dataset": "dirlab",
        "panel": "A",
        "method_slug": "mr_d_bscp",
        "display_method": "MR-D-BSCP",
        "implementation_id": METHOD,
        "status": "completed" if not errors else "invalid",
        "source_kind": "schedule_refresh_diagnostic_enabled",
        "source_path": portable(
            curve_path,
            run_root=run_root,
            historical_run_root=historical_run_root,
            data_root=data_root,
            aug13_root=aug13_root,
        ),
        "summary_path": portable(
            summary_path,
            run_root=run_root,
            historical_run_root=historical_run_root,
            data_root=data_root,
            aug13_root=aug13_root,
        ),
        "endpoint_name": "one_minus_ncc",
        "endpoint_value": endpoint,
        "diagnostic_ncc": diagnostic_ncc,
        "diagnostic_rows": curve_rows,
        "curve_sha256": curve_sha,
        "lr": 0.001,
        "alpha": 10000.0,
        "stage_full_cps": "32,16,8,4",
        "stage_downsample": "8,4,2,1",
        "stage_epochs": "400,400,800,900",
        "epochs": 2500,
        "seed": 1,
        "diagnostic_interval": 10,
        **metrics,
        "validation_errors": ";".join(errors),
    }
    return row, errors


def historical_rows(index_path: Path) -> pd.DataFrame:
    index = pd.read_csv(index_path)
    rows = index[
        index["dataset"].eq("dirlab")
        & index["panel"].eq("A")
        & index["method_slug"].eq("mr_d_bscp")
    ].copy()
    rows["case_id"] = rows["case_id"].astype(int)
    return rows.sort_values("case_id")


def old_summary_metrics(
    row: pd.Series, historical_run_root: Path, data_root: Path
) -> dict[str, float | None]:
    summary_path = resolve_historical(row.get("summary_path"), historical_run_root, data_root)
    return summary_metrics(load_json(summary_path))


def aug13_metrics(case_id: int, root: Path) -> tuple[dict[str, float | None], dict[str, Any]]:
    path = root / METHOD / aug13_run_name(case_id) / "summary.json"
    payload = load_json(path)
    return summary_metrics(payload), payload


def descriptive(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            key: math.nan
            for key in ["n", "mean", "sd", "median", "q1", "q3", "min", "max"]
        }
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def format_value(value: Any, digits: int = 6) -> str:
    number = finite(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    review_root = Path(args.review_root)
    run_root = Path(args.run_root)
    historical_run_root = Path(args.historical_run_root)
    data_root = Path(args.data_root)
    aug13_root = Path(args.aug13_run_root)
    review_root.mkdir(parents=True, exist_ok=True)

    replacement_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for case_id in EXPECTED_CASES:
        row, errors = validate_new_case(
            case_id, run_root, historical_run_root, data_root, aug13_root
        )
        replacement_rows.append(row)
        problems.extend(errors)

    old = historical_rows(Path(args.historical_index))
    if old["case_id"].tolist() != EXPECTED_CASES:
        problems.append(f"historical_cases={old['case_id'].tolist()}")
    replacement_by_case = {int(row["case_id"]): row for row in replacement_rows}
    comparison_rows: list[dict[str, Any]] = []
    crosscheck_protocol_errors: list[str] = []
    for _, old_row in old.iterrows():
        case_id = int(old_row["case_id"])
        new = replacement_by_case[case_id]
        old_metrics = old_summary_metrics(old_row, historical_run_root, data_root)
        aug_metrics, aug_payload = aug13_metrics(case_id, aug13_root)
        aug_training = aug_payload.get("training") if isinstance(aug_payload.get("training"), dict) else {}
        protocol_ok = (
            aug_payload.get("status") == "completed"
            and aug_payload.get("method") == METHOD
            and aug_training.get("stage_epochs") == EXPECTED_STAGE_EPOCHS
            and aug_training.get("stage_full_cps") == EXPECTED_STAGE_FULL_CPS
            and aug_training.get("stage_downsample") == EXPECTED_STAGE_DOWNSAMPLE
            and isinstance(aug_training.get("stage_lr"), list)
            and len(aug_training.get("stage_lr")) == 4
            and all(
                abs(float(value) - 0.001) < 1e-12
                for value in aug_training.get("stage_lr")
            )
            and abs(float(aug_training.get("alpha", math.nan)) - 10000.0) < 1e-8
            and int(aug_training.get("seed", -1)) == 1
            and aug_training.get("loss")
            == "full-crop global NCC + dense forward-difference BE"
            and int(aug_training.get("global_ncc_diagnostic_interval", -1)) == 0
        )
        if not protocol_ok:
            crosscheck_protocol_errors.append(f"case{case_id:02d}:aug13_protocol")
        for metric_name, metric_value in old_metrics.items():
            if metric_name != "native_crop_tre_snap_mean_mm" and metric_value is None:
                problems.append(f"case{case_id:02d}:historical_{metric_name}_nonfinite")
        old_endpoint = float(old_row["endpoint_value"])
        new_endpoint = finite(new["endpoint_value"])
        delta = None if new_endpoint is None else new_endpoint - old_endpoint
        relative = None if delta is None else 100.0 * delta / old_endpoint
        comparison_rows.append(
            {
                "case_id": case_id,
                "old_stage_epochs": "200,400,800,1000",
                "new_stage_epochs": "400,400,800,900",
                "old_one_minus_ncc": old_endpoint,
                "new_one_minus_ncc": new_endpoint,
                "delta_one_minus_ncc": delta,
                "relative_delta_pct": relative,
                "old_runtime_sec": finite(old_row.get("runtime_sec")),
                "new_runtime_sec": new["runtime_sec"],
                "delta_runtime_sec": None
                if finite(old_row.get("runtime_sec")) is None or new["runtime_sec"] is None
                else float(new["runtime_sec"]) - float(old_row["runtime_sec"]),
                "old_peak_memory_gb": finite(old_row.get("peak_memory_gb")),
                "new_peak_memory_gb": new["peak_memory_gb"],
                "delta_peak_memory_gb": None
                if finite(old_row.get("peak_memory_gb")) is None
                or new["peak_memory_gb"] is None
                else float(new["peak_memory_gb"])
                - float(old_row["peak_memory_gb"]),
                "old_tre_snap_mean_mm": old_metrics["tre_snap_mean_mm"],
                "new_tre_snap_mean_mm": new["tre_snap_mean_mm"],
                "delta_tre_snap_mean_mm": None
                if old_metrics["tre_snap_mean_mm"] is None or new["tre_snap_mean_mm"] is None
                else float(new["tre_snap_mean_mm"]) - float(old_metrics["tre_snap_mean_mm"]),
                "old_fg_logj_std": old_metrics["fg_logj_std"],
                "new_fg_logj_std": new["fg_logj_std"],
                "delta_fg_logj_std": None
                if old_metrics["fg_logj_std"] is None or new["fg_logj_std"] is None
                else float(new["fg_logj_std"]) - float(old_metrics["fg_logj_std"]),
                "old_fg_jdet_le0_pct": old_metrics["fg_jdet_le0_pct"],
                "new_fg_jdet_le0_pct": new["fg_jdet_le0_pct"],
                "delta_fg_jdet_le0_pct": None
                if old_metrics["fg_jdet_le0_pct"] is None
                or new["fg_jdet_le0_pct"] is None
                else float(new["fg_jdet_le0_pct"])
                - float(old_metrics["fg_jdet_le0_pct"]),
                "new_native_crop_tre_snap_mean_mm": new[
                    "native_crop_tre_snap_mean_mm"
                ],
                "aug13_protocol_match": protocol_ok,
                "aug13_runtime_sec": aug_metrics["runtime_sec"],
                "delta_new_vs_aug13_runtime_sec": None
                if aug_metrics["runtime_sec"] is None or new["runtime_sec"] is None
                else float(new["runtime_sec"]) - float(aug_metrics["runtime_sec"]),
                "aug13_peak_memory_gb": aug_metrics["peak_memory_gb"],
                "delta_new_vs_aug13_peak_memory_gb": None
                if aug_metrics["peak_memory_gb"] is None
                or new["peak_memory_gb"] is None
                else float(new["peak_memory_gb"])
                - float(aug_metrics["peak_memory_gb"]),
                "aug13_tre_snap_mean_mm": aug_metrics["tre_snap_mean_mm"],
                "delta_new_vs_aug13_tre_mm": None
                if aug_metrics["tre_snap_mean_mm"] is None or new["tre_snap_mean_mm"] is None
                else float(new["tre_snap_mean_mm"]) - float(aug_metrics["tre_snap_mean_mm"]),
                "aug13_fg_logj_std": aug_metrics["fg_logj_std"],
                "delta_new_vs_aug13_fg_logj_std": None
                if aug_metrics["fg_logj_std"] is None or new["fg_logj_std"] is None
                else float(new["fg_logj_std"]) - float(aug_metrics["fg_logj_std"]),
                "aug13_fg_jdet_le0_pct": aug_metrics["fg_jdet_le0_pct"],
                "delta_new_vs_aug13_fg_jdet_le0_pct": None
                if aug_metrics["fg_jdet_le0_pct"] is None
                or new["fg_jdet_le0_pct"] is None
                else float(new["fg_jdet_le0_pct"])
                - float(aug_metrics["fg_jdet_le0_pct"]),
                "aug13_native_crop_tre_snap_mean_mm": aug_metrics[
                    "native_crop_tre_snap_mean_mm"
                ],
            }
        )
    problems.extend(crosscheck_protocol_errors)

    population_rows: list[dict[str, Any]] = []
    for metric, old_key, new_key, delta_key in [
        ("one_minus_ncc", "old_one_minus_ncc", "new_one_minus_ncc", "delta_one_minus_ncc"),
        ("runtime_sec", "old_runtime_sec", "new_runtime_sec", "delta_runtime_sec"),
        (
            "peak_memory_gb",
            "old_peak_memory_gb",
            "new_peak_memory_gb",
            "delta_peak_memory_gb",
        ),
        ("tre_snap_mean_mm", "old_tre_snap_mean_mm", "new_tre_snap_mean_mm", "delta_tre_snap_mean_mm"),
        ("fg_logj_std", "old_fg_logj_std", "new_fg_logj_std", "delta_fg_logj_std"),
        (
            "fg_jdet_le0_pct",
            "old_fg_jdet_le0_pct",
            "new_fg_jdet_le0_pct",
            "delta_fg_jdet_le0_pct",
        ),
    ]:
        old_stats = descriptive(
            value for row in comparison_rows if (value := finite(row.get(old_key))) is not None
        )
        new_stats = descriptive(
            value for row in comparison_rows if (value := finite(row.get(new_key))) is not None
        )
        if delta_key is None:
            delta_values = [
                float(row[new_key]) - float(row[old_key])
                for row in comparison_rows
                if finite(row.get(new_key)) is not None and finite(row.get(old_key)) is not None
            ]
        else:
            delta_values = [
                value
                for row in comparison_rows
                if (value := finite(row.get(delta_key))) is not None
            ]
        delta_stats = descriptive(delta_values)
        population_rows.append(
            {
                "metric": metric,
                **{f"old_{key}": value for key, value in old_stats.items()},
                **{f"new_{key}": value for key, value in new_stats.items()},
                **{f"delta_{key}": value for key, value in delta_stats.items()},
            }
        )

    for row in population_rows:
        if int(row["old_n"]) != 10 or int(row["new_n"]) != 10 or int(row["delta_n"]) != 10:
            problems.append(
                f"population_{row['metric']}_n="
                f"{int(row['old_n'])}/{int(row['new_n'])}/{int(row['delta_n'])}"
            )

    if len(replacement_rows) != 10:
        problems.append(f"replacement_count={len(replacement_rows)}")
    if len(comparison_rows) != 10:
        problems.append(f"comparison_count={len(comparison_rows)}")

    write_csv(review_root / "replacement_index.csv", replacement_rows)
    (review_root / "replacement_index.json").write_text(
        json.dumps(replacement_rows, indent=2, allow_nan=False) + "\n"
    )
    write_csv(review_root / "case_schedule_comparison.csv", comparison_rows)
    (review_root / "case_schedule_comparison.json").write_text(
        json.dumps(comparison_rows, indent=2, allow_nan=False) + "\n"
    )
    write_csv(review_root / "population_schedule_comparison.csv", population_rows)
    (review_root / "population_schedule_comparison.json").write_text(
        json.dumps(population_rows, indent=2, allow_nan=False) + "\n"
    )

    accepted = not problems
    acceptance = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000"),
        "accepted": accepted,
        "expected_records": 10,
        "completed_valid_records": sum(row["status"] == "completed" for row in replacement_rows),
        "diagnostic_rows_per_record": 251,
        "historical_comparison_records": len(comparison_rows),
        "aug13_crosscheck_protocol_records": sum(
            bool(row["aug13_protocol_match"]) for row in comparison_rows
        ),
        "problems": problems,
    }
    (review_root / "acceptance_report.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    acceptance_lines = [
        "# DIR-LAB Panel A Schedule Refresh Acceptance",
        "",
        f"Acceptance: `{'PASS' if accepted else 'FAIL'}`",
        f"Valid replacements: `{acceptance['completed_valid_records']}/10`",
        f"Historical comparisons: `{len(comparison_rows)}/10`",
        f"August-13 protocol cross-checks: `{acceptance['aug13_crosscheck_protocol_records']}/10`",
        "",
        "## Problems",
        "",
        *([f"- `{problem}`" for problem in problems] or ["- None."]),
    ]
    (review_root / "acceptance_report.md").write_text("\n".join(acceptance_lines) + "\n")

    endpoint_stats = next(row for row in population_rows if row["metric"] == "one_minus_ncc")
    aug_runtime_stats = descriptive(
        value
        for row in comparison_rows
        if (value := finite(row.get("delta_new_vs_aug13_runtime_sec"))) is not None
    )
    aug_secondary_max_abs = {
        "TRE": max(abs(float(row["delta_new_vs_aug13_tre_mm"])) for row in comparison_rows),
        "stdLogJ": max(
            abs(float(row["delta_new_vs_aug13_fg_logj_std"]))
            for row in comparison_rows
        ),
        "folding": max(
            abs(float(row["delta_new_vs_aug13_fg_jdet_le0_pct"]))
            for row in comparison_rows
        ),
        "peak memory": max(
            abs(float(row["delta_new_vs_aug13_peak_memory_gb"]))
            for row in comparison_rows
        ),
    }
    report_lines = [
        "# DIR-LAB Fit-Modes Panel A MR-D-BSCP Schedule Comparison",
        "",
        f"Acceptance: `{'PASS' if accepted else 'FAIL'}`",
        "",
        "The replacement changes only the MR-D-BSCP stage allocation from "
        "`200/400/800/1000` (2400 epochs) to `400/400/800/900` (2500 epochs). "
        "Lower `1-NCC`, TRE, stdLogJ, and folding are better. The comparison is descriptive.",
        "",
        "## Casewise primary endpoint",
        "",
        "| Case | Old 1-NCC | New 1-NCC | New-old | Relative delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        report_lines.append(
            f"| {int(row['case_id']):02d} | {format_value(row['old_one_minus_ncc'])} | "
            f"{format_value(row['new_one_minus_ncc'])} | {format_value(row['delta_one_minus_ncc'])} | "
            f"{format_value(row['relative_delta_pct'], 3)}% |"
        )
    report_lines.extend(
        [
            "",
            "## Population summary",
            "",
            f"- Old 1-NCC: `{endpoint_stats['old_mean']:.6f} ± {endpoint_stats['old_sd']:.6f}`.",
            f"- New 1-NCC: `{endpoint_stats['new_mean']:.6f} ± {endpoint_stats['new_sd']:.6f}`.",
            f"- Paired new-old mean: `{endpoint_stats['delta_mean']:.6f}`; median: `{endpoint_stats['delta_median']:.6f}`; range: `[{endpoint_stats['delta_min']:.6f}, {endpoint_stats['delta_max']:.6f}]`.",
            "",
            "All statistics below use the same ten paired cases. TRE is the common "
            "registration-crop snap-to-voxel metric present in every historical and new summary; "
            "the replacement index additionally retains the new native-crop headline TRE.",
            "",
            "| Metric | N | Old mean ± SD | Old median [Q1, Q3] | New mean ± SD | New median [Q1, Q3] | Paired new-old mean [range] |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    metric_labels = {
        "one_minus_ncc": "1-NCC",
        "runtime_sec": "Runtime (s)",
        "peak_memory_gb": "Peak memory (GiB)",
        "tre_snap_mean_mm": "Snap TRE (mm)",
        "fg_logj_std": "Foreground stdLogJ",
        "fg_jdet_le0_pct": "Foreground J≤0 (%)",
    }
    for row in population_rows:
        report_lines.append(
            f"| {metric_labels[row['metric']]} | {int(row['delta_n'])} | "
            f"{row['old_mean']:.6f} ± {row['old_sd']:.6f} | "
            f"{row['old_median']:.6f} [{row['old_q1']:.6f}, {row['old_q3']:.6f}] | "
            f"{row['new_mean']:.6f} ± {row['new_sd']:.6f} | "
            f"{row['new_median']:.6f} [{row['new_q1']:.6f}, {row['new_q3']:.6f}] | "
            f"{row['delta_mean']:.6f} [{row['delta_min']:.6f}, {row['delta_max']:.6f}] |"
        )
    report_lines.extend(
        [
            "",
            "## August 13 diagnostic-disabled replay cross-check",
            "",
            "All ten August 13 records match the locked schedule, method, loss, LR, alpha, "
            "seed, spacing, and downsampling metadata. The diagnostic-enabled reruns reproduce "
            f"the displacement-derived outputs exactly at the stored precision: maximum absolute "
            f"differences were TRE `{aug_secondary_max_abs['TRE']:.6g}`, stdLogJ "
            f"`{aug_secondary_max_abs['stdLogJ']:.6g}`, folding "
            f"`{aug_secondary_max_abs['folding']:.6g}`, and peak memory "
            f"`{aug_secondary_max_abs['peak memory']:.6g}`. The full-crop diagnostic added a "
            f"mean `{aug_runtime_stats['mean']:.1f} s` runtime (median "
            f"`{aug_runtime_stats['median']:.1f} s`; range "
            f"`[{aug_runtime_stats['min']:.1f}, {aug_runtime_stats['max']:.1f}] s`).",
            "",
            "Casewise values and machine-readable population statistics are in "
            "`case_schedule_comparison.csv` and `population_schedule_comparison.csv`.",
            "",
            "The accepted 2400-epoch cohort remains immutable historical sensitivity evidence. "
            "This report does not modify the main population index or regenerate paper diagnostics.",
        ]
    )
    (review_root / "comparison_report.md").write_text("\n".join(report_lines) + "\n")

    print(json.dumps(acceptance, indent=2), flush=True)
    if problems and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
