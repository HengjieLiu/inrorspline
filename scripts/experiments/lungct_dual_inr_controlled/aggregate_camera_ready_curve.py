#!/usr/bin/env python3
"""Aggregate the compact DUAL-INR camera-ready TRE/stdlogJ curve sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments.lungct_dual_inr_author.portable_paths import (
    portable_review_value,
)
from scripts.experiments.lungct_dual_inr_controlled.aggregate_lungmask_revision import (
    add_regularity,
    inspect_record,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CASES = tuple(range(1, 11))
VARIANTS = ("PL1D", "PL1B", "PL1A", "PL1C")
NATIVE_MULTIPLIERS = (0.01, 0.1, 1.0, 10.0, 100.0)
PTV_WEIGHTS = (0.0055, 0.017, 0.055, 0.17, 0.55)
P27_ROOT = Path("/database/inrorspline/runs/lungct_dual_inr_1mm_lungmask_revision_v1")
P28_ROOT = Path("/database/inrorspline/runs/lungct_dual_inr_1mm_lungmask_author_schedule_v1")
P29_STAGE1_ROOT = Path(
    "/database/inrorspline/runs/lungct_dual_inr_camera_ready_curve_stage1_pl1c_default_v1"
)
PTVREG_ROWS = REPOSITORY_ROOT / (
    "experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/"
    "pareto_visual_review_midratio_extension/combined_case_rows_with_ptvreg.csv"
)
MR_ROWS = REPOSITORY_ROOT / (
    "experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/"
    "directcp_refined_case_rows.csv"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        if fieldnames:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(jsonable(row.get(key)), sort_keys=True)
                        if isinstance(row.get(key), (dict, list, tuple))
                        else "" if row.get(key) is None else row.get(key)
                        for key in fieldnames
                    }
                )
    os.replace(tmp, path)


def format_token(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p").replace("+", "")


def mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None, None
    return statistics.fmean(finite), statistics.stdev(finite) if len(finite) > 1 else 0.0


@dataclass(frozen=True)
class CurvePoint:
    curve_key: str
    variant: str
    curve_label: str
    weight_label: str
    weight_value: float
    regularization_multiplier: float | None
    ptv_isotv_weight: float | None
    run_root: Path
    run_tag: str | None
    source: str

    @property
    def inspect_root(self) -> Path:
        return self.run_root / self.run_tag if self.run_tag else self.run_root


def stage2_points(stage2_run_root: Path) -> list[CurvePoint]:
    points: list[CurvePoint] = []
    for variant in ("PL1D", "PL1B", "PL1A"):
        for multiplier in NATIVE_MULTIPLIERS:
            label = f"m{format_token(multiplier)}"
            default = math.isclose(multiplier, 1.0)
            if default and variant in {"PL1D", "PL1B"}:
                root, tag, source = P27_ROOT, None, "P27 default reuse"
            elif default and variant == "PL1A":
                root, tag, source = P28_ROOT, None, "P28 default reuse"
            else:
                root, tag, source = (
                    stage2_run_root,
                    f"{variant.lower()}_{label}",
                    "P29 Stage 2 compact sweep",
                )
            points.append(
                CurvePoint(
                    curve_key=f"{variant}_{label}",
                    variant=variant,
                    curve_label=f"{variant} compact sweep",
                    weight_label=label,
                    weight_value=float(multiplier),
                    regularization_multiplier=float(multiplier),
                    ptv_isotv_weight=None,
                    run_root=root,
                    run_tag=tag,
                    source=source,
                )
            )
    for weight in PTV_WEIGHTS:
        label = f"w{format_token(weight)}"
        default = math.isclose(weight, 0.055, rel_tol=0.0, abs_tol=1e-12)
        root, tag, source = (
            (P29_STAGE1_ROOT, None, "P29 Stage 1 default reuse")
            if default
            else (stage2_run_root, f"pl1c_{label}", "P29 Stage 2 compact sweep")
        )
        points.append(
            CurvePoint(
                curve_key=f"PL1C_{label}",
                variant="PL1C",
                curve_label="PL1C compact sweep",
                weight_label=label,
                weight_value=float(weight),
                regularization_multiplier=None,
                ptv_isotv_weight=float(weight),
                run_root=root,
                run_tag=tag,
                source=source,
            )
        )
    return points


def finite(record: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def weight_contract_error(record: Mapping[str, Any], point: CurvePoint) -> str | None:
    if not record.get("technical_accepted"):
        return None
    multiplier = record.get("regularization_multiplier")
    weights = record.get("resolved_regularization_weights")
    if point.ptv_isotv_weight is not None:
        actual = None
        if isinstance(weights, Mapping):
            actual = weights.get("ptv_isotv_weight")
        if actual is None and point.source.endswith("default reuse"):
            actual = point.ptv_isotv_weight
        try:
            actual_f = float(actual)
        except (TypeError, ValueError):
            return "missing pTV isoTV weight metadata"
        if not math.isclose(actual_f, point.ptv_isotv_weight, rel_tol=1e-6, abs_tol=1e-9):
            return f"pTV isoTV weight {actual_f} != expected {point.ptv_isotv_weight}"
        return None
    if point.regularization_multiplier is not None:
        if multiplier is None and point.source.endswith("default reuse"):
            return None
        try:
            actual_m = float(multiplier)
        except (TypeError, ValueError):
            return "missing regularization multiplier metadata"
        if not math.isclose(
            actual_m, point.regularization_multiplier, rel_tol=1e-6, abs_tol=1e-9
        ):
            return (
                f"regularization multiplier {actual_m} != "
                f"expected {point.regularization_multiplier}"
            )
    return None


def load_reference_rows() -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if PTVREG_ROWS.is_file():
        rows = [
            row
            for row in csv.DictReader(PTVREG_ROWS.open(encoding="utf-8"))
            if row.get("method") == "ptvreg_reference"
        ]
        refs.extend(reference_summary("pTVReg reference", rows, "original_snap_tre_mean_mm"))
    if MR_ROWS.is_file():
        rows = []
        for row in csv.DictReader(MR_ROWS.open(encoding="utf-8")):
            try:
                reg_value = float(row.get("reg_value", "nan"))
            except ValueError:
                continue
            if (
                row.get("method") == "direct_cp_32_16_8_4_ptv_lcc_isotv"
                and math.isclose(reg_value, 0.055, rel_tol=0.0, abs_tol=1e-12)
            ):
                rows.append(row)
        refs.extend(
            reference_summary(
                "MR-D-BSCP pTV-loss w=0.055",
                rows,
                "original_snap_tre_mean_mm",
            )
        )
    return refs


def reference_summary(label: str, rows: Sequence[Mapping[str, str]], tre_key: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    tre = [float(row[tre_key]) for row in rows if row.get(tre_key)]
    logj = [float(row["fg_logj_std"]) for row in rows if row.get("fg_logj_std")]
    fold = [
        float(row["fg_jdet_le0_pct"])
        for row in rows
        if row.get("fg_jdet_le0_pct")
    ]
    tre_mean, tre_sd = mean_sd(tre)
    logj_mean, logj_sd = mean_sd(logj)
    fold_mean, fold_sd = mean_sd(fold)
    return [
        {
            "row_type": "reference",
            "label": label,
            "variant": "",
            "weight_label": "",
            "n": len(tre),
            "headline_native_snap_tre_mm_mean": tre_mean,
            "headline_native_snap_tre_mm_sd": tre_sd,
            "regularity_log_jdet_std_fg_mean": logj_mean,
            "regularity_log_jdet_std_fg_sd": logj_sd,
            "regularity_jdet_le0_pct_fg_mean": fold_mean,
            "regularity_jdet_le0_pct_fg_sd": fold_sd,
        }
    ]


def aggregate(stage2_run_root: Path, review_root: Path, seed: int) -> dict[str, Any]:
    points = stage2_points(stage2_run_root.resolve())
    records: list[dict[str, Any]] = []
    for point in points:
        for case_id in CASES:
            record = inspect_record(
                run_root=point.inspect_root,
                variant=point.variant,
                case_id=case_id,
                seed=seed,
            )
            record.update(asdict(point))
            error = weight_contract_error(record, point)
            if error:
                record["technical_accepted"] = False
                record.setdefault("technical_errors", []).append(error)
            records.append(record)
    regularity = add_regularity(records)

    point_rows: list[dict[str, Any]] = []
    for point in points:
        group = [
            row
            for row in records
            if row["curve_key"] == point.curve_key and row["technical_accepted"]
        ]
        reg_group = [row for row in group if row.get("regularity_status") == "complete"]
        row: dict[str, Any] = {
            **asdict(point),
            "row_type": "curve_point",
            "label": f"{point.variant} {point.weight_label}",
            "n": len(group),
            "regularity_complete_n": len(reg_group),
            "technical_complete": len(group) == len(CASES),
        }
        for metric in (
            "headline_native_snap_tre_mm",
            "headline_native_subvoxel_tre_mm",
            "registration_grid_subvoxel_tre_mm",
            "regularity_log_jdet_std_fg",
            "regularity_jdet_le0_pct_fg",
            "total_wall_seconds",
        ):
            values = [value for value in (finite(item, metric) for item in group) if value is not None]
            mean, sd = mean_sd(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        point_rows.append(row)

    references = load_reference_rows()
    best_by_variant: list[dict[str, Any]] = []
    for variant in VARIANTS:
        candidates = [
            row
            for row in point_rows
            if row["variant"] == variant
            and row["technical_complete"]
            and row.get("headline_native_snap_tre_mm_mean") is not None
        ]
        if candidates:
            best_by_variant.append(
                min(candidates, key=lambda row: float(row["headline_native_snap_tre_mm_mean"]))
            )
    overall_best = (
        min(best_by_variant, key=lambda row: float(row["headline_native_snap_tre_mm_mean"]))
        if best_by_variant
        else None
    )

    review_root = review_root.resolve()
    review_root.mkdir(parents=True, exist_ok=True)
    portable_roots = (
        (stage2_run_root.resolve(), "${STAGE2_RUN_ROOT}"),
        (P27_ROOT, "${P27_RUN_ROOT}"),
        (P28_ROOT, "${P28_RUN_ROOT}"),
        (P29_STAGE1_ROOT, "${P29_STAGE1_RUN_ROOT}"),
        (review_root, "${REVIEW_ROOT}"),
        (REPOSITORY_ROOT, "${REPO_ROOT}"),
    )
    atomic_csv(
        review_root / "stage2_curve_case_rows.csv",
        portable_review_value(records, portable_roots),
    )
    atomic_json(
        review_root / "stage2_curve_case_rows.json",
        portable_review_value({"schema_version": 1, "records": records}, portable_roots),
    )
    atomic_csv(
        review_root / "stage2_curve_points.csv",
        portable_review_value(point_rows, portable_roots),
    )
    atomic_json(
        review_root / "stage2_curve_points.json",
        portable_review_value({"schema_version": 1, "rows": point_rows}, portable_roots),
    )
    atomic_csv(
        review_root / "stage2_best_by_variant_with_references.csv",
        portable_review_value(references + best_by_variant, portable_roots),
    )
    acceptance = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage2_run_root": str(stage2_run_root.resolve()),
        "review_root": str(review_root),
        "expected_records": len(points) * len(CASES),
        "accepted_records": sum(bool(row["technical_accepted"]) for row in records),
        "regularity_complete_records": sum(row["status"] == "complete" for row in regularity),
        "all_curve_points_complete": all(row["technical_complete"] for row in point_rows),
        "overall_best": overall_best,
        "best_by_variant": best_by_variant,
        "technical_failures": [
            row
            for row in records
            if not row.get("technical_accepted")
        ],
    }
    atomic_json(
        review_root / "stage2_compact_all_case_sweep_acceptance.json",
        portable_review_value(acceptance, portable_roots),
    )
    plot_path = review_root / "stage2_coarse_tre_stdlogj_curve.png"
    render_plot(point_rows, references, plot_path)
    report = render_report(point_rows, references, best_by_variant, overall_best, plot_path)
    atomic_text(review_root / "stage2_compact_all_case_sweep_report.md", report)
    return acceptance


def fmt(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{f:.{digits}f}" if math.isfinite(f) else "n/a"


def render_plot(
    point_rows: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    colors = {
        "PL1D": "#1f77b4",
        "PL1B": "#d62728",
        "PL1A": "#2ca02c",
        "PL1C": "#9467bd",
    }
    markers = {"PL1D": "o", "PL1B": "s", "PL1A": "^", "PL1C": "D"}
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=180)
    for variant in VARIANTS:
        group = [
            row
            for row in point_rows
            if row["variant"] == variant
            and row.get("regularity_log_jdet_std_fg_mean") is not None
            and row.get("headline_native_snap_tre_mm_mean") is not None
        ]
        group = sorted(group, key=lambda row: float(row["weight_value"]))
        if not group:
            continue
        x = [float(row["regularity_log_jdet_std_fg_mean"]) for row in group]
        y = [float(row["headline_native_snap_tre_mm_mean"]) for row in group]
        ax.plot(
            x,
            y,
            marker=markers[variant],
            color=colors[variant],
            linewidth=1.5,
            markersize=5,
            label=variant,
        )
        for row, xx, yy in zip(group, x, y):
            ax.annotate(
                str(row["weight_label"]),
                (xx, yy),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                color=colors[variant],
            )
    for ref in references:
        if ref.get("regularity_log_jdet_std_fg_mean") is None:
            continue
        ax.scatter(
            [float(ref["regularity_log_jdet_std_fg_mean"])],
            [float(ref["headline_native_snap_tre_mm_mean"])],
            marker="x",
            s=52,
            color="#333333",
            label=str(ref["label"]),
        )
    ax.set_xlabel("Foreground stdlogJ")
    ax.set_ylabel("Native snap TRE mean (mm)")
    ax.set_title("DIR-Lab 4DCT coarse DUAL-INR TRE/stdlogJ sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def render_report(
    point_rows: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    best_by_variant: Sequence[Mapping[str, Any]],
    overall_best: Mapping[str, Any] | None,
    plot_path: Path,
) -> str:
    lines = [
        "# Stage 2 Compact All-Case Sweep Report",
        "",
        "## Status",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Expected curve records: `{len(point_rows) * len(CASES)}`",
        f"- Complete curve points: `{sum(bool(row.get('technical_complete')) for row in point_rows)}/{len(point_rows)}`",
        f"- Review plot: `{plot_path.name}`",
        "",
    ]
    if overall_best is not None:
        lines.extend(
            [
                "## Best TRE DUAL-INR Point",
                "",
                (
                    f"The best TRE point among the four DUAL-INR families is "
                    f"`{overall_best['variant']} {overall_best['weight_label']}` "
                    f"with native snap TRE `{fmt(overall_best['headline_native_snap_tre_mm_mean'])} +/- "
                    f"{fmt(overall_best['headline_native_snap_tre_mm_sd'])} mm` and "
                    f"foreground stdlogJ `{fmt(overall_best['regularity_log_jdet_std_fg_mean'], 4)}`."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Best-by-family Compared With References",
            "",
            "| Row | Weight | Native snap TRE mean +/- SD | FG stdlogJ mean +/- SD | FG J<=0 pct mean +/- SD | n |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for ref in references:
        lines.append(
            f"| {ref['label']} | n/a | "
            f"{fmt(ref.get('headline_native_snap_tre_mm_mean'))} +/- {fmt(ref.get('headline_native_snap_tre_mm_sd'))} | "
            f"{fmt(ref.get('regularity_log_jdet_std_fg_mean'), 4)} +/- {fmt(ref.get('regularity_log_jdet_std_fg_sd'), 4)} | "
            f"{fmt(ref.get('regularity_jdet_le0_pct_fg_mean'), 5)} +/- {fmt(ref.get('regularity_jdet_le0_pct_fg_sd'), 5)} | "
            f"{int(ref.get('n', 0))} |"
        )
    for row in best_by_variant:
        lines.append(
            f"| {row['variant']} best | {row['weight_label']} | "
            f"{fmt(row.get('headline_native_snap_tre_mm_mean'))} +/- {fmt(row.get('headline_native_snap_tre_mm_sd'))} | "
            f"{fmt(row.get('regularity_log_jdet_std_fg_mean'), 4)} +/- {fmt(row.get('regularity_log_jdet_std_fg_sd'), 4)} | "
            f"{fmt(row.get('regularity_jdet_le0_pct_fg_mean'), 5)} +/- {fmt(row.get('regularity_jdet_le0_pct_fg_sd'), 5)} | "
            f"{int(row.get('n', 0))} |"
        )
    lines.extend(
        [
            "",
            "## All Coarse Curve Points",
            "",
            "| Variant | Weight | Native snap TRE | Reg-grid subvoxel TRE | FG stdlogJ | FG J<=0 pct | Runtime | Complete |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(point_rows, key=lambda r: (str(r["variant"]), float(r["weight_value"]))):
        lines.append(
            f"| {row['variant']} | {row['weight_label']} | "
            f"{fmt(row.get('headline_native_snap_tre_mm_mean'))} +/- {fmt(row.get('headline_native_snap_tre_mm_sd'))} | "
            f"{fmt(row.get('registration_grid_subvoxel_tre_mm_mean'))} +/- {fmt(row.get('registration_grid_subvoxel_tre_mm_sd'))} | "
            f"{fmt(row.get('regularity_log_jdet_std_fg_mean'), 4)} +/- {fmt(row.get('regularity_log_jdet_std_fg_sd'), 4)} | "
            f"{fmt(row.get('regularity_jdet_le0_pct_fg_mean'), 5)} +/- {fmt(row.get('regularity_jdet_le0_pct_fg_sd'), 5)} | "
            f"{fmt(row.get('total_wall_seconds_mean'), 1)} +/- {fmt(row.get('total_wall_seconds_sd'), 1)} | "
            f"{row.get('n', 0)}/10 |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `stage2_curve_case_rows.csv`",
            "- `stage2_curve_points.csv`",
            "- `stage2_best_by_variant_with_references.csv`",
            "- `stage2_coarse_tre_stdlogj_curve.png`",
            "",
            "Acceptance is technical only. TRE and stdlogJ are used for curve selection after all cases are complete.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = aggregate(args.stage2_run_root, args.review_root, int(args.seed))
    print(
        f"Accepted {result['accepted_records']}/{result['expected_records']} records; "
        f"regularity complete {result['regularity_complete_records']}; "
        f"review: {args.review_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
