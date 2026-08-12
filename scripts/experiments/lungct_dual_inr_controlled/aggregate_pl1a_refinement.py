#!/usr/bin/env python3
"""Aggregate PL1A refinement multipliers for final TRE/stdLogJ figures."""

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

import numpy as np

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
VARIANT = "PL1A"
CASES = tuple(range(1, 11))
MULTIPLIERS = (1.0 / 3.0, 3.0)


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


def finite(record: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(record[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class Point:
    curve_key: str
    variant: str
    curve_label: str
    weight_label: str
    weight_value: float
    regularization_multiplier: float
    ptv_isotv_weight: None
    run_root: Path
    run_tag: str
    source: str

    @property
    def inspect_root(self) -> Path:
        return self.run_root / self.run_tag


def points(stage3_run_root: Path) -> list[Point]:
    out: list[Point] = []
    for multiplier in MULTIPLIERS:
        label = f"m{format_token(multiplier)}"
        out.append(
            Point(
                curve_key=f"{VARIANT}_{label}",
                variant=VARIANT,
                curve_label="PL1A refinement sweep",
                weight_label=label,
                weight_value=float(multiplier),
                regularization_multiplier=float(multiplier),
                ptv_isotv_weight=None,
                run_root=stage3_run_root,
                run_tag=f"pl1a_{label}",
                source="P29 Stage 3 PL1A refinement",
            )
        )
    return out


def weight_contract_error(record: Mapping[str, Any], point: Point) -> str | None:
    if not record.get("technical_accepted"):
        return None
    try:
        actual_m = float(record.get("regularization_multiplier"))
    except (TypeError, ValueError):
        return "missing regularization multiplier metadata"
    if not math.isclose(actual_m, point.regularization_multiplier, rel_tol=1e-6, abs_tol=1e-9):
        return f"regularization multiplier {actual_m} != expected {point.regularization_multiplier}"
    return None


def fmt(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{f:.{digits}f}" if math.isfinite(f) else "n/a"


def aggregate(stage3_run_root: Path, review_root: Path, seed: int) -> dict[str, Any]:
    stage3_run_root = stage3_run_root.resolve()
    review_root = review_root.resolve()
    pts = points(stage3_run_root)
    records: list[dict[str, Any]] = []
    for point in pts:
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
    for point in pts:
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

    portable_roots = (
        (stage3_run_root, "${STAGE3_RUN_ROOT}"),
        (review_root, "${REVIEW_ROOT}"),
        (REPOSITORY_ROOT, "${REPO_ROOT}"),
    )
    atomic_csv(
        review_root / "stage3_pl1a_refinement_case_rows.csv",
        portable_review_value(records, portable_roots),
    )
    atomic_json(
        review_root / "stage3_pl1a_refinement_case_rows.json",
        portable_review_value({"schema_version": 1, "records": records}, portable_roots),
    )
    atomic_csv(
        review_root / "stage3_pl1a_refinement_curve_points.csv",
        portable_review_value(point_rows, portable_roots),
    )
    atomic_json(
        review_root / "stage3_pl1a_refinement_curve_points.json",
        portable_review_value({"schema_version": 1, "rows": point_rows}, portable_roots),
    )
    acceptance = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage3_run_root": str(stage3_run_root),
        "review_root": str(review_root),
        "expected_records": len(pts) * len(CASES),
        "accepted_records": sum(bool(row["technical_accepted"]) for row in records),
        "regularity_complete_records": sum(row["status"] == "complete" for row in regularity),
        "all_curve_points_complete": all(row["technical_complete"] for row in point_rows),
        "technical_failures": [row for row in records if not row.get("technical_accepted")],
    }
    atomic_json(
        review_root / "stage3_pl1a_refinement_acceptance.json",
        portable_review_value(acceptance, portable_roots),
    )
    lines = [
        "# Stage 3 PL1A Refinement Report",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Accepted records: `{acceptance['accepted_records']}/{acceptance['expected_records']}`",
        f"- Regularity complete: `{acceptance['regularity_complete_records']}`",
        "",
        "| Weight | Native snap TRE | Reg-grid subvoxel TRE | FG stdLogJ | FG J<=0 pct | Runtime | Complete |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in point_rows:
        lines.append(
            f"| {row['weight_label']} | "
            f"{fmt(row.get('headline_native_snap_tre_mm_mean'))} +/- {fmt(row.get('headline_native_snap_tre_mm_sd'))} | "
            f"{fmt(row.get('registration_grid_subvoxel_tre_mm_mean'))} +/- {fmt(row.get('registration_grid_subvoxel_tre_mm_sd'))} | "
            f"{fmt(row.get('regularity_log_jdet_std_fg_mean'), 4)} +/- {fmt(row.get('regularity_log_jdet_std_fg_sd'), 4)} | "
            f"{fmt(row.get('regularity_jdet_le0_pct_fg_mean'), 5)} +/- {fmt(row.get('regularity_jdet_le0_pct_fg_sd'), 5)} | "
            f"{fmt(row.get('total_wall_seconds_mean'), 1)} +/- {fmt(row.get('total_wall_seconds_sd'), 1)} | "
            f"{row.get('n', 0)}/10 |"
        )
    lines.append("")
    atomic_text(review_root / "stage3_pl1a_refinement_report.md", "\n".join(lines))
    return acceptance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = aggregate(args.stage3_run_root, args.review_root, int(args.seed))
    print(
        f"Accepted {result['accepted_records']}/{result['expected_records']} records; "
        f"regularity complete {result['regularity_complete_records']}; "
        f"review: {args.review_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
