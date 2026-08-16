#!/usr/bin/env python
"""Write a casewise review report for the LungCT cps4 FD-BE alpha sweep."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]

DEFAULT_SOURCE = (
    REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/"
    / "v011_current_tre_stdlogj_fdbe_cps4_alpha_extension/"
    / "fdbe_cps4_alpha_sweep_case_rows.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10/"
    / "casewise_six_fdbe_methods"
)

EXPECTED_METHODS = ("inr_bscp", "d_bscp")
EXPECTED_ALPHAS = (10000.0, 100000.0, 1000000.0)
EXPECTED_CASES = tuple(range(1, 11))
DELTA_BASELINE_ALPHA = 10000.0

METHOD_LABELS = {
    "inr_bscp": "INR-BSCP cps4 (FD-BE)",
    "d_bscp": "D-BSCP cps4 (FD-BE)",
}
METHOD_ORDER = {method: index for index, method in enumerate(EXPECTED_METHODS)}
ALPHA_ORDER = {alpha: index for index, alpha in enumerate(EXPECTED_ALPHAS)}

NUMERIC_COLUMNS = {
    "case_id",
    "alpha",
    "lr",
    "cps",
    "epochs",
    "seed",
    "original_snap_tre_mean_mm",
    "original_snap_tre_std_mm",
    "original_subvoxel_tre_mean_mm",
    "original_subvoxel_tre_std_mm",
    "reg1mm_snap_tre_mean_mm",
    "reg1mm_snap_tre_std_mm",
    "reg1mm_subvoxel_tre_mean_mm",
    "reg1mm_subvoxel_tre_std_mm",
    "fg_logj_std",
    "fg_jdet_le0_pct",
    "fg_bending_energy",
    "fg_diffusion_l2",
    "full_crop_one_minus_ncc",
    "fixed_t00_lung_mask_one_minus_ncc",
    "elapsed_sec",
    "peak_mem_gb",
    "final_data_loss",
    "final_reg_loss",
    "final_loss",
}

LONG_COLUMNS = (
    "case_id",
    "case_key",
    "method_slug",
    "display_method",
    "method_id",
    "alpha",
    "alpha_label",
    "combo_label",
    "lr",
    "cps",
    "epochs",
    "seed",
    "original_snap_tre_mean_mm",
    "fg_logj_std",
    "fg_jdet_le0_pct",
    "fg_bending_energy",
    "fg_diffusion_l2",
    "full_crop_one_minus_ncc",
    "fixed_t00_lung_mask_one_minus_ncc",
    "elapsed_sec",
    "peak_mem_gb",
    "tre_source",
    "source_summary_path",
    "summary_path",
    "masked_loss_used",
    "status",
    "validation_errors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-alphas", default=",".join(f"{alpha:g}" for alpha in EXPECTED_ALPHAS))
    parser.add_argument("--delta-baseline-alpha", type=float, default=DELTA_BASELINE_ALPHA)
    parser.add_argument("--report-filename", default="casewise_six_fdbe_methods_report.md")
    return parser.parse_args()


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def parse_int(value: object) -> int | None:
    parsed = parse_float(value)
    return None if parsed is None else int(round(parsed))


def fmt(value: object, digits: int = 3) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "NA"
    return f"{parsed:.{digits}f}"


def fmt_runtime(value: object) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "NA"
    if parsed >= 600.0:
        return f"{parsed / 60.0:.1f}m"
    return f"{parsed:.0f}s"


def alpha_label(value: object) -> str:
    alpha = parse_float(value)
    if alpha is None:
        return "NA"
    if alpha > 0:
        exponent = int(math.floor(math.log10(alpha)))
        mantissa = alpha / (10.0**exponent)
        for candidate in (1.0, 2.0, 5.0):
            if abs(mantissa - candidate) < 1.0e-9:
                prefix = "" if candidate == 1.0 else f"{candidate:g}"
                return f"{prefix}e{exponent}"
    return f"{alpha:g}"


def alpha_sort(value: object) -> int:
    return ALPHA_ORDER.get(parse_float(value), len(ALPHA_ORDER))


def method_sort(value: object) -> int:
    return METHOD_ORDER.get(str(value), len(METHOD_ORDER))


def combo_label(row: dict[str, object]) -> str:
    method = str(row.get("method_slug", ""))
    return f"{METHOD_LABELS.get(method, method)} alpha={alpha_label(row.get('alpha'))}"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def source_status(row: dict[str, object]) -> str:
    alpha = parse_float(row.get("alpha"))
    if alpha == 10000.0:
        return "legacy alpha=1e4 source"
    status = str(row.get("status", "")).strip()
    return status or "new alpha-extension source"


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for raw in reader:
            row: dict[str, object] = dict(raw)
            for column in NUMERIC_COLUMNS:
                if column in row:
                    parsed = parse_float(row[column])
                    row[column] = "" if parsed is None else parsed
            if "case_id" in row:
                case_id = parse_int(row["case_id"])
                row["case_id"] = "" if case_id is None else case_id
            row["alpha_label"] = alpha_label(row.get("alpha"))
            row["combo_label"] = combo_label(row)
            row["source_status"] = source_status(row)
            rows.append(row)
    return rows


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    fieldnames = tuple(fieldnames)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def clean_long_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {column: row.get(column, "") for column in LONG_COLUMNS}
        for row in sorted(
            rows,
            key=lambda item: (
                int(item.get("case_id") or 0),
                method_sort(item.get("method_slug")),
                alpha_sort(item.get("alpha")),
            ),
        )
    ]


def write_wide_metric(path: Path, rows: list[dict[str, object]], metric: str) -> None:
    labels = [
        f"{METHOD_LABELS[method]} alpha={alpha_label(alpha)}"
        for method in EXPECTED_METHODS
        for alpha in EXPECTED_ALPHAS
    ]
    index: dict[tuple[int, str], object] = {}
    case_keys: dict[int, str] = {}
    for row in rows:
        case_id = int(row.get("case_id") or 0)
        label = combo_label(row)
        case_keys[case_id] = str(row.get("case_key", f"case{case_id:02d}"))
        index[(case_id, label)] = row.get(metric, "")

    fieldnames = ("case_id", "case_key", *labels)
    output_rows: list[dict[str, object]] = []
    for case_id in EXPECTED_CASES:
        out: dict[str, object] = {"case_id": case_id, "case_key": case_keys.get(case_id, f"case{case_id:02d}")}
        for label in labels:
            out[label] = index.get((case_id, label), "")
        output_rows.append(out)
    write_csv(path, fieldnames, output_rows)


def mean(values: Iterable[object]) -> float | None:
    parsed = [value for value in (parse_float(item) for item in values) if value is not None]
    if not parsed:
        return None
    return statistics.fmean(parsed)


def stdev(values: Iterable[object]) -> float | None:
    parsed = [value for value in (parse_float(item) for item in values) if value is not None]
    if len(parsed) < 2:
        return None
    return statistics.stdev(parsed)


def group_rows(rows: Iterable[dict[str, object]], *keys: str) -> dict[tuple[object, ...], list[dict[str, object]]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return groups


def overall_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method in EXPECTED_METHODS:
        for alpha in EXPECTED_ALPHAS:
            group = [
                row
                for row in rows
                if row.get("method_slug") == method and parse_float(row.get("alpha")) == alpha
            ]
            output.append(
                {
                    "method_slug": method,
                    "display_method": METHOD_LABELS[method],
                    "alpha": alpha,
                    "alpha_label": alpha_label(alpha),
                    "cases": len(group),
                    "tre_mean_mm": mean(row.get("original_snap_tre_mean_mm") for row in group),
                    "tre_sd_mm": stdev(row.get("original_snap_tre_mean_mm") for row in group),
                    "sdlogj_mean": mean(row.get("fg_logj_std") for row in group),
                    "sdlogj_sd": stdev(row.get("fg_logj_std") for row in group),
                    "folding_pct_mean": mean(row.get("fg_jdet_le0_pct") for row in group),
                    "full_crop_one_minus_ncc_mean": mean(row.get("full_crop_one_minus_ncc") for row in group),
                    "fixed_t00_lung_mask_one_minus_ncc_mean": mean(
                        row.get("fixed_t00_lung_mask_one_minus_ncc") for row in group
                    ),
                    "runtime_sec_mean": mean(row.get("elapsed_sec") for row in group),
                    "peak_mem_gb_mean": mean(row.get("peak_mem_gb") for row in group),
                }
            )
    return output


def best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    grouped = group_rows(rows, "case_id", "method_slug")
    for key in sorted(grouped, key=lambda item: (int(item[0]), method_sort(item[1]))):
        candidates = [
            row
            for row in grouped[key]
            if parse_float(row.get("original_snap_tre_mean_mm")) is not None
        ]
        if not candidates:
            continue
        best = sorted(
            candidates,
            key=lambda row: (
                parse_float(row.get("original_snap_tre_mean_mm")) or math.inf,
                parse_float(row.get("fg_logj_std")) or math.inf,
                alpha_sort(row.get("alpha")),
            ),
        )[0]
        output.append(
            {
                "case_id": best.get("case_id", ""),
                "case_key": best.get("case_key", ""),
                "method_slug": best.get("method_slug", ""),
                "display_method": best.get("display_method", ""),
                "best_alpha": best.get("alpha", ""),
                "alpha_label": best.get("alpha_label", ""),
                "tre_mean_mm": best.get("original_snap_tre_mean_mm", ""),
                "fg_logj_std": best.get("fg_logj_std", ""),
                "fg_jdet_le0_pct": best.get("fg_jdet_le0_pct", ""),
            }
        )
    return output


def delta_rows(rows: list[dict[str, object]], baseline_alpha: float) -> list[dict[str, object]]:
    grouped = group_rows(rows, "case_id", "method_slug")
    output: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: (int(item[0]), method_sort(item[1]))):
        group = grouped[key]
        baseline = next((row for row in group if parse_float(row.get("alpha")) == float(baseline_alpha)), None)
        if baseline is None:
            continue
        base_tre = parse_float(baseline.get("original_snap_tre_mean_mm"))
        base_logj = parse_float(baseline.get("fg_logj_std"))
        base_fold = parse_float(baseline.get("fg_jdet_le0_pct"))
        for row in sorted(group, key=lambda item: alpha_sort(item.get("alpha"))):
            alpha = parse_float(row.get("alpha"))
            if alpha == float(baseline_alpha):
                continue
            tre = parse_float(row.get("original_snap_tre_mean_mm"))
            logj = parse_float(row.get("fg_logj_std"))
            fold = parse_float(row.get("fg_jdet_le0_pct"))
            output.append(
                {
                    "case_id": row.get("case_id", ""),
                    "case_key": row.get("case_key", ""),
                    "method_slug": row.get("method_slug", ""),
                    "display_method": row.get("display_method", ""),
                    "alpha": row.get("alpha", ""),
                    "alpha_label": row.get("alpha_label", ""),
                    "baseline_alpha": baseline_alpha,
                    "baseline_alpha_label": alpha_label(baseline_alpha),
                    "tre_mean_mm": row.get("original_snap_tre_mean_mm", ""),
                    "baseline_tre_mean_mm": baseline.get("original_snap_tre_mean_mm", ""),
                    "delta_tre_mm_vs_baseline": "" if tre is None or base_tre is None else tre - base_tre,
                    "fg_logj_std": row.get("fg_logj_std", ""),
                    "baseline_fg_logj_std": baseline.get("fg_logj_std", ""),
                    "delta_fg_logj_std_vs_baseline": "" if logj is None or base_logj is None else logj - base_logj,
                    "fg_jdet_le0_pct": row.get("fg_jdet_le0_pct", ""),
                    "baseline_fg_jdet_le0_pct": baseline.get("fg_jdet_le0_pct", ""),
                    "delta_folding_pct_vs_baseline": "" if fold is None or base_fold is None else fold - base_fold,
                }
            )
    return output


def validation_notes(rows: list[dict[str, object]], source: Path) -> list[str]:
    notes: list[str] = []
    expected_count = len(EXPECTED_CASES) * len(EXPECTED_METHODS) * len(EXPECTED_ALPHAS)
    if len(rows) == expected_count:
        notes.append(f"PASS: source has {expected_count} rows.")
    else:
        notes.append(f"FAIL: source has {len(rows)} rows; expected {expected_count}.")

    for method in EXPECTED_METHODS:
        for alpha in EXPECTED_ALPHAS:
            count = sum(
                1
                for row in rows
                if row.get("method_slug") == method and parse_float(row.get("alpha")) == alpha
            )
            status = "PASS" if count == len(EXPECTED_CASES) else "FAIL"
            notes.append(f"{status}: {method} alpha={alpha_label(alpha)} has {count}/10 cases.")

    masked_rows = [
        row
        for row in rows
        if str(row.get("masked_loss_used", "")).strip().lower() not in {"false", "0", ""}
        or "masked" in str(row.get("method_id", "")).lower()
        or "masked" in str(row.get("source_summary_path", "")).lower()
        or "masked" in str(row.get("summary_path", "")).lower()
    ]
    if not masked_rows:
        notes.append("PASS: no masked-loss method IDs or masked source paths detected.")
    else:
        notes.append(f"FAIL: {len(masked_rows)} rows look masked-loss related.")

    missing_main = [
        row
        for row in rows
        if parse_float(row.get("original_snap_tre_mean_mm")) is None
        or parse_float(row.get("fg_logj_std")) is None
    ]
    if not missing_main:
        notes.append("PASS: all rows have finite original_snap TRE and foreground SDlogJ.")
    else:
        notes.append(f"FAIL: {len(missing_main)} rows are missing TRE or SDlogJ.")

    missing_diagnostic = [
        row
        for row in rows
        if parse_float(row.get("alpha")) != 10000.0
        and (
            parse_float(row.get("full_crop_one_minus_ncc")) is None
            or parse_float(row.get("fixed_t00_lung_mask_one_minus_ncc")) is None
        )
    ]
    if not missing_diagnostic:
        notes.append("PASS: new alpha=1e5/1e6 rows include full-crop and fixed-mask 1-NCC diagnostics.")
    else:
        notes.append(f"FAIL: {len(missing_diagnostic)} new rows are missing NCC diagnostics.")

    notes.append(f"Source: `{display_path(source)}`")
    return notes


def markdown_table(headers: tuple[str, ...], rows: Iterable[Iterable[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, rows: list[dict[str, object]], source: Path, baseline_alpha: float) -> None:
    overall = overall_rows(rows)
    best = best_rows(rows)
    deltas = delta_rows(rows, baseline_alpha)

    lines: list[str] = []
    lines.append("# LungCT FD-BE cps4 alpha sweep: casewise report")
    lines.append("")
    lines.append(f"- Updated: `{utc_text()}`")
    lines.append("- Scope: INR-BSCP cps4 and D-BSCP cps4, unmasked full-crop global NCC + dense FD-BE.")
    lines.append("- Cases: DIR-Lab 4DCT cases 01-10, pTVReg iso1mm landmark-crop protocol.")
    lines.append("- Headline metric: original/native snap-to-voxel TRE in mm; regularity uses foreground SDlogJ.")
    lines.append("- Fixed T00 lung-mask NCC is a diagnostic/evaluation field only, not the optimization loss domain.")
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    for note in validation_notes(rows, source):
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Overall Means")
    lines.append("")
    lines.append(
        markdown_table(
            (
                "Method",
                "alpha",
                "Cases",
                "TRE mean",
                "TRE SD",
                "SDlogJ mean",
                "Folding %",
                "Full 1-NCC",
                "Mask 1-NCC",
                "Runtime",
                "Peak GB",
            ),
            (
                (
                    row["display_method"],
                    row["alpha_label"],
                    row["cases"],
                    fmt(row["tre_mean_mm"], 3),
                    fmt(row["tre_sd_mm"], 3),
                    fmt(row["sdlogj_mean"], 4),
                    fmt(row["folding_pct_mean"], 4),
                    fmt(row["full_crop_one_minus_ncc_mean"], 5),
                    fmt(row["fixed_t00_lung_mask_one_minus_ncc_mean"], 5),
                    fmt_runtime(row["runtime_sec_mean"]),
                    fmt(row["peak_mem_gb_mean"], 2),
                )
                for row in overall
            ),
        )
    )
    lines.append("")
    lines.append("## Best Alpha By Case")
    lines.append("")
    lines.append(
        markdown_table(
            ("Case", "Method", "Best alpha", "TRE", "SDlogJ", "Folding %"),
            (
                (
                    f"{int(row['case_id']):02d}",
                    row["display_method"],
                    row["alpha_label"],
                    fmt(row["tre_mean_mm"], 3),
                    fmt(row["fg_logj_std"], 4),
                    fmt(row["fg_jdet_le0_pct"], 4),
                )
                for row in best
            ),
        )
    )
    lines.append("")
    lines.append(f"## Delta Versus alpha={alpha_label(baseline_alpha)}")
    lines.append("")
    lines.append(
        markdown_table(
            ("Case", "Method", "alpha", "Delta TRE", "Delta SDlogJ", "Delta folding %"),
            (
                (
                    f"{int(row['case_id']):02d}",
                    row["display_method"],
                    row["alpha_label"],
                    fmt(row["delta_tre_mm_vs_baseline"], 3),
                    fmt(row["delta_fg_logj_std_vs_baseline"], 4),
                    fmt(row["delta_folding_pct_vs_baseline"], 4),
                )
                for row in deltas
            ),
        )
    )
    lines.append("")
    lines.append("## Case Tables")
    for case_id in EXPECTED_CASES:
        case_rows = [
            row
            for row in rows
            if int(row.get("case_id") or 0) == case_id
        ]
        case_rows = sorted(case_rows, key=lambda row: (method_sort(row.get("method_slug")), alpha_sort(row.get("alpha"))))
        lines.append("")
        lines.append(f"### Case {case_id:02d}")
        lines.append("")
        lines.append(
            markdown_table(
                (
                    "Method",
                    "alpha",
                    "TRE",
                    "SDlogJ",
                    "Folding %",
                    "BE",
                    "Diff L2",
                    "Full 1-NCC",
                    "Mask 1-NCC",
                    "Runtime",
                    "Peak GB",
                    "Source",
                ),
                (
                    (
                        row.get("display_method", ""),
                        row.get("alpha_label", ""),
                        fmt(row.get("original_snap_tre_mean_mm"), 3),
                        fmt(row.get("fg_logj_std"), 4),
                        fmt(row.get("fg_jdet_le0_pct"), 4),
                        fmt(row.get("fg_bending_energy"), 4),
                        fmt(row.get("fg_diffusion_l2"), 4),
                        fmt(row.get("full_crop_one_minus_ncc"), 5),
                        fmt(row.get("fixed_t00_lung_mask_one_minus_ncc"), 5),
                        fmt_runtime(row.get("elapsed_sec")),
                        fmt(row.get("peak_mem_gb"), 2),
                        row.get("source_status", ""),
                    )
                    for row in case_rows
                ),
            )
        )

    lines.append("")
    lines.append("## Companion CSVs")
    lines.append("")
    lines.append("- `casewise_long.csv`: cleaned source rows for the six method/alpha combinations.")
    lines.append("- `casewise_tre_wide.csv`: case-by-case original_snap TRE matrix.")
    lines.append("- `casewise_stdlogj_wide.csv`: case-by-case foreground SDlogJ matrix.")
    lines.append("- `casewise_best_by_case_method.csv`: lowest-TRE alpha per case and method.")
    lines.append(f"- `casewise_delta_vs_alpha{int(float(baseline_alpha))}.csv`: per-case deltas versus alpha={alpha_label(baseline_alpha)}.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    global ALPHA_ORDER
    global DELTA_BASELINE_ALPHA
    args = parse_args()
    expected = tuple(
        float(token.strip())
        for token in str(args.expected_alphas).split(",")
        if token.strip()
    )
    if expected:
        globals()["EXPECTED_ALPHAS"] = expected
        ALPHA_ORDER = {alpha: index for index, alpha in enumerate(EXPECTED_ALPHAS)}
    DELTA_BASELINE_ALPHA = float(args.delta_baseline_alpha)
    rows = load_rows(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_rows = clean_long_rows(rows)
    overall = overall_rows(rows)
    best = best_rows(rows)
    deltas = delta_rows(rows, DELTA_BASELINE_ALPHA)

    write_csv(args.output_dir / "casewise_long.csv", LONG_COLUMNS, clean_rows)
    write_csv(
        args.output_dir / "overall_means.csv",
        (
            "method_slug",
            "display_method",
            "alpha",
            "alpha_label",
            "cases",
            "tre_mean_mm",
            "tre_sd_mm",
            "sdlogj_mean",
            "sdlogj_sd",
            "folding_pct_mean",
            "full_crop_one_minus_ncc_mean",
            "fixed_t00_lung_mask_one_minus_ncc_mean",
            "runtime_sec_mean",
            "peak_mem_gb_mean",
        ),
        overall,
    )
    write_wide_metric(args.output_dir / "casewise_tre_wide.csv", rows, "original_snap_tre_mean_mm")
    write_wide_metric(args.output_dir / "casewise_stdlogj_wide.csv", rows, "fg_logj_std")
    write_csv(
        args.output_dir / "casewise_best_by_case_method.csv",
        (
            "case_id",
            "case_key",
            "method_slug",
            "display_method",
            "best_alpha",
            "alpha_label",
            "tre_mean_mm",
            "fg_logj_std",
            "fg_jdet_le0_pct",
        ),
        best,
    )
    write_csv(
        args.output_dir / f"casewise_delta_vs_alpha{int(DELTA_BASELINE_ALPHA)}.csv",
        (
            "case_id",
            "case_key",
            "method_slug",
            "display_method",
            "alpha",
            "alpha_label",
            "baseline_alpha",
            "baseline_alpha_label",
            "tre_mean_mm",
            "baseline_tre_mean_mm",
            "delta_tre_mm_vs_baseline",
            "fg_logj_std",
            "baseline_fg_logj_std",
            "delta_fg_logj_std_vs_baseline",
            "fg_jdet_le0_pct",
            "baseline_fg_jdet_le0_pct",
            "delta_folding_pct_vs_baseline",
        ),
        deltas,
    )
    write_report(args.output_dir / str(args.report_filename), rows, args.source, DELTA_BASELINE_ALPHA)

    print(f"Wrote casewise report package to {args.output_dir}")


if __name__ == "__main__":
    main()
