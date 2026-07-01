#!/usr/bin/env python
"""Build automatic paper-facing result selection tables."""

from __future__ import annotations

import csv
import json
import math
import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SPEC_PATH = HERE / "table_specs.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.paper_artifacts.paper_artifact_common import (  # noqa: E402
    canonical_lr,
    display_name as artifact_display_name,
    format_lr,
    load_nicknames,
    load_standard,
    method_family,
    parse_effective_lr,
    sanitize_public_records,
    selected_hypervariant_records,
    validate_lr_records,
    validation_report_markdown,
)


NO_SWEEP_COLUMN_MAP = {
    "native_crop_snap_tre_mean_mm": "original_snap_tre_mean_mm",
    "native_crop_sub_tre_mean_mm": "original_subvoxel_tre_mean_mm",
    "reg1mm_snap_tre_mean_mm": "reg1mm_snap_tre_mean_mm",
    "reg1mm_sub_tre_mean_mm": "reg1mm_subvoxel_tre_mean_mm",
    "fg_log_jdet_std": "fg_logj_std",
    "fg_jdet_le0_pct": "fg_jdet_le0_pct",
    "fg_bending_energy": "fg_bending_energy",
    "fg_diffusion_l2": "fg_diffusion_l2"
}


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float):
        return value
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def is_finite(value: Any) -> bool:
    return math.isfinite(safe_float(value))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def population_mean_std(values: Iterable[Any]) -> Tuple[float, float]:
    arr = [safe_float(value) for value in values]
    arr = [value for value in arr if math.isfinite(value)]
    if not arr:
        return float("nan"), float("nan")
    mean = sum(arr) / len(arr)
    var = sum((value - mean) ** 2 for value in arr) / len(arr)
    return mean, math.sqrt(var)


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _standard_selected_record(
    standard: Mapping[str, Any] | None,
    artifact: str,
    dataset_key: str,
    method_id: str,
) -> Dict[str, Any]:
    if standard is None:
        return {}
    for record in selected_hypervariant_records(artifact, standard):
        if str(record.get("dataset")) == str(dataset_key) and str(record.get("method_id")) == str(method_id):
            return dict(record)
    return {}


def _selected_lr(
    row: Mapping[str, Any],
    standard: Mapping[str, Any] | None,
    dataset_key: str,
    method_id: str,
) -> float:
    parsed = parse_effective_lr(row)
    if parsed is not None:
        return parsed
    standard_record = _standard_selected_record(standard, "best_result_table", dataset_key, method_id)
    return safe_float(standard_record.get("lr"))


def _write_validation_outputs(
    output_dir: Path,
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    stem: str = "standard_validation",
) -> None:
    fieldnames = [
        "artifact",
        "dataset",
        "method_id",
        "method_family",
        "observed_lr",
        "observed_lr_label",
        "canonical_lr",
        "canonical_lr_label",
        "status",
        "selected_setting",
        "source",
        "note",
    ]
    validation_rows = sanitize_public_records(validation_rows)
    write_csv(output_dir / f"{stem}.csv", validation_rows, fieldnames)
    (output_dir / f"{stem}_report.md").write_text(
        validation_report_markdown("Best Result Table LR Validation", validation_rows)
    )


def transformed_sort_value(row: Mapping[str, Any], criterion: Mapping[str, Any]) -> float:
    value = safe_float(row.get(str(criterion["metric"])))
    if not math.isfinite(value):
        return math.inf
    mode = str(criterion.get("mode", "min"))
    if mode == "max":
        return -value
    if mode == "min":
        return value
    raise ValueError(f"Unknown selection mode: {mode}")


def selection_key(row: Mapping[str, Any], selection: Mapping[str, Any]) -> Tuple[float, ...]:
    criteria = [
        {"metric": selection["metric"], "mode": selection.get("mode", "min")},
        *selection.get("tie_breakers", [])
    ]
    return tuple(transformed_sort_value(row, criterion) for criterion in criteria)


def format_fixed(value: Any, digits: int) -> str:
    f = safe_float(value)
    if not math.isfinite(f):
        return ""
    return f"{f:.{digits}f}"


def format_adaptive_percent(value: Any) -> str:
    f = safe_float(value)
    if not math.isfinite(f):
        return ""
    magnitude = abs(f)
    if magnitude == 0:
        return "0.00"
    if magnitude < 1e-4:
        return f"{f:.2e}"
    if magnitude < 0.1:
        return f"{f:.4f}"
    return f"{f:.2f}"


def format_adaptive_small(value: Any) -> str:
    f = safe_float(value)
    if not math.isfinite(f):
        return ""
    magnitude = abs(f)
    if magnitude == 0:
        return "0"
    if magnitude < 1e-5:
        return f"{f:.2e}"
    if magnitude < 0.01:
        return f"{f:.6f}"
    if magnitude < 1:
        return f"{f:.4f}"
    return f"{f:.2f}"


def format_value(value: Any, metric_spec: Mapping[str, Any]) -> str:
    fmt = str(metric_spec.get("format", "fixed"))
    if fmt == "fixed":
        return format_fixed(value, int(metric_spec.get("digits", 3)))
    if fmt == "adaptive_percent":
        return format_adaptive_percent(value)
    if fmt == "adaptive_small":
        return format_adaptive_small(value)
    raise ValueError(f"Unknown metric format: {fmt}")


def metric_columns(metric_spec: Mapping[str, Any], stat: str) -> List[str]:
    direct_key = metric_spec.get(stat)
    if direct_key:
        return [str(direct_key)]
    candidates_key = f"{stat}_candidates"
    return [str(value) for value in metric_spec.get(candidates_key, [])]


def select_metric_value(
    row: Mapping[str, Any],
    metric_spec: Mapping[str, Any],
    stat: str,
) -> Tuple[float, str, Dict[str, str] | None]:
    columns = metric_columns(metric_spec, stat)
    for idx, column in enumerate(columns):
        value = safe_float(row.get(column))
        if math.isfinite(value):
            fallback = None
            if idx > 0:
                fallback = {
                    "metric": str(metric_spec["key"]),
                    "stat": stat,
                    "selected_column": column,
                    "preferred_column": columns[0],
                    "note": str(metric_spec.get("fallback_note", "Used fallback metric column."))
                }
            return value, column, fallback
    return float("nan"), columns[0] if columns else "", None


def format_mean_std(
    row: Mapping[str, Any],
    metric_spec: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    mean, mean_column, mean_fallback = select_metric_value(row, metric_spec, "mean")
    std, std_column, std_fallback = select_metric_value(row, metric_spec, "std")
    display = ""
    if math.isfinite(mean) and math.isfinite(std):
        display = f"{format_value(mean, metric_spec)} +/- {format_value(std, metric_spec)}"
    raw = {
        "mean": mean,
        "std": std,
        "mean_column": mean_column,
        "std_column": std_column
    }
    fallbacks = [item for item in [mean_fallback, std_fallback] if item is not None]
    return display, raw, fallbacks


def markdown_table(headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |"
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def selected_setting(row: Mapping[str, Any], dataset_key: str) -> str:
    if dataset_key == "oasis20":
        alpha = safe_float(row.get("alpha"))
        if math.isfinite(alpha):
            return f"alpha={alpha:g}"
        return "reference"
    reg_label = str(row.get("reg_label", "")).strip()
    if reg_label:
        return reg_label
    reg_value = safe_float(row.get("reg_value"))
    if math.isfinite(reg_value):
        return f"reg={reg_value:g}"
    return "reference"


def load_oasis_rows(spec: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    source = repo_path(str(spec["source"]))
    rows = read_csv(source)
    for row in rows:
        row["_source_name"] = "oasis20_population"
        row["_source_path"] = str(source.relative_to(REPO_ROOT))
    return rows, [{"name": "oasis20_population", "path": str(source.relative_to(REPO_ROOT)), "rows": len(rows)}]


def aggregate_no_sweep_rows(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    label_by_method: Mapping[str, str],
    source_name: str,
    source_path: Path,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", ""))
        if method in methods:
            grouped[method].append(row)

    out: List[Dict[str, Any]] = []
    for method in methods:
        group = grouped.get(method, [])
        row_out: Dict[str, Any] = {
            "method": method,
            "method_label": label_by_method.get(method, method),
            "reg_value": "",
            "reg_label": "default",
            "ratio_to_best_tre_weight": "",
            "cases": len({str(row.get("case_id")) for row in group if str(row.get("case_id", "")).strip()}),
            "_source_name": source_name,
            "_source_path": str(source_path.relative_to(REPO_ROOT))
        }
        for source_column, output_prefix in NO_SWEEP_COLUMN_MAP.items():
            mean, std = population_mean_std(row.get(source_column) for row in group)
            row_out[f"{output_prefix}_mean"] = mean
            row_out[f"{output_prefix}_std"] = std
        out.append(row_out)
    return out


def load_lungct_rows(spec: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    label_by_method = {str(item["id"]): str(item["label"]) for item in spec["methods"]}
    all_rows: List[Dict[str, Any]] = []
    source_summaries: List[Dict[str, Any]] = []

    for source_spec in spec["sources"]:
        source_path = repo_path(str(source_spec["path"]))
        rows = read_csv(source_path)
        methods = [str(method) for method in source_spec.get("methods", [])]
        source_summaries.append(
            {
                "name": source_spec["name"],
                "type": source_spec["type"],
                "path": str(source_path.relative_to(REPO_ROOT)),
                "rows": len(rows),
                "methods": methods
            }
        )
        if source_spec["type"] == "mean_rows":
            for row in rows:
                if str(row.get("method")) in methods:
                    row["_source_name"] = str(source_spec["name"])
                    row["_source_path"] = str(source_path.relative_to(REPO_ROOT))
                    all_rows.append(row)
        elif source_spec["type"] == "no_sweep_case_rows":
            all_rows.extend(
                aggregate_no_sweep_rows(
                    rows,
                    methods,
                    label_by_method,
                    str(source_spec["name"]),
                    source_path,
                )
            )
        else:
            raise ValueError(f"Unknown LungCT source type: {source_spec['type']}")

    return all_rows, source_summaries


def method_id_column(dataset_key: str) -> str:
    return "method_key" if dataset_key == "oasis20" else "method"


def select_rows(
    dataset_key: str,
    spec: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    nicknames: Mapping[str, Mapping[str, str]] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    id_column = method_id_column(dataset_key)
    selected: List[Dict[str, Any]] = []
    pool_summaries: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for method_spec in spec["methods"]:
        method_id = str(method_spec["id"])
        pool = [dict(row) for row in candidates if str(row.get(id_column)) == method_id]
        pool_summaries.append({"method": method_id, "candidate_rows": len(pool)})
        if not pool:
            warnings.append(f"No candidate rows found for {method_id}.")
            continue
        pool.sort(key=lambda row: selection_key(row, spec["selection"]))
        row = pool[0]
        row["_display_label"] = artifact_display_name(method_id, str(method_spec["label"]), nicknames)
        row["_original_label"] = str(method_spec["label"])
        row["_selected_setting"] = selected_setting(row, dataset_key)
        row["_selection_pool_size"] = len(pool)
        selected.append(row)

    return selected, pool_summaries, warnings


def build_dataset(
    dataset_key: str,
    spec: Mapping[str, Any],
    *,
    output_dir: Path = HERE,
    nicknames: Mapping[str, Mapping[str, str]] | None = None,
    standard: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if dataset_key == "oasis20":
        candidates, sources = load_oasis_rows(spec)
    elif dataset_key == "lungct10":
        candidates, sources = load_lungct_rows(spec)
    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    selected, pools, warnings = select_rows(dataset_key, spec, candidates, nicknames)
    table_rows: List[Dict[str, Any]] = []
    selected_json_rows: List[Dict[str, Any]] = []
    validation_input_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for row in selected:
        display_row: Dict[str, Any] = {
            "Method": row["_display_label"],
            "Selected": row["_selected_setting"],
            "Cases": str(int(safe_float(row.get(str(spec["cases_column"])))))
        }
        raw_metrics: Dict[str, Any] = {}
        fallbacks: List[Dict[str, str]] = []

        for metric_spec in spec["metrics"]:
            display, raw, metric_fallbacks = format_mean_std(row, metric_spec)
            display_row[str(metric_spec["label"])] = display
            raw_metrics[str(metric_spec["key"])] = raw
            fallbacks.extend(metric_fallbacks)
            if display == "":
                errors.append(f"{dataset_key}: blank display metric {metric_spec['key']} for {row['_display_label']}")

        method_id = str(row.get(method_id_column(dataset_key)))
        selected_lr = _selected_lr(row, standard, dataset_key, method_id)
        method_family_name = method_family(method_id, standard) if standard is not None else ""
        canonical_method_lr = canonical_lr(method_id, standard) if standard is not None else None

        table_rows.append(display_row)
        validation_input_rows.append(
            {
                "dataset": dataset_key,
                "method_id": method_id,
                "lr": selected_lr,
                "selected_setting": row["_selected_setting"],
                "source": row.get("_source_path", ""),
            }
        )
        selected_json_rows.append(
            {
                "method": method_id,
                "label": row["_display_label"],
                "original_label": row.get("_original_label", row["_display_label"]),
                "selected_setting": row["_selected_setting"],
                "selection_pool_size": row["_selection_pool_size"],
                "source_name": row.get("_source_name"),
                "source_path": row.get("_source_path"),
                "cases": int(safe_float(row.get(str(spec["cases_column"])))),
                "method_family": method_family_name,
                "lr": selected_lr,
                "lr_label": format_lr(selected_lr),
                "canonical_lr": canonical_method_lr,
                "canonical_lr_label": format_lr(canonical_method_lr),
                "selection_metric": spec["selection"]["metric"],
                "selection_metric_value": safe_float(row.get(str(spec["selection"]["metric"]))),
                "raw_metrics": raw_metrics,
                "fallbacks": fallbacks,
                "raw_row": row
            }
        )

    if len(table_rows) != int(spec["expected_rows"]):
        errors.append(f"{dataset_key}: expected {spec['expected_rows']} output rows, found {len(table_rows)}")
    for row in selected_json_rows:
        if int(row["cases"]) != int(spec["expected_cases"]):
            errors.append(f"{dataset_key}: {row['label']} cases {row['cases']} != {spec['expected_cases']}")

    headers = ["Method", "Selected", "Cases", *[str(metric["label"]) for metric in spec["metrics"]]]
    stem = dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f"{stem}_summary.csv", table_rows, headers)
    (output_dir / f"{stem}_summary.md").write_text(markdown_table(headers, table_rows) + "\n")
    selected_json_rows = sanitize_public_records(selected_json_rows)
    (output_dir / f"{stem}_selected_rows.json").write_text(json.dumps(selected_json_rows, indent=2, sort_keys=True) + "\n")

    validation_rows = (
        validate_lr_records(validation_input_rows, artifact="best_result_table", standard=standard)
        if standard is not None
        else []
    )

    result = {
        "dataset": dataset_key,
        "title": spec["title"],
        "sources": sources,
        "candidate_rows_total": len(candidates),
        "selection": spec["selection"],
        "pool_summaries": pools,
        "output_rows": len(table_rows),
        "expected_rows": spec["expected_rows"],
        "expected_cases": spec["expected_cases"],
        "warnings": warnings,
        "errors": errors,
        "table_headers": headers,
        "table_rows": table_rows,
        "selected_rows": selected_json_rows,
        "validation_rows": validation_rows,
    }
    if dataset_key == "lungct10" and "failure_analysis" in spec:
        result["failure_analysis"] = build_failure_analysis(spec["failure_analysis"], nicknames=nicknames)
    return result


def build_failure_analysis(
    spec: Mapping[str, Any],
    *,
    nicknames: Mapping[str, Mapping[str, str]] | None = None,
) -> Dict[str, Any]:
    source = repo_path(str(spec["source"]))
    rows = read_csv(source)
    metric = str(spec["metric"])
    threshold = safe_float(spec["threshold_mm"])
    method_specs = spec.get("methods", [])
    out_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for method_spec in method_specs:
        method_id = str(method_spec["id"])
        label = artifact_display_name(method_id, str(method_spec["label"]), nicknames)
        method_rows = [row for row in rows if str(row.get("method")) == method_id]
        failures = [
            row
            for row in method_rows
            if math.isfinite(safe_float(row.get(metric))) and safe_float(row.get(metric)) > threshold
        ]
        failures.sort(key=lambda row: int(safe_float(row.get("case_id"))))
        if not method_rows:
            warnings.append(f"No case rows found for failure-analysis method {method_id}.")
        worst = max(failures, key=lambda row: safe_float(row.get(metric))) if failures else None
        case_values = [
            f"{int(safe_float(row.get('case_id'))):02d}: {safe_float(row.get(metric)):.3f}"
            for row in failures
        ]
        out_rows.append(
            {
                "Method": label,
                "Failed cases": ", ".join(case_values) if case_values else "none",
                "Fail count": f"{len(failures)}/{len(method_rows)}",
                "Worst case": (
                    f"{int(safe_float(worst.get('case_id'))):02d}: {safe_float(worst.get(metric)):.3f}"
                    if worst is not None
                    else "none"
                )
            }
        )

    headers = ["Method", "Failed cases", "Fail count", "Worst case"]
    return {
        "title": spec.get("title", "Failure analysis"),
        "source": str(source.relative_to(REPO_ROOT)),
        "metric": metric,
        "metric_label": spec.get("metric_label", metric),
        "threshold_mm": threshold,
        "warnings": warnings,
        "headers": headers,
        "rows": out_rows
    }


def write_review_report(results: Sequence[Mapping[str, Any]], *, output_dir: Path = HERE) -> None:
    lines: List[str] = [
        "# Automatic Result Selection Tables",
        "",
        f"- Generated UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "- OASIS20 selection maximizes mean Dice and reports first20 mean +/- standard deviation.",
        "- LungCT10 selection minimizes original/native-crop snap TRE and keeps original subvoxel TRE as the compact diagnostic.",
        "- LungCT pTVReg is a protocol reference row, not a clean method-only comparison.",
        "- OASIS foreground regularity uses fixed raw image-positive foreground; LungCT foreground regularity uses fixed T00 lung mask on the pTV 1mm crop.",
        "",
    ]
    for result in results:
        title = str(result["title"])
        dataset = str(result["dataset"])
        lines.extend([f"## {title}", ""])
        markdown_path = output_dir / f"{dataset}_summary.md"
        lines.append(markdown_path.read_text().strip())
        lines.append("")
        failure_analysis = result.get("failure_analysis")
        if isinstance(failure_analysis, dict):
            lines.extend(
                [
                    f"## {failure_analysis['title']}",
                    "",
                    (
                        f"Failure is defined as `{failure_analysis['metric_label']}` "
                        f"> `{failure_analysis['threshold_mm']:.1f}` mm, using the same "
                        "original/native-crop snap TRE convention as the LungCT headline table."
                    ),
                    "",
                    markdown_table(failure_analysis["headers"], failure_analysis["rows"]),
                    "",
                ]
            )

    lines.extend(
        [
            "## Generated Files",
            "",
            "- `oasis20_summary.md`, `oasis20_summary.csv`, `oasis20_selected_rows.json`",
            "- `lungct10_summary.md`, `lungct10_summary.csv`, `lungct10_selected_rows.json`",
            "- `selection_manifest.json`",
            "",
        ]
    )
    (output_dir / "review_report.md").write_text("\n".join(lines).rstrip() + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-path", type=Path, default=SPEC_PATH)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--standard-path", type=Path, default=None)
    parser.add_argument("--nickname-path", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    spec_path = repo_path(str(args.spec_path))
    output_dir = repo_path(str(args.output_dir))
    standard_path = repo_path(str(args.standard_path)) if args.standard_path is not None else None
    nickname_path = repo_path(str(args.nickname_path)) if args.nickname_path is not None else None
    standard = load_standard(standard_path) if standard_path is not None else None
    nicknames = load_nicknames(nickname_path) if nickname_path is not None else None

    spec = json.loads(spec_path.read_text())
    results = [
        build_dataset(
            dataset_key,
            spec[dataset_key],
            output_dir=output_dir,
            nicknames=nicknames,
            standard=standard,
        )
        for dataset_key in ["oasis20", "lungct10"]
    ]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec_path": str(spec_path.relative_to(REPO_ROOT)),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "standard_path": str(standard_path.relative_to(REPO_ROOT)) if standard_path else "",
        "nickname_path": str(nickname_path.relative_to(REPO_ROOT)) if nickname_path else "",
        "display_name_mapping": {
            str(result["dataset"]): [
                {
                    "method_id": row["method"],
                    "original_name": row["original_label"],
                    "display_name": row["label"],
                    "method_family": row["method_family"],
                }
                for row in result["selected_rows"]
            ]
            for result in results
        },
        "formatting": {
            "mean_std_separator": "+/-",
            "adaptive_percent": "2 decimals for ordinary percentages, 4 decimals near zero, scientific for tiny nonzero values",
            "adaptive_small": "preserves small regularity values instead of rounding them to zero"
        },
        "datasets": {
            str(result["dataset"]): {
                key: value
                for key, value in result.items()
                if key not in {"table_rows", "selected_rows"}
            }
            for result in results
        }
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_review_report(results, output_dir=output_dir)

    validation_rows = [
        row
        for result in results
        for row in result.get("validation_rows", [])
    ]
    if validation_rows:
        _write_validation_outputs(output_dir, validation_rows)

    errors = [error for result in results for error in result["errors"]]
    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors))

    for result in results:
        print(f"{result['dataset']}: wrote {result['output_rows']} rows")


if __name__ == "__main__":
    main()
