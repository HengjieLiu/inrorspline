#!/usr/bin/env python
"""Build publication-formatted best-result tables from selected-row artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_SOURCE_DIR = HERE / "v002_candidate"
DEFAULT_OUTPUT_DIR = HERE / "v002_publication"


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def format_fixed(value: Any, digits: int) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def format_adaptive_percent(value: Any) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return ""
    magnitude = abs(number)
    if magnitude == 0:
        return "0.00"
    if magnitude < 1e-4:
        return f"{number:.2e}"
    if magnitude < 0.1:
        return f"{number:.4f}"
    return f"{number:.2f}"


def format_metric(value: Any, metric_spec: Mapping[str, Any]) -> str:
    fmt = str(metric_spec.get("format", "fixed"))
    if fmt == "fixed":
        return format_fixed(value, int(metric_spec.get("digits", 3)))
    if fmt == "adaptive_percent":
        return format_adaptive_percent(value)
    raise ValueError(f"Unknown metric format: {fmt}")


def format_mean_std(row: Mapping[str, Any], metric_spec: Mapping[str, Any]) -> str:
    raw_metric = row["raw_metrics"][str(metric_spec["key"])]
    mean = format_metric(raw_metric["mean"], metric_spec)
    std = format_metric(raw_metric["std"], metric_spec)
    if not mean or not std:
        return ""
    return f"{mean} ({std})"


def read_selected_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(path.read_text())
    return {str(row["method"]): row for row in rows}


def write_csv(path: Path, rows: Sequence[Mapping[str, str]], headers: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def markdown_table(headers: Sequence[str], rows: Sequence[Mapping[str, str]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_header(label: str) -> str:
    if label == "J<=0%":
        return r"$J\leq 0$ (\%)"
    return latex_escape(label)


def latex_table(
    headers: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    *,
    caption: str,
    label: str,
) -> str:
    alignment = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_header(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(header, "")) for header in headers) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


OASIS_METRICS = [
    {"key": "dice", "label": "Dice", "format": "fixed", "digits": 3},
    {"key": "hd95", "label": "HD95", "format": "fixed", "digits": 2},
    {"key": "fg_logj", "label": "SDLogJ", "format": "fixed", "digits": 2},
    {"key": "fg_jdet_le0_pct", "label": "J<=0%", "format": "adaptive_percent"},
]

LUNGCT_METRICS = [
    {"key": "original_snap_tre", "label": "TRE (mm)", "format": "fixed", "digits": 3},
    {"key": "fg_logj", "label": "SDLogJ", "format": "fixed", "digits": 3},
    {"key": "fg_jdet_le0_pct", "label": "J<=0%", "format": "adaptive_percent"},
]

TABLE_SPECS = [
    {
        "stem": "oasis20_publication_main",
        "title": "OASIS seed-3 first20 compact publication table",
        "caption": "OASIS brain registration results on the seed-3 first20 subset.",
        "latex_label": "tab:oasis20-publication-main",
        "dataset": "oasis20",
        "source_json": "oasis20_selected_rows.json",
        "metrics": OASIS_METRICS,
        "rows": [
            ("idir_original_dense_siren_be_autograd", "INR-Dense"),
            ("sinr_cps2_be_bspline_analytic", "INR-BSCP (cps2)"),
            ("direct_cp_cps2_be_bspline_analytic", "D-BSCP (cps2)"),
            ("direct_cp_8_4_2_be_bspline_analytic_multistage", "MR-D-BSCP (8-4-2)"),
            ("bs4_sitreg_v0", "SITReg (SOTA, learning-based)"),
        ],
    },
    {
        "stem": "oasis20_publication_extended",
        "title": "OASIS seed-3 first20 extended publication table",
        "caption": "OASIS brain registration results with additional spacing and baseline rows.",
        "latex_label": "tab:oasis20-publication-extended",
        "dataset": "oasis20",
        "source_json": "oasis20_selected_rows.json",
        "metrics": OASIS_METRICS,
        "rows": [
            ("idir_original_dense_siren_be_autograd", "INR-Dense"),
            ("sinr_cps4_be_bspline_analytic", "INR-BSCP (cps4)"),
            ("direct_cp_cps4_be_bspline_analytic", "D-BSCP (cps4)"),
            ("sinr_cps2_be_bspline_analytic", "INR-BSCP (cps2)"),
            ("direct_cp_cps2_be_bspline_analytic", "D-BSCP (cps2)"),
            ("direct_cp_8_4_2_be_bspline_analytic_multistage", "MR-D-BSCP (8-4-2)"),
            ("greedy", "Greedy"),
            ("bs1_vxm_v0_2", "VoxelMorph"),
            ("bs4_sitreg_v0", "SITReg (SOTA, learning-based)"),
        ],
    },
    {
        "stem": "oasis20_publication_main_with_references",
        "title": "OASIS seed-3 first20 compact publication table with references",
        "caption": "OASIS brain registration results with compact spline rows and reference methods.",
        "latex_label": "tab:oasis20-publication-main-with-references",
        "dataset": "oasis20",
        "source_json": "oasis20_selected_rows.json",
        "metrics": OASIS_METRICS,
        "rows": [
            ("idir_original_dense_siren_be_autograd", "INR-Dense"),
            ("sinr_cps2_be_bspline_analytic", "INR-BSCP (cps2)"),
            ("direct_cp_cps2_be_bspline_analytic", "D-BSCP (cps2)"),
            ("direct_cp_8_4_2_be_bspline_analytic_multistage", "MR-D-BSCP (8-4-2)"),
            ("greedy", "Greedy"),
            ("bs1_vxm_v0_2", "VoxelMorph"),
            ("bs4_sitreg_v0", "SITReg (SOTA)"),
        ],
    },
    {
        "stem": "lungct10_publication",
        "title": "LungCT 4DCT all10 publication table",
        "caption": "LungCT registration results using original snap TRE as the headline TRE.",
        "latex_label": "tab:lungct10-publication",
        "dataset": "lungct10",
        "source_json": "lungct10_selected_rows.json",
        "metrics": LUNGCT_METRICS,
        "rows": [
            ("idir_be_default_lungmask", "INR-Dense"),
            ("sinr_cps4_ptv_lcc_isotv", "INR-BSCP"),
            ("direct_cp_cps4_ptv_lcc_isotv", "D-BSCP"),
            ("direct_cp_32_16_8_4_global_ncc_fd_be", "MR-D-BSCP"),
            ("direct_cp_32_16_8_4_ptv_lcc_isotv", "MR-D-BSCP (pTVReg loss)"),
            ("ptvreg_reference", "pTVReg (SOTA)"),
        ],
    },
    {
        "stem": "lungct10_publication_multistage_focus",
        "title": "LungCT 4DCT all10 multistage-focused publication table",
        "caption": "LungCT registration results focused on multistage DirectCP variants.",
        "latex_label": "tab:lungct10-publication-multistage-focus",
        "dataset": "lungct10",
        "source_json": "lungct10_selected_rows.json",
        "metrics": LUNGCT_METRICS,
        "rows": [
            ("idir_be_default_lungmask", "INR-Dense"),
            ("direct_cp_32_16_8_4_global_ncc_fd_be", "MR-D-BSCP (32-16-8-4)"),
            ("direct_cp_32_16_8_4_ptv_lcc_isotv", "MR-D-BSCP (32-16-8-4) *"),
            ("ptvreg_reference", "pTVReg (SOTA)"),
        ],
    },
]


def build_table(
    spec: Mapping[str, Any],
    *,
    selected_rows_by_dataset: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[list[str], list[dict[str, str]], list[dict[str, Any]]]:
    rows_by_method = selected_rows_by_dataset[str(spec["dataset"])]
    headers = ["Method", *[str(metric["label"]) for metric in spec["metrics"]]]
    table_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, Any]] = []

    for method_id, display_name in spec["rows"]:
        if method_id not in rows_by_method:
            raise KeyError(f"{spec['stem']}: missing selected row for {method_id}")
        source_row = rows_by_method[method_id]
        output_row = {"Method": display_name}
        for metric in spec["metrics"]:
            output_row[str(metric["label"])] = format_mean_std(source_row, metric)
        table_rows.append(output_row)
        manifest_rows.append(
            {
                "method_id": method_id,
                "display_name": display_name,
                "source_label": source_row.get("label", ""),
                "selected_setting": source_row.get("selected_setting", ""),
                "cases": source_row.get("cases", ""),
                "source_path": source_row.get("source_path", ""),
                "metrics": {
                    str(metric["label"]): source_row["raw_metrics"][str(metric["key"])]
                    for metric in spec["metrics"]
                },
            }
        )

    return headers, table_rows, manifest_rows


def write_readme(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# Publication Best-Result Tables",
        "",
        "Publication-formatted tables derived from `best_result_table/v002_candidate`.",
        "Values are formatted as `mean (std)` using the selected-row raw metrics.",
        "",
        "## Files",
        "",
    ]
    for table in manifest["tables"]:
        stem = table["stem"]
        lines.append(f"- `{stem}.md`, `{stem}.csv`, `{stem}.tex`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- OASIS rows use the seed-3 first20 selected rows from the v002 candidate table.",
            "- LungCT `TRE (mm)` is the original/native-crop snap TRE column.",
            "- LungCT rows remain protocol-tagged comparison rows; do not interpret them as clean method-only comparisons unless preprocessing, metric, mask, and loss conventions are matched.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python experiments/paper_artifacts/best_result_table/build_publication_tables.py",
            "```",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines).rstrip() + "\n")


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir if args.source_dir.is_absolute() else REPO_ROOT / args.source_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows_by_dataset = {
        "oasis20": read_selected_rows(source_dir / "oasis20_selected_rows.json"),
        "lungct10": read_selected_rows(source_dir / "lungct10_selected_rows.json"),
    }

    manifest_tables = []
    combined_md_lines = ["# Publication Best-Result Tables", ""]
    combined_tex_parts = []

    for spec in TABLE_SPECS:
        headers, rows, manifest_rows = build_table(spec, selected_rows_by_dataset=selected_rows_by_dataset)
        stem = str(spec["stem"])
        write_csv(output_dir / f"{stem}.csv", rows, headers)
        table_md = markdown_table(headers, rows)
        (output_dir / f"{stem}.md").write_text(f"# {spec['title']}\n\n{table_md}\n")
        table_tex = latex_table(
            headers,
            rows,
            caption=str(spec["caption"]),
            label=str(spec["latex_label"]),
        )
        (output_dir / f"{stem}.tex").write_text(table_tex)

        combined_md_lines.extend([f"## {spec['title']}", "", table_md, ""])
        combined_tex_parts.append(table_tex)
        manifest_tables.append(
            {
                "stem": stem,
                "title": spec["title"],
                "dataset": spec["dataset"],
                "source_json": repo_relative(source_dir / str(spec["source_json"])),
                "headers": headers,
                "rows": manifest_rows,
            }
        )

    (output_dir / "publication_tables.md").write_text("\n".join(combined_md_lines).rstrip() + "\n")
    (output_dir / "publication_tables.tex").write_text("\n".join(combined_tex_parts).rstrip() + "\n")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dir": repo_relative(source_dir),
        "output_dir": repo_relative(output_dir),
        "format": "mean (std)",
        "source_version": "v002_candidate",
        "tables": manifest_tables,
    }
    (output_dir / "publication_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_readme(output_dir, manifest)

    for table in manifest_tables:
        print(f"wrote {table['stem']}: {len(table['rows'])} rows")


if __name__ == "__main__":
    main()
