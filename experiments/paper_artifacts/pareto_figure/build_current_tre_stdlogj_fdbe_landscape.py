#!/usr/bin/env python3
"""Build the current 2x2 Pareto preview with the FD-BE cps4 alpha landscape."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

EXPERIMENT_SCRIPTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SCRIPTS_DIR))

import build_current_tre_stdlogj_expanded as base  # noqa: E402
import artifact_paths  # noqa: E402


ARTIFACT_EXPERIMENT_KEY = "paper_artifacts/pareto_figure/v012_current_tre_stdlogj_fdbe_cps4_alpha_landscape"
EXPECTED_FDBE_ALPHAS = (10000.0, 20000.0, 50000.0, 100000.0, 200000.0, 500000.0, 1000000.0)


def alpha_label(value: float) -> str:
    if value > 0:
        exponent = int(math.floor(math.log10(value)))
        mantissa = value / (10.0**exponent)
        for candidate in (1.0, 2.0, 5.0):
            if abs(mantissa - candidate) < 1.0e-9:
                prefix = "" if candidate == 1.0 else f"{candidate:g}"
                return f"{prefix}e{exponent}"
    return f"{value:g}"


def load_current_fdbe_case_rows(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [base.load_fdbe_alpha_extension_case_rows(path) for path in paths]
    rows = pd.concat(frames, ignore_index=True, sort=False)
    rows["alpha_numeric"] = pd.to_numeric(rows["alpha"], errors="coerce")
    rows = rows.sort_values(["method_slug", "alpha_numeric", "case_id"]).copy()
    duplicate_keys = rows.duplicated(["method_slug", "alpha_numeric", "case_id"], keep=False)
    if duplicate_keys.any():
        duplicates = rows.loc[duplicate_keys, ["method_slug", "alpha_numeric", "case_id", "source_case_rows"]]
        raise RuntimeError(f"Duplicate FD-BE case rows in current sources:\n{duplicates.to_string(index=False)}")
    return rows.drop(columns=["alpha_numeric"])


def write_alpha1e4_audit(
    output_dir: Path,
    *,
    legacy_case_rows: Path,
    current_case_rows: pd.DataFrame,
) -> dict[str, Any]:
    legacy = pd.read_csv(legacy_case_rows)
    legacy = legacy[
        legacy["method_slug"].astype(str).isin(base.FDBE_SINGLE_POINT_SPECS)
        & (pd.to_numeric(legacy["alpha"], errors="coerce") == 10000.0)
    ].copy()
    current = current_case_rows[
        current_case_rows["method_slug"].astype(str).isin(base.FDBE_SINGLE_POINT_SPECS)
        & (pd.to_numeric(current_case_rows["alpha"], errors="coerce") == 10000.0)
    ].copy()
    merged = legacy.merge(
        current,
        on=["method_slug", "case_id", "case_key"],
        how="outer",
        suffixes=("_legacy_v011", "_rerun_current"),
        indicator=True,
    )
    merged["legacy_tre"] = pd.to_numeric(merged["original_snap_tre_mean_mm_legacy_v011"], errors="coerce")
    merged["rerun_tre"] = pd.to_numeric(merged["original_snap_tre_mean_mm_rerun_current"], errors="coerce")
    merged["delta_tre_rerun_minus_legacy"] = merged["rerun_tre"] - merged["legacy_tre"]
    merged["abs_delta_tre"] = merged["delta_tre_rerun_minus_legacy"].abs()
    audit_path = output_dir / "fdbe_alpha1e4_legacy_vs_rerun_audit.csv"
    merged.to_csv(audit_path, index=False)

    method_summaries: dict[str, dict[str, float | int | bool]] = {}
    review_only = False
    for method, group in merged.groupby("method_slug", dropna=False):
        legacy_mean = float(group["legacy_tre"].mean())
        rerun_mean = float(group["rerun_tre"].mean())
        mean_delta = rerun_mean - legacy_mean
        max_abs_case_delta = float(group["abs_delta_tre"].max())
        method_review = abs(mean_delta) > 0.05 or max_abs_case_delta > 0.25
        review_only = review_only or method_review
        method_summaries[str(method)] = {
            "rows": int(len(group)),
            "legacy_mean_tre": legacy_mean,
            "rerun_mean_tre": rerun_mean,
            "mean_delta_tre": mean_delta,
            "max_abs_case_delta_tre": max_abs_case_delta,
            "review_only": bool(method_review),
        }
    return {
        "audit_csv": base.repo_relative(audit_path),
        "legacy_case_rows": base.repo_relative(legacy_case_rows),
        "review_only": bool(review_only),
        "thresholds": {"mean_abs_delta_tre_mm": 0.05, "case_abs_delta_tre_mm": 0.25},
        "method_summaries": method_summaries,
    }


def write_review_report(
    output_dir: Path,
    manifest: Mapping[str, Any],
    lung: pd.DataFrame,
    fdbe_case_rows: pd.DataFrame,
) -> None:
    panel4 = lung[lung["panel4_include"].astype(bool)].copy()
    lines = [
        "# Current TRE/stdLogJ Expanded Pareto Preview with FD-BE cps4 Alpha Landscape",
        "",
        f"- Generated: `{manifest['generated_utc']}`",
        "- OASIS panels are copied from the OASIS100 paper-ready source rows.",
        "- Bottom-left keeps the original LungCT TRE/stdLogJ structure with ellipses.",
        "- Bottom-right is expanded LungCT TRE/stdLogJ without ellipses and includes FD-BE cps4 alpha landscapes.",
        f"- Alpha=1e4 rerun audit review-only flag: `{manifest['alpha1e4_rerun_audit']['review_only']}`.",
        "- Masked-loss method IDs are not included in `source_rows.csv`.",
        "",
        "## Outputs",
        "",
    ]
    for output in manifest["outputs"]:
        lines.append(f"- `{output}`")
    lines.extend(["", "## Panel 4 Rows", "", "| Method | Points | TRE range | SDlogJ range |", "|---|---:|---:|---:|"])
    for method in base.PANEL4_METHODS:
        group = panel4[panel4["method_id"].astype(str).eq(method)]
        if group.empty:
            continue
        label = base.METHOD_SPECS[method]["label"]
        lines.append(
            f"| {label} | {len(group)} | "
            f"{group[base.Y_COL].min():.3f}-{group[base.Y_COL].max():.3f} | "
            f"{group[base.X_COL].min():.4f}-{group[base.X_COL].max():.4f} |"
        )
    lines.extend(
        [
            "",
            "## FD-BE cps4 Alpha-Landscape Case Rows",
            "",
            "| Method | Alpha | Cases | TRE mean | SDlogJ mean | Source rows |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    fdbe_case_rows = fdbe_case_rows.copy()
    fdbe_case_rows["alpha_numeric"] = pd.to_numeric(fdbe_case_rows["alpha"], errors="coerce")
    for (method_slug, alpha), group in fdbe_case_rows.groupby(["method_slug", "alpha_numeric"], sort=True):
        sources = ", ".join(sorted(group.get("source_case_rows", pd.Series(dtype=str)).astype(str).unique()))
        lines.append(
            f"| {group['display_method'].iloc[0]} | {alpha_label(float(alpha))} | {len(group)} | "
            f"{group['original_snap_tre_mean_mm'].mean():.3f} | "
            f"{group['fg_logj_std'].mean():.4f} | {sources} |"
        )
    (output_dir / "review_report.md").write_text("\n".join(lines) + "\n")


def build(
    *,
    output_dir: Path,
    oasis_source: Path,
    lung_schedule_source: Path,
    legacy_w0022_source: Path,
    dual_stage2_source: Path,
    dual_stage3_source: Path,
    landscape_case_rows: Path,
    accepted_extension_case_rows: Path,
    legacy_v011_case_rows: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    oasis = base.load_oasis_rows(oasis_source)
    lung_base = base.load_schedule_refreshed_lung_rows(lung_schedule_source, legacy_w0022_source)
    dual_pl1a = base.load_dual_pl1a_rows(dual_stage2_source, dual_stage3_source)
    fdbe_case_rows = load_current_fdbe_case_rows([landscape_case_rows, accepted_extension_case_rows])
    fdbe_case_rows_path = output_dir / "fdbe_cps4_alpha_sweep_case_rows.csv"
    fdbe_case_rows.to_csv(fdbe_case_rows_path, index=False)
    alpha1e4_audit = write_alpha1e4_audit(
        output_dir,
        legacy_case_rows=legacy_v011_case_rows,
        current_case_rows=fdbe_case_rows,
    )
    fdbe_points = base.aggregate_fdbe_points(
        fdbe_case_rows,
        source_path=base.repo_relative(fdbe_case_rows_path),
    )

    lung = pd.concat([lung_base, dual_pl1a, fdbe_points], ignore_index=True, sort=False)
    lung["panel3_include"] = lung["panel3_include"].fillna(False).astype(bool)
    lung["panel4_include"] = lung["panel4_include"].fillna(False).astype(bool)
    lung["masked_loss_used"] = lung["masked_loss_used"].fillna(False).astype(bool)
    lung["legacy_context"] = lung["legacy_context"].fillna(False).astype(bool)

    oasis_output = output_dir / "oasis100_pareto_source_rows.csv"
    lung_output = output_dir / "source_rows.csv"
    shutil.copyfile(oasis_source, oasis_output)
    lung.to_csv(lung_output, index=False)

    outputs = [
        base.write_composite(
            output_dir,
            oasis,
            lung,
            annotate=False,
            filename="pareto_four_panel_current_fdbe_alpha_landscape_no_annotations.png",
        ),
        base.write_composite(
            output_dir,
            oasis,
            lung,
            annotate=False,
            filename=(
                "pareto_four_panel_current_fdbe_alpha_landscape_"
                "panel4_matched_panel3_y_no_annotations.png"
            ),
            panel4_match_panel3_y=True,
        ),
        base.write_composite(
            output_dir,
            oasis,
            lung,
            annotate=True,
            filename="pareto_four_panel_current_fdbe_alpha_landscape_annotated_review.png",
        ),
        base.write_composite(
            output_dir,
            oasis,
            lung,
            annotate=True,
            filename="pareto_four_panel_current_fdbe_alpha_landscape_zoomed_review.png",
            panel4_zoom=((0.07, 0.50), (0.86, 2.15)),
        ),
    ]

    validation = base.validate_outputs(oasis_source, output_dir, lung)
    expected_by_method = {
        spec["method_id"]: list(EXPECTED_FDBE_ALPHAS)
        for spec in base.FDBE_SINGLE_POINT_SPECS.values()
    }
    validation["expected_fdbe_cps4_alpha_values"] = expected_by_method
    validation["fdbe_case_row_count"] = int(len(fdbe_case_rows))
    validation["fdbe_expected_case_row_count"] = 140
    validation["fdbe_case_row_count_ok"] = bool(len(fdbe_case_rows) == 140)

    manifest: dict[str, Any] = {
        "artifact": "pareto_current_tre_stdlogj_fdbe_cps4_alpha_landscape",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_inputs": {
            "oasis100_base": base.repo_relative(oasis_source),
            "lungct_schedule_refresh": base.repo_relative(lung_schedule_source),
            "legacy_w0022_source": base.repo_relative(legacy_w0022_source),
            "dual_pl1a_stage2": base.repo_relative(dual_stage2_source),
            "dual_pl1a_stage3": base.repo_relative(dual_stage3_source),
            "fdbe_landscape_case_rows": base.repo_relative(landscape_case_rows),
            "fdbe_accepted_1e5_1e6_case_rows": base.repo_relative(accepted_extension_case_rows),
            "fdbe_legacy_v011_case_rows_for_alpha1e4_audit": base.repo_relative(legacy_v011_case_rows),
        },
        "source_rows": base.repo_relative(lung_output),
        "oasis_source_rows": base.repo_relative(oasis_output),
        "fdbe_cps4_alpha_sweep_case_rows": base.repo_relative(fdbe_case_rows_path),
        "outputs": [base.repo_relative(path) for path in outputs],
        "panel_layout": {
            "A": "Dice vs SDlogJ, OASIS N=100, unchanged from v003 source rows",
            "B": "HD95 vs SDlogJ, OASIS N=100, unchanged from v003 source rows",
            "C": "TRE vs SDlogJ, LungCT N=10, original structure plus 2500 MR-D-BSCP refresh",
            "D": "Expanded TRE vs SDlogJ, LungCT N=10, no ellipses, FD-BE cps4 alpha landscape",
        },
        "new_fdbe_alpha_landscape_runs_included": True,
        "masked_loss_rows_included": False,
        "review_only_due_to_alpha1e4_audit": bool(alpha1e4_audit["review_only"]),
        "alpha1e4_rerun_audit": alpha1e4_audit,
        "fdbe_cps4_alpha_sweep_policy": (
            "Use current-source unmasked full-crop global-NCC + dense FD-BE cps4 rows for "
            "alpha=1e4/2e4/5e4/2e5/5e5 plus accepted current alpha=1e5/1e6 rows."
        ),
        "validation": validation,
    }
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_review_report(output_dir, manifest, lung, fdbe_case_rows)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--oasis-source", type=Path, required=True)
    parser.add_argument("--lung-schedule-source", type=Path, required=True)
    parser.add_argument("--legacy-w0022-source", type=Path, required=True)
    parser.add_argument("--dual-stage2-source", type=Path, required=True)
    parser.add_argument("--dual-stage3-source", type=Path, required=True)
    parser.add_argument("--landscape-case-rows", type=Path, required=True)
    parser.add_argument("--accepted-extension-case-rows", type=Path, required=True)
    parser.add_argument("--legacy-v011-case-rows", type=Path, required=True)
    return parser.parse_args(argv)


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve the external destination and prior generated-input files."""

    args.review_root = artifact_paths.resolve_review_root(
        ARTIFACT_EXPERIMENT_KEY,
        args.review_root,
        repo_root=REPO_ROOT,
    )
    for attribute, option in (
        ("oasis_source", "--oasis-source"),
        ("lung_schedule_source", "--lung-schedule-source"),
        ("legacy_w0022_source", "--legacy-w0022-source"),
        ("dual_stage2_source", "--dual-stage2-source"),
        ("dual_stage3_source", "--dual-stage3-source"),
        ("landscape_case_rows", "--landscape-case-rows"),
        ("accepted_extension_case_rows", "--accepted-extension-case-rows"),
        ("legacy_v011_case_rows", "--legacy-v011-case-rows"),
    ):
        setattr(
            args,
            attribute,
            artifact_paths.resolve_external_input_file(
                getattr(args, attribute),
                label=option,
                repo_root=REPO_ROOT,
            ),
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = resolve_artifact_paths(parse_args(argv))
    manifest = build(
        output_dir=args.review_root,
        oasis_source=args.oasis_source,
        lung_schedule_source=args.lung_schedule_source,
        legacy_w0022_source=args.legacy_w0022_source,
        dual_stage2_source=args.dual_stage2_source,
        dual_stage3_source=args.dual_stage3_source,
        landscape_case_rows=args.landscape_case_rows,
        accepted_extension_case_rows=args.accepted_extension_case_rows,
        legacy_v011_case_rows=args.legacy_v011_case_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
