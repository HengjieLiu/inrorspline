#!/usr/bin/env python3
"""Build a LungCT TRE/SDlogJ paper-style overlay with selected DUAL-INR rows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "experiments/paper_artifacts"
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))

import build_standardized_pareto as pareto  # noqa: E402


DEFAULT_BASE_ROWS = (
    REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v002_candidate/lungct10_pareto_source_rows.csv"
)
DEFAULT_STAGE2_POINTS = (
    REPO_ROOT
    / "experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage2_curve_points.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v004_candidate_dual_inr_pl1a_pl1c"
)
DEFAULT_BASE_METHODS = [
    "idir_be_default_lungmask",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "ptvreg_reference",
]

DUAL_METHODS = {
    "PL1A": {
        "method": "dual_inr_pl1a_author_blur_native_loss",
        "method_label": "DUAL-INR PL1A original blur native loss",
        "display_method": "DUAL-INR (original DUAL loss)",
        "method_family": "dual_inr",
        "color": "#7E57C2",
        "marker": "X",
        "weight_prefix": "m",
    },
    "PL1C": {
        "method": "dual_inr_pl1c_author_blur_ptv_loss",
        "method_label": "DUAL-INR PL1C original blur pTVReg loss",
        "display_method": "DUAL-INR (pTVReg loss)",
        "method_family": "dual_inr",
        "color": "#C44E52",
        "marker": "v",
        "weight_prefix": "w",
    },
}

BASE_METHOD_ORDER = [
    "idir_be_default_lungmask",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "dual_inr_pl1a_author_blur_native_loss",
    "dual_inr_pl1c_author_blur_ptv_loss",
    "ptvreg_reference",
]


def _fmt_weight(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:g}"


def _stage2_to_paper_rows(
    stage2: pd.DataFrame,
    variants: list[str],
    source_path: Path,
    exclude_labels: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    complete = stage2[stage2["technical_complete"].astype(str).str.lower() == "true"].copy()
    for variant in variants:
        spec = DUAL_METHODS[variant]
        group = complete[complete["variant"].astype(str) == variant].copy()
        if exclude_labels:
            group = group[~group["weight_label"].astype(str).isin(exclude_labels)].copy()
        if group.empty:
            raise RuntimeError(f"No complete Stage 2 rows found for {variant}")
        best_tre_idx = pd.to_numeric(
            group["headline_native_snap_tre_mm_mean"], errors="coerce"
        ).idxmin()
        best_weight = float(group.loc[best_tre_idx, "weight_value"])
        for _, source in group.iterrows():
            weight = float(source["weight_value"])
            ratio = weight / best_weight if best_weight else float("nan")
            prefix = str(spec["weight_prefix"])
            rows.append(
                {
                    "method": spec["method"],
                    "method_label": spec["method_label"],
                    "reg_value": weight,
                    "reg_label": f"{prefix}={_fmt_weight(weight)}",
                    "ratio_to_best_tre_weight": ratio,
                    "cases": int(source["n"]),
                    "original_snap_tre_mean_mm_mean": float(
                        source["headline_native_snap_tre_mm_mean"]
                    ),
                    "original_snap_tre_mean_mm_std": float(
                        source["headline_native_snap_tre_mm_sd"]
                    ),
                    "original_subvoxel_tre_mean_mm_mean": float(
                        source["headline_native_subvoxel_tre_mm_mean"]
                    ),
                    "original_subvoxel_tre_mean_mm_std": float(
                        source["headline_native_subvoxel_tre_mm_sd"]
                    ),
                    "reg1mm_snap_tre_mean_mm_mean": float("nan"),
                    "reg1mm_snap_tre_mean_mm_std": float("nan"),
                    "reg1mm_subvoxel_tre_mean_mm_mean": float(
                        source["registration_grid_subvoxel_tre_mm_mean"]
                    ),
                    "reg1mm_subvoxel_tre_mean_mm_std": float(
                        source["registration_grid_subvoxel_tre_mm_sd"]
                    ),
                    "fg_logj_std_mean": float(source["regularity_log_jdet_std_fg_mean"]),
                    "fg_logj_std_std": float(source["regularity_log_jdet_std_fg_sd"]),
                    "fg_jdet_le0_pct_mean": float(source["regularity_jdet_le0_pct_fg_mean"]),
                    "fg_jdet_le0_pct_std": float(source["regularity_jdet_le0_pct_fg_sd"]),
                    "fg_bending_energy_mean": float("nan"),
                    "fg_bending_energy_std": float("nan"),
                    "fg_diffusion_l2_mean": float("nan"),
                    "fg_diffusion_l2_std": float("nan"),
                    "fg_disp_mag_mean_vox_mean": float("nan"),
                    "fg_disp_mag_mean_vox_std": float("nan"),
                    "fg_disp_mag_p95_vox_mean": float("nan"),
                    "fg_disp_mag_p95_vox_std": float("nan"),
                    "fg_disp_mag_max_vox_mean": float("nan"),
                    "fg_disp_mag_max_vox_std": float("nan"),
                    "dataset": "lungct10",
                    "method_id": spec["method"],
                    "display_method": spec["display_method"],
                    "method_family": spec["method_family"],
                    "selected_lr": float("nan"),
                    "selected_lr_label": "",
                    "source_path": str(source_path.relative_to(REPO_ROOT)),
                }
            )
    return pd.DataFrame(rows)


def _method_order_for_variants(variants: list[str], base_methods: list[str]) -> list[str]:
    dual_methods = [str(DUAL_METHODS[variant]["method"]) for variant in variants]
    out: list[str] = []
    inserted_dual = False
    for method in base_methods:
        if method == "ptvreg_reference" and not inserted_dual:
            out.extend(dual_methods)
            inserted_dual = True
        out.append(method)
    if not inserted_dual:
        out.extend(dual_methods)
    return out


def _configure_dual_overlay_style(variants: list[str], base_methods: list[str]) -> None:
    pareto.LUNGCT_PLOT_METHOD_ORDER = _method_order_for_variants(variants, base_methods)
    for variant in variants:
        spec = DUAL_METHODS[variant]
        method = str(spec["method"])
        pareto.LUNGCT_METHOD_LABELS[method] = str(spec["method_label"])
        pareto.LUNGCT_METHOD_COLORS[method] = str(spec["color"])
        pareto.LUNGCT_METHOD_MARKERS[method] = str(spec["marker"])
        pareto.METHOD_FAMILY_BY_ID[method] = str(spec["method_family"])
    pareto.LUNGCT_METHOD_LABELS[
        "dual_inr_pl1a_author_blur_native_loss"
    ] = "DUAL-INR (original DUAL loss)"
    pareto.LUNGCT_METHOD_LABELS[
        "dual_inr_pl1c_author_blur_ptv_loss"
    ] = "DUAL-INR (pTVReg loss)"


def _write_outputs(
    *,
    rows: pd.DataFrame,
    output_dir: Path,
    nicknames: dict[str, dict[str, str]],
    variants: list[str],
    base_methods: list[str],
) -> list[Path]:
    _configure_dual_overlay_style(variants, base_methods)
    subset = rows.loc[[not pareto._should_exclude_lung(row) for _, row in rows.iterrows()]].copy()
    outputs: list[Path] = []
    for dirname, annotate in (
        ("lungct10_dual_overlay", True),
        ("lungct10_dual_overlay_paper_no_annotations", False),
    ):
        path = (
            output_dir
            / dirname
            / pareto.LUNGCT_SUBSET_NAME
            / "original_snap_tre_mean_mm_vs_fg_logj_std.png"
        )
        outputs.append(
            pareto.plot_lung_metric_pair(
                subset,
                path,
                acc_key="original_snap_tre_mean_mm",
                acc_label="Original snap TRE (mm)",
                acc_higher_is_better=False,
                reg_key="fg_logj_std",
                reg_label="Foreground stdLogJ",
                nicknames=nicknames,
                annotate=annotate,
            )
        )
    return outputs


def build(
    base_rows: Path,
    stage_points: list[Path],
    output_dir: Path,
    variants: list[str],
    exclude_labels: set[str],
    base_methods: list[str],
) -> dict[str, Any]:
    base = pd.read_csv(base_rows)
    base = base[base["method_id"].astype(str).isin(base_methods)].copy()
    dual_frames = [
        _stage2_to_paper_rows(pd.read_csv(path), variants, path, exclude_labels)
        for path in stage_points
    ]
    dual = pd.concat(dual_frames, ignore_index=True, sort=False)
    combined = pd.concat([base, dual], ignore_index=True, sort=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_out = output_dir / "lungct10_dual_overlay_source_rows.csv"
    combined.to_csv(source_out, index=False)

    nicknames = pareto.load_nicknames()
    outputs = _write_outputs(
        rows=combined,
        output_dir=output_dir,
        nicknames=nicknames,
        variants=variants,
        base_methods=base_methods,
    )

    summary_cols = [
        "method_id",
        "reg_label",
        "cases",
        "original_snap_tre_mean_mm_mean",
        "original_snap_tre_mean_mm_std",
        "fg_logj_std_mean",
        "fg_logj_std_std",
    ]
    dual_method_ids = [str(DUAL_METHODS[variant]["method"]) for variant in variants]
    dual_summary = combined[
        combined["method_id"].isin(dual_method_ids)
    ][summary_cols].copy()
    dual_summary.to_csv(output_dir / "dual_inr_selected_points.csv", index=False)

    manifest = {
        "artifact": "lungct_dual_inr_tre_sdlogj_overlay",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_rows": str(base_rows.relative_to(REPO_ROOT)),
        "stage_points": [str(path.relative_to(REPO_ROOT)) for path in stage_points],
        "source_rows": str(source_out.relative_to(REPO_ROOT)),
        "outputs": [str(path.relative_to(REPO_ROOT)) for path in outputs],
        "included_variants": variants,
        "excluded_dual_weight_labels": sorted(exclude_labels),
        "base_methods": base_methods,
        "plotted_methods": _method_order_for_variants(variants, base_methods),
        "ellipse_definition": "width = 2 * sample SD of foreground stdLogJ across cases; height = 2 * sample SD of native snap TRE across cases",
        "dual_methods": {
            variant: {
                key: value
                for key, value in spec.items()
                if key in {"method", "method_label", "display_method", "method_family"}
            }
            for variant, spec in DUAL_METHODS.items()
            if variant in variants
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "review_report.md").write_text(render_report(manifest, dual_summary))
    return manifest


def render_report(manifest: dict[str, Any], dual_summary: pd.DataFrame) -> str:
    variant_text = "/".join(manifest["included_variants"])
    lines = [
        f"# LungCT DUAL-INR {variant_text} TRE-SDlogJ Overlay",
        "",
        f"- Generated: `{manifest['generated_utc']}`",
        f"- Base source rows: `{manifest['base_rows']}`",
        f"- DUAL source rows: `{manifest['stage_points']}`",
        f"- Combined source rows: `{manifest['source_rows']}`",
        f"- Excluded DUAL weight labels: `{manifest['excluded_dual_weight_labels']}`",
        "- Ellipses show +/-1 sample SD in both axes via full width/height `2 * SD`.",
        "",
        "## Outputs",
        "",
    ]
    for path in manifest["outputs"]:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Added DUAL-INR Points",
            "",
            "| Method | Weight | n | TRE mean +/- SD | stdLogJ mean +/- SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in dual_summary.iterrows():
        lines.append(
            "| "
            f"{row['method_id']} | {row['reg_label']} | {int(row['cases'])} | "
            f"{float(row['original_snap_tre_mean_mm_mean']):.3f} +/- "
            f"{float(row['original_snap_tre_mean_mm_std']):.3f} | "
            f"{float(row['fg_logj_std_mean']):.4f} +/- "
            f"{float(row['fg_logj_std_std']):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-rows", type=Path, default=DEFAULT_BASE_ROWS)
    parser.add_argument("--stage2-points", type=Path, default=DEFAULT_STAGE2_POINTS)
    parser.add_argument(
        "--extra-stage-points",
        type=Path,
        action="append",
        default=[],
        help="Additional curve-point CSVs with the Stage 2 schema.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(DUAL_METHODS),
        default=["PL1A", "PL1C"],
        help="DUAL-INR variants to add to the existing LungCT paper rows.",
    )
    parser.add_argument(
        "--exclude-dual-weight-label",
        action="append",
        default=[],
        help="Weight label to exclude from DUAL rows, for example m100.",
    )
    parser.add_argument(
        "--base-methods",
        nargs="+",
        default=DEFAULT_BASE_METHODS,
        help="Existing paper methods to keep before adding selected DUAL rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(
        args.base_rows.resolve(),
        [args.stage2_points.resolve(), *[path.resolve() for path in args.extra_stage_points]],
        args.output_dir.resolve(),
        list(args.variants),
        set(args.exclude_dual_weight_label),
        list(args.base_methods),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
