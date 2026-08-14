#!/usr/bin/env python3
"""Build LungCT TRE/stdLogJ 2x2 design-review plots from existing runs only."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_ROWS = (
    REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v002_candidate/lungct10_pareto_source_rows.csv"
)
DEFAULT_STAGE2_POINTS = (
    REPO_ROOT
    / "experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage2_curve_points.csv"
)
DEFAULT_STAGE3_POINTS = (
    REPO_ROOT
    / "experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage3_pl1a_refinement_curve_points.csv"
)
DEFAULT_IDIR_PTV_PILOT = (
    REPO_ROOT
    / "experiments/2026-06-26_lungct_idir_sampled_ptvlcc_tv_w0p11_all10/mean_rows.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v008_lungct_existing_run_design_review"
)

X_COL = "fg_logj_std_mean"
Y_COL = "original_snap_tre_mean_mm_mean"
X_STD_COL = "fg_logj_std_std"
Y_STD_COL = "original_snap_tre_mean_mm_std"

METHOD_ORDER = [
    "idir_be_default_lungmask",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "dual_inr_pl1a_author_blur_native_loss",
    "dual_inr_pl1c_author_blur_ptv_loss",
    "idir_sampled_ptv_lcc_tv_pilot",
    "ptvreg_reference",
]

METHOD_SPECS = {
    "idir_be_default_lungmask": {
        "label": "IDIR (IDIR loss)",
        "color": "#0072B2",
        "marker": "o",
        "line": True,
    },
    "direct_cp_32_16_8_4_global_ncc_fd_be": {
        "label": "MR-D-BSCP (IDIR loss)",
        "color": "#009E73",
        "marker": "^",
        "line": True,
    },
    "direct_cp_32_16_8_4_ptv_lcc_isotv": {
        "label": "MR-D-BSCP (pTVReg loss)",
        "color": "#D55E00",
        "marker": "s",
        "line": True,
    },
    "dual_inr_pl1a_author_blur_native_loss": {
        "label": "Dual-INR (original DUAL loss)",
        "color": "#7E57C2",
        "marker": "X",
        "line": True,
    },
    "dual_inr_pl1c_author_blur_ptv_loss": {
        "label": "Dual-INR (pTVReg loss)",
        "color": "#C44E52",
        "marker": "v",
        "line": True,
    },
    "idir_sampled_ptv_lcc_tv_pilot": {
        "label": "IDIR pTV pilot w=0.11",
        "color": "#666666",
        "marker": "D",
        "line": False,
    },
    "ptvreg_reference": {
        "label": "pTVReg (pTVReg loss)",
        "color": "#111111",
        "marker": "*",
        "line": False,
    },
}

PANEL_SPECS = {
    "A1": {
        "title": "A1. Original clean baseline",
        "methods": [
            "idir_be_default_lungmask",
            "direct_cp_32_16_8_4_global_ncc_fd_be",
            "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "ptvreg_reference",
        ],
        "ellipses": True,
        "include_pilot_in_annotated": False,
    },
    "A2": {
        "title": "A2. Existing candidates, no ellipses",
        "methods": [
            "idir_be_default_lungmask",
            "direct_cp_32_16_8_4_global_ncc_fd_be",
            "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "dual_inr_pl1a_author_blur_native_loss",
            "dual_inr_pl1c_author_blur_ptv_loss",
            "ptvreg_reference",
        ],
        "ellipses": False,
        "include_pilot_in_annotated": True,
    },
    "B1": {
        "title": "B1. IDIR/native-loss family",
        "methods": [
            "idir_be_default_lungmask",
            "dual_inr_pl1a_author_blur_native_loss",
            "direct_cp_32_16_8_4_global_ncc_fd_be",
            "ptvreg_reference",
        ],
        "ellipses": True,
        "include_pilot_in_annotated": False,
    },
    "B2": {
        "title": "B2. pTVReg-loss family",
        "methods": [
            "dual_inr_pl1c_author_blur_ptv_loss",
            "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "ptvreg_reference",
        ],
        "ellipses": True,
        "include_pilot_in_annotated": True,
    },
}

BASE_METHODS = {
    "idir_be_default_lungmask",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "ptvreg_reference",
}

BASE_EXCLUDED_RATIOS = {
    "direct_cp_32_16_8_4_global_ncc_fd_be": {0.0, 0.001, 0.01},
    "direct_cp_32_16_8_4_ptv_lcc_isotv": {0.01, 10.0, 100.0},
}


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def close(value: float, target: float) -> bool:
    return math.isfinite(value) and abs(value - target) <= 1e-9 * max(1.0, abs(target))


def fmt_weight(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:g}"


def is_default_excluded(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method_id") or row.get("method"))
    ratio = finite_float(row.get("ratio_to_best_tre_weight"))
    return any(close(ratio, target) for target in BASE_EXCLUDED_RATIOS.get(method, set()))


def load_base_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows[rows["method_id"].astype(str).isin(BASE_METHODS)].copy()
    rows["source_group"] = "base_pareto_v002"
    rows["source_path"] = repo_relative(path)
    rows["display_method"] = rows["method_id"].map(lambda method: METHOD_SPECS[str(method)]["label"])
    rows["plotted_default_exclusions"] = [
        not is_default_excluded(row) for row in rows.to_dict("records")
    ]
    rows["pilot_only"] = False
    rows["include_unannotated"] = rows["plotted_default_exclusions"]
    rows["include_annotated"] = rows["plotted_default_exclusions"]
    return rows


def stage_points_to_rows(path: Path, variants: set[str], *, exclude_weight_labels: set[str]) -> pd.DataFrame:
    source_rows = pd.read_csv(path)
    source_rows = source_rows[source_rows["technical_complete"].astype(str).str.lower() == "true"].copy()
    source_rows = source_rows[source_rows["variant"].astype(str).isin(variants)].copy()
    source_rows = source_rows[~source_rows["weight_label"].astype(str).isin(exclude_weight_labels)].copy()
    out_rows: list[dict[str, Any]] = []
    for variant, group in source_rows.groupby("variant", sort=False):
        if variant == "PL1A":
            method_id = "dual_inr_pl1a_author_blur_native_loss"
            prefix = "m"
        elif variant == "PL1C":
            method_id = "dual_inr_pl1c_author_blur_ptv_loss"
            prefix = "w"
        else:
            continue
        best_idx = pd.to_numeric(group["headline_native_snap_tre_mm_mean"], errors="coerce").idxmin()
        best_weight = finite_float(group.loc[best_idx, "weight_value"])
        for _, row in group.iterrows():
            weight = finite_float(row.get("weight_value"))
            out_rows.append(
                {
                    "method": method_id,
                    "method_id": method_id,
                    "method_label": METHOD_SPECS[method_id]["label"],
                    "display_method": METHOD_SPECS[method_id]["label"],
                    "reg_value": weight,
                    "reg_label": f"{prefix}={fmt_weight(weight)}",
                    "ratio_to_best_tre_weight": weight / best_weight if best_weight else float("nan"),
                    "cases": int(row["n"]),
                    "original_snap_tre_mean_mm_mean": finite_float(row["headline_native_snap_tre_mm_mean"]),
                    "original_snap_tre_mean_mm_std": finite_float(row["headline_native_snap_tre_mm_sd"]),
                    "fg_logj_std_mean": finite_float(row["regularity_log_jdet_std_fg_mean"]),
                    "fg_logj_std_std": finite_float(row["regularity_log_jdet_std_fg_sd"]),
                    "dataset": "lungct10",
                    "source_group": "dual_stage_points",
                    "source_path": repo_relative(path),
                    "variant": variant,
                    "weight_label": str(row["weight_label"]),
                    "plotted_default_exclusions": True,
                    "pilot_only": False,
                    "include_unannotated": True,
                    "include_annotated": True,
                }
            )
    return pd.DataFrame(out_rows)


def load_pilot_row(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    source = pd.read_csv(path)
    if source.empty:
        return pd.DataFrame()
    row = source.iloc[0]
    return pd.DataFrame(
        [
            {
                "method": "idir_sampled_ptv_lcc_tv_pilot",
                "method_id": "idir_sampled_ptv_lcc_tv_pilot",
                "method_label": "IDIR sampled pTV-LCC + TV pilot",
                "display_method": METHOD_SPECS["idir_sampled_ptv_lcc_tv_pilot"]["label"],
                "reg_value": finite_float(row.get("ptv_isotv_weight")),
                "reg_label": "pilot w=0.11",
                "ratio_to_best_tre_weight": float("nan"),
                "cases": int(row.get("cases", 10)),
                "original_snap_tre_mean_mm_mean": finite_float(row.get(Y_COL)),
                "original_snap_tre_mean_mm_std": finite_float(row.get(Y_STD_COL)),
                "fg_logj_std_mean": finite_float(row.get(X_COL)),
                "fg_logj_std_std": finite_float(row.get(X_STD_COL)),
                "dataset": "lungct10",
                "source_group": "idir_ptv_pilot",
                "source_path": repo_relative(path),
                "variant": "pilot_only",
                "weight_label": "pilot_w0p11",
                "plotted_default_exclusions": False,
                "pilot_only": True,
                "include_unannotated": False,
                "include_annotated": True,
            }
        ]
    )


def build_source_rows(
    base_rows: Path,
    stage2_points: Path,
    stage3_points: Path,
    idir_ptv_pilot: Path,
) -> pd.DataFrame:
    frames = [
        load_base_rows(base_rows),
        stage_points_to_rows(stage2_points, {"PL1A", "PL1C"}, exclude_weight_labels={"m100"}),
        stage_points_to_rows(stage3_points, {"PL1A"}, exclude_weight_labels=set()),
        load_pilot_row(idir_ptv_pilot),
    ]
    rows = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False)
    rows["plot_sort"] = pd.to_numeric(rows["ratio_to_best_tre_weight"], errors="coerce")
    rows["reg_sort"] = pd.to_numeric(rows["reg_value"], errors="coerce")
    return rows


def finite_panel_rows(rows: pd.DataFrame, methods: list[str], *, annotate: bool, include_pilot: bool) -> pd.DataFrame:
    allowed = set(methods)
    if annotate and include_pilot:
        allowed.add("idir_sampled_ptv_lcc_tv_pilot")
    include_col = "include_annotated" if annotate else "include_unannotated"
    panel = rows[rows["method_id"].astype(str).isin(allowed)].copy()
    panel = panel[panel[include_col].astype(bool)].copy()
    panel[X_COL] = pd.to_numeric(panel[X_COL], errors="coerce")
    panel[Y_COL] = pd.to_numeric(panel[Y_COL], errors="coerce")
    panel = panel[panel[X_COL].notna() & panel[Y_COL].notna()].copy()
    return panel


def annotation_text(row: Mapping[str, Any]) -> str:
    method = str(row.get("method_id"))
    if method == "ptvreg_reference":
        return "pTVReg"
    if method == "idir_sampled_ptv_lcc_tv_pilot":
        return "pilot"
    label = str(row.get("reg_label") or "")
    return label.replace("=", "")


def draw_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    panel_key: str,
    *,
    annotate: bool,
) -> None:
    spec = PANEL_SPECS[panel_key]
    panel = finite_panel_rows(
        rows,
        list(spec["methods"]),
        annotate=annotate,
        include_pilot=bool(spec["include_pilot_in_annotated"]),
    )
    for method in METHOD_ORDER:
        group = panel[panel["method_id"].astype(str) == method].copy()
        if group.empty:
            continue
        group = group.sort_values(["plot_sort", "reg_sort"], na_position="last")
        mspec = METHOD_SPECS[method]
        color = str(mspec["color"])
        marker = str(mspec["marker"])
        xs = group[X_COL].to_numpy(dtype=float)
        ys = group[Y_COL].to_numpy(dtype=float)
        if bool(mspec["line"]) and len(xs) > 1:
            ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.82, zorder=2)
        if bool(spec["ellipses"]):
            for _, row in group.iterrows():
                xerr = max(finite_float(row.get(X_STD_COL), 0.0), 0.0)
                yerr = max(finite_float(row.get(Y_STD_COL), 0.0), 0.0)
                if xerr > 0 or yerr > 0:
                    alpha = 0.16 if method == "ptvreg_reference" else 0.10
                    ax.add_patch(
                        Ellipse(
                            (finite_float(row[X_COL]), finite_float(row[Y_COL])),
                            width=2.0 * xerr,
                            height=2.0 * yerr,
                            facecolor=color,
                            edgecolor=color,
                            linewidth=0.9,
                            alpha=alpha,
                            zorder=1,
                        )
                    )
        size = 110 if method == "ptvreg_reference" else 42
        if method == "idir_sampled_ptv_lcc_tv_pilot":
            ax.scatter(
                xs,
                ys,
                s=58,
                facecolor="none",
                edgecolor=color,
                marker=marker,
                linewidth=1.2,
                label=str(mspec["label"]),
                zorder=4,
            )
        else:
            ax.scatter(
                xs,
                ys,
                s=size,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.7,
                label=str(mspec["label"]),
                zorder=3,
            )
        if annotate:
            for _, row in group.iterrows():
                ax.annotate(
                    annotation_text(row),
                    (finite_float(row[X_COL]), finite_float(row[Y_COL])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.3,
                    color=color,
                )
    ax.set_title(str(spec["title"]), fontsize=10)
    ax.set_xlabel("Foreground stdLogJ")
    ax.set_ylabel("Original snap TRE (mm)")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=6.8, frameon=True, loc="best")


def axis_limits_for(rows: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = pd.to_numeric(rows[X_COL], errors="coerce").dropna()
    ys = pd.to_numeric(rows[Y_COL], errors="coerce").dropna()
    if xs.empty or ys.empty:
        return (0.0, 1.0), (0.0, 1.0)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_pad = max((x_max - x_min) * 0.08, 0.015)
    y_pad = max((y_max - y_min) * 0.08, 0.035)
    return (max(0.0, x_min - x_pad), x_max + x_pad), (max(0.0, y_min - y_pad), y_max + y_pad)


def set_panel_axis_rules(fig_axes: Mapping[str, plt.Axes], rows: pd.DataFrame, *, annotate: bool) -> None:
    shared_ylim = fig_axes["A1"].get_ylim()
    for key in ("A2", "B1", "B2"):
        fig_axes[key].set_ylim(*shared_ylim)

    b_rows = []
    for key in ("B1", "B2"):
        spec = PANEL_SPECS[key]
        b_rows.append(
            finite_panel_rows(
                rows,
                list(spec["methods"]),
                annotate=annotate,
                include_pilot=bool(spec["include_pilot_in_annotated"]),
            )
        )
    combined = pd.concat(b_rows, ignore_index=True, sort=False)
    xlim, _ylim = axis_limits_for(combined)
    for key in ("B1", "B2"):
        fig_axes[key].set_xlim(*xlim)


def write_figure(rows: pd.DataFrame, output_path: Path, *, annotate: bool) -> Path:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.4), squeeze=False)
    panel_axes = {
        "A1": axes[0][0],
        "A2": axes[0][1],
        "B1": axes[1][0],
        "B2": axes[1][1],
    }
    for key in ("A1", "A2", "B1", "B2"):
        draw_panel(panel_axes[key], rows, key, annotate=annotate)
    set_panel_axis_rules(panel_axes, rows, annotate=annotate)
    title = "LungCT TRE/stdLogJ design review"
    title += " (annotated)" if annotate else " (unannotated)"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260)
    plt.close(fig)
    return output_path


def render_report(manifest: Mapping[str, Any], rows: pd.DataFrame) -> str:
    plotted = rows[rows["include_unannotated"].astype(bool)]
    lines = [
        "# LungCT TRE/stdLogJ Existing-Run Design Review",
        "",
        f"- Generated: `{manifest['generated_utc']}`",
        "- No IDIR pTVReg-loss sweep was launched or required for this preview.",
        "- A2 is the all-candidate no-ellipse panel.",
        "- A2, B1, and B2 use A1's y-axis limits inside each 2x2 figure.",
        "- B1 and B2 also use matched x-axis limits inside each 2x2 figure.",
        "- Ellipse width/height are `2 * sample SD` for foreground stdLogJ and original snap TRE.",
        "- The IDIR pTV pilot is annotated-only and is not included in the unannotated camera-style preview.",
        "",
        "## Outputs",
        "",
    ]
    for output in manifest["outputs"]:
        lines.append(f"- `{output}`")
    lines.extend(
        [
            "",
            "## Plotted Unannotated Rows",
            "",
            "| Method | Points | TRE range | stdLogJ range |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHOD_ORDER:
        group = plotted[plotted["method_id"].astype(str) == method]
        if group.empty:
            continue
        label = METHOD_SPECS[method]["label"]
        lines.append(
            f"| {label} | {len(group)} | "
            f"{group[Y_COL].min():.3f}-{group[Y_COL].max():.3f} | "
            f"{group[X_COL].min():.4f}-{group[X_COL].max():.4f} |"
        )
    lines.extend(
        [
            "",
            "## Exclusions",
            "",
            "- Base MR-D-BSCP/IDIR curves use the existing LungCT default-exclusion convention.",
            "- DUAL PL1A `m100` is omitted from the preview.",
            "- Full included/excluded row metadata is retained in `source_rows.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def build(
    *,
    base_rows: Path,
    stage2_points: Path,
    stage3_points: Path,
    idir_ptv_pilot: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows = build_source_rows(base_rows, stage2_points, stage3_points, idir_ptv_pilot)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source_rows.csv"
    rows.to_csv(source_path, index=False)
    outputs = [
        write_figure(rows, output_dir / "lungct_tre_stdlogj_2x2_annotated.png", annotate=True),
        write_figure(rows, output_dir / "lungct_tre_stdlogj_2x2_unannotated.png", annotate=False),
    ]
    manifest: dict[str, Any] = {
        "artifact": "lungct_tre_stdlogj_existing_run_design_review",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_inputs": {
            "base_rows": repo_relative(base_rows),
            "stage2_points": repo_relative(stage2_points),
            "stage3_points": repo_relative(stage3_points),
            "idir_ptv_pilot": repo_relative(idir_ptv_pilot),
        },
        "source_rows": repo_relative(source_path),
        "outputs": [repo_relative(path) for path in outputs],
        "panel_layout": {
            "A1": "Original clean baseline plot",
            "A2": "All existing candidate curves, no ellipses",
            "B1": "Split ellipse, IDIR/native-loss family",
            "B2": "Split ellipse, pTVReg-loss family",
        },
        "ellipse_definition": "width = 2 * sample SD of foreground stdLogJ; height = 2 * sample SD of original snap TRE",
        "axis_rule": "A2, B1, and B2 share A1's y-axis limits; B1 and B2 also share x-axis limits within each annotated/unannotated figure.",
        "no_new_runs": True,
        "idir_ptv_pilot_policy": "Annotated-only pilot marker; excluded from unannotated camera-style preview.",
        "base_default_excluded_ratios": {
            method: sorted(values) for method, values in BASE_EXCLUDED_RATIOS.items()
        },
        "dual_excluded_weight_labels": ["m100"],
        "row_counts": {
            "source_rows": int(len(rows)),
            "unannotated_rows": int(rows["include_unannotated"].astype(bool).sum()),
            "annotated_rows": int(rows["include_annotated"].astype(bool).sum()),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "review_report.md").write_text(render_report(manifest, rows))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-rows", type=Path, default=DEFAULT_BASE_ROWS)
    parser.add_argument("--stage2-points", type=Path, default=DEFAULT_STAGE2_POINTS)
    parser.add_argument("--stage3-points", type=Path, default=DEFAULT_STAGE3_POINTS)
    parser.add_argument("--idir-ptv-pilot", type=Path, default=DEFAULT_IDIR_PTV_PILOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(
        base_rows=args.base_rows.resolve(),
        stage2_points=args.stage2_points.resolve(),
        stage3_points=args.stage3_points.resolve(),
        idir_ptv_pilot=args.idir_ptv_pilot.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
