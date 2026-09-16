#!/usr/bin/env python3
"""Build the final Pareto figure with DIR-LAB 4DCT bottom-panel styling."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
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


ARTIFACT_EXPERIMENT_KEY = "paper_artifacts/pareto_figure/v013_final_dirlab4dct_axes_labels"

X_COL = base.X_COL
Y_COL = base.Y_COL
X_STD_COL = base.X_STD_COL
Y_STD_COL = base.Y_STD_COL
DAGGER = "\u2020"
X_AXIS_LABEL = "Foreground SDlogJ"

ALL_METHODS = [
    "idir_be_default_lungmask",
    "dual_inr_pl1a_author_blur_native_loss",
    "sinr_cps4_global_ncc_fd_be",
    "direct_cp_cps4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "ptvreg_reference",
]
SELECTED_METHODS = [
    "idir_be_default_lungmask",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "ptvreg_reference",
]

FINAL_METHOD_SPECS = {
    "idir_be_default_lungmask": {
        "label": "INR-Dense",
        "color": "#0072B2",
        "marker": "o",
        "line": True,
    },
    "dual_inr_pl1a_author_blur_native_loss": {
        "label": "MR-INR-Dense",
        "color": "#56B4E9",
        "marker": "o",
        "line": True,
    },
    "sinr_cps4_global_ncc_fd_be": {
        "label": "INR-BSCP (cps4)",
        "color": "#009E73",
        "marker": "o",
        "line": True,
    },
    "direct_cp_cps4_global_ncc_fd_be": {
        "label": "D-BSCP (cps4)",
        "color": "#66C2A5",
        "marker": "o",
        "line": True,
    },
    "direct_cp_32_16_8_4_global_ncc_fd_be": {
        "label": "MR-D-BSCP (32-16-8-4)",
        "color": "#7E57C2",
        "marker": "o",
        "line": True,
    },
    "direct_cp_32_16_8_4_ptv_lcc_isotv": {
        "label": f"MR-D-BSCP (32-16-8-4){DAGGER}",
        "color": "#D55E00",
        "marker": "o",
        "line": True,
    },
    "ptvreg_reference": {
        "label": f"pTVReg{DAGGER}",
        "color": "#111111",
        "marker": "*",
        "line": False,
    },
}

AXIS_POLICY = {
    "xlim": (0.06, 0.50),
    "ylim": (0.60, 2.05),
    "xticks": [0.1, 0.2, 0.3, 0.4, 0.5],
    "yticks": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
}
STYLE = {
    "figsize": (14.0, 10.0),
    "dpi": 300,
    "title_fontsize": 14,
    "axis_label_fontsize": 14,
    "tick_label_fontsize": 12,
    "legend_fontsize": 7.3,
    "legend_title_fontsize": 7.1,
    "marker_size": 48,
    "ptvreg_marker_size": 120,
    "line_width": 1.55,
    "ellipse_alpha": 0.12,
    "grid_alpha": 0.25,
    "wspace": 0.20,
    "hspace": 0.30,
}


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_pair_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out[X_COL] = pd.to_numeric(out[X_COL], errors="coerce")
    out[Y_COL] = pd.to_numeric(out[Y_COL], errors="coerce")
    out = out[out[X_COL].notna() & out[Y_COL].notna()].copy()
    if out.empty:
        return out
    out["sort_ratio"] = pd.to_numeric(out.get("ratio_to_best_tre_weight"), errors="coerce")
    out["sort_reg"] = pd.to_numeric(out.get("reg_value"), errors="coerce")
    return out.sort_values(["sort_ratio", "sort_reg"], na_position="last")


def annotation_text(row: Mapping[str, Any]) -> str:
    method = str(row.get("method_id"))
    if method == "ptvreg_reference":
        return f"pTVReg{DAGGER}"
    label = str(row.get("reg_label") or "")
    return label if label else str(row.get("reg_value", ""))


def draw_lung_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    methods: Sequence[str],
    *,
    title: str,
    annotate: bool,
    ellipses: bool,
) -> None:
    panel = rows[rows["method_id"].astype(str).isin(methods)].copy()
    for method in methods:
        group = finite_pair_rows(panel[panel["method_id"].astype(str).eq(method)])
        if group.empty:
            continue
        spec = FINAL_METHOD_SPECS[method]
        color = str(spec["color"])
        marker = str(spec["marker"])
        xs = group[X_COL].to_numpy(dtype=float)
        ys = group[Y_COL].to_numpy(dtype=float)
        if bool(spec["line"]) and len(group) > 1:
            ax.plot(xs, ys, color=color, linewidth=STYLE["line_width"], alpha=0.78, zorder=2)
        if ellipses:
            for _, row in group.iterrows():
                xerr = max(finite_float(row.get(X_STD_COL), 0.0), 0.0)
                yerr = max(finite_float(row.get(Y_STD_COL), 0.0), 0.0)
                if xerr > 0 or yerr > 0:
                    ax.add_patch(
                        Ellipse(
                            (finite_float(row[X_COL]), finite_float(row[Y_COL])),
                            width=2.0 * xerr,
                            height=2.0 * yerr,
                            facecolor=color,
                            edgecolor=color,
                            linewidth=0.9,
                            alpha=STYLE["ellipse_alpha"],
                            zorder=1,
                        )
                    )
        size = STYLE["ptvreg_marker_size"] if method == "ptvreg_reference" else STYLE["marker_size"]
        ax.scatter(
            xs,
            ys,
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            label=str(spec["label"]),
            zorder=3,
        )
        if annotate:
            for _, row in group.iterrows():
                ax.annotate(
                    annotation_text(row),
                    (finite_float(row[X_COL]), finite_float(row[Y_COL])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.2,
                    color=color,
                    clip_on=True,
                )

    ax.set_title(title, fontsize=STYLE["title_fontsize"])
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel("TRE (mm)")
    ax.set_xlim(*AXIS_POLICY["xlim"])
    ax.set_ylim(*AXIS_POLICY["ylim"])
    ax.set_xticks(AXIS_POLICY["xticks"])
    ax.set_yticks(AXIS_POLICY["yticks"])
    ax.tick_params(axis="both", labelsize=STYLE["tick_label_fontsize"])
    ax.xaxis.label.set_size(STYLE["axis_label_fontsize"])
    ax.yaxis.label.set_size(STYLE["axis_label_fontsize"])
    ax.grid(True, alpha=STYLE["grid_alpha"])
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="none", marker="", color="none"))
    labels.append(f"{DAGGER} pTVReg loss")
    legend = ax.legend(
        handles,
        labels,
        loc="best",
        fontsize=STYLE["legend_fontsize"],
        frameon=True,
    )
    legend_texts = legend.get_texts()
    if legend_texts:
        legend_texts[-1].set_fontsize(STYLE["legend_title_fontsize"])


def draw_composite(
    output_dir: Path,
    oasis: pd.DataFrame,
    lung: pd.DataFrame,
    *,
    annotate: bool,
    filename_base: str,
    formats: Sequence[str],
) -> list[Path]:
    nicknames = base.pareto_base.load_nicknames()
    fig, axes = plt.subplots(2, 2, figsize=STYLE["figsize"], squeeze=False)
    oasis_plot = base.pareto_base._configure_oasis_plot_module(nicknames, annotate=annotate)
    base.pareto_base._draw_composite_oasis_panel(
        axes[0][0],
        oasis_plot,
        oasis,
        title="Dice vs Foreground SDlogJ (OASIS N=100)",
        acc_key="dice_mean",
        annotate=annotate,
    )
    axes[0][0].set_xlabel(X_AXIS_LABEL)
    base.pareto_base._draw_composite_oasis_panel(
        axes[0][1],
        oasis_plot,
        oasis,
        title="HD95 vs Foreground SDlogJ (OASIS N=100)",
        acc_key="hd95_mean",
        annotate=annotate,
    )
    axes[0][1].set_xlabel(X_AXIS_LABEL)
    draw_lung_panel(
        axes[1][0],
        lung,
        ALL_METHODS,
        title="TRE vs Foreground SDlogJ (DIR-LAB 4DCT N=10), all",
        annotate=annotate,
        ellipses=False,
    )
    draw_lung_panel(
        axes[1][1],
        lung,
        SELECTED_METHODS,
        title="TRE vs Foreground SDlogJ (DIR-LAB 4DCT N=10), selected",
        annotate=annotate,
        ellipses=True,
    )
    fig.subplots_adjust(wspace=STYLE["wspace"], hspace=STYLE["hspace"])

    outputs: list[Path] = []
    for fmt in formats:
        output_path = output_dir / f"{filename_base}.{fmt}"
        fig.savefig(output_path, dpi=STYLE["dpi"], bbox_inches="tight")
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def validate_outputs(
    *,
    oasis_source: Path,
    output_dir: Path,
    source_rows: Path,
    lung: pd.DataFrame,
) -> dict[str, Any]:
    oasis_copy = output_dir / "oasis100_pareto_source_rows.csv"
    source_copy = output_dir / "source_rows.csv"
    oasis_rows_match = pd.read_csv(oasis_source).equals(pd.read_csv(oasis_copy))
    source_rows_match = pd.read_csv(source_rows).equals(pd.read_csv(source_copy))
    masked_method_ids = sorted(
        lung.loc[
            lung["method_id"].astype(str).str.contains("masked", case=False, na=False),
            "method_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    masked_source_paths = sorted(
        lung.loc[
            lung.get("source_path", pd.Series("", index=lung.index))
            .astype(str)
            .str.contains("masked_ncc", case=False, na=False),
            "source_path",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    return {
        "oasis_rows_match_v003": bool(oasis_rows_match),
        "source_rows_match_v012": bool(source_rows_match),
        "masked_method_ids_in_source_rows": masked_method_ids,
        "masked_ncc_source_paths_in_source_rows": masked_source_paths,
        "bottom_axis_policy": AXIS_POLICY,
        "panel3_methods_all": ALL_METHODS,
        "panel4_methods_selected": SELECTED_METHODS,
        "panel3_labels_all": [FINAL_METHOD_SPECS[method]["label"] for method in ALL_METHODS],
        "panel4_labels_selected": [
            FINAL_METHOD_SPECS[method]["label"] for method in SELECTED_METHODS
        ],
        "panel3_ellipses": False,
        "panel4_ellipses": True,
    }


def write_review_report(output_dir: Path, manifest: Mapping[str, Any], lung: pd.DataFrame) -> None:
    lines = [
        "# Final Pareto Figure: DIR-LAB 4DCT Axes and Labels",
        "",
        f"- Generated: `{manifest['generated_utc']}`",
        "- Top OASIS panels are unchanged from the v003 OASIS100 source rows; x-axis display label is unified to foreground SDlogJ.",
        "- Bottom-left is the all-method DIR-LAB 4DCT panel without ellipses.",
        "- Bottom-right is the selected DIR-LAB 4DCT panel with ellipses.",
        "- Both bottom panels share x/y limits and ticks; weak-reg FD-BE outliers are clipped.",
        f"- Legend bottom note: `{DAGGER} pTVReg loss`.",
        "",
        "## Outputs",
        "",
    ]
    for output in manifest["outputs"]:
        lines.append(f"- `{output}`")
    lines.extend(["", "## Bottom Panel Method Ranges", ""])
    for title, methods in [
        ("All", ALL_METHODS),
        ("Selected", SELECTED_METHODS),
    ]:
        lines.extend(
            [
                f"### {title}",
                "",
                "| Method | Points | TRE range | Foreground SDlogJ range |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in methods:
            group = finite_pair_rows(lung[lung["method_id"].astype(str).eq(method)])
            if group.empty:
                continue
            lines.append(
                f"| {FINAL_METHOD_SPECS[method]['label']} | {len(group)} | "
                f"{group[Y_COL].min():.3f}-{group[Y_COL].max():.3f} | "
                f"{group[X_COL].min():.4f}-{group[X_COL].max():.4f} |"
            )
        lines.append("")
    (output_dir / "review_report.md").write_text("\n".join(lines) + "\n")


def build(
    *,
    output_dir: Path,
    oasis_source: Path,
    source_rows: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    oasis = base.load_oasis_rows(oasis_source)
    lung = pd.read_csv(source_rows).copy()

    oasis_copy = output_dir / "oasis100_pareto_source_rows.csv"
    source_copy = output_dir / "source_rows.csv"
    shutil.copyfile(oasis_source, oasis_copy)
    shutil.copyfile(source_rows, source_copy)

    outputs = []
    outputs.extend(
        draw_composite(
            output_dir,
            oasis,
            lung,
            annotate=False,
            filename_base="pareto_four_panel_final_dirlab4dct_no_annotations",
            formats=("png", "pdf"),
        )
    )
    outputs.extend(
        draw_composite(
            output_dir,
            oasis,
            lung,
            annotate=True,
            filename_base="pareto_four_panel_final_dirlab4dct_annotated_review",
            formats=("png",),
        )
    )

    manifest: dict[str, Any] = {
        "artifact": "pareto_final_dirlab4dct_axes_labels",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_inputs": {
            "oasis100_base": base.repo_relative(oasis_source),
            "v012_source_rows": base.repo_relative(source_rows),
        },
        "oasis_source_rows": base.repo_relative(oasis_copy),
        "source_rows": base.repo_relative(source_copy),
        "outputs": [base.repo_relative(path) for path in outputs],
        "panel_layout": {
            "A": "Dice vs Foreground SDlogJ, OASIS N=100, unchanged from v003 source rows",
            "B": "HD95 vs Foreground SDlogJ, OASIS N=100, unchanged from v003 source rows",
            "C": "TRE vs Foreground SDlogJ (DIR-LAB 4DCT N=10), all",
            "D": "TRE vs Foreground SDlogJ (DIR-LAB 4DCT N=10), selected",
        },
        "style_policy": {
            "x_axis_label": X_AXIS_LABEL,
            "non_reference_markers": "solid_circle",
            "ptvreg_marker": "black_star",
            "panel3_ellipses": False,
            "panel4_ellipses": True,
            "legend_note": f"{DAGGER} pTVReg loss",
            "legend_note_position": "bottom_entry",
        },
        "method_labels": {
            method: FINAL_METHOD_SPECS[method]["label"] for method in ALL_METHODS
        },
        "method_colors": {
            method: FINAL_METHOD_SPECS[method]["color"] for method in ALL_METHODS
        },
        "validation": validate_outputs(
            oasis_source=oasis_source,
            output_dir=output_dir,
            source_rows=source_rows,
            lung=lung,
        ),
    }
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_review_report(output_dir, manifest, lung)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--oasis-source", type=Path, required=True)
    parser.add_argument("--source-rows", type=Path, required=True)
    return parser.parse_args(argv)


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve the external destination and prior generated-input files."""

    args.review_root = artifact_paths.resolve_review_root(
        ARTIFACT_EXPERIMENT_KEY,
        args.review_root,
        repo_root=REPO_ROOT,
    )
    args.oasis_source = artifact_paths.resolve_external_input_file(
        args.oasis_source,
        label="--oasis-source",
        repo_root=REPO_ROOT,
    )
    args.source_rows = artifact_paths.resolve_external_input_file(
        args.source_rows,
        label="--source-rows",
        repo_root=REPO_ROOT,
    )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = resolve_artifact_paths(parse_args(argv))
    manifest = build(
        output_dir=args.review_root,
        oasis_source=args.oasis_source,
        source_rows=args.source_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
