#!/usr/bin/env python3
"""Build a side-by-side LungCT Figure 2 schedule-refresh comparison."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_SOURCE = (
    REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v002_candidate/lungct10_pareto_source_rows.csv"
)
REFRESH_SOURCE = (
    REPO_ROOT
    / "experiments/2026-08-13_lungct_fig2_mrdbscp_2500_schedule_refresh/mean_rows.csv"
)
COMPARISON_SOURCE = (
    REPO_ROOT
    / "experiments/2026-08-13_lungct_fig2_mrdbscp_2500_schedule_refresh/"
    "schedule_comparison_rows.csv"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/paper_artifacts/pareto_figure/v009_lungct_fig2_schedule_refresh_comparison"
)

X_COL = "fg_logj_std_mean"
Y_COL = "original_snap_tre_mean_mm_mean"
X_STD_COL = "fg_logj_std_std"
Y_STD_COL = "original_snap_tre_mean_mm_std"

METHOD_ORDER = [
    "idir_be_default_lungmask",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "ptvreg_reference",
]
METHOD_SPECS = {
    "idir_be_default_lungmask": {
        "label": "IDIR (IDIR loss)",
        "color": "#0072B2",
        "marker": "o",
    },
    "direct_cp_32_16_8_4_global_ncc_fd_be": {
        "label": "MR-D-BSCP (IDIR loss)",
        "color": "#009E73",
        "marker": "^",
    },
    "direct_cp_32_16_8_4_ptv_lcc_isotv": {
        "label": "MR-D-BSCP (pTVReg loss)",
        "color": "#D55E00",
        "marker": "s",
    },
    "ptvreg_reference": {
        "label": "pTVReg",
        "color": "#111111",
        "marker": "*",
    },
}
SCHEDULE_TITLES = {
    "2400": "Previous MR-D-BSCP schedule",
    "2500": "2500-step MR-D-BSCP refresh",
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


def fmt_reg(method_id: str, value: float) -> str:
    if not math.isfinite(value):
        return "default"
    if method_id == "direct_cp_32_16_8_4_global_ncc_fd_be":
        return f"a={value:g}"
    if method_id == "direct_cp_32_16_8_4_ptv_lcc_isotv":
        return f"w={value:g}"
    return f"{value:g}"


def annotation_text(row: Mapping[str, Any]) -> str:
    method_id = str(row.get("method_id"))
    if method_id == "ptvreg_reference":
        return "pTVReg"
    ratio = finite_float(row.get("ratio_to_best_tre_weight"))
    if math.isfinite(ratio):
        return f"{ratio:g}x"
    return str(row.get("reg_label") or "")


def finite_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out[X_COL] = pd.to_numeric(out[X_COL], errors="coerce")
    out[Y_COL] = pd.to_numeric(out[Y_COL], errors="coerce")
    out = out[out[X_COL].notna() & out[Y_COL].notna()].copy()
    if out.empty:
        return out
    out["sort_ratio"] = pd.to_numeric(out.get("ratio_to_best_tre_weight"), errors="coerce")
    out["sort_reg"] = pd.to_numeric(out.get("reg_value"), errors="coerce")
    return out.sort_values(["sort_ratio", "sort_reg"], na_position="last")


def selected_base_rows() -> pd.DataFrame:
    base = pd.read_csv(BASE_SOURCE)
    rows = base[
        base["method_id"].astype(str).isin({"idir_be_default_lungmask", "ptvreg_reference"})
    ].copy()
    rows["schedule"] = "unchanged"
    rows["comparison_panel"] = "both"
    rows["source_path"] = repo_relative(BASE_SOURCE)
    rows["schedule_role"] = "unchanged_reference"
    rows["display_method"] = rows["method_id"].map(lambda x: METHOD_SPECS[str(x)]["label"])
    return rows


def selected_refresh_rows() -> pd.DataFrame:
    refresh = pd.read_csv(REFRESH_SOURCE)
    refresh = refresh[refresh["schedule"].astype(str).isin({"2400", "2500"})].copy()
    refresh = refresh[
        refresh["method"].astype(str).isin(
            {
                "direct_cp_32_16_8_4_global_ncc_fd_be",
                "direct_cp_32_16_8_4_ptv_lcc_isotv",
            }
        )
    ].copy()
    out = refresh.rename(columns={"method": "method_id"}).copy()
    out["method"] = out["method_id"]
    out["method_label"] = out["method_id"].map(lambda x: METHOD_SPECS[str(x)]["label"])
    out["display_method"] = out["method_label"]
    out["reg_label"] = [fmt_reg(str(m), finite_float(v)) for m, v in zip(out["method_id"], out["reg_value"])]
    out["dataset"] = "lungct10"
    out["method_family"] = "directcp_multistage"
    out["selected_lr"] = 1e-3
    out["selected_lr_label"] = "0.001"
    out["comparison_panel"] = out["schedule"].astype(str)
    out["source_path"] = repo_relative(REFRESH_SOURCE)
    out["schedule_role"] = out["schedule"].map({"2400": "previous", "2500": "refresh"})

    ratio_values: dict[int, float] = {}
    for (_, schedule), group in out.groupby(["method_id", "schedule"], sort=False):
        best_idx = pd.to_numeric(group[Y_COL], errors="coerce").idxmin()
        best = finite_float(out.loc[best_idx, "reg_value"])
        for idx, row in group.iterrows():
            value = finite_float(row["reg_value"])
            ratio_values[int(idx)] = value / best if best else float("nan")
    out["ratio_to_best_tre_weight"] = out.index.map(lambda idx: ratio_values.get(int(idx), float("nan")))
    return out


def build_plot_rows() -> pd.DataFrame:
    base = selected_base_rows()
    refresh = selected_refresh_rows()
    rows = pd.concat([base, refresh], ignore_index=True, sort=False)
    rows = rows[rows["cases"].astype(int) == 10].copy()
    return rows


def rows_for_schedule(rows: pd.DataFrame, schedule: str) -> pd.DataFrame:
    return rows[(rows["comparison_panel"].isin(["both", schedule]))].copy()


def draw_panel(ax: plt.Axes, rows: pd.DataFrame, *, title: str, annotate: bool) -> None:
    for method_id in METHOD_ORDER:
        method_rows = rows[rows["method_id"].astype(str) == method_id].copy()
        method_rows = finite_rows(method_rows)
        if method_rows.empty:
            continue
        spec = METHOD_SPECS[method_id]
        color = str(spec["color"])
        marker = str(spec["marker"])
        xs = method_rows[X_COL].to_numpy(dtype=float)
        ys = method_rows[Y_COL].to_numpy(dtype=float)
        if method_id != "ptvreg_reference" and len(xs) > 1:
            ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.78, zorder=2)
        for _, row in method_rows.iterrows():
            x = finite_float(row.get(X_COL))
            y = finite_float(row.get(Y_COL))
            x_std = max(finite_float(row.get(X_STD_COL)), 0.0)
            y_std = max(finite_float(row.get(Y_STD_COL)), 0.0)
            if math.isfinite(x_std) and math.isfinite(y_std) and (x_std > 0 or y_std > 0):
                ax.add_patch(
                    Ellipse(
                        (x, y),
                        width=2.0 * x_std,
                        height=2.0 * y_std,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.12 if method_id != "ptvreg_reference" else 0.18,
                        linewidth=1.0,
                        zorder=1,
                    )
                )
        size = 120 if method_id == "ptvreg_reference" else 52
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
            for _, row in method_rows.iterrows():
                x = finite_float(row.get(X_COL))
                y = finite_float(row.get(Y_COL))
                if math.isfinite(x) and math.isfinite(y):
                    ax.annotate(
                        annotation_text(row),
                        (x, y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Foreground stdLogJ", fontsize=11)
    ax.set_ylabel("Original snap TRE (mm)", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=7.5, frameon=True, loc="best")


def plot_comparison(rows: pd.DataFrame, output_path: Path) -> Path:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True, sharey=True)
    panels = [
        ("2400", False),
        ("2500", False),
        ("2400", True),
        ("2500", True),
    ]
    for ax, (schedule, annotate) in zip(axes.ravel(), panels):
        panel_rows = rows_for_schedule(rows, schedule)
        suffix = "annotated" if annotate else "unannotated"
        draw_panel(ax, panel_rows, title=f"{SCHEDULE_TITLES[schedule]} ({suffix})", annotate=annotate)

    finite = rows[pd.to_numeric(rows[X_COL], errors="coerce").notna() & pd.to_numeric(rows[Y_COL], errors="coerce").notna()]
    x = pd.to_numeric(finite[X_COL], errors="coerce")
    y = pd.to_numeric(finite[Y_COL], errors="coerce")
    x_pad = max((x.max() - x.min()) * 0.08, 0.02)
    y_pad = max((y.max() - y.min()) * 0.08, 0.03)
    for ax in axes.ravel():
        ax.set_xlim(float(x.min() - x_pad), float(x.max() + x_pad))
        ax.set_ylim(float(y.min() - y_pad), float(y.max() + y_pad))

    fig.suptitle("LungCT Figure 2 left-bottom panel: previous vs 2500-step MR-D-BSCP rows", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def write_report(rows: pd.DataFrame, output_dir: Path) -> None:
    comp = pd.read_csv(COMPARISON_SOURCE)
    comp["family_label"] = comp["family"].map({"idir_fdbe": "MR-D-BSCP (IDIR loss)", "ptv": "MR-D-BSCP (pTVReg loss)"})
    comp["reg_label"] = [
        fmt_reg(
            "direct_cp_32_16_8_4_global_ncc_fd_be" if family == "idir_fdbe" else "direct_cp_32_16_8_4_ptv_lcc_isotv",
            finite_float(value),
        )
        for family, value in zip(comp["family"], comp["reg_value"])
    ]
    lines = [
        "# LungCT Figure 2 Schedule Refresh Plot Comparison",
        "",
        "This review compares the previous 2400-step MR-D-BSCP rows against the",
        "new 2500-step schedule rows in the LungCT Figure 2 TRE/stdLogJ panel.",
        "",
        "The unchanged rows are IDIR (IDIR loss) and pTVReg. The swapped rows are",
        "MR-D-BSCP (IDIR loss) and MR-D-BSCP (pTVReg loss).",
        "",
        "## Outputs",
        "",
        "- `lungct_fig2_left_bottom_schedule_refresh_2x2.png`",
        "- `source_rows.csv`",
        "- `manifest.json`",
        "",
        "## Headline Deltas",
        "",
        "| Method | Reg | Old TRE | New TRE | Delta | Old stdLogJ | New stdLogJ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comp.iterrows():
        lines.append(
            "| {method} | {reg} | {old:.3f} | {new:.3f} | {delta:+.3f} | {oldx:.6f} | {newx:.6f} |".format(
                method=row["family_label"],
                reg=row["reg_label"],
                old=finite_float(row["old_original_snap_tre_mean_mm"]),
                new=finite_float(row["new_original_snap_tre_mean_mm"]),
                delta=finite_float(row["delta_original_snap_tre_mean_mm"]),
                oldx=finite_float(row["old_fg_logj_std_mean"]),
                newx=finite_float(row["new_fg_logj_std_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Both columns use identical x/y limits.",
            "- Ellipse full width/height is `2 * sample SD`, matching the current",
            "  LungCT panel convention.",
            "- Top row has no point annotations. Bottom row labels hyperparameter",
            "  ratio relative to the best TRE point within each displayed curve.",
            "- This artifact does not update Figure 2 plotting source.",
        ]
    )
    (output_dir / "review_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_plot_rows()
    rows.to_csv(OUTPUT_DIR / "source_rows.csv", index=False)
    plot_path = plot_comparison(rows, OUTPUT_DIR / "lungct_fig2_left_bottom_schedule_refresh_2x2.png")
    manifest = {
        "artifact": "lungct_fig2_left_bottom_schedule_refresh_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "unchanged_idir_ptvreg": repo_relative(BASE_SOURCE),
            "mrdbscp_old_new_mean_rows": repo_relative(REFRESH_SOURCE),
            "old_new_comparison": repo_relative(COMPARISON_SOURCE),
        },
        "outputs": {
            "figure": repo_relative(plot_path),
            "source_rows": repo_relative(OUTPUT_DIR / "source_rows.csv"),
            "review_report": repo_relative(OUTPUT_DIR / "review_report.md"),
        },
        "panels": {
            "top_left": "previous 2400 schedule, unannotated",
            "top_right": "2500 schedule refresh, unannotated",
            "bottom_left": "previous 2400 schedule, annotated",
            "bottom_right": "2500 schedule refresh, annotated",
        },
        "x_axis": "foreground stdLogJ",
        "y_axis": "original snap TRE in mm",
        "ellipse_definition": "full width/height equals 2 * sample SD",
        "figure2_source_updated": False,
        "row_count": int(len(rows)),
    }
    write_report(rows, OUTPUT_DIR)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_dir": repo_relative(OUTPUT_DIR), "figure": repo_relative(plot_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
