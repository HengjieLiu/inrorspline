"""Build paper-figure curves for fit-mode method comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REVIEW_ROOT = Path(__file__).resolve().parent
OASIS_REVIEW = REVIEW_ROOT.parent / "2026-06-26_oasis_rank009_sitreg_fit_modes_lr_sweep"
LUNG01_REVIEW = REVIEW_ROOT.parent / "2026-06-26_lungct_case01_ptvreg_fit_modes_lr_sweep"
LUNG08_REVIEW = REVIEW_ROOT.parent / "2026-06-26_lungct_case08_ptvreg_fit_modes_lr_sweep"
IDIR_EVAL_ROOT = Path("/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1")
LUNG_4STAGE_FINEST_DIAG_ROOT = (
    Path("/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_v1")
    / "direct_cp_32_16_8_4_ptv_lcc_isotv"
)
LUNG_4STAGE_FIELD_MSE_DIAG_ROOT = Path(
    "/database/inrorspline/runs/fit_modes_paper_curves_directcp4stage_field_mse_diag_v1"
)

CASE_ORDER = ["OASIS rank-9", "LungCT case01", "LungCT case08"]
METHOD_ORDER = ["IDIR", "SINR cps4", "DirectCP cps4", "DirectCP 4stage cps4"]
METHOD_STYLE = {
    "IDIR": {"color": "#2f5d9e", "linestyle": "-", "marker": "o", "zorder": 5},
    "SINR cps4": {"color": "#0b7a53", "linestyle": "-", "marker": None, "zorder": 4},
    "DirectCP cps4": {"color": "#b23a48", "linestyle": "-", "marker": None, "zorder": 3},
    "DirectCP 4stage cps4": {"color": "#6f4aa2", "linestyle": "--", "marker": None, "zorder": 2},
}


@dataclass(frozen=True)
class Curve:
    case: str
    task: str
    method: str
    lr: float
    epochs: np.ndarray
    values: np.ndarray
    y_label: str
    source: str
    note: str


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _first_row(df: pd.DataFrame, mask: pd.Series, what: str) -> pd.Series:
    subset = df.loc[mask].copy()
    if subset.empty:
        raise ValueError(f"No row matched {what}")
    return subset.iloc[0]


def _select_min(df: pd.DataFrame, metric: str) -> pd.Series:
    subset = df.dropna(subset=[metric]).copy()
    if subset.empty:
        raise ValueError(f"No finite values for {metric}")
    return subset.sort_values(metric, ascending=True).iloc[0]


def _select_max(df: pd.DataFrame, metric: str) -> pd.Series:
    subset = df.dropna(subset=[metric]).copy()
    if subset.empty:
        raise ValueError(f"No finite values for {metric}")
    return subset.sort_values(metric, ascending=False).iloc[0]


def _history_from_run_dir(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    stage = run_dir / "stage_history.csv"
    hist = run_dir / "history.csv"
    loss = run_dir / "loss_history.csv"
    if stage.exists():
        df = pd.read_csv(stage)
        if "global_epoch" not in df.columns:
            df["global_epoch"] = np.arange(1, len(df) + 1, dtype=float)
        return df
    if hist.exists():
        df = pd.read_csv(hist)
        df["global_epoch"] = df.get("global_epoch", df["epoch"])
        return df
    if loss.exists():
        df = pd.read_csv(loss)
        df["global_epoch"] = df.get("global_epoch", df["epoch"])
        return df
    raise FileNotFoundError(f"No history CSV found in {run_dir}")


def _lung_4stage_field_diag_run_dir(case_key: str) -> Path:
    return (
        LUNG_4STAGE_FIELD_MSE_DIAG_ROOT
        / f"lungct_{case_key}"
        / "direct_cp_4stage_ptvreg_field_mse"
        / (
            f"ptvreg_field_fit_direct_cp_4stage_{case_key}_finestmsediag_"
            "sched32-16-8-4_e400-400-800-900_lr0p01"
        )
    )


def _curve_from_dataframe(
    df: pd.DataFrame,
    *,
    case: str,
    task: str,
    method: str,
    lr: float,
    y_col: str,
    y_label: str,
    source: str,
    note: str,
    y_scale: float = 1.0,
) -> Curve:
    if y_col not in df.columns:
        raise KeyError(f"{y_col} not in {source}")
    epoch_col = "global_epoch" if "global_epoch" in df.columns else "epoch"
    rows = df[[epoch_col, y_col]].dropna().copy()
    rows = rows.sort_values(epoch_col)
    return Curve(
        case=case,
        task=task,
        method=method,
        lr=float(lr),
        epochs=rows[epoch_col].to_numpy(float),
        values=rows[y_col].to_numpy(float) * y_scale,
        y_label=y_label,
        source=source,
        note=note,
    )


def selected_runs() -> pd.DataFrame:
    """Return the explicit LR/run selection used for the notebook figure."""
    rows: list[dict[str, object]] = []

    oasis_summary = pd.read_csv(OASIS_REVIEW / "final_metric_summary.csv")
    for method in ["IDIR", "SINR cps4", "DirectCP cps4"]:
        subset = oasis_summary[
            (oasis_summary["objective_group"] == "image_registration")
            & (oasis_summary["method_label"] == method)
        ]
        if method == "IDIR":
            row = _first_row(subset, subset["lr"].eq(1e-4), "OASIS IDIR reg lr=1e-4")
            criterion = "fixed to matched-eval rerun"
        else:
            row = _select_max(subset, "dice_mean")
            criterion = "max final Dice among completed LR sweep"
        rows.append(
            {
                "case": "OASIS rank-9",
                "task": "registration",
                "method": method,
                "lr": row["lr"],
                "selection_metric": criterion,
                "selection_value": row.get("dice_mean", np.nan),
                "curve_metric": "global NCC loss",
                "run_dir": row["summary_path"],
            }
        )

    for method in ["IDIR", "SINR cps4", "DirectCP cps4"]:
        subset = oasis_summary[
            (oasis_summary["objective_group"] == "sitreg_field_fit")
            & (oasis_summary["method_label"] == method)
        ]
        row = _select_min(subset, "field_mse_objective_normalized_grid")
        rows.append(
            {
                "case": "OASIS rank-9",
                "task": "SITReg field fit",
                "method": method,
                "lr": row["lr"],
                "selection_metric": "min objective field MSE",
                "selection_value": row["field_mse_objective_normalized_grid"],
                "curve_metric": "masked normalized-grid MSE",
                "run_dir": row["summary_path"],
            }
        )

    for case, review in [("LungCT case01", LUNG01_REVIEW), ("LungCT case08", LUNG08_REVIEW)]:
        case_key = "case01" if case.endswith("case01") else "case08"
        summary = pd.read_csv(review / "summary_runs.csv")
        for method in ["IDIR", "SINR cps4", "DirectCP cps4", "DirectCP 4stage cps4"]:
            subset = summary[
                (summary["objective_group"] == "image_registration")
                & (summary["method_label"] == method)
            ]
            if method == "IDIR":
                row = _first_row(subset, subset["lr"].eq(1e-4), f"{case} IDIR reg lr=1e-4")
                criterion = "fixed to matched-eval rerun"
            elif method == "DirectCP 4stage cps4":
                row = _first_row(subset, subset["lr"].eq(1e-2), f"{case} DirectCP 4-stage finest diagnostic lr=1e-2")
                criterion = "fixed to existing finest-domain diagnostic"
            else:
                row = _select_min(subset, "native_crop_snap_tre_mean_mm")
                criterion = "min final native-crop snap TRE"
            run_dir = row["run_dir"]
            if method == "DirectCP 4stage cps4":
                run_dir = (
                    LUNG_4STAGE_FINEST_DIAG_ROOT
                    / f"image_reg_direct_cp_4stage_{case_key}_finestdiag_sched32-16-8-4_e400-400-800-900_lr0p01"
                )
            rows.append(
                {
                    "case": case,
                    "task": "registration",
                    "method": method,
                    "lr": row["lr"],
                    "selection_metric": criterion,
                    "selection_value": row["native_crop_snap_tre_mean_mm"],
                    "curve_metric": "pTV LCC data loss",
                    "run_dir": run_dir,
                }
            )

        for method in ["IDIR", "SINR cps4", "DirectCP cps4", "DirectCP 4stage cps4"]:
            subset = summary[
                (summary["objective_group"] == "ptvreg_field_fit")
                & (summary["method_label"] == method)
            ]
            row = _select_min(subset, "field_mse_objective_normalized_grid")
            criterion = "min objective field MSE"
            run_dir = row["run_dir"]
            if method == "DirectCP 4stage cps4":
                row = _first_row(subset, subset["lr"].eq(1e-2), f"{case} DirectCP 4-stage field diagnostic lr=1e-2")
                criterion = "fixed to rerun finest-domain diagnostic"
                run_dir = _lung_4stage_field_diag_run_dir("case01" if case.endswith("case01") else "case08")
            rows.append(
                {
                    "case": case,
                    "task": "pTVReg field fit",
                    "method": method,
                    "lr": row["lr"],
                    "selection_metric": criterion,
                    "selection_value": row["field_mse_objective_normalized_grid"],
                    "curve_metric": "masked normalized-grid MSE",
                    "run_dir": run_dir,
                }
            )

    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    out["case"] = pd.Categorical(out["case"], CASE_ORDER, ordered=True)
    out = out.sort_values(["case", "task", "method"]).reset_index(drop=True)
    return out


def load_curves(selection: pd.DataFrame | None = None) -> tuple[list[Curve], list[Curve]]:
    if selection is None:
        selection = selected_runs()

    reg_curves: list[Curve] = []
    field_curves: list[Curve] = []

    oasis_loss = pd.read_csv(OASIS_REVIEW / "loss_history_augmented.csv")
    oasis_field = pd.read_csv(OASIS_REVIEW / "loss_history_all.csv")

    idir_oasis = pd.read_csv(
        IDIR_EVAL_ROOT
        / "oasis_rank009/idir_original_dense_siren_be_autograd/"
        / "oasis_rank009_idir_a1_lr1e-4_global_ncc_eval/image_similarity_eval.csv"
    )
    reg_curves.append(
        _curve_from_dataframe(
            idir_oasis,
            case="OASIS rank-9",
            task="registration",
            method="IDIR",
            lr=1e-4,
            y_col="similarity_loss",
            y_label="Global NCC loss",
            source="IDIR matched eval CSV",
            note="IDIR matched global NCC evaluated every 10 epochs.",
        )
    )

    for method in ["SINR cps4", "DirectCP cps4"]:
        row = selection[
            (selection["case"].astype(str) == "OASIS rank-9")
            & (selection["task"] == "registration")
            & (selection["method"].astype(str) == method)
        ].iloc[0]
        hist = oasis_loss[
            (oasis_loss["objective_group"] == "image_registration")
            & (oasis_loss["method_label"] == method)
            & np.isclose(oasis_loss["lr"].astype(float), float(row["lr"]))
        ]
        reg_curves.append(
            _curve_from_dataframe(
                hist,
                case="OASIS rank-9",
                task="registration",
                method=method,
                lr=float(row["lr"]),
                y_col="data_loss",
                y_label="Global NCC loss",
                source=str(row["run_dir"]),
                note="Per-epoch SINR global NCC data loss.",
            )
        )

    for method in ["IDIR", "SINR cps4", "DirectCP cps4"]:
        row = selection[
            (selection["case"].astype(str) == "OASIS rank-9")
            & (selection["task"] == "SITReg field fit")
            & (selection["method"].astype(str) == method)
        ].iloc[0]
        hist = oasis_field[
            (oasis_field["objective_group"] == "sitreg_field_fit")
            & (oasis_field["method_label"] == method)
            & np.isclose(oasis_field["lr"].astype(float), float(row["lr"]))
        ]
        field_curves.append(
            _curve_from_dataframe(
                hist,
                case="OASIS rank-9",
                task="SITReg field fit",
                method=method,
                lr=float(row["lr"]),
                y_col="field_mse_objective_normalized_grid",
                y_label="SITReg field MSE",
                source=str(row["run_dir"]),
                note="Masked normalized-grid target-field MSE.",
            )
        )

    for case, case_key in [("LungCT case01", "lungct_case01"), ("LungCT case08", "lungct_case08")]:
        idir_csv = (
            IDIR_EVAL_ROOT
            / case_key
            / "idir_be_default_lungmask"
            / f"{case_key}_idir_lr1e-4_ptv_lcc_eval/image_similarity_eval.csv"
        )
        idir_eval = pd.read_csv(idir_csv)
        reg_curves.append(
            _curve_from_dataframe(
                idir_eval,
                case=case,
                task="registration",
                method="IDIR",
                lr=1e-4,
                y_col="similarity_loss",
                y_label="pTV LCC loss (x1e6)",
                source=str(idir_csv),
                note="IDIR matched pTV LCC evaluated every 10 epochs.",
                y_scale=1e-6,
            )
        )

        for method in ["SINR cps4", "DirectCP cps4", "DirectCP 4stage cps4"]:
            row = selection[
                (selection["case"].astype(str) == case)
                & (selection["task"] == "registration")
                & (selection["method"].astype(str) == method)
            ].iloc[0]
            hist = _history_from_run_dir(row["run_dir"])
            y_col = "finest_data_loss" if method == "DirectCP 4stage cps4" else "data_loss"
            note = (
                "Posthoc finest-domain pTV LCC diagnostic; training still used stage-domain loss."
                if method == "DirectCP 4stage cps4"
                else "Per-epoch pTV LCC data loss."
            )
            reg_curves.append(
                _curve_from_dataframe(
                    hist,
                    case=case,
                    task="registration",
                    method=method,
                    lr=float(row["lr"]),
                    y_col=y_col,
                    y_label="pTV LCC loss (x1e6)",
                    source=str(row["run_dir"]),
                    note=note,
                    y_scale=1e-6,
                )
            )

        for method in ["IDIR", "SINR cps4", "DirectCP cps4", "DirectCP 4stage cps4"]:
            row = selection[
                (selection["case"].astype(str) == case)
                & (selection["task"] == "pTVReg field fit")
                & (selection["method"].astype(str) == method)
            ].iloc[0]
            hist = _history_from_run_dir(row["run_dir"])
            note = "Masked normalized-grid target-field MSE."
            y_col = "field_mse_objective_normalized_grid"
            if method == "DirectCP 4stage cps4":
                y_col = "finest_data_loss"
                note = (
                    "Posthoc finest-domain masked normalized-grid MSE; coarse stages "
                    "evaluated every 50 epochs and final stage every epoch."
                )
            field_curves.append(
                _curve_from_dataframe(
                    hist,
                    case=case,
                    task="pTVReg field fit",
                    method=method,
                    lr=float(row["lr"]),
                    y_col=y_col,
                    y_label="pTVReg field MSE",
                    source=str(row["run_dir"]),
                    note=note,
                )
            )

    return reg_curves, field_curves


def curve_summary(curves: list[Curve]) -> pd.DataFrame:
    rows = []
    for curve in curves:
        rows.append(
            {
                "case": curve.case,
                "task": curve.task,
                "method": curve.method,
                "lr": curve.lr,
                "points": len(curve.epochs),
                "epoch_min": float(np.nanmin(curve.epochs)),
                "epoch_max": float(np.nanmax(curve.epochs)),
                "value_start": float(curve.values[0]),
                "value_end": float(curve.values[-1]),
                "source": curve.source,
                "note": curve.note,
            }
        )
    return pd.DataFrame(rows)


def _smooth(values: np.ndarray, points: int) -> np.ndarray:
    if points <= 1 or len(values) < 3:
        return values
    window = min(points, len(values))
    return pd.Series(values).rolling(window=window, min_periods=1, center=True).median().to_numpy()


def make_main_figure(
    reg_curves: list[Curve],
    field_curves: list[Curve],
    *,
    smooth: bool = True,
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.25), sharex=True, constrained_layout=True)
    panel_titles = {
        "OASIS rank-9": "OASIS rank-9",
        "LungCT case01": "LungCT case01",
        "LungCT case08": "LungCT case08",
    }

    for col, case in enumerate(CASE_ORDER):
        ax = axes[0, col]
        for curve in [c for c in reg_curves if c.case == case]:
            style = METHOD_STYLE[curve.method]
            values = _smooth(curve.values, 3 if curve.method == "IDIR" else 25) if smooth else curve.values
            marker = style["marker"] if curve.method == "IDIR" else None
            markevery = max(1, len(curve.epochs) // 8) if marker else None
            ax.plot(
                curve.epochs,
                values,
                label=curve.method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                marker=marker,
                markersize=2.0,
                markevery=markevery,
                zorder=style["zorder"],
            )
        ax.set_title(panel_titles[case])
        ax.set_ylabel("Global NCC loss" if case == "OASIS rank-9" else "pTV LCC loss (x1e6)")
        ax.grid(True, color="#d8dce2", linewidth=0.5, alpha=0.75)

        ax = axes[1, col]
        for curve in [c for c in field_curves if c.case == case]:
            style = METHOD_STYLE[curve.method]
            values = _smooth(curve.values, 3 if curve.method == "IDIR" else 25) if smooth else curve.values
            marker = style["marker"] if curve.method == "IDIR" else None
            markevery = max(1, len(curve.epochs) // 8) if marker else None
            ax.plot(
                curve.epochs,
                values,
                label=curve.method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                marker=marker,
                markersize=2.0,
                markevery=markevery,
                zorder=style["zorder"],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Target-field MSE")
        ax.grid(True, which="both", color="#d8dce2", linewidth=0.5, alpha=0.75)

    axes[0, 0].text(
        -0.12,
        1.07,
        "A",
        transform=axes[0, 0].transAxes,
        fontweight="bold",
        fontsize=10,
    )
    axes[1, 0].text(
        -0.12,
        1.07,
        "B",
        transform=axes[1, 0].transAxes,
        fontweight="bold",
        fontsize=10,
    )
    axes[0, 0].annotate("Image registration", xy=(0, 1), xycoords="axes fraction", xytext=(0, 16), textcoords="offset points")
    axes[1, 0].annotate("Target-field fitting", xy=(0, 1), xycoords="axes fraction", xytext=(0, 16), textcoords="offset points")

    for ax in axes.ravel():
        ax.set_xlim(0, 2500)
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            linestyle=METHOD_STYLE[method]["linestyle"],
            marker=METHOD_STYLE[method]["marker"] or None,
            markersize=3,
            linewidth=1.8,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        frameon=False,
    )
    return fig


def save_main_figure(fig: plt.Figure, stem: str = "fit_modes_method_comparison_curves") -> dict[str, Path]:
    outputs = {
        "pdf": REVIEW_ROOT / f"{stem}.pdf",
        "png": REVIEW_ROOT / f"{stem}.png",
    }
    for path in outputs.values():
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return outputs


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure, dict[str, Path]]:
    selection = selected_runs()
    reg_curves, field_curves = load_curves(selection)
    summary = pd.concat(
        [
            curve_summary(reg_curves).assign(curve_group="registration"),
            curve_summary(field_curves).assign(curve_group="field_fit"),
        ],
        ignore_index=True,
    )
    selection.to_csv(REVIEW_ROOT / "fit_modes_method_comparison_selected_runs.csv", index=False)
    summary.to_csv(REVIEW_ROOT / "fit_modes_method_comparison_curve_summary.csv", index=False)
    fig = make_main_figure(reg_curves, field_curves)
    outputs = save_main_figure(fig)
    return selection, summary, fig, outputs
