"""Build paper-figure curves for fit-mode method comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REVIEW_ROOT = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.paper_artifacts.paper_artifact_common import (  # noqa: E402
    artifact_output_dir,
    display_name as artifact_display_name,
    format_lr,
    load_nicknames,
    load_standard,
    sanitize_public_value,
    selected_hypervariant_records,
    validate_lr_records,
    validation_report_markdown,
)

OASIS_REVIEW = REVIEW_ROOT.parent / "2026-06-26_oasis_rank009_sitreg_fit_modes_lr_sweep"
OASIS_GREEDY_REVIEW = REVIEW_ROOT.parent / "2026-06-26_oasis_rank009_fit_modes_lr_sweep"
LUNG01_REVIEW = REVIEW_ROOT.parent / "2026-06-26_lungct_case01_ptvreg_fit_modes_lr_sweep"
LUNG08_REVIEW = REVIEW_ROOT.parent / "2026-06-26_lungct_case08_ptvreg_fit_modes_lr_sweep"
IDIR_EVAL_ROOT = Path("/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1")
PANELA_GLOBAL_NCC_ROOT = Path("/database/inrorspline/runs/fit_modes_panelA_global_ncc_v1")
OASIS_842_FIELD_FIT_ROOT = Path("/database/inrorspline/runs/fit_modes_oasis_directcp842_field_fit_v1")
LUNG_4STAGE_FINEST_DIAG_ROOT = (
    Path("/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_v1")
    / "direct_cp_32_16_8_4_ptv_lcc_isotv"
)
LUNG_4STAGE_FIELD_MSE_DIAG_ROOT = Path(
    "/database/inrorspline/runs/fit_modes_paper_curves_directcp4stage_field_mse_diag_v1"
)

CASE_ORDER = ["OASIS rank-9", "LungCT case01", "LungCT case08"]
PANELA_CASE_ORDER = CASE_ORDER
PANELA_PLOT_METHOD_ORDER = [
    "panelA_idir",
    "panelA_sinr_cps4",
    "panelA_direct_cp_cps4",
    "panelA_direct_cp_multistage",
]
PANELA_STYLE_METHOD_BY_PLOT_ID = {
    "panelA_idir": "IDIR",
    "panelA_sinr_cps4": "SINR cps4",
    "panelA_direct_cp_cps4": "DirectCP cps4",
    "panelA_direct_cp_multistage": "DirectCP 4stage cps4",
}
PANELA_PLOT_ID_BY_SOURCE_METHOD = {
    "idir_original_dense_siren_be_autograd": "panelA_idir",
    "idir_be_default_lungmask": "panelA_idir",
    "sinr_cps4_be_bspline_analytic": "panelA_sinr_cps4",
    "sinr_cps4_global_ncc_fd_be": "panelA_sinr_cps4",
    "direct_cp_cps4_be_bspline_analytic": "panelA_direct_cp_cps4",
    "direct_cp_cps4_global_ncc_fd_be": "panelA_direct_cp_cps4",
    "direct_cp_8_4_2_be_bspline_analytic_multistage": "panelA_direct_cp_multistage",
    "direct_cp_32_16_8_4_global_ncc_fd_be": "panelA_direct_cp_multistage",
}
RELATIVE_FIELD_CASE_ORDER = ["OASIS SITReg", "OASIS Greedy", "LungCT case01", "LungCT case08"]
METHOD_ORDER = ["IDIR", "SINR cps4", "DirectCP cps4", "DirectCP 4stage cps4"]
STANDARDIZED_FIELD_FIT_METHOD_ORDER = [
    "IDIR",
    "SINR cps4",
    "DirectCP cps4",
    "DirectCP 8-4-2",
    "DirectCP 4stage cps4",
]
METHOD_STYLE = {
    "IDIR": {"color": "#2C7BB6", "linestyle": "-", "marker": None, "zorder": 5},
    "SINR cps4": {"color": "#F28E2B", "linestyle": "-", "marker": None, "zorder": 4},
    "DirectCP cps4": {"color": "#D62728", "linestyle": "-", "marker": None, "zorder": 3},
    "DirectCP 8-4-2": {"color": "#CC79A7", "linestyle": "-", "marker": None, "zorder": 2},
    "DirectCP 4stage cps4": {"color": "#CC79A7", "linestyle": "-", "marker": None, "zorder": 2},
}
LR_TRAJECTORY_METHOD_LRS = {
    "IDIR": [1e-3, 1e-4],
    "SINR cps4": [1e-3, 1e-4],
    "DirectCP cps4": [1e-2, 1e-3, 1e-4],
    "DirectCP 4stage cps4": [1e-2, 1e-3, 1e-4],
}
LR_TRAJECTORY_LR_ORDER = [1e-2, 1e-3, 1e-4]
LR_TRAJECTORY_STYLE = {
    1e-2: {"color": "#d55e00", "linestyle": "-", "label": "lr 1e-2"},
    1e-3: {"color": "#0072b2", "linestyle": "-", "label": "lr 1e-3"},
    1e-4: {"color": "#009e73", "linestyle": "-", "label": "lr 1e-4"},
}
TARGET_POWER_EPS = 1.0e-12
PRECHECK_TARGET_POWER_BY_CASE = {
    "OASIS rank-9": 5.664249e-04,
    "OASIS SITReg": 5.664249e-04,
    "OASIS Greedy": 6.675104e-04,
    "LungCT case01": 7.440060e-04,
    "LungCT case08": 7.239232e-03,
}
MERGED_PANELB_OASIS_TARGET_OPTIONS = ("OASIS SITReg", "OASIS Greedy")
PANELAB_BASE_AXIS_LABEL_SIZE = 8
PANELAB_AXIS_LABEL_SIZE = 12
PANELAB_BASE_LEGEND_FONT_SIZE = 7
PANELAB_LEGEND_FONT_SIZE = 14
PANELAB_BASE_LEGEND_ANCHOR_Y = -0.025
PANELAB_LEGEND_ANCHOR_Y = -0.100


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


@dataclass(frozen=True)
class PanelACurve:
    case: str
    source_method_id: str
    plot_method_id: str
    lr: float
    epochs: np.ndarray
    values: np.ndarray
    source: str
    optimized_loss: str
    diagnostic_loss: str
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


def _format_lr_tag(lr: float) -> str:
    known = {
        1e-2: "0p01",
        1e-3: "0p001",
        1e-4: "0p0001",
    }
    for value, tag in known.items():
        if np.isclose(float(lr), value):
            return tag
    return f"{float(lr):g}".replace("-", "m").replace(".", "p")


def _format_lr_label(lr: float) -> str:
    known = {
        1e-2: "1e-2",
        1e-3: "1e-3",
        1e-4: "1e-4",
    }
    for value, label in known.items():
        if np.isclose(float(lr), value):
            return label
    return f"{float(lr):g}"


def _lung_4stage_field_diag_run_dir(case_key: str, lr: float = 1e-2) -> Path:
    lr_tag = _format_lr_tag(lr)
    return (
        LUNG_4STAGE_FIELD_MSE_DIAG_ROOT
        / f"lungct_{case_key}"
        / "direct_cp_4stage_ptvreg_field_mse"
        / (
            f"ptvreg_field_fit_direct_cp_4stage_{case_key}_finestmsediag_"
            f"sched32-16-8-4_e400-400-800-900_lr{lr_tag}"
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
    out["method"] = pd.Categorical(out["method"], STANDARDIZED_FIELD_FIT_METHOD_ORDER, ordered=True)
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


def _with_case(curve: Curve, case: str, task: str | None = None) -> Curve:
    return Curve(
        case=case,
        task=curve.task if task is None else task,
        method=curve.method,
        lr=curve.lr,
        epochs=curve.epochs,
        values=curve.values,
        y_label=curve.y_label,
        source=curve.source,
        note=curve.note,
    )


def load_oasis_greedy_field_curves() -> list[Curve]:
    summary = pd.read_csv(OASIS_GREEDY_REVIEW / "final_metric_summary.csv")
    history = pd.read_csv(OASIS_GREEDY_REVIEW / "loss_history_all.csv")
    curves: list[Curve] = []
    for method in ["IDIR", "SINR cps4", "DirectCP cps4"]:
        subset = summary[
            (summary["objective_group"] == "greedy_field_fit")
            & (summary["method_label"] == method)
        ]
        row = _select_min(subset, "field_mse_objective_normalized_grid")
        hist = history[
            (history["objective_group"] == "greedy_field_fit")
            & (history["method_label"] == method)
            & np.isclose(history["lr"].astype(float), float(row["lr"]))
        ]
        curves.append(
            _curve_from_dataframe(
                hist,
                case="OASIS Greedy",
                task="Greedy field fit",
                method=method,
                lr=float(row["lr"]),
                y_col="field_mse_objective_normalized_grid",
                y_label="Greedy field MSE",
                source=str(row["summary_path"]),
                note="Masked normalized-grid target-field MSE against canonical Greedy.",
            )
        )
    return curves


def load_relative_field_fit_curves(selection: pd.DataFrame | None = None) -> list[Curve]:
    _, field_curves = load_curves(selection)
    curves: list[Curve] = []
    for curve in field_curves:
        if curve.case == "OASIS rank-9" and curve.task == "SITReg field fit":
            curves.append(_with_case(curve, "OASIS SITReg", "SITReg field fit"))
        else:
            curves.append(curve)
    curves.extend(load_oasis_greedy_field_curves())
    return curves


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


def target_power_summary(
    field_curves: list[Curve],
    case_order: list[str] | None = None,
) -> pd.DataFrame:
    """Infer target-field power from the zero-control DirectCP cps4 first point."""
    rows: list[dict[str, object]] = []
    if case_order is None:
        case_order = list(dict.fromkeys(curve.case for curve in field_curves))
    for case in case_order:
        matches = [curve for curve in field_curves if curve.case == case and curve.method == "DirectCP cps4"]
        if not matches:
            raise ValueError(f"No DirectCP cps4 field-fit curve found for {case}")
        curve = matches[0]
        target_power = float(curve.values[0])
        if not np.isfinite(target_power) or target_power <= 0.0:
            raise ValueError(f"Invalid target power inferred for {case}: {target_power}")
        precheck = float(PRECHECK_TARGET_POWER_BY_CASE.get(case, np.nan))
        relative_difference = (
            float(abs(target_power - precheck) / precheck)
            if np.isfinite(precheck) and precheck > 0.0
            else np.nan
        )
        rows.append(
            {
                "case": case,
                "target_power_component_mean": target_power,
                "target_rms_component": float(np.sqrt(target_power)),
                "target_power_precheck_component_mean": precheck,
                "relative_difference_vs_precheck": relative_difference,
                "target_power_units": "normalized_grid_displacement_squared",
                "target_power_domain": curve.task,
                "target_power_source": (
                    "DirectCP cps4 epoch-1 masked normalized-grid MSE; "
                    "zero-control initialization makes this mean(target_u ** 2)"
                ),
                "eps": TARGET_POWER_EPS,
            }
        )
    return pd.DataFrame(rows)


def _target_power_map(power_summary: pd.DataFrame) -> dict[str, float]:
    required = {"case", "target_power_component_mean"}
    missing = required - set(power_summary.columns)
    if missing:
        raise KeyError(f"target power summary missing columns: {sorted(missing)}")
    return {
        str(row["case"]): float(row["target_power_component_mean"])
        for _, row in power_summary.iterrows()
    }


def _relative_mse_values(curve: Curve, target_power: float) -> np.ndarray:
    return curve.values / (target_power + TARGET_POWER_EPS)


def _relative_rrmse_percent_values(curve: Curve, target_power: float) -> np.ndarray:
    return 100.0 * np.sqrt(_relative_mse_values(curve, target_power))


def _value_at_epoch(epochs: np.ndarray, values: np.ndarray, epoch: float) -> float:
    matches = np.where(np.isclose(epochs.astype(float), float(epoch)))[0]
    if matches.size == 0:
        return float("nan")
    return float(values[int(matches[0])])


def relative_field_curve_summary(field_curves: list[Curve], power_summary: pd.DataFrame) -> pd.DataFrame:
    power_by_case = _target_power_map(power_summary)
    rows: list[dict[str, object]] = []
    for curve in field_curves:
        target_power = power_by_case[curve.case]
        rmse_values = _relative_mse_values(curve, target_power)
        rrmse_pct_values = _relative_rrmse_percent_values(curve, target_power)
        rmse_epoch10 = _value_at_epoch(curve.epochs, rmse_values, 10.0)
        rrmse_pct_epoch10 = _value_at_epoch(curve.epochs, rrmse_pct_values, 10.0)
        rows.append(
            {
                "case": curve.case,
                "task": curve.task,
                "method": curve.method,
                "lr": curve.lr,
                "points": len(curve.epochs),
                "epoch_min": float(np.nanmin(curve.epochs)),
                "epoch_max": float(np.nanmax(curve.epochs)),
                "mse_start": float(curve.values[0]),
                "mse_end": float(curve.values[-1]),
                "target_power_component_mean": target_power,
                "rMSE_start": float(rmse_values[0]),
                "rMSE_epoch10": rmse_epoch10,
                "rMSE_end": float(rmse_values[-1]),
                "rRMSE_start": float(np.sqrt(rmse_values[0])),
                "rRMSE_end": float(np.sqrt(rmse_values[-1])),
                "rRMSE_percent_start": float(rrmse_pct_values[0]),
                "rRMSE_percent_epoch10": rrmse_pct_epoch10,
                "rRMSE_percent_end": float(rrmse_pct_values[-1]),
                "has_epoch10": bool(np.isfinite(rmse_epoch10)),
                "source": curve.source,
                "note": (
                    f"{curve.note} Relative metrics divide by masked target-field "
                    "component power in the same normalized-grid convention."
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["case"] = pd.Categorical(out["case"], list(power_by_case), ordered=True)
    out["method"] = pd.Categorical(out["method"], STANDARDIZED_FIELD_FIT_METHOD_ORDER, ordered=True)
    return out.sort_values(["case", "method"]).reset_index(drop=True)


def _lr_values_for_method(method: str) -> list[float]:
    if method not in LR_TRAJECTORY_METHOD_LRS:
        raise KeyError(f"No LR trajectory list for method {method}")
    return LR_TRAJECTORY_METHOD_LRS[method]


def _finite_curve_count(curve: Curve) -> int:
    return int(np.isfinite(curve.values).sum())


def _source_from_history_subset(hist: pd.DataFrame, default: str) -> str:
    if "history_path" in hist.columns:
        paths = hist["history_path"].dropna().astype(str)
        if not paths.empty:
            return str(paths.iloc[0])
    if "summary_path" in hist.columns:
        paths = hist["summary_path"].dropna().astype(str)
        if not paths.empty:
            return str(paths.iloc[0])
    return default


def _append_oasis_lr_trajectory_curves(
    curves: list[Curve],
    *,
    review_root: Path,
    objective_group: str,
    case: str,
    task: str,
    y_label: str,
) -> None:
    history = pd.read_csv(review_root / "loss_history_all.csv")
    for method in ["IDIR", "SINR cps4", "DirectCP cps4"]:
        for lr in _lr_values_for_method(method):
            hist = history[
                (history["objective_group"] == objective_group)
                & (history["method_label"] == method)
                & np.isclose(history["lr"].astype(float), float(lr))
            ]
            if hist.empty:
                raise ValueError(f"Missing {case} {method} lr={lr:g} trajectory")
            source = _source_from_history_subset(hist, str(review_root / "loss_history_all.csv"))
            curves.append(
                _curve_from_dataframe(
                    hist,
                    case=case,
                    task=task,
                    method=method,
                    lr=lr,
                    y_col="field_mse_objective_normalized_grid",
                    y_label=y_label,
                    source=source,
                    note=(
                        "Requested LR trajectory for masked normalized-grid target-field "
                        "MSE in the OASIS field-fit sweep."
                    ),
                )
            )


def _oasis_directcp842_field_fit_run_dir(case: str) -> Path:
    run_name = (
        "direct_cp_8_4_2_field_mse_rank009_sched8-4-2_ds4-2-1_"
        "e400-800-1300_a1000_lr0p001"
    )
    if case == "OASIS SITReg":
        return (
            OASIS_842_FIELD_FIT_ROOT
            / "oasis_sitreg_rank009"
            / "direct_cp_8_4_2_field_mse"
            / f"sitreg_field_fit_{run_name}"
        )
    if case == "OASIS Greedy":
        return (
            OASIS_842_FIELD_FIT_ROOT
            / "oasis_greedy_rank009"
            / "direct_cp_8_4_2_field_mse"
            / f"greedy_field_fit_{run_name}"
        )
    raise ValueError(f"Unsupported OASIS DirectCP 8-4-2 field-fit case: {case}")


def _append_oasis_directcp842_field_fit_curve(
    curves: list[Curve],
    *,
    case: str,
    task: str,
    y_label: str,
) -> None:
    run_dir = _oasis_directcp842_field_fit_run_dir(case)
    hist = _history_from_run_dir(run_dir)
    y_col = _finest_field_mse_column(hist, str(run_dir))
    curves.append(
        _curve_from_dataframe(
            hist,
            case=case,
            task=task,
            method="DirectCP 8-4-2",
            lr=1e-3,
            y_col=y_col,
            y_label=y_label,
            source=str(run_dir),
            note=(
                "Canonical OASIS multistage DirectCP field-fit add-on: cps 8/4/2, "
                "downsample 4/2/1, stage epochs 400/800/1300, lr 1e-3; plotted "
                "using finest-domain masked normalized-grid MSE diagnostics."
            ),
        )
    )


def _finest_field_mse_column(df: pd.DataFrame, source: str) -> str:
    for column in ["finest_data_loss", "finest_field_mse_objective_normalized_grid"]:
        if column in df.columns and df[column].notna().any():
            return column
    raise KeyError(f"No finite finest-domain field MSE diagnostic column in {source}")


def _append_lung_lr_trajectory_curves(
    curves: list[Curve],
    *,
    review_root: Path,
    case: str,
    case_key: str,
) -> None:
    summary = pd.read_csv(review_root / "summary_runs.csv")
    for method in METHOD_ORDER:
        for lr in _lr_values_for_method(method):
            if method == "DirectCP 4stage cps4":
                run_dir = _lung_4stage_field_diag_run_dir(case_key, lr)
                hist = _history_from_run_dir(run_dir)
                y_col = _finest_field_mse_column(hist, str(run_dir))
                note = (
                    "Requested LR trajectory using posthoc finest-domain masked "
                    "normalized-grid MSE. Coarse stages log this diagnostic sparsely; "
                    "the final 1mm stage logs every epoch."
                )
            else:
                subset = summary[
                    (summary["objective_group"] == "ptvreg_field_fit")
                    & (summary["method_label"] == method)
                    & np.isclose(summary["lr"].astype(float), float(lr))
                ]
                row = _first_row(subset, subset["status"].astype(str).eq("completed"), f"{case} {method} lr={lr:g}")
                run_dir = Path(row["run_dir"])
                hist = _history_from_run_dir(run_dir)
                y_col = "field_mse_objective_normalized_grid"
                note = "Requested LR trajectory for masked normalized-grid target-field MSE."
            curves.append(
                _curve_from_dataframe(
                    hist,
                    case=case,
                    task="pTVReg field fit",
                    method=method,
                    lr=lr,
                    y_col=y_col,
                    y_label="pTVReg field MSE",
                    source=str(run_dir),
                    note=note,
                )
            )


def load_rrmse_lr_trajectory_curves() -> list[Curve]:
    """Load all requested field-fit LR trajectories for the rRMSE figure."""
    curves: list[Curve] = []
    _append_oasis_lr_trajectory_curves(
        curves,
        review_root=OASIS_REVIEW,
        objective_group="sitreg_field_fit",
        case="OASIS SITReg",
        task="SITReg field fit",
        y_label="SITReg field MSE",
    )
    _append_oasis_directcp842_field_fit_curve(
        curves,
        case="OASIS SITReg",
        task="SITReg field fit",
        y_label="SITReg field MSE",
    )
    _append_oasis_lr_trajectory_curves(
        curves,
        review_root=OASIS_GREEDY_REVIEW,
        objective_group="greedy_field_fit",
        case="OASIS Greedy",
        task="Greedy field fit",
        y_label="Greedy field MSE",
    )
    _append_oasis_directcp842_field_fit_curve(
        curves,
        case="OASIS Greedy",
        task="Greedy field fit",
        y_label="Greedy field MSE",
    )
    _append_lung_lr_trajectory_curves(
        curves,
        review_root=LUNG01_REVIEW,
        case="LungCT case01",
        case_key="case01",
    )
    _append_lung_lr_trajectory_curves(
        curves,
        review_root=LUNG08_REVIEW,
        case="LungCT case08",
        case_key="case08",
    )
    return curves


def rrmse_lr_trajectory_selected_runs(curves: list[Curve]) -> pd.DataFrame:
    by_key = {
        (curve.case, curve.method, _format_lr_label(curve.lr)): curve
        for curve in curves
    }
    rows: list[dict[str, object]] = []
    for case in RELATIVE_FIELD_CASE_ORDER:
        for method in METHOD_ORDER:
            for lr in _lr_values_for_method(method):
                lr_label = _format_lr_label(lr)
                curve = by_key.get((case, method, lr_label))
                available = curve is not None
                note = ""
                source = ""
                points = 0
                if available:
                    source = curve.source
                    points = len(curve.epochs)
                    note = curve.note
                elif case.startswith("OASIS") and method == "DirectCP 4stage cps4":
                    note = "Not run for OASIS; no OASIS DirectCP 4-stage field-fit trajectory is included."
                else:
                    note = "Requested trajectory is missing."
                rows.append(
                    {
                        "case": case,
                        "task": "field_fit",
                        "method": method,
                        "lr": lr,
                        "lr_label": lr_label,
                        "requested": True,
                        "available": available,
                        "points": points,
                        "source": source,
                        "note": note,
                    }
                )
    out = pd.DataFrame(rows)
    out["case"] = pd.Categorical(out["case"], RELATIVE_FIELD_CASE_ORDER, ordered=True)
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    lr_order = [_format_lr_label(lr) for lr in LR_TRAJECTORY_LR_ORDER]
    out["lr_label"] = pd.Categorical(out["lr_label"], lr_order, ordered=True)
    return out.sort_values(["case", "method", "lr_label"]).reset_index(drop=True)


def rrmse_lr_trajectory_curve_summary(curves: list[Curve], power_summary: pd.DataFrame) -> pd.DataFrame:
    power_by_case = _target_power_map(power_summary)
    rows: list[dict[str, object]] = []
    for curve in curves:
        target_power = power_by_case[curve.case]
        rmse_values = _relative_mse_values(curve, target_power)
        rrmse_pct_values = _relative_rrmse_percent_values(curve, target_power)
        rows.append(
            {
                "case": curve.case,
                "task": curve.task,
                "method": curve.method,
                "lr": curve.lr,
                "lr_label": _format_lr_label(curve.lr),
                "points": len(curve.epochs),
                "finite_points": _finite_curve_count(curve),
                "epoch_min": float(np.nanmin(curve.epochs)),
                "epoch_max": float(np.nanmax(curve.epochs)),
                "mse_start": float(curve.values[0]),
                "mse_epoch10": _value_at_epoch(curve.epochs, curve.values, 10.0),
                "mse_epoch50": _value_at_epoch(curve.epochs, curve.values, 50.0),
                "mse_end": float(curve.values[-1]),
                "target_power_component_mean": target_power,
                "rMSE_start": float(rmse_values[0]),
                "rMSE_epoch10": _value_at_epoch(curve.epochs, rmse_values, 10.0),
                "rMSE_epoch50": _value_at_epoch(curve.epochs, rmse_values, 50.0),
                "rMSE_end": float(rmse_values[-1]),
                "rRMSE_percent_start": float(rrmse_pct_values[0]),
                "rRMSE_percent_epoch10": _value_at_epoch(curve.epochs, rrmse_pct_values, 10.0),
                "rRMSE_percent_epoch50": _value_at_epoch(curve.epochs, rrmse_pct_values, 50.0),
                "rRMSE_percent_end": float(rrmse_pct_values[-1]),
                "has_epoch10": bool(np.isfinite(_value_at_epoch(curve.epochs, rrmse_pct_values, 10.0))),
                "has_epoch50": bool(np.isfinite(_value_at_epoch(curve.epochs, rrmse_pct_values, 50.0))),
                "source": curve.source,
                "note": curve.note,
            }
        )
    out = pd.DataFrame(rows)
    out["case"] = pd.Categorical(out["case"], RELATIVE_FIELD_CASE_ORDER, ordered=True)
    out["method"] = pd.Categorical(out["method"], STANDARDIZED_FIELD_FIT_METHOD_ORDER, ordered=True)
    out["lr_label"] = pd.Categorical(
        out["lr_label"],
        [_format_lr_label(lr) for lr in LR_TRAJECTORY_LR_ORDER],
        ordered=True,
    )
    return out.sort_values(["case", "method", "lr_label"]).reset_index(drop=True)


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
        for method in STANDARDIZED_FIELD_FIT_METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
        frameon=False,
    )
    return fig


def make_relative_field_fit_figure(
    field_curves: list[Curve],
    power_summary: pd.DataFrame,
    *,
    smooth: bool = True,
    case_order: list[str] | None = None,
    rmse_ylim: tuple[float, float] = (1.0e-5, 1.0e1),
    rrmse_percent_ylim: tuple[float, float] = (1.0e-1, 1.0e2),
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

    power_by_case = _target_power_map(power_summary)
    if case_order is None:
        case_order = [case for case in RELATIVE_FIELD_CASE_ORDER if case in power_by_case]
    fig, axes = plt.subplots(2, len(case_order), figsize=(9.6, 4.25), sharex=True, constrained_layout=True)
    panel_titles = {
        "OASIS SITReg": "OASIS rank-9\nSITReg target",
        "OASIS Greedy": "OASIS rank-9\nGreedy target",
        "LungCT case01": "LungCT case01",
        "LungCT case08": "LungCT case08",
    }

    for col, case in enumerate(case_order):
        curves = [curve for curve in field_curves if curve.case == case]

        ax = axes[0, col]
        for curve in curves:
            style = METHOD_STYLE[curve.method]
            values = _relative_mse_values(curve, power_by_case[case])
            values = _smooth(values, 3 if curve.method == "IDIR" else 25) if smooth else values
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
        ax.set_yscale("log")
        ax.set_ylim(*rmse_ylim)
        ax.set_ylabel("rMSE")
        ax.grid(True, which="both", color="#d8dce2", linewidth=0.5, alpha=0.75)

        ax = axes[1, col]
        for curve in curves:
            style = METHOD_STYLE[curve.method]
            values = _relative_rrmse_percent_values(curve, power_by_case[case])
            values = _smooth(values, 3 if curve.method == "IDIR" else 25) if smooth else values
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
        ax.set_ylim(*rrmse_percent_ylim)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("rRMSE (%)")
        ax.grid(True, which="both", color="#d8dce2", linewidth=0.5, alpha=0.75)

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


def _trajectory_lr_style(lr: float) -> dict[str, object]:
    for value, style in LR_TRAJECTORY_STYLE.items():
        if np.isclose(float(lr), value):
            return style
    return {"color": "#4c4c4c", "linestyle": "-", "label": _format_lr_label(lr)}


def make_rrmse_lr_trajectory_figure(
    curves: list[Curve],
    power_summary: pd.DataFrame,
    *,
    smooth: bool = True,
    rrmse_percent_ylim: tuple[float, float] = (1.0e-1, 1.0e2),
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

    power_by_case = _target_power_map(power_summary)
    fig, axes = plt.subplots(
        len(METHOD_ORDER),
        len(RELATIVE_FIELD_CASE_ORDER),
        figsize=(10.4, 7.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panel_titles = {
        "OASIS SITReg": "OASIS rank-9\nSITReg target",
        "OASIS Greedy": "OASIS rank-9\nGreedy target",
        "LungCT case01": "LungCT case01",
        "LungCT case08": "LungCT case08",
    }
    curves_by_key = {
        (curve.case, curve.method, _format_lr_label(curve.lr)): curve
        for curve in curves
    }

    for row, method in enumerate(METHOD_ORDER):
        for col, case in enumerate(RELATIVE_FIELD_CASE_ORDER):
            ax = axes[row, col]
            plotted = False
            for lr in _lr_values_for_method(method):
                curve = curves_by_key.get((case, method, _format_lr_label(lr)))
                if curve is None:
                    continue
                style = _trajectory_lr_style(lr)
                values = _relative_rrmse_percent_values(curve, power_by_case[case])
                values = _smooth(values, 3 if method == "IDIR" else 25) if smooth else values
                ax.plot(
                    curve.epochs,
                    values,
                    color=str(style["color"]),
                    linestyle=str(style["linestyle"]),
                    linewidth=1.5,
                    label=str(style["label"]),
                )
                plotted = True
            if row == 0:
                ax.set_title(panel_titles[case])
            if col == 0:
                ax.set_ylabel(f"{method}\nrRMSE (%)")
            if row == len(METHOD_ORDER) - 1:
                ax.set_xlabel("Epoch")
            ax.set_yscale("log")
            ax.set_ylim(*rrmse_percent_ylim)
            ax.set_xlim(0, 2500)
            ax.grid(True, which="both", color="#d8dce2", linewidth=0.5, alpha=0.75)
            if not plotted:
                ax.text(
                    0.5,
                    0.5,
                    "not run",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#6f747c",
                    fontsize=8,
                )

    handles = [
        Line2D(
            [0],
            [0],
            color=str(LR_TRAJECTORY_STYLE[lr]["color"]),
            linestyle=str(LR_TRAJECTORY_STYLE[lr]["linestyle"]),
            linewidth=1.8,
            label=str(LR_TRAJECTORY_STYLE[lr]["label"]),
        )
        for lr in LR_TRAJECTORY_LR_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=len(handles),
        frameon=False,
    )
    return fig


def _standardized_field_fit_lr_by_key(standard: dict[str, object]) -> dict[tuple[str, str], float]:
    rows = selected_hypervariant_records("fitting_figure", standard)
    return {
        (str(row["dataset"]), str(row["method_id"])): float(row["lr"])
        for row in rows
    }


def _display_method(method: str, nicknames: dict[str, dict[str, str]] | None = None) -> str:
    return artifact_display_name(method, method, nicknames)


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(df, "map"):
        return df.map(sanitize_public_value)
    return df.applymap(sanitize_public_value)


def _fitting_figure_options(standard: dict[str, object]) -> dict[str, object]:
    options = standard.get("artifact_figure_options", {})
    if not isinstance(options, dict):
        return {}
    fitting_options = options.get("fitting_figure", {})
    if not isinstance(fitting_options, dict):
        return {}
    return dict(fitting_options)


def _merged_panelb_oasis_target(standard: dict[str, object]) -> str:
    target = str(
        _fitting_figure_options(standard).get(
            "merged_panel_b_oasis_target",
            MERGED_PANELB_OASIS_TARGET_OPTIONS[0],
        )
    )
    if target not in MERGED_PANELB_OASIS_TARGET_OPTIONS:
        valid = ", ".join(MERGED_PANELB_OASIS_TARGET_OPTIONS)
        raise ValueError(f"Invalid merged Panel B OASIS target {target!r}; expected one of: {valid}")
    return target


def _merged_panelb_case_for_panela_case(panel_case: str, oasis_target: str) -> str:
    if panel_case == "OASIS rank-9":
        return oasis_target
    return panel_case


def filter_standardized_field_fit_curves(
    curves: list[Curve],
    standard: dict[str, object],
) -> list[Curve]:
    lr_by_key = _standardized_field_fit_lr_by_key(standard)
    filtered: list[Curve] = []
    for curve in curves:
        lr = lr_by_key.get((curve.case, curve.method))
        if lr is not None and np.isclose(float(curve.lr), float(lr)):
            filtered.append(curve)
    return filtered


def standardized_field_fit_selected_runs(
    curves: list[Curve],
    standard: dict[str, object],
    nicknames: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    by_key = {
        (curve.case, curve.method, _format_lr_label(curve.lr)): curve
        for curve in curves
    }
    rows: list[dict[str, object]] = []
    for record in selected_hypervariant_records("fitting_figure", standard):
        case = str(record["dataset"])
        method = str(record["method_id"])
        lr = float(record["lr"])
        lr_label = _format_lr_label(lr)
        curve = by_key.get((case, method, lr_label))
        rows.append(
            {
                "case": case,
                "task": "field_fit",
                "method": method,
                "display_method": _display_method(method, nicknames),
                "lr": lr,
                "lr_label": lr_label,
                "available": curve is not None,
                "points": len(curve.epochs) if curve is not None else 0,
                "source": curve.source if curve is not None else "",
                "note": curve.note if curve is not None else "Requested standardized fitting trajectory is missing.",
            }
        )
    out = pd.DataFrame(rows)
    out["case"] = pd.Categorical(out["case"], RELATIVE_FIELD_CASE_ORDER, ordered=True)
    out["method"] = pd.Categorical(out["method"], STANDARDIZED_FIELD_FIT_METHOD_ORDER, ordered=True)
    return out.sort_values(["case", "method"]).reset_index(drop=True)


def make_standardized_rrmse_field_fit_figure(
    curves: list[Curve],
    power_summary: pd.DataFrame,
    *,
    nicknames: dict[str, dict[str, str]] | None = None,
    smooth: bool = True,
    rrmse_percent_ylim: tuple[float, float] = (1.0e-1, 1.0e2),
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

    power_by_case = _target_power_map(power_summary)
    fig, axes = plt.subplots(
        1,
        len(RELATIVE_FIELD_CASE_ORDER),
        figsize=(10.4, 2.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panel_titles = {
        "OASIS SITReg": "OASIS rank-9\nSITReg target",
        "OASIS Greedy": "OASIS rank-9\nGreedy target",
        "LungCT case01": "LungCT case01",
        "LungCT case08": "LungCT case08",
    }

    for col, case in enumerate(RELATIVE_FIELD_CASE_ORDER):
        ax = axes[col]
        plotted = False
        for method in STANDARDIZED_FIELD_FIT_METHOD_ORDER:
            matches = [curve for curve in curves if curve.case == case and curve.method == method]
            if not matches:
                continue
            curve = matches[0]
            style = METHOD_STYLE[method]
            values = _relative_rrmse_percent_values(curve, power_by_case[case])
            values = _smooth(values, 3 if method == "IDIR" else 25) if smooth else values
            marker = style["marker"] if method == "IDIR" else None
            markevery = max(1, len(curve.epochs) // 8) if marker else None
            ax.plot(
                curve.epochs,
                values,
                label=_display_method(method, nicknames),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                marker=marker,
                markersize=2.0,
                markevery=markevery,
                zorder=style["zorder"],
            )
            plotted = True
        ax.set_title(panel_titles[case])
        ax.set_yscale("log")
        ax.set_ylim(*rrmse_percent_ylim)
        ax.set_xlim(0, 2500)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("rRMSE (%)")
        ax.grid(True, which="both", color="#d8dce2", linewidth=0.5, alpha=0.75)
        if not plotted:
            ax.text(
                0.5,
                0.5,
                "not run",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#6f747c",
                fontsize=8,
            )

    handles = []
    seen_labels: set[str] = set()
    for method in STANDARDIZED_FIELD_FIT_METHOD_ORDER:
        label = _display_method(method, nicknames)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        handles.append(
            Line2D(
                [0],
                [0],
                color=METHOD_STYLE[method]["color"],
                linestyle=METHOD_STYLE[method]["linestyle"],
                marker=METHOD_STYLE[method]["marker"] or None,
                markersize=3,
                linewidth=1.8,
                label=label,
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(handles),
        frameon=False,
    )
    return fig


def _panelA_plot_id(source_method_id: str) -> str:
    try:
        return PANELA_PLOT_ID_BY_SOURCE_METHOD[str(source_method_id)]
    except KeyError as exc:
        raise KeyError(f"No Panel A plot label configured for {source_method_id}") from exc


def _panelA_run_dir(case_dir: str, method_id: str, run_name: str) -> Path:
    return PANELA_GLOBAL_NCC_ROOT / case_dir / method_id / run_name


def _panelA_curve_from_dataframe(
    df: pd.DataFrame,
    *,
    case: str,
    source_method_id: str,
    lr: float,
    y_col: str,
    source: str,
    optimized_loss: str,
    diagnostic_loss: str,
    note: str,
) -> PanelACurve:
    if y_col not in df.columns:
        raise KeyError(f"{y_col} not in {source}")
    epoch_col = "global_epoch" if "global_epoch" in df.columns else "epoch"
    rows = df[[epoch_col, y_col]].dropna().copy()
    rows = rows.sort_values(epoch_col)
    if rows.empty:
        raise ValueError(f"No finite {y_col} values in {source}")
    return PanelACurve(
        case=case,
        source_method_id=source_method_id,
        plot_method_id=_panelA_plot_id(source_method_id),
        lr=float(lr),
        epochs=rows[epoch_col].to_numpy(float),
        values=rows[y_col].to_numpy(float),
        source=source,
        optimized_loss=optimized_loss,
        diagnostic_loss=diagnostic_loss,
        note=note,
    )


def _append_oasis_panelA_curves(curves: list[PanelACurve]) -> None:
    idir_csv = (
        IDIR_EVAL_ROOT
        / "oasis_rank009/idir_original_dense_siren_be_autograd/"
        / "oasis_rank009_idir_a1_lr1e-4_global_ncc_eval/image_similarity_eval.csv"
    )
    curves.append(
        _panelA_curve_from_dataframe(
            pd.read_csv(idir_csv),
            case="OASIS rank-9",
            source_method_id="idir_original_dense_siren_be_autograd",
            lr=1e-4,
            y_col="similarity_loss",
            source=str(idir_csv),
            optimized_loss="sampled IDIR NCC + BE",
            diagnostic_loss="full-image global NCC",
            note="Existing matched-eval run; diagnostic only, logged every 10 epochs.",
        )
    )

    history = pd.read_csv(OASIS_REVIEW / "loss_history_augmented.csv")
    for source_method_id, lr in [
        ("sinr_cps4_be_bspline_analytic", 1e-4),
        ("direct_cp_cps4_be_bspline_analytic", 1e-3),
    ]:
        hist = history[
            (history["objective_group"] == "image_registration")
            & (history["method_key"] == source_method_id)
            & np.isclose(history["lr"].astype(float), float(lr))
            & np.isclose(history["alpha"].astype(float), 1000.0)
        ].copy()
        if hist.empty:
            raise ValueError(f"Missing OASIS Panel A history for {source_method_id} lr={lr:g}")
        curves.append(
            _panelA_curve_from_dataframe(
                hist,
                case="OASIS rank-9",
                source_method_id=source_method_id,
                lr=lr,
                y_col="data_loss",
                source=_source_from_history_subset(hist, str(OASIS_REVIEW / "loss_history_augmented.csv")),
                optimized_loss="full-image global NCC + analytic B-spline BE",
                diagnostic_loss="full-image global NCC",
                note="Existing OASIS lr-sweep history; plotted data loss is the optimized image term.",
            )
        )

    run_dir = _panelA_run_dir(
        "oasis_rank009",
        "direct_cp_8_4_2_be_bspline_analytic_multistage",
        "panelA_oasis_rank009_directcp842_sched8-4-2_ds4-2-1_e400-800-1300_a1000_lr0p001",
    )
    curves.append(
        _panelA_curve_from_dataframe(
            pd.read_csv(run_dir / "global_ncc_diagnostic.csv"),
            case="OASIS rank-9",
            source_method_id="direct_cp_8_4_2_be_bspline_analytic_multistage",
            lr=1e-3,
            y_col="global_ncc_loss",
            source=str(run_dir / "global_ncc_diagnostic.csv"),
            optimized_loss="stage-domain global NCC + analytic B-spline BE",
            diagnostic_loss="full-image global NCC",
            note=(
                "New OASIS multistage run: cps 8/4/2, downsample 4/2/1, "
                "stage epochs 400/800/1300, alpha 1000."
            ),
        )
    )


def _append_lung_panelA_curves(curves: list[PanelACurve], *, case: str, case_tag: str) -> None:
    case_dir = f"lungct_case{case_tag}"
    specs = [
        {
            "source_method_id": "idir_be_default_lungmask",
            "lr": 1e-4,
            "run_name": f"panelA_lungct_case{case_tag}_idir_a3_lr0p0001",
            "optimized_loss": "native IDIR sampled NCC + IDIR BE",
            "note": "New IDIR rerun with full-crop global NCC diagnostic every 10 epochs; alpha_bending 3.",
        },
        {
            "source_method_id": "sinr_cps4_global_ncc_fd_be",
            "lr": 1e-4,
            "run_name": f"panelA_lungct_case{case_tag}_sinr_cps4_globalncc_fdbe_a10000_lr0p0001",
            "optimized_loss": "full-crop global NCC + dense forward-difference BE",
            "note": "New SINR cps4 global-NCC+FD-BE run; alpha 10000.",
        },
        {
            "source_method_id": "direct_cp_cps4_global_ncc_fd_be",
            "lr": 1e-3,
            "run_name": f"panelA_lungct_case{case_tag}_directcp_cps4_globalncc_fdbe_a10000_lr0p001",
            "optimized_loss": "full-crop global NCC + dense forward-difference BE",
            "note": "New DirectCP cps4 global-NCC+FD-BE run; alpha 10000.",
        },
        {
            "source_method_id": "direct_cp_32_16_8_4_global_ncc_fd_be",
            "lr": 1e-3,
            "run_name": (
                f"panelA_lungct_case{case_tag}_directcp321684_globalncc_fdbe_"
                "sched32-16-8-4_ds8-4-2-1_e200-400-800-1000_a10000_lr0p001"
            ),
            "optimized_loss": "stage-domain full-crop global NCC + dense forward-difference BE",
            "note": (
                "New DirectCP multistage global-NCC+FD-BE run: cps 32/16/8/4, "
                "downsample 8/4/2/1, stage epochs 200/400/800/1000, alpha 10000."
            ),
        },
    ]
    for spec in specs:
        run_dir = _panelA_run_dir(case_dir, str(spec["source_method_id"]), str(spec["run_name"]))
        curves.append(
            _panelA_curve_from_dataframe(
                pd.read_csv(run_dir / "global_ncc_diagnostic.csv"),
                case=case,
                source_method_id=str(spec["source_method_id"]),
                lr=float(spec["lr"]),
                y_col="global_ncc_loss",
                source=str(run_dir / "global_ncc_diagnostic.csv"),
                optimized_loss=str(spec["optimized_loss"]),
                diagnostic_loss="full-crop global NCC",
                note=str(spec["note"]),
            )
        )


def load_standardized_panelA_global_ncc_curves() -> list[PanelACurve]:
    curves: list[PanelACurve] = []
    _append_oasis_panelA_curves(curves)
    _append_lung_panelA_curves(curves, case="LungCT case01", case_tag="01")
    _append_lung_panelA_curves(curves, case="LungCT case08", case_tag="08")
    return curves


def standardized_panelA_global_ncc_selected_runs(
    curves: list[PanelACurve],
    standard: dict[str, object],
    nicknames: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    by_key = {(curve.case, curve.source_method_id): curve for curve in curves}
    rows: list[dict[str, object]] = []
    for record in selected_hypervariant_records("panelA_global_ncc_figure", standard):
        case = str(record["dataset"])
        source_method_id = str(record["method_id"])
        plot_method_id = _panelA_plot_id(source_method_id)
        lr = float(record["lr"])
        curve = by_key.get((case, source_method_id))
        available = curve is not None and np.isclose(curve.lr, lr)
        rows.append(
            {
                "case": case,
                "task": "image_registration_global_ncc",
                "source_method_id": source_method_id,
                "source_method_display": artifact_display_name(source_method_id, source_method_id, nicknames),
                "plot_method_id": plot_method_id,
                "plot_method_display": artifact_display_name(plot_method_id, plot_method_id, nicknames),
                "lr": lr,
                "lr_label": _format_lr_label(lr),
                "selected_setting": str(record.get("selected_setting", "")),
                "available": available,
                "points": len(curve.epochs) if available and curve is not None else 0,
                "epoch_min": float(np.nanmin(curve.epochs)) if available and curve is not None else np.nan,
                "epoch_max": float(np.nanmax(curve.epochs)) if available and curve is not None else np.nan,
                "optimized_loss": curve.optimized_loss if available and curve is not None else "",
                "diagnostic_loss": curve.diagnostic_loss if available and curve is not None else "",
                "source": curve.source if available and curve is not None else "",
                "note": curve.note if available and curve is not None else "Requested Panel A global-NCC curve is missing.",
            }
        )
    out = pd.DataFrame(rows)
    out["case"] = pd.Categorical(out["case"], PANELA_CASE_ORDER, ordered=True)
    out["plot_method_id"] = pd.Categorical(out["plot_method_id"], PANELA_PLOT_METHOD_ORDER, ordered=True)
    return out.sort_values(["case", "plot_method_id"]).reset_index(drop=True)


def panelA_global_ncc_curve_summary(
    curves: list[PanelACurve],
    nicknames: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for curve in curves:
        rows.append(
            {
                "case": curve.case,
                "task": "image_registration_global_ncc",
                "source_method_id": curve.source_method_id,
                "source_method_display": artifact_display_name(
                    curve.source_method_id,
                    curve.source_method_id,
                    nicknames,
                ),
                "plot_method_id": curve.plot_method_id,
                "plot_method_display": artifact_display_name(curve.plot_method_id, curve.plot_method_id, nicknames),
                "lr": curve.lr,
                "lr_label": _format_lr_label(curve.lr),
                "points": len(curve.epochs),
                "finite_points": int(np.isfinite(curve.values).sum()),
                "epoch_min": float(np.nanmin(curve.epochs)),
                "epoch_max": float(np.nanmax(curve.epochs)),
                "global_ncc_loss_start": float(curve.values[0]),
                "global_ncc_loss_epoch10": _value_at_epoch(curve.epochs, curve.values, 10.0),
                "global_ncc_loss_epoch100": _value_at_epoch(curve.epochs, curve.values, 100.0),
                "global_ncc_loss_end": float(curve.values[-1]),
                "global_ncc_value_start": float(-curve.values[0]),
                "global_ncc_value_epoch10": float(-_value_at_epoch(curve.epochs, curve.values, 10.0)),
                "global_ncc_value_epoch100": float(-_value_at_epoch(curve.epochs, curve.values, 100.0)),
                "global_ncc_value_end": float(-curve.values[-1]),
                "optimized_loss": curve.optimized_loss,
                "diagnostic_loss": curve.diagnostic_loss,
                "source": curve.source,
                "note": curve.note,
            }
        )
    out = pd.DataFrame(rows)
    out["case"] = pd.Categorical(out["case"], PANELA_CASE_ORDER, ordered=True)
    out["plot_method_id"] = pd.Categorical(out["plot_method_id"], PANELA_PLOT_METHOD_ORDER, ordered=True)
    return out.sort_values(["case", "plot_method_id"]).reset_index(drop=True)


def make_standardized_global_ncc_panelA_figure(
    curves: list[PanelACurve],
    *,
    nicknames: dict[str, dict[str, str]] | None = None,
    smooth: bool = True,
    global_ncc_ylim: tuple[float, float] = (-1.0, -0.8),
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

    fig, axes = plt.subplots(
        1,
        len(PANELA_CASE_ORDER),
        figsize=(7.8, 2.65),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panel_titles = {
        "OASIS rank-9": "OASIS rank-9",
        "LungCT case01": "LungCT case01",
        "LungCT case08": "LungCT case08",
    }

    for col, case in enumerate(PANELA_CASE_ORDER):
        ax = axes[col]
        for plot_method_id in PANELA_PLOT_METHOD_ORDER:
            matches = [
                curve
                for curve in curves
                if curve.case == case and curve.plot_method_id == plot_method_id
            ]
            if not matches:
                continue
            curve = matches[0]
            style = METHOD_STYLE[PANELA_STYLE_METHOD_BY_PLOT_ID[plot_method_id]]
            values = _smooth(curve.values, 3 if plot_method_id == "panelA_idir" else 25) if smooth else curve.values
            marker = style["marker"] if plot_method_id == "panelA_idir" else None
            markevery = max(1, len(curve.epochs) // 8) if marker else None
            ax.plot(
                curve.epochs,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                marker=marker,
                markersize=2.0,
                markevery=markevery,
                zorder=style["zorder"],
                label=artifact_display_name(plot_method_id, plot_method_id, nicknames),
            )
        ax.set_title(panel_titles[case])
        ax.set_ylim(*global_ncc_ylim)
        ax.set_xlim(0, 2500)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Global NCC loss")
        ax.grid(True, color="#d8dce2", linewidth=0.5, alpha=0.75)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[PANELA_STYLE_METHOD_BY_PLOT_ID[plot_method_id]]["color"],
            linestyle=METHOD_STYLE[PANELA_STYLE_METHOD_BY_PLOT_ID[plot_method_id]]["linestyle"],
            marker=METHOD_STYLE[PANELA_STYLE_METHOD_BY_PLOT_ID[plot_method_id]]["marker"] or None,
            markersize=3,
            linewidth=1.8,
            label=artifact_display_name(plot_method_id, plot_method_id, nicknames),
        )
        for plot_method_id in PANELA_PLOT_METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(handles),
        frameon=False,
    )
    return fig


def make_standardized_panelA_one_minus_ncc_log_figure(
    curves: list[PanelACurve],
    *,
    nicknames: dict[str, dict[str, str]] | None = None,
    smooth: bool = True,
    one_minus_ncc_ylim: tuple[float, float] = (1.0e-3, 5.0e-1),
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

    fig, axes = plt.subplots(
        1,
        len(PANELA_CASE_ORDER),
        figsize=(7.8, 2.65),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panel_titles = {
        "OASIS rank-9": "OASIS rank-9",
        "LungCT case01": "LungCT case01",
        "LungCT case08": "LungCT case08",
    }

    for col, case in enumerate(PANELA_CASE_ORDER):
        ax = axes[col]
        for plot_method_id in PANELA_PLOT_METHOD_ORDER:
            matches = [
                curve
                for curve in curves
                if curve.case == case and curve.plot_method_id == plot_method_id
            ]
            if not matches:
                continue
            curve = matches[0]
            style = METHOD_STYLE[PANELA_STYLE_METHOD_BY_PLOT_ID[plot_method_id]]
            values = 1.0 + curve.values
            values = np.maximum(values, 1.0e-12)
            values = _smooth(values, 3 if plot_method_id == "panelA_idir" else 25) if smooth else values
            marker = style["marker"] if plot_method_id == "panelA_idir" else None
            markevery = max(1, len(curve.epochs) // 8) if marker else None
            ax.plot(
                curve.epochs,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                marker=marker,
                markersize=2.0,
                markevery=markevery,
                zorder=style["zorder"],
                label=artifact_display_name(plot_method_id, plot_method_id, nicknames),
            )
        ax.set_title(panel_titles[case])
        ax.set_yscale("log")
        ax.set_ylim(*one_minus_ncc_ylim)
        ax.set_xlim(0, 2500)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("1 - NCC")
        ax.grid(True, which="both", color="#d8dce2", linewidth=0.5, alpha=0.75)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[PANELA_STYLE_METHOD_BY_PLOT_ID[plot_method_id]]["color"],
            linestyle=METHOD_STYLE[PANELA_STYLE_METHOD_BY_PLOT_ID[plot_method_id]]["linestyle"],
            marker=METHOD_STYLE[PANELA_STYLE_METHOD_BY_PLOT_ID[plot_method_id]]["marker"] or None,
            markersize=3,
            linewidth=1.8,
            label=artifact_display_name(plot_method_id, plot_method_id, nicknames),
        )
        for plot_method_id in PANELA_PLOT_METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(handles),
        frameon=False,
    )
    return fig


def _standardized_panelab_legend_handles(
    nicknames: dict[str, dict[str, str]] | None = None,
) -> list[Line2D]:
    handles: list[Line2D] = []
    seen_labels: set[str] = set()
    for plot_method_id in PANELA_PLOT_METHOD_ORDER:
        label = artifact_display_name(plot_method_id, plot_method_id, nicknames)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        style = METHOD_STYLE[PANELA_STYLE_METHOD_BY_PLOT_ID[plot_method_id]]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"] or None,
                markersize=3,
                linewidth=1.8,
                label=label,
            )
        )
    return handles


def make_standardized_panelAB_figure(
    panelA_curves: list[PanelACurve],
    field_curves: list[Curve],
    power_summary: pd.DataFrame,
    *,
    nicknames: dict[str, dict[str, str]] | None = None,
    panelb_oasis_target: str = "OASIS SITReg",
    smooth: bool = True,
    one_minus_ncc_ylim: tuple[float, float] = (1.0e-3, 5.0e-1),
    rrmse_percent_ylim: tuple[float, float] = (1.0e-1, 1.0e2),
    axis_label_size: int = PANELAB_AXIS_LABEL_SIZE,
    legend_font_size: int = PANELAB_LEGEND_FONT_SIZE,
    legend_anchor_y: float = PANELAB_LEGEND_ANCHOR_Y,
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": axis_label_size,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": legend_font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )

    if panelb_oasis_target not in MERGED_PANELB_OASIS_TARGET_OPTIONS:
        valid = ", ".join(MERGED_PANELB_OASIS_TARGET_OPTIONS)
        raise ValueError(f"Invalid Panel B OASIS target {panelb_oasis_target!r}; expected one of: {valid}")

    power_by_case = _target_power_map(power_summary)
    fig, axes = plt.subplots(
        2,
        len(PANELA_CASE_ORDER),
        figsize=(7.9, 5.15),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    axes = np.asarray(axes)
    top_titles = {
        "OASIS rank-9": "OASIS rank-9",
        "LungCT case01": "LungCT case01",
        "LungCT case08": "LungCT case08",
    }
    bottom_titles = {
        "OASIS SITReg": "SITReg field target",
        "OASIS Greedy": "Greedy field target",
        "LungCT case01": "pTVReg field target",
        "LungCT case08": "pTVReg field target",
    }

    for col, case in enumerate(PANELA_CASE_ORDER):
        ax = axes[0, col]
        for plot_method_id in PANELA_PLOT_METHOD_ORDER:
            matches = [
                curve
                for curve in panelA_curves
                if curve.case == case and curve.plot_method_id == plot_method_id
            ]
            if not matches:
                continue
            curve = matches[0]
            style = METHOD_STYLE[PANELA_STYLE_METHOD_BY_PLOT_ID[plot_method_id]]
            values = np.maximum(1.0 + curve.values, 1.0e-12)
            values = _smooth(values, 3 if plot_method_id == "panelA_idir" else 25) if smooth else values
            ax.plot(
                curve.epochs,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                marker=style["marker"] or None,
                markersize=2.0,
                markevery=max(1, len(curve.epochs) // 8) if style["marker"] else None,
                zorder=style["zorder"],
                label=artifact_display_name(plot_method_id, plot_method_id, nicknames),
            )
        ax.set_title(top_titles[case])
        ax.set_yscale("log")
        ax.set_ylim(*one_minus_ncc_ylim)
        ax.set_xlim(0, 2500)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_ylabel("1 - NCC")
        ax.grid(True, which="both", color="#d8dce2", linewidth=0.5, alpha=0.75)

        field_case = _merged_panelb_case_for_panela_case(case, panelb_oasis_target)
        ax = axes[1, col]
        plotted = False
        for method in STANDARDIZED_FIELD_FIT_METHOD_ORDER:
            matches = [
                curve
                for curve in field_curves
                if curve.case == field_case and curve.method == method
            ]
            if not matches:
                continue
            curve = matches[0]
            style = METHOD_STYLE[method]
            values = _relative_rrmse_percent_values(curve, power_by_case[field_case])
            values = _smooth(values, 3 if method == "IDIR" else 25) if smooth else values
            ax.plot(
                curve.epochs,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                marker=style["marker"] or None,
                markersize=2.0,
                markevery=max(1, len(curve.epochs) // 8) if style["marker"] else None,
                zorder=style["zorder"],
                label=_display_method(method, nicknames),
            )
            plotted = True
        if not plotted:
            ax.text(
                0.5,
                0.5,
                "not run",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#6f747c",
                fontsize=8,
            )
        ax.set_title(bottom_titles[field_case])
        ax.set_yscale("log")
        ax.set_ylim(*rrmse_percent_ylim)
        ax.set_xlim(0, 2500)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("rRMSE (%)")
        ax.grid(True, which="both", color="#d8dce2", linewidth=0.5, alpha=0.75)

    axes[0, 0].text(
        -0.28,
        1.12,
        "A",
        transform=axes[0, 0].transAxes,
        fontweight="bold",
        fontsize=11,
        ha="left",
        va="center",
    )
    axes[1, 0].text(
        -0.28,
        1.12,
        "B",
        transform=axes[1, 0].transAxes,
        fontweight="bold",
        fontsize=11,
        ha="left",
        va="center",
    )
    handles = _standardized_panelab_legend_handles(nicknames)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_anchor_y),
        ncol=len(handles),
        frameon=False,
        fontsize=legend_font_size,
    )
    return fig


def standardized_panelAB_selected_runs(
    panelA_selected: pd.DataFrame,
    field_selected: pd.DataFrame,
    *,
    panelb_oasis_target: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    panelA_method_order = {method: idx for idx, method in enumerate(PANELA_PLOT_METHOD_ORDER)}
    field_method_order = {
        "IDIR": 0,
        "SINR cps4": 1,
        "DirectCP cps4": 2,
        "DirectCP 8-4-2": 3,
        "DirectCP 4stage cps4": 3,
    }
    for _, row in panelA_selected.iterrows():
        rows.append(
            {
                "panel": "A",
                "plot_case": str(row["case"]),
                "source_case": str(row["case"]),
                "target_field": "",
                "task": str(row["task"]),
                "method_id": str(row["source_method_id"]),
                "plot_method_id": str(row["plot_method_id"]),
                "display_method": str(row["plot_method_display"]),
                "method_order": panelA_method_order[str(row["plot_method_id"])],
                "lr": float(row["lr"]),
                "lr_label": str(row["lr_label"]),
                "available": bool(row["available"]),
                "points": int(row["points"]),
                "source": str(row["source"]),
                "note": str(row["note"]),
            }
        )

    field_case_to_plot_case = {
        panelb_oasis_target: "OASIS rank-9",
        "LungCT case01": "LungCT case01",
        "LungCT case08": "LungCT case08",
    }
    for _, row in field_selected.iterrows():
        source_case = str(row["case"])
        if source_case not in field_case_to_plot_case:
            continue
        rows.append(
            {
                "panel": "B",
                "plot_case": field_case_to_plot_case[source_case],
                "source_case": source_case,
                "target_field": source_case,
                "task": str(row["task"]),
                "method_id": str(row["method"]),
                "plot_method_id": str(row["method"]),
                "display_method": str(row["display_method"]),
                "method_order": field_method_order[str(row["method"])],
                "lr": float(row["lr"]),
                "lr_label": str(row["lr_label"]),
                "available": bool(row["available"]),
                "points": int(row["points"]),
                "source": str(row["source"]),
                "note": str(row["note"]),
            }
        )
    out = pd.DataFrame(rows)
    out["panel"] = pd.Categorical(out["panel"], ["A", "B"], ordered=True)
    out["plot_case"] = pd.Categorical(out["plot_case"], PANELA_CASE_ORDER, ordered=True)
    return out.sort_values(["panel", "plot_case", "method_order"]).reset_index(drop=True)


def standardized_panelAB_curve_summary(
    panelA_curves: list[PanelACurve],
    field_curves: list[Curve],
    power_summary: pd.DataFrame,
    *,
    nicknames: dict[str, dict[str, str]] | None = None,
    panelb_oasis_target: str,
) -> pd.DataFrame:
    power_by_case = _target_power_map(power_summary)
    rows: list[dict[str, object]] = []
    method_order = {
        **{method: idx for idx, method in enumerate(PANELA_PLOT_METHOD_ORDER)},
        "IDIR": 0,
        "SINR cps4": 1,
        "DirectCP cps4": 2,
        "DirectCP 8-4-2": 3,
        "DirectCP 4stage cps4": 3,
    }
    for curve in panelA_curves:
        values = np.maximum(1.0 + curve.values, 1.0e-12)
        rows.append(
            {
                "panel": "A",
                "plot_case": curve.case,
                "source_case": curve.case,
                "metric": "one_minus_ncc",
                "method_id": curve.source_method_id,
                "plot_method_id": curve.plot_method_id,
                "display_method": artifact_display_name(curve.plot_method_id, curve.plot_method_id, nicknames),
                "method_order": method_order[curve.plot_method_id],
                "lr": curve.lr,
                "lr_label": _format_lr_label(curve.lr),
                "points": len(curve.epochs),
                "epoch_min": float(np.nanmin(curve.epochs)),
                "epoch_max": float(np.nanmax(curve.epochs)),
                "value_start": float(values[0]),
                "value_epoch10": _value_at_epoch(curve.epochs, values, 10.0),
                "value_epoch100": _value_at_epoch(curve.epochs, values, 100.0),
                "value_end": float(values[-1]),
                "source": curve.source,
                "note": curve.note,
            }
        )

    for panel_case in PANELA_CASE_ORDER:
        field_case = _merged_panelb_case_for_panela_case(panel_case, panelb_oasis_target)
        for method in STANDARDIZED_FIELD_FIT_METHOD_ORDER:
            matches = [
                curve
                for curve in field_curves
                if curve.case == field_case and curve.method == method
            ]
            if not matches:
                continue
            curve = matches[0]
            values = _relative_rrmse_percent_values(curve, power_by_case[field_case])
            rows.append(
                {
                    "panel": "B",
                    "plot_case": panel_case,
                    "source_case": field_case,
                    "metric": "rRMSE_percent",
                    "method_id": curve.method,
                    "plot_method_id": curve.method,
                    "display_method": _display_method(curve.method, nicknames),
                    "method_order": method_order[curve.method],
                    "lr": curve.lr,
                    "lr_label": _format_lr_label(curve.lr),
                    "points": len(curve.epochs),
                    "epoch_min": float(np.nanmin(curve.epochs)),
                    "epoch_max": float(np.nanmax(curve.epochs)),
                    "value_start": float(values[0]),
                    "value_epoch10": _value_at_epoch(curve.epochs, values, 10.0),
                    "value_epoch100": _value_at_epoch(curve.epochs, values, 100.0),
                    "value_end": float(values[-1]),
                    "source": curve.source,
                    "note": curve.note,
                }
            )
    out = pd.DataFrame(rows)
    out["panel"] = pd.Categorical(out["panel"], ["A", "B"], ordered=True)
    out["plot_case"] = pd.Categorical(out["plot_case"], PANELA_CASE_ORDER, ordered=True)
    return out.sort_values(["panel", "plot_case", "method_order"]).reset_index(drop=True)


def _write_panelAB_validation(
    output_dir: Path,
    panelA_selected: pd.DataFrame,
    field_selected: pd.DataFrame,
    *,
    panelb_oasis_target: str,
    standard: dict[str, object],
) -> None:
    panelA_input = [
        {
            "dataset": row["case"],
            "method_id": row["source_method_id"],
            "lr": row["lr"],
            "selected_setting": row["selected_setting"],
            "source": row["source"],
            "note": row["note"],
        }
        for _, row in panelA_selected.iterrows()
    ]
    validation_rows = validate_lr_records(
        panelA_input,
        artifact="panelA_global_ncc_figure",
        standard=standard,
    )
    for row in validation_rows:
        row["panel"] = "A"

    field_cases = {panelb_oasis_target, "LungCT case01", "LungCT case08"}
    field_input = [
        {
            "dataset": row["case"],
            "method_id": row["method"],
            "lr": row["lr"],
            "selected_setting": row["lr_label"],
            "source": row["source"],
            "note": row["note"],
        }
        for _, row in field_selected.iterrows()
        if str(row["case"]) in field_cases
    ]
    field_validation = validate_lr_records(field_input, artifact="fitting_figure", standard=standard)
    for row in field_validation:
        row["panel"] = "B"
    validation_rows.extend(field_validation)
    validation_rows = [sanitize_public_value(row) for row in validation_rows]
    pd.DataFrame(validation_rows).to_csv(
        output_dir / "fit_modes_standardized_panelAB_standard_validation.csv",
        index=False,
    )
    (output_dir / "fit_modes_standardized_panelAB_standard_validation_report.md").write_text(
        validation_report_markdown("Merged Panel A/B LR Validation", validation_rows)
    )


def _save_figure_to_dir(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    output_formats: tuple[str, ...] = ("png",),
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {fmt: output_dir / f"{stem}.{fmt}" for fmt in output_formats}
    for path in outputs.values():
        fig.savefig(path, dpi=300, bbox_inches="tight")
    for stale_fmt in {"pdf", "png"} - set(output_formats):
        stale_path = output_dir / f"{stem}.{stale_fmt}"
        if stale_path.exists():
            stale_path.unlink()
    return outputs


def _write_fitting_validation(
    output_dir: Path,
    selected: pd.DataFrame,
    standard: dict[str, object],
) -> None:
    validation_input = [
        {
            "dataset": row["case"],
            "method_id": row["method"],
            "lr": row["lr"],
            "selected_setting": row["lr_label"],
            "source": row["source"],
            "note": row["note"],
        }
        for _, row in selected.iterrows()
    ]
    validation_rows = validate_lr_records(validation_input, artifact="fitting_figure", standard=standard)
    validation_rows = [sanitize_public_value(row) for row in validation_rows]
    pd.DataFrame(validation_rows).to_csv(output_dir / "standard_validation.csv", index=False)
    (output_dir / "standard_validation_report.md").write_text(
        validation_report_markdown("Fitting Figure LR Validation", validation_rows)
    )


def _write_panelA_global_ncc_validation(
    output_dir: Path,
    selected: pd.DataFrame,
    standard: dict[str, object],
) -> None:
    validation_input = [
        {
            "dataset": row["case"],
            "method_id": row["source_method_id"],
            "lr": row["lr"],
            "selected_setting": row["selected_setting"],
            "source": row["source"],
            "note": row["note"],
        }
        for _, row in selected.iterrows()
    ]
    validation_rows = validate_lr_records(
        validation_input,
        artifact="panelA_global_ncc_figure",
        standard=standard,
    )
    validation_rows = [sanitize_public_value(row) for row in validation_rows]
    pd.DataFrame(validation_rows).to_csv(
        output_dir / "fit_modes_standardized_global_ncc_panelA_standard_validation.csv",
        index=False,
    )
    (output_dir / "fit_modes_standardized_global_ncc_panelA_standard_validation_report.md").write_text(
        validation_report_markdown("Panel A Global-NCC LR Validation", validation_rows)
    )


def build_standardized_global_ncc_panelA_outputs(
    *,
    standard_path: str | Path | None = None,
    nickname_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure, dict[str, Path]]:
    standard = load_standard(standard_path)
    nicknames = load_nicknames(nickname_path)
    output_dir = artifact_output_dir("fitting_figure", standard)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = load_standardized_panelA_global_ncc_curves()
    selected = standardized_panelA_global_ncc_selected_runs(curves, standard, nicknames)
    missing = selected.loc[~selected["available"].astype(bool)]
    if not missing.empty:
        missing_text = "\n".join(
            f"{row.case} {row.source_method_id} lr={row.lr_label}"
            for row in missing.itertuples(index=False)
        )
        raise RuntimeError(f"Missing standardized Panel A global-NCC trajectories:\n{missing_text}")

    summary = panelA_global_ncc_curve_summary(curves, nicknames)
    selected = _sanitize_dataframe(selected)
    summary = _sanitize_dataframe(summary)
    selected.to_csv(output_dir / "fit_modes_standardized_global_ncc_panelA_selected_runs.csv", index=False)
    summary.to_csv(output_dir / "fit_modes_standardized_global_ncc_panelA_curve_summary.csv", index=False)
    fig = make_standardized_global_ncc_panelA_figure(curves, nicknames=nicknames)
    outputs = _save_figure_to_dir(
        fig,
        output_dir,
        "fit_modes_standardized_global_ncc_panelA_curves",
        output_formats=("png",),
    )
    one_minus_fig = make_standardized_panelA_one_minus_ncc_log_figure(curves, nicknames=nicknames)
    one_minus_outputs = _save_figure_to_dir(
        one_minus_fig,
        output_dir,
        "fit_modes_standardized_global_ncc_panelA_1minusncc_log_curves",
        output_formats=("png",),
    )
    _write_panelA_global_ncc_validation(output_dir, selected, standard)
    (output_dir / "fit_modes_standardized_global_ncc_panelA_manifest.json").write_text(
        json.dumps(
            sanitize_public_value(
                {
                    "artifact": "panelA_global_ncc_figure",
                    "version": standard["artifact_versions"]["fitting_figure"]["current"],
                    "standard_path": "experiments/paper_artifacts/artifact_standard.json",
                    "nickname_path": "experiments/paper_artifacts/method_nicknames.csv",
                    "metric": "global NCC loss = -NCC over the full OASIS image or full LungCT pTV crop",
                    "global_ncc_ylim": [-1.0, -0.8],
                    "one_minus_ncc_metric": "1 - NCC = 1 + global_ncc_loss",
                    "one_minus_ncc_log_ylim": [1.0e-3, 5.0e-1],
                    "selected_runs_csv": "fit_modes_standardized_global_ncc_panelA_selected_runs.csv",
                    "curve_summary_csv": "fit_modes_standardized_global_ncc_panelA_curve_summary.csv",
                    "validation_csv": "fit_modes_standardized_global_ncc_panelA_standard_validation.csv",
                    "validation_report": "fit_modes_standardized_global_ncc_panelA_standard_validation_report.md",
                    "outputs": {
                        "global_ncc_loss_png": outputs["png"].name,
                        "one_minus_ncc_log_png": one_minus_outputs["png"].name,
                    },
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return selected, summary, fig, outputs


def build_standardized_rrmse_field_fit_outputs(
    *,
    standard_path: str | Path | None = None,
    nickname_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, plt.Figure, dict[str, Path]]:
    standard = load_standard(standard_path)
    nicknames = load_nicknames(nickname_path)
    output_dir = artifact_output_dir("fitting_figure", standard)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / ".ipynb_checkpoints"
    if checkpoint_dir.exists():
        import shutil

        shutil.rmtree(checkpoint_dir)

    reference_field_curves = load_relative_field_fit_curves(selected_runs())
    powers = target_power_summary(reference_field_curves, RELATIVE_FIELD_CASE_ORDER)
    all_curves = load_rrmse_lr_trajectory_curves()
    curves = filter_standardized_field_fit_curves(all_curves, standard)
    selected = standardized_field_fit_selected_runs(curves, standard, nicknames)
    missing = selected.loc[~selected["available"].astype(bool)]
    if not missing.empty:
        missing_text = "\n".join(
            f"{row.case} {row.method} lr={row.lr_label}"
            for row in missing.itertuples(index=False)
        )
        raise RuntimeError(f"Missing standardized fitting trajectories:\n{missing_text}")

    summary = rrmse_lr_trajectory_curve_summary(curves, powers)
    summary.insert(3, "display_method", [
        _display_method(str(method), nicknames) for method in summary["method"]
    ])
    powers.to_csv(output_dir / "fit_modes_standardized_target_power_summary.csv", index=False)
    selected = _sanitize_dataframe(selected)
    summary = _sanitize_dataframe(summary)
    selected.to_csv(output_dir / "fit_modes_standardized_rrmse_selected_runs.csv", index=False)
    summary.to_csv(output_dir / "fit_modes_standardized_rrmse_curve_summary.csv", index=False)
    fig = make_standardized_rrmse_field_fit_figure(curves, powers, nicknames=nicknames)
    outputs = _save_figure_to_dir(
        fig,
        output_dir,
        "fit_modes_standardized_rrmse_field_fit_curves",
        output_formats=("png",),
    )
    _write_fitting_validation(output_dir, selected, standard)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(
            {
                "artifact": "fitting_figure",
                "version": standard["artifact_versions"]["fitting_figure"]["current"],
                "standard_path": "experiments/paper_artifacts/artifact_standard.json",
                "nickname_path": "experiments/paper_artifacts/method_nicknames.csv",
                "metric": "rRMSE = 100 * sqrt(mse / (target_power + 1e-12))",
                "selected_runs_csv": "fit_modes_standardized_rrmse_selected_runs.csv",
                "curve_summary_csv": "fit_modes_standardized_rrmse_curve_summary.csv",
                "target_power_csv": "fit_modes_standardized_target_power_summary.csv",
                "outputs": {key: path.name for key, path in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return selected, powers, summary, fig, outputs


def build_standardized_panelAB_outputs(
    *,
    standard_path: str | Path | None = None,
    nickname_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure, dict[str, Path]]:
    standard = load_standard(standard_path)
    nicknames = load_nicknames(nickname_path)
    output_dir = artifact_output_dir("fitting_figure", standard)
    output_dir.mkdir(parents=True, exist_ok=True)

    panelb_oasis_target = _merged_panelb_oasis_target(standard)

    panelA_curves = load_standardized_panelA_global_ncc_curves()
    panelA_selected = standardized_panelA_global_ncc_selected_runs(panelA_curves, standard, nicknames)
    missing_panelA = panelA_selected.loc[~panelA_selected["available"].astype(bool)]
    if not missing_panelA.empty:
        missing_text = "\n".join(
            f"{row.case} {row.source_method_id} lr={row.lr_label}"
            for row in missing_panelA.itertuples(index=False)
        )
        raise RuntimeError(f"Missing standardized Panel A global-NCC trajectories:\n{missing_text}")

    reference_field_curves = load_relative_field_fit_curves(selected_runs())
    powers = target_power_summary(reference_field_curves, RELATIVE_FIELD_CASE_ORDER)
    all_field_curves = load_rrmse_lr_trajectory_curves()
    field_curves = filter_standardized_field_fit_curves(all_field_curves, standard)
    field_selected = standardized_field_fit_selected_runs(field_curves, standard, nicknames)
    merged_selected = standardized_panelAB_selected_runs(
        panelA_selected,
        field_selected,
        panelb_oasis_target=panelb_oasis_target,
    )
    missing_merged = merged_selected.loc[~merged_selected["available"].astype(bool)]
    if not missing_merged.empty:
        missing_text = "\n".join(
            f"Panel {row.panel} {row.plot_case} {row.method_id} lr={row.lr_label}"
            for row in missing_merged.itertuples(index=False)
        )
        raise RuntimeError(f"Missing standardized merged Panel A/B trajectories:\n{missing_text}")

    summary = standardized_panelAB_curve_summary(
        panelA_curves,
        field_curves,
        powers,
        nicknames=nicknames,
        panelb_oasis_target=panelb_oasis_target,
    )
    merged_selected = _sanitize_dataframe(merged_selected)
    summary = _sanitize_dataframe(summary)
    merged_selected.to_csv(output_dir / "fit_modes_standardized_panelAB_selected_runs.csv", index=False)
    summary.to_csv(output_dir / "fit_modes_standardized_panelAB_curve_summary.csv", index=False)

    fig = make_standardized_panelAB_figure(
        panelA_curves,
        field_curves,
        powers,
        nicknames=nicknames,
        panelb_oasis_target=panelb_oasis_target,
    )
    outputs = _save_figure_to_dir(
        fig,
        output_dir,
        "fit_modes_standardized_panelAB_curves",
        output_formats=("png",),
    )
    _write_panelAB_validation(
        output_dir,
        panelA_selected,
        field_selected,
        panelb_oasis_target=panelb_oasis_target,
        standard=standard,
    )
    (output_dir / "fit_modes_standardized_panelAB_manifest.json").write_text(
        json.dumps(
            sanitize_public_value(
                {
                    "artifact": "fitting_figure_merged_panelAB",
                    "version": standard["artifact_versions"]["fitting_figure"]["current"],
                    "standard_path": "experiments/paper_artifacts/artifact_standard.json",
                    "nickname_path": "experiments/paper_artifacts/method_nicknames.csv",
                    "editable_selector": (
                        "artifact_figure_options.fitting_figure.merged_panel_b_oasis_target"
                    ),
                    "panel_b_oasis_target": panelb_oasis_target,
                    "panel_b_oasis_target_options": list(MERGED_PANELB_OASIS_TARGET_OPTIONS),
                    "panel_a_metric": "1 - NCC = 1 + global_ncc_loss over the full OASIS image or full LungCT pTV crop",
                    "panel_a_one_minus_ncc_log_ylim": [1.0e-3, 5.0e-1],
                    "panel_b_metric": "rRMSE = 100 * sqrt(mse / (target_power + 1e-12))",
                    "panel_b_rrmse_percent_ylim": [1.0e-1, 1.0e2],
                    "typography_and_spacing": {
                        "axis_label_size_pt": PANELAB_AXIS_LABEL_SIZE,
                        "axis_label_size_previous_pt": PANELAB_BASE_AXIS_LABEL_SIZE,
                        "axis_label_size_scale": PANELAB_AXIS_LABEL_SIZE / PANELAB_BASE_AXIS_LABEL_SIZE,
                        "legend_font_size_pt": PANELAB_LEGEND_FONT_SIZE,
                        "legend_font_size_previous_pt": PANELAB_BASE_LEGEND_FONT_SIZE,
                        "legend_font_size_scale": PANELAB_LEGEND_FONT_SIZE / PANELAB_BASE_LEGEND_FONT_SIZE,
                        "legend_anchor_y": PANELAB_LEGEND_ANCHOR_Y,
                        "legend_anchor_y_previous": PANELAB_BASE_LEGEND_ANCHOR_Y,
                    },
                    "selected_runs_csv": "fit_modes_standardized_panelAB_selected_runs.csv",
                    "curve_summary_csv": "fit_modes_standardized_panelAB_curve_summary.csv",
                    "validation_csv": "fit_modes_standardized_panelAB_standard_validation.csv",
                    "validation_report": "fit_modes_standardized_panelAB_standard_validation_report.md",
                    "outputs": {key: path.name for key, path in outputs.items()},
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return merged_selected, summary, fig, outputs


def save_main_figure(fig: plt.Figure, stem: str = "fit_modes_method_comparison_curves") -> dict[str, Path]:
    outputs = {
        "pdf": REVIEW_ROOT / f"{stem}.pdf",
        "png": REVIEW_ROOT / f"{stem}.png",
    }
    for path in outputs.values():
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return outputs


def save_relative_field_fit_figure(
    fig: plt.Figure,
    stem: str = "fit_modes_relative_field_fit_prelim_curves",
) -> dict[str, Path]:
    return save_main_figure(fig, stem=stem)


def save_rrmse_lr_trajectory_figure(
    fig: plt.Figure,
    stem: str = "fit_modes_rrmse_lr_trajectory_curves",
) -> dict[str, Path]:
    return save_main_figure(fig, stem=stem)


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


def build_relative_field_fit_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, plt.Figure, dict[str, Path]]:
    selection = selected_runs()
    field_curves = load_relative_field_fit_curves(selection)
    powers = target_power_summary(field_curves, RELATIVE_FIELD_CASE_ORDER)
    summary = relative_field_curve_summary(field_curves, powers)
    powers.to_csv(REVIEW_ROOT / "fit_modes_target_power_summary.csv", index=False)
    summary.to_csv(REVIEW_ROOT / "fit_modes_relative_field_fit_curve_summary.csv", index=False)
    fig = make_relative_field_fit_figure(field_curves, powers, case_order=RELATIVE_FIELD_CASE_ORDER)
    outputs = save_relative_field_fit_figure(fig)
    return selection, powers, summary, fig, outputs


def build_rrmse_lr_trajectory_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, plt.Figure, dict[str, Path]]:
    selection = selected_runs()
    reference_field_curves = load_relative_field_fit_curves(selection)
    powers = target_power_summary(reference_field_curves, RELATIVE_FIELD_CASE_ORDER)
    curves = load_rrmse_lr_trajectory_curves()
    selected = rrmse_lr_trajectory_selected_runs(curves)
    summary = rrmse_lr_trajectory_curve_summary(curves, powers)
    selected.to_csv(REVIEW_ROOT / "fit_modes_rrmse_lr_trajectory_selected_runs.csv", index=False)
    summary.to_csv(REVIEW_ROOT / "fit_modes_rrmse_lr_trajectory_curve_summary.csv", index=False)
    fig = make_rrmse_lr_trajectory_figure(curves, powers)
    outputs = save_rrmse_lr_trajectory_figure(fig)
    return selected, powers, summary, fig, outputs
