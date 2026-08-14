#!/usr/bin/env python
"""Posthoc fixed-lung-mask diagnostics for DIR-LAB fit-modes case panels.

This is a focused audit/repair script for the camera-ready diagnostic question:
registration Panel A curves were optimized and plotted with full-crop global NCC,
while IDIR's LungCT training domain is the fixed T00 lung mask.  The script does
not rerun optimization and does not modify the camera-ready source index.  It
loads the final saved dense fields, reconstructs warped moving images, and
reports final registration ``1-NCC`` inside the fixed T00 lung mask for case01
and case08.  Panel B field-fitting metrics are read from the existing summaries,
whose objective metric is already fixed-mask MSE/rRMSE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from plot_case_diagnostics import METHOD_ORDER, METHOD_STYLE


REPO_ROOT = Path(__file__).resolve().parents[3]
LUNGCT_RUNNER_DIR = REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro"
if str(LUNGCT_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(LUNGCT_RUNNER_DIR))
import common as lung_common  # noqa: E402


DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT / "experiments/2026-08-14_dirlab_case01_case08_lungmask_diagnostics"
)
DEFAULT_SOURCE_INDEX = (
    REPO_ROOT
    / "experiments/paper_artifacts/fit_modes_population_diagnostics/v001_schedule_matched"
    / "camera_ready_source_index.csv"
)
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "DATA_ROOT",
        "/database/inrorspline",
    )
)
DEFAULT_PTV_ROOT = Path(
    os.environ.get(
        "DIRLAB_PTVREG_ISO1MM_ROOT",
        "/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm",
    )
)
DEFAULT_POPULATION_RUN_ROOT = Path(
    os.environ.get("POPULATION_RUN_ROOT", str(DEFAULT_DATA_ROOT / "runs/fit_modes_population_n10_v1"))
)
DEFAULT_REFRESH_RUN_ROOT = Path(
    os.environ.get(
        "DIRLAB_REFRESH_RUN_ROOT",
        str(DEFAULT_DATA_ROOT / "runs/fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1"),
    )
)

DATA_TOKEN = "${DATA_ROOT}"
POPULATION_TOKEN = "${POPULATION_RUN_ROOT}"
REFRESH_TOKEN = "${DIRLAB_REFRESH_RUN_ROOT}"
PTV_TOKEN = "${DIRLAB_PTVREG_ISO1MM_ROOT}"
TARGET_CASES = ("case01", "case08")
CASE_IDS = {"case01": 1, "case08": 8}
EXPECTED_MRDBSCP_CPS = "32,16,8,4"
EXPECTED_MRDBSCP_DOWNSAMPLE = "8,4,2,1"
EXPECTED_MRDBSCP_EPOCHS = "400,400,800,900"
EXPECTED_TOTAL_EPOCHS = 2500
FULL_CROP_NCC_TOLERANCE = 1.0e-4


@dataclass(frozen=True)
class ArgsForWorker:
    data_root: str
    ptv_root: str
    population_run_root: str
    refresh_run_root: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ptv-root", type=Path, default=DEFAULT_PTV_ROOT)
    parser.add_argument("--population-run-root", type=Path, default=DEFAULT_POPULATION_RUN_ROOT)
    parser.add_argument("--refresh-run-root", type=Path, default=DEFAULT_REFRESH_RUN_ROOT)
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated CUDA device ids for Panel-A recomputation; use 'cpu' for CPU.",
    )
    parser.add_argument("--full-crop-ncc-tolerance", type=float, default=FULL_CROP_NCC_TOLERANCE)
    return parser.parse_args()


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> dict[str, Any]:
    def run_git(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "commit": run_git(["git", "rev-parse", "HEAD"]),
        "branch": run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run_git(["git", "status", "--porcelain"])),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def portable_path(path: Path, *, data_root: Path, ptv_root: Path, population_run_root: Path, refresh_run_root: Path) -> str:
    text = str(path)
    for root, token in [
        (refresh_run_root, REFRESH_TOKEN),
        (population_run_root, POPULATION_TOKEN),
        (ptv_root, PTV_TOKEN),
        (data_root, DATA_TOKEN),
        (REPO_ROOT, "${REPO_ROOT}"),
    ]:
        root_text = str(root)
        if text == root_text or text.startswith(root_text + "/"):
            return text.replace(root_text, token, 1)
    return path.name


def resolve_token_path(
    value: Any,
    *,
    data_root: Path,
    ptv_root: Path,
    population_run_root: Path,
    refresh_run_root: Path,
) -> Path:
    text = str(value or "")
    replacements = {
        DATA_TOKEN: str(data_root),
        POPULATION_TOKEN: str(population_run_root),
        REFRESH_TOKEN: str(refresh_run_root),
        PTV_TOKEN: str(ptv_root),
        "${REPO_ROOT}": str(REPO_ROOT),
    }
    for token, root in replacements.items():
        text = text.replace(token, root)
    return Path(text)


def parse_gpus(value: str) -> list[str]:
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise ValueError("--gpus cannot be empty")
    if len(tokens) == 1 and tokens[0].lower() == "cpu":
        return ["cpu"]
    return tokens


def select_records(index: pd.DataFrame) -> pd.DataFrame:
    selected = index[
        index["dataset"].eq("dirlab")
        & index["case_key"].isin(TARGET_CASES)
        & index["panel"].isin(["A", "B"])
        & index["method_slug"].isin(METHOD_ORDER)
    ].copy()
    if len(selected) != 16:
        raise RuntimeError(f"Expected 16 case01/case08 DIR-LAB records, found {len(selected)}")
    if selected.duplicated(["case_key", "panel", "method_slug"]).any():
        raise RuntimeError("Duplicate selected records in source index")
    selected["case_order"] = selected["case_key"].map({case: i for i, case in enumerate(TARGET_CASES)})
    selected["panel_order"] = selected["panel"].map({"A": 0, "B": 1})
    selected["method_order"] = selected["method_slug"].map({method: i for i, method in enumerate(METHOD_ORDER)})
    return selected.sort_values(["case_order", "panel_order", "method_order"]).reset_index(drop=True)


def assert_mrdbscp_refresh(row: pd.Series, summary: dict[str, Any], summary_path: Path) -> None:
    if str(row["panel"]) != "A" or str(row["method_slug"]) != "mr_d_bscp":
        return
    checks = {
        "source_kind": str(row.get("source_kind", "")) == "schedule_refresh_diagnostic_enabled",
        "summary_path_refresh_root": str(row.get("summary_path", "")).startswith(REFRESH_TOKEN),
        "cps": str(row.get("cps", "")) == EXPECTED_MRDBSCP_CPS,
        "stage_downsample": str(row.get("stage_downsample", "")) == EXPECTED_MRDBSCP_DOWNSAMPLE,
        "stage_epochs": str(row.get("stage_epochs", "")) == EXPECTED_MRDBSCP_EPOCHS,
        "epochs": int(float(row.get("epochs"))) == EXPECTED_TOTAL_EPOCHS,
        "diagnostic_interval": int(float(row.get("diagnostic_interval"))) == 10,
        "summary_stage_cps": nested(summary, "training", "stage_full_cps") == [32, 16, 8, 4],
        "summary_stage_downsample": nested(summary, "training", "stage_downsample") == [8, 4, 2, 1],
        "summary_stage_epochs": nested(summary, "training", "stage_epochs") == [400, 400, 800, 900],
        "summary_diagnostic_interval": int(nested(summary, "training", "global_ncc_diagnostic_interval") or -1)
        == 10,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"MR-D-BSCP refresh validation failed for {summary_path}: {failed}")


def masked_one_minus_ncc(
    fixed_t: torch.Tensor,
    warped_t: torch.Tensor,
    mask_t: torch.Tensor,
    eps: float = 1.0e-10,
) -> float:
    fixed_values = fixed_t[mask_t]
    warped_values = warped_t[mask_t]
    if int(fixed_values.numel()) <= 1:
        raise RuntimeError("Cannot compute masked NCC on an empty/singleton mask")
    cc = ((fixed_values - fixed_values.mean()) * (warped_values - warped_values.mean())).mean()
    std = torch.std(fixed_values) * torch.std(warped_values)
    ncc = cc / (std + eps)
    return 1.0 - float(ncc.detach().cpu())


def recompute_panel_a_worker(payload: tuple[dict[str, Any], str, ArgsForWorker]) -> dict[str, Any]:
    row_dict, gpu, worker_args = payload
    data_root = Path(worker_args.data_root)
    ptv_root = Path(worker_args.ptv_root)
    population_run_root = Path(worker_args.population_run_root)
    refresh_run_root = Path(worker_args.refresh_run_root)

    if gpu == "cpu":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        device = torch.device(f"cuda:{int(gpu)}")

    row = pd.Series(row_dict)
    case_key = str(row["case_key"])
    case_id = CASE_IDS[case_key]
    summary_path = resolve_token_path(
        row["summary_path"],
        data_root=data_root,
        ptv_root=ptv_root,
        population_run_root=population_run_root,
        refresh_run_root=refresh_run_root,
    )
    source_path = resolve_token_path(
        row["source_path"],
        data_root=data_root,
        ptv_root=ptv_root,
        population_run_root=population_run_root,
        refresh_run_root=refresh_run_root,
    )
    run_dir = summary_path.parent
    disp_path = run_dir / "disp_norm_xyz_reg1mm.npy"
    curve_path = source_path
    required = [summary_path, curve_path, disp_path]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Panel-A files:\n" + "\n".join(map(str, missing)))

    summary = load_json(summary_path)
    assert_mrdbscp_refresh(row, summary, summary_path)
    case = lung_common.load_case(ptv_root, case_id, dataset="4dct")
    fixed_t = torch.from_numpy(case.fixed.astype(np.float32, copy=False)).to(device)
    moving_t = torch.from_numpy(case.moving.astype(np.float32, copy=False)).to(device)
    mask_t = torch.from_numpy(case.fixed_mask.astype(bool, copy=False)).to(device)
    disp_t = torch.from_numpy(np.load(disp_path).astype(np.float32, copy=False)).to(device)
    if tuple(disp_t.shape) != tuple(case.fixed.shape) + (3,):
        raise ValueError(f"{case_key}/{row['method_slug']}: displacement shape {tuple(disp_t.shape)}")

    id_grid = lung_common.make_coordinate_grid(case.fixed.shape, device).detach()
    with torch.no_grad():
        warped_t = lung_common.warp_full(moving_t, id_grid + disp_t, mode="bilinear")
        full_loss = lung_common.ncc_loss(fixed_t[None, None], warped_t[None, None])
        full_one_minus_ncc = 1.0 + float(full_loss.detach().cpu())
        lung_one_minus_ncc = masked_one_minus_ncc(fixed_t, warped_t, mask_t)

    source_one_minus_ncc = float(row["endpoint_value"])
    full_abs_diff = abs(full_one_minus_ncc - source_one_minus_ncc)
    if gpu != "cpu" and device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))
        torch.cuda.empty_cache()
    else:
        peak_memory_gb = float("nan")

    return {
        "job_id": str(row["job_id"]),
        "dataset": "dirlab",
        "case_key": case_key,
        "case_id": int(case_id),
        "panel": "A",
        "objective": str(row["objective"]),
        "method_slug": str(row["method_slug"]),
        "display_method": METHOD_STYLE[str(row["method_slug"])]["label"],
        "implementation_id": str(row["implementation_id"]),
        "metric_domain": "fixed_t00_lung_mask_reg1mm",
        "metric_name": "lung_mask_one_minus_ncc",
        "lung_mask_one_minus_ncc": float(lung_one_minus_ncc),
        "source_full_crop_one_minus_ncc": source_one_minus_ncc,
        "recomputed_full_crop_one_minus_ncc": float(full_one_minus_ncc),
        "full_crop_recompute_abs_diff": float(full_abs_diff),
        "fixed_lung_mask_voxels": int(case.fixed_mask.sum()),
        "fixed_lung_mask_fraction": float(case.fixed_mask.mean()),
        "shape_zyx": "x".join(str(int(v)) for v in case.fixed.shape),
        "summary_path": portable_path(
            summary_path,
            data_root=data_root,
            ptv_root=ptv_root,
            population_run_root=population_run_root,
            refresh_run_root=refresh_run_root,
        ),
        "source_path": portable_path(
            source_path,
            data_root=data_root,
            ptv_root=ptv_root,
            population_run_root=population_run_root,
            refresh_run_root=refresh_run_root,
        ),
        "disp_path": portable_path(
            disp_path,
            data_root=data_root,
            ptv_root=ptv_root,
            population_run_root=population_run_root,
            refresh_run_root=refresh_run_root,
        ),
        "lr": finite(row.get("lr")),
        "alpha": finite(row.get("alpha")),
        "cps": row.get("cps", ""),
        "stage_downsample": row.get("stage_downsample", ""),
        "stage_epochs": row.get("stage_epochs", ""),
        "epochs": int(float(row.get("epochs"))),
        "seed": int(float(row.get("seed"))),
        "diagnostic_interval": int(float(row.get("diagnostic_interval"))),
        "tre_snap_to_voxel_mean_mm": finite(nested(summary, "tre_snap_to_voxel", "tre_mean_mm")),
        "runtime_sec": finite(row.get("runtime_sec")),
        "peak_memory_gb_source": finite(row.get("peak_memory_gb")),
        "recompute_gpu": gpu,
        "recompute_peak_memory_gb": peak_memory_gb,
        "status": "completed",
    }


def panel_b_row(
    row: pd.Series,
    *,
    data_root: Path,
    ptv_root: Path,
    population_run_root: Path,
    refresh_run_root: Path,
) -> dict[str, Any]:
    case_key = str(row["case_key"])
    case_id = CASE_IDS[case_key]
    summary_path = resolve_token_path(
        row["summary_path"],
        data_root=data_root,
        ptv_root=ptv_root,
        population_run_root=population_run_root,
        refresh_run_root=refresh_run_root,
    )
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = load_json(summary_path)
    if str(row["method_slug"]) == "mr_d_bscp":
        training = summary.get("training", {})
        checks = {
            "cps": str(row.get("cps", "")) == EXPECTED_MRDBSCP_CPS,
            "stage_downsample": str(row.get("stage_downsample", "")) == EXPECTED_MRDBSCP_DOWNSAMPLE,
            "stage_epochs": str(row.get("stage_epochs", "")) == EXPECTED_MRDBSCP_EPOCHS,
            "epochs": int(float(row.get("epochs"))) == EXPECTED_TOTAL_EPOCHS,
            "summary_stage_epochs": training.get("stage_epochs") == [400, 400, 800, 900],
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"Panel-B MR-D-BSCP schedule validation failed for {summary_path}: {failed}")

    case = lung_common.load_case(ptv_root, case_id, dataset="4dct")
    mse = finite(summary.get("field_mse_objective_normalized_grid"))
    rmse = finite(summary.get("field_rmse_objective_normalized_grid"))
    target_power = finite(summary.get("field_target_power_masked_normalized_grid"))
    if target_power is None:
        target_power = finite(row.get("field_target_power_masked_normalized_grid"))
    rrmse = finite(summary.get("field_rrmse_objective_percent"))
    if rrmse is None:
        rrmse = finite(row.get("endpoint_value"))
    if rrmse is None and mse is not None and target_power is not None:
        rrmse = 100.0 * math.sqrt(float(mse) / (float(target_power) + 1.0e-12))
    if None in (mse, rmse, target_power, rrmse):
        raise RuntimeError(f"Missing finite Panel-B masked metric in {summary_path}")

    return {
        "job_id": str(row["job_id"]),
        "dataset": "dirlab",
        "case_key": case_key,
        "case_id": int(case_id),
        "panel": "B",
        "objective": str(row["objective"]),
        "method_slug": str(row["method_slug"]),
        "display_method": METHOD_STYLE[str(row["method_slug"])]["label"],
        "implementation_id": str(row["implementation_id"]),
        "metric_domain": "fixed_t00_lung_mask_reg1mm",
        "metric_name": "field_fit_masked_rrmse_percent",
        "field_mse_objective_normalized_grid": float(mse),
        "field_rmse_objective_normalized_grid": float(rmse),
        "field_target_power_masked_normalized_grid": float(target_power),
        "field_rrmse_objective_percent": float(rrmse),
        "source_index_rrmse_percent": float(row["endpoint_value"]),
        "fixed_lung_mask_voxels": int(case.fixed_mask.sum()),
        "fixed_lung_mask_fraction": float(case.fixed_mask.mean()),
        "summary_path": portable_path(
            summary_path,
            data_root=data_root,
            ptv_root=ptv_root,
            population_run_root=population_run_root,
            refresh_run_root=refresh_run_root,
        ),
        "source_path": str(row["source_path"]),
        "lr": finite(row.get("lr")),
        "alpha": finite(row.get("alpha")),
        "cps": row.get("cps", ""),
        "stage_downsample": row.get("stage_downsample", ""),
        "stage_epochs": row.get("stage_epochs", ""),
        "epochs": int(float(row.get("epochs"))),
        "seed": int(float(row.get("seed"))),
        "diagnostic_interval": int(float(row.get("diagnostic_interval"))),
        "runtime_sec": finite(row.get("runtime_sec")),
        "peak_memory_gb_source": finite(row.get("peak_memory_gb")),
        "status": "completed",
    }


def source_record(row: pd.Series) -> dict[str, Any]:
    return {
        key: row.get(key, "")
        for key in [
            "job_id",
            "dataset",
            "case_key",
            "case_id",
            "panel",
            "objective",
            "method_slug",
            "display_method",
            "implementation_id",
            "source_kind",
            "source_path",
            "summary_path",
            "endpoint_name",
            "endpoint_value",
            "lr",
            "alpha",
            "cps",
            "stage_downsample",
            "stage_epochs",
            "epochs",
            "seed",
            "diagnostic_interval",
        ]
    }


def summary_stats(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for panel in ["A", "B"]:
        for case_key in TARGET_CASES:
            subset = [r for r in rows if r["panel"] == panel and r["case_key"] == case_key]
            values = [
                float(r["lung_mask_one_minus_ncc"] if panel == "A" else r["field_rrmse_objective_percent"])
                for r in subset
            ]
            out.append(
                {
                    "case_key": case_key,
                    "panel": panel,
                    "n_methods": len(values),
                    "metric": "lung_mask_one_minus_ncc" if panel == "A" else "field_rrmse_objective_percent",
                    "mean_across_methods": float(np.mean(values)) if values else float("nan"),
                    "min_across_methods": float(np.min(values)) if values else float("nan"),
                    "max_across_methods": float(np.max(values)) if values else float("nan"),
                }
            )
    return out


def render_report(path: Path, rows: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> None:
    panel_a = [r for r in rows if r["panel"] == "A"]
    panel_b = [r for r in rows if r["panel"] == "B"]

    lines = [
        "# DIR-LAB case01/case08 fixed-lung-mask diagnostic audit",
        "",
        f"Generated: `{manifest['generated_at_utc']}`",
        "",
        "This diagnostic-only pass recomputes/report final metrics for DIR-LAB case01 and case08 using the fixed T00 lung mask. It does not rerun optimization, edit the source index, or regenerate figures.",
        "",
        "## Key convention",
        "",
        "- Panel A registration metric: final `1-NCC` recomputed inside the fixed T00 lung mask on the canonical 1 mm crop.",
        "- Panel A comparison column: existing source-index full-crop `1-NCC` endpoint.",
        "- Panel B supervised fitting metric: fixed-mask objective MSE/rRMSE from the accepted summaries.",
        "- MR-D-BSCP Panel A uses the accepted refreshed `400/400/800/900` schedule with total 2500 epochs.",
        "",
        "## Panel A: registration",
        "",
        "| Case | Method | Lung-mask 1-NCC | Full-crop 1-NCC | Full-crop recompute diff | TRE (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in panel_a:
        lines.append(
            "| {case_key} | {display_method} | {lung_mask_one_minus_ncc:.6f} | "
            "{source_full_crop_one_minus_ncc:.6f} | {full_crop_recompute_abs_diff:.2e} | "
            "{tre_snap_to_voxel_mean_mm:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Panel B: supervised field fitting",
            "",
            "| Case | Method | Masked rRMSE (%) | Masked MSE | Target power |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in panel_b:
        lines.append(
            "| {case_key} | {display_method} | {field_rrmse_objective_percent:.6f} | "
            "{field_mse_objective_normalized_grid:.6e} | "
            "{field_target_power_masked_normalized_grid:.6e} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
        ]
    )
    for key, value in manifest["validation"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation note",
            "",
            "This is the fairer LungCT diagnostic domain for comparing the case01 and case08 registration panels against IDIR, because all methods are evaluated inside the same fixed T00 lung mask.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def image_record(path: Path, output_dir: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": os.path.relpath(path, output_dir),
        "bytes": path.stat().st_size,
        "width_px": width,
        "height_px": height,
        "sha256": sha256(path),
    }


def render_before_after_2x2(path: Path, rows: Sequence[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    by_key = {(row["case_key"], row["panel"], row["method_slug"]): row for row in rows}
    colors = [METHOD_STYLE[method]["color"] for method in METHOD_ORDER]
    labels = [METHOD_STYLE[method]["label"] for method in METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), dpi=200, sharex=True)
    for col, case_key in enumerate(TARGET_CASES):
        ax = axes[0, col]
        before = np.array(
            [float(by_key[(case_key, "A", method)]["source_full_crop_one_minus_ncc"]) for method in METHOD_ORDER]
        )
        after = np.array(
            [float(by_key[(case_key, "A", method)]["lung_mask_one_minus_ncc"]) for method in METHOD_ORDER]
        )
        ax.bar(x - width / 2, before, width=width, color=colors, alpha=0.32, edgecolor=colors, linewidth=1.1)
        ax.bar(x + width / 2, after, width=width, color=colors, alpha=0.88, edgecolor=colors, linewidth=1.1)
        ax.set_yscale("log")
        ax.set_ylim(1.0e-3, 2.0e-1)
        ax.set_title(f"DIR-LAB {case_key}: registration", fontsize=13)
        ax.set_ylabel("Panel A 1−NCC", fontsize=12)
        ax.grid(True, which="both", color="#D8DEE9", linewidth=0.75, alpha=0.75)

        ax = axes[1, col]
        before_b = np.array(
            [float(by_key[(case_key, "B", method)]["source_index_rrmse_percent"]) for method in METHOD_ORDER]
        )
        after_b = np.array(
            [float(by_key[(case_key, "B", method)]["field_rrmse_objective_percent"]) for method in METHOD_ORDER]
        )
        ax.bar(x - width / 2, before_b, width=width, color=colors, alpha=0.32, edgecolor=colors, linewidth=1.1)
        ax.bar(x + width / 2, after_b, width=width, color=colors, alpha=0.88, edgecolor=colors, linewidth=1.1)
        ax.set_yscale("log")
        ax.set_ylim(1.0e-1, 2.0e1)
        ax.set_title(f"DIR-LAB {case_key}: supervised fitting", fontsize=13)
        ax.set_ylabel("Panel B rRMSE (%)", fontsize=12)
        ax.grid(True, which="both", color="#D8DEE9", linewidth=0.75, alpha=0.75)
        ax.text(
            0.02,
            0.96,
            "unchanged: already fixed-mask",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
        )

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=10)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.9)

    legend_handles = [
        Line2D([0], [0], color="#333333", lw=8, alpha=0.32, label="before: source-index/full-crop for Panel A"),
        Line2D([0], [0], color="#333333", lw=8, alpha=0.88, label="after: fixed T00 lung mask"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Before/after LungCT metric-domain fix for case01 and case08", fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.18, wspace=0.20, hspace=0.34)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return image_record(path, output_dir)


def main() -> None:
    args = parse_args()
    index = pd.read_csv(args.source_index)
    selected = select_records(index)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = parse_gpus(args.gpus)
    worker_args = ArgsForWorker(
        data_root=str(args.data_root),
        ptv_root=str(args.ptv_root),
        population_run_root=str(args.population_run_root),
        refresh_run_root=str(args.refresh_run_root),
    )

    source_records = [source_record(row) for _, row in selected.iterrows()]
    write_csv(output_dir / "lungmask_diagnostic_source_index.csv", source_records)

    panel_a_selected = selected[selected["panel"].eq("A")].reset_index(drop=True)
    tasks = [
        (row.to_dict(), gpus[i % len(gpus)], worker_args)
        for i, (_, row) in enumerate(panel_a_selected.iterrows())
    ]
    panel_a_rows: list[dict[str, Any]] = []
    max_workers = 1 if gpus == ["cpu"] else min(len(gpus), len(tasks))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(recompute_panel_a_worker, task) for task in tasks]
        for future in as_completed(futures):
            panel_a_rows.append(future.result())

    panel_b_rows = [
        panel_b_row(
            row,
            data_root=args.data_root,
            ptv_root=args.ptv_root,
            population_run_root=args.population_run_root,
            refresh_run_root=args.refresh_run_root,
        )
        for _, row in selected[selected["panel"].eq("B")].iterrows()
    ]
    rows = sorted(
        panel_a_rows + panel_b_rows,
        key=lambda row: (
            TARGET_CASES.index(row["case_key"]),
            0 if row["panel"] == "A" else 1,
            METHOD_ORDER.index(row["method_slug"]),
        ),
    )

    full_crop_diffs = [float(row["full_crop_recompute_abs_diff"]) for row in panel_a_rows]
    validation = {
        "selected_records": len(selected),
        "panel_a_records": len(panel_a_rows),
        "panel_b_records": len(panel_b_rows),
        "unique_case_panel_method_records": len(
            {(row["case_key"], row["panel"], row["method_slug"]) for row in rows}
        ),
        "panel_a_finite_lung_mask_one_minus_ncc": sum(
            math.isfinite(float(row["lung_mask_one_minus_ncc"])) for row in panel_a_rows
        ),
        "panel_b_finite_masked_rrmse": sum(
            math.isfinite(float(row["field_rrmse_objective_percent"])) for row in panel_b_rows
        ),
        "max_full_crop_recompute_abs_diff": max(full_crop_diffs) if full_crop_diffs else float("nan"),
        "full_crop_recompute_tolerance": float(args.full_crop_ncc_tolerance),
        "full_crop_recompute_pass": bool(
            full_crop_diffs and max(full_crop_diffs) <= float(args.full_crop_ncc_tolerance)
        ),
        "mrdbscp_panelA_refresh_records": sum(
            row["panel"] == "A"
            and row["method_slug"] == "mr_d_bscp"
            and str(row["summary_path"]).startswith(REFRESH_TOKEN)
            and str(row["stage_epochs"]) == EXPECTED_MRDBSCP_EPOCHS
            for row in rows
        ),
        "expected_mrdbscp_panelA_refresh_records": 2,
        "case01_mask_voxels": next(
            int(row["fixed_lung_mask_voxels"]) for row in rows if row["case_key"] == "case01"
        ),
        "case08_mask_voxels": next(
            int(row["fixed_lung_mask_voxels"]) for row in rows if row["case_key"] == "case08"
        ),
    }
    validation["pass"] = bool(
        validation["selected_records"] == 16
        and validation["panel_a_records"] == 8
        and validation["panel_b_records"] == 8
        and validation["unique_case_panel_method_records"] == 16
        and validation["panel_a_finite_lung_mask_one_minus_ncc"] == 8
        and validation["panel_b_finite_masked_rrmse"] == 8
        and validation["full_crop_recompute_pass"]
        and validation["mrdbscp_panelA_refresh_records"] == 2
    )

    write_csv(output_dir / "lungmask_metric_summary.csv", rows)
    write_csv(output_dir / "lungmask_metric_case_averages.csv", summary_stats(rows))

    figure_record = render_before_after_2x2(output_dir / "lungmask_before_after_2x2.png", rows, output_dir)

    manifest = {
        "generated_at_utc": utc_text(),
        "script": "scripts/experiments/fit_modes_population/recompute_dirlab_lungmask_case_diagnostics.py",
        "source_index": str(args.source_index),
        "source_index_sha256": sha256(args.source_index),
        "outputs": {
            "source_index": "lungmask_diagnostic_source_index.csv",
            "summary": "lungmask_metric_summary.csv",
            "case_averages": "lungmask_metric_case_averages.csv",
            "report": "lungmask_metric_comparison.md",
            "technical_log": "technical_log.md",
            "before_after_figure": "lungmask_before_after_2x2.png",
        },
        "figures": {"before_after_2x2": figure_record},
        "roots": {
            "data_root_token": DATA_TOKEN,
            "population_run_root_token": POPULATION_TOKEN,
            "refresh_run_root_token": REFRESH_TOKEN,
            "ptv_root_token": PTV_TOKEN,
        },
        "gpu_request": args.gpus,
        "git": git_revision(),
        "validation": validation,
    }
    write_json(output_dir / "lungmask_metric_manifest.json", manifest)
    render_report(output_dir / "lungmask_metric_comparison.md", rows, manifest)

    technical_log = [
        "# Technical log",
        "",
        "## Known limitations",
        "",
        "- Diagnostic-only pass; no optimization was rerun.",
        "- The output covers only DIR-LAB case01 and case08.",
        "- Panel A reports final masked NCC endpoints only, not full masked diagnostic curves over all epochs.",
        "- Existing full-crop figures and `camera_ready_source_index.csv` are intentionally unchanged.",
        "",
        "## Execution record",
        "",
        f"- Generated: `{manifest['generated_at_utc']}`",
        f"- Script: `{manifest['script']}`",
        f"- GPU request: `{args.gpus}`",
        f"- Validation pass: `{validation['pass']}`",
        f"- MR-D-BSCP Panel A schedule: `{EXPECTED_MRDBSCP_EPOCHS}` / total `{EXPECTED_TOTAL_EPOCHS}`",
    ]
    (output_dir / "technical_log.md").write_text("\n".join(technical_log) + "\n")

    print(json.dumps({"output_dir": str(output_dir), "validation": validation}, indent=2, sort_keys=True))
    if not validation["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
