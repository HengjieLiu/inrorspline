#!/usr/bin/env python
"""Plot DirectCP 4-stage finest-domain loss diagnostics next to sweep plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import aggregate_report as base_report  # noqa: E402
import common as c  # noqa: E402


METHOD_DIRECT_CP_4STAGE = "direct_cp_32_16_8_4_ptv_lcc_isotv"
DIAG_PLOT_NAME = "best_lr_registration_loss_spline_methods_finest_domain_directcp4stage_lr0p01.png"
COLORS = {
    "sinr_cps4": "#2ca02c",
    "direct_cp_cps4": "#d62728",
    "direct_cp_4stage": "#9467bd",
}
LABELS = {
    "sinr_cps4": "SINR cps4",
    "direct_cp_cps4": "DirectCP cps4",
    "direct_cp_4stage": "DirectCP 4stage cps4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default="1,8")
    parser.add_argument(
        "--diag-root",
        type=Path,
        default=Path("/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_v1"),
    )
    parser.add_argument("--accepted-review-root-template", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    return parser.parse_args()


def case_review_root(case_id: int, template: str) -> Path:
    if template:
        return Path(template.format(case_id=int(case_id), case_id02=f"{int(case_id):02d}"))
    return c.REPO_ROOT / f"experiments/2026-06-26_lungct_case{int(case_id):02d}_ptvreg_fit_modes_lr_sweep"


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_float_id(value: float) -> str:
    return c.format_float(float(value))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_history(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    return read_csv(path)


def accepted_best_row(review_root: Path, alias: str) -> Mapping[str, str]:
    rows = read_csv(review_root / "summary_runs.csv")
    candidates = [
        row
        for row in rows
        if row.get("objective_group") == "image_registration"
        and row.get("method_alias") == alias
        and math.isfinite(as_float(row.get("native_crop_snap_tre_mean_mm")))
    ]
    if not candidates:
        raise RuntimeError(f"No accepted image-registration row for {alias} under {review_root}")
    return min(candidates, key=lambda row: as_float(row.get("native_crop_snap_tre_mean_mm")))


def accepted_curve(row: Mapping[str, str]) -> Tuple[List[float], List[float]]:
    run_dir = Path(str(row["run_dir"]))
    hist = read_history(run_dir / "loss_history.csv")
    pairs = []
    for item in hist:
        x = as_float(item.get("epoch"))
        y = as_float(item.get("loss"))
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if not pairs:
        raise RuntimeError(f"No finite accepted curve values in {run_dir}")
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def find_diag_run(diag_root: Path, case_id: int, lr: float) -> Tuple[Path, Dict[str, object]]:
    method_dir = diag_root / METHOD_DIRECT_CP_4STAGE
    pattern = f"*case{int(case_id):02d}*lr{format_float_id(lr)}*/summary.json"
    candidates: List[Tuple[float, Path, Dict[str, object]]] = []
    for summary_path in method_dir.glob(pattern):
        payload = json.loads(summary_path.read_text())
        training = payload.get("training", {})
        if not isinstance(training, dict):
            training = {}
        stage_lr = training.get("stage_lr", [])
        first_lr = as_float(stage_lr[0]) if isinstance(stage_lr, list) and stage_lr else float("nan")
        if (
            int(as_float(payload.get("case_id"))) == int(case_id)
            and payload.get("method") == METHOD_DIRECT_CP_4STAGE
            and payload.get("status") == "completed"
            and bool(training.get("finest_loss_diagnostic"))
            and abs(first_lr - float(lr)) <= 1e-12
        ):
            candidates.append((summary_path.stat().st_mtime, summary_path, payload))
    if not candidates:
        raise RuntimeError(f"No completed finest-domain diagnostic run found for case{case_id:02d} under {diag_root}")
    _mtime, summary_path, payload = sorted(candidates, key=lambda item: item[0])[-1]
    return summary_path.parent, payload


def diag_curve(run_dir: Path) -> Tuple[List[float], List[float]]:
    hist = read_history(run_dir / "stage_history.csv")
    pairs = []
    for item in hist:
        x = as_float(item.get("global_epoch"))
        y = as_float(item.get("finest_loss"))
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if not pairs:
        raise RuntimeError(f"No finite finest-domain loss values in {run_dir / 'stage_history.csv'}")
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def diag_native_snap_tre(summary_path: Path, payload: Mapping[str, object]) -> float:
    native = base_report.native_crop_tre_from_summary(summary_path, dict(payload))
    if not isinstance(native, dict):
        return float("nan")
    snap = native.get("snap", {})
    if not isinstance(snap, dict):
        return float("nan")
    return as_float(snap.get("tre_mean_mm"))


def plot_case(case_id: int, args: argparse.Namespace) -> Path:
    review_root = case_review_root(case_id, str(args.accepted_review_root_template))
    if args.output_dir is None:
        out_path = review_root / "plots" / DIAG_PLOT_NAME
    else:
        out_path = Path(args.output_dir) / f"case{int(case_id):02d}_{DIAG_PLOT_NAME}"

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2))
    for alias in ["sinr_cps4", "direct_cp_cps4"]:
        row = accepted_best_row(review_root, alias)
        xs, ys = accepted_curve(row)
        lr = as_float(row.get("lr"))
        tre = as_float(row.get("native_crop_snap_tre_mean_mm"))
        ax.plot(
            xs,
            ys,
            color=COLORS[alias],
            linewidth=1.8,
            label=f"{LABELS[alias]} lr={lr:g} (TRE={tre:.4f} mm)",
        )

    diag_run_dir, payload = find_diag_run(Path(args.diag_root), case_id, float(args.lr))
    xs, ys = diag_curve(diag_run_dir)
    diag_tre = diag_native_snap_tre(diag_run_dir / "summary.json", payload)
    ax.plot(
        xs,
        ys,
        color=COLORS["direct_cp_4stage"],
        linewidth=1.8,
        label=f"{LABELS['direct_cp_4stage']} lr={float(args.lr):g} finest-domain (TRE={diag_tre:.4f} mm)",
    )

    ax.set_xlabel("epoch")
    ax.set_ylabel("full 1mm-domain objective")
    ax.set_title(f"Case{int(case_id):02d} Registration Loss: Spline Methods")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    case_ids = [int(token) for token in str(args.cases).replace(" ", "").split(",") if token]
    outputs = [plot_case(case_id, args) for case_id in case_ids]
    print(json.dumps({"plots": [str(path) for path in outputs]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
