#!/usr/bin/env python
"""Sampled SINR-vs-DirectCP follow-up for the update-alignment run.

This first follow-up implementation reuses deterministic checkpoint samples
from the overnight update-alignment run. It measures whether the observed SINR
updates are close to a scalar-rescaled direct control-point update on the same
sampled dense domain. It does not form the autograd CP kernel J J^T.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


THIS_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "docs").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root(THIS_DIR)
DEFAULT_SOURCE_UPDATE_ROOT = Path("/database/inrorspline/runs/update_alignment_overnight_20260629_v1")
DEFAULT_OUT_ROOT = Path("/database/inrorspline/runs/sinr_cp_kernel_followup_20260629_v1")
DEFAULT_REVIEW_ROOT = REPO_ROOT / "experiments/2026-06-29_sinr_cp_kernel_followup"


CASE_SPECS = [
    {
        "dataset": "oasis",
        "case_key": "oasis_rank009",
        "sinr_method": "sinr_cps4_be_bspline_analytic",
        "directcp_method": "direct_cp_cps4_be_bspline_analytic",
    },
    {
        "dataset": "lungct",
        "case_key": "lungct_case01",
        "sinr_method": "sinr_cps4_ptv_lcc_isotv",
        "directcp_method": "direct_cp_cps4_ptv_lcc_isotv",
    },
    {
        "dataset": "lungct",
        "case_key": "lungct_case08",
        "sinr_method": "sinr_cps4_ptv_lcc_isotv",
        "directcp_method": "direct_cp_cps4_ptv_lcc_isotv",
    },
]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str) + "\n")


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fieldnames and not isinstance(value, (list, dict)):
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def phase_name(args: argparse.Namespace) -> str:
    return "smoke" if args.smoke else "full"


def source_phase_root(args: argparse.Namespace) -> Path:
    return Path(args.source_update_root) / phase_name(args)


def run_phase_root(args: argparse.Namespace) -> Path:
    return Path(args.out_root) / phase_name(args)


def review_phase_root(args: argparse.Namespace) -> Path:
    if args.smoke:
        return Path(args.review_root) / "smoke"
    return Path(args.review_root)


def vector_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    av = a.reshape(-1).astype(np.float64, copy=False)
    bv = b.reshape(-1).astype(np.float64, copy=False)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= eps:
        return float("nan")
    return float(np.dot(av, bv) / (denom + eps))


def norm_flat(a: np.ndarray) -> float:
    return float(np.linalg.norm(a.reshape(-1).astype(np.float64, copy=False)))


def rmse(a: np.ndarray) -> float:
    return float(math.sqrt(np.mean(np.square(a.astype(np.float64, copy=False)))))


def fit_scale(source: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    sv = source.reshape(-1).astype(np.float64, copy=False)
    tv = target.reshape(-1).astype(np.float64, copy=False)
    denom = float(np.dot(sv, sv))
    if denom <= eps:
        return float("nan")
    return float(np.dot(sv, tv) / denom)


def epoch_index(epochs: np.ndarray, epoch: int) -> Optional[int]:
    hits = np.where(epochs.astype(np.int64) == int(epoch))[0]
    if len(hits) == 0:
        return None
    return int(hits[0])


def choose_checkpoint_targets(epochs: np.ndarray, *, smoke: bool) -> List[Tuple[str, int]]:
    available = set(int(v) for v in epochs.astype(np.int64).tolist())
    if smoke:
        return [(f"epoch{epoch}", epoch) for epoch in (1, 2) if epoch in available]
    targets: List[Tuple[str, int]] = []
    for label, epoch in [("init_step", 1), ("epoch50", 50), ("epoch250", 250)]:
        if epoch in available:
            targets.append((label, epoch))
    return targets


def load_checkpoint_samples(run_dir: Path) -> Dict[str, np.ndarray]:
    path = run_dir / "checkpoint_samples.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint sample file: {path}")
    data = np.load(path)
    return {
        "epochs": data["epochs"].astype(np.int64),
        "flat_indices": data["flat_indices"].astype(np.int64),
        "disp": data["disp_vox_xyz"].astype(np.float32),
    }


def job_lookup(manifest: Dict[str, object]) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list):
        raise RuntimeError("job_manifest.json does not contain a jobs list.")
    lookup: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in jobs:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("dataset")), str(row.get("case_key")), str(row.get("method")))
        lookup[key] = row
    return lookup


def compare_case(
    spec: Dict[str, str],
    lookup: Dict[Tuple[str, str, str], Dict[str, object]],
    *,
    smoke: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    dataset = spec["dataset"]
    case_key = spec["case_key"]
    sinr_key = (dataset, case_key, spec["sinr_method"])
    cp_key = (dataset, case_key, spec["directcp_method"])
    if sinr_key not in lookup:
        raise RuntimeError(f"Missing SINR job in manifest: {sinr_key}")
    if cp_key not in lookup:
        raise RuntimeError(f"Missing DirectCP job in manifest: {cp_key}")
    sinr_job = lookup[sinr_key]
    cp_job = lookup[cp_key]
    sinr_samples = load_checkpoint_samples(Path(str(sinr_job["run_dir"])))
    cp_samples = load_checkpoint_samples(Path(str(cp_job["run_dir"])))
    if not np.array_equal(sinr_samples["flat_indices"], cp_samples["flat_indices"]):
        raise RuntimeError(f"Sample mismatch for {dataset} {case_key}")
    if not np.array_equal(sinr_samples["epochs"], cp_samples["epochs"]):
        raise RuntimeError(f"Epoch mismatch for {dataset} {case_key}")
    epochs = sinr_samples["epochs"]
    sinr_disp = sinr_samples["disp"]
    cp_disp = cp_samples["disp"]
    idx0 = epoch_index(epochs, 0)
    if idx0 is None:
        raise RuntimeError(f"Missing epoch 0 samples for {dataset} {case_key}")

    metric_rows: List[Dict[str, object]] = []
    spectrum_rows: List[Dict[str, object]] = []
    for checkpoint_label, target_epoch in choose_checkpoint_targets(epochs, smoke=smoke):
        idx_t = epoch_index(epochs, target_epoch)
        if idx_t is None:
            continue
        sinr_update = sinr_disp[idx_t] - sinr_disp[idx0]
        cp_update = cp_disp[idx_t] - cp_disp[idx0]
        alpha = fit_scale(cp_update, sinr_update)
        scaled_residual = sinr_update - alpha * cp_update if np.isfinite(alpha) else np.full_like(sinr_update, np.nan)
        sinr_norm = norm_flat(sinr_update)
        cp_norm = norm_flat(cp_update)
        residual_norm = norm_flat(scaled_residual)
        metric_rows.append(
            {
                "dataset": dataset,
                "case_key": case_key,
                "checkpoint": checkpoint_label,
                "from_epoch": 0,
                "to_epoch": int(target_epoch),
                "sample_count": int(sinr_update.shape[0]),
                "D_U_proxy_cos_sinr_directcp": vector_cosine(sinr_update, cp_update),
                "R_U_proxy_scaled_residual": residual_norm / sinr_norm if sinr_norm > 0 else float("nan"),
                "alpha_directcp_to_sinr": alpha,
                "A_U_proxy_norm_sinr_over_directcp": sinr_norm / cp_norm if cp_norm > 0 else float("nan"),
                "sinr_update_norm": sinr_norm,
                "directcp_update_norm": cp_norm,
                "scaled_residual_norm": residual_norm,
                "rmse_sinr_update_vox": rmse(sinr_update),
                "rmse_directcp_update_vox": rmse(cp_update),
            }
        )
        component_norms_sinr = np.linalg.norm(sinr_update.astype(np.float64), axis=1)
        component_norms_cp = np.linalg.norm(cp_update.astype(np.float64), axis=1)
        ratio = component_norms_sinr / np.maximum(component_norms_cp, 1e-12)
        spectrum_rows.append(
            {
                "dataset": dataset,
                "case_key": case_key,
                "checkpoint": checkpoint_label,
                "diagnostic": "sampled_update_norm_ratio_proxy",
                "p05": float(np.percentile(ratio, 5)),
                "p50": float(np.percentile(ratio, 50)),
                "p95": float(np.percentile(ratio, 95)),
                "p99": float(np.percentile(ratio, 99)),
                "max": float(np.max(ratio)),
                "note": "Proxy only; autograd CP-kernel spectrum was not formed.",
            }
        )
    return metric_rows, spectrum_rows


def validate_rows(rows: Sequence[Dict[str, object]], *, smoke: bool) -> List[str]:
    errors: List[str] = []
    expected_checkpoints = {"epoch1", "epoch2"} if smoke else {"init_step", "epoch50", "epoch250"}
    for spec in CASE_SPECS:
        case_rows = [
            row
            for row in rows
            if row.get("dataset") == spec["dataset"] and row.get("case_key") == spec["case_key"]
        ]
        seen = {str(row.get("checkpoint")) for row in case_rows}
        missing = sorted(expected_checkpoints.difference(seen))
        if missing:
            errors.append(f"{spec['dataset']} {spec['case_key']} missing checkpoints: {missing}")
    for row in rows:
        cosine = float(row["D_U_proxy_cos_sinr_directcp"])
        residual = float(row["R_U_proxy_scaled_residual"])
        amp = float(row["A_U_proxy_norm_sinr_over_directcp"])
        if not np.isfinite(cosine) or cosine < -1.000001 or cosine > 1.000001:
            errors.append(f"Invalid D_U proxy for {row.get('case_key')} {row.get('checkpoint')}: {cosine}")
        if not np.isfinite(residual) or residual < 0:
            errors.append(f"Invalid R_U proxy for {row.get('case_key')} {row.get('checkpoint')}: {residual}")
        if not np.isfinite(amp) or amp < 0:
            errors.append(f"Invalid A_U proxy for {row.get('case_key')} {row.get('checkpoint')}: {amp}")
    return errors


def make_plots(plots_dir: Path, rows: Sequence[Dict[str, object]], spectrum_rows: Sequence[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(plots_dir / "plot_error.json", {"error": str(exc)})
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (str(row["dataset"]), str(row["case_key"]), int(row["to_epoch"])))
    labels = [f"{row['case_key']}\n{row['checkpoint']}" for row in ordered]
    cosines = [float(row["D_U_proxy_cos_sinr_directcp"]) for row in ordered]
    amps = [float(row["A_U_proxy_norm_sinr_over_directcp"]) for row in ordered]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].bar(range(len(cosines)), cosines)
    axes[0].set_ylabel("D_U proxy cosine")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(range(len(amps)), amps)
    axes[1].set_ylabel("A_U proxy")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "update_agreement_and_amplification.png", dpi=180)
    plt.close(fig)

    if spectrum_rows:
        spec_ordered = sorted(spectrum_rows, key=lambda row: (str(row["dataset"]), str(row["case_key"]), str(row["checkpoint"])))
        labels = [f"{row['case_key']}\n{row['checkpoint']}" for row in spec_ordered]
        p50 = [float(row["p50"]) for row in spec_ordered]
        p95 = [float(row["p95"]) for row in spec_ordered]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(labels)), p50, marker="o", label="p50")
        ax.plot(range(len(labels)), p95, marker="o", label="p95")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("sampled SINR/DirectCP norm ratio")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "kernel_spectrum_optional.png", dpi=180)
        plt.close(fig)


def write_report(review_root: Path, summary: Dict[str, object], rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# SINR CP-Kernel Follow-Up Report",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This run is a sampled dense-update proxy for the CP-kernel diagnostic. It reuses the deterministic checkpoint samples from the update-alignment run and compares SINR cumulative updates against the paired single-stage DirectCP cumulative updates.",
        "",
        "## Metrics",
        "",
        "| Dataset | Case | Checkpoint | D_U proxy | R_U proxy | alpha | A_U proxy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (str(item["dataset"]), str(item["case_key"]), int(item["to_epoch"]))):
        lines.append(
            "| {dataset} | {case_key} | {checkpoint} | {du:.4f} | {ru:.4f} | {alpha:.4g} | {au:.4f} |".format(
                dataset=row["dataset"],
                case_key=row["case_key"],
                checkpoint=row["checkpoint"],
                du=float(row["D_U_proxy_cos_sinr_directcp"]),
                ru=float(row["R_U_proxy_scaled_residual"]),
                alpha=float(row["alpha_directcp_to_sinr"]),
                au=float(row["A_U_proxy_norm_sinr_over_directcp"]),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `D_U proxy` is cosine similarity between sampled SINR and DirectCP cumulative dense updates.",
            "- `R_U proxy` is the relative residual after fitting one scalar from DirectCP to SINR.",
            "- `A_U proxy` is the sampled dense-update norm ratio, SINR over DirectCP.",
            "- This does not estimate `K = J_theta J_theta^T`; it is the overnight-safe continuation artifact that identifies whether the full autograd kernel follow-up is worth running.",
        ]
    )
    errors = summary.get("errors", [])
    if errors:
        lines.extend(["", "## Errors", ""])
        for error in errors:
            lines.append(f"- {error}")
    review_root.mkdir(parents=True, exist_ok=True)
    (review_root / "sinr_cp_kernel_report.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> Dict[str, object]:
    source_root = source_phase_root(args)
    manifest_path = source_root / "job_manifest.json"
    review_root = review_phase_root(args)
    run_root = run_phase_root(args)
    tables_dir = review_root / "tables"
    plots_dir = review_root / "plots"
    if args.dry_run:
        plan = {
            "status": "dry_run",
            "source_manifest": str(manifest_path),
            "source_manifest_exists": manifest_path.exists(),
            "out_root": str(run_root),
            "review_root": str(review_root),
            "cases": CASE_SPECS,
            "phase": phase_name(args),
        }
        write_json(run_root / "dry_run_plan.json", plan)
        return plan
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")

    manifest_obj = read_json(manifest_path)
    if not isinstance(manifest_obj, dict):
        raise RuntimeError(f"Manifest is not a dict: {manifest_path}")
    lookup = job_lookup(manifest_obj)
    metric_rows: List[Dict[str, object]] = []
    spectrum_rows: List[Dict[str, object]] = []
    errors: List[str] = []
    for spec in CASE_SPECS:
        try:
            rows, spec_rows = compare_case(spec, lookup, smoke=bool(args.smoke))
            metric_rows.extend(rows)
            spectrum_rows.extend(spec_rows)
        except Exception as exc:
            errors.append(f"{spec['dataset']} {spec['case_key']}: {exc}")
            if not args.allow_partial:
                break
    if not errors or args.allow_partial:
        errors.extend(validate_rows(metric_rows, smoke=bool(args.smoke)))
    status = "completed" if not errors else "incomplete"
    run_root.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": status,
        "errors": errors,
        "phase": phase_name(args),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_root": str(source_root),
        "out_root": str(run_root),
        "review_root": str(review_root),
        "metric_rows": len(metric_rows),
        "spectrum_rows": len(spectrum_rows),
        "implementation_note": "Sampled dense-update proxy; autograd CP-kernel J J^T was not formed.",
    }
    write_csv(tables_dir / "kernel_metrics.csv", metric_rows)
    write_csv(tables_dir / "sampled_spectrum_metrics.csv", spectrum_rows)
    write_json(review_root / "summary.json", summary)
    write_json(run_root / "summary.json", summary)
    write_report(review_root, summary, metric_rows)
    make_plots(plots_dir, metric_rows, spectrum_rows)
    if errors and not args.allow_partial:
        raise RuntimeError(f"Follow-up validation failed: {errors}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("first3",), default="first3")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--source-update-root", type=Path, default=DEFAULT_SOURCE_UPDATE_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.dry_run or args.smoke or args.run_full):
        raise SystemExit("Choose one of --dry-run, --smoke, or --run-full.")
    if args.run_full:
        args.smoke = False
    result = run(args)
    print(json.dumps({"followup": result}, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
