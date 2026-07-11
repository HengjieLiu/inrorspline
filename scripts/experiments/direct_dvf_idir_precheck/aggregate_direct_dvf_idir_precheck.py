#!/usr/bin/env python
"""Aggregate the direct dense-DVF vs IDIR precheck."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REVIEW_ROOT = Path("experiments/2026-06-30_direct_dvf_idir_precheck")
OASIS_DIRECT_ROOT = Path("/database/inrorspline/runs/oasis_rank009_direct_dvf_idir_loss_precheck_v1")
LUNG_DIRECT_ROOT = Path("/database/inrorspline/runs/lungct_case01_case08_direct_dvf_idir_loss_precheck_v1")


def run_roots() -> List[Path]:
    roots: List[Path] = []
    env_root = os.environ.get("INROSPLINE_RUNS_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.extend(
        [
            Path("/database/inrorspline/runs"),
            Path("/data/hengjie/inrorspline/runs"),
        ]
    )
    deduped: List[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def existing_path(path: Path) -> Path:
    if path.exists():
        return path
    text = str(path)
    if text.startswith("/database/"):
        host_path = Path("/data/hengjie") / Path(text).relative_to("/database")
        if host_path.exists():
            return host_path
    return path


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_nested(data: Dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def last_history_value(summary: Dict[str, Any], key: str) -> Any:
    history = summary.get("history_tail")
    if not isinstance(history, list) or not history:
        return None
    last = history[-1]
    if not isinstance(last, dict):
        return None
    return last.get(key)


def find_one(candidates: Iterable[Path], label: str) -> Path:
    found = [path for path in candidates if path.exists()]
    if len(found) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(found)}: {[str(p) for p in found]}")
    return found[0]


def find_oasis_direct() -> Path:
    root = existing_path(OASIS_DIRECT_ROOT)
    return find_one(
        root.glob("direct_dvf_sampled_ncc_fd_be/*/summary.json"),
        "OASIS direct-DVF summary",
    )


def find_oasis_idir_anchor() -> Path:
    candidates: List[Path] = []
    for root in run_roots():
        candidates.extend(
            root.glob(
                "oasis_seed3_first10_alpha_sweep_be_2500e_v1/"
                "idir_original_dense_siren_be_autograd/"
                "idir_original_dense_siren_be_autograd_rank009_a1_e2500_lr0p0001/"
                "summary.json"
            )
        )
    return find_one(candidates, "OASIS IDIR rank9 alpha1 anchor")


def find_lung_direct(case_id: int) -> Path:
    root = existing_path(LUNG_DIRECT_ROOT)
    pattern = (
        "direct_dvf_lungmask_sampled_ncc_fd_be/"
        f"direct_dvf_lungmask_sampled_ncc_fd_be_case{case_id:02d}_*_seed1/"
        "summary.json"
    )
    matches = []
    for path in root.glob(pattern):
        summary = read_json(path)
        if int(summary.get("case_id", -1)) == int(case_id) and int(summary.get("training", {}).get("epochs", -1)) == 2500:
            matches.append(path)
    return find_one(matches, f"LungCT case{case_id:02d} direct-DVF summary")


def find_lung_idir_anchor(case_id: int) -> Path:
    candidates: List[Path] = []
    for root in run_roots():
        for base in [
            "lungct_ptvprep_no_sweep_repro_v1",
            "lungct_ptvprep_reg_sweep_idir_directcp32_v1",
        ]:
            candidates.extend(
                root.glob(
                    f"{base}/idir_be_default_lungmask/"
                    f"idir_be_default_lungmask_case{case_id:02d}_*_bend10_seed1/"
                    "summary.json"
                )
            )
    matches: List[Path] = []
    for path in candidates:
        summary = read_json(path)
        training = summary.get("training", {})
        if (
            int(summary.get("case_id", -1)) == int(case_id)
            and summary.get("status") == "completed"
            and as_float(training.get("alpha_bending")) == 10.0
            and as_float(training.get("lr")) == 1e-4
        ):
            matches.append(path)
    if not matches:
        raise RuntimeError(f"No LungCT case{case_id:02d} IDIR anchor found")
    return sorted(matches, key=lambda p: ("no_sweep" not in str(p), str(p)))[0]


def oasis_row(summary_path: Path, role: str) -> Dict[str, Any]:
    s = read_json(summary_path)
    return {
        "dataset": "oasis",
        "case": f"rank{int(s.get('case_rank', -1)):03d}",
        "role": role,
        "method": s.get("method_key", s.get("method")),
        "summary_path": str(summary_path),
        "alpha": s.get("alpha"),
        "lr": s.get("lr"),
        "epochs": s.get("epochs"),
        "sample_domain": s.get("sample_domain"),
        "regularizer": s.get("regularizer"),
        "native_analytic_be_formula": s.get("native_analytic_be_formula"),
        "dice_mean": s.get("dice_mean"),
        "hd95_mean": s.get("hd95_mean"),
        "asd_symmetric_mean": s.get("asd_symmetric_mean"),
        "tre_snap_mean_mm": None,
        "tre_subvoxel_mean_mm": None,
        "be_fd_forward_fixed_fg_voxel": s.get("be_fd_forward_fixed_fg_voxel"),
        "be_dense_or_fg": s.get("be_fd_forward_fixed_fg_voxel"),
        "diffusion_fd_forward_fixed_fg_voxel": s.get("diffusion_fd_forward_fixed_fg_voxel"),
        "diffusion_dense_or_fg": s.get("diffusion_fd_forward_fixed_fg_voxel"),
        "log_jdet_std_fg": s.get("log_jdet_std_fg"),
        "jdet_le0_pct_fg": s.get("jdet_le0_pct_fg"),
        "ndv_frac_fg": s.get("ndv_frac_fg"),
        "runtime_sec": s.get("training_elapsed_sec"),
        "peak_mem_gb": s.get("training_peak_mem_gb"),
        "final_loss": s.get("final_loss"),
        "final_data_loss": s.get("final_data_loss"),
        "final_reg_loss": s.get("final_reg_loss"),
    }


def lung_row(summary_path: Path, role: str) -> Dict[str, Any]:
    s = read_json(summary_path)
    reg = s.get("regularity_fg_registration_crop", {})
    training = s.get("training", {})
    return {
        "dataset": "lungct",
        "case": f"case{int(s.get('case_id', -1)):02d}",
        "role": role,
        "method": s.get("method"),
        "summary_path": str(summary_path),
        "alpha": training.get("alpha_bending"),
        "lr": training.get("lr"),
        "epochs": training.get("epochs"),
        "sample_domain": training.get("sample_domain", s.get("loss_domain")),
        "regularizer": training.get("regularizer"),
        "native_analytic_be_formula": training.get("native_analytic_be_formula"),
        "dice_mean": None,
        "hd95_mean": None,
        "asd_symmetric_mean": None,
        "tre_snap_mean_mm": get_nested(s, "tre_snap_to_voxel", "tre_mean_mm"),
        "tre_subvoxel_mean_mm": get_nested(s, "tre_subvoxel", "tre_mean_mm"),
        "be_fd_forward_fixed_fg_voxel": None,
        "be_dense_or_fg": reg.get("bending_energy_fg"),
        "diffusion_fd_forward_fixed_fg_voxel": None,
        "diffusion_dense_or_fg": reg.get("diffusion_l2_fg"),
        "log_jdet_std_fg": reg.get("log_jdet_std_fg"),
        "jdet_le0_pct_fg": reg.get("jdet_le0_pct_fg"),
        "ndv_frac_fg": reg.get("ndv_frac_fg"),
        "runtime_sec": s.get("elapsed_sec"),
        "peak_mem_gb": s.get("peak_mem_gb"),
        "final_loss": last_history_value(s, "loss"),
        "final_data_loss": last_history_value(s, "data_loss"),
        "final_reg_loss": last_history_value(s, "reg_loss"),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}g}"


def report_table(rows: List[Dict[str, Any]], dataset: str) -> List[str]:
    subset = [row for row in rows if row["dataset"] == dataset]
    if dataset == "oasis":
        header = "| Case | Role | Dice | HD95 | ASD | BE fd fg vox | Diff fd fg vox | logJ std | fold % | NDV | sec | GB |"
        sep = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        body = [
            "| {case} | {role} | {dice} | {hd95} | {asd} | {be} | {diff} | {logj} | {fold} | {ndv} | {sec} | {gb} |".format(
                case=row["case"],
                role=row["role"],
                dice=fmt(row["dice_mean"]),
                hd95=fmt(row["hd95_mean"]),
                asd=fmt(row["asd_symmetric_mean"]),
                be=fmt(row["be_dense_or_fg"]),
                diff=fmt(row["diffusion_dense_or_fg"]),
                logj=fmt(row["log_jdet_std_fg"]),
                fold=fmt(row["jdet_le0_pct_fg"]),
                ndv=fmt(row["ndv_frac_fg"]),
                sec=fmt(row["runtime_sec"]),
                gb=fmt(row["peak_mem_gb"]),
            )
            for row in subset
        ]
    else:
        header = "| Case | Role | snap TRE | subvox TRE | BE fg | Diff fg | logJ std | fold % | NDV | sec | GB |"
        sep = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        body = [
            "| {case} | {role} | {snap} | {sub} | {be} | {diff} | {logj} | {fold} | {ndv} | {sec} | {gb} |".format(
                case=row["case"],
                role=row["role"],
                snap=fmt(row["tre_snap_mean_mm"]),
                sub=fmt(row["tre_subvoxel_mean_mm"]),
                be=fmt(row["be_dense_or_fg"]),
                diff=fmt(row["diffusion_dense_or_fg"]),
                logj=fmt(row["log_jdet_std_fg"]),
                fold=fmt(row["jdet_le0_pct_fg"]),
                ndv=fmt(row["ndv_frac_fg"]),
                sec=fmt(row["runtime_sec"]),
                gb=fmt(row["peak_mem_gb"]),
            )
            for row in subset
        ]
    return [header, sep, *body]


def main() -> None:
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "oasis_direct": find_oasis_direct(),
        "oasis_idir": find_oasis_idir_anchor(),
        "lung01_direct": find_lung_direct(1),
        "lung01_idir": find_lung_idir_anchor(1),
        "lung08_direct": find_lung_direct(8),
        "lung08_idir": find_lung_idir_anchor(8),
    }
    rows = [
        oasis_row(paths["oasis_idir"], "IDIR anchor"),
        oasis_row(paths["oasis_direct"], "direct dense DVF"),
        lung_row(paths["lung01_idir"], "IDIR anchor"),
        lung_row(paths["lung01_direct"], "direct dense DVF"),
        lung_row(paths["lung08_idir"], "IDIR anchor"),
        lung_row(paths["lung08_direct"], "direct dense DVF"),
    ]
    write_csv(REVIEW_ROOT / "direct_dvf_idir_precheck_rows.csv", rows)
    (REVIEW_ROOT / "direct_dvf_idir_precheck_rows.json").write_text(
        json.dumps({"paths": {k: str(v) for k, v in paths.items()}, "rows": rows}, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    report = [
        "# Direct Dense-DVF vs IDIR Precheck",
        "",
        "Direct dense-DVF uses IDIR-style sampled NCC at 10,000 fixed-domain grid points.",
        "Bending energy is dense finite difference; native continuous analytic BE is unavailable for dense-grid DVF.",
        "",
        "## OASIS Rank9",
        "",
        *report_table(rows, "oasis"),
        "",
        "## LungCT pTV-Prep Diagnostic Domain",
        "",
        *report_table(rows, "lungct"),
        "",
        "## Source Summaries",
        "",
    ]
    report.extend(f"- `{name}`: `{path}`" for name, path in paths.items())
    (REVIEW_ROOT / "review_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"review_root": str(REVIEW_ROOT), "rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
