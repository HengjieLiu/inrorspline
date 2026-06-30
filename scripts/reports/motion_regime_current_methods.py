#!/usr/bin/env python
"""Current-run OASIS/LungCT deformation-regime characterization.

This report uses already optimized dense displacement fields. It does not train
or modify experiment outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.lungct_no_sweep_repro import common as lung_c  # noqa: E402
from scripts.experiments.oasis_alpha_sweep import common as oasis_c  # noqa: E402


OUT_ROOT = REPO_ROOT / "experiments/2026-06-25_motion_regime_seed3_first10_current_methods"
OASIS_SUMMARY_CSV = REPO_ROOT / "experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/summary_runs.csv"
LUNG_SUMMARY_CSV = REPO_ROOT / "experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/summary_runs.csv"

METHODS = [
    {
        "dataset": "OASIS",
        "role": "primary",
        "method_key": "sinr_cps4_be_bspline_analytic",
        "method_label": "OASIS SINR cps4 alpha1000",
        "alpha": 1000.0,
        "summary_csv": OASIS_SUMMARY_CSV,
    },
    {
        "dataset": "OASIS",
        "role": "sensitivity",
        "method_key": "direct_cp_cps4_be_bspline_analytic",
        "method_label": "OASIS DirectCP cps4 alpha1000",
        "alpha": 1000.0,
        "summary_csv": OASIS_SUMMARY_CSV,
    },
    {
        "dataset": "4DCT",
        "role": "primary",
        "method_key": "idir_be_default_lungmask",
        "method_label": "4DCT IDIR default",
        "summary_csv": LUNG_SUMMARY_CSV,
    },
    {
        "dataset": "4DCT",
        "role": "sensitivity",
        "method_key": "direct_cp_32_16_8_4_ptv_lcc_isotv",
        "method_label": "4DCT DirectCP 32-16-8-4",
        "summary_csv": LUNG_SUMMARY_CSV,
    },
]

LAGS_MM = (2, 4, 8, 16, 32, 64)
LOWPASS_SIGMAS_MM = (2, 4, 8, 16, 32)
COARSE_SPACINGS_MM = (64, 32, 16, 8, 4, 2)
RNG_SEED = 12345


@dataclass
class CaseField:
    dataset: str
    role: str
    method_key: str
    method_label: str
    case_index: int
    case_label: str
    fixed_id: str
    moving_id: str
    run_dir: Path
    summary_path: Path
    disp_xyz_mm: np.ndarray
    mask: np.ndarray
    meta: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--oasis-summary-csv", type=Path, default=OASIS_SUMMARY_CSV)
    parser.add_argument("--lung-summary-csv", type=Path, default=LUNG_SUMMARY_CSV)
    parser.add_argument("--oasis-root", type=Path, default=oasis_c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--lung-ptvprep-root", type=Path, default=lung_c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--case-count", type=int, default=10)
    parser.add_argument("--max-pairs-per-lag-axis", type=int, default=300_000)
    parser.add_argument("--pca-sample", type=int, default=150_000)
    parser.add_argument("--hash-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-sanity-tests", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fieldnames and not isinstance(value, (list, tuple, dict)):
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True, default=str) + "\n")


def fmt(value: object, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return str(val)
    if abs(val) >= 100:
        return f"{val:.1f}"
    if abs(val) >= 10:
        return f"{val:.2f}"
    return f"{val:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return df.to_markdown(index=False)


def resolve_path(path_text: str) -> Path:
    candidates = [Path(path_text)]
    if path_text.startswith("/database/"):
        candidates.append(Path("/data/hengjie") / path_text[len("/database/") :])
    if path_text.startswith("/homebase/"):
        candidates.append(Path("/home/hengjie") / path_text[len("/homebase/") :])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def norm_disp_xyz_to_mm(disp_norm: np.ndarray, shape_zyx: Sequence[int]) -> np.ndarray:
    z, y, x = (int(v) for v in shape_zyx)
    scale = np.asarray([(x - 1) / 2.0, (y - 1) / 2.0, (z - 1) / 2.0], dtype=np.float32)
    return disp_norm.astype(np.float32, copy=False) * scale.reshape((1, 1, 1, 3))


def select_rows(summary_csv: Path, spec: Mapping[str, object], case_count: int) -> List[Dict[str, str]]:
    rows = read_csv(summary_csv)
    key = str(spec["method_key"])
    selected = [row for row in rows if row.get("method_key", row.get("method", "")) == key or row.get("method") == key]
    if "alpha" in spec:
        alpha = float(spec["alpha"])
        selected = [row for row in selected if abs(float(row["alpha"]) - alpha) < 1e-9]
    if str(spec["dataset"]) == "OASIS":
        selected = [row for row in selected if int(row["case_rank"]) < case_count]
        selected.sort(key=lambda row: int(row["case_rank"]))
    else:
        selected = [row for row in selected if int(row["case_id"]) <= case_count]
        selected.sort(key=lambda row: int(row["case_id"]))
    if len(selected) != case_count:
        raise RuntimeError(
            f"Expected {case_count} rows for {spec['method_label']}, found {len(selected)} in {summary_csv}"
        )
    return selected


def load_oasis_field(row: Mapping[str, str], spec: Mapping[str, object], oasis_root: Path) -> CaseField:
    run_dir = resolve_path(row.get("run_dir_actual") or row["run_dir"])
    summary_path = resolve_path(row["summary_path"])
    fixed_id = row["fixed_id"]
    moving_id = row["moving_id"]
    disp_path = run_dir / f"disp_norm_{fixed_id}_{moving_id}.npy"
    mask_audit_path = run_dir / "mask_audit.json"
    missing = [str(p) for p in (run_dir, summary_path, disp_path, mask_audit_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing OASIS input files:\n" + "\n".join(missing))

    audit = json.loads(mask_audit_path.read_text())
    if audit.get("foreground_rule") != "fixed raw image > 0":
        raise RuntimeError(f"Unexpected OASIS foreground rule in {mask_audit_path}: {audit.get('foreground_rule')}")

    batch = oasis_c.load_oasis_case(
        rank=int(row["case_rank"]),
        device=torch.device("cpu"),
        oasis_root=oasis_root,
        pair_file=oasis_c.DEFAULT_PAIR_FILE,
    )
    if str(batch["fixed_id"]) != fixed_id or str(batch["moving_id"]) != moving_id:
        raise RuntimeError(
            "OASIS summary row does not match seed-3 shuffled pair file at "
            f"rank {row['case_rank']}: summary {fixed_id}->{moving_id}, "
            f"pair file {batch['fixed_id']}->{batch['moving_id']}"
        )
    mask = np.asarray(batch["fixed_mask"].detach().cpu().numpy() > 0, dtype=bool)
    disp_norm = np.load(disp_path).astype(np.float32, copy=False)
    if tuple(disp_norm.shape[:3]) != tuple(oasis_c.IMAGE_SHAPE):
        raise RuntimeError(f"Unexpected OASIS displacement shape {disp_norm.shape} in {disp_path}")
    if tuple(mask.shape) != tuple(disp_norm.shape[:3]):
        raise RuntimeError(f"OASIS mask/field shape mismatch for {disp_path}: {mask.shape} vs {disp_norm.shape}")
    disp_mm = norm_disp_xyz_to_mm(disp_norm, disp_norm.shape[:3])
    return CaseField(
        dataset="OASIS",
        role=str(spec["role"]),
        method_key=str(spec["method_key"]),
        method_label=str(spec["method_label"]),
        case_index=int(row["case_rank"]),
        case_label=f"rank{int(row['case_rank']):03d} {fixed_id}->{moving_id}",
        fixed_id=fixed_id,
        moving_id=moving_id,
        run_dir=run_dir,
        summary_path=summary_path,
        disp_xyz_mm=disp_mm,
        mask=mask,
        meta={
            "summary_row": dict(row),
            "mask_audit": audit,
            "disp_path": str(disp_path),
            "mask_rule": "fixed raw image > 0",
            "pair_file": str(batch["pair_file"]),
            "sorted_pair_file": str(batch["sorted_pair_file"]),
        },
    )


def load_lung_field(row: Mapping[str, str], spec: Mapping[str, object], ptvprep_root: Path) -> CaseField:
    summary_path = resolve_path(row["summary_path"])
    run_dir = summary_path.parent
    disp_path = run_dir / "disp_norm_xyz_reg1mm.npy"
    missing = [str(p) for p in (summary_path, disp_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing LungCT input files:\n" + "\n".join(missing))

    summary = json.loads(summary_path.read_text())
    protocol = str(summary.get("protocol_tag", ""))
    mask_rule = str(summary.get("mask_rule", ""))
    if "fixed_t00_lung_mask" not in protocol.lower() and "fixed_t00_lung_mask" not in mask_rule.lower():
        raise RuntimeError(f"Unexpected LungCT mask convention in {summary_path}: {protocol} / {mask_rule}")

    case_id = int(row["case_id"])
    case = lung_c.load_case(ptvprep_root, case_id)
    mask = case.fixed_mask.astype(bool, copy=False)
    disp_norm = np.load(disp_path).astype(np.float32, copy=False)
    if tuple(mask.shape) != tuple(disp_norm.shape[:3]):
        raise RuntimeError(f"LungCT mask/field shape mismatch for {disp_path}: {mask.shape} vs {disp_norm.shape}")
    disp_mm = norm_disp_xyz_to_mm(disp_norm, disp_norm.shape[:3])
    return CaseField(
        dataset="4DCT",
        role=str(spec["role"]),
        method_key=str(spec["method_key"]),
        method_label=str(spec["method_label"]),
        case_index=case_id,
        case_label=f"case{case_id:02d}",
        fixed_id=f"case{case_id:02d}_T00",
        moving_id=f"case{case_id:02d}_T50",
        run_dir=run_dir,
        summary_path=summary_path,
        disp_xyz_mm=disp_mm,
        mask=mask,
        meta={
            "summary_row": dict(row),
            "summary": summary,
            "disp_path": str(disp_path),
            "mask_rule": "fixed T00 lung mask on cropped 1mm grid",
        },
    )


def masked_vectors(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = field[mask]
    if values.size == 0:
        raise RuntimeError("Empty foreground mask")
    return values.astype(np.float32, copy=False)


def finite_mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def gradient_components(field: np.ndarray) -> Dict[str, np.ndarray]:
    derivs: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(("x", "y", "z")):
        dz, dy, dx = np.gradient(field[..., idx].astype(np.float32, copy=False), edge_order=1)
        derivs[f"{name}_x"] = dx.astype(np.float32, copy=False)
        derivs[f"{name}_y"] = dy.astype(np.float32, copy=False)
        derivs[f"{name}_z"] = dz.astype(np.float32, copy=False)
    return derivs


def bending_energy_from_derivs(derivs: Mapping[str, np.ndarray], mask: np.ndarray) -> float:
    total = np.zeros(mask.shape, dtype=np.float32)
    for name in ("x", "y", "z"):
        gx = derivs[f"{name}_x"]
        gy = derivs[f"{name}_y"]
        gz = derivs[f"{name}_z"]
        gx_z, gx_y, gx_x = np.gradient(gx, edge_order=1)
        gy_z, gy_y, _ = np.gradient(gy, edge_order=1)
        gz_z, _, _ = np.gradient(gz, edge_order=1)
        total += (gx_x * gx_x + gy_y * gy_y + gz_z * gz_z + 2.0 * (gx_y * gx_y + gx_z * gx_z + gy_z * gy_z))
    return float(np.mean(total[mask]))


def smoothness_metrics(field: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    derivs = gradient_components(field)
    ux_x, ux_y, ux_z = derivs["x_x"], derivs["x_y"], derivs["x_z"]
    uy_x, uy_y, uy_z = derivs["y_x"], derivs["y_y"], derivs["y_z"]
    uz_x, uz_y, uz_z = derivs["z_x"], derivs["z_y"], derivs["z_z"]

    grad_sq = ux_x**2 + ux_y**2 + ux_z**2 + uy_x**2 + uy_y**2 + uy_z**2 + uz_x**2 + uz_y**2 + uz_z**2
    diffusion = float(np.mean(grad_sq[mask]))
    tv = float(np.mean(np.sqrt(grad_sq[mask])))

    e_xx, e_yy, e_zz = ux_x, uy_y, uz_z
    e_xy = 0.5 * (ux_y + uy_x)
    e_xz = 0.5 * (ux_z + uz_x)
    e_yz = 0.5 * (uy_z + uz_y)
    strain_sq = e_xx**2 + e_yy**2 + e_zz**2 + 2.0 * (e_xy**2 + e_xz**2 + e_yz**2)
    strain = np.sqrt(strain_sq[mask])

    divergence = ux_x + uy_y + uz_z
    curl_x = uz_y - uy_z
    curl_y = ux_z - uz_x
    curl_z = uy_x - ux_y
    curl_norm = np.sqrt((curl_x**2 + curl_y**2 + curl_z**2)[mask])

    a11 = 1.0 + ux_x
    a12 = ux_y
    a13 = ux_z
    a21 = uy_x
    a22 = 1.0 + uy_y
    a23 = uy_z
    a31 = uz_x
    a32 = uz_y
    a33 = 1.0 + uz_z
    jdet = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
    jdet_m = jdet[mask].astype(np.float64, copy=False)
    positive = np.clip(jdet_m, 1e-6, None)

    return {
        "diffusion_energy": diffusion,
        "bending_energy": bending_energy_from_derivs(derivs, mask),
        "displacement_tv": tv,
        "strain_norm_mean": float(np.mean(strain)),
        "strain_norm_p95": float(np.percentile(strain, 95)),
        "divergence_abs_mean": float(np.mean(np.abs(divergence[mask]))),
        "curl_norm_mean": float(np.mean(curl_norm)),
        "log_jdet_std": float(np.std(np.log(positive))),
        "jdet_le0_pct": float(np.mean(jdet_m <= 0.0) * 100.0),
        "jdet_min": float(np.min(jdet_m)),
        "jdet_p1": float(np.percentile(jdet_m, 1)),
        "jdet_p99": float(np.percentile(jdet_m, 99)),
    }


def amplitude_metrics(case: CaseField) -> Dict[str, object]:
    vectors = masked_vectors(case.disp_xyz_mm, case.mask).astype(np.float64, copy=False)
    mag = np.linalg.norm(vectors, axis=1)
    abs_comp = np.mean(np.abs(vectors), axis=0)
    abs_sum = float(abs_comp.sum())
    comp_frac = abs_comp / abs_sum if abs_sum > 0 else np.full(3, np.nan)
    signed_mean = np.mean(vectors, axis=0)
    signed_std = np.std(vectors, axis=0)
    row: Dict[str, object] = {
        "dataset": case.dataset,
        "role": case.role,
        "method_key": case.method_key,
        "method_label": case.method_label,
        "case_index": case.case_index,
        "case_label": case.case_label,
        "fixed_id": case.fixed_id,
        "moving_id": case.moving_id,
        "run_dir": str(case.run_dir),
        "summary_path": str(case.summary_path),
        "disp_path": str(case.meta["disp_path"]),
        "pair_file": str(case.meta.get("pair_file", "")),
        "mask_rule": str(case.meta["mask_rule"]),
        "field_shape_zyx": "x".join(str(v) for v in case.disp_xyz_mm.shape[:3]),
        "component_order": "xyz",
        "axis_order": "zyx",
        "spacing_mm": "1x1x1",
        "mask_voxels": int(case.mask.sum()),
        "mask_fraction": float(case.mask.mean()),
        "mag_mean_mm": float(np.mean(mag)),
        "mag_std_mm": float(np.std(mag)),
        "mag_median_mm": float(np.percentile(mag, 50)),
        "mag_p25_mm": float(np.percentile(mag, 25)),
        "mag_p75_mm": float(np.percentile(mag, 75)),
        "mag_p90_mm": float(np.percentile(mag, 90)),
        "mag_p95_mm": float(np.percentile(mag, 95)),
        "mag_p98_mm": float(np.percentile(mag, 98)),
        "mag_p99_mm": float(np.percentile(mag, 99)),
        "mag_max_mm": float(np.max(mag)),
        "pct_gt_5mm": float(np.mean(mag > 5.0) * 100.0),
        "pct_gt_10mm": float(np.mean(mag > 10.0) * 100.0),
        "ux_mean_mm": float(signed_mean[0]),
        "uy_mean_mm": float(signed_mean[1]),
        "uz_mean_mm": float(signed_mean[2]),
        "ux_std_mm": float(signed_std[0]),
        "uy_std_mm": float(signed_std[1]),
        "uz_std_mm": float(signed_std[2]),
        "abs_ux_mean_mm": float(abs_comp[0]),
        "abs_uy_mean_mm": float(abs_comp[1]),
        "abs_uz_mean_mm": float(abs_comp[2]),
        "abs_ux_fraction": float(comp_frac[0]),
        "abs_uy_fraction": float(comp_frac[1]),
        "abs_uz_fraction": float(comp_frac[2]),
    }
    row.update(smoothness_metrics(case.disp_xyz_mm, case.mask))
    return row


def pca_metrics(case: CaseField, sample_count: int, rng: np.random.Generator) -> Dict[str, float]:
    vectors = masked_vectors(case.disp_xyz_mm, case.mask).astype(np.float64, copy=False)
    if vectors.shape[0] > sample_count:
        vectors = vectors[rng.choice(vectors.shape[0], size=sample_count, replace=False)]
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    moment = (vectors.T @ vectors) / float(vectors.shape[0])
    cov = (centered.T @ centered) / float(max(vectors.shape[0] - 1, 1))
    moment_eig = np.linalg.eigvalsh(moment)[::-1]
    cov_eig = np.linalg.eigvalsh(cov)[::-1]
    moment_frac = moment_eig / moment_eig.sum() if moment_eig.sum() > 0 else np.full(3, np.nan)
    cov_frac = cov_eig / cov_eig.sum() if cov_eig.sum() > 0 else np.full(3, np.nan)
    return {
        "pca_sample_voxels": int(vectors.shape[0]),
        "moment_frac1": float(moment_frac[0]),
        "moment_frac2": float(moment_frac[1]),
        "moment_frac3": float(moment_frac[2]),
        "centered_frac1": float(cov_frac[0]),
        "centered_frac2": float(cov_frac[1]),
        "centered_frac3": float(cov_frac[2]),
        "moment_eig1_over_eig2": float(moment_eig[0] / moment_eig[1]) if moment_eig[1] > 0 else float("nan"),
        "centered_eig1_over_eig2": float(cov_eig[0] / cov_eig[1]) if cov_eig[1] > 0 else float("nan"),
    }


def shifted_slices(axis: int, lag: int) -> Tuple[Tuple[slice, slice, slice], Tuple[slice, slice, slice]]:
    src = [slice(None), slice(None), slice(None)]
    dst = [slice(None), slice(None), slice(None)]
    src[axis] = slice(0, -lag)
    dst[axis] = slice(lag, None)
    return tuple(src), tuple(dst)  # type: ignore[return-value]


def vector_autocorr(case: CaseField, lags: Sequence[int], max_pairs: int, rng: np.random.Generator) -> Tuple[List[Dict[str, object]], float, bool]:
    rows: List[Dict[str, object]] = []
    for lag in lags:
        dot_sum = e1_sum = e2_sum = diff_sum = 0.0
        pair_count = 0
        for axis in range(3):
            if lag >= case.mask.shape[axis]:
                continue
            src, dst = shifted_slices(axis, int(lag))
            valid = case.mask[src] & case.mask[dst]
            idx = np.flatnonzero(valid.reshape(-1))
            if idx.size == 0:
                continue
            if idx.size > max_pairs:
                idx = rng.choice(idx, size=max_pairs, replace=False)
            u1 = case.disp_xyz_mm[src].reshape(-1, 3)[idx].astype(np.float64, copy=False)
            u2 = case.disp_xyz_mm[dst].reshape(-1, 3)[idx].astype(np.float64, copy=False)
            diff = u2 - u1
            dot_sum += float(np.sum(u1 * u2))
            e1_sum += float(np.sum(u1 * u1))
            e2_sum += float(np.sum(u2 * u2))
            diff_sum += float(np.sum(diff * diff))
            pair_count += int(idx.size)
        corr = dot_sum / math.sqrt(e1_sum * e2_sum) if e1_sum > 0 and e2_sum > 0 else float("nan")
        semivar = 0.5 * diff_sum / pair_count if pair_count > 0 else float("nan")
        rows.append(
            {
                "dataset": case.dataset,
                "role": case.role,
                "method_key": case.method_key,
                "method_label": case.method_label,
                "case_index": case.case_index,
                "case_label": case.case_label,
                "lag_mm": int(lag),
                "autocorrelation": float(corr),
                "semivariogram_mm2": float(semivar),
                "sampled_pairs": int(pair_count),
            }
        )

    corr_length = float(lags[-1])
    censored = True
    prev_lag = 0.0
    prev_corr = 1.0
    for row in rows:
        corr = float(row["autocorrelation"])
        lag = float(row["lag_mm"])
        if math.isfinite(corr) and corr < 0.5:
            corr_length = lag if prev_corr == corr else prev_lag + (0.5 - prev_corr) * (lag - prev_lag) / (corr - prev_corr)
            censored = False
            break
        prev_lag, prev_corr = lag, corr
    return rows, corr_length, censored


def energy_denom(case: CaseField) -> float:
    values = masked_vectors(case.disp_xyz_mm, case.mask).astype(np.float64, copy=False)
    return float(np.sum(values * values))


def residual_stats(field: np.ndarray, recon: np.ndarray, mask: np.ndarray, denom: float) -> Tuple[float, float, float, float]:
    residual = (field - recon)[mask].astype(np.float64, copy=False)
    residual_energy = float(np.sum(residual * residual) / denom) if denom > 0 else float("nan")
    mag = np.linalg.norm(residual, axis=1)
    return residual_energy, 1.0 - residual_energy, float(np.mean(mag)), float(np.percentile(mag, 95))


def lowpass_curves(case: CaseField, sigmas: Sequence[int]) -> List[Dict[str, object]]:
    denom = energy_denom(case)
    rows: List[Dict[str, object]] = []
    for sigma in sigmas:
        low = np.empty_like(case.disp_xyz_mm, dtype=np.float32)
        for component in range(3):
            low[..., component] = gaussian_filter(case.disp_xyz_mm[..., component], sigma=float(sigma), mode="nearest")
        residual_energy, explained, residual_mean, residual_p95 = residual_stats(case.disp_xyz_mm, low, case.mask, denom)
        rows.append(
            {
                "dataset": case.dataset,
                "role": case.role,
                "method_key": case.method_key,
                "method_label": case.method_label,
                "case_index": case.case_index,
                "case_label": case.case_label,
                "sigma_mm": int(sigma),
                "residual_energy": residual_energy,
                "explained_energy": explained,
                "residual_mean_mm": residual_mean,
                "residual_p95_mm": residual_p95,
            }
        )
    return rows


def coarse_shape(shape_zyx: Sequence[int], spacing: int) -> Tuple[int, int, int]:
    return tuple(max(2, int(math.ceil((int(size) - 1) / float(spacing))) + 1) for size in shape_zyx)


def coarse_grid_curves(case: CaseField, spacings: Sequence[int]) -> List[Dict[str, object]]:
    denom = energy_denom(case)
    field_t = torch.from_numpy(case.disp_xyz_mm).permute(3, 0, 1, 2).unsqueeze(0).float()
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for spacing in spacings:
            shape = coarse_shape(case.disp_xyz_mm.shape[:3], int(spacing))
            coarse = F.interpolate(field_t, size=shape, mode="trilinear", align_corners=True)
            recon = F.interpolate(coarse, size=case.disp_xyz_mm.shape[:3], mode="trilinear", align_corners=True)
            recon_np = recon.squeeze(0).permute(1, 2, 3, 0).numpy().astype(np.float32, copy=False)
            residual_energy, explained, residual_mean, residual_p95 = residual_stats(case.disp_xyz_mm, recon_np, case.mask, denom)
            rows.append(
                {
                    "dataset": case.dataset,
                    "role": case.role,
                    "method_key": case.method_key,
                    "method_label": case.method_label,
                    "case_index": case.case_index,
                    "case_label": case.case_label,
                    "spacing_mm": int(spacing),
                    "coarse_shape_zyx": "x".join(str(v) for v in shape),
                    "approx_dof": int(np.prod(shape) * 3),
                    "residual_energy": residual_energy,
                    "explained_energy": explained,
                    "residual_mean_mm": residual_mean,
                    "residual_p95_mm": residual_p95,
                }
            )
    return rows


def load_cases(args: argparse.Namespace) -> Tuple[List[CaseField], List[Dict[str, object]]]:
    cases: List[CaseField] = []
    manifest: List[Dict[str, object]] = []
    specs = []
    for spec in METHODS:
        spec = dict(spec)
        if spec["dataset"] == "OASIS":
            spec["summary_csv"] = args.oasis_summary_csv
        else:
            spec["summary_csv"] = args.lung_summary_csv
        specs.append(spec)

    for spec in specs:
        rows = select_rows(Path(spec["summary_csv"]), spec, args.case_count)
        for row in rows:
            if spec["dataset"] == "OASIS":
                case = load_oasis_field(row, spec, args.oasis_root)
            else:
                case = load_lung_field(row, spec, args.lung_ptvprep_root)
            cases.append(case)
            input_paths = [case.summary_path, Path(str(case.meta["disp_path"]))]
            mask_path = case.run_dir / "mask_audit.json"
            if mask_path.exists():
                input_paths.append(mask_path)
            for path in input_paths:
                entry = {
                    "dataset": case.dataset,
                    "role": case.role,
                    "method_key": case.method_key,
                    "case_index": case.case_index,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
                if args.hash_inputs:
                    entry["sha256"] = sha256_file(path)
                manifest.append(entry)
    counts = pd.DataFrame([{"dataset": c.dataset, "role": c.role, "method_key": c.method_key} for c in cases])
    expected = {
        ("OASIS", "primary"): args.case_count,
        ("OASIS", "sensitivity"): args.case_count,
        ("4DCT", "primary"): args.case_count,
        ("4DCT", "sensitivity"): args.case_count,
    }
    for key, n_expected in expected.items():
        n_found = int(((counts["dataset"] == key[0]) & (counts["role"] == key[1])).sum())
        if n_found != n_expected:
            raise RuntimeError(f"Expected {n_expected} {key} fields, found {n_found}")
    return cases, manifest


def aggregate_case_metrics(case_rows: List[Mapping[str, object]]) -> List[Dict[str, object]]:
    metrics = [
        "mag_mean_mm",
        "mag_p95_mm",
        "mag_p99_mm",
        "mag_max_mm",
        "pct_gt_10mm",
        "diffusion_energy",
        "bending_energy",
        "displacement_tv",
        "strain_norm_mean",
        "divergence_abs_mean",
        "curl_norm_mean",
        "log_jdet_std",
        "jdet_le0_pct",
        "corr_length_mm",
        "moment_frac1",
        "centered_frac1",
    ]
    grouped: Dict[Tuple[str, str, str, str], List[Mapping[str, object]]] = {}
    for row in case_rows:
        key = (str(row["dataset"]), str(row["role"]), str(row["method_key"]), str(row["method_label"]))
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, object]] = []
    for (dataset, role, method_key, method_label), rows in grouped.items():
        agg: Dict[str, object] = {
            "dataset": dataset,
            "role": role,
            "method_key": method_key,
            "method_label": method_label,
            "n_cases": len(rows),
        }
        for metric in metrics:
            mean, std = finite_mean_std([float(row[metric]) for row in rows])
            agg[f"{metric}_mean"] = mean
            agg[f"{metric}_std"] = std
        out.append(agg)
    out.sort(key=lambda r: (str(r["role"]) != "primary", str(r["dataset"]), str(r["method_label"])))
    return out


def make_paper_table(aggregate_rows: List[Mapping[str, object]], coarse_rows: List[Mapping[str, object]]) -> pd.DataFrame:
    coarse = pd.DataFrame(coarse_rows)
    coarse_mean = (
        coarse.groupby(["dataset", "role", "method_key", "method_label", "spacing_mm"], as_index=False)
        .agg(explained_energy=("explained_energy", "mean"), residual_p95_mm=("residual_p95_mm", "mean"))
    )
    records: List[Dict[str, object]] = []
    for row in aggregate_rows:
        rec = {
            "Dataset": row["dataset"],
            "Role": row["role"],
            "Method": row["method_label"],
            "Mean disp mm": row["mag_mean_mm_mean"],
            "P95 disp mm": row["mag_p95_mm_mean"],
            "% >10mm": row["pct_gt_10mm_mean"],
            "Diffusion": row["diffusion_energy_mean"],
            "Bending": row["bending_energy_mean"],
            "logJ std": row["log_jdet_std_mean"],
            "J<=0 %": row["jdet_le0_pct_mean"],
            "Corr len mm": row["corr_length_mm_mean"],
        }
        sub = coarse_mean[
            (coarse_mean["dataset"] == row["dataset"])
            & (coarse_mean["role"] == row["role"])
            & (coarse_mean["method_key"] == row["method_key"])
        ]
        for spacing in (16, 32, 64):
            hit = sub[sub["spacing_mm"] == spacing]
            rec[f"Coarse E@{spacing}mm"] = float(hit["explained_energy"].iloc[0]) if not hit.empty else float("nan")
        records.append(rec)
    return pd.DataFrame(records)


def make_contrast_table(paper_df: pd.DataFrame) -> pd.DataFrame:
    numeric = [col for col in paper_df.columns if col not in ("Dataset", "Role", "Method")]
    records: List[Dict[str, object]] = []
    for role in ("primary", "sensitivity"):
        sub = paper_df[paper_df["Role"] == role]
        oasis = sub[sub["Dataset"] == "OASIS"]
        lung = sub[sub["Dataset"] == "4DCT"]
        if oasis.empty or lung.empty:
            continue
        row = {"Role": role, "Contrast": "4DCT / OASIS", "OASIS method": oasis["Method"].iloc[0], "4DCT method": lung["Method"].iloc[0]}
        for col in numeric:
            denom = float(oasis[col].iloc[0])
            row[col] = float(lung[col].iloc[0]) / denom if denom != 0 else float("nan")
        records.append(row)
    return pd.DataFrame(records)


def plot_box_metric(df: pd.DataFrame, metrics: Sequence[Tuple[str, str]], path: Path, title: str, primary_only: bool = True) -> None:
    plot_df = df[df["role"] == "primary"] if primary_only else df
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for ax, (metric, label) in zip(axes[0], metrics):
        labels = []
        data = []
        for method, sub in plot_df.groupby("method_label", sort=False):
            labels.append(method.replace(" ", "\n"))
            data.append(sub[metric].astype(float).to_numpy())
        try:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)
        for i, vals in enumerate(data, 1):
            ax.scatter(np.full(len(vals), i), vals, s=24, alpha=0.75)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_curves(df: pd.DataFrame, x: str, y: str, path: Path, title: str, xlabel: str, ylabel: str, primary_only: bool = True) -> None:
    plot_df = df[df["role"] == "primary"] if primary_only else df
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, sub in plot_df.groupby("method_label", sort=False):
        stats = sub.groupby(x)[y].agg(["mean", "std"]).reset_index()
        ax.plot(stats[x], stats["mean"], marker="o", label=method)
        ax.fill_between(stats[x], stats["mean"] - stats["std"], stats["mean"] + stats["std"], alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_representative_slices(cases: List[CaseField], case_rows: pd.DataFrame, path: Path) -> None:
    primary = case_rows[case_rows["role"] == "primary"]
    selected: List[Tuple[str, CaseField]] = []
    for method, sub in primary.groupby("method_label", sort=False):
        ordered = sub.sort_values("mag_p95_mm")
        for label, idx in (("median P95", len(ordered) // 2), ("high P95", len(ordered) - 1)):
            row = ordered.iloc[idx]
            match = next(
                c
                for c in cases
                if c.method_label == method and c.case_index == int(row["case_index"]) and c.role == "primary"
            )
            selected.append((f"{method}\n{label}\n{match.case_label}", match))
    vmax = 0.0
    mags: List[np.ndarray] = []
    slices: List[int] = []
    for _, case in selected:
        mag = np.linalg.norm(case.disp_xyz_mm, axis=3)
        mags.append(mag)
        areas = case.mask.sum(axis=(1, 2))
        slices.append(int(np.argmax(areas)))
        vmax = max(vmax, float(np.percentile(mag[case.mask], 99)))
    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4), squeeze=False)
    for ax, (title, case), mag, z in zip(axes[0], selected, mags, slices):
        im = ax.imshow(np.where(case.mask[z], mag[z], np.nan), cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Displacement magnitude (mm)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_tables(out_dir: Path, case_rows: List[Dict[str, object]], aggregate_rows: List[Dict[str, object]], paper_df: pd.DataFrame, contrast_df: pd.DataFrame) -> None:
    write_csv(out_dir / "tables/case_metrics_comprehensive.csv", case_rows)
    write_csv(out_dir / "tables/aggregate_metrics_comprehensive.csv", aggregate_rows)
    paper_df.to_csv(out_dir / "tables/paper_table_organized.csv", index=False)
    contrast_df.to_csv(out_dir / "tables/contrast_table_4dct_over_oasis.csv", index=False)

    case_cols = [
        "dataset",
        "role",
        "method_label",
        "case_label",
        "mag_mean_mm",
        "mag_p95_mm",
        "pct_gt_10mm",
        "diffusion_energy",
        "bending_energy",
        "log_jdet_std",
        "jdet_le0_pct",
        "corr_length_mm",
    ]
    case_md = pd.DataFrame(case_rows)[case_cols]
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables/case_metrics_comprehensive.md").write_text(markdown_table(case_md) + "\n")
    (out_dir / "tables/paper_table_organized.md").write_text(markdown_table(paper_df) + "\n")
    (out_dir / "tables/contrast_table_4dct_over_oasis.md").write_text(markdown_table(contrast_df) + "\n")


def write_report(out_dir: Path, paper_df: pd.DataFrame, contrast_df: pd.DataFrame) -> None:
    report = f"""# Current-Run Motion-Regime Characterization

## Scope

This report characterizes method-estimated dense displacement fields, not
ground-truth physical motion. OASIS uses the seed-3 shuffled first10 protocol
with the fixed raw-image-positive foreground mask (`image > 0`). 4DCT uses the
fixed T00 lung mask on the cropped 1mm pTV-prep grid.

Primary methods are OASIS SINR cps4 alpha1000 and 4DCT IDIR default. Sensitivity
methods are OASIS DirectCP cps4 alpha1000 and 4DCT DirectCP 32-16-8-4.

## Organized Paper Table

{markdown_table(paper_df)}

## 4DCT / OASIS Contrast

{markdown_table(contrast_df)}

## Figures And Interpretation

### Figure 1: Primary Motion Amplitude

![Primary amplitude](plots/primary_amplitude_boxplots.png)

This plot compares displacement magnitude on the correct foreground masks. Large
4DCT amplitudes indicate respiratory-motion scale; OASIS amplitudes are smaller
but should be interpreted together with smoothness and coherence.

### Figure 2: Primary Smoothness And Jacobian Properties

![Primary smoothness](plots/primary_smoothness_boxplots.png)

Diffusion, bending, strain, log-Jacobian spread, and folding quantify local
deformation burden. These metrics expose whether a large-amplitude field is
also locally irregular.

### Figure 3: Primary Spatial Coherence

![Primary autocorrelation](plots/primary_autocorrelation.png)

Vector autocorrelation over increasing lag summarizes spatial coherence. A
slowly decaying curve means motion remains coherent over larger physical
distances.

![Primary semivariogram](plots/primary_semivariogram.png)

The semivariogram gives the complementary view: larger values at short lags
indicate stronger local variation.

### Figure 4: Primary Coarse-Grid Explainability

![Primary coarse explained](plots/primary_coarse_explained_energy.png)

Coarse-grid explained energy asks how much of the displacement field can be
captured by local control sampled at coarse spacings. This is the most direct
off-grid/local-control proxy in this analysis.

![Primary coarse residual](plots/primary_coarse_residual_p95.png)

Residual P95 shows the magnitude of field content not captured at each coarse
spacing.

### Figure 5: Primary Low-Pass Explainability

![Primary lowpass](plots/primary_lowpass_explained_energy.png)

Gaussian low-pass explained energy separates broad smooth components from
higher-frequency local corrections.

### Figure 6: Representative Magnitude Slices

![Representative slices](plots/primary_representative_magnitude_slices.png)

Slices are selected from median-P95 and high-P95 cases for each primary method.
The color scale is shared across displayed examples.

## Sensitivity Figures

![All amplitude](plots/all_methods_amplitude_boxplots.png)

![All smoothness](plots/all_methods_smoothness_boxplots.png)

![All autocorrelation](plots/all_methods_autocorrelation.png)

![All coarse explained](plots/all_methods_coarse_explained_energy.png)

## Output Files

- `tables/case_metrics_comprehensive.csv`
- `tables/aggregate_metrics_comprehensive.csv`
- `tables/paper_table_organized.csv`
- `tables/contrast_table_4dct_over_oasis.csv`
- `curves/autocorrelation_curves.csv`
- `curves/lowpass_curves.csv`
- `curves/coarse_grid_curves.csv`
- `input_manifest.csv`
- `input_manifest_sha256.csv`
"""
    (out_dir / "review_report.md").write_text(report)


def run_sanity_tests() -> None:
    shape = (16, 18, 20)
    mask = np.ones(shape, dtype=bool)
    zero = np.zeros((*shape, 3), dtype=np.float32)
    z = smoothness_metrics(zero, mask)
    assert abs(z["diffusion_energy"]) < 1e-8
    assert abs(z["bending_energy"]) < 1e-8
    assert abs(z["log_jdet_std"]) < 1e-8
    assert abs(z["jdet_le0_pct"]) < 1e-8

    const = np.zeros((*shape, 3), dtype=np.float32)
    const[..., 0] = 3.0
    c = smoothness_metrics(const, mask)
    assert abs(c["diffusion_energy"]) < 1e-8
    assert abs(c["bending_energy"]) < 1e-8

    rng = np.random.default_rng(1)
    noise = rng.normal(size=(*shape, 3)).astype(np.float32)
    smooth = np.empty_like(noise)
    for component in range(3):
        smooth[..., component] = gaussian_filter(noise[..., component], sigma=3.0)
    fake_noise = CaseField("test", "primary", "noise", "noise", 0, "noise", "", "", Path("."), Path("."), noise, mask, {})
    fake_smooth = CaseField("test", "primary", "smooth", "smooth", 0, "smooth", "", "", Path("."), Path("."), smooth, mask, {})
    coarse_noise = coarse_grid_curves(fake_noise, [8])[0]["explained_energy"]
    coarse_smooth = coarse_grid_curves(fake_smooth, [8])[0]["explained_energy"]
    assert float(coarse_smooth) > float(coarse_noise)


def run_analysis(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "curves").mkdir(exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    cases, manifest = load_cases(args)
    write_csv(out_dir / "input_manifest.csv", [{k: v for k, v in row.items() if k != "sha256"} for row in manifest])
    write_csv(out_dir / "input_manifest_sha256.csv", manifest)

    case_rows: List[Dict[str, object]] = []
    corr_rows: List[Dict[str, object]] = []
    lowpass_rows: List[Dict[str, object]] = []
    coarse_rows: List[Dict[str, object]] = []
    data_audit: List[Dict[str, object]] = []

    for case in cases:
        row = amplitude_metrics(case)
        row.update(pca_metrics(case, args.pca_sample, rng))
        c_rows, corr_length, censored = vector_autocorr(case, LAGS_MM, args.max_pairs_per_lag_axis, rng)
        row["corr_length_mm"] = float(corr_length)
        row["corr_length_censored"] = bool(censored)
        case_rows.append(row)
        corr_rows.extend(c_rows)
        lowpass_rows.extend(lowpass_curves(case, LOWPASS_SIGMAS_MM))
        coarse_rows.extend(coarse_grid_curves(case, COARSE_SPACINGS_MM))
        data_audit.append(
            {
                "dataset": case.dataset,
                "role": case.role,
                "method_key": case.method_key,
                "method_label": case.method_label,
                "case_index": case.case_index,
                "case_label": case.case_label,
                "fixed_id": case.fixed_id,
                "moving_id": case.moving_id,
                "field_shape_zyx": "x".join(str(v) for v in case.disp_xyz_mm.shape[:3]),
                "component_order": "xyz",
                "axis_order": "zyx",
                "spacing_mm": "1x1x1",
                "mask_voxels": int(case.mask.sum()),
                "mask_fraction": float(case.mask.mean()),
                "mask_rule": case.meta["mask_rule"],
                "run_dir": str(case.run_dir),
                "summary_path": str(case.summary_path),
                "disp_path": str(case.meta["disp_path"]),
                "pair_file": str(case.meta.get("pair_file", "")),
            }
        )

    aggregate_rows = aggregate_case_metrics(case_rows)
    paper_df = make_paper_table(aggregate_rows, coarse_rows)
    contrast_df = make_contrast_table(paper_df)

    write_csv(out_dir / "curves/autocorrelation_curves.csv", corr_rows)
    write_csv(out_dir / "curves/lowpass_curves.csv", lowpass_rows)
    write_csv(out_dir / "curves/coarse_grid_curves.csv", coarse_rows)
    write_csv(out_dir / "data_audit.csv", data_audit)
    write_tables(out_dir, case_rows, aggregate_rows, paper_df, contrast_df)

    case_df = pd.DataFrame(case_rows)
    corr_df = pd.DataFrame(corr_rows)
    lowpass_df = pd.DataFrame(lowpass_rows)
    coarse_df = pd.DataFrame(coarse_rows)

    plot_box_metric(
        case_df,
        [("mag_mean_mm", "Mean magnitude"), ("mag_p95_mm", "P95 magnitude"), ("mag_p99_mm", "P99 magnitude"), ("pct_gt_10mm", "% voxels > 10mm")],
        out_dir / "plots/primary_amplitude_boxplots.png",
        "Primary method displacement amplitude",
        primary_only=True,
    )
    plot_box_metric(
        case_df,
        [("diffusion_energy", "Diffusion"), ("bending_energy", "Bending"), ("strain_norm_mean", "Strain mean"), ("log_jdet_std", "logJ std"), ("jdet_le0_pct", "J<=0 %")],
        out_dir / "plots/primary_smoothness_boxplots.png",
        "Primary method smoothness and Jacobian metrics",
        primary_only=True,
    )
    plot_curves(corr_df, "lag_mm", "autocorrelation", out_dir / "plots/primary_autocorrelation.png", "Primary vector autocorrelation", "Lag (mm)", "Vector autocorrelation", True)
    plot_curves(corr_df, "lag_mm", "semivariogram_mm2", out_dir / "plots/primary_semivariogram.png", "Primary semivariogram", "Lag (mm)", "Semivariogram (mm^2)", True)
    plot_curves(coarse_df, "spacing_mm", "explained_energy", out_dir / "plots/primary_coarse_explained_energy.png", "Primary coarse-grid explained energy", "Coarse spacing (mm)", "Explained energy", True)
    plot_curves(coarse_df, "spacing_mm", "residual_p95_mm", out_dir / "plots/primary_coarse_residual_p95.png", "Primary coarse-grid residual P95", "Coarse spacing (mm)", "Residual P95 (mm)", True)
    plot_curves(lowpass_df, "sigma_mm", "explained_energy", out_dir / "plots/primary_lowpass_explained_energy.png", "Primary low-pass explained energy", "Gaussian sigma (mm)", "Explained energy", True)
    plot_representative_slices(cases, case_df, out_dir / "plots/primary_representative_magnitude_slices.png")

    plot_box_metric(
        case_df,
        [("mag_mean_mm", "Mean magnitude"), ("mag_p95_mm", "P95 magnitude"), ("pct_gt_10mm", "% voxels > 10mm")],
        out_dir / "plots/all_methods_amplitude_boxplots.png",
        "All selected methods displacement amplitude",
        primary_only=False,
    )
    plot_box_metric(
        case_df,
        [("diffusion_energy", "Diffusion"), ("bending_energy", "Bending"), ("log_jdet_std", "logJ std"), ("jdet_le0_pct", "J<=0 %")],
        out_dir / "plots/all_methods_smoothness_boxplots.png",
        "All selected methods smoothness and Jacobian metrics",
        primary_only=False,
    )
    plot_curves(corr_df, "lag_mm", "autocorrelation", out_dir / "plots/all_methods_autocorrelation.png", "All selected methods vector autocorrelation", "Lag (mm)", "Vector autocorrelation", False)
    plot_curves(coarse_df, "spacing_mm", "explained_energy", out_dir / "plots/all_methods_coarse_explained_energy.png", "All selected methods coarse-grid explained energy", "Coarse spacing (mm)", "Explained energy", False)

    write_report(out_dir, paper_df, contrast_df)
    write_json(
        out_dir / "analysis_metadata.json",
        {
            "rng_seed": RNG_SEED,
            "lags_mm": list(LAGS_MM),
            "lowpass_sigmas_mm": list(LOWPASS_SIGMAS_MM),
            "coarse_spacings_mm": list(COARSE_SPACINGS_MM),
            "n_cases": len(cases),
            "notes": [
                "Fields are method-estimated DVFs, not ground-truth physical motion.",
                "OASIS foreground uses fixed raw image > 0.",
                "4DCT foreground uses fixed T00 lung mask on cropped 1mm grid.",
            ],
        },
    )


def main() -> None:
    args = parse_args()
    if args.run_sanity_tests:
        run_sanity_tests()
    if not args.skip_analysis:
        run_analysis(args)


if __name__ == "__main__":
    main()
