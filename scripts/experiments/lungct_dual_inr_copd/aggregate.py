#!/usr/bin/env python3
"""Aggregate COPD A/N/P DUAL-INR records with fixed pTVReg and MR references."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from scripts.experiments.lungct_dual_inr_author.posthoc_author_regularity import (
    REPORT_METRICS,
    compute_common_regularity,
)


CASES = tuple(range(1, 11))
NEW_ARMS = ("A0", "A1", "N0", "N1", "P0", "P1")
REFERENCE_ARMS = ("pTVReg official", "MR-D-BSCP P2e")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in records:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list, tuple)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def _summary_tre(summary: Mapping[str, Any], arm: str) -> dict[str, float | None]:
    if arm.startswith("A"):
        metrics = _nested(summary, "landmark_metrics", "dense_dvf")
        return {
            "headline_native_snap_mm": _finite(_nested(metrics, "snap", "mean_mm"), f"{arm}.dense.snap"),
            "native_subvoxel_mm": _finite(_nested(metrics, "subvoxel", "mean_mm"), f"{arm}.dense.sub"),
            "reg1mm_snap_mm": None,
            "reg1mm_subvoxel_mm": None,
        }
    metrics = _nested(summary, "metrics")
    if arm.startswith("N"):
        return {
            "headline_native_snap_mm": _finite(
                _nested(metrics, "headline_tre", "tre_mean_mm"), f"{arm}.headline"
            ),
            "native_subvoxel_mm": _finite(
                _nested(metrics, "tre_subvoxel", "tre_mean_mm"), f"{arm}.native.sub"
            ),
            "reg1mm_snap_mm": None,
            "reg1mm_subvoxel_mm": None,
        }
    headline = _nested(metrics, "headline_tre", "tre_mean_mm")
    native_sub = metrics.get("native_crop_tre_subvoxel")
    reg_snap = metrics.get("tre_snap_to_voxel")
    reg_sub = metrics.get("tre_subvoxel")
    return {
        "headline_native_snap_mm": _finite(headline, f"{arm}.headline"),
        "native_subvoxel_mm": _finite(native_sub["tre_mean_mm"], f"{arm}.native.sub") if isinstance(native_sub, Mapping) else None,
        "reg1mm_snap_mm": _finite(reg_snap["tre_mean_mm"], f"{arm}.reg.snap") if isinstance(reg_snap, Mapping) else None,
        "reg1mm_subvoxel_mm": _finite(reg_sub["tre_mean_mm"], f"{arm}.reg.sub") if isinstance(reg_sub, Mapping) else None,
    }


def _author_crop_mask(summary: Mapping[str, Any], full_mask: np.ndarray) -> np.ndarray:
    raw_bbox = summary.get("crop_bbox")
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 3:
        raise ValueError("Author summary does not contain a three-axis crop_bbox")
    slices = []
    for axis in raw_bbox:
        if not isinstance(axis, Mapping):
            raise ValueError("Malformed author crop_bbox")
        slices.append(slice(axis.get("start"), axis.get("stop"), axis.get("step")))
    return np.ascontiguousarray(full_mask[tuple(slices)].astype(np.uint8, copy=False))


def _regularity(field: np.ndarray, mask: np.ndarray, spacing_xyz: Iterable[float]) -> dict[str, float]:
    result = compute_common_regularity(
        disp_vox_xyz=field,
        fixed_lobe_union_xyz=mask,
        spacing_xyz_mm=tuple(float(value) for value in spacing_xyz),
    )
    return {key: _finite(result["metrics"][key], key) for key in REPORT_METRICS}


def _new_record(
    *,
    arm: str,
    case: int,
    author_run_root: Path,
    controlled_run_root: Path,
    author_data_root: Path,
) -> dict[str, Any]:
    if arm.startswith("A"):
        case_dir = author_run_root / arm.lower() / f"case_{case:02d}"
        field_name = "dense_dvf_crop_voxel_xyz.npy"
        full_mask = np.load(
            author_data_root / f"case_{case:02d}" / "metric_masks" / "fixed_iBH_r231_metricmask_author_full_xyz.npy",
            allow_pickle=False,
        )
        summary = _read_json(case_dir / "summary.json")
        mask = _author_crop_mask(summary, full_mask)
        coverage = float(mask.sum() / full_mask.sum()) if full_mask.sum() else float("nan")
        protocol = "dual_author_native_lungbbox_totalseg"
        preprocessing = "author native lobe-union crop; fixed iBH five-lobe sampling"
        spacing = summary["image_spacing_xyz"]
    else:
        case_dir = controlled_run_root / arm.lower() / f"case_{case:02d}"
        field_name = "dense_dvf_xyz_voxel.npy"
        summary = _read_json(case_dir / "summary.json")
        mask = np.load(case_dir / "fixed_metric_mask_xyz.npy", allow_pickle=False)
        coverage = 1.0
        protocol = str(summary.get("protocol_tags", {}).get("dataset", "copd"))
        preprocessing = str(summary.get("data", {}).get("data_provenance", {}).get("protocol_handle", arm))
        spacing = summary["spacing_xyz_mm"]
    if not (case_dir / "DONE").is_file():
        raise ValueError(f"{arm} case{case:02d} is missing DONE")
    field = np.load(case_dir / field_name, allow_pickle=False)
    if field.shape[:3] != mask.shape:
        raise ValueError(f"{arm} case{case:02d} field/mask geometry mismatch")
    tre = _summary_tre(summary, arm)
    metrics = _regularity(field, mask, spacing)
    return {
        "method": arm,
        "case_id": case,
        "kind": "new_dual_inr",
        "protocol": protocol,
        "preprocessing": preprocessing,
        "headline_native_snap_mm": tre["headline_native_snap_mm"],
        "native_subvoxel_mm": tre["native_subvoxel_mm"],
        "reg1mm_snap_mm": tre["reg1mm_snap_mm"],
        "reg1mm_subvoxel_mm": tre["reg1mm_subvoxel_mm"],
        "metric_mask_voxels": int(mask.sum()),
        "metric_mask_coverage_of_available_fixed_r231": coverage,
        "regularity_available": True,
        **metrics,
    }


def _ptv_reference(path: Path) -> list[dict[str, Any]]:
    from scipy.io import loadmat

    payload = loadmat(path)
    values = np.asarray(payload["TRE_e"], dtype=float).reshape(-1)
    if values.size != 10 or not np.isfinite(values).all():
        raise ValueError("Official COPD_TRE_best.mat does not contain ten finite TRE_e values")
    return [
        {
            "method": "pTVReg official",
            "case_id": case,
            "kind": "reference",
            "protocol": "official_ptvreg_copd_best_1mm",
            "preprocessing": "official pTVReg 1 mm landmark crop; exact MATLAB crop convention",
            "headline_native_snap_mm": float(values[case - 1]),
            "native_subvoxel_mm": None,
            "reg1mm_snap_mm": None,
            "reg1mm_subvoxel_mm": None,
            "metric_mask_voxels": None,
            "metric_mask_coverage_of_available_fixed_r231": None,
            "regularity_available": False,
            **{metric: None for metric in REPORT_METRICS},
        }
        for case in CASES
    ]


def _mr_reference(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    if len(source) != 10:
        raise ValueError("MR-D-BSCP P2e case table must contain exactly ten rows")
    by_case = {int(row["case_id"]): row for row in source}
    if sorted(by_case) != list(CASES):
        raise ValueError("MR-D-BSCP P2e table must contain cases 01-10 exactly once")
    records = []
    for case in CASES:
        row = by_case[case]
        records.append(
            {
                "method": "MR-D-BSCP P2e",
                "case_id": case,
                "kind": "reference",
                "protocol": "ptv_directcp_multistage",
                "preprocessing": "legacy pTV-style 1 mm crop; matched to COPD P0/P1",
                "headline_native_snap_mm": _finite(row["native_crop_snap_tre_mean_mm"], "MR headline"),
                "native_subvoxel_mm": _finite(row["native_crop_sub_tre_mean_mm"], "MR native sub"),
                "reg1mm_snap_mm": _finite(row["reg1mm_snap_tre_mean_mm"], "MR reg snap"),
                "reg1mm_subvoxel_mm": _finite(row["reg1mm_sub_tre_mean_mm"], "MR reg sub"),
                "metric_mask_voxels": None,
                "metric_mask_coverage_of_available_fixed_r231": None,
                "regularity_available": True,
                "diffusion_l2_fg": _finite(row["fg_diffusion_l2"], "MR diffusion"),
                "bending_energy_fg": _finite(row["fg_bending_energy"], "MR bending"),
                "log_jdet_std_fg": _finite(row["fg_log_jdet_std"], "MR SDlogJ"),
                "jdet_le0_pct_fg": _finite(row["fg_jdet_le0_pct"], "MR fold"),
                **{metric: None for metric in REPORT_METRICS if metric not in {"diffusion_l2_fg", "bending_energy_fg", "log_jdet_std_fg", "jdet_le0_pct_fg"}},
            }
        )
    return records


def _mean_sd(values: Iterable[Any]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None, None
    return float(statistics.fmean(finite)), float(statistics.stdev(finite)) if len(finite) > 1 else 0.0


def _population(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    methods = [*NEW_ARMS, *REFERENCE_ARMS]
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        if len(group) != 10:
            raise ValueError(f"{method} does not have ten case rows")
        row: dict[str, Any] = {
            "method": method,
            "n_cases": len(group),
            "protocol": group[0]["protocol"],
            "preprocessing": group[0]["preprocessing"],
        }
        for metric in ("headline_native_snap_mm", "native_subvoxel_mm", "reg1mm_snap_mm", "reg1mm_subvoxel_mm", "diffusion_l2_fg", "bending_energy_fg", "log_jdet_std_fg", "jdet_le0_pct_fg"):
            mean, sd = _mean_sd(item.get(metric) for item in group)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sample_sd"] = sd
        output.append(row)
    return output


def _render_report(rows: list[Mapping[str, Any]], population: list[Mapping[str, Any]], acceptance: Mapping[str, Any]) -> str:
    by_case = {case: {row["method"]: row for row in rows if row["case_id"] == case} for case in CASES}
    lines = [
        "# COPD DUAL-INR six-arm camera-ready readout",
        "",
        "## Cohort identity",
        "",
        "- **Dataset:** DIR-Lab COPD cases 01-10, not the separate DIR-Lab 4DCT T00/T50 cohort.",
        "- **Direction:** moving eBHCT to fixed iBHCT (eBH -> iBH), with 300 COPD landmarks per case.",
        "- **Source convention:** native records use the COPD iBH/eBH image-pair layout; P records use the COPD legacy Python-bounds pTV-style 1 mm artifact.",
        "",
        "All six new arms use one seed and all ten COPD cases. TRE values are physical mm; pTVReg is intentionally TRE-only because its official artifact has no dense field.",
        "",
        "## Casewise headline native/crop snap TRE",
        "",
        "| COPD case | A0 | A1 | N0 | N1 | P0 | P1 | pTVReg official | MR-D-BSCP P2e |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        items = by_case[case]
        values = [items[method]["headline_native_snap_mm"] for method in [*NEW_ARMS, *REFERENCE_ARMS]]
        lines.append("| " + str(case) + " | " + " | ".join(f"{float(value):.4f}" for value in values) + " |")
    lines.extend(["", "## Population summary", "", "| Method | TRE mean ± sample SD (mm) | SDlogJ | J≤0 (%) | Bending | Diffusion |", "|---|---:|---:|---:|---:|---:|"])
    for row in population:
        def fmt(metric: str) -> str:
            mean, sd = row.get(f"{metric}_mean"), row.get(f"{metric}_sample_sd")
            return "N/A" if mean is None else f"{mean:.4f} ± {sd:.4f}"
        lines.append(
            f"| {row['method']} | {fmt('headline_native_snap_mm')} | {fmt('log_jdet_std_fg')} | "
            f"{fmt('jdet_le0_pct_fg')} | {fmt('bending_energy_fg')} | {fmt('diffusion_l2_fg')} |"
        )
    lines.extend(
        [
            "",
            "## Protocol limits",
            "",
            "- A0/A1 retain the author native TotalSegmentator five-lobe crop and sampling. A1 removes only the semantic loss and replays its paired A0 blur transitions.",
            "- N0/N1 use native full-volume fixed-iBH R231 sampling. P0/P1 use the existing legacy Python pTV-style 1 mm crop and no lung mask in optimization. Only N0–N1 and P0–P1 are schedule-controlled contrasts.",
            "- Official pTVReg uses exact MATLAB crop bounds and is a contextual TRE-only reference. MR-D-BSCP P2e is the single lowest population-TRE completed 1 mm COPD cohort, used whole rather than per-case selection.",
            "- Diffusion and bending are voxel-coordinate finite-difference diagnostics; cross-domain magnitude comparisons remain protocol-tagged. SDlogJ and folding use the available fixed-iBH R231 metric ROI, and A rows report its crop coverage.",
            "",
            f"Technical acceptance: {acceptance['accepted_records']}/{acceptance['expected_records']} new records complete; regularity computed for {acceptance['regularity_records']} new/reference-field records.",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    author_root = args.author_run_root.resolve()
    controlled_root = args.controlled_run_root.resolve()
    author_data = args.author_data_root.resolve()
    review = args.review_root.resolve()
    records = [
        _new_record(arm=arm, case=case, author_run_root=author_root, controlled_run_root=controlled_root, author_data_root=author_data)
        for arm in NEW_ARMS
        for case in CASES
    ]
    records.extend(_ptv_reference(args.ptvreg_reference.resolve()))
    records.extend(_mr_reference(args.mr_reference.resolve()))
    population = _population(records)
    accepted = sum(1 for row in records if row["kind"] == "new_dual_inr")
    regularity_records = sum(1 for row in records if row["regularity_available"])
    acceptance = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "pass" if accepted == 60 else "fail",
        "expected_records": 60,
        "accepted_records": accepted,
        "regularity_records": regularity_records,
        "references": {
            "ptvreg": "external/pTVreg/examples_ptv/results/COPD_TRE_best.mat; TRE only",
            "mr_d_bscp": "P2e complete 1 mm cohort selected by population headline TRE",
        },
        "scientific_performance_gate": None,
    }
    review.mkdir(parents=True, exist_ok=True)
    _write_csv(review / "casewise_tre_and_regularity.csv", records)
    _write_csv(review / "population_summary.csv", population)
    _write_json(review / "acceptance_report.json", acceptance)
    (review / "camera_ready_readout.md").write_text(_render_report(records, population, acceptance), encoding="utf-8")
    (review / "acceptance_report.md").write_text(
        f"# COPD six-arm acceptance\n\nStatus: **{acceptance['status']}**. New records: {accepted}/60. "
        "TRE and regularity values were not execution gates.\n",
        encoding="utf-8",
    )
    return {"status": acceptance["status"], "review_root": str(review), "records": len(records)}


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-run-root", type=Path, required=True)
    parser.add_argument("--controlled-run-root", type=Path, required=True)
    parser.add_argument("--author-data-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--ptvreg-reference", type=Path, default=root / "external/pTVreg/examples_ptv/results/COPD_TRE_best.mat")
    parser.add_argument("--mr-reference", type=Path, default=root / "experiments/2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity/copd_case_rows.csv")
    return parser


def main() -> None:
    result = aggregate(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
