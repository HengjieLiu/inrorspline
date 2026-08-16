#!/usr/bin/env python
"""Aggregate LungCT case01/case08 masked-NCC FD-BE quick A/B results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


DEFAULT_MASKED_RUN_ROOT = Path("/database/inrorspline/runs/lungct_case01_case08_masked_ncc_fdbe_quick_v1")
DEFAULT_UNMASKED_RUN_ROOT = Path("/database/inrorspline/runs/dirlab_case01_case08_panelA_lungmask_diag_curves_v1")
DEFAULT_REVIEW_DIR = REPO_ROOT / "experiments/2026-08-14_lungct_case01_case08_masked_ncc_fdbe_quick"


@dataclass(frozen=True)
class MethodPair:
    slug: str
    label: str
    unmasked_impl: str
    masked_impl: str
    lr: float
    alpha: float
    old_name_kind: str
    epochs: int | None = 2500
    stage_full_cps: str | None = None
    stage_downsample: str | None = None
    stage_epochs: str | None = None


METHODS = (
    MethodPair(
        slug="inr_bscp",
        label="INR-BSCP",
        unmasked_impl="sinr_cps4_global_ncc_fd_be",
        masked_impl="sinr_cps4_masked_ncc_fd_be",
        lr=1.0e-4,
        alpha=10000.0,
        old_name_kind="inr_bscp",
    ),
    MethodPair(
        slug="d_bscp",
        label="D-BSCP",
        unmasked_impl="direct_cp_cps4_global_ncc_fd_be",
        masked_impl="direct_cp_cps4_masked_ncc_fd_be",
        lr=1.0e-3,
        alpha=10000.0,
        old_name_kind="d_bscp",
    ),
    MethodPair(
        slug="mr_d_bscp",
        label="MR-D-BSCP",
        unmasked_impl="direct_cp_32_16_8_4_global_ncc_fd_be",
        masked_impl="direct_cp_32_16_8_4_masked_ncc_fd_be",
        lr=1.0e-3,
        alpha=10000.0,
        old_name_kind="mr_d_bscp",
        epochs=None,
        stage_full_cps="32,16,8,4",
        stage_downsample="8,4,2,1",
        stage_epochs="400,400,800,900",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masked-run-root", type=Path, default=DEFAULT_MASKED_RUN_ROOT)
    parser.add_argument("--unmasked-run-root", type=Path, default=DEFAULT_UNMASKED_RUN_ROOT)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--cases", default="1,8")
    return parser.parse_args()


def utc_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S +0000")


def parse_cases(text: str) -> tuple[int, ...]:
    cases: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            cases.append(int(token))
    return tuple(dict.fromkeys(cases))


def old_run_name(case_id: int, method: MethodPair) -> str:
    if method.old_name_kind == "inr_bscp":
        return f"lungmask_curves_case{case_id:02d}_panelA_inr_bscp_e2500_lr1em4_seed1_diag10"
    if method.old_name_kind == "d_bscp":
        return f"lungmask_curves_case{case_id:02d}_panelA_d_bscp_e2500_lr1em3_seed1_diag10"
    if method.old_name_kind == "mr_d_bscp":
        return (
            f"lungmask_curves_case{case_id:02d}_panelA_mr_d_bscp_"
            "sched32-16-8-4_ds8-4-2-1_e400-400-800-900_a10000_lr0p001_seed1_diag10"
        )
    raise ValueError(method.old_name_kind)


def masked_run_name(case_id: int, method: MethodPair) -> str:
    case_token = f"case{case_id:02d}"
    if method.stage_epochs is not None:
        stage_full = str(method.stage_full_cps).replace(",", "-")
        stage_down = str(method.stage_downsample).replace(",", "-")
        stage_epochs = str(method.stage_epochs).replace(",", "-")
        return (
            f"{method.masked_impl}_{case_token}_sched{stage_full}_ds{stage_down}_"
            f"e{stage_epochs}_lr{c.format_float(method.lr)}_a{c.format_float(method.alpha)}_seed1"
        )
    return (
        f"{method.masked_impl}_{case_token}_e{int(method.epochs or 2500)}_"
        f"lr{c.format_float(method.lr)}_a{c.format_float(method.alpha)}_seed1"
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def last_curve_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    rows = list(csv.DictReader(path.open()))
    return rows[-1] if rows else {}


def nested(summary: dict[str, Any], name: str, field: str) -> float | None:
    block = summary.get(name)
    if not isinstance(block, dict):
        return None
    return finite_float(block.get(field))


def history_tail(summary: dict[str, Any], field: str) -> float | None:
    tail = summary.get("history_tail")
    if not isinstance(tail, list) or not tail:
        return None
    last = tail[-1]
    if not isinstance(last, dict):
        return None
    return finite_float(last.get(field))


def row_from_summary(
    *,
    case_id: int,
    method: MethodPair,
    condition: str,
    impl: str,
    run_dir: Path,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    curve_path = run_dir / "global_ncc_diagnostic.csv"
    summary = read_json(summary_path)
    curve = last_curve_row(curve_path)
    regularity = summary.get("regularity_fg_registration_crop")
    if not isinstance(regularity, dict):
        regularity = {}
    global_ncc_value = finite_float(curve.get("global_ncc_value"))
    lung_mask_one_minus_ncc = finite_float(curve.get("lung_mask_one_minus_ncc"))
    return {
        "case_key": f"case{case_id:02d}",
        "case_id": case_id,
        "method_slug": method.slug,
        "display_method": method.label,
        "condition": condition,
        "implementation_id": impl,
        "status": summary.get("status", "missing"),
        "summary_path": str(summary_path),
        "curve_path": str(curve_path),
        "loss_domain": summary.get("loss_domain", ""),
        "mask_rule": summary.get("mask_rule", ""),
        "protocol_tag": summary.get("protocol_tag", ""),
        "headline_tre_metric": summary.get("headline_tre_metric", ""),
        "headline_tre_mean_mm": nested(summary, "native_crop_tre_snap_to_voxel", "tre_mean_mm"),
        "headline_tre_std_mm": nested(summary, "native_crop_tre_snap_to_voxel", "tre_std_mm"),
        "reg1mm_snap_tre_mean_mm": nested(summary, "tre_snap_to_voxel", "tre_mean_mm"),
        "reg1mm_subvoxel_tre_mean_mm": nested(summary, "tre_subvoxel", "tre_mean_mm"),
        "fg_stdlogj": finite_float(regularity.get("log_jdet_std_fg")),
        "fg_jdet_le0_pct": finite_float(regularity.get("jdet_le0_pct_fg")),
        "fg_bending_energy": finite_float(regularity.get("bending_energy_fg")),
        "fg_diffusion_l2": finite_float(regularity.get("diffusion_l2_fg")),
        "full_crop_one_minus_ncc": None if global_ncc_value is None else 1.0 - global_ncc_value,
        "fixed_t00_lung_mask_one_minus_ncc": lung_mask_one_minus_ncc,
        "final_loss": history_tail(summary, "loss"),
        "final_data_loss": history_tail(summary, "data_loss"),
        "final_reg_loss": history_tail(summary, "reg_loss"),
        "elapsed_sec": finite_float(summary.get("elapsed_sec")),
        "peak_mem_gb": finite_float(summary.get("peak_mem_gb")),
    }


def prefix(condition: str, row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "status",
        "implementation_id",
        "loss_domain",
        "mask_rule",
        "headline_tre_mean_mm",
        "fg_stdlogj",
        "fg_jdet_le0_pct",
        "fg_bending_energy",
        "fg_diffusion_l2",
        "full_crop_one_minus_ncc",
        "fixed_t00_lung_mask_one_minus_ncc",
        "final_loss",
        "final_data_loss",
        "final_reg_loss",
        "elapsed_sec",
        "peak_mem_gb",
        "summary_path",
    ]
    return {f"{condition}_{key}": row.get(key) for key in keep}


def delta(masked: Any, unmasked: Any) -> float | None:
    masked_f = finite_float(masked)
    unmasked_f = finite_float(unmasked)
    if masked_f is None or unmasked_f is None:
        return None
    return masked_f - unmasked_f


def side_by_side(unmasked: dict[str, Any], masked: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_key": unmasked["case_key"],
        "case_id": unmasked["case_id"],
        "method_slug": unmasked["method_slug"],
        "display_method": unmasked["display_method"],
    }
    row.update(prefix("unmasked", unmasked))
    row.update(prefix("masked", masked))
    for field in [
        "headline_tre_mean_mm",
        "fg_stdlogj",
        "fg_jdet_le0_pct",
        "fg_bending_energy",
        "fg_diffusion_l2",
        "full_crop_one_minus_ncc",
        "fixed_t00_lung_mask_one_minus_ncc",
        "final_data_loss",
        "final_reg_loss",
        "elapsed_sec",
        "peak_mem_gb",
    ]:
        row[f"delta_{field}"] = delta(masked.get(field), unmasked.get(field))
    return row


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def fmt(value: Any, digits: int = 4) -> str:
    value_f = finite_float(value)
    if value_f is None:
        return "NA"
    return f"{value_f:.{digits}f}"


def write_report(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# LungCT masked-NCC FD-BE quick A/B",
        "",
        f"- Updated: `{utc_text()}`",
        "- Comparison: prior full-crop NCC FD-BE runs versus new fixed-T00 lung-mask NCC FD-BE runs.",
        "- Scope: DIR-Lab 4DCT case01 and case08 only; this is a quick protocol check, not a population conclusion.",
        "- Caution: this isolates the NCC data-loss domain inside the FD-BE family; it is not a native-IDIR equivalence claim.",
        "",
        "| Case | Method | TRE old | TRE masked | Delta | stdLogJ old | stdLogJ masked | Lung 1-NCC old | Lung 1-NCC masked |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {method} | {old_tre} | {new_tre} | {delta_tre} | "
            "{old_logj} | {new_logj} | {old_lung} | {new_lung} |".format(
                case=row["case_key"],
                method=row["display_method"],
                old_tre=fmt(row.get("unmasked_headline_tre_mean_mm"), 3),
                new_tre=fmt(row.get("masked_headline_tre_mean_mm"), 3),
                delta_tre=fmt(row.get("delta_headline_tre_mean_mm"), 3),
                old_logj=fmt(row.get("unmasked_fg_stdlogj"), 4),
                new_logj=fmt(row.get("masked_fg_stdlogj"), 4),
                old_lung=fmt(row.get("unmasked_fixed_t00_lung_mask_one_minus_ncc"), 5),
                new_lung=fmt(row.get("masked_fixed_t00_lung_mask_one_minus_ncc"), 5),
            )
        )
    lines.extend(
        [
            "",
            "Generated files:",
            "",
            "- `masked_vs_unmasked_long.csv`",
            "- `masked_vs_unmasked_summary.csv`",
            "- `masked_vs_unmasked_manifest.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    long_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    for case_id in parse_cases(args.cases):
        for method in METHODS:
            unmasked_dir = Path(args.unmasked_run_root) / method.unmasked_impl / old_run_name(case_id, method)
            masked_dir = Path(args.masked_run_root) / method.masked_impl / masked_run_name(case_id, method)
            unmasked = row_from_summary(
                case_id=case_id,
                method=method,
                condition="unmasked_full_crop_fdbe",
                impl=method.unmasked_impl,
                run_dir=unmasked_dir,
            )
            masked = row_from_summary(
                case_id=case_id,
                method=method,
                condition="masked_fixed_t00_lungmask_fdbe",
                impl=method.masked_impl,
                run_dir=masked_dir,
            )
            long_rows.extend([unmasked, masked])
            side_rows.append(side_by_side(unmasked, masked))
    args.review_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.review_dir / "masked_vs_unmasked_long.csv", long_rows)
    write_csv(args.review_dir / "masked_vs_unmasked_summary.csv", side_rows)
    write_report(args.review_dir / "masked_vs_unmasked_report.md", side_rows)
    manifest = {
        "updated_at_utc": utc_text(),
        "masked_run_root": str(args.masked_run_root),
        "unmasked_run_root": str(args.unmasked_run_root),
        "review_dir": str(args.review_dir),
        "cases": list(parse_cases(args.cases)),
        "rows": len(side_rows),
        "completed_masked_rows": sum(row.get("masked_status") == "completed" for row in side_rows),
        "completed_unmasked_rows": sum(row.get("unmasked_status") == "completed" for row in side_rows),
    }
    (args.review_dir / "masked_vs_unmasked_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
