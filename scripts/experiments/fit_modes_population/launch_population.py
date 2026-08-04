#!/usr/bin/env python
"""Launch the OASIS10/DIR-LAB10 population fit-modes experiment.

The launcher intentionally delegates optimization to the audited OASIS and
LungCT one-job runners.  Its responsibilities are cohort locking, exact reuse
of the current paper-figure records, four-GPU scheduling, resumability, and
durable progress/ETA reporting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "experiments/2026-08-02_fit_modes_population_n10"
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data"))
DEFAULT_OUT_ROOT = Path(
    os.environ.get(
        "RUN_ROOT", "/workspace/scratch/runs/fit_modes_population_n10_v1"
    )
)
DEFAULT_SMOKE_OUT_ROOT = Path(
    os.environ.get(
        "SMOKE_RUN_ROOT",
        "/workspace/scratch/runs/fit_modes_population_n10_smoke_v1",
    )
)
PAIR_FILE = REPO_ROOT / "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"
REUSE_CSV = (
    REPO_ROOT
    / "experiments/paper_artifacts/fitting_figure/v002_candidate/fit_modes_standardized_panelAB_selected_runs.csv"
)
DEFAULT_OASIS_ROOT = Path(
    os.environ.get("OASIS_ROOT", "/workspace/data/brain/oasis_organized/split_v1")
)
DEFAULT_SITREG_ROOT = Path(
    os.environ.get("SITREG_ROOT", "/workspace/data/brain/sitreg_targets")
)
DEFAULT_DIRLAB_DATA_ROOT = Path(
    os.environ.get(
        "DIRLAB_PTVREG_ISO1MM_ROOT",
        "/workspace/data/lungct/processed/pTVReg_iso1mm",
    )
)
DEFAULT_PTVREG_FIELD_ROOT = Path(
    os.environ.get("PTVREG_FIELD_ROOT", "/workspace/data/lungct/ptvreg_fields/4DCT")
)
ORIGINAL_FIRST10_INDICES = [158, 62, 14, 69, 47, 137, 110, 6, 116, 88]
VARIANT = "fit_modes_population_n10_v1"


HISTORICAL_SECONDS: dict[tuple[str, str, str], float] = {
    ("oasis", "A", "idir"): 195.0,
    ("oasis", "A", "inr_bscp"): 185.0,
    ("oasis", "A", "d_bscp"): 165.0,
    ("oasis", "A", "mr_d_bscp"): 210.0,
    ("oasis", "B", "idir"): 45.0,
    ("oasis", "B", "inr_bscp"): 125.0,
    ("oasis", "B", "d_bscp"): 95.0,
    ("oasis", "B", "mr_d_bscp"): 210.0,
    ("dirlab", "A", "idir"): 220.0,
    ("dirlab", "A", "inr_bscp"): 175.0,
    ("dirlab", "A", "d_bscp"): 135.0,
    ("dirlab", "A", "mr_d_bscp"): 285.0,
    ("dirlab", "B", "idir"): 20.0,
    ("dirlab", "B", "inr_bscp"): 135.0,
    ("dirlab", "B", "d_bscp"): 90.0,
    ("dirlab", "B", "mr_d_bscp"): 260.0,
}


@dataclass
class CaseSpec:
    dataset: str
    case_key: str
    case_rank: int | None = None
    original_case_index: int | None = None
    fixed_id: str = ""
    moving_id: str = ""
    case_id: int | None = None
    fixed_phase: str = ""
    moving_phase: str = ""
    target_ref: str = ""
    target_runtime_path: str = ""
    target_exists: bool = False
    target_size_bytes: int = 0
    target_sha256: str = ""


@dataclass
class Job:
    job_id: str
    dataset: str
    case_key: str
    panel: str
    objective: str
    direction: str
    target_ref: str
    target_units: str
    mask_rule: str
    method_slug: str
    display_method: str
    implementation_id: str
    reuse_method_id: str
    lr: float
    alpha: float | None
    alpha_role: str
    epochs: int
    cps: str
    stage_downsample: str
    stage_epochs: str
    seed: int
    diagnostic_interval: int
    output_schema: str
    compatibility_signature: str
    run_dir: str
    summary_path: str
    curve_path: str
    command: list[str]
    estimated_sec: float
    case_rank: int | None = None
    original_case_index: int | None = None
    fixed_id: str = ""
    moving_id: str = ""
    case_id: int | None = None
    status: str = "pending"
    source_kind: str = "new_run"
    source_path: str = ""
    reuse_validation: str = ""
    gpu: str = ""
    attempts: int = 0
    elapsed_sec: float | None = None
    returncode: int | None = None
    log_path: str = ""
    failure_note: str = ""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime | None = None) -> str:
    return (value or utc_now()).strftime("%Y-%m-%d %H:%M:%S +0000")


def format_float(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-targets", action="store_true")
    parser.add_argument("--hash-targets", action="store_true")
    parser.add_argument("--gpus", default="auto")
    parser.add_argument(
        "--container-image",
        default=os.environ.get("INRORSPLINE_DOCKER_IMAGE", "unspecified"),
        help="Portable image identifier recorded in runtime provenance.",
    )
    parser.add_argument("--max-concurrent", type=int, default=0)
    parser.add_argument("--poll-sec", type=int, default=15)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=PAIR_FILE)
    parser.add_argument("--sitreg-root", type=Path, default=DEFAULT_SITREG_ROOT)
    parser.add_argument("--dirlab-data-root", type=Path, default=DEFAULT_DIRLAB_DATA_ROOT)
    parser.add_argument("--ptvreg-field-root", type=Path, default=DEFAULT_PTVREG_FIELD_ROOT)
    parser.add_argument("--reuse-selected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-csv", type=Path, default=REUSE_CSV)
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()
    if args.poll_sec < 1 or args.poll_sec > 60:
        raise ValueError("--poll-sec must be between 1 and 60 seconds")
    if args.out_root is None:
        args.out_root = DEFAULT_SMOKE_OUT_ROOT if args.smoke else DEFAULT_OUT_ROOT
    if args.smoke:
        args.review_root = Path(args.review_root) / "smoke"
    return args


def parse_oasis_pairs(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        ids = re.findall(r"\d+", line)
        if len(ids) != 2:
            raise ValueError(f"Cannot parse OASIS pair line: {line!r}")
        rows.append((ids[0].zfill(4), ids[1].zfill(4)))
    if len(rows) < 10:
        raise RuntimeError(f"Expected at least 10 OASIS pairs in {path}, found {len(rows)}")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_cases(args: argparse.Namespace) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for rank, (fixed_id, moving_id) in enumerate(parse_oasis_pairs(Path(args.pair_file))[:10]):
        target = Path(args.sitreg_root) / f"disp_{fixed_id}_{moving_id}.nii.gz"
        cases.append(
            CaseSpec(
                dataset="oasis",
                case_key=f"rank{rank:03d}",
                case_rank=rank,
                original_case_index=ORIGINAL_FIRST10_INDICES[rank],
                fixed_id=fixed_id,
                moving_id=moving_id,
                target_ref=f"${{SITREG_ROOT}}/disp_{fixed_id}_{moving_id}.nii.gz",
                target_runtime_path=str(target),
                target_exists=target.is_file(),
                target_size_bytes=target.stat().st_size if target.is_file() else 0,
            )
        )
    for case_id in range(1, 11):
        target = Path(args.ptvreg_field_root) / f"case{case_id:02d}_T50_to_T00_deformation.mat"
        cases.append(
            CaseSpec(
                dataset="dirlab",
                case_key=f"case{case_id:02d}",
                case_id=case_id,
                fixed_phase="T00",
                moving_phase="T50",
                target_ref=f"${{PTVREG_FIELD_ROOT}}/case{case_id:02d}_T50_to_T00_deformation.mat",
                target_runtime_path=str(target),
                target_exists=target.is_file(),
                target_size_bytes=target.stat().st_size if target.is_file() else 0,
            )
        )
    if args.hash_targets:
        for case in cases:
            if case.target_exists:
                case.target_sha256 = file_sha256(Path(case.target_runtime_path))
    return cases


def oasis_method_specs(panel: str) -> list[dict[str, Any]]:
    if panel == "A":
        return [
            dict(slug="idir", display="INR-Dense", impl="idir_original", reuse="idir_original_dense_siren_be_autograd", lr=1e-4, alpha=1.0, cps="dense"),
            dict(slug="inr_bscp", display="INR-BSCP", impl="sinr_cps2_be_bspline_analytic", reuse="sinr_cps2_be_bspline_analytic", lr=1e-4, alpha=1000.0, cps="2"),
            dict(slug="d_bscp", display="D-BSCP", impl="direct_cp_cps2_be_bspline_analytic", reuse="direct_cp_cps2_be_bspline_analytic", lr=1e-3, alpha=1000.0, cps="2"),
            dict(slug="mr_d_bscp", display="MR-D-BSCP", impl="direct_cp_8_4_2_be_bspline_analytic_multistage", reuse="direct_cp_8_4_2_be_bspline_analytic_multistage", lr=1e-3, alpha=1000.0, cps="8,4,2"),
        ]
    return [
        dict(slug="idir", display="INR-Dense", impl="idir_original", reuse="IDIR", lr=1e-4, alpha=1.0, cps="dense"),
        dict(slug="inr_bscp", display="INR-BSCP", impl="sinr_cps2_be_bspline_analytic", reuse="SINR cps2", lr=1e-4, alpha=1000.0, cps="2"),
        dict(slug="d_bscp", display="D-BSCP", impl="direct_cp_cps2_be_bspline_analytic", reuse="DirectCP cps2", lr=1e-3, alpha=1000.0, cps="2"),
        dict(slug="mr_d_bscp", display="MR-D-BSCP", impl="direct_cp_8_4_2_be_bspline_analytic_multistage", reuse="DirectCP 8-4-2", lr=1e-3, alpha=1000.0, cps="8,4,2"),
    ]


def dirlab_method_specs(panel: str) -> list[dict[str, Any]]:
    if panel == "A":
        return [
            dict(slug="idir", display="INR-Dense", impl="idir_be_default_lungmask", reuse="idir_be_default_lungmask", lr=1e-4, alpha=3.0, cps="dense"),
            dict(slug="inr_bscp", display="INR-BSCP", impl="sinr_cps4_global_ncc_fd_be", reuse="sinr_cps4_global_ncc_fd_be", lr=1e-4, alpha=10000.0, cps="4"),
            dict(slug="d_bscp", display="D-BSCP", impl="direct_cp_cps4_global_ncc_fd_be", reuse="direct_cp_cps4_global_ncc_fd_be", lr=1e-3, alpha=10000.0, cps="4"),
            dict(slug="mr_d_bscp", display="MR-D-BSCP", impl="direct_cp_32_16_8_4_global_ncc_fd_be", reuse="direct_cp_32_16_8_4_global_ncc_fd_be", lr=1e-3, alpha=10000.0, cps="32,16,8,4"),
        ]
    return [
        dict(slug="idir", display="INR-Dense", impl="idir", reuse="IDIR", lr=1e-4, alpha=None, cps="dense"),
        dict(slug="inr_bscp", display="INR-BSCP", impl="sinr_cps4", reuse="SINR cps4", lr=1e-4, alpha=None, cps="4"),
        dict(slug="d_bscp", display="D-BSCP", impl="direct_cp_cps4", reuse="DirectCP cps4", lr=1e-3, alpha=None, cps="4"),
        dict(slug="mr_d_bscp", display="MR-D-BSCP", impl="direct_cp_4stage", reuse="DirectCP 4stage cps4", lr=1e-3, alpha=None, cps="32,16,8,4"),
    ]


def oasis_output_key(panel: str, impl: str) -> str:
    if panel == "A":
        return "idir_original_dense_siren_be_autograd" if impl == "idir_original" else impl
    return {
        "idir_original": "idir_original_dense_siren_field_mse",
        "sinr_cps2_be_bspline_analytic": "sinr_cps2_field_mse",
        "direct_cp_cps2_be_bspline_analytic": "direct_cp_cps2_field_mse",
        "direct_cp_8_4_2_be_bspline_analytic_multistage": "direct_cp_8_4_2_field_mse",
    }[impl]


def dirlab_output_key(panel: str, impl: str) -> str:
    if panel == "A":
        return impl
    return {
        "idir": "idir_ptvreg_field_mse",
        "sinr_cps4": "sinr_cps4_ptvreg_field_mse",
        "direct_cp_cps4": "direct_cp_cps4_ptvreg_field_mse",
        "direct_cp_4stage": "direct_cp_4stage_ptvreg_field_mse",
    }[impl]


def build_oasis_command(args: argparse.Namespace, case: CaseSpec, panel: str, spec: dict[str, Any], run_name: str, smoke: bool) -> list[str]:
    assert case.case_rank is not None and case.original_case_index is not None
    diagnostic = 1 if smoke else 10
    eval_interval = 1 if smoke else 250
    multistage = spec["slug"] == "mr_d_bscp"
    epochs = 3 if (smoke and multistage) else 2 if smoke else 2500
    out_root = Path(args.out_root) / "oasis" / ("panelA_registration" if panel == "A" else "panelB_sitreg_field_fit")
    if panel == "A":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/experiments/oasis_alpha_sweep/run_one.py"),
            "--method", spec["impl"], "--alpha", f"{spec['alpha']:g}", "--lr", f"{spec['lr']:g}",
            "--gpu", "__GPU__", "--case-rank", str(case.case_rank), "--epochs", str(epochs),
            "--eval-interval", str(eval_interval), "--global-ncc-diagnostic-interval", str(diagnostic),
            "--seed", "1", "--variant", VARIANT, "--out-root", str(out_root),
            "--oasis-root", str(args.oasis_root), "--pair-file", str(args.pair_file),
            "--expected-original-case-index", str(case.original_case_index),
            "--expected-fixed-id", case.fixed_id, "--expected-moving-id", case.moving_id,
            "--run-name", run_name,
        ]
        if multistage:
            cmd.extend(["--stage-full-cps", "8,4,2", "--stage-downsample", "4,2,1", "--stage-epochs", "1,1,1" if smoke else "400,800,1300"])
        if smoke:
            cmd.append("--no-save-final-dense")
        return cmd
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/experiments/oasis_alpha_sweep/run_rank009_sitreg_field_fit_one.py"),
        "--method", spec["impl"], "--alpha", f"{spec['alpha']:g}", "--lr", f"{spec['lr']:g}",
        "--gpu", "__GPU__", "--case-rank", str(case.case_rank), "--epochs", str(epochs),
        "--eval-interval", str(eval_interval), "--seed", "1", "--variant", VARIANT,
        "--out-root", str(out_root), "--oasis-root", str(args.oasis_root), "--pair-file", str(args.pair_file),
        "--sitreg-root", str(args.sitreg_root), "--expected-original-case-index", str(case.original_case_index),
        "--expected-fixed-id", case.fixed_id, "--expected-moving-id", case.moving_id, "--run-name", run_name,
    ]
    if multistage:
        cmd.extend([
            "--stage-full-cps", "8,4,2", "--stage-downsample", "4,2,1",
            "--stage-epochs", "1,1,1" if smoke else "400,800,1300",
            "--finest-mse-diagnostic", "--finest-mse-every", str(diagnostic),
        ])
    if smoke:
        cmd.append("--no-save-final-dense")
    return cmd


def build_dirlab_command(args: argparse.Namespace, case: CaseSpec, panel: str, spec: dict[str, Any], run_name: str, smoke: bool) -> list[str]:
    assert case.case_id is not None
    diagnostic = 1 if smoke else 10
    eval_interval = 1 if smoke else 250
    multistage = spec["slug"] == "mr_d_bscp"
    out_root = Path(args.out_root) / "dirlab" / ("panelA_registration" if panel == "A" else "panelB_ptvreg_field_fit")
    if panel == "A":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro/run_one.py"),
            "--method", spec["impl"], "--case-id", str(case.case_id), "--data-root", str(args.dirlab_data_root),
            "--out-root", str(out_root), "--run-name", run_name, "--gpu", "__GPU__", "--seed", "1",
            "--eval-interval", str(eval_interval), "--global-ncc-diagnostic-interval", str(diagnostic),
        ]
        if spec["slug"] == "idir":
            cmd.extend(["--epochs", "2" if smoke else "2500", "--lr", "1e-4", "--alpha-bending", "3"])
        elif not multistage:
            cmd.extend(["--epochs", "2" if smoke else "2500", "--lr", f"{spec['lr']:g}", "--alpha", "10000"])
        else:
            cmd.extend([
                "--lr", "1e-3", "--alpha", "10000", "--stage-full-cps", "32,16,8,4",
                "--stage-downsample", "8,4,2,1", "--stage-epochs", "1,1,1,1" if smoke else "200,400,800,1000",
            ])
        if smoke:
            cmd.append("--no-save-dense-dvf")
        return cmd
    epochs = 4 if (smoke and multistage) else 2 if smoke else 2500
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro/run_case01_ptvreg_field_fit_one.py"),
        "--method", spec["impl"], "--case-id", str(case.case_id), "--data-root", str(args.dirlab_data_root),
        "--ptvreg-mat", str(Path(args.ptvreg_field_root) / f"case{case.case_id:02d}_T50_to_T00_deformation.mat"),
        "--out-root", str(out_root), "--run-name", run_name, "--gpu", "__GPU__", "--seed", "1",
        "--epochs", str(epochs), "--lr", f"{spec['lr']:g}", "--eval-interval", str(eval_interval),
    ]
    if multistage:
        cmd.extend([
            "--stage-full-cps", "32,16,8,4", "--stage-downsample", "8,4,2,1",
            "--stage-epochs", "1,1,1,1" if smoke else "400,400,800,900",
            "--finest-mse-diagnostic", "--finest-mse-every", "1",
        ])
    if smoke:
        cmd.append("--no-save-dense-dvf")
    return cmd


def build_jobs(args: argparse.Namespace, cases: Sequence[CaseSpec]) -> list[Job]:
    smoke_cases = {("oasis", "rank000"), ("dirlab", "case02")}
    selected_cases = [case for case in cases if not args.smoke or (case.dataset, case.case_key) in smoke_cases]
    jobs: list[Job] = []
    for case in selected_cases:
        for panel in ("A", "B"):
            specs = oasis_method_specs(panel) if case.dataset == "oasis" else dirlab_method_specs(panel)
            for spec in specs:
                multistage = spec["slug"] == "mr_d_bscp"
                if args.smoke:
                    epochs = 3 if case.dataset == "oasis" and multistage else 4 if case.dataset == "dirlab" and multistage else 2
                    stage_epochs = "1,1,1" if case.dataset == "oasis" and multistage else "1,1,1,1" if multistage else ""
                else:
                    epochs = 2400 if case.dataset == "dirlab" and panel == "A" and multistage else 2500
                    stage_epochs = (
                        "400,800,1300" if case.dataset == "oasis" and multistage
                        else "200,400,800,1000" if case.dataset == "dirlab" and panel == "A" and multistage
                        else "400,400,800,900" if case.dataset == "dirlab" and panel == "B" and multistage
                        else ""
                    )
                run_name = (
                    f"population_{case.dataset}_{case.case_key}_panel{panel}_{spec['slug']}"
                    f"_lr{format_float(spec['lr'])}_seed1"
                )
                if case.dataset == "oasis":
                    command = build_oasis_command(args, case, panel, spec, run_name, args.smoke)
                    out_root = Path(args.out_root) / "oasis" / ("panelA_registration" if panel == "A" else "panelB_sitreg_field_fit")
                    run_dir = out_root / oasis_output_key(panel, spec["impl"]) / run_name
                else:
                    command = build_dirlab_command(args, case, panel, spec, run_name, args.smoke)
                    out_root = Path(args.out_root) / "dirlab" / ("panelA_registration" if panel == "A" else "panelB_ptvreg_field_fit")
                    run_dir = out_root / dirlab_output_key(panel, spec["impl"]) / run_name
                curve_name = "global_ncc_diagnostic.csv" if panel == "A" else "stage_history.csv" if multistage else "history.csv"
                if args.smoke:
                    diagnostic_interval = 1
                elif panel == "A":
                    diagnostic_interval = 10
                elif not multistage or case.dataset == "dirlab":
                    diagnostic_interval = 1
                else:
                    diagnostic_interval = 10
                direction = (
                    f"moving_{case.moving_id}_to_fixed_{case.fixed_id}"
                    if case.dataset == "oasis"
                    else "moving_T50_to_fixed_T00"
                )
                if panel == "B":
                    mask_rule = "fixed_image_positive" if case.dataset == "oasis" else "fixed_t00_lung_mask_reg1mm"
                elif case.dataset == "oasis" and spec["slug"] == "idir":
                    mask_rule = "training_fixed_image_positive;diagnostic_full_image"
                elif case.dataset == "oasis":
                    mask_rule = "training_full_image;diagnostic_full_image;regularity_fixed_image_positive"
                elif spec["slug"] == "idir":
                    mask_rule = "training_fixed_t00_lung_mask;diagnostic_full_1mm_crop"
                else:
                    mask_rule = "training_full_1mm_crop;diagnostic_full_1mm_crop;regularity_fixed_t00_lung_mask"
                compatibility_payload = {
                    "dataset": case.dataset,
                    "case_key": case.case_key,
                    "direction": direction,
                    "target_ref": case.target_ref if panel == "B" else "image_pair",
                    "objective": "image_registration_global_ncc" if panel == "A" else "field_fit_mse",
                    "target_units": "not_applicable_image_pair" if panel == "A" else "normalized_grid_displacement",
                    "mask_rule": mask_rule,
                    "implementation_id": spec["impl"],
                    "cps": spec["cps"],
                    "stage_downsample": "4,2,1" if case.dataset == "oasis" and multistage else "8,4,2,1" if multistage else "",
                    "stage_epochs": stage_epochs,
                    "lr": float(spec["lr"]),
                    "alpha": spec["alpha"],
                    "alpha_role": "training_regularization_weight" if panel == "A" else "recorded_not_used_mse_only",
                    "seed": 1,
                    "epochs": epochs,
                    "diagnostic_interval": diagnostic_interval,
                    "output_schema": "fit_modes_population_runner_outputs_v1",
                }
                compatibility_signature = hashlib.sha256(
                    json.dumps(compatibility_payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                jobs.append(
                    Job(
                        job_id=f"{case.dataset}_{case.case_key}_panel{panel}_{spec['slug']}",
                        dataset=case.dataset,
                        case_key=case.case_key,
                        panel=panel,
                        objective="image_registration_global_ncc" if panel == "A" else "field_fit_mse",
                        direction=direction,
                        target_ref=case.target_ref if panel == "B" else "image_pair",
                        target_units="not_applicable_image_pair" if panel == "A" else "normalized_grid_displacement",
                        mask_rule=mask_rule,
                        method_slug=spec["slug"], display_method=spec["display"], implementation_id=spec["impl"],
                        reuse_method_id=spec["reuse"], lr=float(spec["lr"]), alpha=spec["alpha"],
                        alpha_role="training_regularization_weight" if panel == "A" else "recorded_not_used_mse_only",
                        epochs=epochs, cps=spec["cps"],
                        stage_downsample=("4,2,1" if case.dataset == "oasis" and multistage else "8,4,2,1" if multistage else ""),
                        stage_epochs=stage_epochs, seed=1, diagnostic_interval=diagnostic_interval,
                        output_schema="fit_modes_population_runner_outputs_v1",
                        compatibility_signature=compatibility_signature,
                        run_dir=str(run_dir), summary_path=str(run_dir / "summary.json"), curve_path=str(run_dir / curve_name),
                        command=command,
                        estimated_sec=HISTORICAL_SECONDS[(case.dataset, panel, spec["slug"])] * (0.18 if args.smoke else 1.0),
                        case_rank=case.case_rank, original_case_index=case.original_case_index,
                        fixed_id=case.fixed_id, moving_id=case.moving_id, case_id=case.case_id,
                    )
                )
    expected = 16 if args.smoke else 160
    if len(jobs) != expected or len({job.job_id for job in jobs}) != expected:
        raise RuntimeError(f"Job matrix validation failed: expected {expected}, got {len(jobs)}")
    return jobs


def job_is_complete(job: Job) -> bool:
    summary_path = Path(job.summary_path)
    curve_path = Path(job.curve_path)
    if not summary_path.is_file() or not curve_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return False
    if job.dataset == "dirlab" and job.panel == "A":
        return summary.get("status") == "completed"
    return (Path(job.run_dir) / "DONE").is_file() or summary.get("status") == "completed"


def finite_csv_source(path: Path) -> bool:
    candidates = [path]
    if path.is_dir():
        candidates = [path / "history.csv", path / "stage_history.csv", path / "global_ncc_diagnostic.csv"]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open(newline="") as handle:
                reader = csv.DictReader(handle)
                row = next(reader, None)
            if row and any(math.isfinite(float(value)) for value in row.values() if value not in (None, "", "True", "False") and re.fullmatch(r"[-+0-9.eE]+", str(value))):
                return True
        except Exception:
            continue
    return False


def reuse_case_label(job: Job) -> str:
    if job.dataset == "oasis":
        return "OASIS rank-9"
    return f"LungCT case{int(job.case_id or 0):02d}"


def load_reuse_catalog(args: argparse.Namespace, jobs: Sequence[Job]) -> dict[str, tuple[str, str]]:
    if args.smoke or not args.reuse_selected:
        return {}
    wanted = {job.job_id: job for job in jobs if (job.dataset == "oasis" and job.case_rank == 9) or (job.dataset == "dirlab" and job.case_id in {1, 8})}
    catalog: dict[str, tuple[str, str]] = {}
    with Path(args.reuse_csv).open(newline="") as handle:
        for row in csv.DictReader(handle):
            panel = str(row["panel"])
            for job_id, job in wanted.items():
                if panel != job.panel or str(row["plot_case"]) != reuse_case_label(job):
                    continue
                if str(row["method_id"]) != job.reuse_method_id:
                    continue
                if abs(float(row["lr"]) - job.lr) > 1e-12 or str(row["available"]).lower() != "true":
                    continue
                source = str(row["source"]).replace("${DATA_ROOT}", str(args.data_root))
                source_path = Path(source)
                text_blob = " ".join([str(row.get("method_id", "")), str(row.get("note", "")), source])
                spacing_ok = True
                if job.method_slug in {"inr_bscp", "d_bscp"}:
                    spacing_ok = f"cps{job.cps}" in text_blob.lower()
                elif job.method_slug == "mr_d_bscp":
                    spacing_ok = ("8-4-2" in text_blob or "8_4_2" in text_blob) if job.dataset == "oasis" else ("32-16-8-4" in text_blob or "321684" in text_blob)
                if not source_path.exists() or not finite_csv_source(source_path) or not spacing_ok:
                    continue
                validation = (
                    "accepted_current_v002_selected_run: case/panel/method/lr/final-spacing/source-exists/finite-curve"
                )
                catalog[job_id] = (source, validation)
    expected = set(wanted)
    missing = sorted(expected - set(catalog))
    if missing:
        raise RuntimeError(f"Exact selected-run reuse validation failed for: {missing}")
    return catalog


def initialize_statuses(args: argparse.Namespace, jobs: Sequence[Job], reuse: dict[str, tuple[str, str]]) -> None:
    partial: list[str] = []
    for job in jobs:
        if job_is_complete(job):
            job.status = "completed_existing"
            job.source_kind = "population_run_resume"
            job.source_path = job.curve_path
            continue
        if Path(job.run_dir).exists():
            partial.append(job.job_id)
        if job.job_id in reuse:
            job.status = "reused"
            job.source_kind = "exact_compatible_current_figure"
            job.source_path, job.reuse_validation = reuse[job.job_id]
    if partial and not args.resume:
        raise RuntimeError(
            "Partial population run directories exist; rerun with --resume after inspecting them: "
            + ", ".join(partial[:10])
        )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cooked = {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
            writer.writerow(cooked)


def command_for_gpu(job: Job, gpu: str) -> list[str]:
    return [gpu if token == "__GPU__" else token for token in job.command]


def git_metadata() -> dict[str, Any]:
    def run(*tokens: str) -> str:
        result = subprocess.run(tokens, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        return result.stdout.strip()
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty_paths": run("git", "status", "--short").splitlines(),
    }


def sanitize_for_review(value: Any, args: argparse.Namespace) -> Any:
    replacements = [
        (str(args.out_root), "${RUN_ROOT}"),
        (str(args.oasis_root), "${OASIS_ROOT}"),
        (str(args.sitreg_root), "${SITREG_ROOT}"),
        (str(args.dirlab_data_root), "${DIRLAB_PTVREG_ISO1MM_ROOT}"),
        (str(args.ptvreg_field_root), "${PTVREG_FIELD_ROOT}"),
        (str(args.data_root), "${DATA_ROOT}"),
        (str(REPO_ROOT), "${REPO_ROOT}"),
    ]
    if isinstance(value, dict):
        return {key: sanitize_for_review(item, args) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_review(item, args) for item in value]
    if isinstance(value, str):
        for source, replacement in replacements:
            value = value.replace(source, replacement)
    return value


def write_manifests(args: argparse.Namespace, cases: Sequence[CaseSpec], jobs: Sequence[Job]) -> None:
    review = Path(args.review_root)
    review.mkdir(parents=True, exist_ok=True)
    if cases:
        safe_cases: list[dict[str, Any]] = []
        for case in cases:
            row = asdict(case)
            row.pop("target_runtime_path", None)
            safe_cases.append(row)
        write_csv(review / "cohort_manifest.csv", safe_cases)
    job_rows = [sanitize_for_review(asdict(job), args) for job in jobs]
    write_csv(review / "job_manifest.csv", job_rows)
    (review / "job_manifest.json").write_text(json.dumps(job_rows, indent=2, allow_nan=False) + "\n")
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1
    launcher_state = sanitize_for_review({
        "updated_at": utc_text(),
        "mode": "smoke" if args.smoke else "full",
        "out_root": str(args.out_root),
        "status_counts": counts,
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status,
                "gpu": job.gpu,
                "attempts": job.attempts,
                "elapsed_sec": job.elapsed_sec,
                "returncode": job.returncode,
                "log_path": job.log_path,
                "source_kind": job.source_kind,
            }
            for job in jobs
        ],
    }, args)
    (review / "launcher_state.json").write_text(json.dumps(launcher_state, indent=2, allow_nan=False) + "\n")
    runtime_manifest = {
        "generated_at": utc_text(), "mode": "smoke" if args.smoke else "full",
        "out_root": str(args.out_root), "review_root": str(args.review_root),
        "container_python": sys.executable,
        "container_hostname": os.environ.get("HOSTNAME", ""),
        "declared_container_image": str(args.container_image),
        "git": git_metadata(),
        "pair_file": str(args.pair_file),
        "pair_file_sha256": file_sha256(Path(args.pair_file)),
        "environment_variables": {
            "REPO_ROOT": str(REPO_ROOT),
            "DATA_ROOT": str(args.data_root),
            "RUN_ROOT": str(args.out_root),
            "OASIS_ROOT": str(args.oasis_root),
            "SITREG_ROOT": str(args.sitreg_root),
            "DIRLAB_PTVREG_ISO1MM_ROOT": str(args.dirlab_data_root),
            "PTVREG_FIELD_ROOT": str(args.ptvreg_field_root),
        },
        "target_runtime_paths": {case.case_key: case.target_runtime_path for case in cases},
        "target_sha256": {case.case_key: case.target_sha256 for case in cases},
    }
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    runtime_manifest_path = Path(args.out_root) / "runtime_manifest.json"
    if cases or not runtime_manifest_path.exists():
        runtime_manifest_path.write_text(json.dumps(runtime_manifest, indent=2) + "\n")


def discover_gpus(text: str) -> list[str]:
    if text != "auto":
        gpus = [token.strip() for token in text.split(",") if token.strip()]
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True, capture_output=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"GPU discovery failed: {result.stderr.strip()}")
        gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(gpus) < 4:
        raise RuntimeError(f"The approved launch requires four visible GPUs; found {gpus}")
    return gpus[:4]


def observed_estimates(jobs: Sequence[Job]) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for job in jobs:
        if job.status == "completed" and job.elapsed_sec and math.isfinite(job.elapsed_sec):
            grouped.setdefault((job.dataset, job.panel, job.method_slug), []).append(float(job.elapsed_sec))
    return {key: statistics.median(values) for key, values in grouped.items()}


def progress_payload(jobs: Sequence[Job], gpus: Sequence[str], started: float) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1
    observed = observed_estimates(jobs)
    remaining_work = 0.0
    for job in jobs:
        if job.status not in {"pending", "running"}:
            continue
        estimate = observed.get((job.dataset, job.panel, job.method_slug), job.estimated_sec)
        if job.status == "running" and job.elapsed_sec:
            estimate = max(estimate - float(job.elapsed_sec), estimate * 0.1)
        remaining_work += estimate
    eta_sec = remaining_work / max(len(gpus), 1)
    finished_new = sum(1 for job in jobs if job.status == "completed")
    active = {
        job.gpu: {"job_id": job.job_id, "elapsed_sec": job.elapsed_sec, "log_path": job.log_path}
        for job in jobs if job.status == "running"
    }
    return {
        "updated_at": utc_text(),
        "elapsed_sec": time.time() - started,
        "counts": counts,
        "total": len(jobs),
        "active_gpus": active,
        "gpus": list(gpus),
        "eta_provisional": finished_new < 8,
        "eta_seconds": eta_sec,
        "estimated_completion_utc": utc_text(utc_now() + timedelta(seconds=eta_sec)),
        "observed_runtime_groups": {"|".join(key): value for key, value in observed.items()},
        "recent_failures": [
            {"job_id": job.job_id, "returncode": job.returncode, "log_path": job.log_path, "note": job.failure_note}
            for job in jobs if job.status == "failed"
        ][-10:],
    }


def duration_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def write_monitor(args: argparse.Namespace, jobs: Sequence[Job], gpus: Sequence[str], started: float) -> dict[str, Any]:
    payload = sanitize_for_review(progress_payload(jobs, gpus, started), args)
    review = Path(args.review_root)
    (review / "monitor_status.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    counts = payload["counts"]
    lines = [
        "# Population Fit-Modes Monitor", "", f"Updated: `{payload['updated_at']}`",
        f"Elapsed: `{duration_text(payload['elapsed_sec'])}`",
        f"ETA: `{'provisional; ' if payload['eta_provisional'] else ''}{duration_text(payload['eta_seconds'])}`",
        f"Estimated completion: `{payload['estimated_completion_utc']}`", "",
        "| Status | Count |", "|---|---:|",
    ]
    for status in ["reused", "completed_existing", "completed", "running", "pending", "failed"]:
        lines.append(f"| {status} | {counts.get(status, 0)} |")
    lines.extend(["", "## Active GPUs", "", "| GPU | Job | Elapsed |", "|---:|---|---:|"])
    for gpu in gpus:
        item = payload["active_gpus"].get(gpu)
        lines.append(f"| {gpu} | {item['job_id'] if item else 'idle'} | {duration_text(item['elapsed_sec'] or 0) if item else ''} |")
    if payload["recent_failures"]:
        lines.extend(["", "## Recent Failures", ""])
        for item in payload["recent_failures"]:
            lines.append(f"- `{item['job_id']}` rc={item['returncode']} log=`{item['log_path']}`")
    (review / "monitor_status.md").write_text("\n".join(lines) + "\n")
    return payload


def run_scheduler(args: argparse.Namespace, jobs: list[Job], gpus: list[str]) -> int:
    pending = sorted([job for job in jobs if job.status == "pending"], key=lambda item: item.estimated_sec, reverse=True)
    running: dict[str, tuple[Job, subprocess.Popen[str], Any, float]] = {}
    max_concurrent = min(len(gpus), int(args.max_concurrent) if int(args.max_concurrent) > 0 else len(gpus))
    active_gpus = gpus[:max_concurrent]
    started = time.time()
    log_root = Path(args.out_root) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    write_monitor(args, jobs, active_gpus, started)
    while pending or running:
        free = [gpu for gpu in active_gpus if gpu not in running]
        while free and pending:
            gpu = free.pop(0)
            job = pending.pop(0)
            job.gpu = gpu
            job.attempts += 1
            job.status = "running"
            job.elapsed_sec = 0.0
            log_path = log_root / f"{job.job_id}_gpu{gpu}.log"
            job.log_path = str(log_path)
            log_handle = log_path.open("w")
            command = command_for_gpu(job, gpu)
            log_handle.write(json.dumps({"started_at": utc_text(), "job_id": job.job_id, "gpu": gpu, "command": command}) + "\n")
            log_handle.flush()
            proc = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            running[gpu] = (job, proc, log_handle, time.time())
            print(json.dumps({"started": job.job_id, "gpu": gpu, "pending": len(pending)}), flush=True)
        time.sleep(min(args.poll_sec, 60))
        for gpu, (job, proc, log_handle, job_started) in list(running.items()):
            job.elapsed_sec = time.time() - job_started
            returncode = proc.poll()
            if returncode is None:
                continue
            log_handle.flush()
            log_handle.close()
            job.returncode = int(returncode)
            if returncode == 0 and job_is_complete(job):
                job.status = "completed"
                job.source_kind = "population_run"
                job.source_path = job.curve_path
                print(json.dumps({"completed": job.job_id, "gpu": gpu, "elapsed_sec": job.elapsed_sec}), flush=True)
            else:
                job.status = "failed"
                job.failure_note = "nonzero return code" if returncode != 0 else "expected outputs missing"
                print(json.dumps({"failed": job.job_id, "gpu": gpu, "returncode": returncode, "log": job.log_path}), flush=True)
            del running[gpu]
        write_manifests(args, [], jobs)
        payload = write_monitor(args, jobs, active_gpus, started)
        print(json.dumps({"progress": payload}), flush=True)
    return 1 if any(job.status == "failed" for job in jobs) else 0


def validate_targets(cases: Sequence[CaseSpec]) -> None:
    missing = [case.target_ref for case in cases if not case.target_exists]
    if missing:
        raise RuntimeError(f"Missing target fields: {missing}")
    counts = {dataset: sum(case.dataset == dataset for case in cases) for dataset in ("oasis", "dirlab")}
    if counts != {"oasis": 10, "dirlab": 10}:
        raise RuntimeError(f"Target cohort count mismatch: {counts}")


def extract_last_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    parsed: list[tuple[int, int, dict[str, Any]]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append((end, index, value))
    if not parsed:
        raise ValueError("validation command produced no JSON object")
    return max(parsed, key=lambda item: (item[0], item[1]))[2]


def sanitize_validation(value: Any, args: argparse.Namespace) -> Any:
    return sanitize_for_review(value, args)


def deep_validate_targets(args: argparse.Namespace, cases: Sequence[CaseSpec]) -> None:
    """Load every field with the audited runner and persist sanitized audits."""
    gpus = discover_gpus(args.gpus)
    work: list[tuple[int, CaseSpec, list[str]]] = []
    for index, case in enumerate(cases):
        if case.dataset == "oasis":
            assert case.case_rank is not None and case.original_case_index is not None
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/experiments/oasis_alpha_sweep/run_rank009_sitreg_field_fit_one.py"),
                "--method", "idir_original", "--alpha", "1", "--lr", "1e-4",
                "--gpu", gpus[index % len(gpus)], "--case-rank", str(case.case_rank),
                "--validate-target-only", "--out-root", str(Path(args.out_root) / "target_validation_no_outputs"),
                "--oasis-root", str(args.oasis_root), "--pair-file", str(args.pair_file),
                "--sitreg-root", str(args.sitreg_root),
                "--expected-original-case-index", str(case.original_case_index),
                "--expected-fixed-id", case.fixed_id, "--expected-moving-id", case.moving_id,
            ]
        else:
            assert case.case_id is not None
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro/run_case01_ptvreg_field_fit_one.py"),
                "--method", "idir", "--case-id", str(case.case_id), "--lr", "1e-4",
                "--data-root", str(args.dirlab_data_root),
                "--ptvreg-mat", str(Path(args.ptvreg_field_root) / f"case{case.case_id:02d}_T50_to_T00_deformation.mat"),
                "--validate-target-only", "--out-root", str(Path(args.out_root) / "target_validation_no_outputs"),
            ]
        work.append((index, case, command))

    def run_one(item: tuple[int, CaseSpec, list[str]]) -> tuple[int, dict[str, Any]]:
        index, case, command = item
        started = time.time()
        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        record: dict[str, Any] = {
            "dataset": case.dataset,
            "case_key": case.case_key,
            "target_ref": case.target_ref,
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "elapsed_sec": time.time() - started,
        }
        if result.returncode == 0:
            try:
                record["audit"] = sanitize_validation(extract_last_json(result.stdout), args)
            except Exception as error:
                record["status"] = "failed"
                record["parse_error"] = str(error)
        else:
            record["stderr_tail"] = sanitize_validation(result.stderr[-2000:], args)
            record["stdout_tail"] = sanitize_validation(result.stdout[-2000:], args)
        return index, record

    ordered: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_one, item) for item in work]
        for future in as_completed(futures):
            index, record = future.result()
            ordered[index] = record
            print(json.dumps({key: record[key] for key in ("dataset", "case_key", "status", "elapsed_sec")}), flush=True)
    records = [ordered[index] for index in range(len(work))]
    review = Path(args.review_root)
    review.mkdir(parents=True, exist_ok=True)
    (review / "target_validation.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    write_csv(
        review / "target_validation.csv",
        [
            {
                "dataset": record["dataset"],
                "case_key": record["case_key"],
                "target_ref": record["target_ref"],
                "status": record["status"],
                "returncode": record["returncode"],
                "elapsed_sec": record["elapsed_sec"],
            }
            for record in records
        ],
    )
    failed = [record["case_key"] for record in records if record["status"] != "passed"]
    if failed:
        raise RuntimeError(f"Deep target validation failed: {failed}")


def main() -> None:
    args = parse_args()
    cases = build_cases(args)
    if args.validate_targets or args.dry_run or args.smoke:
        validate_targets(cases)
    if args.validate_targets:
        deep_validate_targets(args, cases)
    jobs = build_jobs(args, cases)
    reuse = load_reuse_catalog(args, jobs)
    initialize_statuses(args, jobs, reuse)
    write_manifests(args, cases, jobs)
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1
    summary = {
        "mode": "smoke" if args.smoke else "full", "jobs": len(jobs), "status_counts": counts,
        "targets": {"oasis": 10, "dirlab": 10}, "out_root": str(args.out_root),
        "review_root": str(args.review_root),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return
    gpus = discover_gpus(args.gpus)
    returncode = run_scheduler(args, jobs, gpus)
    if not args.skip_aggregate:
        aggregate_cmd = [
            sys.executable,
            str(Path(__file__).with_name("aggregate_population.py")),
            "--review-root", str(args.review_root),
            "--run-root", str(args.out_root),
            "--data-root", str(args.data_root),
        ]
        aggregate = subprocess.run(aggregate_cmd, cwd=REPO_ROOT, check=False)
        returncode = returncode or aggregate.returncode
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
