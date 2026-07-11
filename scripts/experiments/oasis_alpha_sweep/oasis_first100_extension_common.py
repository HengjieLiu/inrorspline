"""Shared constants for the OASIS seed-3 first100 extension."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import common as c


REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-07-04_oasis_seed3_first100_extension"
TRACKER_ROOT = REVIEW_ROOT / "tracker"
PROGRESS_LOG = REVIEW_ROOT / "progress_log.md"
TRACKER_FIGURE = TRACKER_ROOT / "oasis_first100_progress_tracker.png"
FIRST100_ARTIFACT_ROOT = REVIEW_ROOT / "first100_artifacts"

FIRST10_MAIN_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1")
FIRST10_CPS2_PRELIM_ROOT = Path(
    "/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1"
)
FIRST10_CPS2_COMPLETION_ROOT = Path(
    "/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_completion_a0_10_100000_v1"
)
NEXT10_ROOT = Path("/database/inrorspline/runs/oasis_seed3_next10_alpha_sweep_be_2500e_5models_v1")
FIRST20_BASELINE_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first10_saved_baselines_v1")
FIRST20_DCP842_ROOT = Path("/database/inrorspline/runs/oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_v1")

BASELINE_EXTENSION_VARIANT = "oasis_seed3_ranks020_099_saved_baselines_v1"
SINGLE_STAGE_EXTENSION_VARIANT = "oasis_seed3_ranks020_099_alpha_sweep_be_2500e_5models_v1"
DCP842_EXTENSION_VARIANT = "oasis_seed3_ranks020_099_directcp842_lr1e3_alpha_sweep_v1"

BASELINE_EXTENSION_ROOT = Path(f"/database/inrorspline/runs/{BASELINE_EXTENSION_VARIANT}")
SINGLE_STAGE_EXTENSION_ROOT = Path(f"/database/inrorspline/runs/{SINGLE_STAGE_EXTENSION_VARIANT}")
DCP842_EXTENSION_ROOT = Path(f"/database/inrorspline/runs/{DCP842_EXTENSION_VARIANT}")

RANKS_FIRST20 = list(range(20))
RANKS_EXTENSION = list(range(20, 100))
RANKS_FIRST100 = list(range(100))

IDIR_ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
SPLINE_ALPHAS = [0.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

DCP842_METHOD = "direct_cp_8_4_2_be_bspline_analytic_multistage"
DCP842_LABEL = "MR-D-BSCP (8-4-2)"
DCP842_STAGE_FULL_CPS = "8,4,2"
DCP842_STAGE_DOWNSAMPLE = "4,2,1"
DCP842_STAGE_EPOCHS = "400,800,1300"
DCP842_LR = 1e-3

BASELINE_METHODS = [
    "bs1_vxm_v0_2",
    "bs2_tm_v0",
    "bs3_vfa_v0",
    "bs4_sitreg_v0",
    "greedy",
]


@dataclass(frozen=True)
class MethodSpec:
    method: str
    method_key: str
    display_name: str
    lr: float
    alphas: Sequence[float]


SINGLE_STAGE_METHOD_SPECS = [
    MethodSpec(
        "direct_cp_cps2_be_bspline_analytic",
        "direct_cp_cps2_be_bspline_analytic",
        "D-BSCP (cps2)",
        1e-3,
        SPLINE_ALPHAS,
    ),
    MethodSpec(
        "sinr_cps2_be_bspline_analytic",
        "sinr_cps2_be_bspline_analytic",
        "INR-BSCP (cps2)",
        1e-4,
        SPLINE_ALPHAS,
    ),
    MethodSpec(
        "idir_original",
        "idir_original_dense_siren_be_autograd",
        "INR-Dense",
        1e-4,
        IDIR_ALPHAS,
    ),
    MethodSpec(
        "direct_cp_cps4_be_bspline_analytic",
        "direct_cp_cps4_be_bspline_analytic",
        "D-BSCP (cps4)",
        1e-3,
        SPLINE_ALPHAS,
    ),
    MethodSpec(
        "sinr_cps4_be_bspline_analytic",
        "sinr_cps4_be_bspline_analytic",
        "INR-BSCP (cps4)",
        1e-4,
        SPLINE_ALPHAS,
    ),
]


def parse_int_ranges(text: str) -> List[int]:
    values: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start = int(start_text)
            stop = int(stop_text)
            step = 1 if stop >= start else -1
            values.extend(range(start, stop + step, step))
        else:
            values.append(int(token))
    out: List[int] = []
    for value in values:
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError(f"No integer ranks parsed from {text!r}")
    return out


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            value = float(token)
            if value not in values:
                values.append(value)
    if not values:
        raise ValueError(f"No floats parsed from {text!r}")
    return values


def resolve_gpus(text: str) -> List[str]:
    requested = str(text).strip().lower()
    if requested not in {"auto", "all"}:
        gpus = [token.strip() for token in str(text).split(",") if token.strip()]
        if not gpus:
            raise ValueError(f"No GPUs parsed from {text!r}")
        return gpus
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not detect GPUs with nvidia-smi; pass --gpus explicitly.") from exc
    gpus = [line.strip() for line in raw.splitlines() if line.strip()]
    if not gpus:
        raise RuntimeError("nvidia-smi did not report any visible GPUs.")
    return gpus


def thread_env(cpu_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, int(cpu_threads)))
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ]:
        env[key] = threads
    return env


def method_specs_by_key(specs: Sequence[MethodSpec]) -> Dict[str, MethodSpec]:
    return {spec.method_key: spec for spec in specs}

