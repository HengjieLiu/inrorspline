#!/usr/bin/env python
"""Run ``ptvreg-python`` on one COPD case/protocol variant."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists():
            return parent
    raise RuntimeError("Could not find repository root.")


REPO_ROOT = repo_root()
THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
LUNG_COMMON_PATH = REPO_ROOT / "scripts" / "experiments" / "lungct_no_sweep_repro" / "common.py"
DATA_LUNGCT_PATH = REPO_ROOT / "scripts" / "data" / "common" / "lungct.py"
PTVREG_PYTHON_SRC = REPO_ROOT / "external" / "ptvreg-python" / "src"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths

METHOD = "ptvreg_python_admm"
PROTOCOL_CANONICAL = "ptvreg_python_admm+canonical_ptv_landmark_crop_1mm_no_lungmask"
PROTOCOL_PAPER_HU = "ptvreg_python_admm+paper_hu_landmark_crop_1mm_no_lungmask"
PROTOCOL_PAPER_HU_NATIVE = "ptvreg_python_admm+paper_hu_native_landmark_crop_no_lungmask"
PROTOCOL_CHOICES = ("canonical_1mm", "paper_hu_iso1mm", "paper_hu_native")
DEFAULT_CASES = tuple(range(1, 11))
MARGIN_XYZ = np.asarray([10, 10, 5], dtype=np.int64)
MARGIN_ZYX = MARGIN_XYZ[::-1].copy()
MASK_MODEL_NAME = "R231"
RUN_KEY = "ptvreg_python_copd_fixed_all10_v1"


def configured_default_run_root() -> Path:
    """Return an import-safe display fallback; runtime resolution is strict."""

    configured = os.environ.get("INRORSPLINE_PTVREG_PYTHON_RUN_ROOT")
    if configured:
        return Path(configured)
    artifact_root = os.environ.get(artifact_paths.ARTIFACT_ROOT_ENV)
    if artifact_root:
        return Path(artifact_root) / artifact_paths.RUNS_DIRNAME / RUN_KEY
    return Path("/database/inrorspline") / artifact_paths.RUNS_DIRNAME / RUN_KEY


# Kept for import compatibility.  Actual launches call ``resolve_run_root``
# and therefore require ARTIFACT_ROOT instead of trusting this fallback.
DEFAULT_RUN_ROOT = configured_default_run_root()
DEFAULT_PAPER_HU_ISO1MM_ROOT = Path(
    os.environ.get(
        "INRORSPLINE_PTVREG_PYTHON_PAPER_HU_ROOT",
        str(DEFAULT_RUN_ROOT / "_paper_hu_iso1mm_inputs"),
    )
)
DEFAULT_PAPER_HU_NATIVE_ROOT = Path(
    os.environ.get(
        "INRORSPLINE_PTVREG_PYTHON_PAPER_HU_NATIVE_ROOT",
        str(DEFAULT_RUN_ROOT / "_paper_hu_native_inputs"),
    )
)
DEFAULT_PAPER_HU_ROOT = DEFAULT_PAPER_HU_ISO1MM_ROOT


def resolve_run_root(requested_run_root: Optional[Path] = None) -> Path:
    """Resolve the ptvreg run root under the configured external artifact store."""

    requested = requested_run_root
    if requested is None:
        configured = os.environ.get("INRORSPLINE_PTVREG_PYTHON_RUN_ROOT")
        requested = Path(configured) if configured else None
    return artifact_paths.resolve_run_root(
        RUN_KEY,
        requested,
        repo_root=REPO_ROOT,
    )


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.out_root = resolve_run_root(args.out_root)
    args.paper_hu_root = resolve_paper_hu_root(str(args.protocol), args.paper_hu_root)
    for attribute, label in (
        ("audit_output", "--audit-output"),
        ("synthetic_smoke_output", "--synthetic-smoke-output"),
        ("prepare_output", "--prepare-output"),
    ):
        value = getattr(args, attribute)
        if value is not None:
            setattr(
                args,
                attribute,
                artifact_paths.resolve_managed_path(
                    value,
                    allowed_root=args.out_root,
                    label=label,
                ),
            )
    return args


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c = load_module("ptvreg_python_lungct_common", LUNG_COMMON_PATH)
raw_lungct = load_module("ptvreg_python_data_lungct", DATA_LUNGCT_PATH)


def protocol_tag(protocol: str) -> str:
    if protocol == "canonical_1mm":
        return PROTOCOL_CANONICAL
    if protocol == "paper_hu_iso1mm":
        return PROTOCOL_PAPER_HU
    if protocol == "paper_hu_native":
        return PROTOCOL_PAPER_HU_NATIVE
    raise ValueError(f"Unsupported protocol: {protocol!r}")


def protocol_slug(protocol: str) -> str:
    return {
        "canonical_1mm": "canonical",
        "paper_hu_iso1mm": "paperhu",
        "paper_hu_native": "paperhu_native",
    }[protocol]


def paper_hu_root_for_protocol(protocol: str, explicit_root: Optional[Path]) -> Path:
    if explicit_root is not None:
        return Path(explicit_root)
    if protocol == "paper_hu_native":
        return DEFAULT_PAPER_HU_NATIVE_ROOT
    return DEFAULT_PAPER_HU_ISO1MM_ROOT


def resolve_paper_hu_root(protocol: str, explicit_root: Optional[Path]) -> Path:
    """Keep generated paper-HU inputs in the configured external artifact store."""

    return artifact_paths.resolve_managed_path(
        paper_hu_root_for_protocol(protocol, explicit_root),
        allowed_root=artifact_paths.artifact_root(repo_root=REPO_ROOT),
        label="--paper-hu-root",
    )


def parse_triplet(value: str, cast, name: str) -> Tuple:
    vals = [token for token in str(value).replace(" ", "").split(",") if token]
    if len(vals) == 1:
        vals = vals * 3
    if len(vals) != 3:
        raise ValueError(f"{name} expects one value or three comma-separated values, got {value!r}")
    return tuple(cast(token) for token in vals)


def fmt_float(value: float) -> str:
    return c.format_float(float(value))


def run_name_for_case(protocol: str, case_id: int, opts: Dict[str, object]) -> str:
    grid = opts["grid_spacing"]
    metric_param = opts["metric_param"]
    assert isinstance(grid, tuple)
    assert isinstance(metric_param, tuple)
    return (
        f"case{int(case_id):02d}_{protocol_slug(protocol)}_"
        f"gs{'x'.join(str(int(v)) for v in grid)}_"
        f"tv{fmt_float(float(opts['isoTV']))}_"
        f"lcc{fmt_float(float(metric_param[0]))}_admm"
    )


def default_raw_dirlab_root(ptvprep_root: Path) -> Path:
    override = os.environ.get("INRORSPLINE_DIRLAB_ROOT")
    if override:
        return Path(override)
    root = Path(ptvprep_root)
    if root.name == "pTVReg_iso1mm" and root.parent.name == "processed":
        return root.parent.parent
    if Path("/database").exists():
        return Path("/database/datasets/lungCT/DIR-Lab")
    return Path(f"/data/{os.environ.get('USER', 'hengjie')}/datasets/lungCT/DIR-Lab")


def default_native_mask_root(raw_root: Path) -> Path:
    override = os.environ.get("INRORSPLINE_DIRLAB_NATIVE_MASK_ROOT")
    if override:
        return Path(override)
    return Path(raw_root) / "processed" / "lungmask_R231_native"


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name!r}")


def build_opts(args: argparse.Namespace, device: torch.device | str) -> Dict[str, object]:
    opts: Dict[str, object] = {
        "device": str(device),
        "dtype": dtype_from_name(str(args.dtype)),
        "opt_method": str(args.opt_method),
        "grid_spacing": parse_triplet(str(args.grid_spacing), int, "grid_spacing"),
        "isoTV": float(args.isoTV),
        "metric": str(args.metric),
        "metric_param": parse_triplet(str(args.metric_param), float, "metric_param"),
        "scale_metric_param": bool(args.scale_metric_param),
        "pix_resolution": parse_triplet(str(args.pix_resolution), float, "pix_resolution"),
        "spline_order": int(args.spline_order),
        "k_down": float(args.k_down),
        "border_mask": int(args.border_mask),
        "interp_type": int(args.interp_type),
        "max_iters": int(args.max_iters),
        "cp_refinements": int(args.cp_refinements),
        "csqrt": float(args.csqrt),
        "verbose": bool(args.verbose),
        "admm_max_outer": int(args.admm_max_outer),
        "admm_lbfgs_iters": int(args.admm_lbfgs_iters),
        "admm_mu": float(args.admm_mu),
        "admm_tau": float(args.admm_tau),
        "admm_tol": float(args.admm_tol),
    }
    if args.admm_rho is not None:
        opts["admm_rho"] = float(args.admm_rho)
    if args.nlvl is not None:
        opts["nlvl"] = int(args.nlvl)
    return opts


def opts_for_json(opts: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in opts.items():
        if isinstance(value, torch.dtype):
            out[key] = str(value).replace("torch.", "")
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(args: Sequence[str], cwd: Path = REPO_ROOT) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return None
    return proc.stdout.strip()


def git_metadata_dir(worktree: Path) -> Optional[Path]:
    dotgit = Path(worktree) / ".git"
    if dotgit.is_dir():
        return dotgit
    if dotgit.is_file():
        text = dotgit.read_text().strip()
        if text.startswith("gitdir:"):
            raw = text.split(":", 1)[1].strip()
            path = Path(raw)
            if not path.is_absolute():
                path = dotgit.parent / path
            return path.resolve()
    return None


def read_packed_ref(gitdir: Path, ref: str) -> Optional[str]:
    packed = gitdir / "packed-refs"
    if not packed.exists():
        return None
    for line in packed.read_text().splitlines():
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            return parts[0].strip()
    return None


def git_head_fallback(worktree: Path) -> Tuple[Optional[str], Optional[str]]:
    gitdir = git_metadata_dir(worktree)
    if gitdir is None:
        return None, None
    head_path = gitdir / "HEAD"
    if not head_path.exists():
        return None, None
    head = head_path.read_text().strip()
    if not head.startswith("ref:"):
        return head, None
    ref = head.split(":", 1)[1].strip()
    ref_path = gitdir / ref
    commit = ref_path.read_text().strip() if ref_path.exists() else read_packed_ref(gitdir, ref)
    branch = ref.rsplit("/", 1)[-1]
    return commit, branch


def source_provenance() -> Dict[str, object]:
    script_paths = [
        THIS_DIR / "__init__.py",
        THIS_DIR / "run_copd_case.py",
        THIS_DIR / "launch_copd_fixed_all10.py",
        THIS_DIR / "aggregate_copd_fixed_all10.py",
    ]
    status = git_output(["status", "--short", "--untracked-files=no"])
    repo_commit = git_output(["rev-parse", "HEAD"])
    repo_branch = git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    if repo_commit is None or repo_branch is None:
        fallback_commit, fallback_branch = git_head_fallback(REPO_ROOT)
        repo_commit = repo_commit or fallback_commit
        repo_branch = repo_branch or fallback_branch
    external_root = REPO_ROOT / "external" / "ptvreg-python"
    external_commit = git_output(["rev-parse", "HEAD"], cwd=external_root)
    if external_commit is None:
        external_commit = git_head_fallback(external_root)[0]
    return {
        "repo_commit": repo_commit,
        "repo_branch": repo_branch,
        "repo_dirty_tracked": None if status is None else bool(status),
        "external_ptvreg_python_commit": external_commit,
        "external_ptvreg_python_path": str(external_root),
        "adapter_source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in script_paths
            if path.exists()
        },
    }


def import_register_pair_min():
    if not PTVREG_PYTHON_SRC.is_dir():
        raise FileNotFoundError(
            f"Missing ptvreg-python source at {PTVREG_PYTHON_SRC}. "
            "Run `git submodule update --init external/ptvreg-python`."
        )
    if str(PTVREG_PYTHON_SRC) not in sys.path:
        sys.path.insert(0, str(PTVREG_PYTHON_SRC))
    from ptv.register_min import register_pair_min

    return register_pair_min


def setup_device(args: argparse.Namespace) -> torch.device:
    if str(args.device) == "cpu":
        return torch.device("cpu")
    return c.set_cuda_device(str(args.gpu))


def clip_normalize_paper_hu(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    return ((np.clip(arr, -1000.0, 500.0) + 1000.0) / 1500.0).astype(np.float32, copy=False)


def landmarks_xyz_one_based_to_zyx_zero_based(landmarks_xyz: np.ndarray) -> np.ndarray:
    return np.asarray(landmarks_xyz, dtype=np.float64)[:, [2, 1, 0]] - 1.0


def points_inside_shape(points_zyx: np.ndarray, shape_zyx: Sequence[int]) -> Dict[str, object]:
    pts = np.asarray(points_zyx, dtype=np.float64)
    shape = np.asarray(tuple(int(v) for v in shape_zyx), dtype=np.float64)
    inside = np.all((pts >= 0.0) & (pts <= (shape[None, :] - 1.0)), axis=1)
    bad = np.where(~inside)[0]
    return {
        "all_inside": bool(np.all(inside)),
        "inside_fraction": float(np.mean(inside)) if inside.size else float("nan"),
        "n_points": int(inside.size),
        "outside_indices_first10": [int(v) for v in bad[:10]],
    }


def ptvreg_formula_crop_bounds_zyx_from_landmarks_xyz(
    fixed_xyz: np.ndarray,
    moving_xyz: np.ndarray,
    shape_zyx: Sequence[int],
) -> np.ndarray:
    both = np.concatenate([np.asarray(fixed_xyz), np.asarray(moving_xyz)], axis=0)
    lo1_xyz = np.floor(np.min(both, axis=0)).astype(np.int64)
    hi1_xyz = np.ceil(np.max(both, axis=0)).astype(np.int64)
    shape_xyz = np.asarray(tuple(int(v) for v in shape_zyx)[::-1], dtype=np.int64)
    start1_xyz = np.maximum(1, lo1_xyz - MARGIN_XYZ)
    end1_xyz = np.minimum(shape_xyz, hi1_xyz + MARGIN_XYZ)
    start0_zyx = start1_xyz[[2, 1, 0]] - 1
    end0_zyx = end1_xyz[[2, 1, 0]] - 1
    return np.stack([start0_zyx, end0_zyx], axis=1).astype(np.int64)


def legacy_python_crop_bounds_zyx_from_landmarks_xyz(
    fixed_xyz: np.ndarray,
    moving_xyz: np.ndarray,
    shape_zyx: Sequence[int],
) -> np.ndarray:
    fixed_zyx = np.asarray(fixed_xyz, dtype=np.float64)[:, [2, 1, 0]]
    moving_zyx = np.asarray(moving_xyz, dtype=np.float64)[:, [2, 1, 0]]
    both = np.concatenate([fixed_zyx, moving_zyx], axis=0)
    lo = np.floor(np.min(both, axis=0)).astype(np.int64)
    hi = np.ceil(np.max(both, axis=0)).astype(np.int64)
    shape = np.asarray(tuple(int(v) for v in shape_zyx), dtype=np.int64)
    start = np.maximum(0, lo - MARGIN_ZYX)
    end = np.minimum(shape - 1, hi + MARGIN_ZYX)
    return np.stack([start, end], axis=1).astype(np.int64)


def transform_landmarks_native0_to_crop(points_zyx: np.ndarray, bounds_zyx: np.ndarray) -> np.ndarray:
    return np.asarray(points_zyx, dtype=np.float64) - np.asarray(bounds_zyx[:, 0], dtype=np.float64)[None, :]


def crop_zyx(vol: np.ndarray, bounds_zyx: np.ndarray) -> np.ndarray:
    return vol[
        bounds_zyx[0, 0] : bounds_zyx[0, 1] + 1,
        bounds_zyx[1, 0] : bounds_zyx[1, 1] + 1,
        bounds_zyx[2, 0] : bounds_zyx[2, 1] + 1,
    ]


def resize_image_trilinear_zyx(vol: np.ndarray, new_shape_zyx: Sequence[int], device: torch.device) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(vol, dtype=np.float32))[None, None].to(device=device)
        out = F.interpolate(
            tensor,
            size=tuple(int(v) for v in new_shape_zyx),
            mode="trilinear",
            align_corners=True,
        )
    return out[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)


def paper_hu_paths(root: Path, case_id: int) -> Dict[str, Path]:
    case_dir = Path(root) / "COPD" / f"case{int(case_id):02d}"
    return {
        "case_dir": case_dir,
        "fixed_reg": case_dir / "fixed_iBH_reg1mm_trilinear_zyx.npy",
        "moving_reg": case_dir / "moving_eBH_reg1mm_trilinear_zyx.npy",
        "audit": case_dir / "preprocess_audit.json",
    }


def paper_hu_native_paths(root: Path, case_id: int) -> Dict[str, Path]:
    case_dir = Path(root) / "COPD" / f"case{int(case_id):02d}"
    return {
        "case_dir": case_dir,
        "fixed_crop": case_dir / "fixed_iBH_crop_paperhu_native_zyx.npy",
        "moving_crop": case_dir / "moving_eBH_crop_paperhu_native_zyx.npy",
        "fixed_mask_crop": case_dir / f"fixed_iBH_{MASK_MODEL_NAME}_lungmask_crop_zyx.npy",
        "moving_mask_crop": case_dir / f"moving_eBH_{MASK_MODEL_NAME}_lungmask_crop_zyx.npy",
        "landmarks_fixed_crop": case_dir / "landmarks_fixed_iBH_crop_zyx.npy",
        "landmarks_moving_crop": case_dir / "landmarks_moving_eBH_crop_zyx.npy",
        "audit": case_dir / "preprocess_audit.json",
    }


def all_exist(paths: Iterable[Path]) -> bool:
    return all(Path(path).exists() for path in paths)


def load_native_copd_mask(
    *,
    mask_root: Path,
    case_id: int,
    phase: str,
    shape_zyx: Sequence[int],
    fallback_ones: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    path = raw_lungct.mask_root_case_dir(Path(mask_root), int(case_id), dataset="copd") / (
        f"case{int(case_id):02d}_{phase}_{MASK_MODEL_NAME}_binary_zyx.npy"
    )
    metadata: Dict[str, object] = {
        "phase": phase,
        "model": MASK_MODEL_NAME,
        "path": str(path),
        "source": "native_lungmask_R231_binary",
        "fallback_ones": False,
    }
    if not path.exists():
        if not fallback_ones:
            raise FileNotFoundError(f"Missing native COPD mask: {path}")
        mask = np.ones(tuple(int(v) for v in shape_zyx), dtype=np.uint8)
        metadata.update({"source": "all_ones_fallback", "fallback_ones": True})
        return mask, metadata
    mask = (np.load(path) > 0).astype(np.uint8, copy=False)
    if mask.shape != tuple(int(v) for v in shape_zyx):
        raise ValueError(f"Native mask {path} shape {mask.shape} != image shape {tuple(shape_zyx)}")
    metadata.update(
        {
            "foreground_voxels": int(mask.sum()),
            "foreground_fraction": float(mask.mean()),
        }
    )
    return mask, metadata


def create_or_load_paper_hu_case(
    *,
    case_id: int,
    canonical_root: Path,
    raw_root: Path,
    paper_hu_root: Path,
    device: torch.device,
    overwrite: bool,
) -> Tuple[object, Dict[str, object]]:
    canonical = c.load_case(canonical_root, case_id, dataset="copd")
    paths = paper_hu_paths(paper_hu_root, case_id)
    required = [paths["fixed_reg"], paths["moving_reg"], paths["audit"]]
    created = False

    if overwrite or not all_exist(required):
        raw_case = raw_lungct.resolve_copd_case(int(case_id), Path(raw_root))
        if raw_case.status != "complete":
            raise RuntimeError(f"COPD raw case {case_id} is {raw_case.status}; notes={raw_case.notes}")
        bounds = np.asarray(canonical.preprocess_audit["crop_bounds_zyx_inclusive"], dtype=np.int64)
        if bounds.shape != (3, 2):
            raise ValueError(f"Unexpected crop bounds shape for case {case_id}: {bounds.shape}")
        target_shape = tuple(int(v) for v in canonical.shape_zyx)
        fixed_raw = raw_lungct.load_raw_copd_image(raw_case.image_ibh, raw_case.shape_zyx)
        moving_raw = raw_lungct.load_raw_copd_image(raw_case.image_ebh, raw_case.shape_zyx)
        fixed_crop = crop_zyx(clip_normalize_paper_hu(fixed_raw), bounds)
        moving_crop = crop_zyx(clip_normalize_paper_hu(moving_raw), bounds)
        fixed_reg = resize_image_trilinear_zyx(fixed_crop, target_shape, device)
        moving_reg = resize_image_trilinear_zyx(moving_crop, target_shape, device)
        paths["case_dir"].mkdir(parents=True, exist_ok=True)
        np.save(paths["fixed_reg"], fixed_reg.astype(np.float32, copy=False))
        np.save(paths["moving_reg"], moving_reg.astype(np.float32, copy=False))
        audit = {
            "protocol_name": "ptvreg_python_paper_hu_iso1mm",
            "dataset": "copd",
            "case_id": int(case_id),
            "fixed_phase": "iBH",
            "moving_phase": "eBH",
            "direction": "eBHCT -> iBHCT",
            "intensity_rule": "clip [-1000, 500], add 1000, divide by 1500",
            "crop_rule": "reuse canonical pTVReg_iso1mm crop_bounds_zyx_inclusive from fixed/moving landmark bbox margin xyz [10,10,5]",
            "crop_bounds_zyx_inclusive": bounds.astype(int).tolist(),
            "crop_shape_zyx": [int(v) for v in canonical.crop_shape_zyx],
            "registration_shape_zyx": list(target_shape),
            "registration_spacing_zyx_mm": [1.0, 1.0, 1.0],
            "native_spacing_zyx_mm": [float(v) for v in canonical.native_spacing_zyx],
            "raw_shape_zyx": [int(v) for v in raw_case.shape_zyx],
            "raw_array_order": "zyx",
            "image_resampling": "PyTorch trilinear, align_corners=True",
            "landmark_source": "canonical pTVReg_iso1mm landmarks reused without change",
            "mask_source": "canonical pTVReg_iso1mm masks reused for metric/regularity only",
            "canonical_ptvprep_root": str(canonical_root),
            "raw_dirlab_root": str(raw_root),
            "image_paths": {
                "fixed_ibh": raw_case.image_ibh,
                "moving_ebh": raw_case.image_ebh,
            },
            "array_paths": {
                "fixed_iBH_reg1mm_trilinear_zyx": str(paths["fixed_reg"]),
                "moving_eBH_reg1mm_trilinear_zyx": str(paths["moving_reg"]),
            },
            "verification": {
                "fixed_reg_shape_matches_canonical": tuple(fixed_reg.shape) == tuple(canonical.shape_zyx),
                "moving_reg_shape_matches_canonical": tuple(moving_reg.shape) == tuple(canonical.shape_zyx),
                "fixed_reg_finite": bool(np.isfinite(fixed_reg).all()),
                "moving_reg_finite": bool(np.isfinite(moving_reg).all()),
                "fixed_reg_min": float(np.min(fixed_reg)),
                "fixed_reg_max": float(np.max(fixed_reg)),
                "moving_reg_min": float(np.min(moving_reg)),
                "moving_reg_max": float(np.max(moving_reg)),
            },
        }
        audit["verification_pass"] = bool(
            audit["verification"]["fixed_reg_shape_matches_canonical"]
            and audit["verification"]["moving_reg_shape_matches_canonical"]
            and audit["verification"]["fixed_reg_finite"]
            and audit["verification"]["moving_reg_finite"]
        )
        c.write_json(paths["audit"], audit)
        created = True

    fixed = np.load(paths["fixed_reg"]).astype(np.float32, copy=False)
    moving = np.load(paths["moving_reg"]).astype(np.float32, copy=False)
    audit = json.loads(paths["audit"].read_text())
    if tuple(fixed.shape) != tuple(canonical.shape_zyx):
        raise ValueError(f"Paper-HU fixed shape {fixed.shape} != canonical shape {canonical.shape_zyx}")
    if tuple(moving.shape) != tuple(canonical.shape_zyx):
        raise ValueError(f"Paper-HU moving shape {moving.shape} != canonical shape {canonical.shape_zyx}")

    case = c.LungCTCase(
        case_id=int(case_id),
        dataset="copd",
        case_dir=paths["case_dir"],
        fixed_label=canonical.fixed_label,
        moving_label=canonical.moving_label,
        fixed=fixed,
        moving=moving,
        fixed_mask=canonical.fixed_mask,
        moving_mask=canonical.moving_mask,
        landmarks_fixed=canonical.landmarks_fixed,
        landmarks_moving=canonical.landmarks_moving,
        landmarks_fixed_crop=canonical.landmarks_fixed_crop,
        landmarks_moving_crop=canonical.landmarks_moving_crop,
        native_spacing_zyx=canonical.native_spacing_zyx,
        crop_shape_zyx=canonical.crop_shape_zyx,
        preprocess_audit=audit,
    )
    metadata = {
        "input_protocol": "paper_hu_iso1mm",
        "paper_hu_created_this_run": created,
        "paper_hu_root": str(paper_hu_root),
        "paper_hu_case_dir": str(paths["case_dir"]),
        "paper_hu_audit": str(paths["audit"]),
        "canonical_ptvprep_root": str(canonical_root),
        "raw_dirlab_root": str(raw_root),
        "intensity_rule": "clip [-1000, 500], normalize to [0,1]",
        "crop_bounds_zyx_inclusive": audit.get("crop_bounds_zyx_inclusive"),
        "registration_shape_zyx": audit.get("registration_shape_zyx"),
        "verification_pass": bool(audit.get("verification_pass")),
    }
    return case, metadata


def create_or_load_paper_hu_native_case(
    *,
    case_id: int,
    raw_root: Path,
    paper_hu_root: Path,
    mask_root: Path,
    overwrite: bool,
    allow_mask_fallback: bool,
) -> Tuple[object, Dict[str, object]]:
    paths = paper_hu_native_paths(paper_hu_root, case_id)
    required = [
        paths["fixed_crop"],
        paths["moving_crop"],
        paths["fixed_mask_crop"],
        paths["moving_mask_crop"],
        paths["landmarks_fixed_crop"],
        paths["landmarks_moving_crop"],
        paths["audit"],
    ]
    created = False

    if overwrite or not all_exist(required):
        raw_case = raw_lungct.resolve_copd_case(int(case_id), Path(raw_root))
        if raw_case.status != "complete":
            raise RuntimeError(f"COPD raw case {case_id} is {raw_case.status}; notes={raw_case.notes}")
        fixed_xyz = raw_lungct.load_landmarks_xyz(raw_case.landmarks_ibh)
        moving_xyz = raw_lungct.load_landmarks_xyz(raw_case.landmarks_ebh)
        fixed_native0_zyx = landmarks_xyz_one_based_to_zyx_zero_based(fixed_xyz)
        moving_native0_zyx = landmarks_xyz_one_based_to_zyx_zero_based(moving_xyz)
        ptvreg_bounds = ptvreg_formula_crop_bounds_zyx_from_landmarks_xyz(fixed_xyz, moving_xyz, raw_case.shape_zyx)
        legacy_bounds = legacy_python_crop_bounds_zyx_from_landmarks_xyz(fixed_xyz, moving_xyz, raw_case.shape_zyx)
        crop_bounds = ptvreg_bounds
        crop_shape = tuple((crop_bounds[:, 1] - crop_bounds[:, 0] + 1).astype(int).tolist())

        fixed_raw = raw_lungct.load_raw_copd_image(raw_case.image_ibh, raw_case.shape_zyx)
        moving_raw = raw_lungct.load_raw_copd_image(raw_case.image_ebh, raw_case.shape_zyx)
        fixed_crop = crop_zyx(clip_normalize_paper_hu(fixed_raw), crop_bounds)
        moving_crop = crop_zyx(clip_normalize_paper_hu(moving_raw), crop_bounds)
        fixed_mask_native, fixed_mask_meta = load_native_copd_mask(
            mask_root=mask_root,
            case_id=int(case_id),
            phase="iBH",
            shape_zyx=raw_case.shape_zyx,
            fallback_ones=allow_mask_fallback,
        )
        moving_mask_native, moving_mask_meta = load_native_copd_mask(
            mask_root=mask_root,
            case_id=int(case_id),
            phase="eBH",
            shape_zyx=raw_case.shape_zyx,
            fallback_ones=allow_mask_fallback,
        )
        fixed_mask_crop = crop_zyx(fixed_mask_native, crop_bounds).astype(np.uint8, copy=False)
        moving_mask_crop = crop_zyx(moving_mask_native, crop_bounds).astype(np.uint8, copy=False)
        fixed_crop_zyx = transform_landmarks_native0_to_crop(fixed_native0_zyx, crop_bounds)
        moving_crop_zyx = transform_landmarks_native0_to_crop(moving_native0_zyx, crop_bounds)

        paths["case_dir"].mkdir(parents=True, exist_ok=True)
        np.save(paths["fixed_crop"], fixed_crop.astype(np.float32, copy=False))
        np.save(paths["moving_crop"], moving_crop.astype(np.float32, copy=False))
        np.save(paths["fixed_mask_crop"], fixed_mask_crop)
        np.save(paths["moving_mask_crop"], moving_mask_crop)
        np.save(paths["landmarks_fixed_crop"], fixed_crop_zyx.astype(np.float64, copy=False))
        np.save(paths["landmarks_moving_crop"], moving_crop_zyx.astype(np.float64, copy=False))

        verification = {
            "fixed_crop_shape_matches_expected": tuple(fixed_crop.shape) == tuple(crop_shape),
            "moving_crop_shape_matches_expected": tuple(moving_crop.shape) == tuple(crop_shape),
            "fixed_mask_shape_matches_crop": tuple(fixed_mask_crop.shape) == tuple(crop_shape),
            "moving_mask_shape_matches_crop": tuple(moving_mask_crop.shape) == tuple(crop_shape),
            "fixed_crop_finite": bool(np.isfinite(fixed_crop).all()),
            "moving_crop_finite": bool(np.isfinite(moving_crop).all()),
            "fixed_crop_landmarks_inside": points_inside_shape(fixed_crop_zyx, crop_shape),
            "moving_crop_landmarks_inside": points_inside_shape(moving_crop_zyx, crop_shape),
            "fixed_crop_min": float(np.min(fixed_crop)),
            "fixed_crop_max": float(np.max(fixed_crop)),
            "moving_crop_min": float(np.min(moving_crop)),
            "moving_crop_max": float(np.max(moving_crop)),
            "fixed_mask_voxels": int(fixed_mask_crop.sum()),
            "moving_mask_voxels": int(moving_mask_crop.sum()),
        }
        verification["pass"] = bool(
            verification["fixed_crop_shape_matches_expected"]
            and verification["moving_crop_shape_matches_expected"]
            and verification["fixed_mask_shape_matches_crop"]
            and verification["moving_mask_shape_matches_crop"]
            and verification["fixed_crop_finite"]
            and verification["moving_crop_finite"]
            and verification["fixed_crop_landmarks_inside"]["all_inside"]
            and verification["moving_crop_landmarks_inside"]["all_inside"]
        )
        audit = {
            "protocol_name": "ptvreg_python_paper_hu_native_crop",
            "dataset": "copd",
            "case_id": int(case_id),
            "fixed_phase": "iBH",
            "moving_phase": "eBH",
            "direction": "eBHCT -> iBHCT",
            "intensity_rule": "clip [-1000, 500], add 1000, divide by 1500",
            "crop_rule": "pTVReg landmark bbox margin xyz [10,10,5], exact 1-based MATLAB formula converted to 0-based zyx",
            "crop_convention": "ptvreg_formula_1based_xyz_to_python_0based_zyx",
            "crop_margin_xyz": [int(v) for v in MARGIN_XYZ],
            "crop_margin_zyx": [int(v) for v in MARGIN_ZYX],
            "crop_bounds_zyx_inclusive": crop_bounds.astype(int).tolist(),
            "ptvreg_formula_crop_bounds_zyx_inclusive": ptvreg_bounds.astype(int).tolist(),
            "legacy_task4_crop_bounds_zyx_inclusive": legacy_bounds.astype(int).tolist(),
            "legacy_minus_ptvreg_bounds_zyx": (legacy_bounds - ptvreg_bounds).astype(int).tolist(),
            "raw_shape_zyx": [int(v) for v in raw_case.shape_zyx],
            "native_spacing_zyx_mm": [float(v) for v in raw_case.spacing_zyx],
            "crop_shape_zyx": [int(v) for v in crop_shape],
            "registration_shape_zyx": [int(v) for v in crop_shape],
            "registration_spacing_zyx_mm": [float(v) for v in raw_case.spacing_zyx],
            "registration_grid": "native_landmark_crop_anisotropic_spacing",
            "image_resampling": "none; raw COPD native voxels retained after cropping",
            "mask_source": f"native lungmask {MASK_MODEL_NAME} crop for posthoc regularity only",
            "landmark_source_order": "xyz, pTVReg/MATLAB 1-based voxel coordinates",
            "landmark_saved_order": "zyx, Python 0-based crop voxel coordinates",
            "raw_dirlab_root": str(raw_root),
            "mask_root": str(mask_root),
            "image_paths": {
                "fixed_ibh": raw_case.image_ibh,
                "moving_ebh": raw_case.image_ebh,
            },
            "landmark_paths": {
                "fixed_ibh": raw_case.landmarks_ibh,
                "moving_ebh": raw_case.landmarks_ebh,
            },
            "mask_metadata": {
                "fixed_iBH": fixed_mask_meta,
                "moving_eBH": moving_mask_meta,
            },
            "array_paths": {
                "fixed_iBH_crop_paperhu_native_zyx": str(paths["fixed_crop"]),
                "moving_eBH_crop_paperhu_native_zyx": str(paths["moving_crop"]),
                "fixed_iBH_lungmask_crop_zyx": str(paths["fixed_mask_crop"]),
                "moving_eBH_lungmask_crop_zyx": str(paths["moving_mask_crop"]),
                "landmarks_fixed_iBH_crop_zyx": str(paths["landmarks_fixed_crop"]),
                "landmarks_moving_eBH_crop_zyx": str(paths["landmarks_moving_crop"]),
            },
            "verification": verification,
            "verification_pass": verification["pass"],
        }
        c.write_json(paths["audit"], audit)
        created = True

    fixed = np.load(paths["fixed_crop"]).astype(np.float32, copy=False)
    moving = np.load(paths["moving_crop"]).astype(np.float32, copy=False)
    fixed_mask = (np.load(paths["fixed_mask_crop"]) > 0).astype(np.uint8, copy=False)
    moving_mask = (np.load(paths["moving_mask_crop"]) > 0).astype(np.uint8, copy=False)
    landmarks_fixed_crop = np.load(paths["landmarks_fixed_crop"]).astype(np.float64, copy=False)
    landmarks_moving_crop = np.load(paths["landmarks_moving_crop"]).astype(np.float64, copy=False)
    audit = json.loads(paths["audit"].read_text())
    spacing = tuple(float(v) for v in audit["native_spacing_zyx_mm"])
    crop_shape = tuple(int(v) for v in audit["crop_shape_zyx"])
    if tuple(fixed.shape) != crop_shape:
        raise ValueError(f"Native paper-HU fixed shape {fixed.shape} != audit crop shape {crop_shape}")
    if tuple(moving.shape) != crop_shape:
        raise ValueError(f"Native paper-HU moving shape {moving.shape} != audit crop shape {crop_shape}")
    if tuple(fixed_mask.shape) != crop_shape:
        raise ValueError(f"Native paper-HU fixed mask shape {fixed_mask.shape} != audit crop shape {crop_shape}")
    if tuple(moving_mask.shape) != crop_shape:
        raise ValueError(f"Native paper-HU moving mask shape {moving_mask.shape} != audit crop shape {crop_shape}")

    case = c.LungCTCase(
        case_id=int(case_id),
        dataset="copd",
        case_dir=paths["case_dir"],
        fixed_label="iBH",
        moving_label="eBH",
        fixed=fixed,
        moving=moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        landmarks_fixed=landmarks_fixed_crop,
        landmarks_moving=landmarks_moving_crop,
        landmarks_fixed_crop=landmarks_fixed_crop,
        landmarks_moving_crop=landmarks_moving_crop,
        native_spacing_zyx=spacing,
        crop_shape_zyx=crop_shape,
        preprocess_audit=audit,
    )
    metadata = {
        "input_protocol": "paper_hu_native",
        "paper_hu_created_this_run": created,
        "paper_hu_root": str(paper_hu_root),
        "paper_hu_case_dir": str(paths["case_dir"]),
        "paper_hu_audit": str(paths["audit"]),
        "raw_dirlab_root": str(raw_root),
        "mask_root": str(mask_root),
        "intensity_rule": "clip [-1000, 500], normalize to [0,1]",
        "crop_rule": audit.get("crop_rule"),
        "crop_convention": audit.get("crop_convention"),
        "crop_bounds_zyx_inclusive": audit.get("crop_bounds_zyx_inclusive"),
        "registration_shape_zyx": audit.get("registration_shape_zyx"),
        "registration_spacing_zyx_mm": audit.get("registration_spacing_zyx_mm"),
        "registration_grid": audit.get("registration_grid"),
        "verification_pass": bool(audit.get("verification_pass")),
    }
    return case, metadata


def load_protocol_case(args: argparse.Namespace, device: torch.device) -> Tuple[object, Dict[str, object]]:
    if args.protocol == "canonical_1mm":
        case = c.load_case(args.data_root, args.case_id, dataset="copd")
        return case, {
            "input_protocol": "canonical_1mm",
            "canonical_ptvprep_root": str(args.data_root),
            "case_dir": str(case.case_dir),
            "intensity_rule": case.preprocess_audit.get("intensity_rule"),
            "crop_bounds_zyx_inclusive": case.preprocess_audit.get("crop_bounds_zyx_inclusive"),
            "registration_shape_zyx": case.preprocess_audit.get("registration_shape_zyx"),
            "verification_pass": bool(case.preprocess_audit.get("verification_pass", True)),
        }
    raw_root = args.raw_data_root or default_raw_dirlab_root(args.data_root)
    paper_hu_root = paper_hu_root_for_protocol(str(args.protocol), args.paper_hu_root)
    if args.protocol == "paper_hu_iso1mm":
        return create_or_load_paper_hu_case(
            case_id=int(args.case_id),
            canonical_root=Path(args.data_root),
            raw_root=Path(raw_root),
            paper_hu_root=paper_hu_root,
            device=device,
            overwrite=bool(args.overwrite_paper_hu),
        )
    if args.protocol == "paper_hu_native":
        mask_root = args.mask_root or default_native_mask_root(Path(raw_root))
        return create_or_load_paper_hu_native_case(
            case_id=int(args.case_id),
            raw_root=Path(raw_root),
            paper_hu_root=paper_hu_root,
            mask_root=Path(mask_root),
            overwrite=bool(args.overwrite_paper_hu),
            allow_mask_fallback=bool(args.allow_mask_fallback),
        )
    raise ValueError(f"Unsupported protocol: {args.protocol!r}")


def tensor_from_volume(volume: np.ndarray) -> torch.Tensor:
    arr = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    return torch.from_numpy(arr)[None, None]


def ptv_tmm_to_disp_vox_zyx(
    tmm: torch.Tensor | np.ndarray,
    pix_resolution_zyx: Sequence[float],
) -> np.ndarray:
    arr = tmm.detach().cpu().numpy() if isinstance(tmm, torch.Tensor) else np.asarray(tmm)
    if arr.ndim == 5:
        if int(arr.shape[0]) != 1 or int(arr.shape[1]) != 3:
            raise ValueError(f"Expected Tmm shape [1,3,D,H,W], got {arr.shape}")
        arr = arr[0]
    if arr.ndim != 4 or int(arr.shape[0]) != 3:
        raise ValueError(f"Expected Tmm shape [3,D,H,W] after batch squeeze, got {arr.shape}")
    spacing = np.asarray(tuple(float(v) for v in pix_resolution_zyx), dtype=np.float64).reshape(3, 1, 1, 1)
    disp_vox_channels = arr.astype(np.float64, copy=False) / spacing
    return np.moveaxis(disp_vox_channels, 0, -1).astype(np.float32, copy=False)


def disp_vox_zyx_to_dense_norm_xyz(disp_vox_zyx: np.ndarray) -> np.ndarray:
    disp = np.asarray(disp_vox_zyx, dtype=np.float32)
    if disp.ndim != 4 or int(disp.shape[-1]) != 3:
        raise ValueError(f"Expected displacement [D,H,W,3], got {disp.shape}")
    d, h, w = disp.shape[:3]
    scale_x = max((float(w) - 1.0) / 2.0, 1.0)
    scale_y = max((float(h) - 1.0) / 2.0, 1.0)
    scale_z = max((float(d) - 1.0) / 2.0, 1.0)
    return np.stack(
        (
            disp[..., 2] / scale_x,
            disp[..., 1] / scale_y,
            disp[..., 0] / scale_z,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def serialize_level_infos(infos: Sequence[object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, info in enumerate(infos):
        hist = getattr(info, "admm_history", None)
        rows.append(
            {
                "level_index": idx,
                "data_loss": float(getattr(info, "data_loss", float("nan"))),
                "reg_loss": float(getattr(info, "reg_loss", float("nan"))),
                "total_loss": float(getattr(info, "total_loss", float("nan"))),
                "n_iters": int(getattr(info, "n_iters", 0)),
                "grid_spacing": list(getattr(info, "grid_spacing", []) or []),
                "image_size": list(getattr(info, "image_size", []) or []),
                "admm_history": hist,
            }
        )
    return rows


def save_knots(path: Path, knots: torch.Tensor, level_infos: Sequence[object], opts: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "knots": knots.detach().cpu(),
        "level_infos": serialize_level_infos(level_infos),
        "opts": opts_for_json(opts),
    }
    torch.save(payload, path)


def field_stats(disp_vox_zyx: np.ndarray, dense_norm_xyz: np.ndarray, expected_shape: Sequence[int]) -> Dict[str, object]:
    finite_vox = np.isfinite(disp_vox_zyx)
    finite_norm = np.isfinite(dense_norm_xyz)
    mag = np.sqrt(np.sum(np.square(disp_vox_zyx.astype(np.float64)), axis=-1))
    return {
        "dense_field_shape_zyx": [int(v) for v in disp_vox_zyx.shape[:3]],
        "dense_field_shape_matches_input": tuple(disp_vox_zyx.shape[:3]) == tuple(int(v) for v in expected_shape),
        "dense_field_component_order": "zyx voxel displacement; fixed-grid pull field toward moving",
        "dense_norm_component_order": "xyz normalized PyTorch grid displacement",
        "dense_field_all_finite": bool(finite_vox.all() and finite_norm.all()),
        "dense_field_nan_count": int(np.isnan(disp_vox_zyx).sum() + np.isnan(dense_norm_xyz).sum()),
        "dense_field_inf_count": int(np.isinf(disp_vox_zyx).sum() + np.isinf(dense_norm_xyz).sum()),
        "disp_vox_abs_max": float(np.max(np.abs(disp_vox_zyx))) if disp_vox_zyx.size else float("nan"),
        "disp_vox_mag_mean": float(np.mean(mag)) if mag.size else float("nan"),
        "disp_vox_mag_p95": float(np.percentile(mag, 95)) if mag.size else float("nan"),
        "disp_vox_mag_max": float(np.max(mag)) if mag.size else float("nan"),
    }


def effective_opts_for_case(args: argparse.Namespace, opts: Dict[str, object], case: object) -> Dict[str, object]:
    if str(args.protocol) == "paper_hu_native":
        opts = dict(opts)
        opts["pix_resolution"] = tuple(float(v) for v in case.native_spacing_zyx)
        opts["effective_pix_resolution_source"] = "native COPD spacing"
    return opts


def final_metrics_from_native_crop_disp(
    *,
    disp_vox_native_crop_zyx: np.ndarray,
    dense_norm_xyz: np.ndarray,
    case: object,
    out_dir: Path,
    save_dense_dvf: bool,
    log_base: str,
) -> Dict[str, object]:
    identity = c.compute_tre_mm(case.landmarks_fixed_crop, case.landmarks_moving_crop, case.native_spacing_zyx)
    tre_snap = c.landmark_tre_from_dense_disp(
        disp_vox_native_crop_zyx,
        case.landmarks_fixed_crop,
        case.landmarks_moving_crop,
        sample_mode="nearest",
        snap_output=True,
        spacing_zyx=case.native_spacing_zyx,
        dense_disp_units="native_crop_voxel",
    )
    tre_sub = c.landmark_tre_from_dense_disp(
        disp_vox_native_crop_zyx,
        case.landmarks_fixed_crop,
        case.landmarks_moving_crop,
        sample_mode="trilinear",
        snap_output=False,
        spacing_zyx=case.native_spacing_zyx,
        dense_disp_units="native_crop_voxel",
    )
    regularity = c.displacement_regularities(
        disp_vox_native_crop_zyx,
        foreground_mask=case.fixed_mask.astype(bool),
        log_base=log_base,
    )
    c.save_landmark_csv(
        out_dir / "per_landmark_tre.csv",
        disp_vox_native_crop_zyx,
        case.landmarks_fixed_crop,
        case.landmarks_moving_crop,
        spacing_zyx=case.native_spacing_zyx,
        dense_disp_units="native_crop_voxel",
    )
    c.save_landmark_csv(
        out_dir / "per_landmark_tre_native_crop.csv",
        disp_vox_native_crop_zyx,
        case.landmarks_fixed_crop,
        case.landmarks_moving_crop,
        spacing_zyx=case.native_spacing_zyx,
        dense_disp_units="native_crop_voxel",
    )
    if save_dense_dvf:
        np.save(out_dir / "disp_vox_native_crop_zyx.npy", disp_vox_native_crop_zyx.astype(np.float32))
        np.save(out_dir / "disp_norm_xyz_native_crop.npy", dense_norm_xyz.astype(np.float32))
    return {
        "identity_tre": identity,
        "tre_snap_to_voxel": tre_snap,
        "tre_subvoxel": tre_sub,
        "native_crop_identity_tre": identity,
        "native_crop_tre_snap_to_voxel": tre_snap,
        "native_crop_tre_subvoxel": tre_sub,
        "headline_tre_metric": "native_crop_tre_snap_to_voxel",
        "regularity_fg_registration_crop": regularity,
        "regularity_spacing_basis": "native_crop_voxel_derivatives_on_anisotropic_grid",
        "registration_grid_spacing_zyx_mm": [float(v) for v in case.native_spacing_zyx],
        "dense_field_shape_zyx": list(disp_vox_native_crop_zyx.shape[:3]),
    }


def final_metrics_for_protocol(
    *,
    protocol: str,
    disp_vox_zyx: np.ndarray,
    dense_norm_xyz: np.ndarray,
    case: object,
    out_dir: Path,
    save_dense_dvf: bool,
    log_base: str,
) -> Dict[str, object]:
    if protocol == "paper_hu_native":
        return final_metrics_from_native_crop_disp(
            disp_vox_native_crop_zyx=disp_vox_zyx,
            dense_norm_xyz=dense_norm_xyz,
            case=case,
            out_dir=out_dir,
            save_dense_dvf=save_dense_dvf,
            log_base=log_base,
        )
    return c.final_metrics_from_dense(
        torch.from_numpy(dense_norm_xyz),
        case,
        out_dir,
        save_dense_dvf=save_dense_dvf,
        log_base=log_base,
    )


def output_file_manifest(out_dir: Path, protocol: str, save_dense_dvf: bool) -> Dict[str, object]:
    if protocol == "paper_hu_native":
        return {
            "summary_json": str(out_dir / "summary.json"),
            "disp_vox_zyx": str(out_dir / "disp_vox_native_crop_zyx.npy") if save_dense_dvf else None,
            "disp_norm_xyz": str(out_dir / "disp_norm_xyz_native_crop.npy") if save_dense_dvf else None,
            "disp_vox_native_crop_zyx": str(out_dir / "disp_vox_native_crop_zyx.npy") if save_dense_dvf else None,
            "disp_norm_xyz_native_crop": str(out_dir / "disp_norm_xyz_native_crop.npy") if save_dense_dvf else None,
            "per_landmark_tre": str(out_dir / "per_landmark_tre.csv"),
            "per_landmark_tre_native_crop": str(out_dir / "per_landmark_tre_native_crop.csv"),
            "ptvreg_python_knots": str(out_dir / "ptvreg_python_knots.pt"),
        }
    return {
        "summary_json": str(out_dir / "summary.json"),
        "disp_vox_zyx": str(out_dir / "disp_vox_reg1mm_zyx.npy") if save_dense_dvf else None,
        "disp_norm_xyz": str(out_dir / "disp_norm_xyz_reg1mm.npy") if save_dense_dvf else None,
        "disp_vox_reg1mm_zyx": str(out_dir / "disp_vox_reg1mm_zyx.npy") if save_dense_dvf else None,
        "disp_norm_xyz_reg1mm": str(out_dir / "disp_norm_xyz_reg1mm.npy") if save_dense_dvf else None,
        "per_landmark_tre": str(out_dir / "per_landmark_tre.csv"),
        "per_landmark_tre_native_crop": str(out_dir / "per_landmark_tre_native_crop.csv"),
        "ptvreg_python_knots": str(out_dir / "ptvreg_python_knots.pt"),
    }


def peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return float("nan")
    allocated = torch.cuda.max_memory_allocated(device) / (1024.0**3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024.0**3)
    return float(max(allocated, reserved))


def make_synthetic_pair(shape_zyx: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    z = torch.linspace(-1.0, 1.0, int(shape_zyx[0]), dtype=torch.float32)
    y = torch.linspace(-1.0, 1.0, int(shape_zyx[1]), dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, int(shape_zyx[2]), dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    fixed = torch.exp(-8.0 * ((xx + 0.10) ** 2 + (yy - 0.05) ** 2 + (zz + 0.02) ** 2))
    moving = torch.exp(-8.0 * ((xx + 0.02) ** 2 + (yy - 0.02) ** 2 + (zz + 0.04) ** 2))
    return moving[None, None], fixed[None, None]


def run_synthetic_smoke(args: argparse.Namespace) -> Dict[str, object]:
    device = setup_device(args)
    opts = build_opts(args, device)
    opts["nlvl"] = 1
    opts["max_iters"] = 1
    opts["admm_max_outer"] = 1
    opts["admm_lbfgs_iters"] = 1
    register_pair_min = import_register_pair_min()
    shape = parse_triplet(str(args.synthetic_shape), int, "synthetic_shape")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    moving, fixed = make_synthetic_pair(shape)
    t0 = time.perf_counter()
    _voldef, tmm, knots, infos = register_pair_min(moving, fixed, opts)
    elapsed = time.perf_counter() - t0
    pix_resolution = opts["pix_resolution"]
    assert isinstance(pix_resolution, tuple)
    disp_vox_zyx = ptv_tmm_to_disp_vox_zyx(tmm, pix_resolution)
    dense_norm_xyz = disp_vox_zyx_to_dense_norm_xyz(disp_vox_zyx)
    stats = field_stats(disp_vox_zyx, dense_norm_xyz, shape)
    payload: Dict[str, object] = {
        "status": "completed",
        "smoke_type": "synthetic_tiny_external_registration",
        "created_wall": c.now_text(),
        "elapsed_sec": float(elapsed),
        "peak_mem_gb": peak_memory_gb(device),
        "shape_zyx": [int(v) for v in shape],
        "opts": opts_for_json(opts),
        "field_conversion": {
            "external_tmm_shape": list(tmm.shape),
            "external_tmm_component_order": "z,y,x",
            "external_field_direction": "fixed-grid pull displacement sampling moving at identity + displacement",
            "repo_saved_disp_norm_component_order": "x,y,z",
        },
        "level_info": serialize_level_infos(infos),
        "knots_shape": list(knots.shape),
        "source_provenance": source_provenance(),
    }
    payload.update(stats)
    if args.synthetic_smoke_output is not None:
        c.write_json(args.synthetic_smoke_output, payload)
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def prepare_paper_hu_inputs(args: argparse.Namespace) -> Dict[str, object]:
    device = setup_device(args)
    raw_root = args.raw_data_root or default_raw_dirlab_root(args.data_root)
    prepare_protocol = str(args.protocol)
    if prepare_protocol == "canonical_1mm":
        prepare_protocol = "paper_hu_iso1mm"
    paper_hu_root = paper_hu_root_for_protocol(prepare_protocol, args.paper_hu_root)
    mask_root = args.mask_root or default_native_mask_root(Path(raw_root))
    rows: List[Dict[str, object]] = []
    for case_id in c.parse_int_list(args.prepare_cases):
        if prepare_protocol == "paper_hu_native":
            _case, metadata = create_or_load_paper_hu_native_case(
                case_id=int(case_id),
                raw_root=Path(raw_root),
                paper_hu_root=paper_hu_root,
                mask_root=Path(mask_root),
                overwrite=bool(args.overwrite_paper_hu),
                allow_mask_fallback=bool(args.allow_mask_fallback),
            )
        else:
            _case, metadata = create_or_load_paper_hu_case(
                case_id=int(case_id),
                canonical_root=Path(args.data_root),
                raw_root=Path(raw_root),
                paper_hu_root=paper_hu_root,
                device=device,
                overwrite=bool(args.overwrite_paper_hu),
            )
        rows.append({"case_id": int(case_id), **metadata})
    payload = {
        "status": "completed",
        "created_wall": c.now_text(),
        "dataset": "copd",
        "protocol": prepare_protocol,
        "paper_hu_root": str(paper_hu_root),
        "canonical_ptvprep_root": str(args.data_root),
        "raw_dirlab_root": str(raw_root),
        "mask_root": str(mask_root) if prepare_protocol == "paper_hu_native" else None,
        "case_ids": [int(row["case_id"]) for row in rows],
        "n_cases": len(rows),
        "all_verification_pass": all(bool(row.get("verification_pass")) for row in rows),
        "rows": rows,
        "source_provenance": source_provenance(),
    }
    if args.prepare_output is not None:
        c.write_json(args.prepare_output, payload)
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def run_case(args: argparse.Namespace) -> Dict[str, object]:
    device = setup_device(args)
    tag = protocol_tag(args.protocol)
    base_opts = build_opts(args, device)
    name = args.run_name or run_name_for_case(args.protocol, int(args.case_id), base_opts)
    out_dir = Path(args.out_root) / tag / name
    summary_path = out_dir / "summary.json"
    started_wall = c.now_text()
    try:
        case, input_metadata = load_protocol_case(args, device)
        opts = effective_opts_for_case(args, base_opts, case)
        name = args.run_name or run_name_for_case(args.protocol, int(args.case_id), opts)
        out_dir = Path(args.out_root) / tag / name
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not args.overwrite:
            existing = c.read_json(summary_path)
            if isinstance(existing, dict) and existing.get("status") == "completed":
                print(json.dumps({"skipped_completed": str(summary_path)}, indent=2), flush=True)
                return existing

        out_dir.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        register_pair_min = import_register_pair_min()
        fixed_t = tensor_from_volume(case.fixed)
        moving_t = tensor_from_volume(case.moving)
        start_payload: Dict[str, object] = {
            "status": "running",
            "dataset": "copd",
            "method": METHOD,
            "run_name": name,
            "protocol": str(args.protocol),
            "protocol_tag": tag,
            "case_id": int(args.case_id),
            "fixed_label": case.fixed_label,
            "moving_label": case.moving_label,
            "direction": f"{case.moving_label}CT -> {case.fixed_label}CT",
            "out_dir": str(out_dir),
            "started_wall": started_wall,
            "input_shape_zyx": [int(v) for v in case.shape_zyx],
            "input_metadata": input_metadata,
            "opts": opts_for_json(opts),
            "source_provenance": source_provenance(),
        }
        c.write_json(summary_path, start_payload)
        print(
            json.dumps(
                {
                    "started": str(summary_path),
                    "case_id": int(args.case_id),
                    "protocol": str(args.protocol),
                    "input_shape_zyx": [int(v) for v in case.shape_zyx],
                    "opts": opts_for_json(opts),
                },
                indent=2,
            ),
            flush=True,
        )
        t0 = time.perf_counter()
        voldef, tmm, knots, infos = register_pair_min(moving_t, fixed_t, opts)
        elapsed = time.perf_counter() - t0
        pix_resolution = opts["pix_resolution"]
        assert isinstance(pix_resolution, tuple)
        disp_vox_zyx = ptv_tmm_to_disp_vox_zyx(tmm, pix_resolution)
        dense_norm_xyz = disp_vox_zyx_to_dense_norm_xyz(disp_vox_zyx)
        stats = field_stats(disp_vox_zyx, dense_norm_xyz, case.shape_zyx)
        if not stats["dense_field_all_finite"]:
            raise RuntimeError(f"Non-finite displacement field for case {case.case_id}: {stats}")
        metrics = final_metrics_for_protocol(
            protocol=str(args.protocol),
            disp_vox_zyx=disp_vox_zyx,
            dense_norm_xyz=dense_norm_xyz,
            case=case,
            out_dir=out_dir,
            save_dense_dvf=bool(args.save_dense_dvf),
            log_base=str(args.log_base),
        )
        save_knots(out_dir / "ptvreg_python_knots.pt", knots, infos, opts)
        level_infos = serialize_level_infos(infos)
        summary: Dict[str, object] = {
            "status": "completed",
            "dataset": "copd",
            "method": METHOD,
            "run_name": name,
            "protocol": str(args.protocol),
            "protocol_tag": tag,
            "case_id": int(args.case_id),
            "fixed_label": case.fixed_label,
            "moving_label": case.moving_label,
            "direction": f"{case.moving_label}CT -> {case.fixed_label}CT",
            "out_dir": str(out_dir),
            "started_wall": started_wall,
            "completed_wall": c.now_text(),
            "elapsed_sec": float(elapsed),
            "peak_mem_gb": peak_memory_gb(device),
            "input_shape_zyx": [int(v) for v in case.shape_zyx],
            "native_spacing_zyx_mm": [float(v) for v in case.native_spacing_zyx],
            "crop_shape_zyx": [int(v) for v in case.crop_shape_zyx],
            "input_metadata": input_metadata,
            "opts": opts_for_json(opts),
            "ptvreg_python_api": "ptv.register_min.register_pair_min(moving, fixed, opts)",
            "source_provenance": source_provenance(),
            "field_conversion": {
                "external_tmm_shape": list(tmm.shape),
                "external_tmm_units": "mm",
                "external_tmm_component_order": "z,y,x",
                "external_field_direction": "fixed-grid pull displacement sampling moving at identity + displacement",
                "repo_saved_disp_vox_component_order": "z,y,x",
                "repo_saved_disp_norm_component_order": "x,y,z",
                "pix_resolution_zyx": [float(v) for v in pix_resolution],
            },
            "level_info": level_infos,
            "output_files": output_file_manifest(out_dir, str(args.protocol), bool(args.save_dense_dvf)),
        }
        summary.update(stats)
        summary.update(metrics)
        c.write_json(summary_path, summary)
        headline = summary["native_crop_tre_snap_to_voxel"]["tre_mean_mm"]
        print(
            json.dumps(
                {
                    "completed": str(summary_path),
                    "case_id": int(args.case_id),
                    "protocol": str(args.protocol),
                    "headline_native_crop_snap_tre_mm": headline,
                    "elapsed_sec": elapsed,
                    "peak_mem_gb": summary["peak_mem_gb"],
                },
                indent=2,
            ),
            flush=True,
        )
        return summary
    except Exception as exc:
        failure = {
            "status": "failed",
            "dataset": "copd",
            "method": METHOD,
            "run_name": name,
            "protocol": str(args.protocol),
            "protocol_tag": tag,
            "case_id": int(args.case_id),
            "out_dir": str(out_dir),
            "started_wall": started_wall,
            "failed_wall": c.now_text(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "source_provenance": source_provenance(),
        }
        c.write_json(summary_path, failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=PROTOCOL_CHOICES, default="canonical_1mm")
    parser.add_argument("--case-id", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--raw-data-root", type=Path, default=None)
    parser.add_argument("--paper-hu-root", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--save-dense-dvf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-paper-hu", action="store_true")
    parser.add_argument("--allow-mask-fallback", action="store_true")
    parser.add_argument("--audit-loader", action="store_true")
    parser.add_argument("--audit-cases", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--synthetic-shape", type=str, default="20,24,28")
    parser.add_argument("--synthetic-smoke-output", type=Path, default=None)
    parser.add_argument("--prepare-paper-hu-only", action="store_true")
    parser.add_argument("--prepare-cases", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--prepare-output", type=Path, default=None)
    parser.add_argument("--log-base", choices=("e", "2"), default="e")

    parser.add_argument("--opt-method", choices=("admm", "lbfgs", "adam"), default="admm")
    parser.add_argument("--grid-spacing", type=str, default="8,8,8")
    parser.add_argument("--isoTV", type=float, default=0.12)
    parser.add_argument("--metric", choices=("lcc", "ssd", "sad"), default="lcc")
    parser.add_argument("--metric-param", type=str, default="2.1,2.1,2.1")
    parser.add_argument("--scale-metric-param", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pix-resolution", type=str, default="1,1,1")
    parser.add_argument("--spline-order", type=int, default=1)
    parser.add_argument("--k-down", type=float, default=0.7)
    parser.add_argument("--border-mask", type=int, default=5)
    parser.add_argument("--interp-type", type=int, default=1)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--cp-refinements", type=int, default=0)
    parser.add_argument("--csqrt", type=float, default=5e-3)
    parser.add_argument("--nlvl", type=int, default=None)
    parser.add_argument("--admm-max-outer", type=int, default=10)
    parser.add_argument("--admm-lbfgs-iters", type=int, default=30)
    parser.add_argument("--admm-rho", type=float, default=None)
    parser.add_argument("--admm-mu", type=float, default=10.0)
    parser.add_argument("--admm-tau", type=float, default=1.3)
    parser.add_argument("--admm-tol", type=float, default=1e-2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if int(args.case_id) not in DEFAULT_CASES:
        raise ValueError(f"COPD case id must be 1-10, got {args.case_id}")
    return args


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    if args.audit_loader:
        audit = c.loader_audit(args.data_root, c.parse_int_list(args.audit_cases), dataset="copd")
        if args.audit_output is not None:
            c.write_json(args.audit_output, audit)
        print(json.dumps(audit, indent=2), flush=True)
        return
    if args.synthetic_smoke:
        run_synthetic_smoke(args)
        return
    if args.prepare_paper_hu_only:
        prepare_paper_hu_inputs(args)
        return
    run_case(args)


if __name__ == "__main__":
    main()
