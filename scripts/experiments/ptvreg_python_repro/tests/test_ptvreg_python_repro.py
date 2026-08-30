from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_copd_case as runner  # noqa: E402
import aggregate_copd_fixed_all10 as aggregate  # noqa: E402
import launch_copd_fixed_all10 as launcher  # noqa: E402


def default_args(**overrides):
    values = {
        "dtype": "float64",
        "opt_method": "admm",
        "grid_spacing": "8,8,8",
        "isoTV": 0.12,
        "metric": "lcc",
        "metric_param": "2.1,2.1,2.1",
        "scale_metric_param": True,
        "pix_resolution": "1,1,1",
        "spline_order": 1,
        "k_down": 0.7,
        "border_mask": 5,
        "interp_type": 1,
        "max_iters": 100,
        "cp_refinements": 0,
        "csqrt": 5e-3,
        "verbose": False,
        "admm_max_outer": 10,
        "admm_lbfgs_iters": 30,
        "admm_mu": 10.0,
        "admm_tau": 1.3,
        "admm_tol": 1e-2,
        "admm_rho": None,
        "nlvl": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_default_opts_match_fixed_plan():
    opts = runner.build_opts(default_args(), "cuda")
    assert opts["opt_method"] == "admm"
    assert opts["grid_spacing"] == (8, 8, 8)
    assert opts["isoTV"] == pytest.approx(0.12)
    assert opts["metric"] == "lcc"
    assert opts["metric_param"] == pytest.approx((2.1, 2.1, 2.1))
    assert opts["scale_metric_param"] is True
    assert opts["pix_resolution"] == pytest.approx((1.0, 1.0, 1.0))
    assert opts["spline_order"] == 1
    assert opts["k_down"] == pytest.approx(0.7)
    assert opts["border_mask"] == 5
    assert opts["interp_type"] == 1
    assert opts["dtype"] == torch.float64


def test_native_protocol_overrides_pix_resolution_from_case():
    opts = runner.build_opts(default_args(), "cuda")
    case = argparse.Namespace(native_spacing_zyx=(2.5, 0.586, 0.586))
    args = argparse.Namespace(protocol="paper_hu_native")
    effective = runner.effective_opts_for_case(args, opts, case)
    assert effective["pix_resolution"] == pytest.approx((2.5, 0.586, 0.586))
    assert effective["effective_pix_resolution_source"] == "native COPD spacing"
    assert opts["pix_resolution"] == pytest.approx((1.0, 1.0, 1.0))


def test_paper_hu_root_defaults_are_protocol_specific():
    assert runner.paper_hu_root_for_protocol("paper_hu_iso1mm", None) == runner.DEFAULT_PAPER_HU_ISO1MM_ROOT
    assert runner.paper_hu_root_for_protocol("paper_hu_native", None) == runner.DEFAULT_PAPER_HU_NATIVE_ROOT


def test_ptv_tmm_to_repo_field_axis_order_and_units():
    shape = (3, 4, 5)
    tmm = np.zeros((1, 3, *shape), dtype=np.float64)
    tmm[:, 0] = 2.0
    tmm[:, 1] = -3.0
    tmm[:, 2] = 4.0
    disp = runner.ptv_tmm_to_disp_vox_zyx(tmm, pix_resolution_zyx=(1.0, 1.5, 2.0))
    assert disp.shape == (*shape, 3)
    assert disp[0, 0, 0].tolist() == pytest.approx([2.0, -2.0, 2.0])

    dense_norm = runner.disp_vox_zyx_to_dense_norm_xyz(disp)
    assert dense_norm.shape == (*shape, 3)
    assert dense_norm[0, 0, 0].tolist() == pytest.approx([1.0, -2.0 / 1.5, 2.0])
    roundtrip = runner.c.dense_norm_xyz_to_vox_zyx(dense_norm)
    assert roundtrip == pytest.approx(disp)


def test_fixed_grid_pull_sign_matches_landmark_metric():
    shape = (5, 6, 7)
    tmm = np.zeros((1, 3, *shape), dtype=np.float64)
    tmm[:, 0] = 1.0
    tmm[:, 1] = 2.0
    tmm[:, 2] = 3.0
    disp = runner.ptv_tmm_to_disp_vox_zyx(tmm, pix_resolution_zyx=(1.0, 1.0, 1.0))
    fixed = np.asarray([[2.0, 2.0, 2.0], [1.0, 3.0, 4.0]], dtype=np.float64)
    moving = fixed + np.asarray([1.0, 2.0, 3.0], dtype=np.float64)[None, :]
    tre = runner.c.landmark_tre_from_dense_disp(
        disp,
        fixed,
        moving,
        sample_mode="trilinear",
        snap_output=False,
    )
    assert tre["tre_mean_mm"] == pytest.approx(0.0, abs=1e-6)


def test_native_crop_metrics_use_native_spacing(tmp_path):
    shape = (5, 6, 7)
    disp = np.zeros((*shape, 3), dtype=np.float32)
    disp[..., 0] = 1.0
    disp[..., 1] = 2.0
    disp[..., 2] = 3.0
    dense_norm = runner.disp_vox_zyx_to_dense_norm_xyz(disp)
    fixed = np.asarray([[2.0, 2.0, 2.0], [1.0, 3.0, 4.0]], dtype=np.float64)
    moving = fixed + np.asarray([1.0, 2.0, 3.0], dtype=np.float64)[None, :]
    case = argparse.Namespace(
        case_id=8,
        landmarks_fixed_crop=fixed,
        landmarks_moving_crop=moving,
        native_spacing_zyx=(2.5, 0.586, 0.586),
        fixed_mask=np.ones(shape, dtype=np.uint8),
    )
    metrics = runner.final_metrics_from_native_crop_disp(
        disp_vox_native_crop_zyx=disp,
        dense_norm_xyz=dense_norm,
        case=case,
        out_dir=tmp_path,
        save_dense_dvf=False,
        log_base="e",
    )
    assert metrics["native_crop_tre_snap_to_voxel"]["tre_mean_mm"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["tre_snap_to_voxel"]["tre_mean_mm"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["identity_tre"]["tre_mean_mm"] > 0.0
    assert metrics["registration_grid_spacing_zyx_mm"] == pytest.approx([2.5, 0.586, 0.586])


def test_external_import_path_resolves_register_pair_min():
    register_pair_min = runner.import_register_pair_min()
    assert callable(register_pair_min)
    module = importlib.import_module("ptv.register_min")
    assert getattr(module, "register_pair_min") is register_pair_min


def test_git_head_fallback_reads_regular_and_indirected_gitdir(tmp_path):
    gitdir = tmp_path / ".git"
    refdir = gitdir / "refs" / "heads"
    refdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    (refdir / "main").write_text("a" * 40 + "\n")
    assert runner.git_head_fallback(tmp_path) == ("a" * 40, "main")

    worktree = tmp_path / "submodule"
    real_gitdir = tmp_path / "modules" / "submodule"
    (real_gitdir / "refs" / "heads").mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: ../modules/submodule\n")
    (real_gitdir / "HEAD").write_text("ref: refs/heads/dev\n")
    (real_gitdir / "refs" / "heads" / "dev").write_text("b" * 40 + "\n")
    assert runner.git_head_fallback(worktree) == ("b" * 40, "dev")


def test_launcher_builds_expected_summary_path(tmp_path):
    args = argparse.Namespace(
        protocol="canonical_1mm",
        data_root=tmp_path / "data",
        raw_data_root=None,
        paper_hu_root=None,
        mask_root=None,
        out_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
        max_iters=1,
        admm_max_outer=1,
        admm_lbfgs_iters=3,
        dtype="float64",
        nlvl=1,
        verbose=False,
        save_dense_dvf=False,
        force=True,
        allow_mask_fallback=False,
    )
    job = launcher.build_job(args, case_id=1)
    assert job["protocol_tag"] == runner.PROTOCOL_CANONICAL
    assert str(job["summary_path"]).endswith("/summary.json")
    assert "--case-id" in job["cmd"]
    assert "--no-save-dense-dvf" in job["cmd"]
    assert "--nlvl" in job["cmd"]


def test_launcher_builds_native_protocol_with_mask_root(tmp_path):
    args = argparse.Namespace(
        protocol="paper_hu_native",
        data_root=tmp_path / "ptvprep",
        raw_data_root=tmp_path / "dirlab",
        paper_hu_root=None,
        mask_root=tmp_path / "masks",
        out_root=tmp_path / "runs",
        log_root=tmp_path / "logs",
        max_iters=1,
        admm_max_outer=1,
        admm_lbfgs_iters=1,
        dtype="float64",
        nlvl=1,
        verbose=False,
        save_dense_dvf=False,
        force=True,
        allow_mask_fallback=False,
    )
    job = launcher.build_job(args, case_id=8)
    assert job["protocol_tag"] == runner.PROTOCOL_PAPER_HU_NATIVE
    assert job["paper_hu_root"] == str(runner.DEFAULT_PAPER_HU_NATIVE_ROOT)
    assert job["mask_root"] == str(tmp_path / "masks")
    assert "--mask-root" in job["cmd"]
    assert "--case-id" in job["cmd"]


def test_aggregator_acceptance_flags_missing_or_invalid():
    rows = [
        {
            "case_id": 1,
            "native_crop_snap_tre_mean_mm": 1.0,
            "native_crop_sub_tre_mean_mm": 0.9,
            "reg1mm_snap_tre_mean_mm": 1.1,
            "reg1mm_sub_tre_mean_mm": 1.0,
            "fg_log_jdet_std": 0.1,
            "fg_jdet_le0_pct": 0.0,
            "fg_bending_energy": 0.0,
            "fg_diffusion_l2": 0.0,
            "dense_field_shape_matches_input": True,
            "dense_field_all_finite": True,
            "dense_field_nan_count": 0,
        }
    ]
    acc = aggregate.acceptance_from_rows(rows, [1, 2])
    assert acc["passed"] is False
    assert acc["missing_cases"] == [2]
    bad = dict(rows[0])
    bad["fg_log_jdet_std"] = float("nan")
    acc_bad = aggregate.acceptance_from_rows([bad], [1])
    assert acc_bad["passed"] is False
    assert "case01:fg_log_jdet_std" in acc_bad["invalid_items"]


def test_collect_rows_reads_protocol_summaries(tmp_path):
    tag = runner.PROTOCOL_CANONICAL
    summary_path = tmp_path / tag / "case01" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    payload = {
        "status": "completed",
        "dataset": "copd",
        "method": runner.METHOD,
        "protocol": "canonical_1mm",
        "case_id": 1,
        "run_name": "case01",
        "protocol_tag": tag,
        "native_crop_tre_snap_to_voxel": {"tre_mean_mm": 1.0},
        "native_crop_tre_subvoxel": {"tre_mean_mm": 0.9},
        "native_crop_identity_tre": {"tre_mean_mm": 20.0},
        "tre_snap_to_voxel": {"tre_mean_mm": 1.1},
        "tre_subvoxel": {"tre_mean_mm": 1.0},
        "identity_tre": {"tre_mean_mm": 21.0},
        "regularity_fg_registration_crop": {
            "log_jdet_std_fg": 0.1,
            "jdet_le0_pct_fg": 0.0,
            "bending_energy_fg": 0.0,
            "diffusion_l2_fg": 0.0,
        },
        "opts": {"opt_method": "admm"},
        "source_provenance": {"external_ptvreg_python_commit": "abc"},
        "input_metadata": {"input_protocol": "canonical_1mm"},
        "dense_field_shape_matches_input": True,
        "dense_field_all_finite": True,
        "dense_field_nan_count": 0,
    }
    summary_path.write_text(json.dumps(payload) + "\n")
    rows = aggregate.collect_rows(tmp_path, "canonical_1mm", [1])
    assert len(rows) == 1
    assert rows[0]["native_crop_snap_tre_mean_mm"] == pytest.approx(1.0)
