from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "experiments" / "oasis_alpha_sweep"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("oasis842dense_test_runner", "run_one.py")
launch = load_module("oasis842dense_test_launch", "launch_oasis_mrdbscp842_dense_lambda_sweep.py")


def make_job(alpha: float, gpu: str | None = "0"):
    return launch.Job(0, alpha, gpu, 0, 158, "0333", "0115")


def test_method_identity_and_unique_dense_lambda_names() -> None:
    assert launch.METHOD in runner.METHODS
    assert runner.method_cps(launch.METHOD) is None
    names = {launch.run_name(make_job(alpha)) for alpha in (10, 100, 1000, 10000, 100000)}
    assert len(names) == 5
    assert all("sched8-4-2-dense" in name for name in names)


def test_dense_command_keeps_prefix_fixed_and_exposes_dense_schedule() -> None:
    args = argparse.Namespace(
        eval_interval=250,
        seed=1,
        out_root=Path("/tmp/runs"),
        oasis_root=Path("/tmp/oasis"),
        pair_file=Path("/tmp/pairs.txt"),
        overwrite=False,
    )
    command = launch.build_command(args, make_job(10.0), "3", "abc123-dirty")
    joined = " ".join(command)
    assert "--stage-full-cps 8,4,2" in joined
    assert "--stage-alpha 1000.0,1000.0,1000.0" in joined
    assert "--dense-stage-alpha 10" in joined
    assert "--dense-stage-lr 0.001" in joined
    assert "--dense-stage-epochs 1300" in joined
    assert "--dense-stage-early-stop-mode adaptive" in joined
    assert "--dense-stage-min-epochs 130" in joined
    assert "--dense-stage-patience 50" in joined
    assert "--dense-stage-check-interval 25" in joined
    assert "--dense-stage-rel-delta-total 0.0001" in joined
    assert "--dense-stage-rel-delta-data 5e-05" in joined
    assert "--cudnn-benchmark false --cudnn-deterministic true" in joined
    assert "--source-revision abc123-dirty" in joined


def test_default_run_name_encodes_dense_stopping_and_lambda() -> None:
    args = argparse.Namespace(
        method=launch.METHOD,
        run_name=None,
        case_rank=0,
        stage_full_cps="8,4,2",
        stage_downsample="4,2,1",
        stage_epochs="400,800,1300",
        stage_alpha="1000,1000,1000",
        stage_early_stop_mode="adaptive",
        stage_min_epochs="40,80,130",
        stage_patience="20,35,50",
        alpha=1000.0,
        epochs=3800,
        dct_size=32,
        dense_stage_alpha=10.0,
        dense_stage_lr=1e-3,
        dense_stage_epochs=1300,
        dense_stage_early_stop_mode="adaptive",
        dense_stage_min_epochs=130,
        dense_stage_patience=50,
        dense_stage_check_interval=25,
        dense_stage_rel_delta_total=1e-4,
        dense_stage_rel_delta_data=5e-5,
    )
    name = runner.build_run_name(args, 0.01)
    assert "sched8-4-2" in name
    assert "sa1000-1000-1000" in name
    assert "dense_e1300_a10_lr0p001_esadaptive_min130_pat50_ci25" in name


def test_dense_optimizer_uses_full_global_ncc_and_unmasked_fd_be(monkeypatch) -> None:
    shape = (5, 6, 7)
    monkeypatch.setattr(runner.c, "IMAGE_SHAPE", shape)
    calls = {"ncc": 0, "be": 0}
    original_ncc = runner.c.ncc_loss
    original_be = runner.c.forward_valid_be

    def checked_ncc(fixed, warped):
        assert tuple(fixed.shape) == (1, 1, *shape)
        assert tuple(warped.shape) == (1, 1, *shape)
        calls["ncc"] += 1
        return original_ncc(fixed, warped)

    def checked_be(disp, mask, *, component_reduction):
        assert tuple(disp.shape) == (1, 3, *shape)
        assert mask is None
        assert component_reduction == "mean"
        calls["be"] += 1
        return original_be(disp, mask, component_reduction=component_reduction)

    monkeypatch.setattr(runner.c, "ncc_loss", checked_ncc)
    monkeypatch.setattr(runner.c, "forward_valid_be", checked_be)
    torch.manual_seed(7)
    fixed = torch.rand(shape)
    moving = torch.rand(shape)
    initial = torch.zeros((*shape, 3))
    args = SimpleNamespace(
        alpha=1000.0,
        dense_stage_alpha=10.0,
        dense_stage_lr=1e-3,
        dense_stage_epochs=1,
        dense_stage_min_epochs=1,
        dense_stage_patience=1,
        dense_stage_check_interval=1,
        dense_stage_rel_delta_total=1e-4,
        dense_stage_rel_delta_data=5e-5,
        dense_stage_early_stop_mode="none",
        eval_interval=1,
        method=launch.METHOD,
    )
    dense, summary, rows = runner.optimize_dense_stage(
        args, fixed, moving, initial, 3, 0, 1, sampler=None, timer=None
    )
    assert tuple(dense.shape) == (*shape, 3)
    assert torch.isfinite(dense).all()
    assert calls == {"ncc": 1, "be": 1}
    assert summary["representation"] == "dense_voxel"
    assert summary["bspline_order"] is None
    assert summary["regularizer"] == "unmasked_forward_valid_dense_fd_be"
    assert all(float(value) == 0.0 for key, value in summary.items() if key.startswith("dense_init_"))
    assert len(rows) == 1
