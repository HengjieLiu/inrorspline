"""CPU contracts for the independent frozen-VFA dense refinement runner."""
import dataclasses
import itertools
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments/oasis_alpha_sweep"))
import common as c
import vfa_feature_refinement as f
import launch_oasis_vfa_feature_dense as launcher
import run_oasis_vfa_feature_dense_one as worker


def fake_row(config, dice=.78, hd95=2.2, asd=.48):
    return {"config": dataclasses.asdict(config), "dice": dice, "hd95": hd95, "asd": asd}


def test_search_counts_and_unique_names():
    a = launcher.phase_a()
    rows_a = [fake_row(x) for x in a]
    b = launcher.phase_b(rows_a)
    rows_ab = rows_a + [fake_row(x) for x in b]
    cc = launcher.phase_c(rows_ab)
    controls = launcher.controls(rows_ab, min(rows_ab, key=f.selection_key))
    assert list(map(len, (a, b, cc, controls))) == [48, 24, 20, 4]
    names = [f.config_name(x, 0) for x in a + b + cc + controls]
    assert len(set(names)) == 96
    assert f.config_name(a[0], 1) != names[0]
    assert f.config_name(dataclasses.replace(a[0], max_epochs=1, min_epochs=0), 0) != names[0]


def test_match_and_selection():
    good = fake_row(f.FitConfig(), **f.REFERENCE)
    poor = fake_row(f.FitConfig(), dice=.82, hd95=4, asd=.4)
    assert f.metric_deficit(good) == 0
    assert f.selection_key(good) < f.selection_key(poor)
    better = {**good, "dice": good["dice"] + .00001}
    assert f.selection_key(better) < f.selection_key(good)


def test_cli_and_legacy_prefix():
    args = worker.parse_args(["--oasis-root", "/tmp/data", "--dl-root", "/tmp/dl", "--vfa-root", "/tmp/vfa",
                              "--checkpoint", "/tmp/model", "--cache-dir", "/tmp/cache", "--out-dir", "/tmp/out",
                              "--feature-order", "project_then_warp", "--feature-loss", "local_ncc", "--alpha", "3"])
    assert worker.config_from_args(args).alpha == 3
    assert worker.config_from_args(args).loss == "local_ncc"
    prefix = f.prefix_args()
    assert prefix.stage_full_cps == "8,4,2"
    assert prefix.stage_alpha == "1000,1000,1000"
    assert prefix.stage_lr == "0.03,0.01,0.003"
    assert prefix.method == "direct_cp_8_4_2_be_bspline_analytic_multistage"
    for kwargs in ({"alpha": -1}, {"lr": 0}, {"alpha": float("nan")}, {"max_epochs": 0}, {"support": "seg"}):
        with unittest.TestCase().assertRaises(ValueError):
            dataclasses.replace(f.FitConfig(), **kwargs).validate()


def test_voxel_warp_axes_and_border():
    shape = (7, 8, 9)
    identity = c.make_coordinate_grid(shape, torch.device("cpu"))
    field = torch.zeros(*shape, 3)
    image = torch.arange(shape[2]).float()[None, None, None, None].expand(1, 1, *shape)
    field[..., 0] = 1
    warped = f.warp_features(image, field, identity)
    assert torch.allclose(warped[..., :-1], image[..., 1:], atol=1e-5)
    assert torch.allclose(warped[..., -1], image[..., -1], atol=1e-5)
    field.zero_()
    assert torch.allclose(f.warp_features(image, field, identity), image, atol=1e-5)


def test_diffusion_voxel_units_component_and_axis_means():
    shape = (7, 8, 9)
    field = torch.zeros(*shape, 3)
    field[..., 0] = torch.arange(shape[2]).float()[None, None]
    assert math.isclose(float(f.diffusion(field)), 1/9, rel_tol=1e-6)
    assert f.diffusion(field + 100) == f.diffusion(field)
    assert f.diffusion(torch.zeros_like(field)) == 0


def test_loss_identity_and_finite_gradients():
    for loss in ("mse", "global_ncc", "unit_mse", "local_ncc"):
        torch.manual_seed(1)
        a = torch.randn(1, 3, 11, 12, 13)
        b = a.clone().requires_grad_(True)
        value = f.feature_loss(a, b, loss)
        assert abs(float(value)) < 1e-5
        value.backward()
        assert torch.isfinite(b.grad).all()


def test_full_volume_and_masked_objectives():
    a, b = torch.zeros(1, 2, 5, 5, 5), torch.ones(1, 2, 5, 5, 5)
    mask = torch.zeros(5, 5, 5)
    mask[2] = 1
    b[:, :, 2] = 0
    assert math.isclose(float(f.feature_loss(a, b, "mse")), .8, rel_tol=1e-6)
    assert f.feature_loss(a, b, "mse", mask) == 0
    assert float(f.feature_loss(torch.ones_like(a), -torch.ones_like(a), "global_ncc")) == 1


def fake_cache():
    torch.manual_seed(1)
    shape = (7, 8, 9)
    project = torch.nn.Conv3d(16, 8, 3, padding=1)
    moving = torch.randn(1, 16, *shape)
    with torch.no_grad():
        projected = f.moving_project(project, moving)
    return {"prefix_voxel_xyz": torch.zeros(*shape, 3), "projection_state": project.state_dict(),
            "moving_encoder": moving, "moving_projected": projected,
            "fixed_projected": projected + torch.randn_like(projected) * .1,
            "fixed_mask": torch.ones(shape), "fixed_image": torch.rand(shape), "moving_image": torch.rand(shape)}


def test_projection_order_and_frozen_gradients():
    cache = fake_cache()
    first = f.FeatureObjective(cache, f.FitConfig(), torch.device("cpu"))
    second = f.FeatureObjective(cache, f.FitConfig(order="project_then_warp"), torch.device("cpu"))
    # Identity agrees within interpolation rounding; nonconstant displacement
    # does not commute with a spatial convolution.
    assert torch.allclose(first.raw(first.initial), second.raw(second.initial), atol=1e-6)
    field = (torch.randn_like(first.initial) * .3).requires_grad_(True)
    assert not torch.allclose(first.raw(field), second.raw(field), atol=1e-5)
    first(field)[0].backward()
    assert torch.isfinite(field.grad).all() and field.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in first.project.parameters())


def test_best_checkpoint_matches_its_loss_and_legacy_untouched():
    with tempfile.TemporaryDirectory(prefix="vfa-feature-test-") as directory:
        _check_checkpoint(Path(directory))


def _check_checkpoint(tmp_path):
    cfg = f.FitConfig(max_epochs=2, min_epochs=0, lr=.01)
    objective = f.FeatureObjective(fake_cache(), cfg, torch.device("cpu"))
    original = objective.initial.clone()
    result, summary = f.optimize(objective, cfg, tmp_path)
    state = torch.load(tmp_path / "best_checkpoint.pt", weights_only=False)
    assert torch.equal(objective.initial, original)
    assert torch.equal(state["dense_voxel_xyz"], result)
    assert math.isclose(float(objective(result)[0]), summary["best_loss"], abs_tol=1e-7)
    assert summary["optimizer_steps"] == 2
    assert state["cps"] is None and state["control_shape"] is None


def test_plateau_requires_post_reduction_patience():
    cfg = f.FitConfig()
    rows = [{"epoch": i, "loss": 1., "data_loss": 1.} for i in range(301)]
    assert f.plateau(rows, cfg, 0)
    assert not f.plateau(rows, cfg, 250)
    rows.extend({"epoch": i, "loss": 1., "data_loss": 1.} for i in range(301, 351))
    assert f.plateau(rows, cfg, 250)


def test_foreground_fallback_does_not_change_regularizer():
    cache = fake_cache()
    objective = f.FeatureObjective(cache, f.FitConfig(support="fixed_positive"), torch.device("cpu"))
    field = torch.randn_like(objective.initial)
    _, _, regularizer = objective(field)
    assert regularizer == f.diffusion(field)


def test_ncc_keeps_sign_and_local_boundary_counts():
    torch.manual_seed(4)
    a = torch.randn(1, 2, 11, 12, 13)
    assert float(f.feature_loss(a, -a, "global_ncc")) > 1.999
    assert float(f.feature_loss(a, -a, "local_ncc")) > 1.999
    # Native projection central crop equals a convolution with replicate
    # boundaries (the outer zeros of the padded conv never affect the crop).
    project = torch.nn.Conv3d(2, 3, 3, padding=1)
    expected = torch.nn.functional.conv3d(torch.nn.functional.pad(a, (1, 1, 1, 1, 1, 1), mode="replicate"),
                                         project.weight, project.bias)
    assert torch.allclose(f.moving_project(project, a), expected, atol=1e-6)


def test_scheduler_refuses_incomplete_results(tmp_path=None):
    with tempfile.TemporaryDirectory(prefix="vfa-scheduler-test-") as directory:
        with unittest.TestCase().assertRaises(RuntimeError):
            launcher.validate_result(Path(directory), launcher.job(f.FitConfig()))


def test_complete_report_pipeline_and_holdout_means():
    from types import SimpleNamespace
    from unittest.mock import patch
    with tempfile.TemporaryDirectory(prefix="vfa-report-test-") as directory:
        root = Path(directory)
        cfg = f.FitConfig()
        features, controls = [], []
        paths = {}
        for rank in range(10):
            path = root / f"cache{rank}"
            path.mkdir()
            paths[rank] = path
            prepared = {"vfa_metrics": f.REFERENCE, "prefix_metrics": {"dice": .75, "hd95": 2.7, "asd": .6},
                        "fixed_id": "0333", "moving_id": "0115", "prefix_elapsed_sec": 1.}
            c.write_json(path / "summary.json", prepared)
            for collection, config in ((features, cfg), (controls, dataclasses.replace(cfg, loss="raw_ncc"))):
                row = fake_row(config, **f.REFERENCE)
                row.update(run_name=f.config_name(config, rank), phase="A" if rank == 0 else "confirmation",
                           case_rank=rank, fixed_id="0333", moving_id="0115", matches_vfa=True,
                           metric_deficit=0., best_epoch=100, gpu="0", prefix_cache=str(path), run_dir=str(root),
                           deltas_vfa={k: 0. for k in f.REFERENCE})
                collection.append(row)
        args = SimpleNamespace(review_root=root / "reports")
        with patch.object(launcher, "historical_be_row", return_value=None):
            launcher.report_search(args, [features[0], controls[0]], features[0], controls[0])
            launcher.report_cohort(args, SimpleNamespace(cached=lambda rank: paths[rank]), features, controls)
        report = c.read_json(args.review_root / "first10_report.json")
        assert len(report["case_metrics"]) == 40
        assert len(report["means"]) == 8
        holdout = [r for r in report["means"] if r["method"] == "Dense VFA features" and r["subset"] == "ranks_1_9"][0]
        assert holdout["n"] == 9 and holdout["cases_matching_all"] == 9
        assert holdout["mean_matches_all"]
        assert len(c.read_json(args.review_root / "rank0_search.json")["selected_comparison"]) == 5
        assert (args.review_root / "ACCEPTED.json").exists()


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in list(globals().items())
                               if name.startswith("test_") and callable(value))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
