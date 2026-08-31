"""Contract tests for the full-grid Learn2Breath P2e adapter."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "learn2breath_mrdbscp_p2e"
    / "run_pairlocal.py"
)
SPEC = spec_from_file_location("learn2breath_mrdbscp_p2e", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)

LAUNCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "learn2breath_mrdbscp_p2e"
    / "run_all_two_window_audit.py"
)
LAUNCHER_SPEC = spec_from_file_location("learn2breath_mrdbscp_p2e_all", LAUNCHER_PATH)
assert LAUNCHER_SPEC is not None and LAUNCHER_SPEC.loader is not None
LAUNCHER = module_from_spec(LAUNCHER_SPEC)
sys.modules[LAUNCHER_SPEC.name] = LAUNCHER
LAUNCHER_SPEC.loader.exec_module(LAUNCHER)


def test_historical_raw_window_is_exact_on_physical_hu_input():
    values = np.asarray([-1500.0, -920.0, -510.0, -100.0, 400.0], dtype=np.float32)
    observed = RUNNER.normalize_p2e_ct(values)
    np.testing.assert_allclose(observed, np.asarray([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32))


def test_learn2breath_lung_window_is_exact_on_physical_hu_input():
    values = np.asarray([-1200.0, -1000.0, -400.0, 200.0, 500.0], dtype=np.float32)
    observed = RUNNER.normalize_p2e_ct(values, "l2b_lung_hu")
    np.testing.assert_allclose(observed, np.asarray([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32))
    profile = RUNNER.intensity_profile_config("p2e_equiv_hu")
    assert profile["low_hu"] == -920.0
    assert profile["high_hu"] == -100.0


def test_locked_p2e_schedule_has_expected_nested_control_lattices():
    assert RUNNER.STAGE_FULL_CPS == (64, 32, 16, 8, 4)
    assert RUNNER.STAGE_DOWNSAMPLE == (16, 8, 4, 2, 1)
    shapes = RUNNER.p2e.build_nested_cp_shapes((224, 192, 224), 64, RUNNER.STAGE_FULL_CPS)
    assert shapes == {64: (8, 7, 8), 32: (12, 10, 12), 16: (20, 16, 20), 8: (36, 28, 36), 4: (68, 52, 68)}
    RUNNER.p2e.assert_nested_schedule(shapes, RUNNER.STAGE_FULL_CPS)


def test_field_layout_round_trip_preserves_xyz_voxel_components():
    field = np.zeros((3, 7, 6, 5), dtype=np.float32)
    field[0] = 1.25
    field[1] = -0.75
    field[2] = 0.5
    normalised = RUNNER.field_cxyz_to_normalized_zyx(field)
    restored = RUNNER.normalized_zyx_to_field_cxyz(normalised)
    np.testing.assert_allclose(restored, field, rtol=0.0, atol=1.0e-6)
    points = np.asarray([[0.0, 0.0, 0.0], [6.0, 5.0, 4.0], [2.25, 1.5, 3.75]])
    np.testing.assert_allclose(
        RUNNER.sample_field_trilinear_xyz(restored, points),
        np.broadcast_to([1.25, -0.75, 0.5], (len(points), 3)),
    )


def test_challenge_flow_adapter_preserves_xyz_channel_order():
    field = np.zeros((3, 7, 6, 5), dtype=np.float32)
    field[0] = 1.25
    field[1] = -0.75
    field[2] = 0.5
    challenge_flow = RUNNER.challenge_flow_xyzc(field)
    assert challenge_flow.shape == (7, 6, 5, 3)
    np.testing.assert_allclose(challenge_flow[2, 3, 4], [1.25, -0.75, 0.5])


def test_interpolation_audit_records_historical_boundary_difference():
    audit = RUNNER.interpolation_audit()
    assert audit["historical_align_corners"] == 1.0
    assert audit["learn2breath_evaluator_align_corners"] == 0.0
    assert audit["synthetic_max_abs_difference"] > 0.0
    assert audit["synthetic_center_abs_difference"] == 0.0


def test_dedicated_manifest_has_four_fixed_ten_case_lanes():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        args = SimpleNamespace(
            out_root=root / "outputs",
            copd_data_root=root / "copd",
            validation_data_root=root / "validation",
            r231_root=root / "r231",
            learn2breath_repo=root / "learn2breath",
            gpus="0,1,2,3",
            force=False,
        )
        manifest = LAUNCHER.build_manifest(args)
        assert manifest["job_count"] == 40
        for lane in range(4):
            jobs = [job for job in manifest["jobs"] if job["lane"] == lane]
            assert len(jobs) == 10
            assert jobs[0]["temperature"] == "cold"
            assert all(job["temperature"] == "warm" for job in jobs[1:])
