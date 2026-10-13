from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


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


common = load_module("oasis8421_test_common", "common.py")
runner = load_module("oasis8421_test_runner", "run_one.py")
launch = load_module("oasis8421_test_launch", "launch_oasis_mrdbscp8421_lambda_sweep.py")


def make_job(cps1_alpha: float, preferred_gpu: str | None = "0"):
    return launch.Job(
        job_index=0,
        cps1_alpha=cps1_alpha,
        stage_alpha=(1000.0, 1000.0, 1000.0, cps1_alpha),
        preferred_gpu=preferred_gpu,
        case_rank=0,
        original_case_index=17,
        fixed_id="0333",
        moving_id="0115",
    )


def test_stage_alpha_falls_back_to_global_alpha_and_validates_length() -> None:
    assert common.parse_float_list("", 4, 1000.0) == [1000.0] * 4
    assert common.parse_float_list("1000,1000,1000,10", 4, 0.0) == [1000.0, 1000.0, 1000.0, 10.0]
    try:
        common.parse_float_list("1000,10", 4, 0.0)
    except ValueError as exc:
        assert "Expected 4 stage float values" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("A mismatched stage-alpha list must fail")


def test_8421_schedule_is_exactly_nested_and_reports_cps1() -> None:
    shapes = common.build_nested_cp_shapes(common.IMAGE_SHAPE, base_spacing=8, spacings=[8, 4, 2, 1])
    common.assert_nested_schedule(shapes, [8, 4, 2, 1])
    assert shapes == {8: (24, 32, 28), 4: (44, 60, 52), 2: (84, 116, 100), 1: (164, 228, 196)}
    assert runner.method_cps(launch.METHOD) == 1


def test_stage_alpha_is_part_of_default_run_name() -> None:
    args = argparse.Namespace(
        method=launch.METHOD,
        run_name=None,
        case_rank=0,
        stage_full_cps="8,4,2,1",
        stage_downsample="4,2,1,1",
        stage_epochs="400,800,1300,1300",
        stage_alpha="1000,1000,1000,10",
        stage_early_stop_mode="adaptive",
        stage_min_epochs="40,80,130,130",
        stage_patience="20,35,50,50",
        alpha=1000.0,
        epochs=3800,
        dct_size=32,
    )
    name = runner.build_run_name(args, 0.01)
    assert "sched8-4-2-1" in name
    assert "sa1000-1000-1000-10" in name


def test_candidate_names_are_unique_and_commands_pin_complete_schedule() -> None:
    names = {launch.run_name(make_job(alpha)) for alpha in (10.0, 100.0, 1000.0, 10000.0, 100000.0)}
    assert len(names) == 5
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
    assert "--gpu 3" in joined
    assert "--stage-alpha 1000.0,1000.0,1000.0,10.0" in joined
    assert "--stage-lr 0.03,0.01,0.003,0.001" in joined
    assert "--stage-early-stop-mode adaptive" in joined
    assert "--cudnn-benchmark false --cudnn-deterministic true" in joined
    assert "--source-revision abc123-dirty" in joined
