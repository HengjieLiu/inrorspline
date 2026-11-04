"""Lightweight protocol tests for the P71 bounded-three-map extension."""
from __future__ import annotations

import runpy
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "scripts/experiments/oasis_alpha_sweep/run_oasis_bspline_depth_bounded3_first10.py"
mod = runpy.run_path(str(RUNNER), run_name="p71_test_module")


def test_extension_is_one_bounded_three_map_arm():
    config = mod["CONFIGS"]["bounded_3_compose"]
    assert config["bounded"] is True
    assert config["maps_per_level"] == 3
    assert "three_map_composition" in config["method"]
    assert mod["VARIANT"].endswith("bounded3_depth_extension_v1")


def test_depths_have_unique_three_map_run_names():
    base = mod["base"]
    original = (base.CONFIGS, base.CONFIG_ORDER, base.VARIANT, base.PROTOCOL)
    try:
        mod["configure_base"]()
        names = [base.run_name(0, depth, "bounded_3_compose") for depth in range(1, 5)]
        assert len(set(names)) == 4
        for depth in range(1, 5):
            assert len(base.cell_protocol(depth, "bounded_3_compose")["cps_schedule"]) == depth
            assert base.cell_protocol(depth, "bounded_3_compose")["stage_alpha"] == [1.0] * depth
            assert depth * base.CONFIGS["bounded_3_compose"]["maps_per_level"] == depth * 3
    finally:
        base.CONFIGS, base.CONFIG_ORDER, base.VARIANT, base.PROTOCOL = original


def test_combined_report_method_order_is_five_arms_then_two_references():
    assert len(mod["REPORT_METHODS"]) == 5
    assert mod["REPORT_METHODS"][-1] == "bounded_3_compose"
    assert mod["METRICS"] == ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
