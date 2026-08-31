"""Pure-contract tests for the CT-only FireANTs Learn2Breath audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "experiments" / "learn2breath_fireants_greedy" / "run_pairlocal.py"
SCHEDULER_PATH = ROOT / "scripts" / "experiments" / "learn2breath_fireants_greedy" / "run_all_two_window_audit.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runner():
    return load_module("fireants_pairlocal_test", RUNNER_PATH)


def scheduler():
    return load_module("fireants_scheduler_test", SCHEDULER_PATH)


def test_both_physical_hu_normalizers_are_explicit_and_bounded():
    module = runner()
    values = np.asarray([-1200.0, -1000.0, -920.0, -510.0, -100.0, 200.0, 400.0], dtype=np.float32)
    p2e = module.normalize_ct_hu(values.reshape(1, 1, -1), "p2e_equiv_hu").ravel()
    lung = module.normalize_ct_hu(values.reshape(1, 1, -1), "l2b_lung_hu").ravel()
    assert np.allclose(p2e, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    assert np.allclose(lung, [0.0, 0.0, 80.0 / 1200.0, 490.0 / 1200.0, 0.75, 1.0, 1.0])
    assert not module.ct_only_config("p2e_equiv_hu")["intensity"]["legacy_direct_stored_80_900_hu_normalizer_used"]


def test_ct_only_config_excludes_evaluation_assets_from_registration_input():
    config = runner().ct_only_config("l2b_lung_hu")
    assert config["ct_only"] is True
    assert config["registration_inputs"] == "in_memory_scalar_ct_only"
    assert config["registration"]["scales"] == [16, 8, 4, 2, 1]
    assert config["registration"]["iterations"] == [200] * 5
    assert set(config["excluded_from_registration_inputs_and_loss"]) == {
        "copd_landmarks", "r231_masks", "validation_lobe_labels"
    }


def test_fireants_field_axes_and_channel_layout_contract():
    module = runner()
    native = np.zeros((4, 3, 2, 3), dtype=np.float32)  # Z,Y,X,component XYZ.
    native[..., 0] = 1.0
    native[..., 1] = 2.0
    native[..., 2] = 3.0
    candidates = module.channel_last_candidates(native, (2, 3, 4))
    field = candidates["transpose_zyx_to_xyz_xyz"]
    assert field.shape == (2, 3, 4, 3)
    assert np.all(field[..., 0] == 1.0)
    field_cxyz = module.field_cxyz_from_xyzc(np.zeros((224, 192, 224, 3), dtype=np.float32))
    assert field_cxyz.shape == (3, 224, 192, 224)
    assert field_cxyz.dtype == np.float32


def test_four_lane_manifest_contract_and_combined_completeness_gate():
    module = scheduler()
    assert module.parse_gpu_ids("0,1,2,3") == (0, 1, 2, 3)
    with pytest.raises(module.AuditError):
        module.parse_gpu_ids("0,0,1,2")
    rows = []
    for case_id in module.COPD_CASES:
        for method in ("FireANTs Greedy NCC", "MR-D-BSCP P2e"):
            for profile in module.PROFILES:
                rows.append({"case_id": case_id, "method": method, "intensity_profile": profile})
    module.required_combined_keys(rows, "copd")
    with pytest.raises(module.AuditError):
        module.required_combined_keys(rows[:-1], "copd")


def test_validation_completeness_requires_all_lobe_dice():
    module = scheduler()
    rows = []
    for case_id in module.VALIDATION_CASES:
        for method in ("FireANTs Greedy NCC", "MR-D-BSCP P2e"):
            for profile in module.PROFILES:
                row = {"case_id": case_id, "method": method, "intensity_profile": profile}
                row.update({f"dice_lobe_{label}": 0.9 for label in (8, 16, 32, 64, 128)})
                rows.append(row)
    module.required_combined_keys(rows, "validation")
    rows[0]["dice_lobe_64"] = np.nan
    with pytest.raises(module.AuditError):
        module.required_combined_keys(rows, "validation")
