from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_adapter():
    path = REPO_ROOT / "scripts/data/preprocessing/lungct/prepare_dual_author_4dct.py"
    spec = importlib.util.spec_from_file_location(
        "test_prepare_dual_author_4dct_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adapter = load_adapter()


def test_author_raw_image_transform_matches_released_converter(tmp_path: Path) -> None:
    shape_zyx = (2, 2, 3)
    raw = np.arange(np.prod(shape_zyx), dtype=np.int16).reshape(shape_zyx) + 1000
    path = tmp_path / "phase.img"
    raw.tofile(path)

    observed = adapter.author_image_from_raw(path, shape_zyx)
    expected = np.ascontiguousarray(np.flip(raw - 1024, axis=0))

    assert observed.dtype == np.int16
    np.testing.assert_array_equal(observed, expected)


def test_author_landmark_transform_retains_xy_and_flips_z() -> None:
    source = np.asarray([[1.0, 2.0, 3.0], [9.0, 8.0, 10.0]])

    observed = adapter.author_landmarks_xyz(source, (20, 30, 12))

    np.testing.assert_allclose(observed, [[1.0, 2.0, 9.0], [9.0, 8.0, 2.0]])
    np.testing.assert_array_equal(source, [[1.0, 2.0, 3.0], [9.0, 8.0, 10.0]])


def test_dual_union_crop_uses_xyz_order_both_phases_and_padding_two() -> None:
    shape_xyz = (9, 10, 11)
    fixed_xyz = np.zeros(shape_xyz, dtype=np.uint8)
    moving_xyz = np.zeros(shape_xyz, dtype=np.uint8)
    fixed_xyz[2:5, 3:6, 4:7] = 10
    moving_xyz[4:8, 1:4, 6:10] = 14
    # Adapter inputs are SimpleITK arrays in zyx; DUAL swaps 0 and 2 on load.
    fixed_zyx = np.swapaxes(fixed_xyz, 0, 2)
    moving_zyx = np.swapaxes(moving_xyz, 0, 2)

    bbox = adapter.dual_union_bbox_xyz(fixed_zyx, moving_zyx)

    assert [(item.start, item.stop) for item in bbox] == [(0, 9), (0, 8), (2, 11)]
    assert adapter.bbox_payload(bbox) == {
        "start_xyz": [0, 0, 2],
        "stop_xyz_exclusive": [9, 8, 11],
        "shape_xyz": [9, 8, 9],
    }


def test_identity_tre_uses_xyz_spacing_and_population_std() -> None:
    fixed = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    moving = np.asarray([[1.0, 0.0, 0.0], [1.0, 4.0, 3.0]])

    stats = adapter.identity_tre_mm(fixed, moving, (2.0, 3.0, 4.0))

    # Per-point distances are 2 and 6 mm.
    assert stats["mean_mm"] == pytest.approx(4.0)
    assert stats["std_mm_population"] == pytest.approx(2.0)
    assert stats["count"] == 2


def test_checksum_manifest_detects_mutation(tmp_path: Path) -> None:
    (tmp_path / "case_01").mkdir()
    payload = tmp_path / "case_01" / "value.txt"
    payload.write_text("original\n")
    adapter.write_checksums(tmp_path)

    assert adapter.verify_checksums(tmp_path)["status"] == "pass"

    payload.write_text("changed\n")
    audit = adapter.verify_checksums(tmp_path)
    assert audit["status"] == "fail"
    assert audit["failures"] == [
        {"path": "case_01/value.txt", "reason": "sha256 mismatch"}
    ]


def test_checksum_manifest_detects_unlisted_file(tmp_path: Path) -> None:
    (tmp_path / "listed.txt").write_text("listed\n")
    adapter.write_checksums(tmp_path)
    (tmp_path / "late.txt").write_text("late\n")

    audit = adapter.verify_checksums(tmp_path)

    assert audit["status"] == "fail"
    assert audit["failures"] == [
        {"path": "late.txt", "reason": "not listed in checksums.sha256"}
    ]


def test_totalsegmentator_model_provenance_pins_configs_and_checkpoint(
    tmp_path: Path,
) -> None:
    trainer = (
        tmp_path
        / "Dataset291_TotalSegmentator_part1_test"
        / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
    )
    (trainer / "fold_0").mkdir(parents=True)
    (trainer / "dataset.json").write_text("{}\n")
    (trainer / "plans.json").write_text("{}\n")
    (trainer / "fold_0" / "checkpoint_final.pth").write_bytes(b"weights")

    provenance = adapter.totalsegmentator_model_provenance(
        tmp_path,
        require_complete=True,
        hash_files=True,
        task_ids=(291,),
    )

    assert provenance["status"] == "complete"
    assert provenance["resolved_tasks"][0]["task_id"] == 291
    checkpoint = provenance["resolved_tasks"][0]["files"]["checkpoint_fold_0"]
    assert checkpoint["bytes"] == 7
    assert checkpoint["sha256"] == adapter.sha256_file(
        trainer / "fold_0" / "checkpoint_final.pth"
    )


def test_versioned_output_is_refused_inside_git_tree() -> None:
    with pytest.raises(ValueError, match="outside Git repository"):
        adapter.validate_output_location(REPO_ROOT / "outputs", "dual_author_v001")


def test_post_publication_finalizer_pins_script_permissions_and_checksums(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact_v001"
    root.mkdir(mode=0o700)
    (root / "dataset_manifest.json").write_text("{}\n")
    executed_source = tmp_path / "executed.py"
    executed_source.write_text("print('executed')\n")
    executed = adapter.install_executed_script_provenance(root, executed_source)
    adapter.write_json(
        root / "preprocess_contract.json",
        {"reference": {"adapter_script_sha256": executed["sha256"]}},
    )
    adapter.write_checksums(root)

    result = adapter.finalize_existing_artifact(root)

    assert result["status"] == "pass"
    assert result["permission_audit"]["root_mode"] == "0755"
    assert result["permission_audit"]["group_or_other_writable_paths"] == []
    assert result["executed_script_provenance"]["mode"] == "0444"
    assert result["checksum_audit"]["status"] == "pass"
    assert (root / "provenance" / "post_publication_finalization.json").is_file()


def test_case_ids_are_unique_and_limited_to_4dct_cohort() -> None:
    assert adapter.parse_and_validate_case_ids("1-3,8") == [1, 2, 3, 8]
    with pytest.raises(ValueError, match="Duplicate"):
        adapter.parse_and_validate_case_ids("1,1")
    with pytest.raises(ValueError, match=r"\[1, 10\]"):
        adapter.parse_and_validate_case_ids("0,11")
