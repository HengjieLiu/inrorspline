from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.experiments.lungct_dual_inr_author.posthoc_author_regularity as posthoc
from scripts.experiments.lungct_dual_inr_author.posthoc_author_regularity import (
    RegularityError,
    compute_common_regularity,
    generate_posthoc_reports,
    validate_author_geometry,
    xyz_field_to_common_zyx,
)


def test_zero_and_translation_fields_preserve_common_convention_and_metrics():
    shape_xyz = (5, 6, 7)
    mask = np.ones(shape_xyz, dtype=bool)
    spacing_xyz = (0.7, 0.8, 2.5)

    zero = np.zeros(shape_xyz + (3,), dtype=np.float32)
    zero_result = compute_common_regularity(
        disp_vox_xyz=zero,
        fixed_lobe_union_xyz=mask,
        spacing_xyz_mm=spacing_xyz,
    )
    zero_metrics = zero_result["metrics"]
    assert zero_result["conversion"]["target_shape_zyx"] == [7, 6, 5]
    assert zero_metrics["diffusion_l2_full"] == pytest.approx(0.0)
    assert zero_metrics["bending_energy_full"] == pytest.approx(0.0)
    assert zero_metrics["jdet_mean_full"] == pytest.approx(1.0)
    assert zero_metrics["jdet_le0_pct_full"] == pytest.approx(0.0)
    assert zero_metrics["log_jdet_std_full"] == pytest.approx(0.0)

    translation = np.zeros_like(zero)
    translation[...] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    converted = xyz_field_to_common_zyx(translation)
    assert converted.shape == (7, 6, 5, 3)
    assert np.all(converted == np.array([3.0, 2.0, 1.0], dtype=np.float32))
    translation_metrics = compute_common_regularity(
        disp_vox_xyz=translation,
        fixed_lobe_union_xyz=mask,
        spacing_xyz_mm=spacing_xyz,
    )["metrics"]
    expected_voxel_magnitude = np.sqrt(1.0**2 + 2.0**2 + 3.0**2)
    expected_mm_magnitude = np.sqrt((1.0 * 0.7) ** 2 + (2.0 * 0.8) ** 2 + (3.0 * 2.5) ** 2)
    assert translation_metrics["diffusion_l2_full"] == pytest.approx(0.0)
    assert translation_metrics["bending_energy_full"] == pytest.approx(0.0)
    assert translation_metrics["jdet_mean_full"] == pytest.approx(1.0)
    assert translation_metrics["disp_mag_mean_full_vox"] == pytest.approx(
        expected_voxel_magnitude
    )
    assert translation_metrics["disp_mag_mean_full_mm"] == pytest.approx(
        expected_mm_magnitude
    )


def test_known_affine_has_expected_spacing_invariant_determinant():
    shape = (7, 8, 9)
    xx, yy, zz = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    scales = (0.1, -0.05, 0.2)
    field = np.stack((scales[0] * xx, scales[1] * yy, scales[2] * zz), axis=-1)
    metrics = compute_common_regularity(
        disp_vox_xyz=field,
        fixed_lobe_union_xyz=np.ones(shape, dtype=bool),
        spacing_xyz_mm=(0.7, 1.1, 2.5),
    )["metrics"]
    expected_det = np.prod([1.0 + value for value in scales])
    assert metrics["jdet_mean_full"] == pytest.approx(expected_det, abs=1e-6)
    assert metrics["log_jdet_std_full"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["jdet_le0_pct_full"] == pytest.approx(0.0)
    assert metrics["bending_energy_full"] == pytest.approx(0.0, abs=1e-8)


def _fake_pair_and_data(tmp_path: Path):
    shape = (10, 9, 8)
    lobes = np.zeros((5,) + shape, dtype=np.float32)
    for channel in range(5):
        lobes[channel, 2 * channel : 2 * channel + 2] = 1.0
    bbox = (slice(0, 10), slice(0, 9), slice(0, 8))
    bbox_payload = [
        {"start": item.start, "stop": item.stop, "step": item.step} for item in bbox
    ]
    summary = {
        "fixed_phase": 0,
        "moving_phase": 5,
        "image_shape_crop": list(shape),
        "image_shape_original": list(shape),
        "image_spacing_xyz": [0.7, 0.8, 2.5],
        "crop_bbox": bbox_payload,
    }
    a0_field = np.zeros(shape + (3,), dtype=np.float32)
    a1_field = np.zeros_like(a0_field)
    a1_field[..., 0] = 1.0
    pair = {
        "case": 1,
        "a0": {
            "seed": 42,
            "run_mode": "full_length",
            "summary": dict(summary),
            "dense_dvf": a0_field,
            "case_dir": tmp_path / "runs" / "a0" / "case_01",
        },
        "a1": {
            "seed": 42,
            "run_mode": "full_length",
            "summary": dict(summary),
            "dense_dvf": a1_field,
            "case_dir": tmp_path / "runs" / "a1" / "case_01",
        },
    }
    data = {
        "images": {"fixed": np.zeros(shape, dtype=np.float32)},
        "lung_classes_masks": {"fixed": lobes},
        "image_spacing": (0.7, 0.8, 2.5),
        "bbox": bbox,
        "original_shape": shape,
    }
    return pair, data


def test_generate_paired_csv_json_markdown_and_reject_existing(monkeypatch, tmp_path: Path):
    run_root = tmp_path / "runs"
    data_root = tmp_path / "data"
    review_root = tmp_path / "review"
    run_root.mkdir()
    data_root.mkdir()
    pair, data = _fake_pair_and_data(tmp_path)

    fake_api = SimpleNamespace(
        utils=SimpleNamespace(load_and_crop_full_dirlab=lambda *args, **kwargs: data)
    )
    monkeypatch.setattr(posthoc, "load_external_api", lambda: fake_api)
    monkeypatch.setattr(posthoc, "load_paired_case", lambda **kwargs: pair)

    manifest = generate_posthoc_reports(
        run_root=run_root,
        data_root=data_root,
        review_root=review_root,
        cases=[1],
    )
    csv_path = review_root / "case_01_paired_regularity.csv"
    json_path = review_root / "case_01_paired_regularity.json"
    markdown_path = review_root / "case_01_paired_regularity.md"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["variant"] for row in rows] == ["A0", "A1", "A1_minus_A0"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["a1"]["metrics"]["disp_mag_mean_full_mm"] == pytest.approx(0.7)
    expected_source = posthoc.POSTHOC_SOURCE.resolve()
    expected_source_bytes = expected_source.read_bytes()
    expected_source_record = {
        "path": "${REPO_ROOT}/"
        + expected_source.relative_to(posthoc.REPOSITORY_ROOT).as_posix(),
        "repository_relative_path": expected_source.relative_to(
            posthoc.REPOSITORY_ROOT
        ).as_posix(),
        "size_bytes": len(expected_source_bytes),
        "sha256": hashlib.sha256(expected_source_bytes).hexdigest(),
    }
    assert manifest["posthoc_source"] == expected_source_record
    assert payload["posthoc_source"] == expected_source_record
    assert manifest["run_root"] == "${RUN_ROOT}"
    assert manifest["data_root"] == "${DATA_ROOT}"
    assert manifest["review_root"] == "${REVIEW_ROOT}"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "voxel coordinates" in markdown
    assert "similarity transform" in markdown
    assert expected_source_record["path"] in markdown
    assert str(expected_source_record["size_bytes"]) in markdown
    assert expected_source_record["sha256"] in markdown
    assert manifest["conversion"].startswith("(X,Y,Z,xyz)")

    with pytest.raises(RegularityError, match="Refusing to overwrite"):
        generate_posthoc_reports(
            run_root=run_root,
            data_root=data_root,
            review_root=review_root,
            cases=[1],
        )


def test_rejects_invalid_field_and_geometry_mismatch():
    with pytest.raises(RegularityError, match="Expected displacement shape"):
        xyz_field_to_common_zyx(np.zeros((5, 5, 5), dtype=np.float32))
    with pytest.raises(RegularityError, match="DVF/mask shape mismatch"):
        compute_common_regularity(
            disp_vox_xyz=np.zeros((5, 5, 5, 3), dtype=np.float32),
            fixed_lobe_union_xyz=np.ones((5, 5, 4), dtype=bool),
            spacing_xyz_mm=(1.0, 1.0, 1.0),
        )


def test_rejects_saved_spacing_crop_and_dvf_shape_mismatches(tmp_path: Path):
    pair, data = _fake_pair_and_data(tmp_path)
    fixed_union, spacing = validate_author_geometry(pair=pair, data=data, case=1)
    assert fixed_union.shape == (10, 9, 8)
    assert spacing == (0.7, 0.8, 2.5)

    bad_spacing_pair, _ = _fake_pair_and_data(tmp_path)
    bad_spacing_pair["a1"]["summary"]["image_spacing_xyz"] = [1.0, 1.0, 1.0]
    with pytest.raises(RegularityError, match="spacing mismatch"):
        validate_author_geometry(pair=bad_spacing_pair, data=data, case=1)

    bad_bbox_pair, _ = _fake_pair_and_data(tmp_path)
    bad_bbox_pair["a0"]["summary"]["crop_bbox"][0]["stop"] = 9
    with pytest.raises(RegularityError, match="crop bbox mismatch"):
        validate_author_geometry(pair=bad_bbox_pair, data=data, case=1)

    bad_shape_pair, _ = _fake_pair_and_data(tmp_path)
    bad_shape_pair["a1"]["dense_dvf"] = np.zeros((9, 9, 8, 3), dtype=np.float32)
    with pytest.raises(RegularityError, match="DVF crop shape mismatch"):
        validate_author_geometry(pair=bad_shape_pair, data=data, case=1)
