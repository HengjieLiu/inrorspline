from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.lungct_dual_inr_author.external_api import (
    AUTHOR_EXTERNAL_COMMIT,
    load_external_api,
)
from scripts.experiments.lungct_dual_inr_author.plot_author_review import (
    ReviewError,
    generate_review,
    load_paired_case,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_image(api, path: Path, array_xyz: np.ndarray) -> None:
    image = api.utils.sitk.GetImageFromArray(np.swapaxes(array_xyz, 0, 2))
    image.SetSpacing((1.0, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    api.utils.sitk.WriteImage(image, str(path))


def _make_author_case(data_root: Path, case: int = 1) -> tuple[int, int, int]:
    api = load_external_api()
    shape = (8, 8, 8)
    coordinates = np.indices(shape, dtype=np.float32)
    fixed = coordinates[0] + 0.5 * coordinates[1] + 0.25 * coordinates[2]
    moving = np.roll(fixed, shift=1, axis=1)
    labels = np.zeros(shape, dtype=np.int16)
    for offset, label in enumerate((10, 11, 12, 13, 14), start=1):
        labels[offset, 2:6, 2:6] = label
    moving_labels = np.roll(labels, shift=1, axis=1)

    case_dir = data_root / f"case_{case:02d}"
    _write_image(api, case_dir / "images" / "phase_00.nii", fixed)
    _write_image(api, case_dir / "images" / "phase_05.nii", moving)
    _write_image(api, case_dir / "segmentations_00.nii", labels)
    _write_image(api, case_dir / "segmentations_05.nii", moving_labels)
    landmarks = np.tile(np.array([[3.0, 3.0, 3.0]], dtype=np.float32), (300, 1))
    landmark_dir = case_dir / "landmarks_raw"
    landmark_dir.mkdir(parents=True)
    np.savetxt(landmark_dir / "extreme_landmarks_0.csv", landmarks, delimiter=",")
    np.savetxt(landmark_dir / "extreme_landmarks_5.csv", landmarks, delimiter=",")
    return shape


def _make_record(
    run_root: Path,
    data_root: Path,
    *,
    variant: str,
    shape: tuple[int, int, int],
    case: int = 1,
) -> Path:
    case_dir = run_root / variant.lower() / f"case_{case:02d}"
    case_dir.mkdir(parents=True)
    a0_trace = run_root / "a0" / f"case_{case:02d}" / "stage_transitions.json"
    summary = {
        "status": "complete",
        "variant": variant,
        "case": case,
        "seed": 42,
        "run_mode": "short_smoke",
        "smoke_steps": 3,
        "image_shape_crop": list(shape),
        "image_spacing_xyz": [1.0, 1.0, 1.0],
        "a0_transition_trace": str(a0_trace) if variant == "A1" else None,
    }
    state = dict(summary)
    config = {
        "loss": {"mask_weight": 0.1 if variant == "A0" else 0.0},
        "training": {"total_steps": 3},
    }
    provenance = {
        "external_commit": AUTHOR_EXTERNAL_COMMIT,
        "external_git_status_porcelain": "",
        "data_root": str(data_root.resolve()),
    }
    trace = {
        "schema_version": 1,
        "source_variant": variant,
        "case": case,
        "seed": 42,
        "total_steps": 3,
        "blur_sigmas": [4.0, 2.0, 0.0],
        "stage_start_steps": [0],
        "policy": "author_dynamic_convergence" if variant == "A0" else "replay_a0",
        "replayed_manifest_path": str(a0_trace) if variant == "A1" else None,
        "events": [],
    }
    _write_json(case_dir / "summary.json", summary)
    _write_json(case_dir / "RUN_STATE.json", state)
    _write_json(case_dir / "resolved_config.json", config)
    _write_json(case_dir / "provenance.json", provenance)
    _write_json(case_dir / "stage_transitions.json", trace)

    fields = [
        "zero_step",
        "stage_index",
        "blur_sigma",
        "loss_total",
        "loss_similarity",
        "loss_spatial_weighted",
        "loss_mask_weighted",
    ]
    with (case_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in range(3):
            writer.writerow(
                {
                    "zero_step": step,
                    "stage_index": 0,
                    "blur_sigma": 4.0,
                    "loss_total": 1.0 - 0.1 * step,
                    "loss_similarity": 0.8 - 0.05 * step,
                    "loss_spatial_weighted": 0.1,
                    "loss_mask_weighted": 0.1 if variant == "A0" else 0.0,
                }
            )
    dvf = np.zeros(shape + (3,), dtype=np.float32)
    if variant == "A1":
        dvf[..., 1] = 0.2
    np.save(case_dir / "dense_dvf_crop_voxel_xyz.npy", dvf)
    (case_dir / "DONE").touch()
    return case_dir


def test_generate_review_from_complete_synthetic_pair_and_reject_partial(tmp_path: Path):
    data_root = tmp_path / "data"
    run_root = tmp_path / "runs"
    review_root = tmp_path / "review"
    shape = _make_author_case(data_root)
    _make_record(run_root, data_root, variant="A0", shape=shape)
    a1_dir = _make_record(run_root, data_root, variant="A1", shape=shape)

    manifest = generate_review(
        run_root=run_root,
        data_root=data_root,
        review_output=review_root,
        cases=[1],
        slice_axis=2,
        dpi=72,
        allow_smoke=True,
    )
    figure = review_root / "case_01_author_review.png"
    manifest_path = review_root / "author_review_manifest.json"
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert manifest_path.is_file()
    assert manifest["case_reviews"][0]["slice"]["selection_rule"] == (
        "argmax_fixed_lobe_union_area"
    )
    assert manifest["warping"]["image_interpolation"].endswith("order1_mode_nearest")

    (a1_dir / "DONE").unlink()
    with pytest.raises(ReviewError, match="Incomplete A1"):
        load_paired_case(
            run_root=run_root,
            data_root=data_root,
            case=1,
            allow_smoke=True,
        )
