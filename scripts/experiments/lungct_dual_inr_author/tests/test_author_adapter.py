from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.experiments.lungct_dual_inr_author.adapter import (
    AUTHOR_ADAMW_WEIGHT_DECAY,
    build_author_config,
    seed_everything,
    train_author,
    validate_transition_manifest,
)
from scripts.experiments.lungct_dual_inr_author.external_api import (
    AUTHOR_EXTERNAL_COMMIT,
    external_repository,
    load_external_api,
)
from scripts.experiments.lungct_dual_inr_author.run_author import (
    adapter_source_bundle,
    build_parser,
    compute_five_lobe_dice,
    evaluate_landmarks,
    resolve_case_transition_trace,
    validate_roots,
)


def _tiny_arrays(shape: tuple[int, int, int] = (6, 6, 6)):
    grid = np.stack(
        np.meshgrid(
            *[np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in shape],
            indexing="ij",
        ),
        axis=0,
    )
    fixed = (0.5 * grid[0] + 0.3 * grid[1] ** 2 - 0.2 * grid[2]).astype(
        np.float32
    )
    moving = np.roll(fixed, shift=1, axis=0).copy()

    labels = np.indices(shape).sum(axis=0) % 5
    fixed_masks = np.stack(
        [(labels == index).astype(np.float32) for index in range(5)], axis=0
    )
    moving_masks = np.roll(fixed_masks, shift=1, axis=1).copy()
    return fixed, moving, fixed_masks, moving_masks


def _tiny_config(variant: str) -> dict:
    config = build_author_config(variant)
    config["model"]["layers"] = [3, 8, 3]
    config["training"]["total_steps"] = 3
    config["training"]["blur_sigmas"] = [1.0, 0.0]
    config["training"]["blur_convergence_patience"] = 1
    config["training"]["blur_convergence_threshold"] = 1.0
    config["training"]["blur_convergence_threshold_mode"] = "rel"
    config["scheduler"]["type"] = "none"
    config["scheduler"]["warmup_steps"] = 0
    config["data"]["dense_points_per_batch"] = 8
    return config


def test_external_import_is_pinned_and_path_guarded():
    src = str((external_repository() / "src").resolve())
    was_present = src in sys.path
    api = load_external_api()
    assert api.commit == AUTHOR_EXTERNAL_COMMIT
    assert Path(api.trainer.__file__).resolve().is_relative_to(api.package_dir)
    if not was_present:
        assert src not in sys.path


def test_adapter_source_bundle_hashes_exact_executable_bytes():
    bundle = adapter_source_bundle()
    assert bundle["algorithm"] == "sha256"
    assert len(bundle["bundle_sha256"]) == 64
    assert {Path(item["path"]).name for item in bundle["sources"]} == {
        "run_author.py",
        "adapter.py",
        "external_api.py",
    }
    for item in bundle["sources"]:
        data = Path(item["path"]).read_bytes()
        assert item["size_bytes"] == len(data)
        assert item["sha256"] == hashlib.sha256(data).hexdigest()
    snapshot = bundle["main_worktree_snapshot"]
    assert len(snapshot["status_porcelain_sha256"]) == 64
    assert "untracked" in snapshot["limitation"]


def test_a0_a1_configs_change_only_requested_objective_and_explicit_defaults():
    a0 = build_author_config("A0")
    a1 = build_author_config("A1")
    assert a0["loss"]["mask_weight"] == pytest.approx(0.1)
    assert a1["loss"]["mask_weight"] == pytest.approx(0.0)
    assert a0["optimizer"] == {
        "type": "AdamW",
        "lr": 0.001,
        "end_lr": 0.00001,
        "weight_decay": AUTHOR_ADAMW_WEIGHT_DECAY,
    }
    assert a0["training"]["blur_convergence_patience"] == 50
    assert a0["training"]["blur_convergence_threshold"] == pytest.approx(0.005)
    assert a0["scheduler"]["reset_on_blur_change"] is True

    stripped_a0 = copy.deepcopy(a0)
    stripped_a1 = copy.deepcopy(a1)
    stripped_a0["loss"]["mask_weight"] = 0.0
    stripped_a0["adapter"] = stripped_a1["adapter"]
    assert stripped_a0 == stripped_a1


def test_transition_manifest_validation_rejects_mismatches():
    manifest = {
        "schema_version": 1,
        "source_variant": "A0",
        "total_steps": 3,
        "blur_sigmas": [1.0, 0.0],
        "stage_start_steps": [0, 2],
        "events": [],
    }
    normalized = validate_transition_manifest(
        manifest,
        total_steps=3,
        blur_sigmas=[1.0, 0.0],
        require_a0=True,
    )
    assert normalized["stage_start_steps"] == [0, 2]
    with pytest.raises(ValueError, match="total_steps mismatch"):
        validate_transition_manifest(manifest, total_steps=4)
    with pytest.raises(ValueError, match="blur_sigmas mismatch"):
        validate_transition_manifest(manifest, blur_sigmas=[2.0, 0.0])
    bad = dict(manifest, stage_start_steps=[1, 2])
    with pytest.raises(ValueError, match="begin at zero"):
        validate_transition_manifest(bad)


def test_a0_adaptive_trace_is_replayed_exactly_by_a1_on_cpu():
    fixed, moving, fixed_masks, moving_masks = _tiny_arrays()
    seed_everything(42)
    a0 = train_author(
        fixed_image=fixed,
        moving_image=moving,
        fixed_lung_classes_mask=fixed_masks,
        moving_lung_classes_mask=moving_masks,
        config=_tiny_config("A0"),
        device="cpu",
        source_variant="A0",
    )
    trace = dict(a0["transition_manifest"])
    trace["case"] = 1
    assert trace["policy"] == "author_dynamic_convergence"
    assert trace["stage_start_steps"] == [0, 2]
    assert [row["stage_index"] for row in a0["history"]] == [0, 0, 1]
    assert a0["optimizer_settings"]["weight_decay"] == pytest.approx(0.01)
    assert a0["optimizer"].param_groups[0]["weight_decay"] == pytest.approx(0.01)

    seed_everything(42)
    a1 = train_author(
        fixed_image=fixed,
        moving_image=moving,
        fixed_lung_classes_mask=fixed_masks,
        moving_lung_classes_mask=moving_masks,
        config=_tiny_config("A1"),
        device="cpu",
        source_variant="A1",
        replay_transition_manifest=trace,
    )
    assert a1["transition_manifest"]["policy"] == "replay_a0"
    assert a1["transition_manifest"]["stage_start_steps"] == [0, 2]
    assert [row["stage_index"] for row in a1["history"]] == [0, 0, 1]
    assert all(row["loss_mask_weighted"] == pytest.approx(0.0) for row in a1["history"])
    assert all(np.isfinite(row["loss_total"]) for row in a1["history"])
    for row in a1["history"]:
        component_sum = (
            row["loss_similarity"]
            + row["loss_spatial_weighted"]
            + row["loss_mask_weighted"]
        )
        assert row["loss_total"] == pytest.approx(component_sum, abs=1e-6)


def test_a1_rejects_missing_trace_before_training():
    fixed, moving, fixed_masks, moving_masks = _tiny_arrays()
    with pytest.raises(ValueError, match="requires a realized per-case A0"):
        train_author(
            fixed_image=fixed,
            moving_image=moving,
            fixed_lung_classes_mask=fixed_masks,
            moving_lung_classes_mask=moving_masks,
            config=_tiny_config("A1"),
            device="cpu",
            source_variant="A1",
        )


def test_case_trace_resolution_and_cli_seed_default(tmp_path: Path):
    trace_root = tmp_path / "traces"
    trace_path = trace_root / "a0" / "case_03" / "stage_transitions.json"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_variant": "A0",
                "case": 3,
                "total_steps": 3,
                "blur_sigmas": [1.0, 0.0],
                "stage_start_steps": [0, 2],
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    selected, manifest = resolve_case_transition_trace(
        trace_root,
        case=3,
        total_steps=3,
        blur_sigmas=[1.0, 0.0],
        allow_direct_file=False,
    )
    assert selected == trace_path.resolve()
    assert manifest["case"] == 3

    args = build_parser().parse_args(
        [
            "--variant",
            "A0",
            "--case",
            "1",
            "--data-root",
            "/tmp/data",
            "--output-root",
            "/tmp/output",
        ]
    )
    assert args.seed == 42
    smoke_args = build_parser().parse_args(
        [
            "--variant",
            "A0",
            "--case",
            "1",
            "--data-root",
            "/tmp/data",
            "--output-root",
            "/tmp/output",
            "--smoke-steps",
            "3",
        ]
    )
    assert smoke_args.smoke_steps == 3


def test_identity_evaluation_reports_snap_subvoxel_and_lobe_dice():
    api = load_external_api()

    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, coordinates):
            return torch.zeros_like(coordinates) + self.anchor * 0.0

    shape = (4, 4, 4)
    landmarks = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    dense = np.zeros(shape + (3,), dtype=np.float32)
    metrics, arrays = evaluate_landmarks(
        api=api,
        model=ZeroModel(),
        fixed_landmarks=landmarks,
        moving_landmarks=landmarks.copy(),
        image_shape=shape,
        image_spacing=(1.0, 1.0, 1.0),
        dense_dvf=dense,
    )
    assert metrics["direct_inr"]["snap"]["mean_mm"] == pytest.approx(0.0)
    assert metrics["dense_dvf"]["subvoxel"]["mean_mm"] == pytest.approx(0.0)
    assert arrays["direct_subvoxel_tre_mm"].shape == (2,)

    labels = np.indices(shape).sum(axis=0) % 5
    masks = np.stack(
        [(labels == channel).astype(np.float32) for channel in range(5)], axis=0
    )
    dice = compute_five_lobe_dice(
        api=api,
        fixed_masks=masks,
        moving_masks=masks.copy(),
        dense_dvf=dense,
    )
    assert dice["mean_dice"] == pytest.approx(1.0)
    assert [item["label_id"] for item in dice["per_lobe"]] == [10, 11, 12, 13, 14]


def test_output_root_must_be_absolute_and_outside_repo(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(ValueError, match="absolute scratch"):
        validate_roots(data_root, Path("relative-output"))
    with pytest.raises(ValueError, match="outside the Git repository"):
        validate_roots(data_root, external_repository())
    with pytest.raises(ValueError, match="outside the input data root"):
        validate_roots(data_root, data_root / "runs")
