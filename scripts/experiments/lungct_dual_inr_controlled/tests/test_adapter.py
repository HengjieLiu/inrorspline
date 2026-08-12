from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from scripts.experiments.lungct_dual_inr_controlled import adapter


def _released_dual_config() -> dict:
    """Small, dependency-free stand-in for the pinned external DUAL preset."""

    return {
        "model": {
            "is_dual": True,
            "layers": [3, 256, 256, 3],
            "coarse_omega0": 10,
            "fine_omega0": 30,
        },
        "training": {},
        "loss": {
            "laplace_weight": 0.01,
            "jacobian_weight": 0.01,
            "fine_reg_scale": 0.1,
        },
        "data": {
            "dense_points_per_batch": 30000,
            "sampler": "continuous",
        },
        "scheduler": {"warmup_steps": 20},
    }


@pytest.fixture
def stub_external_config(monkeypatch: pytest.MonkeyPatch) -> None:
    api = SimpleNamespace(
        configs=SimpleNamespace(DUAL_CONFIG=_released_dual_config())
    )
    monkeypatch.setattr(adapter, "load_external_api", lambda: api)


def test_locked_four_variant_schedule_and_fine_activation(
    stub_external_config: None,
) -> None:
    assert adapter.LEGACY_VARIANTS == ("N0", "N1", "P0", "P1")
    assert adapter.LUNGMASK_REVISION_VARIANTS == ("PL1D", "PL1B", "PL1A", "PL1C")
    assert adapter.VARIANTS == (
        "N0",
        "N1",
        "P0",
        "P1",
        "PL1D",
        "PL1B",
        "PL1A",
        "PL1C",
    )
    assert adapter.FULL_RESOLUTION_FACTORS == (1, 1, 1, 1)
    assert adapter.MULTIRESOLUTION_FACTORS == (8, 4, 2, 1)
    assert adapter.LOCKED_STAGE_LENGTHS == (400, 400, 800, 900)
    assert adapter.LOCKED_STAGE_STARTS == (0, 400, 800, 1600)
    assert adapter.LOCKED_TOTAL_STEPS == 2500
    assert sum(adapter.LOCKED_STAGE_LENGTHS) == adapter.LOCKED_TOTAL_STEPS

    expected = {
        "N0": ("N", [1, 1, 1, 1]),
        "N1": ("N", [8, 4, 2, 1]),
        "P0": ("P", [1, 1, 1, 1]),
        "P1": ("P", [8, 4, 2, 1]),
        "PL1D": ("PL", [8, 4, 2, 1]),
        "PL1B": ("PL", [8, 4, 2, 1]),
        "PL1A": ("PL", [1, 1, 1]),
        "PL1C": ("PL", [1, 1, 1]),
    }
    for variant, (protocol, factors) in expected.items():
        config = adapter.build_controlled_config(variant)
        assert adapter.variant_protocol(variant) == protocol
        assert list(adapter.variant_factors(variant)) == factors
        assert config["training"]["image_factors"] == factors
        if variant in {"PL1A", "PL1C"}:
            assert config["training"]["stage_lengths"] == [3000]
            assert config["training"]["stage_starts"] == [0]
            assert config["training"]["total_steps"] == 3000
            assert config["training"]["blur_sigmas"] == [4.0, 2.0, 0.0]
            assert adapter.variant_schedule(variant) == adapter.SCHEDULE_AUTHOR_BLUR
        else:
            assert config["training"]["stage_lengths"] == [400, 400, 800, 900]
            assert config["training"]["stage_starts"] == [0, 400, 800, 1600]
            assert config["training"]["total_steps"] == 2500
            assert adapter.variant_schedule(variant) == adapter.SCHEDULE_IMAGE_PYRAMID
        assert config["training"]["fine_start_stage"] == 1
        if variant not in {"PL1A", "PL1C"}:
            assert config["training"]["stage_starts"][
                config["training"]["fine_start_stage"]
            ] == 400


def test_controlled_objective_and_stage_reset_contract(
    stub_external_config: None,
) -> None:
    for variant in adapter.LEGACY_VARIANTS + ("PL1D", "PL1A"):
        config = adapter.build_controlled_config(variant)
        assert config["loss"]["mask_weight"] == pytest.approx(0.0)
        assert config["loss"]["laplace_weight"] == pytest.approx(0.01)
        assert config["loss"]["jacobian_weight"] == pytest.approx(0.01)
        assert config["loss"]["fine_reg_scale"] == pytest.approx(0.1)
        if variant == "PL1A":
            assert config["optimizer"]["reset_each_stage"] is False
            assert config["scheduler"]["reset_each_stage"] is False
            assert config["scheduler"]["reset_on_blur_change"] is True
            assert (
                config["controlled_adapter"]["optimizer_state_policy"]
                == "single_adamw_persistent_moments"
            )
            assert (
                config["controlled_adapter"]["stage_policy"]
                == "author_dynamic_convergence"
            )
        else:
            assert config["optimizer"]["reset_each_stage"] is True
            assert config["scheduler"]["reset_each_stage"] is True
            assert (
                config["controlled_adapter"]["optimizer_state_policy"]
                == "fresh_adamw_each_stage"
            )
        assert config["controlled_adapter"]["semantic_mask_loss_present"] is False
        assert (
            config["controlled_adapter"]["loss_branch_identifier"]
            == adapter.LOSS_BRANCH_DUAL_NATIVE
        )


def test_lungmask_revision_idir_be_branch_disables_dual_native_regularizers(
    stub_external_config: None,
) -> None:
    config = adapter.build_controlled_config("PL1B")

    assert adapter.variant_protocol("PL1B") == "PL"
    assert adapter.variant_loss_branch("PL1B") == adapter.LOSS_BRANCH_IDIR_BE
    assert config["loss"]["mask_weight"] == pytest.approx(0.0)
    assert config["loss"]["laplace_weight"] == pytest.approx(0.0)
    assert config["loss"]["jacobian_weight"] == pytest.approx(0.0)
    assert config["loss"]["fine_reg_scale"] == pytest.approx(0.0)
    assert config["loss"]["idir_bending_weight"] == pytest.approx(10.0)
    assert (
        config["controlled_adapter"]["loss_scope"]
        == "sampled_global_ncc_plus_idir_coordinate_autograd_bending_energy"
    )


def test_lungmask_revision_ptv_lcc_tv_branch_uses_author_blur(
    stub_external_config: None,
) -> None:
    config = adapter.build_controlled_config("PL1C")

    assert adapter.variant_protocol("PL1C") == "PL"
    assert adapter.variant_loss_branch("PL1C") == adapter.LOSS_BRANCH_PTV_LCC_TV
    assert adapter.variant_schedule("PL1C") == adapter.SCHEDULE_AUTHOR_BLUR
    assert config["loss"]["mask_weight"] == pytest.approx(0.0)
    assert config["loss"]["laplace_weight"] == pytest.approx(0.0)
    assert config["loss"]["jacobian_weight"] == pytest.approx(0.0)
    assert config["loss"]["fine_reg_scale"] == pytest.approx(0.0)
    assert config["loss"]["ptv_isotv_weight"] == pytest.approx(0.055)
    assert config["loss"]["ptv_lcc_sigma_mm"] == pytest.approx(2.1)
    assert config["loss"]["sampled_lcc_centers"] == 512
    assert config["loss"]["sampled_lcc_offsets"] == 64
    assert (
        config["controlled_adapter"]["loss_scope"]
        == "sampled_ptv_lcc_plus_sampled_physical_isotv"
    )


def test_regularization_multiplier_scales_variant_specific_weights(
    stub_external_config: None,
) -> None:
    native = adapter.build_controlled_config("PL1D", regularization_multiplier=10.0)
    be = adapter.build_controlled_config("PL1B", regularization_multiplier=0.1)
    ptv = adapter.build_controlled_config("PL1C", ptv_isotv_weight=0.17)

    assert native["loss"]["jacobian_weight"] == pytest.approx(0.1)
    assert native["loss"]["laplace_weight"] == pytest.approx(0.1)
    assert native["controlled_adapter"]["resolved_regularization_weights"][
        "fine_jacobian_weight"
    ] == pytest.approx(0.01)
    assert be["loss"]["idir_bending_weight"] == pytest.approx(1.0)
    assert ptv["loss"]["ptv_isotv_weight"] == pytest.approx(0.17)
    assert ptv["controlled_adapter"]["resolved_regularization_weights"][
        "ptv_isotv_weight"
    ] == pytest.approx(0.17)


def test_factor_one_loss_volume_is_exact_identity() -> None:
    volume = torch.arange(5 * 7 * 9, dtype=torch.float32).reshape(5, 7, 9)
    level, sigma, kernel_size = adapter._make_stage_volume(
        volume,
        1,
        sigma_scale=0.5,
        radius_sigma=3.0,
    )

    assert level.data_ptr() == volume.data_ptr()
    assert torch.equal(level, volume)
    assert sigma == pytest.approx(0.0)
    assert kernel_size == 1


@pytest.mark.parametrize(
    ("factor", "expected_shape"),
    [
        (8, (3, 3, 3)),
        (4, (5, 5, 5)),
        (2, (9, 9, 10)),
    ],
)
def test_antialiased_pyramid_uses_ceil_shapes(
    factor: int,
    expected_shape: tuple[int, int, int],
) -> None:
    volume = torch.linspace(0.0, 1.0, 17 * 18 * 19).reshape(17, 18, 19)
    level, sigma, kernel_size = adapter._make_stage_volume(
        volume,
        factor,
        sigma_scale=0.5,
        radius_sigma=3.0,
    )

    assert tuple(level.shape) == expected_shape
    assert tuple(level.shape) == tuple(
        max(2, math.ceil(size / factor)) for size in volume.shape
    )
    assert sigma == pytest.approx(0.5 * factor)
    assert kernel_size == 2 * math.ceil(3.0 * sigma) + 1
    assert level.dtype == volume.dtype
    assert level.device == volume.device
