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
    assert adapter.VARIANTS == ("N0", "N1", "P0", "P1")
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
    }
    for variant, (protocol, factors) in expected.items():
        config = adapter.build_controlled_config(variant)
        assert adapter.variant_protocol(variant) == protocol
        assert list(adapter.variant_factors(variant)) == factors
        assert config["training"]["image_factors"] == factors
        assert config["training"]["stage_lengths"] == [400, 400, 800, 900]
        assert config["training"]["stage_starts"] == [0, 400, 800, 1600]
        assert config["training"]["total_steps"] == 2500
        assert config["training"]["fine_start_stage"] == 1
        assert config["training"]["stage_starts"][
            config["training"]["fine_start_stage"]
        ] == 400


def test_controlled_objective_and_stage_reset_contract(
    stub_external_config: None,
) -> None:
    for variant in adapter.VARIANTS:
        config = adapter.build_controlled_config(variant)
        assert config["loss"]["mask_weight"] == pytest.approx(0.0)
        assert config["loss"]["laplace_weight"] == pytest.approx(0.01)
        assert config["loss"]["jacobian_weight"] == pytest.approx(0.01)
        assert config["loss"]["fine_reg_scale"] == pytest.approx(0.1)
        assert config["optimizer"]["reset_each_stage"] is True
        assert config["scheduler"]["reset_each_stage"] is True
        assert (
            config["controlled_adapter"]["optimizer_state_policy"]
            == "fresh_adamw_each_stage"
        )
        assert config["controlled_adapter"]["semantic_mask_loss_present"] is False


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
