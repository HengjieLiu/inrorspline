from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.lungct_dual_inr_controlled import data


def _case(protocol: str, sampling: np.ndarray, metric: np.ndarray) -> data.ControlledCase:
    fixed = np.zeros((12, 13, 14), dtype=np.float32)
    moving = np.zeros_like(fixed)
    landmarks = np.array([[1.0, 2.0, 3.0], [8.0, 9.0, 10.0]], dtype=np.float64)
    return data.ControlledCase(
        protocol=protocol,
        case_id=1,
        fixed=fixed,
        moving=moving,
        sampling_mask_xyz=sampling.astype(np.uint8, copy=False),
        metric_mask_xyz=metric.astype(np.uint8, copy=False),
        landmarks_fixed_xyz=landmarks,
        landmarks_moving_xyz=landmarks,
        spacing_xyz=(1.0, 1.0, 1.0),
        source_paths={},
        provenance={},
        ptv_case=object() if protocol in {"P", "PL"} else None,
    )


def test_ptv_lungmask_protocol_requires_sampling_and_metric_mask_match() -> None:
    metric = np.zeros((12, 13, 14), dtype=np.uint8)
    metric[2:10, 3:11, 4:12] = 1
    case = _case("PL", sampling=metric.copy(), metric=metric)

    data._validate_controlled_case(case)

    border = data.make_border_five_mask_xyz(metric.shape)
    with pytest.raises(ValueError, match="Protocol PL sampling and metric masks"):
        data._validate_controlled_case(_case("PL", sampling=border, metric=metric))


def test_border_mask_protocol_remains_distinct_from_lungmask_protocol() -> None:
    metric = np.zeros((12, 13, 14), dtype=np.uint8)
    metric[2:10, 3:11, 4:12] = 1
    border = data.make_border_five_mask_xyz(metric.shape)

    data._validate_controlled_case(_case("P", sampling=border, metric=metric))
    with pytest.raises(ValueError, match="Protocol P sampling mask"):
        data._validate_controlled_case(_case("P", sampling=metric, metric=metric))
