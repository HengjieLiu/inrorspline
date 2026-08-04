from __future__ import annotations

import numpy as np

from scripts.data.preprocessing.lungct.prepare_dual_author_copd import (
    _metric_mask_author_xyz,
    parse_case_ids_str,
)


def test_metric_mask_matches_author_z_flip_and_xyz_axes() -> None:
    source = np.zeros((2, 3, 4), dtype=np.uint8)
    source[0, 1, 2] = 1
    source[1, 2, 3] = 1
    observed = _metric_mask_author_xyz(source)
    assert observed.shape == (4, 3, 2)
    # raw zyx (0,1,2) flips to z=1 and then maps to xyz (2,1,1).
    assert observed[2, 1, 1] == 1
    assert observed[3, 2, 0] == 1
    assert int(observed.sum()) == 2


def test_copd_case_parser_requires_unique_in_range_cases() -> None:
    assert parse_case_ids_str("1-3,10") == [1, 2, 3, 10]
    for value in ("", "1,1", "0", "11"):
        try:
            parse_case_ids_str(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {value!r}")
