from __future__ import annotations

import csv

import numpy as np

from scripts.experiments.lungct_dual_inr_copd.aggregate import (
    REPORT_METRICS,
    _author_crop_mask,
    _mr_reference,
)


def test_author_crop_mask_uses_serialized_xyz_bbox() -> None:
    full = np.ones((5, 6, 7), dtype=np.uint8)
    summary = {
        "crop_bbox": [
            {"start": 1, "stop": 4, "step": None},
            {"start": 2, "stop": 5, "step": None},
            {"start": 3, "stop": 7, "step": None},
        ]
    }
    assert _author_crop_mask(summary, full).shape == (3, 3, 4)


def test_mr_reference_requires_complete_cohort(tmp_path) -> None:
    path = tmp_path / "rows.csv"
    fields = [
        "case_id",
        "native_crop_snap_tre_mean_mm",
        "native_crop_sub_tre_mean_mm",
        "reg1mm_snap_tre_mean_mm",
        "reg1mm_sub_tre_mean_mm",
        "fg_diffusion_l2",
        "fg_bending_energy",
        "fg_log_jdet_std",
        "fg_jdet_le0_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in range(1, 11):
            writer.writerow({name: 1.0 if name != "case_id" else case for name in fields})
    rows = _mr_reference(path)
    assert len(rows) == 10
    assert rows[0]["method"] == "MR-D-BSCP P2e"
    assert rows[0]["regularity_available"] is True
    assert all(metric in rows[0] for metric in REPORT_METRICS)
