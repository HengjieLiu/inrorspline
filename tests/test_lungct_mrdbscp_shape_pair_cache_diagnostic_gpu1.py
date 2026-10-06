from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_DIR = REPO_ROOT / "scripts" / "experiments" / "lungct_no_sweep_repro"
if str(LAUNCHER_DIR) not in sys.path:
    sys.path.insert(0, str(LAUNCHER_DIR))

import launch_mrdbscp_shape_pair_cache_diagnostic_gpu1 as diagnostic  # noqa: E402


def test_each_case_is_an_adjacent_first_repeat_pair() -> None:
    rows = diagnostic.paired_case_sequence()
    assert len(rows) == 20
    for case_id in range(1, 11):
        pair = rows[2 * (case_id - 1) : 2 * case_id]
        assert [row["case_id"] for row in pair] == [case_id, case_id]
        assert [row["timing_condition"] for row in pair] == ["shape_first", "shape_repeat"]
        assert [row["sequence_index"] for row in pair] == [2 * (case_id - 1), 2 * (case_id - 1) + 1]


def test_diagnostic_has_80_unique_measurement_names() -> None:
    rows = [
        diagnostic.paired_run_name(dataset, variant, row)
        for dataset in diagnostic.base.DATASETS
        for variant in diagnostic.base.VARIANTS
        for row in diagnostic.paired_case_sequence()
    ]
    assert len(rows) == diagnostic.EXPECTED_PRIMARY_ROWS == 80
    assert len(set(rows)) == 80


def test_smoke_is_one_case_and_does_not_claim_shape_repeat() -> None:
    assert diagnostic.paired_case_sequence(smoke=True) == [
        {
            "case_id": 1,
            "sequence_index": 0,
            "timing_condition": "smoke",
            "case_occurrence": "smoke",
        }
    ]


def test_base_controller_exposes_diagnostic_configuration_hooks() -> None:
    source = (
        REPO_ROOT
        / "scripts/experiments/lungct_no_sweep_repro/launch_mrdbscp_adaptive_fixed50_gpu1.py"
    ).read_text()
    assert "PRIMARY_ROWS_PER_ARM" in source
    assert "WORKER_SCRIPT_NAME" in source
    assert "AGGREGATOR_SCRIPT_NAME" in source


def test_runner_accepts_explicit_shape_pair_timing_labels() -> None:
    source = (
        REPO_ROOT / "scripts/experiments/lungct_no_sweep_repro/run_one.py"
    ).read_text()
    assert '"shape_first"' in source
    assert '"shape_repeat"' in source
