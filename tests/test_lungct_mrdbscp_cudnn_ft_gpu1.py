from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_DIR = REPO_ROOT / "scripts" / "experiments" / "lungct_no_sweep_repro"
if str(LAUNCHER_DIR) not in sys.path:
    sys.path.insert(0, str(LAUNCHER_DIR))

import launch_mrdbscp_cudnn_ft_gpu1 as study  # noqa: E402


RUN_ID = "lungct_mrdbscp_cudnn_ft_gpu1_20260929T230000Z"


def test_backend_command_explicitly_requests_false_true() -> None:
    args = SimpleNamespace(
        data_root=Path("/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm"),
        attempt_root=Path("/database/inrorspline/runs/test/attempt"),
    )
    row = study.base.case_sequence()[0]
    command = study.backend_run_one_argv(args, "4dct", "fixed50", row, smoke=False)
    assert "--no-cudnn-benchmark" in command
    assert "--cudnn-deterministic" in command


def test_backend_manifest_keeps_44_row_original_layout() -> None:
    payload = study.backend_manifest(SimpleNamespace(run_id=RUN_ID))
    assert payload["experiment_id"] == study.EXPERIMENT_ID
    assert len(payload["primary_jobs"]) == 44
    assert payload["backend"] == {"cudnn_benchmark": False, "cudnn_deterministic": True}
    assert payload["comparison_references"]["original_cold_warm"]["run_id"] == study.P34_RUN_ID
    assert payload["comparison_references"]["adjacent_shape_pairs"]["run_id"] == study.P35_RUN_ID


def test_runner_exposes_backend_flags_and_records_effective_values() -> None:
    source = (LAUNCHER_DIR / "run_one.py").read_text()
    assert '"--cudnn-benchmark"' in source
    assert '"--cudnn-deterministic"' in source
    assert '"effective_cudnn_benchmark"' in source
    assert '"effective_cudnn_deterministic"' in source


def test_common_defaults_apply_future_false_true_policy() -> None:
    source = (LAUNCHER_DIR / "common.py").read_text()
    assert "cudnn_benchmark: bool = False" in source
    assert "cudnn_deterministic: bool = True" in source


def test_source_provenance_includes_new_launcher_aggregator_test_and_logs() -> None:
    original = {
        "EXPERIMENT_ID": study.base.EXPERIMENT_ID,
        "RUN_PREFIX": study.base.RUN_PREFIX,
        "WORKER_SCRIPT_NAME": study.base.WORKER_SCRIPT_NAME,
        "AGGREGATOR_SCRIPT_NAME": study.base.AGGREGATOR_SCRIPT_NAME,
        "run_one_argv": study.base.run_one_argv,
        "immutable_manifest": study.base.immutable_manifest,
        "SOURCE_PATHS": study.base.SOURCE_PATHS,
    }
    try:
        study.configure_base()
        paths = {str(path) for path in study.base.SOURCE_PATHS}
        assert study.base.EXPERIMENT_ID == study.EXPERIMENT_ID
        assert study.base.RUN_PREFIX == study.RUN_PREFIX
        assert study.base.WORKER_SCRIPT_NAME == "launch_mrdbscp_cudnn_ft_gpu1.py"
        assert study.base.AGGREGATOR_SCRIPT_NAME == "aggregate_mrdbscp_cudnn_ft_gpu1.py"
        assert "tests/test_lungct_mrdbscp_cudnn_ft_gpu1.py" in paths
        assert "experiments/2026-09-29_lungct_mrdbscp_cudnn_ft_gpu1/technical_log.md" in paths
    finally:
        for key, value in original.items():
            setattr(study.base, key, value)
