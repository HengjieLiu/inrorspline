from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_DIR = REPO_ROOT / "scripts" / "experiments" / "lungct_no_sweep_repro"
import sys

if str(LAUNCHER_DIR) not in sys.path:
    sys.path.insert(0, str(LAUNCHER_DIR))

import launch_mrdbscp_adaptive_fixed50_gpu1 as launch  # noqa: E402


RUN_ID = "lungct_mrdbscp_adaptive_fixed50_gpu1_20260929T120000Z"


def test_exact_frozen_schedules() -> None:
    four_adaptive = launch.variant_spec("4dct", "adaptive")
    assert four_adaptive.stage_full_cps == (32, 16, 8, 4)
    assert four_adaptive.stage_downsample == (8, 4, 2, 1)
    assert four_adaptive.stage_epochs == (1400, 400, 400, 400)
    assert four_adaptive.stage_lr == (0.01, 0.003, 0.003, 0.003)
    assert four_adaptive.min_epochs == (210, 60, 60, 60)
    assert four_adaptive.patience == (50, 25, 25, 25)
    assert four_adaptive.total_thresholds == (5e-5,) * 4
    assert four_adaptive.data_thresholds == (2e-5,) * 4

    copd_adaptive = launch.variant_spec("copd", "adaptive")
    assert copd_adaptive.stage_full_cps == (64, 32, 16, 8, 4)
    assert copd_adaptive.stage_downsample == (16, 8, 4, 2, 1)
    assert copd_adaptive.stage_epochs == (800, 600, 400, 400, 400)
    assert copd_adaptive.stage_lr == (0.03, 0.03, 0.01, 0.01, 0.003)
    assert copd_adaptive.min_epochs == (120, 90, 60, 60, 60)
    assert copd_adaptive.patience == (50, 38, 25, 25, 25)
    assert copd_adaptive.total_thresholds == (5e-5,) * 5
    assert copd_adaptive.data_thresholds == (2e-5,) * 5

    for dataset, stages in (("4dct", 4), ("copd", 5)):
        fixed = launch.variant_spec(dataset, "fixed50")
        assert fixed.stage_epochs == (50,) * stages
        assert fixed.early_stop_mode == "disabled"


def test_manifest_has_44_unique_cold_warm_measurements() -> None:
    rows = launch.expected_jobs()
    assert len(rows) == 44
    assert len({row["run_name"] for row in rows}) == 44
    assert len([row for row in rows if row["timing_condition"] == "cold"]) == 4
    assert len([row for row in rows if row["timing_condition"] == "warm"]) == 40
    for dataset in launch.DATASETS:
        for variant in launch.VARIANTS:
            arm = [row for row in rows if row["dataset"] == dataset and row["variant"] == variant]
            assert [(row["case_id"], row["timing_condition"]) for row in arm] == [
                (1, "cold"),
                (1, "warm"),
                *((case_id, "warm") for case_id in range(2, 11)),
            ]


def test_manifest_validation_is_json_roundtrip_idempotent(tmp_path: Path) -> None:
    controller = launch.Controller.__new__(launch.Controller)
    controller.args = SimpleNamespace(run_id=RUN_ID)
    controller.run_root = tmp_path / "run"
    controller.review_root = tmp_path / "review"
    controller.run_root.mkdir()
    controller.review_root.mkdir()
    controller.ensure_manifest()
    controller.ensure_manifest()
    written = json.loads((controller.run_root / "study_manifest.json").read_text())
    assert written["manifest"]["expected_primary_rows"] == 44


def test_gpu_one_and_fresh_run_id_are_enforced() -> None:
    with pytest.raises(SystemExit):
        launch.parse_args(["--run-id", RUN_ID, "--gpu", "0"])
    with pytest.raises(SystemExit):
        launch.parse_args(["--run-id", "wrong_prefix", "--gpu", "1"])
    args = launch.parse_args(["--run-id", RUN_ID, "--gpu", "1"])
    assert args.gpu == 1


def test_external_artifact_routing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_root = tmp_path / "artifact-store"
    artifact_root.mkdir()
    args = launch.parse_args(
        ["--run-id", RUN_ID, "--gpu", "1", "--artifact-host", str(artifact_root)]
    )
    resolved = launch.resolve_artifact_paths(args)
    assert resolved.out_root == artifact_root / "runs" / RUN_ID
    assert resolved.review_root == artifact_root / "repo_experiments" / launch.EXPERIMENT_ID / RUN_ID
    assert not str(resolved.out_root).startswith(str(launch.REPO_ROOT))

    monkeypatch.delenv("ARTIFACT_ROOT", raising=False)
    missing = launch.parse_args(["--run-id", RUN_ID, "--gpu", "1"])
    with pytest.raises(launch.StudyError, match="ARTIFACT_ROOT"):
        launch.resolve_artifact_paths(missing)


def test_contaminated_attempt_restarts_complete_arm(tmp_path: Path) -> None:
    controller = launch.Controller.__new__(launch.Controller)
    controller.args = SimpleNamespace(max_arm_attempts=3)
    controller.run_root = tmp_path
    controller.accepted = {}
    controller.accepted_path = tmp_path / "accepted_arms.json"
    controller.accepted_arm_valid = lambda key, smoke: False
    calls: list[Path] = []

    def fake_attempt(dataset: str, variant: str, attempt_root: Path, *, smoke: bool) -> dict[str, object]:
        calls.append(attempt_root)
        return {
            "accepted": len(calls) == 2,
            "contaminated": len(calls) == 1,
            "dataset": dataset,
            "variant": variant,
            "smoke": smoke,
            "attempt_root": str(attempt_root),
        }

    controller.run_attempt = fake_attempt
    verdict = controller.run_arm("4dct", "adaptive", smoke=False)
    assert verdict["accepted"] is True
    assert [path.name for path in calls] == ["attempt_00", "attempt_01"]
    assert controller.accepted["primary:4dct:adaptive"]["attempt_root"].endswith("attempt_01")


def test_resume_requires_complete_accepted_attempt(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt_00"
    attempt.mkdir()
    (attempt / "worker_result.json").write_text(json.dumps({"status": "completed", "row_count": 11}))
    (attempt / "attempt_verdict.json").write_text(json.dumps({"accepted": True}))
    controller = launch.Controller.__new__(launch.Controller)
    controller.accepted = {
        "primary:copd:fixed50": {"accepted": True, "attempt_root": str(attempt)}
    }
    assert controller.accepted_arm_valid("primary:copd:fixed50", smoke=False)
    (attempt / "worker_result.json").write_text(json.dumps({"status": "completed", "row_count": 10}))
    assert not controller.accepted_arm_valid("primary:copd:fixed50", smoke=False)


def test_worker_command_masks_physical_gpu_one(tmp_path: Path) -> None:
    controller = launch.Controller.__new__(launch.Controller)
    controller.args = SimpleNamespace(
        repo_container="/homebase/code_sync/inrorspline",
        container="py310sitreg_v1_artifact_store",
        run_id=RUN_ID,
        data_root=Path("/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm"),
        artifact_host=tmp_path,
        artifact_container=Path("/database/inrorspline"),
    )
    attempt = tmp_path / "runs" / RUN_ID / "attempt"
    command = controller.docker_command("copd", "adaptive", attempt, "token", smoke=False)
    assert "CUDA_VISIBLE_DEVICES=1" in command
    assert "--groups=1009,1020" in command
    assert "--reuid=1009" in command
    assert "--regid=1009" in command
    assert command[command.index("--gpu") + 1] == "1"


def test_historical_launcher_explicitly_preserves_true_false_backend() -> None:
    args = SimpleNamespace(
        data_root=Path("/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm"),
        attempt_root=Path("/database/inrorspline/runs/test/attempt"),
    )
    command = launch.run_one_argv(args, "4dct", "fixed50", launch.case_sequence()[0], smoke=False)
    assert "--cudnn-benchmark" in command
    assert "--no-cudnn-deterministic" in command
