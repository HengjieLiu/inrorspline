#!/usr/bin/env python3
"""Launch the COPD A0/A1 cohort with validated author-format data."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from scripts.data.preprocessing.lungct import prepare_dual_author_copd as adapter
from scripts.experiments.lungct_dual_inr_author import launch_author as core


def audit_prepared_data(data_root: Path) -> dict[str, Any]:
    required = (
        data_root / "dataset_manifest.json",
        data_root / "preprocess_contract.json",
        data_root / "checksums.sha256",
    )
    binding = {
        "dataset_manifest_sha256": core.file_sha256(required[0]) if required[0].is_file() else None,
        "preprocess_contract_sha256": core.file_sha256(required[1]) if required[1].is_file() else None,
        "checksums_manifest_sha256": core.file_sha256(required[2]) if required[2].is_file() else None,
    }
    if any(not path.is_file() for path in required):
        return {"status": "fail", "errors": ["prepared COPD artifact is incomplete"], "binding": binding}
    try:
        validation = adapter.validate_existing(data_root, "1-10")
        manifest = core._load_json(data_root / "dataset_manifest.json")
        contract = core._load_json(data_root / "preprocess_contract.json")
        errors = []
        if manifest.get("adapter_id") != adapter.ADAPTER_ID:
            errors.append("dataset manifest adapter ID is not the COPD author adapter")
        if contract.get("adapter_id") != adapter.ADAPTER_ID:
            errors.append("preprocess contract adapter ID is not the COPD author adapter")
        if manifest.get("case_ids") != list(range(1, 11)):
            errors.append("prepared artifact must contain COPD cases 01-10")
        if validation.get("status") != "pass":
            errors.append("prepared COPD artifact validation failed")
        return {
            "status": "pass" if not errors else "fail",
            "errors": errors,
            "binding": binding,
            "adapter_id": adapter.ADAPTER_ID,
            "case_ids": list(range(1, 11)),
            "validation": validation,
        }
    except Exception as exc:
        return {"status": "fail", "errors": [f"{type(exc).__name__}: {exc}"], "binding": binding}


def main() -> int:
    core.RUNNER_MODULE = "scripts.experiments.lungct_dual_inr_author_copd.run_author"
    core.audit_prepared_data = audit_prepared_data
    core.validate_done_summary = partial(core.validate_done_summary, identity_tre_mm=None)
    original_plan = core.immutable_plan_payload
    original_state = core.initial_state_payload
    original_code_binding = core.code_sha256_binding

    def copd_code_binding() -> dict[str, str]:
        binding = original_code_binding()
        root = core.repository_root()
        for relative in (
            "scripts/data/preprocessing/lungct/prepare_dual_author_copd.py",
            "scripts/experiments/lungct_dual_inr_author_copd/launch_author.py",
            "scripts/experiments/lungct_dual_inr_author_copd/run_author.py",
        ):
            binding[relative] = core.file_sha256(root / relative)
        return binding

    def copd_plan(*args, **kwargs):
        payload = original_plan(*args, **kwargs)
        payload["experiment_id"] = "dual_inr_copd_six_arm_v001"
        payload["dataset"] = "copd"
        payload["registration_direction"] = "eBH_to_iBH"
        return payload

    def copd_state(*args, **kwargs):
        payload = original_state(*args, **kwargs)
        payload["experiment_id"] = "dual_inr_copd_six_arm_v001"
        payload["dataset"] = "copd"
        return payload

    core.immutable_plan_payload = copd_plan
    core.initial_state_payload = copd_state
    core.code_sha256_binding = copd_code_binding
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
