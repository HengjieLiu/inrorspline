"""Independently verify a generated gallery and retain CPU-test evidence."""
import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np
from PIL import Image
import torch

from .preview import MODULE, REPO, digest, write_json


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src"):
                self.targets.append(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    artifact_root = Path(os.environ["ARTIFACT_ROOT"]).resolve()
    if root == artifact_root or not root.is_relative_to(artifact_root) or root.is_relative_to(REPO):
        raise ValueError("Verification can only write inside an external artifact run")
    manifest = json.loads((root / "manifest.json").read_text())
    cases = json.loads((root / "cases.json").read_text())
    checks = {}
    checks["expected_stage"] = manifest["status"] == "awaiting_case_selection"
    checks["output_hashes"] = all(digest(root / p) == h for p, h in manifest["outputs_sha256"].items())
    checks["source_snapshots"] = all(digest(root / "source_snapshot" / p) == h for p, h in manifest["sources"].items())
    checks["source_not_drifted"] = all(digest(REPO / p) == h for p, h in manifest["sources"].items())
    checks["ten_cases"] = len(cases) == 10 and [r["rank"] for r in cases] == list(range(10))
    checks["pngs_readable_1080p"] = True
    checks["finite_slices_valid_labels"] = True
    for case in cases:
        with Image.open(root / case["figure"]) as picture:
            checks["pngs_readable_1080p"] &= picture.size == (1920, 1080)
            picture.verify()
        with np.load(root / case["slices"], allow_pickle=False) as archive:
            checks["finite_slices_valid_labels"] &= set(archive.files) == {
                "moving", "fixed", "warped", "moving_seg", "fixed_seg", "warped_seg", "grid"}
            for key in archive.files:
                arr = archive[key]
                checks["finite_slices_valid_labels"] &= bool(arr.shape == (224, 160) and np.isfinite(arr).all())
                if key.endswith("_seg"):
                    checks["finite_slices_valid_labels"] &= bool(np.isin(arr, np.arange(36)).all())
    links = Links()
    links.feed((root / "index.html").read_text())
    checks["html_local_assets_resolve"] = all(
        p.startswith("#") or (root / p).is_file() for p in links.targets)
    checks["no_remote_dependencies"] = all("://" not in p for p in links.targets)
    for source in MODULE.glob("*.py"):
        ast.parse(source.read_text(), filename=str(source))
    checks["module_ast_parses"] = True
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("scripts.visualization.mrdbscp_progress.test_progress")
    with redirect_stdout(output), redirect_stderr(output):
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    checks["unit_tests"] = result.wasSuccessful()
    (root / "unit_tests.txt").write_text(output.getvalue())
    diff = subprocess.run(["git", "diff", "--check"], cwd=REPO, capture_output=True, text=True)
    checks["git_diff_check"] = diff.returncode == 0
    (root / "diff_check.txt").write_text(diff.stdout + diff.stderr)
    checks["cuda_not_initialized"] = not torch.cuda.is_initialized()
    passed = all(checks.values())
    # Verification itself may be added after rendering; record its exact source separately.
    (root / "verification_source.py").write_text(Path(__file__).read_text())
    write_json(root / "delivery_validation.json", dict(passed=passed, checks=checks,
        unit_tests_run=result.testsRun, verified_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=digest(Path(__file__)), registration_launched=False,
        next_gate="user_case_selection"))
    print(json.dumps(dict(passed=passed, checks=checks, unit_tests_run=result.testsRun)))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
