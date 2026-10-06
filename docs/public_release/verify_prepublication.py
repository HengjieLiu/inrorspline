"""Offline acceptance inside the cloned-candidate image, with no data mounts."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path("/opt/rightpriordir")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "release_manifest.json").read_text())
    for relative, record in manifest["files"].items():
        assert sha(root / relative) == record["sha256"], relative
        assert (root / relative).stat().st_size == record["bytes"], relative
    assert not list(root.rglob(".git"))
    assert not list(root.rglob(".env*"))
    assert not list(root.rglob(".ruff_cache"))
    expected = json.loads((root / "docs/environment.json").read_text())["packages"]
    for name, version in expected.items():
        assert importlib.metadata.version(name) == version, name
    for name, arguments in [
        ("tests", ["pytest", "-q", "-p", "no:cacheprovider", "tests", f"--junitxml={out / 'tests.xml'}"]),
        ("verify", ["rightpriordir", "verify"]),
        ("pip-check", ["pip", "check"]),
        ("frozen", ["rightpriordir", "frozen", "--out", str(out / "frozen")]),
    ]:
        result = subprocess.run([sys.executable, "-m", *arguments], cwd=root,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out / f"{name}.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(result.stdout)
    expected_frozen = {
        "figure2.png": "eb0db83bbd1f96d34c3afc9c25576e56c7944c07cc04ce74ece6f615d23e3e16",
        "updated_oasis_timing.md": "3f8368c14a273dd80636c3325c1f3ef56e3a1381be9837a7a8172dacce5aaebf",
        "historical_oasis100_accuracy.md": "190b650a9d8876d4855802473352eaab0f8d512ac7eb38dd31ebe07c6247c54e",
        "table1_published.md": "6cba9b322bca632e3ba9bb46889dbcc36c1db344187d6fd0e83f259438b1fbcb",
        "copd_extension.md": "04ddeb681db27bc3945afff76d77de0946d502f4f56b7bb7395b8f98e8d13ef9",
    }
    for name, digest in expected_frozen.items():
        assert sha(out / "frozen" / name) == digest, name
    suites = list(ET.parse(out / "tests.xml").getroot().iter("testsuite"))
    assert sum(int(s.attrib["failures"]) + int(s.attrib["errors"]) for s in suites) == 0
    report = {
        "tests_passed": sum(int(s.attrib["tests"]) for s in suites),
        "test_seconds": sum(float(s.attrib["time"]) for s in suites),
        "manifest_sha256": sha(root / "release_manifest.json"),
        "manifest_files_verified": len(manifest["files"]),
        "installed_package_versions_verified": len(expected),
        "frozen_outputs_sha256": expected_frozen,
        "pip_check": "passed",
        "no_git_env_or_ruff_cache_in_image": True,
        "optimization_executed": False,
        "publication_clearance": False,
        "permission_status": "SINR_pending",
        "evidence_sha256": {name: sha(out / name) for name in ["tests.xml", "verify.log", "pip-check.log", "frozen/render_manifest.json"]},
    }
    (out / "technical_acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
