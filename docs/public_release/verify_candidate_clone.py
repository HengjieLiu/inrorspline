"""Validate a disposable Git candidate; never commit in the official checkout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(directory, *args):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    result = subprocess.run(
        ["git", "-c", "credential.helper=", "-c", "core.hooksPath=/dev/null", "-C", str(directory), *args],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    public, snapshot, work = args.public.resolve(), args.snapshot.resolve(), args.work.resolve()
    if work.is_relative_to(public) or work.is_relative_to(snapshot):
        raise ValueError("Acceptance work must be separate from source/snapshot")
    work.mkdir(parents=True, exist_ok=False)
    before_head = git(public, "rev-parse", "HEAD")
    before_index = sha(public / ".git/index")
    manifest = json.loads((snapshot / "release_manifest.json").read_text())
    pins = manifest["submodule_commits"]
    subpaths = {f"third_party/{name}" for name in pins}
    ordinary = sorted(p for p in manifest["files"] if not any(p.startswith(s + "/") for s in subpaths))
    ordinary.append("release_manifest.json")
    for path, record in manifest["files"].items():
        if sha(snapshot / path) != record["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {path}")
    candidate, fresh = work / "candidate", work / "fresh-clone"
    git(work, "clone", "--no-hardlinks", "--no-checkout", str(public), str(candidate))
    git(candidate, "read-tree", before_head)
    for relative in ordinary:
        destination = candidate / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / relative, destination)
    git(candidate, "add", "--", *ordinary)
    for name, pin in pins.items():
        git(candidate, "update-index", "--add", "--cacheinfo", "160000", pin, f"third_party/{name}")
    assert set(git(candidate, "ls-files").splitlines()) == set(ordinary) | subpaths
    git(candidate, "diff", "--cached", "--check")
    git(candidate, "-c", "user.name=Local release verification", "-c", "user.email=release-check@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-m", "Disposable publication candidate acceptance; not an official release")
    candidate_commit = git(candidate, "rev-parse", "HEAD")
    git(work, "clone", "--no-hardlinks", "--recurse-submodules", str(candidate), str(fresh))
    assert not git(fresh, "status", "--porcelain")
    for path, record in manifest["files"].items():
        assert sha(fresh / path) == record["sha256"], path
        assert (fresh / path).stat().st_size == record["bytes"], path
    for name, pin in pins.items():
        assert git(fresh / "third_party" / name, "rev-parse", "HEAD") == pin
        assert not git(fresh / "third_party" / name, "status", "--porcelain")
    assert git(public, "rev-parse", "HEAD") == before_head
    assert sha(public / ".git/index") == before_index
    result = {
        "official_head": before_head,
        "official_head_and_index_unchanged": True,
        "disposable_candidate_commit": candidate_commit,
        "fresh_recursive_clone": True,
        "upstream_sources": "public_HTTPS_URLs_from_gitmodules",
        "submodule_pins": pins,
        "manifest_sha256": sha(fresh / "release_manifest.json"),
        "manifest_files_verified": len(manifest["files"]),
        "parent_git_entries": len(ordinary) + len(subpaths),
        "explicit_staging_paths": sorted(ordinary + list(subpaths)),
        "publication_clearance": False,
        "reason": "SINR_permission_pending",
    }
    (work / "candidate_acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
