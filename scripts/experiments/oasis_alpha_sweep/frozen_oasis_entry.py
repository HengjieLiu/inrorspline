"""Run an approved OASIS worker against a verified, read-only repo snapshot.

Only historical repository paths are redirected for hash verification. External
datasets, model code, dependencies, checkpoints and reports are never exempted.
"""
from pathlib import Path
import hashlib
import json
import os
import runpy
import sys


def install():
    path = Path(os.environ['OASIS_SOURCE_SNAPSHOT'])
    manifest = json.loads(path.read_text())
    root = Path(manifest['snapshot_repo'])
    original = Path(manifest['original_repo'])
    if Path(__file__).resolve().parents[3] != root:
        raise RuntimeError('Bootstrap must execute from the frozen repository')
    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for p, expected in manifest['snapshot_sha256'].items():
        if digest(p) != expected:
            raise RuntimeError('Frozen source changed: '+p)
    import sitreg_training as native
    def verify(hashes):
        for p, expected in hashes.items():
            target = Path(p)
            if target.is_relative_to(original):
                target = root / target.relative_to(original)
            if native.sha256(target) != expected:
                raise RuntimeError(f'Source/data drift: {p} (verified at {target})')
    native.verify_hashes = verify
    return manifest


def main():
    install()
    script = Path(sys.argv[1]).resolve()
    if not script.is_relative_to(Path(__file__).resolve().parents[3]):
        raise RuntimeError('Worker outside frozen source tree')
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
