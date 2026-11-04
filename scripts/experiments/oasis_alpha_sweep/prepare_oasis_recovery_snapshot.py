"""Create a new source-only execution snapshot; never edit historical inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

EXPECTED_EVALUATOR = '35504fab4e9174d6034f05174415433f46cffeaf0097877d4c8b080e8a2a2172'
CURRENT_EVALUATOR = 'e5507606b48f04da1195467a864e3d9d410e35872f7459b221e1010da5badfcd'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError('Require UID:GID 1009:1020')
    repo = Path(__file__).resolve().parents[3]
    a.out.mkdir(parents=True, exist_ok=False)
    snapshot = a.out/'repo'
    snapshot.mkdir()
    paths = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).decode().split('\0')
    paths = sorted(set(x for x in paths if x and Path(x).parts[0] in ('scripts', 'configs', 'tests', 'src')))
    source = {}
    for rel in paths:
        src, dst = repo/rel, snapshot/rel
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        source[str(src)] = hashlib.sha256(src.read_bytes()).hexdigest()
    evaluator = snapshot/'scripts/experiments/oasis_alpha_sweep/run_oasis_fedda_first100_eval.py'
    data = evaluator.read_bytes()
    if hashlib.sha256(data).hexdigest() != CURRENT_EVALUATOR or hashlib.sha256(data+b'\n').hexdigest() != EXPECTED_EVALUATOR:
        raise RuntimeError('Historical evaluator is not exactly current bytes plus one trailing newline')
    # Exact byte recovery, verified against the pre-interruption gate; no AST or
    # numerical change, no edit to the working repository.
    evaluator.write_bytes(data+b'\n')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    metadata = snapshot/'.git'
    (metadata/'objects').mkdir(parents=True)
    (metadata/'refs').mkdir()
    (metadata/'HEAD').write_text(revision+'\n')
    (metadata/'config').write_text('[core]\nrepositoryformatversion = 0\nbare = false\n')
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.rglob('*') if f.is_file()}
    manifest = dict(original_repo=str(repo), snapshot_repo=str(snapshot), revision=revision,
                    source_sha256=source, snapshot_sha256=hashes,
                    evaluator_recovery=dict(original_sha256=EXPECTED_EVALUATOR, current_sha256=CURRENT_EVALUATOR,
                                            transformation='append exactly one trailing newline'),
                    external_dependencies='unchanged external paths, still checked by historical source hashes')
    (a.out/'snapshot.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for f in snapshot.rglob('*'):
        if f.is_file():
            f.chmod(0o444)
    print(json.dumps(dict(snapshot=str(a.out/'snapshot.json'), files=len(hashes), revision=revision)))


if __name__ == '__main__':
    main()
