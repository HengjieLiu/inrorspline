"""Small provenance and atomic-file helpers for the isolated P81 execution."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
KEY = 'oasis_seed3_first10_adaptive_local_affine_fieldfit_v1'
FAMILIES = ('bspline', 'gaussian_translation', 'gaussian_affine',
            'tree_translation_uniform', 'tree_translation_adaptive',
            'tree_affine_uniform', 'tree_affine_adaptive')
CAPS = (768, 1536, 3072)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def clean(value):
    import math
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return clean(value.item())
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(clean(value), stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return json.loads(Path(path).read_text())


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def code_hashes():
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(HERE.glob('*.py'))}


def assert_external(path):
    path = Path(path).resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError('Generated P81 outputs must be outside the Git worktree')
    return path
