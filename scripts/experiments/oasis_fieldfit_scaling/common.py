"""Isolated P82 runtime, immutable provenance, and cumulative active timers."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
KEY = 'oasis_rank000_fieldfit_parameter_scaling_v1'
FAMILIES = ('bspline', 'gaussian_translation', 'gaussian_affine',
            'tree_translation_uniform', 'tree_translation_adaptive',
            'tree_affine_uniform', 'tree_affine_adaptive')
CAPS = (3072, 24576, 196608, 1572864, 12582912, 20643840)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    if isinstance(x, Path): return str(x)
    if hasattr(x, 'item'): return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x

def write_json(path, x):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name+'.')
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(clean(x), f, indent=2, allow_nan=False); f.write('\n')
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def read_json(path): return json.loads(Path(path).read_text())

def external(path):
    p = Path(path).resolve()
    if p == REPO or REPO in p.parents: raise ValueError('Outputs must be external')
    return p

def sources():
    files = list(HERE.glob('*.py'))
    files += [REPO/'scripts/experiments/oasis_polyaffine_fieldfit/basis.py']
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(files)}

class Deadline(Exception): pass

class Budget:
    """Only instantiated after input loading. All stage/transfer caps share it."""
    def __init__(self, seconds=1200., spent=0.):
        self.seconds, self.spent = float(seconds), float(spent)
        self.start = time.monotonic()
        self.limit = self.seconds
    @property
    def elapsed(self): return self.spent + time.monotonic()-self.start
    def check(self):
        if self.elapsed >= min(self.seconds,self.limit): raise Deadline('active_time_limit')
    def stage(self, cumulative): self.limit = min(self.seconds, float(cumulative))

def jobs():
    return [dict(id=f'{target}__{family}__{cap}__{version}', target=target,
                 family=family, cap=cap, version=version)
            for cap in CAPS for target in ('rank000_sitreg','rank000_unbounded')
            for family in FAMILIES for version in ('native','multiresolution')]
