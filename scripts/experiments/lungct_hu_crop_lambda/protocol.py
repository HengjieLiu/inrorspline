"""Pure-stdlib campaign identities, atomic receipts and adaptive search rules."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import datetime

WINDOWS = {'H0': (-944., -124.), 'H1': (-1000., 1000.), 'H2': (-1000., 2000.)}
CROPS = ('C0', 'C1', 'C2')
FIRST = [('4dct', 1), ('4dct', 8), ('copd', 1), ('copd', 8)]
REST = [(ds, i) for ds in ('4dct', 'copd') for i in range(1, 11) if i not in (1, 8)]
BASE = .055
CAP = 8
TOL = 1e-6
ITERATIONS = [400, 400, 400, 400, 900]

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def weight(k):
    if isinstance(k, bool) or int(k) != k or abs(k) > CAP: raise ValueError(k)
    return math.ldexp(BASE, int(k))

def sid(ds, case, hu, crop):
    return f'{ds}_{case:02d}_{hu}_{crop}'

def combinations(pairs):
    return [dict(dataset=ds, case=case, hu=hu, crop=crop, id=sid(ds,case,hu,crop))
            for ds,case in pairs for crop in CROPS for hu in WINDOWS]

def direction(results, sign):
    """Failures break adjacency. Each branch independently compares its center."""
    previous = results.get('0', {}).get('score')
    streak = 0
    for distance in range(1, CAP+1):
        k = sign*distance
        r = results.get(str(k))
        if r is None:
            return dict(status='active', next=k, streak=streak)
        score = r.get('score') if r.get('status') == 'accepted' else None
        if score is None:
            previous = None; streak = 0
        else:
            if not math.isfinite(score): raise ValueError('Nonfinite accepted TRE')
            streak = streak+1 if previous is not None and score > previous+TOL else 0
            previous = score
            if streak == 2: return dict(status='two_increases', next=None, streak=2, last=k)
    return dict(status='cap_reached', next=None, streak=streak, last=sign*CAP)

def sweep_state(results, preparation_failure=False):
    if preparation_failure:
        return dict(done=True, status='preparation_failed', next=[], lower=None, upper=None, best=None)
    if '0' not in results:
        return dict(done=False, status='center_pending', next=[0], lower=None, upper=None, best=None)
    branches = [direction(results, s) for s in (1,-1)]
    valid = [(int(k), r) for k,r in results.items() if r.get('status') == 'accepted']
    best = None
    if valid:
        minimum = min(r['score'] for _,r in valid)
        ties = [(k,r) for k,r in valid if r['score'] <= minimum+TOL]
        k,r = min(ties, key=lambda v:(abs(v[0]),-v[0]))
        best = dict(k=k, weight=weight(k), score=r['score'], candidate=r)
    done = all(b['next'] is None for b in branches)
    return dict(done=done, status='finished' if done else 'searching',
                next=[b['next'] for b in branches if b['next'] is not None],
                upper=branches[0], lower=branches[1], best=best)

def candidate_key(config, k):
    return f"{config['id']}/k{k:+03d}"
