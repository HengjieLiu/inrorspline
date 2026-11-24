"""Pure-stdlib experiment identity, coarse bracketing, and atomic metadata."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

ARMS = tuple(f"{loss}_{backend}" for loss in ("diffusion", "be", "kl_c1")
             for backend in ("fd", "analytic"))
COARSE = tuple(10.**k for k in range(-5, 4))
VERSION = "oasis_first10_regularizer_fd_analytic_v1"
PROTOCOL = dict(version=VERSION, cps=[8,4,2], factors=[4,2,1], lr=[.03,.01,.003],
    max_updates=[400,800,1300], min_updates=[40,80,130], patience=[20,35,50],
    check_interval=25, median_window=25, relative_total=1e-4, relative_data=5e-5,
    seed=1, kl_epsilon=.05, kl_kappa=120., best_state_including_initial=True,
    within_stage_lr_reductions=0, image_loss="squared_local_NCC9",
    training_mask=None, parameter_units="normalized_grid_XYZ_align_corners_true",
    derivative_units="current_evaluation_grid_voxel", cudnn_benchmark=False,
    cudnn_deterministic=True, tf32=False, jacobian_crop=2,
    cpu_threads=6, interop_threads=1, sample_seconds=.2)


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [clean(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True)+"\n")
    tmp.replace(path)


def read(path): return json.loads(Path(path).read_text())


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""): h.update(block)
    return h.hexdigest()


def stamp(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def setting(arm, weight):
    if arm not in (*ARMS,"none") or not math.isfinite(weight) or weight < 0:
        raise ValueError("Invalid regularizer setting")
    if (arm=="none") != (weight==0): raise ValueError("Use the shared none/zero control")
    return f"{arm}_w{weight:.9g}".replace(".","p").replace("-","m").replace("+","")


def sequence(phase):
    if phase=="smoke": return [(0,"cold")]
    if phase.startswith("pilot"): return [(0,"cold"),(0,"warm")]
    if phase=="cohort": return [(0,"cold")]+[(r,"warm") for r in range(10)]
    raise ValueError(phase)


def bracket(rows):
    """Power-of-ten rows only; failed/missing scores cannot witness a bracket."""
    valid=[r for r in rows if isinstance(r.get("dice"),(int,float)) and math.isfinite(r["dice"])]
    if not valid: return dict(status="no_valid_results", extensions=[])
    best=min(valid, key=lambda r:(-r["dice"],r.get("asd",math.inf),r.get("hd95",math.inf),r["weight"]))
    below=[r for r in valid if r["weight"]<best["weight"] and best["dice"]-r["dice"]>1e-5]
    above=[r for r in valid if r["weight"]>best["weight"] and best["dice"]-r["dice"]>1e-5]
    tested=[r["weight"] for r in rows]
    extensions=[]
    if len(below)<2 and min(tested)>1e-8*(1+1e-9): extensions.append(min(tested)/10)
    if len(above)<2 and max(tested)<1e6*(1-1e-9): extensions.append(max(tested)*10)
    ok=len(below)>=2 and len(above)>=2
    return dict(status="bracketed" if ok else "extend" if extensions else "flat_or_unbracketed",
                best=best, lower_witnesses=below, upper_witnesses=above,
                extensions=[] if ok else extensions)


def refined(center):
    k=round(math.log10(center))
    if not math.isclose(center,10.**k,rel_tol=1e-8): raise ValueError("Coarse decade required")
    return [v for e in range(k-2,k+2) for v in (float(f"1e{e}"),float(f"3e{e}"))]+[10.**(k+2)]
