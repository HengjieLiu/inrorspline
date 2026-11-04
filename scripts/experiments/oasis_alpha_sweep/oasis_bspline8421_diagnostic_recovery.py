#!/usr/bin/env python
"""P80 execution_v2: coordinate-autograd-only off-grid diagnostic correction.

Preserves the first smoke execution and every original source byte. Optimization,
parameters, losses, stopping and transition checks are delegated unchanged.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch

import bspline8421_matched_prefix as e
import run_oasis_bspline8421_matched_prefix as worker
import queue_oasis_bspline8421_matched_prefix as queue

BASE_SOURCES=e.sources
BASE_DIAGNOSTICS=worker.diagnostics
HERE=Path(__file__).resolve()


def sources(sitreg_root):
    return {"execution_v1":BASE_SOURCES(sitreg_root),"diagnostic_recovery_v2":{
        str(HERE.relative_to(e.c.REPO_ROOT)):e.sha(HERE),
        "tests/test_oasis_bspline8421_diagnostic_recovery.py":e.sha(e.c.REPO_ROOT/"tests/test_oasis_bspline8421_diagnostic_recovery.py")}}


def diagnostics(spec,field,controls,atomics,batch,out):
    if spec.family!="bounded2":
        return BASE_DIAGNOSTICS(spec,field,controls,atomics,batch,out)
    # Only off-grid sample coordinates require derivatives. The implementation
    # detaches all map fields; this does not enable network/control optimization.
    with torch.enable_grad():
        result=worker.b.offgrid_diagnostics(atomics,field,batch["fixed_mask"],count=100000,seed=1)
    return {"offgrid_audit":result}


def install():
    e.sources=sources
    worker.diagnostics=diagnostics
    queue.WORKER=HERE
    queue.REPORT=HERE


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    install()
    if argv and argv[0] in ("worker","audit","fixture"):
        args=worker.parse_args(argv); m=e.c.read_json(args.manifest)
        spec=e.Spec(args.case_rank,args.family,args.alpha,args.smoke)
        try:
            {"worker":worker.worker,"audit":worker.audit,"fixture":worker.fixture}[args.action](args,m,spec)
        except Exception:
            import time,traceback
            if args.out_dir.exists():
                e.atomic_json(args.out_dir/f"FAILED_{args.action}_{time.time_ns()}.json",{"traceback":traceback.format_exc(),"at":e.c.now_text()})
            raise
    elif "--run-root" in argv:
        sys.path.insert(0,str(e.c.REPO_ROOT/"scripts/reports"))
        import report_oasis_bspline8421_matched_prefix as report
        p=argparse.ArgumentParser(description="P80 corrected diagnostic report adapter")
        p.add_argument("--artifact-root",type=Path,required=True);p.add_argument("--run-root",type=Path,required=True)
        report.main(p.parse_args(argv))
    else:
        p=argparse.ArgumentParser(description=__doc__)
        p.add_argument("--artifact-root",type=Path,default=Path("/database/inrorspline"))
        p.add_argument("--stop-after",choices=("smoke","full"),default="full")
        p.add_argument("--execution",default="execution_v2")
        queue.main(p.parse_args(argv))


if __name__=="__main__":main()
