#!/usr/bin/env python
"""Native-volume worker/serialization smoke after numerical resource profiles."""
import argparse
import copy
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from runtime import read_json,atomic_json,now,REPO,code_hashes
from worker import Worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--revision',default='v3');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('GPU0 mask required')
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=read_json(REPO/'experiments/2026-11-05_oasis_adaptive_local_affine_fieldfit/protocol_v1.3.json')
    config.update(fixed_grid_steps=[[4,1],[2,1],[1,1]],tree_growth_steps_per_split=1,
                  tree_growth_total_steps=3,tree_half_steps=[1,1,1],tree_native_steps=[1,1,1])
    config['nonlinear']['check_interval']=1
    target=next(t for t in read_json(a.root/'input_manifest.json')['targets'] if t['id']=='rank000_sitreg')
    rows=[];source=code_hashes()
    for mode,family,budgets in [('d0','tree_affine_uniform',[48,96,192]),
                               ('svf','tree_affine_adaptive',[48,96,192]),
                               ('d1','bspline',[192])]:
        ident='smoke_'+mode+'_'+family
        job=dict(id=ident,target_id=target['id'],mode=mode,family=family,budgets=budgets,
                 initialization={'d1':'ls','svf':'global_affine_ls'}.get(mode,'zero'),kind='worker_smoke')
        smoke_root=a.root/f'worker_smoke_{a.revision}'
        folder=smoke_root/ident
        try:
            if folder.exists():raise FileExistsError('Smoke attempt exists; preserve it and use a new revision')
            torch.cuda.reset_peak_memory_stats()
            w=Worker(smoke_root,job,folder,copy.deepcopy(config),target,smoke=True)
            w.run()
            if len(w.endpoints)!=len(budgets):raise RuntimeError('Missing serialized smoke budgets')
            rows.append(dict(job_id=ident,status='passed',endpoints=w.endpoints))
            del w;torch.cuda.empty_cache()
        except Exception as exc:
            rows.append(dict(job_id=ident,status='failed',error=str(exc),traceback=traceback.format_exc()))
        result=dict(at=now(),passed=len(rows)==3 and all(r['status']=='passed' for r in rows),
                    rows=rows,source_hashes=source,limitations=['One-update stage smoke; no optimizer-convergence claim.'])
        atomic_json(a.root/f'WORKER_SMOKE_{a.revision}.json',result)
        atomic_json(a.root/'WORKER_SMOKE.json',result)
        print(rows[-1],flush=True)


if __name__=='__main__':main()
