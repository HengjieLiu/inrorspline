"""Freeze the P82 manifest, validate source reconstruction, and profile operators."""
from __future__ import annotations
import argparse
import gc
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from common import HERE,REPO,KEY,FAMILIES,CAPS,jobs,sources,sha,now,external,read_json,write_json
from operators import Grid,GridOperator,lattice_shape
from trees import Tree,TreeOperator
from worker import Worker

def replay(target):
    path=REPO/'scripts/experiments/oasis_alpha_sweep/common.py'
    spec=importlib.util.spec_from_file_location('p82_original_spline',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    cp=Path(target['native_checkpoint_path'])
    state=torch.load(cp,map_location='cpu',weights_only=False)
    controls=state['parameter_norm_xyz'].cuda()
    dense=module.expand_bspline_controls(controls,tuple(controls.shape[:3]),tuple(target['shape']),2)
    original=np.load(target['source_path'],mmap_mode='r')
    normalized_rmse=float(np.sqrt(np.mean((dense.cpu().numpy().astype(np.float64)-original)**2)))
    if not np.allclose(np.asarray(target['affine'])[:3,:3],np.eye(3)):raise ValueError('Unexpected physical affine')
    mm=dense.flip(-1)*dense.new_tensor((np.asarray(target['shape'])-1)/2)
    ref=torch.tensor(np.load(target['canonical_path']),device='cuda',dtype=torch.float64)
    rmse=float((mm.double()-ref).square().sum(-1).mean().sqrt())
    result=dict(status='passed' if rmse<1e-4 else 'failed',full_rmse_mm=rmse,
                normalized_component_rmse=normalized_rmse,actual_params=controls.numel(),
                control_shape=list(controls.shape[:3]),checkpoint_sha256=sha(cp),implementation_sha256=sha(path))
    if rmse>=1e-4:raise RuntimeError(f'Native cps2 reconstruction failed: {result}')
    return result

def profile(family,cap,lo,hi,shape):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    if family.startswith('tree_'):
        model=Tree(family,lo,hi);goal=cap//12 if model.affine else (cap-9)//3
        while model.k<goal:model.split(goal,np.arange(model.k,dtype=float) if family.endswith('adaptive') else None)
        op=TreeOperator(model,shape,'cuda')
    else:
        k=cap//12 if family=='gaussian_affine' else (cap-9)//3 if family=='gaussian_translation' else cap//3
        dims=lattice_shape(k,[b-a for a,b in zip(lo,hi)],4 if family=='bspline' else 2)
        model=Grid(family,dims,lo,hi).at_factor(1);op=GridOperator(model,shape,'cuda')
    c=torch.zeros(model.m,3,device='cuda',dtype=torch.float64)
    norms=op.norms();y=op.forward(c);g=op.adjoint(y)
    if float(y.abs().max())!=0 or float(g.abs().max())!=0 or not bool(torch.isfinite(norms).all()):
        raise ValueError('Native zero/norm smoke failed')
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_reserved()/2**20
    result=dict(family=family,cap=cap,actual_params=model.params,seconds=time.monotonic()-start,
                peak_reserved_mib=peak,shape=list(shape),status='passed')
    del op,c,y,g,norms;gc.collect();torch.cuda.empty_cache()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reference-manifest',required=True)
    a=p.parse_args();root=external(a.root);root.mkdir(parents=True,exist_ok=False)
    cfg=read_json(REPO/'experiments/2026-11-05_oasis_rank000_fieldfit_parameter_scaling/config.json')
    targets={t['id']:t for t in read_json(a.reference_manifest)['targets']
             if t['id'] in ('rank000_sitreg','rank000_unbounded')}
    if len(targets)!=2:raise ValueError('Missing two rank0 sources')
    for t in targets.values():
        if (t['fixed_id'],t['moving_id'],t['shape'])!=('0333','0115',[160,224,192]):raise ValueError('Wrong pair/shape')
        if sha(t['canonical_path'])!=t['canonical_sha256']:raise ValueError('Canonical input hash mismatch')
        if sha(t['source_path'])!=t['source_sha256']:raise ValueError('Original source hash mismatch')
        if not np.isfinite(np.load(t['canonical_path'],mmap_mode='r')).all():raise ValueError('Nonfinite source')
    manifest=dict(created_at=now(),experiment=KEY,jobs=jobs(),targets=targets,config=cfg,source_hashes=sources(),
                  reference_manifest_sha256=sha(a.reference_manifest),repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  config_sha256=sha(REPO/'experiments/2026-11-05_oasis_rank000_fieldfit_parameter_scaling/config.json'),
                  environment=dict(torch=torch.__version__,cuda=torch.version.cuda,uid=os.getuid(),gid=os.getgid(),
                                   container_image_id=os.environ.get('P82_IMAGE_ID','unrecorded'),
                                   requested_backend=cfg['backend'],effective_backend=cfg['backend']))
    write_json(root/'manifest.json',manifest)
    try:
        unit=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_oasis_fieldfit_scaling.py','-v'],
                            cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (root/'unit_tests.log').write_text(unit.stdout)
        if unit.returncode:raise RuntimeError('Unit contracts failed; see unit_tests.log')
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        result=replay(targets['rank000_unbounded']);write_json(root/'native_cps2_replay.json',result)
        print('Native cps2 replay:',result,flush=True)
        profiles=[];t=targets['rank000_sitreg']
        for family in FAMILIES:
            row=profile(family,3072,t['domain_min'],t['domain_max'],t['shape']);profiles.append(row)
            write_json(root/'profiles.json',profiles);print('PROFILE',row,flush=True)
        for family in FAMILIES[:3]:
            row=profile(family,CAPS[-1],t['domain_min'],t['domain_max'],t['shape']);profiles.append(row)
            write_json(root/'profiles.json',profiles);print('PROFILE',row,flush=True)
        for family,version in [('bspline','multiresolution'),('tree_affine_adaptive','native')]:
            j=next(j for j in manifest['jobs'] if j['target']=='rank000_sitreg' and j['family']==family and j['cap']==3072 and j['version']==version)
            w=Worker(root,j,root/'smoke'/j['id'],seconds=20);r=w.run();print('SMOKE',j['id'],r['metrics'],flush=True)
            del w;gc.collect();torch.cuda.empty_cache()
        write_json(root/'preflight.json',dict(status='passed',at=now(),source_hashes=sources(),profiles=len(profiles),
                     unit_tests='passed',native_cps2_replay=result,native_worker_smokes=2))
    except Exception as e:
        import traceback
        write_json(root/'preflight.json',dict(status='failed',at=now(),error=str(e),traceback=traceback.format_exc()))
        raise
    print(str(root),flush=True)

if __name__=='__main__':main()
