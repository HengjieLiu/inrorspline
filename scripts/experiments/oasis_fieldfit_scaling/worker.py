"""One independent P82 target/family/cap/version fit, checkpointed on deadline."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from common import Budget,Deadline,external,read_json,write_json,sha,now,sources
from operators import Grid,GridOperator,lattice_shape,prolong_spline
from trees import Tree,TreeOperator
from solver import solve

def pyramid(field,factor):
    if factor==1:return field
    sigma=.5*factor;radius=math.ceil(3*sigma)
    coords=torch.arange(-radius,radius+1,device=field.device,dtype=field.dtype)
    kernel=torch.exp(-.5*(coords/sigma).square());kernel/=kernel.sum()
    x=field.permute(3,0,1,2)[None]
    for axis in range(3):
        pad=[0]*6;pad[2*(2-axis)]=radius;pad[2*(2-axis)+1]=radius
        sh=[1,1,1];sh[axis]=len(kernel)
        weight=kernel.reshape(1,1,*sh).expand(3,1,*sh)
        x=F.conv3d(F.pad(x,pad,mode='replicate'),weight,groups=3)
    shape=[math.ceil(n/factor) for n in field.shape[:3]]
    return F.interpolate(x,size=shape,mode='trilinear',align_corners=True)[0].permute(1,2,3,0).contiguous()

def operator(model,shape,device,check=lambda:None,tile=128,leaves=False):
    if isinstance(model,Grid):return GridOperator(model,shape,device,check)
    return TreeOperator(model,shape,device,check,tile=tile,leaves=leaves)

def save_state(folder,model,coeff,metadata):
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    # A versioned immutable payload and atomically replaced pointer keep a crash
    # from pairing new metadata with old coefficients or topology.
    stamp=time.time_ns();p=folder/f'state_{stamp}.npz'
    arrays={'coefficients':coeff.detach().cpu().numpy()}
    if isinstance(model,Tree):arrays['nodes']=model.data[:model.count]
    with open(p,'wb') as f:np.savez(f,**arrays)
    old=read_json(folder/'checkpoint.json') if (folder/'checkpoint.json').exists() else None
    write_json(folder/'checkpoint.json',dict(model=model.metadata(),payload=str(p),**metadata))
    if old and old['payload']!=str(p):Path(old['payload']).unlink(missing_ok=True)

def load_state(folder,device):
    d=read_json(Path(folder)/'checkpoint.json');meta=d['model']
    with np.load(d['payload'],allow_pickle=False) as a:
        if meta['type']=='grid':m=Grid.from_meta(meta)
        else:
            m=Tree(meta['family'],meta['lo'],meta['hi']);m.data=a['nodes'].copy();m.count=meta['count'];m.k=meta['k']
        c=torch.tensor(a['coefficients'],device=device)
    return m,c,d

class Worker:
    def __init__(self,root,job,out,*,device='cuda:0',resume=None,spent=0,tile=128,seconds=1200):
        self.root=external(root);self.job=job;self.out=external(out);self.out.mkdir(exist_ok=True,parents=True)
        self.device=device;self.tile=tile;self.last_saved=0;self.stages=[];self.transfers=[]
        manifest=read_json(self.root/'manifest.json');self.target=manifest['targets'][job['target']]
        t=time.monotonic()
        if sha(self.target['canonical_path'])!=self.target['canonical_sha256']:raise ValueError('Target hash drift')
        self.field=torch.tensor(np.load(self.target['canonical_path']),device=device,dtype=torch.float64)
        self.mask=np.load(self.target['mask_path']).astype(bool)
        self.load_seconds=time.monotonic()-t;self.model=None;self.coeff=None;self.stage_index=-1
        self.resume=resume;self.resume_state=None;self.model_stage=-1
        if resume:
            self.model,self.coeff,self.resume_state=load_state(resume,device)
            self.stages=self.resume_state.get('stages',[]);self.transfers=self.resume_state.get('transfers',[])
            self.model_stage=self.resume_state.get('model_stage',self.resume_state['stage_index'])
        self.budget=Budget(seconds,spent)
        self.limit_seconds=seconds
    def progress(self,phase,**extra):
        row=dict(at=now(),phase=phase,job=self.job['id'],active_seconds=self.budget.elapsed,stage_index=self.stage_index,**extra)
        if str(self.device).startswith('cuda'):
            row.update(reserved_mib=torch.cuda.memory_reserved()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        write_json(self.out/'progress.json',row)
    def checkpoint(self,force=False):
        if self.model is None or self.coeff is None:return
        if force or time.monotonic()-self.last_saved>60:
            save_state(self.out,self.model,self.coeff,dict(stage_index=self.stage_index,
                active_seconds=self.budget.elapsed,stages=self.stages,transfers=self.transfers,
                phase='stage_ready',model_stage=self.model_stage,job=self.job))
            self.last_saved=time.monotonic()
    def callback(self,c,meta):
        self.coeff=c;self.progress('solving',solver=meta,actual_params=self.model.params)
        self.checkpoint()
    def fit(self,op,y,iterations=20000):
        self.coeff,meta=solve(op,y,self.coeff,self.budget.check,self.callback,iterations)
        self.progress('solve_finished',solver=meta,actual_params=self.model.params)
        return meta
    def run(self):
        j=self.job;lo=self.target['domain_min'];hi=self.target['domain_max'];tree=j['family'].startswith('tree_')
        final_k=(j['cap']//12 if 'affine' in j['family'] else (j['cap']-9)//3) if tree else None
        final=None
        if not tree:
            k=j['cap']//12 if j['family']=='gaussian_affine' else (j['cap']-9)//3 if j['family']=='gaussian_translation' else j['cap']//3
            dims=lattice_shape(k,[b-a for a,b in zip(lo,hi)],4 if j['family']=='bspline' else 2)
            final=Grid(j['family'],dims,lo,hi).at_factor(1)
        factors=[1] if j['version']=='native' else [4,2,1]
        limits=[self.limit_seconds] if len(factors)==1 else [self.limit_seconds*.1,self.limit_seconds*.3,self.limit_seconds]
        resume_index=self.resume_state['stage_index'] if self.resume_state else 0
        try:
            for si,(f,limit) in enumerate(zip(factors,limits)):
                if si<resume_index:continue
                self.stage_index=si;self.budget.stage(limit)
                start=self.budget.elapsed;self.progress('stage_start',factor=f)
                try:
                    self.budget.check();y=pyramid(self.field,f);shape=y.shape[:3]
                    already_ready=self.resume_state is not None and self.model_stage==si
                    if tree:
                        if self.model is None:
                            self.model=Tree(j['family'],lo,hi);self.coeff=y.new_zeros((4,3))
                        self.model_stage=si
                        self.checkpoint(force=True)
                        goal=max(1,math.ceil(final_k/(f**3)))
                        # Short intermediate solves permit growth; final geometry
                        # receives the entire remainder of the current stage.
                        op=operator(self.model,shape,self.device,self.budget.check,self.tile)
                        meta=self.fit(op,y,iterations=8 if self.model.k<goal else 20000)
                        while self.model.k<goal:
                            self.budget.check();scores=None
                            if j['family'].endswith('adaptive'):
                                r=op.forward(self.coeff)-y
                                leafop=operator(self.model,shape,self.device,self.budget.check,self.tile,leaves=True)
                                scores=leafop.adjoint(r.square().sum(-1,keepdim=True)).reshape(-1).cpu().numpy()
                                del leafop,r
                            # Freeze last usable state until operator construction succeeds.
                            updated=self.model.clone();added=updated.split(goal,scores,self.budget.check)
                            if not added:raise RuntimeError('tree_geometry_limit')
                            nc=torch.cat((self.coeff,self.coeff.new_zeros((updated.m-self.model.m,3))),0)
                            op=operator(updated,shape,self.device,self.budget.check,self.tile)
                            self.model,self.coeff=updated,nc
                            meta=self.fit(op,y,iterations=8 if self.model.k<goal else 20000)
                    else:
                        new=final if f==1 else final.at_factor(f)
                        if self.model is None:
                            self.model=new;self.coeff=y.new_zeros((new.m,3))
                            self.model_stage=si
                        elif not already_ready:
                            transfer_start=self.budget.elapsed
                            if j['family']=='bspline':
                                nc=prolong_spline(self.model,new,self.coeff,self.budget.check)
                                self.transfers.append(dict(stage=si,kind='exact_knot_insertion',seconds=self.budget.elapsed-transfer_start))
                            else:
                                original_limit=self.budget.limit
                                self.budget.limit=min(original_limit,transfer_start+60)
                                try:
                                    oldop=operator(self.model,shape,self.device,self.budget.check,self.tile)
                                    inherited=oldop.forward(self.coeff);del oldop
                                    newop=operator(new,shape,self.device,self.budget.check,self.tile)
                                    nc,tm=solve(newop,inherited,check=self.budget.check)
                                finally:self.budget.limit=original_limit
                                self.transfers.append(dict(stage=si,kind='gaussian_ls_projection',solver=tm,
                                    rmse_mm=math.sqrt(tm['mse_mm2']) if 'mse_mm2' in tm else None,seconds=self.budget.elapsed-transfer_start))
                                del inherited,newop
                            self.model,self.coeff=new,nc
                            self.model_stage=si
                        self.checkpoint(force=True)
                        op=operator(self.model,shape,self.device,self.budget.check,self.tile)
                        meta=self.fit(op,y)
                    self.stages.append(dict(index=si,factor=f,shape=list(shape),params=self.model.params,
                        elapsed_seconds=self.budget.elapsed-start,solver=meta))
                    self.checkpoint(force=True)
                    del op,y
                except Deadline:
                    self.stages.append(dict(index=si,factor=f,elapsed_seconds=self.budget.elapsed-start,
                        params=self.model.params if self.model else 0,solver=dict(converged=False,reason='time_limit')))
                    self.progress('stage_timed_out',factor=f)
                    self.checkpoint(force=True)
                self.resume_state=None
                if self.budget.elapsed>=self.limit_seconds:break
        except Deadline:pass
        if self.model is None or self.coeff is None:raise RuntimeError('No finite checkpoint before deadline')
        self.checkpoint(force=True)
        active=self.budget.elapsed;self.fit_active_final=active
        self.progress('evaluating',fit_active_seconds=active)
        reached=self.model.k==final_k if tree else self.model.shape==final.shape
        start=time.monotonic()
        op=operator(self.model,self.field.shape[:3],self.device,tile=self.tile)
        prediction=op.forward(self.coeff)
        if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('Nonfinite final field')
        sq=(prediction-self.field).square().sum(-1);mask=torch.as_tensor(self.mask,device=self.device)
        metrics=dict(full_rmse_mm=float(sq.mean().sqrt()),foreground_rmse_mm=float(sq[mask].mean().sqrt()),
                     background_rmse_mm=float(sq[~mask].mean().sqrt()),max_error_mm=float(sq.max().sqrt()))
        np.save(self.out/'field_mm.npy',prediction.cpu().numpy().astype(np.float32))
        last=next((s['solver'] for s in reversed(self.stages) if s['factor']==1),{})
        converged=bool(reached and last.get('converged',False))
        status='converged' if converged else 'unresolved' if reached else 'budget_not_reached'
        result=dict(**j,status=status,converged=converged,budget_reached=reached,actual_params=self.model.params,
                    metrics=metrics,stages=self.stages,transfers=self.transfers,fit_active_seconds=active,
                    input_loading_seconds=self.load_seconds,evaluation_seconds=time.monotonic()-start,
                    field_path=str(self.out/'field_mm.npy'),checkpoint=str(self.out/'checkpoint.json'),
                    source_sha256=self.target['canonical_sha256'],finished_at=now(),
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if str(self.device).startswith('cuda') else 0)
        write_json(self.out/'result.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--job',required=True)
    p.add_argument('--out',required=True);p.add_argument('--resume');p.add_argument('--spent',type=float,default=0)
    p.add_argument('--tile',type=int,default=128);p.add_argument('--seconds',type=float,default=1200);a=p.parse_args()
    manifest=read_json(Path(a.root)/'manifest.json')
    if sources()!=manifest['source_hashes']:raise RuntimeError('Frozen source hashes changed')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(0);np.random.seed(0);torch.cuda.set_device(0)
    j=next(j for j in manifest['jobs'] if j['id']==a.job);w=None
    try:
        w=Worker(a.root,j,a.out,resume=a.resume,spent=a.spent,tile=a.tile,seconds=a.seconds);w.run()
    except Exception as e:
        oom=isinstance(e,torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()
        elapsed=getattr(w,'fit_active_final',w.budget.elapsed) if w else a.spent
        write_json(Path(a.out)/'failure.json',dict(at=now(),error=str(e),traceback=traceback.format_exc(),
                   oom=oom,active_seconds=elapsed))
        raise

if __name__=='__main__':main()
