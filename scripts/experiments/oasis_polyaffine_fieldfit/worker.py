#!/usr/bin/env python
"""P81 isolated field-fitting worker; all generated files are external."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch

from runtime import atomic_json, read_json, sha, now, code_hashes, assert_external
from basis import Basis, create_basis
from data import load_target, indices_to_mm
from numerics import solve_d0, render_field, optimize_nonlinear, sample_field_points, grid_lipschitz_bound

MODES={'d0':'direct_dvf','d1':'bounded_dvf','svf':'svf'}


def nonlinear_progress_guard(callback, config):
    """Record a check before aborting finite runaway instead of accepting a fallback."""
    initial={}
    def checked(row):
        callback(row)
        for key in ('loss_mm2','validation_mm2'):
            if key not in row:continue
            value=float(row[key])
            if not math.isfinite(value):
                raise FloatingPointError('solver_diverged: nonfinite '+key)
            initial.setdefault(key,value)
            limit=min(float(config.get('divergence_ceiling_mm2',1e6)),
                      max(float(config.get('divergence_floor_mm2',100.)),
                          float(config.get('divergence_factor',100.))*initial[key]))
            if value>limit:
                raise FloatingPointError(f'solver_diverged: {key}={value:g}, initial={initial[key]:g}, limit={limit:g}')
    return checked


def coarse_geometry(shape, affine, factor):
    new=tuple(max(2, int(round((n-1)/factor))+1) for n in shape)
    h=np.array(affine,dtype=np.float64,copy=True)
    h[:3,:3]=h[:3,:3] @ np.diag([(n-1)/(m-1) for n,m in zip(shape,new)])
    return new,h


def tree_leaves(family, cap):
    return cap//12 if 'affine' in family else (cap-9)//3


class Worker:
    def __init__(self, root, job, output, config, target, *, smoke=False):
        self.root,self.job,self.out=Path(root),job,assert_external(output)
        self.out.mkdir(parents=True,exist_ok=True)
        self.config,self.target,self.smoke=config,target,smoke
        self.start=time.monotonic(); self.solves=[]; self.events=[]; self.endpoints=[]
        self.excluded_evaluation_sec=0.
        self.last_inherited=None
        self.data=load_target(target)
        self.device=torch.device('cuda:0')
        self.points=torch.tensor(self.data['points_mm'],device=self.device,dtype=torch.float64)
        self.values=torch.tensor(self.data['targets_mm'],device=self.device,dtype=torch.float64)
        self.shape=tuple(self.data['shape']); self.affine=np.asarray(self.data['affine'])
        vi=self.data['val_indices']
        self.valpoints=torch.tensor(indices_to_mm(vi,self.shape,self.affine),device=self.device,dtype=torch.float32)
        self.valvalues=torch.tensor(np.asarray(self.data['field']).reshape(-1,3)[vi],device=self.device,dtype=torch.float32)
        self.progress(phase='loaded')

    def progress(self, **row):
        payload={'at':now(),'job_id':self.job['id'],'elapsed_sec':time.monotonic()-self.start,
                 'current_reserved_mib':torch.cuda.memory_reserved()/2**20,
                 'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                 'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,**row}
        atomic_json(self.out/'progress.json',payload)

    def callback(self,row):
        self.progress(**{'phase':'fitting',**row})

    def linear(self,basis,coeff=None,final=False):
        c=dict(self.config['lsqr'])
        c['iter_lim']=self.config['lsqr']['final_iter_lim' if final else 'iter_lim']
        coeff,meta=solve_d0(basis,self.points,self.values,initial=coeff,config=c,callback=self.callback)
        self.solves.append(meta)
        if final and not meta['converged']:
            c['iter_lim']*=2
            coeff,second=solve_d0(basis,self.points,self.values,initial=coeff,config=c,callback=self.callback)
            second['extension_of_unconverged_solve']=True;self.solves.append(second)
        return coeff

    def nonlinear(self,basis,coeff,steps,factor=1,min_steps=0):
        if steps<=0:return coeff
        shape,affine=coarse_geometry(self.shape,self.affine,factor)
        c={**self.config['nonlinear'],'max_steps':steps,'min_steps':min(min_steps,steps)}
        c.update(steps=self.config['integration_steps'],chunk=self.config['render_chunk'],rho=self.config['rho'])
        coeff,meta=optimize_nonlinear(basis,coeff.float(),shape,affine,
                    fit_targets_mm=self.values.float(),mode=MODES[self.job['mode']],
                    validation_targets_mm=self.valvalues,config=c,callback=nonlinear_progress_guard(self.callback,c),
                    fit_points_mm=self.points.float(),validation_points_mm=self.valpoints)
        meta['grid_factor']=factor;self.solves.append(meta)
        return coeff

    def predict_samples(self,basis,coeff,factor=4):
        with torch.no_grad():
            if self.job['mode']=='d0':
                return basis.evaluate(self.points,coeff.double())
            shape,h=coarse_geometry(self.shape,self.affine,factor)
            field=render_field(basis,coeff.float(),shape,h,MODES[self.job['mode']],
                               chunk=self.config['render_chunk'],steps=self.config['integration_steps'])
            return sample_field_points(field,self.points.float(),h).double()

    def save_endpoint(self,basis,coeff,cap,*,supplement=False):
        from metrics import evaluate_fields
        from diagnostics import global_affine_residual_normalization, blending_derivative_diagnostics, inverse_consistency
        phase='supplement' if supplement else 'endpoint'
        folder=self.out/phase/f'cap{cap}_p{basis.actual_motion_params}'
        folder.mkdir(parents=True,exist_ok=False)
        self.progress(phase='rendering',budget_cap=cap,actual_params=basis.actual_motion_params)
        before=time.monotonic()
        with torch.no_grad():
            field,render_meta=render_field(basis,coeff.float(),self.shape,self.affine,
                         MODES[self.job['mode']],chunk=self.config['render_chunk'],
                         steps=self.config['integration_steps'],rho=self.config['rho'],return_metadata=True)
            if not torch.isfinite(field).all():raise FloatingPointError('Nonfinite final rendered field')
            if self.job['mode']=='d1':
                bound,certificate=grid_lipschitz_bound(field.double(),self.affine,return_metadata=True)
                bound=float(bound);render_meta['certificate']=certificate
                render_meta['serialized_precision_bound']=bound
                if bound>self.config['rho']+1e-7:raise RuntimeError('Final D1 certificate failed')
            np.save(folder/'field_mm.npy',field.cpu().numpy())
            fi=torch.as_tensor(self.data['fit_indices'],device=self.device)
            train_mse=float((field.reshape(-1,3)[fi].double()-self.values).square().sum(-1).mean())
            if self.job['mode']=='svf':
                finer=render_field(basis,coeff.float(),self.shape,self.affine,'svf',
                         chunk=self.config['render_chunk'],steps=self.config['integration_steps']+1)
                error=(finer-field).square().sum(-1)
                fg=torch.as_tensor(np.asarray(self.data['mask']),device=self.device,dtype=torch.bool)
                change=float(error[fg].double().mean().sqrt())
                render_meta['integration_sensitivity']={
                    'primary_steps':self.config['integration_steps'],
                    'diagnostic_steps':self.config['integration_steps']+1,
                    'foreground_vector_rmse_mm':change,
                    'max_component_mm':float((finer-field).abs().max()),
                    'flag_threshold_mm':max(.001,.05*train_mse**.5),
                    'sensitive':change>max(.001,.05*train_mse**.5)}
                del finer,error,fg
                inverse=render_field(basis,-coeff.float(),self.shape,self.affine,'svf',
                         chunk=self.config['render_chunk'],steps=self.config['integration_steps'])
                render_meta['inverse']=inverse_consistency(field,inverse,self.affine,mask=self.data['mask'])
                render_meta['inverse']['construction']='same_velocity_coefficients_negated_then_exponentiated'
                del inverse
            del field
        np.save(folder/'coefficients.npy',coeff.detach().cpu().numpy())
        atomic_json(folder/'basis.json',basis.to_dict())
        render_sec=time.monotonic()-before
        self.progress(phase='evaluating',budget_cap=cap)
        eval_start=time.monotonic()
        reloaded=np.load(folder/'field_mm.npy',mmap_mode='r')
        metrics=evaluate_fields(reloaded,self.data,surface_metrics=not self.smoke)
        metrics['global_affine_residual']=global_affine_residual_normalization(reloaded,self.data)
        metrics['analytic_blending']=blending_derivative_diagnostics(basis,coeff,self.points.float())
        metrics['analytic_blending']['vector_field_role']={'d0':'displacement','d1':'preconstraint_displacement','svf':'velocity'}[self.job['mode']]
        if self.job['mode']!='svf':
            render_meta['inverse']={'status':'not_evaluated','reason':'No validated inverse extension/domain contract for this direct-map result'}
        meta={'status':'complete','job_id':self.job['id'],'target_id':self.target['id'],
              'source_method':self.target['source_method'],'pair_rank':self.target['pair_rank'],
              'family':self.job['family'],'mode':self.job['mode'],'initialization':self.job['initialization'],
              'kind':self.job['kind'],'budget_cap':cap,'actual_motion_params':basis.actual_motion_params,
              'leaf_count':basis.leaf_count,'scalar_features':basis.feature_count,'supplement':supplement,
              'train_mse_mm2':train_mse,'metrics':metrics,'renderer':render_meta,
              'basis_path':str(folder/'basis.json'),'coefficients_path':str(folder/'coefficients.npy'),
              'inherited_coefficients_path':self.last_inherited,
              'inherited_coefficients_sha256':sha(self.last_inherited) if self.last_inherited else None,
              'field_path':str(folder/'field_mm.npy'), 'field_sha256':sha(folder/'field_mm.npy'),
              'basis_sha256':sha(folder/'basis.json'),'coefficients_sha256':sha(folder/'coefficients.npy'),
              'input_sha256':self.target.get('canonical_sha256',self.target.get('hashes')),
              'elapsed_fit_and_structure_sec':eval_start-self.start-render_sec-self.excluded_evaluation_sec,
              'final_render_sec':render_sec,'evaluation_sec':time.monotonic()-eval_start,
              'elapsed_worker_sec':time.monotonic()-self.start,
              'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
              'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
              'peak_cpu_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'model_serialized_bytes':(folder/'basis.json').stat().st_size+(folder/'coefficients.npy').stat().st_size,
              'dense_field_bytes':(folder/'field_mm.npy').stat().st_size,
              'solves':self.solves.copy(),'split_events':self.events.copy(),'created_at':now(),
              'scientific_status':'solver_unresolved' if self.job['mode']=='d0' and not self.solves[-1]['converged'] else 'evaluated'}
        atomic_json(folder/'metrics.json',meta)
        self.excluded_evaluation_sec+=time.monotonic()-before
        if not supplement:
            pointer=self.root/'endpoints'/self.job['id']/f'cap{cap}.json'
            if pointer.exists():raise FileExistsError('Refusing to overwrite existing endpoint '+str(pointer))
            atomic_json(pointer,meta)
            self.endpoints.append(str(pointer))
        return meta

    def run(self):
        j=self.job;budgets=j['budgets']; family=j['family']
        frozen=j.get('geometry_from')
        if frozen:
            source=read_json(self.root/frozen)
            for kind in ('basis','coefficients'):
                if sha(source[kind+'_path'])!=source[kind+'_sha256']:
                    raise RuntimeError('Frozen geometry dependency changed')
            basis=Basis.from_dict(read_json(source['basis_path']))
        else:
            basis=create_basis(family,max(budgets),self.target['domain_min'],self.target['domain_max'],
                               start_root=family.startswith('tree_'),
                               gaussian_sigma_over_h=self.config['gaussian_sigma_over_h'])
        coeff=torch.zeros((basis.feature_count,3),device=self.device,dtype=torch.float64)
        if j['kind']=='solver_diagnostic':
            from diagnostics import optimize_sampled_d0
            cold_start=time.monotonic();cold=self.linear(basis,final=True);cold_meta=self.solves[-1]
            cold_seconds=time.monotonic()-cold_start
            inherited=source.get('inherited_coefficients_path') or source['coefficients_path']
            if source.get('inherited_coefficients_path') and sha(inherited)!=source['inherited_coefficients_sha256']:
                raise RuntimeError('Inherited split coefficients changed')
            warm_init=torch.tensor(np.load(inherited),device=self.device,dtype=torch.float64)
            warm_start=time.monotonic();warm=self.linear(basis,warm_init,final=True);warm_meta=self.solves[-1]
            warm_seconds=time.monotonic()-warm_start
            mse=lambda c:float((basis.evaluate(self.points,c.double())-self.values).square().sum(-1).mean())
            coeff,diag=optimize_sampled_d0(basis,self.points.float(),self.values.float(),coeff.float(),
                                         max_steps=1000,lbfgs_evals=100,callback=self.callback,
                                         conditioning_config=self.config['nonlinear'])
            cold_mse=mse(cold);diag.update(solver='sampled_adam_lbfgs_vs_ls',converged=cold_meta['converged'],
                cold_ls_mse_mm2=cold_mse,warm_ls_mse_mm2=mse(warm),
                warm_start_kind='split_inherited' if source.get('inherited_coefficients_path') else 'saved_ls_solution',
                adam_training_gap_mm2=diag['adam_endpoint_mse_mm2']-cold_mse,
                final_training_gap_mm2=diag['final_endpoint_mse_mm2']-cold_mse,
                cold_ls_seconds=cold_seconds,warm_ls_seconds=warm_seconds)
            self.solves.append(diag);self.save_endpoint(basis,coeff,budgets[0]);return
        if j['mode']=='svf' and j['initialization']=='global_affine_ls':
            from initialization import fit_global_affine_initialization
            coeff,meta=fit_global_affine_initialization(basis,self.points,self.values,self.config['lsqr'])
            self.solves.append(meta)
        elif j['mode']=='d0' or j['initialization']=='ls':
            coeff=self.linear(basis,final=bool(frozen) or not family.startswith('tree_'))
        if frozen or not family.startswith('tree_'):
            if j['mode']!='d0':
                for factor,steps in self.config['fixed_grid_steps']:
                    coeff=self.nonlinear(basis,coeff,steps,factor,min_steps=150 if factor==1 else 0)
            self.save_endpoint(basis,coeff,budgets[0]);return
        # A single growing path shares all three main budgets. Zero contrasts
        # preserve the complete field before every refit, including SVF velocity.
        growth_used=0
        samek_saved=set()
        for index,cap in enumerate(budgets):
            goal=tree_leaves(family,cap)
            while basis.leaf_count<goal:
                pred=self.predict_samples(basis,coeff)
                old=basis.evaluate(self.points[:512],coeff.double()).detach()
                before=time.monotonic()
                next_samek=[k for k in (64,128,256) if 'translation' in family and basis.leaf_count<k<=goal]
                stop=min([goal]+next_samek)
                new,new_coeff,event=basis.split(coeff,self.points,pred-self.values,max_leaves=stop,growth=.25)
                if new.leaf_count==basis.leaf_count:raise RuntimeError('Tree cannot reach requested budget under frozen geometric limits')
                error=float((new.evaluate(self.points[:512],new_coeff.double())-old).abs().max())
                if error>1e-6:raise RuntimeError('Split does not preserve represented field')
                event.update(sample_preservation_max_mm=error,split_transfer_sec=time.monotonic()-before)
                self.events.append(event);basis,coeff=new,new_coeff
                if basis.leaf_count==goal:
                    inherited=self.out/'split_initial'/f'leaf{goal}.npy'
                    inherited.parent.mkdir(exist_ok=True)
                    np.save(inherited,coeff.detach().cpu().numpy());self.last_inherited=str(inherited)
                if j['mode']=='d0':coeff=self.linear(basis,coeff)
                else:
                    steps=min(self.config['tree_growth_steps_per_split'],self.config['tree_growth_total_steps']-growth_used)
                    coeff=self.nonlinear(basis,coeff,steps,4);growth_used+=steps
                if 'translation' in family and basis.leaf_count in (64,128,256) and basis.leaf_count not in samek_saved:
                    if j['mode']=='d0':coeff=self.linear(basis,coeff,final=True)
                    self.save_endpoint(basis,coeff,basis.actual_motion_params,supplement=True)
                    samek_saved.add(basis.leaf_count)
            if j['mode']=='d0':coeff=self.linear(basis,coeff,final=True)
            else:
                coeff=self.nonlinear(basis,coeff,self.config['tree_half_steps'][index],2)
                coeff=self.nonlinear(basis,coeff,self.config['tree_native_steps'][index],1,min_steps=150)
            self.save_endpoint(basis,coeff,cap)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job-id',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output=assert_external(a.output);a.output.mkdir(parents=True,exist_ok=True)
    try:
        manifest=read_json(a.root/'launch_manifest.json')
        if code_hashes()!=manifest['code_hashes']:raise RuntimeError('Worker source hash mismatch')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Worker requires explicit physical GPU0 mask')
        torch.set_num_threads(manifest['queue_policy']['cpu_threads'])
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(0);np.random.seed(0);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
        job=next(j for j in manifest['jobs'] if j['id']==a.job_id)
        inputs=read_json(a.root/'input_manifest.json')
        target=next(t for t in inputs['targets'] if t['id']==job['target_id'])
        if target['status']!='passed':raise RuntimeError('Target failed coordinate/input audit')
        if sha(target['canonical_path'])!=target['canonical_sha256']:
            raise RuntimeError('Prepared reference field changed after input audit')
        w=Worker(a.root,job,a.output,manifest['fit_config'],target)
        w.run()
        atomic_json(a.output/'result.json',{'status':'complete','job_id':job['id'],
                    'endpoints':w.endpoints,'at':now(),'elapsed_sec':time.monotonic()-w.start,
                    'requested_backend':{'cudnn_benchmark':False,'cudnn_deterministic':True,'tf32':False},
                    'effective_backend':{'cudnn_benchmark':torch.backends.cudnn.benchmark,
                         'cudnn_deterministic':torch.backends.cudnn.deterministic,
                         'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32}})
    except BaseException as exc:
        atomic_json(a.output/'failure.json',{'status':'failed','job_id':a.job_id,
                    'error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc(),'at':now()})
        raise


if __name__=='__main__':main()
