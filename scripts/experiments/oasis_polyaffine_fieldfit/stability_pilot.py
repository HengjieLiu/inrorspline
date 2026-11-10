#!/usr/bin/env python
"""P81 v1.3 multi-step regression for foreground-support conditioning failure.

Uses the actual previously failing rank0 bounded-two source and production LS
limits. It preserves all v2 artifacts and writes a separate stability gate.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from basis import create_basis
from data import load_target, indices_to_mm
from numerics import solve_d0, optimize_nonlinear, render_field, sample_field_points
from initialization import fit_global_affine_initialization
from worker import nonlinear_progress_guard
from runtime import REPO,HERE,read_json,atomic_json,sha,now,assert_external,code_hashes

SCIENTIFIC_NAMES=("basis.py","data.py","numerics.py","stability_pilot.py","runtime.py",
                  "initialization.py","worker.py")


def scientific_hashes():
    return {name:sha(HERE/name) for name in SCIENTIFIC_NAMES}


def coarse_geometry(shape, affine, factor):
    coarse=tuple(max(2,int(round((n-1)/factor))+1) for n in shape)
    h=np.array(affine,dtype=np.float64,copy=True)
    h[:3,:3]=h[:3,:3]@np.diag([(n-1)/(m-1) for n,m in zip(shape,coarse)])
    return coarse,h


def full_field_stats(field, coefficient, fit_indices, target):
    if not torch.isfinite(field).all() or not torch.isfinite(coefficient).all():
        raise FloatingPointError("Nonfinite pilot coefficient/native field")
    power=field.double().square().sum(-1)
    return dict(coefficient_abs_max_mm=float(coefficient.abs().max()),
                full_box_vector_max_mm=float(power.max().sqrt()),
                full_box_vector_rms_mm=float(power.mean().sqrt()),
                fit_endpoint_mse_mm2=float((field.reshape(-1,3)[fit_indices].double()-target.double()).square().sum(-1).mean()))


def run(root, config_path):
    root=assert_external(root)
    folder=root/"stability"/"full_box_v2"
    folder.mkdir(parents=True,exist_ok=False)
    config=read_json(config_path);config_hash=sha(config_path)
    start_hashes=scientific_hashes()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!='0':
        raise RuntimeError("Stability pilot requires physical GPU0 mask")
    torch.set_num_threads(3);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();rows=[]

    def progress(phase, **extra):
        payload=dict(at=now(),phase=phase,elapsed_sec=time.monotonic()-started,
                     current_reserved_mib=torch.cuda.memory_reserved()/2**20,
                     peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        payload.update(extra);atomic_json(folder/"progress.json",payload)
        print(payload,flush=True)

    try:
        inputs=read_json(root/"input_manifest.json")
        record=next(t for t in inputs['targets'] if t['id']=='rank000_bounded2')
        if record['status']!='passed' or sha(record['canonical_path'])!=record['canonical_sha256']:
            raise RuntimeError("Stability pilot requires hash-verified rank000_bounded2")
        data=load_target(record);shape=tuple(data['shape']);affine=data['affine']
        points=torch.tensor(data['points_mm'],device='cuda:0',dtype=torch.float64)
        values=torch.tensor(data['targets_mm'],device='cuda:0',dtype=torch.float64)
        fit_indices=torch.tensor(np.asarray(data['fit_indices']).copy(),device='cuda:0',dtype=torch.long)
        val_indices=np.asarray(data['val_indices'])
        val_points=torch.tensor(indices_to_mm(val_indices,shape,affine),device='cuda:0',dtype=torch.float32)
        val_values=torch.tensor(np.asarray(data['field']).reshape(-1,3)[val_indices],device='cuda:0')
        diagonal=float(np.linalg.norm(np.asarray(record['domain_max'])-record['domain_min']))
        identity_loss=float(values.square().sum(-1).mean())
        sanity=dict(maximum_full_box_displacement_mm=diagonal,
                    maximum_coefficient_abs_mm=100*diagonal,
                    maximum_observed_endpoint_mse_mm2=1e6)
        chosen_basis=chosen_coeff=None
        for cap in (768,1536,3072):
            progress('linear_start',budget_cap=cap)
            basis=create_basis('bspline',cap,record['domain_min'],record['domain_max'])
            ls_config={**config['lsqr'],'iter_lim':config['lsqr']['final_iter_lim']}
            coeff,meta=solve_d0(basis,points,values,config=ls_config,
                callback=lambda row:progress('linear_component',budget_cap=cap,**row))
            attempts=[meta]
            if not meta['converged']:
                ls_config['iter_lim']*=2
                coeff,meta=solve_d0(basis,points,values,initial=coeff,config=ls_config,
                    callback=lambda row:progress('linear_extension',budget_cap=cap,**row))
                attempts.append(meta)
            with torch.no_grad():
                field=render_field(basis,coeff.float(),shape,affine,'direct_dvf',chunk=config['render_chunk'])
                stats=full_field_stats(field,coeff,fit_indices,values)
            row=dict(kind='production_limit_lsqr',budget_cap=cap,actual_parameters=basis.actual_motion_params,
                     attempts=attempts,stats=stats,identity_endpoint_mse_mm2=identity_loss,
                     converged=meta['converged'])
            # D0 is deliberately unconstrained outside its foreground loss.
            # Finite iteration-limit results remain explicitly unresolved.
            row['passed']=stats['fit_endpoint_mse_mm2']<identity_loss
            row['scientific_status']='converged' if meta['converged'] else 'solver_unresolved'
            row['outside_foreground_constraint']='none; extrapolation is recorded, not bounded'
            rows.append(row);atomic_json(folder/f'lsqr_cap{cap}.json',row)
            np.save(folder/f'lsqr_cap{cap}_coeff.npy',coeff.cpu().numpy())
            del field
            if cap==1536:chosen_basis,chosen_coeff=basis,coeff
            progress('linear_finished',budget_cap=cap,passed=row['passed'],**stats)
        if not all(r['passed'] for r in rows):
            raise RuntimeError("Production-limit LSQR finite/improvement diagnostic failed")
        for initialization in ('zero','global_affine_ls'):
            if initialization=='zero':
                coeff=torch.zeros_like(chosen_coeff).float(); init_meta={'kind':'zero'}
            else:
                coeff,init_meta=fit_global_affine_initialization(chosen_basis,points,values,config)
                coeff=coeff.float()
            with torch.no_grad():
                initial_field=render_field(chosen_basis,coeff,shape,affine,'svf',chunk=config['render_chunk'],steps=config['integration_steps'])
                initial_stats=full_field_stats(initial_field,coeff,fit_indices,values)
                del initial_field
            phases=[]
            for factor,steps in ((4,100),(2,25),(1,5)):
                gshape,h=coarse_geometry(shape,affine,factor)
                nl={**config['nonlinear'],'max_steps':steps,'min_steps':steps,
                    'check_interval':5,'patience':1000,'chunk':config['render_chunk'],
                    'steps':config['integration_steps'],'rho':config['rho']}
                progress('svf_stage_start',initialization=initialization,factor=factor,steps=steps)
                coeff,meta=optimize_nonlinear(chosen_basis,coeff,gshape,h,
                    fit_targets_mm=values.float(),fit_points_mm=points.float(),
                    validation_targets_mm=val_values,validation_points_mm=val_points,
                    mode='svf',config=nl,
                    callback=nonlinear_progress_guard(lambda row:progress('nonlinear',**row),nl))
                phases.append(dict(factor=factor,requested_steps=steps,optimizer=meta))
                np.save(folder/f'svf_{initialization}_factor{factor}_coeff.npy',coeff.cpu().numpy())
            with torch.no_grad():
                final_field=render_field(chosen_basis,coeff,shape,affine,'svf',chunk=config['render_chunk'],steps=config['integration_steps'])
                final_stats=full_field_stats(final_field,coeff,fit_indices,values)
                sampled=sample_field_points(final_field,val_points,affine)
                validation=float((sampled-val_values).double().square().sum(-1).mean())
                extra=render_field(chosen_basis,coeff,shape,affine,'svf',chunk=config['render_chunk'],steps=config['integration_steps']+1)
                integration_rmse=float((extra-final_field).double().square().sum(-1).mean().sqrt())
                del extra,final_field
            row=dict(kind='multistep_svf',budget_cap=1536,initialization=initialization,
                     initialization_details=init_meta,
                     phases=phases,initial=initial_stats,final=final_stats,validation_endpoint_mse_mm2=validation,
                     final_integration_7_8_rmse_mm=integration_rmse,
                     passed=(final_stats['fit_endpoint_mse_mm2']<initial_stats['fit_endpoint_mse_mm2']
                       and final_stats['full_box_vector_max_mm']<sanity['maximum_full_box_displacement_mm']
                       and final_stats['coefficient_abs_max_mm']<sanity['maximum_coefficient_abs_mm']
                       and validation<1e6 and integration_rmse<.05))
            rows.append(row);atomic_json(folder/f'svf_{initialization}.json',row)
            progress('svf_finished',initialization=initialization,passed=row['passed'],**final_stats)
        if scientific_hashes()!=start_hashes or sha(config_path)!=config_hash:
            raise RuntimeError("Scientific source/config changed during stability pilot")
        receipt=dict(passed=all(r['passed'] for r in rows),at=now(),
                     input_manifest_sha256=sha(root/'input_manifest.json'),config_sha256=config_hash,
                     config_path=str(config_path),source_hashes=code_hashes(),scientific_source_hashes=start_hashes,
                     target_id=record['id'],target_sha256=record['canonical_sha256'],sanity_limits=sanity,
                     rows=rows,elapsed_sec=time.monotonic()-started,
                     peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                     limitations=['A conditioning regression on the formerly failing pair, not completed scientific evaluation.',
                       'No regularizer or fitting-loss change; full-box scaling is an invertible coefficient-coordinate transform.',
                       'LS nullspace/minimum-correction selection can depend on scaling and initialization.',
                       'Unconstrained local D0 extrapolation and iteration-limit nonconvergence remain explicit diagnostics.',
                       'SVF uses zero or a globally fitted affine displacement interpreted as initial velocity, not its affine logarithm.',
                       'Concurrent GPU0 validation load makes pilot timing diagnostic.'])
        atomic_json(folder/'summary.json',receipt)
        if not receipt['passed']:raise RuntimeError("Multi-step SVF stability gate failed")
        atomic_json(root/'STABILITY_PASSED.json',receipt)
        return receipt
    except BaseException as exc:
        atomic_json(folder/'FAILED.json',dict(passed=False,at=now(),error=type(exc).__name__+': '+str(exc),
                    rows=rows,scientific_source_hashes=start_hashes,config_sha256=config_hash,
                    traceback=traceback.format_exc(),elapsed_sec=time.monotonic()-started))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=REPO/'experiments/2026-11-05_oasis_adaptive_local_affine_fieldfit/protocol_v1.3.json')
    args=parser.parse_args();r=run(args.root,args.config)
    print({'passed':r['passed'],'rows':len(r['rows']),'elapsed_sec':r['elapsed_sec']},flush=True)


if __name__=='__main__':main()
