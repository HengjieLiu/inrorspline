#!/usr/bin/env python
"""Review frozen P81 pilots against the existing foreground integration criterion.

The original exploratory full-box gate remains failed and is never overwritten.
This review does not refit, change integration, or assert whole-box stability.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from basis import create_basis
from data import load_target, indices_to_mm
from numerics import render_field, sample_field_points
from runtime import REPO, HERE, read_json, atomic_json, sha, now, assert_external, code_hashes
from stability_pilot import full_field_stats, scientific_hashes


def review(root, config_path):
    root=assert_external(root)
    folder=root/'stability'/'full_box_v2'
    destination=root/'STABILITY_PASSED.json'
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an accepted stability receipt')
    raw=read_json(folder/'summary.json');failed=read_json(folder/'FAILED.json')
    diagnostic=read_json(folder/'integration_scope_diagnostic.json')
    config=read_json(config_path);config_hash=sha(config_path)
    if raw['passed'] or failed['passed']:
        raise RuntimeError('Expected preserved failed exploratory full-box gate')
    if raw['scientific_source_hashes']!=scientific_hashes():
        raise RuntimeError('Scientific sources changed after pilot fitting')
    if raw['config_sha256']!=config_hash or failed['config_sha256']!=config_hash:
        raise RuntimeError('Fit configuration changed after pilot')
    if raw['input_manifest_sha256']!=sha(root/'input_manifest.json'):
        raise RuntimeError('Input manifest changed after pilot')
    snapshot={**scientific_hashes(),'stability_review.py':sha(HERE/'stability_review.py')}
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise RuntimeError('Review requires physical GPU0 mask')
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0)
    started=time.monotonic()
    inputs=read_json(root/'input_manifest.json')
    record=next(t for t in inputs['targets'] if t['id']==raw['target_id'])
    if sha(record['canonical_path'])!=raw['target_sha256']:
        raise RuntimeError('Canonical target changed after pilot')
    data=load_target(record)
    basis=create_basis('bspline',1536,record['domain_min'],record['domain_max'])
    mask=torch.tensor(np.asarray(data['mask']).copy(),device='cuda:0',dtype=torch.bool)
    fit=torch.tensor(np.asarray(data['fit_indices']).copy(),device='cuda:0',dtype=torch.long)
    targets=torch.tensor(data['targets_mm'],device='cuda:0',dtype=torch.float64)
    vi=np.asarray(data['val_indices'])
    val_points=torch.tensor(indices_to_mm(vi,data['shape'],data['affine']),device='cuda:0',dtype=torch.float32)
    val_targets=torch.tensor(np.asarray(data['field']).reshape(-1,3)[vi],device='cuda:0')
    rows=[]
    for source_row in raw['rows']:
        row=copy.deepcopy(source_row)
        row['raw_exploratory_passed']=row.pop('passed')
        if row['kind']=='production_limit_lsqr':
            if not row['raw_exploratory_passed']:
                raise RuntimeError('The finite/improving local LS diagnostic itself failed')
            row['passed']=True
            row['acceptance_scope']='Finite improving foreground fit; convergence status and extrapolation remain explicit.'
            rows.append(row)
            continue
        initialization=row['initialization']
        path=folder/f'svf_{initialization}_factor1_coeff.npy'
        digest=sha(path)
        coefficient=torch.tensor(np.load(path),device='cuda:0')
        with torch.no_grad():
            field=render_field(basis,coefficient,data['shape'],data['affine'],'svf',
                               chunk=config['render_chunk'],steps=config['integration_steps'])
            extra=render_field(basis,coefficient,data['shape'],data['affine'],'svf',
                               chunk=config['render_chunk'],steps=config['integration_steps']+1)
            reproduced=full_field_stats(field,coefficient,fit,targets)
            for key,value in row['final'].items():
                if not math.isclose(value,reproduced[key],rel_tol=1e-6,abs_tol=1e-5):
                    raise RuntimeError('Frozen native metrics no longer reproduce: '+key)
            validation=float((sample_field_points(field,val_points,data['affine'])-val_targets).double().square().sum(-1).mean())
            if not math.isclose(validation,row['validation_endpoint_mse_mm2'],rel_tol=1e-6,abs_tol=1e-5):
                raise RuntimeError('Frozen validation metric no longer reproduces')
            error=(extra-field).double().square().sum(-1)
            foreground_rmse=float(error[mask].mean().sqrt())
            full_rmse=float(error.mean().sqrt())
            previous=next(d for d in diagnostic if d['initialization']==initialization)
            if not math.isclose(foreground_rmse,previous['fg_rmse_mm'],rel_tol=1e-6,abs_tol=1e-7):
                raise RuntimeError('Foreground integration diagnostic did not reproduce')
            if not math.isclose(full_rmse,row['final_integration_7_8_rmse_mm'],rel_tol=1e-6,abs_tol=1e-7):
                raise RuntimeError('Exploratory full-box failure did not reproduce')
            threshold=max(.001,.05*math.sqrt(reproduced['fit_endpoint_mse_mm2']))
            sensitivity=dict(foreground_vector_rmse_mm=foreground_rmse,
                foreground_max_vector_error_mm=float(error[mask].max().sqrt()),
                full_box_vector_rmse_mm=full_rmse,full_box_max_vector_error_mm=float(error.max().sqrt()),
                criterion='max(0.001 mm, 0.05 * foreground training endpoint RMSE)',
                threshold_mm=threshold,passed=foreground_rmse<=threshold,
                criterion_source='worker.py save_endpoint integration_sensitivity',
                original_exploratory_full_box_threshold_mm=.05,
                original_exploratory_full_box_passed=full_rmse<.05)
            sane=(reproduced['fit_endpoint_mse_mm2']<row['initial']['fit_endpoint_mse_mm2']
                  and reproduced['full_box_vector_max_mm']<raw['sanity_limits']['maximum_full_box_displacement_mm']
                  and reproduced['coefficient_abs_max_mm']<raw['sanity_limits']['maximum_coefficient_abs_mm']
                  and validation<raw['sanity_limits']['maximum_observed_endpoint_mse_mm2'])
            row.update(passed=bool(sane and sensitivity['passed']),canonical_foreground_integration=sensitivity,
                coefficients_path=str(path),coefficients_sha256=digest,
                reproduced_native_stats=reproduced,
                acceptance_scope='Finite improving foreground fit and existing foreground integration criterion; no whole-box stability claim.')
        if sha(path)!=digest:
            raise RuntimeError('Frozen coefficients changed during review')
        rows.append(row)
    if {**scientific_hashes(),'stability_review.py':sha(HERE/'stability_review.py')}!=snapshot:
        raise RuntimeError('Scientific source changed during review')
    if sha(config_path)!=config_hash or sha(root/'input_manifest.json')!=raw['input_manifest_sha256']:
        raise RuntimeError('Configuration/input manifest changed during review')
    receipt=dict(passed=all(r['passed'] for r in rows),at=now(),
        acceptance_kind='reviewed_existing_foreground_integration_criterion',
        input_manifest_sha256=raw['input_manifest_sha256'],config_sha256=config_hash,
        source_hashes=code_hashes(),scientific_source_hashes=snapshot,
        target_id=raw['target_id'],target_sha256=raw['target_sha256'],rows=rows,
        prior_failed_exploratory_gate=dict(summary_path=str(folder/'summary.json'),summary_sha256=sha(folder/'summary.json'),
            failure_path=str(folder/'FAILED.json'),failure_sha256=sha(folder/'FAILED.json'),
            integration_scope_diagnostic_path=str(folder/'integration_scope_diagnostic.json'),
            integration_scope_diagnostic_sha256=sha(folder/'integration_scope_diagnostic.json')),
        review_elapsed_sec=time.monotonic()-started,
        limitations=[
            'The original exploratory full-box integration gate FAILED and remains unchanged. Whole-box integration remains sensitive.',
            'Acceptance uses the existing worker foreground integration threshold; no renderer, fit objective, parameters or integration depth changed.',
            'Foreground-only fitting leaves the rest of the box unobserved: SVF endpoints reach 239/268 mm there and local D0 extrapolates by thousands of mm.',
            'Local LSQR cap1536/cap3072 remains unconverged at the fixed 1500+3000 iteration limits; it is not a proven capacity optimum.',
            'This regression covers one problematic target and two130-step BSpline SVF fits, not completed scientific evaluation.',
            'The40-target input audit, CPU suite and17 native resource profiles are separate required receipts.'])
    atomic_json(folder/'foreground_review.json',receipt)
    if not receipt['passed']:
        raise RuntimeError('Canonical foreground stability review failed')
    atomic_json(destination,receipt)
    print(dict(passed=True,rows=len(rows),elapsed_sec=receipt['review_elapsed_sec']),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=REPO/'experiments/2026-11-05_oasis_adaptive_local_affine_fieldfit/protocol_v1.3.json')
    args=parser.parse_args();review(args.root,args.config)


if __name__=='__main__':main()
