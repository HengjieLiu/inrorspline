#!/usr/bin/env python
"""Freeze an explicit full D0/SVF matrix after input and native resource gates."""
from __future__ import annotations
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys

from runtime import CAPS,FAMILIES,HERE,REPO,atomic_json,read_json,sha,now,code_hashes,assert_external
from controller import gpu_snapshot


def job_id(target,mode,family,cap,initialization,kind='main'):
    return f'{target}__{mode}__{family}__{cap}__{initialization}__{kind}'


def build_jobs(targets):
    jobs=[];known={}
    def add(target,mode,family,budgets,initialization,kind='main',geometry=None):
        token='path' if len(budgets)>1 else str(budgets[0])
        ident=job_id(target['id'],mode,family,token,initialization,kind)
        j={'id':ident,'target_id':target['id'],'mode':mode,'family':family,'budgets':list(budgets),
           'initialization':initialization,'kind':kind,'profile_key':mode+'__'+family,'dependencies':[]}
        if geometry:
            parent,cap=geometry
            path=f'endpoints/{parent}/cap{cap}.json'
            j['dependencies']=[{'job_id':parent,'path':path}];j['geometry_from']=path
        jobs.append(j);known[ident]=j;return ident
    for target in sorted(targets,key=lambda t:(t['pair_rank'],t['source_method'])):
        add(target,'d0','global_affine',[12],'zero','anchor')
        for family in FAMILIES:
            groups=[CAPS] if family.startswith('tree_') else [[c] for c in (1536,768,3072)]
            for budgets in groups:
                add(target,'d0',family,budgets,'zero')
                for init in ('zero','global_affine_ls'):add(target,'svf',family,budgets,init)
    for target in targets:
        if target['pair_rank'] not in (0,1):continue
        for family in ('bspline','tree_affine_uniform','tree_affine_adaptive'):
            token='path' if family.startswith('tree_') else '1536'
            d0=job_id(target['id'],'d0',family,token,'zero')
            add(target,'d1',family,[1536],'ls','bounded_followup',(d0,1536))
            add(target,'d0',family,[1536],'zero','solver_diagnostic',(d0,1536))
        family='tree_affine_adaptive'
        d0=job_id(target['id'],'d0',family,'path','zero')
        v=job_id(target['id'],'svf',family,'path','zero')
        # Existing D0, D1 on D0 tree and V on V tree are reused; these are the
        # three missing cells of the bidirectional topology replay per target.
        for init in ('zero','global_affine_ls'):
            add(target,'svf',family,[1536],init,'replay_d0_tree',(d0,1536))
        add(target,'d0',family,[1536],'zero','replay_v_tree',(v,1536))
        add(target,'d1',family,[1536],'ls','replay_v_tree',(v,1536))
    assert len({j['id'] for j in jobs})==len(jobs)
    assert all(d['job_id'] in known for j in jobs for d in j['dependencies'])
    return jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,default=REPO/'experiments/2026-11-05_oasis_adaptive_local_affine_fieldfit/protocol_v1.3.json')
    p.add_argument('--start',action='store_true');a=p.parse_args();root=assert_external(a.root)
    if (root/'launch_manifest.json').exists():raise FileExistsError('Launch already frozen; use controller resume')
    if (root/'SUPERSEDED.json').exists():raise RuntimeError('Cannot launch a superseded execution')
    accepted=read_json(root/'INPUTS_ACCEPTED.json');inputs=read_json(root/'input_manifest.json')
    if not accepted.get('passed',accepted.get('complete',False)):
        raise RuntimeError('Input acceptance missing')
    if accepted['input_manifest_sha256']!=sha(root/'input_manifest.json'):
        raise RuntimeError('Input manifest no longer matches its acceptance')
    if accepted['rank000_metric_replay_sha256']!=sha(root/'rank000_metric_replay.json'):
        raise RuntimeError('Source metric replay no longer matches acceptance')
    if accepted['data_source_sha256']!=sha(HERE/'data.py'):
        raise RuntimeError('Input adapter changed after accepted preparation')
    profile=read_json(root/'preflight'/'summary.json')
    if not profile.get('passed',False):raise RuntimeError('Native preflight incomplete or failed')
    if profile['input_manifest_sha256']!=sha(root/'input_manifest.json'):
        raise RuntimeError('Native preflight belongs to another input manifest')
    for path,digest in profile['code_hashes'].items():
        full=Path(path) if Path(path).is_absolute() else HERE/path
        if sha(full)!=digest:raise RuntimeError('Native preflight source changed: '+path)
    smoke=read_json(root/'WORKER_SMOKE.json')
    if not smoke.get('passed'):raise RuntimeError('Serialized worker smoke failed/incomplete')
    for name in ('worker.py','metrics.py','diagnostics.py','basis.py','numerics.py'):
        key=str((HERE/name).relative_to(REPO))
        if smoke['source_hashes'].get(key)!=sha(HERE/name):
            raise RuntimeError('Worker smoke source changed: '+name)
    stability=read_json(root/'STABILITY_PASSED.json')
    if stability.get('passed') is not True:
        raise RuntimeError('Multi-step solver stability pilot failed/incomplete')
    if stability['input_manifest_sha256']!=sha(root/'input_manifest.json'):
        raise RuntimeError('Stability pilot belongs to another input manifest')
    if stability.get('config_sha256')!=sha(a.config):
        raise RuntimeError('Fit protocol changed after stability pilot')
    for name in ('numerics.py','basis.py','data.py','worker.py','runtime.py','initialization.py','stability_pilot.py','stability_review.py'):
        key=str((HERE/name).relative_to(REPO))
        if stability['source_hashes'].get(key)!=sha(HERE/name):
            raise RuntimeError('Stability pilot source changed: '+name)
    cpu=read_json(root/'validation'/'cpu_tests.json')
    if not cpu.get('passed') or cpu.get('exit_code')!=0:
        raise RuntimeError('CPU validation suite failed/incomplete')
    if cpu['source_hashes']!=code_hashes():
        raise RuntimeError('Implementation changed after CPU validation')
    jobs=build_jobs(inputs['targets']);config=read_json(a.config)
    profiles=profile['memory_profiles_mib']
    profiles.setdefault('d0__global_affine',max(v for k,v in profiles.items() if k.startswith('d0__')))
    missing={j['profile_key'] for j in jobs}-profiles.keys()
    if missing:raise RuntimeError('Missing memory profiles: '+str(missing))
    manifest={'version':config['version'],'created_at':now(),'gpu':0,'gpu0_uuid':gpu_snapshot()['uuid'],
              'input_manifest_sha256':sha(root/'input_manifest.json'),
              'input_acceptance_sha256':sha(root/'INPUTS_ACCEPTED.json'),
              'preflight_sha256':sha(root/'preflight'/'summary.json'),
              'cpu_validation_sha256':sha(root/'validation'/'cpu_tests.json'),
              'worker_smoke_sha256':sha(root/'WORKER_SMOKE.json'),
              'stability_pilot_sha256':sha(root/'STABILITY_PASSED.json'),
              'config_sha256':sha(a.config),'fit_config':config,'code_hashes':code_hashes(),
              'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
              'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)),
              'jobs':jobs,'memory_profiles_mib':profiles,
              'queue_policy':{'max_jobs':8,'gpu':0,'cpu_threads':3,'poll_sec':15,'headroom_mib':2048,
                              'min_disk_gib':300,'min_host_gib':24,'stall_timeout_sec':3600,
                              'job_timeout_sec':43200,'kill_grace_sec':60,'admission_timeout_sec':3600,
                              'automatic_retries':0,'failures':'record_preserve_continue'},
              'coverage':{'d0_main_endpoints':840,'svf_main_configurations':840,
                          'svf_initialization_trials':1680,'anchors':40,'d1_followup':24,
                          'new_replay_configurations':24,'new_replay_trials':32,
                          'solver_diagnostics':24,'worker_jobs':len(jobs)}}
    atomic_json(root/'launch_manifest.json',manifest)
    atomic_json(root/'PREFLIGHT_PASSED.json',{'passed':True,'at':now(),
                'launch_manifest_sha256':sha(root/'launch_manifest.json'),
                'input_gate_sha256':sha(root/'INPUTS_ACCEPTED.json'),'native_gate_sha256':sha(root/'preflight'/'summary.json')})
    # A portable source snapshot preserves the exact uncommitted implementation.
    import tarfile
    with tarfile.open(root/'executed_sources.tar.gz','w:gz') as archive:
        for path in HERE.glob('*.py'):archive.add(path,arcname=str(path.relative_to(REPO)))
        archive.add(a.config,arcname=str(a.config.relative_to(REPO)))
    if a.start:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1')
        with (root/'controller.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen([sys.executable,'-B',str(HERE/'controller.py'),'--root',str(root)],
                                  stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        print({'controller_pid':proc.pid,'jobs':len(jobs),'root':str(root)},flush=True)
    else:print({'frozen':True,'jobs':len(jobs),'root':str(root)},flush=True)


if __name__=='__main__':main()
