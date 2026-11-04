#!/usr/bin/env python
"""P74: preserve278 accepted fits; continue only two failed cps1 stages."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import run_oasis_bspline_svf_sweep_recovery as r
import svf_transition_validation_v2 as v2
from queue_oasis_svf_continue_on_error import gpu_snapshot

FOLDER = 'continuation_v2'
MANIFEST = 'launch_manifest_r1.json'
FIXTURES = 'fixtures_r1'
RANKS = (2, 5)
OPTIMIZE = r.s.optimize
COPY_NAMES = ('best.pt', 'last.pt', 'history.csv', 'history.jsonl', 'summary.json')


def spec_for(rank):
    if rank not in RANKS:
        raise ValueError('Only the two approved ranks may be continued')
    return r.s.Spec(rank, '8421', 'displacement', .01)


def source_hashes():
    paths = [Path(__file__), Path(v2.__file__),
             r.c.REPO_ROOT/'tests/test_oasis_svf_cps1_continuation.py']
    return {str(p.relative_to(r.c.REPO_ROOT)): r.sha(p) for p in paths}


def assert_manifest(manifest):
    r.assert_manifest(manifest)
    info = manifest['continuation_v2']
    if info['source_hashes'] != source_hashes() or info['policy'] != v2.POLICY:
        raise RuntimeError('Continuation source/policy drift')
    if set(info['targets']) != {'2', '5'}:
        raise RuntimeError('Unexpected continuation targets')


def hash_files(paths):
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(lambda p: (str(p), r.sha(p)), paths))


def verify_inventory(folder, manifest):
    path = folder/manifest['continuation_v2'].get('inventory_file','preserved_files.json')
    if r.sha(path) != manifest['continuation_v2']['inventory_sha256']:
        raise RuntimeError('Preservation inventory changed')
    old = r.c.read_json(path)
    actual = hash_files([Path(p) for p in old])
    if actual != old:
        raise RuntimeError(f'Preexisting artifact changed: {[p for p in old if old[p] != actual[p]][:10]}')


def prepare(artifact):
    root = artifact/'runs'/r.s.VARIANT
    folder = root/FOLDER
    folder.mkdir(exist_ok=True)
    path = folder/MANIFEST
    if path.exists():
        manifest = r.c.read_json(path); assert_manifest(manifest)
        return manifest
    initial = folder/'launch_manifest.json'
    if initial.exists():
        # Preserve the pre-optimization adapter-smoke failure and its frozen manifest.
        previous = r.c.read_json(initial)
        r.assert_manifest(previous)
        old_inventory = folder/'preserved_files.json'
        if r.sha(old_inventory) != previous['continuation_v2']['inventory_sha256']:
            raise RuntimeError('Original preservation inventory changed')
        inventory = r.c.read_json(old_inventory)
        inventory.update(hash_files([p for p in folder.rglob('*') if p.is_file()]))
        inventory_file = 'preserved_files_r1.json'
        r.atomic_json(folder/inventory_file,inventory)
        info = {**previous['continuation_v2'], 'source_hashes':source_hashes(),
                'inventory_file':inventory_file,'inventory_sha256':r.sha(folder/inventory_file),
                'preserved_file_count':len(inventory),'created_at':r.c.now_text(),
                'adapter_revision':1,'supersedes_manifest_sha256':r.sha(initial),
                'amendment':'Expose legacy POLICY metadata alongside v2 transition dispatcher; no optimizer ran in failed fixtures'}
        manifest = {**previous,'continuation_v2':info}
        r.atomic_json(path,manifest)
        print(json.dumps({'prepared':True,'preserved_files':len(inventory),'adapter_revision':1}),flush=True)
        return manifest
    parent_path = root/'continue_on_error_v1/launch_manifest.json'
    parent = r.c.read_json(parent_path)
    r.assert_manifest(parent)
    state = r.c.read_json(root/'continue_on_error_v1/state.json')
    if Counter(j['status'] for j in state['jobs']) != {'complete': 278, 'failed_deferred': 2}:
        raise RuntimeError('Require drained278+2 queue')
    targets, accepted = {}, {}
    for job in state['jobs']:
        spec = r.s.Spec(**job['spec']); out = Path(job['out'])
        if job['status'] == 'complete':
            r.validate(out, parent, spec)
            accepted[spec.name()] = str(out)
        else:
            if spec != spec_for(spec.case_rank):
                raise RuntimeError('Failed configuration differs from approved recovery')
            old_manifest = r.c.read_json(out/'manifest.json')
            failure = r.c.read_json(out/'FAILED_worker.json')
            elapsed = (datetime.strptime(failure['created_at'], '%Y-%m-%d %H:%M:%S %z') -
                       datetime.strptime(old_manifest['started_at'], '%Y-%m-%d %H:%M:%S %z')).total_seconds()
            if elapsed <= 0 or (out/'stage3_cps1').exists():
                raise RuntimeError('Expected pre-cps1 failure with positive elapsed time')
            targets[str(spec.case_rank)] = {'spec': asdict(spec), 'prefix_dir': str(out),
                'failure_sha256': r.sha(out/'FAILED_worker.json'), 'manifest_sha256': r.sha(out/'manifest.json'),
                'prefix_gpu': old_manifest['gpu'], 'original_failed_worker_sec_estimate': elapsed,
                'prefix_stage_hashes': {str(i): r.sha(out/f'stage{i}_cps{cps}'/'validated.json')
                                        for i, cps in enumerate((8,4,2))}}
    if (root/'progress.json').exists():
        shutil.copy2(root/'progress.json', folder/'progress_before_resume.json')
    print('Hashing preexisting runs, failures and reports for preservation', flush=True)
    files = [p for candidate in root.glob('rank*') for p in candidate.rglob('*') if p.is_file()]
    for sub in ('recovery_v1', 'parallel8_v1', 'continue_on_error_v1'):
        files += [p for p in (root/sub).rglob('*') if p.is_file()]
    reports = artifact/'repo_experiments'/r.s.VARIANT
    files += [p for p in reports.rglob('*') if p.is_file() and not p.is_relative_to(reports/FOLDER)]
    inventory = hash_files(sorted(set(files)))
    r.atomic_json(folder/'preserved_files.json', inventory)
    manifest = {**parent, 'continuation_v2': {'policy': v2.POLICY, 'source_hashes': source_hashes(),
        'targets': targets, 'accepted_paths': accepted, 'created_at': r.c.now_text(),
        'parent_manifest_sha256': r.sha(parent_path), 'inventory_sha256': r.sha(folder/'preserved_files.json'),
        'preserved_file_count': len(inventory), 'source_revision': subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=r.c.REPO_ROOT,text=True).strip(),
        'optimization_changed': False, 'resume_stage': 3, 'initial_gpu_state': gpu_snapshot()}}
    r.atomic_json(path, manifest)
    print(json.dumps({'prepared': True, 'preserved_files': len(inventory), 'targets': list(targets)}),flush=True)
    return manifest


def check_prefix_block(target, stage, manifest):
    prefix = Path(target['prefix_dir'])
    if r.sha(prefix/'manifest.json') != target['manifest_sha256'] or r.sha(prefix/'FAILED_worker.json') != target['failure_sha256']:
        raise RuntimeError('Original attempt metadata changed')
    old_manifest = r.c.read_json(prefix/'manifest.json')
    spec = spec_for(old_manifest['case_rank'])
    if old_manifest['spec'] != asdict(spec) or old_manifest['source_hashes'] != manifest['source_hashes']:
        raise RuntimeError('Prefix provenance mismatch')
    for k in ('fixed_id','moving_id','input_hashes'):
        if old_manifest[k] != manifest['cases'][str(spec.case_rank)][k]:
            raise RuntimeError('Prefix case/input mismatch')
    folder = prefix/f'stage{stage.index}_cps{stage.cps}'
    if r.sha(folder/'validated.json') != target['prefix_stage_hashes'][str(stage.index)]:
        raise RuntimeError('Prefix stage record changed')
    block = r.c.read_json(folder/'validated.json')
    for name, key in (('best.pt','model_sha256'),('last.pt','last_model_sha256'),('history.csv','history_sha256')):
        if r.sha(folder/name) != block[key]:
            raise RuntimeError('Prefix checkpoint/history changed')
    if r.sha(block['dvf_path']) != block['dvf_sha256']:
        raise RuntimeError('Prefix field changed')
    if block['optimizer']['stage'] != asdict(stage) or block['alpha'] != .01:
        raise RuntimeError('Prefix schedule/weight mismatch')
    return folder, block


def replay_prefix(initial, objective, out, target, manifest):
    """No optimizer constructed for a completed prefix; copy bytes, verify selected state."""
    folder, block = check_prefix_block(target, objective.stage, manifest)
    state = torch.load(folder/'best.pt',map_location=initial.device,weights_only=False)
    cp, info = state['velocity_controls_norm_xyz'], state['summary']
    if info != block['optimizer'] or cp.shape != initial.shape:
        raise RuntimeError('Prefix selected control metadata mismatch')
    with torch.no_grad():
        values = [float(x) for x in objective(cp)]
        initial_values = [float(x) for x in objective(initial)]
        if any(not r.close(v, info[k]) for v,k in zip(values,('best_loss','best_data_loss','best_reg_loss'))):
            raise RuntimeError('Prefix selected loss replay mismatch')
        if any(not r.close(a,b) for a,b in zip(initial_values, block['initial_loss_data_reg'])):
            raise RuntimeError('Prefix initialization replay mismatch')
        field = objective.field(cp,native=True)
        if not np.array_equal(field.cpu().numpy(),np.load(block['dvf_path'])):
            raise RuntimeError('Prefix selected field does not replay exactly')
    out.mkdir(parents=True,exist_ok=False)
    for name in COPY_NAMES:
        shutil.copy2(folder/name,out/name)
        if r.sha(folder/name) != r.sha(out/name):
            raise RuntimeError('Prefix copy not byte-identical')
    return cp.detach(), info


@contextmanager
def worker_adapter(target, manifest, fixture=False):
    """Process-local delegation only; old worker/auditor Python files are untouched."""
    original_transition, original_optimize = r.transition, r.s.optimize
    checks = []
    def verify(*args, **kwargs):
        result = v2.verify_transition(*args, **kwargs)
        checks.append(result)
        return result
    def optimize(initial, objective, out):
        if objective.stage.index < 3:
            return replay_prefix(initial, objective, out, target, manifest)
        if not fixture:
            return OPTIMIZE(initial, objective, out)
        objective.stage = replace(objective.stage,max_epochs=1,min_epochs=1)
        best, info = OPTIMIZE(initial, objective, out)
        with torch.no_grad():
            saved = torch.load(out/'best.pt',map_location=initial.device,weights_only=False)['velocity_controls_norm_xyz']
            if not torch.equal(best,saved) or info['actual_epochs'] != 1:
                raise RuntimeError('One-update checkpoint smoke failed')
            u = objective.field(saved,native=True)
            if list(u.shape) != [160,224,192,3] or not torch.isfinite(u).all():
                raise RuntimeError('Smoke field invalid')
            path = out/'smoke_disp.npy'
            r.save_field(path,u)
            if not np.array_equal(u.cpu().numpy(),np.load(path)):
                raise RuntimeError('Smoke DVF serialization failed')
            batch = objective._continuation_batch
            metrics = r.metric_row(u,batch,out/'canonical_metrics')
        r.atomic_json(out.parent/'FIXTURE_PASSED.json',{'passed':True,'transition_checks':checks,
            'one_adam_update':True,'metrics':metrics,'source_hashes':source_hashes(),
            'policy':v2.POLICY,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'prefix':target,'created_at':r.c.now_text()})
        raise FixtureFinished()
    original_objective = r.s.Objective
    # Fixture needs the canonical batch only; capture it via the unchanged setup routine.
    original_setup = r.setup
    holder = {}
    def setup(*args, **kwargs):
        result = original_setup(*args, **kwargs)
        holder['batch'] = result[1]
        return result
    def objective(*args, **kwargs):
        obj = original_objective(*args, **kwargs)
        if fixture:
            obj._continuation_batch = holder['batch']
        return obj
    r.transition = SimpleNamespace(POLICY=original_transition.POLICY, verify_transition=verify)
    r.s.optimize = optimize
    if fixture:
        r.setup, r.s.Objective = setup, objective
    try:
        yield checks
    finally:
        r.transition, r.s.optimize = original_transition, original_optimize
        r.setup, r.s.Objective = original_setup, original_objective


class FixtureFinished(Exception):
    pass


def worker_args(args):
    return SimpleNamespace(**vars(args), schedule='8421', regularize='displacement', alpha=.01,
                           smoke=False, integration_steps=7)


def run_worker(args, fixture=False):
    manifest = r.c.read_json(args.manifest); assert_manifest(manifest)
    target = manifest['continuation_v2']['targets'][str(args.case_rank)]
    spec_for(args.case_rank)
    try:
        with worker_adapter(target,manifest,fixture):
            r.worker(worker_args(args))
    except FixtureFinished:
        if not fixture: raise
        return
    if fixture:
        raise RuntimeError('Fixture did not stop after one update')
    row = r.c.read_json(args.out_dir/'summary.json')
    for stage, block in zip(r.s.stages(spec_for(args.case_rank))[:3],row['blocks'][:3]):
        _, old = check_prefix_block(target,stage,manifest)
        expected = {**old,'model_path':block['model_path'],'dvf_path':block['dvf_path']}
        if block != expected:
            raise RuntimeError('Resumed prefix records differ beyond copied paths')
    row['continuation_v2'] = {'policy':v2.POLICY,'source_hashes':source_hashes(), 'prefix':target,
        'new_optimizer_stages':[1], 'new_adam_updates':row['blocks'][-1]['optimizer']['actual_epochs'],
        'prefix_adam_updates_reused':sum(b['optimizer']['actual_epochs'] for b in row['blocks'][:3]),
        'continuation_worker_sec':row['worker_sec'],
        'worker_time_accounting':'original failed-worker elapsed (1-second timestamp precision) plus continuation worker elapsed; no downtime',
        'source_revision':manifest['continuation_v2']['source_revision']}
    row['worker_sec'] += target['original_failed_worker_sec_estimate']
    row['peak_cuda_gib'] = max(row['peak_cuda_gib'],*(b['optimizer']['peak_cuda_gib'] for b in row['blocks'][:3]))
    row['peak_reserved_gib'] = max(row['peak_reserved_gib'],*(b['optimizer']['peak_reserved_gib'] for b in row['blocks'][:3]))
    assert_manifest(manifest)
    r.atomic_json(args.out_dir/'summary.json',row)
    r.atomic_json(args.out_dir/'CONTINUATION_DONE.json',{'summary_sha256':r.sha(args.out_dir/'summary.json'),
        'source_hashes':source_hashes(),'completed_at':r.c.now_text()})


def audit(args):
    manifest = r.c.read_json(args.manifest); assert_manifest(manifest)
    row = r.c.read_json(args.out_dir/'summary.json')
    target = manifest['continuation_v2']['targets'][str(row['case_rank'])]
    if row['continuation_v2']['prefix'] != target:
        raise RuntimeError('Wrong continuation source')
    done = r.c.read_json(args.out_dir/'CONTINUATION_DONE.json')
    if done['summary_sha256'] != r.sha(args.out_dir/'summary.json'):
        raise RuntimeError('Incomplete continuation summary')
    # Independent old audit replays every copied prefix checkpoint, history and field.
    with worker_adapter(target,manifest):
        r.audit(args)
    assert_manifest(manifest)
    r.atomic_json(args.out_dir/'CONTINUATION_AUDIT.json',{'passed':True,'source_hashes':source_hashes(),
        'policy':v2.POLICY,'audit_sha256':r.sha(args.out_dir/'independent_audit.json'),
        'summary_sha256':r.sha(args.out_dir/'summary.json'),'completed_at':r.c.now_text()})


def validate(out, manifest, spec):
    row = r.validate(out,manifest,spec)
    check = r.c.read_json(out/'CONTINUATION_AUDIT.json')
    if (not check['passed'] or check['source_hashes'] != source_hashes() or check['policy'] != v2.POLICY
        or check['audit_sha256'] != r.sha(out/'independent_audit.json')
        or check['summary_sha256'] != r.sha(out/'summary.json')
        or row['continuation_v2']['source_hashes'] != source_hashes()):
        raise RuntimeError('Continuation audit/provenance mismatch')
    return row


def report(artifact):
    from report_oasis_bspline_svf_sweep import generate
    root = artifact/'runs'/r.s.VARIANT; folder=root/FOLDER
    manifest = r.c.read_json(folder/MANIFEST); assert_manifest(manifest)
    rows=[]
    for spec in r.s.matrix():
        if spec.name() in manifest['continuation_v2']['accepted_paths']:
            rows.append(r.validate(Path(manifest['continuation_v2']['accepted_paths'][spec.name()]),manifest,spec))
        else:
            paths = [p for p in (root/spec.name()).glob('attempt*') if (p/'CONTINUATION_AUDIT.json').exists()]
            if len(paths)!=1: raise RuntimeError('Require exactly one audited continuation')
            rows.append(validate(paths[0],manifest,spec))
    verify_inventory(folder,manifest)
    final_out = artifact/'repo_experiments'/r.s.VARIANT/FOLDER
    if final_out.exists(): raise RuntimeError('Published continuation report is immutable')
    final_out.parent.mkdir(parents=True,exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='continuation_v2_report_',dir=final_out.parent))
    generate(rows,manifest,out)
    # The original report generator is immutable. Correct its historical concurrency wording
    # only in this new release, then rehash the generated acceptance record.
    path=out/'report.md'; content=path.read_text()
    content=content.replace('One new worker/audit per GPU0–3, with other workloads untouched.',
        'Mixed concurrency: initially one, then two workers/auditors per GPU; the final two cps1 continuations used separate GPU slots. Other workloads remained untouched.')
    content=content.replace('## Timing and memory interpretation',
        '## Checkpoint continuation\n\n278 accepted registrations are unchanged. Ranks2/5,8421/displacement/λ0.01, '
        'completed only cps1 from their own verified cps2 checkpoints under validation policyv2. '
        'FP32 optimization and seven-step integration are unchanged. For these two rows, worker time is the '
        'original failed-attempt elapsed time (approximately1-second precision) plus the continuation worker; '
        'idle downtime is excluded. Optimizer time sums original prefix and new cps1 optimization; '
        'no prefix update was repeated. All four stages and7/8/10 diagnostics were independently audited.\n\n## Timing and memory interpretation')
    path.write_text(content)
    accepted=r.c.read_json(out/'ACCEPTED.json')
    accepted['output_sha256']['report.md']=r.sha(path)
    accepted['continuation_count']=2
    r.atomic_json(out/'ACCEPTED.json',accepted)
    for name,digest in accepted['output_sha256'].items():
        if r.sha(out/name)!=digest: raise RuntimeError('Final report hash mismatch')
    out.rename(final_out)
    out = final_out
    r.atomic_json(folder/'COMPLETE.json',{'count':280,'continued':2,'preserved':278,
        'preserved_file_count':manifest['continuation_v2']['preserved_file_count'],'report':str(out),'completed_at':r.c.now_text()})
    r.atomic_json(root/'COMPLETE.json',r.c.read_json(folder/'COMPLETE.json'))
    r.atomic_json(root/'progress.json',{'phase':'accepted','completed':280,'total':280,
        'failed_deferred':0,'remaining':0,'report':str(out),'updated_at':r.c.now_text()})


def phase(args, manifest, action):
    root=args.artifact_root/'runs'/r.s.VARIANT; folder=root/FOLDER
    jobs=[{'rank':rank,'status':'pending','tries':0} for rank in RANKS]
    active={}
    for job in jobs:
        candidates = ([folder/FIXTURES/f'rank{job["rank"]}'/f'attempt{n}' for n in (0,1)] if action=='fixture'
                      else [root/spec_for(job['rank']).name()/f'attempt{n}' for n in (1,2)])
        marker='FIXTURE_PASSED.json' if action=='fixture' else 'CONTINUATION_AUDIT.json'
        completed=[p for p in candidates if (p/marker).exists()]
        if completed:
            if len(completed)!=1: raise RuntimeError('Duplicate continuation result')
            if action=='worker': validate(completed[0],manifest,spec_for(job['rank']))
            else:
                check=r.c.read_json(completed[0]/marker)
                if check['source_hashes']!=source_hashes() or not check['passed']: raise RuntimeError('Invalid fixture')
            job.update(status='complete',out=str(completed[0]));continue
        existing=[p for p in candidates if p.exists()]
        if existing:
            # A worker that finished successfully but lacks its audit can be audited without rerunning.
            done=[p for p in existing if (p/'CONTINUATION_DONE.json').exists() and not list(p.glob('FAILED*'))]
            if action=='worker' and len(done)==1:
                job.update(status='audit_pending',out=str(done[0]))
            else:
                raise RuntimeError(f'Preserved incomplete continuation needs audit: {existing}')
    while any(j['status'] not in ('complete','failed') for j in jobs) or active:
        for pid,item in list(active.items()):
            code=item['proc'].poll()
            if code is None: continue
            item['stream'].close();job=item['job'];out=Path(job['out'])
            try:
                if code: raise RuntimeError(f'{item["action"]} exited{code}; see {item["log"]}')
                if item['action']=='worker':
                    if not (out/'CONTINUATION_DONE.json').exists(): raise RuntimeError('Missing continuation marker')
                    job['status']='audit_pending'
                elif item['action']=='audit':
                    validate(out,manifest,spec_for(job['rank']));job['status']='complete'
                else:
                    check=r.c.read_json(out/'FIXTURE_PASSED.json')
                    if not check['passed'] or check['source_hashes']!=source_hashes(): raise RuntimeError('Fixture failed')
                    job['status']='complete'
            except Exception:
                tail=Path(item['log']).read_text(errors='replace')[-16000:]
                oom='cuda' in tail.lower() and 'out of memory' in tail.lower()
                r.atomic_json(folder/f'failure_{time.time_ns()}.json',{'rank':job['rank'],'action':item['action'],
                    'traceback':traceback.format_exc(),'log':item['log'],'out':str(out),'oom':oom})
                if oom and job['tries']==0 and item['action']!='audit':
                    job.update(status='pending',tries=1)
                else: job['status']='failed'
            del active[pid]
        assert_manifest(manifest)
        snapshot=gpu_snapshot()
        occupied={item['job']['gpu'] for item in active.values()}
        required=8192
        for check in (folder/FIXTURES).glob('rank*/attempt*/FIXTURE_PASSED.json'):
            required=max(required,int((r.c.read_json(check)['peak_reserved_gib']+2)*1024))
        for gpu,row in sorted(snapshot['rows'].items(),key=lambda x:x[1]['memory_free_mib'],reverse=True):
            if gpu in occupied or row['memory_free_mib']<required: continue
            job=next((j for j in jobs if j['status'] in ('pending','audit_pending')),None)
            if job is None: break
            command='audit' if job['status']=='audit_pending' else action
            if command=='audit': out=Path(job['out'])
            elif action=='fixture': out=folder/FIXTURES/f'rank{job["rank"]}'/f'attempt{job["tries"]}'
            else: out=root/spec_for(job['rank']).name()/f'attempt{1+job["tries"]}'
            if command!='audit' and out.exists(): raise RuntimeError(f'Refuse overwrite:{out}')
            log=folder/f'{command}_rank{job["rank"]}_{time.time_ns()}.log';stream=log.open('x')
            cmd=[sys.executable,'-u','-B',str(Path(__file__).resolve()),command,'--manifest',str(folder/MANIFEST),
                 '--out-dir',str(out),'--case-rank',str(job['rank']),'--gpu',gpu]
            proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,
                'CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
            job.update(status='auditing' if command=='audit' else 'running',gpu=gpu,out=str(out),pid=proc.pid)
            active[proc.pid]={'proc':proc,'stream':stream,'job':job,'action':command,'log':str(log)}
            occupied.add(gpu)
            r.atomic_json(folder/f'launch_{time.time_ns()}.json',{'job':job,'command':cmd,'gpu_state':snapshot,
                'required_free_mib':required,'created_at':r.c.now_text()})
            print(json.dumps({'launched':command,'rank':job['rank'],'gpu':gpu,'pid':proc.pid}),flush=True)
        status={'phase':f'continuation_v2_{action}','completed':278+sum(j['status']=='complete' for j in jobs) if action=='worker' else 278,
                'total':280,'jobs':jobs,'gpu_state':snapshot,'updated_at':r.c.now_text()}
        r.atomic_json(folder/f'{action}_progress_r1.json',status);r.atomic_json(root/'progress.json',status)
        if active:time.sleep(5)
        elif any(j['status'] in ('pending','audit_pending') for j in jobs):time.sleep(10)
    if any(j['status']=='failed' for j in jobs): raise RuntimeError('Continuation failure preserved; successful peer completed')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('prepare','launch','report'):
        q=sub.add_parser(command);q.add_argument('--artifact-root',type=Path,default=Path('/database/inrorspline'))
    for command in ('fixture','worker','audit'):
        q=sub.add_parser(command)
        q.add_argument('--manifest',type=Path,required=True);q.add_argument('--out-dir',type=Path,required=True)
        q.add_argument('--case-rank',type=int,choices=RANKS,required=True);q.add_argument('--gpu',choices=('0','1','2','3'),required=True)
    args=p.parse_args(argv)
    out=args.artifact_root if hasattr(args,'artifact_root') else args.out_dir
    if out.resolve().is_relative_to(r.c.REPO_ROOT):p.error('External artifacts only')
    return args


def main(args):
    if (os.getuid(),os.getgid())!=(1009,1020):raise RuntimeError('Require UID:GID1009:1020')
    if args.command in ('fixture','worker'):run_worker(args,fixture=args.command=='fixture')
    elif args.command=='audit':audit(args)
    else:
        root=args.artifact_root/'runs'/r.s.VARIANT
        with (root/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.command=='report':report(args.artifact_root);return
            manifest=prepare(args.artifact_root)
            if args.command=='prepare':return
            phase(args,manifest,'fixture');phase(args,manifest,'worker');report(args.artifact_root)


if __name__=='__main__':
    args=parse_args()
    try:main(args)
    except BaseException:
        folder=args.out_dir if hasattr(args,'out_dir') else args.artifact_root/'runs'/r.s.VARIANT/FOLDER
        folder.mkdir(parents=True,exist_ok=True)
        r.atomic_json(folder/f'FAILED_{args.command}_{time.time_ns()}.json',{'traceback':traceback.format_exc(),'created_at':r.c.now_text()})
        raise
