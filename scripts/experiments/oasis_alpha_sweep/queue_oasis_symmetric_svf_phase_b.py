#!/usr/bin/env python
"""Approved 15 -> 9 -> 4 Phase-B queue; immutable plans and deferred OOM retry."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_oasis_symmetric_svf_phase_b as r


def oom(text):
    value=text.lower()
    return 'cuda out of memory' in value or 'cuda error: out of memory' in value or 'torch.cuda.outofmemoryerror' in value


def plan(root, name, entries, dependencies):
    value={'name':name,'entries':[{'spec':asdict(s),'pinned_gpu':g,'parent_run':parent} for s,g,parent in entries],
           'dependencies':{x['run_name']:r.sha(Path(x['run_dir'])/'summary.json') for x in dependencies}}
    path=root/(name+'_plan.json')
    if path.exists():
        if r.c.read_json(path)!=value:raise RuntimeError('Selection plan drift')
    else:r.atomic_json(path,value)
    return value


def run_group(root,manifest,name,entries,required,previous_count=0):
    jobs=[]
    progress=root/(name+'_progress.json')
    old={j['spec_name']:j for j in r.c.read_json(progress)['jobs']} if progress.exists() else {}
    for spec,pinned,parent in entries:
        prior=old.get(spec.name(),{})
        path=Path(prior.get('path',str(root/spec.name()/'attempt0')))
        status='pending'
        if (path/'independent_audit.json').is_file():
            r.validate(path,manifest,spec);status='complete'
        elif prior.get('status')=='failed':status='failed'
        elif (path/'DONE').is_file():status='audit_pending'
        elif path.exists():status='failed'
        jobs.append({'spec':spec,'spec_name':spec.name(),'path':path,'status':status,
                     'pinned_gpu':pinned,'parent_run':parent,'gpu':prior.get('gpu'),
                     'retry_count':prior.get('retry_count',0),'defer_release':0,
                     'failures':prior.get('failures',[]),'attempt':prior.get('attempt',0)})
    active={};released=0;last_print=0.;started=time.monotonic()
    while any(j['status'] not in ('complete','failed') for j in jobs):
        for gpu,item in list(active.items()):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();job=item['job'];released+=1
            error=None
            if code:error=f'{item["action"]} exited {code}'
            elif item['action']=='worker':job['status']='audit_pending'
            else:
                try:r.validate(job['path'],manifest,job['spec']);job['status']='complete'
                except Exception as exc:error=repr(exc)
            if error:
                text=item['log_path'].read_text(errors='replace')[-100000:]
                record={'error':error,'action':item['action'],'log':str(item['log_path']),
                        'attempt':job['attempt'],'gpu':gpu,'time':r.c.now_text(),'cuda_oom':oom(text)}
                job['failures'].append(record)
                if record['cuda_oom'] and job['retry_count']==0:
                    job['retry_count']=1;job['defer_release']=released+1
                    if item['action']=='worker':
                        job['attempt']+=1;job['path']=root/job['spec_name']/f'attempt{job["attempt"]}'
                        job['status']='pending'
                    else:job['status']='audit_pending'
                else:job['status']='failed'
                print(json.dumps({'failure_preserved':record,'next_status':job['status']}),flush=True)
            del active[gpu]
        state=r.p.legacy.gpu_snapshot()
        available={str(x['physical_gpu']):x for x in state.get('rows',[])
                   if str(x['physical_gpu']) in manifest['gpus'] and str(x['physical_gpu']) not in active
                   and x['memory_free_mib']>=required}
        for job in sorted(jobs,key=lambda j:(j['status']!='audit_pending',bool(j['retry_count']))):
            if job['status'] not in ('pending','audit_pending') or not available:continue
            if job['defer_release']>released and active:continue
            eligible=available if job['pinned_gpu'] is None else {g:v for g,v in available.items() if g==job['pinned_gpu']}
            if not eligible:continue
            r.assert_manifest(manifest)
            gpu=max(eligible,key=lambda g:eligible[g]['memory_free_mib'])
            if job['status']=='audit_pending' and job['gpu'] in eligible:gpu=job['gpu']
            action='audit' if job['status']=='audit_pending' else 'worker'
            path=job['path'];path.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-u','-B',str(Path(r.__file__)),action,'--manifest',str(root/'launch_manifest.json'),
                     '--out-dir',str(path),'--gpu',gpu]
            spec=job['spec']
            if action=='worker':
                command+=['--arm',spec.arm,'--alpha',str(spec.alpha),'--beta',str(spec.beta)]
                for flag in ('smoke','swapped','repeat'):
                    if getattr(spec,flag):command+=['--'+flag]
            log_path=path.parent/f'{action}_attempt{job["attempt"]}_{time.time_ns()}.log'
            stream=log_path.open('x')
            env={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':gpu,
                 'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
                 'MPLCONFIGDIR':'/tmp/inrorspline_p78_mpl'}
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
            job.update(status='running',gpu=gpu)
            active[gpu]={'process':process,'job':job,'action':action,'log':stream,'log_path':log_path}
            event={'time':r.c.now_text(),'group':name,'spec':asdict(spec),'action':action,'gpu':gpu,
                   'pid':process.pid,'gpu_before':eligible[gpu],'log':str(log_path),'command':command}
            with (root/'launch_events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True);del available[gpu]
        complete=[j for j in jobs if j['status']=='complete']
        durations=[]
        for j in complete:
            row=r.c.read_json(j['path']/'summary.json');audit=r.c.read_json(j['path']/'independent_audit.json')
            durations.append(row['worker_sec']+audit['audit_sec'])
        total=4 if name=='smoke' else 28
        ncomplete=previous_count+len(complete)
        eta=(sum(durations)/len(durations)*(total-ncomplete)/4) if durations else None
        status={'group':name,'completed':len(complete),'group_total':len(jobs),
                'experiment_completed':ncomplete,'experiment_total':total,'estimated_remaining_sec':eta,
                'elapsed_sec':time.monotonic()-started,'updated_at':r.c.now_text(),'required_free_mib':required,
                'jobs':[{**{k:v for k,v in j.items() if k not in ('spec','path')},'spec':asdict(j['spec']),
                         'path':str(j['path'])} for j in jobs],
                'active':[{'gpu':g,'action':v['action'],'spec_name':v['job']['spec_name'],'pid':v['process'].pid,
                           'worker_progress':r.c.read_json(v['job']['path']/'progress.json')
                           if (v['job']['path']/'progress.json').exists() else {}} for g,v in active.items()]}
        r.atomic_json(progress,status);r.atomic_json(root/'live_progress.json',status)
        if time.monotonic()-last_print>=30:print(json.dumps(status),flush=True);last_print=time.monotonic()
        if any(j['status'] not in ('complete','failed') for j in jobs):time.sleep(5)
    result={'passed':all(j['status']=='complete' for j in jobs),'source_hashes':r.source_hashes(),
            'queue_source_sha256':r.sha(__file__),'completed':sum(j['status']=='complete' for j in jobs),
            'elapsed_sec':time.monotonic()-started,'completed_at':r.c.now_text(),'runs':[],
            'failures':[{'spec_name':j['spec_name'],'status':j['status'],'records':j['failures']} for j in jobs if j['failures'] or j['status']=='failed'],
            'peak_gib':0.}
    for job in jobs:
        if job['status']!='complete':continue
        row=r.validate(job['path'],manifest,job['spec'])
        result['runs'].append({'spec':asdict(job['spec']),'path':str(job['path']),'gpu':row['gpu'],
                               'pinned_gpu':job['pinned_gpu'],'parent_run':job['parent_run']})
        result['peak_gib']=max(result['peak_gib'],row['peak_cuda_gib'],row['peak_reserved_gib'],
                              row['audit']['audit_peak_cuda_gib'],row['audit']['audit_peak_reserved_gib'])
    r.atomic_json(root/(name.upper()+'_FINISHED.json'),result)
    if not result['passed']:raise RuntimeError(f'{name}: incomplete fits preserved; unrelated jobs have finished')
    return result


def main(args):
    root=args.artifact_root/'runs'/r.m.VARIANT/r.m.PHASE
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=r.c.read_json(root/'launch_manifest.json');r.assert_manifest(manifest)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if str(root/'launch_manifest.json').encode() in cmd and any(x.endswith(b'/run_oasis_symmetric_svf_phase_b.py') for x in cmd):
            raise RuntimeError('Existing worker/replay process; do not duplicate it')
    started=time.monotonic()
    if args.phase=='smoke':
        entries=[(s,str(i),None) for i,s in enumerate(r.m.smoke_matrix())]
        plan(root,'smoke',entries,[])
        result=run_group(root,manifest,'smoke',entries,8192.)
        r.atomic_json(root/'SMOKES_PASSED.json',result);return
    gate=r.c.read_json(root/'SMOKES_PASSED.json')
    if not gate['passed'] or gate['source_hashes']!=r.source_hashes():raise RuntimeError('Verified smokes required')
    for item in gate['runs']:r.validate(Path(item['path']),manifest,r.m.Spec(**item['spec']))
    required=max(8192.,(gate['peak_gib']+2)*1024)
    entries=[(s,'0' if s.arm=='base' else None,None) for s in r.m.initial_matrix()]
    plan(root,'initial',entries,[])
    first=run_group(root,manifest,'initial',entries,required)
    rows=[r.validate(Path(x['path']),manifest,r.m.Spec(**x['spec'])) for x in first['runs']]
    entries=[(s,None,None) for s in r.m.combined_matrix(rows)]
    plan(root,'combined',entries,rows)
    second=run_group(root,manifest,'combined',entries,required,15)
    rows += [r.validate(Path(x['path']),manifest,r.m.Spec(**x['spec'])) for x in second['runs']]
    entries=r.m.diagnostic_matrix(rows)
    plan(root,'diagnostics',entries,rows)
    third=run_group(root,manifest,'diagnostics',entries,required,24)
    result={'passed':True,'completed':28,'runs':first['runs']+second['runs']+third['runs'],
            'source_hashes':r.source_hashes(),'queue_source_sha256':r.sha(__file__),
            'elapsed_sec':time.monotonic()-started,'completed_at':r.c.now_text(),
            'failures':first['failures']+second['failures']+third['failures'],
            'peak_gib':max(first['peak_gib'],second['peak_gib'],third['peak_gib'])}
    r.atomic_json(root/'PRODUCTION_FINISHED.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,default=Path('/database/inrorspline'))
    parser.add_argument('--phase',choices=('smoke','production'),required=True)
    main(parser.parse_args())
