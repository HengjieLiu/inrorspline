"""Persistent memory-admitted workers; first-batch report automatically releases rest."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .protocol import FIRST,REST,combinations,sweep_state,candidate_key,dump,read,sha,stamp
from .report import generate

def gpu_memory():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu',
                      '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    return [dict(index=int(v[0]),total_gib=float(v[1])/1024,free_gib=float(v[2])/1024,utilization=float(v[3]))
            for v in (line.split(',') for line in r.stdout.strip().splitlines())]

def outcome(path, returncode):
    p=path/'outcome.json'
    if p.exists():return read(p)
    result=dict(status='failed',score=None,error=f'Worker exited {returncode} without outcome',finished=stamp())
    dump(p,result);return result

class Controller:
    def __init__(self,config):
        self.config=config;self.run=Path(config['run_root']);self.review=Path(config['review_root'])
        self.prepared=Path(config['prepared_root']);self.frozen=Path(config['frozen_root'])
        self.state_path=self.run/'controller_state.json';self.active={};self.handles={}
        self.state=read(self.state_path) if self.state_path.exists() else dict(
            schema='lungct_hu_crop_lambda_v1',stage='first',sweeps=[],profiles={},events=[],started=stamp())
        self.last_report=0

    def event(self,event,**details):
        item=dict(time=stamp(),event=event,**details);self.state['events'].append(item)
        print(item,flush=True)

    def persist(self):
        self.state['updated']=stamp();self.state['active']=[dict(key=k,**{x:v[x] for x in ['gpu','path','smoke','pid']}) for k,v in self.active.items()]
        dump(self.state_path,self.state)

    def validate_source(self):
        manifest=read(self.frozen/'source_manifest.json')
        for name,digest in manifest['source_hashes'].items():
            if sha(self.frozen/name)!=digest:raise RuntimeError(f'Frozen source modified: {name}')
        if sha(self.run/'config.json')!=manifest['config_sha256']:raise RuntimeError('Configuration drift')

    def prepare(self,batch):
        receipt=self.prepared/f'preparation_{batch}.json'
        if receipt.exists() and read(receipt).get('complete'):return
        cmd=[sys.executable,'-B','-m','scripts.experiments.lungct_hu_crop_lambda.prepare',
             '--raw-root',self.config['raw_root'],'--baseline-root',self.config['baseline_root'],
             '--out-root',str(self.prepared),'--batch',batch]
        with (self.run/f'prepare_{batch}.log').open('a') as log:
            r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=self.frozen)
        if r.returncode: self.event('preparation_process_failure',batch=batch,returncode=r.returncode)
        if receipt.exists():
            data=read(receipt);data['complete']=True;dump(receipt,data)

    def add_batch(self,batch):
        self.prepare(batch)
        known={s['id'] for s in self.state['sweeps']}
        for s in combinations(FIRST if batch=='first' else REST):
            if s['id'] in known:continue
            s.update(batch=batch,results={})
            data=self.prepared/f"{s['hu']}_{s['crop']}"/('4DCT' if s['dataset']=='4dct' else 'COPD')/f"case{s['case']:02d}"
            s['data_path']=str(data)
            if not (data/'accepted.json').exists():
                s.update(preparation_failure=True,error='No accepted preparation; inspect preparation log')
            else:
                s['voxels']=__import__('math').prod(read(data/'preprocess_audit.json')['registration_shape_zyx'])
            self.state['sweeps'].append(s)
        self.event('batch_added',batch=batch);self.persist()

    def collect_existing(self):
        # A container restart must not start duplicate workers still alive in it.
        # PID1 death stops container children; external source/receipts stay intact.
        for s in self.state['sweeps']:
            for k in range(-8,9):
                path=self.run/'candidates'/candidate_key(s,k)
                if (path/'outcome.json').exists() and str(k) not in s['results']:
                    r=read(path/'outcome.json');s['results'][str(k)]=dict(r,path=str(path))
                elif path.exists() and str(k) not in s['results']:
                    r=outcome(path,'interrupted');s['results'][str(k)]=dict(r,path=str(path))
        for path in (self.run/'profiles').glob('*/outcome.json'):
            self.state['profiles'][path.parent.name]=read(path)

    def pending(self):
        sweeps=[s for s in self.state['sweeps'] if s['batch']==self.state['stage']]
        center_pending=any(not s.get('preparation_failure') and '0' not in s['results'] for s in sweeps)
        values=[]
        for s in sweeps:
            state=sweep_state(s['results'],s.get('preparation_failure',False))
            for k in state['next']:
                if center_pending and k!=0:continue
                key=candidate_key(s,k)
                if key not in self.active:values.append((s,k))
        return sorted(values,key=lambda v:(abs(v[1]),v[1]<0,-v[0].get('voxels',0),v[0]['id']))

    def profile_id(self,s):return f"{s['dataset']}_{s['case']:02d}_{s['crop']}"

    def launch(self,s,k,gpu,smoke=False):
        profile=self.profile_id(s)
        key='profile/'+profile if smoke else candidate_key(s,k)
        path=self.run/'profiles'/profile if smoke else self.run/'candidates'/key
        path.mkdir(parents=True,exist_ok=False)
        hu='H2' if smoke else s['hu']
        data_root=self.prepared/f"{hu}_{s['crop']}"
        cmd=[sys.executable,'-B','-m','scripts.experiments.lungct_hu_crop_lambda.worker',
             '--data-root',str(data_root),'--dataset',s['dataset'],'--case',str(s['case']),
             '--hu',hu,'--crop',s['crop'],'--k',str(k),'--gpu',str(gpu['index']),'--out',str(path)]
        if smoke:cmd.append('--smoke')
        dump(path/'launch.json',dict(command=cmd,gpu_context=gpu,source_manifest_sha256=sha(self.frozen/'source_manifest.json'),time=stamp()))
        log=(path/'worker.log').open('a');env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu['index']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=self.frozen,env=env)
        self.active[key]=dict(proc=proc,pid=proc.pid,log=log,s=s,k=k,gpu=gpu['index'],path=str(path),smoke=smoke)
        self.event('started',key=key,gpu=gpu['index'],smoke=smoke);self.persist()

    def reap(self):
        for key,item in list(self.active.items()):
            if item['proc'].poll() is None:continue
            r=outcome(Path(item['path']),item['proc'].returncode);item['log'].close()
            if item['smoke']:
                self.state['profiles'][self.profile_id(item['s'])]=r
            else:
                item['s']['results'][str(item['k'])]=dict(r,path=item['path'],gpu=item['gpu'])
            self.event('finished',key=key,status=r['status'],score=r.get('score'),error=r.get('error'))
            del self.active[key];self.persist()

    def skip_geometry(self,s,reason):
        profile=self.profile_id(s)
        for t in self.state['sweeps']:
            if self.profile_id(t)==profile:
                t.update(preparation_failure=True,unavailable_kind='memory',error=reason)
        self.event('geometry_unavailable',profile=profile,reason=reason);self.persist()

    def dispatch(self):
        gpus=[g for g in gpu_memory() if g['index'] in self.config['gpu_pool']]
        if not gpus:raise RuntimeError('No configured GPU is visible')
        busy={v['gpu'] for v in self.active.values()}
        available=sorted([g for g in gpus if g['index'] not in busy],key=lambda g:-g['free_gib'])
        capacity=max(g['total_gib'] for g in gpus)
        for gpu in available:
            jobs=self.pending()
            for s,k in jobs:
                pid=self.profile_id(s);profile=self.state['profiles'].get(pid)
                if profile and profile['status']!='profiled':
                    self.skip_geometry(s,'Memory smoke failed; '+profile.get('error','unknown'));continue
                if profile:
                    need=1.2*profile['peak_reserved_gib']+2
                    if need>capacity:
                        self.skip_geometry(s,f'Measured memory admission {need:.2f}GiB exceeds device capacity {capacity:.2f}GiB');continue
                    if gpu['free_gib']>=need:self.launch(s,k,gpu);break
                else:
                    if 'profile/'+pid in self.active:continue
                    # Initial estimate is conservative relative to accepted P87 peaks.
                    estimate=s['voxels']*700/1024**3+1
                    if gpu['free_gib']>=1.2*estimate+2 or gpu['free_gib']>=.94*gpu['total_gib']:
                        self.launch(s,0,gpu,smoke=True);break

    def run_loop(self):
        self.validate_source()
        if not self.state['sweeps']:self.add_batch('first')
        self.collect_existing();self.persist()
        while self.state['stage']!='complete':
            self.reap()
            current=[s for s in self.state['sweeps'] if s['batch']==self.state['stage']]
            done=current and all(sweep_state(s['results'],s.get('preparation_failure',False))['done'] for s in current)
            if done and not self.active:
                checkpoint='first_batch' if self.state['stage']=='first' else 'final'
                generate(self.state,self.review,checkpoint,seal=True)
                self.event('report_checkpoint',report=str(self.review/checkpoint/'report.md'))
                if self.state['stage']=='first':
                    self.state['stage']='rest';self.persist();self.add_batch('rest')
                    self.event('automatic_continuation',cases=16)
                else:
                    self.state['stage']='complete';self.persist();break
            try:self.dispatch()
            except Exception as e:self.event('scheduler_error',error=str(e),traceback=traceback.format_exc())
            if time.time()-self.last_report>45:
                generate(self.state,self.review,'live');self.last_report=time.time();self.persist()
            time.sleep(5)
        generate(self.state,self.review,'live')
        self.event('campaign_finished');self.persist()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    config=read(a.config);run=Path(config['run_root']);run.mkdir(parents=True,exist_ok=True)
    with (run/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        Controller(config).run_loop()

if __name__=='__main__':main()
