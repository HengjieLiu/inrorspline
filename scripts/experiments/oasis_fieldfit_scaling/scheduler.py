"""Durable memory-admitted four-GPU queue; workers never share a fit budget."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import HERE,external,read_json,write_json,now,sources

TERMINAL={'converged','unresolved','budget_not_reached','failed','resource_limited'}

def snapshots():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a):dict(index=int(a),uuid=u,total=int(t),free=int(f))
            for a,u,t,f in ([x.strip() for x in line.split(',')] for line in text.splitlines())}

def alive(pid,out):
    try:
        if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':return False
        return str(out).encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except OSError:return False

def host_available():
    line=next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    return int(line.split()[1])/1024

def resident(pid):
    try:
        l=next(l for l in Path(f'/proc/{pid}/status').read_text().splitlines() if l.startswith('VmRSS:'))
        return int(l.split()[1])/1024
    except (OSError,StopIteration):return 0

def plan_batch(free,active,candidates,limit,headroom=2048):
    available=free-headroom-sum(max(0,a['required']-a.get('current',0)) for a in active)
    chosen=[]
    for ident,required in candidates:
        if len(active)+len(chosen)>=limit:break
        if required<=available:chosen.append((ident,required));available-=required
    possible=len(active)+len(chosen)
    tier=next((n for n in (16,8,4,2,1) if n<=limit and n<=possible),0)
    return chosen[:max(0,tier-len(active))]

def retry_decision(job,failure,gpu_total):
    """Pure policy: retain elapsed time; never spin on an exclusive tile1 OOM."""
    tile=job.get('tile',128);previous=job.get('required_mib',4096)
    ceiling=gpu_total-2560
    if tile==1 and job.get('exclusive',False) and previous>=ceiling:
        return dict(status='resource_limited',spent=max(job.get('spent',0),failure.get('active_seconds',0)))
    return dict(status='pending',tile=max(1,tile//2),
                required_mib=min(ceiling,max(previous*1.3,previous+2048)),
                spent=max(job.get('spent',0),failure.get('active_seconds',0)),retry_after=time.time()+60)

class Queue:
    def __init__(self,root):
        self.root=external(root);self.manifest=read_json(self.root/'manifest.json');self.cfg=self.manifest['config']
        self.lock=open(self.root/'queue.lock','a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pre=read_json(self.root/'preflight.json')
        if pre['status']!='passed' or pre['source_hashes']!=sources() or self.manifest['source_hashes']!=sources():
            raise RuntimeError('Preflight/frozen source mismatch')
        self.profiles=read_json(self.root/'profiles.json');self.children={};self.report=None;self.last_report=0
        self.state=read_json(self.root/'queue_state.json') if (self.root/'queue_state.json').exists() else dict(
            created_at=now(),jobs=[dict(**j,status='pending',attempts=[],spent=0,tile=self.cfg['operator_rank_tile']) for j in self.manifest['jobs']])
        self.state['controller_pid']=os.getpid();self.save()
    def save(self):
        self.state.update(updated_at=now(),counts=dict(Counter(j['status'] for j in self.state['jobs'])))
        write_json(self.root/'queue_state.json',self.state)
    def requirement(self,j):
        if 'required_mib' in j:return j['required_mib']
        rows=[p for p in self.profiles if p['family']==j['family']]
        chosen=min(rows,key=lambda p:abs(p['cap']-j['cap']))
        # Profiling covers forward+adjoint; reserve additional LSQR fields,
        # coefficient vectors, pyramid temporaries and the CUDA context.
        coeff=j['cap']*8*10/2**20
        return math.ceil(max(5120,chosen['peak_reserved_mib']*1.5+2560+coeff)/256)*256
    def recover(self):
        for j in self.state['jobs']:
            if j['status']!='running':continue
            a=j['attempts'][-1];out=Path(a['out']);p=a['pid']
            if alive(p,out):continue
            child=self.children.pop(p,None)
            if child is not None:child.wait()
            result=out/'result.json'
            if result.exists():
                r=read_json(result);j.update(status=r['status'],result=str(result),spent=r['fit_active_seconds'])
                a.update(outcome=r['status'],finished_at=now())
            else:
                failure=read_json(out/'failure.json') if (out/'failure.json').exists() else dict(error='worker_exit_without_result',oom=False)
                a.update(outcome='oom' if failure.get('oom') else 'failed',failure=failure,finished_at=now())
                if (out/'checkpoint.json').exists():j['resume']=str(out)
                if failure.get('oom'):
                    j.update(retry_decision(j,failure,a['gpu_total']))
                else:j.update(status='failed',failure=failure)
            self.save()
    def running(self,gpu=None):
        return [j for j in self.state['jobs'] if j['status']=='running' and (gpu is None or j['attempts'][-1]['gpu']==gpu)]
    def launch(self,j,gpu,required):
        snap=snapshots()[gpu]
        active=self.running(gpu);reservations=[]
        for aj in active:
            a=aj['attempts'][-1];progress=Path(a['out'])/'progress.json'
            p=read_json(progress) if progress.exists() else {}
            reservations.append(dict(required=aj['required_mib'],current=p.get('reserved_mib',0)))
        if not plan_batch(snap['free'],reservations,[(j['id'],required)],self.cfg['gpu_limits'][str(gpu)],self.cfg['gpu_headroom_mib']):
            # A tier must be admitted atomically as a group. A single spawn
            # after planning may have e.g.3 active workers; only memory applies.
            avail=snap['free']-self.cfg['gpu_headroom_mib']-sum(max(0,a['required']-a['current']) for a in reservations)
            if required>avail or len(active)>=self.cfg['gpu_limits'][str(gpu)]:return False
        held=sum(max(0,self.cfg['host_reservation_mib_per_worker']-resident(a['attempts'][-1]['pid'])) for a in self.running())
        if host_available()-held<self.cfg['host_reservation_mib_per_worker']+8192:return False
        out=self.root/'jobs'/j['id']/f'attempt{len(j["attempts"])}';out.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=snap['uuid'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        args=[sys.executable,'-B',str(HERE/'worker.py'),'--root',str(self.root),'--job',j['id'],'--out',str(out),
              '--spent',str(j.get('spent',0)),'--tile',str(j['tile']),'--seconds',str(self.cfg['fit_seconds'])]
        if j.get('resume'):args+=['--resume',j['resume']]
        with open(out/'worker.log','ab') as log:child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        self.children[child.pid]=child
        j.update(status='running',required_mib=required,exclusive=not active)
        j['attempts'].append(dict(pid=child.pid,out=str(out),gpu=gpu,gpu_uuid=snap['uuid'],gpu_total=snap['total'],
                                  required_mib=required,tile=j['tile'],started_at=now(),started_epoch=time.time()))
        self.save();return True
    def update_report(self,final=False):
        if self.report is not None and self.report.poll() is None:
            if final:self.report.wait()
            else:return
        if not final and time.time()-self.last_report<120:return
        args=[sys.executable,'-B',str(HERE/'report.py'),'--root',str(self.root)]
        if final:args+=['--final']
        with open(self.root/'report.log','ab') as log:self.report=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        self.last_report=time.time()
        if final:
            code=self.report.wait();self.state['final_report_returncode']=code
    def run(self):
        while True:
            self.recover()
            if all(j['status'] in TERMINAL for j in self.state['jobs']):break
            snaps=snapshots()
            for gpu in sorted(snaps,key=lambda g:-snaps[g]['free']):
                if str(gpu) not in self.cfg['gpu_limits']:continue
                active=[]
                for j in self.running(gpu):
                    p=Path(j['attempts'][-1]['out'])/'progress.json';d=read_json(p) if p.exists() else {}
                    active.append(dict(required=j['required_mib'],current=d.get('reserved_mib',0)))
                pending=[j for j in self.state['jobs'] if j['status']=='pending' and j.get('retry_after',0)<=time.time()]
                candidates=[(j['id'],self.requirement(j)) for j in pending]
                chosen=plan_batch(snaps[gpu]['free'],active,candidates,self.cfg['gpu_limits'][str(gpu)],self.cfg['gpu_headroom_mib'])
                byid={j['id']:j for j in pending}
                for ident,required in chosen:self.launch(byid[ident],gpu,required)
            self.save();self.update_report();time.sleep(self.cfg['poll_seconds'])
        self.state['finished_at']=now();self.save();self.update_report(final=True);self.save()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();Queue(a.root).run()
if __name__=='__main__':main()
