#!/usr/bin/env python
"""GPU-0-only, bounded-memory, durable P81 queue with isolated job failures."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from runtime import HERE, atomic_json, read_json, now, code_hashes, sha, assert_external

TERMINAL = {'complete', 'failed', 'dependency_failed', 'unavailable', 'resource_deferred'}


def process_identity(pid):
    try:
        parts = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        if parts[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': parts[19], 'uid': (Path('/proc') / str(pid)).stat().st_uid}
    except (OSError, IndexError):
        return None


def same_process(identity):
    return identity is not None and process_identity(identity['pid']) == identity


def recover_identity(folder):
    """Recover this exact worker after a controller death in the spawn window."""
    matches=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            argv=(p/'cmdline').read_bytes().split(b'\0')
            if str(HERE/'worker.py').encode() not in argv or b'--output' not in argv:continue
            pos=argv.index(b'--output')
            if pos+1<len(argv) and argv[pos+1].decode()==str(folder):
                identity=process_identity(int(p.name))
                if identity:matches.append(identity)
        except (OSError,UnicodeError):continue
    if len(matches)>1:raise RuntimeError('Multiple workers found for one durable launch intent')
    return matches[0] if matches else None


def gpu_snapshot():
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    for line in r.stdout.splitlines():
        idx, uuid, total, free = [s.strip() for s in line.split(',')]
        if idx == '0':
            return {'gpu': 0, 'uuid': uuid, 'total_mib': int(total), 'free_mib': int(free)}
    raise RuntimeError('GPU0 telemetry missing')


def safe_progress(path):
    try:
        value=read_json(path)
        return value if isinstance(value,dict) else {}
    except (OSError, ValueError):
        return {}


def can_admit(snapshot, active_reservations, required, headroom=2048, max_jobs=8):
    # Reserve the not-yet-observed part of measured peaks. Current allocation is
    # already reflected in nvidia-smi free memory; missing progress is conservative.
    reserve = sum(max(0, x['required_mib'] - x.get('current_reserved_mib', 0))
                  for x in active_reservations)
    return len(active_reservations) < min(8, max_jobs) and snapshot['free_mib'] - reserve >= required + headroom


def activity(identity, folder):
    result = []
    try:
        p = Path('/proc') / str(identity['pid'])
        stat = (p/'stat').read_text().rsplit(')', 1)[1].split()
        result.extend(stat[11:13])
        result.append((p/'io').read_text())
    except OSError:
        pass
    for name in ('progress.json', 'worker.log', 'result.json'):
        try:
            st = (folder/name).stat(); result.append((name, st.st_size, st.st_mtime_ns))
        except OSError:
            pass
    return result


class Queue:
    def __init__(self, root, manifest):
        self.root, self.manifest = assert_external(root), manifest
        self.path = self.root/'queue_state.json'
        self.children = {}
        self.policy = manifest['queue_policy']
        if self.path.exists():
            self.state = read_json(self.path)
            if self.state['manifest_sha256'] != sha(self.root/'launch_manifest.json'):
                raise RuntimeError('Queue manifest changed; refusing incompatible resume')
        else:
            self.state = {'manifest_sha256': sha(self.root/'launch_manifest.json'),
                          'created_at': now(), 'jobs': [dict(j, status='pending', attempts=[]) for j in manifest['jobs']]}
        self.last_progress = time.time()
        self.last_guard = 0.

    def save(self, status='running', warning=None):
        self.state.update(updated_at=now(), status=status, warning=warning,
                          counts=dict(Counter(j['status'] for j in self.state['jobs'])))
        atomic_json(self.path, self.state)

    def failure(self, job, reason, status='failed'):
        event = {'at': now(), 'job_id': job['id'], 'reason': reason,
                 'attempt': job.get('active_attempt'), 'status': status}
        atomic_json(self.root/'failures'/f"{job['id']}_{time.time_ns()}.json", event)
        job.update(status=status, failure=event)
        self.last_progress = time.time()

    def check_dependencies(self, job):
        by_id = {j['id']: j for j in self.state['jobs']}
        for dep in job.get('dependencies', []):
            parent = by_id[dep['job_id']]
            p = self.root/dep['path']
            if p.is_file():
                try:
                    value=read_json(p)
                    if value.get('status')!='complete' or value.get('job_id')!=parent['id']:
                        raise ValueError('Wrong endpoint identity/status')
                    for kind in ('basis','coefficients'):
                        if sha(value[kind+'_path'])!=value[kind+'_sha256']:
                            raise ValueError(kind+' checksum mismatch')
                except Exception as exc:
                    self.failure(job,'Invalid dependency: '+str(exc),'dependency_failed')
                    return False
                continue
            if parent['status'] in TERMINAL:
                self.failure(job, 'Required checkpoint unavailable: '+dep['path'], 'dependency_failed')
                return False
            return False
        return True

    def reconcile(self):
        for job in self.state['jobs']:
            if job['status'] not in ('launching', 'running', 'terminating'):
                continue
            a = job['active_attempt']; folder = Path(a['folder'])
            identity = a.get('identity')
            if identity is None:
                identity=recover_identity(folder)
                if identity:a['identity']=identity
            if same_process(identity):
                job['status'] = 'running' if job['status'] == 'launching' else job['status']
                state = activity(identity, folder)
                if state != a.get('activity'):
                    a.update(activity=state, last_activity=time.time())
                elapsed = time.time()-a['started_epoch']
                stalled = time.time()-a.get('last_activity', a['started_epoch'])
                if job['status'] == 'terminating':
                    if time.time()-a['term_epoch'] > self.policy['kill_grace_sec']:
                        try:os.killpg(identity['pid'], signal.SIGKILL)
                        except ProcessLookupError:pass
                elif elapsed > self.policy['job_timeout_sec'] or stalled > self.policy['stall_timeout_sec']:
                    a['terminate_reason'] = 'wall_time_limit' if elapsed > self.policy['job_timeout_sec'] else 'no_cpu_io_or_file_progress'
                    try:os.killpg(identity['pid'], signal.SIGTERM)
                    except ProcessLookupError:pass
                    a['term_epoch'] = time.time(); job['status'] = 'terminating'
                continue
            child = self.children.pop(job['id'], None)
            if child:
                child.wait()
            result = safe_progress(folder/'result.json')
            if result.get('status') == 'complete' and result.get('job_id') == job['id']:
                try:
                    expected=[self.root/'endpoints'/job['id']/f'cap{c}.json' for c in job['budgets']]
                    if set(result.get('endpoints',[]))!={str(p) for p in expected}:
                        raise ValueError('Worker completion does not cover expected budgets')
                    for path in expected:
                        endpoint=read_json(path)
                        if endpoint.get('status')!='complete' or endpoint.get('job_id')!=job['id']:
                            raise ValueError('Endpoint identity/status mismatch')
                        for kind in ('basis','coefficients','field'):
                            if sha(endpoint[kind+'_path'])!=endpoint[kind+'_sha256']:
                                raise ValueError('Serialized '+kind+' checksum mismatch')
                    job.update(status='complete', result_path=str(folder/'result.json'))
                    self.last_progress = time.time()
                except Exception as exc:self.failure(job,'completion_validation_failed: '+str(exc))
            else:
                err = safe_progress(folder/'failure.json')
                reason = a.get('terminate_reason') or err.get('error') or 'worker_exit_without_valid_completion'
                self.failure(job, reason)

    def launch(self, job, required):
        attempt = len(job['attempts'])
        folder = self.root/'jobs'/job['id']/f'attempt{attempt}'
        if folder.exists():
            identity=recover_identity(folder)
            if identity:
                a={'folder':str(folder),'started_epoch':time.time(),'required_mib':required,
                   'last_activity':time.time(),'identity':identity,'recovered_unpersisted_intent':True}
                job['attempts'].append(a);job.update(status='running',active_attempt=a)
            else:self.failure(job,'preexisting_attempt_directory_preserved')
            self.save();return
        try:folder.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            self.failure(job,'attempt_directory_failed: '+str(exc));self.save();return
        a = {'folder': str(folder), 'started_epoch': time.time(), 'required_mib': required,
             'last_activity': time.time(), 'identity': None}
        job['attempts'].append(a); job.update(status='launching', active_attempt=a)
        self.save()  # Persist intent before spawning.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS=str(self.policy['cpu_threads']), MKL_NUM_THREADS=str(self.policy['cpu_threads']),
                   OPENBLAS_NUM_THREADS=str(self.policy['cpu_threads']), NUMEXPR_NUM_THREADS=str(self.policy['cpu_threads']))
        try:
            with (folder/'worker.log').open('ab', buffering=0) as log:
                p = subprocess.Popen([sys.executable, '-B', str(HERE/'worker.py'), '--root', str(self.root),
                                      '--job-id', job['id'], '--output', str(folder)],
                                     stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            a['identity'] = process_identity(p.pid)
            self.children[job['id']] = p
            job['status'] = 'running'
        except Exception as exc:
            self.failure(job, 'spawn_failed: '+str(exc))
        self.save()

    def run(self):
        while True:
            try:self.reconcile()
            except Exception as exc:
                self.save('reconcile_warning',str(exc))
            if all(j['status'] in TERMINAL for j in self.state['jobs']):
                self.save('finished')
                subprocess.run([sys.executable, '-B', str(HERE/'report.py'), '--root', str(self.root)], check=False)
                return
            try:
                if time.time()-self.last_guard > 120:
                    if code_hashes() != self.manifest['code_hashes']:
                        raise RuntimeError('Numerical/controller source hashes changed')
                    if sha(self.root/'input_manifest.json') != self.manifest['input_manifest_sha256']:
                        raise RuntimeError('Input manifest changed')
                    self.last_guard = time.time()
                snap = gpu_snapshot()
                if snap['uuid'] != self.manifest['gpu0_uuid']:
                    raise RuntimeError('GPU0 identity changed')
                free_gib = shutil.disk_usage(self.root).free/2**30
                mem = {k.rstrip(':'): int(v.split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
                if free_gib < self.policy['min_disk_gib'] or mem['MemAvailable']/2**20 < self.policy['min_host_gib']:
                    raise RuntimeError('Shared disk/RAM admission guard')
                active = []
                for j in self.state['jobs']:
                    if j['status'] in ('running','launching','terminating'):
                        a=j['active_attempt']; progress=safe_progress(Path(a['folder'])/'progress.json')
                        import math
                        current=progress.get('current_reserved_mib',0)
                        if not isinstance(current,(int,float)) or not math.isfinite(current) or current<0:current=0
                        active.append({'required_mib': a['required_mib'], 'current_reserved_mib': current})
                for job in self.state['jobs']:
                    if job['status'] != 'pending' or not self.check_dependencies(job):
                        continue
                    required=self.manifest['memory_profiles_mib'][job['profile_key']]
                    if required+self.policy['headroom_mib'] > snap['total_mib']:
                        self.failure(job,'measured_peak_exceeds_gpu_capacity','resource_deferred'); continue
                    if can_admit(snap, active, required, self.policy['headroom_mib'], self.policy['max_jobs']):
                        self.launch(job, required)
                        active.append({'required_mib': required, 'current_reserved_mib': 0})
                self.save()
            except Exception as exc:
                self.save('waiting_infrastructure', str(exc))
            # Admission cannot wait forever after the last active worker exits.
            active_any=any(j['status'] in ('running','launching','terminating') for j in self.state['jobs'])
            if not active_any and time.time()-self.last_progress > self.policy['admission_timeout_sec']:
                for job in self.state['jobs']:
                    if job['status']=='pending':
                        self.failure(job,'admission_deadline_exceeded','resource_deferred')
                self.save('deferred')
            time.sleep(self.policy['poll_sec'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a=p.parse_args()
    root=assert_external(a.root)
    if (root/'SUPERSEDED.json').exists():
        raise RuntimeError('This execution was superseded; preserve it and use its recorded successor')
    with (root/'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=read_json(root/'launch_manifest.json')
        if m.get('gpu') != 0 or not 1 <= m['queue_policy']['max_jobs'] <= 8:
            raise ValueError('P81 is GPU0 only, maximum eight jobs')
        if not (root/'PREFLIGHT_PASSED.json').is_file():
            raise RuntimeError('Launch requires verified preflight')
        gate=read_json(root/'PREFLIGHT_PASSED.json')
        if gate.get('passed') is not True or gate.get('launch_manifest_sha256')!=sha(root/'launch_manifest.json'):
            raise RuntimeError('Preflight gate does not bind this launch manifest')
        atomic_json(root/'controller_identity.json', {'identity': process_identity(os.getpid()), 'started_at': now()})
        Queue(root,m).run()


if __name__ == '__main__':
    main()
