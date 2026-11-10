#!/usr/bin/env python
"""Independent four-GPU admission, durable handoff, and automatic OOM requeue."""
import argparse
from collections import Counter
import fcntl
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from support import (HERE, SCIENCE, atomic_json, read_json, sha, now,
                     assert_external, validate_revision, cuda_oom)
from controller import Queue, TERMINAL, process_identity, same_process, safe_progress


def gpu_snapshots():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    snapshots = {}
    for line in result.stdout.splitlines():
        index, uuid, name, total, free = [x.strip() for x in line.split(',')]
        snapshots[int(index)] = dict(gpu=int(index), uuid=uuid, name=name,
                                     total_mib=int(total), free_mib=int(free))
    return snapshots


def plan_batch(snapshot, active, candidates, limit, headroom=2048):
    """Largest fitting total-worker tier; reservations include unobserved peaks.

    candidates is [(job_id, required_mib), ...] in stable queue order. A second
    memory read before each spawn prevents stale batches oversubscribing a GPU.
    Existing workers are never evicted when external occupancy increases.
    """
    if limit not in (1, 2, 4, 8, 16):
        raise ValueError('Unsupported per-GPU limit')
    reserved = sum(max(0., a['required_mib']-a.get('current_reserved_mib', 0.)) for a in active)
    available = snapshot['free_mib']-headroom-reserved
    fitting = []
    for ident, required in candidates:
        if len(active)+len(fitting) >= limit:
            break
        if required <= available:
            fitting.append((ident, required))
            available -= required
    possible = len(active)+len(fitting)
    tier = next((n for n in (16, 8, 4, 2, 1) if n <= limit and n <= possible), 0)
    return tier, fitting[:max(0, tier-len(active))]


def recover_worker(folder):
    matches = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            argv = (path/'cmdline').read_bytes().split(b'\0')
            entry = str(HERE/'worker_entry.py').encode()
            legacy = str(SCIENCE/'worker.py').encode()
            if not any(x in argv for x in (entry, legacy)) or b'--output' not in argv:
                continue
            pos = argv.index(b'--output')
            if argv[pos+1].decode() == str(folder):
                identity = process_identity(int(path.name))
                if identity:
                    matches.append(identity)
        except (OSError, IndexError, UnicodeError):
            continue
    if len(matches) > 1:
        raise RuntimeError('Multiple workers for the same persisted attempt')
    return matches[0] if matches else None


class MultiGPUQueue(Queue):
    def __init__(self, root, manifest, revision_path):
        super().__init__(root, manifest)
        self.revision_path = Path(revision_path)
        self.revision = read_json(self.revision_path)
        self.policy = self.revision['queue_policy']
        self.limits = {int(k): v for k, v in self.revision['gpu_limits'].items()}
        self.state['scheduler_revision'] = str(self.revision_path)
        self.state['scheduler_revision_sha256'] = sha(self.revision_path)
        for job in self.state['jobs']:
            for attempt in job.get('attempts', []):
                attempt.setdefault('physical_gpu', 0)
                attempt.setdefault('worker_generation', 'original_v3')
            if job.get('active_attempt'):
                job['active_attempt'].setdefault('physical_gpu', 0)
                job['active_attempt'].setdefault('worker_generation', 'original_v3')
        for job in self.state['jobs']:
            if job['status'] == 'failed':
                reason = job.get('failure', {}).get('reason', '')
                detail = safe_progress(Path(job.get('active_attempt', {}).get('folder', '/nonexistent'))/'failure.json')
                if cuda_oom(reason) or detail.get('cuda_oom') is True:
                    self.failure(job, reason or 'CUDA out of memory from previous scheduler')

    def save(self, status='running', warning=None):
        self.state['running_per_gpu'] = dict(Counter(
            str(j['active_attempt'].get('physical_gpu', 0)) for j in self.state['jobs']
            if j['status'] in ('running', 'launching', 'terminating')))
        self.state['gpu_limits'] = {str(k): v for k, v in self.limits.items()}
        super().save(status, warning)

    def failure(self, job, reason, status='failed'):
        attempt = job.get('active_attempt', {})
        detail = safe_progress(Path(attempt.get('folder', '/nonexistent'))/'failure.json')
        oom = cuda_oom(reason) or (detail.get('cuda_oom') is True and
                                  reason in (detail.get('error'), 'worker_exit_without_valid_completion'))
        if status != 'failed' or not oom:
            return super().failure(job, reason, status)
        count = job.get('oom_retries', 0)+1
        previous = attempt.get('required_mib', self.manifest['memory_profiles_mib'][job['profile_key']])
        peak = safe_progress(Path(attempt.get('folder', '/nonexistent'))/'progress.json').get('peak_reserved_mib', 0)
        if not isinstance(peak, (int, float)) or not math.isfinite(peak):
            peak = 0
        ceiling = max(self.revision['gpu_total_mib'].values())-self.policy['headroom_mib']-512
        required = min(ceiling, math.ceil(max(previous*1.25, previous+2048, peak+2048)/256)*256)
        delay = min(900, 60*(2**min(count-1, 4)))
        event = dict(at=now(), job_id=job['id'], reason=reason, status='oom_requeued',
                     attempt=attempt, retry_number=count, next_required_mib=required, cooldown_sec=delay)
        atomic_json(self.root/'failures'/f"{job['id']}_{time.time_ns()}.json", event)
        attempt.update(outcome='cuda_oom', failed_at=now())
        job.update(status='pending', failure=event, oom_retries=count,
                   required_mib_override=required, retry_not_before=time.time()+delay)
        self.last_progress = time.time()

    def reconcile(self):
        for job in self.state['jobs']:
            if job['status'] in ('launching', 'running', 'terminating'):
                attempt = job['active_attempt']
                if attempt.get('identity') is None:
                    attempt['identity'] = recover_worker(Path(attempt['folder']))
        super().reconcile()

    def reservations(self, gpu):
        active = []
        for job in self.state['jobs']:
            if job['status'] not in ('launching', 'running', 'terminating'):
                continue
            attempt = job['active_attempt']
            if attempt.get('physical_gpu', 0) != gpu:
                continue
            progress = safe_progress(Path(attempt['folder'])/'progress.json')
            current = progress.get('current_reserved_mib', 0)
            if not isinstance(current, (int, float)) or not math.isfinite(current) or current < 0:
                current = 0
            active.append(dict(required_mib=attempt['required_mib'], current_reserved_mib=current))
        return active

    def launch_on(self, job, required, gpu):
        number = len(job['attempts'])
        folder = self.root/'jobs'/job['id']/f'attempt{number}'
        if folder.exists():
            identity = recover_worker(folder)
            if not identity:
                self.failure(job, 'preexisting_attempt_directory_preserved')
                return
            attempt = dict(folder=str(folder), started_epoch=time.time(), last_activity=time.time(),
                           required_mib=required, physical_gpu=gpu, identity=identity,
                           worker_generation='multigpu_v1', recovered_unpersisted_intent=True)
            job['attempts'].append(attempt)
            job.update(status='running', active_attempt=attempt)
            self.save()
            return
        folder.mkdir(parents=True, exist_ok=False)
        attempt = dict(folder=str(folder), started_epoch=time.time(), last_activity=time.time(),
                       required_mib=required, physical_gpu=gpu, identity=None,
                       gpu_uuid=self.revision['gpu_uuids'][str(gpu)], worker_generation='multigpu_v1',
                       scheduler_revision_sha256=sha(self.revision_path))
        job['attempts'].append(attempt)
        job.update(status='launching', active_attempt=attempt)
        self.save()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=attempt['gpu_uuid'], PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS=str(self.policy['cpu_threads']), MKL_NUM_THREADS=str(self.policy['cpu_threads']),
                   OPENBLAS_NUM_THREADS=str(self.policy['cpu_threads']), NUMEXPR_NUM_THREADS=str(self.policy['cpu_threads']))
        try:
            with (folder/'worker.log').open('ab', buffering=0) as log:
                child = subprocess.Popen([sys.executable, '-B', str(HERE/'worker_entry.py'),
                    '--root', str(self.root), '--revision', str(self.revision_path),
                    '--job-id', job['id'], '--output', str(folder), '--gpu', str(gpu)],
                    stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            attempt['identity'] = process_identity(child.pid)
            self.children[job['id']] = child
            job['status'] = 'running'
        except Exception as exc:
            self.failure(job, 'spawn_failed: '+str(exc))
        self.save()

    def admit_once(self):
        validate_revision(self.root, self.revision)
        if shutil.disk_usage(self.root).free/2**30 < self.policy['min_disk_gib']:
            raise RuntimeError('Shared disk admission guard')
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0])/2**20 < self.policy['min_host_gib']:
            raise RuntimeError('Shared host-memory admission guard')
        ready = []
        for job in self.state['jobs']:
            if job['status'] == 'pending' and time.time() >= job.get('retry_not_before', 0) and self.check_dependencies(job):
                ready.append(job)
        # Original pending jobs precede retries; repeated OOM never blocks the rest.
        ready.sort(key=lambda j: j.get('oom_retries', 0))
        self.state['gpu_admission'] = {}
        for gpu, limit in self.limits.items():
            # Fresh telemetry before each independent GPU batch.
            snap = gpu_snapshots().get(gpu)
            if not snap or snap['uuid'] != self.revision['gpu_uuids'][str(gpu)]:
                self.state['gpu_admission'][str(gpu)] = {'status': 'telemetry_or_identity_unavailable', 'at': now()}
                continue
            active = self.reservations(gpu)
            candidates = [(j['id'], j.get('required_mib_override', self.manifest['memory_profiles_mib'][j['profile_key']]))
                          for j in ready if j['status'] == 'pending']
            tier, batch = plan_batch(snap, active, candidates, limit, self.policy['headroom_mib'])
            self.state['gpu_admission'][str(gpu)] = dict(at=now(), snapshot=snap, active=len(active),
                selected_tier=tier, planned_batch=len(batch), status='admitting' if batch else 'waiting_memory_or_slots')
            by_id = {j['id']: j for j in ready}
            for ident, required in batch:
                fresh = gpu_snapshots().get(gpu)
                if not fresh or fresh['uuid'] != snap['uuid']:
                    break
                active = self.reservations(gpu)
                reserve = sum(max(0, a['required_mib']-a['current_reserved_mib']) for a in active)
                if len(active) >= limit or fresh['free_mib']-reserve < required+self.policy['headroom_mib']:
                    break
                self.launch_on(by_id[ident], required, gpu)

    def run(self):
        while True:
            tick_started = time.monotonic()
            try:
                self.reconcile()
                if all(j['status'] in TERMINAL for j in self.state['jobs']):
                    self.save('finished')
                    subprocess.run([sys.executable, '-B', str(SCIENCE/'report.py'), '--root', str(self.root)], check=False)
                    return
                self.admit_once()
                self.save()
            except Exception as exc:
                self.save('waiting_infrastructure', str(exc))
            # A busy GPU is never converted into terminal job failures. Other
            # GPUs are checked independently on the next one-minute pass.
            time.sleep(max(.1, self.policy['poll_sec']-(time.monotonic()-tick_started)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--revision', type=Path, required=True)
    args = parser.parse_args()
    root = assert_external(args.root)
    revision = read_json(args.revision)
    if revision['gpu_limits'] != {'0': 16, '1': 16, '2': 4, '3': 4} or revision['queue_policy']['poll_sec'] != 60:
        raise ValueError('This continuation requires the authorized 16/16/4/4 limits and 60-second polling')
    validate_revision(root, revision)
    if (root/'SUPERSEDED.json').exists():
        raise RuntimeError('Refusing a superseded execution')
    with (root/'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(root/'controller_identity.json', dict(identity=process_identity(os.getpid()), started_at=now(),
                    initial_run_started_at=read_json(root/'launch_manifest.json')['created_at'],
                    scheduler_revision=str(args.revision), scheduler_revision_sha256=sha(args.revision)))
        MultiGPUQueue(root, read_json(root/'launch_manifest.json'), args.revision).run()


if __name__ == '__main__':
    main()
