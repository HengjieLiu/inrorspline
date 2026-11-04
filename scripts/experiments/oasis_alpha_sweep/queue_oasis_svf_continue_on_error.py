#!/usr/bin/env python
"""P74 durable continue-on-error queue. Numerical worker/auditor remain immutable."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import run_oasis_bspline_svf_sweep_recovery as r
from queue_oasis_svf_eight_workers import process_info, same_process, option, experiment_processes

FOLDER = 'continue_on_error_v1'
GPUS = ('0', '1', '2', '3')
WORKER = Path(r.__file__).resolve()
REPORT = Path(__file__).with_name('report_oasis_svf_queue_outcome.py')
TERMINAL = ('complete', 'failed_deferred')
POLICY = {'max_concurrent': 8, 'workers_per_gpu': 2, 'required_free_mib': 8192,
          'reservation': 'full 8192MiB per active child persists until exit; conservative double accounting',
          'oom_retries': 1, 'retry_order': 'after all unattempted workers and pending audits',
          'stall_warning_sec': 1800, 'stall_terminate_sec': 3600, 'terminate_grace_sec': 60,
          'min_host_available_gib': 32, 'min_artifact_free_gib': 400,
          'failure_policy': 'defer individual configuration, never stop unrelated configurations',
          'scientific_validation': 'unchanged recovery worker and auditor',
          'infrastructure_policy': 'pause admissions on shared storage/provenance failure; retry checks'}


def sources():
    paths = [Path(__file__), REPORT, r.c.REPO_ROOT/'tests/test_oasis_svf_continue_on_error.py',
             Path(__file__).with_name('queue_oasis_svf_eight_workers.py')]
    return {str(p.relative_to(r.c.REPO_ROOT)): r.sha(p) for p in paths}


@contextmanager
def deadline(seconds=30):
    """Bound interruptible controller I/O; kernel uninterruptible NFS remains a system limitation."""
    def expired(*_):
        raise TimeoutError(f'Controller operation exceeded {seconds}s')
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def safe_progress(path):
    try:
        with deadline(5):
            return r.c.read_json(path) if path.exists() else {'phase': 'loading'}
    except (OSError, ValueError, TimeoutError) as exc:
        return {'phase': 'progress_unavailable', 'warning': str(exc)}


def log_tail(path):
    if not path:
        return ''
    try:
        with deadline(5), Path(path).open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-32768))
            return stream.read().decode(errors='replace')
    except (OSError, TimeoutError) as exc:
        return f'[log unavailable: {exc}]'


def parse_gpus(output):
    rows, errors = {}, []
    for line in output.splitlines():
        try:
            gpu, free, used, util = [v.strip() for v in line.split(',')]
            if gpu not in GPUS:
                continue
            rows[gpu] = {'physical_gpu': gpu, 'memory_free_mib': int(free),
                         'memory_used_mib': int(used), 'utilization_gpu_percent': int(util)}
        except ValueError:
            errors.append(f'Invalid GPU telemetry row: {line}')
    return {'rows': rows, 'errors': errors, 'missing_gpus': sorted(set(GPUS)-set(rows))}


def gpu_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return parse_gpus(result.stdout)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {'rows': {}, 'missing_gpus': list(GPUS), 'errors': [str(exc)]}


def capacity(gpu, active, free_mib, required):
    count = sum(x['job'].get('gpu') == gpu for x in active.values())
    return gpu in GPUS and len(active) < 8 and count < 2 and free_mib-count*required >= required


def choose_job(jobs, gpu, affinity):
    ordinary = any(j['status'] in ('pending', 'running', 'launching', 'audit_pending', 'auditing') for j in jobs)
    choices = []
    for index, job in enumerate(jobs):
        status = job['status']
        if status not in ('pending', 'audit_pending', 'oom_retry'):
            continue
        if status == 'oom_retry' and ordinary:
            continue
        preferred = job.get('gpu') if status == 'audit_pending' else affinity.get(job['spec'].case_rank)
        choices.append(((status != 'audit_pending', preferred != gpu, index), job))
    return min(choices, key=lambda x: x[0])[1] if choices else None


def serialized(job):
    return {**{k: v for k, v in job.items() if k not in ('spec', 'row')}, 'spec': asdict(job['spec'])}


def activity(record, job):
    """Use both on-disk progress and actual CPU/IO activity, not progress JSON alone."""
    values = []
    try:
        stat = (Path('/proc')/str(record['pid'])/'stat').read_text().rsplit(')', 1)[1].split()
        values += stat[11:13]  # utime/stime
        values.append((Path('/proc')/str(record['pid'])/'io').read_text())
    except OSError:
        pass
    out = Path(job['out'])
    paths = [out/'progress.json', out/'summary.json', out/'independent_audit.json']
    if job.get('log_path'):
        paths.append(Path(job['log_path']))
    paths += list(out.glob('stage*/history*')) + list(out.glob('stage*/best.pt'))
    for path in paths:
        try:
            stat = path.stat()
            values.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            pass
    return values


class Queue:
    def __init__(self, root, manifest, required=8192):
        self.root, self.folder, self.manifest = root, root/FOLDER, manifest
        self.required = required
        self.jobs, self.active, self.affinity = [], {}, {}
        self.started = time.time()
        self.last_source_check = 0.
        self.initial_terminal = 0
        self.events = []

    def save(self):
        with deadline():
            r.atomic_json(self.folder/'state.json', {'updated_at': r.c.now_text(),
                          'jobs': [serialized(x) for x in self.jobs], 'policy': POLICY})

    def failure(self, job, action, reason, *, tail='', allow_oom=False):
        oom = allow_oom and 'cuda' in tail.lower() and 'out of memory' in tail.lower()
        event = {'id': f'{time.time_ns()}', 'created_at': r.c.now_text(), 'spec': asdict(job['spec']),
                 'run_name': job['spec'].name(), 'action': action, 'reason': reason, 'oom': oom,
                 'gpu': job.get('gpu'), 'attempt': job['attempt'], 'out': job.get('out'),
                 'log_path': job.get('log_path'), 'log_tail': tail,
                 'stage': safe_progress(Path(job['out'])/'progress.json') if job.get('out') else None,
                 **{k: self.manifest['cases'][str(job['spec'].case_rank)][k] for k in ('fixed_id', 'moving_id')}}
        with deadline():
            r.atomic_json(self.folder/'failures'/f"{event['id']}.json", event)
        retry = oom and job.get('oom_retries', 0) < POLICY['oom_retries']
        previous = dict(job)
        job.update(status='oom_retry' if retry else 'failed_deferred', failure=event,
                   retry_action=action, oom_retries=job.get('oom_retries', 0)+int(retry))
        try:
            self.save()  # Retry budget is durable BEFORE another dispatch.
        except Exception:
            job.clear()
            job.update(previous)
            raise
        print(json.dumps({'failure': event['run_name'], 'action': action, 'reason': reason,
                          'disposition': job['status']}), flush=True)

    def reconcile(self, live):
        old = r.c.read_json(self.folder/'state.json') if (self.folder/'state.json').exists() else {'jobs': []}
        by_name = {r.s.Spec(**x['spec']).name(): x for x in old['jobs']}
        by_path = {x['out']: x for x in live}
        if len(by_path) != len(live):
            raise RuntimeError('Duplicate active processes for one attempt')
        for spec in r.s.matrix():
            saved = by_name.get(spec.name(), {})
            job = {**saved, 'spec': spec, 'status': 'pending', 'attempt': saved.get('attempt', 0),
                   'gpu': saved.get('gpu'), 'oom_retries': saved.get('oom_retries', 0)}
            self.jobs.append(job)
            paths = sorted([p for p in (self.root/spec.name()).glob('attempt*')
                            if p.is_dir() and p.name[7:].isdigit()], key=lambda p: int(p.name[7:]))
            accepted = [p for p in paths if (p/'independent_audit.json').exists()]
            procs = [by_path.pop(str(p)) for p in paths if str(p) in by_path]
            # A child can start before its attempt directory exists (crash after Popen).
            procs += [by_path.pop(p) for p in list(by_path) if Path(p).parent == self.root/spec.name()]
            if len(procs) > 1:
                raise RuntimeError(f'Duplicate live configuration: {spec.name()}')
            if procs:
                rec = procs[0]
                job.update(out=rec['out'], gpu=rec['gpu'], attempt=int(Path(rec['out']).name[7:]),
                           status='running' if rec['action'] == 'worker' else 'auditing', record=rec)
                # Find exactly the current invocation log through /proc; never scan historical tails for OOM.
                try:
                    job['log_path'] = os.readlink(f"/proc/{rec['pid']}/fd/1")
                except OSError:
                    pass
                self.active[rec['pid']] = {'record': rec, 'process': None, 'job': job,
                                           'action': rec['action'], 'last_activity': time.time(), 'signature': None}
            elif accepted:
                job.update(out=str(accepted[-1]), attempt=int(accepted[-1].name[7:]))
                try:
                    if len(accepted) != 1:
                        raise ValueError('Duplicate accepted attempts')
                    row = r.validate(accepted[0], self.manifest, spec)
                    job.update(status='complete', row=row, gpu=row['gpu'])
                except Exception:
                    self.failure(job, 'resume_validation', traceback.format_exc())
            elif saved.get('status') in ('failed_deferred', 'oom_retry'):
                job['status'] = saved['status']
            elif paths:
                path = paths[-1]
                job.update(out=str(path), attempt=int(path.name[7:]))
                failure_files = list(path.glob('FAILED*.json'))
                if failure_files:
                    self.failure(job, 'historical_failure', 'Preserved historical failed attempt; review after queue',
                                 tail='\n'.join(log_tail(p) for p in failure_files))
                elif (path/'DONE').exists():
                    job['status'] = 'audit_pending'
                else:
                    self.failure(job, 'interrupted_attempt', 'No live process or completion marker; preserve and defer')
            elif saved.get('status') == 'launching':
                self.failure(job, 'interrupted_launch', 'Durable launch intent without surviving process or output')
            if job.get('gpu') is not None:
                self.affinity.setdefault(spec.case_rank, job['gpu'])
        if by_path:
            raise RuntimeError(f'Unrecognized live experiment processes: {by_path}')
        if len(self.active) > 8 or any(n > 2 for n in Counter(x['job']['gpu'] for x in self.active.values()).values()):
            raise RuntimeError('Live children exceed approved concurrency')
        self.initial_terminal = sum(j['status'] in TERMINAL for j in self.jobs)
        self.save()

    def finished_code(self, item):
        if item['process'] is not None:
            return item['process'].poll()
        if same_process(item['record']):
            return None
        marker = 'DONE' if item['action'] == 'worker' else 'independent_audit.json'
        return 0 if (Path(item['job']['out'])/marker).exists() else 1

    def poll(self):
        for pid, item in list(self.active.items()):
            code = self.finished_code(item)
            if code is None:
                self.watchdog(item)
                continue
            job = item['job']
            if code or item.get('terminated'):
                self.failure(job, item['action'], item.get('terminated') or f'Process exited {code}',
                             tail=log_tail(job.get('log_path')), allow_oom=not item.get('terminated'))
            elif item['action'] == 'worker':
                if (Path(job['out'])/'DONE').exists():
                    job['status'] = 'audit_pending'
                else:
                    self.failure(job, 'worker', 'Exit zero without DONE marker')
            else:
                try:
                    row = r.validate(Path(job['out']), self.manifest, job['spec'])
                except Exception:
                    self.failure(job, 'parent_validation', traceback.format_exc())
                else:
                    job.update(status='complete', row=row)
            del self.active[pid]
            self.save()

    def watchdog(self, item):
        now, rec = time.time(), item['record']
        if not rec.get('start_ticks'):
            current = process_info(rec['pid'])
            if current:
                item['record'] = current
            return
        try:
            with deadline(5):
                signature = activity(rec, item['job'])
        except (OSError, TimeoutError):
            return  # Storage observation alone is not evidence of a hung computation.
        if signature != item.get('signature'):
            item.update(signature=signature, last_activity=now)
        idle = now-item['last_activity']
        if idle >= POLICY['stall_warning_sec']:
            item['warning'] = f'No observed CPU/IO/file activity for {idle:.0f}s'
        if idle >= POLICY['stall_terminate_sec'] and not item.get('terminated'):
            current = process_info(rec['pid'])
            if current and current['uid'] == 1009 and current['argv'] == rec['argv'] and same_process(rec):
                os.kill(rec['pid'], signal.SIGTERM)
                item.update(terminated='Watchdog: 60 minutes without observed activity', terminated_at=now)
        if item.get('terminated') and now-item['terminated_at'] >= POLICY['terminate_grace_sec']:
            current = process_info(rec['pid'])
            if current and current['uid'] == 1009 and current['argv'] == rec['argv'] and same_process(rec):
                os.kill(rec['pid'], signal.SIGKILL)
        # The slot stays occupied until actual exit, including uninterruptible D-state.

    def launch(self, job, gpu, snapshot):
        before_launch = dict(job)
        retry = job['status'] == 'oom_retry'
        action = job.get('retry_action', 'worker') if retry else ('audit' if job['status'] == 'audit_pending' else 'worker')
        if retry and action == 'worker':
            job['attempt'] += 1
        path = Path(job['out']) if action == 'audit' else self.root/job['spec'].name()/f"attempt{job['attempt']}"
        job.update(out=str(path), gpu=gpu, action=action)
        try:
            if action == 'worker' and path.exists():
                raise FileExistsError(f'Refuse overwrite: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            if retry and action == 'audit':
                # Keep every prior audit artifact before the unchanged auditor overwrites its own paths.
                archive = path.parent/f"audit_oom_preserved_{time.time_ns()}"
                archive.mkdir()
                for p in list(path.glob('independent_*'))+list(path.glob('FAILED_audit*')):
                    if p.is_dir():
                        shutil.copytree(p, archive/p.name)
                    else:
                        shutil.copy2(p, archive/p.name)
            invocation = str(time.time_ns())
            log_path = path.parent/f"attempt{job['attempt']}_{action}_continue_{invocation}.log"
            log = log_path.open('x')
        except (OSError, ValueError):
            self.failure(job, 'launch_preparation', traceback.format_exc())
            return
        job.update(status='launching', log_path=str(log_path), invocation=invocation, launch_at=time.time())
        proc = None
        try:
            self.save()  # durable intent before process creation
            command = [sys.executable, '-u', '-B', str(WORKER), action, '--manifest',
                       str(self.folder/'launch_manifest.json'), '--out-dir', str(path), '--gpu', gpu]
            if action == 'worker':
                spec = job['spec']
                command += ['--case-rank', str(spec.case_rank), '--schedule', spec.schedule,
                            '--regularize', spec.regularize, '--alpha', str(spec.alpha)]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                   'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONDONTWRITEBYTECODE': '1'}
            try:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            except OSError:
                self.failure(job, 'process_spawn', traceback.format_exc())
                return
            rec = process_info(proc.pid) or {'pid': proc.pid, 'start_ticks': None, 'argv': command, 'uid': os.getuid()}
            job.update(status='running' if action == 'worker' else 'auditing', record=rec)
            self.active[proc.pid] = {'record': rec, 'process': proc, 'job': job, 'action': action,
                                     'signature': None, 'last_activity': time.time()}
            self.affinity.setdefault(job['spec'].case_rank, gpu)
            event = {'at': r.c.now_text(), 'run': job['spec'].name(), 'action': action, 'gpu': gpu,
                     'pid': proc.pid, 'log_path': str(log_path), 'out': str(path), 'snapshot': snapshot,
                     'required_free_mib': self.required, 'retry': retry}
            r.atomic_json(self.folder/'launches'/f'{invocation}.json', event)
            self.save()
            print(json.dumps(event), flush=True)
        except Exception:
            if proc is None and job['status'] == 'launching':
                job.clear()
                job.update(before_launch)
            raise
        finally:
            log.close()

    def infrastructure(self):
        with deadline(30):
            if time.time()-self.last_source_check > 60:
                r.assert_manifest(self.manifest)
                if self.manifest['execution']['source_hashes'] != sources():
                    raise RuntimeError('Controller/report source drift')
                self.last_source_check = time.time()
            free = shutil.disk_usage(self.root).free/2**30
            if free < POLICY['min_artifact_free_gib']:
                raise RuntimeError(f'Shared artifact capacity {free:.1f}GiB below guard')
            available = next(int(x.split()[1])/2**20 for x in Path('/proc/meminfo').read_text().splitlines()
                             if x.startswith('MemAvailable:'))
            if available < POLICY['min_host_available_gib']:
                raise RuntimeError(f'Host available RAM {available:.1f}GiB below guard')
            self.save()  # writable shared journal is required for every admission cycle

    def progress(self, snapshot, infrastructure_error=None):
        counts = Counter(j['status'] for j in self.jobs)
        terminal = sum(counts[s] for s in TERMINAL)
        elapsed = time.time()-self.started
        processed = terminal-self.initial_terminal
        pending = len(self.jobs)-terminal
        detail = [{'run': x['job']['spec'].name(), 'gpu': x['job']['gpu'], 'action': x['action'],
                   'pid': x['record']['pid'], 'warning': x.get('warning'),
                   'detail': safe_progress(Path(x['job']['out'])/'progress.json')} for x in self.active.values()]
        phase = ('paused_infrastructure' if infrastructure_error else
                 'waiting_gpu_telemetry' if not snapshot['rows'] else 'production_continue_on_error')
        if not self.active and pending and infrastructure_error is None and snapshot['rows']:
            phase = 'waiting_gpu_memory'
        if not self.active and not pending:
            phase = 'queue_finished_with_failures' if counts['failed_deferred'] else 'queue_finished'
        result = {'phase': phase, 'completed': counts['complete'], 'failed_deferred': counts['failed_deferred'],
                  'processed': terminal, 'total': len(self.jobs), 'remaining': pending,
                  'status_counts': dict(counts), 'active_phases': detail, 'gpu_state': snapshot,
                  'infrastructure_error': infrastructure_error, 'updated_at': r.c.now_text(),
                  'elapsed_sec': elapsed, 'max_concurrent': 8, 'workers_per_gpu': 2,
                  'eta_sec_observed_processed_throughput': elapsed/processed*pending if processed >= 4 else None,
                  'eta_scope': 'optimization plus independent audits; final report not included',
                  'jobs': [serialized(j) for j in self.jobs]}
        r.atomic_json(self.root/'progress.json', result)
        return result

    def run(self):
        while True:
            error = None
            try:
                self.poll()
                self.infrastructure()
            except Exception:
                error = traceback.format_exc()
                print(json.dumps({'infrastructure_pause': error, 'at': r.c.now_text()}), flush=True)
            snapshot = gpu_snapshot()
            if error is None:
                try:
                    for _ in range(2):
                        # At the retry tail, highest free memory wins before case affinity.
                        for gpu in sorted(GPUS, key=lambda g: snapshot['rows'].get(g, {}).get('memory_free_mib', 0), reverse=True):
                            free = snapshot['rows'].get(gpu, {}).get('memory_free_mib', 0)
                            if not capacity(gpu, self.active, free, self.required):
                                continue
                            job = choose_job(self.jobs, gpu, self.affinity)
                            if job:
                                self.launch(job, gpu, snapshot)
                except Exception:
                    error = traceback.format_exc()
                    print(json.dumps({'infrastructure_pause': error}), flush=True)
            try:
                with deadline():
                    progress = self.progress(snapshot, error)
            except Exception:
                print(json.dumps({'progress_write_failure': traceback.format_exc()}), flush=True)
                progress = None
            if not self.active and all(j['status'] in TERMINAL for j in self.jobs) and error is None:
                self.save()
                r.atomic_json(self.folder/'QUEUE_FINISHED.json', {'at': r.c.now_text(),
                              'accepted': sum(j['status'] == 'complete' for j in self.jobs),
                              'failed_deferred': sum(j['status'] == 'failed_deferred' for j in self.jobs),
                              'scientific_acceptance': all(j['status'] == 'complete' for j in self.jobs)})
                return
            time.sleep(5)


def inventory(root, folder):
    """All preexisting attempt bytes, not just already accepted summaries, are read-only."""
    path = folder/'preserved_files.json'
    if path.exists():
        return r.sha(path)
    if (root/'progress.json').exists():
        shutil.copy2(root/'progress.json', folder/'progress_before_resume.json')
    files = {}
    for candidate in root.glob('rank*'):
        for p in candidate.rglob('*'):
            if p.is_file():
                files[str(p)] = r.sha(p)
    for sub in ('parallel8_v1', 'recovery_v1'):
        for p in (root/sub).glob('*'):
            if p.is_file():
                files[str(p)] = r.sha(p)
    if (folder/'progress_before_resume.json').exists():
        files[str(folder/'progress_before_resume.json')] = r.sha(folder/'progress_before_resume.json')
    r.atomic_json(path, files)
    return r.sha(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', type=Path, default=Path('/database/inrorspline'))
    p.add_argument('--gpus', choices=('0,1,2,3',), default='0,1,2,3')
    p.add_argument('--max-concurrent', type=int, choices=(8,), default=8)
    p.add_argument('--workers-per-gpu', type=int, choices=(2,), default=2)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args(argv)
    if args.artifact_root.resolve().is_relative_to(r.c.REPO_ROOT):
        p.error('External artifacts only')
    return args


def main(args):
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError('Require UID:GID 1009:1020')
    root = args.artifact_root/'runs'/r.s.VARIANT
    folder = root/FOLDER
    folder.mkdir(exist_ok=True)
    # Every previous sweep controller uses this exact lock. Never signal unrelated processes.
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        live = experiment_processes(root)
        if any(x['action'] == 'launch' for x in live):
            raise RuntimeError('Old controller still alive; no takeover needed/authorized here')
        parent_path = root/'parallel8_v1/launch_manifest.json'
        manifest_path = folder/'launch_manifest.json'
        manifest = r.c.read_json(manifest_path if manifest_path.exists() else parent_path)
        r.assert_manifest(manifest)
        r.verify_preserved(root, manifest)
        if not manifest_path.exists():
            if live:
                raise RuntimeError('Initial preservation requires drained queue; subsequent restarts adopt active children')
            digest = inventory(root, folder)
            manifest = {**manifest, 'execution': {'policy': POLICY, 'source_hashes': sources(),
                        'parent_manifest_sha256': r.sha(parent_path), 'preserved_inventory_sha256': digest,
                        'created_at': r.c.now_text(), 'initial_gpu_state': gpu_snapshot(),
                        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=r.c.REPO_ROOT, text=True).strip()}}
            r.atomic_json(manifest_path, manifest)
        elif manifest['execution']['source_hashes'] != sources() or manifest['execution']['policy'] != POLICY:
            raise RuntimeError('Resume orchestration source/policy drift')
        q = Queue(root, manifest, max(8192, r.c.read_json(root/'recovery_v1/smoke_passed.json')['required_free_mib']))
        q.reconcile(live)
        q.infrastructure()
        print(json.dumps({'preflight': dict(Counter(j['status'] for j in q.jobs)),
                          'active': len(q.active), 'gpu_state': gpu_snapshot()}), flush=True)
        if args.preflight_only:
            return
        q.run()
        # Report is a separate process: reporting failure cannot retroactively stop registrations.
        log_path = folder/f'report_{time.time_ns()}.log'
        report_code = None
        try:
            with log_path.open('x') as log:
                result = subprocess.run([sys.executable, '-u', '-B', str(REPORT), '--artifact-root', str(args.artifact_root)],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=3600)
            report_code = result.returncode
            if report_code:
                raise RuntimeError(f'Report process exit {report_code}')
        except Exception:
            r.atomic_json(folder/'REPORT_FAILED.json', {'returncode': report_code, 'log': str(log_path),
                          'traceback': traceback.format_exc()})
        for path, digest in r.c.read_json(folder/'preserved_files.json').items():
            if r.sha(path) != digest:
                raise RuntimeError(f'Preserved artifact changed: {path}')
        r.atomic_json(folder/'PRESERVATION_VERIFIED.json', {'at': r.c.now_text()})
        if all(j['status'] == 'complete' for j in q.jobs) and report_code == 0:
            r.report(args.artifact_root)


if __name__ == '__main__':
    main(parse_args())
