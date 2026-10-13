"""Approved FeDDA-1MF queue, independent of active VFA/SITReg controllers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import fedda_training as s

ASSIGNMENTS = ((0, '3'), (0, '10'), (1, '0.1'), (2, '1over3'))


def run_tag(key):
    if key not in s.LAMBDAS:
        raise ValueError('Unapproved lambda')
    return f'fedda_1mf_lambda_{key}_seed42_attempt01'


def clean_evaluation(job):
    if job['status'] != 'complete' or not job.get('eval_dir') or not job.get('train_dir'):
        return False
    attempts = job.get('attempts', [])
    return bool(attempts and attempts[-1]['stage'] == 'eval' and attempts[-1]['returncode'] == 0
                and (Path(job['eval_dir']) / 'COMPLETED').is_file()
                and (Path(job['train_dir']) / 'COMPLETED').is_file())


def dependency_state(job, jobs, vfa_queue):
    key = job['lambda_key']
    if key == '3':
        return 'ready'
    if key == '10':
        dependency = next(j for j in jobs if j['lambda_key'] == '3')
    else:
        dependency = next(j for j in vfa_queue['jobs'] if j['lambda_key'] == key)
    if dependency['status'] == 'failed':
        return 'dependency_failed'
    return 'ready' if clean_evaluation(dependency) else 'waiting_for_dependency'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--report-root', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--vfa-root', type=Path, required=True)
    a = p.parse_args()
    gate = json.loads((a.preflight / 'preflight.json').read_text())
    if not (a.preflight / 'PASSED').exists() or gate['status'] != 'passed':
        raise RuntimeError('Require FeDDA reference and memory preflight')
    s.verify_hashes(gate['source_sha256'])
    native = json.loads((a.preflight / 'manifest.json').read_text())['arguments']
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root / 'LAUNCH_CLAIMED').write_text(str(os.getpid()) + '\n')
    jobs = [{'gpu': gpu, 'lambda_key': key, 'status': 'pending', 'next_stage': 'train',
             'pid': None, 'train_dir': None, 'eval_dir': None, 'attempts': []}
            for gpu, key in ASSIGNMENTS]
    state = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
             'jobs': jobs, 'preflight': str(a.preflight), 'vfa_root': str(a.vfa_root),
             'report_root': str(a.report_root), 'source_sha256': gate['source_sha256'],
             'min_train_free_gib': gate['min_train_free_gib'], 'retries': 'none',
             'container_image_id': os.environ.get('INRORSPLINE_CONTAINER_IMAGE_ID')}
    active = {}
    terminal = {'complete', 'failed'}
    try:
        while not all(j['status'] in terminal for j in jobs):
            s.verify_hashes(gate['source_sha256'])
            vfa = json.loads((a.vfa_root / 'queue.json').read_text())
            for job in jobs:
                key = job['lambda_key']
                if job['status'] in terminal:
                    continue
                if key in active:
                    child, log, stage, output = active[key]
                    code = child.poll()
                    if code is None:
                        progress = output / 'progress.json'
                        if progress.exists():
                            job['progress'] = json.loads(progress.read_text())
                        continue
                    log.close()
                    del active[key]
                    job['pid'] = None
                    job['attempts'][-1].update(returncode=code, finished_unix=time.time())
                    if code == 0 and (output / 'COMPLETED').exists():
                        job.pop('progress', None)
                        if stage == 'train':
                            job.update(next_stage='eval', status='pending_eval')
                        else:
                            job.update(status='complete', finished_unix=time.time())
                    else:
                        job.update(status='failed', failure_stage=stage, returncode=code)
                    continue
                dep = dependency_state(job, jobs, vfa)
                if dep != 'ready':
                    job['status'] = 'failed' if dep == 'dependency_failed' else dep
                    if dep == 'dependency_failed':
                        job['failure_stage'] = dep
                    continue
                if any(j['gpu'] == job['gpu'] and j['lambda_key'] in active for j in jobs):
                    job['status'] = 'waiting_for_gpu'
                    continue
                # Never overlap a currently active P50 worker, even if its mapping changes.
                if any(j['gpu'] == job['gpu'] and j['status'] in ('training', 'evaluating')
                       for j in vfa['jobs']):
                    job['status'] = 'waiting_for_vfa_gpu'
                    continue
                stage = job['next_stage']
                device = next(r for r in s.gpu_rows() if r['index'] == job['gpu'])
                required = gate['min_train_free_gib'] if stage == 'train' else 24
                job['gpu_snapshot'] = device
                if device['free_mib'] < required * 1024:
                    job['status'] = 'waiting_for_memory'
                    continue
                tag = run_tag(key)
                if stage == 'train':
                    output = a.out_root / tag
                    job['train_dir'] = str(output)
                    command = [sys.executable, '-B', str(Path(__file__).with_name('run_oasis_fedda_training.py')),
                               '--mode', 'train', '--lambda-key', key, '--gpu', str(job['gpu']),
                               '--preflight', str(a.preflight), '--out', str(output),
                               '--min-free-gib', str(required)]
                    for arg in ('wavereg_root', 'data_root', 'reference_checkpoint', 'reference_field'):
                        command.extend(['--' + arg.replace('_', '-'), native[arg]])
                else:
                    output = a.out_root / 'first100' / tag
                    job['eval_dir'] = str(output)
                    command = [sys.executable, '-B', str(Path(__file__).with_name('run_oasis_fedda_first100_eval.py')),
                               '--training-dir', job['train_dir'], '--out', str(output), '--gpu', str(job['gpu'])]
                log = (a.out_root / f'{tag}_{stage}.log').open('x')
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                active[key] = child, log, stage, output
                job.update(status='training' if stage == 'train' else 'evaluating', pid=child.pid)
                job['attempts'].append({'stage': stage, 'gpu': job['gpu'], 'pid': child.pid,
                                        'out': str(output), 'command': command, 'returncode': None,
                                        'started_unix': time.time(), 'gpu_at_launch': device})
            state['updated_unix'] = time.time()
            s.write_json(a.out_root / 'queue.json', state)
            if not all(j['status'] in terminal for j in jobs):
                time.sleep(30)
        state['status'] = 'reporting' if all(j['status'] == 'complete' for j in jobs) else 'failures_preserved'
        s.write_json(a.out_root / 'queue.json', state)
        if state['status'] == 'reporting':
            command = [sys.executable, '-B', str(Path(__file__).with_name('report_oasis_fedda_training.py')),
                       '--training-root', str(a.out_root), '--out', str(a.report_root),
                       '--data-root', native['data_root'],
                       '--reference-root', str(Path(native['reference_checkpoint']).parents[1])]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ''
            with (a.out_root / 'report.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
            state['status'] = 'complete'
            (a.out_root / 'COMPLETED').write_text('4 trainings,400 cases,14000 label rows,and comparison report\n')
        state['finished_unix'] = time.time()
        s.write_json(a.out_root / 'queue.json', state)
    except BaseException:
        state['status'] = 'controller_failed'
        s.write_json(a.out_root / 'queue.json', state)
        (a.out_root / 'CONTROLLER_FAILED.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
