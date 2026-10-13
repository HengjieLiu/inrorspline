"""Per-GPU SITReg-evaluation -> VFA training -> detailed first100 queue."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import vfa_training as v

ASSIGNMENTS = ((0, '10'), (1, '0.1'), (2, '1over3'), (3, '3'))


def is_cuda_oom(output):
    p = Path(output) / 'progress.json'
    if p.exists() and json.loads(p.read_text()).get('failure_kind') == 'cuda_oom':
        return True
    p = Path(output) / 'FAILED.txt'
    return p.exists() and 'out of memory' in p.read_text().lower() and 'cuda' in p.read_text().lower()


def retry_allowed(job, *, oom):
    return bool(oom and job['lambda_key'] == '10' and job['gpu'] == 0 and not job['retried_on_gpu3'])


def sitreg_finished(queue, gpu):
    job = next(j for j in queue['jobs'] if j['gpu'] == gpu)
    return (job['status'] == 'complete' and job['returncode'] == 0
            and (Path(job['out']) / 'COMPLETED').is_file())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--sitreg-eval-root', type=Path, required=True)
    a = p.parse_args()
    gate = json.loads((a.preflight / 'preflight.json').read_text())
    if not (a.preflight / 'PASSED').exists() or gate['status'] != 'passed':
        raise RuntimeError('Require native VFA parity/update/serialization gate')
    v.verify_hashes(gate['source_sha256'])
    native_args = json.loads((a.preflight / 'manifest.json').read_text())['arguments']
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root / 'LAUNCH_CLAIMED').write_text(str(os.getpid()) + '\n')
    jobs = [{'lambda_key': key, 'gpu': gpu, 'initial_gpu': gpu, 'status': 'pending',
             'next_stage': 'train', 'retried_on_gpu3': False, 'attempt': 1, 'pid': None,
             'train_dir': None, 'eval_dir': None, 'attempts': []} for gpu, key in ASSIGNMENTS]
    state = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
             'preflight': str(a.preflight), 'sitreg_eval_root': str(a.sitreg_eval_root),
             'jobs': jobs, 'source_sha256': gate['source_sha256'],
             'gpu0_memory_override': 'explicit user approval; lambda10 OOM gets one GPU3 retry',
             'retry_order': 'after GPU3 lambda3 training and first100 evaluation finish',
             'container_image_id': os.environ.get('INRORSPLINE_CONTAINER_IMAGE_ID')}
    active = {}
    terminal = {'complete', 'failed'}
    try:
        while not all(j['status'] in terminal for j in jobs):
            v.verify_hashes(gate['source_sha256'])
            sq = json.loads((a.sitreg_eval_root / 'queue.json').read_text())
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
                    attempt = job['attempts'][-1]
                    attempt.update(returncode=code, finished_unix=time.time())
                    job['pid'] = None
                    if code == 0 and (output / 'COMPLETED').exists():
                        if stage == 'train':
                            job.update(next_stage='eval', status='pending_eval')
                        else:
                            job.update(status='complete', finished_unix=time.time())
                        continue
                    oom = is_cuda_oom(output)
                    attempt['cuda_oom'] = oom
                    if retry_allowed(job, oom=oom):
                        job.update(gpu=3, attempt=2, retried_on_gpu3=True,
                                   status='waiting_for_gpu3_retry', next_stage=stage)
                        if stage == 'train':
                            job['train_dir'] = None
                    else:
                        job.update(status='failed', failure_stage=stage, returncode=code)
                    continue
                if job['retried_on_gpu3']:
                    gpu3_primary = next(j for j in jobs if j['lambda_key'] == '3')
                    if gpu3_primary['status'] not in terminal:
                        job['status'] = 'waiting_for_gpu3_retry'
                        continue
                if not sitreg_finished(sq, job['gpu']):
                    old = next(j for j in sq['jobs'] if j['gpu'] == job['gpu'])
                    job['status'] = 'waiting_for_sitreg_eval'
                    if old['status'] in ('failed', 'training_failed'):
                        job.update(status='failed', failure_stage='sitreg_dependency')
                    continue
                if any(j['gpu'] == job['gpu'] and j['lambda_key'] in active for j in jobs):
                    job['status'] = 'waiting_for_gpu'
                    continue
                stage = job['next_stage']
                gpu_row = next(r for r in v.gpu_rows() if r['index'] == job['gpu'])
                tight = stage == 'train' and key == '10' and job['gpu'] == 0 and job['attempt'] == 1
                required = v.MIN_TRAIN_FREE_GIB if stage == 'train' else 24
                job['gpu_snapshot'] = gpu_row
                if not tight and gpu_row['free_mib'] < required * 1024:
                    job['status'] = 'waiting_for_memory'
                    continue
                tag = f'vfa_lambda_{key}_seed42_attempt{job["attempt"]:02d}'
                if stage == 'train':
                    output = a.out_root / tag
                    job['train_dir'] = str(output)
                    command = [sys.executable, '-B', str(Path(__file__).with_name('run_oasis_vfa_training.py')),
                               '--mode', 'train', '--lambda-key', key, '--gpu', str(job['gpu']),
                               '--preflight', str(a.preflight), '--out', str(output)]
                    for arg in ('wavereg_root', 'data_root', 'reference_checkpoint', 'reference_field'):
                        command.extend(['--' + arg.replace('_', '-'), native_args[arg]])
                    if tight:
                        command.append('--allow-tight-memory')
                else:
                    output = a.out_root / 'first100' / tag
                    job['eval_dir'] = str(output)
                    command = [sys.executable, '-B', str(Path(__file__).with_name('run_oasis_vfa_first100_eval.py')),
                               '--training-dir', job['train_dir'], '--out', str(output), '--gpu', str(job['gpu'])]
                log = (a.out_root / f'{tag}_{stage}.log').open('x')
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                active[key] = child, log, stage, output
                job.update(status='training' if stage == 'train' else 'evaluating', pid=child.pid)
                job['attempts'].append({'stage': stage, 'gpu': job['gpu'], 'attempt': job['attempt'],
                                        'pid': child.pid, 'out': str(output), 'command': command,
                                        'started_unix': time.time(), 'returncode': None,
                                        'gpu_at_launch': gpu_row, 'tight_memory_override': tight})
            state['updated_unix'] = time.time()
            v.write_json(a.out_root / 'queue.json', state)
            if not all(j['status'] in terminal for j in jobs):
                time.sleep(30)
        state['status'] = 'complete' if all(j['status'] == 'complete' for j in jobs) else 'failures_preserved'
        state['finished_unix'] = time.time()
        v.write_json(a.out_root / 'queue.json', state)
        if state['status'] == 'complete':
            summaries = {j['lambda_key']: json.loads((Path(j['eval_dir']) / 'summary.json').read_text()) for j in jobs}
            v.write_json(a.out_root / 'all_four_first100_summary.json', summaries)
            (a.out_root / 'COMPLETED').write_text('4 trainings,400 cases,14000 label rows\n')
    except BaseException:
        (a.out_root / 'CONTROLLER_FAILED.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
