"""Queue immutable SITReg best-checkpoint evaluation after each training exit."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import sitreg_training as s
from run_oasis_sitreg_first100_eval import evaluation_sources, training_ready


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--smoke-dir', type=Path, required=True)
    a = p.parse_args()
    if not (a.smoke_dir / 'PASSED').is_file():
        raise RuntimeError('Require successful full-geometry reused/fresh smoke')
    smoke = json.loads((a.smoke_dir / 'summary.json').read_text())
    source_hashes = evaluation_sources()
    if smoke['source_sha256'] != source_hashes:
        raise RuntimeError('Source changed since smoke')
    required_gib = max(24, math.ceil(smoke['peak_reserved_gib'] + 3))
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root / 'LAUNCH_CLAIMED').write_text(str(os.getpid()) + '\n')
    launch = json.loads((a.training_root / 'launch.json').read_text())
    jobs = [{'lambda_key': j['lambda_key'], 'gpu': j['gpu'], 'training_dir': j['output'],
             'out': str(a.out_root / f'sitreg_lambda_{j["lambda_key"]}_seed42'),
             'status': 'waiting_for_training', 'pid': None, 'returncode': None} for j in launch['jobs']]
    if [(j['gpu'], j['lambda_key']) for j in jobs] != list(enumerate(s.LAMBDAS)):
        raise ValueError('Unexpected training assignment')
    state = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
             'training_root': str(a.training_root), 'smoke_dir': str(a.smoke_dir),
             'source_sha256': source_hashes, 'required_free_gib': required_gib, 'jobs': jobs}
    active = {}
    terminal = {'complete', 'failed', 'training_failed'}
    while not all(j['status'] in terminal for j in jobs):
        s.verify_hashes(source_hashes)
        training = json.loads((a.training_root / 'launch.json').read_text())
        parents = {j['lambda_key']: j for j in training['jobs']}
        for j in jobs:
            if j['status'] in terminal:
                continue
            if j['lambda_key'] in active:
                child, log = active[j['lambda_key']]
                code = child.poll()
                if code is None:
                    progress_path = Path(j['out']) / 'progress.json'
                    if progress_path.exists():
                        j['progress'] = json.loads(progress_path.read_text())
                    continue
                j['returncode'] = code
                j['status'] = 'complete' if code == 0 and (Path(j['out']) / 'COMPLETED').is_file() else 'failed'
                j['finished_unix'] = time.time()
                log.close()
                del active[j['lambda_key']]
                continue
            ready, reason = training_ready(j['training_dir'])
            parent_exit = parents[j['lambda_key']]['returncode']
            if reason == 'training_failed' or parent_exit not in (None, 0):
                j['status'] = 'training_failed'
                continue
            if not ready or parent_exit is None:
                j['status'] = 'waiting_for_training'
                continue
            row = next(r for r in s.gpu_rows() if r['index'] == j['gpu'])
            if row['free_mib'] < required_gib * 1024:
                j['status'] = 'waiting_for_gpu_memory'
                j['gpu_snapshot'] = row
                continue
            command = [sys.executable, '-B', str(Path(__file__).with_name('run_oasis_sitreg_first100_eval.py')),
                       '--training-dir', j['training_dir'], '--out', j['out'], '--gpu', str(j['gpu']),
                       '--min-free-gib', str(required_gib)]
            log = (a.out_root / f'lambda_{j["lambda_key"]}.log').open('x')
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            active[j['lambda_key']] = child, log
            j.update(status='evaluating', pid=child.pid, command=command, started_unix=time.time())
        state['updated_unix'] = time.time()
        s.write_json(a.out_root / 'queue.json', state)
        if not all(j['status'] in terminal for j in jobs):
            time.sleep(30)
    state['status'] = 'complete' if all(j['status'] == 'complete' for j in jobs) else 'failures_preserved'
    state['finished_unix'] = time.time()
    s.write_json(a.out_root / 'queue.json', state)
    if state['status'] == 'complete':
        summaries = {j['lambda_key']: json.loads((Path(j['out']) / 'summary.json').read_text()) for j in jobs}
        s.write_json(a.out_root / 'all_four_summary.json', summaries)
        (a.out_root / 'COMPLETED').write_text('400 case evaluations;14000 per-label rows\n')


if __name__ == '__main__':
    main()
