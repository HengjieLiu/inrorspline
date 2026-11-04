"""Four independent GNCC² trainings, per-GPU evaluation, then strict comparison."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import sitreg_gncc2_training as g


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get('ARTIFACT_ROOT', '/database/inrorspline'))
    p.add_argument('--out-root', type=Path, default=root/'runs'/g.VARIANT)
    p.add_argument('--eval-root', type=Path, default=root/'runs'/g.EVAL_VARIANT)
    p.add_argument('--review-root', type=Path, default=root/'repo_experiments'/g.VARIANT)
    p.add_argument('--baseline-training-root', type=Path, default=root/'runs/oasis_sitreg_ncc9_diffusion_training_lambda_sweep_v1')
    p.add_argument('--baseline-report', type=Path, default=root/'repo_experiments/oasis_seed3_first100_mrdbscp_vfa_sitreg_sweeps_comparison_v1')
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--max-concurrent', type=int, default=4)
    a = p.parse_args(argv)
    if a.gpus != '0,1,2,3' or a.max_concurrent != 4:
        p.error('Approved assignment is exactly four workers on GPUs0,1,2,3')
    outputs = [a.out_root.resolve(), a.eval_root.resolve(), a.review_root.resolve()]
    inputs = [a.baseline_training_root.resolve(), a.baseline_report.resolve(), Path(__file__).resolve().parents[3]]
    if any(x.is_relative_to(y) or y.is_relative_to(x) for x in outputs for y in inputs):
        p.error('Output overlaps protected input')
    if any(x.is_relative_to(y) or y.is_relative_to(x) for i, x in enumerate(outputs) for y in outputs[i+1:]):
        p.error('Outputs overlap')
    return a


def baseline_hashes(root):
    accepted = json.loads((root/'ACCEPTED.json').read_text())
    if not accepted['complete'] or accepted['sitreg_label_rows_validated'] != 14000:
        raise RuntimeError('Require accepted historical SITReg comparison')
    hashes = {str(root/name): digest for name, digest in accepted['output_sha256'].items()}
    hashes[str(root/'ACCEPTED.json')] = g.sha256(root/'ACCEPTED.json')
    result = json.loads((root/'results.json').read_text())
    hashes.update({r['checkpoint']: r['checkpoint_sha256'] for r in result['checkpoints'].values()})
    g.verify_hashes(hashes)
    return hashes


def execute(a):
    g.require_identity()
    if not (a.preflight/'GNCC2_PASSED').is_file():
        raise RuntimeError('Extended GNCC² smoke gate required')
    gate = json.loads((a.preflight/'preflight.json').read_text())
    g.require_protocol(gate)
    if gate['gncc2_gate'] != 'passed':
        raise RuntimeError('Failed GNCC² gate')
    g.verify_hashes(gate['source_sha256']); g.verify_hashes(gate['baseline_sources_sha256'])
    refs = baseline_hashes(a.baseline_report)
    old = json.loads((a.baseline_training_root/'sitreg_lambda_0.1_seed42/manifest.json').read_text())['arguments']
    required_gib = max(37, math.ceil(gate['peak_reserved_gib']+1))
    a.out_root.mkdir(parents=True, exist_ok=True)
    a.eval_root.mkdir(parents=True, exist_ok=False)
    a.review_root.mkdir(parents=True, exist_ok=False)
    state_path = a.out_root/'launch.json'
    if state_path.exists():
        raise FileExistsError('Preserve existing launch; no implicit restart')
    jobs = [dict(lambda_key=key, alpha=alpha, gpu=gpu, status='waiting_for_training_memory',
                 train_dir=str(a.out_root/g.run_name(key)), eval_dir=str(a.eval_root/g.run_name(key)))
            for gpu, (key, alpha) in enumerate(g.LAMBDAS.items())]
    state = dict(status='running', pid=os.getpid(), started_unix=time.time(), jobs=jobs,
                 protocol=g.PROTOCOL, required_train_free_gib=required_gib,
                 preflight=str(a.preflight), preflight_sha256=g.sha256(a.preflight/'preflight.json'),
                 source_sha256=gate['source_sha256'], reference_sha256=refs,
                 container_image_id=os.environ.get('INRORSPLINE_CONTAINER_IMAGE_ID'))
    active = {}
    terminal = {'complete', 'train_failed', 'eval_failed'}

    def save():
        state['updated_unix'] = time.time(); g.write_json(state_path, state)

    save()
    while not all(j['status'] in terminal for j in jobs):
        g.verify_hashes(gate['source_sha256'])
        for j in jobs:
            key = j['lambda_key']
            if j['status'] in terminal:
                continue
            if key in active:
                process, handle, phase = active[key]
                code = process.poll()
                progress = Path(j['train_dir'] if phase == 'train' else j['eval_dir'])/'progress.json'
                if progress.exists():
                    j['progress'] = json.loads(progress.read_text())
                if code is None:
                    continue
                handle.close(); del active[key]
                path = Path(j['train_dir'] if phase == 'train' else j['eval_dir'])
                complete = code == 0 and (path/'COMPLETED').is_file() and not (path/'FAILED.txt').exists()
                j[phase+'_exit'] = code; j[phase+'_finished_unix'] = time.time()
                j['status'] = ('waiting_for_evaluation_memory' if phase == 'train' else 'complete') if complete else phase+'_failed'
                print(json.dumps({'event': j['status'], 'lambda_key': key, 'gpu': j['gpu']}), flush=True)
                save()
                if j['status'] in terminal:
                    continue
            phase = 'train' if j['status'] == 'waiting_for_training_memory' else 'eval'
            memory = next(r for r in g.gpu_rows() if r['index'] == j['gpu'])
            if memory['free_mib'] < (required_gib if phase == 'train' else 24)*1024:
                j['waiting_gpu_snapshot'] = memory
                continue
            here = Path(__file__).resolve().parent
            if phase == 'train':
                command = [sys.executable, '-B', str(here/'frozen_oasis_entry.py'), str(here/'run_oasis_sitreg_gncc2_training.py'), '--mode', 'train',
                           '--lambda-key', key, '--gpu', str(j['gpu']), '--out', j['train_dir'],
                           '--preflight', str(a.preflight), '--baseline-preflight', str(a.baseline_training_root/'preflight_attempt2')]
                for name in ('wavereg_root', 'data_root', 'reference_checkpoint', 'reference_field'):
                    command.extend(['--'+name.replace('_', '-'), old[name]])
            else:
                command = [sys.executable, '-B', str(here/'frozen_oasis_entry.py'), str(here/'run_oasis_sitreg_gncc2_eval.py'), '--training-dir', j['train_dir'],
                           '--out', j['eval_dir'], '--gpu', str(j['gpu']), '--min-free-gib', '24']
            log = a.out_root/f'{phase}_lambda_{key}.log'
            handle = log.open('x')
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
            active[key] = process, handle, phase
            j.update(status='training' if phase == 'train' else 'evaluating', pid=process.pid)
            j[phase+'_started_unix'] = time.time(); j[phase+'_command'] = command; j[phase+'_gpu_start'] = memory
            print(json.dumps({'event': j['status'], 'lambda_key': key, 'gpu': j['gpu'], 'pid': process.pid}), flush=True)
            save()
        save()
        if not all(j['status'] in terminal for j in jobs):
            time.sleep(15)
    if not all(j['status'] == 'complete' for j in jobs):
        state['status'] = 'failures_preserved'; save(); return
    state['status'] = 'reporting'; save()
    g.verify_hashes(refs)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'reports'))
    from oasis_sitreg_gncc2_report import write_report
    write_report(a.out_root, a.eval_root, a.baseline_report, a.review_root, refs)
    state['status'] = 'complete'; state['finished_unix'] = time.time(); save()
    (a.out_root/'COMPLETED').write_text('Four trainings and400 detailed case evaluations plus comparison complete\n')
    print(json.dumps({'event': 'complete', 'report': str(a.review_root/'report.md')}), flush=True)


if __name__ == '__main__':
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root/'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (args.out_root/f'launcher_{time.time_ns()}.log').open('x', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    execute(args)
                except BaseException:
                    traceback.print_exc(); raise

