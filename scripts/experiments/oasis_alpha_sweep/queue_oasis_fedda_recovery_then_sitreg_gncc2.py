"""Approved persistent all-four FeDDA barrier, then SITReg GNCC² train/eval."""
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

import fedda_gncc2_training as f
import sitreg_gncc2_training as s

ROOT = Path(os.environ.get('ARTIFACT_ROOT', '/database/inrorspline'))
HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def command(script, *arguments):
    return [sys.executable, '-B', str(HERE/'frozen_oasis_entry.py'), str(HERE/script), *map(str, arguments)]


def require_fedda_barrier(train_root, eval_root):
    if not (train_root/'COMPLETED').is_file():
        raise RuntimeError('All-four FeDDA report/completion barrier not met')
    for key in f.LAMBDAS:
        train, evaluated = train_root/f.run_name(key), eval_root/f.run_name(key)
        for directory in (train, evaluated):
            if not (directory/'COMPLETED').is_file() or (directory/'FAILED.txt').exists():
                raise RuntimeError('FeDDA train/eval barrier not met for '+key)
        record = read(train/'training_summary.json')
        summary = read(evaluated/'summary.json')
        if record['epoch'] != 200 or record['adam_updates'] != 20000 or summary['case_count'] != 100 or summary['label_rows'] != 3500:
            raise RuntimeError('Incomplete FeDDA training/evaluation budget')
        if sorted(r['rank'] for r in read(evaluated/'per_case.json')) != list(range(100)):
            raise RuntimeError('Wrong first100 cohort')


def execute(out):
    f.require_identity()
    gate = ROOT/'runs/oasis_fedda_gncc2_training_preflight_v1'
    failed = ROOT/'runs'/f.VARIANT
    recovered = ROOT/'runs'/(f.VARIANT+'_recovery_v1')
    evaluated = ROOT/'runs'/(f.EVAL_VARIANT+'_recovery_v1')
    sitreg_gate = ROOT/'runs/oasis_sitreg_gncc2_training_preflight_v1'
    state = dict(status='validating_recovery', pid=os.getpid(), started_unix=time.time(),
                 source_snapshot=os.environ['OASIS_SOURCE_SNAPSHOT'],
                 fedda=dict(resume_root=str(failed), train_root=str(recovered), eval_root=str(evaluated)),
                 sitreg=dict(status='queued', dependency='all_four_fedda_train_eval_and_report_complete',
                             train_root=str(ROOT/'runs'/s.VARIANT), eval_root=str(ROOT/'runs'/s.EVAL_VARIANT),
                             preflight=str(sitreg_gate)),
                 mapping=[dict(gpu=i, lambda_key=k, alpha=a) for i, (k, a) in enumerate(f.LAMBDAS.items())])
    def save():
        state['updated_unix'] = time.time()
        f.write_json(out/'queue.json', state)
    def run_stage(name, cmd, completed):
        state['status'] = name
        log = out/(name+'.log')
        with log.open('x') as handle:
            worker = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
            state[name] = dict(pid=worker.pid, command=cmd, log=str(log), started_unix=time.time())
            save()
            while worker.poll() is None:
                save(); time.sleep(15)
            state[name].update(exit_code=worker.returncode, finished_unix=time.time())
            save()
            if worker.returncode or not completed.is_file():
                raise RuntimeError(f'{name} failed; preserve {log}; subsequent stages blocked')
    save()
    # Explicit independent recovery checks on all four saved checkpoints before
    # any resumed optimization. Each constructs the native model from seed42,
    # replays the iterator, and checks optimizer/RNG state without an update.
    handles, active = [], []
    try:
        for gpu, key in enumerate(f.LAMBDAS):
            original = failed/f.run_name(key)
            old = read(original/'manifest.json')['arguments']
            target = out/f'validate_lambda_{key}'
            cmd = command('resume_oasis_fedda_gncc2.py', '--mode', 'train', '--validate-only',
                          '--resume-from', original, '--lambda-key', key, '--gpu', gpu,
                          '--out', target, '--preflight', gate, '--min-free-gib', '24')
            for name in ('wavereg_root', 'data_root', 'reference_checkpoint', 'reference_field', 'baseline_preflight', 'baseline_training_root'):
                cmd.extend(['--'+name.replace('_', '-'), old[name]])
            handle = (out/f'validate_lambda_{key}.log').open('x')
            handles.append(handle)
            worker = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
            active.append((worker, target))
        while any(w.poll() is None for w, _ in active):
            save(); time.sleep(10)
        if any(w.returncode or not (p/'RECOVERY_PASSED').is_file() for w, p in active):
            raise RuntimeError('Recovery validation failed; no production started')
    finally:
        for h in handles:
            h.close()
    state['recovery_validation'] = 'all_four_passed'; save()
    run_stage('fedda_recovery', command('launch_oasis_fedda_gncc2_recovery.py', '--preflight', gate), recovered/'COMPLETED')
    require_fedda_barrier(recovered, evaluated)
    state['sitreg']['status'] = 'preflight_pending'; save()
    while next(r for r in s.gpu_rows() if r['index'] == 3)['free_mib'] < 38*1024:
        state['status'] = 'sitreg_preflight_waiting_for_gpu3_memory'; save(); time.sleep(15)
    baseline = ROOT/'runs/oasis_sitreg_ncc9_diffusion_training_lambda_sweep_v1'
    old = read(baseline/'sitreg_lambda_0.1_seed42/manifest.json')['arguments']
    cmd = command('run_oasis_sitreg_gncc2_training.py', '--mode', 'preflight', '--gpu', 3,
                  '--out', sitreg_gate, '--baseline-preflight', baseline/'preflight_attempt2')
    for name in ('wavereg_root', 'data_root', 'reference_checkpoint', 'reference_field'):
        cmd.extend(['--'+name.replace('_', '-'), old[name]])
    run_stage('sitreg_preflight', cmd, sitreg_gate/'GNCC2_PASSED')
    state['sitreg']['status'] = 'training_then_automatic_evaluation'; save()
    run_stage('sitreg_sweep', command('launch_oasis_sitreg_gncc2.py', '--preflight', sitreg_gate), ROOT/'runs'/s.VARIANT/'COMPLETED')
    state.update(status='complete', finished_unix=time.time())
    state['sitreg']['status'] = 'complete'; save()
    (out/'COMPLETED').write_text('Recovered FeDDA and fresh SITReg four-lambda train/eval pipelines completed\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    with (a.out/'queue.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (a.out/'controller.log').open('x', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    execute(a.out)
                except BaseException:
                    failure = traceback.format_exc()
                    (a.out/'FAILED.txt').write_text(failure)
                    path = a.out/'queue.json'
                    value = read(path) if path.exists() else {}
                    value.update(status='failed', failure=failure, updated_unix=time.time())
                    f.write_json(path, value)
                    traceback.print_exc()
                    raise


if __name__ == '__main__':
    main()
