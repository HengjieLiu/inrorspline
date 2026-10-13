"""Launch exactly four approved SITReg jobs, one per physical GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import sitreg_training as s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--wavereg-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--reference-checkpoint', type=Path, required=True)
    p.add_argument('--reference-field', type=Path, required=True)
    a = p.parse_args()
    gate = json.loads((a.preflight / 'preflight.json').read_text())
    assert (a.preflight / 'PASSED').exists() and gate['status'] == 'passed'
    s.verify_hashes(gate['source_sha256'])
    rows = s.gpu_rows()
    for index in range(4):
        s.memory_fraction(next(r for r in rows if r['index'] == index))
    # Claim this launch before spawning. Never overwrite a prior launch.
    a.root.mkdir(parents=True, exist_ok=True)
    with (a.root / 'LAUNCH_CLAIMED').open('x') as marker:
        marker.write(str(time.time()) + '\n')
    children, records = [], []
    for gpu, key in enumerate(s.LAMBDAS):
        output = a.root / f'sitreg_lambda_{key}_seed42'
        command = [sys.executable, '-B', str(Path(__file__).with_name('run_oasis_sitreg_training.py')),
                   '--mode', 'train', '--lambda-key', key, '--gpu', str(gpu),
                   '--wavereg-root', str(a.wavereg_root), '--data-root', str(a.data_root),
                   '--reference-checkpoint', str(a.reference_checkpoint),
                   '--reference-field', str(a.reference_field), '--preflight', str(a.preflight),
                   '--out', str(output)]
        log = (a.root / f'worker_gpu{gpu}.log').open('x')
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        children.append((child, log))
        records.append(dict(gpu=gpu, lambda_key=key, alpha=s.LAMBDAS[key], pid=child.pid,
                            output=str(output), command=command, returncode=None))
    state = dict(started=time.time(), pid=os.getpid(), gpu_before=rows, jobs=records,
                 controller_sha256=s.sha256(__file__),
                 container_image_id=os.environ.get('INRORSPLINE_CONTAINER_IMAGE_ID'),
                 gpu_processes_at_launch=subprocess.check_output([
                     'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                     '--format=csv'], text=True),
                 preflight=str(a.preflight), status='running')
    s.write_json(a.root / 'launch.json', state)
    print(json.dumps(state), flush=True)
    while any(child.poll() is None for child, _ in children):
        for record, (child, _) in zip(records, children):
            record['returncode'] = child.poll()
        state['updated'] = time.time()
        s.write_json(a.root / 'launch.json', state)
        time.sleep(30)
    for record, (child, log) in zip(records, children):
        record['returncode'] = child.returncode
        log.close()
    state['status'] = 'complete' if all(r['returncode'] == 0 for r in records) else 'failed_candidates_preserved'
    state['finished'] = time.time()
    s.write_json(a.root / 'launch.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
