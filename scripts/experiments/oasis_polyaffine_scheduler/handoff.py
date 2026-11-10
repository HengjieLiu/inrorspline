#!/usr/bin/env python
"""Validate and atomically hand the existing queue to the authorized scheduler.

Only the old controller is stopped. Worker processes and published outputs
remain in place. The original launch/scientific manifests are never rewritten.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

from support import (HERE, SCIENCE, atomic_json, read_json, sha, now,
                     code_hashes, scheduler_hashes, assert_external)
from dispatcher import gpu_snapshots
from controller import same_process, process_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--revision-dir', type=Path, required=True)
    args = parser.parse_args()
    root, folder = assert_external(args.root), assert_external(args.revision_dir)
    if (folder/'manifest.json').exists():
        raise FileExistsError('Revision already frozen; resume its dispatcher instead')
    manifest = read_json(root/'launch_manifest.json')
    if manifest['code_hashes'] != code_hashes():
        raise RuntimeError('Scientific code must remain byte-identical to the running launch')
    old_gate = read_json(root/'PREFLIGHT_PASSED.json')
    if not old_gate['passed'] or old_gate['launch_manifest_sha256'] != sha(root/'launch_manifest.json'):
        raise RuntimeError('Original validated launch binding failed')
    if sha(root/'input_manifest.json') != manifest['input_manifest_sha256']:
        raise RuntimeError('Original inputs changed')
    cpu_path = folder/'validation'/'cpu_tests.json'
    cpu = read_json(cpu_path)
    if not cpu['passed'] or cpu['exit_code'] != 0 or cpu['scheduler_source_hashes'] != scheduler_hashes() or cpu['scientific_source_hashes'] != code_hashes():
        raise RuntimeError('Final scheduler CPU validation is absent or stale')
    snapshots = gpu_snapshots()
    receipts = [dict(path=str(cpu_path), sha256=sha(cpu_path))]
    for gpu in (0, 1, 2, 3):
        smoke_path = folder/'gpu_smoke'/f'gpu{gpu}'/'receipt.json'
        smoke = read_json(smoke_path)
        if (not smoke['passed'] or smoke['physical_gpu'] != gpu or smoke['gpu_uuid'] != snapshots[gpu]['uuid']
            or smoke['scientific_source_hashes'] != code_hashes()
            or smoke['base_launch_sha256'] != sha(root/'launch_manifest.json')
            or smoke['input_manifest_sha256'] != sha(root/'input_manifest.json')
            or smoke['worker_entry_sha256'] != sha(HERE/'worker_entry.py')
            or smoke['verify_source_sha256'] != sha(HERE/'verify_gpu.py')):
            raise RuntimeError('GPU smoke failed or is stale for GPU '+str(gpu))
        receipts.append(dict(path=str(smoke_path), sha256=sha(smoke_path)))
    policy = dict(manifest['queue_policy'])
    policy.pop('max_jobs', None)
    policy.pop('gpu', None)
    policy.pop('admission_timeout_sec', None)
    policy.update(poll_sec=60, automatic_retries='cuda_oom_only',
                  oom_cooldown_initial_sec=60, oom_cooldown_max_sec=900,
                  external_occupancy='wait_and_poll_without_terminal_deferral',
                  failures='record_preserve_continue; CUDA OOM requeues with increased reservation')
    revision = dict(version='scheduler_v1', created_at=now(),
        authorization='User: GPU0/GPU1 max16 each; GPU2/GPU3 max4 each; shared queue with per-GPU memory tiers; 60s polling; automatic OOM requeue.',
        base_launch_sha256=sha(root/'launch_manifest.json'), input_manifest_sha256=sha(root/'input_manifest.json'),
        gpu_limits={'0': 16, '1': 16, '2': 4, '3': 4},
        gpu_uuids={str(g): snapshots[g]['uuid'] for g in (0,1,2,3)},
        gpu_total_mib={str(g): snapshots[g]['total_mib'] for g in (0,1,2,3)},
        gpu_models={str(g): snapshots[g]['name'] for g in (0,1,2,3)},
        queue_policy=policy, scientific_source_hashes=code_hashes(),
        scheduler_source_hashes=scheduler_hashes(), validation_receipts=receipts,
        numerical_protocol='unchanged v1.3; preserve original fitting, fields, checkpoints and source hashes',
        timing_protocol='record physical GPU, concurrency and OOM attempts; do not treat shared wall times as exclusive speed')
    atomic_json(folder/'manifest.json', revision)
    with tarfile.open(folder/'scheduler_sources.tar.gz','w:gz') as archive:
        for path in HERE.glob('*.py'):
            archive.add(path, arcname=path.name)
    identity = read_json(root/'controller_identity.json')
    atomic_json(folder/'controller_identity_before.json', identity)
    if same_process(identity['identity']):
        os.kill(identity['identity']['pid'], signal.SIGTERM)
        deadline = time.monotonic()+10
        while same_process(identity['identity']) and time.monotonic() < deadline:
            time.sleep(.1)
        if same_process(identity['identity']):
            raise RuntimeError('Old controller did not stop; no workers were signalled')
    with (root/'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Preserve exact queue bytes after the prior writer has released its lock.
        snapshot = (root/'queue_state.json').read_bytes()
        (folder/'queue_state_before.json').write_bytes(snapshot)
        state = read_json(root/'queue_state.json')
        workers = [j['active_attempt'] for j in state['jobs'] if j['status'] in ('running','launching','terminating')]
        atomic_json(folder/'handoff.json', dict(at=now(), counts_before=state['counts'],
                    inherited_attempts=workers, workers_signalled=0,
                    base_launch_sha256=sha(root/'launch_manifest.json'),
                    queue_snapshot_sha256=sha(folder/'queue_state_before.json')))
        atomic_json(root/'ACTIVE_SCHEDULER.json', dict(path=str(folder/'manifest.json'),
                    sha256=sha(folder/'manifest.json'), activated_at=now()))
    with (folder/'controller.log').open('ab',buffering=0) as log:
        child = subprocess.Popen([sys.executable,'-B',str(HERE/'dispatcher.py'),
                    '--root',str(root),'--revision',str(folder/'manifest.json')],
                    stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
    atomic_json(folder/'controller_spawn.json',dict(at=now(),identity=process_identity(child.pid)))
    print(dict(controller_pid=child.pid, revision=str(folder/'manifest.json'),
               counts_before=state['counts'], inherited_workers=len(workers), workers_signalled=0),flush=True)


if __name__=='__main__':
    main()
