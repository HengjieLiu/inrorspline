#!/usr/bin/env python
"""Native-volume serialized SVF smoke on one explicitly masked physical GPU."""
import argparse
import copy
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from support import atomic_json, read_json, sha, now, code_hashes, assert_external, scheduler_hashes
from dispatcher import gpu_snapshots
from worker_entry import ContinuationWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    output = assert_external(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources, scheduling = code_hashes(), scheduler_hashes()
    started = time.monotonic()
    try:
        snap = gpu_snapshots()[args.gpu]
        if os.environ.get('CUDA_VISIBLE_DEVICES') != snap['uuid'] or torch.cuda.device_count() != 1:
            raise RuntimeError('GPU smoke mask must equal the selected physical GPU UUID')
        if snap['free_mib'] < 6144:
            raise RuntimeError('GPU smoke memory admission: need 6 GiB free')
        manifest = read_json(args.root/'launch_manifest.json')
        if sources != manifest['code_hashes']:
            raise RuntimeError('Scientific source changed')
        torch.set_num_threads(3)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        config = copy.deepcopy(manifest['fit_config'])
        config['fixed_grid_steps'] = [[4, 1], [2, 1], [1, 1]]
        config['nonlinear']['check_interval'] = 1
        target = next(t for t in read_json(args.root/'input_manifest.json')['targets'] if t['id'] == 'rank000_sitreg')
        if sha(target['canonical_path']) != target['canonical_sha256']:
            raise RuntimeError('GPU smoke target changed after acceptance')
        job = dict(id=f'gpu{args.gpu}_svf_smoke', target_id=target['id'], mode='svf', family='bspline',
                   budgets=[192], initialization='global_affine_ls', kind='scheduler_smoke')
        worker = ContinuationWorker(output, job, output/'attempt0', config, target,
                                    smoke=True, physical_gpu=args.gpu, gpu_uuid=snap['uuid'])
        worker.run()
        if len(worker.endpoints) != 1:
            raise RuntimeError('Missing serialized smoke endpoint')
        endpoint = read_json(worker.endpoints[0])
        for kind in ('basis', 'coefficients', 'field'):
            if sha(endpoint[kind+'_path']) != endpoint[kind+'_sha256']:
                raise RuntimeError('Serialized smoke checksum mismatch')
        if code_hashes() != sources:
            raise RuntimeError('Scientific source changed during GPU smoke')
        receipt = dict(passed=True, at=now(), physical_gpu=args.gpu, gpu_uuid=snap['uuid'],
                       base_launch_sha256=sha(args.root/'launch_manifest.json'),
                       input_manifest_sha256=sha(args.root/'input_manifest.json'),
                       device_name=torch.cuda.get_device_name(0), endpoints=worker.endpoints,
                       scientific_source_hashes=sources,
                       worker_entry_sha256=sha(Path(__file__).with_name('worker_entry.py')),
                       verify_source_sha256=sha(__file__),
                       scheduler_source_hashes_at_start=scheduling,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                       elapsed_sec=time.monotonic()-started)
        atomic_json(output/'receipt.json', receipt)
        print({k: receipt[k] for k in ('passed', 'physical_gpu', 'peak_reserved_mib', 'elapsed_sec')}, flush=True)
    except BaseException as exc:
        atomic_json(output/'failure.json', dict(passed=False, at=now(), physical_gpu=args.gpu,
                    error=str(exc), traceback=traceback.format_exc(), scientific_source_hashes=sources))
        raise


if __name__ == '__main__':
    main()
