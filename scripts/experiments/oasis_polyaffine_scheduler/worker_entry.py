#!/usr/bin/env python
"""Select a physical GPU and reuse the frozen P81 worker without changing fits."""
import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from support import (atomic_json, read_json, sha, now, assert_external,
                     validate_revision, cuda_oom)
from worker import Worker


class ContinuationWorker(Worker):
    def __init__(self, *args, physical_gpu, gpu_uuid, **kwargs):
        self.physical_gpu, self.gpu_uuid = physical_gpu, gpu_uuid
        self.reused_endpoints = []
        super().__init__(*args, **kwargs)

    def progress(self, **row):
        super().progress(**{**row, 'physical_gpu': self.physical_gpu, 'gpu_uuid': self.gpu_uuid})

    def save_endpoint(self, basis, coeff, cap, *, supplement=False):
        pointer = self.root/'endpoints'/self.job['id']/f'cap{cap}.json'
        if not supplement and pointer.exists():
            # An OOM can happen after a tree checkpoint was already published.
            # Preserve that endpoint (and any dependent jobs), and restore its
            # exact coefficients before continuing the remaining tree budgets.
            old = read_json(pointer)
            if old.get('status') != 'complete' or old.get('job_id') != self.job['id']:
                raise RuntimeError('Invalid previously published checkpoint')
            for kind in ('basis', 'coefficients', 'field'):
                if sha(old[kind+'_path']) != old[kind+'_sha256']:
                    raise RuntimeError('Previously published checkpoint changed: '+kind)
            if basis.to_dict() != read_json(old['basis_path']):
                raise RuntimeError('Retry tree differs from published checkpoint; preserving original and refusing to mix paths')
            saved = torch.as_tensor(np.load(old['coefficients_path']), device=coeff.device, dtype=coeff.dtype)
            if saved.shape != coeff.shape:
                raise RuntimeError('Retry checkpoint coefficient shape mismatch')
            with torch.no_grad():
                coeff.copy_(saved)
            self.endpoints.append(str(pointer))
            self.reused_endpoints.append({'pointer': str(pointer), 'sha256': sha(pointer),
                                          'restored_coefficients': True})
            self.progress(phase='restored_published_checkpoint', budget_cap=cap)
            return old
        return super().save_endpoint(basis, coeff, cap, supplement=supplement)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--revision', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    root, output = assert_external(args.root), assert_external(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        revision = read_json(args.revision)
        validate_revision(root, revision)
        uuid = revision['gpu_uuids'][str(args.gpu)]
        if str(args.gpu) not in revision['gpu_limits'] or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
            raise RuntimeError('Worker GPU assignment does not match the frozen scheduler revision')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Each worker must see exactly its assigned GPU')
        manifest = read_json(root/'launch_manifest.json')
        torch.set_num_threads(revision['queue_policy']['cpu_threads'])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        job = next(j for j in manifest['jobs'] if j['id'] == args.job_id)
        target = next(t for t in read_json(root/'input_manifest.json')['targets'] if t['id'] == job['target_id'])
        if target['status'] != 'passed' or sha(target['canonical_path']) != target['canonical_sha256']:
            raise RuntimeError('Prepared reference no longer matches its acceptance')
        worker = ContinuationWorker(root, job, output, manifest['fit_config'], target,
                                    physical_gpu=args.gpu, gpu_uuid=uuid)
        worker.run()
        atomic_json(output/'result.json', {
            'status': 'complete', 'job_id': job['id'], 'endpoints': worker.endpoints,
            'at': now(), 'elapsed_sec': time.monotonic()-worker.start,
            'physical_gpu': args.gpu, 'gpu_uuid': uuid,
            'scheduler_revision_sha256': sha(args.revision),
            'reused_endpoints': worker.reused_endpoints,
            'requested_backend': manifest['fit_config']['backend'],
            'effective_backend': {'cudnn_benchmark': torch.backends.cudnn.benchmark,
                                  'cudnn_deterministic': torch.backends.cudnn.deterministic,
                                  'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                                  'tf32_cudnn': torch.backends.cudnn.allow_tf32}})
    except BaseException as exc:
        atomic_json(output/'failure.json', {
            'status': 'failed', 'job_id': args.job_id, 'at': now(),
            'physical_gpu': args.gpu, 'error': type(exc).__name__+': '+str(exc),
            'cuda_oom': isinstance(exc, torch.cuda.OutOfMemoryError) or cuda_oom(str(exc)),
            'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
