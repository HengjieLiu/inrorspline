"""Certify a completed, zero-issue FeDDA audit without rerunning or hiding failures.

This establishes adapter correctness, not repeatability of historical cuDNN
benchmark mode. The stricter saved-field equality check cannot certify a merely
nearby field. All prior failed preflights must remain linked in the certificate.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

import fedda_training as s


def validate_contract(result, manifest):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(result['status'] == 'audit_only' and result['issues'] == [], 'Unsuccessful audit')
    parity = result['parity']
    require(parity['bitwise_field_reproduction'] is True and
            parity['max_abs_voxel_error'] == parity['rmse_voxel'] == 0,
            'Certification requires an exactly reproduced saved field')
    require((parity['fixed_id'], parity['moving_id']) == ('0333', '0115'), 'Wrong case')
    for key in ('dice_mean', 'hd95_mean', 'asd_symmetric_mean', 'log_jdet_std_fg', 'jdet_le0_pct_fg'):
        value = parity['actual_metrics'][key]
        require(math.isfinite(value) and value == parity['reference_metrics'][key],
                f'Exact-field metric mismatch: {key}')
    require(abs(result['native_validation']['dice'] - result['historical_validation_dice']) <= 1e-4,
            'Native validation mismatch')
    require(manifest['backend'] == {'benchmark': True, 'deterministic': False,
                                    'cudnn_tf32': True, 'matmul_tf32': False}, 'Backend drift')
    require((manifest['uid'], manifest['gid'], manifest['physical_gpu']) == (1009, 1020, 0),
            'Wrong execution identity or GPU')
    require(manifest['model_key'] == 'fedda_1mf' and manifest['seed'] == 42 and
            manifest['updates_per_pair'] == 2 and manifest['use_batch_parallel'] is False,
            'Protocol drift')
    steps = result['native_steps']
    require(len(steps) == 4 and {row['lambda_key'] for row in steps} == set(s.LAMBDAS),
            'Incomplete lambda smokes')
    for row in steps:
        require(row['lambda'] == s.LAMBDAS[row['lambda_key']] and row['adam_updates'] == 2 and
                row['finite_gradients'] and row['checkpoint_roundtrip_exact'], 'Invalid smoke')
        require([r['direction'] for r in row['directions']] == ['forward', 'reverse'],
                'Incorrect directional updates')
        require(all(math.isfinite(r[key]) for r in row['directions']
                    for key in ('loss', 'similarity', 'diffusion', 'weighted_diffusion')),
                'Nonfinite training loss')
    require(result['warm_memory']['adam_updates'] == 6, 'Missing warm-state smoke')
    peaks = [row['peak_reserved_gib'] for row in steps] + [result['warm_memory']['peak_reserved_gib']]
    require(all(math.isfinite(v) and v > 0 for v in peaks), 'Invalid memory measurement')
    require(result['min_train_free_gib'] >= max(24, max(peaks) + 2), 'Unsafe memory gate')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--prior-failure', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError('Use container UID:GID1009:1020')
    if not (a.audit / 'DIAGNOSTIC_COMPLETED').is_file() or (a.audit / 'FAILED.txt').exists():
        raise RuntimeError('Audit did not finish successfully')
    result = json.loads((a.audit / 'preflight.json').read_text())
    manifest = json.loads((a.audit / 'manifest.json').read_text())
    validate_contract(result, manifest)
    s.verify_hashes(result['source_sha256'])
    native = manifest['arguments']
    if not s.sha256(native['reference_checkpoint']) == result['reference_checkpoint_sha256'] == s.CHECKPOINT_SHA:
        raise ValueError('Checkpoint mismatch')
    assert s.sha256(native['reference_field']) == result['reference_field_sha256']
    assert s.sha256(Path(native['data_root']) / 'dataset.json') == result['data_sha256'] == s.DATA_SHA
    import nibabel as nib
    import numpy as np
    import torch
    torch.set_num_threads(4)
    field = np.asarray(nib.load(str(a.audit / 'reference_rank000_dvf.nii.gz')).dataobj)
    reference = np.asarray(nib.load(native['reference_field']).dataobj)
    assert field.shape == reference.shape == (3, *s.SHAPE)
    assert np.isfinite(field).all() and np.array_equal(field, reference)
    labels = json.loads((a.audit / 'smoke_per_label.json').read_text())
    assert len(labels) == 35
    for key, alpha in s.LAMBDAS.items():
        checkpoint = torch.load(a.audit / f'smoke_lambda_{key}.pth', map_location='cpu', weights_only=False)
        assert checkpoint['lambda'] == alpha
        assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
        assert all(int(v['step']) == 2 for v in checkpoint['optimizer']['state'].values())
    prior = []
    evidence = [a.audit / name for name in ('preflight.json', 'manifest.json', 'field_parity.json',
                'reference_rank000_dvf.nii.gz', 'smoke_per_label.json', 'DIAGNOSTIC_COMPLETED')]
    evidence.extend(a.audit / f'smoke_lambda_{key}.pth' for key in s.LAMBDAS)
    for path in a.prior_failure:
        failure = path / 'FAILED.txt'
        if not failure.is_file():
            raise ValueError(f'Missing preserved failed preflight: {path}')
        prior.append({'directory': str(path), 'failure': failure.read_text(),
                      'field_parity': json.loads((path / 'field_parity.json').read_text())})
        evidence.extend([failure, path / 'field_parity.json'])
    certified = copy.deepcopy(result)
    certified.update(status='passed', certification={
        'kind': 'independently_validated_completed_audit', 'audit': str(a.audit),
        'certified_unix': time.time(), 'prior_failed_preflights': prior,
        'evidence_sha256': {str(path): s.sha256(path) for path in evidence},
        'limitation': 'Exact adapter reproduction demonstrated; historical benchmark mode is not '
                      'cross-process repeatable. Prior HD95 gate failure remains recorded; no '
                      'claim that every fresh inference passes those metric tolerances.'})
    certified['source_sha256'][str(Path(__file__).resolve())] = s.sha256(__file__)
    a.out.mkdir(parents=True, exist_ok=False)
    s.write_json(a.out / 'manifest.json', {**manifest, 'certified_from': str(a.audit)})
    s.write_json(a.out / 'preflight.json', certified)
    (a.out / 'PASSED').write_text('Exact saved-field, native validation and training-memory audit certified\n')
    print(json.dumps({'certified': str(a.out), 'audit': str(a.audit),
                      'min_train_free_gib': certified['min_train_free_gib'], 'prior_failures_retained': len(prior)}))


if __name__ == '__main__':
    main()
