"""Independently certify a complete zero-issue GNCC audit, preserving failures.

No new inference or changed tolerance. Require BITWISE historical-field and
native/adapter parity, plus all originally requested validation/update/metric
checks. This certifies adapter correctness, not cross-process repeatability.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import time

import fedda_gncc_training as g


def validate_contract(audit, diagnostic, manifest):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    for value in (audit, manifest):
        g.require_protocol(value)
    require(audit['status'] == 'audit_only' and audit['issues'] == [], 'Audit has failed historical gates')
    require(diagnostic['status'] == 'diagnostic_only' and diagnostic['production_qualified'] is False and
            diagnostic['historical_parity_issues'] == [], 'Not a clean non-accepting diagnostic')
    parity = audit['parity']
    require(parity['bitwise_field_reproduction'] and parity['max_abs_voxel_error'] == parity['rmse_voxel'] == 0,
            'Require exact saved-field reproduction, not relaxed numerical bounds')
    require((parity['fixed_id'], parity['moving_id']) == ('0333', '0115'), 'Wrong reference case')
    for metric in ('dice_mean', 'hd95_mean', 'asd_symmetric_mean', 'log_jdet_std_fg', 'jdet_le0_pct_fg'):
        value = parity['actual_metrics'][metric]
        require(math.isfinite(value) and value == parity['reference_metrics'][metric], 'Reference metric inequality')
    require(abs(audit['native_validation']['dice']-audit['historical_validation_dice']) <= 1e-4,
            'Native validation gate failed')
    comparison = diagnostic['independent_native_vs_gncc_model']
    require(comparison['bitwise_equal'] and comparison['identical_state_hash'] and
            comparison['max_abs_voxel'] == comparison['rmse_voxel'] == 0, 'Native/adapter model mismatch')
    require((comparison['native_similarity_class'], comparison['native_keep_sign'], comparison['new_similarity_class'],
             comparison['native_parameters']) == ('NCC_vfa', False, 'GlobalNCC', 2854079), 'Wrong loss/model identity')
    require(manifest['backend'] == g.PROTOCOL['backend'] and
            (manifest['uid'], manifest['gid']) == (1009, 1020) and manifest['physical_gpu'] in range(4),
            'Backend/execution identity drift')
    require(manifest['model_key'] == 'fedda_1mf' and manifest['updates_per_pair'] == 2 and
            manifest['seed'] == 42 and manifest['use_batch_parallel'] is False, 'FeDDA protocol drift')
    steps = audit['native_steps']
    require(len(steps) == 4 and {r['lambda_key'] for r in steps} == set(g.LAMBDAS), 'Incomplete four-lambda smokes')
    for row in steps:
        require(row['lambda'] == g.LAMBDAS[row['lambda_key']] and row['adam_updates'] == 2 and
                row['finite_gradients'] and row['checkpoint_roundtrip_exact'], 'Invalid native smoke')
        require([r['direction'] for r in row['directions']] == ['forward', 'reverse'], 'Wrong sequential directions')
        require(all(math.isfinite(r[k]) for r in row['directions'] for k in
                    ('loss', 'similarity', 'diffusion', 'weighted_diffusion')), 'Nonfinite fresh update')
    require(audit['warm_memory']['adam_updates'] == 6, 'Warm optimizer smoke missing')
    require(diagnostic['trained_checkpoint_adam_updates'] == 2 and
            [r['direction'] for r in diagnostic['trained_checkpoint_directions']] == ['forward', 'reverse'],
            'Trained GNCC two-update smoke missing')
    require(all(math.isfinite(r[k]) for r in diagnostic['trained_checkpoint_directions'] for k in
                ('loss', 'similarity', 'diffusion', 'weighted_diffusion')), 'Nonfinite trained update')
    require([r['rank'] for r in diagnostic['evaluation_smokes']] == [0, 10] and
            all(r['label_rows'] == 35 for r in diagnostic['evaluation_smokes']), 'Incomplete detailed metric smoke')
    peaks = [r['peak_reserved_gib'] for r in steps]+[audit['warm_memory']['peak_reserved_gib']]
    require(all(math.isfinite(v) and v > 0 for v in peaks), 'Invalid measured memory')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--prior-failure', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    g.require_identity()
    if not (a.audit/'DIAGNOSTIC_COMPLETED').is_file() or (a.audit/'FAILED.txt').exists():
        raise RuntimeError('Incomplete/failed diagnostic')
    audit, diagnostic, manifest = [json.loads((a.audit/n).read_text()) for n in
                                  ('preflight.json', 'diagnostic.json', 'manifest.json')]
    validate_contract(audit, diagnostic, manifest)
    g.verify_hashes(audit['source_sha256']); g.verify_hashes(diagnostic['source_sha256'])
    if diagnostic['native_audit_sha256'] != g.sha256(a.audit/'preflight.json'):
        raise ValueError('Audit changed after diagnostic')
    args = manifest['arguments']
    baseline_dir = Path(args['baseline_preflight'])
    baseline = json.loads((baseline_dir/'preflight.json').read_text())
    g.verify_hashes(baseline['source_sha256']); g.verify_hashes(baseline['certification']['evidence_sha256'])
    if diagnostic['baseline_certificate_sha256'] != g.sha256(baseline_dir/'preflight.json'):
        raise ValueError('Historical certificate changed')
    if g.sha256(args['reference_checkpoint']) != audit['reference_checkpoint_sha256'] or audit['reference_checkpoint_sha256'] != g.CHECKPOINT_SHA:
        raise ValueError('Wrong original checkpoint')
    if g.sha256(args['reference_field']) != audit['reference_field_sha256']:
        raise ValueError('Original field changed')
    if g.sha256(Path(args['data_root'])/'dataset.json') != audit['data_sha256'] or audit['data_sha256'] != g.DATA_SHA:
        raise ValueError('Dataset changed')
    pair_hashes = g.baseline_pair_hashes(Path(args['baseline_training_root']), audit['initial_state_sha256'])
    if diagnostic['expected_pair_hashes'] != pair_hashes or baseline['initial_state_sha256'] != audit['initial_state_sha256']:
        raise ValueError('Initialization or baseline pair hashes disagree')
    import nibabel as nib
    import numpy as np
    import torch
    torch.set_num_threads(4)
    actual = np.asarray(nib.load(str(a.audit/'reference_rank000_dvf.nii.gz')).dataobj)
    reference = np.asarray(nib.load(args['reference_field']).dataobj)
    if actual.shape != (3, *g.SHAPE) or not np.isfinite(actual).all() or not np.array_equal(actual, reference):
        raise ValueError('Saved audit field is not bitwise historical reference')
    evidence = [a.audit/n for n in ('preflight.json', 'diagnostic.json', 'manifest.json', 'field_parity.json',
                 'reference_rank000_dvf.nii.gz', 'smoke_per_label.json', 'DIAGNOSTIC_COMPLETED')]
    if len(json.loads((a.audit/'smoke_per_label.json').read_text())) != 35:
        raise ValueError('Missing reference label details')
    for key, alpha in g.LAMBDAS.items():
        path = a.audit/f'smoke_lambda_{key}.pth'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        g.require_protocol(checkpoint)
        if checkpoint['lambda'] != alpha or not all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values()):
            raise ValueError('Invalid fresh checkpoint')
        if not all(int(v['step']) == 2 for v in checkpoint['optimizer']['state'].values()):
            raise ValueError('Invalid optimizer checkpoint')
        evidence.append(path)
    for row in diagnostic['evaluation_smokes']:
        path = Path(row['field_path'])
        field = np.asarray(nib.load(str(path)).dataobj)
        if field.shape != (3, *g.SHAPE) or not np.isfinite(field).all() or g.sha256(path) != row['field_sha256']:
            raise ValueError('Invalid detailed-smoke field')
        metrics_path = path.parent/'metrics.json'
        saved = json.loads(metrics_path.read_text())
        if saved['summary'] != row or sorted(r['label_id'] for r in saved['per_label']) != list(range(1, 36)):
            raise ValueError('Detailed evaluation record mismatch')
        evidence.extend([path, metrics_path, path.parent/'per_label.csv'])
    prior = []
    for directory in a.prior_failure:
        failure = directory/'FAILED.txt'
        if not failure.is_file():
            raise ValueError('Missing preserved preflight failure')
        prior.append(dict(directory=str(directory), failure=failure.read_text()))
        evidence.append(failure)
        for name in ('field_parity.json', 'manifest.json', 'stdout.log'):
            if (directory/name).is_file(): evidence.append(directory/name)
    peaks = [r['peak_reserved_gib'] for r in audit['native_steps']]+[audit['warm_memory']['peak_reserved_gib']]
    certified = copy.deepcopy(audit)
    certified.update(status='passed', gncc_gate='passed', peak_reserved_gib=max(peaks),
                     min_train_free_gib=max(24, math.ceil(max(peaks)+2)),
                     expected_pair_hashes=pair_hashes, trained_weight_step=diagnostic['trained_checkpoint_directions'],
                     evaluation_smokes=diagnostic['evaluation_smokes'], baseline_sources_sha256=baseline['source_sha256'],
                     baseline_evidence_sha256=baseline['certification']['evidence_sha256'],
                     baseline_preflight_sha256=g.sha256(baseline_dir/'preflight.json'),
                     source_sha256={**diagnostic['source_sha256'], str(Path(__file__).resolve()): g.sha256(__file__)},
                     certification=dict(kind='independently_verified_zero_issue_exact_field_and_native_adapter_audit',
                        audit=str(a.audit), certified_unix=time.time(), prior_failed_preflights=prior,
                        evidence_sha256={str(path): g.sha256(path) for path in evidence},
                        limitation='Certifies adapter correctness with unchanged tolerances, not repeatability '
                                   'of historical CUDA benchmark mode. Earlier metric-gate failure remains recorded.'))
    a.out.mkdir(parents=True, exist_ok=False)
    g.write_json(a.out/'manifest.json', {**manifest, 'certified_from': str(a.audit)})
    g.write_json(a.out/'preflight.json', certified)
    (a.out/'PASSED').write_text('Original strict gates and exact-field/native-adapter audit independently verified\n')
    (a.out/'GNCC_PASSED').write_text('All original FeDDA GNCC gates independently verified from complete audit\n')
    print(json.dumps({'certified': str(a.out), 'peak_reserved_gib': max(peaks),
                      'min_train_free_gib': certified['min_train_free_gib'], 'prior_failures_retained': len(prior)}), flush=True)


if __name__ == '__main__':
    main()
