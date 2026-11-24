"""Independent completed-campaign metadata and landmark replay audit.

Dense fields were scanned, hashed and reloaded by each worker. This audit checks
their headers and the sealed receipts, without rehashing terabytes of fields.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .protocol import dump, read, sha, stamp, sweep_state, weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config = read(args.config)
    run = Path(config['run_root'])
    review = Path(config['review_root'])
    frozen = Path(config['frozen_root'])
    state = read(run / 'controller_state.json')
    errors = []
    accepted = 0
    failures = []
    capped = []
    point_count = 0
    max_error = 0.

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(state['stage'] == 'complete', 'Campaign is not complete')
    require(len(state['sweeps']) == 180, 'Expected180 sweeps')
    manifest = read(frozen / 'source_manifest.json')
    require(sha(args.config) == manifest['config_sha256'], 'Config drift')
    for name, digest in manifest['source_hashes'].items():
        require(sha(frozen / name) == digest, f'Frozen source drift: {name}')
    for sw in state['sweeps']:
        search = sweep_state(sw['results'], sw.get('preparation_failure', False))
        require(search['done'], f"Incomplete sweep: {sw['id']}")
        for direction in ('upper', 'lower'):
            if search.get(direction) and search[direction]['status'] == 'cap_reached':
                capped.append(dict(sweep=sw['id'], direction=direction))
        for k, result in sw['results'].items():
            path = Path(result['path'])
            if result['status'] != 'accepted':
                failures.append(dict(sweep=sw['id'], k=int(k), error=result.get('error')))
                continue
            try:
                evaluation = read(path / 'evaluation.json')
                require(sha(path / 'evaluation.json') == result['evaluation_sha256'], 'Evaluation drift')
                require(evaluation['landmarks'] == 300, 'Incomplete landmarks')
                require(evaluation['weight'] == weight(int(k)), 'Wrong lambda')
                training = read(path / 'training.json')
                require([s['actual_epochs'] for s in training['stage_summaries']] == [400]*4+[900], 'Wrong iterations')
                backend = training['backend']
                require(backend['effective_cudnn_benchmark'] is False and
                        backend['effective_cudnn_deterministic'] is True, 'Backend drift')
                geometry = evaluation['geometry']
                for name, shape in [('disp_vox_reg1mm_zyx.npy', geometry['registration_shape_zyx']),
                                    ('disp_vox_native_crop_zyx.npy', geometry['crop_shape_zyx'])]:
                    field = np.load(path / name, mmap_mode='r')
                    require(field.dtype == np.float32 and field.shape == tuple(shape)+(3,), 'Invalid field header')
                    del field
                with (path / 'per_landmark.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                require(len(rows) == 300 and {int(r['landmark_id']) for r in rows} == set(range(300)), 'Landmark IDs')
                spacing = np.asarray(geometry['native_spacing_zyx_mm'])
                for endpoint in ('snap', 'subvoxel'):
                    predicted = np.array([[float(r[f'predicted_{endpoint}_{a}_native']) for a in 'zyx'] for r in rows])
                    moving = np.array([[float(r[f'moving_{a}_native']) for a in 'zyx'] for r in rows])
                    replay = np.linalg.norm((predicted-moving)*spacing, axis=1)
                    stored = np.array([float(r[f'original_{endpoint}']) for r in rows])
                    delta = float(np.max(np.abs(replay-stored)))
                    max_error = max(max_error, delta)
                    require(np.isfinite(replay).all() and delta < 1e-10, 'Native TRE replay failed')
                    require(abs(float(replay.mean())-evaluation[f'original_{endpoint}']) < 1e-10, 'TRE mean differs')
                    require(abs(float(replay.std(ddof=1))-evaluation[f'original_{endpoint}_sd']) < 1e-10, 'TRE SD differs')
                require(evaluation['original_snap'] == result['score'], 'Controller score differs')
                with (path / 'stage_history.csv').open() as stream:
                    require(sum(1 for _ in csv.DictReader(stream)) == 2500, 'Incomplete loss history')
                anatomy = read(path / 'anatomy_metrics.json')
                require(anatomy['status'] == 'accepted_automatic_proxy', 'Missing anatomy metrics')
                require(0 <= anatomy['source_inside_fraction'] <= 1, 'Invalid source coverage')
                accepted += 1
                point_count += 300
                if accepted % 200 == 0:
                    print('AUDITED', accepted, flush=True)
            except Exception as error:
                errors.append(dict(path=str(path), error=str(error)))
    for report in ('first_batch', 'final', 'anatomy_final'):
        folder = review / report
        receipt = read(folder / 'accepted.json')
        for name, digest in receipt['receipts'].items():
            require(sha(folder / name) == digest, f'{report} sealed report drift: {name}')
    with (review / 'anatomy_final/cohort_summaries.csv').open() as stream:
        cohort = list(csv.DictReader(stream))
    require(all(int(r['n']) == 10 for r in cohort if r['endpoint'] in ('fixed_tre', 'best_tre')), 'Missing cohort TRE')
    output = dict(status='accepted' if not errors else 'failed', time=stamp(),
                  sweeps=len(state['sweeps']), accepted_candidates=accepted,
                  correspondence_rows=point_count, max_native_tre_replay_error_mm=max_error,
                  failures=failures, capped_branches=capped, errors=errors,
                  source_manifest_sha256=sha(frozen / 'source_manifest.json'),
                  audit_source_sha256=sha(Path(__file__)),
                  scope='all accepted metadata/field headers/landmarks/histories/anatomy metrics; sealed report hashes; no dense-field rehash')
    dump(args.out, output)
    print(output, flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
