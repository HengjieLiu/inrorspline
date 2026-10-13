"""Canonical FeDDA historical baseline and complete-only first100 sweep report."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics

import fedda_training as s
from run_oasis_sitreg_first100_eval import label_rows, json_safe, write_report

METRICS = {'dice': 'dice_mean', 'hd95': 'hd95_mean', 'asd': 'asd_symmetric_mean',
           'log_jdet_std_fg': 'log_jdet_std_fg', 'jdet_le0_pct_fg': 'jdet_le0_pct_fg'}
REFERENCE_TABLE = Path('/database/inrorspline/repo_experiments/oasis_seed3_first100_mrdbscp_vfa_sitreg_sweeps_comparison_v1/per_case.csv')
BASELINE_ARGS = None


def validate_cohort(rows, pairs):
    if len(rows) != 100 or sorted(int(r['rank']) for r in rows) != list(range(100)):
        raise ValueError('Require100 unique canonical ranks')
    for row in rows:
        if (row['fixed_id'], row['moving_id']) != pairs[int(row['rank'])]:
            raise ValueError('Pair identity mismatch')
        if any(not math.isfinite(float(row[k])) for k in METRICS):
            raise ValueError('Nonfinite case metrics')


def baseline_case(rank):
    import nibabel as nib
    import numpy as np
    import torch
    import common as c
    from evaluate_saved_baselines_first10 import voxel_zyx_to_norm_dhw3
    torch.set_num_threads(1)
    args = BASELINE_ARGS
    pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)
    fixed, moving = pairs[rank]
    path = args.reference_root / 'inference/inference_bestdice' / f'disp_{fixed}_{moving}.nii.gz'
    field = np.ascontiguousarray(np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)[None])
    if field.shape != (1, 3, *s.SHAPE) or not np.isfinite(field).all():
        raise ValueError('Invalid original FeDDA field')
    batch = c.load_oasis_case(rank=rank, device=torch.device('cpu'), oasis_root=args.data_root)
    dense = torch.from_numpy(voxel_zyx_to_norm_dhw3(field))
    metrics = c.evaluate_from_dense_coords(c.make_coordinate_grid(s.SHAPE, torch.device('cpu')) + dense, batch)
    reg = c.jacobian_metrics_voxel(field, batch['fixed_mask'].bool().numpy())
    row = {'method': 'FeDDA-1MF lambda1 historical', 'alpha': 1.0, 'rank': rank,
           'fixed_id': fixed, 'moving_id': moving, 'direction': 'moving_to_fixed',
           **{k: float(({**metrics, **reg})[v]) for k, v in METRICS.items()},
           'field_path': str(path), 'field_sha256': s.sha256(path)}
    details = label_rows(metrics, batch['fixed_seg'], batch['moving_seg'],
                        {k: row[k] for k in ('method', 'rank', 'fixed_id', 'moving_id')})
    row['input_sha256'] = {str(p): s.sha256(p) for ident in (fixed, moving)
                           for p in (args.data_root / 'test' / f'img{ident}.nii.gz',
                                     args.data_root / 'test' / f'seg{ident}.nii.gz')}
    return row, json_safe(details)


def baseline(args):
    global BASELINE_ARGS
    import common as c
    cache = args.out / 'historical_lambda1'
    pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)
    sources = {str(Path(__file__).resolve()): s.sha256(Path(__file__)),
               str(Path(c.__file__).resolve()): s.sha256(Path(c.__file__))}
    checkpoint = args.reference_root / 'checkpoint_final/dice0.8141_epoch0196.pth.tar'
    if s.sha256(checkpoint) != s.CHECKPOINT_SHA:
        raise ValueError('Wrong historical FeDDA checkpoint')
    if (cache / 'COMPLETED').exists():
        manifest = json.loads((cache / 'manifest.json').read_text())
        s.verify_hashes(manifest['source_sha256'])
        rows = json.loads((cache / 'per_case.json').read_text())
        validate_cohort(rows, pairs)
        for row in rows:
            s.verify_hashes({row['field_path']: row['field_sha256'], **row['input_sha256']})
        return rows
    cache.mkdir(parents=True, exist_ok=False)
    s.check_dataset(args.data_root)
    BASELINE_ARGS = args
    rows, labels = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool:
        for row, details in pool.map(baseline_case, range(100)):
            rows.append(row)
            labels.extend(details)
            if len(rows) % 10 == 0:
                print(json.dumps({'baseline_cases': len(rows)}), flush=True)
    validate_cohort(rows, pairs)
    if len(labels) != 3500:
        raise ValueError('Incomplete historical label metrics')
    s.write_json(cache / 'per_case.json', rows)
    c.write_csv(cache / 'per_case.csv', [{k: v for k, v in r.items() if k != 'input_sha256'} for r in rows])
    s.write_json(cache / 'per_label_per_case.json', labels)
    c.write_csv(cache / 'per_label_per_case.csv', labels)
    s.write_json(cache / 'manifest.json', {'checkpoint': str(checkpoint), 'checkpoint_sha256': s.CHECKPOINT_SHA,
                 'source_sha256': sources, 'pair_file_sha256': s.sha256(c.DEFAULT_PAIR_FILE),
                 'foreground': 'fixed_raw_image_positive', 'label_rows': len(labels),
                 'selection': 'historical_best_validation_Dice_epoch196', 'cases': 100})
    (cache / 'COMPLETED').write_text('100 original saved fields,3500 label rows\n')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--baseline-only', action='store_true')
    a = p.parse_args()
    if not a.baseline_only and not a.training_root:
        p.error('Full report needs --training-root')
    import torch
    import common as c
    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'ACCEPTED.json').exists():
        raise FileExistsError('Preserve completed report')
    rows = baseline(a)
    if a.baseline_only:
        print(json.dumps({'baseline': 'complete', 'mean': {k: statistics.mean(r[k] for r in rows) for k in METRICS}}), flush=True)
        return
    pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)
    queue = json.loads((a.training_root / 'queue.json').read_text())
    if len(queue['jobs']) != 4 or any(j['status'] != 'complete' for j in queue['jobs']):
        raise ValueError('Only aggregate all four successful candidates')
    provenance = {str(REFERENCE_TABLE): s.sha256(REFERENCE_TABLE)}
    for job in queue['jobs']:
        output = Path(job['eval_dir'])
        if not (output / 'COMPLETED').exists():
            raise ValueError('Missing evaluation completion marker')
        source = json.loads((output / 'per_case.json').read_text())
        labels = json.loads((output / 'per_label_per_case.json').read_text())
        if len(labels) != 3500:
            raise ValueError('Require3500 label rows per candidate')
        manifest = json.loads((output / 'manifest.json').read_text())
        s.verify_hashes({manifest['checkpoint']: manifest['checkpoint_sha256']})
        s.verify_hashes(manifest['source_sha256'])
        s.verify_hashes(manifest['native_source_sha256'])
        alpha = s.LAMBDAS[job['lambda_key']]
        subset = []
        for r in source:
            s.verify_hashes({r['field_path']: r['field_sha256']})
            subset.append({'method': f'FeDDA-1MF lambda{alpha:g}', 'alpha': alpha,
                           **{k: r[k] for k in ('rank', 'fixed_id', 'moving_id')},
                           **{k: r[v] for k, v in METRICS.items()}, 'gpu': job['gpu'],
                           'best_epoch': manifest['best_epoch'], 'best_val_dice': manifest['best_val_dice']})
        validate_cohort(subset, pairs)
        rows.extend(subset)
        provenance[str(output / 'per_case.json')] = s.sha256(output / 'per_case.json')
        provenance[str(output / 'per_label_per_case.json')] = s.sha256(output / 'per_label_per_case.json')
    with REFERENCE_TABLE.open() as f:
        references = list(csv.DictReader(f))
    for name in ('VFA (original)', 'SITReg λ=1 (original)'):
        subset = [{**r, 'rank': int(r['rank']), 'alpha': 1.0,
                   **{k: float(r[k]) for k in METRICS}} for r in references if r['method'] == name]
        validate_cohort(subset, pairs)
        rows.extend(subset)
    groups = sorted(set(r['method'] for r in rows))
    summary = []
    for name in groups:
        subset = [r for r in rows if r['method'] == name]
        row = {'method': name, 'alpha': subset[0]['alpha'], 'n': len(subset)}
        for k in METRICS:
            row[k + '_mean'] = statistics.mean(float(r[k]) for r in subset)
            row[k + '_std'] = statistics.stdev(float(r[k]) for r in subset)
        summary.append(row)
    export = [{k: r[k] for k in ('method', 'alpha', 'rank', 'fixed_id', 'moving_id', *METRICS)} for r in rows]
    c.write_csv(a.out / 'per_case.csv', export)
    c.write_csv(a.out / 'mean_std.csv', summary)
    s.write_json(a.out / 'results.json', {'summary': summary, 'per_case': export, 'input_sha256': provenance})
    lines = ['# FeDDA-1MF first100 training lambda sweep', '',
             'Same seed3 shuffled100 pairs;mean ± across-case sample SD. Historical lambda1 is not a fresh replication.',
             'Native best-validation-Dice checkpoints; no test-selected epochs. Foreground=fixed raw image>0;',
             'SD(logJ) uses positive J; folding is J<=0 percent. Canonical pooled-area symmetric ASD in mm.', '',
             '| Method | Dice | HD95 mm | ASD mm | FG SDlogJ | FG folding % |', '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| ' + r['method'] + ' | ' + ' | '.join(f'{r[k + "_mean"]:.6f} ± {r[k + "_std"]:.6f}' for k in METRICS) + ' |')
    write_report(a.out / 'report.md', lines)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve = sorted((r for r in summary if r['method'].startswith('FeDDA')), key=lambda r: r['alpha'])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, metric in zip(axes, ('dice', 'hd95', 'asd')):
        ax.plot([r['log_jdet_std_fg_mean'] for r in curve], [r[metric + '_mean'] for r in curve], '-o', label='FeDDA-1MF')
        for r in curve:
            ax.annotate(f'{r["alpha"]:g}', (r['log_jdet_std_fg_mean'], r[metric + '_mean']), xytext=(4, 4), textcoords='offset points', fontsize=8)
        for r in summary:
            if not r['method'].startswith('FeDDA'):
                ax.scatter(r['log_jdet_std_fg_mean'], r[metric + '_mean'], marker='*', s=90, label=r['method'])
        ax.set(xlabel='Foreground SD(log J)', ylabel={'dice': 'Dice', 'hd95': 'HD95 (mm)', 'asd': 'Symmetric ASD (mm)'}[metric])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(a.out / f'accuracy_regularity_lines.{ext}', dpi=180)
    plt.close(fig)
    s.write_json(a.out / 'ACCEPTED.json', {'case_rows': len(rows), 'groups': len(summary), 'new_candidates': 4,
                 'new_label_rows': 14000, 'input_sha256': provenance,
                 'report_sha256': {p.name: s.sha256(p) for p in a.out.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
