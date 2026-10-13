"""Complete 13-setting FeDDA comparison; immutable signed GNCC/LNCC inputs."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/oasis_alpha_sweep'))
import common as c
import fedda_gncc2_training as g
import oasis_sitreg_ncc_arc_lines_top as arc
import oasis_ncc_three_regularity_arc as fg

CORE = ('dice', 'hd95', 'asd', 'log_jdet_std_fg', 'jdet_le0_pct_fg')
METRICS = (*CORE, 'diffusion_l2_fg', 'diffusion_l2_full')
SOURCE = ('dice_mean', 'hd95_mean', 'asd_symmetric_mean', 'log_jdet_std_fg', 'jdet_le0_pct_fg')
SERIES = {'lncc9': [('10', 10.), ('3', 3.), ('1', 1.), ('1over3', 1/3), ('0.1', .1)],
          'gncc': [('3', 3.), ('1', 1.), ('1over3', 1/3), ('0.1', .1)],
          'gncc2': [('3', 3.), ('1', 1.), ('1over3', 1/3), ('0.1', .1)]}
NAMES = {'lncc9': '1−LNCC² (9³)', 'gncc': '1−GNCC', 'gncc2': '1−GNCC²'}
COLORS = {'lncc9': '#0072B2', 'gncc': '#D55E00', 'gncc2': '#009E73'}
MARKERS = {'lncc9': 's', 'gncc': 'o', 'gncc2': '^'}
STEM = 'fedda_three_ncc_stdlogj_arc_2x3'


def read(path):
    return json.loads(Path(path).read_text())


def accepted_hashes(root):
    certificate = read(root/'ACCEPTED.json')
    if certificate.get('complete') is not True or not (root/'DONE').is_file():
        raise RuntimeError('Require accepted complete reference: '+str(root))
    hashes = {str(root/name): digest for name, digest in certificate['outputs_sha256'].items()}
    hashes[str(root/'ACCEPTED.json')] = g.sha256(root/'ACCEPTED.json')
    hashes[str(root/'DONE')] = g.sha256(root/'DONE')
    g.verify_hashes(hashes)
    return hashes


def reference_roots(baseline_root):
    artifact = Path(baseline_root).parents[1]
    return (artifact/'repo_experiments/oasis_seed3_first100_fedda_gncc_vs_lncc_v1',
            artifact/'repo_experiments/oasis_seed3_first100_sitreg_fedda_ncc_foreground_three_arc_v1')


def reference_hashes(baseline_root):
    signed, foreground = reference_roots(baseline_root)
    return {**accepted_hashes(signed), **accepted_hashes(foreground)}


def reference_rows(baseline_root):
    signed, foreground = reference_roots(baseline_root)
    data = read(signed/'results.json')
    arc.validated_means(data)  # Historical nine-setting validator stays unchanged.
    records = [r for r in read(foreground/'per_case.json') if r['model'] == 'fedda']
    if len(records) != 900:
        raise RuntimeError('Require 900 foreground reference records')
    mapping = {(r['loss'], r['lambda_key'], r['rank']): r for r in records}
    if len(mapping) != 900:
        raise RuntimeError('Duplicate reference')
    for row in data['per_case']:
        other = mapping[row['loss'], row['lambda_key'], row['rank']]
        for key in ('fixed_id', 'moving_id', 'alpha', *CORE):
            if row[key] != other[key]:
                raise RuntimeError('Foreground reference disagrees with accepted core metric')
    return [dict(r) for r in records], data


def summarize(rows, ranks=range(100)):
    expected = {(loss, key) for loss, group in SERIES.items() for key, _ in group}
    if {(r['loss'], r['lambda_key']) for r in rows} != expected:
        raise RuntimeError('Require exactly 13 approved settings')
    pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)
    result = []
    for loss, groups in SERIES.items():
        for key, alpha in groups:
            group = [r for r in rows if r['loss'] == loss and r['lambda_key'] == key and r['rank'] in ranks]
            if sorted(r['rank'] for r in group) != list(ranks):
                raise RuntimeError('Incomplete or duplicate comparison cohort')
            for r in group:
                if (r['fixed_id'], r['moving_id']) != pairs[r['rank']] or r['alpha'] != alpha:
                    raise RuntimeError('Mismatched case/lambda')
            row = dict(loss=loss, lambda_key=key, alpha=alpha, n=len(group),
                       historical=loss == 'lncc9' and key == '1')
            for metric in METRICS:
                values = [r[metric] for r in group]
                if not all(math.isfinite(v) for v in values):
                    raise RuntimeError('Nonfinite comparison metric')
                row[metric+'_mean'], row[metric+'_std'] = statistics.mean(values), statistics.stdev(values)
            result.append(row)
    return result


def build_figure(means):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D
    lookup = {(r['loss'], r['lambda_key']): r for r in means}
    if len(means) != 13 or set(lookup) != {(l, k) for l, group in SERIES.items() for k, _ in group}:
        raise ValueError('Plot requires all thirteen settings')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), sharex=True, sharey='col')
    for ri in range(2):
        for ci, (metric, title) in enumerate(arc.Y):
            ax = axes[ri, ci]
            for loss, series in SERIES.items():
                color = COLORS[loss]
                for (left, _), (right, _) in zip(series, series[1:]):
                    historical = loss == 'lncc9' and '1' in (left, right)
                    ax.plot([lookup[loss, k][arc.X+'_mean'] for k in (left, right)],
                            [lookup[loss, k][metric+'_mean'] for k in (left, right)],
                            color=color, lw=2.1, ls='--' if historical else '-', zorder=3)
                for key, _ in series:
                    r = lookup[loss, key]
                    x, y = r[arc.X+'_mean'], r[metric+'_mean']
                    if ri:
                        ax.add_patch(Ellipse((x, y), 2*r[arc.X+'_std'], 2*r[metric+'_std'],
                                             facecolor=color, edgecolor=color, alpha=.13, lw=1.2, zorder=1))
                    ax.plot(x, y, marker='D' if r['historical'] else MARKERS[loss], ls='none',
                            color=color, markerfacecolor='white' if r['historical'] else color, ms=6.5, zorder=4)
                    label = '1/3' if key == '1over3' else key
                    ax.annotate('λ='+label+('*' if r['historical'] else ''), (x, y),
                                xytext={'lncc9': (5, -15), 'gncc': (5, 9), 'gncc2': (-26, 15)}[loss],
                                textcoords='offset points', color=color, fontsize=8, zorder=5)
            ax.set_title(f"{'abcdef'[ri*3+ci]}) {title}", loc='left')
            ax.set_xlabel('Foreground SD(log J) ↓'); ax.set_ylabel(title)
            ax.set_xlim(*fg.limits(means, arc.X)); ax.set_ylim(*fg.limits(means, metric))
            ax.grid(alpha=.18); ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelbottom=True)
    handles = [Line2D([], [], color=COLORS[l], marker=MARKERS[l], lw=2, label=NAMES[l]) for l in SERIES]
    handles.append(Line2D([], [], color=COLORS['lncc9'], marker='D', markerfacecolor='white',
                          ls='--', label='LNCC² λ=1* historical'))
    fig.suptitle('FeDDA-1MF · three NCC losses · OASIS first100', fontsize=19, y=.985)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .948), ncol=4, frameon=False)
    fig.text(.5, .89, 'TOP: mean curves only', ha='center', fontsize=12)
    fig.text(.5, .485, 'BOTTOM: same curves with ±1 across-case SD ellipses', ha='center', fontsize=12)
    fig.text(.5, .04, '100 paired cases per point · Marginal sample SD, not confidence/covariance regions · Validation-selected checkpoints', ha='center', fontsize=9)
    fig.text(.5, .018, 'Fixed raw image > 0; canonical crop2; positive J for log statistics. Equal λ does not mean equal achieved regularity.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.06, right=.985, bottom=.12, top=.85, wspace=.26, hspace=.51)
    return fig, axes


def plot(means, out):
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, _ = build_figure(means)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out/f'{STEM}.{ext}', dpi=200, facecolor='white')
        plt.close(fig)


def table(means):
    lines = ['| Loss | λ | Dice ± SD | HD95 mm ± SD | ASD mm ± SD | FG stdlogJ ± SD | FG folding % ± SD | FG diffusion ± SD | Full diffusion ± SD |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in means:
        name = NAMES[row['loss']]+(' historical' if row['historical'] else '')
        key = '1/3' if row['lambda_key'] == '1over3' else row['lambda_key']
        lines.append(f'| {name} | {key} | '+' | '.join(f"{row[m+'_mean']:.6g} ± {row[m+'_std']:.6g}" for m in METRICS)+' |')
    return lines


def augment_diffusion(rows, data_root, out, provenance):
    import nibabel as nib
    import numpy as np
    directory = out/'diffusion_cases'
    directory.mkdir(exist_ok=False)
    # Read one mask per worker rather than retaining all 65 masks.
    def measure(row):
        path = data_root/'test'/f"img{row['fixed_id']}.nii.gz"
        mask, mask_sha = fg.load_mask(path)
        field = np.ascontiguousarray(nib.load(row['field_path']).dataobj, dtype=np.float32)
        if field.shape != (3, *g.SHAPE) or not np.isfinite(field).all():
            raise ValueError('Invalid field for diffusion')
        diffusion = fg.diffusion(field, mask)
        if not math.isclose(diffusion['diffusion_l2_full'], row['reference_diffusion_l2_full'],
                            rel_tol=2e-6, abs_tol=1e-8):
            raise ValueError('Native/posthoc full-volume diffusion mismatch')
        result = {**row, **diffusion, 'mask_sha256': mask_sha}
        c.write_json(directory/f"{row['lambda_key']}_rank{row['rank']:03d}.json", result)
        return result, str(path), mask_sha
    with ThreadPoolExecutor(max_workers=4) as pool:
        measured = list(pool.map(measure, [r for r in rows if r['loss'] == 'gncc2']))
    provenance.update({path: digest for _, path, digest in measured})
    return [r for r in rows if r['loss'] != 'gncc2']+[r for r, _, _ in measured]


def paired_rows(rows):
    mapping = {(r['loss'], r['lambda_key'], r['rank']): r for r in rows}
    output = []
    for left, right in (('gncc2', 'gncc'), ('gncc2', 'lncc9'), ('gncc', 'lncc9')):
        for key, alpha in g.LAMBDAS.items():
            for rank in range(100):
                a, b = mapping[left, key, rank], mapping[right, key, rank]
                output.append(dict(loss=left, reference_loss=right, lambda_key=key, alpha=alpha, rank=rank,
                    fixed_id=a['fixed_id'], moving_id=a['moving_id'],
                    **{'delta_'+m: a[m]-b[m] for m in METRICS}))
    return output


def write_report(train_root, eval_root, baseline_root, out, reference_hashes):
    if (out/'ACCEPTED.json').exists() or (out/'DONE').exists():
        raise FileExistsError('Preserve accepted report')
    g.verify_hashes(reference_hashes)
    rows, prior = reference_rows(baseline_root)
    labels, training, provenance, data_roots = [], [], dict(reference_hashes), set()
    expected_pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)[:100]
    for key, alpha in g.LAMBDAS.items():
        train, root = train_root/g.run_name(key), eval_root/g.run_name(key)
        for directory in (train, root):
            if not (directory/'COMPLETED').is_file() or (directory/'FAILED.txt').exists():
                raise RuntimeError('Incomplete GNCC train/evaluation')
        tm, tr, em, summary = read(train/'manifest.json'), read(train/'training_summary.json'), read(root/'manifest.json'), read(root/'summary.json')
        for value in (tm, tr, em, summary):
            g.require_protocol(value)
        g.verify_hashes(tm['source_sha256']); g.verify_hashes(em['source_sha256']); g.verify_hashes(em['native_source_sha256'])
        if (tm['model_key'], tm['updates_per_pair'], tr['epoch'], tr['adam_updates']) != ('fedda_1mf', 2, 200, 20000):
            raise RuntimeError('Wrong FeDDA training/update protocol')
        if em['alpha'] != alpha or em['variant'] != g.EVAL_VARIANT or em['smoke']:
            raise RuntimeError('Wrong evaluation identity')
        if (em['foreground'], em['jacobian'], em['asd_definition']) != (
                'fixed_raw_image_positive', 'canonical_LUMIR_crop2_positive_J_log_std',
                'pooled_surfel_area_weighted_bidirectional_mean'):
            raise RuntimeError('Changed canonical metric definitions')
        if summary['case_count'] != 100 or summary['label_rows'] != 3500 or summary['status'] != 'complete':
            raise RuntimeError('Incomplete detailed evaluation')
        if em['best_epoch'] != tr['best_epoch'] or em['best_val_dice'] != tr['best_val_dice']:
            raise RuntimeError('Best-validation checkpoint disagreement')
        g.verify_hashes({em['checkpoint']: em['checkpoint_sha256']})
        provenance[em['checkpoint']] = em['checkpoint_sha256']
        cases, details = read(root/'per_case.json'), read(root/'per_label_per_case.json')
        if sorted(r['rank'] for r in cases) != list(range(100)) or len(details) != 3500:
            raise RuntimeError('Missing or duplicate cases/labels')
        for case in cases:
            rank = case['rank']
            if (case['fixed_id'], case['moving_id']) != expected_pairs[rank] or case['direction'] != 'moving_to_fixed':
                raise RuntimeError('Case identity/direction mismatch')
            if case['alpha'] != alpha or case['checkpoint_sha256'] != em['checkpoint_sha256']:
                raise RuntimeError('Case checkpoint/lambda mismatch')
            directory = root/'cases'/f"rank{rank:03d}_{case['fixed_id']}_{case['moving_id']}"
            saved = read(directory/'metrics.json')
            current = [r for r in details if r['rank'] == rank]
            if saved['summary'] != case or saved['per_label'] != current or sorted(r['label_id'] for r in current) != list(range(1, 36)) or not (directory/'COMPLETED').is_file():
                raise RuntimeError('Per-label/case completion mismatch')
            for source, label_metric in zip(SOURCE[:3], ('dice', 'hd95_mm', 'asd_symmetric_mm')):
                valid = [r[label_metric] for r in current if r['included_in_case_mean']]
                if not math.isclose(statistics.mean(valid), case[source], abs_tol=1e-12, rel_tol=1e-12):
                    raise RuntimeError('Label-to-case reconstruction failed')
            g.verify_hashes({case['field_path']: case['field_sha256']})
            provenance[case['field_path']] = case['field_sha256']
            row = {k: case[k] for k in ('rank', 'fixed_id', 'moving_id')}
            row.update(loss='gncc2', lambda_key=key, alpha=alpha, method=f'FeDDA-1MF GNCC² λ{key}',
                       **{m: case[s] for m, s in zip(METRICS, SOURCE)})
            row.update(field_path=case['field_path'], field_sha256=case['field_sha256'],
                       reference_diffusion_l2_full=case['diffusion_l2_full'],
                       inference_seconds=case['inference_seconds'], physical_gpu=case['physical_gpu'],
                       best_epoch=case['best_epoch'], historical=False, model='fedda')
            rows.append(row)
        for source in SOURCE:
            values = [r[source] for r in cases]
            if not math.isclose(statistics.mean(values), summary['metrics'][source]['mean'], abs_tol=1e-12, rel_tol=1e-12) or not math.isclose(statistics.stdev(values), summary['metrics'][source]['sample_sd'], abs_tol=1e-12, rel_tol=1e-12):
                raise RuntimeError('Saved cohort summary mismatch')
        data_roots.add(tm['arguments']['data_root'])
        labels.extend({**r, 'loss': 'gncc2', 'lambda_key': key} for r in details)
        for name in ('manifest.json', 'summary.json', 'per_case.json', 'per_label_per_case.json'):
            provenance[str(root/name)] = g.sha256(root/name)
        training.append(dict(loss='gncc2', lambda_key=key, alpha=alpha, best_epoch=tr['best_epoch'], best_val_dice=tr['best_val_dice'],
                             elapsed_seconds=tr['elapsed_seconds'], gpu=tm['physical_gpu'],
                             initial_state_sha256=read(train/'initialization.json')['state_sha256'],
                             pair_stream_sha256=tr['pair_stream_sha256']))
    if len(rows) != 1300 or len(labels) != 14000 or len(data_roots) != 1:
        raise RuntimeError('Incomplete three-loss cohort')
    if len({r['initial_state_sha256'] for r in training}) != 1 or len({r['pair_stream_sha256'] for r in training}) != 1:
        raise RuntimeError('Independent trainings diverged from common initialization/pair stream')
    for old in prior['training']:
        if old['initial_state_sha256'] != training[0]['initial_state_sha256'] or old['pair_stream_sha256'] != training[0]['pair_stream_sha256']:
            raise RuntimeError('Squared/signed training seeds or pair streams differ')
    rows = augment_diffusion(rows, Path(next(iter(data_roots))), out, provenance)
    for row in rows:
        provenance[row['field_path']] = row['field_sha256']
    g.verify_hashes(provenance)
    means, first10 = summarize(rows), summarize(rows, range(10))
    selected = max(training, key=lambda r: (r['best_val_dice'], -r['alpha']))
    paired = paired_rows(rows)
    c.write_csv(out/'mean_std.csv', means); c.write_csv(out/'first10_mean_std.csv', first10)
    c.write_csv(out/'per_case.csv', rows); c.write_csv(out/'paired_differences.csv', paired)
    c.write_csv(out/'training_summary.csv', training); c.write_csv(out/'per_label_per_case.csv', labels)
    c.write_json(out/'per_label_per_case.json', labels)
    data = dict(complete=True, means=means, first10=first10, per_case=rows, paired=paired,
                training=training, reference_training=prior['training'],
                validation_selected=selected, input_sha256=provenance, protocol=g.PROTOCOL)
    c.write_json(out/'results.json', data)
    plot(means, out)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    history_rows = []
    signed_train = Path(baseline_root).parents[1]/'runs/oasis_fedda_gncc_diffusion_training_lambda_sweep_v1'
    for loss, root in (('gncc2', train_root), ('gncc', signed_train)):
        for key in g.LAMBDAS:
            name = g.run_name(key) if loss == 'gncc2' else f'fedda_1mf_gncc_lambda_{key}_seed42'
            path = root/name/'epochs.jsonl'
            epochs = [json.loads(line) for line in path.read_text().splitlines()]
            if [r['epoch'] for r in epochs] != list(range(1, 201)):
                raise ValueError('Incomplete validation history')
            history_rows.extend(dict(loss=loss, lambda_key=key, epoch=r['epoch'],
                                     validation_dice=r['validation']['dice']) for r in epochs)
            ax.plot([r['epoch'] for r in epochs], [r['validation']['dice'] for r in epochs],
                    ls='-' if loss == 'gncc2' else '--', label=f"{NAMES[loss]} λ{key}")
            provenance[str(path)] = g.sha256(path)
    ax.set(xlabel='Epoch', ylabel='Native forward validation Dice'); ax.grid(alpha=.2)
    ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    fig.savefig(out/'validation_curves.png', dpi=200); plt.close(fig)
    c.write_csv(out/'validation_history.csv', history_rows)
    lines = ['# FeDDA three NCC losses', '',
             'Four new squared-GNCC trainings; 400 new test cases and 14000 per-label records. Nine immutable reference settings.',
             'All points use the same shuffled seed3 first100 and validation-selected checkpoints.',
             'Native 200 epochs, 20000 sequential-directional Adam updates, LR1e-4; unmasked total voxel diffusion.',
             'Same correlation moment convention does not imply calibrated loss scale or matched regularity at equal lambda.',
             'GNCC versus GNCC² isolates squaring on global support; GNCC² versus LNCC² changes local/global statistics.',
             'LNCC lambda1 is historical, not a fresh matched-seed control. Single training seed and previously inspected test cohort.',
             '', f"GNCC² validation-selected lambda: {selected['lambda_key']}; Dice {selected['best_val_dice']:.8f}; epoch {selected['best_epoch']}.",
             f"Selected at searched boundary: {selected['alpha'] in (.1, 3.)}. No automatic range extension.",
             '', '## First100', '', *table(means), '', '## First10', '', *table(first10), '',
             'Paired differences cover all three contrasts at four common lambdas. Negative differences favor the new loss for distance/regularity metrics; positive differences favor Dice.',
             'Foreground diffusion restricts valid edges to two foreground endpoints and is not the unmasked training objective.',
             'FG means fixed raw image >0; Jacobian crop2, stdlogJ on positive J; inspect folding separately.',
             'Top mean curves; bottom marginal ±1 across-case sample SD, not confidence/covariance regions.',
             '', f'![Three NCC ARC]({STEM}.png)', '', '![Validation histories](validation_curves.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    c.write_json(out/'results.json', data)
    g.verify_hashes(reference_hashes)
    hashes = {p.name: g.sha256(p) for p in out.iterdir() if p.is_file() and p.name not in ('ACCEPTED.json', 'DONE')}
    c.write_json(out/'ACCEPTED.json', dict(complete=True, settings=13, cases=1300, new_trainings=4,
        new_cases=400, new_label_rows=14000, reference_cases=900, selection='validation_only', outputs_sha256=hashes))
    (out/'DONE').write_text('Four GNCC2 trainings; 400 detailed evaluations; 13-setting three-loss ARC accepted\n')
