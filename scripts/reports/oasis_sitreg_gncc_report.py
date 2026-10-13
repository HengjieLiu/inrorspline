"""Strict first100 GNCC versus immutable LNCC SITReg comparison."""
from __future__ import annotations
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/oasis_alpha_sweep'))
import common as c
import sitreg_gncc_training as g
from run_oasis_sitreg_first100_eval import SUMMARY_METRICS, label_rows, stats

METRICS = ('dice', 'hd95', 'asd', 'log_jdet_std_fg', 'jdet_le0_pct_fg')
SOURCE = ('dice_mean', 'hd95_mean', 'asd_symmetric_mean', 'log_jdet_std_fg', 'jdet_le0_pct_fg')
OLD = {'SITReg λ=0.1': ('0.1', .1), 'SITReg λ=1/3': ('1over3', 1/3),
       'SITReg λ=1 (original)': ('1', 1.), 'SITReg λ=3': ('3', 3.), 'SITReg λ=10': ('10', 10.)}


def read(path):
    return json.loads(Path(path).read_text())


def summarize(rows, ranks=range(100)):
    output = []
    keys = [('gncc', key, alpha) for key, alpha in g.LAMBDAS.items()]+[('lncc9', key, alpha) for key, alpha in OLD.values()]
    for loss, key, alpha in keys:
        group = [r for r in rows if r['loss'] == loss and r['lambda_key'] == key and r['rank'] in ranks]
        if sorted(r['rank'] for r in group) != list(ranks):
            raise RuntimeError('Incomplete/duplicate comparison cohort')
        row = dict(loss=loss, lambda_key=key, alpha=alpha, n=len(group), historical=loss == 'lncc9' and key == '1')
        for metric in METRICS:
            values = [r[metric] for r in group]
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError('Nonfinite comparison metric')
            row[metric+'_mean'], row[metric+'_std'] = statistics.mean(values), statistics.stdev(values)
        output.append(row)
    return output


def plot(means, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    for rowidx in range(2):
        for ax, metric in zip(axes[rowidx], METRICS[:3]):
            for loss, color, label in [('gncc', '#D55E00', 'SITReg: global 1−NCC'), ('lncc9', '#0072B2', 'SITReg: local 1−NCC² (window9)')]:
                group = sorted([r for r in means if r['loss'] == loss], key=lambda r: r['alpha'])
                for row in group:
                    x, y = row['log_jdet_std_fg_mean'], row[metric+'_mean']
                    if rowidx == 0:
                        ax.add_patch(Ellipse((x, y), 2*row['log_jdet_std_fg_std'], 2*row[metric+'_std'], facecolor=color, edgecolor=color, alpha=.13))
                    key = '1/3' if row['lambda_key'] == '1over3' else row['lambda_key']
                    ax.annotate(key, (x, y), xytext=(4, 5), textcoords='offset points', fontsize=8, color=color)
                    if row['historical']:
                        ax.plot(x, y, marker='*', color=color, markersize=14)
                ax.plot([r['log_jdet_std_fg_mean'] for r in group], [r[metric+'_mean'] for r in group], '-o', color=color, label=label)
            ax.set_xlabel('Foreground SD(log J) ↓')
            ax.set_ylabel({'dice': 'Dice ↑', 'hd95': 'HD95 (mm) ↓', 'asd': 'Symmetric ASD (mm) ↓'}[metric])
            ax.set_title('With ±1 sample-SD ellipses' if rowidx == 0 else 'Lines only — identical operating points')
            ax.grid(alpha=.2)
    for col in range(3):
        for getter, setter in [('get_xlim', 'set_xlim'), ('get_ylim', 'set_ylim')]:
            lim = [getattr(ax, getter)() for ax in axes[:, col]]
            for ax in axes[:, col]:
                getattr(ax, setter)((min(v[0] for v in lim), max(v[1] for v in lim)))
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=2)
    fig.text(.5, .018, 'Seed3 first100; checkpoint selection by validation Dice. Star: historical LNCC λ1, not newly retrained.\n'
             'Ellipses: marginal across-case ±1 sample SD, not confidence regions. Positive-J stdlogJ; folding reported separately.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out/f'sitreg_gncc_vs_lncc_arc_2x3.{ext}', dpi=200)
    plt.close(fig)


def table(means):
    lines = ['| Loss | λ | Dice ± SD | HD95 mm ± SD | ASD mm ± SD | FG stdlogJ ± SD | FG folding % ± SD |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in means:
        name = row['loss']+(' (historical)' if row['historical'] else '')
        key = '1/3' if row['lambda_key'] == '1over3' else row['lambda_key']
        lines.append(f'| {name} | {key} | '+' | '.join(f"{row[m+'_mean']:.6f} ± {row[m+'_std']:.6f}" for m in METRICS)+' |')
    return lines


def write_report(train_root, eval_root, baseline_root, out, reference_hashes):
    g.verify_hashes(reference_hashes)
    baseline = read(baseline_root/'results.json')
    rows, labels, training, provenance = [], [], [], dict(reference_hashes)
    expected_pairs = c.parse_pair_file(c.DEFAULT_PAIR_FILE)[:100]
    for old in baseline['per_case']:
        if old['method'] in OLD:
            key, alpha = OLD[old['method']]
            if (old['fixed_id'], old['moving_id']) != expected_pairs[old['rank']]:
                raise RuntimeError('Historical pair mismatch')
            rows.append({**old, 'loss': 'lncc9', 'lambda_key': key, 'alpha': alpha})
    for key, alpha in g.LAMBDAS.items():
        train, root = train_root/g.run_name(key), eval_root/g.run_name(key)
        for folder in (train, root):
            if not (folder/'COMPLETED').is_file() or (folder/'FAILED.txt').exists():
                raise RuntimeError('Incomplete GNCC train/evaluation')
        tm, tr, em, summary = read(train/'manifest.json'), read(train/'training_summary.json'), read(root/'manifest.json'), read(root/'summary.json')
        for value in (tm, tr, em, summary):
            g.require_protocol(value)
        g.verify_hashes(tm['source_sha256']); g.verify_hashes(em['source_sha256'])
        if tr['epoch'] != 200 or tr['adam_updates'] != 10000 or em['alpha'] != alpha or em['variant'] != g.EVAL_VARIANT:
            raise RuntimeError('Wrong training/evaluation protocol')
        if summary['case_count'] != 100 or summary['label_rows'] != 3500 or summary['status'] != 'complete':
            raise RuntimeError('Incomplete detailed evaluation')
        if em['best_epoch'] != tr['best_epoch'] or em['best_val_dice'] != tr['best_val_dice']:
            raise RuntimeError('Checkpoint selection disagreement')
        g.verify_hashes({em['checkpoint']: em['checkpoint_sha256']})
        provenance[em['checkpoint']] = em['checkpoint_sha256']
        cases, details = read(root/'per_case.json'), read(root/'per_label_per_case.json')
        if sorted(r['rank'] for r in cases) != list(range(100)) or len(details) != 3500:
            raise RuntimeError('Missing/duplicate cases or labels')
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
                if not math.isclose(statistics.mean(valid), case[source], abs_tol=1e-12):
                    raise RuntimeError('Label-to-case reconstruction failed')
            g.verify_hashes({case['field_path']: case['field_sha256']})
            provenance[case['field_path']] = case['field_sha256']
            row = {k: case[k] for k in ('rank', 'fixed_id', 'moving_id')}
            row.update(loss='gncc', lambda_key=key, alpha=alpha, method=f'SITReg GNCC λ{key}',
                       **{m: case[s] for m, s in zip(METRICS, SOURCE)})
            rows.append(row)
        for metric, source in zip(METRICS, SOURCE):
            values = [r[source] for r in cases]
            if not math.isclose(statistics.mean(values), summary['metrics'][source]['mean'], abs_tol=1e-12) or not math.isclose(statistics.stdev(values), summary['metrics'][source]['sample_sd'], abs_tol=1e-12):
                raise RuntimeError('Saved cohort summary mismatch')
        labels.extend({**r, 'loss': 'gncc'} for r in details)
        for name in ('manifest.json', 'summary.json', 'per_case.json', 'per_label_per_case.json'):
            provenance[str(root/name)] = g.sha256(root/name)
        training.append({'lambda_key': key, 'alpha': alpha, 'best_epoch': tr['best_epoch'], 'best_val_dice': tr['best_val_dice'],
                         'elapsed_seconds': tr['elapsed_seconds'], 'gpu': tm['physical_gpu'],
                         'initial_state_sha256': read(train/'initialization.json')['state_sha256'],
                         'pair_stream_sha256': tr['pair_stream_sha256']})
    if len(rows) != 900 or len(labels) != 14000 or len({r['initial_state_sha256'] for r in training}) != 1 or len({r['pair_stream_sha256'] for r in training}) != 1:
        raise RuntimeError('Cohort or matched-training invariant failed')
    means, first10 = summarize(rows), summarize(rows, range(10))
    selected = max(training, key=lambda r: (r['best_val_dice'], -r['alpha']))
    paired = []
    for row in rows:
        if row['loss'] == 'gncc':
            ref = next(r for r in rows if r['loss'] == 'lncc9' and r['lambda_key'] == row['lambda_key'] and r['rank'] == row['rank'])
            paired.append({**row, **{f'delta_vs_LNCC_{m}': row[m]-ref[m] for m in METRICS}})
    c.write_csv(out/'mean_std.csv', means); c.write_csv(out/'first10_mean_std.csv', first10)
    c.write_csv(out/'per_case.csv', rows); c.write_csv(out/'paired_differences.csv', paired)
    c.write_csv(out/'training_summary.csv', training); c.write_csv(out/'per_label_per_case.csv', labels)
    c.write_json(out/'per_label_per_case.json', labels)
    c.write_json(out/'results.json', dict(complete=True, means=means, first10=first10, per_case=rows, paired=paired,
                                       training=training, validation_selected=selected, input_sha256=provenance, protocol=g.PROTOCOL))
    plot(means, out)
    # Validation curves only; never select epochs from test metrics.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for record in training:
        epochs = [json.loads(line) for line in (train_root/g.run_name(record['lambda_key'])/'epochs.jsonl').read_text().splitlines()]
        ax.plot([r['epoch'] for r in epochs], [r['validation']['dice'] for r in epochs], label=f"GNCC λ{record['lambda_key']}")
    ax.set(xlabel='Epoch', ylabel='Native 30-pair validation Dice'); ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(out/'validation_curves.png', dpi=200); plt.close(fig)
    lines = ['# SITReg global versus local NCC training', '',
             'Four fresh GNCC trainings;400 new case evaluations and14000 new label rows. Five immutable LNCC reference settings.',
             'Same seed3 first100; means and sample SD weight cases equally. LNCC λ1 is historical, not a new matched training.',
             'Locality and squaring both change. Equal λ does not mean equal effective regularization.', '',
             f"GNCC lambda selected by best validation Dice: {selected['lambda_key']} (Dice={selected['best_val_dice']:.8f}, epoch={selected['best_epoch']}).",
             f"Validation selection at searched boundary: {selected['alpha'] in (min(g.LAMBDAS.values()), max(g.LAMBDAS.values()))}. No automatic range extension.",
             '', '## First100', '', *table(means), '', '## First10 (previously exploratory)', '', *table(first10), '',
             'Per-case paired differences and all per-label records are in the companion CSV/JSON files.',
             'Foreground is fixed raw image>0; stdlogJ excludes nonpositive Jacobians. Inspect folding alongside stdlogJ.',
             'Single-seed experiments on a previously explored test cohort; no statistical superiority claim.', '',
             '![ARC comparison](sitreg_gncc_vs_lncc_arc_2x3.png)', '', '![Validation](validation_curves.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    hashes = {p.name: g.sha256(p) for p in out.iterdir() if p.is_file() and p.name not in ('ACCEPTED.json', 'DONE')}
    c.write_json(out/'ACCEPTED.json', dict(complete=True, new_trainings=4, new_cases=400, new_label_rows=14000,
                                        reference_cases=500, selection='validation_only', outputs_sha256=hashes))
    (out/'DONE').write_text('All train/eval/report gates passed\n')
