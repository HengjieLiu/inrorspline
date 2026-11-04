"""Coverage-aware P74 report; missing configurations never become cohort means."""
from collections import defaultdict
from dataclasses import asdict
import argparse
import json
from pathlib import Path

import run_oasis_bspline_svf_sweep_recovery as r
from report_oasis_bspline_svf_sweep import METRICS, RESOURCES, summarize, fmt


def generate(jobs, rows, manifest, out):
    out.mkdir(parents=True, exist_ok=True)
    lookup = {x['run_name']: x for x in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate accepted configuration')
    groups, coverage, flat, stages, sensitivity, paired = defaultdict(list), [], [], [], [], []
    for job in jobs:
        spec = r.s.Spec(**job['spec'])
        base = {**job['spec'], 'run_name': spec.name(), 'status': job['status'],
                'attempt': job['attempt'], 'gpu': job.get('gpu'),
                **{k: manifest['cases'][str(spec.case_rank)][k] for k in ('fixed_id', 'moving_id')}}
        coverage.append({**base, 'failure': job.get('failure'), 'run_dir': job.get('out')})
        if job['status'] != 'complete':
            continue
        row = lookup[spec.name()]
        item = {**base, **{k: row.get(k) for k in (*METRICS, *RESOURCES)}, 'run_dir': row['run_dir']}
        flat.append(item)
        groups[(spec.schedule, spec.regularize, spec.alpha)].append(item)
        for block in row.get('blocks', []):
            stages.append({**base, 'cps': block['cps'], **block['optimizer']})
        for diag in row.get('integration_diagnostics', []):
            sensitivity.append({**base, **diag})
    summaries, group_coverage = [], []
    expected = sorted({(j['spec']['schedule'], j['spec']['regularize'], j['spec']['alpha']) for j in jobs})
    for schedule, reg, alpha in expected:
        items = groups[(schedule, reg, alpha)]
        ranks = [x['case_rank'] for x in items]
        complete = len(ranks) == 10 and set(ranks) == set(range(10))
        base = {'schedule': schedule, 'regularize': reg, 'alpha': alpha}
        group_coverage.append({**base, 'accepted_n': len(ranks), 'complete_first10': complete,
                               'missing_ranks': sorted(set(range(10))-set(ranks))})
        if complete:
            summaries.append({**base, **summarize(items)})
    by_spec = {(x['schedule'], x['regularize'], x['alpha'], x['case_rank']): x for x in flat}
    ref_groups = defaultdict(list)
    for x in manifest.get('references', []):
        ref_groups[x['method']].append(x)
    for x in flat:
        rank, schedule, reg, alpha = (x[k] for k in ('case_rank', 'schedule', 'regularize', 'alpha'))
        opponents = [(name, next((v for v in vals if v['case_rank'] == rank), None))
                     for name, vals in ref_groups.items()]
        opponents += [('same setting lambda1', by_spec.get((schedule, reg, 1., rank))),
                      ('other penalty same lambda', by_spec.get((schedule, 'displacement' if reg == 'velocity' else 'velocity', alpha, rank))),
                      ('other depth same lambda', by_spec.get(('8421' if schedule == '842' else '842', reg, alpha, rank)))]
        for name, ref in opponents:
            paired.append({'run_name': x['run_name'], 'case_rank': rank, 'reference': name,
                           'pair_available': ref is not None,
                           **{f'delta_{k}': x[k]-ref[k] if ref is not None else None for k in METRICS}})
    reference_summaries = [{'method': name, **summarize(values)} for name, values in ref_groups.items()
                           if len(values) == 10 and {x['case_rank'] for x in values} == set(range(10))]
    failures = [x for x in coverage if x['status'] == 'failed_deferred']
    payload = {'all_scientifically_accepted': len(rows) == len(jobs), 'accepted': len(rows),
               'total': len(jobs), 'deferred_failures': failures, 'coverage': coverage,
               'group_coverage': group_coverage, 'per_case': flat, 'summary': summaries,
               'references': manifest.get('references', []), 'reference_summary': reference_summaries,
               'paired_differences': paired, 'stage_diagnostics': stages,
               'integration_sensitivity': sensitivity, 'provenance': manifest}
    for name, values in [('coverage', coverage), ('group_coverage', group_coverage), ('per_case', flat),
                         ('summary', summaries), ('failures', failures), ('paired_differences', paired),
                         ('stage_diagnostics', stages), ('integration_sensitivity', sensitivity)]:
        r.c.write_csv(out/f'{name}.csv', values)
    r.atomic_json(out/'results.json', payload)
    text = ['# P74 queue outcome (coverage-aware)', '',
            f'Accepted {len(rows)}/{len(jobs)}; deferred failures {len(failures)}. '
            'Queue completion is not scientific acceptance. Exploratory test-case tuning.', '',
            'Means and sample SD below require all ten unique cases. Incomplete groups are listed '
            'in group_coverage.csv, without partial-cohort means. Timings reflect mixed concurrency '
            'and other workloads. All canonical metrics and numerical checks are unchanged.', '',
            '| Schedule / penalty | λ | Dice | HD95 | ASD | FG SD(log J) | FG folding % | Optimizer s | Worker s | Peak allocated GiB | Peak reserved GiB |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in summaries:
        text.append(f"| {x['schedule']} / {x['regularize']} | {x['alpha']:g} | " +
                    ' | '.join(fmt(x, k) for k in (*METRICS, 'optimizer_sec', 'worker_sec', 'peak_cuda_gib', 'peak_reserved_gib'))+' |')
    for x in reference_summaries:
        text.append(f"| {x['method']} | n/a | "+' | '.join(fmt(x, k) for k in METRICS)+' | n/a | n/a | n/a | n/a |')
    text += ['', '## Deferred configurations', '']
    for x in failures:
        text.append(f"- `{x['run_name']}`: {x.get('failure')}")
    (out/'report.md').write_text('\n'.join(text)+'\n')
    return payload


def main(root):
    run = root/'runs'/r.s.VARIANT
    folder = run/'continue_on_error_v1'
    manifest = r.c.read_json(folder/'launch_manifest.json')
    r.assert_manifest(manifest)
    state = r.c.read_json(folder/'state.json')
    rows = [r.validate(Path(j['out']), manifest, r.s.Spec(**j['spec']))
            for j in state['jobs'] if j['status'] == 'complete']
    out = root/'repo_experiments'/r.s.VARIANT/'continue_on_error_v1'
    generate(state['jobs'], rows, manifest, out)
    r.atomic_json(folder/'REPORT_FINISHED.json', {'at': r.c.now_text(), 'out': str(out), 'accepted': len(rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, default=Path('/database/inrorspline'))
    main(parser.parse_args().artifact_root)
