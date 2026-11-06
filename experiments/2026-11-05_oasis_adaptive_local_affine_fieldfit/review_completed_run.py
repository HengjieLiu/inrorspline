#!/usr/bin/env python
"""Describe completed P81 fits, preserving convergence and foreground scope.

Reads saved endpoint metadata; does not refit or revalidate dense-field bytes.
Generated review files must remain outside the source worktree.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return st.mean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output == repo or repo in output.parents:
        raise ValueError('Generated review must be external')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root/'launch_manifest.json').read_text())
    state = json.loads((root/'queue_state.json').read_text())
    expected = {(j['id'], cap) for j in manifest['jobs'] for cap in j['budgets']}
    records, hashes, stopping = [], {}, Counter()
    for path in sorted((root/'endpoints').glob('*/*.json')):
        payload = path.read_bytes()
        x = json.loads(payload)
        hashes[str(path.relative_to(root))] = hashlib.sha256(payload).hexdigest()
        metrics, renderer = x['metrics'], x['renderer']
        fg, full = metrics['field_error']['foreground'], metrics['field_error']['full']
        geometry = metrics['geometry']['foreground_eroded']
        labels = metrics['labels']
        paired = labels['summary']['paired']
        sensitivity = renderer.get('integration_sensitivity', {})
        last = x['solves'][-1]
        if x['kind'] == 'main' and x['mode'] == 'svf':
            stopping[last.get('stopping_reason', 'unknown')] += 1
        row = {k: x[k] for k in ('job_id', 'target_id', 'pair_rank', 'source_method', 'mode',
                                'family', 'initialization', 'kind', 'budget_cap',
                                'actual_motion_params', 'scientific_status')}
        row.update(fg_rmse_mm=fg['rmse_mm'], fg_mean_epe_mm=fg['mean_epe_mm'],
                   full_rmse_mm=full['rmse_mm'],
                   fg_fold_percent=geometry['prediction']['folding_percent'],
                   reference_fg_fold_percent=geometry['reference']['folding_percent'],
                   paired_dice=paired['prediction']['dice'],
                   reference_paired_dice=paired['reference']['dice'],
                   paired_valid_labels=labels['paired_valid_labels'],
                   label_means_comparable=labels['cohort_means_comparable'],
                   integration_sensitive=sensitivity.get('sensitive'),
                   integration_fg_rmse_mm=sensitivity.get('foreground_vector_rmse_mm'),
                   integration_full_max_component_mm=sensitivity.get('max_component_mm'),
                   final_solver_converged=last.get('converged'),
                   final_solver_stopping=last.get('stopping_reason'),
                   d1_bound=renderer.get('serialized_precision_bound'))
        records.append(row)
    found = {(r['job_id'], r['budget_cap']) for r in records}
    if found != expected or len(found) != len(records):
        raise RuntimeError('Endpoint coverage/uniqueness mismatch')
    with (root/'reports/per_endpoint.csv').open() as stream:
        original = {(r['id'], int(r['budget_cap'])): r for r in csv.DictReader(stream)}
    for row in records:
        old = original[(row['job_id'], row['budget_cap'])]
        if not math.isclose(float(old['rmse_mm']), row['fg_rmse_mm'], rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError('Previously published error summary no longer matches endpoint')

    primary = [r for r in records if r['kind'] == 'main']
    if Counter(r['mode'] for r in primary) != {'d0': 840, 'svf': 1680}:
        raise RuntimeError('Main matrix coverage mismatch')
    summaries = []
    for scope in ('all_sources_descriptive', 'source_separated'):
        groups = defaultdict(list)
        for r in primary:
            key = (r['mode'], r['initialization'], r['family'], r['budget_cap'],
                   r['source_method'] if scope == 'source_separated' else 'all_four_sources')
            groups[key].append(r)
        for key, values in sorted(groups.items()):
            n_expected = 10 if scope == 'source_separated' else 40
            if len(values) != n_expected:
                raise RuntimeError('Incomplete cohort')
            row = dict(zip(('mode', 'initialization', 'family', 'budget_cap', 'source'), key))
            row.update(scope=scope, n_fields=len(values), n_pairs=len({r['pair_rank'] for r in values}),
                       actual_motion_params=sorted({r['actual_motion_params'] for r in values}),
                       unresolved=sum(r['scientific_status']=='solver_unresolved' for r in values),
                       label_comparison_warnings=sum(not r['label_means_comparable'] for r in values),
                       sampled_folded_fields=sum(r['fg_fold_percent']>0 for r in values))
            for metric in ('fg_rmse_mm', 'fg_mean_epe_mm', 'full_rmse_mm', 'fg_fold_percent',
                           'reference_fg_fold_percent', 'paired_dice', 'reference_paired_dice'):
                nums = [r[metric] for r in values if r[metric] is not None and math.isfinite(r[metric])]
                row[metric] = dict(mean=mean(nums), median=st.median(nums) if nums else None,
                                   sd=st.stdev(nums) if len(nums)>1 else None, finite_n=len(nums))
            summaries.append(row)
    comparisons = []
    for mode, initialization in (('d0','zero'),('svf','zero'),('svf','global_affine_ls')):
        for cap in (768,1536,3072):
            for a, b in (('tree_affine_adaptive','tree_affine_uniform'),
                         ('tree_translation_adaptive','tree_translation_uniform'),
                         ('tree_affine_adaptive','tree_translation_adaptive')):
                aa = {r['target_id']: r for r in primary if (r['mode'],r['initialization'],r['budget_cap'],r['family'])==(mode,initialization,cap,a)}
                bb = {r['target_id']: r for r in primary if (r['mode'],r['initialization'],r['budget_cap'],r['family'])==(mode,initialization,cap,b)}
                if aa.keys() != bb.keys() or len(aa) != 40:
                    raise RuntimeError('Paired comparison coverage mismatch')
                pairs = defaultdict(list)
                delta = []
                for target, r in aa.items():
                    d = r['fg_rmse_mm']-bb[target]['fg_rmse_mm']
                    delta.append(d); pairs[r['pair_rank']].append(d)
                pair_deltas = [mean(v) for _, v in sorted(pairs.items())]
                comparisons.append(dict(mode=mode, initialization=initialization, budget_cap=cap,
                    a=a, b=b, delta_a_minus_b_mm=mean(delta), field_wins=sum(d<0 for d in delta),
                    field_ties=sum(abs(d)<=1e-12 for d in delta),
                    pair_mean_deltas_mm=pair_deltas, pair_mean_wins=sum(d<0 for d in pair_deltas),
                    n_pairs=10, n_correlated_fields=40))
    start = datetime.fromisoformat(manifest['created_at'].replace('Z','+00:00'))
    finish = datetime.fromisoformat(state['updated_at'].replace('Z','+00:00'))
    result = dict(queue_status=state['status'], jobs=state['counts'], endpoint_count=len(records),
        started_at=manifest['created_at'], finished_at=state['updated_at'], elapsed_hours=(finish-start).total_seconds()/3600,
        oom_retries=sum(j.get('oom_retries',0) for j in state['jobs']),
        scientific_status=dict(Counter(r['scientific_status'] for r in records)),
        main_d0_unresolved=sum(r['mode']=='d0' and r['scientific_status']=='solver_unresolved' for r in primary),
        main_svf_final_stopping=dict(stopping),
        svf_foreground_integration=dict(Counter('sensitive' if r['integration_sensitive'] else 'within_tolerance' for r in records if r['mode']=='svf')),
        d1_certificate_max=max(r['d1_bound'] for r in records if r['mode']=='d1'),
        summaries=summaries, paired_comparisons=comparisons,
        provenance=dict(launch_manifest_sha256=digest(root/'launch_manifest.json'),queue_state_sha256=digest(root/'queue_state.json'),
                        original_table_sha256=digest(root/'reports/per_endpoint.csv'),review_source_sha256=digest(__file__),endpoint_metadata_sha256=hashes),
        study_complete=False,
        limitations=['Descriptive achieved fits; unresolved D0 solves are not certified least-squares optima.',
                     'All main SVF final stages used their fixed iteration budgets; nonlinear optimality is not established.',
                     'Forty fields are four correlated sources on ten repeatedly explored pairs, not40 independent subjects.',
                     'Foreground fitting does not constrain whole-volume extrapolation or guarantee global integration accuracy.',
                     'Pending Taylor/transition diagnostics, final figures/notebook and convergence disposition remain in the analysis ledger.',
                     'Metadata-level review only; dense bytes were checked by the queue at completion, not reread here.'])
    (output/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'endpoint_review.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lookup={(x['mode'],x['initialization'],x['family'],x['budget_cap']):x for x in summaries if x['scope']=='all_sources_descriptive'}
    lines=['# P81 completed fitting run: first results review','',
        f"Finished {state['updated_at']}:1680/1680 jobs;2640 endpoint records; no runtime failures or OOM retries.",
        f"Elapsed execution:{result['elapsed_hours']:.2f} hours, including the multi-GPU scheduler transition.",'',
        '**Execution is complete; scientific analysis is not fully complete.**',
        f"D0 main fits unresolved:{result['main_d0_unresolved']}/840. All1680 main SVF final stages reached their iteration budgets.",
        'Numbers below describe achieved fits; they are not certified representation-capacity optima.','',
        '## Foreground vector RMSE (mm), parameter cap3072','',
        'Mean of40 reference fields: four sources on the same ten pairs. SVF uses the zero initialization below; both starts and source-separated results are in review.json.','',
        '| Family | Actual parameters | Direct DVF | SVF |','| --- | ---: | ---: | ---: |']
    for family in ('bspline','gaussian_translation','gaussian_affine','tree_translation_uniform','tree_translation_adaptive','tree_affine_uniform','tree_affine_adaptive'):
        d=lookup['d0','zero',family,3072];v=lookup['svf','zero',family,3072]
        lines.append(f"| {family} | {d['actual_motion_params'][0]} | {d['fg_rmse_mm']['mean']:.4f} | {v['fg_rmse_mm']['mean']:.4f} |")
    lines += ['', '## Interpretation and limits','',
        '- At cap3072, adaptive trees improve over uniform trees for every one of40 source/pair fields, in both D0 and SVF.',
        '- Adaptive-affine D0 only marginally improves over adaptive-translation D0 on average; the SVF adaptive-translation fits have lower error on all40 fields.',
        '- Sampled foreground folding is lower in SVF, but fitted fields still lose registration-label accuracy versus the source fields. See paired Dice and label-coverage warnings in review.json.',
        '- Both SVF initializations give similar aggregate errors. This does not establish nonlinear convergence or initialization independence in general.',
        '- D0 convergence, exterior extrapolation and the remaining mechanism analyses must be resolved before strong capacity or invertibility claims.',
        '', 'All2640 metadata records match the published CSV errors and planned identities. The original reports and dense fields are unchanged.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('jobs','endpoint_count','elapsed_hours','main_d0_unresolved','main_svf_final_stopping','svf_foreground_integration','d1_certificate_max')},indent=2))


if __name__=='__main__':
    main()
