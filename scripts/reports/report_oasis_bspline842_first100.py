#!/usr/bin/env python
"""P79 complete-cohort accuracy/regularity/resource tables and 2x3 ARC."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/oasis_alpha_sweep'))
import oasis_bspline842_first100 as e

RESOURCES=('optimizer_sec','worker_sec','audit_sec','actual_adam_updates','peak_allocated_gib','peak_reserved_gib')
COHORTS={'first10':range(10),'additional90':range(10,100),'first100':range(100)}


def summarize(rows, ranks):
    if sorted(r['case_rank'] for r in rows)!=list(ranks):
        raise ValueError('Only a complete, unique cohort can be summarized')
    result={'n':len(rows)}
    for key in (*e.METRICS,*RESOURCES):
        vals=[float(r[key]) for r in rows if r.get(key) is not None]
        if key in e.METRICS and len(vals)!=len(rows):raise ValueError('Missing primary metric')
        if any(not math.isfinite(v) for v in vals):raise ValueError('Nonfinite statistic')
        # Explicit n for resources: historical reserved-memory/audit timing may
        # be unavailable, and must not become invented zeros.
        result[key+'_n']=len(vals)
        for suffix,value in [('mean',statistics.mean(vals) if vals else None),
                             ('sd',statistics.stdev(vals) if len(vals)>1 else None),
                             ('min',min(vals) if vals else None),('max',max(vals) if vals else None)]:
            result[key+'_'+suffix]=value
    return result


def flat(row,spec,kind):
    p=Path(row['run_dir']);audit=e.c.read_json(p/'independent_audit.json') if (p/'independent_audit.json').exists() else {}
    if spec.family=='displacement':
        blocks=[{'cps':x['cps'],'optimizer':x} for x in row['stages']]
        allocated=max(x['peak_cuda_gib'] for x in row['stages']);reserved=None
    else:
        blocks=row['blocks'];allocated=row['peak_cuda_gib'];reserved=row['peak_reserved_gib']
    extra=e.c.read_json(p/'extension_provenance.json') if (p/'extension_provenance.json').exists() else {}
    if extra:allocated=extra['peak_allocated_gib'];reserved=extra['peak_reserved_gib']
    output={'family':spec.family,'method':e.LABELS[spec.family],'alpha':spec.alpha,'case_rank':spec.case_rank,
        'fixed_id':row['fixed_id'],'moving_id':row['moving_id'],'direction':row['direction'],
        **{k:row[k] for k in e.METRICS},'source_kind':kind,'gpu':row['gpu'],'run_dir':row['run_dir'],
        'dvf_sha256':row['dvf_sha256'],'optimizer_sec':row.get('optimizer_sec',row.get('training_elapsed_sec')),
        'worker_sec':row.get('worker_sec',row.get('total_elapsed_sec')),
        'audit_sec':audit.get('audit_sec',audit.get('audit_elapsed_sec')),
        'actual_adam_updates':sum(x['optimizer']['actual_epochs'] for x in blocks),
        'peak_allocated_gib':allocated,'peak_reserved_gib':reserved,
        'audit_peak_allocated_gib':audit.get('audit_peak_cuda_gib',audit.get('peak_cuda_gib')),
        'audit_peak_reserved_gib':audit.get('audit_peak_reserved_gib',audit.get('peak_reserved_gib')),
        'valid_label_count':row['valid_label_count'],'excluded_labels':row['excluded_labels'],
        'stage_actual_updates':'/'.join(str(x['optimizer']['actual_epochs']) for x in blocks),
        'stage_best_updates':'/'.join(str(x['optimizer']['best_epoch']) for x in blocks)}
    return output,blocks


def write_csv(path, rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def arc(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size':15,'axes.labelsize':17,'axes.titlesize':18,'xtick.labelsize':14,
                         'ytick.labelsize':14,'legend.fontsize':14})
    fig,axes=plt.subplots(2,3,figsize=(18,11))
    styles={'displacement':('#d55e00','o','-'),'bounded2':('#0072b2','s','--'),'svf_velocity':('#009e73','^','-.')}
    metrics=('dice','hd95','asd');titles=('Dice ↑','HD95 (mm) ↓','Symmetric ASD (mm) ↓')
    xmin=min(r['log_jdet_std_fg_mean']-r['log_jdet_std_fg_sd'] for r in rows)
    xmax=max(r['log_jdet_std_fg_mean']+r['log_jdet_std_fg_sd'] for r in rows)
    xpad=max(.02,.05*(xmax-xmin))
    for col,(metric,title) in enumerate(zip(metrics,titles)):
        for band in range(2):
            ax=axes[band,col]
            for fam,(color,marker,line) in styles.items():
                data=sorted([r for r in rows if r['family']==fam],key=lambda r:r['alpha'])
                ax.plot([r['log_jdet_std_fg_mean'] for r in data],[r[metric+'_mean'] for r in data],
                        color=color,marker=marker,linestyle=line,linewidth=2.4,markersize=7,label=e.LABELS[fam])
                if band:
                    for r in data:
                        ax.add_patch(Ellipse((r['log_jdet_std_fg_mean'],r[metric+'_mean']),
                            2*r['log_jdet_std_fg_sd'],2*r[metric+'_sd'],facecolor=color,edgecolor=color,alpha=.12))
            for method,color,marker in [('SITReg','#332288','D'),('VFA','#aa4499','P')]:
                r=next(r for r in rows if r['family']==method)
                ax.scatter(r['log_jdet_std_fg_mean'],r[metric+'_mean'],c=color,marker=marker,s=85,label=method,zorder=5)
                if band:ax.add_patch(Ellipse((r['log_jdet_std_fg_mean'],r[metric+'_mean']),
                    2*r['log_jdet_std_fg_sd'],2*r[metric+'_sd'],facecolor=color,edgecolor=color,alpha=.12))
            low=min(r[metric+'_mean']-(r[metric+'_sd'] if band else 0) for r in rows)
            high=max(r[metric+'_mean']+(r[metric+'_sd'] if band else 0) for r in rows)
            pad=max(.003,.07*(high-low))
            ax.set(xlim=(max(0,xmin-xpad),xmax+xpad),ylim=(low-pad,high+pad),xlabel='Foreground SD(log J)',ylabel=title)
            ax.set_title(title+(' · means ± sample SD' if band else ' · means'))
            ax.xaxis.set_major_locator(MaxNLocator(6));ax.yaxis.set_major_locator(MaxNLocator(6));ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.02))
    fig.suptitle('OASIS seed-3 first100 · 8→4→2 NCC9/diffusion sweeps',fontsize=21)
    fig.text(.5,.005,'Ellipses: ±1 across-case sample SD, not confidence regions. Folding is reported separately.',ha='center',fontsize=13)
    fig.tight_layout(rect=(0,.12,1,.95))
    for ext in ('png','pdf','svg'):fig.savefig(out/f'arc_2x3.{ext}',dpi=180)
    plt.close(fig)


def run(root, execution="original"):
    runroot=e.run_root(root,execution);m=e.c.read_json(runroot/'launch_manifest.json');e.assert_manifest(m)
    state=e.c.read_json(runroot/'controller/state.json')
    if len(state['jobs'])!=1530:raise RuntimeError('Wrong production job count')
    records={}
    for name,item in m['reused'].items():
        e.verify_receipt(e.c.read_json(runroot/'reuse_receipts'/f'{name}.json'))
        records[name]=(item['row'],e.Spec(**item['spec']),'reused_accepted')
    failures=[]
    for job in state['jobs']:
        spec=e.Spec(**job['spec'])
        if job['status']!='complete':failures.append(job);continue
        row=e.validate(Path(job['out']),m,spec)
        if spec.name() in records:raise RuntimeError('Duplicate registration')
        records[spec.name()]=(row,spec,'new_first100_extension')
    complete=len(records)==2100 and not failures and set(records)=={s.name() for s in e.matrix()}
    out=root/'repo_experiments'/e.VARIANT/f'release_{time.time_ns()}'
    out.mkdir(parents=True,exist_ok=False)
    flatrows,stages,labels,sensitivity=[],[],[],[]
    for row,spec,kind in records.values():
        r,blocks=flat(row,spec,kind);flatrows.append(r)
        common={k:r[k] for k in ('family','method','alpha','case_rank','fixed_id','moving_id','source_kind')}
        for i,b in enumerate(blocks):
            stages.append({**common,'block':i,'cps':b['cps'],**b['optimizer'],
                           'bound_checks':b.get('atomic_corner_audit'), 'initialization':b.get('initialization')})
        labelpath=Path(row['run_dir'])/('case_metrics.csv' if spec.family=='displacement' else 'final_metrics/case_metrics.csv')
        with labelpath.open() as f:lr=next(csv.DictReader(f))
        if lr['fixed_id']!=row['fixed_id'] or lr['moving_id']!=row['moving_id']:raise RuntimeError('Label case mismatch')
        reconstructed={k:[] for k in ('dice','hd95','asd')}
        for label in e.c.OASIS_LABELS:
            values={k:float(lr[f'{source}_label_{label:02d}']) for k,source in [('dice','dice'),('hd95','hd95'),('asd','asd_symmetric')]}
            for k,v in values.items():
                if math.isfinite(v):reconstructed[k].append(v)
            labels.append({**common,'label':label,**{k:v if math.isfinite(v) else None for k,v in values.items()}})
        for k,values in reconstructed.items():
            if not math.isclose(statistics.mean(values),row[k],rel_tol=0,abs_tol=1e-10):raise RuntimeError('Label-to-case mismatch')
        for diag in row.get('integration_diagnostics',[]):sensitivity.append({**common,**diag})
    references=[]
    for row in m['references']:
        references.append({'family':row['representation'],'method':row['representation'],'alpha':None,
            **{k:row[k] for k in ('case_rank','fixed_id','moving_id',*e.METRICS)},'source_kind':'original_DL',
            'source_path':row['source_path'],'source_sha256':row['source_sha256']})
    by={(r['family'],r['alpha'],r['case_rank']):r for r in flatrows}
    refs={(r['family'],r['case_rank']):r for r in references}
    paired=[]
    for row in flatrows:
        targets=[(model,refs[(model,row['case_rank'])]) for model in ('VFA','SITReg')]
        for fam in e.FAMILIES:
            other=by.get((fam,row['alpha'],row['case_rank']))
            if other and fam!=row['family']:targets.append((fam,other))
        for name,reference in targets:
            paired.append({k:row[k] for k in ('family','alpha','case_rank','fixed_id','moving_id')}|
                {'reference':name}|{f'delta_{k}':row[k]-reference[k] for k in e.METRICS})
    summaries=[];resources=[]
    for cohort,ranks in COHORTS.items():
        groups=defaultdict(list)
        for r in flatrows+references:
            if r['case_rank'] in ranks:groups[(r['family'],r['alpha'])].append(r)
        for (fam,alpha),rows in groups.items():
            if sorted(r['case_rank'] for r in rows)!=list(ranks):continue
            summaries.append({'cohort':cohort,'family':fam,'method':rows[0]['method'],'alpha':alpha,**summarize(rows,ranks)})
    for (fam,kind),rows in group_resources(flatrows).items():
        resources.append({'family':fam,'source_kind':kind,'n':len(rows),**resource_stats(rows)})
    first100=[r for r in summaries if r['cohort']=='first100']
    if complete and len(first100)!=23:raise RuntimeError('Require21 experimental and2 reference complete groups')
    write_csv(out/'per_case.csv',flatrows+references);write_csv(out/'summary.csv',summaries)
    write_csv(out/'paired_differences.csv',paired);write_csv(out/'stage_diagnostics.csv',stages)
    write_csv(out/'per_label.csv',labels);write_csv(out/'resources_by_provenance.csv',resources)
    write_csv(out/'integration_sensitivity.csv',sensitivity)
    e.atomic_json(out/'results.json',{'complete':complete,'candidate_count':len(flatrows),'reference_count':200,
        'per_case':flatrows,'references':references,'summary':summaries,'resource_by_provenance':resources,
        'failures':failures,'provenance':{'manifest':str(runroot/'launch_manifest.json'),'sha256':e.sha(runroot/'launch_manifest.json')}})
    lines=['# First100 B-spline842 NCC9/diffusion comparison','',
        f'Accepted individual registrations: {len(flatrows)}/2100; deferred failures: {len(failures)}.',
        '', 'Mean ± across-case sample SD. Accuracy and FG SD(log J) rounded to three decimals; folding in percent, scientific notation.',
        'Velocity diffusion and displacement diffusion are not equal-strength penalties at equal lambda. Bounded2 has twice the per-level optimizer budget.',
        'Ranks0–9 informed exploratory tuning; ranks10–99 are also previously inspected test data, not an untouched confirmatory set. Shared-load timing is diagnostic; historical and new timing are separated.', '']
    for cohort in COHORTS:
        lines += [f'## {cohort}', '', '| Method | λ | n | Dice ↑ | HD95 mm ↓ | ASD mm ↓ | FG SD(log J) | FG folding % |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        rows=[r for r in summaries if r['cohort']==cohort]
        for fam in ('SITReg','VFA',*e.FAMILIES):
            for r in sorted([r for r in rows if r['family']==fam],key=lambda x:x['alpha'] or 0):
                vals=[f"{r[k+'_mean']:.3e} ± {r[k+'_sd']:.3e}" if k=='jdet_le0_pct_fg' else f"{r[k+'_mean']:.3f} ± {r[k+'_sd']:.3f}" for k in e.METRICS]
                lines.append(f"| {r['method']} | {r['alpha'] if r['alpha'] is not None else 'original'} | {r['n']} | "+' | '.join(vals)+' |')
        lines.append('')
    lines += ['## Resource diagnostics by provenance','',
              'Seconds and GiB, mean ± sample SD; missing historical measurements remain n/a. Shared workloads prevent controlled speed claims.', '',
              '| Family | Provenance | n | Optimizer s | Worker s | Replay s | Peak allocated GiB | Peak reserved GiB |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in resources:
        values=[]
        for k in ('optimizer_sec','worker_sec','audit_sec','peak_allocated_gib','peak_reserved_gib'):
            value='n/a' if r[k+'_mean'] is None else f"{r[k+'_mean']:.2f} ± {r[k+'_sd']:.2f} (n={r[k+'_n']})" if r[k+'_sd'] is not None else f"{r[k+'_mean']:.2f} (n={r[k+'_n']})"
            values.append(value)
        lines.append(f"| {e.LABELS[r['family']]} | {r['source_kind']} | {r['n']} | "+' | '.join(values)+' |')
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    if complete:arc(first100,out)
    # Independently re-read the serialized CSV and reproduce every primary statistic.
    with (out/'per_case.csv').open() as f:readback=list(csv.DictReader(f))
    checks=0
    for r in summaries:
        vals=[x for x in readback if x['family']==r['family'] and (float(x['alpha']) if x['alpha'] else None)==r['alpha'] and int(x['case_rank']) in COHORTS[r['cohort']]]
        if len(vals)!=r['n']:raise RuntimeError('CSV cohort mismatch')
        for k in e.METRICS:
            numbers=[float(x[k]) for x in vals]
            if not math.isclose(statistics.mean(numbers),r[k+'_mean'],abs_tol=1e-12,rel_tol=0) or not math.isclose(statistics.stdev(numbers),r[k+'_sd'],abs_tol=1e-12,rel_tol=0):raise RuntimeError('CSV statistic mismatch')
            checks+=1
    e.assert_manifest(m)
    hashes={p.name:e.sha(p) for p in out.iterdir() if p.is_file()}
    result={'complete':complete,'candidate_count':len(flatrows),'new_count':len(flatrows)-570,'reused_count':570,
            'failures':len(failures),'statistic_checks':checks,'output_sha256':hashes,'completed_at':e.c.now_text()}
    e.atomic_json(out/('ACCEPTED.json' if complete else 'INCOMPLETE.json'),result)
    e.atomic_json(out.parent/'LATEST.json',{'path':str(out),'complete':complete,'at':e.c.now_text()})
    print(json.dumps({'report':str(out),'complete':complete,'candidate_count':len(flatrows)}),flush=True)


def group_resources(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['family'],row['source_kind'])].append(row)
    return groups


def resource_stats(rows):
    result={}
    for k in RESOURCES:
        v=[float(r[k]) for r in rows if r.get(k) is not None]
        result.update({k+'_n':len(v),k+'_mean':statistics.mean(v) if v else None,
                       k+'_sd':statistics.stdev(v) if len(v)>1 else None,k+'_min':min(v) if v else None,k+'_max':max(v) if v else None})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact-root',type=Path,default=Path('/database/inrorspline'))
    p.add_argument('--execution',choices=e.EXECUTIONS,default='original')
    args=p.parse_args()
    run(args.artifact_root,args.execution)
