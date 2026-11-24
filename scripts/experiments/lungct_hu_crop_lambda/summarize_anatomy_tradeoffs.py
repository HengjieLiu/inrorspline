"""Summarize fixed-lambda and TRE-selected anatomy tradeoffs without rerunning DIR."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import statistics

from .protocol import read, dump, sha, stamp, sweep_state

BONE_GROUPS = ('bone', 'ribs', 'vertebrae', 'sternum')
METRICS = ['tre', 'bone_dice', 'bone_assd_mm', 'lung_dice', 'lung_assd_mm',
           'heart_dice', 'aorta_dice', 'esophagus_dice', 'ribs_dice',
           'vertebrae_dice', 'sternum_dice', 'bone_strain', 'bone_volume_error',
           'folding_lung_pct', 'folding_full_pct', 'source_inside_fraction',
           'bone_coverage_fraction']


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_metrics(path):
    path = Path(path)
    anatomy = read(path / 'anatomy_metrics.json')
    deformation = read(path / 'deformation_metrics.json')
    assert anatomy['status'] == 'accepted_automatic_proxy'
    result = {}
    for group in ['bone', 'lung', 'heart', 'aorta', 'esophagus', 'ribs', 'vertebrae', 'sternum']:
        result[group + '_dice'] = anatomy['groups'][group]['dice']
        if group in ['bone', 'lung']:
            result[group + '_assd_mm'] = anatomy['groups'][group]['assd_mm']
    result.update(bone_strain=anatomy['bone_rigidity']['mean_frobenius_JtJ_minus_I'],
                  bone_volume_error=anatomy['bone_rigidity']['mean_abs_jdet_minus_one'],
                  folding_lung_pct=deformation['lung']['folding_pct'],
                  folding_full_pct=deformation['full']['folding_pct'],
                  source_inside_fraction=anatomy['source_inside_fraction'],
                  bone_coverage_fraction=anatomy['additional_fov_coverage']['fixed_bone_voxels_common'] /
                      anatomy['additional_fov_coverage']['fixed_bone_voxels_full'])
    return result


def describe(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    run = Path(config['run_root'])
    state = read(run / 'controller_state.json')
    assert state['stage'] == 'complete'
    args.out.mkdir(parents=True, exist_ok=False)
    selected = []
    for sweep in state['sweeps']:
        best = sweep_state(sweep['results'])['best']
        for mode, candidate, k in [('fixed', sweep['results']['0'], 0),
                                    ('TRE_tuned', best['candidate'], best['k'])]:
            selected.append(dict(dataset=sweep['dataset'], case=sweep['case'], hu=sweep['hu'],
                                 crop=sweep['crop'], mode=mode, k=k, lambda_value=.055*2**k,
                                 tre=candidate['score'], path=candidate['path']))
    paths = sorted({r['path'] for r in selected})
    with ThreadPoolExecutor(max_workers=8) as pool:
        cache = dict(zip(paths, pool.map(load_metrics, paths)))
    rows = [dict(r, **cache[r['path']]) for r in selected]
    assert len(rows) == 360
    aggregates = []
    for ds in ['4dct', 'copd']:
        for mode in ['fixed', 'TRE_tuned']:
            for hu in ['H0', 'H1', 'H2']:
                for crop in ['C0', 'C1', 'C2']:
                    group = [r for r in rows if (r['dataset'],r['mode'],r['hu'],r['crop']) == (ds,mode,hu,crop)]
                    assert len(group) == 10
                    aggregate = dict(dataset=ds, mode=mode, hu=hu, crop=crop)
                    for metric in METRICS:
                        aggregate.update({metric+'_'+k:v for k,v in describe([r[metric] for r in group]).items()})
                    aggregates.append(aggregate)
    index = {(r['dataset'],r['case'],r['mode'],r['hu'],r['crop']):r for r in rows}
    paired = []
    for ds in ['4dct','copd']:
        comparisons = []
        for mode in ['fixed','TRE_tuned']:
            for crop in ['C0','C1','C2']:
                for old,new in [('H0','H1'),('H0','H2'),('H1','H2')]:
                    comparisons.append((mode+' '+new+'-'+old+' '+crop, (mode,old,crop), (mode,new,crop)))
            for hu in ['H0','H1','H2']:
                for old,new in [('C0','C1'),('C0','C2'),('C1','C2')]:
                    comparisons.append((mode+' '+new+'-'+old+' '+hu, (mode,hu,old), (mode,hu,new)))
        for hu in ['H0','H1','H2']:
            for crop in ['C0','C1','C2']:
                comparisons.append(('tuning '+hu+' '+crop, ('fixed',hu,crop), ('TRE_tuned',hu,crop)))
        for name,old,new in comparisons:
            for case in range(1,11):
                left=index[(ds,case,*old)];right=index[(ds,case,*new)]
                result=dict(dataset=ds,comparison=name,case=case,old_lambda=left['lambda_value'],new_lambda=right['lambda_value'])
                result.update({m:right[m]-left[m] if left[m] is not None and right[m] is not None else None for m in METRICS})
                paired.append(result)
    paired_summary = []
    for ds,name in sorted({(r['dataset'],r['comparison']) for r in paired}):
        group = [r for r in paired if (r['dataset'],r['comparison']) == (ds,name)]
        summary=dict(dataset=ds,comparison=name)
        for metric in METRICS:
            values=[r[metric] for r in group if r[metric] is not None]
            summary.update({metric+'_'+k:v for k,v in describe(values).items()})
            sign=1 if metric.endswith('_dice') or metric in ['source_inside_fraction','bone_coverage_fraction'] else -1
            summary[metric+'_improved']=sum(sign*v>1e-6 for v in values)
            summary[metric+'_worsened']=sum(sign*v < -1e-6 for v in values)
        for component in BONE_GROUPS:
            metric=component+'_dice'
            summary[f'TRE_worse_{metric}_better']=sum(r['tre']>1e-6 and r[metric] is not None and r[metric]>1e-6 for r in group)
            summary[f'TRE_better_{metric}_worse']=sum(r['tre'] < -1e-6 and r[metric] is not None and r[metric] < -1e-6 for r in group)
        paired_summary.append(summary)
    write_csv(args.out/'selected_cases.csv',rows)
    write_csv(args.out/'aggregate.csv',aggregates)
    write_csv(args.out/'paired_cases.csv',paired)
    write_csv(args.out/'paired_summary.csv',paired_summary)
    bone_cases=[{k:r[k] for k in ['dataset','case','hu','crop','mode','k','lambda_value','tre']+
                [group+'_dice' for group in BONE_GROUPS]+['path']} for r in rows]
    bone_aggregates=[]
    for r in aggregates:
        result={k:r[k] for k in ['dataset','mode','hu','crop','tre_mean','tre_sample_sd','tre_n']}
        for group in BONE_GROUPS:
            for stat in ['mean','sample_sd']:
                value=r[group+'_dice_'+stat]
                result[group+'_dice_pct_'+stat]=100*value if value is not None else None
            result[group+'_dice_n']=r[group+'_dice_n']
        bone_aggregates.append(result)
    write_csv(args.out/'bone_components_cases.csv',bone_cases)
    write_csv(args.out/'bone_components_aggregate.csv',bone_aggregates)
    dump(args.out/'results.json',dict(aggregates=aggregates,paired_summary=paired_summary))
    lines=['# HU/crop and landmark/anatomy tradeoffs','',
           'Independent automatic labels; visual QC pending. All anatomy evaluated on the same case-specific C0/C1/C2 intersection, not the whole thorax.',
           'Fixed means lambda0.055. TRE_tuned selects each case/lambda by its own300 native snap landmarks; anatomy does not select lambda.',
           'Combined bone Dice is retained from the union of ribs+vertebrae+sternum. It is not the arithmetic average of component Dice. Ribs and vertebrae each combine all visible labels in that group, rather than scoring individual ribs or vertebral levels.',
           'All table entries below are unweighted ten-case means; Dice is shown as percent. CSVs retain sample SDs, counts, casewise paired changes, surface and rigidity metrics.','']
    for ds in ['4dct','copd']:
        for mode in ['fixed','TRE_tuned']:
            lines += [f'## {ds.upper()} / {mode}','',
                      '| HU | Crop | TRE mm ↓ | Bone Dice % ↑ | Bone ASSD mm ↓ | Lung Dice % ↑ | Heart Dice % ↑ | Aorta Dice % ↑ |',
                      '|---|---|---:|---:|---:|---:|---:|---:|']
            for r in aggregates:
                if (r['dataset'],r['mode']) != (ds,mode):continue
                lines.append(f"|{r['hu']}|{r['crop']}|{r['tre_mean']:.4f}|{100*r['bone_dice_mean']:.2f}|{r['bone_assd_mm_mean']:.3f}|{100*r['lung_dice_mean']:.2f}|{100*r['heart_dice_mean']:.2f}|{100*r['aorta_dice_mean']:.2f}|")
            lines.append('')
            lines += [f'### Bone components: {ds.upper()} / {mode}','',
                      'Unweighted mean ± sample SD across ten cases. All Dice values are percent; complete counts and individual case values are in the bone component CSVs.','',
                      '| HU | Crop | TRE, mm | Combined bone Dice % | Ribs Dice % | Vertebrae Dice % | Sternum Dice % |',
                      '|---|---|---:|---:|---:|---:|---:|']
            for r in aggregates:
                if (r['dataset'],r['mode']) != (ds,mode):continue
                components='|'.join(f"{100*r[group+'_dice_mean']:.2f} ± {100*r[group+'_dice_sample_sd']:.2f}" if r[group+'_dice_n']>1 else '—' for group in BONE_GROUPS)
                lines.append(f"|{r['hu']}|{r['crop']}|{r['tre_mean']:.4f} ± {r['tre_sample_sd']:.4f}|{components}|")
            lines.append('')
    lines += ['## Interpretation limits','',
              '- TRE measures lung correspondences; organ/bone overlap and surface distance measure different regions/features.',
              '- Bone is the union of ribs, vertebrae and sternum, not label-by-label rib/vertebra matching.',
              '- Large lung Dice does not establish alignment of small intrapulmonary features.',
              '- Automatic segmentation errors and clipped common-region surfaces affect these metrics.',
              '- Folding diagnostics in the CSV retain each run\'s own registration grid; they are not common-ROI anatomy measurements.',
              '- Caps, one failed candidate and CUDA repeatability limitations remain as in the sealed campaign reports.',
              '- Group-mean plots are descriptive; repeated settings from the same cases are not independent subjects.','']
    lines += ['## Paired HU changes at C0 after TRE selection','',
              '| Dataset | H1 minus H0 TRE, mm | Bone Dice, percentage points | Bone ASSD, mm | Heart Dice, percentage points | Aorta Dice, percentage points |',
              '|---|---:|---:|---:|---:|---:|']
    for ds in ['4dct','copd']:
        r=next(r for r in paired_summary if r['dataset']==ds and r['comparison']=='TRE_tuned H1-H0 C0')
        lines.append(f"|{ds}|{r['tre_mean']:+.4f}|{100*r['bone_dice_mean']:+.2f}|{r['bone_assd_mm_mean']:+.3f}|{100*r['heart_dice_mean']:+.2f}|{100*r['aorta_dice_mean']:+.2f}|")
    tuning=[r for r in paired_summary if r['comparison'].startswith('tuning ')]
    count=sum(r['TRE_better_bone_dice_worse'] for r in tuning)
    lines += ['',f'In{count}/180 case/preparation combinations, TRE selection reduced TRE but also reduced bone Dice relative to lambda0.055. These combinations repeat the same20 subjects.','']
    lines += ['## Component-specific tradeoffs after TRE selection','',
              'Paired means of new-minus-old within the same ten cases. Dice changes are percentage points; positive TRE changes are worse. Improvements are descriptive, not independent-subject hypothesis tests.','',
              '| Dataset | Comparison | TRE change, mm | Combined bone Dice change | Ribs Dice change | Vertebrae Dice change | Sternum Dice change |',
              '|---|---|---:|---:|---:|---:|---:|']
    wanted=['TRE_tuned H1-H0 C0','TRE_tuned H1-H0 C1','TRE_tuned H1-H0 C2',
            'TRE_tuned C1-C0 H1','TRE_tuned C2-C1 H1','TRE_tuned H2-H1 C1']
    for ds in ['4dct','copd']:
        for name in wanted:
            r=next(r for r in paired_summary if r['dataset']==ds and r['comparison']==name)
            values='|'.join(f"{100*r[group+'_dice_mean']:+.2f}" for group in BONE_GROUPS)
            lines.append(f"|{ds}|{name}|{r['tre_mean']:+.4f}|{values}|")
    lines += ['','### Lambda selection and component overlap','',
              '| Component | Case/preparation combinations with lower TRE but lower Dice after tuning, out of180 |',
              '|---|---:|']
    for group in BONE_GROUPS:
        total=sum(r[f'TRE_better_{group}_dice_worse'] for r in tuning)
        lines.append(f'|{group}|{total}|')
    lines.append('')
    for ds in ['4dct','copd']:
        values=[r['bone_coverage_fraction'] for r in rows if r['dataset']==ds and r['mode']=='fixed' and r['hu']=='H0' and r['crop']=='C0']
        lines.append(f"Common-region fraction of fixed segmented bone in{ds}: mean{100*statistics.mean(values):.2f}%, range{100*min(values):.2f}–{100*max(values):.2f}%. This excludes bone outside the acquired image and outside the common evaluation region.")
    (args.out/'report.md').write_text('\n'.join(lines))
    plot(aggregates,args.out)
    plot_bone_components(aggregates,args.out)
    dump(args.out/'accepted.json',dict(status='summarized',time=stamp(),cases=20,selected_rows=len(rows),
         analysis_source_sha256=sha(Path(__file__)),controller_state_sha256=sha(run/'controller_state.json'),
         config_sha256=sha(args.config),annotation_protocol_sha256=sha(Path(config['anatomy_root'])/'protocol.json'),
         artifacts={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}))
    print('SUMMARIZED',args.out,flush=True)
    for ds in ['4dct','copd']:
        for r in aggregates:
            if r['dataset']==ds and r['mode']=='TRE_tuned':
                print(ds,r['hu'],r['crop'],*[round(r[m+'_mean'],4) for m in ['tre','bone_dice','bone_assd_mm','lung_dice','heart_dice','aorta_dice','bone_strain']],flush=True)


def plot(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={'H0':'#2563eb','H1':'#d97706','H2':'#b91c1c'}
    markers={'C0':'o','C1':'s','C2':'^'}
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for i,ds in enumerate(['4dct','copd']):
        for j,metric in enumerate(['bone_dice','heart_dice','lung_dice']):
            ax=axes[i,j]
            for hu in colors:
                for crop in markers:
                    pair=[next(r for r in rows if (r['dataset'],r['hu'],r['crop'],r['mode'])==(ds,hu,crop,mode)) for mode in ['fixed','TRE_tuned']]
                    ax.plot([r['tre_mean'] for r in pair],[100*r[metric+'_mean'] for r in pair],color=colors[hu],lw=.7,alpha=.45)
                    for mode,r in zip(['fixed','TRE_tuned'],pair):
                        ax.scatter(r['tre_mean'],100*r[metric+'_mean'],s=65,marker=markers[crop],edgecolors=colors[hu],facecolors='white' if mode=='fixed' else colors[hu],zorder=3)
            ax.set_xlabel('Native snap TRE (mm; lower is better)')
            ax.set_ylabel(metric.replace('_',' ').title()+' (%)')
            ax.set_title(ds.upper()+' / '+metric.split('_')[0]);ax.grid(alpha=.2)
    legends=[Line2D([0],[0],color=color,label=hu) for hu,color in colors.items()]
    legends += [Line2D([0],[0],color='gray',marker=marker,lw=0,label=crop) for crop,marker in markers.items()]
    legends += [Line2D([0],[0],color='gray',marker='o',mfc=face,lw=0,label=label) for face,label in [('white','Fixed λ = 0.055'),('gray','TRE-selected λ')]]
    fig.legend(handles=legends,loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Landmark and anatomy tradeoffs: ten-case means, common crop intersection\nConnected hollow/filled points show lambda selection; automatic labels, visual QC pending')
    fig.tight_layout(rect=(0,.11,1,.92))
    fig.savefig(out/'tradeoffs.png',dpi=180)
    fig.savefig(out/'tradeoffs.svg')
    plt.close(fig)


def plot_bone_components(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={'H0':'#2563eb','H1':'#d97706','H2':'#b91c1c'}
    markers={'C0':'o','C1':'s','C2':'^'}
    titles={'bone':'Combined bone','ribs':'Ribs','vertebrae':'Vertebrae','sternum':'Sternum'}
    fig,axes=plt.subplots(2,4,figsize=(17,8.5))
    for i,ds in enumerate(['4dct','copd']):
        for j,group in enumerate(BONE_GROUPS):
            ax=axes[i,j];metric=group+'_dice'
            for hu,color in colors.items():
                for crop,marker in markers.items():
                    pair=[next(r for r in rows if (r['dataset'],r['hu'],r['crop'],r['mode'])==(ds,hu,crop,mode)) for mode in ['fixed','TRE_tuned']]
                    ax.plot([r['tre_mean'] for r in pair],[100*r[metric+'_mean'] for r in pair],color=color,lw=.7,alpha=.45)
                    for mode,r in zip(['fixed','TRE_tuned'],pair):
                        ax.scatter(r['tre_mean'],100*r[metric+'_mean'],s=60,marker=marker,edgecolors=color,facecolors='white' if mode=='fixed' else color,zorder=3)
            ax.set_xlabel('Native snap TRE (mm; lower is better)')
            ax.set_ylabel('Dice (%)');ax.set_title(ds.upper()+' / '+titles[group]);ax.grid(alpha=.2)
    legend=[Line2D([0],[0],color=color,label=hu) for hu,color in colors.items()]
    legend += [Line2D([0],[0],color='gray',marker=marker,lw=0,label=crop) for crop,marker in markers.items()]
    legend += [Line2D([0],[0],color='gray',marker='o',mfc=face,lw=0,label=label) for face,label in [('white','Fixed λ = 0.055'),('gray','TRE-selected λ')]]
    fig.legend(handles=legend,loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Landmark TRE versus bone component overlap: ten-case means\nSame common evaluation crop; automatic labels, full visual QC pending')
    fig.tight_layout(rect=(0,.11,1,.92))
    fig.savefig(out/'bone_component_tradeoffs.png',dpi=180)
    fig.savefig(out/'bone_component_tradeoffs.svg')
    plt.close(fig)


if __name__=='__main__':main()
