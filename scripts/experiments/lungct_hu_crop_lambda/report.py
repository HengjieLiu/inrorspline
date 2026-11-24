"""Atomic live tables and immutable first-batch/final reports."""
from __future__ import annotations
import csv
import io
import math
from pathlib import Path
import statistics
from .protocol import dump,read,sha,stamp,sweep_state,WINDOWS,CROPS

BONE_GROUPS = ('bone', 'ribs', 'vertebrae', 'sternum')
BONE_DICE_ENDPOINTS = tuple(f'{prefix}_{group}_dice' for prefix in ('fixed', 'best') for group in BONE_GROUPS)

def text_atomic(path, text):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)

def csv_atomic(path, rows):
    if not rows:return
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);text_atomic(path,stream.getvalue())

def generate(state, review, name='live', seal=False):
    review=Path(review);dest=review/name;dest.mkdir(parents=True,exist_ok=True)
    if seal and (dest/'accepted.json').exists():return
    rows=[];candidates=[];anatomy_cache={}
    def anatomy(candidate,group,metric):
        if not candidate or not candidate.get('path'):return None
        path=Path(candidate['path'])/'anatomy_metrics.json'
        if not path.exists():return None
        if path not in anatomy_cache:anatomy_cache[path]=read(path)
        data=anatomy_cache[path]
        return data.get('groups',{}).get(group,{}).get(metric)
    for s in state['sweeps']:
        status=sweep_state(s['results'],s.get('preparation_failure',False))
        best=status['best'];center=s['results'].get('0',{})
        rows.append(dict(dataset=s['dataset'],case=s['case'],hu=s['hu'],crop=s['crop'],
          status='memory_unavailable' if s.get('unavailable_kind')=='memory' else status['status'],fixed_lambda=.055,fixed_tre=center.get('score'),
          fixed_sd=center.get('score_sd'),best_lambda=best['weight'] if best else None,
          best_k=best['k'] if best else None,best_tre=best['score'] if best else None,
          best_sd=best['candidate'].get('score_sd') if best else None,
          fixed_bone_assd_mm=anatomy(center,'bone','assd_mm'),
          best_bone_assd_mm=anatomy(best['candidate'] if best else None,'bone','assd_mm'),
          **{f'{prefix}_{group}_dice':anatomy(candidate,group,'dice')
             for prefix,candidate in [('fixed',center),('best',best['candidate'] if best else None)]
             for group in BONE_GROUPS},
          best_lung_dice=anatomy(best['candidate'] if best else None,'lung','dice'),
          best_heart_dice=anatomy(best['candidate'] if best else None,'heart','dice'),
          best_aorta_dice=anatomy(best['candidate'] if best else None,'aorta','dice'),
          tested=len(s['results']),failures=sum(r['status']!='accepted' for r in s['results'].values()),
          upper_stop=status['upper']['status'] if status['upper'] else None,
          lower_stop=status['lower']['status'] if status['lower'] else None,
          error=s.get('error','')))
        for k,r in s['results'].items():
            candidates.append(dict(dataset=s['dataset'],case=s['case'],hu=s['hu'],crop=s['crop'],
                k=int(k),weight=.055*2**int(k),status=r['status'],tre=r.get('score'),sd=r.get('score_sd'),
                gpu=r.get('gpu'),seconds=r.get('seconds'),path=r.get('path'),error=r.get('error','')))
    csv_atomic(dest/'case_conditions.csv',rows);csv_atomic(dest/'candidates.csv',candidates)
    dump(dest/'results.json',dict(updated=stamp(),stage=state['stage'],rows=rows,candidates=candidates))
    lines=['# MR-D-BSCP HU/crop/lambda '+name,'',
       'Per-case lambda selection uses the same300 native snap landmarks being reported. Best means best tested; caps/failures remain explicit.',
       '',f"Stage: {state['stage']}; {len(candidates)} finished candidates; {sum(c['status']=='accepted' for c in candidates)} accepted.",
       '', 'H0[-944,-124], H1[-1000,1000], H2[-1000,2000]HU; common intensity gain. C0 landmark crop, C1 paired lung-mask box+20/20/10mm, C2 full acquired FOV.',
       'Fixed0.055 and tuned results are reported separately. Training protocols are otherwise unchanged.','']
    lines.extend(['Anatomy metrics are independent automatic-label proxies on shared native coverage. Missing values mean pending/failed annotations or absent structures; they are not perfect alignment. Original reports remain sealed; later anatomy supplements add results.',''])
    lines.extend(['Combined bone is the union of ribs, vertebrae and sternum; its Dice is retained directly and is not the arithmetic mean of component Dice. Ribs and vertebrae are each grouped unions, not individual rib/vertebra-level scores. All anatomy uses the same C0/C1/C2 intersection. Dice in CSV/JSON is a fraction; dedicated bone tables display percent.',''])
    summaries=[]
    for dataset in ('4dct','copd'):
        lines.extend([f'## {dataset.upper()}','',
          '| Case | HU | Crop | TRE at0.055, mm | Best tested TRE, mm | Selected lambda | Candidates / failures | Upper / lower stop |',
          '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |'])
        def fmt(mean,sd):return '—' if mean is None else f'{mean:.4f} ± {sd:.4f}'
        ds=[r for r in rows if r['dataset']==dataset]
        for r in sorted(ds,key=lambda v:(v['case'],v['crop'],v['hu'])):
            lines.append(f"| {r['case']:02d} | {r['hu']} | {r['crop']} | {fmt(r['fixed_tre'],r['fixed_sd'])} | {fmt(r['best_tre'],r['best_sd'])} | {r['best_lambda'] if r['best_lambda'] is not None else '—'} | {r['tested']} / {r['failures']} | {r['upper_stop']} / {r['lower_stop']} |")
        lines.extend(['','### Casewise bone components','',
          '| Case | HU | Crop | Lambda selection | Lambda | TRE, mm | Combined bone Dice % | Ribs Dice % | Vertebrae Dice % | Sternum Dice % |',
          '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
        def number(value,scale=1):return '—' if value is None else f'{value*scale:.4f}'
        for r in sorted(ds,key=lambda v:(v['case'],v['crop'],v['hu'])):
            for prefix,label in [('fixed','Fixed'),('best','TRE-selected')]:
                dice=' | '.join(number(r[f'{prefix}_{group}_dice'],100) for group in BONE_GROUPS)
                lines.append(f"| {r['case']:02d} | {r['hu']} | {r['crop']} | {label} | {number(r[f'{prefix}_lambda'])} | {number(r[f'{prefix}_tre'])} | {dice} |")
        lines.extend(['','### Bone component cohort means ± sample SD','',
          '| HU | Crop | Lambda selection | TRE, mm | Combined bone Dice % | Ribs Dice % | Vertebrae Dice % | Sternum Dice % |',
          '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |'])
        def cohort_value(rs,endpoint,scale=1):
            vals=[r[endpoint]*scale for r in rs if r[endpoint] is not None]
            if not vals:return '— (n=0)'
            mean=statistics.mean(vals);sd=statistics.stdev(vals) if len(vals)>1 else None
            return f'{mean:.4f} ± {sd:.4f} (n={len(vals)})' if sd is not None else f'{mean:.4f} (n=1)'
        for hu in WINDOWS:
            for crop in CROPS:
                rs=[r for r in ds if r['hu']==hu and r['crop']==crop]
                for prefix,label in [('fixed','Fixed'),('best','TRE-selected')]:
                    dice=' | '.join(cohort_value(rs,f'{prefix}_{group}_dice',100) for group in BONE_GROUPS)
                    lines.append(f"| {hu} | {crop} | {label} | {cohort_value(rs,prefix+'_tre')} | {dice} |")
        lines.extend(['','### All cohort summaries','',
          '| HU | Crop | Endpoint | Units | n / expected | Mean across cases | Sample SD |','| --- | --- | --- | --- | ---: | ---: | ---: |'])
        expected=len({r['case'] for r in ds})
        for hu in WINDOWS:
            for crop in CROPS:
                rs=[r for r in ds if r['hu']==hu and r['crop']==crop]
                for endpoint in ('fixed_tre','best_tre','fixed_bone_assd_mm','best_bone_assd_mm','best_lung_dice','best_heart_dice','best_aorta_dice')+BONE_DICE_ENDPOINTS:
                    vals=[r[endpoint] for r in rs if r[endpoint] is not None]
                    complete=len(vals)==expected
                    mean=statistics.mean(vals) if vals else None
                    sd=statistics.stdev(vals) if len(vals)>1 else None
                    unit='fraction' if endpoint.endswith('_dice') else 'mm'
                    summaries.append(dict(dataset=dataset,hu=hu,crop=crop,endpoint=endpoint,unit=unit,n=len(vals),expected=expected,
                        complete=complete,mean=mean,sample_sd=sd))
                    lines.append(f"| {hu} | {crop} | {endpoint} | {unit} | {len(vals)} / {expected} | {mean:.4f} | {sd:.4f} |" if sd is not None else f'| {hu} | {crop} | {endpoint} | {unit} | {len(vals)} / {expected} | {number(mean)} | — |')
        lines.append('')
    failures=[c for c in candidates if c['status']!='accepted']
    lines.extend(['## Failures','', 'No failure is converted to infinite TRE or silently removed. Partial cohorts retain their n.',''])
    for c in failures:lines.append(f"- {c['dataset']}{c['case']:02d} {c['hu']}/{c['crop']} k={c['k']}: {c['error']}")
    for r in rows:
        if r['error']:lines.append(f"- Condition ({r['status']}): {r['dataset']}{r['case']:02d} {r['hu']}/{r['crop']}: {r['error']}")
    text_atomic(dest/'report.md','\n'.join(lines)+'\n');csv_atomic(dest/'cohort_summaries.csv',summaries)
    if seal:
        plot_curves(candidates,dest)
        dump(dest/'accepted.json',dict(status='reported',created=stamp(),stage=state['stage'],
             cases=sorted({f"{r['dataset']}_{r['case']:02d}" for r in rows}),
             report_source_sha256=sha(Path(__file__)),
             receipts={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='accepted.json'}))

def plot_curves(candidates,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids=sorted({(c['dataset'],c['case']) for c in candidates})
    for ds,case in ids:
        fig,axes=plt.subplots(3,3,figsize=(13,10),squeeze=False)
        for i,crop in enumerate(CROPS):
            for j,hu in enumerate(WINDOWS):
                points=sorted([c for c in candidates if (c['dataset'],c['case'],c['hu'],c['crop'])==(ds,case,hu,crop) and c['status']=='accepted'],key=lambda c:c['k'])
                ax=axes[i,j]
                if points:ax.plot([c['k'] for c in points],[c['tre'] for c in points],'o-')
                ax.axvline(0,color='gray',linestyle=':');ax.set_title(f'{hu}/{crop}');ax.set_xlabel('log2(lambda/0.055)');ax.set_ylabel('Native snap TRE (mm)');ax.grid(alpha=.2)
        fig.suptitle(f'{ds.upper()} case{case:02d}: per-case tuning, tested candidates only');fig.tight_layout()
        fig.savefig(dest/f'{ds}_{case:02d}_lambda_curves.png',dpi=140);plt.close(fig)
