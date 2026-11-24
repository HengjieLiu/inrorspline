"""Ordered casewise comparisons and a machine-readable metric appendix."""
from pathlib import Path
import numpy as np
from core import load_csv,read_json,sha,write_csv,write_json


LIMITATIONS = ('Headline: original/native snap-to-voxel TRE in mm. Per-case SD is sample SD over300 landmarks; '
               'aggregate SD is sample SD over10 case means. Legacy and 4DCT MR/pTV are historical references. '
               'COPD MR weight0.055 in both crop variants; historical reuse is not a contemporaneous crop-only ablation. '
               'CorrField uses unclipped HU plus fixed R231 mask, while MR/pTV use pTV intensities and no lung-mask weighting. '
               'Corrected COPD01=272 Python-resize slices; legacy275; pTV MATLAB273. No tuning or forced TRE equivalence.')


def table(title,rows,columns,delta=False):
    text=f'## {title}\n\n| Case | '+' | '.join(columns)+' |\n| --- | '+' | '.join(['---:']*len(columns))+' |\n'
    for r in rows:
        cells=[]
        for col in columns:
            if col=='Corrected - legacy':cells.append(f"{r['delta']:+.4f}")
            else:cells.append(f"{r[col]:.4f} ± {r[col+'_sd']:.4f}")
        text+=f"| {r['case']:02d} | "+' | '.join(cells)+' |\n'
    cells=[]
    for col in columns:
        vals=np.asarray([r['delta' if col=='Corrected - legacy' else col] for r in rows])
        cells.append(f'{vals.mean():+.4f} ± {vals.std(ddof=1):.4f}' if col=='Corrected - legacy' else f'{vals.mean():.4f} ± {vals.std(ddof=1):.4f}')
    text+='| Mean ± SD | '+' | '.join(cells)+' |\n'
    return text


def build(run_root,reference_root,review_root,require_all=False):
    run_root=Path(run_root);review_root=Path(review_root);out=review_root/'reports';out.mkdir(parents=True,exist_ok=True)
    refs=read_json(Path(reference_root)/'accepted.json')['evaluations']
    records=list(refs)
    for p in run_root.glob('*_*/evaluation.json'):
        if (p.parent/'accepted.json').exists():records.append(read_json(p))
    def find(method,ds,case):
        rr=[r for r in records if r['method']==method and r['dataset']==ds and r['case']==case]
        if len(rr)!=1:raise LookupError((method,ds,case,len(rr)))
        return rr[0]
    sections=[];tables={}
    try:
        rows=[]
        for c in range(1,11):
            old=find('mrdbscp_legacy','copd',c);new=find('mrdbscp','copd',c)
            rows.append(dict(case=c,Legacy=old['original_snap'],Legacy_sd=old['original_snap_sd'],
                             Corrected=new['original_snap'],Corrected_sd=new['original_snap_sd'],
                             delta=new['original_snap']-old['original_snap']))
        cols=['Legacy','Corrected','Corrected - legacy']
        text=table('1. COPD MR-D-BSCP: corrected versus legacy',rows,cols)
        (out/'01_copd_crop_comparison.md').write_text(text+'\n'+LIMITATIONS+'\n')
        write_csv(out/'01_copd_crop_comparison.csv',rows);write_json(out/'01_copd_crop_comparison.json',rows)
        sections.append(text);tables['copd_crop_comparison']=rows
    except LookupError:
        if require_all:raise
    for index,ds in [(2,'4dct'),(3,'copd')]:
        try:
            rows=[]
            for c in range(1,11):
                row=dict(case=c)
                for method,label in [('ptvreg','pTVReg'),('mrdbscp','MR-D-BSCP corrected'),('corrfield','CorrField')]:
                    r=find(method,ds,c);row[label]=r['original_snap'];row[label+'_sd']=r['original_snap_sd']
                rows.append(row)
            cols=['pTVReg','MR-D-BSCP corrected','CorrField']
            text=table(f'{index}. {ds.upper()}: pTVReg, corrected MR-D-BSCP and CorrField',rows,cols)
            stem=f'{index:02d}_{ds}_three_methods'
            (out/(stem+'.md')).write_text(text+'\n'+LIMITATIONS+'\n')
            write_csv(out/(stem+'.csv'),rows);write_json(out/(stem+'.json'),rows)
            sections.append(text);tables[ds]=rows
        except LookupError:
            if require_all:raise
    if require_all:
        assert len(records)==70 and len(tables)==3
        assert all(r['landmarks']==300 and np.isfinite(r['original_snap']) for r in records)
    if sections:
        (out/'report.md').write_text('# Corrected1mm LungCT comparisons\n\n'+LIMITATIONS+'\n\n'+'\n'.join(sections))
        write_json(out/'results.json',dict(status='complete' if require_all else 'partial',tables=tables,evaluations=records,limitations=LIMITATIONS))
    if require_all:
        diagnostic=[];landmarks=[];receipts=[]
        for r in records:
            diagnostic.append({k:r.get(k) for k in ['dataset','case','method','historical','original_snap','original_snap_sd',
                                                   'original_subvoxel','reg1mm_snap','reg1mm_subvoxel','intensity_profile','registration_mask']})
            for v in load_csv(r['per_landmark_path']):
                landmarks.append(dict(dataset=r['dataset'],case=r['case'],method=r['method'],landmark_id=int(v['landmark_id']),
                                      original_snap=float(v['original_snap']),original_subvoxel=float(v['original_subvoxel'])))
            assert sha(r['field_path'])==r['field_sha256']
            receipts.append(dict(dataset=r['dataset'],case=r['case'],method=r['method'],field_path=r['field_path'],sha256=r['field_sha256']))
        write_csv(out/'metric_diagnostics.csv',diagnostic);write_csv(out/'all_per_landmark_tre.csv',landmarks)
        write_json(out/'accepted.json',dict(status='accepted',evaluations=70,new_production=30,landmark_metric_rows=len(landmarks),
                                           field_receipts=receipts,report_sha256=sha(out/'report.md')))
    return tables
