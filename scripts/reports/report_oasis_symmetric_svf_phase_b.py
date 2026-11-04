#!/usr/bin/env python
"""Accept the approved complete Phase-B ablation, with swap/repeat diagnostics."""
import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/experiments/oasis_alpha_sweep'))
import run_oasis_symmetric_svf_phase_b as r
sys.path.insert(0,str(ROOT/'scripts/reports'))
from report_oasis_symmetric_svf_phase_a import difference


def primary_rows(rows):
    return [x for x in rows if not x['spec']['swapped'] and not x['spec']['repeat']]


def validate_design(rows,gate,root):
    if len(rows)!=28 or len({x['run_name'] for x in rows})!=28:raise RuntimeError('Require 28 unique fits')
    initial=[x for x in primary_rows(rows) if x['spec']['arm']!='combined']
    combined=r.m.combined_matrix(initial)
    expected={s.name() for s in r.m.initial_matrix()+combined}
    if {x['run_name'] for x in primary_rows(rows)}!=expected:raise RuntimeError('Incorrect primary matrix')
    diagnostics=r.m.diagnostic_matrix(primary_rows(rows))
    lookup={x['run_name']:x for x in rows}
    if set(lookup)!=expected|{s.name() for s,g,p in diagnostics}:raise RuntimeError('Wrong diagnostic selections')
    for spec,gpu,parent in diagnostics:
        if lookup[spec.name()]['gpu']!=gpu:raise RuntimeError('Diagnostic GPU not matched')
    for group,deps in (('initial',[]),('combined',initial),('diagnostics',primary_rows(rows))):
        plan=r.c.read_json(root/(group+'_plan.json'))
        expected_entries=([(s,'0' if s.arm=='base' else None,None) for s in r.m.initial_matrix()] if group=='initial'
                          else [(s,None,None) for s in combined] if group=='combined' else diagnostics)
        if plan['entries']!=[{'spec':asdict(s),'pinned_gpu':gpu,'parent_run':parent} for s,gpu,parent in expected_entries]:
            raise RuntimeError('Selection plan entries differ from approved deterministic rule')
        if plan['dependencies']!={x['run_name']:r.sha(Path(x['run_dir'])/'summary.json') for x in deps}:
            raise RuntimeError('Selection dependency hash changed')
        for entry in plan['entries']:
            spec=r.m.Spec(**entry['spec'])
            if entry['pinned_gpu'] is not None and lookup[spec.name()]['gpu']!=entry['pinned_gpu']:
                raise RuntimeError('Pinned GPU mismatch')
    for row in rows:
        spec=r.m.Spec(**row['spec']);spec.validate()
        pair=('0115','0333') if spec.swapped else ('0333','0115')
        if (row['fixed_id'],row['moving_id'])!=pair or spec.smoke:raise RuntimeError('Wrong production pair')
        if [x['integration_steps'] for x in row['evaluations']]!=[7,8,10]:raise RuntimeError('Incomplete integration checks')
        if row['diffusion_coefficient']!=2 or row['amplitude_weight']!=spec.alpha or row['balance_weight']!=spec.beta:
            raise RuntimeError('Objective metadata mismatch')
        for ev in row['evaluations']:
            for side in ('forward','reverse'):
                field=ev[side]
                if not all(math.isfinite(field['metrics'][k]) for k in r.METRICS):raise RuntimeError('Nonfinite metric')
                if r.sha(field['dvf_path'])!=field['dvf_sha256']:raise RuntimeError('Field changed')
    return diagnostics


def flatten(row):
    spec=row['spec'];side='reverse' if spec['swapped'] else 'forward'
    final=row['blocks'][-1]['branches'];energies=final['full_volume']
    return {'run_name':row['run_name'],'arm':spec['arm'],'alpha':spec['alpha'],'beta':spec['beta'],
            'swapped':spec['swapped'],'repeat':spec['repeat'],'export_direction':side,
            **{k:row['evaluations'][0][side]['metrics'][k] for k in r.METRICS},
            **{k:energies[k] for k in ('a1_mm','a2_mm','E1_mm2','E2_mm2','common_mode_rms_mm',
                                      'opposing_mode_rms_mm','relative_rms_imbalance','velocity_cosine','R1_native','R2_native')},
            'midpoint_positive1':final['branch1_midpoint']['image_positive_fraction'],
            'midpoint_positive2':final['branch2_midpoint']['image_positive_fraction'],
            'optimizer_sec':row['optimizer_sec'],'worker_sec':row['worker_sec'],'audit_sec':row['audit']['audit_sec'],
            'peak_cuda_gib':max(row['peak_cuda_gib'],row['audit']['audit_peak_cuda_gib']),
            'peak_reserved_gib':max(row['peak_reserved_gib'],row['audit']['audit_peak_reserved_gib']),
            'actual_adam_updates':row['actual_adam_updates'],'gpu':row['gpu'],
            'stage_updates':'/'.join(str(x['optimizer']['actual_epochs']) for x in row['blocks']),
            'stage_best':'/'.join(str(x['optimizer']['best_epoch']) for x in row['blocks'])}


def pareto(rows):
    def values(x):return (-x['dice'],x['hd95'],x['asd'],x['log_jdet_std_fg'],x['jdet_le0_pct_fg'])
    return [x['run_name'] for x in rows if not any(all(a<=b for a,b in zip(values(y),values(x)))
             and any(a<b for a,b in zip(values(y),values(x))) for y in rows)]


def compare_fields(reference,candidate,masks):
    result={}
    for side in ('forward','reverse'):
        other=('reverse' if side=='forward' else 'forward') if candidate['spec'].get('swapped',False) else side
        a,b=reference['evaluations'][0][side],candidate['evaluations'][0][other]
        av,bv=np.load(a['dvf_path']),np.load(b['dvf_path'])
        result[side]={'full_volume':difference(av,bv),'raw_foreground':difference(av,bv,masks[side]),
                      'metric_delta_candidate_minus_reference':{k:b['metrics'][k]-a['metrics'][k] for k in r.METRICS}}
    result.update(reference=reference['run_name'],candidate=candidate['run_name'],
                  reference_gpu=reference['gpu'],candidate_gpu=candidate['gpu'],
                  reference_updates=[x['optimizer']['actual_epochs'] for x in reference['blocks']],
                  candidate_updates=[x['optimizer']['actual_epochs'] for x in candidate['blocks']])
    return result


def table(rows):
    lines=['| Method | α | β | Dice ↑ | HD95 mm ↓ | ASD mm ↓ | FG SD(log J) | FG folding % |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        lines.append(f"| {x.get('method',x['arm'])} | {x.get('alpha','—')} | {x.get('beta','—')} | {x['dice']:.6f} | {x['hd95']:.6f} | {x['asd']:.6f} | {x['log_jdet_std_fg']:.6f} | {x['jdet_le0_pct_fg']:.6e} |")
    return lines


def main(args):
    root=args.artifact_root/'runs'/r.m.VARIANT/r.m.PHASE
    out=args.artifact_root/'repo_experiments'/r.m.VARIANT/r.m.PHASE
    manifest=r.c.read_json(root/'launch_manifest.json');r.assert_manifest(manifest)
    gate=r.c.read_json(root/'PRODUCTION_FINISHED.json')
    if not gate['passed'] or gate['completed']!=28 or gate['source_hashes']!=r.source_hashes():
        raise RuntimeError('Incomplete or changed production')
    rows=[r.validate(Path(x['path']),manifest,r.m.Spec(**x['spec'])) for x in gate['runs']]
    diagnostic_specs=validate_design(rows,gate,root)
    primaries=primary_rows(rows);flat=[flatten(x) for x in rows]
    lookup={x['run_name']:x for x in rows};flat_lookup={x['run_name']:x for x in flat}
    selected={arm:min((x for x in primaries if x['spec']['arm']==arm),key=r.m.rank_key)['run_name']
              for arm in ('base','amplitude','balance','combined')}
    winner=min(primaries,key=r.m.rank_key)['run_name']
    refs=[{'arm':x['series'],'method':x['method']+'; historical',**{k:x[k] for k in r.METRICS}}
          for x in manifest['references']]
    selected_table=refs+[flat_lookup[name] for name in selected.values()]
    paired=[]
    for x in flat:
        for ref in refs+[flat_lookup[selected['base']]]:
            paired.append({'run_name':x['run_name'],'reference':ref.get('run_name',ref['arm']),
                           **{k+'_delta':x[k]-ref[k] for k in r.METRICS}})
    masks={side:r.c.load_nifti_reoriented(Path(manifest['settings']['oasis_root'])/'test'/f'img{subject}.nii.gz',dtype=np.float32)>0
           for side,subject in (('forward','0333'),('reverse','0115'))}
    discrepancies={}
    for spec,gpu,parent in diagnostic_specs:
        discrepancies[spec.name()]=compare_fields(lookup[parent],lookup[spec.name()],masks)
    old=r.c.read_json(args.artifact_root/'repo_experiments'/r.m.VARIANT/'phase_a'/'results.json')
    old_a1=next(x for x in old['candidates'] if x['spec']['arm']=='A1')
    discrepancies['fresh_vs_historical_A1']=compare_fields(old_a1,lookup[selected['base']],masks)
    branches=[];stages=[];support=[];integrations=[];labels=[]
    for row in rows:
        name=row['run_name']
        for block in row['blocks']:
            info=block['optimizer'];values=block['branches']['full_volume']
            branches.append({'run_name':name,'cps':block['cps'],**{k:v for k,v in values.items() if not isinstance(v,list)},
                             **{f'mean_v{j}_{axis}_mm':values[f'mean_v{j}_xyz_mm'][i] for j in (1,2) for i,axis in enumerate('xyz')}})
            stages.append({'run_name':name,'cps':block['cps'],'gpu':row['gpu'],
                           **{k:info[k] for k in ('actual_epochs','best_epoch','stopping_reason','training_elapsed_sec',
                                                'peak_cuda_gib','peak_reserved_gib','best_loss','best_data_loss','best_reg_loss')},
                           **info['selected_terms']})
            for j in (1,2):support.append({'run_name':name,'cps':block['cps'],'branch':j,**block['branches'][f'branch{j}_midpoint']})
        for ev in row['evaluations']:
            for side in ('forward','reverse'):
                entry=ev[side]
                integrations.append({'run_name':name,'steps':ev['integration_steps'],'direction':side,
                                     **{k:entry['metrics'][k] for k in r.METRICS},
                                     **{k+'_delta_ss7':entry['metrics'][k]-row['evaluations'][0][side]['metrics'][k] for k in r.METRICS}})
                if ev['integration_steps']==7:
                    with (Path(entry['dvf_path']).parent/side/'case_metrics.csv').open() as stream:
                        labels.extend({'run_name':name,'direction':side,**x} for x in csv.DictReader(stream))
    result={'complete':True,'phase':r.m.PHASE,'single_case_exploratory':True,'candidates':rows,
            'all_candidates_original_direction':flat,'references':refs,'selected_by_family':selected,'overall_winner':winner,
            'selected_comparison':selected_table,'pareto_run_names':pareto([flat_lookup[x['run_name']] for x in primaries]),
            'discrepancies':discrepancies,'phase_a_swap_reference':old['input_swap'],
            'paired':paired,'stage_diagnostics':stages,'branch_energies':branches,
            'midpoint_support':support,'integration_sensitivity':integrations,'queue':gate,
            'manifest_sha256':r.sha(root/'launch_manifest.json'),'report_source_sha256':r.sha(__file__),
            'report_dependency_sha256':r.sha(ROOT/'scripts/reports/report_oasis_symmetric_svf_phase_a.py'),
            'report_tests_sha256':r.sha(ROOT/'tests/test_oasis_symmetric_svf_phase_b_report.py'),
            'selection_plan_sha256':{group:r.sha(root/(group+'_plan.json')) for group in ('initial','combined','diagnostics')}}
    out.mkdir(parents=True,exist_ok=False)
    for name,values in (('all_candidates',flat),('selected_comparison',selected_table),('paired_differences',paired),
                        ('stages',stages),('branch_energies',branches),('midpoint_support',support),
                        ('integration_sensitivity',integrations),('per_label',labels)):
        r.c.write_csv(out/(name+'.csv'),values)
    r.atomic_json(out/'results.json',result);r.atomic_json(out/'swap_repeat_reproduction.json',discrepancies)
    write_markdown(out,result,flat_lookup,lookup)
    with (out/'all_candidates.csv').open() as stream:saved=list(csv.DictReader(stream))
    if len(saved)!=28:raise RuntimeError('Incorrect saved candidate count')
    for x,y in zip(saved,flat):
        for k in r.METRICS:
            if float(x[k])!=y[k]:raise RuntimeError('Saved metric mismatch')
    if len(stages)!=84 or len(integrations)!=168:raise RuntimeError('Incomplete stage/integration table')
    r.assert_manifest(manifest)
    acceptance={'complete':True,'fits':28,'independent_replays':28,'primary_fits':24,'diagnostic_fits':4,
                'canonical_statistic_checks':140,'stage_rows':84,'integration_rows':168,'accepted_at':r.c.now_text(),
                'source_unchanged':True,'output_sha256':{p.name:r.sha(p) for p in out.iterdir() if p.is_file()}}
    r.atomic_json(out/'ACCEPTED.json',acceptance)
    print(json.dumps({'report':str(out/'report.md'),'winner':flat_lookup[winner],
                      'selected':[flat_lookup[x] for x in selected.values()],
                      'acceptance':{k:v for k,v in acceptance.items() if k!='output_sha256'}},indent=2))


def write_markdown(out,result,flat_lookup,lookup):
    lines=['# Symmetric B-spline SVF Phase B: rank-0 amplitude/balance ablation','',
           'Single exploratory test case: fixed 0333, moving 0115, cps8→4→2. No across-case SD. All 28 fits start from identity; 24 primary settings and four swap/repeat checks. Both branches refine total SVFs independently. No pyramid-map composition.','',
           'Loss = midpoint NCC9 + 2(R1+R2) + α(E1+E2) + β(√(E1+1e−8)−√(E2+1e−8))². E is native full-volume mean squared velocity-vector norm in mm². Diffusion retains evaluation-grid voxel units. No mask, endpoint loss or segmentation supervision.','',
           'Weights are selected by original-direction Dice, then ASD, HD95 and lower coefficients. Epochs use lowest complete training loss, not labels. Total objective values across different weights are not a common ranking score. Swapped diagnostics use their reverse export for the original-direction table.','',
           '## Selected comparison','']+table(result['selected_comparison'])
    lines += ['',f"Overall Dice-first winner: {result['overall_winner']}.",'',
              '## All primary settings','']+table([x for x in result['all_candidates_original_direction'] if not x['swapped'] and not x['repeat']])
    lines += ['','## Selected branch motion and midpoint coverage','',
              '| Family | RMS v1/v2 mm | Shared/opposing RMS mm | RMS imbalance % | Velocity cosine | Midpoint positive % |',
              '|---|---:|---:|---:|---:|---:|']
    for arm,name in result['selected_by_family'].items():
        x=flat_lookup[name]
        lines.append(f"| {arm} | {x['a1_mm']:.4f}/{x['a2_mm']:.4f} | {x['common_mode_rms_mm']:.4f}/{x['opposing_mode_rms_mm']:.4f} | {100*x['relative_rms_imbalance']:.4f} | {x['velocity_cosine']:.6f} | {100*x['midpoint_positive1']:.3f}/{100*x['midpoint_positive2']:.3f} |")
    lines += ['','Amplitude penalizes shared and opposing motion; a lower RMS alone is not evidence of better registration. Balance equalizes global RMS, not directions. The inherited empty-window NCC behavior is unchanged.','',
              '## Input exchange, same-order repeat and historical reproduction','',
              '| Check | Forward FG EPE mean / p95 / max mm | Reverse FG EPE mean / p95 / max mm | Forward Dice change |',
              '|---|---:|---:|---:|']
    for name,entry in result['discrepancies'].items():
        a,b=entry['forward']['raw_foreground'],entry['reverse']['raw_foreground']
        label=name if name=='fresh_vs_historical_A1' else flat_lookup[name]['arm']+(' swap' if flat_lookup[name]['swapped'] else ' repeat')
        lines.append(f"| {label} | {a['mean_endpoint_mm']:.6f}/{a['p95_endpoint_mm']:.6f}/{a['max_endpoint_mm']:.6f} | {b['mean_endpoint_mm']:.6f}/{b['p95_endpoint_mm']:.6f}/{b['max_endpoint_mm']:.6f} | {entry['forward']['metric_delta_candidate_minus_reference']['dice']:+.6f} |")
    if 'phase_a_swap_reference' in result:
        previous=result['phase_a_swap_reference']['forward']['field_difference_raw_foreground']
        lines += ['',f"Historical Phase-A A1 input-swap foreground EPE mean/p95/max: {previous['mean_endpoint_mm']:.6f}/{previous['p95_endpoint_mm']:.6f}/{previous['max_endpoint_mm']:.6f} mm, evaluated on the same original fixed-image foreground support."]
    lines += ['','These are between-registration differences, not within-fit inverse-cycle residuals. CPU/nonzero-field gradient symmetry does not guarantee identical adaptive GPU optimization. Matched-GPU checks do not establish a unique cause for discrepancies.','',
              '## Selected resource diagnostics','',
              '| Family | Optimizer / worker / replay s | Peak allocated / reserved GiB | Updates by stage | Best updates by stage | GPU |',
              '|---|---:|---:|---:|---:|---:|']
    for arm,name in result['selected_by_family'].items():
        x=flat_lookup[name]
        lines.append(f"| {arm} | {x['optimizer_sec']:.2f}/{x['worker_sec']:.2f}/{x['audit_sec']:.2f} | {x['peak_cuda_gib']:.3f}/{x['peak_reserved_gib']:.3f} | {x['stage_updates']} | {x['stage_best']} | {x['gpu']} |")
    lines += ['','## Selected within-fit inverse and export diagnostics','',
              '100,000 seeded off-grid raw-positive foreground points per direction; all-point errors are shown with in-bounds coverage, not silently restricted to valid points.','',
              '| Family | Direction | Cycle mean / p95 mm | In-bounds % | Export vs retained mean / p95 mm |',
              '|---|---|---:|---:|---:|']
    for arm,name in result['selected_by_family'].items():
        for side,diag in lookup[name]['evaluations'][0]['offgrid'].items():
            if side not in ('forward','reverse'):continue
            cycle,export=diag['cycle_all'],diag['export_vs_retained']
            lines.append(f"| {arm} | {side} | {cycle['mean_mm']:.6f}/{cycle['p95_mm']:.6f} | {100*diag['inbounds_fraction']:.3f} | {export['mean_mm']:.6f}/{export['p95_mm']:.6f} |")
    lines += ['',f"Production queue wall time: {result['queue']['elapsed_sec']/60:.2f} minutes. Shared-load timings are diagnostic; existing jobs are untouched. Seven-step fields remain primary; 8/10-step sensitivity, full histories, per-label scores, scaled-map derivatives, inverse cycles and export/interpolant discrepancies are retained in the CSV/JSON/checkpoint companions.",'',
              'Pareto candidates use all five canonical metrics (higher Dice; lower HD95, ASD, SDlogJ and folding), without inventing an overall regularity score:','']
    lines += ['- '+x for x in result['pareto_run_names']]
    lines += ['','All completed primary candidates are shown; no grid expansion or first-ten evaluation. Historical sources and accepted outputs remain unchanged. Numerical acceptance is not a guarantee of accuracy, invertibility or exact input symmetry.','']
    (out/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,default=Path('/database/inrorspline'))
    main(parser.parse_args())
