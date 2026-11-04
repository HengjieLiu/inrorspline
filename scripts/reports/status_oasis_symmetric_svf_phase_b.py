#!/usr/bin/env python
"""Read-only concise progress; only independently replayed runs have metrics."""
import argparse
import json
from pathlib import Path


def main(args):
    root=args.artifact_root/'runs'/'oasis_seed3_rank000_bspline_two_total_svf_midpoint_ablation_v1'/'phase_b_v1'
    status=json.loads((root/'live_progress.json').read_text())
    result={k:status[k] for k in ('updated_at','group','experiment_completed','experiment_total','estimated_remaining_sec','elapsed_sec')}
    result['active']=[{k:item[k] for k in ('gpu','action','spec_name','worker_progress')} for item in status['active']]
    jobs=[]
    for group in ('initial','combined','diagnostics'):
        p=root/(group+'_progress.json')
        if p.exists():jobs.extend(json.loads(p.read_text())['jobs'])
    result['failures']=[{'run':x['spec_name'],'records':x['failures']} for x in jobs if x['failures'] or x['status']=='failed']
    completed=[]
    for job in jobs:
        if job['status']!='complete':continue
        row=json.loads((Path(job['path'])/'summary.json').read_text())
        audit=json.loads((Path(job['path'])/'independent_audit.json').read_text())
        if not audit['passed']:raise RuntimeError('Invalid completed job')
        spec=row['spec'];side='reverse' if spec['swapped'] else 'forward'
        metrics=row['evaluations'][0][side]['metrics']
        energies=row['blocks'][-1]['branches']['full_volume']
        completed.append({**{k:spec[k] for k in ('arm','alpha','beta','swapped','repeat')},
                          **{k:metrics[k] for k in ('dice','hd95','asd','log_jdet_std_fg','jdet_le0_pct_fg')},
                          'common_rms_mm':energies['common_mode_rms_mm'],
                          'rms1_mm':energies['a1_mm'],'rms2_mm':energies['a2_mm'],
                          'gpu':row['gpu'],'worker_sec':row['worker_sec'],'audit_sec':audit['audit_sec']})
    result['complete_candidates']=completed if args.all else completed[-4:]
    result['complete_candidates_shown']='all' if args.all else 'last_four_in_plan_order'
    result['production_finished']=(root/'PRODUCTION_FINISHED.json').is_file()
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',type=Path,default=Path('/database/inrorspline'))
    parser.add_argument('--all',action='store_true')
    main(parser.parse_args())
