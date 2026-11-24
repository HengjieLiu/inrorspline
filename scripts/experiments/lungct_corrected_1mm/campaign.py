"""Preflight gates, immutable source snapshot, and one-job-per-GPU queue."""
from __future__ import annotations
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from core import REPO,case_data,read_json,sha,write_json
from report import build


def production_jobs():
    return ([dict(method='mrdbscp',dataset='copd',case=c) for c in range(1,11)]+
            [dict(method='corrfield',dataset=ds,case=c) for ds in ['4dct','copd'] for c in range(1,11)])


def key(job):return f"{job['method']}_{job['dataset']}_{job['case']:02d}"


def source_files():
    return sorted(set(list(Path(__file__).parent.glob('*.py'))+
                      list((REPO/'scripts/experiments/lungct_no_sweep_repro').glob('*.py'))+
                      list((REPO/'scripts/data/preprocessing/lungct').glob('*.py'))+
                      list((REPO/'scripts/data/common').glob('*.py'))+
                      [REPO/'tests/test_lungct_corrected_1mm.py']))


def freeze(args):
    receipt=args.review_root/'source_manifest.json'
    files=source_files();hashes={str(p):sha(p) for p in files}
    external={str(p):sha(p) for p in sorted((args.corrfield_source/'corrfield').glob('*.py'))}
    if receipt.exists():
        old=read_json(receipt)
        if old['source_hashes']!=hashes or old['corrfield_hashes']!=external:raise RuntimeError('Frozen source drift')
        return old
    snap=args.review_root/'source_snapshot'
    for p in files:
        target=snap/p.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in external:
        p=Path(name);target=snap/'corrfield_source'/p.relative_to(args.corrfield_source)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    raw={}
    projected=0
    for ds in ['4dct','copd']:
        for c in range(1,11):
            a=read_json(case_data(args.data_root,ds,c)/'preprocess_audit.json')
            for name in list(a['image_paths'].values())+list(a['landmark_paths'].values()):
                raw[name]=sha(name)
            import numpy as np
            projected+=int((np.prod(a['registration_shape_zyx'])+np.prod(a['crop_shape_zyx']))*12*(2 if ds=='copd' else 1))
    free=shutil.disk_usage(args.run_root.parent).free
    if free<projected*2:raise RuntimeError('Insufficient free artifact storage')
    r=dict(source_hashes=hashes,corrfield_hashes=external,raw_input_hashes=raw,
           repo_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
           created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           projected_new_field_bytes=projected,available_bytes=free,
           planned_production=production_jobs(),prepared_receipt_sha256=sha(args.preparation_receipt),
           reference_receipt_sha256=sha(args.reference_root/'accepted.json'),
           runtime=read_json(args.review_root/'runtime.json'))
    write_json(receipt,r);return r


def queue(jobs,args,phase,manifest):
    pending=list(jobs);running=[];failed=[];done=[]
    args.run_root.mkdir(parents=True,exist_ok=True);logs=args.review_root/'logs';logs.mkdir(exist_ok=True)
    while pending or running:
        while pending and len(running)<len(args.gpus):
            job=pending.pop(0);name=key(job);smoke=job.get('smoke',False)
            dest=args.run_root/('smokes' if smoke else '')/name
            if (dest/'accepted.json').exists():done.append(name);continue
            if dest.exists():raise RuntimeError(f'Unaccepted existing attempt retained: {dest}')
            gpu=next(g for g in args.gpus if g not in [x['gpu'] for x in running])
            # Fail before touching a case if numerical source drifted.
            if any(sha(p)!=v for p,v in {**manifest['source_hashes'],**manifest['corrfield_hashes']}.items()):raise RuntimeError('Source drift')
            cmd=[sys.executable,str(Path(__file__).with_name('runner.py')),'--method',job['method'],
                 '--dataset',job['dataset'],'--case',str(job['case']),'--data-root',str(args.data_root),
                 '--out-dir',str(dest),'--gpu',str(gpu),'--corrfield-source',str(args.corrfield_source)]
            if smoke:cmd.append('--smoke')
            log=(logs/f'{phase}_{name}.log').open('w')
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            running.append(dict(job=job,name=name,gpu=gpu,process=proc,log=log,dest=dest,cmd=cmd))
            print('LAUNCHED',phase,name,'gpu',gpu,flush=True)
        for item in list(running):
            rc=item['process'].poll()
            if rc is None:continue
            item['log'].close();running.remove(item)
            if rc or not (item['dest']/'accepted.json').exists():failed.append(dict(job=item['job'],returncode=rc))
            else:done.append(item['name'])
            print('FINISHED',phase,item['name'],rc,flush=True)
            if phase=='production':build(args.run_root,args.reference_root,args.review_root)
        write_json(args.review_root/'state.json',dict(phase=phase,done=done,failed=failed,pending=len(pending),
                   running=[dict(name=x['name'],gpu=x['gpu'],pid=x['process'].pid,cmd=x['cmd']) for x in running]))
        if running:time.sleep(2)
    if failed:raise RuntimeError(f'{phase} failures: {failed}')
    return done


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--review-root',type=Path,required=True)
    p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--preparation-receipt',type=Path,required=True)
    p.add_argument('--corrfield-source',type=Path,default=Path('/opt/corrfield-source'))
    p.add_argument('--gpus',default='0,1,2,3');args=p.parse_args();args.gpus=[int(v) for v in args.gpus.split(',')]
    assert read_json(args.preparation_receipt)['status']=='accepted'
    assert read_json(args.reference_root/'accepted.json')['status']=='accepted'
    args.run_root.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.review_root/'campaign_arguments.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    manifest=freeze(args)
    smokes=[dict(method='mrdbscp',dataset='copd',case=c,smoke=True) for c in [1,2]]
    for ds in ['4dct','copd']:
        import numpy as np
        largest=max(range(1,11),key=lambda c:int(np.prod(read_json(case_data(args.data_root,ds,c)/'preprocess_audit.json')['registration_shape_zyx'])))
        smokes.append(dict(method='corrfield',dataset=ds,case=largest))
    queue(smokes,args,'preflight',manifest)
    write_json(args.review_root/'preflight_accepted.json',dict(status='accepted',smokes=smokes,production_reuse=2))
    queue(production_jobs(),args,'production',manifest)
    assert len(list(args.run_root.glob('*_*/accepted.json')))==30
    build(args.run_root,args.reference_root,args.review_root,require_all=True)
    write_json(args.review_root/'campaign_accepted.json',dict(status='accepted',new_registrations=30,
               report_receipt_sha256=sha(args.review_root/'reports/accepted.json')))
    print('CAMPAIGN ACCEPTED',flush=True)


if __name__=='__main__':main()
