"""Recover only approved 4DCT04–06; retain the complete original attempts."""
from __future__ import annotations
import argparse
import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import time

from core import CORRFIELD,REPO,read_json,sha,write_csv,write_json
from report import build

CASES=(4,5,6)
VARIANT='corrfield_componentwise_forest_v1'
NOTE=('CorrField4DCT04,05,06 use the user-approved component-wise forest guard. '
      'It runs the same upstream MST/message passing independently on existing graph components, '
      'adding no edges and changing no parameters. Other CorrField rows retain original successful outputs. '
      'Original failed attempts and the earlier partial report are retained under forest_guard_v1/previous_reports '
      'and the original run directories. Guarded rows are explicitly tagged; they are not unmodified-upstream results.')


def verify_original_sources(review):
    manifest=read_json(review/'source_manifest.json')
    for name,expected in {**manifest['source_hashes'],**manifest['corrfield_hashes']}.items():
        if sha(name)!=expected:
            raise RuntimeError(f'Original numerical source changed: {name}')
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--review-root',type=Path,required=True)
    parser.add_argument('--gpus',default='0,1,2')
    args=parser.parse_args();review=args.review_root;config=read_json(review/'campaign_arguments.json')
    gpus=[int(value) for value in args.gpus.split(',')]
    if len(gpus)!=3 or len(set(gpus))!=3:
        raise ValueError('Specify three distinct idle GPUs for the three approved cases')
    verify_original_sources(review)
    original=Path(config['run_root']);recovery=original/'forest_guard_v1';guard_review=review/'forest_guard_v1'
    recovery.mkdir(exist_ok=False);guard_review.mkdir(exist_ok=False)
    shutil.copytree(review/'reports',guard_review/'previous_reports')
    here=Path(__file__).parent
    files=[here/'forest_guard.py',here/'guarded_runner.py',here/'recover_forest.py',
           REPO/'scripts/reports/audit_lungct_corrected_1mm.py']
    # The test lives at the repository root, outside scripts/.
    files.append(REPO/'tests/test_lungct_corrfield_forest_guard.py')
    hashes={str(path):sha(path) for path in files}
    for path in files:
        shutil.copy2(path,guard_review/path.name)
    commands=[]
    for case,gpu in zip(CASES,gpus):
        cmd=[sys.executable,str(here/'guarded_runner.py'),'--method','corrfield','--dataset','4dct',
             '--case',str(case),'--data-root',config['data_root'],'--out-dir',str(recovery/f'corrfield_4dct_{case:02d}'),
             '--gpu',str(gpu),'--corrfield-source',config['corrfield_source']]
        commands.append(dict(case=case,gpu=gpu,command=cmd))
    write_json(guard_review/'manifest.json',dict(variant=VARIANT,cases=list(CASES),commands=commands,
               source_hashes=hashes,parameters=CORRFIELD,
               approved='User explicitly approved the tested component-wise graph guard for 4DCT04–06 on2026-11-19.',
               original_manifest_sha256=sha(review/'source_manifest.json'),
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    running=[]
    for item in commands:
        log=(guard_review/f"case{item['case']:02d}.log").open('w')
        process=subprocess.Popen(item['command'],stdout=log,stderr=subprocess.STDOUT)
        running.append((item,process,log))
    failures=[]
    for item,process,log in running:
        rc=process.wait();log.close()
        if rc:
            failures.append(dict(case=item['case'],returncode=rc))
    if failures:
        write_json(guard_review/'failed.json',failures)
        raise RuntimeError(f'Guarded failures retained: {failures}')
    verify_original_sources(review)
    for name,expected in hashes.items():
        assert sha(name)==expected
    assembly=guard_review/'assembled_cases';assembly.mkdir(exist_ok=False)
    for method,ds,cases in [('mrdbscp','copd',range(1,11)),('corrfield','4dct',range(1,11)),('corrfield','copd',range(1,11))]:
        for case in cases:
            key=f'{method}_{ds}_{case:02d}'
            target=recovery/key if method=='corrfield' and ds=='4dct' and case in CASES else original/key
            receipt=read_json(target/'accepted.json');evaluation=read_json(target/'evaluation.json')
            assert receipt['status']=='accepted'
            assert receipt['evaluation_sha256']==sha(target/'evaluation.json')
            assert sha(evaluation['field_path'])==evaluation['field_sha256']
            if target.parent==recovery:
                assert evaluation['implementation_variant']==VARIANT
                assert evaluation['parameters']==CORRFIELD
            (assembly/key).symlink_to(target,target_is_directory=True)
    # Append recovery disclosure to the original reporting interface in memory.
    # No frozen numerical/reporting source file is changed.
    import report
    report.LIMITATIONS+=' '+NOTE
    build(assembly,Path(config['reference_root']),review,require_all=True)
    results=read_json(review/'reports/results.json')
    results['recovery_variant']=VARIANT
    results['guarded_cases']=[dict(dataset='4dct',case=case) for case in CASES]
    for dataset,table_name in [('4dct','02_4dct_three_methods'),('copd','03_copd_three_methods')]:
        rows=results['tables'][dataset]
        for row in rows:
            row['CorrField_implementation']=(VARIANT if dataset=='4dct' and row['case'] in CASES else 'original_standalone')
        write_csv(review/'reports'/f'{table_name}.csv',rows)
        write_json(review/'reports'/f'{table_name}.json',rows)
    write_json(review/'reports/results.json',results)
    write_json(guard_review/'accepted.json',dict(status='accepted',cases=list(CASES),variant=VARIANT,
               assembled_cases=str(assembly),source_hashes=hashes,
               report_receipt_sha256=sha(review/'reports/accepted.json'),
               completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    write_json(review/'campaign_accepted.json',dict(status='accepted',new_registrations=30,
               successful_original=27,successful_guarded=3,original_failed_attempts_retained=3,
               original_source_hashes_unchanged=True,assembled_cases=str(assembly),
               recovery_manifest_sha256=sha(guard_review/'manifest.json'),
               report_receipt_sha256=sha(review/'reports/accepted.json')))
    print('GUARDED RECOVERY ACCEPTED',flush=True)


if __name__=='__main__':
    main()
