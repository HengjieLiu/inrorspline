"""Create a fresh campaign identity and immutable executable source snapshot."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
from .protocol import dump,sha,stamp

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--execution-id',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[3]
    run=a.artifact_root/'runs'/a.execution_id
    run.mkdir(parents=True,exist_ok=True)
    if (run/'config.json').exists():raise FileExistsError('Existing campaign identity is immutable')
    frozen=run/'source';frozen.mkdir(exist_ok=False)
    selected=[]
    for name in ['scripts/experiments/lungct_hu_crop_lambda','scripts/experiments/lungct_no_sweep_repro',
                 'scripts/experiments/lungct_corrected_1mm','scripts/data/common','scripts/data/preprocessing/lungct',
                 'experiments/2026-11-19_lungct_hu_crop_lambda']:
        selected.extend(p for p in (repo/name).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.m','.yaml','.txt') and '__pycache__' not in p.parts)
    selected += [repo/name for name in ['AGENTS.md','docs/implementation/lungct_hu_crop_lambda.md','scripts/experiments/common_dctfield.py','tests/test_lungct_hu_crop_lambda.py','tests/test_lungct_corrected_1mm.py']]
    hashes={}
    for src in sorted(set(selected)):
        rel=src.relative_to(repo);target=frozen/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,target);hashes[str(rel)]=sha(target)
    config=dict(execution_id=a.execution_id,created=stamp(),run_root=str(run),
        prepared_root=str(a.artifact_root/'prepared'/a.execution_id),
        review_root=str(a.artifact_root/'repo_experiments'/a.execution_id),
        anatomy_root=str(a.artifact_root/'prepared'/a.execution_id/'anatomy'),
        raw_root=str(a.raw_root),baseline_root=str(a.baseline_root),frozen_root=str(frozen),
        gpu_pool=[0,1,2,3],memory_multiplier=1.2,memory_reserve_gib=2,
        numerical_image_id='sha256:dab700c415830dcf07c16ef7f2901deed95bb104eda388514c3ce74a43354ba8',
        anatomy_image_id='sha256:047cfe9e32a9001de0f55d04bffdb1ad81bf59b8f9f7de049103cf5d15d0acb7',
        lambda_center=.055,exponent_cap=8,tolerance_mm=1e-6,
        source_git_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        dirty_source_snapshot=True,first_batch=['4dct_01','4dct_08','copd_01','copd_08'])
    dump(run/'config.json',config)
    dump(frozen/'source_manifest.json',dict(source_hashes=hashes,config_sha256=sha(run/'config.json'),created=stamp()))
    Path(config['review_root']).mkdir(parents=True,exist_ok=True)
    dump(Path(config['review_root'])/'execution_identity.json',dict(config_sha256=sha(run/'config.json'),source_manifest_sha256=sha(frozen/'source_manifest.json')))
    print(run/'config.json',flush=True)

if __name__=='__main__':main()
