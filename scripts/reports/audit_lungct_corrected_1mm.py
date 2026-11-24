"""Independent native-coordinate audit of the corrected LungCT campaign.

Uses raw text landmarks, NumPy indexing and SciPy interpolation, without
importing the campaign evaluator or optimizer. Does not launch registration.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--review-root',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,help='Optional assembled cohort containing references to accepted case directories.')
    parser.add_argument('--allow-incomplete',action='store_true',help='Audit completed cases while preserving failed/missing cases explicitly.')
    args=parser.parse_args();review=args.review_root
    if not args.allow_incomplete:
        assert read(review/'campaign_accepted.json')['status']=='accepted'
    config=read(review/'campaign_arguments.json')
    run_root=args.run_root or Path(config['run_root']);data_root=Path(config['data_root'])
    records=[]
    for path in sorted(run_root.glob('*_*/evaluation.json')):
        result=read(path);case=result['case'];dataset=result['dataset']
        data=data_root/('4DCT' if dataset=='4dct' else 'COPD')/f'case{case:02d}'
        geometry=read(data/'preprocess_audit.json')
        assert geometry['crop_convention']=='corrected'
        raw={}
        for role in ['fixed','moving']:
            source=next(v for k,v in geometry['landmark_paths'].items() if k.startswith(role))
            raw[role]=np.loadtxt(source)[:,::-1]-1
            assert raw[role].shape==(300,3)
        origin=np.asarray(geometry['crop_bounds_zyx_inclusive'])[:,0]
        spacing=np.asarray(geometry['native_spacing_zyx_mm'])
        positions=raw['fixed']-origin
        np.testing.assert_array_equal(positions,np.rint(positions))
        indices=positions.astype(np.int64)
        native=np.load(result['native_field_path'],mmap_mode='r')
        displacement=native[indices[:,0],indices[:,1],indices[:,2]].astype(np.float64)
        snap_pred=np.rint(raw['fixed']+displacement)
        snap=np.linalg.norm((snap_pred-raw['moving'])*spacing,axis=1)
        sub=np.linalg.norm((raw['fixed']+displacement-raw['moving'])*spacing,axis=1)
        saved=csv_rows(result['per_landmark_path'])
        np.testing.assert_array_equal(snap,[float(r['original_snap']) for r in saved])
        np.testing.assert_array_equal(sub,[float(r['original_subvoxel']) for r in saved])
        assert abs(snap.mean()-result['original_snap'])<1e-12
        assert abs(snap.std(ddof=1)-result['original_snap_sd'])<1e-12
        field=np.load(result['field_path'],mmap_mode='r')
        reg_shape=np.asarray(geometry['registration_shape_zyx'])
        crop_shape=np.asarray(geometry['crop_shape_zyx'])
        coordinates=(positions*(reg_shape-1)/(crop_shape-1)).T
        sampled=np.stack([map_coordinates(field[...,c],coordinates,order=1,mode='nearest',prefilter=False)
                          for c in range(3)],axis=-1)
        independent_native=(sampled/spacing.astype(np.float32)).astype(np.float32)
        resize_error=float(np.max(np.abs(independent_native-displacement)))
        assert resize_error<2e-4, (path,resize_error)
        assert digest(result['field_path'])==result['field_sha256']
        assert digest(result['native_field_path'])==result['native_field_sha256']
        backend=result['backend'];is_mr=result['method']=='mrdbscp'
        assert backend['requested_cudnn_benchmark']==backend['effective_cudnn_benchmark']==is_mr
        assert backend['requested_cudnn_deterministic']==backend['effective_cudnn_deterministic']==(not is_mr)
        if is_mr:
            summary=read(result['solver_summary_path'])
            assert summary['training']['stage_epochs']==[400,400,400,400,900]
            assert summary['training']['stage_full_cps']==[64,32,16,8,4]
            assert summary['training']['stage_downsample']==[16,8,4,2,1]
            assert summary['training']['ptv_isotv_weight']==0.055
            history=csv_rows(Path(result['solver_summary_path']).parent/'stage_history.csv')
            assert len(history)==2500
            for key in ['loss','data_loss','reg_loss','reg_weight','lr']:
                assert np.isfinite([float(row[key]) for row in history]).all()
        else:
            assert result['intensity_profile']=='raw_minus1024_unclipped_HU'
            assert result['registration_mask']=='fixed_R231_lung_only'
        records.append(dict(dataset=dataset,case=case,method=result['method'],landmarks=300,
                            native_snap_max_error_mm=0.0,native_subvoxel_max_error_mm=0.0,
                            independent_resize_max_native_voxel_error=resize_error))
        print('audited',dataset,case,result['method'],resize_error,flush=True)
    assert len(records)==30 if not args.allow_incomplete else 0<len(records)<30
    manifest=read(review/'source_manifest.json')
    for name,expected in {**manifest['source_hashes'],**manifest['corrfield_hashes'],**manifest['raw_input_hashes']}.items():
        assert digest(name)==expected,name
    preparation=read(review/'preparation.json')
    for case in preparation['cases']:
        d=data_root/('4DCT' if case['dataset']=='4dct' else 'COPD')/f"case{case['case']:02d}"
        for name,expected in case['array_hashes'].items():
            assert digest(d/name)==expected,str(d/name)
    results=read(review/'reports/results.json')
    assert len(results['evaluations'])==40+len(records)
    if not args.allow_incomplete:
        assert len(csv_rows(review/'reports/all_per_landmark_tre.csv'))==21000
    for key in results['tables']:
        assert [r['case'] for r in results['tables'][key]]==list(range(1,11))
    payload=dict(status='accepted_partial' if args.allow_incomplete else 'accepted',new_method_cases=len(records),native_landmark_pairs_checked=300*len(records),
                 native_snap_max_error_mm=0.0,native_subvoxel_max_error_mm=0.0,
                 prepared_and_raw_input_hashes_unchanged=True,numerical_source_hashes_unchanged=True,
                 records=records,audit_source_sha256=digest(__file__),
                 limitation='Independent resize tolerance tests floating interpolation; no additional optimization was run.')
    target=review/('posthoc_partial_audit.json' if args.allow_incomplete else 'posthoc_audit.json')
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(payload,indent=2)+'\n')
    print('INDEPENDENT AUDIT ACCEPTED',flush=True)


if __name__=='__main__':
    main()
