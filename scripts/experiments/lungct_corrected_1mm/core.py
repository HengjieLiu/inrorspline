"""Explicit field geometry, immutable receipts, and shared TRE evaluation."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
LEGACY_DIR = REPO/'scripts/experiments/lungct_no_sweep_repro'
spec = importlib.util.spec_from_file_location('_lungct_numeric_common', LEGACY_DIR/'common.py')
numeric = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = numeric
spec.loader.exec_module(numeric)

ARCHIVE_SHA = 'a2aaa91add1286ef3cc77a0bf049a4d3ceed57dd4dfa4bcd39b7cac559a57472'
CORRFIELD = dict(alpha=2.5,beta=150.,gamma=5.,delta=1,lambd=0.,sigma=1.4,sigma1=1.,
                 L=[16,8],N=[6,3],Q=[2,1],R=[3,2],T=['n','n'])


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows: raise ValueError('Cannot write an empty result table')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def phases(dataset):
    return ('T00','T50') if dataset.lower()=='4dct' else ('iBH','eBH')


def case_data(root,dataset,case):
    return Path(root)/('4DCT' if dataset.lower()=='4dct' else 'COPD')/f'case{case:02d}'


def historical_path(path,artifact_root):
    # The private audit records host paths; resolve only the known artifact
    # relative suffix, never a guessed arbitrary host prefix.
    text=str(path);marker='/workspace/inrorspline/'
    if marker in text:return Path(artifact_root)/text.split(marker,1)[1]
    return Path(path)


def corrfield_to_zyx(native,shape_zyx):
    a=np.asarray(native,dtype=np.float32)
    expected=tuple(shape_zyx)[::-1]+(3,)
    if a.shape!=expected or not np.isfinite(a).all():
        raise ValueError(f'Expected finite XYZ CorrField output {expected}, got {a.shape}')
    return np.ascontiguousarray(a.transpose(2,1,0,3)[...,::-1])


def finite_field(field,shape):
    if field.shape!=(*tuple(shape),3) or field.dtype!=np.float32:
        raise ValueError(f'Incorrect field contract: {field.shape}/{field.dtype}')
    for z in range(0,len(field),16):
        if not np.isfinite(field[z:z+16]).all():raise ValueError('Non-finite field')


def scores(field,fixed,moving,spacing):
    fixed=np.asarray(fixed,dtype=np.float64);moving=np.asarray(moving,dtype=np.float64)
    shape=np.asarray(field.shape[:3])
    if fixed.shape!=(300,3) or moving.shape!=(300,3):raise ValueError('Expected300 landmark pairs')
    if np.any(fixed<0) or np.any(fixed>shape-1):raise ValueError('Fixed landmark outside field')
    d0=numeric.sample_dense_disp_nearest(field,fixed)
    d1=numeric.sample_dense_disp_trilinear(field,fixed)
    pred0=np.rint(fixed+d0);pred1=fixed+d1
    e0=np.linalg.norm((pred0-moving)*spacing,axis=1)
    e1=np.linalg.norm((pred1-moving)*spacing,axis=1)
    return dict(snap_mean=float(e0.mean()),snap_sd=float(e0.std(ddof=1)),
                subvoxel_mean=float(e1.mean()),subvoxel_sd=float(e1.std(ddof=1))), (d0,d1,pred0,pred1,e0,e1)


def evaluate(field_path,data_dir,out_dir,identity):
    """One evaluator for both methods, with native absolute-coordinate proof."""
    field_path=Path(field_path);data_dir=Path(data_dir);out_dir=Path(out_dir)
    a=read_json(data_dir/'preprocess_audit.json');fx,mv=phases(a['dataset'])
    field=np.load(field_path,mmap_mode='r');shape=a['registration_shape_zyx']
    finite_field(field,shape)
    spacing=np.asarray(a['native_spacing_zyx_mm']);origin=np.asarray(a['crop_bounds_zyx_inclusive'])[:,0]
    f=np.load(data_dir/f'landmarks_fixed_{fx}_crop_zyx.npy')
    m=np.load(data_dir/f'landmarks_moving_{mv}_crop_zyx.npy')
    native=numeric.resize_reg1mm_disp_to_native_crop_vox(field,a['crop_shape_zyx'],spacing)
    native_path=out_dir/'disp_vox_native_crop_zyx.npy'
    if native_path.exists():raise FileExistsError(native_path)
    np.save(native_path,native,allow_pickle=False)
    stats,v=scores(native,f,m,spacing)
    d0,d1,p0,p1,e0,e1=v
    absolute=np.linalg.norm((np.rint(f+origin+d0)-(m+origin))*spacing,axis=1)
    if not np.array_equal(e0,absolute):raise ValueError('Crop/full native TRE disagreement')
    regstats,_=scores(field,np.load(data_dir/f'landmarks_fixed_{fx}_reg1mm_zyx.npy'),
                       np.load(data_dir/f'landmarks_moving_{mv}_reg1mm_zyx.npy'),np.ones(3))
    rows=[]
    for i in range(300):
        r=dict(landmark_id=i,original_snap=float(e0[i]),original_subvoxel=float(e1[i]))
        for axis,j in zip('zyx',range(3)):
            r.update({f'fixed_{axis}_native':float(f[i,j]+origin[j]),f'moving_{axis}_native':float(m[i,j]+origin[j]),
                      f'disp_{axis}_native_vox':float(d1[i,j]),f'predicted_snap_{axis}_native':float(p0[i,j]+origin[j]),
                      f'predicted_subvoxel_{axis}_native':float(p1[i,j]+origin[j])})
        rows.append(r)
    write_csv(out_dir/'per_landmark.csv',rows)
    for z in range(0,len(field),16):
        if not np.array_equal(field[z:z+16],np.load(field_path,mmap_mode='r')[z:z+16]):
            raise ValueError('Field reload mismatch')
    result=dict(**identity,dataset=a['dataset'],case=int(a['case_id']),landmarks=300,
                status='completed',original_snap=stats['snap_mean'],original_snap_sd=stats['snap_sd'],
                original_subvoxel=stats['subvoxel_mean'],original_subvoxel_sd=stats['subvoxel_sd'],
                reg1mm_snap=regstats['snap_mean'],reg1mm_subvoxel=regstats['subvoxel_mean'],
                baseline_mean=float(np.linalg.norm((f-m)*spacing,axis=1).mean()),
                crop_full_tre_max_error_mm=float(np.max(np.abs(e0-absolute))),
                field_path=str(field_path),field_sha256=sha(field_path),native_field_path=str(native_path),
                native_field_sha256=sha(native_path),per_landmark_path=str(out_dir/'per_landmark.csv'),
                field_contract=dict(layout='ZYX3',components='ZYX',units='nominal_1mm_grid_voxels',
                                    semantics='moving(fixed+u)'),
                geometry=dict(crop_convention=a.get('crop_convention','corrected' if a['dataset']=='4dct' else 'legacy'),
                              **{k:a[k] for k in ['crop_bounds_zyx_inclusive','crop_shape_zyx',
                                                 'registration_shape_zyx','native_spacing_zyx_mm']}),
                prepared_audit_sha256=sha(data_dir/'preprocess_audit.json'))
    write_json(out_dir/'evaluation.json',result)
    return result
