"""Prepare immutable, physical-coordinate case bundles from accepted P87 outputs."""
from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.ndimage import map_coordinates

from .geometry import PhysicalField, compact, display_geometry, invert_points, to_lps

METHODS = [('mrdbscp', 'MR-D-BSCP'), ('ptvreg', 'pTVReg'), ('corrfield', 'CorrField')]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def save_array(out, key, array, channels=1, tolerance=2.):
    if array.dtype == np.uint8:
        a, err = array, 0.
    else:
        a, err = compact(array, tolerance)
    path = out/f'{key}.bin.gz'
    with gzip.GzipFile(filename=str(path), mode='wb', compresslevel=1, mtime=0) as f:
        f.write(a.tobytes(order='C'))
    # Exact display-array round trip; scientific precision assessed separately.
    with gzip.open(path, 'rb') as f:
        decoded = np.frombuffer(f.read(), dtype=a.dtype).reshape(a.shape)
    if not np.array_equal(a, decoded):
        raise ValueError('Display asset reload mismatch')
    return dict(file=path.name, dtype=a.dtype.str, channels=channels, shape=list(a.shape),
                bytes=a.nbytes, compressed_bytes=path.stat().st_size, sha256=sha(path),
                max_display_error=err)


def prepare_case(args, ds, case, records):
    key = f'{ds}_{case:02d}'; out = args.output_root/'cases'/key
    if out.exists():
        if args.resume and (out/'accepted.json').exists():
            receipt=read(out/'accepted.json')
            assert sha(out/'manifest.json') == receipt['manifest_sha256']
            manifest=read(out/'manifest.json')
            for a in manifest['assets'].values():
                assert sha(out/a['file']) == a['sha256']
            print('accepted bundle reused', key, flush=True)
            return manifest
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started=time.monotonic()
    prepared=args.data_root/('4DCT' if ds=='4dct' else 'COPD')/f'case{case:02d}'
    audit=read(prepared/'preprocess_audit.json')
    assert audit['crop_convention']=='corrected'
    geom=display_geometry(audit)
    shape=tuple(audit['registration_shape_zyx']); rawshape=tuple(audit['raw_shape_zyx'])
    sp=np.asarray(audit['native_spacing_zyx_mm'][::-1]); bounds=np.asarray(audit['crop_bounds_zyx_inclusive'])
    scale=(np.asarray(audit['crop_shape_zyx'])-1)/(np.asarray(shape)-1)
    origin=bounds[:,0]
    fixed_phase=audit['fixed_phase']; moving_phase=audit['moving_phase']
    fpts=np.load(prepared/f'landmarks_fixed_{fixed_phase}_native0_zyx.npy')[:,::-1].astype(np.float64)
    mpts=np.load(prepared/f'landmarks_moving_{moving_phase}_native0_zyx.npy')[:,::-1].astype(np.float64)
    fworld=to_lps(fpts, sp, rawshape[0]); mworld=to_lps(mpts, sp, rawshape[0])
    dsdir='4DCT' if ds=='4dct' else 'COPD'
    maskpaths={role:args.mask_root/dsdir/f'case{case:02d}'/f'case{case:02d}_{ph}_R231_labels_zyx.npy'
               for role,ph in [('fixed',fixed_phase),('moving',moving_phase)]}
    masks={k:np.load(p,mmap_mode='r') for k,p in maskpaths.items()}
    side=masks['fixed'][tuple(np.rint(fpts[:,::-1]).astype(int).T)]
    mside=masks['moving'][tuple(np.rint(mpts[:,::-1]).astype(int).T)]
    assert np.all(np.isin(side,[1,2])) and np.array_equal(side,mside)
    imagepaths=audit['image_paths']
    fixed_path=next(Path(v) for k,v in imagepaths.items() if k.startswith('fixed'))
    moving_path=next(Path(v) for k,v in imagepaths.items() if k.startswith('moving'))
    rawmov=np.memmap(moving_path,dtype=np.int16,mode='r',shape=rawshape)
    fixed=np.load(prepared/f'fixed_{fixed_phase}_reg1mm_hu_zyx.npy',mmap_mode='r')
    moving=np.load(prepared/f'moving_{moving_phase}_reg1mm_hu_zyx.npy',mmap_mode='r')
    assets={}; source_files=[prepared/'preprocess_audit.json',fixed_path,moving_path,*maskpaths.values()]
    for role,image in [('fixed',fixed),('moving',moving)]:
        assets[f'{role}_image']=save_array(out,f'{role}_image',image)
        label=np.empty(shape,dtype=np.uint8)
        for z in range(0,shape[0],12):
            zz,yy,xx=np.meshgrid(np.arange(z,min(z+12,shape[0])),np.arange(shape[1]),np.arange(shape[2]),indexing='ij')
            coords=[origin[i]+v*scale[i] for i,v in enumerate((zz,yy,xx))]
            label[z:z+12]=map_coordinates(masks[role],coords,order=0,mode='constant',cval=0,prefilter=False)
        assets[f'{role}_mask']=save_array(out,f'{role}_mask',label)
    initial=np.linalg.norm((fpts-mpts)*sp,axis=1)
    landmarks=[dict(id=i,side='right' if side[i]==1 else 'left',fixed_lps_mm=fworld[i].tolist(),
                    moving_lps_mm=mworld[i].tolist(),tre_mm=dict(initial=float(initial[i])),methods={}) for i in range(300)]
    methods=[]; validation={}; umax=0.; diffmax=float(np.max(np.abs(np.asarray(fixed)-moving)))
    for method,name in METHODS:
        r=next(r for r in records if r['method']==method and r['dataset']==ds and r['case']==case)
        path=Path(r['field_path']); assert sha(path)==r['field_sha256']
        source_files.extend([path,Path(r['per_landmark_path'])])
        if method=='ptvreg':
            with h5py.File(path,'r') as f:
                data=f['Tptv_rsz'][:].astype(np.float32)
                cb=np.asarray(f['crop_v_saved']).T-1
            field=PhysicalField(data,sp,cb,ptv=True)
        else:
            data=np.load(path,mmap_mode='r')
            g=r['geometry']; cb=np.asarray(g['crop_bounds_zyx_inclusive'])[::-1]
            field=PhysicalField(data,sp,cb,reg_geometry=g)
        saved=rows(r['per_landmark_path']); assert len(saved)==300
        if method=='ptvreg':
            disp=data[:,fpts[:,2].astype(int),fpts[:,1].astype(int),fpts[:,0].astype(int)].T.astype(np.float64)/sp
        else:
            native=np.load(r['native_field_path'],mmap_mode='r')
            native_origin=np.asarray(r['geometry']['crop_bounds_zyx_inclusive'])[:,0]
            fi=(fpts[:,::-1]-native_origin).astype(int)
            disp=native[tuple(fi.T)][:,::-1].astype(np.float64)
            source_files.append(Path(r['native_field_path']))
        residual=(np.rint(fpts+disp)-mpts)*sp
        # Preserve the audited reduction order, including last-bit FP64 sums.
        tre=np.linalg.norm(residual if method=='ptvreg' else residual[:,::-1],axis=1)
        expected=np.asarray([float(p['original_snap']) for p in saved])
        np.testing.assert_array_equal(tre,expected)
        inverse=invert_points(field,mpts)
        for i,p in enumerate(inverse):
            point=p.pop('point_native_xyz')
            p['point_lps_mm']=None if point is None else to_lps(point,sp,rawshape[0]).tolist()
            landmarks[i]['methods'][method]=p
            landmarks[i]['tre_mm'][method]=float(expected[i])
        warped=np.empty(shape,dtype=np.float32)
        warpedmask=np.empty(shape,dtype=np.uint8)
        maps=np.empty((*shape,4),dtype=np.float32)
        valid_count=0
        for z in range(0,shape[0],12):
            zz,yy,xx=np.meshgrid(np.arange(z,min(z+12,shape[0])),np.arange(shape[1]),np.arange(shape[2]),indexing='ij')
            native_zyx=np.stack([origin[i]+v*scale[i] for i,v in enumerate((zz,yy,xx))],axis=-1)
            points=native_zyx[...,::-1]; valid=field.valid(points)
            u=field.sample(points); u[~valid]=0
            samples=(points+u/sp)[...,::-1]
            coords=np.moveaxis(samples,-1,0)
            w=map_coordinates(rawmov,coords,order=1,mode='constant',cval=0,prefilter=False,output=np.float32)-1024
            lab=map_coordinates(masks['moving'],coords,order=0,mode='constant',cval=0,prefilter=False)
            mag=np.linalg.norm(u,axis=-1)
            warped[z:z+12]=w; warpedmask[z:z+12]=np.where(valid,lab,0)
            maps[z:z+12]=np.stack([u[...,0],-u[...,2],mag,valid.astype(np.float32)],axis=-1)
            valid_count+=int(valid.sum()); umax=max(umax,float(mag.max()))
            if valid.any():diffmax=max(diffmax,float(np.abs(w-np.asarray(moving[z:z+12]))[valid].max()))
        assets[f'{method}_image']=save_array(out,f'{method}_image',warped)
        assets[f'{method}_mask']=save_array(out,f'{method}_mask',warpedmask)
        assets[f'{method}_maps']=save_array(out,f'{method}_maps',maps,channels=4,tolerance=.1)
        variant=r.get('implementation_variant','original_standalone' if method=='corrfield' else 'historical_reference' if r.get('historical') else 'corrected_w0.055')
        methods.append(dict(id=method,name=name,variant=variant,historical=bool(r.get('historical')),
                            source_geometry=r.get('geometry',dict(native_crop_bounds_xyz=cb.tolist())),
                            source_units='raw XYZ mm' if method=='ptvreg' else 'ZYX nominal1mm displacement',
                            field_sha256=r['field_sha256'],valid_voxels=valid_count))
        validation[method]=dict(tre_max_error_mm=float(np.max(np.abs(tre-expected))),
                                inverse_counts={s:sum(v['status']==s for v in inverse) for s in ('resolved','ambiguous','unresolved')},
                                mean_native_snap_mm=float(tre.mean()))
        print(key,method,validation[method],flush=True)
        del data,field,warped,warpedmask,maps
    manifest=dict(schema='dirlab_viewer_v1',id=key,dataset=ds,case=case,geometry=geom,
                  phases=dict(fixed=fixed_phase,moving=moving_phase),assets=assets,methods=methods,
                  landmarks=landmarks,scale_max_mm=float(np.ceil(umax)),difference_max_hu=float(np.ceil(diffmax)),
                  lung_counts=dict(right=int((side==1).sum()),left=int((side==2).sum())),
                  protocol_note='Common raw-HU display. MR/pTV optimized pTV intensities; CorrField used HU/fixed R231. Native snap TRE is historical/audited; inverse markers are visualization only.',
                  crop_convention='corrected',validation=validation,
                  source_hashes={str(p):sha(p) for p in dict.fromkeys(source_files)},
                  preparation_seconds=time.monotonic()-started)
    write(out/'manifest.json',manifest)
    write(out/'accepted.json',dict(status='accepted',landmarks=300,manifest_sha256=sha(out/'manifest.json')))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-review',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--mask-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--cases',default='all',help='all or comma-separated 4dct:01,copd:10')
    p.add_argument('--resume',action='store_true',help='Verify/reuse accepted bundles; never replace cases')
    a=p.parse_args(); a.output_root.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[3]
    if a.output_root.resolve().is_relative_to(repo):raise ValueError('Generated assets must be external')
    assert read(a.campaign_review/'campaign_accepted.json')['status']=='accepted'
    report=read(a.campaign_review/'reports/results.json'); assert report['status']=='complete'
    records=report['evaluations']
    selected=[(ds,c) for ds in ['4dct','copd'] for c in range(1,11)] if a.cases=='all' else [(s.split(':')[0],int(s.split(':')[1])) for s in a.cases.split(',')]
    sources={str(p.relative_to(repo)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    provenance=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources,
                    campaign_receipt_sha256=sha(a.campaign_review/'campaign_accepted.json'),argv=sys.argv,
                    numpy=np.__version__,python=sys.version)
    write(a.output_root/f'preparation_{int(time.time())}.json',provenance)
    for ds,c in selected:
        if ds not in ('4dct','copd') or c not in range(1,11):raise ValueError('Unknown case')
        prepare_case(a,ds,c,records)
        entries=[]
        for receipt in sorted((a.output_root/'cases').glob('*/accepted.json')):
            m=read(receipt.parent/'manifest.json')
            entries.append(dict(id=m['id'],dataset=m['dataset'],case=m['case'],url=f"/assets/cases/{m['id']}/manifest.json"))
        write(a.output_root/'catalog.json',dict(schema='dirlab_viewer_v1',cases=entries))
    print('PREPARATION COMPLETE',len(entries),'accepted bundles',flush=True)


if __name__=='__main__':main()
