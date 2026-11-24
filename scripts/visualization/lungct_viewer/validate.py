"""Independent sample reconstruction and complete immutable-bundle verification."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import map_coordinates
from .prepare import read, write, sha, rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root',type=Path,required=True)
    p.add_argument('--campaign-review',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--mask-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-incomplete',action='store_true',help='Smoke audit only; never seals a complete cohort')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    catalog=read(a.bundle_root/'catalog.json')['cases']
    cohort={(x['dataset'],x['case']) for x in catalog}
    expected_cohort={(d,c) for d in ['4dct','copd'] for c in range(1,11)}
    assert cohort<=expected_cohort and (a.allow_incomplete or cohort==expected_cohort)
    records=read(a.campaign_review/'reports/results.json')['evaluations']
    checked={};results=[];asset_bytes=0;landmark_methods=0
    for entry in catalog:
        root=a.bundle_root/'cases'/entry['id'];m=read(root/'manifest.json')
        assert sha(root/'manifest.json')==read(root/'accepted.json')['manifest_sha256']
        for name,expected in m['source_hashes'].items():
            if name not in checked:checked[name]=sha(name)
            assert checked[name]==expected,name
        dsdir='4DCT' if m['dataset']=='4dct' else 'COPD'
        prepared=a.data_root/dsdir/f"case{m['case']:02d}"
        audit=read(prepared/'preprocess_audit.json')
        shape=np.array(m['geometry']['shape_zyx']);sp=np.array(audit['native_spacing_zyx_mm'])[::-1]
        rawshape=tuple(audit['raw_shape_zyx']);origin=np.asarray(audit['crop_bounds_zyx_inclusive'])[:,0]
        scale=(np.asarray(audit['crop_shape_zyx'])-1)/(shape-1)
        rng=np.random.default_rng(1);index=rng.integers(0,shape,size=(160,3));index[:2]=[np.zeros(3),shape-1]
        native_zyx=origin+index*scale;native_xyz=native_zyx[:,::-1]
        # Original text labels, never inferred from a rendered position.
        fpath=next(v for k,v in audit['landmark_paths'].items() if k.startswith('fixed'))
        mpath=next(v for k,v in audit['landmark_paths'].items() if k.startswith('moving'))
        fp=np.loadtxt(fpath)-1;mp=np.loadtxt(mpath)-1
        for role,points in [('fixed',fp),('moving',mp)]:
            world=np.asarray([x[f'{role}_lps_mm'] for x in m['landmarks']]);world[:,2]=(rawshape[0]-1)*sp[2]-world[:,2]
            np.testing.assert_allclose(world/sp,points,rtol=0,atol=1e-10)
        samples={}
        for key,asset in m['assets'].items():
            path=root/asset['file'];assert sha(path)==asset['sha256']
            with gzip.open(path,'rb') as f:data=np.frombuffer(f.read(),dtype=asset['dtype']).reshape(asset['shape'])
            assert np.isfinite(data).all() and data.nbytes==asset['bytes']
            assert asset['max_display_error'] <= (.1 if key.endswith('maps') else 2)
            samples[key]=data[tuple(index.T)].astype(np.float64);asset_bytes+=path.stat().st_size
            del data
        imagepaths={role:next(v for k,v in audit['image_paths'].items() if k.startswith(role)) for role in ['fixed','moving']}
        raw={role:np.memmap(v,dtype=np.int16,mode='r',shape=rawshape) for role,v in imagepaths.items()}
        masks={role:np.load(a.mask_root/dsdir/f"case{m['case']:02d}"/f"case{m['case']:02d}_{audit[role+'_phase']}_R231_labels_zyx.npy",mmap_mode='r') for role in ['fixed','moving']}
        for role in ['fixed','moving']:
            v=map_coordinates(raw[role],native_zyx.T,order=1,mode='constant',cval=0,output=np.float32)-1024
            np.testing.assert_allclose(v,samples[f'{role}_image'],rtol=0,atol=2.05)
            lab=map_coordinates(masks[role],native_zyx.T,order=0,mode='constant',cval=0)
            np.testing.assert_array_equal(lab,samples[f'{role}_mask'])
        case_result=dict(id=m['id'],methods={},assets=len(samples),landmarks=300)
        for method in ['mrdbscp','ptvreg','corrfield']:
            r=next(r for r in records if r['method']==method and r['dataset']==m['dataset'] and r['case']==m['case'])
            if method=='ptvreg':
                with h5py.File(r['field_path'],'r') as f:field=f['Tptv_rsz'][:];bounds=np.asarray(f['crop_v_saved']).T-1
                def sample(points):
                    return np.stack([map_coordinates(field[c],points[:,::-1].T,order=1,mode='nearest') for c in range(3)],axis=1).astype(float)
            else:
                field=np.load(r['field_path'],mmap_mode='r');g=r['geometry'];bounds=np.asarray(g['crop_bounds_zyx_inclusive'])[::-1]
                off=np.asarray(g['crop_bounds_zyx_inclusive'])[:,0];factor=(np.asarray(g['registration_shape_zyx'])-1)/(np.asarray(g['crop_shape_zyx'])-1)
                def sample(points):
                    coords=((points[:,::-1]-off)*factor).T
                    return np.stack([map_coordinates(field[...,c],coords,order=1,mode='nearest') for c in [2,1,0]],axis=1).astype(float)
            u=sample(native_xyz);valid=np.all((native_xyz>=bounds[:,0]-1e-6)&(native_xyz<=bounds[:,1]+1e-6),axis=1);u[~valid]=0
            expected_maps=np.column_stack([u[:,0],-u[:,2],np.linalg.norm(u,axis=1),valid])
            np.testing.assert_allclose(expected_maps,samples[f'{method}_maps'],rtol=0,atol=.10001)
            coords=(native_xyz+u/sp)[:,::-1].T
            w=map_coordinates(raw['moving'],coords,order=1,mode='constant',cval=0,output=np.float32)-1024
            np.testing.assert_allclose(w,samples[f'{method}_image'],rtol=0,atol=2.0001)
            lab=map_coordinates(masks['moving'],coords,order=0,mode='constant',cval=0);lab[~valid]=0
            np.testing.assert_array_equal(lab,samples[f'{method}_mask'])
            saved=rows(r['per_landmark_path']);assert len(saved)==300
            np.testing.assert_array_equal([x['tre_mm'][method] for x in m['landmarks']],[float(x['original_snap']) for x in saved])
            resolved=[i for i,x in enumerate(m['landmarks']) if x['methods'][method]['status']!='unresolved']
            world=np.asarray([m['landmarks'][i]['methods'][method]['point_lps_mm'] for i in resolved])
            world[:,2]=(rawshape[0]-1)*sp[2]-world[:,2];q=world/sp
            err=np.linalg.norm((q-mp[resolved])*sp+sample(q),axis=1)
            assert np.max(err,initial=0)<=.050001
            case_result['methods'][method]=dict(resolved=len(resolved),inverse_max_residual_mm=float(np.max(err,initial=0)),native_tre_exact=True)
            landmark_methods+=300
            del field
        results.append(case_result);print('validated',m['id'],flush=True)
    repo=Path(__file__).resolve().parents[3]
    source={str(p.relative_to(repo)):sha(p) for p in Path(__file__).parent.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(a.output,dict(status='accepted' if cohort==expected_cohort else 'accepted_partial_smoke',cases=len(results),landmark_method_records=landmark_methods,
                        compressed_asset_bytes=asset_bytes,original_source_files_unchanged=len(checked),
                        source_hashes=source,records=results))
    print('ALL20 BUNDLES ACCEPTED' if cohort==expected_cohort else f'PARTIAL SMOKE ACCEPTED {len(results)}',flush=True)


if __name__=='__main__':main()
