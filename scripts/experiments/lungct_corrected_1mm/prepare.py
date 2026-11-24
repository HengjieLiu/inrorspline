"""Prepare versioned products and verify every reused 4DCT input exactly."""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from core import REPO,case_data,phases,read_json,sha,write_json


def validate(data_root,old_root):
    records=[]
    for ds in ['4dct','copd']:
        fx,mv=phases(ds)
        for case in range(1,11):
            d=case_data(data_root,ds,case);a=read_json(d/'preprocess_audit.json')
            assert a['crop_convention']=='corrected' and a['verification']['pass']
            origin=np.asarray(a['crop_bounds_zyx_inclusive'])[:,0]
            shape=np.asarray(a['crop_shape_zyx']);reg=np.asarray(a['registration_shape_zyx'])
            parity=[];hashes={}
            for p in sorted(d.glob('*.npy')):
                arr=np.load(p,mmap_mode='r')
                for z in range(0,len(arr),16):assert np.isfinite(arr[z:z+16]).all(),p
                hashes[p.name]=sha(p)
                if ds=='4dct' and 'reg1mm_hu' not in p.name:
                    old=case_data(old_root,ds,case)/p.name
                    assert np.array_equal(arr,np.load(old,mmap_mode='r')),p
                    parity.append(p.name)
            for role,phase in [('fixed',fx),('moving',mv)]:
                native=np.load(d/f'landmarks_{role}_{phase}_native0_zyx.npy')
                registered=np.load(d/f'landmarks_{role}_{phase}_reg1mm_zyx.npy')
                np.testing.assert_allclose(registered*(shape-1)/(reg-1)+origin,native,rtol=0,atol=1e-10)
                assert native.shape==(300,3)
                hu=np.load(d/f'{role}_{phase}_reg1mm_hu_zyx.npy',mmap_mode='r')
                assert tuple(hu.shape)==tuple(reg)
                if case in [1,2]:
                    p=next(v for k,v in a['image_paths'].items() if k.startswith(role))
                    raw=np.memmap(p,mode='r',dtype=np.int16,shape=tuple(a['raw_shape_zyx']))
                    bounds=a['crop_bounds_zyx_inclusive'];sl=tuple(slice(lo,hi+1) for lo,hi in bounds)
                    inp=np.asarray(raw[sl],dtype=np.float32)-1024.
                    expected=F.interpolate(torch.from_numpy(inp)[None,None],size=tuple(reg),mode='trilinear',align_corners=True)[0,0].numpy()
                    assert np.array_equal(hu,expected)
                    mask_native=np.load(d/f'{role}_{phase}_lungmask_native_zyx.npy',mmap_mode='r')
                    expected_mask=F.interpolate(torch.from_numpy(mask_native[sl].copy().astype(np.float32))[None,None],size=tuple(reg),mode='nearest')[0,0].numpy()
                    assert np.array_equal(expected_mask,np.load(d/f'{role}_{phase}_lungmask_reg1mm_nearest_zyx.npy'))
            records.append(dict(dataset=ds,case=case,array_hashes=hashes,exact_historical_arrays=parity,
                                registration_shape_zyx=reg.tolist(),landmarks_per_phase=300))
            print('validated',ds,case,reg.tolist(),flush=True)
    assert records[10]['registration_shape_zyx'][0]==272
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--old-root',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    for ds in ['4dct','copd']:
        script=REPO/f'scripts/data/preprocessing/lungct/generate_{ds}_ptvprep.py'
        subprocess.run([sys.executable,str(script),'--data-root',str(a.raw_root),
                        '--mask-root',str(a.raw_root/'processed/lungmask_R231_native'),'--out-root',str(a.out_root),
                        '--crop-convention','corrected','--save-hu','--device','cpu'],check=True)
    write_json(a.receipt,dict(status='accepted',cases=validate(a.out_root,a.old_root)))


if __name__=='__main__':main()
