"""One isolated GPU case; the numerical MR runner and CorrField stay intact."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['mrdbscp','corrfield'],required=True)
    p.add_argument('--dataset',choices=['4dct','copd'],required=True);p.add_argument('--case',type=int,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--corrfield-source',type=Path,default=Path('/opt/corrfield-source'));a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    import numpy as np
    import torch
    from core import (ARCHIVE_SHA,CORRFIELD,LEGACY_DIR,case_data,corrfield_to_zyx,evaluate,
                      phases,read_json,sha,write_json)
    torch.set_num_threads(2)
    if a.out_dir.exists():raise FileExistsError(a.out_dir)
    a.out_dir.mkdir(parents=True)
    ds=case_data(a.data_root,a.dataset,a.case);fx,mv=phases(a.dataset)
    audit=read_json(ds/'preprocess_audit.json')
    assert audit['crop_convention']=='corrected'
    torch.manual_seed(1);np.random.seed(1)
    bench=a.method=='mrdbscp';det=not bench
    torch.backends.cudnn.benchmark=bench;torch.backends.cudnn.deterministic=det
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    backend=dict(requested_cudnn_benchmark=bench,requested_cudnn_deterministic=det,
                 effective_cudnn_benchmark=torch.backends.cudnn.benchmark,
                 effective_cudnn_deterministic=torch.backends.cudnn.deterministic,
                 torch_version=torch.__version__,cuda_version=torch.version.cuda,
                 cudnn_version=torch.backends.cudnn.version(),device_name=torch.cuda.get_device_name(0),
                 physical_gpu=a.gpu,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                 tf32_cudnn=torch.backends.cudnn.allow_tf32)
    identity=dict(method=a.method,backend=backend,smoke=a.smoke,seed=1,
                  intensity_profile='ptv_clip80_900_scale01' if bench else 'raw_minus1024_unclipped_HU',
                  registration_mask='border5_only' if bench else 'fixed_R231_lung_only')
    inputs=[ds/'preprocess_audit.json',ds/f'fixed_{fx}_reg1mm_{"trilinear" if bench else "hu"}_zyx.npy',
            ds/f'moving_{mv}_reg1mm_{"trilinear" if bench else "hu"}_zyx.npy']
    if not bench:inputs.append(ds/f'fixed_{fx}_lungmask_reg1mm_nearest_zyx.npy')
    input_hashes={str(p):sha(p) for p in inputs}
    write_json(a.out_dir/'start.json',dict(**identity,inputs=input_hashes))
    started=time.time();field_path=a.out_dir/'disp_vox_reg1mm_zyx.npy'
    if bench:
        if a.dataset!='copd':raise ValueError('Only COPD MR reruns authorized')
        cmd=[sys.executable,str(LEGACY_DIR/'run_one.py'),'--dataset','copd',
             '--method','direct_cp_32_16_8_4_ptv_lcc_isotv','--case-id',str(a.case),
             '--data-root',str(a.data_root),'--out-root',str(a.out_dir/'solver'),
             '--run-name','registration','--gpu',a.gpu,'--seed','1','--lr','0.001',
             '--stage-full-cps','64,32,16,8,4','--stage-downsample','16,8,4,2,1',
             '--stage-epochs','1,1,1,1,1' if a.smoke else '400,400,400,400,900',
             '--ptv-lcc-sigma-mm','2.1','--ptv-isotv-weight','0.055','--ptv-csqrt','0.005',
             '--border-mask-voxels','5','--cudnn-benchmark','--no-cudnn-deterministic','--save-dense-dvf']
        write_json(a.out_dir/'command.json',cmd)
        subprocess.run(cmd,check=True)
        found=list((a.out_dir/'solver').rglob('summary.json'))
        if len(found)!=1:raise ValueError('Ambiguous MR output')
        solver=read_json(found[0]);assert solver['status']=='completed'
        identity['solver_summary_path']=str(found[0]);identity['training']=solver['training']
        identity['peak_allocated_gb']=solver['peak_mem_gb']
        identity['backend']=solver['backend']
        # One canonical entry points at the immutable solver field, no extra GB copy.
        field_path.symlink_to((found[0].parent/'disp_vox_reg1mm_zyx.npy').relative_to(a.out_dir))
    else:
        assert sha(a.corrfield_source/'corrfield.zip')==ARCHIVE_SHA
        sys.path.insert(0,str(a.corrfield_source))
        from corrfield.corrfield import corrfield
        def tensor(path):
            arr=np.load(path)
            return torch.from_numpy(np.ascontiguousarray(arr.transpose(2,1,0),dtype=np.float32))[None,None].cuda()
        fixed=tensor(inputs[1]);moving=tensor(inputs[2]);mask=tensor(inputs[3])
        assert torch.count_nonzero(mask)>0
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.time()
        with torch.no_grad():
            output,kf,km=corrfield(fixed,mask,moving,**CORRFIELD)
        torch.cuda.synchronize()
        identity['optimizer_seconds']=time.time()-t
        identity['peak_allocated_gb']=torch.cuda.max_memory_allocated()/1024**3
        identity['parameters']=CORRFIELD
        identity['corrfield_source_hashes']={str(p.relative_to(a.corrfield_source)):sha(p)
                                           for p in sorted((a.corrfield_source/'corrfield').glob('*.py'))}
        field=corrfield_to_zyx(output[0].cpu().numpy(),audit['registration_shape_zyx'])
        np.save(field_path,field,allow_pickle=False)
        keypoints=np.concatenate([kf[0].cpu().numpy(),km[0].cpu().numpy()],axis=1)
        np.savetxt(a.out_dir/'internal_correspondences_xyz.csv',keypoints,delimiter=',',header='fixed_x,fixed_y,fixed_z,mapped_x,mapped_y,mapped_z',comments='')
        identity['internal_keypoints']=len(keypoints)
        del fixed,moving,mask,output,kf,km,field
        torch.cuda.empty_cache()
    identity['registration_and_export_seconds']=time.time()-started
    result=evaluate(field_path,ds,a.out_dir,identity)
    assert {str(p):sha(p) for p in inputs}==input_hashes
    write_json(a.out_dir/'accepted.json',dict(status='accepted',evaluation_sha256=sha(a.out_dir/'evaluation.json'),
                                           input_hashes=input_hashes,full_case_seconds=time.time()-started))
    print('ACCEPTED',a.method,a.dataset,a.case,result['original_snap'],flush=True)


if __name__=='__main__':main()
