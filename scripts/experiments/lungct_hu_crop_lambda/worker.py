"""One fresh MR optimization using the frozen, unchanged numerical solver."""
from __future__ import annotations
import argparse
import gc
import hashlib
import math
import os
from pathlib import Path
import sys
import time
import traceback
from .protocol import dump,read,sha,weight,stamp

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    if hasattr(value,'item'):return clean(value.item())
    return value

def jacobian_stats(field, mask):
    """Bounded-memory centered-difference diagnostics; explicitly tagged stencil."""
    import numpy as np
    total={k:dict(n=0,fold=0,js=0.,logs=0.,logs2=0.,positive=0) for k in ('full','lung')}
    for start in range(1,field.shape[0]-1,12):
        stop=min(start+12,field.shape[0]-1)
        block=np.asarray(field[start-1:stop+1],dtype=np.float64)
        gradients=[np.gradient(block[...,j],axis=(0,1,2),edge_order=1) for j in range(3)]
        entries=[[gradients[j][i][1:-1,1:-1,1:-1]+(1. if i==j else 0.) for i in range(3)] for j in range(3)]
        a,b,c=entries[0];d,e,f=entries[1];g,h,i=entries[2]
        det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
        selections={'full':det.reshape(-1),'lung':det[np.asarray(mask[start:stop,1:-1,1:-1])>0]}
        for key,v in selections.items():
            t=total[key];t['n']+=v.size;t['fold']+=int(np.sum(v<=0));t['js']+=float(v.sum())
            logs=np.log(v[v>0]);t['positive']+=logs.size;t['logs']+=float(logs.sum());t['logs2']+=float(np.square(logs).sum())
    out={'stencil':'centered differences, one-voxel outer exclusion, nominal1mm units'}
    for key,t in total.items():
        n=t['n'];p=t['positive']
        out[key]=dict(voxels=n,folding_pct=100*t['fold']/n if n else None,
                      jdet_mean=t['js']/n if n else None,
                      std_log_jdet_positive=math.sqrt(max(0,t['logs2']/p-(t['logs']/p)**2)) if p else None)
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--dataset',choices=['4dct','copd'],required=True);p.add_argument('--case',type=int,required=True)
    p.add_argument('--hu',required=True);p.add_argument('--crop',required=True);p.add_argument('--k',type=int,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--save-profile-field',action='store_true');a=p.parse_args()
    # The controller creates the directory to attach immutable launch/log records.
    if (a.out/'outcome.json').exists():raise FileExistsError(a.out/'outcome.json')
    a.out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
        import numpy as np
        import torch
        torch.set_num_threads(2)
        repo=Path(__file__).resolve().parents[3]
        frozen=repo/'source_manifest.json'
        if frozen.exists():
            for name,digest in read(frozen)['source_hashes'].items():
                if sha(repo/name)!=digest:raise RuntimeError(f'Frozen source drift: {name}')
        sys.path.insert(0,str(repo/'scripts/experiments/lungct_no_sweep_repro'))
        import run_one as solver
        previous=sys.argv
        sys.argv=['run_one.py','--method',solver.METHOD_DIRECT_CP_32_PTV,'--dataset',a.dataset,
          '--case-id',str(a.case),'--data-root',str(a.data_root),'--gpu',a.gpu,'--lr','0.001',
          '--seed','1','--stage-full-cps','64,32,16,8,4','--stage-downsample','16,8,4,2,1',
          '--stage-epochs','2,2,2,2,2' if a.smoke else '400,400,400,400,900',
          '--ptv-isotv-weight',repr(weight(a.k)),'--ptv-lcc-sigma-mm','2.1','--ptv-csqrt','0.005',
          '--border-mask-voxels','5','--no-cudnn-benchmark','--cudnn-deterministic']
        dump(a.out/'solver_arguments.json',sys.argv)
        args=solver.parse_args();sys.argv=previous
        device=solver.c.set_cuda_device(a.gpu,cudnn_benchmark=False,cudnn_deterministic=True)
        data_dir=solver.c.case_dir(a.data_root,a.case,a.dataset)
        prepared=read(data_dir/'accepted.json')
        for name,digest in prepared['input_hashes'].items():
            if sha(data_dir/name)!=digest:raise ValueError(f'Input drift {data_dir/name}')
        case=solver.c.load_case(a.data_root,a.case,a.dataset)
        backend=dict(requested_cudnn_benchmark=False,requested_cudnn_deterministic=True,
            effective_cudnn_benchmark=torch.backends.cudnn.benchmark,
            effective_cudnn_deterministic=torch.backends.cudnn.deterministic,
            torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),physical_gpu=a.gpu,
            device_name=torch.cuda.get_device_name(0),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32,cpu_threads=torch.get_num_threads(),
            reproducibility_scope='cuDNN settings do not guarantee bitwise determinism of all CUDA operations; original-CLI repeats also differ')
        dump(a.out/'start.json',dict(started=stamp(),dataset=a.dataset,case=a.case,hu=a.hu,crop=a.crop,
             k=a.k,weight=weight(a.k),smoke=a.smoke,backend=backend,prepared_receipt_sha256=sha(data_dir/'accepted.json')))
        torch.cuda.reset_peak_memory_stats()
        result=solver.train_multistage(args,case,device,a.out,sampler=None)
        if not a.smoke and sum(s['actual_epochs'] for s in result['stage_summaries'])!=2500:
            raise RuntimeError('Production run did not complete the required 2500 iterations')
        torch.cuda.synchronize()
        peak_reserved=torch.cuda.max_memory_reserved()/1024**3
        peak_allocated=torch.cuda.max_memory_allocated()/1024**3
        dense=result.pop('dense_disp_norm_xyz')
        field=solver.c.dense_norm_xyz_to_vox_zyx(dense).astype(np.float32,copy=False)
        for z in range(0,len(field),16):
            if not np.isfinite(field[z:z+16]).all():raise ValueError('Nonfinite output field')
        dump(a.out/'training.json',clean(dict(result,backend=backend,peak_reserved_gib=peak_reserved,
             peak_allocated_gib=peak_allocated,optimizer_end_utc=stamp())))
        del dense;gc.collect();torch.cuda.empty_cache()
        if a.smoke:
            if a.save_profile_field:np.save(a.out/'profile_field_zyx.npy',field,allow_pickle=False)
            outcome=dict(status='profiled',smoke=True,score=None,peak_reserved_gib=peak_reserved,
                peak_allocated_gib=peak_allocated,voxels=int(np.prod(field.shape[:3])),seconds=time.time()-started,
                field_bytes_sha256=hashlib.sha256(np.ascontiguousarray(field).data).hexdigest())
        else:
            path=a.out/'disp_vox_reg1mm_zyx.npy';np.save(path,field,allow_pickle=False)
            from scripts.experiments.lungct_corrected_1mm.core import evaluate
            identity=dict(method='mrdbscp',hu=a.hu,crop=a.crop,k=a.k,weight=weight(a.k),historical=False,
                          backend=backend,peak_reserved_gib=peak_reserved,training=result['training'])
            metrics=evaluate(path,data_dir,a.out,identity)
            dump(a.out/'deformation_metrics.json',jacobian_stats(field,case.fixed_mask))
            outcome=dict(status='accepted',score=metrics['original_snap'],score_sd=metrics['original_snap_sd'],
                peak_reserved_gib=peak_reserved,peak_allocated_gib=peak_allocated,
                voxels=int(np.prod(field.shape[:3])),seconds=time.time()-started,
                evaluation_path=str(a.out/'evaluation.json'),evaluation_sha256=sha(a.out/'evaluation.json'),
                prepared_receipt_sha256=sha(data_dir/'accepted.json'))
        dump(a.out/'outcome.json',dict(outcome,finished=stamp()))
        print('WORKER',a.dataset,a.case,a.hu,a.crop,a.k,outcome['status'],outcome['score'],flush=True)
    except Exception as e:
        outcome=dict(status='failed',score=None,error_type=type(e).__name__,error=str(e),
            traceback=traceback.format_exc(),seconds=time.time()-started,finished=stamp())
        try:
            import torch
            if torch.cuda.is_initialized():outcome['peak_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
        except Exception:pass
        dump(a.out/'outcome.json',outcome)
        traceback.print_exc();raise SystemExit(1)

if __name__=='__main__':main()
