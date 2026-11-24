"""Replay audited historical fields without running registration or changing them."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import map_coordinates
from core import (case_data,evaluate,historical_path,phases,read_json,sha,write_csv,write_json)


def ptv_result(path,data_dir,out):
    out.mkdir(parents=True,exist_ok=False)
    a=read_json(data_dir/'preprocess_audit.json');fx,mv=phases(a['dataset'])
    with h5py.File(path,'r') as f:
        field=f['Tptv_rsz'].astype('float32')[:]
        if not np.isfinite(field).all():raise ValueError('Non-finite pTV field')
        fixed=np.asarray(f['pts_fix']).T-1;moving=np.asarray(f['pts_mov']).T-1
        spacing=np.asarray(f['spc_orig_saved']).reshape(-1)
        saved=float(f['TRE_phys'][0,0]);saved_std=float(f['TREstd_case'][0,0])
    for role,phase,pts in [('fixed',fx,fixed),('moving',mv,moving)]:
        assert np.array_equal(pts[:,::-1],np.load(data_dir/f'landmarks_{role}_{phase}_native0_zyx.npy'))
    idx=np.rint(fixed).astype(int)
    disp=field[:,idx[:,2],idx[:,1],idx[:,0]].T.astype(np.float64)
    pred0=np.rint(fixed+disp/spacing);pred1=fixed+disp/spacing
    snap=np.linalg.norm((pred0-moving)*spacing,axis=1)
    sub=np.linalg.norm((pred1-moving)*spacing,axis=1)
    assert abs(snap.mean()-saved)<1e-8
    assert abs(snap.std(ddof=1)-saved_std)<1e-8
    rows=[]
    for i in range(300):
        row=dict(landmark_id=i,original_snap=float(snap[i]),original_subvoxel=float(sub[i]))
        for axis,j in zip('xyz',range(3)):
            row.update({f'fixed_{axis}_native':float(fixed[i,j]),f'moving_{axis}_native':float(moving[i,j]),
                        f'predicted_snap_{axis}_native':float(pred0[i,j]),f'predicted_subvoxel_{axis}_native':float(pred1[i,j])})
        rows.append(row)
    write_csv(out/'per_landmark.csv',rows)
    # Diagnostic pTV field sampling on the campaign's corrected nominal1mm grid.
    origin=np.asarray(a['crop_bounds_zyx_inclusive'])[:,0]
    scale=(np.asarray(a['crop_shape_zyx'])-1)/(np.asarray(a['registration_shape_zyx'])-1)
    fr=np.load(data_dir/f'landmarks_fixed_{fx}_reg1mm_zyx.npy')
    mr=np.load(data_dir/f'landmarks_moving_{mv}_reg1mm_zyx.npy')
    values=[]
    for points,snap_output in [(np.rint(fr),True),(fr,False)]:
        coords=(origin+points*scale).T
        u=np.stack([map_coordinates(field[k],coords,order=1,mode='nearest') for k in [2,1,0]],axis=-1)
        pred=fr+u
        if snap_output:pred=np.rint(pred)
        values.append(float(np.linalg.norm(pred-mr,axis=1).mean()))
    r=dict(method='ptvreg',dataset=a['dataset'],case=a['case_id'],status='completed',historical=True,
           landmarks=300,original_snap=float(snap.mean()),original_snap_sd=float(snap.std(ddof=1)),
           original_subvoxel=float(sub.mean()),original_subvoxel_sd=float(sub.std(ddof=1)),
           reg1mm_snap=values[0],reg1mm_subvoxel=values[1],
           diagnostic_grid='campaign_corrected_nominal1mm; resampled native physical field',
           field_path=str(path),field_sha256=sha(path),per_landmark_path=str(out/'per_landmark.csv'),
           saved_tre_mean_difference_mm=float(snap.mean()-saved),
           intensity_profile='historical_pTV',registration_mask='no_lung_mask')
    write_json(out/'evaluation.json',r);return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--old-root',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True)
    a=p.parse_args();old=read_json(a.audit);records=[]
    a.out_root.mkdir(parents=True,exist_ok=False)
    for row in old['exact_poster_table_records']:
        ds=row['dataset'].lower();case=row['case']
        d=case_data(a.data_root,ds,case)
        pp=historical_path(row['ptv_field_path'],a.artifact_root)
        pr=ptv_result(pp,d,a.out_root/f'ptvreg_{ds}_{case:02d}');records.append(pr)
        assert abs(pr['original_snap']-row['ptv_snap_mean_mm'])<1e-8
        md=case_data(a.old_root,ds,case) if ds=='copd' else d
        out=a.out_root/f'mrdbscp_{ds}_{case:02d}';out.mkdir()
        mp=historical_path(row['mr_field_path'],a.artifact_root)
        r=evaluate(mp,md,out,dict(method='mrdbscp_legacy' if ds=='copd' else 'mrdbscp',historical=True,
                                  intensity_profile='ptv_clip80_900_scale01',registration_mask='border5_only'))
        saved=np.asarray([float(v['snap_tre_mm']) for v in __import__('core').load_csv(mp.parent/'per_landmark_tre_native_crop.csv')])
        np.testing.assert_array_equal(saved,np.asarray([float(v['original_snap']) for v in __import__('core').load_csv(out/'per_landmark.csv')]))
        assert abs(r['original_snap']-row['mr_snap_mean_mm'])<1e-8
        records.append(r);print('reference accepted',ds,case,pr['original_snap'],r['original_snap'],flush=True)
    write_json(a.out_root/'accepted.json',dict(status='accepted',evaluations=records))


if __name__=='__main__':main()
