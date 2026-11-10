"""Coverage-aware P82 figures, machine-readable tables and external notebook."""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import REPO,FAMILIES,CAPS,read_json,write_json,now,external

COLORS=dict(zip(FAMILIES,plt.get_cmap('tab10').colors[:7]))
LABELS={f:f.replace('gaussian_','Gaussian ').replace('tree_','Tree ').replace('_',' ') for f in FAMILIES}
LABELS['bspline']='B-spline'

def collect(root):
    manifest=read_json(root/'manifest.json')
    state=read_json(root/'queue_state.json') if (root/'queue_state.json').exists() else {'jobs':manifest['jobs']}
    rows=[]
    for job in state['jobs']:
        row={k:job[k] for k in ('id','target','family','cap','version')}
        row.update(status=job.get('status','pending'),actual_params=None,full_rmse_mm=None,foreground_rmse_mm=None,
                   background_rmse_mm=None,fit_active_seconds=None,converged=False,budget_reached=False)
        if job.get('result'):
            r=read_json(job['result']);row.update({k:r[k] for k in ('status','actual_params','fit_active_seconds','converged','budget_reached')})
            row.update(r['metrics']);row.update(result_path=job['result'],field_path=r['field_path'],
                stages=r['stages'],transfers=r['transfers'],evaluation_seconds=r['evaluation_seconds'],input_loading_seconds=r['input_loading_seconds'])
        if job.get('failure'):row['failure']=job['failure']
        row['attempts']=job.get('attempts',[]);rows.append(row)
    if len(rows)!=168 or len({r['id'] for r in rows})!=168:raise ValueError('Coverage identities are not the requested168')
    return manifest,state,rows

def curve(ax,rows,family,metric,color,style='-',label=None):
    points=sorted([r for r in rows if r['family']==family],key=lambda r:r['cap'])
    previous=None
    for r in points:
        if r[metric] is None or not r['budget_reached']:
            previous=None;continue
        x=r['actual_params'];y=max(r[metric],1e-12)
        if previous is not None:ax.plot([previous[0],x],[previous[1],y],style,color=color,lw=1.3)
        ax.plot(x,y,'o',color=color,ms=5,markerfacecolor=color if r['converged'] else 'white',label=label)
        label=None;previous=(x,y)
    # Legend entry even if no endpoint has completed.
    if label is not None:ax.plot([],[],style,color=color,label=label)

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def panels(rows,out,metric,name,replay):
    fig,axes=plt.subplots(2,2,figsize=(14,10),sharex=True,sharey='row')
    for i,target in enumerate(('rank000_sitreg','rank000_unbounded')):
        values=[max(r[metric],1e-12) for r in rows if r['target']==target and r[metric] is not None and r['budget_reached']]
        ylim=(min(values)/1.5,max(values)*1.5) if values else (1e-3,10)
        for j,version in enumerate(('native','multiresolution')):
            ax=axes[i,j];subset=[r for r in rows if r['target']==target and r['version']==version]
            for f in FAMILIES:curve(ax,subset,f,metric,COLORS[f],label=LABELS[f])
            ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,which='both',alpha=.2)
            ax.axvline(CAPS[-1],color='grey',ls=':',lw=.8);ax.set_xlim(2000,CAPS[-1]*1.4)
            ax.set_ylim(*ylim)
            missing=sum(r[metric] is None or not r['budget_reached'] for r in subset)
            ax.set_title(('SITReg' if i==0 else 'Unbounded B-spline842 λ=1')+' — '+version)
            ax.text(.02,.02,f'{missing}/42 requested points unavailable or budget not reached',transform=ax.transAxes,fontsize=8)
            if i==1:ax.text(.02,.94,f'Original cps2 replay: {replay["full_rmse_mm"]:.2g} mm (control only)',transform=ax.transAxes,fontsize=8)
            ax.set_xlabel('Actual scalar motion parameters');ax.set_ylabel('Full-box vector RMSE (mm)' if metric=='full_rmse_mm' else 'Foreground vector RMSE (mm)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9)
    fig.suptitle('20-minute maximum per independent fit; filled = converged, hollow = unresolved',fontsize=12)
    fig.tight_layout(rect=(0,.08,1,.95));save(fig,out,name)

def paired(rows,out):
    fig,axes=plt.subplots(2,7,figsize=(25,8),sharex=True,sharey='row')
    for i,target in enumerate(('rank000_sitreg','rank000_unbounded')):
        values=[max(r['full_rmse_mm'],1e-12) for r in rows if r['target']==target and r['full_rmse_mm'] is not None and r['budget_reached']]
        ylim=(min(values)/1.5,max(values)*1.5) if values else (1e-3,10)
        for j,f in enumerate(FAMILIES):
            ax=axes[i,j]
            for version,style,color in [('native','-','tab:blue'),('multiresolution','--','tab:orange')]:
                sub=[r for r in rows if r['target']==target and r['version']==version]
                curve(ax,sub,f,'full_rmse_mm',color,style,label=version)
            ax.set_xscale('log');ax.set_yscale('log');ax.grid(True,alpha=.2);ax.set_title(LABELS[f],fontsize=9)
            ax.set_xlim(2000,CAPS[-1]*1.4);ax.set_ylim(*ylim)
            ax.set_xlabel('Parameters');ax.set_ylabel(('SITReg' if i==0 else 'Unbounded')+' RMSE (mm)')
    axes[0,0].legend(fontsize=8);fig.tight_layout();save(fig,out,'paired_versions')

def error_maps(rows,manifest,out):
    for target in ('rank000_sitreg','rank000_unbounded'):
        reference=np.load(manifest['targets'][target]['canonical_path'],mmap_mode='r');z=reference.shape[2]//2
        fig,axes=plt.subplots(2,7,figsize=(24,7));images=[]
        for i,version in enumerate(('native','multiresolution')):
            for j,f in enumerate(FAMILIES):
                ax=axes[i,j];row=next(r for r in rows if r['target']==target and r['version']==version and r['family']==f and r['cap']==CAPS[0])
                ax.axis('off');ax.set_title(f'{LABELS[f]}\n{version}',fontsize=8)
                if not row.get('field_path'):ax.text(.1,.5,row['status'],transform=ax.transAxes);continue
                fitted=np.load(row['field_path'],mmap_mode='r')
                error=np.linalg.norm(fitted[:,:,z].astype(np.float64)-reference[:,:,z],axis=-1)
                im=ax.imshow(error.T,origin='lower',vmin=0,vmax=5,cmap='magma');images.append(im)
                ax.text(.01,.01,f'P={row["actual_params"]}; {row["status"]}',color='white',fontsize=6,transform=ax.transAxes)
        if images:fig.colorbar(images[0],ax=axes.ravel().tolist(),label='Endpoint error (mm)',fraction=.015,pad=.015)
        fig.suptitle(f'{target}: matched native central slice, requested cap3072; display clipped at5mm')
        save(fig,out,f'error_maps_{target}')

def notebook(root,out):
    import nbformat
    from nbclient import NotebookClient
    source=REPO/'experiments/2026-11-05_oasis_rank000_fieldfit_parameter_scaling/review.ipynb'
    nb=nbformat.read(source,as_version=4)
    # Substitute only the external execution copy, preserving output-free source.
    nb.cells[1].source=f"RUN_ROOT = {str(root)!r}\n"+nb.cells[1].source
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(out)}}).execute()
    nbformat.write(nb,out/'review_executed.ipynb')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--final',action='store_true');a=p.parse_args()
    root=external(a.root);out=root/'reports';out.mkdir(exist_ok=True)
    manifest,state,rows=collect(root);replay=read_json(root/'native_cps2_replay.json')
    counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}
    data=dict(at=now(),counts=counts,expected=168,rows=rows,native_cps2_replay=replay,
              execution_complete=all(r['status'] in ('converged','unresolved','budget_not_reached','failed','resource_limited') for r in rows))
    write_json(out/'results.json',data)
    columns=['id','target','family','cap','version','status','actual_params','budget_reached','converged','full_rmse_mm',
             'foreground_rmse_mm','background_rmse_mm','fit_active_seconds','input_loading_seconds','evaluation_seconds']
    with open(out/'results.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    with open(out/'stage_history.csv','w',newline='') as f:
        columns2=['id','stage','factor','params','elapsed_seconds','converged','reason','iterations']
        w=csv.DictWriter(f,fieldnames=columns2);w.writeheader()
        for r in rows:
            for s in r.get('stages',[]):
                w.writerow(dict(id=r['id'],stage=s['index'],factor=s['factor'],params=s.get('params'),elapsed_seconds=s['elapsed_seconds'],
                    **{k:s.get('solver',{}).get(k) for k in ('converged','reason','iterations')}))
    panels(rows,out,'full_rmse_mm','full_box_curves',replay);panels(rows,out,'foreground_rmse_mm','foreground_curves',replay)
    paired(rows,out)
    if a.final:error_maps(rows,manifest,out)
    report=['# P82 native versus multiresolution scaling','',f'Updated {now()}. Coverage: {counts}.',
        '',f'Original cps2 checkpoint replay RMSE: {replay["full_rmse_mm"]:.8g} mm.',
        '', 'All curves use full-box supervised field fitting; no SVF or image registration. Each endpoint has a1200-second active budget.',
        'Filled markers are numerically converged. Hollow markers are unresolved achieved fits. Missing or budget-not-reached points break lines.',
        'Multiresolution stages and Gaussian projection consume the same budget; native fits start independently.',
        'Exact-basis high-parameter trees may not reach their requested budget. These are not valid completed-cap points.',
        '', '![Full-box curves](full_box_curves.png)', '', '![Foreground curves](foreground_curves.png)',
        '', '![Paired versions](paired_versions.png)', '', 'One image pair and two correlated source fields; no population inference.']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    if a.final:notebook(root,out)
    print(counts,flush=True)
if __name__=='__main__':main()
