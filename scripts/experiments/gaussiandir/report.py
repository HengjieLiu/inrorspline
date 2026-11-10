"""Collect all four P83 endpoints and the identical-evaluator reference rows."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import atomic_json, external, false_to_mm, load_inputs, sha256


LABELS = {"identity":"Identity", "sitreg":"SITReg (saved lambda 1)",
          "released_control_supervised":"Released control — field fit",
          "paper_reconstruction_supervised":"Paper reconstruction — field fit",
          "released_control_unsupervised":"Released control — image registration",
          "paper_reconstruction_unsupervised":"Paper reconstruction — image registration"}


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item,name))
        elif isinstance(item,(str,int,float,bool)) or item is None:
            result[name] = item
    return result


def write_csv(path, rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build(config_path, run_root, reference_root, out):
    out=external(out)
    out.mkdir(parents=True,exist_ok=False)
    config=json.loads(config_path.read_text())
    inputs=load_inputs(config)
    results, summaries, directories, receipts = {}, {}, {}, {}
    for key in LABELS:
        directory=reference_root/key if key in ("identity","sitreg") else run_root/key/"evaluation"
        directories[key]=directory
        results[key]=json.loads((directory/"metrics.json").read_text())
        receipts[key]=dict(metrics_sha256=sha256(directory/"metrics.json"))
        if key not in ("identity","sitreg"):
            summary=json.loads((run_root/key/"summary.json").read_text())
            manifest=json.loads((run_root/key/"manifest.json").read_text())
            if summary["smoke"] or manifest["config"] != config:
                raise ValueError(f"Wrong frozen configuration for {key}")
            digest=sha256(run_root/key/"field_false_xyz.npy")
            if digest!=summary["field_sha256"] or digest!=results[key]["field_receipt"]["sha256"]:
                raise ValueError(f"Exported field changed after evaluation: {key}")
            receipts[key].update(field_sha256=digest,manifest_sha256=sha256(run_root/key/"manifest.json"))
            summaries[key]=summary
    (out/"report_source.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(out/"report_receipt.json",dict(inputs=receipts,config_sha256=sha256(config_path),
                report_source_sha256=sha256(__file__),
                note="Posthoc report/plot source captured separately from frozen training snapshots"))
    rows=[dict(method=key,**flatten({k:v for k,v in result.items() if k not in
          ("input_receipts","field_receipt","conventions")}),**flatten(summaries.get(key,{}),"training"))
          for key,result in results.items()]
    write_csv(out/"all_metrics.csv",rows)
    write_csv(out/"per_label.csv",[dict(method=key,**flatten(row)) for key,result in results.items()
                                 for row in result["per_label"]])
    atomic_json(out/"comparison.json",dict(results=results,training=summaries,
                run_root=str(run_root),reference_root=str(reference_root)))
    lines=["# GaussianDIR rank-0 pilot", "", "Fixed 0333 / moving 0115; seed-3 shuffled rank 0. "
           "All rows use the same full-box field evaluator and raw fixed-image-positive foreground.", "",
           "| Method | Labels | Dice ↑ | HD95 mm ↓ | ASD mm ↓ | FG folds % ↓ | FG SDlogJ ↓ | Full-box field RMSE mm ↓ |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key,r in results.items():
        a,g,e=r["accuracy"],r["regularity"],r["field_error"]["full"]
        lines.append(f"| {LABELS[key]} | {r['eligible_labels']}/35 | {a['dice_mean']:.6f} | {a['hd95_mean']:.4f} | "
                     f"{a['asd_symmetric_mean']:.4f} | {g['jdet_le0_pct_fg']:.6g} | "
                     f"{g['log_jdet_std_fg']:.6f} | {e['rmse_mm']:.6f} |")
    lines += ["", "Field RMSE measures agreement with SITReg, not anatomical accuracy. "
              "No labels or SITReg field enter the image-registration objective.", "",
              "| Candidate | Status | Steps | Active seconds | Nodes | Stored parameters | Peak GPU MiB |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for key,s in summaries.items():
        lines.append(f"| {LABELS[key]} | {s['outcome']} | {s['total_steps']}/{s['expected_steps']} | "
                     f"{s['active_seconds']:.1f} | {s['final_counts']['nodes']} | "
                     f"{s['final_counts']['parameters']} | {s['training_memory']['peak_allocated_mib']:.1f} |")
    lines += ["", "The corrected model uses anisotropy, local rigid motion and a frozen coarse field plus "
              "a fine residual. Its two-scale schedule is an explicit reconstruction of underspecified paper "
              "details. Its signed Pearson NCC, direction-summed TV and position-gradient population control "
              "also differ from the released control; this comparison does not isolate representation alone.", "",
              "Supervised fitting uses fixed populations and no regularizer. Each endpoint starts from identity, "
              "uses 2,000 scheduled steps and a 1,200-second active watchdog. Final export and evaluation are "
              "timed separately. Active time includes initialization, optimization, adaptation and checkpointing.", "",
              "This repeatedly explored single case and one optimization seed support an exploratory pilot only. "
              "The achieved field fit is not a certified capacity optimum. SITReg is a saved historical learned "
              "endpoint; no training or inference speed comparison is made.", "",
              "[All scalar metrics](all_metrics.csv) include full/foreground image errors, diffusion, BE, "
              "NDV, Jacobian statistics and field/Jacobian fidelity. [Per-label metrics](per_label.csv) include "
              "all 35 labels, counts and explicit exclusions. Analytic B-spline/control-grid metrics are "
              "inapplicable and omitted; the reported BE/diffusion metrics use the common dense-field stencils.", "",
              "On this case, the corrected image registration improves Dice, HD95, ASD and foreground "
              "folding over the released control, but remains below SITReg's accuracy and folding performance. "
              "Its lower positive-J SDlogJ than SITReg does not compensate for its much larger folded fraction. "
              "Full-box behavior also matters: the corrected model has higher whole-box NDV fraction and BE "
              "than the released control despite better foreground metrics. Both fitted fields retain higher "
              "folding and lower Dice than their SITReg target. A lower displacement MSE alone does not "
              "guarantee preservation of derivatives or anatomical overlap.", "",
              "![Accuracy and regularity](accuracy_regularity.png)", "",
              "![Optimization traces](convergence.png)", "", "![Fields and warps](slices.png)", ""]
    (out/"report.md").write_text("\n".join(lines))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for key in summaries:
        checkpoint=torch.load(run_root/key/"checkpoint.pt",map_location="cpu",weights_only=False)
        history=checkpoint["history"]
        ax=axes[0 if key.endswith("_supervised") else 1]
        ax.plot([h["total_steps"] for h in history],[h["loss"] for h in history],label=LABELS[key].split(" — ")[0])
    for ax,title in zip(axes,("Supervised sampled vector MSE (mm²)","Image-registration objective (different normalizations)")):
        ax.set(xlabel="Optimization step",ylabel="Loss",title=title); ax.legend(); ax.grid(alpha=.2)
    axes[0].set_yscale("log")
    fig.savefig(out/"convergence.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    colors=("gray","black","tab:blue","tab:orange","tab:blue","tab:orange")
    markers=("x","*","s","s","o","o")
    for (key,r),color,marker in zip(results.items(),colors,markers):
        for ax,metric in zip(axes,("log_jdet_std_fg","jdet_le0_pct_fg")):
            ax.scatter(r["regularity"][metric],r["accuracy"]["dice_mean"],
                       c=color,marker=marker,s=95,label=LABELS[key])
    for ax,title in zip(axes,("Foreground SD(log J), positive J only","Foreground J ≤ 0 (%)")):
        ax.set(xlabel=title,ylabel="Mean Dice (35 eligible labels)")
        ax.grid(alpha=.2)
    axes[1].set_xscale("symlog",linthresh=1e-4)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=2,fontsize=9)
    fig.subplots_adjust(bottom=.28,wspace=.3)
    fig.savefig(out/"accuracy_regularity.png",dpi=170); plt.close(fig)
    from scripts.experiments.oasis_alpha_sweep import common
    fig,axes=plt.subplots(6,4,figsize=(13,17),layout="constrained")
    middle=inputs["mask"].shape[1]//2
    for row,key in enumerate(LABELS):
        warped=np.load(directories[key]/"warped_image.npy")
        if key=="identity":
            mm=np.zeros_like(inputs["field_mm"])
        elif key=="sitreg":
            mm=inputs["field_mm"]
        else:
            field=np.load(run_root/key/"field_false_xyz.npy")
            mm=false_to_mm(torch.from_numpy(field),inputs["mask"].shape,inputs["affine"]).numpy()
        error=np.linalg.norm(mm-inputs["field_mm"],axis=-1)
        voxel=np.moveaxis(mm,-1,0)[None]  # This verified pair has an identity 1mm linear affine.
        det=common.jacobian_determinant_zyx(voxel.astype(np.float64))
        images=[inputs["fixed_image"][:,middle,:],warped[:,middle,:],error[:,middle,:],det[:,middle-2,:]]
        for col,(ax,pic) in enumerate(zip(axes[row],images)):
            im=ax.imshow(pic.T,origin="lower",cmap="gray" if col<2 else ("magma" if col==2 else "coolwarm"),
                         **({"vmin":0,"vmax":5} if col==2 else {"vmin":0,"vmax":2} if col==3 else {}))
            ax.set_xticks([]); ax.set_yticks([])
            if col>=2: fig.colorbar(im,ax=ax,shrink=.65)
            if row==0: ax.set_title(("Fixed","Warped moving","EPE to SITReg (mm)","Jacobian determinant (crop 2)")[col])
        axes[row,0].set_ylabel(LABELS[key],fontsize=9)
    fig.savefig(out/"slices.png",dpi=140); plt.close(fig)
    print(out/"report.md",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--reference-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    build(args.config,args.run_root,args.reference_root,args.out)
