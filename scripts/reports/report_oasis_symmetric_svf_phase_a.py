#!/usr/bin/env python
"""Accept only four independently replayed Phase-A symmetric-SVF fits."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
import run_oasis_symmetric_svf as r

LABELS = {"A1": "Two total SVFs; effective lambda 1 (coefficient 2)",
          "A2": "Two total SVFs; coefficient 1 (equal-split effective lambda 0.5)",
          "A3": "Tied +/-w/2 midpoint; R(w)",
          "A1_swap": "A1 swapped inputs; reverse export in original direction"}


def difference(a, b, mask=None):
    if a.shape != (160,224,192,3) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("Invalid finite native field")
    scale = np.array([191/2,223/2,159/2],dtype=np.float64)
    delta = (a.astype(np.float64)-b.astype(np.float64))*scale
    if mask is not None:
        if mask.shape!=a.shape[:3] or not mask.any():raise RuntimeError("Invalid difference support")
        delta=delta[mask]
    epe = np.linalg.norm(delta,axis=-1)
    return {"component_rms_mm": float(np.sqrt(np.mean(delta**2))),
            "max_abs_component_mm": float(np.abs(delta).max()), "mean_endpoint_mm": float(epe.mean()),
            "p95_endpoint_mm": float(np.quantile(epe,.95)), "max_endpoint_mm": float(epe.max())}


def validate_rows(rows):
    if len(rows)!=4 or {x["spec"]["arm"] for x in rows}!={"A1","A2","A3","A1_swap"}:
        raise RuntimeError("Require all four unique Phase-A fits")
    for row in rows:
        pair=("0115","0333") if row["spec"]["arm"]=="A1_swap" else ("0333","0115")
        if (row["fixed_id"],row["moving_id"])!=pair or row["spec"]["smoke"]:
            raise RuntimeError("Wrong production pair")
        if [x["integration_steps"] for x in row["evaluations"]]!=[7,8,10]:
            raise RuntimeError("Incomplete integration diagnostics")
        for ev in row["evaluations"]:
            for direction in ("forward","reverse"):
                if not all(math.isfinite(ev[direction]["metrics"][k]) for k in r.METRICS):
                    raise RuntimeError("Nonfinite canonical metric")


def main(args):
    root=args.artifact_root/"runs"/r.m.VARIANT/"phase_a"
    out=args.artifact_root/"repo_experiments"/r.m.VARIANT/"phase_a"
    manifest=r.c.read_json(root/"launch_manifest.json")
    r.assert_manifest(manifest)
    gate=r.c.read_json(root/"PRODUCTION_FINISHED.json")
    if not gate["passed"] or gate["completed"]!=4 or gate["source_hashes"]!=r.source_hashes():
        raise RuntimeError("Production incomplete")
    rows=[r.validate(Path(j["path"]),manifest,r.m.Spec(**j["spec"])) for j in gate["runs"]]
    rows.sort(key=lambda x:list(LABELS).index(x["spec"]["arm"]))
    validate_rows(rows)
    cases,branches,stage_rows,paired,label_rows,integrations,support=[],[],[],[],[],[],[]
    for row in rows:
        arm=row["spec"]["arm"]
        direction="reverse" if arm=="A1_swap" else "forward"
        metrics=row["evaluations"][0][direction]["metrics"]
        cases.append({"arm":arm,"method":LABELS[arm],"fixed_id":"0333","moving_id":"0115",
                      "export_direction":direction,"diffusion_coefficient":row["diffusion_coefficient"],
                      **{k:metrics[k] for k in r.METRICS},"optimizer_sec":row["optimizer_sec"],
                      "worker_sec":row["worker_sec"],"audit_sec":row["audit"]["audit_sec"],
                      "peak_cuda_gib":max(row["peak_cuda_gib"],row["audit"]["audit_peak_cuda_gib"]),
                      "peak_reserved_gib":max(row["peak_reserved_gib"],row["audit"]["audit_peak_reserved_gib"]),
                      "actual_adam_updates":row["actual_adam_updates"],"gpu":row["gpu"]})
        for baseline in manifest["references"]:
            paired.append({"arm":arm,"reference":baseline["series"],
                           **{k+"_delta":metrics[k]-baseline[k] for k in r.METRICS}})
        for block in row["blocks"]:
            vals=block["branches"]["full_volume"]
            entry={"arm":arm,"cps":block["cps"],**{k:v for k,v in vals.items() if not isinstance(v,list)},
                   **{f"mean_v{j}_{axis}_mm":vals[f"mean_v{j}_xyz_mm"][i] for j in (1,2) for i,axis in enumerate("xyz")},
                   "R1_loss_grid":block["branches"]["loss_grid_diffusion"]["R1"],
                   "R2_loss_grid":block["branches"]["loss_grid_diffusion"]["R2"]}
            branches.append(entry)
            for branch in (1,2):
                support.append({"arm":arm,"cps":block["cps"],"branch":branch,
                                **block["branches"][f"branch{branch}_midpoint"]})
            info=block["optimizer"]
            stage_rows.append({"arm":arm,"cps":block["cps"],"gpu":row["gpu"],
                               **{k:info[k] for k in ("actual_epochs","best_epoch","stopping_reason","training_elapsed_sec",
                                                       "peak_cuda_gib","peak_reserved_gib","best_loss","best_data_loss","best_reg_loss")}})
        for ev in row["evaluations"]:
            for direction in ("forward","reverse"):
                entry=ev[direction]
                if r.sha(entry["dvf_path"])!=entry["dvf_sha256"]:raise RuntimeError("Export hash mismatch")
                integrations.append({"arm":arm,"steps":ev["integration_steps"],"direction":direction,
                                     **{k:entry["metrics"][k] for k in r.METRICS},
                                     **{k+"_delta_from_ss7":entry["metrics"][k]-row["evaluations"][0][direction]["metrics"][k] for k in r.METRICS}})
                if ev["integration_steps"]==7:
                    with (Path(entry["dvf_path"]).parent/direction/"case_metrics.csv").open() as f:
                        label_rows.extend({"arm":arm,"direction":direction,**item} for item in csv.DictReader(f))
    lookup={row["spec"]["arm"]:row for row in rows}
    swap={}
    for normal,swapped in (("forward","reverse"),("reverse","forward")):
        a,b=lookup["A1"]["evaluations"][0][normal],lookup["A1_swap"]["evaluations"][0][swapped]
        target="0333" if normal=="forward" else "0115"
        mask=r.c.load_nifti_reoriented(Path(manifest["settings"]["oasis_root"])/"test"/f"img{target}.nii.gz",dtype=np.float32)>0
        swap[normal]={"field_difference":difference(np.load(a["dvf_path"]),np.load(b["dvf_path"])),
                      "field_difference_raw_foreground":difference(np.load(a["dvf_path"]),np.load(b["dvf_path"]),mask),
                      "metric_deltas_swapped_minus_normal":{k:b["metrics"][k]-a["metrics"][k] for k in r.METRICS}}
    swap["stage_updates"]={arm:[b["optimizer"]["actual_epochs"] for b in lookup[arm]["blocks"]] for arm in ("A1","A1_swap")}
    swap["stage_best_epochs"]={arm:[b["optimizer"]["best_epoch"] for b in lookup[arm]["blocks"]] for arm in ("A1","A1_swap")}
    swap["interpretation"]="Measured input-exchange discrepancy; CUDA backward need not be bitwise deterministic. This is separate from inverse residual within one fit."
    references=[{"arm":x["series"],"method":x["method"]+"; historical lambda 1",**{k:x[k] for k in r.METRICS}} for x in manifest["references"]]
    result={"phase":"a","complete":True,"single_case_exploratory":True,"phase_b_started":False,
            "candidates":rows,"comparison":references+cases,"branches":branches,"stage_diagnostics":stage_rows,
            "paired":paired,"integration_sensitivity":integrations,"input_swap":swap,"midpoint_support":support,
            "manifest_sha256":r.sha(root/"launch_manifest.json"),"report_source_sha256":r.sha(__file__)}
    out.mkdir(parents=True,exist_ok=False)
    r.c.write_csv(out/"comparison.csv",references+cases)
    for name,values in (("branch_energies",branches),("stages",stage_rows),("paired_differences",paired),
                        ("integration_sensitivity",integrations),("per_label",label_rows),("midpoint_support",support)):
        r.c.write_csv(out/f"{name}.csv",values)
    r.atomic_json(out/"results.json",result)
    r.atomic_json(out/"input_swap.json",swap)
    lines=["# Rank-0 symmetric B-spline SVF: Phase A", "",
           "Fixed 0333, moving 0115; seed-3 rank 0. All new fits start from identity, cps8→4→2. Single exploratory case: no across-case SD.", "",
           "A1/A2 use two independently refined total SVFs, not pyramid-map composition. A1 weights R(v1)+R(v2) by 2; A2 weights it by 1. A3 ties v1=w/2, v2=-w/2 and uses R(w). Actual velocities are half the normalized parameter fields; parameter LR is 0.03/0.01/0.003. Seven squarings per branch, unit time. In the tied arm, half-flow composition has its own discrete integration error and is not assumed identical to the historical seven-step full-w export.","",
           "The swapped run is evaluated using its reverse export for the original 0115→0333 comparison below. All accuracy and regularity values describe one complete output field, not midpoint branches.","",
           "| Method | Dice ↑ | HD95 mm ↓ | ASD mm ↓ | FG SD(log J) | FG folding % |",
           "|---|---:|---:|---:|---:|---:|"]
    for x in references+cases:
        lines.append(f"| {x['method']} | {x['dice']:.6f} | {x['hd95']:.6f} | {x['asd']:.6f} | {x['log_jdet_std_fg']:.6f} | {x['jdet_le0_pct_fg']:.6e} |")
    lines += ["", "## Final branch energies", "", "Native physical units, full-volume voxel mean of squared vector norm. Smoothness is mean squared forward differences over components and axes. Equal magnitude is not a proof of a geometric midpoint.", "",
              "| Arm | RMS v1 mm | RMS v2 mm | Relative RMS gap % | R1 native | R2 native | Common-mode RMS mm |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for x in branches:
        if x["cps"]==2:lines.append(f"| {x['arm']} | {x['a1_mm']:.6f} | {x['a2_mm']:.6f} | {100*x['relative_rms_imbalance']:.4f} | {x['R1_native']:.6f} | {x['R2_native']:.6f} | {x['common_mode_rms_mm']:.6f} |")
    lines += ["", "## Midpoint support and common-motion audit", "",
              "| Arm | Original moving/fixed positive % | Midpoint branch 1/2 positive % | Velocity cosine |",
              "|---|---:|---:|---:|"]
    for row in rows:
        b=row["blocks"][-1]["branches"]
        a,z=b["branch1_midpoint"],b["branch2_midpoint"]
        lines.append(f"| {row['spec']['arm']} | {100*a['source_positive_fraction']:.3f}/{100*z['source_positive_fraction']:.3f} | {100*a['image_positive_fraction']:.3f}/{100*z['image_positive_fraction']:.3f} | {b['full_volume']['velocity_cosine']:.6f} |")
    lines += ["", "An equal-magnitude pair need not be opposite or represent a centered midpoint. Large common-mode RMS and increasing midpoint foreground coverage indicate shared coordinate deformation. In the inherited NCC9 arithmetic, an all-zero window contributes correlation zero (loss one), even when both images are zero. Changing both coordinate maps can therefore change the weighting of informative versus empty regions in the unmasked midpoint average. The measured coverage/energy changes are consistent with this objective-level incentive; they are not a failed gradient or map-replay check. A balance-only penalty cannot remove equally large common motion. No mask or objective correction was silently applied."]
    lines += ["", "## Timing and memory", "", "Shared-load diagnostic timings; worker includes metrics/diagnostics, replay is separate. Peaks are the larger worker/replay allocation; they exclude other GPU jobs and some context memory.", "",
              "| Arm | Optimizer s | Worker s | Replay s | Peak allocated/reserved GiB | Joint updates | GPU |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for x in cases:lines.append(f"| {x['arm']} | {x['optimizer_sec']:.2f} | {x['worker_sec']:.2f} | {x['audit_sec']:.2f} | {x['peak_cuda_gib']:.3f}/{x['peak_reserved_gib']:.3f} | {x['actual_adam_updates']} | {x['gpu']} |")
    lines += ["", "## Input exchange and inverse diagnostics", "",
              f"Input-swapped A1 versus normal A1, original forward direction: component RMS {swap['forward']['field_difference']['component_rms_mm']:.6g} mm; maximum endpoint difference {swap['forward']['field_difference']['max_endpoint_mm']:.6g} mm. This is an optimization-symmetry diagnostic, not the within-fit inverse-consistency error.", "",
              "Each fit separately records cycle errors at 100,000 off-grid raw-positive foreground points in each direction, all-sample and in-bounds errors/coverage, and exported-interpolant versus retained-composition differences. Reported 8/10-step sensitivity is diagnostic; primary fields remain seven-step. No unconditional zero-folding guarantee.", "",
              "| Arm | Forward cycle mean / p95 mm | Forward in-bounds % | Reverse cycle mean / p95 mm | Reverse in-bounds % |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        off=row["evaluations"][0]["offgrid"]
        a,b=off["forward"],off["reverse"]
        lines.append(f"| {row['spec']['arm']} | {a['cycle_all']['mean_mm']:.6f} / {a['cycle_all']['p95_mm']:.6f} | {100*a['inbounds_fraction']:.3f} | {b['cycle_all']['mean_mm']:.6f} / {b['cycle_all']['p95_mm']:.6f} | {100*b['inbounds_fraction']:.3f} |")
    lines += ["",
              "All four fits and independent replays are required. Histories, per-label metrics, stage energies, translation/common-mode values, midpoint coverage, source/input hashes and checkpoint paths are in the JSON/CSV companions. Original references remain read-only. No amplitude or balance term was used; Phase B has not been launched.", ""]
    (out/"report.md").write_text("\n".join(lines))
    # Independent saved-table readback, including historical and original-direction swap routing.
    with (out/"comparison.csv").open() as f:saved=list(csv.DictReader(f))
    if len(saved)!=6:raise RuntimeError("Incomplete comparison table")
    for x,y in zip(saved,references+cases):
        for k in r.METRICS:
            if float(x[k])!=y[k]:raise RuntimeError("Saved metric mismatch")
    r.assert_manifest(manifest)
    accepted={"complete":True,"fits":4,"independent_replays":4,"comparison_rows":6,"stage_rows":12,
              "integration_rows":24,"canonical_statistic_checks":30,"phase_b_started":False,
              "accepted_at":datetime.now(timezone.utc).isoformat(),"source_unchanged":True,
              "output_sha256":{p.name:r.sha(p) for p in out.iterdir() if p.is_file()}}
    r.atomic_json(out/"ACCEPTED.json",accepted)
    print(json.dumps({"report":str(out/"report.md"),"acceptance":{k:v for k,v in accepted.items() if k!="output_sha256"},
                      "comparison":references+cases,"final_energies":[x for x in branches if x["cps"]==2],"input_swap":swap},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root",type=Path,default=Path("/database/inrorspline"))
    main(parser.parse_args())
