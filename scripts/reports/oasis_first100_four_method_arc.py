#!/usr/bin/env python
"""Read-only aggregation and 2x3 ARC export for MR-D-BSCP/SITReg/VFA/FeDDA."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import oasis_first100_with_sitreg_sweep as s

REPO = Path(__file__).resolve().parents[2]
X = "log_jdet_std_fg"
METRICS = (("dice", "Dice ↑"), ("hd95", "HD95 (mm) ↓"), ("asd", "Symmetric ASD (mm) ↓"))
COLORS = {"MR-D-BSCP":"#D55E00", "SITReg":"#0072B2", "VFA":"#009E73", "FeDDA-1MF":"#9C4AA0"}
MARKERS = {"MR-D-BSCP":"o", "SITReg":"s", "VFA":"^", "FeDDA-1MF":"P"}


def check_rows(rows, pairs):
    s.summarize(rows)  # complete unique ranks and finite metrics for each group
    for row in rows:
        if (row["fixed_id"], row["moving_id"]) != pairs[row["rank"]]:
            raise ValueError("Case identity mismatch")


def check_summary(rows, saved):
    calculated = s.summarize(rows)
    expected = {r["method"]:r for r in saved}
    for row in calculated:
        for metric in s.METRICS:
            for suffix in ("mean", "std"):
                key = metric+"_"+suffix
                if not math.isclose(row[key], expected[row["method"]][key], abs_tol=1e-12, rel_tol=1e-12):
                    raise ValueError("Stored/recomputed metric mismatch")
    return calculated


def load_all(args):
    provenance = {}
    def record(path, expected=None):
        actual = s.sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Checksum mismatch: {path}")
        provenance[str(path)] = actual
    split = REPO/"configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"
    record(split)
    pairs = re.findall(r"\((\d{4}),\s*(\d{4})\)", split.read_text())[:100]
    if len(pairs) != 100:
        raise ValueError("Wrong split")
    accepted = s.read(args.base/"ACCEPTED.json")
    if not accepted["complete"] or accepted["groups"] != 10:
        raise ValueError("Incomplete MR-D-BSCP/SITReg source")
    record(args.base/"ACCEPTED.json")
    for name, digest in accepted["output_sha256"].items():
        record(args.base/name, digest)
    base = s.read(args.base/"results.json")
    rows = list(base["per_case"])
    check_rows(rows, pairs)
    check_summary(rows, base["means"])
    common_hash = {v for p,v in base["input_sha256"].items() if Path(p).name == "common.py"}
    if len(common_hash) != 1:
        raise ValueError("Ambiguous canonical source")
    queue = s.read(args.vfa/"queue.json")
    record(args.vfa/"queue.json")
    if queue["status"] != "complete" or not (args.vfa/"COMPLETED").exists():
        raise ValueError("VFA sweep incomplete")
    if sorted(j["lambda_key"] for j in queue["jobs"]) != sorted(s.LAMBDAS):
        raise ValueError("Wrong VFA lambda matrix")
    checkpoints = {}
    for job in queue["jobs"]:
        root = Path(job["eval_dir"])
        if job["status"] != "complete" or not (root/"COMPLETED").exists() or (root/"FAILED.txt").exists():
            raise ValueError("Incomplete VFA evaluation")
        for name in ("manifest.json", "summary.json", "per_case.json"):
            record(root/name)
        manifest, summary = s.read(root/"manifest.json"), s.read(root/"summary.json")
        if (manifest["direction"], manifest["foreground"], manifest["jacobian"], manifest["asd_definition"]) != (
            "moving_to_fixed", "fixed_raw_image_positive", "canonical_LUMIR_crop2_positive_J_log_std",
            "pooled_surfel_area_weighted_bidirectional_mean"):
            raise ValueError("VFA metric convention mismatch")
        if manifest["smoke"] or manifest["pair_file_sha256"] != s.sha(split) or summary["case_count"] != 100 or summary["status"] != "complete":
            raise ValueError("Incomplete/wrong VFA cohort")
        for path, digest in manifest["source_sha256"].items():
            if Path(path).name == "common.py":
                if digest not in common_hash:
                    raise ValueError("Different canonical evaluator")
                record(Path(path), digest)
        key, alpha = job["lambda_key"], s.LAMBDAS[job["lambda_key"]]
        if manifest["alpha"] != alpha:
            raise ValueError("Wrong lambda")
        record(Path(manifest["checkpoint"]), manifest["checkpoint_sha256"])
        checkpoints[key] = {k:manifest[k] for k in ("checkpoint", "checkpoint_sha256", "best_epoch", "backend")}
        name = "VFA λ="+("1/3" if key == "1over3" else key)
        subset = []
        for row in s.read(root/"per_case.json"):
            if row["direction"] != "moving_to_fixed" or row["checkpoint_sha256"] != manifest["checkpoint_sha256"] or row["alpha"] != alpha:
                raise ValueError("Wrong case provenance")
            case = root/"cases"/f"rank{row['rank']:03d}_{row['fixed_id']}_{row['moving_id']}"
            if not (case/"COMPLETED").exists():
                raise ValueError("Missing case completion")
            subset.append(dict(method=name, rank=row["rank"], fixed_id=row["fixed_id"], moving_id=row["moving_id"],
                               **{m:row[k] for m,k in zip(s.METRICS, s.SOURCE_METRICS)}))
        check_rows(subset, pairs)
        expected = dict(method=name, n=100)
        for metric, original in zip(s.METRICS, s.SOURCE_METRICS):
            stat = summary["metrics"][original]
            if stat["n"] != 100:
                raise ValueError("Incomplete metric")
            expected.update({metric+"_mean":stat["mean"], metric+"_std":stat["sample_sd"]})
        check_summary(subset, [expected])
        rows.extend(subset)
    accepted = s.read(args.fedda/"ACCEPTED.json")
    record(args.fedda/"ACCEPTED.json")
    if (accepted["groups"],accepted["case_rows"],accepted["new_candidates"]) != (7,700,4):
        raise ValueError("Incomplete FeDDA report")
    for name, digest in accepted["report_sha256"].items():
        record(args.fedda/name, digest)
    for path, digest in accepted["input_sha256"].items():
        record(Path(path), digest)
    historical = s.read(args.fedda/"historical_lambda1/manifest.json")
    for path, digest in historical["source_sha256"].items():
        if Path(path).name == "common.py":
            if digest not in common_hash:
                raise ValueError("FeDDA canonical source mismatch")
            record(Path(path), digest)
    fedda = s.read(args.fedda/"results.json")
    check_rows(fedda["per_case"], pairs)
    check_summary(fedda["per_case"], fedda["summary"])
    for row in fedda["per_case"]:
        if row["method"].startswith("FeDDA"):
            rows.append({k:row[k] for k in ("method", "rank", "fixed_id", "moving_id", *s.METRICS)})
    check_rows(rows,pairs)
    means = s.summarize(rows)
    if len(rows) != 1900 or len(means) != 19:
        raise ValueError("Wrong four-method matrix")
    for row in means:
        name = row["method"]
        row["family"] = next(m for m in COLORS if name.startswith(m))
        row["historical"] = "original" in name or "historical" in name
        if row["historical"]:
            row["alpha"] = 1.
        else:
            value = name.split("λ=")[-1] if "λ=" in name else name.split("lambda")[-1]
            row["alpha"] = 1/3 if value in ("1/3","0.333333") else float(value)
    return rows, means, provenance, checkpoints


def build_figure(means):
    fig,axes = plt.subplots(2,3,figsize=(18,10.8),sharex=True,sharey="col")
    xmin = min(r[X+"_mean"]-r[X+"_std"] for r in means)
    xmax = max(r[X+"_mean"]+r[X+"_std"] for r in means)
    for row_index in range(2):
        for col,(metric,title) in enumerate(METRICS):
            ax=axes[row_index,col]
            for family,color in COLORS.items():
                curve=sorted((r for r in means if r["family"]==family),key=lambda r:r["alpha"],reverse=True)
                # Caption states left-to-right lambda order: enforce it.
                if any(a[X+"_mean"] >= b[X+"_mean"] for a,b in zip(curve,curve[1:])):
                    raise ValueError("Nonmonotone regularity: revise lambda-order caption")
                for a,b in zip(curve,curve[1:]):
                    ax.plot([a[X+"_mean"],b[X+"_mean"]],[a[metric+"_mean"],b[metric+"_mean"]],
                            color=color,lw=2.1,ls="--" if a["historical"] or b["historical"] else "-",zorder=3)
                for r in curve:
                    x,y=r[X+"_mean"],r[metric+"_mean"]
                    if row_index:
                        ax.add_patch(Ellipse((x,y),2*r[X+"_std"],2*r[metric+"_std"],
                            facecolor=to_rgba(color,.065),edgecolor=to_rgba(color,.28),lw=.8,zorder=1))
                    ax.plot(x,y,marker="D" if r["historical"] else MARKERS[family],ms=6.3,
                            color=color,mfc="white" if r["historical"] else color,mew=1.25,ls="none",zorder=4)
            ax.set_title(f"{'abcdef'[3*row_index+col]}) {title}",loc="left",fontsize=13,fontweight="semibold")
            ax.set_xlabel("Foreground SD(log J) ↓",fontsize=11)
            ax.set_ylabel(title,fontsize=11)
            ax.set_xlim(max(0,xmin-.035),xmax+.04)
            lo=min(r[metric+"_mean"]-r[metric+"_std"] for r in means)
            hi=max(r[metric+"_mean"]+r[metric+"_std"] for r in means)
            ax.set_ylim(max(0,lo-.07*(hi-lo)),hi+.07*(hi-lo))
            ax.grid(alpha=.18,lw=.7)
            ax.spines[["top","right"]].set_visible(False)
            ax.tick_params(labelbottom=True)
    handles=[Line2D([],[],color=c,marker=MARKERS[m],lw=2.1,label=m) for m,c in COLORS.items()]
    handles.append(Line2D([],[],color="#444444",marker="D",mfc="white",ls="--",label="Historical DL λ=1"))
    fig.suptitle("OASIS first100 · accuracy–regularity tradeoffs",fontsize=19,y=.985)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.949),ncol=5,frameon=False,fontsize=11)
    fig.text(.5,.884,"TOP: mean curves only",ha="center",fontsize=12,fontweight="semibold")
    fig.text(.5,.495,"BOTTOM: same curves with ±1 across-case SD ellipses",ha="center",fontsize=12,fontweight="semibold")
    fig.text(.5,.080,"Left → right: MR-D-BSCP λ = 3, 1, 0.3, 0.1; each DL curve λ = 10, 3, 1 (historical), 1/3, 0.1.",ha="center",fontsize=10)
    fig.text(.5,.055,"MR-D-BSCP: pyramid 8→4→2→dense · 100 paired cases/point · Ellipses: marginal sample SD, not confidence/covariance regions.",ha="center",fontsize=10)
    fig.text(.5,.030,"Hollow diamonds/dashed segments involve historical DL checkpoints, not new λ=1 retraining. SD(log J) excludes J≤0; it is not folding.",ha="center",fontsize=9.8,color="#555555")
    fig.subplots_adjust(left=.06,right=.985,bottom=.15,top=.847,wspace=.26,hspace=.51)
    return fig,axes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("base","vfa","fedda","out"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.out.resolve().is_relative_to(REPO):
        raise ValueError("Generated artifacts must be external")
    rows,means,provenance,checkpoints=load_all(args)
    print("Validated 19 settings / 1900 paired case rows",flush=True)
    args.out.mkdir(parents=True,exist_ok=False)
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"svg.fonttype":"none"}):
        fig,_=build_figure(means)
        for ext in ("png","pdf","svg"):
            fig.savefig(args.out/f"arc_four_methods_first100_2x3.{ext}",dpi=200,facecolor="white")
        plt.close(fig)
    s.write_csv(args.out/"plotted_mean_std.csv",means)
    s.write_csv(args.out/"per_case.csv",rows)
    s.write_json(args.out/"results.json",dict(means=means,per_case=rows,vfa_checkpoints=checkpoints))
    if any(s.sha(Path(path))!=digest for path,digest in provenance.items()):
        raise ValueError("Source changed during plotting")
    s.write_json(args.out/"plot_manifest.json",dict(complete=True,points=19,cases_per_point=100,case_rows=1900,
        layout="2x3: mean-only top; marginal sample-SD ellipses bottom",ellipse_width_height="2*sample SD; ddof=1; not CI",
        source_sha256=provenance,script_sha256=s.sha(Path(__file__)),
        metrics="Existing complete canonical evaluations; cohort means/SD recomputed, no new inference",
        historical_lambda1="VFA, SITReg and original 200-epoch FeDDA-1MF; hollow diamonds and dashed adjacent segments",
        output_sha256={f.name:s.sha(f) for f in args.out.iterdir() if f.is_file()}))
    print(args.out,flush=True)


if __name__=="__main__":
    main()
