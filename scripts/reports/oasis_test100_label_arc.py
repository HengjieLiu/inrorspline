#!/usr/bin/env python
"""Versioned label-wise ARC from saved scalar metrics only (no field evaluation)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

REPO = Path(__file__).resolve().parents[2]
KEY = "oasis_seed3_test100_label_arc"
ARC = "oasis_seed3_first100_four_method_sweeps_arc_v1"
MR = "oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1"
FEDDA = "oasis_seed3_first100_fedda_training_lambda_sweep_v1"
ACC = ("dice", "hd95", "asd")
REG = ("log_jdet_std_fg", "jdet_le0_pct_fg")
METRICS = ACC + REG
WIDE = dict(dice="dice", hd95="hd95", asd="asd_symmetric")
LONG = dict(dice="dice", hd95="hd95_mm", asd="asd_symmetric_mm")
TITLES = dict(dice="Dice ↑", hd95="HD95 (mm) ↓", asd="Symmetric ASD (mm) ↓")
COLORS = {"MR-D-BSCP dense": "#D55E00", "VFA": "#009E73", "SITReg": "#0072B2",
          "FeDDA-1MF": "#9C4AA0", "MR-D-BSCP prefix": "#595959"}
MARKERS = dict(zip(COLORS, ("o", "^", "s", "P", "v")))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b, message):
    if not math.isclose(float(a), float(b), abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError(f"{message}: {a} != {b}")


def number(value):
    if value is None or value == "":
        return None
    x = float(value)
    if math.isnan(x):
        return None
    if not math.isfinite(x):
        raise ValueError("Infinite metric is not canonical absence")
    return x


def label_names(text):
    names = {}
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0].isdigit() and 1 <= int(parts[0]) <= 35:
            label = int(parts[0])
            if label in names:
                raise ValueError("Duplicate label ID")
            names[label] = parts[1]
    if set(names) != set(range(1, 36)):
        raise ValueError("Require official labels1–35")
    return names


def alpha_key(alpha):
    return "1over3" if abs(alpha - 1/3) < 1e-12 else f"{alpha:g}"


class Audit:
    def __init__(self):
        self.inputs = {}
        self.expected = {}
        self.fields = {}

    def record(self, path):
        path = Path(path)
        digest = sha(path)
        expected = self.expected.get(str(path), self.inputs.get(str(path)))
        if expected is not None and digest != expected:
            raise ValueError(f"Source hash mismatch: {path}")
        self.inputs[str(path)] = digest
        return path

    def read(self, path):
        return json.loads(self.record(path).read_text())

    def csv(self, path):
        with self.record(path).open() as f:
            return list(csv.DictReader(f))

    def accepted(self, directory, name="ACCEPTED.json", hash_key="report_sha256"):
        directory = Path(directory)
        data = self.read(directory / name)
        for filename, digest in data[hash_key].items():
            self.expected[str(directory / filename)] = digest
        return data

    def marker(self, directory, name):
        self.record(Path(directory) / name)

    def field(self, path, digest):
        path = Path(path)
        stat = path.stat()
        if not path.is_file() or stat.st_size == 0 or len(digest) != 64:
            raise ValueError(f"Missing field / recorded digest: {path}")
        value = dict(recorded_sha256=digest, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if str(path) in self.fields and self.fields[str(path)] != value:
            raise ValueError("Conflicting field identity")
        self.fields[str(path)] = value

    def finish(self):
        for path, digest in self.inputs.items():
            if sha(path) != digest or self.expected.get(path, digest) != digest:
                raise ValueError(f"Source changed during report: {path}")
        for path, old in self.fields.items():
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns) != (old["size"], old["mtime_ns"]):
                raise ValueError(f"Field changed during report: {path}")


def validate_group(cases, labels, pairs):
    if sorted(r["rank"] for r in cases) != list(range(len(pairs))):
        raise ValueError("Missing/duplicate case rank")
    expected = {(r, label) for r in range(len(pairs)) for label in range(1, 36)}
    keys = [(r["rank"], r["label_id"]) for r in labels]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing/duplicate label slot")
    by_rank = defaultdict(list)
    for row in labels:
        if (row["fixed_id"], row["moving_id"]) != pairs[row["rank"]]:
            raise ValueError("Label pair mismatch")
        valid = all(row[m] is not None for m in ACC)
        if any(row[m] is not None for m in ACC) != valid or valid != row["included"]:
            raise ValueError("Inconsistent absent-label handling")
        if valid and not (0 <= row["dice"] <= 1 and row["hd95"] >= 0 and row["asd"] >= 0):
            raise ValueError("Invalid accuracy metric")
        by_rank[row["rank"]].append(row)
    for row in cases:
        if (row["fixed_id"], row["moving_id"]) != pairs[row["rank"]] or row["direction"] != "moving_to_fixed":
            raise ValueError("Case pair/direction mismatch")
        if not all(math.isfinite(row[m]) for m in METRICS):
            raise ValueError("Nonfinite case metric")
        for m in ACC:
            values = [r[m] for r in by_rank[row["rank"]] if r["included"]]
            close(statistics.mean(values), row[m], "Label-to-case reconstruction")


def load_sources(args):
    audit = Audit()
    audit.record(Path(__file__))
    audit.record(REPO / "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt")
    split = REPO / "configs/splits/oasis/pair_test_200_shuffled_seed000003.txt"
    pairs = re.findall(r"\((\d{4}),\s*(\d{4})\)", split.read_text())[:100]
    if len(pairs) != 100:
        raise ValueError("Wrong split")
    names = label_names(audit.record(args.label_map).read_text())
    reports, runs = args.artifact_root / "repo_experiments", args.artifact_root / "runs"
    arc_manifest = audit.accepted(reports / ARC, "plot_manifest.json", "output_sha256")
    if not arc_manifest["complete"] or arc_manifest["points"] != 19:
        raise ValueError("Unaccepted original ARC")
    audit.expected.update(arc_manifest["source_sha256"])
    original = audit.read(reports / ARC / "results.json")
    original_cases = {(r["method"], r["rank"]): r for r in original["per_case"]}
    accepted = audit.accepted(reports / MR)
    if not accepted["technical_completion"]:
        raise ValueError("Unaccepted MR source")
    mr = audit.read(reports / MR / "results.json")
    configs, cases, labels = [], [], []

    def add_config(config_id, family, alpha, historical, source, old_name):
        config = dict(config_id=config_id, family=family, alpha=alpha, lambda_key=alpha_key(alpha),
                      historical=historical, source=str(source), original_method=old_name,
                      main=family != "MR-D-BSCP prefix", loss="local_squared_NCC9", regularizer="voxel_diffusion",
                      stage_lambdas=[alpha]* (3 if family == "MR-D-BSCP prefix" else 4) if family.startswith("MR") else None)
        configs.append(config)
        return config

    def add_case(config, source_row, rank, field, digest, metric_source, long_rows=None):
        audit.field(field, digest)
        row = dict(config_id=config["config_id"], rank=rank, fixed_id=source_row["fixed_id"],
                   moving_id=source_row["moving_id"], direction="moving_to_fixed",
                   field_path=str(field), recorded_field_sha256=digest, metric_source=str(metric_source))
        for m in METRICS:
            row[m] = float(source_row.get(m, source_row.get(WIDE.get(m, m)+"_mean")))
        if config["main"]:
            reference = original_cases[(config["original_method"], rank)]
            for m in METRICS:
                close(row[m], reference[m], f"Accepted ARC parity {config['config_id']} rank{rank} {m}")
        cases.append(row)
        for label in range(1, 36):
            detail = long_rows[label] if long_rows is not None else None
            values = {m: number(detail[LONG[m]] if detail else source_row[f"{WIDE[m]}_label_{label:02d}"]) for m in ACC}
            valid = all(v is not None for v in values.values())
            if detail is not None and bool(detail["included_in_case_mean"]) != valid:
                raise ValueError("Canonical label inclusion mismatch")
            labels.append(dict(config_id=config["config_id"], rank=rank, fixed_id=row["fixed_id"],
                               moving_id=row["moving_id"], label_id=label, label_name=names[label],
                               included=valid, exclusion_reason=detail["exclusion_reason"] if detail else
                               ("" if valid else "canonical_absent_label; legacy detailed reason not stored"),
                               **values))

    # MR prefix and dense; every per-run CSV retains all35 labels.
    for representation, family in (("dense_voxel", "MR-D-BSCP dense"), ("bspline_control", "MR-D-BSCP prefix")):
        for alpha in (.1, .3, 1., 3.):
            config = add_config(f"mr_{'dense' if representation == 'dense_voxel' else 'prefix'}_{alpha_key(alpha)}",
                                family, alpha, False, reports/MR, f"MR-D-BSCP λ={alpha:g}")
            subset = [r for r in mr["candidates"] if r["representation"] == representation and r["spec"]["alpha"] == alpha]
            for r in subset:
                directory = Path(r["run_dir"])
                audit.marker(directory, "DONE")
                saved = audit.read(directory / "summary.json")
                if saved["dvf_sha256"] != r["dvf_sha256"] or saved["spec"] != r["spec"] or r["direction"] != "moving_to_fixed":
                    raise ValueError("MR source identity mismatch")
                wide = audit.csv(directory / "case_metrics.csv")
                if len(wide) != 1 or (wide[0]["fixed_id"], wide[0]["moving_id"]) != (r["fixed_id"], r["moving_id"]):
                    raise ValueError("MR label CSV identity mismatch")
                for m in METRICS:
                    close(saved[m], r[m], "MR summary parity")
                add_case(config, {**wide[0], **{k:r[k] for k in METRICS}}, r["case_rank"], r["dvf_path"], r["dvf_sha256"], directory/"case_metrics.csv")

    # Original VFA/SITReg: same field, older saved per-label metrics, current canonical regularity.
    for family, old_id, old_name in (("VFA", "bs3_vfa_v0", "VFA (original)"), ("SITReg", "bs4_sitreg_v0", "SITReg λ=1 (original)")):
        config = add_config(f"{family.lower()}_1_historical", family, 1., True, reports/MR, old_name)
        references = {r["case_rank"]:r for r in mr["case_metrics"] if r["representation"] == family}
        for root in ("oasis_seed3_first10_saved_baselines_v1", "oasis_seed3_ranks020_099_saved_baselines_v1"):
            for path in sorted((runs/root/old_id).glob("*/summary.json")):
                saved = audit.read(path)
                rank = saved["case_rank"]
                if not 0 <= rank < 100:
                    continue
                audit.marker(path.parent, "DONE")
                ref = references[rank]
                # Legacy mount alias must resolve to the same relative wavereg field.
                if saved["source_disp_path"].split("/wavereg/")[-1] != ref["source_path"].split("/wavereg/")[-1]:
                    raise ValueError("Historical field path alias mismatch")
                for m in ACC:
                    close(saved[WIDE[m]+"_mean"], ref[m], "Historical accuracy parity")
                add_case(config, {**saved, **{m:ref[m] for m in METRICS}}, rank, ref["source_path"], ref["source_sha256"], path)

    # Detailed new DL evaluations, and historical FeDDA.
    detailed = []
    for key, alpha in (("0.1",.1),("1over3",1/3),("3",3.),("10",10.)):
        detailed.append(("SITReg", alpha, False, runs/"oasis_seed3_first100_sitreg_training_lambda_eval_v1"/f"sitreg_lambda_{key}_seed42", f"SITReg λ={'1/3' if key == '1over3' else key}"))
    for path in arc_manifest["source_sha256"]:
        if not path.endswith("/per_case.json"):
            continue
        directory = Path(path).parent
        if "/first100/" not in path:
            continue
        family = "VFA" if "vfa_lambda_" in directory.name else "FeDDA-1MF"
        match = re.search(r"lambda_(0\.1|1over3|3|10)_", directory.name)
        if not match:
            raise ValueError("Unrecognized DL configuration")
        key = match[1]
        alpha = 1/3 if key == "1over3" else float(key)
        old_name = f"VFA λ={'1/3' if key == '1over3' else key}" if family == "VFA" else f"FeDDA-1MF lambda{alpha:g}"
        detailed.append((family, alpha, False, directory, old_name))
    fedda_accept = audit.accepted(reports/FEDDA)
    if fedda_accept["new_candidates"] != 4:
        raise ValueError("Wrong FeDDA source")
    audit.expected.update(fedda_accept["input_sha256"])
    detailed.append(("FeDDA-1MF", 1., True, reports/FEDDA/"historical_lambda1", "FeDDA-1MF lambda1 historical"))
    for family, alpha, historical, directory, old_name in detailed:
        audit.marker(directory, "COMPLETED")
        manifest = audit.read(directory/"manifest.json")
        if manifest["foreground"] != "fixed_raw_image_positive" or manifest["pair_file_sha256"] != sha(split):
            raise ValueError("DL foreground/split mismatch")
        if not historical:
            expected = dict(direction="moving_to_fixed", jacobian="canonical_LUMIR_crop2_positive_J_log_std",
                            asd_definition="pooled_surfel_area_weighted_bidirectional_mean", smoke=False)
            if any(manifest[k] != value for k, value in expected.items()):
                raise ValueError("Detailed metric convention mismatch")
            close(manifest["alpha"], alpha, "DL alpha mismatch")
        config = add_config(f"{family.lower()}_{alpha_key(alpha)}"+("_historical" if historical else ""), family, alpha, historical, directory, old_name)
        config.update(checkpoint=manifest["checkpoint"], checkpoint_sha256=manifest["checkpoint_sha256"])
        long = audit.read(directory/"per_label_per_case.json")
        grouped = defaultdict(dict)
        for r in long:
            if r["label_id"] in grouped[r["rank"]] or (r["fixed_id"],r["moving_id"]) != pairs[r["rank"]]:
                raise ValueError("Detailed label identity/duplicate")
            grouped[r["rank"]][r["label_id"]] = r
        if len(long) != 3500:
            raise ValueError("Incomplete detailed label table")
        for r in audit.read(directory/"per_case.json"):
            if not historical and r["direction"] != "moving_to_fixed":
                raise ValueError("Wrong DL direction")
            add_case(config, r, r["rank"], r["field_path"], r["field_sha256"], directory/"per_label_per_case.json", grouped[r["rank"]])
    if len(configs) != 23 or len({r["config_id"] for r in configs}) != 23:
        raise ValueError("Expected23 unique configurations")
    for config in configs:
        cid = config["config_id"]
        validate_group([r for r in cases if r["config_id"] == cid], [r for r in labels if r["config_id"] == cid], pairs)
    print(f"Validated {len(cases)} cases, {len(labels)} label slots; no field arrays loaded", flush=True)
    return audit, configs, cases, labels, names, original


def summarize(configs, cases, labels):
    case_lookup = {(r["config_id"], r["rank"]):r for r in cases}
    label_lookup = {(r["config_id"], r["label_id"], r["rank"]):r for r in labels}
    supports, rows = [], []
    for label in sorted({r["label_id"] for r in labels}):
        ranks = sorted(set.intersection(*[set(r["rank"] for r in labels if r["config_id"] == c["config_id"] and r["label_id"] == label and r["included"]) for c in configs]))
        if len(ranks) < 2:
            raise ValueError("Insufficient common support")
        supports.append(dict(label_id=label, n=len(ranks), valid_ranks=ranks,
                             excluded_ranks=sorted(set(r["rank"] for r in cases)-set(ranks))))
        for c in configs:
            row = {k:c[k] for k in ("config_id", "family", "alpha", "lambda_key", "historical", "main")}
            row.update(label_id=label, label_name=label_lookup[c["config_id"],label,ranks[0]]["label_name"], n=len(ranks))
            for m in METRICS:
                values = [(label_lookup[c["config_id"],label,rank] if m in ACC else case_lookup[c["config_id"],rank])[m] for rank in ranks]
                row[m+"_mean"], row[m+"_sd"] = statistics.mean(values), statistics.stdev(values)
            rows.append(row)
    return rows, supports


def trend_tables(configs, cases, labels, summary, supports):
    trends, paired, pareto = [], [], []
    for label in range(1, 36):
        for family in COLORS:
            group = sorted((r for r in summary if r["label_id"] == label and r["family"] == family), key=lambda r:r["alpha"])
            for m in ACC:
                sign = 1 if m == "dice" else -1
                best = max(group, key=lambda r:sign*r[m+"_mean"])
                deltas = [sign*(b[m+"_mean"]-a[m+"_mean"]) for a,b in zip(group,group[1:])]
                trends.append(dict(label_id=label, label_name=best["label_name"], family=family, metric=m,
                                   best_alpha=best["alpha"], best_mean=best[m+"_mean"], best_sd=best[m+"_sd"],
                                   min_sd=min(r[m+"_sd"] for r in group), max_sd=max(r[m+"_sd"] for r in group),
                                   nonmonotone_vs_lambda=any(d>1e-12 for d in deltas) and any(d< -1e-12 for d in deltas)))
        for m in ACC:
            group = [r for r in summary if r["label_id"] == label and r["main"]]
            sign = 1 if m == "dice" else -1
            for r in group:
                dominators = [s["config_id"] for s in group if s["log_jdet_std_fg_mean"] <= r["log_jdet_std_fg_mean"] and sign*s[m+"_mean"] >= sign*r[m+"_mean"] and
                              (s["log_jdet_std_fg_mean"] < r["log_jdet_std_fg_mean"] or sign*s[m+"_mean"] > sign*r[m+"_mean"])]
                pareto.append(dict(label_id=label, metric=m, config_id=r["config_id"], nondominated=not dominators, dominators=";".join(dominators)))
    lookup = {(r["config_id"], r["label_id"], r["rank"]):r for r in labels}
    cs = {(r["config_id"],r["rank"]):r for r in cases}
    for support in supports:
        label, ranks = support["label_id"], support["valid_ranks"]
        for alpha in (.1,.3,1.,3.):
            a,b = "mr_dense_"+alpha_key(alpha), "mr_prefix_"+alpha_key(alpha)
            row = dict(label_id=label, label_name=lookup[a,label,ranks[0]]["label_name"], alpha=alpha,n=len(ranks))
            for m in METRICS:
                diffs = [(lookup[a,label,r][m]-lookup[b,label,r][m]) if m in ACC else cs[a,r][m]-cs[b,r][m] for r in ranks]
                row[m+"_delta_mean"],row[m+"_delta_sd"] = statistics.mean(diffs),statistics.stdev(diffs)
            paired.append(row)
    return trends, paired, pareto


def plot_curve(ax, rows, metric, ellipses=False, annotate=False):
    for family in COLORS:
        group = sorted((r for r in rows if r["family"] == family), key=lambda r:r["alpha"], reverse=True)
        color = COLORS[family]
        for a,b in zip(group,group[1:]):
            ax.plot([a["log_jdet_std_fg_mean"],b["log_jdet_std_fg_mean"]], [a[metric+"_mean"],b[metric+"_mean"]], color=color, lw=1.7,
                    ls="--" if a["historical"] or b["historical"] else "-", zorder=3)
        for r in group:
            x,y = r["log_jdet_std_fg_mean"],r[metric+"_mean"]
            if ellipses:
                ax.add_patch(Ellipse((x,y), 2*r["log_jdet_std_fg_sd"],2*r[metric+"_sd"],facecolor=color,edgecolor=color,alpha=.10,zorder=1))
            ax.plot(x,y,marker="D" if r["historical"] else MARKERS[family],color=color,mfc="white" if r["historical"] else color,ms=4.5,zorder=4,ls="none")
            if annotate and (r is group[0] or r is group[-1]):
                ax.annotate("λ"+r["lambda_key"],(x,y),xytext=(4,5),textcoords="offset points",fontsize=6.5,color=color)
    ax.grid(alpha=.17)
    ax.spines[["top","right"]].set_visible(False)


def legend(families):
    return [Line2D([],[],color=COLORS[f],marker=MARKERS[f],label=f,lw=1.7) for f in families] + [Line2D([],[],color="black",marker="D",mfc="white",ls="--",label="Historical DL λ=1")]


def label_figure(rows, label, name, mode):
    fig, axes = plt.subplots(2,3,figsize=(17,10),sharex=True,sharey="col")
    for i in range(2):
        for j,m in enumerate(ACC):
            ax = axes[i,j]
            plot_curve(ax, rows, m, ellipses=bool(i))
            ax.set_title(("Mean only · " if i == 0 else "Mean ± sample SD · ")+TITLES[m],fontsize=12)
            ax.set_xlabel("Whole fixed-foreground SD(log J) ↓")
            ax.set_ylabel(TITLES[m])
            vals = [r[m+"_mean"] + sign*r[m+"_sd"] for r in rows for sign in (-1,1)]
            lo,hi = min(vals),max(vals)
            pad = max((hi-lo)*.07,.003)
            ax.set_ylim(max(0,lo-pad), min(1,hi+pad) if m == "dice" else hi+pad)
            ax.tick_params(labelbottom=True)
    families = [f for f in COLORS if any(r["family"] == f for r in rows)]
    fig.suptitle(f"{label:02d} · {name.replace('-', ' ')} · n={rows[0]['n']} paired cases",fontsize=18,y=.98)
    fig.legend(handles=legend(families),loc="upper center",bbox_to_anchor=(.5,.945),ncol=3 if len(families)>4 else 5,frameon=False,fontsize=10)
    fig.text(.5,.075,"Lines connect measured settings in λ order; no smoothing. MR λ: 0.1, 0.3, 1, 3; DL λ: 0.1, 1/3, 1, 3, 10.",ha="center",fontsize=9)
    fig.text(.5,.052,"X is whole-foreground regularity, NOT regularity inside this label. SD(log J) excludes J≤0; folding is tabulated separately.",ha="center",fontsize=9)
    fig.text(.5,.029,"Ellipses: marginal ±1 across-case sample SD (ddof=1), not confidence/covariance regions. Historical λ1 segments are dashed.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.065,right=.98,top=.83,bottom=.15,wspace=.27,hspace=.38)
    return fig


def render(root, summary, names, labels_to_plot):
    counts = {}
    for mode in ("main", "supplementary"):
        selected = [r for r in summary if mode == "supplementary" or r["main"]]
        dest = root/"figures"/mode
        dest.mkdir(parents=True, exist_ok=True)
        with PdfPages(dest/"all_35_labels.pdf") as book:
            for label in labels_to_plot:
                rows = [r for r in selected if r["label_id"] == label]
                fig = label_figure(rows,label,names[label],mode)
                stem = f"label_{label:02d}_{names[label].lower()}"
                for ext in ("png","svg"):
                    fig.savefig(dest/(stem+"."+ext),dpi=145)
                book.savefig(fig)
                plt.close(fig)
                print(f"Rendered {mode} label {label:02d}",flush=True)
        atlas = root/"atlases"/mode
        atlas.mkdir(parents=True,exist_ok=True)
        for metric in ACC:
            fig,axes = plt.subplots(7,5,figsize=(23,27),sharex=True)
            for label,ax in enumerate(axes.flat,1):
                rows = [r for r in selected if r["label_id"] == label]
                plot_curve(ax,rows,metric)
                ax.set_title(f"{label:02d} {names[label].replace('-', ' ')}\nn={rows[0]['n']}",fontsize=9)
                ax.tick_params(labelsize=8)
                if label>30:
                    ax.set_xlabel("FG SD(log J) ↓",fontsize=9)
            fig.suptitle(f"OASIS seed-3 test100 · {TITLES[metric]} by label · {mode}",fontsize=21,y=.994)
            families = [f for f in COLORS if any(r["family"] == f for r in selected)]
            fig.legend(handles=legend(families),loc="upper center",bbox_to_anchor=(.5,.98),ncol=6,frameon=False)
            fig.text(.5,.008,"Mean curves only; label-specific y scales. Whole-foreground regularity, not label-local. Historical λ1 dashed; λ-ordered segments.",ha="center",fontsize=11)
            fig.subplots_adjust(left=.045,right=.985,top=.947,bottom=.035,hspace=.43,wspace=.25)
            for ext in ("png","pdf","svg"):
                fig.savefig(atlas/(metric+"_35_labels."+ext),dpi=140)
            plt.close(fig)
        counts[mode] = len(labels_to_plot)
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root",type=Path,required=True)
    p.add_argument("--label-map",type=Path,required=True)
    p.add_argument("--release",default="v001")
    p.add_argument("--smoke",action="store_true",help="Full validation, render labels1/34 only into a separate smoke release")
    args = p.parse_args(argv)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.release):
        p.error("Invalid release name")
    if args.smoke and "smoke" not in args.release:
        p.error("Smoke requires a distinct release name containing smoke")
    base = args.artifact_root/"repo_experiments"/KEY
    root = base/"releases"/args.release
    if root.resolve().is_relative_to(REPO) or root.exists() or (base/"registry"/(args.release+".json")).exists():
        raise ValueError("Require a new external release directory")
    audit,configs,cases,labels,names,original = load_sources(args)
    summary,supports = summarize(configs,cases,labels)
    expected_n = {i:98 if i==34 else 100 for i in range(1,36)}
    if {r["label_id"]:r["n"] for r in supports} != expected_n:
        raise ValueError("Unexpected missing-label support; review before acceptance")
    trends,paired,pareto = trend_tables(configs,cases,labels,summary,supports)
    # Preserve old ARC aggregation order: mean over case-level macro means.
    case_summary = []
    for config in configs:
        subset = [r for r in cases if r["config_id"]==config["config_id"]]
        row = {k:config[k] for k in ("config_id","family","alpha","historical")}
        for m in METRICS:
            values=[r[m] for r in subset]
            row[m+"_mean"],row[m+"_sd"] = statistics.mean(values),statistics.stdev(values)
            if config["main"]:
                old=next(r for r in original["means"] if r["method"]==config["original_method"])
                close(row[m+"_mean"],old[m+"_mean"],"Original ARC mean")
                close(row[m+"_sd"],old[m+"_std"],"Original ARC SD")
        case_summary.append(row)
    root.mkdir(parents=True,exist_ok=False)
    for filename,rows in (("case_metrics",cases),("label_metrics",labels),("label_summary",summary),
                          ("case_summary",case_summary),("lambda_trends",trends),("dense_minus_prefix",paired),("observed_pareto",pareto)):
        table(root/"tables"/(filename+".csv"),rows)
        dump(root/"tables"/(filename+".json"),rows)
    dump(root/"audit/label_support.json",supports)
    dump(root/"audit/excluded_label_rows.json",[r for r in labels if not r["included"]])
    dump(root/"audit/input_sha256.json",audit.inputs)
    dump(root/"audit/field_inventory.json",audit.fields)
    metadata=dict(schema_version=1,release=args.release,created_at=datetime.now(timezone.utc).isoformat(),
                  source_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
                  source_worktree_status=subprocess.check_output(["git","status","--short"],cwd=REPO,text=True),
                  uid=os.getuid(),gid=os.getgid(),device="CPU only",matplotlib_version=matplotlib.__version__,
                  source_script_sha256=sha(Path(__file__)),configs=configs,labels=names,
                  protocol="saved canonical accuracy; whole fixed-raw-positive FG crop2 positive-J log SD; sample SD ddof1",
                  no_new_evaluation=True,field_verification="existence/size/mtime + previously recorded SHA256; no fresh field hashing",
                  future_additions="append a new registry snapshot and new release; never overwrite accepted releases")
    dump(root/"manifest.json",metadata)
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"svg.fonttype":"none"}):
        counts = render(root,summary,names,[1,34] if args.smoke else range(1,36))
    lines=["# OASIS test100 label-wise ARC", "", "Saved-metric aggregation only: no registration, inference, field loading or metric reevaluation.", "",
           "Main: 19 NCC9/diffusion settings. Supplement: same curves plus four matched-lambda 8→4→2 prefixes (23 settings total).",
           "Cohort: seed-3 ranks0–99; moving→fixed. Rank4 retained. All23 configurations have100 saved fields and3500 label slots.",
           "Right-Vessel (34):98 common valid cases; ranks35/58 excluded. Other34 labels:100 cases. Both axes use the same label-specific cases.",
           "X: whole fixed-raw-image-positive foreground SD(log J), positive Jacobians, canonical crop2. NOT within-label regularity; folding is in tables.",
           "Y: saved Dice, HD95 and pooled-surface-area-weighted symmetric ASD. Historical DL λ1 uses hollow diamonds/dashed adjacent segments.",
           "Ellipses: marginal ±1 sample SD (ddof1), not confidence intervals or covariance contours. Lines follow λ; no fitted/extrapolated curves.", "",
           "## Figures", "", "- [Main35-label PDF](figures/main/all_35_labels.pdf)",
           "- [Prefix comparison35-label PDF](figures/supplementary/all_35_labels.pdf)",
           "- [Dice atlas](atlases/main/dice_35_labels.png)", "- [HD95 atlas](atlases/main/hd95_35_labels.png)",
           "- [Symmetric ASD atlas](atlases/main/asd_35_labels.png)", "", "## Descriptive trends", "",
           "Best λ below maximizes/minimizes a label mean within each measured family; this is exploratory test-set description, not a deployable per-label tuned field.",
           "Observed Pareto tables compare measured points only, not interpolated equal-regularity estimates or significance tests.", "",
           "| Family | Metric | Best-λ counts across35 labels | Nonmonotone labels |", "|---|---|---|---:|"]
    from collections import Counter
    for family in COLORS:
        for metric in ACC:
            rs=[r for r in trends if r["family"]==family and r["metric"]==metric]
            counts_by_alpha=Counter(r["best_alpha"] for r in rs)
            lines.append(f"| {family} | {metric} | "+", ".join(f"{alpha_key(a)}: {n}" for a,n in sorted(counts_by_alpha.items()))+f" | {sum(r['nonmonotone_vs_lambda'] for r in rs)} |")
    lines += ["", "## Dense refinement versus its matched prefix", "",
              "Counts below use paired label-mean differences, not statistical significance; diffusion λ is unchanged across prefix and dense stages.",
              "", "| λ | Labels with better Dice | Better HD95 | Better ASD |", "|---:|---:|---:|---:|"]
    for alpha in (.1,.3,1.,3.):
        rs=[r for r in paired if r["alpha"]==alpha]
        lines.append(f"| {alpha:g} | {sum(r['dice_delta_mean']>0 for r in rs)}/35 | {sum(r['hd95_delta_mean']<0 for r in rs)}/35 | {sum(r['asd_delta_mean']<0 for r in rs)}/35 |")
    lines += ["", "## Reuse and provenance", "", "CSV/JSON tables retain per-case/per-label records, exclusions, all five metric summaries, lambda trends and paired dense-prefix changes.",
              "Source scalar files are SHA256-checked before/after generation; original19 case-level mean/SD values are reproduced. Historical field path aliases are resolved without editing source reports.",
              "Field arrays were not loaded or freshly hashed: inventory records existing field digests plus presence/size/mtime. No GPU jobs were modified.",
              "Add future settings via new source adapters/configuration identities and an append-only registry/release; incomplete cohorts or incompatible protocols must fail validation."]
    lines += ["", "## Individual label figures", "", "| Label | Main | Prefix comparison |", "|---|---|---|"]
    for label in ([1,34] if args.smoke else range(1,36)):
        filename=f"label_{label:02d}_{names[label].lower()}.png"
        lines.append(f"| {label:02d} {names[label]} | [PNG](figures/main/{filename}) | [PNG](figures/supplementary/{filename}) |")
    (root/"report.md").write_text("\n".join(lines)+"\n")
    audit.finish()
    acceptance=dict(complete=True,smoke=args.smoke,configurations=len(configs),case_rows=len(cases),label_rows=len(labels),
                    label_summary_rows=len(summary),figure_pages=counts,source_hashes_unchanged=True,
                    original_arc_parity=True,no_new_evaluation=True,
                    output_sha256={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file()})
    dump(root/"ACCEPTED.json",acceptance)
    registry=base/"registry"/(args.release+".json")
    if registry.exists():
        raise ValueError("Registry snapshot already exists")
    dump(registry,dict(**metadata,release_directory=str(root),acceptance_sha256=sha(root/"ACCEPTED.json")))
    print(f"ACCEPTED: {root}",flush=True)


if __name__ == "__main__":
    main()
