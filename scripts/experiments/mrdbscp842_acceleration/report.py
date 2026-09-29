"""Rebuild machine tables, preregistered decisions and readable report from artifacts.

Run in Docker (NumPy dependency); this does not initialize CUDA or execute registration.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import shutil
from pathlib import Path
import numpy as np


def dump(path, value):
    def clean(v):
        if isinstance(v,float) and not math.isfinite(v):
            return None
        if isinstance(v,dict):
            return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):
            return [clean(x) for x in v]
        return v
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False, default=str)+"\n")


def performance(rows, comparison, mode, phase="pilot"):
    rows = [r for r in rows if r.get("comparison") == comparison and r.get("mode") == mode
            and r["block"].startswith(phase+"_") and r["clean"]]
    cases = sorted(set(r["rank"] for r in rows))
    expected_cases, expected_repeats = ({0,7,9},3) if phase == "pilot" else (set(range(10)),5 if phase == "confirm" else 1)
    if set(cases) != expected_cases:
        return {"complete":False,"reason":"case coverage incomplete","observed_cases":cases}
    ratios, memory, details = [], [], []
    for rank in cases:
        ref = [r for r in rows if r["rank"] == rank and r["variant"] == "reference"]
        alt = [r for r in rows if r["rank"] == rank and r["variant"] != "reference"]
        if not ref or len(ref) != len(alt) or len(ref) != expected_repeats:
            return {"complete":False, "reason":"unpaired/incomplete repetitions"}
        t0, t1 = np.median([r["elapsed_sec"] for r in ref]), np.median([r["elapsed_sec"] for r in alt])
        m0, m1 = max(r["peak_gib"] for r in ref), max(r["peak_gib"] for r in alt)
        ratios.append(float(t1/t0))
        memory.append(float(m1/m0))
        details.append(dict(rank=rank, reference_median_sec=float(t0), candidate_median_sec=float(t1),
                            ratio=float(t1/t0), memory_ratio=float(m1/m0), repeats=len(ref)))
    if not ratios:
        return {"complete":False}
    rng = np.random.default_rng(20260919)
    sample = rng.choice(ratios, size=(10000,len(ratios)), replace=True)
    boot = np.median(sample, axis=1)
    ci = np.quantile(boot, [.025,.975]).tolist()
    median = float(np.median(ratios))
    speed = median <= .95 and ci[1] < 1 and np.mean(np.array(ratios) <= 1.02) >= .9
    memory_ok = max(memory) <= 1.10
    memory_only = max(memory) <= .8 and median <= 1.02
    return dict(complete=True, case_count=len(cases), primary_rows=len(rows), case_details=details,
                median_ratio=median, bootstrap95=ci, speed_gate=bool(speed and memory_ok),
                memory_only_gate=bool(memory_only), max_memory_ratio=max(memory),
                regression_cases=[d["rank"] for d in details if d["ratio"]>1.1])


def quality(row, reference, metrics):
    if row["id"] not in metrics or reference["id"] not in metrics:
        return {"pass":False,"reason":"evaluation missing"}
    a, b = metrics[row["id"]], metrics[reference["id"]]
    da, db = np.array(a["dice"]), np.array(b["dice"])
    nan_ok = bool(np.array_equal(np.isnan(da), np.isnan(db)))
    finite = np.isfinite(da) & np.isfinite(db)
    label_delta = float(np.max(np.abs(da[finite]-db[finite]))) if finite.any() else float("inf")
    mean_delta = abs(a["mean_dice"]-b["mean_dice"])
    jac_delta = a["nonpositive_jacobian_fraction"]-b["nonpositive_jacobian_fraction"]
    same_steps = row["steps"] == reference["steps"]
    def categories(r):
        return [s.split(":",1)[0] for s in (r.get("stop_reasons") or [])]
    same_stops = categories(row) == categories(reference)
    rms = a.get("dvf_vector_rms_voxel", float("inf"))
    maximum = a.get("dvf_vector_max_voxel", float("inf"))
    registration_metrics_pass = nan_ok and mean_delta <= 1e-4 and label_delta <= 1e-3 and jac_delta <= 1e-5
    dvf_pass = rms <= .005 and maximum <= .05
    stopping_pass = same_steps and same_stops
    passed = registration_metrics_pass and dvf_pass and stopping_pass
    result = dict(id=row["id"], reference_id=reference["id"], passed=bool(passed),
                registration_metrics_pass=bool(registration_metrics_pass),dvf_equivalence_pass=bool(dvf_pass),
                stopping_pass=bool(stopping_pass),
                same_steps=same_steps, same_stop_categories=same_stops, same_nan_pattern=nan_ok,
                mean_dice_abs_delta=mean_delta, max_label_dice_abs_delta=label_delta,
                nonpositive_jacobian_fraction_increase=jac_delta, dvf_vector_rms_voxel=rms,
                dvf_vector_max_voxel=maximum)
    result["mean_dice_delta"] = a["mean_dice"]-b["mean_dice"]
    if "surface_metrics" in a and "surface_metrics" in b:
        ha, hb = a["surface_metrics"], b["surface_metrics"]
        av, bv = np.array(ha["hd95"]), np.array(hb["hd95"])
        valid = np.isfinite(av) & np.isfinite(bv)
        result.update(hd95_mean_delta_mm=ha["hd95_mean"]-hb["hd95_mean"],
                      reference_hd95_mean_mm=hb["hd95_mean"], candidate_hd95_mean_mm=ha["hd95_mean"],
                      hd95_same_nan_pattern=bool(np.array_equal(np.isnan(av),np.isnan(bv))),
                      hd95_paired_label_count=int(valid.sum()),
                      hd95_label_deltas_mm=(av-bv).tolist())
    return result


def paired_references(rows):
    pairs = {}
    index = {(r.get("comparison"), r.get("mode"), r.get("repeat"), r["rank"],
              r["block"].split("_",1)[0]):r for r in rows if r["variant"] == "reference" and r.get("comparison")}
    aa = [r for r in rows if r["block"] == "reference_AA"]
    for row in aa[1:]:
        pairs[row["id"]] = aa[0]
    for row in rows:
        if row.get("comparison") and row["variant"] != "reference":
            key = (row["comparison"],row["mode"],row["repeat"],row["rank"],row["block"].split("_",1)[0])
            if key in index:
                pairs[row["id"]] = index[key]
    return pairs


def build(root, review):
    review.mkdir(parents=True, exist_ok=True)
    if (review/"report.md").exists():
        history = review/"history"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        history.mkdir(parents=True)
        for name in ("report.md","decisions.json","timings.csv","evaluations.json","acceptance.json"):
            if (review/name).exists():
                shutil.copy2(review/name,history/name)
    rows = json.loads((root/"primary_rows.json").read_text())
    attempts = json.loads((root/"attempts.json").read_text())
    checks = json.loads((root/"gpu_checks.json").read_text())
    metrics = {}
    for path in root.glob("evaluation_*.json"):
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            metrics[row["id"]] = row
    counts_by_hash = {(r["rank"],r["dvf_sha256"]):r for r in metrics.values() if "label_voxel_counts" in r}
    for row in metrics.values():
        key = (row["rank"],row["dvf_sha256"])
        if "label_voxel_counts" not in row and key in counts_by_hash:
            row["label_voxel_counts"] = counts_by_hash[key]["label_voxel_counts"]
            row["label_counts_reused_from_identical_field_id"] = counts_by_hash[key]["id"]
    pairs = paired_references(rows)
    gates = [quality(r, pairs[r["id"]], metrics) for r in rows if r["id"] in pairs]
    by_id = {g.get("id"):g for g in gates}
    aa = [r for r in rows if r["block"] == "reference_AA"]
    aaw = [r["elapsed_sec"] for r in aa if r["condition"] == "warm"]
    aaquality = all(by_id.get(r["id"],{}).get("passed",False) for r in aa[1:]) and len(aa) == 6
    robust_spread = float((np.quantile(aaw,.9)-np.quantile(aaw,.1))/np.median(aaw)) if aaw else None
    decisions = {}
    for variant in sorted(set(r.get("comparison") for r in rows if r.get("comparison"))):
        entry = {}
        for phase in ("pilot", "characterize", "confirm"):
            subset = [r for r in rows if r.get("comparison") == variant and r["variant"] != "reference"
                      and r["block"].startswith(phase+"_")]
            if not subset:
                continue
            perf = {m:performance(rows,variant,m,phase) for m in ("fixed","adaptive")}
            qpass = all(by_id.get(r["id"],{}).get("passed",False) for r in subset)
            metric_pass = all(by_id.get(r["id"],{}).get("registration_metrics_pass",False) for r in subset)
            trajectory_pass = all(by_id.get(r["id"],{}).get("dvf_equivalence_pass",False)
                                  and by_id.get(r["id"],{}).get("stopping_pass",False) for r in subset)
            numeric = ("features" in subset[0] and all(checks["passed"].get(f,False) for f in subset[0]["features"]))
            entry[phase] = dict(performance=perf, quality_pass=qpass, registration_metrics_pass=metric_pass,
                                dvf_and_stopping_pass=trajectory_pass,numerical_equivalence_pass=numeric,
                                reference_quality_stable=aaquality,
                                strict_pass=bool(numeric and qpass and aaquality and all(p.get("speed_gate",False) for p in perf.values())))
        decisions[variant] = entry
    be_summary = {}
    def mean_sd(values):
        return {"mean":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1)),"n":len(values)}
    for mode in ("fixed","adaptive"):
        candidates = [r for r in rows if r["block"].startswith("characterize_cpbe_") and r["variant"] == "cpbe" and r.get("mode") == mode]
        if len(candidates) != 10 or not all(r["id"] in metrics and pairs[r["id"]]["id"] in metrics for r in candidates):
            continue
        refs = [pairs[r["id"]] for r in candidates]
        cm = [metrics[r["id"]] for r in candidates]
        rm = [metrics[r["id"]] for r in refs]
        cg = [by_id[r["id"]] for r in candidates]
        be_summary[mode] = dict(reference_time_sec=mean_sd([r["elapsed_sec"] for r in refs]),
                               candidate_time_sec=mean_sd([r["elapsed_sec"] for r in candidates]),
                               reference_dice=mean_sd([r["mean_dice"] for r in rm]),
                               candidate_dice=mean_sd([r["mean_dice"] for r in cm]),
                               paired_dice_delta=mean_sd([g["mean_dice_delta"] for g in cg]),
                               reference_hd95_mm=mean_sd([r["surface_metrics"]["hd95_mean"] for r in rm]),
                               candidate_hd95_mm=mean_sd([r["surface_metrics"]["hd95_mean"] for r in cm]),
                               paired_hd95_delta_mm=mean_sd([g["hd95_mean_delta_mm"] for g in cg]),
                               maximum_case_dice_abs_delta=max(g["mean_dice_abs_delta"] for g in cg),
                               maximum_case_hd95_abs_delta_mm=max(abs(g["hd95_mean_delta_mm"]) for g in cg),
                               maximum_dvf_rms_voxel=max(g["dvf_vector_rms_voxel"] for g in cg),
                               maximum_dvf_vector_error_voxel=max(g["dvf_vector_max_voxel"] for g in cg),
                               reference_peak_gib=max(r["peak_gib"] for r in refs),
                               candidate_peak_gib=max(r["peak_gib"] for r in candidates),
                               reference_nonpositive_jacobian_fraction=mean_sd([r["nonpositive_jacobian_fraction"] for r in rm]),
                               candidate_nonpositive_jacobian_fraction=mean_sd([r["nonpositive_jacobian_fraction"] for r in cm]),
                               changed_steps=[dict(rank=a["rank"],reference=b["steps"],candidate=a["steps"])
                                              for a,b in zip(candidates,refs) if a["steps"] != b["steps"]])
        labels = []
        for row,a,b in zip(candidates,cm,rm):
            for label,(da,db,ha,hb) in enumerate(zip(a["dice"],b["dice"],a["surface_metrics"]["hd95"],b["surface_metrics"]["hd95"]),start=1):
                labels.append(dict(mode=mode,rank=row["rank"],label=label,dice_delta=da-db,hd95_delta_mm=ha-hb))
        be_summary[mode]["worst_label_dice"] = max((r for r in labels if np.isfinite(r["dice_delta"])),key=lambda r:abs(r["dice_delta"]))
        be_summary[mode]["worst_label_hd95"] = max((r for r in labels if np.isfinite(r["hd95_delta_mm"])),key=lambda r:abs(r["hd95_delta_mm"]))
        be_summary[mode]["worst_label_dice_decline"] = min((r for r in labels if np.isfinite(r["dice_delta"])),key=lambda r:r["dice_delta"])
        be_summary[mode]["worst_label_hd95_worsening"] = max((r for r in labels if np.isfinite(r["hd95_delta_mm"])),key=lambda r:r["hd95_delta_mm"])
    result = dict(protocol="OASIS seed3 first10 diagnostic, MR_A_fast_decay alpha1000 seed1; physical GPU0 F/T",
                  numerical_gates=checks["passed"], reference_AA_quality_pass=aaquality, BE_first10_summary=be_summary,
                  reference_AA_robust_10_90_spread=robust_spread, decisions=decisions,
                  quality_gates=gates, evaluation_rows=len(metrics), primary_rows=len(rows),
                  contaminated_attempts=sum(not a["clean"] for a in attempts))
    acceptance = json.loads((review/"acceptance.json").read_text()) if (review/"acceptance.json").exists() else {}
    result["study_complete"] = (acceptance.get("complete",False) and acceptance.get("primary_count") == len(rows)
                                and acceptance.get("evaluation_count") == len(metrics))
    dump(review/"decisions.json", result)
    columns = ["id","block","variant","comparison","mode","repeat","rank","condition","elapsed_sec","peak_gib","steps","setup_device_sec","clean","dvf","timing"]
    with (review/"timings.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    dump(review/"evaluations.json", list(metrics.values()))
    lines = ["# MR-D-BSCP fast 8-4-2 acceleration: GPU 0", "", "## Scope and decision", "",
             "Status: "+("completed and acceptance-checked." if result["study_complete"] else "in progress; this report contains completed evidence only."), "",
             "This is a first-10 OASIS diagnostic, not a publication OASIS-100 result. No paper files were changed.",
             "The reference is rerun on GPU 0 under benchmark=False, deterministic=True; GPU-3 historical times are not speedup denominators.", "",
             f"Recorded {len(rows)} clean primary rows and {len(metrics)} evaluated fields; {result['contaminated_attempts']} contaminated attempts retained.",
             f"Reference A/A quality gate: **{aaquality}**. Warm 10th–90th percentile spread / median: {robust_spread!r}.", "",
             *( [f"BE first10 adaptive result: mean runtime {be_summary['adaptive']['reference_time_sec']['mean']:.3f} → "
                 f"{be_summary['adaptive']['candidate_time_sec']['mean']:.3f} s; peak allocated memory "
                 f"{be_summary['adaptive']['reference_peak_gib']:.3f} → {be_summary['adaptive']['candidate_peak_gib']:.3f} GiB. "
                 f"Mean Dice changes by {be_summary['adaptive']['paired_dice_delta']['mean']:+.6f}; mean HD95 changes by "
                 f"{be_summary['adaptive']['paired_hd95_delta_mm']['mean']:+.6f} mm. These small observed label-metric changes "
                 "must be distinguished from failure of strict numerical/field equivalence and changed stopping in one case. "
                 "No clinical noninferiority claim follows from ten cases.", ""] if "adaptive" in be_summary else []),
             "## Candidate decisions", "",
             "| Candidate | Stage | Fixed median ratio (95% CI) | Adaptive median ratio (95% CI) | Dice/Jac gate | DVF/steps gate | Numerical gate | Strict speed promotion |",
             "|---|---|---|---|---|---|---|---|"]
    for variant, phases in decisions.items():
        for phase, d in phases.items():
            cells = []
            for mode in ("fixed","adaptive"):
                p = d["performance"][mode]
                cells.append(f"{p['median_ratio']:.4f} [{p['bootstrap95'][0]:.4f}, {p['bootstrap95'][1]:.4f}]" if p.get("complete") else "incomplete")
            lines.append(f"| {variant} | {phase} | {cells[0]} | {cells[1]} | {d['registration_metrics_pass']} | {d['dvf_and_stopping_pass']} | {d['numerical_equivalence_pass']} | {d['strict_pass']} |")
    selection_path = root/"selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        variant = selection["variant"]
        confirmation = decisions.get(variant,{}).get("confirm",{})
        lines += ["", f"Confirmation selection: `{variant}` with features `{selection['features']}`. "
                  +selection["rule"]]
        if confirmation:
            lines += [f"Confirmed strict replacement gate: **{confirmation['strict_pass']}**. "
                      "This decision is separate from the exploratory BE result."]
    lines += ["", "Ratios <1 are faster. Per-case repeat medians and a 10,000-draw case-cluster bootstrap are used; pilot intervals have only three cases.",
              "A speed gate alone does not establish method equivalence. Failed quality/step/numerical gates prohibit strict replacement.", "",
              "## Numerical findings", "", f"Operator gates: `{checks['passed']}`.",
              "The factored control-space BE has an FP64 sampled-field identity and avoids a subtractive quadratic form. It nevertheless fails strict FP32 trajectory compatibility near the regularizer nullspace under repeated Adam updates. At the user's request it is also tested in registration as a flagged exploratory arm; this numerical failure is not itself evidence of a meaningful Dice/HD95 loss.", "",
              "## Timing and memory interpretation", "",
              "Runtime spans case_load_complete through dvf_file_ready: optimization, final full-resolution expansion, CPU transfer and NPY write. Metrics are evaluated in a later, separate GPU process. Endpoint-only timing is used for paired comparisons; deep diagnostics and the short profiler trace are not headline samples.",
              "Cold uses a fresh worker/context. Every paired warm block has an excluded registration warmup. Peak memory is PyTorch peak allocated GiB, not total driver-reserved memory.",
              "The host verifies the exact worker PID and polls compute processes with interval boundary checks. Contaminated attempts remain in attempts.json; only clean attempts enter timings.csv. Sampling cannot rule out jobs shorter than the polling interval.", "",
              "## Reproducibility and evidence", "",
              "- `decisions.json`: numeric gates, paired ratios, confidence intervals, quality errors and decisions.",
              "- `timings.csv`: all primary rows, including labeled smoke/diagnostic/reference rows; filter block/mode for comparisons.",
              "- `evaluations.json`: label Dice, missing-label patterns, Jacobian and field differences. Identical-field metric reuse is SHA-256 keyed and explicit.",
              "- Run root `primary_rows.json`, `attempts.json`, `gpu_monitor.jsonl`, `workers/`, `jobs/`: raw observations, exact commands, logs, fields, monitoring and retries.",
              "- Run root `provenance/`: Git HEAD, dirty patch/status, source copies/hashes, Docker inspection, hardware and invocation for each phase.",
              "- `gpu_checks.json`, `microbenchmark.json`, and diagnostic/profile job directories: numerical and phase evidence.", "",
              "Rebuild this report with `python -B -m scripts.experiments.mrdbscp842_acceleration.report --root RUN_ROOT --review REVIEW_ROOT` inside the recorded Docker environment.",
              "Full replay instructions and scope limits are in the source-only experiment execution specification.", ""]
    characterized = [r for r in rows if r["block"].startswith("characterize_cpbe_") and r["variant"] == "cpbe"]
    if characterized:
        lines += ["## BE first10 registration endpoints", "",
                  "HD95 follows the existing repository convention: 35 labels, 1 mm isotropic spacing, symmetric area-weighted robust Hausdorff 95th percentile; missing labels are excluded consistently. Differences are candidate minus paired reference. No clinical/noninferiority margin is asserted.", "",
                  "Case-level macro averages (not voxel-pooled), with sample SD across ten cases:", "",
                  "| Mode | Backend | Time (s) | Dice | HD95 (mm) | Peak allocated GiB |",
                  "|---|---|---:|---:|---:|---:|"]
        for mode,s in be_summary.items():
            for backend,key in (("reference","reference"),("BE","candidate")):
                t,d,h = s[key+"_time_sec"],s[key+"_dice"],s[key+"_hd95_mm"]
                lines.append(f"| {mode} | {backend} | {t['mean']:.3f} ± {t['sample_sd']:.3f} | "
                             f"{d['mean']:.6f} ± {d['sample_sd']:.6f} | {h['mean']:.4f} ± {h['sample_sd']:.4f} | {s[key+'_peak_gib']:.4f} |")
        lines += ["", "Per-case paired results:", "",
                  "| Mode | Rank | Reference time (s) | BE time (s) | Ref / BE Dice | Dice Δ | Ref / BE HD95 (mm) | HD95 Δ (mm) | DVF RMS / max (vox) | Steps same |",
                  "|---|---:|---:|---:|---|---:|---|---:|---|---|"]
        for r in sorted(characterized,key=lambda v:(v["mode"],v["rank"])):
            g = by_id.get(r["id"],{})
            ref = pairs[r["id"]]
            a,b = metrics.get(r["id"],{}),metrics.get(ref["id"],{})
            if not g.get("id"):
                continue
            lines.append(f"| {r['mode']} | {r['rank']} | {ref['elapsed_sec']:.3f} | {r['elapsed_sec']:.3f} | "
                         f"{b.get('mean_dice',float('nan')):.6f} / {a.get('mean_dice',float('nan')):.6f} | "
                         f"{g['mean_dice_delta']:+.6f} | {g.get('reference_hd95_mean_mm',float('nan')):.4f} / "
                         f"{g.get('candidate_hd95_mean_mm',float('nan')):.4f} | {g.get('hd95_mean_delta_mm',float('nan')):+.4f} | "
                         f"{g['dvf_vector_rms_voxel']:.6f} / {g['dvf_vector_max_voxel']:.6f} | {g['same_steps']} |")
        lines += ["", "Per-label HD95 deltas and all quality gate failures are retained in decisions.json; metric values and field hashes are in evaluations.json.", ""]
        for mode,s in be_summary.items():
            d,h = s["worst_label_dice"],s["worst_label_hd95"]
            lines.append(f"{mode}: largest per-label absolute Dice change is {d['dice_delta']:+.6f} (rank {d['rank']}, label {d['label']}); "
                         f"largest per-label absolute HD95 change is {h['hd95_delta_mm']:+.4f} mm (rank {h['rank']}, label {h['label']}). "
                         f"Mean full-domain nonpositive-Jacobian fraction: {s['reference_nonpositive_jacobian_fraction']['mean']:.8f} → "
                         f"{s['candidate_nonpositive_jacobian_fraction']['mean']:.8f}.")
            hd = s["worst_label_hd95_worsening"]
            lines.append(f"The largest HD95 **worsening**, separately from the largest absolute change, is {hd['hd95_delta_mm']:+.4f} mm (rank {hd['rank']}, label {hd['label']}).")
        lines += ["", "Largest Dice decline, with structure-size context when measured:", "",
                  "| Mode | Rank | Label | Reference Dice | BE Dice | Fixed voxels | Reference warped / overlap | BE warped / overlap |",
                  "|---|---:|---:|---:|---:|---:|---|---|"]
        for mode,s in be_summary.items():
            outlier = s["worst_label_dice_decline"]
            row = next(r for r in characterized if r["mode"] == mode and r["rank"] == outlier["rank"])
            a,b = metrics[row["id"]],metrics[pairs[row["id"]]["id"]]
            label_index = outlier["label"]-1
            ac,bc = a.get("label_voxel_counts"),b.get("label_voxel_counts")
            fixed = ac["fixed"][label_index] if ac else "not recorded"
            av = f"{ac['warped'][label_index]} / {ac['intersection'][label_index]}" if ac else "not recorded"
            bv = f"{bc['warped'][label_index]} / {bc['intersection'][label_index]}" if bc else "not recorded"
            lines.append(f"| {mode} | {row['rank']} | {outlier['label']} | {b['dice'][label_index]:.6f} | "
                         f"{a['dice'][label_index]:.6f} | {fixed} | {bv} | {av} |")
        lines += ["", "Structure size can explain sensitivity of Dice to a few voxel changes; it does not waive the per-label quality gate.", ""]
    cold = [r for r in rows if r["block"].startswith("cold_")]
    if cold:
        lines += ["## Repeated cold and warm timing", "",
                  "| Backend | Cold mean ± sample SD (s) | Adaptive warm mean ± sample SD (s) | Cold − warm mean (s) | Max allocated GiB |",
                  "|---|---:|---:|---:|---:|"]
        for arm in sorted(set(r["variant"] for r in cold)):
            cv = [r["elapsed_sec"] for r in cold if r["variant"] == arm]
            warm = [r for r in rows if r["block"].startswith("confirm_") and r["variant"] == arm and r.get("mode") == "adaptive"]
            wv = [r["elapsed_sec"] for r in warm]
            if not wv:
                continue
            lines.append(f"| {arm} | {np.mean(cv):.3f} ± {np.std(cv,ddof=1):.3f} | {np.mean(wv):.3f} ± {np.std(wv,ddof=1):.3f} | "
                         f"{np.mean(cv)-np.mean(wv):.3f} | {max(r['peak_gib'] for r in warm):.4f} |")
        lines += ["", "Warm SD describes the 50 case/repetition observations, not an independent-subject inferential SE. Paired speed inference uses per-case repeat medians above.", ""]
    micro = json.loads((root/"microbenchmark.json").read_text())
    refs = {(tuple(r["shape"]),r["operator"]):r["median_sec"] for r in micro["rows"] if r["variant"] == "reference"}
    lines += ["## Isolated operator measurements", "",
              "These are forward + backward calls with fresh graphs: 10 warmups, 5 groups of 25 calls, median group time per call. They exclude registration, image warping and output writing; do not interpret their ratios as full-method speedups.", "",
              "| Loss shape | Control shape | Variant | Operator | ms/call | Candidate/reference |",
              "|---|---|---|---|---:|---:|"]
    for r in micro["rows"]:
        lines.append(f"| {tuple(r['shape'])} | {tuple(r.get('control_shape',[]))} | {r['variant']} | {r['operator']} | "
                     f"{1000*r['median_sec']:.3f} | {r['median_sec']/refs[(tuple(r['shape']),r['operator'])]:.4f} |")
    bridge = [r for r in rows if r["block"] == "reference_bridge" and r["condition"] == "warm"]
    if bridge:
        lines += ["", "## Instrumentation bridge", "",
                  "Same reference worker and rank 0, three warm repetitions per timing mode, alternating order. No control snapshots or profiler are enabled in this bridge.", "",
                  "| Mode | Warm mean ± sample SD (s) |", "|---|---:|"]
        for mode in ("endpoints","deep"):
            values = [r["elapsed_sec"] for r in bridge if r["command"].get("detail") == mode]
            lines.append(f"| {mode} | {np.mean(values):.3f} ± {np.std(values,ddof=1):.3f} |")
        lines += ["", "Endpoint-mode per-phase counters are host dispatch/wait measurements, not CUDA kernel durations. Use the separate deep CUDA-event diagnostics and profiler for phase attribution; do not add overlapping host/GPU intervals."]
    manifests = sorted(root.glob("provenance/*/manifest.json"))
    if manifests:
        manifest = json.loads(manifests[-1].read_text())
        backend = next((r["backend"] for r in rows if "backend" in r),{})
        lines += ["", "## Frozen execution environment", "",
                  f"Git HEAD: `{manifest['git_head']}`; dirty tree recorded: `{bool(manifest['git_status'])}`. No commit created.",
                  f"GPU: {backend.get('cuda_device_name')}; physical GPU 0 masked to logical CUDA 0. "
                  f"PyTorch {backend.get('torch_version')}, CUDA runtime {backend.get('cuda_runtime_version')}, cuDNN {backend.get('cudnn_version')}.",
                  "Requested and effective cuDNN flags: benchmark=False, deterministic=True. Matmul TF32=False; cuDNN TF32=True; deterministic_algorithms=False; float32 precision=highest; 6 intra-op / 1 inter-op CPU threads; Torch/NumPy seed 1.",
                  "These flags are not a universal proof of bitwise determinism. A/A and final-field comparisons provide the local empirical check.",
                  "Data/environment manifest records all first10 image/segmentation hashes, both pair-file hashes, NIfTI shape/spacing, case IDs and installed package versions. Source snapshots and per-worker hashes retain implementation changes made during preflight.", ""]
    region_path = review/"field_region_errors.json"
    if region_path.exists():
        regions = json.loads(region_path.read_text())
        lines += ["## Where do the BE field differences occur?", "",
                  "Diagnostic foreground is the raw fixed image >0, never segmentation >0. This localization does not replace the full-domain gate. Errors below are displacement-vector errors in full-resolution voxel units.", "",
                  "| Mode | Rank | Foreground RMS | Foreground p95 | Foreground max | Background RMS | Background max | Full max inside foreground? |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in regions["rows"]:
            f,b = r["foreground"],r["background"]
            lines.append(f"| {r['mode']} | {r['rank']} | {f['vector_rms_voxel']:.6f} | {f['p95_vector_error_voxel']:.6f} | "
                         f"{f['max_vector_error_voxel']:.6f} | {b['vector_rms_voxel']:.6f} | {b['max_vector_error_voxel']:.6f} | {r['max_is_fixed_foreground']} |")
        lines += ["", "`field_region_errors.json` also records p50/p99, voxel counts and the ZYX location of the full-volume maximum."]
    if (root/"real_control_checks.json").exists():
        real = json.loads((root/"real_control_checks.json").read_text())["rows"]
        lines += ["", "## Checks at actual optimization states", "",
                  "Fifteen saved reference controls: initial, first update, midpoint, 25 steps before stop, and final state in each of three stages. Full NCC + 1000×BE is differentiated. Adam sensitivity uses fresh moments at each snapshot, not the original historical optimizer moments.", "",
                  "| Variant | States | Max relative total-gradient L2 error | Max absolute 1000×BE difference | Max fresh-Adam control update difference (voxel-equivalent) |",
                  "|---|---:|---:|---:|---:|"]
        for variant in sorted({r["variant"] for r in real}):
            group = [r for r in real if r["variant"] == variant]
            lines.append(f"| {variant} | {len(group)} | {max(r['gradient_error']['relative_l2'] for r in group):.6g} | "
                         f"{max(r['weighted_be_abs_delta'] for r in group):.6g} | {max(r['fresh_adam_max_voxel_delta'] for r in group):.6g} |")
        lines += ["", "The final column is a control-coefficient difference scaled into full-resolution voxel-equivalent units, not a measured dense-DVF error. Buffering does not replace a loss operator; its history/stopping behavior is checked separately and its complete trajectories are tested in registration."]
    (review/"report.md").write_text("\n".join(lines))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    a = p.parse_args()
    result = build(a.root,a.review)
    print(json.dumps({"reference_AA_quality_pass":result["reference_AA_quality_pass"],
                      "decisions":result["decisions"]},indent=2))


if __name__ == "__main__":
    main()
