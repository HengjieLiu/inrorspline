"""CPU-only first10 comparisons, component accounting, and acceptance audit."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np
from .report import dump

ARMS = ("reference","cpbe","combo_cache_batch_cpbe")
FEATURES = ([],["cpbe"],["cache","batch","cpbe"])


def stats(values):
    values = np.asarray(values,dtype=float)
    values = values[np.isfinite(values)]
    return dict(mean=float(values.mean()) if len(values) else None,
                sd=float(values.std(ddof=1)) if len(values)>1 else None,n=len(values))


def table(path,rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def field_error(a,b):
    av,bv = np.load(a,mmap_mode="r"),np.load(b,mmap_mode="r")
    total,maximum,count = 0.,0.,0
    location = None
    for z in range(av.shape[0]):
        delta = (av[z].astype(np.float64)-bv[z])*np.array([191/2,223/2,159/2])
        squared = np.sum(delta*delta,axis=-1)
        total += float(squared.sum())
        count += squared.size
        local = float(squared.max())
        if local >= maximum:
            maximum = local
            location = [z,*map(int,np.unravel_index(np.argmax(squared),squared.shape))]
    return dict(vector_rms_voxel=float(np.sqrt(total/count)),vector_max_voxel=float(np.sqrt(maximum)),
                max_location_zyx=location)


def categories(row):
    return [s.split(":",1)[0] for s in row["stop_reasons"]]


def parity(a,b):
    errors = field_error(a["dvf"],b["dvf"])
    result = dict(id=a["id"],reference_id=b["id"],same_steps=a["steps"]==b["steps"],
                  same_stop_categories=categories(a)==categories(b),**errors)
    result["passed"] = (result["same_steps"] and result["same_stop_categories"]
                        and errors["vector_rms_voxel"]<=.005 and errors["vector_max_voxel"]<=.05)
    return result


def smoke(root):
    rows = [r for r in json.loads((root/"primary_rows.json").read_text()) if r["block"].startswith("smoke_")]
    checks = []
    for arm in ARMS:
        h = next(r for r in rows if r["block"] == f"smoke_endpoints_{arm}")
        d = next(r for r in rows if r["block"] == f"smoke_components_{arm}")
        result = parity(d,h)
        timing = json.loads(Path(d["timing"]).read_text())
        comp = timing["components"]
        names = {r["name"] for r in comp["rows"]}
        required = {"forward","image_warp","ncc","be","backward","adam_step","zero_grad",
                    "bspline_field_expansion","warp_coordinates","scalar_transfers","early_stop_check"}
        result.update(variant=arm,phases_present=required<=names,
                      features_correct=h["features"]==d["features"]==FEATURES[ARMS.index(arm)],
                      steps_one_each=h["steps"]==d["steps"]==[1,1,1],
                      no_metric_phases=not any("evaluation" in x for x in names),counters=comp["counters"])
        result["passed"] &= result["phases_present"] and result["features_correct"] and result["steps_one_each"] and result["no_metric_phases"]
        if arm != "reference":
            result["factor_builds"] = sum(v for k,v in comp["counters"].items() if k.endswith("factor_build"))
            result["passed"] &= result["factor_builds"] == 27
        checks.append(result)
    payload = dict(complete=len(rows)==6 and all(c["passed"] for c in checks),checks=checks)
    dump(root/"smoke_validation.json",payload)
    return payload


def build(root,review):
    review.mkdir(parents=True,exist_ok=True)
    if (review/"report.md").exists():
        history = review/"history"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        history.mkdir(parents=True)
        for path in review.iterdir():
            if path.is_file():
                shutil.copy2(path,history/path.name)
    all_rows = json.loads((root/"primary_rows.json").read_text())
    headline = [r for r in all_rows if r["block"].startswith("headline_")]
    diagnostic = [r for r in all_rows if r["block"].startswith("diagnostic_")]
    evaluated = json.loads((root/"evaluation.json").read_text())
    metrics = {r["id"]:r for r in evaluated["rows"]}
    lookup = {(r["variant"],r["job_index"]):r for r in headline}
    pairs = [dict(variant=r["variant"],index=r["job_index"],rank=r["rank"],condition=r["condition"],
                  **parity(r,lookup[(r["variant"],r["job_index"])])) for r in diagnostic]
    summaries,timing_rows = {},[]
    for arm in ARMS:
        warm = [r for r in headline if r["variant"]==arm and r["condition"]=="warm"]
        cold = next(r for r in headline if r["variant"]==arm and r["condition"]=="cold")
        time = stats([r["elapsed_sec"] for r in warm])
        summaries[arm] = dict(cold_sec=cold["elapsed_sec"],warm_sec=time,gap_sec=cold["elapsed_sec"]-time["mean"],
                             peak_allocated_gib=max(r["peak_gib"] for r in warm+[cold]),
                             dice=stats([metrics[r["id"]]["mean_dice"] for r in warm]),
                             hd95_mm=stats([metrics[r["id"]]["surface_metrics"]["hd95_mean"] for r in warm]),
                             regularity={k:stats([metrics[r["id"]]["regularity"][k] for r in warm]) for k in
                                 ("log_jdet_std_fg","log_jdet_std_full","jdet_le0_pct_fg","jdet_le0_pct_full",
                                  "log_jdet_pos_frac_fg","log_jdet_pos_frac_full")})
    comparisons,label_rows = [],[]
    for arm,refarm in (("cpbe","reference"),("combo_cache_batch_cpbe","reference"),("combo_cache_batch_cpbe","cpbe")):
        for index in range(11):
            a,b = lookup[(arm,index)],lookup[(refarm,index)]
            am,bm = metrics[a["id"]],metrics[b["id"]]
            dice = np.asarray(am["dice"])-np.asarray(bm["dice"])
            hd = np.asarray(am["surface_metrics"]["hd95"])-np.asarray(bm["surface_metrics"]["hd95"])
            entry = dict(variant=arm,reference=refarm,index=index,rank=a["rank"],condition=a["condition"],
                         seconds_delta=a["elapsed_sec"]-b["elapsed_sec"],time_ratio=a["elapsed_sec"]/b["elapsed_sec"],
                         dice_delta=am["mean_dice"]-bm["mean_dice"],
                         hd95_delta_mm=am["surface_metrics"]["hd95_mean"]-bm["surface_metrics"]["hd95_mean"],
                         sdlogj_fg_delta=am["regularity"]["log_jdet_std_fg"]-bm["regularity"]["log_jdet_std_fg"],
                         folding_fg_percentage_point_delta=am["regularity"]["jdet_le0_pct_fg"]-bm["regularity"]["jdet_le0_pct_fg"],
                         reference_steps=b["steps"],candidate_steps=a["steps"],
                         same_label_nan_pattern=bool(np.array_equal(np.isnan(am["dice"]),np.isnan(bm["dice"]))),
                         **parity(a,b))
            comparisons.append(entry)
            for label,(dd,hh) in enumerate(zip(dice,hd),1):
                label_rows.append(dict(variant=arm,reference=refarm,condition=a["condition"],rank=a["rank"],label=label,
                                       dice_delta=float(dd),hd95_delta_mm=float(hh),
                                       reference_dice=bm["dice"][label-1],candidate_dice=am["dice"][label-1],
                                       reference_hd95_mm=bm["surface_metrics"]["hd95"][label-1],
                                       candidate_hd95_mm=am["surface_metrics"]["hd95"][label-1],
                                       fixed_voxels=am["label_voxel_counts"]["fixed"][label-1],
                                       reference_warped_voxels=bm["label_voxel_counts"]["warped"][label-1],
                                       candidate_warped_voxels=am["label_voxel_counts"]["warped"][label-1]))
    for r in headline:
        m = metrics[r["id"]]
        timing_rows.append({k:r.get(k) for k in ("id","variant","condition","rank","elapsed_sec","steps",
                                               "stop_reasons","peak_gib","setup_device_sec","dvf","timing")}
                          | dict(mean_dice=m["mean_dice"],hd95_mm=m["surface_metrics"]["hd95_mean"],**m["regularity"]))
    component_rows,case_components = [],{}
    for r in diagnostic:
        comp = json.loads(Path(r["timing"]).read_text())["components"]
        case_components[r["id"]] = comp
        for row in comp["rows"]:
            steps = r["steps"][row["stage"]] if row["stage"]>=0 else sum(r["steps"])
            niter = (1 if row["iteration_group"]=="first_iteration" else
                     steps-1 if row["iteration_group"]=="remaining_iterations" else None)
            component_rows.append(dict(id=r["id"],variant=r["variant"],condition=r["condition"],rank=r["rank"],
                stage_actual_steps=steps,iteration_count=niter,
                wall_ms_per_iteration=1000*row["wall_inclusive_sec"]/niter if niter else None,
                cuda_span_ms_per_iteration=1000*row["cuda_span_inclusive_sec"]/niter if niter else None,**row))
    grouped = defaultdict(list)
    for r in component_rows:
        key = (r["variant"],r["condition"],r["stage"],r["path"],r["iteration_group"],r["window"])
        grouped[key].append(r)
    component_summary = []
    for key,group in grouped.items():
        arm,condition,stage,path,iteration,window = key
        n = 1 if condition=="cold" else 10
        out = dict(variant=arm,condition=condition,stage=stage,path=path,iteration_group=iteration,window=window)
        for quantity in ("calls","wall_inclusive_sec","wall_exclusive_sec","cuda_span_inclusive_sec","cuda_span_exclusive_sec"):
            values = [r[quantity] for r in group]+[0.]*(n-len(group))
            s = stats(values)
            out[quantity+"_mean"],out[quantity+"_sd"] = s["mean"],s["sd"]
        out["wall_ms_per_call"] = 1000*sum(r["wall_inclusive_sec"] for r in group)/sum(r["calls"] for r in group)
        out["cuda_span_ms_per_call"] = 1000*sum(r["cuda_span_inclusive_sec"] for r in group)/sum(r["calls"] for r in group)
        for quantity in ("wall_ms_per_iteration","cuda_span_ms_per_iteration"):
            values = [r[quantity] for r in group if r[quantity] is not None]
            s = stats(values+[0.]*(n-len(group))) if values else dict(mean=None,sd=None)
            out[quantity+"_mean"],out[quantity+"_sd"] = s["mean"],s["sd"]
        component_summary.append(out)
    checks = []
    def check(name,passed,details=None):
        checks.append(dict(name=name,passed=bool(passed),details=details))
    check("33_headline_33_diagnostic",len(headline)==len(diagnostic)==33)
    check("33_evaluations",len(metrics)==33 and all(r["id"] in metrics for r in headline))
    check("unique_clean_primary",len({r["id"] for r in all_rows})==len(all_rows) and all(r["clean"] for r in all_rows))
    expected_ranks = [0,0,1,2,3,4,5,6,7,8,9]
    check("ordered_cold_warm_sequences",all([r["rank"] for r in group if r["variant"]==arm]==expected_ranks
                                          for group in (headline,diagnostic) for arm in ARMS))
    check("cold_and_warm_labels",all([r["condition"] for r in group if r["variant"]==arm]==["cold"]+["warm"]*10
                                     for group in (headline,diagnostic) for arm in ARMS))
    check("HD95_label_accounting",all(m["hd95_nan_pattern_matches_dice"] for m in metrics.values()))
    label_counts_ok = True
    for m in metrics.values():
        counts = m["label_voxel_counts"]
        for i,value in enumerate(m["dice"]):
            present = all(counts[k][i]>0 for k in ("fixed","moving","warped"))
            expected = 2.*counts["intersection"][i]/(counts["fixed"][i]+counts["warped"][i]+1e-5)
            label_counts_ok &= abs(value-expected)<1e-12 if present else bool(np.isnan(value))
    check("Dice_reconstructed_from_label_counts",label_counts_ok)
    check("evaluation_after_all_registration",evaluated["monitor"]["start_unix"] >=
          max(r["monitor"]["end_unix"] for r in headline+diagnostic))
    check("evaluation_backend_fixed",evaluated["backend"]["cudnn_benchmark"] is False
          and evaluated["backend"]["cudnn_deterministic"] is True)
    bad = []
    frozen = json.loads((root/"formal_source_sha256.json").read_text())
    for row in all_rows:
        f = np.load(row["dvf"],mmap_mode="r")
        timing = json.loads(Path(row["timing"]).read_text())
        summary = json.loads(Path(row["summary"]).read_text())
        marks = {m["name"]:m["offset_sec"] for m in timing["marks"]}
        correct = (f.shape==(160,224,192,3) and f.dtype==np.float32 and bool(np.isfinite(f).all())
                   and row["features"]==FEATURES[ARMS.index(row["variant"])]
                   and timing["cudnn_requested"]==dict(benchmark=False,deterministic=True)
                   and timing["cuda_backend"]["cudnn_benchmark"] is False and timing["cuda_backend"]["cudnn_deterministic"] is True
                   and abs(row["elapsed_sec"]-marks["dvf_file_ready"]+marks["case_load_complete"])<1e-8
                   and summary["dense_output_units"]=="normalized_grid_xyz_displacement"
                   and "dice" in summary["evaluation_skipped"])
        extra = row["backend_extra"]
        correct &= (extra["physical_gpu"]==0 and extra["cuda_visible_devices"]=="0"
                    and extra["num_threads"]==6 and extra["num_interop_threads"]==1
                    and extra["torch_seed"]==extra["numpy_seed"]==1
                    and extra["matmul_allow_tf32"] is False and extra["cudnn_allow_tf32"] is True
                    and extra["deterministic_algorithms"] is False and extra["float32_matmul_precision"]=="highest")
        if row in headline+diagnostic:
            hashes = json.loads((root/"workers"/row["worker"]/"source_sha256.json").read_text())
            correct &= all(frozen.get(k)==v for k,v in hashes.items())
            correct &= (root/"workers"/row["worker"]/"host_ack.json").exists()
        if not correct:
            bad.append(row["id"])
    check("DVF_flags_features_endpoints_frozen_sources",not bad,bad)
    check("smokes_passed",json.loads((root/"smoke_validation.json").read_text())["complete"])
    check("final_source_checks_passed",json.loads((root/"final_source_checks.json").read_text())["complete"])
    cache_ok = all(sum(v for k,v in case_components[r["id"]]["counters"].items() if k.endswith("factor_build"))
                   == (27 if r["condition"]=="cold" else 0) for r in diagnostic if r["variant"]!="reference")
    check("cold_factor_builds_and_warm_reuse",cache_ok)
    check("diagnostic_parity_checked",len(pairs)==33)
    acceptance = dict(complete=all(c["passed"] for c in checks),checks=checks,
                      diagnostic_parity_pass=all(r["passed"] for r in pairs),
                      note="Execution acceptance is separate from BE equivalence. Diagnostic parity mismatches remain flagged.")
    attempts = json.loads((root/"attempts.json").read_text())
    result = dict(summaries=summaries,comparisons=comparisons,diagnostic_parity=pairs,
                  acceptance=acceptance,contaminated_attempts=sum(not r["clean"] for r in attempts),
                  renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  BE_warning="Control-space BE is an exploratory, previously non-equivalent arm; no clinical noninferiority claim.")
    dump(review/"results.json",result)
    dump(review/"acceptance.json",acceptance)
    dump(review/"evaluations.json",evaluated)
    dump(review/"component_cases.json",case_components)
    for name,rows in (("timings_metrics",timing_rows),("paired_comparisons",comparisons),("label_differences",label_rows),
                      ("component_raw",component_rows),("component_summary",component_summary),("diagnostic_parity",pairs)):
        table(review/(name+".csv"),rows)
    lines = ["# MR-D-BSCP fast: control-space BE + cache/batch, GPU 0","",
             "Status: "+("complete and acceptance-checked" if acceptance["complete"] else "incomplete / failed acceptance"),"",
             "First10 seed3 OASIS diagnostic, alpha1000 seed1, original adaptive MR_A_fast_decay. All three arms use fresh GPU-0 workers under cuDNN benchmark=False/deterministic=True. No paper change.",
             "Headline sequence per arm: rank0 cold, rank0 warm, ranks1–9 warm. No extra warmup; cold does not imply cleared OS caches. Thirty-three headline and thirty-three separate diagnostic jobs; thirty-three headline metric records.",
             f"Foreign-GPU/query-error contaminated attempts: {result['contaminated_attempts']}. All attempts retained. Sampling cannot exclude shorter-than-poll jobs.","",
             f"Metric records: {len(metrics)}; fresh case/field computations: {sum(not m['reused_identical_dvf_metrics'] for m in metrics.values())}; identical case/DVF-hash reuse: {sum(m['reused_identical_dvf_metrics'] for m in metrics.values())}. Separate cold/warm IDs are retained."+
             (" The original evaluator continued unchanged under an overlapping long-job recovery monitor; details are in evaluation_watchdog_recovery/manifest.json." if (root/"evaluation_watchdog_recovery/manifest.json").exists() else ""),"",
             "## Headline endpoint timing and memory","",
             "case_load_complete → dvf_file_ready: optimization, final field, CPU conversion and completed NPY serialization; no fsync durability guarantee. Setup/loading and separate metrics are excluded. Peak is training PyTorch allocated memory, not driver reserved memory.","",
             "| Arm | Cold s (n=1) | Warm s mean ± SD (n=10) | Cold − warm s | Peak GiB |",
             "|---|---:|---:|---:|---:|"]
    for arm,s in summaries.items():
        lines.append(f"| {arm} | {s['cold_sec']:.3f} | {s['warm_sec']['mean']:.3f} ± {s['warm_sec']['sd']:.3f} | {s['gap_sec']:.3f} | {s['peak_allocated_gib']:.4f} |")
    lines += ["","| Candidate / reference | Cold Δ s (%) | Warm mean Δ s (%) | Gap Δ s (%) |","|---|---:|---:|---:|"]
    for arm,ref in (("cpbe","reference"),("combo_cache_batch_cpbe","reference"),("combo_cache_batch_cpbe","cpbe")):
        a,b = summaries[arm],summaries[ref]
        cells = []
        for av,bv in ((a["cold_sec"],b["cold_sec"]),(a["warm_sec"]["mean"],b["warm_sec"]["mean"]),(a["gap_sec"],b["gap_sec"])):
            cells.append(f"{av-bv:+.3f} ({100*(av-bv)/bv:+.2f}%)" if bv else f"{av-bv:+.3f} (undefined)")
        lines.append(f"| {arm} / {ref} | "+" | ".join(cells)+" |")
    lines += ["","Cold has one sample; warm SD is sample SD across cases, not repeated-run uncertainty. Arm order is fixed; interpret small incremental gains cautiously.","",
              "## Registration endpoints","",
              "Warm ten-case macro means ± sample SD. Dice/HD95: 35 labels, nearest label warp, existing absent-label NaNs, 1mm area-weighted symmetric robust surface HD95. SDlogJ: natural log, J>0 only, population SD within each case; voxel-unit float64 Jacobian, two-voxel border crop. Foreground = reoriented RAW fixed image >0, not segmentation.","",
              "| Arm | Dice | HD95 mm | FG SDlogJ | Full SDlogJ | FG folding % | Full folding % | FG J>0 fraction | Full J>0 fraction |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm,s in summaries.items():
        values = [s["dice"],s["hd95_mm"]]+[s["regularity"][k] for k in ("log_jdet_std_fg","log_jdet_std_full","jdet_le0_pct_fg","jdet_le0_pct_full","log_jdet_pos_frac_fg","log_jdet_pos_frac_full")]
        lines.append("| "+arm+" | "+" | ".join(f"{v['mean']:.6f} ± {v['sd']:.6f}" for v in values)+" |")
    lines += ["","CP-BE retains its previously failed strict numerical/trajectory equivalence warning. Small average Dice/HD95 changes do not establish equivalent fields or clinical noninferiority.","",
              "| Comparison | Mean Dice Δ | Mean HD95 Δ mm | Mean FG SDlogJ Δ | Max field RMS / pointwise vox | Changed-step warm ranks |",
              "|---|---:|---:|---:|---|---|"]
    outlier_notes = []
    for arm,ref in (("cpbe","reference"),("combo_cache_batch_cpbe","reference"),("combo_cache_batch_cpbe","cpbe")):
        group = [r for r in comparisons if r["variant"]==arm and r["reference"]==ref and r["condition"]=="warm"]
        lines.append(f"| {arm} / {ref} | {np.mean([r['dice_delta'] for r in group]):+.6f} | {np.mean([r['hd95_delta_mm'] for r in group]):+.6f} | "
                     f"{np.mean([r['sdlogj_fg_delta'] for r in group]):+.6f} | {max(r['vector_rms_voxel'] for r in group):.6f} / {max(r['vector_max_voxel'] for r in group):.6f} | {[r['rank'] for r in group if not r['same_steps']]} |")
        labels = [r for r in label_rows if r["variant"]==arm and r["reference"]==ref and r["condition"]=="warm"]
        dd = min((r for r in labels if np.isfinite(r["dice_delta"])),key=lambda r:r["dice_delta"])
        hh = max((r for r in labels if np.isfinite(r["hd95_delta_mm"])),key=lambda r:r["hd95_delta_mm"])
        if all(r["vector_max_voxel"]==0 for r in group):
            outlier_notes.append(f"{arm}/{ref}: identical fields and label metrics on all ten warm cases; no nonzero label outlier.")
        else:
            outlier_notes.append(f"{arm}/{ref}: worst label Dice decline {dd['dice_delta']:+.6f} (rank {dd['rank']}, label {dd['label']}; "
                f"{dd['reference_dice']:.6f} → {dd['candidate_dice']:.6f}, fixed {dd['fixed_voxels']} voxels, "
                f"warped {dd['reference_warped_voxels']} → {dd['candidate_warped_voxels']} voxels); largest HD95 worsening "
                f"{hh['hd95_delta_mm']:+.6f} mm (rank {hh['rank']}, label {hh['label']}).")
    for note in outlier_notes:
        lines += ["",note]
    lines += ["","## Instrumentation overhead and parity","",
              f"All diagnostic/headline field, step and stop-category parity checks passed: **{acceptance['diagnostic_parity_pass']}**. Thresholds: vector RMS .005 vox, max .05 vox, identical steps/structured stop categories. Diagnostic times NEVER replace headline times.","",
              "| Arm | Headline / diagnostic cold s | Headline / diagnostic warm mean s | Diagnostic warm overhead % | Setup.device cold s (headline) |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        ds = [r for r in diagnostic if r["variant"]==arm]
        dc = next(r for r in ds if r["condition"]=="cold")
        dw = np.mean([r["elapsed_sec"] for r in ds if r["condition"]=="warm"])
        hc = lookup[(arm,0)]
        hw = summaries[arm]["warm_sec"]["mean"]
        lines.append(f"| {arm} | {hc['elapsed_sec']:.3f} / {dc['elapsed_sec']:.3f} | {hw:.3f} / {dw:.3f} | {100*(dw/hw-1):+.2f} | {hc['setup_device_sec']:.3f} |")
    lines += ["","## Detailed component interpretation","",
              "All following times belong to the separately instrumented runs. Host times are dispatch/wait wall time; CUDA-event spans are stream elapsed time including possible host-fed idle gaps, NOT pure kernel busy time. They overlap and must not be summed. Inclusive parent forward/BE/expansion rows overlap their children. Field expansion IS B-spline expansion; no neural-network forward is present.",
              "A zero CUDA entry for CPU-only scopes (QR construction, scalar transfers, logging, stopping, optimizer creation or device setup) means no CUDA event pair was recorded there, not proof of zero associated GPU work. Consult the host columns and enclosing spans.",
              "component_raw.csv retains per-case, stage, hierarchical path, first/remaining iteration, call counts and inclusive/exclusive times. It and component_summary.csv include milliseconds per iteration (null for outside-iteration work); the summary also includes case mean/sample SD and milliseconds per call. Missing cold-only construction components count as zero in warm averages. First/remaining iteration rows allow per-iteration analysis without attributing stopping-related work reductions to faster kernels.","",
              "Component tables show aggregate occurrences by name, across nested paths. Parent/child rows are intentionally not additive."]
    names = ["setup.device","setup.case_load","control_initialization","loss_volume","gaussian_smoothing","downsample_interpolate","identity_grid","optimizer_creation",
             "forward","bspline_field_expansion","kernel_build","kernel_cache_hit","axis_z","axis_y","axis_x","layout_input","crop_layout_output",
             "warp_coordinates","image_warp","ncc","be","factor_basis_qr_cpu","factor_transfer","factor_lookup","control_differences",
             "projection_z","projection_y","projection_x","energy_reduction","energy_density_reduction","energy_mean_reduction","loss_assembly",
             "zero_grad","backward","adam_step","host_logging_and_checks","scalar_transfers","early_stop_check","instrumentation_stage_resolve",
             "method_execution.dvf_file_prep","method_execution.dvf_file_write"]
    names += [f"derivative_{t}" for t in ("200","020","002","110","101","011")]
    names += [f"stage_{i}.{s}" for i in range(3) for s in ("prolongation","control_summary_stats","final_output_dense_expand")]
    def per_case(arm,name,quantity,stage=None,iteration=None):
        result = []
        for row in diagnostic:
            if row["variant"]!=arm:
                continue
            values = [r for r in case_components[row["id"]]["rows"] if r["name"]==name
                      and (stage is None or r["stage"]==stage) and (iteration is None or r["iteration_group"]==iteration)]
            result.append((row,sum(r[quantity] for r in values)))
        return result
    for arm in ARMS:
        lines += ["",f"### {arm}: whole-run components","",
                  "| Component | Cold host s | Warm host s mean ± SD | Cold CUDA span s | Warm CUDA span s mean ± SD | Cold / warm mean calls |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name in names:
            host = per_case(arm,name,"wall_inclusive_sec")
            gpu = per_case(arm,name,"cuda_span_inclusive_sec")
            count = per_case(arm,name,"calls")
            if not any(v for _,v in count):
                continue
            h = stats([v for r,v in host if r["condition"]=="warm"])
            g = stats([v for r,v in gpu if r["condition"]=="warm"])
            lines.append(f"| {name} | {host[0][1]:.6f} | {h['mean']:.6f} ± {h['sd']:.6f} | {gpu[0][1]:.6f} | {g['mean']:.6f} ± {g['sd']:.6f} | {count[0][1]} / {np.mean([v for r,v in count if r['condition']=='warm']):.1f} |")
        lines += ["",f"### {arm}: per-stage iteration costs","",
                  "Warm means; ms/iteration uses actual iteration counts. Forward is inclusive. First-iteration and remaining-iteration costs are separate.","",
                  "| Stage | Component | Cold host s | Warm host s | Cold total CUDA s | Warm total CUDA s | Warm first CUDA ms/iteration | Warm remaining CUDA ms/iteration |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for stage in range(3):
            for name in ("forward","bspline_field_expansion","image_warp","ncc","be","backward","adam_step","host_logging_and_checks"):
                gpu = per_case(arm,name,"cuda_span_inclusive_sec",stage)
                host = per_case(arm,name,"wall_inclusive_sec",stage)
                per_iteration = []
                for iteration in ("first_iteration","remaining_iterations"):
                    times = per_case(arm,name,"cuda_span_inclusive_sec",stage,iteration)
                    counts = per_case(arm,name,"calls",stage,iteration)
                    values = [1000*t/n for (r,t),(_,n) in zip(times,counts) if r["condition"]=="warm" and n]
                    per_iteration.append(np.mean(values) if values else 0.)
                lines.append(f"| {stage} | {name} | {host[0][1]:.6f} | {np.mean([v for r,v in host if r['condition']=='warm']):.6f} | {gpu[0][1]:.6f} | {np.mean([v for r,v in gpu if r['condition']=='warm']):.6f} | {per_iteration[0]:.4f} | {per_iteration[1]:.4f} |")
        uncovered = [case_components[r["id"]]["host_uncovered_sec"] for r in diagnostic if r["variant"]==arm]
        lines += [f"\nHost time outside all recorded scopes within the headline interval: cold {uncovered[0]:.6f} s, warm mean {np.mean(uncovered[1:]):.6f} s. Within-parent unattributed time is retained separately as wall_exclusive_sec, not silently assigned to a kernel."]
    manifests = sorted(root.glob("provenance/*/manifest.json"))
    provenance = json.loads(manifests[-1].read_text())
    lines += ["","## Reproduction and evidence","",
              f"Git HEAD `{provenance['git_head']}`; dirty tree `{bool(provenance['git_status'])}`; image `{provenance['docker'][0]['Image']}`. Source hashes/copies, dirty patch, data hashes and exact commands retained. No commit created.",
              f"Report renderer SHA256 `{result['renderer_sha256']}`. Frozen registration-worker hashes remain in formal_source_sha256.json and each worker snapshot; later presentation/orchestration amendments do not retroactively change those measurements.",
              "Runtime/backend metadata are in each timing JSON and data_environment.json. Physical GPU0 is masked to logical CUDA0; seeds1; CPU6/1; matmul TF32=False, cuDNN TF32=True, deterministic_algorithms=False. F/T is not a universal bitwise-determinism guarantee.",
              "Replay: the source experiment codex_execution_spec.md documents run_be_combo_first10 --phase dry-run/preflight/all/report with explicit container/data/artifact arguments. Run roots never overwrite prior P2ac evidence.",
              "results.json and acceptance.json: summaries, field/step checks and completion gates. timings_metrics.csv: all headline cases including cold duplicate. paired_comparisons.csv and label_differences.csv: paired errors/outliers. evaluations.json: raw label/regularity metrics and field hashes. component_raw.csv/component_summary.csv/component_cases.json: full hierarchy, CPU/GPU intervals and cache counts. diagnostic_parity.csv: all 33 headline/profile comparisons.",""]
    (review/"report.md").write_text("\n".join(lines))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--review",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    a = p.parse_args()
    result = smoke(a.root) if a.smoke else build(a.root,a.review)
    print(json.dumps(result if a.smoke else {k:result[k] for k in ("summaries","acceptance","contaminated_attempts")},indent=2))
    if not (result["complete"] if a.smoke else result["acceptance"]["complete"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
