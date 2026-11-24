"""Complete first10 combined-loss analysis from accepted immutable observations.

All generated files are external. Case-paired bootstrap intervals are exploratory;
the ten warm cases, rather than labels or cold repeats, are the sampling units.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

import numpy as np

from scripts.experiments.oasis_regularizer_sweep.protocol import read, sha256, stamp
from scripts.experiments.oasis_regularizer_sweep.report import flatten
from scripts.experiments.oasis_diffusion_kl_first10.protocol import BACKENDS, GAMMAS
from scripts.reports.analyze_oasis_regularizer_fd_analytic import clean, csv_write, dump, stat, paired

REPO = Path(__file__).resolve().parents[2]
SCIENCE = ("dice", "hd95", "asd", "ncc9", "log_jdet_std_fg", "jdet_le0_pct_fg",
           "log_jdet_pos_frac_fg", "analytic_jacobian_fg_std_logj",
           "analytic_jacobian_fg_fraction_le_zero", "ndv_frac_fg",
           "diffusion_native_analytic_fixed_fg_voxel", "be_native_analytic_fixed_fg_voxel",
           "diffusion_fd_forward_fixed_fg_voxel", "be_fd_forward_fixed_fg_voxel",
           "fd_jacobian_fg_kl_c1", "analytic_jacobian_fg_kl_c1")
PERFORMANCE = ("registration_sec", "peak_allocated", "peak_reserved", "nvml_peak_mib", "host_rss_peak", "updates", "loop_sec_per_update")


def f(value, precision=5):
    return "NA" if value is None else f"{value:.{precision}g}"


def table(lines, headers, rows):
    lines.extend(["", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    lines.append("")


def analyze(root, out, pilot=None):
    if out.resolve().is_relative_to(REPO):
        raise ValueError("Generated artifacts must be outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    accepted = read(root/"ACCEPTED.json")
    assert accepted["complete"] and accepted["primary_warm_cases"] == 120
    manifest = read(root/"manifest.json")
    source_checks = {name: sha256(REPO/name) == digest for name, digest in manifest["source_hashes"].items()}
    assert all(source_checks.values()), "Frozen execution source changed"
    assert not list(root.rglob("*FAILED*")) and not list(root.rglob("*FAILURE*"))
    pairs = re.findall(r"\((\d+)\s*,\s*(\d+)\)", (REPO/"configs/splits/oasis/pair_test_200_shuffled_seed000003.txt").read_text())[:10]
    payloads = []
    input_files = {str(root/"manifest.json"): sha256(root/"manifest.json"), str(root/"ACCEPTED.json"): sha256(root/"ACCEPTED.json")}
    blocks = []
    for backend in BACKENDS:
        path = root/"baselines"/f"{backend}.json"
        reference = read(path)
        input_files[str(path)] = sha256(path)
        blocks.append(dict(backend=backend, gamma=0., phase="reference", source=reference["source_block"],
                           source_kind="historical_gamma0", cases=reference["cases"]))
    for path in sorted(root.glob("blocks/*/*/attempt*/block.json")):
        for name in ("register_ACCEPTED.json", "eval_ACCEPTED.json", "EVALUATED.json"):
            assert read(path.parent/name)["complete"]
        block = read(path)
        job = block["job"]
        input_files[str(path)] = sha256(path)
        records = []
        for case in block["cases"]:
            metric_path = Path(case["dvf_path"]).parent/"evaluation.json"
            records.append(dict(case=case, metrics=read(metric_path)))
            input_files[str(metric_path)] = sha256(metric_path)
        blocks.append(dict(backend=job["arm"].removeprefix("diffusion_"), gamma=job["gamma"], phase=job["phase"],
                           source=str(path.parent), source_kind="fresh_combined", cases=records))
    assert len(blocks) == 16
    all_rows, labels, stages, resources = [], [], [], []
    input_by_rank = {}
    history_states = 0
    for block in blocks:
        meta = {key: block[key] for key in ("backend", "gamma", "phase", "source", "source_kind")}
        observations = []
        for item in block["cases"]:
            case, metric = item["case"], item["metrics"]
            rank = case["case_rank"]
            assert (case["fixed_id"], case["moving_id"]) == pairs[rank]
            assert case["input_hashes"] == metric["input_hashes"]
            if rank in input_by_rank:
                assert input_by_rank[rank] == case["input_hashes"]
            input_by_rank[rank] = case["input_hashes"]
            assert sha256(case["dvf_path"]) == case["dvf_sha256"] == metric["dvf_sha256"]
            assert sha256(case["model_path"]) == case["model_sha256"]
            assert metric["replay"]["field_error"]["max_voxel"] == 0
            assert case["backend"]["cudnn_benchmark"] is False and case["backend"]["cudnn_deterministic"] is True
            if block["phase"] != "reference":
                assert case["source_hashes"] == manifest["source_hashes"]
                assert case["lambda_diffusion"] == 1 and case["lambda_kl"] == block["gamma"]
                for stage in case["stages"]:
                    path = Path(case["dvf_path"]).parent/f"stage{stage['index']}"/"history.jsonl"
                    states = [json.loads(line) for line in path.read_text().splitlines()]
                    assert len(states) == stage["actual_epochs"]+1
                    assert stage["best_epoch"] == min(states, key=lambda s: s["loss"])["epoch"]
                    for state in states:
                        assert state["gamma"] == block["gamma"]
                        assert math.isclose(state["loss"], state["data_loss"]+state["diffusion_loss"]+
                                            block["gamma"]*state["kl_c1_loss"], rel_tol=1e-6, abs_tol=2e-7)
                    history_states += len(states)
            row = dict(meta, case_rank=rank, condition=case["condition"], fixed_id=case["fixed_id"], moving_id=case["moving_id"],
                       gpu=case["gpu"], lambda_diffusion=1., kl_backend="analytic", dvf_sha256=case["dvf_sha256"],
                       dice=metric["dice"], hd95=metric["hd95"], asd=metric["asd"], ncc9=metric["ncc9"],
                       eligible_labels=sum(v["eligible"] for v in metric["per_label"]),
                       **flatten(metric["regularity"]), **flatten(metric["fd_jacobian"], "fd_jacobian_"),
                       **flatten(metric["analytic_jacobian"], "analytic_jacobian_"))
            row.update({k: metric["accuracy"][k] for k in ("dense_image_loss", "asd_warped_to_fixed_mean", "asd_fixed_to_warped_mean")})
            row.update({k: case.get(k) for k in ("registration_sec", "worker_startup_sec", "launch_to_dvf_sec", "case_load_sec",
                "case_to_dvf_sec", "finalize_dvf_sec", "peak_allocated", "peak_reserved", "nvml_peak_mib", "host_rss_peak")})
            updates = sum(s["actual_epochs"] for s in case["stages"])
            row.update(updates=updates, selected_updates=sum(s["best_epoch"] for s in case["stages"]),
                       capped_stages=sum(s["stopping_reason"] == "max_epochs" for s in case["stages"]),
                       loop_sec=sum(s["optimizer_loop_sec"] for s in case["stages"]),
                       preparation_sec=sum(s["preparation_sec"] for s in case["stages"]))
            row["loop_sec_per_update"] = row["loop_sec"]/updates
            assert all(math.isfinite(row[k]) for k in SCIENCE)
            all_rows.append(row)
            observations.append(row)
            payloads.append(dict(meta, case=case, metrics=metric))
            for label in metric["per_label"]:
                labels.append(dict(meta, case_rank=rank, condition=case["condition"], **flatten(label),
                                   exclusion_reason=";".join(label["exclusion_reason"])))
            for stage in case["stages"]:
                stages.append(dict(meta, case_rank=rank, condition=case["condition"], **flatten(stage)))
        source = Path(block["source"])
        telemetry = read(source/"eval_telemetry.json")
        assert not telemetry["contaminated"] and not read(source/"register_telemetry.json")["contaminated"]
        cold = next(r for r in observations if r["condition"] == "cold")
        warm = [r for r in observations if r["condition"] == "warm"]
        resource = dict(meta, cold_registration_sec=cold["registration_sec"], cold_launch_to_dvf_sec=cold["launch_to_dvf_sec"],
                        cold_startup_sec=cold["worker_startup_sec"], scoring_seconds=sum(i["metrics"]["metric_seconds"] for i in block["cases"]),
                        scoring_unique_fields=sum(not i["metrics"].get("alias_of") for i in block["cases"]),
                        scoring_process_gpu_peak_mib=telemetry["process_gpu_peak_mib"], scoring_host_rss_peak=telemetry["host_rss_peak"],
                        scoring_memory_observations=len(read(source/"block.json")["cases"]))
        for key in PERFORMANCE + ("case_load_sec", "case_to_dvf_sec", "finalize_dvf_sec", "preparation_sec", "loop_sec"):
            resource["cold_"+key] = cold[key]
            if warm:
                resource.update({"warm_"+key+"_"+k: v for k, v in stat([r[key] for r in warm]).items()})
        if warm:
            warm0 = next(r for r in warm if r["case_rank"] == 0)
            resource["rank0_cold_minus_warm_sec"] = cold["registration_sec"]-warm0["registration_sec"]
            resource["cold_warm_same_field"] = cold["dvf_sha256"] == warm0["dvf_sha256"]
        resources.append(resource)
    assert len(all_rows) == 136 and len(labels) == 136*35
    assert Counter(r["phase"] for r in all_rows) == {"cohort": 110, "reference": 22, "smoke": 4}
    primary = [r for r in all_rows if r["phase"] != "smoke" and r["condition"] == "warm"]
    primary.sort(key=lambda r: (r["backend"], r["gamma"], r["case_rank"]))
    groups = defaultdict(list)
    for row in primary:
        groups[row["backend"], row["gamma"]].append(row)
    assert len(groups) == 12 and len(primary) == 120
    summaries = []
    metadata = {"gamma", "case_rank", "gpu", "lambda_diffusion"}
    numeric = [key for key, value in primary[0].items() if (isinstance(value, (int, float)) or value is None) and key not in metadata]
    for (backend, gamma), rows in groups.items():
        assert [r["case_rank"] for r in rows] == list(range(10))
        summary = dict(backend=backend, gamma=gamma, n=10, eligible_labels_by_case=[r["eligible_labels"] for r in rows])
        for key in numeric:
            summary.update({key+"_"+k: v for k, v in stat([r.get(key) for r in rows]).items()})
        summaries.append(summary)
    bootstrap = np.random.default_rng(20261110).integers(0, 10, size=(20000, 10))
    comparisons, paired_cases, common_labels = [], [], []
    label_map = defaultdict(dict)
    for row in labels:
        if row["phase"] != "smoke" and row["condition"] == "warm":
            label_map[row["backend"], row["gamma"], row["case_rank"]][row["label"]] = row

    def compare(kind, a_key, b_key):
        a, b = groups[a_key], groups[b_key]
        meta = dict(kind=kind, a_backend=a_key[0], a_gamma=a_key[1], b_backend=b_key[0], b_gamma=b_key[1])
        for key in SCIENCE + PERFORMANCE:
            comparisons.append(dict(meta, metric=key, **paired(a, b, key, bootstrap)))
        for ar, br in zip(a, b):
            paired_cases.append(dict(meta, case_rank=ar["case_rank"], **{key: br[key]-ar[key] for key in SCIENCE+PERFORMANCE}))
            la, lb = label_map[(*a_key, ar["case_rank"])], label_map[(*b_key, br["case_rank"])]
            shared = [i for i in la if la[i]["eligible"] and lb[i]["eligible"]]
            common_labels.append(dict(meta, case_rank=ar["case_rank"], shared_labels=len(shared),
                **{metric+"_delta": float(np.mean([lb[i][metric]-la[i][metric] for i in shared])) for metric in ("dice", "hd95", "asd_symmetric")}))
    for backend in BACKENDS:
        for gamma in GAMMAS[1:]:
            compare("gamma_minus_baseline", (backend, 0.), (backend, gamma))
    for gamma in GAMMAS:
        compare("analytic_minus_fd", ("fd", gamma), ("analytic", gamma))
    selected = {}
    for backend in BACKENDS:
        rows = [r for r in summaries if r["backend"] == backend]
        winner = min(rows, key=lambda r: (-r["dice_mean"], r["asd_mean"], r["hd95_mean"], r["gamma"]))
        selected[backend] = dict(gamma=winner["gamma"], **{k: winner[k] for k in ("dice_mean", "dice_sd", "hd95_mean", "asd_mean")})
    for name, records in (("all_metrics", all_rows), ("primary_first10", primary), ("cohort_summary", summaries),
                          ("per_label", labels), ("stage_metrics", stages), ("setting_resources", resources),
                          ("paired_summary", comparisons), ("paired_cases", paired_cases), ("common_label_sensitivity", common_labels)):
        dump(out/(name+".json"), records)
        csv_write(out/(name+".csv"), records)
    dump(out/"evaluation_payloads.json", payloads)
    dump(out/"best_tested.json", selected)
    pilot_checks = []
    if pilot is not None:
        assert read(pilot/"ACCEPTED.json")["complete"]
        for path in pilot.glob("blocks/pilot/*/attempt*/block.json"):
            block = read(path)
            case = next(r for r in block["cases"] if r["condition"] == "warm")
            key = (block["job"]["arm"].removeprefix("diffusion_"), block["job"]["gamma"])
            row = groups[key][0]
            pilot_checks.append(dict(backend=key[0], gamma=key[1], same_field=case["dvf_sha256"] == row["dvf_sha256"]))
        assert len(pilot_checks) == 10 and all(r["same_field"] for r in pilot_checks)
    audit = dict(at=stamp(), complete=True, scored_observations=len(all_rows), primary_cases=len(primary), settings=len(groups),
                 labels=len(labels), eligible_label_counts=dict(Counter(r["eligible_labels"] for r in primary)),
                 component_history_states=history_states, source_checks=source_checks, inputs=input_files,
                 analysis_source_sha256=sha256(__file__), helper_source_sha256=sha256(REPO/"scripts/reports/analyze_oasis_regularizer_fd_analytic.py"),
                 bootstrap_seed=20261110, bootstrap_replicates=20000, delta_convention="B-A", paired_unit="case; n=10",
                 pilot_field_checks=pilot_checks, capped_primary_stages=sum(r["capped_stages"] for r in primary),
                 inference="exploratory test-set tuning; percentile case-bootstrap; no multiplicity correction")
    dump(out/"audit.json", audit)
    render(out, primary, summaries, comparisons, resources, audit)
    make_plots(out, groups, summaries)
    dump(out/"report_status.json", dict(complete=True, at=stamp(), primary_settings=12, primary_cases=120, scored_observations=136))
    print(json.dumps(dict(complete=True, selected=selected, eligible_labels=audit["eligible_label_counts"], capped_stages=audit["capped_primary_stages"])))


def render(out, primary, summaries, comparisons, resources, audit):
    summary = {(r["backend"], r["gamma"]): r for r in summaries}
    def ms(row, key, scale=1.):
        return f(row[key+"_mean"]*scale)+" ± "+f(row[key+"_sd"]*scale)
    lines = ["# OASIS first10: diffusion(lambda1) plus analytic KL-C1", "", f"Updated: {stamp()}", "",
        "Gamma0/1e-4/1e-3/1e-2/0.1/1; FD and analytic diffusion; analytic KL in both. "
        "NCC9, prolonged cubic cps8/4/2, pyramid4/2/1, epsilon0.05/kappa120 and P84/P85 optimization are unchanged. "
        "All120 primary endpoints are warm ranks0..9 after seed-3 shuffle; all136 observations include12 cold QA repeats and4 smokes.", "",
        "Gamma0 reuses accepted, source/input/field-audited P84 cohorts. Positive-gamma settings are fresh cohorts; "
        "their rank0 fits are rerun rather than spliced from the pilot. Means ± sample SD summarize10 case metrics. "
        "Paired percentile95% intervals resample the ten paired cases20000 times, seed20261110. "
        "These are exploratory test-set selections, without multiplicity correction or held-out confirmation.", "",
        "## Accuracy", "",
        "Dice is mean35-label overlap. HD95 and symmetric area-weighted ASD are mm; missing labels retain canonical eligibility rules. "
        "NCC9 denotes the minimized raw squared-local-NCC loss, not a correlation score."]
    table(lines, ["Diffusion", "Gamma", "Dice mean ± SD", "HD95 mm mean ± SD", "ASD mm mean ± SD", "NCC9 loss mean", "Eligible labels min-max"],
        [[r["backend"], f(r["gamma"]), ms(r,"dice"), ms(r,"hd95"), ms(r,"asd"), f(r["ncc9_mean"]),
          f"{min(r['eligible_labels_by_case'])}-{max(r['eligible_labels_by_case'])}"] for r in summaries])
    lines += ["## Foreground geometry", "",
        "Foreground is fixed raw image>0. Central and analytic derivatives describe the same saved field: "
        "central differences differentiate dense samples; analytic derivatives differentiate the cubic spline at voxel centers. "
        "Both Jacobians use crop2 and SD(logJ) uses only positive J, with within-volume population SD. "
        "NDV uses crop1 and a multi-stencil negative-volume index, not a folded-voxel percentage. "
        "Sampled zero folding does not establish invertibility."]
    table(lines, ["Diffusion", "Gamma", "Central SDlogJ", "Analytic SDlogJ", "Central folds %", "Analytic folds %", "NDV index"],
        [[r["backend"], f(r["gamma"]), ms(r,"log_jdet_std_fg"), ms(r,"analytic_jacobian_fg_std_logj"),
          ms(r,"jdet_le0_pct_fg"), ms(r,"analytic_jacobian_fg_fraction_le_zero",100), ms(r,"ndv_frac_fg")] for r in summaries])
    lines += ["## Foreground energies in voxel units", "",
        "These are common posthoc evaluations, independent of the active training diffusion backend. "
        "Full-volume values, normalized-grid units, Jacobian coverage/quantiles/KL and every other scalar appear in the complete appendix/CSV."]
    keys = ("diffusion_fd_forward_fixed_fg_voxel", "diffusion_native_analytic_fixed_fg_voxel", "be_fd_forward_fixed_fg_voxel", "be_native_analytic_fixed_fg_voxel")
    table(lines, ["Diffusion", "Gamma", "FD diffusion", "Analytic diffusion", "FD BE", "Analytic BE"],
          [[r["backend"], f(r["gamma"])] + [ms(r,k) for k in keys] for r in summaries])
    lines += ["## Changes against the matching gamma0 baseline", "",
        "All deltas are positive gamma minus gamma0. Dice is shown in percentage points; negative HD95/ASD favors the new setting. "
        "Wins/ties/losses refer to the ten paired Dice changes, tolerance1e-6 in raw Dice. "
        "The SDlogJ/NDV percentage changes below are ratios of cohort means, not mean per-case percentage changes."]
    rows = []
    for backend in BACKENDS:
        base = summary[backend,0.]
        for gamma in GAMMAS[1:]:
            r = summary[backend,gamma]
            p = next(p for p in comparisons if p["kind"] == "gamma_minus_baseline" and p["b_backend"] == backend and p["b_gamma"] == gamma and p["metric"] == "dice")
            rows.append([backend, f(gamma), f(100*p["mean_delta"]), f"[{f(100*p['ci95_low'])}, {f(100*p['ci95_high'])}]",
                f"{p['b_better']}/{p['ties']}/{p['b_worse']}", f(r["hd95_mean"]-base["hd95_mean"]), f(r["asd_mean"]-base["asd_mean"]),
                f(100*(r["analytic_jacobian_fg_std_logj_mean"]/base["analytic_jacobian_fg_std_logj_mean"]-1)),
                f(100*(r["ndv_frac_fg_mean"]/base["ndv_frac_fg_mean"]-1))])
    table(lines, ["Diffusion", "Gamma", "Delta Dice pp", "95% paired CI pp", "Dice W/T/L", "Delta HD95 mm", "Delta ASD mm", "Analytic SDlogJ change %", "NDV change %"], rows)
    lines += ["The complete paired_summary.csv/json also contains medians, case delta ranges, mean-case ratios, "
              "win counts and intervals for every primary geometry metric and resource measure. "
              "Per-case deltas and same-label sensitivity are separate files; label counts are not independent sample sizes.", "",
              "## Analytic-minus-FD diffusion at matching gamma", "",
              "KL is analytic in both rows; this comparison changes only the diffusion backend under the fixed numerical protocol. "
              "Negative geometry/surface deltas favor analytic diffusion; positive Dice favors analytic diffusion."]
    rows = []
    for gamma in GAMMAS:
        ps = {r["metric"]: r for r in comparisons if r["kind"] == "analytic_minus_fd" and r["b_gamma"] == gamma}
        d = ps["dice"]
        rows.append([f(gamma), f(100*d["mean_delta"]), f"[{f(100*d['ci95_low'])}, {f(100*d['ci95_high'])}]",
                     f(ps["hd95"]["mean_delta"]), f(ps["asd"]["mean_delta"]), f(ps["analytic_jacobian_fg_std_logj"]["mean_delta"]),
                     f(100*ps["analytic_jacobian_fg_fraction_le_zero"]["mean_delta"]), f(ps["ndv_frac_fg"]["mean_delta"])])
    table(lines, ["Gamma", "Delta Dice pp", "95% paired CI pp", "Delta HD95", "Delta ASD", "Delta analytic SDlogJ", "Delta analytic folds pp", "Delta NDV"], rows)
    lines += ["## Cold/warm registration resources", "",
        "Each positive-gamma worker runs cold rank0 then independently initialized warm0..9. "
        "Cold denotes a fresh process's first fit, not an uncached filesystem. Warm mean/SD covers ten different pairs; "
        "use rank0 cold-minus-warm for startup-related within-pair differences. Registration spans case_load_complete "
        "to dvf_file_ready, excluding loading and separate scoring. Startup/load/output/stage times are in the CSV. "
        "Torch allocated, reserved, sampled per-process GPU memory and host RSS are separate scopes; NVML/RSS sample every200ms. "
        "Gamma0 measurements are historical; GPUs and update counts can differ, so endpoint timing does not isolate loss-computation overhead."]
    production = [r for r in resources if r["phase"] != "smoke"]
    production.sort(key=lambda r: (r["backend"],r["gamma"]))
    rows = []
    for r in production:
        rows.append([r["backend"],f(r["gamma"]),f(r["cold_registration_sec"],4),
            f(r["warm_registration_sec_mean"],4)+" ± "+f(r["warm_registration_sec_sd"],3),
            f"{f(r['warm_registration_sec_min'],4)}–{f(r['warm_registration_sec_max'],4)}",f(r["rank0_cold_minus_warm_sec"],4),
            f(r["warm_updates_mean"],4),f(r["cold_peak_allocated"]/1024**3,4)+" / "+f(r["cold_peak_reserved"]/1024**3,4),
            f(r["warm_peak_allocated_max"]/1024**3,4)+" / "+f(r["warm_peak_reserved_max"]/1024**3,4),
            f(r["warm_nvml_peak_mib_max"]/1024,4),f(r["warm_host_rss_peak_max"]/1024**3,4)])
    table(lines, ["Diffusion","Gamma","Cold s","Warm s mean ± SD","Warm range s","Rank0 cold-warm s","Warm mean updates", "Cold alloc/res GiB", "Warm max alloc/res GiB","Warm process GPU max GiB","Warm RSS max GiB"],rows)
    lines += ["### Separate scoring", "",
        "Scoring times sum unique field evaluations; explicitly aliased identical cold/warm fields have zero additional metric time. "
        "Memory peaks cover complete11-observation cohort scoring workers and are not registration peaks."]
    table(lines,["Diffusion","Gamma","Unique scoring s","Unique fields","Scoring process GPU GiB","Scoring RSS GiB"],
        [[r["backend"],f(r["gamma"]),f(r["scoring_seconds"]),r["scoring_unique_fields"],
          f(r["scoring_process_gpu_peak_mib"]/1024),f(r["scoring_host_rss_peak"]/1024**3)] for r in production])
    lines += ["## Best tested settings and limits", ""]
    for backend in BACKENDS:
        rows = [r for r in summaries if r["backend"] == backend]
        best = min(rows, key=lambda r: (-r["dice_mean"], r["asd_mean"], r["hd95_mean"], r["gamma"]))
        base = summary[backend,0.]
        lines.append(f"- {backend}: maximum tested mean Dice at gamma={best['gamma']:g}: "
                     f"{best['dice_mean']:.6f} ± {best['dice_sd']:.6f}, delta={100*(best['dice_mean']-base['dice_mean']):+.3f}pp. "
                     f"HD95={best['hd95_mean']:.5f}mm; ASD={best['asd_mean']:.5f}mm.")
    lines += ["",f"Capped primary stages: {audit['capped_primary_stages']}; eligible-label counts across120 cases: {audit['eligible_label_counts']}. "
        f"Checked{audit['component_history_states']} history states and all saved input/source/field/model receipts. "
        f"All12 cold/warm rank0 pairs have identical fields: {all(r['cold_warm_same_field'] for r in production)}. "
        "Stage rates0.03/0.01/0.003 and the prior plateau stopping remain fixed; no within-stage LR reduction. "
        "Best mean Dice need not minimize surface distance, SDlogJ, folding, NDV or bending energy. "
        "No automatic gamma refinement or additional cases were run.", "",
        "## Files", "", "- primary_first10.csv/json: all120 warm endpoints; all_metrics.csv/json includes cold/smoke observations.",
        "- cohort_summary.csv/json and [complete scalar appendix](all_metric_summary.md): every metric's mean/sample SD/count/range.",
        "- paired_summary.csv/json, paired_cases.csv/json: matching-baseline and matching-gamma backend comparisons.",
        "- per_label.csv/json, common_label_sensitivity.csv/json: label eligibility and common-label paired accuracy.",
        "- setting_resources.csv/json, stage_metrics.csv/json: cold/warm registration and separate scoring resources, stage components/stopping.",
        "- evaluation_payloads.json: original nested metrics and case receipts; audit.json: hashes/bootstrap/provenance.",
        "- gamma_accuracy_regularity.png/pdf and accuracy_geometry_tradeoff.png/pdf: cohort curves and trade-offs.", ""]
    (out/"report.md").write_text("\n".join(lines))
    appendix = ["# Every scalar metric: first10 warm case summaries", "", "Within-setting SD is sample SD across10 cases. "
                "Raw fractions retain fraction units; only fields named pct are percentages. "
                "Positive-only SDlogJ excludes J<=0; read coverage and folds alongside it. "
                "Counts, raw J distributions and normalized-grid/voxel energy units retain their original keys."]
    keys = [key.removesuffix("_mean") for key in summaries[0] if key.endswith("_mean")]
    for key in keys:
        appendix.extend(["",f"## {key}"])
        table(appendix,["Diffusion","Gamma","Finite cases","Mean","Sample SD","Min","Max"],
            [[r["backend"],f(r["gamma"]),r[key+"_finite_n"],f(r[key+"_mean"],8),f(r[key+"_sd"],8),
              f(r[key+"_min"],8),f(r[key+"_max"],8)] for r in summaries])
    (out/"all_metric_summary.md").write_text("\n".join(appendix))


def make_plots(out, groups, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs = (("dice","Dice ↑",1),("hd95","HD95 mm ↓",1),("asd","ASD mm ↓",1),
             ("log_jdet_std_fg","Central SD(logJ) ↓",1),("analytic_jacobian_fg_std_logj","Analytic SD(logJ) ↓",1),
             ("analytic_jacobian_fg_fraction_le_zero","Analytic folds % ↓",100),("ndv_frac_fg","NDV index ↓",1),
             ("diffusion_native_analytic_fixed_fg_voxel","Analytic diffusion FG ↓",1),("be_native_analytic_fixed_fg_voxel","Analytic BE FG ↓",1))
    fig, axes = plt.subplots(3,3,figsize=(15,12))
    for backend in BACKENDS:
        rows = sorted([r for r in summaries if r["backend"] == backend],key=lambda r:r["gamma"])
        for ax,(key,label,scale) in zip(axes.flat,specs):
            ax.plot([r["gamma"] for r in rows],[scale*r[key+"_mean"] for r in rows],"o-",label=backend+" diffusion")
            ax.set_xscale("symlog",linthresh=1e-4)
            ax.set(xlabel="Gamma; diffusion lambda=1",ylabel=label)
    axes[0,0].legend()
    fig.suptitle("OASIS seed-3 first10: mean case metrics; analytic KL-C1 in both arms")
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(out/f"gamma_accuracy_regularity.{ext}",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for backend in BACKENDS:
        rows=sorted([r for r in summaries if r["backend"]==backend],key=lambda r:r["gamma"])
        for ax,(key,label) in zip(axes,(("analytic_jacobian_fg_std_logj","Analytic SD(logJ)"),
                                     ("analytic_jacobian_fg_fraction_le_zero","Analytic folds fraction"),("ndv_frac_fg","NDV index"))):
            ax.plot([r[key+"_mean"] for r in rows],[r["dice_mean"] for r in rows],"o-",label=backend+" diffusion")
            for r in rows:
                ax.annotate(f"{r['gamma']:g}",(r[key+"_mean"],r["dice_mean"]),fontsize=8,xytext=(4,3),textcoords="offset points")
            ax.set(xlabel=label+" ↓",ylabel="Mean Dice ↑")
    axes[0].legend()
    fig.suptitle("First10 accuracy–geometry trade-offs; labels are gamma")
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(out/f"accuracy_geometry_tradeoff.{ext}",dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--pilot-root",type=Path)
    args=p.parse_args()
    analyze(args.root,args.out,args.pilot_root)
