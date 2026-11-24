"""Audit the fixed rank0 combined-loss pilot and render complete metric/resource tables.

Run inside the experiment container. Inputs and output directory must be outside
the worktree. This posthoc review never modifies the accepted execution files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from scripts.experiments.oasis_regularizer_sweep.protocol import dump, read, sha256, stamp
from scripts.experiments.oasis_regularizer_sweep.report import csv_write

REPO = Path(__file__).resolve().parents[2]
GAMMAS = (0., 1e-4, 1e-3, 1e-2, 1e-1, 1.)


def table(lines, headers, rows):
    lines.extend(["", "| " + " | ".join(headers) + " |",
                  "| " + " | ".join("---" for _ in headers) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    lines.append("")


def f(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def review(root, parent, out):
    assert not out.resolve().is_relative_to(REPO), "Generated review must be external"
    out.mkdir(parents=True, exist_ok=True)
    accepted = read(root / "ACCEPTED.json")
    assert accepted["complete"] and not accepted["first10_extension"]
    assert read(parent / "report_status.json")["complete"]
    manifest = read(root / "manifest.json")
    for name, digest in manifest["source_hashes"].items():
        assert sha256(REPO / name) == digest, f"Frozen source changed: {name}"
    payloads = read(parent / "evaluation_payloads.json")
    primary = read(parent / "primary_rank0.json")
    assert len(payloads) == 28 and len(primary) == 12
    assert Counter(r["phase"] for r in payloads) == {"pilot": 20, "reference": 4, "smoke": 4}
    assert len(list(root.glob("blocks/*/*/attempt*/eval_ACCEPTED.json"))) == 14
    assert not list(root.rglob("*FAILED.json"))
    for backend in ("fd", "analytic"):
        assert sorted(r["gamma"] for r in primary if r["backend"] == backend) == list(GAMMAS)
    identity = payloads[0]["case"]["input_hashes"]
    history_states = 0
    for rec in payloads:
        case, metric = rec["case"], rec["metrics"]
        assert (case["case_rank"], case["fixed_id"], case["moving_id"]) == (0, "0333", "0115")
        assert case["input_hashes"] == metric["input_hashes"] == identity
        assert metric["dvf_sha256"] == case["dvf_sha256"] == sha256(Path(case["dvf_path"]))
        assert sha256(Path(case["model_path"])) == case["model_sha256"]
        assert len(metric["per_label"]) == 35 and all(v["eligible"] for v in metric["per_label"])
        assert metric["replay"]["field_error"]["max_voxel"] == 0
        if rec["phase"] != "reference":
            for stage in case["stages"]:
                history = Path(case["dvf_path"]).parent / f"stage{stage['index']}" / "history.jsonl"
                states = [json.loads(line) for line in history.read_text().splitlines()]
                assert len(states) == stage["actual_epochs"] + 1
                assert stage["best_epoch"] == min(states, key=lambda r: r["loss"])["epoch"]
                for state in states:
                    assert state["gamma"] == rec["gamma"]
                    assert math.isclose(state["loss"], state["data_loss"] + state["diffusion_loss"]
                                        + rec["gamma"] * state["kl_c1_loss"], rel_tol=1e-6, abs_tol=2e-7)
                history_states += len(states)
    paired = []
    for row in primary:
        records = [p for p in payloads if p["phase"] != "smoke" and
                   (p["backend"], p["gamma"]) == (row["backend"], row["gamma"])]
        cold = next(p for p in records if p["case"]["condition"] == "cold")
        warm = next(p for p in records if p["case"]["condition"] == "warm")
        paired.append(dict(backend=row["backend"], gamma=row["gamma"],
                           same_field=cold["case"]["dvf_sha256"] == warm["case"]["dvf_sha256"],
                           dice_delta=warm["metrics"]["dice"] - cold["metrics"]["dice"]))
    audit = dict(at=stamp(), complete=True, source_hashes_checked=len(manifest["source_hashes"]),
                 primary_settings=12, scored_observations=28, eligible_labels_per_observation=35,
                 component_history_states_checked=history_states, cold_warm_pairs=paired,
                 first10_extension=False, analysis_source_sha256=sha256(Path(__file__)))
    dump(out / "audit.json", audit)
    lines = ["# Rank0 diffusion plus analytic KL-C1: detailed review", "", f"Generated: {stamp()}", "",
             "Objective: NCC9 loss + diffusion (lambda=1) + gamma * analytic KL-C1. "
             "The first shuffled seed-3 test pair is fixed0333 / moving0115. "
             "Both diffusion backends were retained; KL is analytic in both, with epsilon0.05/kappa120. "
             "Gamma0 uses hash-audited historical P84 results. No first10 jobs were run.", "",
             "All12 warm endpoints and28 scored observations passed input/field/source checks; "
             "all35 labels remain eligible. Cold/warm are independently initialized repeated executions "
             "of one subject pair, not independent cases. Every stage selects its minimum total "
             "objective checkpoint, without labels. No population confidence interval is appropriate.", "",
             "## Interpretation", "",
             "Adding analytic KL-C1 provides a strong geometry control alongside diffusion. "
             "Its accuracy benefit is small or absent: FD diffusion's best Dice is at gamma1e-4 "
             "(+0.112 percentage points); analytic diffusion's best remains gamma0. "
             "The FD gamma1e-4 endpoint has slightly worse HD95 and essentially unchanged ASD. "
             "Thus it does not improve all accuracy metrics.", "",
             "At gamma1e-3, FD diffusion loses only0.028 percentage points of Dice, improves both "
             "surface metrics, and lowers NDV about30%, but analytic SDlogJ barely changes. "
             "At gamma1e-2, both backends lower NDV about78-79% and analytic folding about53-54%, "
             "for0.34-0.51 percentage points less Dice. Analytic SDlogJ decreases only about2.7%. "
             "At gamma0.1, analytic SDlogJ falls35-36%, folding falls about99%, and NDV falls98%, "
             "with1.11-1.25 percentage points less Dice and worse ASD. Gamma1 costs about8 Dice "
             "percentage points; its zero sampled central/analytic folds coexist with nonzero NDV. "
             "No measured folding is not a proof of invertibility.", "",
             "A later accuracy/geometry study could retain gamma0,1e-3,1e-2,0.1; "
             "gamma1e-4 can test whether the small FD Dice gain repeats. Intermediate0.003/0.03 "
             "would refine the transition. These are recommendations, not launched work. "
             "The rank0 Dice-selected weights are exploratory selections, not general optima.", "",
             "## Accuracy and matching-baseline changes", "",
             "Dice is the mean over35 eligible labels. HD95 and symmetric area-weighted ASD "
             "are in mm. NCC9 below is the minimized raw squared-local-NCC loss (lower is better), "
             "not a correlation coefficient. Percentage-point Dice deltas multiply the raw change by100."]
    table(lines, ["Diffusion", "Gamma", "Dice", "Delta Dice pp", "HD95 mm", "ASD mm", "NCC9 loss"],
          [[r["backend"], f(r["gamma"]), f(r["dice"]), f(100*r["delta_vs_gamma0_dice"]),
            f(r["hd95"]), f(r["asd"]), f(r["ncc9"])] for r in primary])
    lines += ["## Regularity: foreground and whole volume", "",
              "FG means fixed raw image >0. Both Jacobians use crop2; SD(logJ) uses only J>0 "
              "and population SD. Central differences differentiate the exported grid; analytic "
              "derivatives differentiate the represented spline at voxel centers. All failed "
              "determinants, coverage, low-J fractions and raw-J summaries appear below. "
              "NDV uses crop1 and the multi-stencil negative-volume convention; its normalized "
              "index is not a folded-voxel percentage. A smaller SD alone cannot establish validity."]
    for scope in ("fg", "full"):
        lines.append(f"### {scope.upper()} geometry")
        table(lines, ["Diffusion", "Gamma", "Central SDlogJ", "Analytic SDlogJ", "Central folds %", "Analytic folds %", "NDV index", "Raw NDV"],
              [[r["backend"], f(r["gamma"]), f(r[f"log_jdet_std_{scope}"]), f(r[f"analytic_jacobian_{scope}_std_logj"]),
                f(r[f"jdet_le0_pct_{scope}"]), f(100*r[f"analytic_jacobian_{scope}_fraction_le_zero"]),
                f(r[f"ndv_frac_{scope}"]), f(r[f"ndv_{scope}"])] for r in primary])
        for derivative in ("fd", "analytic"):
            lines.append(f"### {scope.upper()} / {derivative} determinant distribution")
            keys = ("count", "positive_count", "positive_fraction", "fraction_lt_epsilon", "mean", "median", "min", "p01", "p99", "kl_c1")
            table(lines, ["Diffusion", "Gamma"] + list(keys),
                  [[r["backend"], f(r["gamma"])] + [f(r[f"{derivative}_jacobian_{scope}_{k}"]) for k in keys] for r in primary])
    lines += ["## Diffusion and bending energies", "",
              "FD energies use valid forward stencils; analytic energies use spline derivatives. "
              "Voxel units and normalized-grid component units are separate. These posthoc "
              "FG means are not the whole-domain loss used in optimization."]
    for region in ("full", "fixed_fg"):
        for unit in ("voxel", "normalized_grid"):
            lines.append(f"### {region} / {unit}")
            keys = [f"{family}_{backend}_{region}_{unit}" for family in ("diffusion", "be")
                    for backend in ("fd_forward", "native_analytic")]
            table(lines, ["Diffusion", "Gamma", "FD diffusion", "Analytic diffusion", "FD BE", "Analytic BE"],
                  [[r["backend"], f(r["gamma"])] + [f(r[k]) for k in keys] for r in primary])
    lines += ["## Selected native-stage objective components", "",
              "These are whole-domain optimizer terms at the selected final-stage checkpoint, "
              "not the foreground geometry means above. Lambda_diffusion=1 throughout. "
              "Gamma0 bypasses KL, so its historical raw-KL component is unavailable; "
              "its weighted-KL contribution is exactly zero."]
    component_rows = []
    for row in primary:
        rec = next(p for p in payloads if p["phase"] != "smoke" and p["case"]["condition"] == "warm"
                   and (p["backend"], p["gamma"]) == (row["backend"], row["gamma"]))
        stage = rec["case"]["stages"][-1]
        component_rows.append([row["backend"], f(row["gamma"]), f(stage["best_loss"]), f(stage["best_data_loss"]),
                               f(stage.get("best_diffusion_loss", stage["best_reg_loss"])),
                               f(stage.get("best_kl_c1_loss")), f(stage.get("best_weighted_kl_c1_loss", 0.)),
                               stage["best_epoch"], stage["actual_epochs"]])
    table(lines, ["Diffusion", "Gamma", "Total", "Data", "Diffusion", "Raw KL", "Gamma * KL", "Selected epoch", "Actual updates"], component_rows)
    lines += ["## Cold and warm resources", "",
              "Registration excludes loading and independent scoring: case_load_complete to "
              "dvf_file_ready. Cold is first registration in a fresh worker; warm follows in the "
              "same process but restarts from identity. It does not guarantee a cold OS file cache. "
              "Worker startup is shared metadata, not charged twice; cold launch-to-DVF includes "
              "startup and load. Torch allocated/reserved and sampled per-process GPU memory "
              "are different measures. Host RSS is shown separately. Monitoring interval is200ms. "
              "Gamma0 time/memory are historical and GPUs differ across some settings; relative "
              "speed is observational, not an isolated loss-overhead benchmark."]
    production = [r for r in read(parent / "all_metrics.json") if r["phase"] != "smoke"]
    production.sort(key=lambda r: (r["backend"], r["gamma"], r["condition"]))
    table(lines, ["Diffusion", "Gamma", "Condition", "GPU", "Registration s", "Startup s", "Load s", "Launch-to-DVF s", "Updates", "Torch alloc/reserved GiB", "Process GPU GiB", "RSS GiB"],
          [[r["backend"], f(r["gamma"]), r["condition"], r["gpu"], f(r["registration_sec"]),
            f(r["worker_startup_sec"]), f(r["case_load_sec"]), f(r["launch_to_dvf_sec"]), r["updates"],
            f(r["peak_allocated"]/1024**3)+" / "+f(r["peak_reserved"]/1024**3),
            f(r["nvml_peak_mib"]/1024), f(r["host_rss_peak"]/1024**3)] for r in production])
    lines += ["Positive-gamma FD warm times are29.62-32.25s; analytic diffusion warm times "
              "are25.38-27.61s. Allocated peaks are2.154GiB (FD) and1.930GiB (analytic), while "
              "reserved peaks are2.479 and2.516GiB. Sharing analytic derivatives reduces work "
              "in the analytic-diffusion implementation. Update counts also differ, so endpoint "
              "runtime is not a pure per-step comparison. All production stages stopped by the "
              "configured plateau rule; none hit their update cap. Stage learning rates retain "
              "the P84 schedule0.03/0.01/0.003; there were no within-stage LR reductions.", "",
              "### Independent scoring resources", "",
              "Scoring workers are separate from registration. Identical cold/warm fields can "
              "share an explicitly identified metric evaluation; an alias's metric_seconds=0 "
              "does not mean scoring is free. The time column includes only unique rank0 "
              "evaluations. Sampled memory peaks cover the complete original scoring block: "
              "11 observations for historical cohort references,2 for fresh pilots. Historical "
              "peaks therefore cannot be attributed to rank0 alone; the scope is explicit below."]
    scoring = []
    for row in primary:
        recs = [r for r in payloads if r["phase"] != "smoke" and
                (r["backend"], r["gamma"]) == (row["backend"], row["gamma"])]
        block = Path(recs[0]["source"])
        tele = read(block / "eval_telemetry.json")
        scoring.append(dict(backend=row["backend"], gamma=row["gamma"],
                            source_kind=row["source_kind"], metric_seconds=sum(r["metrics"]["metric_seconds"] for r in recs),
                            memory_block_observations=len(read(block / "block.json")["cases"]),
                            **tele))
    table(lines, ["Diffusion", "Gamma", "Unique rank0 scoring s", "Memory block observations", "Process GPU GiB", "Host RSS GiB"],
          [[r["backend"], f(r["gamma"]), f(r["metric_seconds"]), r["memory_block_observations"],
            f(r["process_gpu_peak_mib"]/1024), f(r["host_rss_peak"]/1024**3)] for r in scoring])
    assert all(not r["contaminated"] for r in scoring)
    assert all(r["capped_stages"] == 0 for r in production)
    dump(out / "scoring_resources.json", scoring)
    csv_write(out / "scoring_resources.csv", scoring)
    lines += ["## Audit and complete files", "",
              f"Validated{history_states} per-step component states, all saved field/model hashes, "
              f"and{len(manifest['source_hashes'])} frozen source hashes. "
              f"Exact cold/warm field matches:{sum(r['same_field'] for r in paired)}/12. "
              "No observed registration/scoring failure or GPU contamination. "
              "Preflight passed19 existing checks,5 combined-loss tests, a historical baseline "
              "audit, and4 independently replayed native-volume smokes.", "",
              "The parent directory retains all_metrics.csv/json, original evaluation_payloads.json, "
              "per_label.csv/json, stage_components.csv/json and backend_deltas.csv/json. "
              "All35 labels, directed surface distances, all three stage loss components and "
              "time/memory scopes remain available. The run root retains histories, checkpoints, "
              "fields, monitor samples and source/input/backend receipts. "
              "See [summary report](../report.md) and [gamma curves](../gamma_accuracy_regularity.png).", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    review(args.root, args.review, args.out)
