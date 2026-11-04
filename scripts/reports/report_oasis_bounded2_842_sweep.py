#!/usr/bin/env python
"""Accept the complete P76 sweep; preserve historical lambda1 and figure inputs."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/oasis_alpha_sweep"))
import run_oasis_bounded2_842_sweep as r

P75 = "oasis_seed3_first10_ncc9_method_family_arc_panels_v1"
METRICS = r.METRICS_TO_REPORT
DIAGNOSTICS = ("optimizer_sec", "worker_sec", "audit_sec", "actual_adam_updates", "peak_allocated_gib", "peak_reserved_gib")


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def summarize(rows, group, alpha):
    if sorted(int(x["case_rank"]) for x in rows) != list(range(10)):
        raise ValueError("Require ten distinct cases per accepted group")
    out = {"method": group, "alpha": alpha, "n": 10}
    for key in (*METRICS, *DIAGNOSTICS):
        if key not in rows[0]:
            continue
        values = [float(x[key]) for x in rows]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite reported metric")
        out.update({key + "_mean": statistics.mean(values), key + "_sd": statistics.stdev(values),
                    key + "_min": min(values), key + "_max": max(values)})
    return out


def require_matrix(rows):
    keys = [(int(x["case_rank"]), float(x["alpha"])) for x in rows]
    if len(keys) != 70 or len(set(keys)) != 70 or set(keys) != {(k, a) for a in r.LAMBDAS for k in range(10)}:
        raise ValueError("Require all 70 distinct registrations")


def verify_history(block):
    info = block["optimizer"]
    history = read_csv(Path(block["model_path"]).parent / "history.csv")
    if [int(x["epoch"]) for x in history] != list(range(info["actual_epochs"] + 1)):
        raise ValueError("History update sequence is incomplete")
    for x in history:
        if int(x["optimizer_updates_completed"]) != int(x["epoch"]) or float(x["lr"]) != info["stage"]["lr"]:
            raise ValueError("History optimizer schedule mismatch")
        data, reg, loss, weighted = [float(x[k]) for k in ("data_loss", "reg_loss", "loss", "weighted_reg_loss")]
        if not all(math.isfinite(v) for v in (data, reg, loss, weighted)):
            raise ValueError("Nonfinite history component")
        if not math.isclose(loss, data + block["alpha"] * reg, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("History total/data/regularizer weights disagree")
        if not math.isclose(weighted, block["alpha"] * reg, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("History weighted regularizer disagrees")
    best = min(history, key=lambda x: float(x["loss"]))
    if int(best["epoch"]) != info["best_epoch"]:
        raise ValueError("History checkpoint selection differs")
    return len(history)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def field_difference(a_path, b_path):
    a, b = np.load(a_path, mmap_mode="r"), np.load(b_path, mmap_mode="r")
    if a.shape != b.shape or a.shape != (160, 224, 192, 3):
        raise ValueError("Reproduction field geometry mismatch")
    total, maximum, count = 0., 0., 0
    for z in range(0, 160, 8):
        delta = (a[z:z+8].astype(np.float64) - b[z:z+8]) * np.array([95.5, 111.5, 79.5])
        total += float(np.square(delta).sum()); maximum = max(maximum, float(np.abs(delta).max())); count += delta.size
    return {"dvf_rms_voxel": math.sqrt(total/count), "dvf_max_abs_voxel": maximum}


def run(root):
    run_root, out = root / "runs" / r.VARIANT, root / "repo_experiments" / r.VARIANT
    if out.exists():
        raise FileExistsError("Preserve previous report release")
    manifest = r.c.read_json(run_root / "launch_manifest.json")
    r.assert_manifest(manifest)
    queue = r.c.read_json(run_root / "QUEUE_FINISHED.json")
    state = r.c.read_json(run_root / "production_state.json")
    if queue.get("complete") is not True or len(state["jobs"]) != 70 or any(j["status"] != "complete" for j in state["jobs"]):
        raise RuntimeError("Do not aggregate incomplete queue")
    candidates, cases, audits, hashes = [], [], [], {}
    for job in sorted(state["jobs"], key=lambda j: (j["alpha"], j["rank"])):
        path = Path(job["out"])
        row = r.validate(path, manifest, job["rank"], job["alpha"])
        audit = r.audit_ok(row)
        candidates.append(row); audits.append(audit)
        for name in ("summary.json", "independent_audit.json", "DONE", "final_metrics/case_metrics.csv"):
            hashes[str(path / name)] = r.sha(path / name)
        cases.append({"method": "Bounded 2 maps/level · 8→4→2", "alpha": row["alpha"],
                      "case_rank": row["case_rank"], "fixed_id": row["fixed_id"], "moving_id": row["moving_id"],
                      "direction": row["direction"], **{k: row[k] for k in METRICS},
                      "optimizer_sec": row["training_elapsed_sec"], "worker_sec": row["total_elapsed_sec"],
                      "audit_sec": audit["audit_elapsed_sec"], "actual_adam_updates": row["actual_adam_updates"],
                      "peak_allocated_gib": max(row["peak_cuda_gib"], audit["peak_cuda_gib"]),
                      "peak_reserved_gib": max(row["peak_reserved_gib"], audit["peak_reserved_gib"]),
                      "gpu": row["gpu"], "run_dir": row["run_dir"], "dvf_sha256": row["dvf_sha256"],
                      "chain_folding_percent": row["offgrid_audit"]["retained_composition"]["nonpositive_percent"],
                      "offgrid_export_folding_percent": row["offgrid_audit"]["interpolated_export"]["nonpositive_percent"]})
        print(f"Accepted replay/hashes: {len(cases)}/70", flush=True)
    require_matrix(cases)
    source = root / "repo_experiments" / P75
    accepted = r.c.read_json(source / "ACCEPTED.json")
    if not accepted["complete"]:
        raise ValueError("P75 must be accepted")
    for name, digest in accepted["output_sha256"].items():
        if r.sha(source / name) != digest:
            raise ValueError("P75 source drift")
        hashes[str(source / name)] = digest
    hashes[str(source / "ACCEPTED.json")] = r.sha(source / "ACCEPTED.json")
    references = []
    wanted = {("A", "sitreg"): "SITReg", ("B", "842"): "Displacement B-spline · 8→4→2",
              ("E", "842_velocity"): "SVF velocity · 8→4→2"}
    for x in read_csv(source / "per_case.csv"):
        if (x["panel"], x["series"]) in wanted and float(x["alpha"]) == 1.:
            references.append({"method": wanted[(x["panel"], x["series"])], "alpha": 1.,
                               "case_rank": int(x["case_rank"]), "fixed_id": x["fixed_id"], "moving_id": x["moving_id"],
                               "direction": x["direction"], **{k: float(x[k]) for k in METRICS}})
    if len(references) != 30:
        raise ValueError("Expected three matched reference groups")
    for row in [*cases, *references]:
        case = manifest["cases"][str(row["case_rank"])]
        if any(row[k] != case[k] for k in ("fixed_id", "moving_id")) or row["direction"] != "moving_to_fixed":
            raise ValueError("Report case identity/direction mismatch")
    summaries = [summarize([x for x in cases if x["alpha"] == a], "Bounded 2 maps/level · 8→4→2", a) for a in r.LAMBDAS]
    for name in wanted.values():
        summaries.append(summarize([x for x in references if x["method"] == name], name, 1.))
    paired = []
    for row in cases:
        for reference in [x for x in references if x["case_rank"] == row["case_rank"]] + [x for x in cases if x["case_rank"] == row["case_rank"] and x["alpha"] == 1.]:
            paired.append({"case_rank": row["case_rank"], "alpha": row["alpha"], "reference": reference["method"],
                           **{k + "_difference": row[k] - reference[k] for k in METRICS}})
    blocks, labels, reproduction = [], [], []
    checked_history_rows = 0
    for row in candidates:
        for block in row["blocks"]:
            checked_history_rows += verify_history(block)
            opt = block["optimizer"]
            blocks.append({"case_rank": row["case_rank"], "alpha": row["alpha"], "gpu": row["gpu"],
                           "step": block["step_index"], "cps": block["cps"], "map": block["map_index"],
                           **{k: opt[k] for k in ("actual_epochs", "best_epoch", "stopping_reason", "initial_loss", "best_loss", "best_data_loss", "best_reg_loss", "training_elapsed_sec", "peak_cuda_gib", "saturated_coefficient_fraction_gt_95pct", "max_abs_coefficient_ratio")},
                           "peak_reserved_gib": block["peak_reserved_gib"], "objective_improvement": block["objective_improvement"],
                           "contraction_bound_passed": block["atomic_corner_audit"]["bound_below_one"]})
        label_row = read_csv(Path(row["run_dir"]) / "final_metrics/case_metrics.csv")[0]
        for label in r.c.OASIS_LABELS:
            values = {name: float(label_row[f"{name}_label_{label:02d}"]) for name in
                      ("dice", "hd95", "asd_warped_to_fixed", "asd_fixed_to_warped", "asd_symmetric")}
            labels.append({"case_rank": row["case_rank"], "alpha": row["alpha"], "label": label,
                           "included": label not in row["excluded_labels"], **{k: v if math.isfinite(v) else None for k, v in values.items()}})
        if row["alpha"] == 1.:
            old = next(x for x in manifest["historical_lambda1"] if x["case_rank"] == row["case_rank"])
            if r.sha(Path(old["dvf_path"])) != old["dvf_sha256"]:
                raise ValueError("Historical lambda1 field changed")
            reproduction.append({"case_rank": row["case_rank"], **{k + "_difference": row[k] - old[k] for k in METRICS},
                                 **field_difference(row["dvf_path"], old["dvf_path"]),
                                 "optimizer_updates_difference": row["actual_adam_updates"] - old["actual_adam_updates"],
                                 "old_run": old["run_dir"], "new_run": row["run_dir"]})
    if len(blocks) != 420 or len(reproduction) != 10 or len(labels) != 70*len(r.c.OASIS_LABELS) or len(paired) != 280:
        raise ValueError("Report cardinality failure")
    selected = max(summaries[:7], key=lambda x: (x["dice_mean"], -x["asd_mean"], -x["hd95_mean"]))
    out.mkdir(parents=True, exist_ok=False)
    tables = {"per_case.csv": cases, "references.csv": references, "summary.csv": summaries,
              "paired_differences.csv": paired, "map_diagnostics.csv": blocks, "per_label.csv": labels,
              "fresh_vs_historical_lambda1.csv": reproduction}
    for name, rows in tables.items():
        write_csv(out / name, rows)
    result = {"complete": True, "candidates": candidates, "per_case": cases, "references": references,
              "summary": summaries, "paired": paired, "maps": blocks, "per_label": labels,
              "reproduction": reproduction, "independent_audits": audits, "selected_dice_first_lambda": selected["alpha"],
              "source_hashes": hashes, "report_source_sha256": r.sha(Path(__file__)), "manifest": manifest,
              "verified_history_rows": checked_history_rows}
    r.c.write_json(out / "results.json", result)
    lines = ["# First10 bounded two-map 8→4→2 lambda sweep", "",
             "Seventy fresh identity registrations, six retained bounded maps per fit. The selected lambda applies to all three levels and both maps. Bounds, loss, LR and update budgets are unchanged from P70.", "",
             "Mean ± across-case sample SD (n=10, ddof=1). Canonical Dice/HD95/symmetric ASD; fixed raw-positive foreground, positive Jacobians for SD(log J), J≤0 for folding percent. Exploratory test-case tuning, not an unbiased generalization benchmark.", "",
             "| Method | λ | Dice ↑ | HD95 mm ↓ | ASD mm ↓ | FG SD(log J) | FG folding % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append("| " + row["method"] + f" | {row['alpha']:g} | " + " | ".join(f"{row[k+'_mean']:.6f} ± {row[k+'_sd']:.6f}" for k in METRICS) + " |")
    lines += ["", f"Dice-first lambda: {selected['alpha']:g}; ties broken by ASD then HD95. Accuracy and regularity tradeoffs remain separate.", "",
              "## Runtime and memory", "", "| λ | Optimizer s | Worker s | Replay s | Actual Adam updates | Peak allocated GiB | Peak reserved GiB |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries[:7]:
        lines.append(f"| {row['alpha']:g} | " + " | ".join(f"{row[k+'_mean']:.2f} ± {row[k+'_sd']:.2f} [{row[k+'_min']:.2f}, {row[k+'_max']:.2f}]" for k in DIAGNOSTICS) + " |")
    lines += ["", "Optimizer time sums all six maps. Worker time includes loading, metrics, topology and export, but not independent replay. Peak memory is the larger worker/replay peak; allocated/reserved values exclude other processes and some CUDA context memory. Existing workloads remain active, so timing is diagnostic, not a controlled speed benchmark.", "",
              "All 70 independent replays passed. Each map satisfies its contraction gate, but measured exported-field folding remains an outcome. The lambda1 reproduction appendix reports differences from P70 rather than requiring cross-run bitwise identity. The new sweep replaces the three-level bounded-two point only; it is not a four-level sweep.", "",
              "See CSV/JSON for per-case/per-label results, paired differences, map stopping/best updates, bound saturation, second-map objective gains and all histories/checkpoint paths."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    # Independent numerical read-back, separate arithmetic library and saved CSV.
    for row in read_csv(out / "summary.csv"):
        values = cases if row["method"].startswith("Bounded") else references
        cohort = [x for x in values if x["method"] == row["method"] and x["alpha"] == float(row["alpha"])]
        for key in METRICS:
            array = np.asarray([x[key] for x in cohort], dtype=np.float64)
            if len(array) != 10 or abs(array.mean()-float(row[key+"_mean"])) > 1e-12 or abs(array.std(ddof=1)-float(row[key+"_sd"])) > 1e-12:
                raise ValueError("Independent saved-table statistic mismatch")
    for path, digest in hashes.items():
        if r.sha(Path(path)) != digest:
            raise ValueError("Source changed during report")
    r.assert_manifest(manifest)
    r.atomic_json(out / "ACCEPTED.json", {"complete": True, "production_registrations": 70, "independent_replays": 70,
                  "smokes": 3, "lambda_groups": 7, "map_records": 420, "label_records": len(labels),
                  "verified_history_rows": checked_history_rows,
                  "statistic_checks": 100, "source_unchanged": True, "accepted_at": r.c.now_text(),
                  "output_sha256": {p.name: r.sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({"report": str(out), "accepted": True, "dice_first_lambda": selected["alpha"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/database/inrorspline"))
    run(parser.parse_args().artifact_root)
