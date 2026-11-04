#!/usr/bin/env python
"""Export the approved eleven lambda-one rows from the accepted P75 release."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
ORDER = (
    ("A", "sitreg", "SITReg", "—"),
    ("A", "vfa", "VFA", "—"),
    ("A", "fedda", "FeDDA-1MF", "—"),
    ("B", "842", "Displacement B-spline", "8→4→2"),
    ("B", "8421", "Displacement B-spline", "8→4→2→1"),
    ("C", "bounded_1", "Bounded, 1 map/level", "8→4→2"),
    ("D", "bounded_1", "Bounded, 1 map/level", "8→4→2→1"),
    ("C", "bounded_2_compose", "Bounded, 2 maps/level", "8→4→2"),
    ("D", "bounded_2_compose", "Bounded, 2 maps/level", "8→4→2→1"),
    ("E", "842_velocity", "SVF, velocity penalty", "8→4→2"),
    ("E", "8421_velocity", "SVF, velocity penalty", "8→4→2→1"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select(summary, cases, pairs):
    selected, selected_cases = [], []
    for index, (panel, series, method, schedule) in enumerate(ORDER, 1):
        matches = lambda r: r["panel"] == panel and r["series"] == series and float(r["alpha"]) == 1.
        saved = [r for r in summary if matches(r)]
        cohort = sorted((r for r in cases if matches(r)), key=lambda r: int(r["case_rank"]))
        if len(saved) != 1 or int(saved[0]["n"]) != 10 or [int(r["case_rank"]) for r in cohort] != list(range(10)):
            raise ValueError("Missing or duplicate lambda-one cohort")
        for row in cohort:
            if [row["fixed_id"], row["moving_id"]] != list(pairs[int(row["case_rank"])]):
                raise ValueError("Case identity mismatch")
            if row["direction"] != "moving_to_fixed":
                raise ValueError("Wrong registration direction")
        result = dict(order=index, method=method, schedule=schedule, alpha=1., n=10,
                      source_panel=panel, source_series=series,
                      stage_weights="training λ=1" if panel == "A" else "/".join(["1"] * len(schedule.split("→"))))
        for metric in METRICS:
            values = [float(r[metric]) for r in cohort]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite metric")
            for suffix, value in (("mean", statistics.mean(values)), ("sd", statistics.stdev(values))):
                if not math.isclose(value, float(saved[0][metric + "_" + suffix]), rel_tol=0., abs_tol=1e-12):
                    raise ValueError("Saved/recomputed statistic mismatch")
                result[metric + "_" + suffix] = value
        selected.append(result)
        selected_cases.extend({"table_order": index, **r} for r in cohort)
    return selected, selected_cases


def markdown(rows):
    lines = ["# First10 NCC9 comparison at λ=1", "",
             "Same seed-3 shuffled ranks0–9; mean ± across-case sample SD (n=10, ddof=1). All active optimization stages use λ=1; DL uses λ=1 training checkpoints.", "",
             "| Method | Schedule | Dice ↑ | HD95 (mm) ↓ | Symmetric ASD (mm) ↓ | FG SD(log J) | FG folding (%) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        values = [f"{r[m+'_mean']:.{6 if m == 'jdet_le0_pct_fg' else 4}f} ± {r[m+'_sd']:.{6 if m == 'jdet_le0_pct_fg' else 4}f}" for m in METRICS]
        lines.append("| " + " | ".join([r["method"], r["schedule"], *values]) + " |")
    lines += ["", "Foreground: fixed raw image >0. SD(log J) uses J>0; folding counts J≤0, as percent. Accuracy uses canonical absent-label handling and pooled-area symmetric ASD.", "",
              "Displacement rows use pyramid-loss prolonged B-splines, including genuine cps1 controls; no dense or unbounded-composition substitution. Bounded rows use matched P70 retained composition and all-one stage weights, not the earlier1151 or duplicate P69 runs. SVF rows refine one velocity spline and use velocity diffusion with the primary seven-step integration.", "",
              "DL λ=1 checkpoints are historical. Two-map methods have larger optimizer budgets. Equal nominal λ does not equate constraints across representations. This is exploratory first-ten method comparison, not a pure bounding ablation or an unbiased generalization estimate. Zero measured folding is not a continuous-invertibility guarantee.", ""]
    return "\n".join(lines)


def run(args):
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite an existing table release")
    accepted = json.loads((args.source / "ACCEPTED.json").read_text())
    if not accepted.get("complete"):
        raise ValueError("Source release is not accepted")
    hashes = {str(args.source / name): digest for name, digest in accepted["output_sha256"].items()}
    hashes[str(args.source / "ACCEPTED.json")] = sha(args.source / "ACCEPTED.json")
    for name, digest in hashes.items():
        if sha(name) != digest:
            raise ValueError(f"Source hash changed: {name}")
    manifest = json.loads((args.source / "manifest.json").read_text())
    rows, cases = select(read_csv(args.source / "summary.csv"), read_csv(args.source / "per_case.csv"), manifest["pairs"])
    text = markdown(rows)
    args.out.mkdir(parents=True, exist_ok=False)
    for filename, records in (("summary.csv", rows), ("per_case.csv", cases)):
        with (args.out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    payload = {"summary": rows, "per_case": cases, "source_sha256": hashes,
               "script_sha256": sha(__file__), "created_at": datetime.now(timezone.utc).isoformat(),
               "sample_sd_ddof": 1, "no_registration_or_metric_reevaluation": True}
    (args.out / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    (args.out / "table.md").write_text(text, encoding="utf-8")
    for name, digest in hashes.items():
        if sha(name) != digest:
            raise ValueError(f"Source changed during export: {name}")
    acceptance = {"complete": True, "rows": 11, "case_rows": 110, "recomputed_statistics": 110,
                  "source_files_unchanged": True, "accepted_at": datetime.now(timezone.utc).isoformat(),
                  "output_sha256": {p.name: sha(p) for p in sorted(args.out.iterdir()) if p.is_file()}}
    (args.out / "ACCEPTED.json").write_text(json.dumps(acceptance, indent=2)+"\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
