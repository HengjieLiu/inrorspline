#!/usr/bin/env python
"""Combine accepted P48 results with complete canonical P49 SITReg sweeps."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
SOURCE_METRICS = ("dice_mean", "hd95_mean", "asd_symmetric_mean", "log_jdet_std_fg", "jdet_le0_pct_fg")
LAMBDAS = {"0.1": 0.1, "1over3": 1/3, "3": 3., "10": 10.}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize(rows):
    groups = list(dict.fromkeys(r["method"] for r in rows))
    result = []
    for group in groups:
        subset = [r for r in rows if r["method"] == group]
        if sorted(r["rank"] for r in subset) != list(range(100)):
            raise ValueError(f"Incomplete/duplicate group: {group}")
        item = {"method": group, "n": 100}
        for metric in METRICS:
            values = [float(r[metric]) for r in subset]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Nonfinite metric")
            item[metric + "_mean"] = statistics.mean(values)
            item[metric + "_std"] = statistics.stdev(values)
        result.append(item)
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    accepted = read(args.base / "ACCEPTED.json")
    if not accepted["technical_completion"]:
        raise ValueError("Unaccepted base cohort")
    inputs = {}

    def record(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Checksum mismatch: {path}")
        inputs[str(path)] = actual

    for name, expected in accepted["report_sha256"].items():
        record(args.base / name, expected)
    base = read(args.base / "results.json")
    pairs = {r["case_rank"]: (r["fixed_id"], r["moving_id"]) for r in base["case_metrics"] if r["representation"] == "VFA"}
    rows = []
    for row in base["case_metrics"]:
        method = (f"MR-D-BSCP λ={row['alpha']:g}" if row["representation"] == "842_dense"
                  else "SITReg λ=1 (original)" if row["representation"] == "SITReg" else "VFA (original)")
        rows.append(dict(method=method, rank=row["case_rank"], fixed_id=row["fixed_id"],
                         moving_id=row["moving_id"], **{m: row[m] for m in METRICS}))
    if not (args.sitreg / "COMPLETED").is_file():
        raise ValueError("SITReg evaluation queue incomplete")
    reference_common = {v for p, v in base["provenance"]["evaluator_hashes"].items() if Path(p).name == "common.py"}
    if len(reference_common) != 1:
        raise ValueError("Ambiguous base metric source")
    checkpoints = {}
    for key, alpha in LAMBDAS.items():
        root = args.sitreg / f"sitreg_lambda_{key}_seed42"
        if not (root / "COMPLETED").is_file() or (root / "FAILED.txt").exists():
            raise ValueError(f"Incomplete SITReg {key}")
        manifest, summary = read(root / "manifest.json"), read(root / "summary.json")
        for name in ("manifest.json", "summary.json", "per_case.json", "per_label_per_case.json"):
            record(root / name)
        expected = {"alpha": alpha, "direction": "moving_to_fixed", "foreground": "fixed_raw_image_positive",
                    "jacobian": "canonical_LUMIR_crop2_positive_J_log_std",
                    "asd_definition": "pooled_surfel_area_weighted_bidirectional_mean", "smoke": False}
        if any(manifest[k] != v for k, v in expected.items()):
            raise ValueError("SITReg metric protocol mismatch")
        if summary["status"] != "complete" or summary["case_count"] != 100 or summary["label_rows"] != 3500:
            raise ValueError("Incomplete SITReg summary")
        record(Path(manifest["pair_file"]), manifest["pair_file_sha256"])
        for path, expected_hash in manifest["source_sha256"].items():
            record(Path(path), expected_hash)
            if Path(path).name == "common.py" and expected_hash not in reference_common:
                raise ValueError("Canonical evaluator mismatch")
        record(Path(manifest["checkpoint"]), manifest["checkpoint_sha256"])
        checkpoints[key] = {k: manifest[k] for k in ("checkpoint", "checkpoint_sha256", "best_epoch", "best_val_dice", "backend")}
        cases, labels = read(root / "per_case.json"), read(root / "per_label_per_case.json")
        if sorted(r["rank"] for r in cases) != list(range(100)) or len(labels) != 3500:
            raise ValueError("Incomplete case/label coverage")
        for row in cases:
            rank = row["rank"]
            if (row["fixed_id"], row["moving_id"]) != pairs[rank] or row["direction"] != "moving_to_fixed":
                raise ValueError("Case identity mismatch")
            if row["alpha"] != alpha or row["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
                raise ValueError("Checkpoint/lambda mismatch")
            case_dir = root / "cases" / f"rank{rank:03d}_{row['fixed_id']}_{row['moving_id']}"
            if not (case_dir / "COMPLETED").is_file():
                raise ValueError("Missing case completion marker")
            details = read(case_dir / "metrics.json")
            if details["summary"] != row:
                raise ValueError("Case summary mismatch")
            record(case_dir / "metrics.json")
            record(Path(row["field_path"]), row["field_sha256"])
            subset = [r for r in labels if r["rank"] == rank]
            if sorted(r["label_id"] for r in subset) != list(range(1,36)) or subset != details["per_label"]:
                raise ValueError("Label rows mismatch")
            for source, label_metric in zip(SOURCE_METRICS[:3], ("dice", "hd95_mm", "asd_symmetric_mm")):
                valid = [r[label_metric] for r in subset if r["included_in_case_mean"]]
                if not math.isclose(statistics.mean(valid), row[source], abs_tol=1e-12):
                    raise ValueError("Label-to-case reconstruction failed")
            rows.append(dict(method=f"SITReg λ={'1/3' if key == '1over3' else key}", rank=rank,
                             fixed_id=row["fixed_id"], moving_id=row["moving_id"],
                             **{m: row[s] for m, s in zip(METRICS, SOURCE_METRICS)}))
        new = summarize(rows[-100:])[0]
        for metric, source in zip(METRICS, SOURCE_METRICS):
            previous = summary["metrics"][source]
            if previous["n"] != 100 or not math.isclose(new[metric+"_mean"], previous["mean"], abs_tol=1e-12) or not math.isclose(new[metric+"_std"], previous["sample_sd"], abs_tol=1e-12):
                raise ValueError("Stored/recomputed statistics mismatch")
        print(f"Validated SITReg lambda {key}: 100 cases, 3500 labels, field hashes", flush=True)
    means = summarize(rows)
    if len(means) != 10 or len(rows) != 1000:
        raise ValueError("Wrong combined matrix")
    # Preserve the source reports; never overwrite an existing combined export.
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / "mean_std.csv", means)
    write_csv(args.out / "per_case.csv", rows)
    write_json(args.out / "results.json", {"means": means, "per_case": rows,
                                           "input_sha256": inputs, "checkpoints": checkpoints,
                                           "script_sha256": sha(Path(__file__))})
    text = ["# First100 MR-D-BSCP and SITReg lambda sweeps", "",
            "Same seed-3 OASIS ranks0–99. Mean ± across-case sample SD (ddof=1).",
            "MR-D-BSCP: pyramid 8→4→2→dense, NCC9 plus diffusion. SITReg sweep: independent seed42 training, best-validation-Dice checkpoints.",
            "SITReg λ=1 is the historical original, not a newly retrained sweep point. Inference backend/preprocessing remains native per method; this is not a fully controlled lambda-only comparison against that historical point.",
            "Foreground is fixed raw-image-positive; SD(log J) uses positive Jacobians and folding reports J≤0 as percent. Accuracy uses canonical absent-label handling and pooled-area symmetric ASD.", "",
            "| Method | Dice ↑ | HD95 (mm) ↓ | ASD (mm) ↓ | FG SD(log J) ↓ | FG folding (%) ↓ |",
            "|---|---:|---:|---:|---:|---:|"]
    for row in means:
        text.append("| " + row["method"] + " | " + " | ".join(f"{row[m+'_mean']:.6f} ± {row[m+'_std']:.6f}" for m in METRICS) + " |")
    (args.out / "report.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    write_json(args.out / "ACCEPTED.json", {"complete": True, "groups": 10, "per_case_rows": 1000,
        "sitreg_label_rows_validated": 14000, "all_saved_sitreg_field_hashes_verified": True,
        "metric_recalculation": "Recomputed cohort statistics from validated canonical per-case metrics; no inference rerun needed",
        "output_sha256": {name: sha(args.out/name) for name in ("mean_std.csv", "per_case.csv", "results.json", "report.md")}})
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "sitreg", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    run(parser.parse_args())
