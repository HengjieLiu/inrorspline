"""Aggregate new OASIS100 CP-BE arms with historical analytic-BE anchors."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

from .run_suite import dump

NEW_VARIANTS = (
    "fixed_50_50_50_cached_batch",
    "fixed_50_50_50_amsgrad",
)
BASELINES = ("original_fast_analytic_be", "baseline_2500_analytic_be")
COMMON = (
    "dice", "hd95", "asd_symmetric", "folding_fg_pct", "sdlogj_fg",
    "be_fd_fg_voxel", "diffusion_fg",
)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean_sd(values):
    values = [float(value) for value in values if finite(value)]
    return (
        statistics.fmean(values) if values else math.nan,
        statistics.stdev(values) if len(values) > 1 else math.nan,
        len(values),
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def new_case_rows(root):
    result = []
    quality = json.loads((root / "quality_rows.json").read_text())
    for row in quality:
        surface, regularity = row["surface_metrics"], row["regularity"]
        result.append({
            "variant": row["variant"], "rank": int(row["rank"]),
            "source": "new_current_evaluator",
            "dice": row["mean_dice"], "hd95": surface["hd95_mean"],
            "asd_symmetric": surface["asd_symmetric_mean"],
            "asd_warped_to_fixed": surface["asd_warped_to_fixed_mean"],
            "ncc": row["ncc_full"],
            "folding_fg_pct": regularity["jdet_le0_pct_fg"],
            "folding_full_pct": regularity["jdet_le0_pct_full"],
            "sdlogj_fg": regularity["log_jdet_std_fg"],
            "sdlogj_full": regularity["log_jdet_std_full"],
            "positive_j_fg": regularity["log_jdet_pos_frac_fg"],
            "positive_j_full": regularity["log_jdet_pos_frac_full"],
            "be_fd_fg_voxel": regularity["be_fd_forward_fixed_fg_voxel"],
            "diffusion_fg": regularity["diffusion_l2_fg"],
            "valid_labels": row["valid_labels"],
            "_dice_labels": row["dice"], "_hd95_labels": surface["hd95"],
        })
    return result


def load_summaries(paths, predicate):
    chosen = {}
    for root in paths:
        for path in root.rglob("summary.json"):
            value = json.loads(path.read_text())
            if not predicate(value):
                continue
            rank = int(value["case_rank"])
            if rank in chosen:
                raise RuntimeError(f"Duplicate historical rank {rank}: {path} and {chosen[rank][0]}")
            chosen[rank] = (path, value)
    if sorted(chosen) != list(range(100)):
        missing = sorted(set(range(100)) - set(chosen))
        raise RuntimeError(f"Historical resolver did not produce ranks 0..99; missing={missing}")
    return [chosen[rank][1] for rank in range(100)]


def stage_sum(value):
    return sum(int(item) for item in value.get("stage_actual_epochs", []))


def historical_rows(artifact):
    runs = artifact / "runs"
    fast = load_summaries(
        [
            runs / "oasis_unified_acceleration_sweep_alpha1000_v1",
            runs / "oasis_selected_fast_first100_alpha1000_v1",
        ],
        lambda value: (
            value.get("runtime_candidate") == "MR_A_fast_decay"
            and float(value.get("alpha", -1)) == 1000
            and 0 <= int(value.get("case_rank", -1)) < 100
            and stage_sum(value) > 100
        ),
    )
    baseline = load_summaries(
        [
            runs / "oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_v1",
            runs / "oasis_seed3_ranks020_099_directcp842_lr1e3_alpha_sweep_v1",
        ],
        lambda value: (
            value.get("method") == "direct_cp_8_4_2_be_bspline_analytic_multistage"
            and float(value.get("alpha", -1)) == 1000
            and value.get("stage_epochs") == [400, 800, 1300]
            and value.get("stage_lr") == [0.001, 0.001, 0.001]
            and 0 <= int(value.get("case_rank", -1)) < 100
        ),
    )
    result = []
    for variant, values in zip(BASELINES, (fast, baseline)):
        for value in values:
            result.append({
                "variant": variant, "rank": int(value["case_rank"]),
                "source": "historical_saved_summary",
                "dice": value["dice_mean"], "hd95": value["hd95_mean"],
                "asd_symmetric": value["asd_symmetric_mean"],
                "asd_warped_to_fixed": value["asd_warped_to_fixed_mean"],
                "ncc": math.nan,
                "folding_fg_pct": value["jdet_le0_pct_fg"],
                "folding_full_pct": value["jdet_le0_pct_full"],
                "sdlogj_fg": value["log_jdet_std_fg"],
                "sdlogj_full": value["log_jdet_std_full"],
                "positive_j_fg": 1 - value["jdet_le0_pct_fg"] / 100,
                "positive_j_full": 1 - value["jdet_le0_pct_full"] / 100,
                "be_fd_fg_voxel": value["be_fd_forward_fixed_fg_voxel"],
                "diffusion_fg": value["diffusion_l2_fg"],
                "valid_labels": sum(
                    finite(value.get(f"dice_label_{label:02d}"))
                    for label in range(1, 36)
                ),
                "_dice_labels": [
                    value.get(f"dice_label_{label:02d}", math.nan)
                    for label in range(1, 36)
                ],
                "_hd95_labels": [
                    value.get(f"hd95_label_{label:02d}", math.nan)
                    for label in range(1, 36)
                ],
            })
    return result


def population(rows):
    output = []
    for scope, allowed in (("first100", range(100)), ("ranks10_99", range(10, 100))):
        ranks = set(allowed)
        for variant in NEW_VARIANTS + BASELINES:
            selected = [row for row in rows if row["variant"] == variant and row["rank"] in ranks]
            result = {"scope": scope, "variant": variant, "case_count": len(selected)}
            for metric in (
                "dice", "hd95", "asd_symmetric", "asd_warped_to_fixed", "ncc",
                "folding_fg_pct", "folding_full_pct", "sdlogj_fg", "sdlogj_full",
                "positive_j_fg", "positive_j_full", "be_fd_fg_voxel", "diffusion_fg",
            ):
                avg, sd, count = mean_sd(row[metric] for row in selected)
                result[metric + "_mean"] = avg
                result[metric + "_sd"] = sd
                result[metric + "_count"] = count
            output.append(result)
    return output


def bootstrap_ci(values, seed):
    values = [float(value) for value in values if finite(value)]
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(2000)
    )
    return samples[49], samples[1949]


def paired(rows):
    index = {(row["variant"], row["rank"]): row for row in rows}
    output = []
    seed = 20260924
    for candidate_name in NEW_VARIANTS:
        for reference in BASELINES:
            for scope, ranks in (("first100", range(100)), ("ranks10_99", range(10, 100))):
                for metric in COMMON:
                    values = [
                        index[(candidate_name, rank)][metric] - index[(reference, rank)][metric]
                        for rank in ranks
                        if finite(index[(candidate_name, rank)][metric])
                        and finite(index[(reference, rank)][metric])
                    ]
                    lo, hi = bootstrap_ci(values, seed)
                    seed += 1
                    output.append({
                        "scope": scope, "candidate": candidate_name,
                        "reference": reference, "metric": metric,
                        "paired_count": len(values),
                        "mean_difference": statistics.fmean(values),
                        "sd_difference": statistics.stdev(values),
                        "bootstrap95_low": lo, "bootstrap95_high": hi,
                        "min_difference": min(values), "max_difference": max(values),
                    })
    return output


def label_differences(rows):
    index = {(row["variant"], row["rank"]): row for row in rows}
    output = []
    for candidate_name in NEW_VARIANTS:
        for reference in BASELINES:
            for metric, key in (("dice", "_dice_labels"), ("hd95", "_hd95_labels")):
                for label in range(35):
                    values = []
                    for rank in range(100):
                        left = index[(candidate_name, rank)][key][label]
                        right = index[(reference, rank)][key][label]
                        if finite(left) and finite(right):
                            values.append(float(left) - float(right))
                    avg, sd, count = mean_sd(values)
                    output.append({
                        "candidate": candidate_name, "reference": reference,
                        "metric": metric, "label": label + 1, "paired_count": count,
                        "mean_difference": avg, "sd_difference": sd,
                        "min_difference": min(values) if values else math.nan,
                        "max_difference": max(values) if values else math.nan,
                    })
    return output


def timing_rows(root, artifact):
    primary = json.loads((root / "primary_rows.json").read_text())
    output = []
    for variant in NEW_VARIANTS:
        selected = [row for row in primary if row.get("variant") == variant]
        cold = [row["elapsed_sec"] for row in selected if row["condition"] == "cold"]
        warm_rows = [row for row in selected if row["condition"] == "warm"]
        warm = [row["elapsed_sec"] for row in warm_rows]
        output.append({
            "variant": variant, "source": "new_full100_same_gpu",
            "registration_gpu": selected[0]["backend_extra"]["physical_gpu"],
            "cold_count": len(cold), "cold_sec": cold[0],
            "warm_count": len(warm), "warm_mean_sec": statistics.fmean(warm),
            "warm_sd_sec": statistics.stdev(warm),
            "cold_minus_warm_sec": cold[0] - statistics.fmean(warm),
            "peak_gib_mean": statistics.fmean(row["peak_gib"] for row in selected),
            "steps_mean": statistics.fmean(sum(row["steps"]) for row in warm_rows),
        })
    timing_path = (
        artifact / "repo_experiments"
        / "2026-09-16_oasis_table1_base_fast_ft_first10_gpu3"
        / "timing_summary.csv"
    )
    with timing_path.open() as stream:
        historical = {
            row["candidate"]: row for row in csv.DictReader(stream)
            if row["candidate"] in {"MR_A_fast_decay", "MR_baseline_lr1x_full"}
        }
    for candidate_name, variant in (
        ("MR_A_fast_decay", "original_fast_analytic_be"),
        ("MR_baseline_lr1x_full", "baseline_2500_analytic_be"),
    ):
        row = historical[candidate_name]
        output.append({
            "variant": variant, "source": "historical_first10_gpu3_ft",
            "registration_gpu": row["gpu"],
            "cold_count": row["cold_count"],
            "cold_sec": row["cold_endpoint_sec"],
            "warm_count": row["warm_count"],
            "warm_mean_sec": row["warm_endpoint_mean_sec"],
            "warm_sd_sec": row["warm_endpoint_sd_sec"],
            "cold_minus_warm_sec": row["cold_minus_warm_gap_sec"],
            "peak_gib_mean": row["warm_training_peak_mem_gb_mean"],
            "steps_mean": row["warm_actual_epochs_mean"],
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    args.review.mkdir(parents=True, exist_ok=True)

    cases = new_case_rows(args.root) + historical_rows(args.artifact_root)
    summaries = population(cases)
    comparisons = paired(cases)
    labels = label_differences(cases)
    timings = timing_rows(args.root, args.artifact_root)

    public_cases = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in cases
    ]
    write_csv(args.review / "case_metrics.csv", public_cases)
    write_csv(args.review / "population_summary.csv", summaries)
    write_csv(args.review / "paired_differences.csv", comparisons)
    write_csv(args.review / "per_label_differences.csv", labels)
    write_csv(args.review / "timing_summary.csv", timings)

    full = {
        row["variant"]: row for row in summaries if row["scope"] == "first100"
    }
    lines = [
        "# OASIS-100 fixed-50 CP-BE evaluation",
        "",
        "Two new CP-BE registrations were run under one current timing protocol. "
        "Analytic-BE quality anchors are historical saved summaries and were not rerun. "
        "Their timing rows are controlled first10 GPU3 F/T references only, not matched "
        "full100 timing comparisons.",
        "",
        "## Full100 quality",
        "",
        "| Method | Dice | HD95 mm | ASD mm | FG SDlogJ | FG folding % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels_map = {
        NEW_VARIANTS[0]: "Fixed 50/50/50 + cache/batch",
        NEW_VARIANTS[1]: "Fixed 50/50/50 + AMSGrad",
        BASELINES[0]: "Original fast, analytic dense BE (historical)",
        BASELINES[1]: "2500-step analytic dense BE (historical)",
    }
    for variant in NEW_VARIANTS + BASELINES:
        row = full[variant]
        lines.append(
            f"| {labels_map[variant]} | {row['dice_mean']:.5f} ± {row['dice_sd']:.5f} "
            f"| {row['hd95_mean']:.4f} ± {row['hd95_sd']:.4f} "
            f"| {row['asd_symmetric_mean']:.4f} ± {row['asd_symmetric_sd']:.4f} "
            f"| {row['sdlogj_fg_mean']:.4f} ± {row['sdlogj_fg_sd']:.4f} "
            f"| {row['folding_fg_pct_mean']:.4f} ± {row['folding_fg_pct_sd']:.4f} |"
        )
    lines += [
        "",
        "## Runtime",
        "",
        "| Method | Source | Cold s | Warm s ± SD | N warm | Peak GiB | Steps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in timings:
        lines.append(
            f"| {labels_map[row['variant']]} | {row['source']} "
            f"| {float(row['cold_sec']):.3f} "
            f"| {float(row['warm_mean_sec']):.3f} ± {float(row['warm_sd_sec']):.3f} "
            f"| {row['warm_count']} | {float(row['peak_gib_mean']):.3f} "
            f"| {float(row['steps_mean']):.1f} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- The first ten cases selected the two CP-BE candidates; population_summary.csv "
        "therefore also reports ranks 10–99 separately.",
        "- CP-BE previously failed strict repeated-Adam numerical equivalence to dense "
        "analytic BE. Registration metrics quantify empirical behavior but do not erase "
        "that warning.",
        "- Historical analytic-BE quality uses the same dataset and metric definitions, "
        "but a different source/runtime generation. Runtime ratios against those rows "
        "are contextual rather than controlled.",
        "- Metric evaluation is offline and excluded from every registration timing.",
        "",
        "Machine-readable details: case_metrics.csv, population_summary.csv, "
        "paired_differences.csv, per_label_differences.csv, and timing_summary.csv.",
    ]
    (args.review / "report.md").write_text("\n".join(lines) + "\n")

    acceptance = {
        "new_timing_rows": len(json.loads((args.root / "primary_rows.json").read_text())),
        "new_quality_rows": len(json.loads((args.root / "quality_rows.json").read_text())),
        "historical_quality_rows": sum(row["source"] == "historical_saved_summary" for row in cases),
        "new_case_counts": {
            variant: sum(row["variant"] == variant for row in public_cases)
            for variant in NEW_VARIANTS
        },
        "baseline_case_counts": {
            variant: sum(row["variant"] == variant for row in public_cases)
            for variant in BASELINES
        },
    }
    acceptance["passed"] = (
        acceptance["new_timing_rows"] == 202
        and acceptance["new_quality_rows"] == 200
        and acceptance["historical_quality_rows"] == 200
        and all(value == 100 for value in acceptance["new_case_counts"].values())
        and all(value == 100 for value in acceptance["baseline_case_counts"].values())
    )
    dump(args.review / "acceptance.json", acceptance)
    dump(args.review / "results.json", {
        "population": summaries, "paired": comparisons,
        "timing": timings, "acceptance": acceptance,
    })
    if not acceptance["passed"]:
        raise RuntimeError(f"Acceptance failed: {acceptance}")


if __name__ == "__main__":
    main()
