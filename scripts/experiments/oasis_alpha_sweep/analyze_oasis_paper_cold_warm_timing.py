#!/usr/bin/env python
"""Analyze OASIS paper-facing cold/warm DVF-only timing."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402
import run_oasis_paper_cold_warm_timing as runner  # noqa: E402


DEFAULT_ACCURACY_TABLE = (
    c.REPO_ROOT / "experiments/2026-08-13_oasis_selected_fast_first100/fast_vs_2500_baselines.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=runner.DEFAULT_OUT_ROOT)
    parser.add_argument("--review-root", type=Path, default=runner.DEFAULT_REVIEW_ROOT)
    parser.add_argument("--variant", type=str, default=runner.VARIANT)
    parser.add_argument("--accuracy-table", type=Path, default=DEFAULT_ACCURACY_TABLE)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def finite_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except Exception:
        return default


def timing_offsets(timing_path: Path) -> Dict[str, float]:
    out = {
        "case_load_complete_sec": float("nan"),
        "dvf_in_memory_ready_sec": float("nan"),
        "dvf_file_ready_sec": float("nan"),
        "script_total_sec": float("nan"),
    }
    if not Path(timing_path).exists():
        return out
    try:
        payload = c.read_json(Path(timing_path))
    except Exception:
        return out
    if not isinstance(payload, dict):
        return out
    out["script_total_sec"] = finite_float(payload.get("script_total_sec"))
    marks = payload.get("marks", [])
    if isinstance(marks, list):
        for mark in marks:
            if not isinstance(mark, dict):
                continue
            name = str(mark.get("name", ""))
            offset = finite_float(mark.get("offset_sec"))
            if name == "case_load_complete":
                out["case_load_complete_sec"] = offset
            elif name == "dvf_in_memory_ready":
                out["dvf_in_memory_ready_sec"] = offset
            elif name == "dvf_file_ready":
                out["dvf_file_ready_sec"] = offset
    return out


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.mean(vals)) if vals else float("nan")


def std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0 if len(vals) == 1 else float("nan")


def max_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return max(vals) if vals else float("nan")


def format_float(value: object, digits: int) -> str:
    val = finite_float(value)
    return "" if not math.isfinite(val) else f"{val:.{digits}f}"


def format_steps(value: object) -> str:
    val = finite_float(value)
    return "" if not math.isfinite(val) else f"{val:.0f}"


def read_progress_rows(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(Path(root).glob("worker_progress_*.csv")):
        if "smoke" in path.name:
            continue
        try:
            rows.extend(c.read_csv(path))
        except Exception:
            continue
    dedup: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = str(row.get("run_dir") or json.dumps(row, sort_keys=True))
        dedup[key] = row
    return list(dedup.values())


def read_gpu_rows(root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(Path(root).glob("gpu_observations_*.csv")):
        if "smoke" in path.name:
            continue
        try:
            rows.extend(c.read_csv(path))
        except Exception:
            continue
    return rows


def expected_jobs() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for phase_group, names in [("fast", runner.FAST_CANDIDATES), ("baseline", runner.BASELINE_CANDIDATES)]:
        for name in names:
            candidate = runner.selected_candidate(name)
            rows.append(
                {
                    "phase_group": phase_group,
                    "candidate": name,
                    "method": candidate.method,
                    "method_label": runner.paper_method_label(candidate.method_label),
                    "timing_condition": "cold",
                    "case_rank": 0,
                    "warm_sequence_index": "",
                }
            )
            for seq, rank in enumerate(range(10)):
                rows.append(
                    {
                        "phase_group": phase_group,
                        "candidate": name,
                        "method": candidate.method,
                        "method_label": runner.paper_method_label(candidate.method_label),
                        "timing_condition": "warm",
                        "case_rank": rank,
                        "warm_sequence_index": seq,
                    }
                )
    return rows


def job_key(row: Dict[str, object]) -> Tuple[str, str, str, int, str]:
    return (
        str(row.get("phase_group", "")),
        str(row.get("candidate", "")),
        str(row.get("timing_condition", "")),
        as_int(row.get("case_rank"), -1),
        str(row.get("warm_sequence_index", "")),
    )


def validated_case_rows(rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    out: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for row in rows:
        normalized = dict(row)
        offsets = timing_offsets(Path(str(row.get("timing_path", ""))))
        full_script_to_dvf_file_sec = finite_float(row.get("full_script_to_dvf_file_sec"))
        if not math.isfinite(full_script_to_dvf_file_sec):
            full_script_to_dvf_file_sec = offsets["dvf_file_ready_sec"]
        loaded_images_to_dvf_file_sec = finite_float(row.get("loaded_images_to_dvf_file_sec"))
        if not math.isfinite(loaded_images_to_dvf_file_sec) and math.isfinite(full_script_to_dvf_file_sec):
            case_load_complete = offsets["case_load_complete_sec"]
            if math.isfinite(case_load_complete):
                loaded_images_to_dvf_file_sec = full_script_to_dvf_file_sec - case_load_complete
        normalized["loaded_images_to_dvf_file_sec"] = loaded_images_to_dvf_file_sec
        normalized["full_script_to_dvf_file_sec"] = full_script_to_dvf_file_sec
        normalized["case_load_complete_sec"] = offsets["case_load_complete_sec"]
        normalized["dvf_in_memory_ready_sec"] = offsets["dvf_in_memory_ready_sec"]
        normalized["dvf_file_ready_sec"] = offsets["dvf_file_ready_sec"]
        normalized["actual_steps"] = row.get("actual_steps", "")
        checks = {
            "done": str(row.get("done", "")).lower() == "true",
            "summary_exists": str(row.get("summary_exists", "")).lower() == "true",
            "timing_json_exists": str(row.get("timing_json_exists", "")).lower() == "true",
            "dvf_file_exists": str(row.get("dvf_file_exists", "")).lower() == "true",
            "output_mode_dvf_only": row.get("output_mode") == "dvf_only",
            "no_evaluation_true": str(row.get("no_evaluation", "")).lower() == "true",
            "loaded_images_to_dvf_file_finite": math.isfinite(finite_float(row.get("loaded_images_to_dvf_file_sec"))),
            "full_script_to_dvf_file_finite": math.isfinite(finite_float(row.get("full_script_to_dvf_file_sec"))),
            "peak_mem_finite": math.isfinite(finite_float(row.get("training_peak_mem_gb"))),
        }
        normalized.update({f"check_{key}": value for key, value in checks.items()})
        normalized["validation_pass"] = all(checks.values()) and row.get("status") in {"completed", "skipped_existing"}
        out.append(normalized)
        if not bool(normalized["validation_pass"]):
            failures.append(normalized)
    return out, failures


def add_missing_failures(rows: Sequence[Dict[str, object]], failures: List[Dict[str, object]]) -> List[Dict[str, object]]:
    observed = {job_key(row) for row in rows}
    for expected in expected_jobs():
        if job_key(expected) not in observed:
            failure = dict(expected)
            failure.update({"status": "missing", "validation_pass": False, "error": "expected job not found"})
            failures.append(failure)
    return failures


def method_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for phase_group, names in [("fast", runner.FAST_CANDIDATES), ("baseline", runner.BASELINE_CANDIDATES)]:
        for name in names:
            candidate_rows = [
                row
                for row in rows
                if row.get("phase_group") == phase_group
                and row.get("candidate") == name
                and bool(row.get("validation_pass"))
            ]
            cold = [row for row in candidate_rows if row.get("timing_condition") == "cold"]
            warm = [row for row in candidate_rows if row.get("timing_condition") == "warm"]
            candidate = runner.selected_candidate(name)
            out.append(
                {
                    "phase_group": phase_group,
                    "candidate": name,
                    "method": candidate.method,
                    "method_label": runner.paper_method_label(candidate.method_label),
                    "family": candidate.family,
                    "template": candidate.template,
                    "cold_case_count": len(cold),
                    "warm_case_count": len(warm),
                    "cold_loaded_images_to_dvf_file_sec": mean(
                        finite_float(row.get("loaded_images_to_dvf_file_sec")) for row in cold
                    ),
                    "warm_loaded_images_to_dvf_file_sec_mean": mean(
                        finite_float(row.get("loaded_images_to_dvf_file_sec")) for row in warm
                    ),
                    "warm_loaded_images_to_dvf_file_sec_std": std(
                        finite_float(row.get("loaded_images_to_dvf_file_sec")) for row in warm
                    ),
                    "cold_full_script_to_dvf_file_sec": mean(
                        finite_float(row.get("full_script_to_dvf_file_sec")) for row in cold
                    ),
                    "warm_full_script_to_dvf_file_sec_mean": mean(
                        finite_float(row.get("full_script_to_dvf_file_sec")) for row in warm
                    ),
                    "warm_full_script_to_dvf_file_sec_std": std(
                        finite_float(row.get("full_script_to_dvf_file_sec")) for row in warm
                    ),
                    "cold_persistent_case_wall_sec": mean(
                        finite_float(row.get("persistent_case_wall_sec")) for row in cold
                    ),
                    "warm_persistent_case_wall_sec_mean": mean(
                        finite_float(row.get("persistent_case_wall_sec")) for row in warm
                    ),
                    "warm_persistent_case_wall_sec_std": std(
                        finite_float(row.get("persistent_case_wall_sec")) for row in warm
                    ),
                    "cold_actual_steps": mean(finite_float(row.get("actual_steps")) for row in cold),
                    "warm_actual_steps_mean": mean(finite_float(row.get("actual_steps")) for row in warm),
                    "warm_actual_steps_std": std(finite_float(row.get("actual_steps")) for row in warm),
                    "peak_mem_gb_max": max_finite(finite_float(row.get("training_peak_mem_gb")) for row in candidate_rows),
                    "cold_peak_mem_gb": max_finite(finite_float(row.get("training_peak_mem_gb")) for row in cold),
                    "warm_peak_mem_gb_max": max_finite(finite_float(row.get("training_peak_mem_gb")) for row in warm),
                }
            )
    return out


def by_candidate(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(row.get("candidate")): row for row in rows}


def saving_pct(base: float, fast: float) -> float:
    return 100.0 * (base - fast) / base if math.isfinite(base) and base > 0 and math.isfinite(fast) else float("nan")


def runtime_vs_baseline(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    lookup = by_candidate(summary_rows)
    out: List[Dict[str, object]] = []
    for fast, baseline in runner.BASELINE_FOR_FAST.items():
        fast_row = lookup.get(fast, {})
        base_row = lookup.get(baseline, {})
        cold_fast = finite_float(fast_row.get("cold_loaded_images_to_dvf_file_sec"))
        cold_base = finite_float(base_row.get("cold_loaded_images_to_dvf_file_sec"))
        warm_fast = finite_float(fast_row.get("warm_loaded_images_to_dvf_file_sec_mean"))
        warm_base = finite_float(base_row.get("warm_loaded_images_to_dvf_file_sec_mean"))
        out.append(
            {
                "fast_candidate": fast,
                "baseline_candidate": baseline,
                "method_label": fast_row.get("method_label", ""),
                "cold_loaded_images_to_dvf_file_base_sec": cold_base,
                "cold_loaded_images_to_dvf_file_fast_sec": cold_fast,
                "cold_loaded_images_to_dvf_file_saving_sec": cold_base - cold_fast,
                "cold_loaded_images_to_dvf_file_saving_pct": saving_pct(cold_base, cold_fast),
                "warm_loaded_images_to_dvf_file_base_sec": warm_base,
                "warm_loaded_images_to_dvf_file_fast_sec": warm_fast,
                "warm_loaded_images_to_dvf_file_saving_sec": warm_base - warm_fast,
                "warm_loaded_images_to_dvf_file_saving_pct": saving_pct(warm_base, warm_fast),
                "cold_actual_steps_base": base_row.get("cold_actual_steps", ""),
                "cold_actual_steps_fast": fast_row.get("cold_actual_steps", ""),
                "warm_actual_steps_base": base_row.get("warm_actual_steps_mean", ""),
                "warm_actual_steps_fast": fast_row.get("warm_actual_steps_mean", ""),
                "peak_mem_gb_base_max": base_row.get("peak_mem_gb_max", ""),
                "peak_mem_gb_fast_max": fast_row.get("peak_mem_gb_max", ""),
            }
        )
    return out


def read_accuracy_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not Path(path).exists():
        return {}
    return {row.get("candidate", ""): row for row in c.read_csv(Path(path))}


def paper_table_rows(summary_rows: Sequence[Dict[str, object]], accuracy_rows: Dict[str, Dict[str, str]]) -> List[Dict[str, object]]:
    lookup = by_candidate(summary_rows)
    out: List[Dict[str, object]] = []
    for fast, baseline in runner.BASELINE_FOR_FAST.items():
        fast_row = lookup.get(fast, {})
        base_row = lookup.get(baseline, {})
        acc = accuracy_rows.get(fast, {})
        out.append(
            {
                "method": fast_row.get("method_label", runner.paper_method_label(runner.selected_candidate(fast).method_label)),
                "fast_candidate": fast,
                "baseline_candidate": baseline,
                "dice_base_to_fast": f"{format_float(acc.get('baseline_dice'), 3)}->{format_float(acc.get('fast_dice'), 3)}",
                "hd95_base_to_fast": f"{format_float(acc.get('baseline_hd95'), 2)}->{format_float(acc.get('fast_hd95'), 2)}",
                "steps_base_to_fast": f"{format_steps(base_row.get('warm_actual_steps_mean'))}->{format_steps(fast_row.get('warm_actual_steps_mean'))}",
                "cold_sec_base_to_fast": (
                    f"{format_float(base_row.get('cold_loaded_images_to_dvf_file_sec'), 1)}"
                    f"->{format_float(fast_row.get('cold_loaded_images_to_dvf_file_sec'), 1)}"
                ),
                "warm_sec_base_to_fast": (
                    f"{format_float(base_row.get('warm_loaded_images_to_dvf_file_sec_mean'), 1)}"
                    f"->{format_float(fast_row.get('warm_loaded_images_to_dvf_file_sec_mean'), 1)}"
                ),
                "warm_sec_std_base_to_fast": (
                    f"{format_float(base_row.get('warm_loaded_images_to_dvf_file_sec_std'), 1)}"
                    f"->{format_float(fast_row.get('warm_loaded_images_to_dvf_file_sec_std'), 1)}"
                ),
                "peak_mem_gb_base_to_fast": (
                    f"{format_float(base_row.get('peak_mem_gb_max'), 2)}"
                    f"->{format_float(fast_row.get('peak_mem_gb_max'), 2)}"
                ),
            }
        )
    return out


def markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[Tuple[str, str]]) -> List[str]:
    lines = []
    lines.append("| " + " | ".join(label for label, _ in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        vals = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append("" if not math.isfinite(value) else f"{value:.2f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def build_report(
    *,
    variant: str,
    case_rows: Sequence[Dict[str, object]],
    summary_rows: Sequence[Dict[str, object]],
    vs_rows: Sequence[Dict[str, object]],
    table_rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# OASIS Paper Cold/Warm DVF Timing")
    lines.append("")
    lines.append(f"- Variant: `{variant}`")
    lines.append(f"- Generated: `{c.now_text()}`")
    lines.append(f"- Completed valid timing rows: `{sum(1 for row in case_rows if row.get('validation_pass'))}/110`")
    lines.append(f"- Missing or failed rows: `{len(failures)}`")
    lines.append("")
    lines.append("## Timing Definitions")
    lines.append("")
    lines.append("- `cold`: first OASIS rank-0 run in a fresh method worker.")
    lines.append("- `warm`: rank 0 rerun plus ranks 1-9 in the same already-initialized worker.")
    lines.append("- `loaded_images_to_dvf_file_sec`: paper endpoint, loaded input images to saved dense DVF.")
    lines.append("- `full_script_to_dvf_file_sec`: script start to saved dense DVF.")
    lines.append("- `persistent_case_wall_sec`: worker-measured wall around one in-process `run_one.main()` call.")
    lines.append("- All runs use `--output-mode dvf_only`, so Dice/HD95/ASD/dense NCC/regularity evaluation is skipped.")
    lines.append("- Step counts are optimizer updates for one image pair; the report labels them as steps, not epochs.")
    lines.append("")
    lines.append("## Method Summary")
    lines.extend(
        markdown_table(
            summary_rows,
            [
                ("Phase", "phase_group"),
                ("Candidate", "candidate"),
                ("Method", "method_label"),
                ("Cold n", "cold_case_count"),
                ("Warm n", "warm_case_count"),
                ("Cold DVF s", "cold_loaded_images_to_dvf_file_sec"),
                ("Warm DVF s", "warm_loaded_images_to_dvf_file_sec_mean"),
                ("Warm SD", "warm_loaded_images_to_dvf_file_sec_std"),
                ("Warm steps", "warm_actual_steps_mean"),
                ("Peak GB", "peak_mem_gb_max"),
            ],
        )
    )
    lines.append("")
    lines.append("## Fast Versus 2500-Step Baseline")
    lines.extend(
        markdown_table(
            vs_rows,
            [
                ("Method", "method_label"),
                ("Fast", "fast_candidate"),
                ("Baseline", "baseline_candidate"),
                ("Cold base", "cold_loaded_images_to_dvf_file_base_sec"),
                ("Cold fast", "cold_loaded_images_to_dvf_file_fast_sec"),
                ("Cold saving %", "cold_loaded_images_to_dvf_file_saving_pct"),
                ("Warm base", "warm_loaded_images_to_dvf_file_base_sec"),
                ("Warm fast", "warm_loaded_images_to_dvf_file_fast_sec"),
                ("Warm saving %", "warm_loaded_images_to_dvf_file_saving_pct"),
            ],
        )
    )
    lines.append("")
    lines.append("## Paper Runtime Table Values")
    lines.extend(
        markdown_table(
            table_rows,
            [
                ("Method", "method"),
                ("Dice", "dice_base_to_fast"),
                ("HD95", "hd95_base_to_fast"),
                ("Steps", "steps_base_to_fast"),
                ("Cold s", "cold_sec_base_to_fast"),
                ("Warm s", "warm_sec_base_to_fast"),
                ("Peak GB", "peak_mem_gb_base_to_fast"),
            ],
        )
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `paper_cold_warm_case_timing.csv/json`")
    lines.append("- `paper_cold_warm_method_summary.csv/json`")
    lines.append("- `paper_fast_vs_baseline_timing.csv`")
    lines.append("- `paper_runtime_table_values.csv`")
    lines.append("- `missing_or_failed_jobs.json`")
    lines.append("- `gpu_observations.csv`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    Path(args.review_root).mkdir(parents=True, exist_ok=True)
    rows = read_progress_rows(Path(args.review_root))
    case_rows, failures = validated_case_rows(rows)
    failures = add_missing_failures(case_rows, failures)
    summary_rows = method_summary(case_rows)
    vs_rows = runtime_vs_baseline(summary_rows)
    accuracy_rows = read_accuracy_rows(Path(args.accuracy_table))
    table_rows = paper_table_rows(summary_rows, accuracy_rows)
    gpu_rows = read_gpu_rows(Path(args.review_root))

    c.write_csv(Path(args.review_root) / "paper_cold_warm_case_timing.csv", case_rows)
    c.write_json(Path(args.review_root) / "paper_cold_warm_case_timing.json", case_rows)
    c.write_csv(Path(args.review_root) / "paper_cold_warm_method_summary.csv", summary_rows)
    c.write_json(Path(args.review_root) / "paper_cold_warm_method_summary.json", summary_rows)
    c.write_csv(Path(args.review_root) / "paper_fast_vs_baseline_timing.csv", vs_rows)
    c.write_csv(Path(args.review_root) / "paper_runtime_table_values.csv", table_rows)
    c.write_json(Path(args.review_root) / "missing_or_failed_jobs.json", failures)
    c.write_csv(Path(args.review_root) / "gpu_observations.csv", gpu_rows)
    report = build_report(
        variant=str(args.variant),
        case_rows=case_rows,
        summary_rows=summary_rows,
        vs_rows=vs_rows,
        table_rows=table_rows,
        failures=failures,
    )
    (Path(args.review_root) / "paper_cold_warm_timing_report.md").write_text(report)

    # Mirror small review summaries to the run root for convenience.
    for name in [
        "paper_cold_warm_case_timing.csv",
        "paper_cold_warm_method_summary.csv",
        "paper_fast_vs_baseline_timing.csv",
        "paper_runtime_table_values.csv",
        "missing_or_failed_jobs.json",
        "gpu_observations.csv",
        "paper_cold_warm_timing_report.md",
    ]:
        src = Path(args.review_root) / name
        if src.exists():
            dst = Path(args.out_root) / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text())

    if failures and not args.allow_incomplete:
        raise RuntimeError(f"Timing analysis found {len(failures)} missing or failed rows")
    print(
        json.dumps(
            {
                "case_rows": len(case_rows),
                "valid_rows": sum(1 for row in case_rows if row.get("validation_pass")),
                "failures": len(failures),
                "report": str(Path(args.review_root) / "paper_cold_warm_timing_report.md"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
