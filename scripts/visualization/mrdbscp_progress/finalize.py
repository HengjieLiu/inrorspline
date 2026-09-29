"""Final independent delivery audit and byte-matched historical metric reuse."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess

import numpy as np

from .preview import digest
from .render import stage_indices
from .render_compact import select_field_cadence


class LocalAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        self.paths.extend(value for key, value in attrs if key in ("src", "href") and value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--historical-review", type=Path, required=True)
    args = parser.parse_args()
    import os
    artifact = Path(os.environ["ARTIFACT_ROOT"]).resolve()
    for path in (args.run_root, args.render_root):
        if path.resolve() == artifact or not path.resolve().is_relative_to(artifact):
            raise ValueError("Only external artifact children can be finalized")
    load = lambda p: json.loads(Path(p).read_text())
    done = load(args.run_root / "capture_DONE.json")
    evaluation = done["evaluation"]
    data = load(evaluation["animation_data"])
    profile = data.get("presentation", {"features": ["cpbe"], "candidate": "MR_A_fast_decay"})
    acceptance = load(args.render_root / "acceptance.json")
    formal = load(args.run_root / "formal_rows.json")
    attempts = load(args.run_root / "attempts.json")
    frozen = load(args.run_root / "formal_source_sha256.json")
    repo = Path(__file__).resolve().parents[3]
    timing_source = data.get("timing_source", {"kind": "paired_current_run"})
    historical_clock = timing_source.get("kind") == "historical_accepted"
    expected_modes = (["baseline", "clock", "capture"] if historical_clock
                      else ["baseline", "clock"]*3 + ["capture"])
    clean_inputs = (data["baseline"]["clean"] and data["clock"]["clean"]
                    and (historical_clock or (data["capture"]["clean"] and evaluation["clean"])))
    checks = dict(render_passed=acceptance["passed"],
        clean_timing_and_acceptable_capture=clean_inputs,
        expected_modes=[r["progress_mode"] for r in formal] == expected_modes,
        exact_protocol=all(r["rank"] == 3 and r["features"] == profile["features"] and
            r["backend"]["cudnn_benchmark"] is False and r["backend"]["cudnn_deterministic"] is True for r in formal),
        frozen_training_source_unchanged=all(digest(repo / p) == h for p, h in frozen.items()),
        every_step_loss=len(data["history"]) == sum(data["steps"]),
        parity_passed=all(r["passed"] for r in data["trajectory_parity"]),
        reconstruction_within_tolerance=data["final_reconstruction"]["vector_rms_voxel"] <= .005 and
            data["final_reconstruction"]["vector_max_voxel"] <= .05,
        overhead_gate=abs(data["median_clock_overhead_percent"]) <= 5.)
    parser_html = LocalAssets()
    parser_html.feed((args.render_root / "index.html").read_text())
    checks["html_local_assets"] = all("://" not in p and (args.render_root / p).is_file() for p in parser_html.paths)
    states = load(args.render_root / "playback.json")
    presentation_timing = load(args.render_root / "presentation_timing.json")
    render_data = select_field_cadence(data, int(presentation_timing["field_step_interval"]))
    expected_frames = round(presentation_timing["video_duration_sec"] * presentation_timing["fps"])
    checks["video_frame_count_matches"] = (
        sum(r["video_frames"] for r in states) == expected_frames
        and acceptance["decoded_frames"] == expected_frames)
    checks["phase_aware_15_second_replay"] = (
        presentation_timing["video_duration_sec"] == 15.
        and presentation_timing["optimization_video_sec"] == 13.7
        and all(row["kind"] in ("optimization", "final_hold")
                for row in presentation_timing["phases"]))
    checks["real_endpoint_preserved"] = states[-1]["elapsed_sec"] == data["endpoint_sec"]
    checks["state_files_present"] = all((args.render_root / r["png"]).is_file() for r in states)
    frozen_columns = True
    previous = {}
    for i, row in enumerate(render_data["snapshots"]):
        indices = stage_indices(render_data, i)
        for stage, idx in previous.items():
            if render_data["snapshots"][idx]["final"]:
                frozen_columns &= indices[stage] == idx
        previous = indices
    checks["completed_columns_frozen"] = frozen_columns
    # Reuse final HD95/regularity only if evaluated field bytes match exactly.
    field = np.load(data["capture"]["dvf"], mmap_mode="r")
    h = hashlib.sha256()
    for slab in field:
        h.update(memoryview(np.ascontiguousarray(slab)))
    if historical_clock:
        checks["historical_capture_array_hash_match"] = (
            h.hexdigest() == timing_source.get("dvf_array_sha256"))
        checks["historical_capture_file_hash_match"] = (
            digest(Path(data["capture"]["dvf"])) == timing_source.get("dvf_file_sha256"))
    historic_payload = load(args.historical_review / "evaluations.json")
    historic = historic_payload["rows"] if isinstance(historic_payload, dict) else historic_payload
    matches = [r for r in historic if r["rank"] == 3
               and r.get("variant", profile["candidate"]) == profile["candidate"]
               and r["dvf_sha256"] == h.hexdigest()]
    quality = dict(matched=bool(matches), dvf_array_sha256=h.hexdigest(),
                   source_review=str(args.historical_review))
    if matches:
        source = matches[0]
        quality.update(source_id=source["id"], dice=source["mean_dice"],
            hd95_mm=source["surface_metrics"]["hd95_mean"], regularity=source["regularity"])
        checks["final_dice_matches_historical"] = bool(np.isclose(data["snapshots"][-1]["dice"], quality["dice"], atol=1e-12, rtol=0))
    summary = dict(passed=all(checks.values()), checks=checks,
        total_attempts=len(attempts), contaminated_attempts=sum(not r["clean"] for r in attempts),
        baseline_seconds=[r["elapsed_sec"] for r in formal if r["progress_mode"] == "baseline"],
        clock_seconds=[r["elapsed_sec"] for r in formal if r["progress_mode"] == "clock"],
        final_quality=quality, rank=3, case_number=4,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        source_sha256=digest(Path(__file__)))
    (args.render_root / "final_quality.json").write_text(json.dumps(quality, indent=2))
    (args.render_root / "delivery_validation.json").write_text(json.dumps(summary, indent=2))
    (args.render_root / "finalizer_source.py").write_text(Path(__file__).read_text())
    stages = []
    for stage, measured in enumerate(presentation_timing["clock"]["stages"]):
        span = measured["optimization_sec"]
        stages.append(dict(stage=stage+1, steps=data["steps"][stage],
            preparation_start_sec=measured["preparation_start_sec"],
            preparation_end_sec=measured["preparation_end_sec"],
            preparation_sec=measured["preparation_sec"],
            start_sec=measured["optimization_start_sec"],
            end_sec=measured["optimization_end_sec"], optimization_span_sec=span,
            amortized_ms_per_step=1000*span/data["steps"][stage]))
    (args.render_root / "stage_timings.json").write_text(json.dumps(stages, indent=2))
    report = args.render_root / "report.md"
    text = report.read_text() + "\n\n## Independent acceptance and provenance\n\n"
    text += f"All checks passed: {summary['passed']}. Contaminated attempts: {summary['contaminated_attempts']}.\n\n"
    text += "GPU monitoring logs, all attempts, worker commands, dirty-tree patch, data/image manifests and frozen training sources are retained in:\n\n"
    text += f"`{args.run_root}`\n\n"
    text += f"Git HEAD: `{summary['git_head']}`; dirty-tree state recorded in run provenance.\n\n"
    text += f"Presentation profile: `{profile['profile']}`; features: `{profile['features']}`; steps: `{data['steps']}`.\n\n"
    if historical_clock:
        text += ("The displayed endpoint is the accepted historical Case04 headline warm "
                 "measurement. The concurrent snapshot run is numerical input only; its "
                 "wall time is excluded. Stage spans are historical measurements. Steps are "
                 "evenly placed inside each measured stage span; contaminated capture-time "
                 "fractions are not reused.\n\n")
    else:
        text += "All three pairs are retained, including the slower first baseline; no fastest-run selection or time correction.\n\n"
        text += "| Pair | Baseline s | Clock s | Relative difference % |\n|---|---:|---:|---:|\n"
        for i, (base, clock) in enumerate(zip(summary["baseline_seconds"], summary["clock_seconds"])):
            text += f"| {i+1} | {base:.6f} | {clock:.6f} | {100*(clock/base-1):+.3f} |\n"
        text += "\nPaired differences include ordinary timing variability; they are not proof that timestamping accelerates optimization.\n\n"
    text += "Stage optimization spans below are accepted historical measurements; per-step numbers are amortized wall times, not kernel timings. Endpoint timing exposes preparation only as a tiny aggregate residual, which the presentation clock splits equally and labels explicitly as an allocation.\n\n"
    text += "| Stage | Steps | Optimization span s | Amortized ms/step |\n|---|---:|---:|---:|\n"
    for row in stages:
        text += f"| {row['stage']} | {row['steps']} | {row['optimization_span_sec']:.4f} | {row['amortized_ms_per_step']:.3f} |\n"
    text += f"\nAllocated preparation/transition residual: {presentation_timing['clock']['preparation_total_sec']:.6f} s; measured final-output interval: {presentation_timing['clock']['output_sec']:.6f} s.\n"
    if matches:
        reg = quality["regularity"]
        text += ("\n## Final field quality (exact-byte historical reuse)\n\n"
                 f"Dice {quality['dice']:.6f}; HD95 {quality['hd95_mm']:.6f} mm; "
                 f"foreground SDlogJ {reg['log_jdet_std_fg']:.6f}; foreground folding {reg['jdet_le0_pct_fg']:.6f}%.\n\n"
                 "These prior metrics are reused only after confirming the selected case and exact DVF array SHA256 match. They were not computed during registration.\n")
    else:
        text += "\nHistorical final field did not match byte-for-byte; HD95/SDlogJ have not been transferred to this new field.\n"
    report.write_text(text)
    files = {p.relative_to(args.render_root).as_posix(): digest(p)
             for p in args.render_root.rglob("*") if p.is_file() and p.name != "output_sha256.json"}
    (args.render_root / "output_sha256.json").write_text(json.dumps(files, indent=2))
    print(json.dumps(summary))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
