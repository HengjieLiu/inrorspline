"""Capture Case04 controls on busy GPU-0 while reusing accepted warm timing."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.experiments.mrdbscp842_acceleration.run_suite import Worker, dump
from .config import candidate_overrides
from .workflow import ProgressSuite


def sha256_array(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_record(review, run_container):
    timing_rows = list(csv.DictReader((review / "timings.csv").open()))
    row = next(item for item in timing_rows
               if item["id"] == "headline_fixed_50_50_50_cached_batch_004_attempt0"
               and item["rank"] == "3" and item["condition"] == "warm")
    stage_rows = [item for item in csv.DictReader((review / "stage_diagnostics.csv").open())
                  if item["id"] == row["id"]]
    if len(stage_rows) != 3 or [int(item["steps"]) for item in stage_rows] != [50, 50, 50]:
        raise ValueError("Historical Case04 stage record is incomplete")
    validation = json.loads((review / "delivery_validation.json").read_text())
    if not (validation.get("accepted_checks_pass") and validation.get("main_complete")):
        raise ValueError("Historical delivery is not accepted")
    evaluations = json.loads((review / "evaluations.json").read_text())
    evaluations = evaluations.get("rows", evaluations) if isinstance(evaluations, dict) else evaluations
    quality = next(item for item in evaluations
                   if item.get("registration_id") == row["id"]
                   and item.get("rank") == 3
                   and item.get("variant") == "fixed_50_50_50_cached_batch")
    history = f"{run_container}/jobs/{row['id']}/history.json"
    return dict(id=row["id"], endpoint_sec=float(row["seconds"]), dvf=row["dvf"],
                history=history,
                dvf_array_sha256=quality["dvf_sha256"],
                stage_sec=[float(item["loop_and_postloop_wall_sec"]) for item in stage_rows],
                source_review=str(review), delivery_validation=str(review / "delivery_validation.json"))


def remap_timeline(capture_progress, historical):
    events = copy.deepcopy(capture_progress["events"])
    groups = [[row for row in events if int(row["stage"]) == stage] for stage in range(3)]
    if any(not rows for rows in groups):
        raise ValueError("Capture timeline lacks a stage")
    starts = [rows[0]["elapsed_sec"] for rows in groups]
    ends = [next(row["elapsed_sec"] for row in reversed(rows) if row["final"])
            for rows in groups]
    capture_endpoint = float(capture_progress["endpoints"]["dvf_file_ready"])
    raw_gaps = [starts[0], starts[1] - ends[0], starts[2] - ends[1],
                capture_endpoint - ends[2]]
    remainder = historical["endpoint_sec"] - sum(historical["stage_sec"])
    if remainder < 0 or any(value < 0 for value in raw_gaps):
        raise ValueError("Invalid historical/capture timing decomposition")
    gap_total = sum(raw_gaps)
    mapped_gaps = ([remainder * value / gap_total for value in raw_gaps]
                   if gap_total else [remainder / 4.] * 4)
    stage_offsets = []
    cursor = mapped_gaps[0]
    for stage in range(3):
        stage_offsets.append(cursor)
        cursor += historical["stage_sec"][stage]
        cursor += mapped_gaps[stage + 1]
    for stage, rows in enumerate(groups):
        source_span = ends[stage] - starts[stage]
        if source_span <= 0:
            raise ValueError("Capture stage has no measurable span")
        for row in rows:
            fraction = (row["elapsed_sec"] - starts[stage]) / source_span
            row["elapsed_sec"] = stage_offsets[stage] + fraction * historical["stage_sec"][stage]
    flat = sorted(events, key=lambda row: (row["elapsed_sec"], int(row["stage"]), int(row["epoch"]), row["final"]))
    if any(second["elapsed_sec"] < first["elapsed_sec"] for first, second in zip(flat, flat[1:])):
        raise ValueError("Mapped historical timeline is nonmonotonic")
    return flat, mapped_gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("container", "repo-container", "artifact-host", "artifact-container",
                 "oasis-root", "run-id", "historical-review-host",
                 "historical-run-container"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", type=int, choices=[0], default=0)
    parser.add_argument("--capture-interval", type=int, choices=[1, 10], default=10)
    args = parser.parse_args()
    args.profile = ("fixed50_cached_batch_dense_capture"
                    if args.capture_interval == 1 else "fixed50_cached_batch")
    args.phase = "run"
    args.experiment_key = "2026-09-22_mrdbscp_fixed50_cached_batch_animation"
    artifact = Path(args.artifact_host).resolve()
    repo = Path.cwd().resolve()
    if (not artifact.is_dir() or artifact == Path("/") or artifact.is_relative_to(repo)
            or repo.is_relative_to(artifact) or "/" in args.run_id
            or not args.run_id.startswith("mrdbscp_progress_case04_")):
        raise ValueError("Require dedicated external run ID and artifact root")
    suite = ProgressSuite(args)
    suite.progress_provenance()
    suite.frozen()
    historical = historical_record(Path(args.historical_review_host),
                                   args.historical_run_container)
    dump(suite.root / "job_manifest.json", dict(
        rank=3, case_number=4, fixed="0117", moving="0322",
        profile=suite.presentation.to_dict(), features=list(suite.presentation.features),
        backend="F/T", timing_source=historical,
        execution="one untimed concurrent capture plus offline evaluation",
        contamination_policy="foreign GPU processes retained; capture wall time rejected"))
    suite.monitor.thread.start()
    # The user explicitly authorized an untimed capture in remaining GPU-0
    # memory. Monitoring remains active for provenance, but does not gate launch.
    suite.monitor.wait_clear = lambda *args, **kwargs: None
    try:
        retained_capture = suite.root / "jobs" / "formal_capture_attempt0" / "accepted_result.json"
        if retained_capture.exists():
            capture = json.loads(retained_capture.read_text())
            dump(suite.root / "capture_resume.json", dict(
                reused=str(retained_capture), reason="post-capture metadata-only validation fix"))
        else:
            worker = Worker(suite, "historical_capture", suite.presentation.features)
            try:
                capture = worker.execute("progress_register", "formal_capture_attempt0",
                    rank=3, progress_mode="capture", smoke=False,
                    presentation_profile=suite.presentation.profile,
                    candidate_overrides=candidate_overrides(suite.presentation))
            finally:
                worker.close()
        capture_progress = json.loads(Path(suite.args.artifact_host,
            Path(capture["progress"]).relative_to(args.artifact_container)).read_text())
        historical_history_host = Path(args.artifact_host) / Path(
            historical["history"]).relative_to(args.artifact_container)
        historical_history = json.loads(historical_history_host.read_text())
        trajectory_keys = ("stage_idx", "global_epoch", "stage_epoch", "full_cps",
                           "downsample", "domain_cps", "loss", "data_loss",
                           "reg_loss", "alpha", "lr")
        same_trajectory = (len(capture_progress["history"]) == len(historical_history) == 150
                           and all(all(first[key] == second[key] for key in trajectory_keys)
                                   for first, second in zip(capture_progress["history"],
                                                            historical_history)))
        if not same_trajectory:
            raise ValueError("Concurrent capture loss trajectory differs from accepted historical trajectory")
        historical_dvf_host = Path(args.artifact_host) / Path(
            historical["dvf"]).relative_to(args.artifact_container)
        capture_dvf_host = Path(args.artifact_host) / Path(
            capture["dvf"]).relative_to(args.artifact_container)
        historical_hash, capture_hash = sha256_array(historical_dvf_host), sha256_array(capture_dvf_host)
        if historical_hash != capture_hash:
            raise ValueError("Concurrent capture final DVF differs byte-for-byte from historical Case04")
        events, mapped_gaps = remap_timeline(capture_progress, historical)
        synthetic = copy.deepcopy(capture_progress)
        synthetic.update(mode="clock", events=events, snapshots=[], history=historical_history,
                         endpoints={"case_load_complete": 0.,
                                    "dvf_file_ready": historical["endpoint_sec"]},
                         clock_semantics=("accepted historical Case04 endpoint and stage spans; "
                                          "within-stage event fractions inherited from matching capture"))
        synthetic_path = suite.root / "historical_clock_progress.json"
        dump(synthetic_path, synthetic)
        presentation = suite.presentation.to_dict()
        common = dict(rank=3, elapsed_sec=historical["endpoint_sec"], steps=[50, 50, 50],
            stop_reasons=["requested_epochs_complete"] * 3, peak_gib=None,
            dvf=historical["dvf"], progress=suite.container(synthetic_path),
            presentation=presentation, features=list(suite.presentation.features), clean=True,
            backend=dict(cudnn_benchmark=False, cudnn_deterministic=True),
            timing_source="accepted_historical_headline")
        baseline = dict(common, id=historical["id"], progress_mode="baseline")
        clock = dict(common, id=historical["id"] + "_mapped_clock", progress_mode="clock")
        rows = [baseline, clock, capture]
        dump(suite.root / "formal_rows.json", rows)
        timing_source = dict(kind="historical_accepted", **historical,
            dvf_file_sha256=historical_hash, mapped_nonoptimization_gaps_sec=mapped_gaps,
            capture_clean=bool(capture["clean"]), capture_elapsed_sec=capture["elapsed_sec"])
        retained_evaluation = (suite.root / "jobs" / "formal_evaluation_attempt0" /
                               "accepted_result.json")
        if retained_evaluation.exists():
            evaluation = json.loads(retained_evaluation.read_text())
            animation_data = Path(args.artifact_host) / Path(
                evaluation["animation_data"]).relative_to(args.artifact_container)
            payload = json.loads(animation_data.read_text())
            payload["timing_source"] = timing_source
            dump(animation_data, payload)
            dump(suite.root / "evaluation_resume.json", dict(
                reused=str(retained_evaluation), updated_timing_provenance=str(animation_data)))
        else:
            evaluator = Worker(suite, "historical_evaluation", ())
            try:
                evaluation = evaluator.execute("progress_evaluate", "formal_evaluation_attempt0",
                    rows=rows, rank=3, smoke=False, timing_source=timing_source)
            finally:
                evaluator.close()
        dump(suite.root / "formal_evaluation.json", evaluation)
        dump(suite.root / "capture_DONE.json", dict(
            evaluation=evaluation, timing_source=timing_source,
            utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(completed="capture_with_historical_timing",
                              evaluation=evaluation, timing_source=timing_source)), flush=True)
    except Exception as error:
        dump(suite.root / "ERROR.json", dict(error=repr(error)))
        raise
    finally:
        suite.monitor.done.set()
        suite.monitor.thread.join(timeout=35)


if __name__ == "__main__":
    main()
