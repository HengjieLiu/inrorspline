#!/usr/bin/env python
"""Report-only, restartable first100 recovery. Never launches a GPU fit.

The launch manifest and numerical code are immutable. A separate recovery
manifest records the report correction; atomic per-case caches survive failure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

# Set before importing numpy/torch, including in spawned interpreters.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import oasis_first100_ncc9_dense as report
from launch_oasis_dl_ncc_diffusion_sweep import validate_result
from run_oasis_dl_ncc_diffusion_one import source_hashes

c, d, cohort = report.c, report.d, report.cohort
sha = report.sha256_file


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def fingerprint(path):
    stat = Path(path).stat()
    return [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino]


def validate_reference(row, identity):
    for key in ("case_rank", "fixed_id", "moving_id", "direction", "source_path", "source_sha256", "representation"):
        if row.get(key) != identity[key]:
            raise ValueError(f"Reference {key} mismatch: {row.get(key)} != {identity[key]}")
    if not all(math.isfinite(float(row[key])) for key in report.METRICS):
        raise ValueError("Nonfinite reference metrics")


def load_cache(path, identity):
    if not path.exists():
        return None
    value = c.read_json(path)
    if value.get("identity") != identity:
        raise ValueError(f"Stale reference cache (preserved): {path}")
    validate_reference(value["row"], identity)
    return value["row"]


def worker_init():
    report.torch.set_num_threads(1)
    report.torch.set_num_interop_threads(1)


def score(task):
    identity, oasis_root, pair_file, dl_root = task
    device = report.torch.device("cpu")
    batch = c.load_oasis_case(rank=identity["case_rank"], device=device,
                             oasis_root=Path(oasis_root), pair_file=Path(pair_file))
    if (str(batch["fixed_id"]), str(batch["moving_id"])) != (identity["fixed_id"], identity["moving_id"]):
        raise ValueError("Loaded case mismatch")
    args = argparse.Namespace(target=identity["representation"].lower(), dl_root=Path(dl_root))
    normalized, audit = report.load_target(args, fixed_id=identity["fixed_id"],
                                           moving_id=identity["moving_id"], device=device)
    if audit["source_sha256"] != identity["source_sha256"]:
        raise ValueError("Reference changed during evaluation")
    field = normalized * d.voxel_scale(c.IMAGE_SHAPE, device=device)
    metrics, _ = report.canonical_metrics(field, batch)
    row = {**metrics, **report._regularity(field, batch)}
    row.update({key: identity[key] for key in ("case_rank", "fixed_id", "moving_id", "direction", "source_path", "source_sha256", "representation")})
    row.update(resolution="reference", alpha=None, group=identity["representation"],
               source_kind="original_saved_DL", gpu="", prefix_gpu="",
               training_elapsed_sec="", pipeline_training_elapsed_sec="")
    validate_reference(row, identity)
    return row


def validate_outputs(run_root, manifest):
    """One full output audit; unchanged resumes reuse stat-bound hash receipts."""
    state = c.read_json(run_root / "launcher_state.json")
    receipt_path = run_root / "report_recovery/output_validation.json"
    old = c.read_json(receipt_path) if receipt_path.exists() else {}
    receipts, rows = {}, []
    for i, spec in enumerate(cohort.specs()):
        path = Path(state["results"][spec.name()])
        row = validate_result(path, spec, manifest["numerical_source_hashes"], verify_field=False)
        key = sha(path / "summary.json")
        stamps = [fingerprint(row[name]) for name in ("dvf_path", "model_path")]
        if old.get(spec.name()) != {"summary_sha256": key, "files": stamps}:
            validate_result(path, spec, manifest["numerical_source_hashes"], verify_field=True)
        receipts[spec.name()] = {"summary_sha256": key, "files": stamps}
        rows.append(row)
        if (i + 1) % 20 == 0:
            atomic_json(receipt_path, {**old, **receipts})
            print(json.dumps({"output_validation": i + 1, "total": 800}), flush=True)
    by_name = {cohort.First100FitSpec(**row["spec"]).name(): row for row in rows}
    for spec in cohort.specs():
        if spec.representation == "842_dense":
            dense, prefix = by_name[spec.name()], by_name[spec.prefix().name()]
            if dense["stages"][:3] != prefix["stages"] or dense["prefix_dvf_sha256"] != prefix["dvf_sha256"]:
                raise ValueError(f"Prefix mismatch: {spec.name()}")
    atomic_json(receipt_path, receipts)
    return rows


def verify_report(root):
    data = c.read_json(root / "results.json")
    if (data["candidate_count"], data["reference_count"], len(data["case_metrics"]), len(data["candidates"])) != (400, 200, 600, 800):
        raise ValueError("Incomplete final report")
    for filename, expected in (("case_metrics.csv", data["case_metrics"]),
                               ("mean_std.csv", data["means"]),
                               ("leave_rank004_out_mean_std.csv", data["leave_rank004_out_means"])):
        with (root / filename).open() as handle:
            actual = list(csv.DictReader(handle))
        if len(actual) != len(expected):
            raise ValueError(f"Wrong row count: {filename}")
        for left, right in zip(actual, expected):
            for key, value in right.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isclose(float(left[key]), float(value), rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError(f"CSV/JSON mismatch: {filename}/{key}")
                elif left[key] != ("" if value is None else str(value)):
                    raise ValueError(f"CSV/JSON mismatch: {filename}/{key}")
    # Recompute summaries independently from the serialized case rows.
    for actual, expected in zip(report.summarize(data["case_metrics"]), data["means"]):
        for key, value in actual.items():
            if value != expected[key]:
                raise ValueError(f"Recomputed summary mismatch: {key}")
    leave = report.summarize(data["case_metrics"], [r for r in range(100) if r != 4])
    if leave != data["leave_rank004_out_means"]:
        raise ValueError("Leave-rank4-out mismatch")
    markdown = (root / "report.md").read_text()
    for row in data["means"] + data["leave_rank004_out_means"]:
        for metric in report.METRICS[:5]:
            text = f"{row[metric + '_mean']:.6f} ± {row[metric + '_std']:.6f}"
            if text not in markdown:
                raise ValueError("Markdown/JSON mismatch")
    return {name: sha(root / name) for name in ("case_metrics.csv", "mean_std.csv", "leave_rank004_out_mean_std.csv", "results.json", "report.md")}


def execute(args):
    run_root, review_root = args.run_root, args.review_root
    recovery = run_root / "report_recovery"
    recovery.mkdir(parents=True, exist_ok=True)
    with (recovery / "finalizer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (review_root / "ACCEPTED.json").exists():
            verify_report(review_root)
            print("Already complete; existing reports verified", flush=True)
            return
        manifest = c.read_json(run_root / "launch_manifest.json")
        if source_hashes() != manifest["numerical_source_hashes"]:
            raise ValueError("Numerical provenance drift")
        settings = manifest["args"]
        pair_file = Path(settings["pair_file"])
        if sha(pair_file) != manifest["pair_file_sha256"]:
            raise ValueError("Pair split changed")
        pairs = c.parse_pair_file(pair_file)
        if len(pairs) != 200:
            raise ValueError("Wrong split length")
        evaluator = {str(path): sha(path) for path in
                     (Path(__file__), Path(report.__file__), report.SWEEP / "common.py",
                      report.SWEEP / "run_oasis_vfa_feature_dense_one.py",
                      report.SWEEP / "run_oasis_saved_field_bspline_fit_one.py")}
        provenance = {"launch_manifest_sha256": sha(run_root / "launch_manifest.json"),
                      "original_launch_manifest": str(run_root / "launch_manifest.json"),
                      "source_revision": manifest["source_revision"], "evaluator_hashes": evaluator,
                      "original_orchestration_hashes": manifest["orchestration_source_hashes"],
                      "scope": "CPU report-only recovery; no numerical code or registrations changed",
                      "workers": args.workers, "pid": os.getpid(), "started_at": c.now_text()}
        atomic_json(recovery / f"provenance_{os.getpid()}.json", provenance)
        legacy_path = run_root / "reference_scoring_progress.json"
        legacy = c.read_json(legacy_path).get("completed_rows", {}) if legacy_path.exists() else {}
        tasks, rows, identities = [], {}, {}
        for method in ("VFA", "SITReg"):
            target_args = argparse.Namespace(target=method.lower(), dl_root=Path(settings["dl_root"]))
            for rank, (fixed_id, moving_id) in enumerate(pairs[:100]):
                key = f"{method}:{rank}"
                source = report.target_path(target_args, fixed_id, moving_id)
                identity = dict(case_rank=rank, fixed_id=fixed_id, moving_id=moving_id,
                                direction="moving_to_fixed", representation=method,
                                source_path=str(source), source_sha256=sha(source),
                                pair_sha256=manifest["pair_file_sha256"], evaluator_hashes=evaluator)
                identities[key] = identity
                path = recovery / "cases" / f"{method}_rank{rank:03d}.json"
                row = load_cache(path, identity)
                origin = "validated_case_cache"
                if row is None and rank < 10:
                    saved = Path(settings["baseline_root"]) / "baseline_cases" / f"{method.lower()}_rank{rank:03d}.json"
                    accepted = c.read_json(saved)
                    for code_key, filename in (("metric_code_sha256", "common.py"),
                                               ("evaluator_code_sha256", "run_oasis_vfa_feature_dense_one.py")):
                        if accepted[code_key] != sha(report.SWEEP / filename):
                            raise ValueError("Accepted reference evaluator drift")
                    row = dict(accepted["row"])
                    row.update(resolution="reference", representation=method, alpha=None, group=method,
                               source_kind="original_saved_DL", gpu="", prefix_gpu="",
                               training_elapsed_sec="", pipeline_training_elapsed_sec="")
                    origin = "accepted_first10"
                if row is None and key in legacy:
                    row = legacy[key]
                    origin = "legacy_progress_import_pending_parity"
                if row is not None:
                    validate_reference(row, identity)
                    rows[key] = row
                    atomic_json(path, {"identity": identity, "row": row, "origin": origin})
                else:
                    tasks.append((identity, settings["oasis_root"], str(pair_file), settings["dl_root"]))
        worker_init()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=worker_init) as pool:
            # Serial/spawn parity on a freshly calculated case from EACH method.
            parity = []
            for method in ("VFA", "SITReg"):
                key = f"{method}:10"
                task = (identities[key], settings["oasis_root"], str(pair_file), settings["dl_root"])
                future = pool.submit(score, task)
                serial, parallel = score(task), future.result()
                for metric in report.METRICS:
                    if not math.isclose(serial[metric], parallel[metric], rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError("Serial/spawn metric mismatch")
                    if key in rows and not math.isclose(serial[metric], rows[key][metric], rel_tol=1e-7, abs_tol=1e-7):
                        raise ValueError("Imported metric parity mismatch")
                parity.append({"key": key, "passed": True})
                rows[key] = parallel
                atomic_json(recovery / "cases" / f"{method}_rank010.json",
                            {"identity": identities[key], "row": parallel, "origin": "serial_spawn_parity"})
                print(json.dumps({"parity": key, "passed": True}), flush=True)
            atomic_json(recovery / "parity.json", parity)
            tasks = [t for t in tasks if f"{t[0]['representation']}:{t[0]['case_rank']}" not in rows]
            start, completed_new = time.monotonic(), 0
            futures = {pool.submit(score, task): task[0] for task in tasks}
            for future in as_completed(futures):
                identity = futures[future]
                row = future.result()
                key = f"{identity['representation']}:{identity['case_rank']}"
                rows[key] = row
                atomic_json(recovery / "cases" / f"{identity['representation']}_rank{identity['case_rank']:03d}.json",
                            {"identity": identity, "row": row, "origin": "fresh_spawn_worker"})
                completed_new += 1
                elapsed = time.monotonic() - start
                progress = {"phase": "reference_scoring", "completed": len(rows), "total": 200,
                            "fresh_completed_this_session": completed_new, "elapsed_seconds": elapsed,
                            "scoring_eta_seconds": (len(tasks) - completed_new) * elapsed / completed_new,
                            "updated_at": c.now_text(), "last_case": key}
                atomic_json(recovery / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        atomic_json(recovery / "progress.json", {"phase": "output_validation", "completed": 200, "total": 200, "updated_at": c.now_text()})
        results = validate_outputs(run_root, manifest)
        references = [rows[f"{m}:{r}"] for m in ("VFA", "SITReg") for r in range(100)]
        provenance["reference_field_hashes"] = {r["source_path"]: r["source_sha256"] for r in references}
        provenance["legacy_progress_sha256"] = sha(legacy_path) if legacy_path.exists() else None
        atomic_json(review_root / "references/reference_manifest.json", provenance)
        report.write_report(results, references, review_root, provenance, mark_accepted=False)
        report_hashes = verify_report(review_root)
        if source_hashes() != manifest["numerical_source_hashes"] or any(sha(Path(path)) != value for path, value in evaluator.items()):
            raise ValueError("Source changed during finalization")
        state = c.read_json(run_root / "launcher_state.json")
        state.update(phase="complete", completed_at=c.now_text(), recovery_provenance=provenance)
        atomic_json(run_root / "launcher_state.json", state)
        atomic_json(recovery / "progress.json", {"phase": "complete", "completed": 200, "total": 200, "updated_at": c.now_text()})
        atomic_json(review_root / "ACCEPTED.json", {"technical_completion": True, "candidate_count": 400,
                    "new_candidate_count": 360, "reused_accepted_first10_count": 40,
                    "reference_count": 200, "configuration_count": 4, "completed_at": c.now_text(),
                    "report_sha256": report_hashes, "parity_passed": True})
        print(json.dumps({"complete": True, "reports": str(review_root)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    execute(parser.parse_args())
