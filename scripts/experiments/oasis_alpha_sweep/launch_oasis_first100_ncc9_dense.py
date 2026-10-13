#!/usr/bin/env python
"""Run the first100 pyramid 8→4→2→dense NCC9/diffusion lambda sweep."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/reports"))

import common as c  # noqa: E402
import dl_ncc_diffusion as d  # noqa: E402
import oasis_ncc9_first100 as cohort  # noqa: E402
import launch_oasis_dl_ncc_diffusion_sweep as base  # noqa: E402
import oasis_first100_ncc9_dense as report  # noqa: E402
from launch_oasis_mrdbscp842_dense_lambda_sweep import source_metadata  # noqa: E402
from run_oasis_saved_field_bspline_fit_one import gpu_snapshot, sha256_file  # noqa: E402
from run_oasis_saved_field_bspline_fit_one import target_path  # noqa: E402
from run_oasis_dl_ncc_diffusion_one import source_hashes  # noqa: E402

def parse_args(argv=None):
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=artifact_root / "runs" / cohort.VARIANT)
    parser.add_argument("--review-root", type=Path, default=artifact_root / "repo_experiments" / cohort.VARIANT)
    parser.add_argument(
        "--reuse-root", type=Path,
        default=artifact_root / "runs/oasis_seed3_first10_mrdbscp_dl_ncc9_diffusion_6way_v1",
    )
    parser.add_argument(
        "--baseline-root", type=Path,
        default=report.BASELINE_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--dl-root", type=Path,
        default=(Path(os.environ["INRORSPLINE_OASIS_DL_ROOT"])
                 if os.environ.get("INRORSPLINE_OASIS_DL_ROOT") else None),
    )
    parser.add_argument("--oasis-root", type=Path, required=True)
    parser.add_argument("--dl-code-root", type=Path, required=True)
    parser.add_argument("--vfa-root", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--gpu-slots", default=",".join(cohort.GPU_SLOTS))
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dl_root is None:
        parser.error("--dl-root or INRORSPLINE_OASIS_DL_ROOT is required")
    try:
        args.gpu_slots = cohort.validate_gpu_slots(args.gpu_slots)
    except ValueError as exc:
        parser.error(str(exc))
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    outputs = [args.out_root.resolve(), args.review_root.resolve()]
    sources = [
        ROOT.resolve(), args.reuse_root.resolve(), args.baseline_root.resolve(),
        args.oasis_root.resolve(), args.dl_root.resolve(), args.dl_code_root.resolve(),
        args.vfa_root.resolve(), args.pair_file.resolve(),
    ]
    if any(out.is_relative_to(src) or src.is_relative_to(out) for out in outputs for src in sources):
        parser.error("Output roots must be disjoint from source/data/reference trees")
    if outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        parser.error("Run and review roots must be disjoint")
    return args


def all_specs(smoke=False):
    return cohort.specs(smoke=smoke)


def preflight(args):
    pairs = c.parse_pair_file(args.pair_file)
    if len(pairs) != 200:
        raise RuntimeError(f"Expected the complete shuffled pair_test_200 list, found {len(pairs)}")
    if not args.oasis_root.is_dir() or not args.pair_file.is_file():
        raise FileNotFoundError("OASIS root or seed-3 shuffled pair file is missing")
    # Verify every reference field exists for the exact fixed→moving pair list.
    for target in ("vfa", "sitreg"):
        target_args = argparse.Namespace(target=target, dl_root=args.dl_root)
        for rank, (fixed_id, moving_id) in enumerate(pairs[:100]):
            path = target_path(target_args, fixed_id, moving_id)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {target} rank{rank:03d} reference {path}")
    accepted = c.read_json(args.baseline_root / "ACCEPTED.json")
    if not accepted.get("complete"):
        raise RuntimeError("First10 VFA/SITReg reference report is not accepted")
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Run workers as container UID:GID1009:1020")
    snapshot = gpu_snapshot()
    if snapshot.get("error"):
        raise RuntimeError(f"Could not inspect current physical GPU state: {snapshot['error']}")
    free = {row["physical_gpu"]: int(row["memory_free_mib"]) for row in snapshot["rows"]}
    slot_counts = {gpu: args.gpu_slots.count(gpu) for gpu in set(args.gpu_slots)}
    required = cohort.MEMORY_RESERVE_MIB_PER_WORKER
    for gpu, count in slot_counts.items():
        if free.get(gpu, 0) < count * required:
            raise RuntimeError(
                f"GPU{gpu} has {free.get(gpu, 0)} MiB free; {count} concurrent worker(s) require "
                f"a conservative {count * required} MiB reservation"
            )
    return pairs, snapshot, accepted


class First100Scheduler(base.Scheduler):
    """Resumable DAG scheduler with two independently accounted GPU3 slots."""

    def command(self, spec, gpu, path):
        cmd = super().command(spec, gpu, path)
        cmd[2] = str(HERE / "run_oasis_dl_ncc_diffusion_first100_one.py")
        return cmd

    def _validate(self, path, spec):
        row = base.validate_result(path, spec, self.hashes)
        if row.get("variant") not in (
            cohort.VARIANT,
            "oasis_seed3_first10_mrdbscp_dl_ncc9_diffusion_6way_v1",
        ):
            raise RuntimeError("Candidate has an unexpected experiment identity")
        return row

    def _available_mib(self, gpu, snapshot, active):
        current = {
            row["physical_gpu"]: int(row["memory_free_mib"])
            for row in snapshot["rows"]
        }.get(gpu, 0)
        baseline = self.state.get("initial_free_mib", {}).get(gpu, current)
        active_same = sum(1 for item in active.values() if item["gpu"] == gpu)
        reserved = baseline - active_same * cohort.MEMORY_RESERVE_MIB_PER_WORKER
        return min(current, reserved)

    def run(self, requested):
        pending, failures = [], []
        for spec in requested:
            key = spec.name()
            if key in self.state["results"]:
                self._validate(Path(self.state["results"][key]), spec)
                continue
            attempts = [item for item in self.state["attempts"] if item["key"] == key]
            recovered = False
            for record in attempts:
                path = Path(record["path"])
                if (path / "DONE").is_file():
                    self._validate(path, spec)
                    self.state["results"][key] = str(path)
                    record["status"] = "complete"
                    recovered = True
                    break
            if recovered:
                continue
            if attempts:
                raise RuntimeError(f"Unresolved prior candidate attempt requires inspection: {key}")
            pending.append((spec, 0, 0))

        active, finished = {}, 0
        self.save()
        while pending or active:
            for slot, info in list(active.items()):
                code = info["process"].poll()
                if code is None:
                    continue
                info["log"].close()
                del active[slot]
                finished += 1
                spec, record = info["spec"], info["record"]
                try:
                    if code != 0:
                        raise RuntimeError(f"Worker exited with status {code}")
                    row = self._validate(Path(record["path"]), spec)
                    self.state["results"][spec.name()] = record["path"]
                    record["status"] = "complete"
                    self.event(
                        "complete", key=spec.name(), gpu=record["gpu"],
                        dice=row["dice"], hd95=row["hd95"], asd=row["asd"],
                    )
                except Exception as exc:
                    record.update(status="failed", exit_code=code, error=str(exc))
                    log_text = Path(record["log_path"]).read_text(errors="replace").lower()
                    if record["attempt"] == 0 and "cuda" in log_text and "out of memory" in log_text:
                        pending.append((spec, 1, finished))
                        self.event("oom_retry_queued", key=spec.name(), gpu=record["gpu"])
                    else:
                        failures.append({"key": spec.name(), "error": str(exc), "log": record["log_path"]})
                        self.event("failure", **failures[-1])
                self.save()

            if failures:
                pending.clear()  # Let already-running workers finish; never stop existing jobs.
            free_slots = [index for index in range(len(self.args.gpu_slots)) if index not in active]
            while pending and free_slots and len(active) < 4:
                eligible = [
                    i for i, (spec, _, ready) in enumerate(pending)
                    if finished >= ready and (
                        spec.representation == "842"
                        or spec.prefix().name() in self.state["results"]
                    )
                ]
                if not eligible:
                    break
                index = min(eligible, key=lambda i: (pending[i][0].representation == "842", i))
                spec, attempt, ready = pending[index]
                snapshot = gpu_snapshot()
                if snapshot.get("error"):
                    raise RuntimeError(f"Cannot validate physical GPU memory: {snapshot['error']}")
                required = cohort.required_memory_mib(
                    self.state.get("smoke_peak_gib", {}).get(spec.representation, 0.0)
                )
                choices = []
                for slot in free_slots:
                    gpu = self.args.gpu_slots[slot]
                    available = self._available_mib(gpu, snapshot, active)
                    if available >= required:
                        choices.append((available, slot, gpu))
                if not choices:
                    break
                if attempt:
                    _, slot, gpu = max(choices)
                else:
                    # Retain the approved slot allocation: GPU0, GPU2, GPU3, GPU3.
                    slot = min(item[1] for item in choices)
                    gpu = self.args.gpu_slots[slot]
                pending.pop(index)
                free_slots.remove(slot)
                if source_hashes() != self.hashes:
                    raise RuntimeError("Numerical source changed while scheduler was active")
                path = self.args.out_root / "jobs" / spec.name() / f"attempt{attempt}"
                log_path = self.args.out_root / "logs" / f"{spec.name()}_attempt{attempt}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() or log_path.exists():
                    raise FileExistsError(f"Refusing to overwrite attempt {path}")
                command = self.command(spec, gpu, path)
                log = log_path.open("w")
                environment = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=gpu,
                    PYTHONDONTWRITEBYTECODE="1",
                    PYTHONUNBUFFERED="1",
                    OMP_NUM_THREADS=str(self.args.cpu_threads),
                    MKL_NUM_THREADS=str(self.args.cpu_threads),
                )
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                record = {
                    "key": spec.name(), "spec": spec.__dict__, "attempt": attempt,
                    "gpu": gpu, "gpu_slot": slot, "pid": process.pid,
                    "path": str(path), "log_path": str(log_path), "status": "running",
                    "command": command, "initial_gpu_state": snapshot,
                    "reserved_memory_mib": required, "started_at": c.now_text(),
                }
                self.state["attempts"].append(record)
                active[slot] = {"process": process, "record": record, "spec": spec, "log": log, "gpu": gpu}
                self.event("launched", key=spec.name(), gpu=gpu, slot=slot, pid=process.pid, attempt=attempt)
                self.save()
            if pending or active:
                time.sleep(3)
        if failures:
            raise RuntimeError(f"Failed candidates; no incomplete aggregation: {failures}")
        return [
            self._validate(Path(self.state["results"][spec.name()]), spec)
            for spec in requested
        ]


def seed_reused_first10(scheduler, reuse_root):
    reused = []
    for rank in range(10):
        for alpha in cohort.LAMBDAS:
            for representation in ("842", "842_dense"):
                spec = cohort.First100FitSpec(rank, "pyramid", alpha, representation)
                path = reuse_root / "jobs" / spec.name() / "attempt0"
                if not (path / "DONE").is_file():
                    raise FileNotFoundError(f"Missing accepted P43 result to reuse: {path}")
                scheduler._validate(path, spec)
                scheduler.state["results"][spec.name()] = str(path)
                reused.append({"name": spec.name(), "path": str(path), "sha256": sha256_file(path / "summary.json")})
    scheduler.state["reused_accepted_first10"] = reused
    scheduler.save()
    return reused


def execute(args):
    pairs, initial_gpu_state, baseline_acceptance = preflight(args)
    if args.dry_run:
        print(json.dumps({
            "variant": cohort.VARIANT,
            "pair_count": 100,
            "lambdas": cohort.LAMBDAS,
            "dense_candidate_count": 400,
            "prefix_and_dense_worker_count": 800,
            "reuse_first10_worker_output_count": 80,
            "new_worker_output_count": 720,
            "gpu_slots": args.gpu_slots,
            "initial_gpu_free_mib": {
                row["physical_gpu"]: row["memory_free_mib"] for row in initial_gpu_state["rows"]
            },
            "safe_reservation_mib_per_worker": cohort.MEMORY_RESERVE_MIB_PER_WORKER,
            "baseline_acceptance": baseline_acceptance,
        }, indent=2))
        return

    if args.out_root.exists() and not (args.out_root / "launcher_state.json").exists():
        raise FileExistsError(f"Refusing to adopt an untracked nonempty run root: {args.out_root}")
    if args.review_root.exists() and not (args.review_root / "launch_manifest.json").exists():
        raise FileExistsError(f"Refusing to adopt an untracked nonempty review root: {args.review_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.review_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scheduler = First100Scheduler(args)
        expected_source_hashes = source_hashes()
        source_files = [
            HERE / "oasis_ncc9_first100.py",
            HERE / "run_oasis_dl_ncc_diffusion_first100_one.py",
            HERE / "launch_oasis_first100_ncc9_dense.py",
            ROOT / "scripts/reports/oasis_first100_ncc9_dense.py",
        ]
        if any(not path.is_file() for path in source_files):
            raise FileNotFoundError("Missing first100 experiment implementation source")
        manifest = {
            "variant": cohort.VARIANT,
            "protocol": d.PROTOCOL,
            "args": {**vars(args), "gpu_slots": list(args.gpu_slots)},
            "source_revision": source_metadata()["source_revision"],
            "numerical_source_hashes": expected_source_hashes,
            "orchestration_source_hashes": {str(path): sha256_file(path) for path in source_files},
            "pair_file": str(args.pair_file),
            "pair_file_sha256": sha256_file(args.pair_file),
            "pair_count": 100,
            "pair_prefix": pairs[:100],
            "baseline_acceptance_sha256": sha256_file(args.baseline_root / "ACCEPTED.json"),
            "reference_field_root": str(args.dl_root),
            "initial_gpu_state": initial_gpu_state,
            "uid": os.getuid(), "gid": os.getgid(),
            "container_image": os.environ.get("INRORSPLINE_CONTAINER_IMAGE", "hengjieliu96/dl_dir:py310sitreg_v1"),
            "container_image_id": os.environ.get("INRORSPLINE_CONTAINER_IMAGE_ID", "sha256:dab700c415830dcf07c16ef7f2901deed95bb104eda388514c3ce74a43354ba8"),
            "no_existing_job_modified": True,
            "memory_policy": "require at least 4 GiB free per newly launched worker; GPU3 reserves 8 GiB for its two slots",
            "created_at": c.now_text(),
        }
        manifest_path = args.out_root / "launch_manifest.json"
        review_manifest = args.review_root / "launch_manifest.json"
        if manifest_path.exists():
            old = c.read_json(manifest_path)
            for key in ("variant", "protocol", "numerical_source_hashes", "orchestration_source_hashes", "pair_file_sha256"):
                if old.get(key) != manifest.get(key):
                    raise RuntimeError(f"Launch provenance drift: {key}")
        else:
            c.write_json(manifest_path, manifest)
            c.write_json(review_manifest, manifest)
        reused = seed_reused_first10(scheduler, args.reuse_root)
        scheduler.state["initial_free_mib"] = {
            row["physical_gpu"]: int(row["memory_free_mib"])
            for row in initial_gpu_state["rows"]
        }
        scheduler.state["phase"] = "smoke"
        scheduler.save()
        smoke = scheduler.run(all_specs(smoke=True))
        c.write_json(args.review_root / "smoke_results.json", {"passed": True, "count": 2, "results": smoke})
        scheduler.state["phase"] = "smoke_passed"
        scheduler.save()
        if args.stop_after == "smoke":
            return

        scheduler.state["phase"] = "production"
        scheduler.save()
        results = scheduler.run(all_specs())
        by_name = {row["spec"]["representation"] + ":" + row["spec"]["resolution"] + ":" +
                   str(row["spec"]["case_rank"]) + ":" + str(row["spec"]["alpha"]): row for row in results}
        for rank in range(100):
            for alpha in cohort.LAMBDAS:
                prefix = by_name["842:pyramid:" + str(rank) + ":" + str(alpha)]
                dense = by_name["842_dense:pyramid:" + str(rank) + ":" + str(alpha)]
                if dense["stages"][:3] != prefix["stages"] or dense["prefix_dvf_sha256"] != prefix["dvf_sha256"]:
                    raise RuntimeError(f"Prefix mismatch for rank{rank} lambda{alpha}")
        scheduler.state["phase"] = "references_and_aggregation"
        scheduler.save()
        references, reference_hashes = report.evaluate_references(
            oasis_root=args.oasis_root,
            pair_file=args.pair_file,
            dl_root=args.dl_root,
            reference_root=args.baseline_root,
            output_root=args.review_root / "references",
        )
        provenance = {
            "source_revision": source_metadata()["source_revision"],
            "launch_manifest": str(manifest_path),
            "launch_manifest_sha256": sha256_file(manifest_path),
            "reference_field_hashes": reference_hashes,
            "reused_accepted_first10": reused,
            "gpu_slots": list(args.gpu_slots),
        }
        report.write_report(results, references, args.review_root, provenance)
        scheduler.state["phase"] = "complete"
        scheduler.state["completed_at"] = c.now_text()
        scheduler.save()
        c.write_json(args.review_root / "ACCEPTED.json", {
            "technical_completion": True,
            "candidate_count": 400,
            "new_candidate_count": 360,
            "reused_accepted_first10_count": 40,
            "reference_count": 200,
            "completed_at": c.now_text(),
        })
        print(json.dumps({"complete": cohort.VARIANT, "candidates": 400, "references": len(references)}), flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    try:
        execute(arguments)
    except Exception as error:
        traceback.print_exc()
        raise
