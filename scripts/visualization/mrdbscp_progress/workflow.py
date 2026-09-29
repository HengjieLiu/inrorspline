"""Host-only sequential GPU-0 Case 04 orchestration; no host data dependencies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from scripts.experiments.mrdbscp842_acceleration.run_be_combo_first10 import ComboSuite
from scripts.experiments.mrdbscp842_acceleration.run_suite import Worker, dump
from .config import candidate_overrides, presentation


class ProgressSuite(ComboSuite):
    def __init__(self, args):
        self.presentation = presentation(args.profile)
        super().__init__(args)

    def hashes(self):
        result = super().hashes()
        for path in Path("scripts/visualization/mrdbscp_progress").glob("*.py"):
            # Presentation-only rendering can be iterated after registration;
            # numerical/capture/evaluation sources cannot change during the run.
            if path.stem in ("capture", "offline", "config", "geometry", "workflow"):
                result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def progress_provenance(self):
        self.provenance()
        target = self.root / "progress_source"
        for source in Path("scripts/visualization/mrdbscp_progress").glob("*"):
            if source.is_file():
                destination = target / source
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())

    def clean(self, worker, action, label, **kwargs):
        attempt = 0
        while (self.root / "jobs" / f"{label}_attempt{attempt}").exists():
            attempt += 1
        while True:
            self.monitor.wait_clear()
            result = worker.execute(action, f"{label}_attempt{attempt}", **kwargs)
            if result["clean"]:
                self.record(result, block=label, condition="warm", variant=self.presentation.candidate)
                print(json.dumps(dict(completed=label, seconds=result.get("elapsed_sec"), steps=result.get("steps"))), flush=True)
                return result
            attempt += 1

    def block_progress(self, name, smoke=False):
        worker = Worker(self, name, self.presentation.features)
        rows = []
        settings = dict(presentation_profile=self.presentation.profile,
                        candidate_overrides=candidate_overrides(self.presentation))
        try:
            if not smoke:
                self.clean(worker, "register", name + "_excluded_warmup", rank=3, **settings)
            for pair in range(1 if smoke else 3):
                for mode in ("baseline", "clock"):
                    self.frozen()
                    rows.append(self.clean(worker, "progress_register", f"{name}_pair{pair}_{mode}",
                        rank=3, progress_mode=mode, smoke=smoke, **settings))
            rows.append(self.clean(worker, "progress_register", name + "_capture", rank=3,
                                   progress_mode="capture", smoke=smoke, **settings))
        finally:
            worker.close()
        dump(self.root / f"{name}_rows.json", rows)
        return rows

    def evaluate_block(self, name, rows, smoke=False):
        # A new worker, with fixed evaluation flags and no accelerated backend.
        result = self.auxiliary("progress_evaluate", name + "_evaluation", rows=rows, rank=3, smoke=smoke)
        dump(self.root / f"{name}_evaluation.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("container", "repo-container", "artifact-host", "artifact-container", "oasis-root", "run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", type=int, choices=[0], default=0)
    parser.add_argument("--profile", choices=("adaptive_cpbe", "fixed50_cached_batch",
                                               "fixed50_cached_batch_dense_capture"),
                        default="adaptive_cpbe")
    parser.add_argument("--phase", choices=("dry-run", "run"), default="dry-run")
    args = parser.parse_args()
    args.experiment_key = ("2026-09-22_mrdbscp_fixed50_cached_batch_animation"
                           if args.profile.startswith("fixed50_cached_batch")
                           else "2026-09-20_mrdbscp_progress_animation")
    artifact = Path(args.artifact_host).resolve()
    repo = Path.cwd().resolve()
    if (not artifact.is_dir() or artifact == Path("/") or artifact.is_relative_to(repo)
            or repo.is_relative_to(artifact) or "/" in args.run_id
            or not args.run_id.startswith("mrdbscp_progress_case04_")):
        raise ValueError("Require dedicated external run ID and artifact root")
    suite = ProgressSuite(args)
    suite.progress_provenance()
    dump(suite.root / "job_manifest.json", dict(rank=3, case_number=4, fixed="0117", moving="0322",
        formal=["excluded_warmup", "baseline0", "clock0", "baseline1", "clock1", "baseline2", "clock2", "capture"],
        smoke=["baseline", "clock", "capture", "separate_evaluation"],
        profile=suite.presentation.to_dict(), features=list(suite.presentation.features), backend="F/T",
        selected_pair="median baseline, never fastest"))
    if args.phase == "dry-run":
        print(json.dumps(dict(root=str(suite.root), registration_launched=False)))
        return
    if (suite.root / "formal_rows.json").exists():
        raise FileExistsError("Use a new run ID; do not silently mix persistent-worker blocks")
    suite.monitor.thread.start()
    try:
        suite.frozen()
        suite.helper("cpu_tests", "unittest", "scripts.visualization.mrdbscp_progress.test_progress", "-v")
        environment = suite.auxiliary("environment", "preflight_environment")
        dump(suite.root / "data_environment.json", environment)
        smoke_rows = suite.block_progress("smoke", smoke=True)
        suite.evaluate_block("smoke", smoke_rows, smoke=True)
        formal = suite.block_progress("formal")
        evaluation = suite.evaluate_block("formal", formal)
        suite.frozen()
        dump(suite.root / "capture_DONE.json", dict(evaluation=evaluation, utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(completed="capture_and_offline_dice", evaluation=evaluation)), flush=True)
    except Exception as error:
        dump(suite.root / "ERROR.json", dict(error=repr(error)))
        raise
    finally:
        suite.monitor.done.set()
        suite.monitor.thread.join(timeout=35)


if __name__ == "__main__":
    main()
