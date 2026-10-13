"""Qualified, resumable two-GPU250-fit IDIR omega/lambda sweep controller."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

import common as c
import idir_ncc_diffusion as p
from sitreg_training import gpu_rows

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(c.REPO_ROOT / "scripts/reports"))
from oasis_dl_ncc_diffusion_first10_arc import load_baselines


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
    parser.add_argument("--out-root", type=Path, default=root / "runs" / p.VARIANT)
    parser.add_argument("--review-root", type=Path, default=root / "repo_experiments" / p.VARIANT)
    parser.add_argument("--baseline-root", type=Path, default=root / "repo_experiments/oasis_seed3_first10_raw_ncc_vfa_sitreg_arc_v1")
    parser.add_argument("--gpus", default="1,3", choices=("1,3",))
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    for name in ("oasis-root", "dl-code-root", "vfa-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = (args.out_root.resolve(), args.review_root.resolve())
    for output in outputs:
        if any(output.is_relative_to(source.resolve()) or source.resolve().is_relative_to(output)
               for source in (c.REPO_ROOT, args.oasis_root, args.dl_code_root, args.vfa_root, args.baseline_root)):
            parser.error("Outputs must be outside source/data trees")
    if outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        parser.error("Run and report roots must be disjoint")
    if args.cpu_threads < 1:
        parser.error("Require positive thread count")
    return args


def smoke_specs():
    # Cover all frequencies, both target GPUs, exact1/3 and largest regularizer;
    # ten updates on one worker provide a warm timing/memory estimate.
    return [p.FitSpec(0, omega, "10" if omega == 128 else "1over3", 10 if omega == 32 else 1)
            for omega in p.OMEGAS]


def validate_result(path, spec, hashes):
    path = Path(path)
    if not (path / "DONE").is_file() or (path / "FAILED.txt").exists():
        raise RuntimeError("Worker lacks clean completion")
    row = json.loads((path / "summary.json").read_text())
    if row["spec"] != asdict(spec) or row["protocol"] != p.PROTOCOL or row["source_sha256"] != hashes:
        raise RuntimeError("Wrong result protocol/spec/source provenance")
    pair = c.parse_pair_file(c.DEFAULT_PAIR_FILE)[spec.rank]
    if (row["fixed_id"], row["moving_id"]) != pair or row["direction"] != "moving_to_fixed":
        raise RuntimeError("Wrong subject identity or direction")
    if row["actual_updates"] != spec.updates or row["checkpoint_selection"] != "terminal":
        raise RuntimeError("Wrong optimization budget/checkpoint")
    if (row["uid"], row["gid"]) != (1009, 1020) or row["physical_gpu"] not in (1, 3):
        raise RuntimeError("Wrong worker identity/GPU")
    if row["backend"]["cudnn_benchmark"] or not row["backend"]["cudnn_deterministic"] or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("Wrong precision/backend")
    if spec.smoke_steps and not (row["native_parity"]["passed"] and row["chunk_parity"]["passed"]):
        raise RuntimeError("Missing smoke native-loss parity")
    for filename, key in ((Path(row["field_path"]).name, "field_sha256"), ("terminal.pt", "checkpoint_sha256"),
                          ("per_label.json", "per_label_sha256"), ("warped_seg.nii.gz", "warped_seg_sha256"),
                          ("history.json", "history_sha256")):
        if p.sha256(path / filename) != row[key]:
            raise RuntimeError(f"Output checksum mismatch: {filename}")
    field = np.load(row["field_path"], mmap_mode="r")
    if field.shape != (*c.IMAGE_SHAPE, 3) or field.dtype != np.float32 or not np.isfinite(field).all():
        raise RuntimeError("Invalid final field")
    if any(not math.isfinite(row[k]) for k in p.METRICS):
        raise RuntimeError("Nonfinite result")
    records = json.loads((path / "per_label.json").read_text())
    p.validate_label_records(records, row)
    for record in records:
        if any(record[key] != row[key] for key in ("rank", "omega", "lambda_key", "fixed_id", "moving_id", "direction")):
            raise RuntimeError("Per-label metadata mismatch")
    history = json.loads((path / "history.json").read_text())
    if len(history) != spec.updates + 1 or [r["state_index"] for r in history] != list(range(spec.updates + 1)):
        raise RuntimeError("Incomplete state-indexed loss history")
    if any(r["lr"] != 1e-4 or not all(math.isfinite(r[k]) for k in ("loss", "ncc9", "diffusion")) for r in history):
        raise RuntimeError("Changed LR or invalid loss history")
    return row


class Controller:
    def __init__(self, args):
        self.args = args
        self.hashes = p.source_hashes()
        self.path = args.out_root / "controller_state.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "variant": p.VARIANT, "source_sha256": self.hashes, "results": {}, "attempts": [], "failed": {},
            "created_unix": time.time(), "protocol": p.PROTOCOL}
        if self.state["source_sha256"] != self.hashes or self.state["protocol"] != p.PROTOCOL:
            raise RuntimeError("Controller provenance drift; preserve old attempt and use a new reviewed root")
        for attempt in self.state["attempts"]:
            if attempt["status"] == "running":
                raise RuntimeError("Prior running worker recorded; reconcile its actual process before resuming")
        self.state["controller_pid"] = os.getpid()
        self.baselines, self.reference_hashes = load_baselines(args.baseline_root)
        if "reference_sha256" in self.state and self.state["reference_sha256"] != self.reference_hashes:
            raise RuntimeError("Reference artifact provenance changed")
        self.state["reference_sha256"] = self.reference_hashes

    def save(self):
        self.state["updated_unix"] = time.time()
        p.write_json(self.path, self.state)

    def event(self, event, **extra):
        item = {"time": c.now_text(), "event": event, **extra}
        print(json.dumps(item), flush=True)
        with (self.args.out_root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")

    def command(self, spec, gpu, path):
        result = [sys.executable, "-B", str(HERE / "run_oasis_idir_ncc_diffusion_one.py"),
                  "--case-rank", str(spec.rank), "--omega", str(spec.omega), "--lambda-key", spec.lambda_key,
                  "--gpu", str(gpu), "--smoke-steps", str(spec.smoke_steps), "--out-dir", str(path),
                  "--cpu-threads", str(self.args.cpu_threads)]
        for name in ("oasis_root", "dl_code_root", "vfa_root", "pair_file"):
            result.extend(["--" + name.replace("_", "-"), str(getattr(self.args, name))])
        return result

    def execute(self, jobs, phase, required_gib):
        self.state["phase"] = phase
        queue = []
        for spec in jobs:
            if spec.name() in self.state["results"]:
                validate_result(self.state["results"][spec.name()], spec, self.hashes)
            elif spec.name() not in self.state["failed"]:
                queue.append(spec)
        running = {}
        last_report = 0.
        while queue or running:
            p.verify_hashes(self.hashes)
            for gpu, (process, spec, path, stream, attempt) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close(); del running[gpu]
                attempt.update(status="exited", exit_code=code, finished_unix=time.time())
                try:
                    if code:
                        raise RuntimeError(f"Worker exit code {code}")
                    row = validate_result(path, spec, self.hashes)
                except Exception as exc:
                    text = (path.with_suffix(".log")).read_text(errors="replace")
                    oom = "CUDA out of memory" in text or "torch.OutOfMemoryError" in text
                    previous = [a for a in self.state["attempts"] if a["name"] == spec.name()]
                    attempt.update(status="failed", error=str(exc), oom=oom)
                    if oom and len(previous) < 2:
                        queue.append(spec)
                        self.event("oom_retry_queued", name=spec.name(), preserved=str(path))
                    else:
                        self.state["failed"][spec.name()] = str(path)
                        self.event("failed", name=spec.name(), error=str(exc), preserved=str(path))
                else:
                    attempt["status"] = "complete"
                    self.state["results"][spec.name()] = str(path)
                    self.event("complete", name=spec.name(), gpu=gpu, seconds=row["training_seconds"],
                               metrics={k: row[k] for k in p.METRICS})
                    if phase == "production":
                        self.report(final=False)
                self.save()
            free = {r["index"]: r for r in gpu_rows()}
            for gpu in (1, 3):
                if not queue or gpu in running or free[gpu]["free_mib"] < required_gib * 1024:
                    continue
                spec = queue.pop(0)
                attempt_number = 1 + sum(a["name"] == spec.name() for a in self.state["attempts"])
                path = self.args.out_root / phase / spec.name() / f"attempt{attempt_number:02d}"
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.with_suffix(".log").open("x")
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=free[gpu]["uuid"],
                           OMP_NUM_THREADS=str(self.args.cpu_threads), MKL_NUM_THREADS=str(self.args.cpu_threads), PYTHONDONTWRITEBYTECODE="1")
                command = self.command(spec, gpu, path)
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                attempt = {"name": spec.name(), "spec": asdict(spec), "phase": phase, "gpu": gpu,
                           "pid": process.pid, "path": str(path), "status": "running", "command": command,
                           "started_unix": time.time(), "gpu_before": free[gpu]}
                self.state["attempts"].append(attempt)
                running[gpu] = (process, spec, path, stream, attempt)
                self.event("launch", **attempt); self.save()
            if time.monotonic() - last_report > 30:
                progress = []
                for gpu, (_, spec, path, _, _) in running.items():
                    file = path / "progress.json"
                    progress.append({"gpu": gpu, "name": spec.name(), "progress": json.loads(file.read_text()) if file.exists() else None})
                self.state["live"] = progress
                self.state["queued"] = len(queue)
                self.save(); last_report = time.monotonic()
            if queue or running:
                time.sleep(5)
        failed = [spec.name() for spec in jobs if spec.name() not in self.state["results"]]
        if failed:
            raise RuntimeError(f"Incomplete {phase}: {failed}")
        return [validate_result(self.state["results"][spec.name()], spec, self.hashes) for spec in jobs]

    def report(self, final):
        rows, labels = [], []
        keys = ("rank", "omega", "lambda_key", "alpha", "fixed_id", "moving_id", "direction", *p.METRICS,
                "diffusion_l2_full", "valid_label_count", "actual_updates", "physical_gpu", "training_seconds",
                "case_seconds", "best_loss_epoch_diagnostic", "terminal_loss", "run_dir", "field_sha256")
        for spec in p.specs():
            path = self.state["results"].get(spec.name())
            if path is None:
                continue
            source = Path(path)
            row = json.loads((source / "summary.json").read_text())
            rows.append({k: row[k] for k in keys})
            labels.extend(json.loads((source / "per_label.json").read_text()))
        root = self.args.review_root
        root.mkdir(parents=True, exist_ok=True)
        p.write_json(root / "per_case.json", rows); c.write_csv(root / "per_case.csv", rows)
        p.write_json(root / "per_label_per_case.json", labels); c.write_csv(root / "per_label_per_case.csv", labels)
        means = []
        for omega in p.OMEGAS:
            for key, alpha in p.LAMBDAS.items():
                subset = [r for r in rows if r["omega"] == omega and r["lambda_key"] == key]
                if len(subset) != 10:
                    continue  # No incomplete-cohort means.
                result = {"omega": omega, "lambda_key": key, "alpha": alpha, "n": 10}
                for metric in p.METRICS:
                    values = np.asarray([r[metric] for r in subset], dtype=np.float64)
                    result[metric + "_mean"] = float(values.mean())
                    result[metric + "_sample_sd"] = float(values.std(ddof=1))
                means.append(result)
        p.write_json(root / "means.json", means); c.write_csv(root / "means.csv", means)
        reference_means = []
        for group in ("VFA", "SITReg"):
            subset = [r for r in self.baselines if r["representation"] == group]
            if len(subset) != 10:
                raise RuntimeError("Incomplete reference cohort")
            result = {"group": group, "n": 10}
            for metric, source in zip(p.METRICS, ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")):
                values = np.asarray([r[source] for r in subset], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise RuntimeError("Invalid reference metrics")
                result[metric + "_mean"] = float(values.mean())
                result[metric + "_sample_sd"] = float(values.std(ddof=1))
            reference_means.append(result)
        p.write_json(root / "references_per_case.json", self.baselines)
        p.write_json(root / "reference_means.json", reference_means)
        c.write_csv(root / "reference_means.csv", reference_means)
        lines = ["# IDIR SIREN omega/lambda sweep", "", f"Completed: {len(rows)}/250 cases; {len(labels)}/8750 label records.",
                 "", "Primary field: update2500. First10 test-case tuning is exploratory. No incomplete-cohort means.",
                 "Foreground std(logJ) excludes nonpositive J; folding is a percentage, reported separately.", "",
                 "| Omega | Lambda | N | Dice mean ± SD | HD95 mm mean ± SD | ASD mm mean ± SD | FG std(logJ) mean ± SD | FG folding % mean ± SD |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in means:
            cells = [str(row["omega"]), row["lambda_key"], "10"]
            cells += [f"{row[k + '_mean']:.6f} ± {row[k + '_sample_sd']:.6f}" for k in p.METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        for row in reference_means:
            cells = [row["group"], "historical1", "10"]
            cells += [f"{row[k + '_mean']:.6f} ± {row[k + '_sample_sd']:.6f}" for k in p.METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if final:
            if len(rows) != 250 or len(labels) != 8750 or len(means) != 25:
                raise RuntimeError("Final report requires250 cases,8750 labels,25 complete means")
            for omega in p.OMEGAS:
                hashes = {json.loads((Path(self.state["results"][s.name()]) / "summary.json").read_text())["initial_state_sha256"]
                          for s in p.specs() if s.omega == omega}
                if len(hashes) != 1:
                    raise RuntimeError("Candidates did not share initialization within frequency")
            p.verify_hashes(self.reference_hashes)
            self.plot(means, reference_means, root)
            p.write_json(root / "ACCEPTED.json", {"complete": True, "cases": 250, "label_rows": 8750,
                         "cohorts": 25, "source_sha256": self.hashes,
                         "per_case_sha256": p.sha256(root / "per_case.json"),
                         "per_label_sha256": p.sha256(root / "per_label_per_case.json"), "completed_unix": time.time()})

    @staticmethod
    def plot(means, references, root):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for ax, metric, title in zip(axes, p.METRICS[:3], ("Dice ↑", "HD95 (mm) ↓", "Symmetric ASD (mm) ↓")):
            for omega in p.OMEGAS:
                rows = sorted([r for r in means if r["omega"] == omega], key=lambda r: r["alpha"])
                ax.plot([r["log_jdet_std_fg_mean"] for r in rows], [r[metric + "_mean"] for r in rows], "-o", label=f"ω={omega}")
                for row in rows:
                    ax.annotate(row["lambda_key"].replace("over", "/"), (row["log_jdet_std_fg_mean"], row[metric + "_mean"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            for row, marker in zip(references, ("D", "s")):
                ax.scatter(row["log_jdet_std_fg_mean"], row[metric + "_mean"], marker=marker, s=70, label=row["group"], zorder=5)
            ax.set_xlabel("Foreground std(log J)"); ax.set_ylabel(title); ax.grid(alpha=.2)
        axes[0].legend(); fig.tight_layout()
        for extension in ("png", "svg", "pdf"):
            fig.savefig(root / f"omega_lambda_arc_lines.{extension}", dpi=180)
        plt.close(fig)


def run(args):
    if args.dry_run:
        print(json.dumps({"jobs": [asdict(s) for s in p.specs()], "gpus": [1, 3], "protocol": p.PROTOCOL}, indent=2))
        return
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require container UID:GID1009:1020")
    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = Controller(args)
        controller.save()
        smoke_rows = controller.execute(smoke_specs(), "smoke", required_gib=12.)
        required = max(12., max(r["peak_reserved_gib"] for r in smoke_rows) + 2.)
        certificate = {"passed": True, "source_sha256": controller.hashes, "required_free_gib": required,
                       "smoke_runs": [r["run_dir"] for r in smoke_rows], "smoke_gpus": sorted({r["physical_gpu"] for r in smoke_rows}),
                       "max_reserved_gib": max(r["peak_reserved_gib"] for r in smoke_rows), "created_unix": time.time()}
        if certificate["smoke_gpus"] != [1, 3]:
            raise RuntimeError("Both approved GPUs must pass a smoke")
        p.write_json(args.out_root / "SMOKE_PASSED.json", certificate)
        if args.stop_after == "smoke":
            controller.state["phase"] = "smoke_complete"; controller.save()
            return
        controller.execute(p.specs(), "production", required)
        controller.report(final=True)
        controller.state["phase"] = "complete"; controller.save()
        (args.out_root / "DONE").write_text("250 fits and8750 label records accepted\n")


def main():
    args = parse_args()
    if args.dry_run:
        run(args)
        return
    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "controller.log").open("a", buffering=1) as stream:
        sys.stdout = stream; sys.stderr = stream
        try:
            run(args)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
