"""Four-GPU Fourier-bandwidth/diffusion search with immutable controls and gates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

import common as c
import idir_sampled_diffusion as old
import idir_fourier_diffusion as p
from launch_oasis_idir_sampled_diffusion import validate_result as validate_old, load_baselines
from sitreg_training import gpu_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("ARTIFACT_ROOT", "/database/inrorspline"))
    parser.add_argument("--out-root", type=Path, default=root / "runs" / p.VARIANT)
    parser.add_argument("--review-root", type=Path, default=root / "repo_experiments" / p.VARIANT)
    parser.add_argument("--prior-root", type=Path, default=root / "runs" / old.VARIANT)
    parser.add_argument("--prior-review", type=Path, default=root / "repo_experiments" / old.VARIANT)
    parser.add_argument("--baseline-root", type=Path, default=root / "repo_experiments/oasis_seed3_first10_raw_ncc_vfa_sitreg_arc_v1")
    parser.add_argument("--oasis-root", type=Path, required=True)
    parser.add_argument("--gpus", choices=("0,1,2,3",), default="0,1,2,3")
    parser.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    outputs = (args.out_root.resolve(), args.review_root.resolve())
    for output in outputs:
        for source in (c.REPO_ROOT, args.oasis_root, args.prior_root, args.prior_review, args.baseline_root):
            if output.is_relative_to(source.resolve()) or source.resolve().is_relative_to(output):
                parser.error("Outputs must not overlap code, datasets or prior evidence")
    if outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        parser.error("Run/review roots must be disjoint")
    return args


def validate_result(path, spec, hashes):
    path = Path(path)
    if not (path / "DONE").exists() or (path / "FAILED.txt").exists():
        raise RuntimeError("Missing clean completion")
    row = json.loads((path / "summary.json").read_text())
    if row["spec"] != asdict(spec) or row["protocol"] != p.PROTOCOL or row["source_sha256"] != hashes:
        raise RuntimeError("Identity/protocol/source mismatch")
    if (row["fixed_id"], row["moving_id"]) != c.parse_pair_file(c.DEFAULT_PAIR_FILE)[spec.rank] or row["direction"] != "moving_to_fixed":
        raise RuntimeError("Wrong case/direction")
    if row["actual_updates"] != spec.updates or row["network_training_coordinates"] != spec.updates*10000 or row["param_count"] != spec.param_count:
        raise RuntimeError("Architecture/work budget mismatch")
    if (row["uid"], row["gid"]) != (1009, 1020) or row["physical_gpu"] not in p.GPUS:
        raise RuntimeError("Wrong user/device")
    if row["backend"]["cudnn_benchmark"] or not row["backend"]["cudnn_deterministic"] or row["tf32_cudnn"] or row["tf32_matmul"]:
        raise RuntimeError("Backend changed")
    for name, digest in row["outputs_sha256"].items():
        if old.sha256(path / name) != digest:
            raise RuntimeError("Modified result artifact")
    if old.sha256(row["field_path"]) != row["field_sha256"] or old.sha256(path / "terminal.pt") != row["checkpoint_sha256"]:
        raise RuntimeError("Modified field/checkpoint")
    field = np.load(row["field_path"], mmap_mode="r")
    if field.shape != (*c.IMAGE_SHAPE, 3) or field.dtype != np.float32 or not np.isfinite(field).all():
        raise RuntimeError("Invalid field geometry/values")
    labels = json.loads((path / "per_label.json").read_text()); old.validate_label_records(labels, row)
    for label in labels:
        if any(label[k] != row[k] for k in ("rank", "family", "sigma", "width", "omega", "alpha", "seed", "fixed_id", "moving_id", "direction")):
            raise RuntimeError("Per-label metadata mismatch")
    history = json.loads((path / "history.json").read_text())
    if len(history) != spec.updates or [r["updates_completed"] for r in history] != list(range(1, spec.updates+1)):
        raise RuntimeError("Incomplete history")
    if any(r["sample_count"] != 10000 or r["lr"] != .0001 or not all(math.isfinite(r[k]) for k in ("loss", "data_loss", "diffusion")) for r in history):
        raise RuntimeError("Changed optimization")
    if spec.sigma and (not row["B0_frozen"] or not row["B0_sha256"]):
        raise RuntimeError("Invalid Fourier matrix")
    if spec.smoke_steps and not row["derivative_check"]["passed"]:
        raise RuntimeError("Coordinate derivative gate failed")
    return row


class Controller:
    def __init__(self, args):
        self.args, self.hashes = args, p.source_hashes()
        self.state_path = args.out_root / "controller_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "variant": p.VARIANT, "protocol": p.PROTOCOL, "source_sha256": self.hashes,
            "results": {}, "attempts": [], "failed": {}, "decisions": [],
            "sigmas": list(p.SIGMAS), "lambdas": list(p.LAMBDAS), "created_unix": time.time()}
        if self.state["source_sha256"] != self.hashes or self.state["protocol"] != p.PROTOCOL:
            raise RuntimeError("Source drift; do not mix versions")
        if any(a["status"] == "running" for a in self.state["attempts"]):
            raise RuntimeError("Reconcile previously running workers before resume")
        self.state["controller_pid"] = os.getpid()
        self.baselines, reference_hashes = load_baselines(args.baseline_root)
        acceptance = json.loads((args.prior_review / "ACCEPTED.json").read_text())
        if not acceptance["complete"] or not (args.prior_root / "DONE").exists():
            raise RuntimeError("Require accepted sampled-IDIR baseline")
        for filename, key in (("per_case.json", "per_case_sha256"), ("per_label_per_case.json", "per_label_sha256")):
            if old.sha256(args.prior_review / filename) != acceptance[key]:
                raise RuntimeError("Prior report changed")
        prior = json.loads((args.prior_root / "controller_state.json").read_text())
        self.reused, self.reused_labels = {}, {}
        for omega in (32, 64):
            for rank in range(10):
                prior_spec = old.FitSpec(rank, 0, omega)
                path = Path(prior["results"][prior_spec.name()])
                source = validate_old(path, prior_spec, old.source_hashes())
                spec = p.FitSpec(rank, 1., omega=omega)
                row = {**source, "spec": asdict(spec), "family": "plain", "sigma": 0., "width": 256, "features": 0,
                       "seed": 1, "provenance": "read_only_P53_reuse", "prior_protocol": source["protocol"]}
                self.reused[spec.name()] = row
                extra = {k:row[k] for k in ("family", "sigma", "width", "features", "seed")}
                self.reused_labels[spec.name()] = [{**r, **extra} for r in json.loads((path / "per_label.json").read_text())]
                for file in (path / "summary.json", path / "terminal.pt", path / "per_label.json", Path(source["field_path"])):
                    reference_hashes[str(file)] = old.sha256(file)
        if "reference_sha256" in self.state and self.state["reference_sha256"] != reference_hashes:
            raise RuntimeError("Reference drift")
        self.state["reference_sha256"] = reference_hashes

    def save(self):
        self.state["updated_unix"] = time.time(); old.write_json(self.state_path, self.state)

    def event(self, event, **detail):
        row = {"time": c.now_text(), "event": event, **detail}
        print(json.dumps(row), flush=True)
        with (self.args.out_root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")

    def check_row(self, row):
        if not row["spec"]["smoke_steps"]:
            reference = self.reused[p.FitSpec(row["rank"], 1.).name()]
            if row["sample_sequence_sha256"] != reference["sample_sequence_sha256"]:
                raise RuntimeError("Changed coordinate sequence versus original IDIR")
            if row["family"] == "plain" and row["width"] == 256 and row["omega"] == 32 and row["seed"] == 1:
                if row["initial_state_sha256"] != reference["initial_state_sha256"]:
                    raise RuntimeError("Plain baseline initialization drift")

    def execute(self, specs, phase, required=6.):
        queue, running = [], {}
        for spec in specs:
            spec.validate()
            if spec.name() in self.reused:
                continue
            if spec.name() in self.state["results"]:
                self.check_row(validate_result(self.state["results"][spec.name()], spec, self.hashes))
            elif spec.name() not in self.state["failed"]:
                queue.append(spec)
        self.state["phase"] = phase; self.save()
        last_monitor = 0.
        while queue or running:
            old.verify_hashes(self.hashes)
            for gpu, (process, spec, path, stream, attempt) in list(running.items()):
                code = process.poll()
                if code is None: continue
                stream.close(); del running[gpu]
                attempt.update(exit_code=code, finished_unix=time.time())
                try:
                    if code: raise RuntimeError(f"Worker exited {code}")
                    row = validate_result(path, spec, self.hashes); self.check_row(row)
                except Exception as exc:
                    log = path.with_suffix(".log").read_text(errors="replace")
                    oom = "CUDA out of memory" in log or "torch.OutOfMemoryError" in log
                    attempt.update(status="failed", error=str(exc), oom=oom)
                    count = sum(a["name"] == spec.name() for a in self.state["attempts"])
                    if oom and count == 1:
                        queue.append(spec); self.event("oom_retry_queued", name=spec.name(), preserved=str(path))
                    else:
                        self.state["failed"][spec.name()] = str(path); self.event("failed", name=spec.name(), error=str(exc))
                else:
                    attempt["status"] = "complete"; self.state["results"][spec.name()] = str(path)
                    self.event("complete", name=spec.name(), gpu=gpu, seconds=row["case_seconds"], metrics={k:row[k] for k in p.METRICS})
                    if not spec.smoke_steps: self.report()
                self.save()
            memory = {r["index"]:r for r in gpu_rows()}
            for gpu in p.GPUS:
                if not queue or gpu in running or memory[gpu]["free_mib"] < required*1024: continue
                spec = queue.pop(0)
                number = 1 + sum(a["name"] == spec.name() for a in self.state["attempts"])
                path = self.args.out_root / ("smoke" if spec.smoke_steps else "production") / spec.name() / f"attempt{number:02d}"
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.with_suffix(".log").open("x")
                command = [sys.executable, "-B", str(Path(__file__).with_name("run_oasis_idir_fourier_one.py")),
                           "--spec-json", json.dumps(asdict(spec)), "--gpu", str(gpu), "--cpu-threads", str(self.args.cpu_threads),
                           "--oasis-root", str(self.args.oasis_root), "--out-dir", str(path)]
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=memory[gpu]["uuid"],
                           OMP_NUM_THREADS=str(self.args.cpu_threads), MKL_NUM_THREADS=str(self.args.cpu_threads), PYTHONDONTWRITEBYTECODE="1")
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                attempt = {"name": spec.name(), "spec": asdict(spec), "phase": phase, "gpu": gpu, "pid": process.pid,
                           "path": str(path), "status": "running", "started_unix": time.time(), "gpu_before": memory[gpu], "command": command}
                self.state["attempts"].append(attempt); running[gpu] = (process, spec, path, stream, attempt)
                self.event("launch", **attempt); self.save()
            if time.monotonic()-last_monitor >= 15:
                live = []
                for gpu, (_, spec, path, _, _) in running.items():
                    progress = path / "progress.json"
                    live.append({"gpu": gpu, "name": spec.name(), "progress": json.loads(progress.read_text()) if progress.exists() else None})
                self.state.update(live=live, queued=len(queue)); self.save(); last_monitor = time.monotonic()
            if queue or running: time.sleep(3)
        self.state.update(live=[], queued=0); self.save()
        missing = [s.name() for s in specs if s.name() not in self.state["results"] and s.name() not in self.reused]
        if missing: raise RuntimeError(f"Incomplete phase; no selection: {missing}")
        return [self.reused[s.name()] if s.name() in self.reused else validate_result(self.state["results"][s.name()], s, self.hashes) for s in specs]

    def report(self):
        rows = list(self.reused.values()); labels = [r for group in self.reused_labels.values() for r in group]
        for path in self.state["results"].values():
            row = json.loads((Path(path) / "summary.json").read_text())
            if row["spec"]["smoke_steps"]: continue
            rows.append(row); labels.extend(json.loads((Path(path) / "per_label.json").read_text()))
        means = p.aggregate(rows)
        root = self.args.review_root; root.mkdir(parents=True, exist_ok=True)
        keys = ("spec", "rank", "family", "sigma", "features", "width", "omega", "alpha", "seed", "fixed_id", "moving_id", "direction",
                *p.METRICS, "param_count", "training_seconds", "case_seconds", "physical_gpu", "run_dir", "sample_sequence_sha256")
        concise = [{k:r[k] for k in keys} for r in rows]
        old.write_json(root / "per_case.json", concise); c.write_csv(root / "per_case.csv", concise)
        old.write_json(root / "per_label_per_case.json", labels); c.write_csv(root / "per_label_per_case.csv", labels)
        old.write_json(root / "means.json", means); c.write_csv(root / "means.csv", means)
        old.write_json(root / "decisions.json", self.state["decisions"])
        refs = []
        for name in ("VFA", "SITReg"):
            group = [r for r in self.baselines if r["representation"] == name]
            if len(group) != 10: raise RuntimeError("Incomplete reference")
            row = {"family": name, "n": 10}
            for metric, key in zip(p.METRICS, ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")):
                values = np.array([r[key] for r in group])
                row[metric+"_mean"], row[metric+"_sd"] = float(values.mean()), float(values.std(ddof=1))
            refs.append(row)
        old.write_json(root / "reference_means.json", refs)
        lines = ["# Fourier-input sampled IDIR: first10 exploratory tuning", "",
                 f"Completed new fits: {len(rows)-20}; read-only P53 references:20; total label rows:{len(labels)}.",
                 "10,000 coordinates/update,2500 Adam updates, LR1e-4; signed global1-NCC and original-coordinate analytic diffusion.",
                 "Native fan-in initialization; no silent rescaling. Only complete10case cohorts are averaged. All SDs are sample SDs.",
                 "Accuracy selection: highest mean Dice, ties lower ASD/HD95. Regularity is a separate reported trade-off.", "",
                 "| Family | Sigma | M | Width | Omega | Lambda | Seed | Dice ± SD | HD95 mm ± SD | ASD mm ± SD | FG std(logJ) ± SD | FG folding % ± SD |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in means:
            cells = [str(r[k]) for k in ("family", "sigma", "features", "width", "omega", "alpha", "seed")]
            cells += [f"{r[k+'_mean']:.6f} ± {r[k+'_sd']:.6f}" for k in p.METRICS]
            lines.append("| "+" | ".join(cells)+" |")
        for r in refs:
            lines.append("| "+" | ".join([r["family"], "—", "—", "—", "—", "1", "—"]+[f"{r[k+'_mean']:.6f} ± {r[k+'_sd']:.6f}" for k in p.METRICS])+" |")
        if "final_decision" in self.state:
            lines += ["", "## Selection", "", "```json", json.dumps(self.state["final_decision"], indent=2), "```"]
        (root / "report.md").write_text("\n".join(lines)+"\n")
        return rows, means, refs

    def plot(self, means, refs):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for ax, metric, title in zip(axes, p.METRICS[:3], ("Dice ↑", "HD95 mm ↓", "Symmetric ASD mm ↓")):
            for sigma in self.state["sigmas"]:
                group = sorted([r for r in means if r["sigma"] == sigma and r["features"] == 64 and r["seed"] == 1 and r["omega"] == 32], key=lambda r:r["alpha"])
                ax.plot([r["log_jdet_std_fg_mean"] for r in group], [r[metric+"_mean"] for r in group], "-o", label=f"Fourier sigma={sigma:g}")
            for width in (256, 286):
                group = sorted([r for r in means if r["family"] == "plain" and r["width"] == width and r["seed"] == 1 and r["omega"] == 32], key=lambda r:r["alpha"])
                ax.plot([r["log_jdet_std_fg_mean"] for r in group], [r[metric+"_mean"] for r in group], "--s", label=f"Plain width{width}")
            for r in refs: ax.scatter(r["log_jdet_std_fg_mean"], r[metric+"_mean"], marker="D", label=r["family"])
            ax.set_xlabel("Foreground std(log J)"); ax.set_ylabel(title); ax.grid(alpha=.2)
        axes[0].legend(fontsize=7); fig.tight_layout()
        for ext in ("png", "pdf", "svg"): fig.savefig(self.args.review_root / f"fourier_sweeps_lines.{ext}", dpi=180)
        plt.close(fig)

    def search(self):
        while True:
            sigmas, lambdas = self.state["sigmas"], self.state["lambdas"]
            self.execute(p.grid(sigmas, lambdas), "bandwidth_lambda_controls")
            rows, means, refs = self.report()
            decision = p.boundary_decision(means, sigmas, lambdas)
            decision["time"] = c.now_text()
            self.state["decisions"].append(decision); self.event("boundary_decision", **decision); self.save(); self.report()
            if decision["extra_sigma"] is None and not decision["extra_lambdas"]: break
            if decision["extra_sigma"] is not None:
                p.FitSpec(0, 1., decision["extra_sigma"]).validate()
                self.state["sigmas"] = sorted([*sigmas, decision["extra_sigma"]])
            for alpha in decision["extra_lambdas"]: p.FitSpec(0, alpha).validate()
            self.state["lambdas"] = sorted(set([*lambdas, *decision["extra_lambdas"]])); self.save()
        self.state["final_decision"] = {**decision, "promising": p.promising(decision),
                "followup_gate": "strictly higher mean Dice and no worse mean HD95/ASD/stdlogJ/folding than both independently selected controls"}
        old.write_json(self.args.review_root / "PRIMARY_SELECTED.json", self.state["final_decision"]); self.save()
        if p.promising(decision):
            self.execute(p.followups(decision), "conditional_omega_features_seed_confirmation")
        self.state["phase"] = "validating"; self.save()
        rows, means, refs = self.report()
        for row in rows:
            if row.get("provenance") != "read_only_P53_reuse":
                self.check_row(validate_result(row["run_dir"], p.FitSpec(**row["spec"]), self.hashes))
        old.verify_hashes(self.state["reference_sha256"])
        self.plot(means, refs)
        self.state["phase"] = "complete"; self.save(); self.report()
        old.write_json(self.args.review_root / "ACCEPTED.json", {"complete": True, "new_case_count": len(rows)-20,
            "reference_case_count": 20, "label_count": 35*len(rows), "source_sha256": self.hashes,
            "final_decision": self.state["final_decision"], "per_case_sha256": old.sha256(self.args.review_root / "per_case.json"),
            "per_label_sha256": old.sha256(self.args.review_root / "per_label_per_case.json"), "completed_unix": time.time()})
        (self.args.out_root / "DONE").write_text("Fourier search accepted\n")


def run(args):
    if (os.getuid(), os.getgid()) != (1009, 1020): raise RuntimeError("Require UID:GID1009:1020")
    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = Controller(args); controller.save()
        smokes = [p.FitSpec(0, .3, .5, smoke_steps=100), p.FitSpec(0, 3., 16., smoke_steps=100),
                  p.FitSpec(0, 1., width=286, smoke_steps=100), p.FitSpec(0, .1, smoke_steps=100)]
        rows = controller.execute(smokes, "qualification_four_gpus")
        if {r["physical_gpu"] for r in rows} != set(p.GPUS): raise RuntimeError("Require smoke on all4GPUs")
        if max(r["peak_reserved_gib"] for r in rows) > 10.: raise RuntimeError("Worker memory allowance needs review")
        # Reproduce the old plain100 smoke and prove independent sampling RNG parity.
        old_state = json.loads((args.prior_root / "controller_state.json").read_text())
        prior_spec = old.FitSpec(0, -1, 32, 100)
        prior = validate_old(old_state["results"][prior_spec.name()], prior_spec, old.source_hashes())
        current = rows[-1]
        if any(current[k] != prior[k] for k in ("initial_state_sha256", "sample_sequence_sha256", "first_sample_sha256")):
            raise RuntimeError("Plain smoke initialization/sampling parity failed")
        difference = float(np.max(np.abs(np.load(current["field_path"])-np.load(prior["field_path"])) ))
        if difference > 1e-5: raise RuntimeError(f"Plain smoke numerical reproduction failed:{difference}")
        warm = [statistics.median(v["seconds"] for v in json.loads((Path(r["run_dir"])/"history.json").read_text())[20:]) for r in rows]
        old.write_json(args.out_root / "SMOKE_PASSED.json", {"passed": True, "source_sha256": controller.hashes,
            "gpus": list(p.GPUS), "plain100_max_field_difference": difference,
            "initial_diagnostics": [r["initial_diagnostics"] for r in rows], "derivatives": [r["derivative_check"] for r in rows],
            "peak_reserved_gib": [r["peak_reserved_gib"] for r in rows], "median_update_seconds": warm,
            "initial230_compute_hours_estimate": 230*2500*statistics.mean(warm)/4/3600})
        if args.stop_after == "smoke":
            controller.state["phase"] = "smoke_complete"; controller.save(); return
        controller.search()


def main():
    args = parse_args(); args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "controller.log").open("a", buffering=1) as stream:
        sys.stdout = stream; sys.stderr = stream
        try: run(args)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
