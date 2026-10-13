"""Four-slot monitored lambda-first IDIR search, then omega at frozen lambda."""
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
import idir_sampled_diffusion as p
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
    parser.add_argument("--be-root", type=Path, default=root / "runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1/idir_original_dense_siren_be_autograd")
    parser.add_argument("--oasis-root", type=Path, required=True)
    parser.add_argument("--gpu-slots", choices=("1,1,3,3",), default="1,1,3,3")
    parser.add_argument("--stop-after", choices=("smoke", "full"), default="full")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    outputs = (args.out_root.resolve(), args.review_root.resolve())
    for output in outputs:
        for source in (c.REPO_ROOT, args.oasis_root, args.baseline_root, args.be_root):
            if output.is_relative_to(source.resolve()) or source.resolve().is_relative_to(output):
                parser.error("Outputs must be disjoint from source/data/reference trees")
    if outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        parser.error("Run/review roots must be disjoint")
    return args


def validate_result(path, spec, hashes):
    path = Path(path)
    if not (path / "DONE").exists() or (path / "FAILED.txt").exists():
        raise RuntimeError("Missing clean worker completion")
    row = json.loads((path / "summary.json").read_text())
    if row["spec"] != asdict(spec) or row["protocol"] != p.PROTOCOL or row["source_sha256"] != hashes:
        raise RuntimeError("Candidate identity/protocol/source mismatch")
    if (row["fixed_id"], row["moving_id"]) != c.parse_pair_file(c.DEFAULT_PAIR_FILE)[spec.rank] or row["direction"] != "moving_to_fixed":
        raise RuntimeError("Wrong case/direction")
    if row["actual_updates"] != spec.updates or row["network_training_coordinates"] != spec.updates * 10000:
        raise RuntimeError("Changed optimizer/evaluated-coordinate budget")
    if row["backend"]["cudnn_benchmark"] or not row["backend"]["cudnn_deterministic"] or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("Changed backend")
    if (row["uid"], row["gid"]) != (1009, 1020) or p.SLOTS[row["slot"]] != row["physical_gpu"]:
        raise RuntimeError("Wrong worker user/GPU slot")
    for name, checksum in row["outputs_sha256"].items():
        if p.sha256(path / name) != checksum:
            raise RuntimeError("Modified metric/history artifact")
    if p.sha256(row["field_path"]) != row["field_sha256"] or p.sha256(path / "terminal.pt") != row["checkpoint_sha256"]:
        raise RuntimeError("Modified field/checkpoint")
    field = np.load(row["field_path"], mmap_mode="r")
    if field.shape != (*c.IMAGE_SHAPE, 3) or field.dtype != np.float32 or not np.isfinite(field).all():
        raise RuntimeError("Invalid dense field")
    labels = json.loads((path / "per_label.json").read_text())
    p.validate_label_records(labels, row)
    for label in labels:
        if any(label[k] != row[k] for k in ("rank", "omega", "exponent", "fixed_id", "moving_id", "direction")):
            raise RuntimeError("Label metadata mismatch")
    history = json.loads((path / "history.json").read_text())
    if len(history) != spec.updates or [r["updates_completed"] for r in history] != list(range(1, spec.updates + 1)):
        raise RuntimeError("Incomplete update history")
    if any(r["sample_count"] != 10000 or r["lr"] != .0001 or not all(math.isfinite(r[k]) for k in ("loss", "data_loss", "diffusion")) for r in history):
        raise RuntimeError("Changed sampled optimizer protocol")
    return row


class Controller:
    def __init__(self, args):
        self.args, self.hashes = args, p.source_hashes()
        self.state_path = args.out_root / "controller_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "variant": p.VARIANT, "protocol": p.PROTOCOL, "source_sha256": self.hashes,
            "results": {}, "attempts": [], "failed": {}, "decisions": [],
            "exponents": list(p.INITIAL_EXPONENTS), "created_unix": time.time()}
        if self.state["source_sha256"] != self.hashes or self.state["protocol"] != p.PROTOCOL:
            raise RuntimeError("Source/protocol changed; do not mix candidate implementations")
        if any(a["status"] == "running" for a in self.state["attempts"]):
            raise RuntimeError("Reconcile prior recorded running workers before resuming")
        self.state["controller_pid"] = os.getpid()
        self.baselines, hashes = load_baselines(args.baseline_root)
        if "reference_sha256" in self.state and self.state["reference_sha256"] != hashes:
            raise RuntimeError("Reference drift")
        self.state["reference_sha256"] = hashes

    def save(self):
        self.state["updated_unix"] = time.time()
        p.write_json(self.state_path, self.state)

    def event(self, event, **detail):
        row = {"time": c.now_text(), "event": event, **detail}
        print(json.dumps(row), flush=True)
        with (self.args.out_root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def command(self, spec, slot, path):
        return [sys.executable, "-B", str(HERE / "run_oasis_idir_sampled_diffusion_one.py"),
                "--rank", str(spec.rank), "--exponent", str(spec.exponent), "--omega", str(spec.omega),
                "--gpu", str(p.SLOTS[slot]), "--slot", str(slot), "--smoke-steps", str(spec.smoke_steps),
                "--cpu-threads", str(self.args.cpu_threads), "--oasis-root", str(self.args.oasis_root), "--out-dir", str(path)]

    def monitor(self, running, queue, phase):
        live, etas = [], []
        for slot, (_, spec, path, _, _) in running.items():
            progress = path / "progress.json"
            row = json.loads(progress.read_text()) if progress.exists() else None
            live.append({"slot": slot, "gpu": p.SLOTS[slot], "name": spec.name(), "progress": row})
            if row and row["updates_completed"] >= 50:
                rate = row["elapsed_seconds"] / row["updates_completed"]
                etas.append((spec.updates - row["updates_completed"]) * rate + 40.)
        self.state.update(phase=phase, queued=len(queue), live=live)
        self.state["current_wave_eta_seconds"] = max(etas) if etas else None
        self.save()

    def execute(self, specs, phase, required_gib):
        queue, running = [], {}
        for spec in specs:
            if spec.name() in self.state["results"]:
                validate_result(self.state["results"][spec.name()], spec, self.hashes)
            elif spec.name() not in self.state["failed"]:
                queue.append(spec)
        self.state["phase"] = phase; self.save()
        last_monitor = 0.
        while queue or running:
            p.verify_hashes(self.hashes)
            for slot, (process, spec, path, stream, attempt) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close(); del running[slot]
                attempt.update(exit_code=code, finished_unix=time.time())
                try:
                    if code:
                        raise RuntimeError(f"Worker exit {code}")
                    row = validate_result(path, spec, self.hashes)
                except Exception as exc:
                    text = path.with_suffix(".log").read_text(errors="replace")
                    oom = "CUDA out of memory" in text or "torch.OutOfMemoryError" in text
                    count = sum(a["name"] == spec.name() for a in self.state["attempts"])
                    attempt.update(status="failed", error=str(exc), oom=oom)
                    if oom and count == 1:
                        queue.append(spec)
                        self.event("oom_retry_queued", name=spec.name(), preserved=str(path))
                    else:
                        self.state["failed"][spec.name()] = str(path)
                        self.event("failed", name=spec.name(), reason=str(exc))
                else:
                    attempt["status"] = "complete"
                    self.state["results"][spec.name()] = str(path)
                    self.event("complete", name=spec.name(), slot=slot, seconds=row["case_seconds"], metrics={k: row[k] for k in p.METRICS})
                    if not spec.smoke_steps:
                        self.report()
                self.save()
            memory = {r["index"]: r for r in gpu_rows()}
            reserved_new = {1: 0., 3: 0.}
            for slot, gpu in enumerate(p.SLOTS):
                if not queue or slot in running:
                    continue
                # Recheck availability and account for workers started in this polling pass.
                if memory[gpu]["free_mib"] - reserved_new[gpu] < required_gib * 1024:
                    continue
                spec = queue.pop(0)
                number = 1 + sum(a["name"] == spec.name() for a in self.state["attempts"])
                path = self.args.out_root / ("smoke" if spec.smoke_steps else "production") / spec.name() / f"attempt{number:02d}"
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.with_suffix(".log").open("x")
                command = self.command(spec, slot, path)
                env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=memory[gpu]["uuid"],
                           OMP_NUM_THREADS=str(self.args.cpu_threads), MKL_NUM_THREADS=str(self.args.cpu_threads), PYTHONDONTWRITEBYTECODE="1")
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                attempt = {"name": spec.name(), "spec": asdict(spec), "phase": phase, "slot": slot, "gpu": gpu,
                           "pid": process.pid, "path": str(path), "status": "running", "started_unix": time.time(),
                           "gpu_before": memory[gpu], "command": command}
                self.state["attempts"].append(attempt); running[slot] = (process, spec, path, stream, attempt)
                reserved_new[gpu] += required_gib * 1024
                self.event("launch", **attempt); self.save()
            if time.monotonic() - last_monitor >= 15:
                self.monitor(running, queue, phase); last_monitor = time.monotonic()
            if queue or running:
                time.sleep(3)
        self.monitor({}, [], phase)
        missing = [s.name() for s in specs if s.name() not in self.state["results"]]
        if missing:
            raise RuntimeError(f"Incomplete phase; no selection allowed: {missing}")
        return [validate_result(self.state["results"][s.name()], s, self.hashes) for s in specs]

    def report(self):
        rows, labels = [], []
        for path in self.state["results"].values():
            source = Path(path)
            row = json.loads((source / "summary.json").read_text())
            if row["spec"]["smoke_steps"]:
                continue
            keys = ("rank", "omega", "exponent", "alpha", "fixed_id", "moving_id", "direction", *p.METRICS,
                    "diffusion_l2_full", "valid_label_count", "training_seconds", "case_seconds", "physical_gpu", "slot",
                    "sample_sequence_sha256", "initial_state_sha256", "run_dir", "field_sha256")
            rows.append({k: row[k] for k in keys})
            labels.extend(json.loads((source / "per_label.json").read_text()))
        rows.sort(key=lambda r: (r["omega"], r["exponent"], r["rank"]))
        means = p.aggregate(rows)
        root = self.args.review_root; root.mkdir(parents=True, exist_ok=True)
        p.write_json(root / "per_case.json", rows); c.write_csv(root / "per_case.csv", rows)
        p.write_json(root / "per_label_per_case.json", labels); c.write_csv(root / "per_label_per_case.csv", labels)
        p.write_json(root / "means.json", means); c.write_csv(root / "means.csv", means)
        references = []
        for group in ("VFA", "SITReg"):
            subset = [r for r in self.baselines if r["representation"] == group]
            if len(subset) != 10:
                raise RuntimeError("Incomplete reference cohort")
            row = {"group": group, "n": 10}
            for metric, key in zip(p.METRICS, ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")):
                values = np.array([r[key] for r in subset])
                row[metric + "_mean"], row[metric + "_sample_sd"] = float(values.mean()), float(values.std(ddof=1))
            references.append(row)
        p.write_json(root / "reference_means.json", references); c.write_csv(root / "reference_means.csv", references)
        p.write_json(root / "decisions.json", self.state["decisions"])
        lines = ["# Sampled-IDIR diffusion-first, then frequency search", "", f"Completed registrations: {len(rows)}; label records: {len(labels)}.",
                 "", "Exploratory first10 test tuning.2500updates,10000foreground samples/update. Signed GLOBAL1-NCC, not DL local NCC9.",
                 "Accuracy winner: highest mean Dice, then lower ASD/HD95. Regularity is reported separately, not a hidden constraint.",
                 "Only complete10case cohorts are averaged. Omega sweep holds the omega32-selected lambda fixed.", "",
                 "| Omega | Lambda | Dice mean ± SD | HD95 mm mean ± SD | ASD mm mean ± SD | FG std(logJ) mean ± SD | FG folding % mean ± SD |",
                 "|---:|---:|---:|---:|---:|---:|---:|"]
        for row in means:
            cells = [str(row["omega"]), f"1e{row['exponent']:+d}"] + [f"{row[k+'_mean']:.6f} ± {row[k+'_sample_sd']:.6f}" for k in p.METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        for row in references:
            lines.append("| " + " | ".join([row["group"], "historical1"] + [f"{row[k+'_mean']:.6f} ± {row[k+'_sample_sd']:.6f}" for k in p.METRICS]) + " |")
        (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return rows, means, references

    def validate_cohort_reproducibility(self, rows):
        for rank in range(10):
            subset = [r for r in rows if r["rank"] == rank]
            if len({r["sample_sequence_sha256"] for r in subset}) != 1:
                raise RuntimeError("Different coordinate sequence across candidates of one case")
        for omega in {r["omega"] for r in rows}:
            if len({r["initial_state_sha256"] for r in rows if r["omega"] == omega}) != 1:
                raise RuntimeError("Different initialization within frequency")

    def search(self, required):
        exponents = self.state["exponents"]
        while True:
            self.execute(p.cohort(exponents), "lambda_omega32", required)
            rows, means, refs = self.report()
            self.validate_cohort_reproducibility(rows)
            decision = p.boundary_decision(means, exponents)
            decision.update(time=c.now_text(), tested_exponents=list(exponents), phase="lambda_omega32")
            if not self.state["decisions"] or self.state["decisions"][-1].get("tested_exponents") != list(exponents):
                self.state["decisions"].append(decision)
                self.event("boundary_decision", **decision)
            self.save(); self.report()
            if decision["extend_exponent"] is None:
                chosen = decision["selected"]["exponent"]
                self.state["selected_exponent"] = chosen
                self.save()
                p.write_json(self.args.review_root / "LAMBDA_SELECTED.json", {**decision, "source_sha256": self.hashes})
                break
            exponents = sorted([*exponents, decision["extend_exponent"]])
            self.state["exponents"] = exponents; self.save()
        # Omega32 reuses only the just-validated identical-protocol selected cohort.
        omega_jobs = [p.FitSpec(rank, chosen, omega) for omega in p.OMEGAS for rank in range(10)]
        self.execute(omega_jobs, "omega_frozen_lambda", required)
        rows, means, refs = self.report()
        self.validate_cohort_reproducibility(rows)
        selected_omega = p.select([r for r in means if r["exponent"] == chosen])
        self.state["selected_omega"] = selected_omega["omega"]
        self.state["phase"] = "complete"; self.save()
        expected = 10 * len(exponents) + 40
        if len(rows) != expected:
            raise RuntimeError("Incomplete staged search")
        p.verify_hashes(self.state["reference_sha256"])
        self.plot(means, refs, chosen)
        self.report()
        p.write_json(self.args.review_root / "ACCEPTED.json", {
            "complete": True, "case_count": expected, "label_count": 35 * expected,
            "tested_exponents": exponents, "selected_exponent_at_omega32": chosen,
            "selected_omega_at_frozen_lambda": selected_omega,
            "source_sha256": self.hashes, "decisions": self.state["decisions"],
            "per_case_sha256": p.sha256(self.args.review_root / "per_case.json"),
            "per_label_sha256": p.sha256(self.args.review_root / "per_label_per_case.json"), "completed_unix": time.time()})
        (self.args.out_root / "DONE").write_text("Both staged searches accepted\n")

    def plot(self, means, refs, chosen):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for row_index, family in enumerate((sorted([r for r in means if r["omega"] == 32], key=lambda r:r["exponent"]),
                                             sorted([r for r in means if r["exponent"] == chosen], key=lambda r:r["omega"]))):
            for ax, key, title in zip(axes[row_index], p.METRICS[:3], ("Dice ↑", "HD95 mm ↓", "Symmetric ASD mm ↓")):
                ax.plot([r["log_jdet_std_fg_mean"] for r in family], [r[key+"_mean"] for r in family], "-o", label="lambda sweep at omega32" if row_index == 0 else "omega sweep at fixed lambda")
                for r in family:
                    label = f"1e{r['exponent']:+d}" if row_index == 0 else f"ω{r['omega']}"
                    ax.annotate(label, (r["log_jdet_std_fg_mean"], r[key+"_mean"]), xytext=(3, 3), textcoords="offset points", fontsize=8)
                for r, marker in zip(refs, ("D", "s")):
                    ax.scatter(r["log_jdet_std_fg_mean"], r[key+"_mean"], marker=marker, label=r["group"])
                ax.set_xlabel("Foreground std(log J)"); ax.set_ylabel(title); ax.grid(alpha=.2)
            axes[row_index, 0].legend()
        fig.tight_layout()
        for ext in ("png", "svg", "pdf"):
            fig.savefig(self.args.review_root / f"staged_sweeps_lines.{ext}", dpi=180)
        plt.close(fig)


def run(args):
    if (os.getuid(), os.getgid()) != (1009, 1020):
        raise RuntimeError("Require UID:GID1009:1020")
    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = Controller(args); controller.save()
        calibration = args.out_root / "calibration"
        if not (calibration / "DONE").exists():
            with (args.out_root / "calibration.log").open("a") as log:
                subprocess.run([sys.executable, "-B", str(HERE / "calibrate_oasis_idir_sampled_diffusion.py"),
                                "--oasis-root", str(args.oasis_root), "--be-root", str(args.be_root), "--out", str(calibration)], stdout=log, stderr=subprocess.STDOUT, check=True)
        diagnostic = json.loads((calibration / "calibration.json").read_text())
        if not diagnostic["complete"] or diagnostic["source_sha256"] != controller.hashes:
            raise RuntimeError("Calibration provenance mismatch")
        p.verify_hashes(diagnostic["historical_sha256"])
        smokes = [p.FitSpec(0, exponent, 32, 100) for exponent in (-3, -1, 1, 3)]
        smoke_rows = controller.execute(smokes, "smoke_four_concurrent", 12.)
        if {r["slot"] for r in smoke_rows} != set(range(4)):
            raise RuntimeError("Require one smoke in every concurrent slot")
        required = max(6., max(r["peak_reserved_gib"] for r in smoke_rows) + 2.)
        if max(r["peak_reserved_gib"] for r in smoke_rows) > 10.:
            raise RuntimeError("Two-worker memory allowance requires review")
        warm = []
        for r in smoke_rows:
            history = json.loads((Path(r["run_dir"]) / "history.json").read_text())
            warm.append(statistics.median(x["seconds"] for x in history[20:]))
        p.write_json(args.out_root / "SMOKE_PASSED.json", {
            "passed": True, "source_sha256": controller.hashes, "slots": list(p.SLOTS),
            "runs": [r["run_dir"] for r in smoke_rows], "required_free_gib_per_worker": required,
            "warm_median_update_seconds_by_slot": warm,
            "initial70_compute_hours_estimate": 70 * 2500 * statistics.mean(warm) / 4 / 3600,
            "initial110_compute_hours_estimate": 110 * 2500 * statistics.mean(warm) / 4 / 3600})
        if args.stop_after == "smoke":
            controller.state["phase"] = "smoke_complete"; controller.save()
            return
        controller.search(required)


def main():
    args = parse_args(); args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "controller.log").open("a", buffering=1) as stream:
        sys.stdout = stream; sys.stderr = stream
        try:
            run(args)
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
