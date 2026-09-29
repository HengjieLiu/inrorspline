"""Adopt an already-running, untimed BE-combination evaluation without restart.

The historical controller has a one-hour command watchdog. This narrowly scoped
recovery keeps the existing evaluator and frozen registration sources unchanged,
starts a replacement monitor BEFORE stopping only the verified host controller,
and retains all old evidence. It never launches registration or kills GPU jobs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from scripts.experiments.mrdbscp842_acceleration.run_be_combo_first10 import ComboSuite, EXPERIMENT
from scripts.experiments.mrdbscp842_acceleration.run_suite import dump, process_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("container", "repo-container", "artifact-host", "artifact-container",
                 "oasis-root", "run-id", "worker-name"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--controller-start-ticks", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    args = parser.parse_args()
    args.gpu, args.experiment_key = 0, EXPERIMENT
    if "/" in args.run_id or not args.run_id.startswith("mrdbscp842_be_combo_"):
        raise ValueError("Expected a dedicated BE-combination run ID")
    artifact = Path(args.artifact_host).resolve()
    root = artifact/"runs"/args.run_id
    if not (root/"diagnostic_DONE.json").is_file() or (root/"evaluation.json").exists():
        raise ValueError("Adoption requires completed registration and unfinished evaluation")
    suite = ComboSuite(args)
    suite.frozen()
    if [sum(r["block"].startswith(p) for r in suite.rows) for p in ("headline_", "diagnostic_")] != [33, 33]:
        raise ValueError("Expected 33+33 registration records")
    mail = (root/"workers"/args.worker_name).resolve()
    if mail.parent != (root/"workers").resolve():
        raise ValueError("Worker must belong to this run")
    command = json.loads((mail/"command_0000.json").read_text())
    expected = [(r["id"], r["rank"], r["dvf"]) for r in suite.rows if r["block"].startswith("headline_")]
    observed = [(r["id"], r["rank"], r["dvf"]) for r in command.get("items", [])]
    if command["action"] != "evaluate" or observed != expected:
        raise ValueError("Worker command does not match the 33 headline fields")
    job = root/"jobs"/Path(command["output"]).name
    if suite.container(job) != command["output"] or (job/"accepted_result.json").exists():
        raise ValueError("Unexpected evaluation output or already accepted job")
    owner = json.loads((mail/"host_ack.json").read_text())
    current = process_identity(owner["pid"])
    if current["start_ticks"] != owner["start_ticks"] or current["argv"] != owner["argv"]:
        raise ValueError("Evaluation worker identity changed")
    controller = process_identity(args.controller_pid)
    required = ("scripts.experiments.mrdbscp842_acceleration.run_be_combo_first10", args.run_id)
    if (controller["start_ticks"] != args.controller_start_ticks
            or not all(v in controller["argv"] for v in required)
            or not Path(controller["argv"][0]).name.startswith("python")):
        raise ValueError("Refusing to signal a different controller")
    recovery = root/"evaluation_watchdog_recovery"
    recovery.mkdir(exist_ok=False)
    source = Path(__file__)
    (recovery/source.name).write_bytes(source.read_bytes())
    dump(recovery/"manifest.json", dict(command=sys.argv, args=vars(args),
         controller=controller, worker=owner, original_evaluation_command=command,
         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         reason="Untimed existing CPU metrics exceed historical one-hour command watchdog; no registration or evaluator restart",
         utc=datetime.now(timezone.utc).isoformat()))
    suite.monitor.path = recovery/"gpu_monitor.jsonl"
    suite.monitor.owner = owner
    suite.monitor.sample()
    suite.monitor.thread.start()
    try:
        # The verified controller alone is replaced. Its docker-exec child and
        # CUDA evaluator continue unchanged; new monitoring already overlaps it.
        os.kill(args.controller_pid, signal.SIGTERM)
        dump(recovery/"controller_replaced.json", dict(unix=time.time(), signal="SIGTERM",
             pid=args.controller_pid, worker_interrupted=False))
        started, last, count = time.monotonic(), 0., -1
        result_file = job/"result.json"
        partial = job/"evaluation_partial.json"
        while not result_file.exists():
            actual = process_identity(owner["pid"])
            if actual["start_ticks"] != owner["start_ticks"]:
                raise RuntimeError("Evaluation worker identity changed while waiting")
            if time.monotonic()-started > args.timeout_seconds:
                raise TimeoutError("Recovery deadline exceeded; worker has NOT been killed")
            n = len(json.loads(partial.read_text())) if partial.exists() else 0
            if n != count or time.monotonic()-last >= 60:
                count, last = n, time.monotonic()
                progress = dict(evaluated=n, total=33, unix=time.time(),
                                waiting_sec=time.monotonic()-started)
                dump(recovery/"progress.json", progress)
                print(json.dumps(progress), flush=True)
            time.sleep(1)
        payload = json.loads(result_file.read_text())
        if not payload["ok"]:
            raise RuntimeError(payload.get("traceback", "Evaluation failed"))
        suite.monitor.sample()
        original = [json.loads(line) for line in (root/"gpu_monitor.jsonl").read_text().splitlines()]
        with suite.monitor.lock:
            suite.monitor.records = original+suite.monitor.records
        verdict = suite.monitor.interval(payload["start_unix"], time.time())
        dump(job/"monitor_verdict.json", verdict)
        if not verdict["clean"]:
            raise RuntimeError("Evaluation monitor was contaminated; retained result is not accepted")
        value = payload["result"]
        if len(value["rows"]) != 33:
            raise RuntimeError("Evaluation row count mismatch")
        value.update(id=job.name, action="evaluate", clean=True, monitor=verdict,
                     worker=args.worker_name, command=payload["command"],
                     monitor_recovery="evaluation_watchdog_recovery/manifest.json")
        dump(job/"accepted_result.json", value)
        suite.attempts.append(value)
        dump(root/"attempts.json", suite.attempts)
        dump(root/"evaluation.json", value)
        # The original controller was stopped before its deadline and never
        # overwrote command 0; the evaluator is now waiting for command 1.
        dump(mail/"command_0001.json", dict(action="exit"))
        deadline = time.monotonic()+120
        while Path(f"/proc/{owner['pid']}").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Evaluation completed but worker has not exited")
            time.sleep(.2)
        suite.monitor.owner = None
        suite.frozen()
        suite.source_checks("final_source_checks")
        suite.report()
        dump(root/"all_DONE.json", dict(utc=datetime.now(timezone.utc).isoformat(),
             recovery="evaluation_watchdog_recovery", registration_sources_unchanged=True))
        print("Evaluation, final contracts, and report completed", flush=True)
    except Exception as exc:
        dump(recovery/"ERROR.json", dict(error=repr(exc)))
        raise
    finally:
        suite.monitor.done.set()
        suite.monitor.thread.join(timeout=35)


if __name__ == "__main__":
    main()
