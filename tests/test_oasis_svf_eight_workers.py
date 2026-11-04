from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
import queue_oasis_svf_eight_workers as q


class EightWorkerTests(unittest.TestCase):
    def test_capacity_two_per_gpu_eight_total_and_memory(self):
        active = {}
        for gpu in q.GPUS:
            for _ in range(2):
                self.assertTrue(q.has_capacity(gpu, active, 20000, 8192))
                active[len(active)] = {"job": {"gpu": gpu}}
            self.assertFalse(q.has_capacity(gpu, active, 20000, 8192))
        self.assertEqual(len(active), 8)
        self.assertFalse(q.has_capacity("0", {}, 8191, 8192))
        self.assertFalse(q.has_capacity("4", {}, 30000, 8192))

    def test_cli_and_unchanged_numerical_entry(self):
        args = q.parse_args(["--gpus", "0,1,2,3", "--max-concurrent", "8", "--workers-per-gpu", "2"])
        self.assertEqual((args.max_concurrent, args.workers_per_gpu), (8, 2))
        with self.assertRaises(SystemExit): q.parse_args(["--max-concurrent", "9"])
        self.assertEqual(q.WORKER.name, "run_oasis_bspline_svf_sweep_recovery.py")
        self.assertEqual(len(q.r.s.matrix()), 280)

    def test_priority_affinity_failure_and_oom_wait(self):
        spec = q.r.s.Spec(0, "842", "velocity", 1.)
        pending = {"spec": spec, "status": "pending", "gpu": None}
        audit = {"spec": spec, "status": "audit_pending", "gpu": "3"}
        retry = {"spec": spec, "status": "retry", "gpu": "1", "retry_after": 10}
        self.assertIs(q.choose_job([pending, audit], "0", {}, [], 10), audit)
        self.assertIsNone(q.choose_job([pending], "0", {}, ["failure"], 10))
        self.assertIsNone(q.choose_job([retry], "0", {}, [], 10))
        self.assertIs(q.choose_job([retry], "0", {}, [], 11), retry)

    def test_resume_adopts_worker_without_overwriting(self):
        spec = q.r.s.Spec(3, "842", "velocity", .01)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/spec.name()/"attempt1"
            path.mkdir(parents=True); marker = path/"progress.json"; marker.write_text("preserve")
            live = {"pid": 123, "start_ticks": "999", "action": "worker", "gpu": "3", "out": str(path)}
            manifest = {"recovery": {"approved_retry": {}}}
            with patch.object(q.r.s, "matrix", return_value=[spec]):
                jobs, active, affinity = q.resume_jobs(root, manifest, [live])
            self.assertEqual(jobs[0]["status"], "running")
            self.assertEqual(active[123]["record"], live)
            self.assertEqual(affinity, {3: "3"})
            self.assertEqual(marker.read_text(), "preserve")
            self.assertFalse((path.parent/"attempt2").exists())

    def test_resume_skips_completed_and_handles_exiting_auditor(self):
        spec = q.r.s.Spec(0, "842", "velocity", 1.)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/spec.name()/"attempt0"
            path.mkdir(parents=True); (path/"independent_audit.json").write_text("preserve")
            manifest = {"recovery": {"approved_retry": {}}}
            with patch.object(q.r.s, "matrix", return_value=[spec]), patch.object(q.r, "validate", return_value={"gpu": "2"}):
                jobs, active, _ = q.resume_jobs(root, manifest, [])
                self.assertEqual(jobs[0]["status"], "complete")
                self.assertFalse(active)
                live = {"pid": 123, "action": "audit", "gpu": "2", "out": str(path)}
                jobs, active, _ = q.resume_jobs(root, manifest, [live])
                self.assertEqual(jobs[0]["status"], "auditing")
                self.assertIn(123, active)
            self.assertEqual((path/"independent_audit.json").read_text(), "preserve")

    def test_adopted_exit_requires_completion_marker_and_pid_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            item = {"process": None, "record": {"pid": 123}, "action": "worker", "job": {"out": path}}
            with patch.object(q, "same_process", return_value=True): self.assertIsNone(q.finished_code(item))
            with patch.object(q, "same_process", return_value=False):
                self.assertEqual(q.finished_code(item), 1)
                (path/"DONE").write_text("done")
                self.assertEqual(q.finished_code(item), 0)
        with patch.object(q, "process_info", return_value={"start_ticks": "new", "state": "R"}):
            self.assertFalse(q.same_process({"pid": 1, "start_ticks": "old"}))


if __name__ == "__main__": unittest.main()
