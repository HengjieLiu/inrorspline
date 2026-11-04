import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments/oasis_alpha_sweep"))
import run_oasis_bounded2_842_sweep as r
import queue_oasis_bounded2_842_sweep as q


class SweepTests(unittest.TestCase):
    def test_matrix_names_and_lambdas(self):
        jobs = q.matrix()
        self.assertEqual(len(jobs), 70)
        self.assertEqual(len({j["name"] for j in jobs}), 70)
        self.assertEqual(len(q.matrix(True)), 3)
        self.assertFalse(set(j["name"] for j in jobs) & set(j["name"] for j in q.matrix(True)))
        for a in r.LAMBDAS:
            p = r.cell_protocol(a)
            self.assertEqual(p["stage_alpha"], [a] * 3)
            self.assertEqual([s["lambda"] for s in p["stage_schedule"]], [a] * 3)
        for a in (-1, 0, .2, float("nan"), float("inf")):
            with self.assertRaises(ValueError): r.alpha_value(a)
        for rank in (-1, 10, True):
            with self.assertRaises(ValueError): r.run_name(rank, 1.)

    def test_cli_and_old_compatibility(self):
        p70 = copy.deepcopy(r.parent.PROTOCOL)
        args = r.parse_args(["worker", "--manifest", "/tmp/m.json", "--out-dir", "/tmp/new_run",
                             "--case-rank", "0", "--alpha", ".3", "--gpu", "2"])
        self.assertEqual((args.depth, args.config, args.alpha, args.gpu), (3, r.CONFIG, .3, "2"))
        self.assertIs(r.b.optimize, r.parent.b.optimize)
        self.assertIs(r.b.Objective, r.parent.b.Objective)
        self.assertEqual(r.parent.PROTOCOL, p70)
        with self.assertRaises(SystemExit):
            r.parse_args(["worker", "--manifest", "/tmp/m.json", "--out-dir", "/tmp/new_run",
                          "--case-rank", "0", "--alpha", ".2", "--gpu", "2"])

    def test_six_map_schedule_and_no_cps1_halving(self):
        stages = r.b.stages(False)[:3]
        blocks = [stage for stage in stages for _ in range(2)]
        self.assertEqual([s.cps for s in blocks], [8, 8, 4, 4, 2, 2])
        self.assertEqual([s.index for s in blocks], [0, 0, 1, 1, 2, 2])
        self.assertEqual([s.lr for s in stages], [.03, .01, .003])
        self.assertEqual(sum(s.max_epochs for s in blocks), 5000)
        self.assertEqual([s.patience for s in stages], [25, 35, 50])
        self.assertTrue(all(s.max_lr_reductions == 0 for s in blocks))

    def test_cap_unit_derivative(self):
        cap = torch.tensor([.002, .003, .004])
        p = torch.zeros(3, requires_grad=True)
        controls = r.b.capped_controls(p, cap)
        gradient, = torch.autograd.grad(controls.sum(), p)
        self.assertTrue(torch.equal(gradient, torch.ones_like(p)))
        self.assertTrue(torch.equal(controls, torch.zeros_like(p)))
        self.assertTrue((r.b.capped_controls(torch.tensor([-100., 100., 10.]), cap).abs() <= cap).all())

    def test_full_objective_parity_and_lambda_scaling(self):
        torch.set_num_threads(2); torch.manual_seed(1)
        fixed, moving = torch.rand(16, 24, 16), torch.rand(16, 24, 16)
        stage = r.b.stages(False)[2]
        bound = r.b.compute_bound(2)
        low = r.b.Objective(fixed, moving, stage, .1, [], bound, True)
        high = r.b.Objective(fixed, moving, stage, 10., [], bound, True)
        self.assertTrue(r.verify_objective_parity(low)["passed"])
        self.assertTrue(r.verify_objective_parity(high)["passed"])
        p = (torch.randn(*low.control_shape, 3) * .002).requires_grad_()
        l0, data0, reg0 = low(p)
        grad0, = torch.autograd.grad(l0, p, retain_graph=True)
        rg, = torch.autograd.grad(reg0, p)
        l1, data1, reg1 = high(p)
        grad1, = torch.autograd.grad(l1, p)
        self.assertEqual(float(data0), float(data1)); self.assertEqual(float(reg0), float(reg1))
        self.assertAlmostEqual(float(l1-l0), 9.9*float(reg0), places=6)
        self.assertTrue(torch.allclose(grad1-grad0, 9.9*rg, atol=2e-6, rtol=1e-4))

    def test_capacity_and_failure_policy(self):
        self.assertTrue(q.capacity("0", [], 8192, 8192))
        self.assertFalse(q.capacity("0", [], 8191, 8192))
        self.assertFalse(q.capacity("0", ["0", "0"], 48000, 8192))
        self.assertTrue(q.capacity("0", ["0"], 16384, 8192))
        self.assertFalse(q.capacity("0", ["0"], 16383, 8192))
        self.assertEqual(q.next_status("worker", 0, True, "", 0), "audit_pending")
        self.assertEqual(q.next_status("audit", 0, True, "", 0), "complete")
        self.assertEqual(q.next_status("worker", 0, False, "", 0), "failed_deferred")
        self.assertEqual(q.next_status("worker", 1, False, "CUDA out of memory", 0), "oom_retry")
        self.assertEqual(q.next_status("worker", 1, False, "CUDA out of memory", 1), "failed_deferred")
        with tempfile.TemporaryDirectory() as temp:
            queue = q.Queue(Path(temp), {}, False, 8192)
            job = queue.jobs[0]
            queue.failure(job, "numerical failure", "NaN")
            self.assertEqual(job["status"], "failed_deferred")
            self.assertTrue(all(j["status"] == "pending" for j in queue.jobs[1:]))
            self.assertEqual(len(list((Path(temp) / "failures").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
