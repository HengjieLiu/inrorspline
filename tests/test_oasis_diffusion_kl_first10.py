"""Scope extension guards; the numerical engine is tested by the frozen pilot suite."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.experiments.oasis_regularizer_sweep import losses as l, worker as w
from scripts.experiments.oasis_regularizer_sweep.protocol import sequence
from scripts.experiments.oasis_diffusion_kl_first10.protocol import GAMMAS, PROTOCOL, validate_job
from scripts.experiments.oasis_diffusion_kl_first10.worker import activate
from scripts.experiments.oasis_diffusion_kl_rank0.model import CombinedObjective, optimize


class First10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_matrix_and_sequence_guards(self):
        self.assertEqual(GAMMAS, (0., 1e-4, 1e-3, 1e-2, .1, 1.))
        self.assertEqual(PROTOCOL["case_ranks"], list(range(10)))
        self.assertEqual(sequence("cohort"), [(0, "cold")] + [(r, "warm") for r in range(10)])
        for backend in ("fd", "analytic"):
            for gamma in GAMMAS[1:]:
                validate_job(dict(phase="cohort", weight=1., arm="diffusion_"+backend, gamma=gamma))
        for change in (dict(phase="pilot"), dict(weight=.1), dict(gamma=0.), dict(gamma=.3), dict(arm="be_fd")):
            job = dict(phase="cohort", weight=1., arm="diffusion_fd", gamma=.01)
            job.update(change)
            with self.assertRaises(ValueError):
                validate_job(job)

    def test_scope_extension_keeps_exact_objective_and_optimizer(self):
        w.torch, w.np, w.l, w.c, w.d = torch, np, l, l.c, l.d
        with tempfile.TemporaryDirectory() as folder:
            def fake_register(job, rank, condition, index, device, backend):
                return dict(dvf_path=str(Path(folder)/"disp.npy"), case_rank=rank)
            with patch.object(w, "register", fake_register):
                original = w.l.Objective, w.optimize, w.register, w.PROTOCOL
                with activate(w, .01):
                    self.assertIs(w.optimize, optimize)
                    self.assertEqual(w.PROTOCOL["case_ranks"], list(range(10)))
                    for rank in range(10):
                        result = w.register({}, rank, "warm", rank+1, None, {})
                        self.assertEqual(result["lambda_kl"], .01)
                        self.assertEqual(result["case_rank"], rank)
                    for rank in (-1, 10):
                        with self.assertRaises(ValueError):
                            w.register({}, rank, "warm", 0, None, {})
                    torch.manual_seed(41)
                    shape = (24, 32, 24)
                    cs = l.c.build_nested_cp_shapes(shape, 8, [8])[8]
                    fixed, moving = torch.rand(shape), torch.rand(shape)
                    stage = l.d.StageSpec(0, 8, 4, .03, 1, 1, 25, 0)
                    for arm in ("diffusion_fd", "diffusion_analytic"):
                        direct = CombinedObjective(fixed, moving, 1., stage, cs, arm, gamma=.01)
                        adapter = w.l.Objective(fixed, moving, 1., stage, cs, arm)
                        cp = torch.zeros(*cs, 3, requires_grad=True)
                        a, b = direct(cp)[0], adapter(cp)[0]
                        self.assertTrue(torch.equal(a, b))
                        ga, = torch.autograd.grad(a, cp)
                        gb, = torch.autograd.grad(b, cp)
                        self.assertTrue(torch.equal(ga, gb))
                self.assertEqual((w.l.Objective, w.optimize, w.register, w.PROTOCOL), original)


if __name__ == "__main__":
    unittest.main()
