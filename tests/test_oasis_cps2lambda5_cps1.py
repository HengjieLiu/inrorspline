from dataclasses import asdict
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments/oasis_alpha_sweep"))
import cps2lambda5_cps1 as f
import torch


class CpsOneComparisonTests(unittest.TestCase):
    def test_schedule_matches_dense_except_representation(self):
        actual = asdict(f.stage_for())
        dense = asdict(f.previous.stage_for(f.previous.Spec(0, 1.)))
        self.assertEqual(actual.pop("cps"), 1)
        self.assertIsNone(dense.pop("cps"))
        self.assertEqual(actual, dense)
        self.assertEqual((actual["lr"], actual["max_epochs"], actual["min_epochs"]), (.001, 2000, 300))
        self.assertEqual(f.stage_for(True).max_epochs, 1)
        self.assertNotEqual(f.run_name(), f.run_name(True))
        self.assertIn("1_1_5_1", f.run_name())
        self.assertNotEqual(f.run_name(), f.previous.Spec(0, 1.).name())

    def test_cli_scope(self):
        args = f.parse_args(["launch", "--artifact-root", "/tmp/test-artifacts"])
        self.assertEqual(args.gpu, "3")
        self.assertEqual(args.stop_after, "full")
        with self.assertRaises(SystemExit):
            f.parse_args(["launch", "--artifact-root", "/tmp/test-artifacts", "--gpu", "2"])
        with self.assertRaises(SystemExit):
            f.parse_args(["launch", "--artifact-root", str(f.c.REPO_ROOT)])

    def test_prolongation_and_dense_objective_gradient_chain(self):
        torch.set_num_threads(1); torch.manual_seed(61)
        shape = (16, 24, 16)
        shapes = f.c.build_nested_cp_shapes(shape, 8, [8, 4, 2, 1])
        controls = torch.randn(*shapes[2], 3) * .02
        fine = f.c.cubic_bspline_dyadic_knot_insert_3d(controls, shapes[1]).requires_grad_()
        fixed, moving = torch.rand(shape), torch.rand(shape)
        obj = f.d.StageObjective(fixed, moving, 1., f.stage_for(), shapes[1])
        field = obj.field(fine)
        coarse_field = f.d.AlignedBSpline(shapes[2], shape, shape, 2, "cpu")(controls)
        check = f.legacy.verify_init(coarse_field, field)
        self.assertLess(check["max_abs_voxel"], 1e-5)
        dense = f.previous.objective_for(fixed, moving, f.previous.Spec(0, 1.))
        field_parameter = field.detach().clone().requires_grad_()
        for a, b in zip(obj(fine), dense(field_parameter)):
            self.assertTrue(torch.equal(a, b))
            control_gradient = torch.autograd.grad(a, fine, retain_graph=True)[0]
            field_gradient = torch.autograd.grad(b, field_parameter, retain_graph=True)[0]
            chain = torch.autograd.grad(field, fine, grad_outputs=field_gradient, retain_graph=True)[0]
            self.assertTrue(torch.equal(control_gradient, chain))
            self.assertTrue(torch.isfinite(control_gradient).all())
        self.assertEqual(f.c.build_nested_cp_shapes(f.c.IMAGE_SHAPE, 8, [1])[1], (164, 228, 196))


if __name__ == "__main__":
    unittest.main()
