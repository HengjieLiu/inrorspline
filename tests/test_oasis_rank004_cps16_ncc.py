import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
import common as c
import cps16_ncc as n
import dl_ncc_diffusion as d
import run_oasis_rank004_cps16_ncc_one as worker
import launch_oasis_rank004_cps16_ncc as launch


class Cps16Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(1)

    def test_five_stage_schedule_preserves_existing_four_stages(self):
        for mode in ("pyramid", "full"):
            actual = n.stages(n.FitSpec(mode))
            old = d.stages(d.FitSpec(4, mode, 1., "842_dense"))
            self.assertEqual([s.cps for s in actual], [16, 8, 4, 2, None])
            self.assertEqual([s.lr for s in actual], [.03, .03, .01, .003, .001])
            self.assertEqual([s.factor for s in actual], [8,4,2,1,1] if mode == "pyramid" else [1]*5)
            self.assertEqual([s.max_lr_reductions for s in actual], [0,0,0,0,2])
            self.assertEqual([s.patience for s in actual], [25,25,35,50,100])
            for a, b in zip(actual[1:], old):
                values = asdict(a); values["index"] -= 1
                self.assertEqual(values, asdict(b))

    def test_native_control_shapes_and_synthetic_prolongation(self):
        self.assertEqual(n.control_shapes()[16], (14,18,16))
        self.assertEqual(n.control_shapes()[8], (24,32,28))
        shape = (32,48,32)
        shapes = c.build_nested_cp_shapes(shape,16,[16,8,4,2])
        parameter = torch.randn(*shapes[16],3)*.01
        for coarse, fine in ((16,8),(8,4),(4,2)):
            refined = c.cubic_bspline_dyadic_knot_insert_3d(parameter,shapes[fine])
            for evaluation in (shape, (4,6,4), (7,11,9)):
                before = d.AlignedBSpline(shapes[coarse],shape,evaluation,coarse,"cpu")(parameter)
                after = d.AlignedBSpline(shapes[fine],shape,evaluation,fine,"cpu")(refined)
                self.assertTrue(torch.allclose(before,after,atol=2e-8,rtol=1e-5))
            parameter = refined
        field = d.AlignedBSpline(shapes[2],shape,shape,2,"cpu")(parameter)
        dense = torch.nn.Parameter(field.detach().clone())
        self.assertTrue(torch.equal(field,dense))
        self.assertNotEqual(field.data_ptr(),dense.data_ptr())
        self.assertEqual(tuple(dense.shape),(*shape,3))

    def test_cps16_full_expansion_values_and_gradients(self):
        shape = (32,48,32)
        cp_shape = c.build_nested_cp_shapes(shape,16,[16])[16]
        cp = (torch.randn(*cp_shape,3)*.01).requires_grad_()
        actual = d.AlignedBSpline(cp_shape,shape,shape,16,"cpu")(cp)
        expected = c.expand_on_shape(cp,cp_shape,16,shape,1)
        self.assertTrue(torch.equal(actual,expected))
        ga, = torch.autograd.grad(actual.square().mean(),cp)
        ge, = torch.autograd.grad(expected.square().mean(),cp)
        self.assertTrue(torch.equal(ga,ge))

    def test_cps16_eighth_resolution_endpoint_alignment(self):
        shape = c.IMAGE_SHAPE
        cp_shape = n.control_shapes()[16]
        cp = torch.zeros(*cp_shape,3)
        cp[...,0] = torch.arange(cp_shape[2])[None,None,:]*.01
        evaluation = c.downsample_shape(shape,8)
        self.assertEqual(evaluation,(20,28,24))
        field = d.AlignedBSpline(cp_shape,shape,evaluation,16,"cpu")(cp)
        expected = (torch.linspace(0,shape[2]-1,evaluation[2])/16+1)*.01
        self.assertTrue(torch.allclose(field[...,0],expected[None,None,:].expand(evaluation),atol=2e-8))

    def test_diffusion_grid_units_for_affine_strain(self):
        for factor in (8,4,2,1):
            shape = c.downsample_shape(c.IMAGE_SHAPE,factor)
            q = c.make_coordinate_grid(shape,torch.device("cpu"))*torch.tensor([.01,.02,.03])
            vox = q*d.voxel_scale(shape,device="cpu")
            energy = c.forward_valid_diffusion(vox.permute(3,0,1,2)[None],None)
            self.assertAlmostEqual(float(energy),(.01**2+.02**2+.03**2)/9,places=9)

    def test_names_scope_smoke_and_cli(self):
        names = [n.FitSpec(mode,smoke).name() for mode in ("pyramid","full") for smoke in (False,True)]
        self.assertEqual(len(set(names)),4)
        self.assertTrue(all("cps16-8-4-2-dense" in name for name in names))
        for spec in (n.FitSpec("bad"),n.FitSpec("full",case_rank=0),n.FitSpec("full",alpha=3.),n.FitSpec("full",seed=2)):
            with self.assertRaises(ValueError): spec.validate()
        argv = ["--resolution","pyramid","--gpu","3","--oasis-root","/tmp/data", "--dl-code-root","/tmp/dl",
                "--vfa-root","/tmp/vfa","--out-dir","/tmp/output","--source-revision","test"]
        self.assertEqual(worker.parse_args(argv).case_rank,4)
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit): worker.parse_args(argv+["--gpu","2"])
        self.assertEqual([s.max_epochs for s in n.stages(n.FitSpec("full",True))],[1]*5)

    def test_cps16_zero_images_remain_zero_no_analytic_regularization(self):
        shape = (32,48,32)
        cp_shape = c.build_nested_cp_shapes(shape,16,[16])[16]
        stage = n.stages(n.FitSpec("pyramid",True))[0]
        with patch.object(c,"analytic_bspline_sampled_be",side_effect=AssertionError("No analytic BE")), \
             patch.object(c,"forward_valid_be",side_effect=AssertionError("No finite-difference BE")):
            objective = d.StageObjective(torch.zeros(shape),torch.zeros(shape),1.,stage,cp_shape)
            q = torch.zeros(*cp_shape,3,requires_grad=True)
            loss,data,reg = objective(q)
            self.assertEqual(float(loss),1.)
            self.assertEqual(float(reg),0.)
            loss.backward()
            self.assertTrue(torch.equal(q.grad,torch.zeros_like(q)))

    def test_report_requires_both_runs_and_correct_label_means(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError): launch.write_report([],[],{},root)
            data = {"fixed_id":"0084","moving_id":"0108"}
            for label in c.OASIS_LABELS:
                for metric,value in (("dice",.8),("hd95",2.),("asd_symmetric",.4)):
                    data[f"{metric}_label_{label:02d}"] = value
            path = root/"labels.csv"; c.write_csv(path,[data])
            launch.verify_label_means({"dice":.8,"hd95":2.,"asd":.4},path)
            with self.assertRaises(RuntimeError): launch.verify_label_means({"dice":.9,"hd95":2.,"asd":.4},path)
            data["dice_label_18"] = float("nan"); c.write_csv(path,[data])
            with self.assertRaises(RuntimeError): launch.verify_label_means({"dice":.8,"hd95":2.,"asd":.4},path)


if __name__ == "__main__":
    unittest.main()
