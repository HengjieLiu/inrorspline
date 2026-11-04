import ast
import copy
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts/experiments/oasis_alpha_sweep"))
import svf_transition_validation as v
import run_oasis_bspline_svf_sweep as original
import run_oasis_bspline_svf_sweep_recovery as recovery


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_physical_limits(self):
        self.assertTrue(v.within_physical_tolerance({"max_abs_voxel": .001, "rms_voxel": 1e-5}))
        self.assertFalse(v.within_physical_tolerance({"max_abs_voxel": .001001, "rms_voxel": 1e-7}))
        self.assertFalse(v.within_physical_tolerance({"max_abs_voxel": .0001, "rms_voxel": 1.001e-5}))

    def test_finite_shape_and_unit_scaling(self):
        a = torch.zeros(2, 3, 4, 3, dtype=torch.float64)
        b = a.clone(); b[..., 0] = .001
        stats = v.error_stats(a, b, torch.tensor([100., 50., 20.]))
        self.assertAlmostEqual(stats["max_abs_voxel"], .1)
        self.assertAlmostEqual(stats["rms_voxel"], .1/(3**.5))
        for bad in (torch.full_like(a, float("nan")), torch.full_like(a, float("inf")), a[:1]):
            with self.assertRaises(RuntimeError): v.error_stats(a, bad, torch.ones(3))

    def test_fallback_requires_high_precision_proof(self):
        z = torch.zeros(40, 40, 40, 3)
        near = z.clone(); near[0, 0, 0, 0] = 3e-6
        with patch.object(v, "fp64_reference", return_value={"passed": True}) as proof:
            result = v.verify_transition(z, z, z, z, z, near, 2, torch.ones(3)*100)
            self.assertFalse(result["original_field_gate_passed"])
            proof.assert_called_once()
        with patch.object(v, "fp64_reference", side_effect=RuntimeError("proof failed")):
            with self.assertRaises(RuntimeError):
                v.verify_transition(z, z, z, z, z, near, 2, torch.ones(3)*100)
        with patch.object(v, "fp64_reference") as proof:
            for bad in (near*10, torch.ones_like(z)*3e-6):
                with self.assertRaises(RuntimeError):
                    v.verify_transition(z, z, z, z, z, bad, 2, torch.ones(3)*100)
            proof.assert_not_called()
        with self.assertRaises(RuntimeError):
            v.verify_transition(z, z, z, torch.ones_like(z)*1e-4, z, z, 2, torch.ones(3))

    def test_real_fp64_knot_preservation_and_wrong_controls(self):
        shape = (16, 24, 16)
        shapes = v.s.c.build_nested_cp_shapes(shape, 8, (8, 4, 2, 1))
        scale = v.s.d.voxel_scale(shape, device=torch.device("cpu"))
        torch.manual_seed(1)
        for cps in (4, 2, 1):
            a = torch.randn(*shapes[2*cps], 3)*.02
            b = v.s.c.cubic_bspline_dyadic_knot_insert_3d(a, shapes[cps])
            proof = v.fp64_reference(a, b, cps, shape, scale)
            self.assertTrue(proof["passed"])
            with self.assertRaises(RuntimeError): v.fp64_reference(a, b+.001, cps, shape, scale)

    def test_worker_has_only_validation_and_provenance_changes(self):
        old = ast.parse(inspect.getsource(original.worker))
        new = ast.parse(inspect.getsource(recovery.worker))
        old_gate = next(n for n in ast.walk(old) if isinstance(n, ast.If)
                        and any(isinstance(x, ast.Constant) and x.value == "Knot insertion changed integrated field materially"
                                for x in ast.walk(n)) and len(n.body) == 1 and isinstance(n.body[0], ast.Raise))

        class Normalize(ast.NodeTransformer):
            def visit_Assign(self, node):
                if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "transition.verify_transition":
                    return copy.deepcopy(old_gate)
                return self.generic_visit(node)

            def visit_Dict(self, node):
                keep = [(k, val) for k, val in zip(node.keys, node.values)
                        if not (isinstance(k, ast.Constant) and k.value == "recovery")]
                node.keys, node.values = [x[0] for x in keep], [x[1] for x in keep]
                return self.generic_visit(node)

        new = Normalize().visit(new)
        self.assertEqual(ast.dump(old), ast.dump(new))
        self.assertIs(recovery.s.Objective, original.s.Objective)
        self.assertIs(recovery.s.optimize, original.s.optimize)
        self.assertEqual(original.source_hashes(), recovery.source_hashes())
        self.assertEqual(inspect.getsource(original.setup), inspect.getsource(recovery.setup))

    def test_resume_only_approved_failure_without_overwrite(self):
        spec = recovery.s.Spec(3, "842", "velocity", .01)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); failed = root/spec.name()/"attempt0"
            failed.mkdir(parents=True); (root/"recovery_v1").mkdir()
            failure_file = failed/"FAILED_worker.json"
            failure_file.write_text("preserved failure")
            digest = recovery.sha(failure_file)
            manifest = {"gpus": [], "recovery": {"approved_retry": {
                "path": str(failed), "failure_sha256": digest}}}
            with patch.object(recovery.s, "matrix", return_value=[spec]), \
                    patch.object(recovery.legacy, "gpu_snapshot", return_value={"rows": []}), \
                    patch.object(recovery.time, "sleep", side_effect=InterruptedError("test stop")):
                with self.assertRaises(InterruptedError): recovery.phase(root, manifest, False, 8192)
            job = recovery.c.read_json(root/"progress.json")["jobs"][0]
            self.assertEqual(job["attempt"], 1)
            self.assertEqual(job["status"], "pending")
            self.assertEqual(recovery.sha(failure_file), digest)
            self.assertFalse((failed.parent/"attempt1").exists())
            manifest["recovery"]["approved_retry"]["failure_sha256"] = "wrong"
            with patch.object(recovery.s, "matrix", return_value=[spec]):
                with self.assertRaises(RuntimeError): recovery.phase(root, manifest, False, 8192)

    def test_resume_skips_accepted_run(self):
        spec = recovery.s.Spec(0, "842", "velocity", 1.)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); accepted = root/spec.name()/"attempt0"
            accepted.mkdir(parents=True); (accepted/"independent_audit.json").write_text("kept")
            with patch.object(recovery.s, "matrix", return_value=[spec]), \
                    patch.object(recovery, "validate", return_value={"gpu": "2"}), \
                    patch.object(recovery.subprocess, "Popen") as launch:
                rows = recovery.phase(root, {}, False, 8192)
                self.assertEqual(rows, [{"gpu": "2"}])
                launch.assert_not_called()
            self.assertEqual((accepted/"independent_audit.json").read_text(), "kept")


if __name__ == "__main__":
    unittest.main()
