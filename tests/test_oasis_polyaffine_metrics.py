import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/experiments/oasis_polyaffine_fieldfit"))
import metrics as m


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.shape = (9, 11, 13)
        self.affine = np.diag([2., 1., .5, 1.])
        self.affine[:3, 3] = [-7, 3, -2]
        self.mask = np.zeros(self.shape, bool); self.mask[2:-2, 2:-2, 2:-2] = True
        self.zero = np.zeros((*self.shape, 3), np.float32)
        self.target = dict(field=self.zero, mask=self.mask, affine=self.affine)

    def test_identity_and_large_translation_metrics(self):
        identity = m.evaluate_fields(self.zero, self.target)
        self.assertEqual(identity["field_error"]["foreground"]["rmse_mm"], 0)
        self.assertIsNone(identity["field_error"]["foreground"]["relative_rmse_pct"])
        geo = identity["geometry"]["foreground_eroded"]
        self.assertEqual(geo["prediction"]["minimum"], 1)
        self.assertEqual(geo["prediction"]["folding_fraction"], 0)
        translated = self.zero.copy(); translated[:] = [30, -40, 0]
        result = m.evaluate_fields(translated, self.target)
        self.assertEqual(result["field_error"]["foreground"]["rmse_mm"], 50)
        self.assertEqual(result["geometry"]["full_interior"]["jacobian_matrix_rmse"], 0)
        json.dumps(result, allow_nan=False)

    def test_physical_derivative_and_fold_locations(self):
        indices = np.stack(np.meshgrid(*(np.arange(n) for n in self.shape), indexing="ij"), -1)
        world = indices @ self.affine[:3, :3].T + self.affine[:3, 3]
        folded = self.zero.copy(); folded[..., 0] = -2 * world[..., 0]
        result = m.evaluate_fields(folded, self.target)
        geometry = result["geometry"]["foreground_eroded"]
        self.assertAlmostEqual(geometry["jacobian_matrix_rmse"], 2)
        self.assertAlmostEqual(geometry["prediction"]["minimum"], -1)
        self.assertEqual(geometry["newly_folded_fraction"], 1)
        self.assertEqual(geometry["common_positive_fraction"], 0)
        self.assertIsNone(geometry["log_jacobian_rmse_common_positive"])
        reverse = m.evaluate_fields(self.zero, dict(self.target, field=folded))
        self.assertEqual(reverse["geometry"]["foreground_eroded"]["removed_reference_fold_fraction"], 1)
        json.dumps(reverse, allow_nan=False)

    def test_chunk_boundaries_and_serialized_reloading(self):
        rng = np.random.default_rng(4)
        reference = rng.normal(size=self.zero.shape).astype(np.float32) * .03
        prediction = reference * .9
        target = dict(self.target, field=reference)
        first = m.evaluate_fields(prediction, target, chunk_depth=1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prediction.npy"; np.save(path, prediction)
            second = m.evaluate_fields(path, target, chunk_depth=5)
        self.assertEqual(first["field_error"], second["field_error"])
        self.assertEqual(first["geometry"], second["geometry"])

    def test_explicit_zero_padding_and_image_domain(self):
        image = np.ones(self.shape, np.float32)
        displacement = self.zero.copy(); displacement[..., 0] = -1 # half of2mm voxel
        warped, valid = m.warp_volume(image, displacement, self.affine, chunk_depth=2)
        np.testing.assert_allclose(warped[0], .5, atol=1e-6)
        np.testing.assert_allclose(warped[1:], 1, atol=1e-6)
        self.assertFalse(valid[0].any())
        result = m.evaluate_fields(displacement, dict(self.target, moving_image=image))
        full = result["warping"]["image"]["full"]
        self.assertEqual(full["count"], np.prod(self.shape))
        self.assertEqual(full["reference_inbounds_coverage"], 1)
        self.assertAlmostEqual(result["warping"]["out_of_bounds"]["full"]["prediction_fraction"], 1 / self.shape[0])
        self.assertGreater(full["mae"], 0) # predicted OOB locations were not excluded

    def test_label_retention_and_surface_metrics(self):
        labels = np.zeros(self.shape, np.int16); labels[3:6, 4:7, 4:8] = 1
        result = m.evaluate_fields(self.zero, dict(self.target, moving_seg=labels, fixed_seg=labels))
        summary = result["labels"]
        self.assertEqual(summary["paired_valid_labels"], 1)
        self.assertEqual(summary["per_label"][0]["prediction"]["hd95_mm"], 0)
        self.assertEqual(summary["per_label"][0]["prediction"]["asd_symmetric_mm"], 0)
        self.assertEqual(summary["summary"]["paired"]["delta_prediction_minus_reference"]["dice"], 0)
        self.assertEqual(summary["per_label"][0]["warped_label_dice"], 1)
        self.assertEqual(summary["excluded_reference_labels"], list(range(2, 36)))
        translated = self.zero.copy(); translated[..., 1] = 1
        shifted = m.evaluate_fields(translated, dict(self.target, moving_seg=labels, fixed_seg=labels))
        self.assertAlmostEqual(shifted["labels"]["per_label"][0]["prediction"]["hd95_mm"], 1)
        self.assertGreater(shifted["labels"]["per_label"][0]["prediction"]["asd_symmetric_mm"], 0)
        json.dumps(shifted, allow_nan=False)

    def test_nonfinite_and_sheared_geometry_rejected(self):
        invalid = self.zero.copy(); invalid[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): m.evaluate_fields(invalid, self.target)
        shear = self.affine.copy(); shear[0, 1] = .2
        with self.assertRaises(ValueError): m.evaluate_fields(self.zero, dict(self.target, affine=shear))


if __name__ == "__main__":
    unittest.main()
