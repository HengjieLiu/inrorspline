import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/reports"))
import oasis_rank4_registration_views as v


def report():
    base = {"case_rank": 4, "fixed_id": "0084", "moving_id": "0108", "direction": "moving_to_fixed"}
    rows = [{**base, "resolution": "full", "representation": "842_dense", "alpha": a,
             "group": f"NCC{a}"} for a in v.LAMBDAS]
    rows += [{**base, "resolution": "reference", "representation": n, "alpha": None, "group": n}
             for n in ("VFA", "SITReg")]
    return {"complete": True, "case_metrics": rows}


class VisualizationTests(unittest.TestCase):
    def test_fixed_selection_order_and_no_pyramid_or_cps1(self):
        data = report()
        data["case_metrics"] += [{**data["case_metrics"][0], "resolution": "pyramid"},
                                  {**data["case_metrics"][0], "representation": "8421"}]
        rows = v.select_rows(data)
        self.assertEqual([r["alpha"] for r in rows], [0.1, 0.3, 1, 3, None, None])
        self.assertEqual([r["group"] for r in rows[-2:]], ["VFA", "SITReg"])

    def test_incomplete_duplicate_missing_wrong_direction_rejected(self):
        for kind in ("incomplete", "duplicate", "missing", "direction"):
            data = report()
            if kind == "incomplete":
                data["complete"] = False
            elif kind == "duplicate":
                data["case_metrics"].append(data["case_metrics"][0])
            elif kind == "missing":
                data["case_metrics"].pop()
            else:
                data["case_metrics"][0]["direction"] = "fixed_to_moving"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                v.select_rows(data)

    def test_saved_field_axis_order_and_voxel_scale(self):
        shape = (5, 7, 9)
        raw = np.zeros((3, *shape), np.float32)
        raw[0], raw[1], raw[2] = 2, 6, 12
        with patch.object(v.c, "IMAGE_SHAPE", shape):
            result = v.normalized_from_axis_voxels(raw)
            np.testing.assert_array_equal(result, np.broadcast_to([3, 2, 1], (*shape, 3)))
            with self.assertRaises(ValueError):
                v.normalized_from_axis_voxels(raw.transpose(1, 2, 3, 0))

    def test_all_view_directions(self):
        axes = np.indices((5, 7, 9))
        for axis, _, _, directions in v.VIEWS:
            remaining = [a for a in range(3) if a != axis]
            horizontal = v.display_slice(axes[remaining[0]], axis, 2)
            vertical = v.display_slice(axes[remaining[1]], axis, 2)
            self.assertTrue(np.all(np.diff(horizontal, axis=1) == -1))
            self.assertTrue(np.all(np.diff(vertical, axis=0) == 1))
        self.assertEqual([x[3] for x in v.VIEWS], [("A", "P", "I", "S"), ("R", "L", "I", "S"), ("R", "L", "P", "A")])


if __name__ == "__main__":
    unittest.main()
