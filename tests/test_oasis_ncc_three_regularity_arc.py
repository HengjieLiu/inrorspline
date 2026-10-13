import ast
import copy
from pathlib import Path
import sys
import unittest
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts/reports'))
import oasis_ncc_three_regularity_arc as arc


class ForegroundArcTests(unittest.TestCase):
    def test_constant_and_affine_fields(self):
        mask = np.ones((7, 8, 9), dtype=bool)
        constant = np.full((3, 7, 8, 9), 12, dtype=np.float32)
        self.assertEqual(arc.diffusion(constant, mask)['diffusion_l2_fg'], 0)
        grid = np.indices(mask.shape).astype(np.float32)
        grid *= np.array([1, 2, 3], dtype=np.float32)[:, None, None, None]
        result = arc.diffusion(grid, mask)
        self.assertAlmostEqual(result['diffusion_l2_fg'], 14 / 9)
        self.assertEqual(result['diffusion_l2_fg'], result['diffusion_l2_full'])

    def test_both_endpoints_required_and_component_mean(self):
        mask = np.zeros((7, 8, 9), dtype=bool)
        mask[2:5, 2:6, 2:7] = True
        field = np.zeros((3, *mask.shape), dtype=np.float32)
        field[:, ~mask] = 100
        result = arc.diffusion(field, mask)
        self.assertEqual(result['diffusion_l2_fg'], 0)
        self.assertGreater(result['diffusion_l2_full'], 0)
        self.assertEqual(result['foreground_edges_by_array_axis'], [40, 45, 48])

    def test_matches_existing_canonical_function(self):
        # Execute only the unchanged pure NumPy function, avoiding CUDA dependencies.
        source = (REPO / 'scripts/experiments/oasis_alpha_sweep/common.py').read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'diffusion_loss_voxel')
        namespace = dict(np=np, Optional=Optional)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<canonical-diffusion>', 'exec'), namespace)
        rng = np.random.default_rng(1)
        field = rng.normal(size=(3, 7, 8, 9)).astype(np.float32)
        mask = rng.random((7, 8, 9)) > .3
        result = arc.diffusion(field, mask)
        for key, support in [('diffusion_l2_full', None), ('diffusion_l2_fg', mask)]:
            expected = namespace['diffusion_loss_voxel'](field[None], support)
            self.assertTrue(np.isclose(result[key], expected, rtol=2e-6, atol=1e-8))

    def test_invalid_inputs(self):
        field = np.zeros((3, 7, 8, 9), dtype=np.float32)
        for mask in [np.zeros((7, 8, 9), bool), np.ones((7, 8, 9)), np.ones((7, 8, 8), bool)]:
            with self.assertRaises(ValueError):
                arc.diffusion(field, mask)
        field[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            arc.diffusion(field, np.ones((7, 8, 9), bool))

    def fixture(self):
        rows = []
        for model in arc.REPORTS:
            for loss, group in arc.old.SERIES.items():
                for key, alpha in group:
                    for rank in range(100):
                        row = dict(model=model, loss=loss, lambda_key=key, alpha=alpha, rank=rank)
                        row.update({m: .2 + rank * .001 + alpha * .0001 for m in arc.METRICS})
                        if key == '10':
                            row['jdet_le0_pct_fg'] = 0.
                        rows.append(row)
        return rows

    def test_complete_case_accounting(self):
        rows = self.fixture()
        means = arc.summarize(rows)
        self.assertEqual(sum(len(m) for m in means.values()), 18)
        for bad in [rows[:-1], rows + [rows[0]]]:
            with self.assertRaises(ValueError):
                arc.summarize(bad)
        bad = copy.deepcopy(rows)
        bad[0]['diffusion_l2_fg'] = float('nan')
        with self.assertRaises(ValueError):
            arc.summarize(bad)

    def test_all_six_figures_and_exact_zero_folding(self):
        import matplotlib.pyplot as plt
        means = arc.summarize(self.fixture())
        all_means = [r for group in means.values() for r in group.values()]
        for model in arc.REPORTS:
            for metric in arc.AXES:
                fig, axes = arc.build_figure(model, means[model], metric, all_means)
                try:
                    self.assertEqual(axes.shape, (2, 3))
                    for col, (y, _) in enumerate(arc.old.Y):
                        self.assertEqual(len(axes[0, col].patches), 0)
                        self.assertEqual(len(axes[1, col].patches), 9)
                        self.assertEqual(axes[0, col].get_xlim(), axes[1, col].get_xlim())
                        self.assertEqual(axes[0, col].get_ylim(), axes[1, col].get_ylim())
                        self.assertEqual(axes[0, col].get_xscale(), 'linear')
                        r = means[model]['gncc', '3']
                        ellipse = axes[1, col].patches[0]
                        self.assertEqual(ellipse.width, 2 * r[metric + '_std'])
                        self.assertEqual(ellipse.height, 2 * r[y + '_std'])
                    if metric == 'jdet_le0_pct_fg':
                        self.assertEqual(axes[0, 0].get_xlim()[0], 0)
                        self.assertEqual(axes[1, 0].patches[4].width, 0)
                finally:
                    plt.close(fig)


if __name__ == '__main__':
    unittest.main()
