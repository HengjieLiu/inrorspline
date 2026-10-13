import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/reports"))
import oasis_sitreg_ncc_arc_lines_top as plot


class SitregArcTests(unittest.TestCase):
    def fixture(self):
        cases, means = [], []
        for loss, group in plot.SERIES.items():
            for key, alpha in group:
                rows = [dict(loss=loss, lambda_key=key, rank=i, fixed_id=f"f{i}", moving_id=f"m{i}",
                             **{m: .2+i*.001+alpha*.0001 for m in plot.METRICS}) for i in range(100)]
                row = dict(loss=loss, lambda_key=key, alpha=alpha, n=100,
                           historical=loss == "lncc9" and key == "1")
                for metric in plot.METRICS:
                    values = [r[metric] for r in rows]
                    row[metric+"_mean"], row[metric+"_std"] = statistics.mean(values), statistics.stdev(values)
                cases.extend(rows); means.append(row)
        return dict(complete=True, means=means, per_case=cases)

    def test_layout_and_ellipse_dimensions(self):
        means = plot.validated_means(self.fixture())
        fig, axes = plot.build_figure(means)
        try:
            self.assertEqual(axes.shape, (2, 3))
            for col, (metric, _) in enumerate(plot.Y):
                self.assertEqual(len(axes[0, col].patches), 0)
                self.assertEqual(len(axes[1, col].patches), 9)
                self.assertEqual(axes[0, col].get_ylim(), axes[1, col].get_ylim())
                self.assertEqual(axes[0, col].get_xlim(), axes[1, col].get_xlim())
                point = means["gncc", "3"]
                ellipse = axes[1, col].patches[0]
                self.assertEqual(ellipse.width, 2*point[plot.X+"_std"])
                self.assertEqual(ellipse.height, 2*point[metric+"_std"])
        finally:
            plot.plt.close(fig)

    def test_incomplete_or_changed_cases_rejected(self):
        for corruption in ("missing", "duplicate", "direction", "stats", "nonfinite", "historical", "not_complete"):
            data = copy.deepcopy(self.fixture())
            if corruption == "missing": data["per_case"].pop()
            elif corruption == "duplicate": data["per_case"][0]["rank"] = 1
            elif corruption == "direction": data["per_case"][101]["moving_id"] = "wrong"
            elif corruption == "stats": data["means"][0]["dice_std"] += .1
            elif corruption == "nonfinite": data["per_case"][0]["dice"] = float("nan")
            elif corruption == "historical": data["means"][0]["historical"] = True
            else: data["complete"] = False
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                plot.validated_means(data)

    def test_export_completion_gate_and_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"report"; root.mkdir()
            out = root/"lines_top_v1"
            with self.assertRaises(ValueError): plot.export(root, out)
            (root/"results.json").write_text(json.dumps(self.fixture()))
            (root/"ACCEPTED.json").write_text(json.dumps(dict(complete=True, new_cases=400,
                reference_cases=500, outputs_sha256={"results.json": plot.sha(root/"results.json")})))
            (root/"DONE").write_text("complete\n")
            plot.export(root, out)
            for ext in ("png", "pdf", "svg"):
                self.assertGreater((out/f"{plot.STEM}.{ext}").stat().st_size, 1000)
            manifest = plot.read(out/"plot_manifest.json")
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["top"], "means_only")
            self.assertEqual(manifest["points"], 9)
            with self.assertRaises(FileExistsError): plot.export(root, out)
            (root/"results.json").write_text("{}")
            with self.assertRaises(ValueError): plot.export(root, root/"bad_export")


if __name__ == "__main__":
    unittest.main()
