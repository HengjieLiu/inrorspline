from __future__ import annotations

import importlib.util
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "2026-09-22_slides_for_midl_yrs"
    / "presentation_arc_plots.py"
)
SPEC = importlib.util.spec_from_file_location("presentation_arc_plots", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PLOTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOTS
SPEC.loader.exec_module(PLOTS)


def archived_sources() -> tuple[Path, Path]:
    configured = PLOTS.artifact_paths.artifact_root(repo_root=REPO_ROOT)
    return PLOTS.default_source_paths(configured)


class PresentationArcPlotsTest(unittest.TestCase):
    def require_archived_sources(self) -> tuple[Path, Path]:
        oasis_source, lung_source = archived_sources()
        if not oasis_source.is_file() or not lung_source.is_file():
            self.skipTest("camera-ready source archive is unavailable")
        return oasis_source, lung_source

    def test_camera_ready_selection_keys_counts_colors_and_limits(self) -> None:
        oasis_source, lung_source = self.require_archived_sources()
        oasis = PLOTS.select_dataset(PLOTS.load_csv(oasis_source), PLOTS.OASIS_SPEC)
        lung_rows = PLOTS.load_csv(lung_source)
        lung = PLOTS.select_dataset(lung_rows, PLOTS.LUNG_SPEC)
        ptvreg = PLOTS.select_series(lung_rows, PLOTS.LUNG_SPEC, PLOTS.PTVREG_SPEC)

        self.assertEqual(
            [series.source_key for series in oasis],
            [
                "idir_original_dense_siren_be_autograd",
                "sinr_cps2_be_bspline_analytic",
                "direct_cp_cps2_be_bspline_analytic",
                "direct_cp_8_4_2_be_bspline_analytic_multistage",
            ],
        )
        self.assertEqual([len(series.points) for series in oasis], [6, 6, 6, 6])
        self.assertEqual(
            [series.source_key for series in lung],
            [
                "idir_be_default_lungmask",
                "sinr_cps4_global_ncc_fd_be",
                "direct_cp_cps4_global_ncc_fd_be",
                "direct_cp_32_16_8_4_ptv_lcc_isotv",
            ],
        )
        self.assertEqual([len(series.points) for series in lung], [7, 7, 7, 6])
        self.assertEqual(ptvreg.source_key, "ptvreg_reference")
        self.assertEqual(len(ptvreg.points), 1)
        self.assertEqual(
            [series.color for series in oasis],
            [PLOTS.METHOD_COLORS[name] for name in PLOTS.METHOD_ORDER],
        )
        self.assertEqual(PLOTS.OASIS_SPEC.axis.xlim, (0.0465148747, 1.6424305623))
        self.assertEqual(PLOTS.OASIS_SPEC.axis.ylim, (0.6515159001, 0.8582796825))
        self.assertEqual(PLOTS.LUNG_SPEC.axis.xlim, (0.06, 0.50))
        self.assertEqual(PLOTS.LUNG_SPEC.axis.ylim, (0.60, 2.05))
        self.assertEqual(PLOTS.AXES_RECT, (0.155, 0.16, 0.585, 0.74))

    def test_resolution_routes_sources_and_outputs_through_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary) / "artifact-root"
            source_root = artifact_root / PLOTS.CAMERA_READY_ARCHIVE
            source_root.mkdir(parents=True)
            (source_root / PLOTS.OASIS_SOURCE_FILENAME).write_text("header\n", encoding="utf-8")
            (source_root / PLOTS.LUNG_SOURCE_FILENAME).write_text("header\n", encoding="utf-8")

            old_value = os.environ.get("ARTIFACT_ROOT")
            os.environ["ARTIFACT_ROOT"] = str(artifact_root)
            try:
                args = PLOTS.resolve_paths(
                    SimpleNamespace(
                        oasis_source=None,
                        lung_source=None,
                        review_root=None,
                        dpi=300,
                        validate_only=True,
                    )
                )
            finally:
                if old_value is None:
                    os.environ.pop("ARTIFACT_ROOT", None)
                else:
                    os.environ["ARTIFACT_ROOT"] = old_value

            self.assertEqual(args.oasis_source, (source_root / PLOTS.OASIS_SOURCE_FILENAME).resolve())
            self.assertEqual(args.lung_source, (source_root / PLOTS.LUNG_SOURCE_FILENAME).resolve())
            self.assertEqual(
                args.review_root,
                (artifact_root / "repo_experiments" / PLOTS.ARTIFACT_EXPERIMENT_KEY).resolve(),
            )
            self.assertFalse(str(args.review_root).startswith(str(REPO_ROOT)))

    def test_rendered_layers_are_aligned_opaque_or_transparent_as_required(self) -> None:
        oasis_source, _lung_source = self.require_archived_sources()
        series = PLOTS.select_dataset(PLOTS.load_csv(oasis_source), PLOTS.OASIS_SPEC)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dpi = 100
            base = root / "base.png"
            layer = root / "layer.png"
            PLOTS.render_base(base, PLOTS.OASIS_SPEC, dpi=dpi)
            PLOTS.render_method_layer(layer, PLOTS.OASIS_SPEC, series[0], 0, dpi=dpi)

            expected = PLOTS.expected_pixel_size(dpi)
            PLOTS.validate_base(base, expected_size=expected)
            PLOTS.validate_overlay(layer, expected_size=expected)
            with Image.open(base) as base_image, Image.open(layer) as layer_image:
                self.assertEqual(base_image.size, expected)
                self.assertEqual(layer_image.size, expected)
                self.assertEqual(base_image.mode, "RGB")
                self.assertEqual(layer_image.mode, "RGBA")

    def test_oasis_asset_inventory_and_final_pixel_identity(self) -> None:
        oasis_source, _lung_source = self.require_archived_sources()
        series = PLOTS.select_dataset(PLOTS.load_csv(oasis_source), PLOTS.OASIS_SPEC)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = PLOTS.render_dataset_assets(root, PLOTS.OASIS_SPEC, series, dpi=100)

            self.assertEqual(
                {path.name for path in outputs["static"]},
                {
                    "01_oasis_no_ellipses.png",
                    "01_oasis_no_ellipses.svg",
                    "02_oasis_with_ellipses.png",
                    "02_oasis_with_ellipses.svg",
                },
            )
            self.assertEqual(
                {path.name for path in outputs["overlays"]},
                {
                    "oasis_anim_base.png",
                    "oasis_anim_01_inr_dense.png",
                    "oasis_anim_02_inr_bscp.png",
                    "oasis_anim_03_d_bscp.png",
                    "oasis_anim_04_mr_d_bscp.png",
                    "oasis_anim_05_uncertainty.png",
                },
            )
            self.assertEqual(
                [path.name for path in outputs["cumulative"]],
                [
                    "oasis_step_01.png",
                    "oasis_step_02.png",
                    "oasis_step_03.png",
                    "oasis_step_04.png",
                    "oasis_step_05_final.png",
                ],
            )
            PLOTS.assert_same_pixels(
                root / "static" / "02_oasis_with_ellipses.png",
                root / "cumulative" / "oasis_step_05_final.png",
            )

    def test_ellipse_dimensions_are_twice_source_standard_deviations(self) -> None:
        point = PLOTS.PlotPoint(0.2, 0.8, 0.03, 0.04, ("0.2", "0.8", "0.03", "0.04"))
        series = PLOTS.PlotSeries("key", "label", "#4E95D9", (point,))
        figure, axis = PLOTS.new_canvas(PLOTS.OASIS_SPEC, decorations=False, transparent=True)
        try:
            PLOTS.draw_ellipses(axis, series)
            self.assertEqual(len(axis.patches), 1)
            ellipse = axis.patches[0]
            self.assertAlmostEqual(ellipse.width, 0.06)
            self.assertAlmostEqual(ellipse.height, 0.08)
            self.assertAlmostEqual(ellipse.center[0], 0.2)
            self.assertAlmostEqual(ellipse.center[1], 0.8)
            self.assertAlmostEqual(ellipse.get_facecolor()[3], PLOTS.STYLE["ellipse_face_alpha"])
            self.assertAlmostEqual(ellipse.get_edgecolor()[3], PLOTS.STYLE["ellipse_edge_alpha"])
        finally:
            PLOTS.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
