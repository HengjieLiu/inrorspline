#!/usr/bin/env python3
"""Build camera-ready ARC plots for the MIDL YRS presentation.

This renderer only restyles accepted camera-ready source rows.  It does not
read raw medical images, rerun registration, or derive new measurements.
Generated assets are routed to the external artifact store.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_SCRIPTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SCRIPTS_DIR))

import artifact_paths  # noqa: E402


ARTIFACT_EXPERIMENT_KEY = "2026-09-22_slides_for_midl_yrs/presentation_arc_assets"
CAMERA_READY_ARCHIVE = Path(
    "repo_experiments/migration_2026-09-02/experiments/paper_artifacts/"
    "pareto_figure/v013_final_dirlab4dct_axes_labels"
)
OASIS_SOURCE_FILENAME = "oasis100_pareto_source_rows.csv"
LUNG_SOURCE_FILENAME = "source_rows.csv"

METHOD_ORDER = ("INR-Dense", "INR-BSCP", "D-BSCP", "MR-D-BSCP")
METHOD_COLORS = {
    "INR-Dense": "#4E95D9",
    "INR-BSCP": "#A783DC",
    "D-BSCP": "#E97132",
    "MR-D-BSCP": "#E0AD46",
}

FIGSIZE = (8.0, 5.33)
DEFAULT_DPI = 300
AXES_RECT = (0.155, 0.16, 0.585, 0.74)
LEGEND_X = 0.775
LEGEND_Y = (0.80, 0.70, 0.60, 0.50)
PTVREG_LEGEND_Y = 0.40

STYLE = {
    "title_fontsize": 21,
    "axis_label_fontsize": 19,
    "tick_fontsize": 16,
    "legend_fontsize": 14,
    "reference_legend_fontsize": 11,
    "linewidth": 2.8,
    "markersize": 7.0,
    "reference_markersize": 5.5,
    "ptvreg_markersize": 13.0,
    "grid_alpha": 0.18,
    "ellipse_face_alpha": 0.12,
    "ellipse_edge_alpha": 0.20,
    "reference_face_alpha": 0.055,
    "reference_edge_alpha": 0.10,
}


@dataclass(frozen=True)
class AxisPolicy:
    title: str
    xlabel: str
    ylabel: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    xticks: tuple[float, ...]
    yticks: tuple[float, ...]


@dataclass(frozen=True)
class SourceSpec:
    source_key: str
    display_label: str
    color: str
    expected_count: int


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    key_column: str
    x_column: str
    y_column: str
    x_std_column: str
    y_std_column: str
    sort_columns: tuple[str, ...]
    methods: tuple[SourceSpec, ...]
    axis: AxisPolicy


@dataclass(frozen=True)
class PlotPoint:
    x: float
    y: float
    x_std: float
    y_std: float
    raw_values: tuple[str, str, str, str]


@dataclass(frozen=True)
class PlotSeries:
    source_key: str
    display_label: str
    color: str
    points: tuple[PlotPoint, ...]


OASIS_SPEC = DatasetSpec(
    slug="oasis",
    key_column="method_key",
    x_column="log_jdet_std_fg_mean",
    y_column="dice_mean_mean",
    x_std_column="log_jdet_std_fg_std",
    y_std_column="dice_mean_std",
    sort_columns=("alpha",),
    methods=(
        SourceSpec("idir_original_dense_siren_be_autograd", "INR-Dense", METHOD_COLORS["INR-Dense"], 6),
        SourceSpec("sinr_cps2_be_bspline_analytic", "INR-BSCP", METHOD_COLORS["INR-BSCP"], 6),
        SourceSpec("direct_cp_cps2_be_bspline_analytic", "D-BSCP", METHOD_COLORS["D-BSCP"], 6),
        SourceSpec(
            "direct_cp_8_4_2_be_bspline_analytic_multistage",
            "MR-D-BSCP",
            METHOD_COLORS["MR-D-BSCP"],
            6,
        ),
    ),
    axis=AxisPolicy(
        title="OASIS Brain MRI (N=100)",
        xlabel="Foreground SD(log J)",
        ylabel="Mean Dice ↑",
        xlim=(0.0465148747, 1.6424305623),
        ylim=(0.6515159001, 0.8582796825),
        xticks=(0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6),
        yticks=(0.675, 0.700, 0.725, 0.750, 0.775, 0.800, 0.825, 0.850),
    ),
)

LUNG_SPEC = DatasetSpec(
    slug="lung",
    key_column="method_id",
    x_column="fg_logj_std_mean",
    y_column="original_snap_tre_mean_mm_mean",
    x_std_column="fg_logj_std_std",
    y_std_column="original_snap_tre_mean_mm_std",
    sort_columns=("ratio_to_best_tre_weight", "reg_value"),
    methods=(
        SourceSpec("idir_be_default_lungmask", "INR-Dense", METHOD_COLORS["INR-Dense"], 7),
        SourceSpec("sinr_cps4_global_ncc_fd_be", "INR-BSCP", METHOD_COLORS["INR-BSCP"], 7),
        SourceSpec("direct_cp_cps4_global_ncc_fd_be", "D-BSCP", METHOD_COLORS["D-BSCP"], 7),
        SourceSpec(
            "direct_cp_32_16_8_4_ptv_lcc_isotv",
            "MR-D-BSCP",
            METHOD_COLORS["MR-D-BSCP"],
            6,
        ),
    ),
    axis=AxisPolicy(
        title="DIR-LAB Lung 4DCT (N=10)",
        xlabel="Foreground SD(log J)",
        ylabel="TRE (mm) ↓",
        xlim=(0.06, 0.50),
        ylim=(0.60, 2.05),
        xticks=(0.1, 0.2, 0.3, 0.4, 0.5),
        yticks=(0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0),
    ),
)

PTVREG_SPEC = SourceSpec("ptvreg_reference", "pTVReg", "#333333", 1)
OASIS_REFERENCE_SPECS = (
    SourceSpec("bs1_vxm_v0_2", "VoxelMorph", "#737373", 1),
    SourceSpec("bs2_tm_v0", "TransMorph", "#737373", 1),
    SourceSpec("bs3_vfa_v0", "VFA", "#737373", 1),
    SourceSpec("bs4_sitreg_v0", "SITReg", "#737373", 1),
    SourceSpec("greedy", "Greedy", "#737373", 1),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oasis-source", type=Path, default=None)
    parser.add_argument("--lung-source", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def default_source_paths(artifact_root: Path) -> tuple[Path, Path]:
    source_root = artifact_root / CAMERA_READY_ARCHIVE
    return source_root / OASIS_SOURCE_FILENAME, source_root / LUNG_SOURCE_FILENAME


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    configured_root = artifact_paths.artifact_root(repo_root=REPO_ROOT)
    default_oasis, default_lung = default_source_paths(configured_root)
    args.oasis_source = artifact_paths.resolve_external_input_file(
        default_oasis if args.oasis_source is None else args.oasis_source,
        label="--oasis-source",
        repo_root=REPO_ROOT,
    )
    args.lung_source = artifact_paths.resolve_external_input_file(
        default_lung if args.lung_source is None else args.lung_source,
        label="--lung-source",
        repo_root=REPO_ROOT,
    )
    args.review_root = artifact_paths.resolve_review_root(
        ARTIFACT_EXPERIMENT_KEY,
        args.review_root,
        repo_root=REPO_ROOT,
    )
    if int(args.dpi) <= 0:
        raise ValueError(f"--dpi must be positive, got {args.dpi!r}")
    return args


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No source rows found in {path}")
    return rows


def finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def optional_sort_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return result if math.isfinite(result) else float("inf")


def _sort_key(row: Mapping[str, str], columns: Sequence[str]) -> tuple[float, ...]:
    return tuple(optional_sort_float(row.get(column)) for column in columns)


def select_series(
    rows: Sequence[Mapping[str, str]],
    dataset: DatasetSpec,
    method: SourceSpec,
) -> PlotSeries:
    selected = [row for row in rows if str(row.get(dataset.key_column, "")) == method.source_key]
    selected.sort(key=lambda row: _sort_key(row, dataset.sort_columns))
    if len(selected) != method.expected_count:
        raise ValueError(
            f"{dataset.slug}:{method.source_key} expected {method.expected_count} rows, "
            f"found {len(selected)}"
        )
    points: list[PlotPoint] = []
    for index, row in enumerate(selected):
        values = tuple(
            str(row.get(column, ""))
            for column in (
                dataset.x_column,
                dataset.y_column,
                dataset.x_std_column,
                dataset.y_std_column,
            )
        )
        x = finite_float(values[0], label=f"{method.source_key}[{index}].x")
        y = finite_float(values[1], label=f"{method.source_key}[{index}].y")
        x_std = finite_float(values[2], label=f"{method.source_key}[{index}].x_std")
        y_std = finite_float(values[3], label=f"{method.source_key}[{index}].y_std")
        if x_std < 0 or y_std < 0:
            raise ValueError(f"{method.source_key}[{index}] has a negative standard deviation")
        points.append(PlotPoint(x, y, x_std, y_std, values))
    series = PlotSeries(method.source_key, method.display_label, method.color, tuple(points))
    assert_direct_source_identity(series, selected, dataset)
    return series


def assert_direct_source_identity(
    series: PlotSeries,
    selected_rows: Sequence[Mapping[str, str]],
    dataset: DatasetSpec,
) -> None:
    columns = (
        dataset.x_column,
        dataset.y_column,
        dataset.x_std_column,
        dataset.y_std_column,
    )
    source_values = tuple(tuple(str(row.get(column, "")) for column in columns) for row in selected_rows)
    plotted_values = tuple(point.raw_values for point in series.points)
    if plotted_values != source_values:
        raise AssertionError(f"Display selection changed numerical source values for {series.source_key}")


def select_dataset(rows: Sequence[Mapping[str, str]], dataset: DatasetSpec) -> tuple[PlotSeries, ...]:
    return tuple(select_series(rows, dataset, method) for method in dataset.methods)


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def series_sha256(series: PlotSeries) -> str:
    payload = json.dumps(
        [point.raw_values for point in series.points],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    red, green, blue = mcolors.to_rgb(color)
    return red, green, blue, alpha


def new_canvas(dataset: DatasetSpec, *, decorations: bool, transparent: bool) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AXES_RECT)
    ax.set_xlim(*dataset.axis.xlim)
    ax.set_ylim(*dataset.axis.ylim)
    ax.set_xticks(dataset.axis.xticks)
    ax.set_yticks(dataset.axis.yticks)
    if decorations:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_title(dataset.axis.title, fontsize=STYLE["title_fontsize"], pad=13)
        ax.set_xlabel(dataset.axis.xlabel, fontsize=STYLE["axis_label_fontsize"], labelpad=8)
        ax.set_ylabel(dataset.axis.ylabel, fontsize=STYLE["axis_label_fontsize"], labelpad=8)
        ax.tick_params(axis="both", labelsize=STYLE["tick_fontsize"])
        ax.grid(True, alpha=STYLE["grid_alpha"], linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.9)
    else:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    return fig, ax


def draw_curve(ax: plt.Axes, series: PlotSeries) -> None:
    xs = [point.x for point in series.points]
    ys = [point.y for point in series.points]
    ax.plot(
        xs,
        ys,
        color=series.color,
        linewidth=STYLE["linewidth"],
        marker="o",
        markersize=STYLE["markersize"],
        markerfacecolor=series.color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        clip_on=True,
        zorder=3,
    )


def draw_ellipses(ax: plt.Axes, series: PlotSeries, *, reference: bool = False) -> None:
    face_alpha = STYLE["reference_face_alpha"] if reference else STYLE["ellipse_face_alpha"]
    edge_alpha = STYLE["reference_edge_alpha"] if reference else STYLE["ellipse_edge_alpha"]
    for point in series.points:
        ax.add_patch(
            Ellipse(
                (point.x, point.y),
                width=2.0 * point.x_std,
                height=2.0 * point.y_std,
                facecolor=rgba(series.color, face_alpha),
                edgecolor=rgba(series.color, edge_alpha),
                linewidth=0.9,
                clip_on=True,
                zorder=1,
            )
        )


def draw_reference_point(ax: plt.Axes, series: PlotSeries) -> None:
    point = series.points[0]
    ax.plot(
        [point.x],
        [point.y],
        linestyle="none",
        marker="D",
        markersize=STYLE["reference_markersize"],
        markerfacecolor=series.color,
        markeredgecolor="white",
        markeredgewidth=0.7,
        clip_on=True,
        zorder=4,
    )


def draw_ptvreg(ax: plt.Axes, series: PlotSeries) -> None:
    point = series.points[0]
    ax.plot(
        [point.x],
        [point.y],
        linestyle="none",
        marker="*",
        markersize=STYLE["ptvreg_markersize"],
        markerfacecolor=series.color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        clip_on=True,
        zorder=5,
    )


def draw_legend_entry(
    fig: plt.Figure,
    label: str,
    color: str,
    row: int,
    *,
    marker: str = "o",
    y_override: float | None = None,
    fontsize: float | None = None,
) -> None:
    y = LEGEND_Y[row] if y_override is None else y_override
    reference_marker = marker in {"*", "D"}
    x_values = [LEGEND_X + 0.020] if reference_marker else [LEGEND_X, LEGEND_X + 0.040]
    y_values = [y] if reference_marker else [y, y]
    artist = Line2D(
        x_values,
        y_values,
        transform=fig.transFigure,
        color=color,
        linewidth=0.0 if reference_marker else STYLE["linewidth"],
        linestyle="none" if reference_marker else "-",
        marker=marker,
        markersize=(STYLE["ptvreg_markersize"] if marker == "*" else STYLE["markersize"]),
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        clip_on=False,
    )
    fig.add_artist(artist)
    fig.text(
        LEGEND_X + 0.055,
        y,
        label,
        ha="left",
        va="center",
        fontsize=STYLE["legend_fontsize"] if fontsize is None else fontsize,
        color="#202020" if color != "#737373" else "#595959",
    )


def save_figure(fig: plt.Figure, path: Path, *, dpi: int, transparent: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        transparent=transparent,
        facecolor="none" if transparent else "white",
        edgecolor="none",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)
    if path.suffix.lower() == ".png" and not transparent:
        with Image.open(path) as image:
            converted = image.convert("RGB")
            converted.save(path, dpi=(dpi, dpi))


def render_base(path: Path, dataset: DatasetSpec, *, dpi: int) -> None:
    fig, _ax = new_canvas(dataset, decorations=True, transparent=False)
    save_figure(fig, path, dpi=dpi, transparent=False)


def render_method_layer(path: Path, dataset: DatasetSpec, series: PlotSeries, row: int, *, dpi: int) -> None:
    fig, ax = new_canvas(dataset, decorations=False, transparent=True)
    draw_curve(ax, series)
    draw_legend_entry(fig, series.display_label, series.color, row)
    save_figure(fig, path, dpi=dpi, transparent=True)


def render_uncertainty_layer(
    path: Path,
    dataset: DatasetSpec,
    series: Sequence[PlotSeries],
    *,
    dpi: int,
) -> None:
    fig, ax = new_canvas(dataset, decorations=False, transparent=True)
    for item in series:
        draw_ellipses(ax, item)
    save_figure(fig, path, dpi=dpi, transparent=True)


def render_ptvreg_layer(path: Path, series: PlotSeries, *, dpi: int) -> None:
    fig, ax = new_canvas(LUNG_SPEC, decorations=False, transparent=True)
    draw_ptvreg(ax, series)
    draw_legend_entry(
        fig,
        series.display_label,
        series.color,
        0,
        marker="*",
        y_override=PTVREG_LEGEND_Y,
    )
    save_figure(fig, path, dpi=dpi, transparent=True)


def composite_layers(paths: Sequence[Path], output_path: Path, *, dpi: int) -> None:
    if not paths:
        raise ValueError("At least one layer is required")
    with Image.open(paths[0]) as first:
        canvas = first.convert("RGBA")
    for path in paths[1:]:
        with Image.open(path) as layer:
            overlay = layer.convert("RGBA")
            if overlay.size != canvas.size:
                raise AssertionError(f"Layer size mismatch: {path} is {overlay.size}, expected {canvas.size}")
            canvas.alpha_composite(overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, dpi=(dpi, dpi))


def render_complete_vector(
    path: Path,
    dataset: DatasetSpec,
    methods: Sequence[PlotSeries],
    *,
    ellipses: bool,
    ptvreg: PlotSeries | None = None,
    references: Sequence[PlotSeries] = (),
    dpi: int,
) -> None:
    fig, ax = new_canvas(dataset, decorations=True, transparent=False)
    if ellipses:
        for series in methods:
            draw_ellipses(ax, series)
    for row, series in enumerate(methods):
        draw_curve(ax, series)
        draw_legend_entry(fig, series.display_label, series.color, row)
    if ptvreg is not None:
        draw_ptvreg(ax, ptvreg)
        draw_legend_entry(
            fig,
            ptvreg.display_label,
            ptvreg.color,
            0,
            marker="*",
            y_override=PTVREG_LEGEND_Y,
        )
    if references:
        for reference in references:
            draw_ellipses(ax, reference, reference=True)
            draw_reference_point(ax, reference)
        fig.text(LEGEND_X, 0.405, "References", fontsize=12, weight="bold", color="#555555")
        for index, reference in enumerate(references):
            draw_legend_entry(
                fig,
                reference.display_label,
                reference.color,
                0,
                marker="D",
                y_override=0.36 - index * 0.052,
                fontsize=STYLE["reference_legend_fontsize"],
            )
    save_figure(fig, path, dpi=dpi, transparent=False)


def render_context_png(
    path: Path,
    methods: Sequence[PlotSeries],
    references: Sequence[PlotSeries],
    *,
    dpi: int,
) -> None:
    render_complete_vector(
        path,
        OASIS_SPEC,
        methods,
        ellipses=True,
        references=references,
        dpi=dpi,
    )


def expected_pixel_size(dpi: int) -> tuple[int, int]:
    return int(round(FIGSIZE[0] * dpi)), int(round(FIGSIZE[1] * dpi))


def validate_image_group(paths: Sequence[Path], *, expected_size: tuple[int, int]) -> None:
    for path in paths:
        with Image.open(path) as image:
            if image.size != expected_size:
                raise AssertionError(f"{path.name}: size {image.size} != {expected_size}")


def validate_overlay(path: Path, *, expected_size: tuple[int, int]) -> None:
    with Image.open(path) as image:
        if image.size != expected_size:
            raise AssertionError(f"{path.name}: size {image.size} != {expected_size}")
        if image.mode != "RGBA":
            raise AssertionError(f"{path.name}: expected RGBA overlay, got {image.mode}")
        alpha = image.getchannel("A")
        extrema = alpha.getextrema()
        if extrema[0] != 0 or extrema[1] == 0:
            raise AssertionError(f"{path.name}: overlay alpha extrema are {extrema}")
        if alpha.getpixel((0, 0)) != 0:
            raise AssertionError(f"{path.name}: top-left background pixel is not transparent")


def validate_base(path: Path, *, expected_size: tuple[int, int]) -> None:
    with Image.open(path) as image:
        if image.size != expected_size:
            raise AssertionError(f"{path.name}: size {image.size} != {expected_size}")
        if image.mode != "RGB":
            raise AssertionError(f"{path.name}: expected opaque RGB base, got {image.mode}")


def assert_same_pixels(left: Path, right: Path) -> None:
    with Image.open(left) as a, Image.open(right) as b:
        if a.mode != b.mode or a.size != b.size or a.tobytes() != b.tobytes():
            raise AssertionError(f"Pixel mismatch between {left.name} and {right.name}")


def make_contact_sheet(paths: Sequence[Path], labels: Sequence[str], output_path: Path, *, dpi: int) -> None:
    if len(paths) != 4 or len(labels) != 4:
        raise ValueError("Contact sheet requires exactly four paths and labels")
    sheet_width, sheet_height = expected_pixel_size(dpi)
    margin = max(18, int(0.02 * sheet_width))
    title_height = max(42, int(0.052 * sheet_height))
    cell_width = (sheet_width - 3 * margin) // 2
    cell_height = (sheet_height - 3 * margin) // 2
    preview_height = cell_height - title_height
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=max(16, int(sheet_width / 115)))
    for index, (path, label) in enumerate(zip(paths, labels)):
        row, column = divmod(index, 2)
        x0 = margin + column * (cell_width + margin)
        y0 = margin + row * (cell_height + margin)
        with Image.open(path) as image:
            preview = ImageOps.contain(image.convert("RGB"), (cell_width, preview_height))
        px = x0 + (cell_width - preview.width) // 2
        py = y0 + title_height + (preview_height - preview.height) // 2
        sheet.paste(preview, (px, py))
        draw.text((x0 + cell_width // 2, y0 + title_height // 2), label, fill="#202020", font=font, anchor="mm")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, dpi=(dpi, dpi))


def render_dataset_assets(
    root: Path,
    dataset: DatasetSpec,
    series: Sequence[PlotSeries],
    *,
    dpi: int,
    ptvreg: PlotSeries | None = None,
) -> dict[str, list[Path]]:
    static_dir = root / "static"
    overlay_dir = root / "overlays"
    cumulative_dir = root / "cumulative"
    static_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    cumulative_dir.mkdir(parents=True, exist_ok=True)

    base = overlay_dir / f"{dataset.slug}_anim_base.png"
    render_base(base, dataset, dpi=dpi)
    method_layers: list[Path] = []
    layer_slugs = ("inr_dense", "inr_bscp", "d_bscp", "mr_d_bscp")
    for index, (item, layer_slug) in enumerate(zip(series, layer_slugs), start=1):
        layer = overlay_dir / f"{dataset.slug}_anim_{index:02d}_{layer_slug}.png"
        render_method_layer(layer, dataset, item, index - 1, dpi=dpi)
        method_layers.append(layer)
    uncertainty = overlay_dir / f"{dataset.slug}_anim_05_uncertainty.png"
    render_uncertainty_layer(uncertainty, dataset, series, dpi=dpi)

    ptvreg_layer: Path | None = None
    if ptvreg is not None:
        ptvreg_layer = overlay_dir / "lung_anim_06_ptvreg.png"
        render_ptvreg_layer(ptvreg_layer, ptvreg, dpi=dpi)

    if dataset.slug == "oasis":
        no_ellipse_png = static_dir / "01_oasis_no_ellipses.png"
        with_ellipse_png = static_dir / "02_oasis_with_ellipses.png"
        no_ellipse_svg = static_dir / "01_oasis_no_ellipses.svg"
        with_ellipse_svg = static_dir / "02_oasis_with_ellipses.svg"
    else:
        no_ellipse_png = static_dir / "03_lung_no_ellipses.png"
        with_ellipse_png = static_dir / "04_lung_with_ellipses.png"
        no_ellipse_svg = static_dir / "03_lung_no_ellipses.svg"
        with_ellipse_svg = static_dir / "04_lung_with_ellipses.svg"

    composite_layers([base, *method_layers], no_ellipse_png, dpi=dpi)
    final_layers = [base, *method_layers, uncertainty]
    if ptvreg_layer is not None:
        final_layers.append(ptvreg_layer)
    composite_layers(final_layers, with_ellipse_png, dpi=dpi)
    render_complete_vector(no_ellipse_svg, dataset, series, ellipses=False, dpi=dpi)
    render_complete_vector(
        with_ellipse_svg,
        dataset,
        series,
        ellipses=True,
        ptvreg=ptvreg,
        dpi=dpi,
    )

    cumulative: list[Path] = []
    for index in range(1, 5):
        path = cumulative_dir / f"{dataset.slug}_step_{index:02d}.png"
        composite_layers([base, *method_layers[:index]], path, dpi=dpi)
        cumulative.append(path)
    final_path = cumulative_dir / f"{dataset.slug}_step_05_final.png"
    shutil.copyfile(with_ellipse_png, final_path)
    cumulative.append(final_path)

    expected_size = expected_pixel_size(dpi)
    overlay_paths = [*method_layers, uncertainty]
    if ptvreg_layer is not None:
        overlay_paths.append(ptvreg_layer)
    validate_base(base, expected_size=expected_size)
    for layer in overlay_paths:
        validate_overlay(layer, expected_size=expected_size)
    validate_image_group([no_ellipse_png, with_ellipse_png, *cumulative], expected_size=expected_size)
    assert_same_pixels(with_ellipse_png, final_path)

    return {
        "static": [no_ellipse_png, no_ellipse_svg, with_ellipse_png, with_ellipse_svg],
        "overlays": [base, *overlay_paths],
        "cumulative": cumulative,
    }


def series_summary(series: PlotSeries) -> dict[str, Any]:
    xs = [point.x for point in series.points]
    ys = [point.y for point in series.points]
    return {
        "source_key": series.source_key,
        "display_label": series.display_label,
        "color": series.color,
        "points": len(series.points),
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "source_array_sha256": series_sha256(series),
    }


def symbolic_path(path: Path, artifact_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(artifact_root.resolve())
    except ValueError:
        return f"<external>/{path.name}"
    return "${ARTIFACT_ROOT}/" + relative.as_posix()


def relative_outputs(paths: Iterable[Path], root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in paths)


def write_review_report(root: Path, manifest: Mapping[str, Any]) -> Path:
    report_path = root / "review_report.md"
    lines = [
        "# Presentation ARC Plot Review",
        "",
        f"- Generated: `{manifest['generated_utc']}`",
        "- Numerical source rows were selected directly from the archived camera-ready v013 CSVs.",
        "- No registration experiment was run and no plotted numerical value was recomputed.",
        "- The lung D-BSCP weak-regularization point above TRE 2.05 is intentionally clipped by the camera-ready axis policy.",
        f"- Canvas: `{manifest['canvas']['pixel_width']} x {manifest['canvas']['pixel_height']} px` at `{manifest['canvas']['dpi']} DPI`.",
        "",
        "## Method colors",
        "",
    ]
    for label in METHOD_ORDER:
        lines.append(f"- `{label}`: `{METHOD_COLORS[label]}`")
    lines.extend(["", "## Selected source series", ""])
    for dataset_name in ("oasis", "lung"):
        lines.append(f"### {dataset_name.upper()}")
        lines.append("")
        lines.append("| Display label | Source key | Points | x range | y range |")
        lines.append("|---|---|---:|---:|---:|")
        for item in manifest["series"][dataset_name]:
            lines.append(
                f"| {item['display_label']} | `{item['source_key']}` | {item['points']} | "
                f"{item['x_min']:.6f}-{item['x_max']:.6f} | "
                f"{item['y_min']:.6f}-{item['y_max']:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## QA",
            "",
            "- Selected x/y means and standard deviations match their source-row strings exactly after deterministic sorting.",
            "- All overlays are aligned RGBA images with transparent backgrounds.",
            "- All base images are opaque RGB images.",
            "- Each cumulative final image is pixel-identical to its corresponding final static PNG.",
            "",
            "## Outputs",
            "",
        ]
    )
    for output in manifest["outputs"]:
        lines.append(f"- `{output}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build(args: argparse.Namespace) -> dict[str, Any]:
    oasis_rows = load_csv(args.oasis_source)
    lung_rows = load_csv(args.lung_source)
    oasis_series = select_dataset(oasis_rows, OASIS_SPEC)
    lung_series = select_dataset(lung_rows, LUNG_SPEC)
    ptvreg = select_series(lung_rows, LUNG_SPEC, PTVREG_SPEC)
    references = tuple(select_series(oasis_rows, OASIS_SPEC, spec) for spec in OASIS_REFERENCE_SPECS)

    validation_summary = {
        "oasis": [series_summary(series) for series in oasis_series],
        "lung": [series_summary(series) for series in lung_series],
        "ptvreg": series_summary(ptvreg),
        "oasis_references": [series_summary(series) for series in references],
    }
    if args.validate_only:
        return {
            "status": "validated",
            "sources": {
                "oasis_sha256": source_sha256(args.oasis_source),
                "lung_sha256": source_sha256(args.lung_source),
            },
            "series": validation_summary,
        }

    root = Path(args.review_root)
    root.mkdir(parents=True, exist_ok=True)
    oasis_outputs = render_dataset_assets(root, OASIS_SPEC, oasis_series, dpi=args.dpi)
    lung_outputs = render_dataset_assets(root, LUNG_SPEC, lung_series, dpi=args.dpi, ptvreg=ptvreg)

    context_png = root / "static" / "oasis_with_ellipses_and_references.png"
    context_svg = root / "static" / "oasis_with_ellipses_and_references.svg"
    render_context_png(context_png, oasis_series, references, dpi=args.dpi)
    render_complete_vector(
        context_svg,
        OASIS_SPEC,
        oasis_series,
        ellipses=True,
        references=references,
        dpi=args.dpi,
    )
    expected_size = expected_pixel_size(args.dpi)
    validate_image_group([context_png], expected_size=expected_size)

    contact_sheet = root / "contact_sheet.png"
    make_contact_sheet(
        [
            root / "static" / "01_oasis_no_ellipses.png",
            root / "static" / "02_oasis_with_ellipses.png",
            root / "static" / "03_lung_no_ellipses.png",
            root / "static" / "04_lung_with_ellipses.png",
        ],
        ("OASIS: curves", "OASIS: curves + uncertainty", "Lung: curves", "Lung: curves + uncertainty + pTVReg"),
        contact_sheet,
        dpi=args.dpi,
    )

    output_paths = [
        *oasis_outputs["static"],
        *oasis_outputs["overlays"],
        *oasis_outputs["cumulative"],
        *lung_outputs["static"],
        *lung_outputs["overlays"],
        *lung_outputs["cumulative"],
        context_png,
        context_svg,
        contact_sheet,
    ]
    artifact_root = artifact_paths.artifact_root(repo_root=REPO_ROOT)
    manifest: dict[str, Any] = {
        "artifact": "camera_ready_arc_presentation_assets",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_inputs": {
            "oasis": symbolic_path(args.oasis_source, artifact_root),
            "lung": symbolic_path(args.lung_source, artifact_root),
            "oasis_sha256": source_sha256(args.oasis_source),
            "lung_sha256": source_sha256(args.lung_source),
            "camera_ready_builder": "experiments/paper_artifacts/pareto_figure/build_final_dirlab4dct_axes_labels.py",
        },
        "canvas": {
            "figsize_inches": list(FIGSIZE),
            "dpi": int(args.dpi),
            "pixel_width": expected_size[0],
            "pixel_height": expected_size[1],
            "axes_rect": list(AXES_RECT),
        },
        "axes": {
            "oasis": {"xlim": list(OASIS_SPEC.axis.xlim), "ylim": list(OASIS_SPEC.axis.ylim)},
            "lung": {"xlim": list(LUNG_SPEC.axis.xlim), "ylim": list(LUNG_SPEC.axis.ylim)},
        },
        "method_order": list(METHOD_ORDER),
        "method_colors": METHOD_COLORS,
        "series": {
            "oasis": validation_summary["oasis"],
            "lung": validation_summary["lung"],
            "ptvreg": validation_summary["ptvreg"],
            "oasis_references": validation_summary["oasis_references"],
        },
        "validation": {
            "direct_source_string_identity": True,
            "overlay_rgba_and_transparent": True,
            "base_rgb_and_opaque": True,
            "all_png_dimensions_match": True,
            "cumulative_final_matches_static_final": True,
            "registration_recomputed": False,
        },
        "outputs": relative_outputs(output_paths, root),
    }
    manifest_path = root / "presentation_arc_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = write_review_report(root, manifest)
    manifest["outputs"] = relative_outputs([*output_paths, manifest_path, report_path], root)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = resolve_paths(parse_args(argv))
    manifest = build(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
