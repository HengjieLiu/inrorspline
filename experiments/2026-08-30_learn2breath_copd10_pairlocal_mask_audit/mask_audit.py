"""Read-only inline visual audit for Learn2Breath COPD10 pair-local masks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba

PHASES = {"1": "INSP / fixed", "2": "EXP / moving"}
EXPECTED_CASE_IDS = [str(case) for case in range(104, 114)]
EXPECTED_IMAGE_SHAPE = (224, 192, 224)
EXPECTED_LOBE_VALUES = {0, 8, 16, 32, 64, 128}
LOBE_VALUES = [8, 16, 32, 64, 128]
LOBE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


@dataclass(frozen=True)
class MaskSource:
    """One supplied segmentation source rendered as a figure row."""

    slug: str
    row_label: str
    directory: str
    filename_suffix: str
    kind: Literal["binary", "lobes"]


SOURCES = (
    MaskSource("r231", "R231 whole lung", "r231", "whole_lung", "binary"),
    MaskSource(
        "totalseg_full_total",
        "TotalSegmentator\nfull-total lobes",
        "totalseg_full_total",
        "lobes",
        "lobes",
    ),
    MaskSource(
        "totalseg_roi_subset_lobes",
        "TotalSegmentator\nROI-subset lobes",
        "totalseg_roi_subset_lobes",
        "lobes",
        "lobes",
    ),
)

LOBE_CMAP = ListedColormap(
    [(0, 0, 0, 0)] + [to_rgba(color, alpha=0.48) for color in LOBE_COLORS]
)
LOBE_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1), LOBE_CMAP.N)


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load one NIfTI and retain array, affine, and voxel spacing."""
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.load(str(path))
    return (
        np.asanyarray(image.dataobj),
        image.affine,
        tuple(float(value) for value in image.header.get_zooms()[:3]),
    )


def _coronal_radiology_view(array_xyz: np.ndarray, y_index: int) -> np.ndarray:
    """Return a coronal view with S up and patient R on screen left."""
    return np.fliplr(np.asarray(array_xyz)[:, int(y_index), :].T)


def _lobe_display_indices(lobes_2d: np.ndarray) -> np.ndarray:
    """Map sparse TotalSegmentator lobe values to contiguous display labels."""
    indices = np.zeros(lobes_2d.shape, dtype=np.int8)
    for index, value in enumerate(LOBE_VALUES, start=1):
        indices[lobes_2d == value] = index
    return indices


def _draw_orientation_markers(axis: plt.Axes) -> None:
    style = {
        "transform": axis.transAxes,
        "color": "white",
        "fontsize": 8,
        "fontweight": "bold",
        "bbox": {
            "boxstyle": "round,pad=0.15",
            "facecolor": "black",
            "alpha": 0.55,
            "edgecolor": "none",
        },
    }
    axis.text(0.50, 0.98, "S", ha="center", va="top", **style)
    axis.text(0.50, 0.02, "I", ha="center", va="bottom", **style)
    axis.text(0.02, 0.50, "R", ha="left", va="center", **style)
    axis.text(0.98, 0.50, "L", ha="right", va="center", **style)


class MaskAudit:
    """Validated source access plus inline rendering for the ten paired cases."""

    def __init__(self, pairlocal_root: Path, mask_root: Path) -> None:
        self.pairlocal_root = pairlocal_root
        self.image_root = pairlocal_root / "imagesTs"
        self.mask_root = mask_root
        self.case_ids = self._discover_case_ids()
        self.validation = self._validate_all()
        self.positions = self._slice_position_table()

    def image_path(self, case_id: str, phase: str) -> Path:
        return self.image_root / f"case_{case_id}_{phase}.nii.gz"

    def mask_path(self, source: MaskSource, case_id: str, phase: str) -> Path:
        return (
            self.mask_root
            / source.directory
            / f"case_{case_id}_{phase}_{source.filename_suffix}.nii.gz"
        )

    def union_path(self, source: MaskSource, case_id: str, phase: str) -> Path:
        return (
            self.mask_root
            / source.directory
            / f"case_{case_id}_{phase}_whole_lung.nii.gz"
        )

    def _discover_case_ids(self) -> list[str]:
        pattern = re.compile(r"^case_(\d+)_[12]\.nii\.gz$")
        found: dict[str, set[str]] = {}
        for path in self.image_root.glob("case_*_[12].nii.gz"):
            match = pattern.match(path.name)
            if match is not None:
                case_id = match.group(1)
                phase = path.name.rsplit("_", 1)[-1].split(".", 1)[0]
                found.setdefault(case_id, set()).add(phase)
        incomplete = {
            case_id: sorted(set(PHASES) - phases)
            for case_id, phases in found.items()
            if set(PHASES) - phases
        }
        if incomplete:
            raise RuntimeError(f"Incomplete image pairs: {incomplete}")
        case_ids = sorted(found, key=int)
        if case_ids != EXPECTED_CASE_IDS:
            raise RuntimeError(
                f"Expected COPD10 case IDs 104--113, found {case_ids}"
            )
        return case_ids

    def _validate_all(self) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for case_id in self.case_ids:
            for phase in PHASES:
                image, image_affine, spacing_xyz = _load_nifti(
                    self.image_path(case_id, phase)
                )
                if image.shape != EXPECTED_IMAGE_SHAPE:
                    raise ValueError(
                        f"case {case_id} phase {phase}: expected "
                        f"{EXPECTED_IMAGE_SHAPE}, found {image.shape}"
                    )
                arrays: dict[str, np.ndarray] = {}
                for source in SOURCES:
                    array, affine, mask_spacing_xyz = _load_nifti(
                        self.mask_path(source, case_id, phase)
                    )
                    if array.shape != image.shape or not np.allclose(
                        affine, image_affine
                    ):
                        raise ValueError(
                            f"case {case_id} phase {phase} {source.slug}: "
                            "image/mask grid mismatch"
                        )
                    if not np.allclose(mask_spacing_xyz, spacing_xyz):
                        raise ValueError(
                            f"case {case_id} phase {phase} {source.slug}: "
                            "spacing mismatch"
                        )
                    labels = set(np.unique(array))
                    if source.kind == "binary" and not labels.issubset({0, 1}):
                        raise ValueError(
                            f"case {case_id} phase {phase}: R231 is not binary"
                        )
                    if source.kind == "lobes" and not labels.issubset(
                        EXPECTED_LOBE_VALUES
                    ):
                        raise ValueError(
                            f"case {case_id} phase {phase} {source.slug}: "
                            f"unexpected lobe values {sorted(labels)}"
                        )
                    if not np.any(array > 0):
                        raise ValueError(
                            f"case {case_id} phase {phase} {source.slug}: empty mask"
                        )
                    arrays[source.slug] = array
                    if source.kind == "lobes":
                        supplied_union, union_affine, _ = _load_nifti(
                            self.union_path(source, case_id, phase)
                        )
                        if supplied_union.shape != image.shape or not np.allclose(
                            union_affine, image_affine
                        ):
                            raise ValueError(
                                f"case {case_id} phase {phase} {source.slug}: "
                                "lobe-union grid mismatch"
                            )
                        if not np.array_equal(array > 0, supplied_union > 0):
                            raise ValueError(
                                f"case {case_id} phase {phase} {source.slug}: "
                                "lobe labels and supplied union disagree"
                            )
                records.append(
                    {
                        "case": case_id,
                        "phase": PHASES[phase],
                        "shape_xyz": str(tuple(int(value) for value in image.shape)),
                        "spacing_mm": str(
                            tuple(round(value, 2) for value in spacing_xyz)
                        ),
                        "R231_voxels": int((arrays["r231"] > 0).sum()),
                        "full-total_lobe_union_voxels": int(
                            (arrays["totalseg_full_total"] > 0).sum()
                        ),
                        "ROI-subset_lobe_union_voxels": int(
                            (arrays["totalseg_roi_subset_lobes"] > 0).sum()
                        ),
                        "validation": "pass",
                    }
                )
        return pd.DataFrame.from_records(records)

    def _pair_slice_positions(self, case_id: str) -> dict[str, int]:
        r231_insp, _, _ = _load_nifti(
            self.mask_path(SOURCES[0], case_id, "1")
        )
        r231_exp, _, _ = _load_nifti(self.mask_path(SOURCES[0], case_id, "2"))
        pair_union = np.logical_or(r231_insp > 0, r231_exp > 0)
        occupied_y = np.flatnonzero(pair_union.any(axis=(0, 2)))
        if not occupied_y.size:
            raise RuntimeError(f"case {case_id}: empty R231 pair-union extent")
        posterior_y = int(occupied_y[0])
        anterior_y = int(occupied_y[-1])
        return {
            "posterior_y": posterior_y,
            "p25_y": int(round(posterior_y + 0.25 * (anterior_y - posterior_y))),
            "p50_y": int(round(posterior_y + 0.50 * (anterior_y - posterior_y))),
            "p75_y": int(round(posterior_y + 0.75 * (anterior_y - posterior_y))),
            "anterior_y": anterior_y,
        }

    def _slice_position_table(self) -> pd.DataFrame:
        records = [
            {"case": case_id, **self._pair_slice_positions(case_id)}
            for case_id in self.case_ids
        ]
        table = pd.DataFrame.from_records(records)
        order = ["posterior_y", "p25_y", "p50_y", "p75_y", "anterior_y"]
        if not np.all(np.diff(table[order].to_numpy(), axis=1) >= 0):
            raise AssertionError("R231 coronal slice positions are not ordered")
        return table

    def _load_case_arrays(self, case_id: str) -> dict[str, dict[str, np.ndarray]]:
        arrays: dict[str, dict[str, np.ndarray]] = {"images": {}}
        for phase in PHASES:
            arrays["images"][phase], _, _ = _load_nifti(
                self.image_path(case_id, phase)
            )
        for source in SOURCES:
            arrays[source.slug] = {}
            for phase in PHASES:
                arrays[source.slug][phase], _, _ = _load_nifti(
                    self.mask_path(source, case_id, phase)
                )
        return arrays

    @staticmethod
    def _draw_overlay(
        axis: plt.Axes,
        image_xyz: np.ndarray,
        mask_xyz: np.ndarray,
        source: MaskSource,
        y_index: int,
    ) -> None:
        image_2d = _coronal_radiology_view(image_xyz, y_index)
        mask_2d = _coronal_radiology_view(mask_xyz, y_index)
        axis.imshow(
            image_2d,
            cmap="gray",
            vmin=-1000,
            vmax=300,
            origin="lower",
            interpolation="nearest",
        )
        if source.kind == "binary":
            if np.any(mask_2d > 0):
                axis.contour(
                    mask_2d > 0,
                    levels=[0.5],
                    colors=["#00e676"],
                    linewidths=1.35,
                )
        else:
            axis.imshow(
                _lobe_display_indices(mask_2d),
                cmap=LOBE_CMAP,
                norm=LOBE_NORM,
                origin="lower",
                interpolation="nearest",
            )
            for index, value in enumerate(LOBE_VALUES):
                selected = mask_2d == value
                if selected.any():
                    axis.contour(
                        selected,
                        levels=[0.5],
                        colors=[LOBE_COLORS[index]],
                        linewidths=0.85,
                    )
        axis.set_xticks([])
        axis.set_yticks([])
        _draw_orientation_markers(axis)

    def render_case(self, case_id: str) -> plt.Figure:
        """Display, but do not save, one requested three-row by six-column figure."""
        if case_id not in self.case_ids:
            raise KeyError(f"Unknown case {case_id}; expected one of {self.case_ids}")
        arrays = self._load_case_arrays(case_id)
        positions = self.positions.set_index("case").loc[case_id].to_dict()
        displayed_positions = (
            ("25%", int(positions["p25_y"])),
            ("50% / central", int(positions["p50_y"])),
            ("75%", int(positions["p75_y"])),
        )
        figure, axes = plt.subplots(3, 6, figsize=(22, 11), layout="constrained")
        figure.suptitle(
            "COPD case "
            f"{case_id}: shared R231 coronal extent "
            f"Y={int(positions['posterior_y'])} (P) to "
            f"Y={int(positions['anterior_y'])} (A)",
            fontsize=15,
            fontweight="bold",
        )
        for row, source in enumerate(SOURCES):
            for position_index, (position_label, y_index) in enumerate(
                displayed_positions
            ):
                for phase_offset, phase in enumerate(PHASES):
                    column = 2 * position_index + phase_offset
                    axis = axes[row, column]
                    self._draw_overlay(
                        axis,
                        arrays["images"][phase],
                        arrays[source.slug][phase],
                        source,
                        y_index,
                    )
                    if row == 0:
                        axis.set_title(
                            f"{position_label} | Y={y_index}\n{PHASES[phase]}",
                            fontsize=10,
                            fontweight="bold",
                        )
                    if column == 0:
                        axis.set_ylabel(
                            source.row_label, fontsize=11, fontweight="bold"
                        )
        plt.show()
        return figure


def run_audit(pairlocal_root: Path | str, mask_root: Path | str) -> MaskAudit:
    """Validate all supplied source grids and return the inline-only audit."""
    audit = MaskAudit(Path(pairlocal_root), Path(mask_root))
    if len(audit.validation) != 20 or not audit.validation["validation"].eq(
        "pass"
    ).all():
        raise AssertionError("Expected 20 successful image/mask validation rows")
    return audit
