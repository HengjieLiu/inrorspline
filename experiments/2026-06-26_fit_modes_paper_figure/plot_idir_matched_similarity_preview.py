#!/usr/bin/env python3
"""Plot the new IDIR matched-similarity eval curves."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


REVIEW_ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path("/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1")

CURVES = [
    {
        "title": "OASIS rank-9",
        "csv": RUN_ROOT
        / "oasis_rank009/idir_original_dense_siren_be_autograd/"
        / "oasis_rank009_idir_a1_lr1e-4_global_ncc_eval/image_similarity_eval.csv",
        "ylabel": "Global NCC loss",
        "scale": 1.0,
        "color": "#2f5d9e",
    },
    {
        "title": "LungCT case01",
        "csv": RUN_ROOT
        / "lungct_case01/idir_be_default_lungmask/"
        / "lungct_case01_idir_lr1e-4_ptv_lcc_eval/image_similarity_eval.csv",
        "ylabel": "pTV LCC loss (x1e6)",
        "scale": 1.0e-6,
        "color": "#0b7a53",
    },
    {
        "title": "LungCT case08",
        "csv": RUN_ROOT
        / "lungct_case08/idir_be_default_lungmask/"
        / "lungct_case08_idir_lr1e-4_ptv_lcc_eval/image_similarity_eval.csv",
        "ylabel": "pTV LCC loss (x1e6)",
        "scale": 1.0e-6,
        "color": "#b23a48",
    },
]


def read_curve(path: Path, scale: float) -> tuple[list[float], list[float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    epochs: list[float] = []
    losses: list[float] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(float(row["epoch"]))
            losses.append(float(row["similarity_loss"]) * scale)
    return epochs, losses


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), constrained_layout=True)
    for ax, spec in zip(axes, CURVES, strict=True):
        epochs, losses = read_curve(spec["csv"], spec["scale"])
        ax.plot(epochs, losses, color=spec["color"], linewidth=1.8, label="IDIR")
        ax.scatter(
            [epochs[0], 10, 100, 200, epochs[-1]],
            [
                losses[epochs.index(1.0)],
                losses[epochs.index(10.0)],
                losses[epochs.index(100.0)],
                losses[epochs.index(200.0)],
                losses[-1],
            ],
            color=spec["color"],
            s=12,
            zorder=3,
        )
        ax.set_title(spec["title"], pad=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(spec["ylabel"])
        ax.set_xlim(0, 2500)
        ax.grid(True, color="#d8dce2", linewidth=0.5, alpha=0.8)
        ax.legend(loc="best", frameon=False)

    for ext in ("pdf", "png"):
        out = REVIEW_ROOT / f"idir_matched_similarity_curves_preview.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(out)


if __name__ == "__main__":
    main()
