#!/usr/bin/env python3
"""Create the fit-modes method-comparison review notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


REVIEW_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = REVIEW_ROOT / "fit_modes_method_comparison.ipynb"


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def main() -> None:
    nb = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        }
    )

    nb.cells = [
        markdown_cell(
            """
# Fit-Modes Method Comparison Curves

This notebook assembles the paper-figure curve preview for the OASIS rank-9
case and LungCT cases 01/08.

Panel A compares image-registration similarity curves. IDIR uses the newly
rerun matched diagnostic metric every 10 epochs; SINR/DirectCP use their
per-epoch matched data-loss histories. LungCT DirectCP 4-stage registration
uses the existing posthoc `finest_data_loss` diagnostic, not stage-domain
training loss. Panel B compares target-field fitting MSE curves against the
SITReg target for OASIS and the pTVReg target for LungCT; LungCT DirectCP
4-stage field fitting uses the rerun posthoc finest-domain MSE diagnostics.
"""
        ),
        markdown_cell(
            """
## Figure Message

The field-fitting objective is easy for local-control spline methods: DirectCP
fits the target fields very well on all three cases. Registration tells a
different story. OASIS is comparatively forgiving, LungCT case01 is easier, and
LungCT case08 exposes the need for a multistage optimization path rather than
a single cps4 spline solve.
"""
        ),
        code_cell(
            """
%matplotlib inline
from pathlib import Path
import sys

import pandas as pd
from IPython.display import Image, Markdown, display

pd.options.display.max_columns = 80
pd.options.display.max_colwidth = 120

candidates = [
    Path.cwd(),
    Path("/homebase/code_sync/inrorspline/experiments/2026-06-26_fit_modes_paper_figure"),
    Path("experiments/2026-06-26_fit_modes_paper_figure").resolve(),
]
for candidate in candidates:
    if (candidate / "fit_modes_method_comparison.py").exists():
        REVIEW_ROOT = candidate
        break
else:
    raise FileNotFoundError("Could not locate fit_modes_method_comparison.py")

sys.path.insert(0, str(REVIEW_ROOT))
print(REVIEW_ROOT)
"""
        ),
        code_cell(
            """
from fit_modes_method_comparison import (
    build_outputs,
    load_curves,
    make_main_figure,
    save_main_figure,
    selected_runs,
    curve_summary,
)
"""
        ),
        markdown_cell(
            """
## Selected Runs

One LR is selected per method/case/task. Registration IDIR is fixed to
`lr=1e-4` because that is the matched-metric rerun. Other registration LRs are
selected by final label metric: Dice for OASIS and native-crop snap TRE for
LungCT, except LungCT DirectCP 4-stage, which is fixed to the existing
`lr=1e-2` finest-domain diagnostic run. Field-fitting LRs are selected by
objective target-field MSE.
"""
        ),
        code_cell(
            """
selection = selected_runs()
display(selection.drop(columns=["run_dir"]))
selection.to_csv(REVIEW_ROOT / "fit_modes_method_comparison_selected_runs.csv", index=False)
"""
        ),
        markdown_cell(
            """
## Curve Inputs

IDIR registration curves have 251 points: epoch 1 plus every 10 epochs through
2500. Other registration histories are per-epoch curves. LungCT DirectCP
4-stage registration uses `finest_data_loss`. LungCT DirectCP 4-stage
field-fitting also uses `finest_data_loss` from the rerun diagnostic histories:
coarse stages are evaluated every 50 epochs, and the final full-resolution
stage is evaluated every epoch.
"""
        ),
        code_cell(
            """
reg_curves, field_curves = load_curves(selection)
summary = pd.concat(
    [
        curve_summary(reg_curves).assign(curve_group="registration"),
        curve_summary(field_curves).assign(curve_group="field_fit"),
    ],
    ignore_index=True,
)
display(summary.drop(columns=["source"]))
summary.to_csv(REVIEW_ROOT / "fit_modes_method_comparison_curve_summary.csv", index=False)
"""
        ),
        markdown_cell(
            """
## Main Comparison Figure

The display applies a centered rolling median only for readability. Set
`smooth=False` in `make_main_figure(...)` below to inspect unsmoothed curves.
The saved PDF/PNG use the same figure shown inline.
"""
        ),
        code_cell(
            """
fig = make_main_figure(reg_curves, field_curves, smooth=True)
outputs = save_main_figure(fig)
display(Markdown("\\n".join([f"- `{name}`: `{path}`" for name, path in outputs.items()])))
display(Image(filename=str(outputs["png"])))
"""
        ),
        markdown_cell(
            """
## Output Files

The notebook writes:

- `fit_modes_method_comparison_curves.pdf`
- `fit_modes_method_comparison_curves.png`
- `fit_modes_method_comparison_selected_runs.csv`
- `fit_modes_method_comparison_curve_summary.csv`
"""
        ),
    ]

    NOTEBOOK_PATH.write_text(nbf.writes(nb))
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
