#!/usr/bin/env python3
"""Create the fit-modes method-comparison review notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


SOURCE_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = SOURCE_ROOT / "fit_modes_method_comparison.ipynb"


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

The final section builds a preliminary field-fit-only figure with relative
target-field MSE (`rMSE`) and relative target-field RMSE (`rRMSE`). It includes
both OASIS rank-9 SITReg-field and Greedy-field fitting, then LungCT pTVReg
case01/case08 field fitting. The relative metrics reuse the same masked
normalized-grid MSE histories and divide by the target-field power in the same
mask and coordinate convention. A second rRMSE-only trajectory figure compares
the requested learning rates for IDIR, SINR cps4, DirectCP cps4, and available
LungCT DirectCP 4-stage diagnostics.
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

import matplotlib.pyplot as plt
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
        SOURCE_ROOT = candidate
        break
else:
    raise FileNotFoundError("Could not locate fit_modes_method_comparison.py")

sys.path.insert(0, str(SOURCE_ROOT))
print(SOURCE_ROOT)
"""
        ),
        code_cell(
            """
from fit_modes_method_comparison import (
    RELATIVE_FIELD_CASE_ORDER,
    build_rrmse_lr_trajectory_outputs,
    load_curves,
    load_relative_field_fit_curves,
    make_main_figure,
    make_relative_field_fit_figure,
    relative_field_curve_summary,
    resolve_fitting_output_dir,
    save_relative_field_fit_figure,
    save_main_figure,
    selected_runs,
    curve_summary,
    target_power_summary,
)

ARTIFACT_OUTPUT_DIR = resolve_fitting_output_dir()
ARTIFACT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Artifact output directory: {ARTIFACT_OUTPUT_DIR}")
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
selection.to_csv(ARTIFACT_OUTPUT_DIR / "fit_modes_method_comparison_selected_runs.csv", index=False)
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
summary.to_csv(ARTIFACT_OUTPUT_DIR / "fit_modes_method_comparison_curve_summary.csv", index=False)
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
display(fig)
plt.close(fig)
existing_outputs = {
    "pdf": ARTIFACT_OUTPUT_DIR / "fit_modes_method_comparison_curves.pdf",
    "png": ARTIFACT_OUTPUT_DIR / "fit_modes_method_comparison_curves.png",
}
display(Markdown(
    "Combined-figure paths in the external artifact store (not rewritten by this cell):\\n"
    + "\\n".join([f"- `{name}`: `{path}`" for name, path in existing_outputs.items()])
))
"""
        ),
        markdown_cell(
            """
## Preliminary Relative Field-Fit Figure

This section leaves the image-registration panel unchanged and replots only the
target-field fitting curves. It adds the OASIS Greedy-field target beside the
OASIS SITReg-field target. The denominator is inferred per target from the
selected DirectCP cps4 first point: DirectCP starts from zero controls, so its
epoch-1 masked MSE equals the masked target-field component power.
"""
        ),
        code_cell(
            """
relative_field_curves = load_relative_field_fit_curves(selection)
target_powers = target_power_summary(relative_field_curves, RELATIVE_FIELD_CASE_ORDER)
display(target_powers)
target_powers.to_csv(ARTIFACT_OUTPUT_DIR / "fit_modes_target_power_summary.csv", index=False)
"""
        ),
        code_cell(
            """
relative_summary = relative_field_curve_summary(relative_field_curves, target_powers)
display(
    relative_summary[
        [
            "case",
            "method",
            "lr",
            "target_power_component_mean",
            "rMSE_start",
            "rMSE_epoch10",
            "rMSE_end",
            "rRMSE_percent_start",
            "rRMSE_percent_epoch10",
            "rRMSE_percent_end",
            "has_epoch10",
        ]
    ]
)
relative_summary.to_csv(ARTIFACT_OUTPUT_DIR / "fit_modes_relative_field_fit_curve_summary.csv", index=False)
"""
        ),
        code_cell(
            """
relative_fig = make_relative_field_fit_figure(
    relative_field_curves,
    target_powers,
    smooth=True,
    case_order=RELATIVE_FIELD_CASE_ORDER,
)
relative_outputs = save_relative_field_fit_figure(
    relative_fig,
    review_root=ARTIFACT_OUTPUT_DIR,
)
display(Markdown("\\n".join([f"- `{name}`: `{path}`" for name, path in relative_outputs.items()])))
display(Image(filename=str(relative_outputs["png"])))
"""
        ),
        markdown_cell(
            """
## rRMSE Learning-Rate Trajectory Figure

This field-fit-only figure compares requested learning-rate trajectories:
IDIR and SINR cps4 at `1e-3`/`1e-4`, DirectCP cps4 at
`1e-2`/`1e-3`/`1e-4`, and LungCT DirectCP 4-stage at
`1e-2`/`1e-3`/`1e-4`. DirectCP 4-stage is not shown for OASIS because no
OASIS 4-stage field-fit run is included in this comparison. The shared y-axis
is capped at `100%` rRMSE.
"""
        ),
        code_cell(
            """
lr_selected, lr_target_powers, lr_summary, lr_fig, lr_outputs = build_rrmse_lr_trajectory_outputs(
    review_root=ARTIFACT_OUTPUT_DIR,
)
display(lr_selected.drop(columns=["source"]))
display(
    lr_summary[
        [
            "case",
            "method",
            "lr_label",
            "points",
            "rRMSE_percent_start",
            "rRMSE_percent_epoch10",
            "rRMSE_percent_epoch50",
            "rRMSE_percent_end",
            "has_epoch10",
            "has_epoch50",
        ]
    ]
)
display(Markdown("\\n".join([f"- `{name}`: `{path}`" for name, path in lr_outputs.items()])))
display(Image(filename=str(lr_outputs["png"])))
plt.close(lr_fig)
"""
        ),
        markdown_cell(
            """
## Output Files

The notebook writes or updates the following files only under the configured
external `ARTIFACT_OUTPUT_DIR` (`$ARTIFACT_ROOT/repo_experiments/...`):

- `fit_modes_method_comparison_selected_runs.csv`
- `fit_modes_method_comparison_curve_summary.csv`
- `fit_modes_relative_field_fit_prelim_curves.pdf`
- `fit_modes_relative_field_fit_prelim_curves.png`
- `fit_modes_relative_field_fit_curve_summary.csv`
- `fit_modes_target_power_summary.csv`
- `fit_modes_rrmse_lr_trajectory_curves.pdf`
- `fit_modes_rrmse_lr_trajectory_curves.png`
- `fit_modes_rrmse_lr_trajectory_curve_summary.csv`
- `fit_modes_rrmse_lr_trajectory_selected_runs.csv`
"""
        ),
    ]

    NOTEBOOK_PATH.write_text(nbf.writes(nb))
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
