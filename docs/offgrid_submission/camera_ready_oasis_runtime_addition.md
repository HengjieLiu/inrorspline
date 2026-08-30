# Camera-Ready OASIS Runtime Addition

This note records the compact runtime text and table intended for the MICCAI
camera-ready paper. The editable LaTeX source corresponding to
`MICCAI_2026_Off_grid_CameraReady_v01.pdf` was not present under
`docs/offgrid_submission/`, so the actual paper-ready LaTeX insert is provided
in `camera_ready_oasis_runtime_addition.tex`.

## Insertion Point

Insert the paragraph and table in the Results section after the Fig. 2
OASIS/DIR-LAB registration-results discussion and before the supervised
target-field fitting diagnostic. This places runtime next to practical
registration performance without expanding the conclusion.

Add only this short conclusion clause if space permits:

> These results also show that direct multiresolution B-Spline optimization can
> be made practically fast, not only accurate.

## Runtime Definition

The table uses OASIS-only timing because OASIS image size is fixed across cases.
DIR-LAB is intentionally omitted from the compact runtime table because case
crop size and motion difficulty vary, making a single small runtime comparison
less controlled.

Runtime is defined as input images to saved dense displacement field:

- includes full pair-specific optimization;
- includes dense DVF output;
- excludes Dice, HD95, ASD, dense NCC, and other evaluation metrics;
- reports cold and warm process timings separately.

The reported step counts are optimizer updates for one image pair.

Cold timing means the first case in a fresh Python/CUDA process. Warm timing
means subsequent cases in a persistent process after one-time CUDA/library setup.

## Table Values

Current values combine OASIS100 accuracy from the selected-fast evaluation with
the 2026-08-15 cold/warm DVF-only timing rerun.

The paper table intentionally omits the fast-schedule name to save space. The
optimized schedules represented by each row are:

| Method | Optimized schedule |
|---|---|
| MR-D-BSCP (8-4-2) | `A_fast_decay` |
| D-BSCP (cps2) | `A_bal_lr10x` |
| D-BSCP (cps4) | `A_bal_lr10x` |
| INR-BSCP (cps2) | `A_bal_lr10x` |
| INR-BSCP (cps4) | `A_bal_lr10x` |

| Method | Dice | HD95 | Steps | Cold s | Warm s |
|---|---:|---:|---:|---:|---:|
| MR-D-BSCP (8-4-2) | 0.810->0.809 | 2.17->2.18 | 2500->438 | 155.2->97.2 | 61.3->10.1 |
| D-BSCP (cps2) | 0.805->0.805 | 2.33->2.33 | 2500->400 | 131.4->63.2 | 80.7->12.9 |
| D-BSCP (cps4) | 0.799->0.797 | 2.29->2.37 | 2500->477 | 125.0->65.9 | 81.3->15.5 |
| INR-BSCP (cps2) | 0.799->0.793 | 2.31->2.24 | 2500->821 | 337.0->163.2 | 275.2->90.2 |
| INR-BSCP (cps4) | 0.796->0.786 | 2.38->2.25 | 2500->671 | 167.4->77.9 | 108.1->28.9 |

Accuracy values come from the OASIS100 selected fast comparison against
alpha-1000 2500-step baselines:

- `experiments/2026-08-13_oasis_selected_fast_first100/fast_vs_2500_baselines.csv`
- `experiments/2026-08-13_oasis_selected_fast_first100/selected_fast_oasis100_report.md`

Cold/warm input-to-DVF-file values come from the OASIS paper cold/warm timing
rerun:

- `experiments/2026-08-15_oasis_paper_cold_warm_timing/paper_runtime_table_values.csv`
- `experiments/2026-08-15_oasis_paper_cold_warm_timing/paper_cold_warm_timing_report.md`

## Later Final Timing Rerun

Before final submission, rerun isolated OASIS timing with one GPU active and
other GPUs idle if possible:

- methods: the five 2500-step baselines plus the five selected fast schedules;
- cold timing: 10 independent fresh subprocess runs per method;
- warm timing: one persistent worker per method, one unreported warmup case,
  then 10 measured cases;
- endpoint: loaded input images to saved dense DVF;
- report mean and standard deviation for cold and warm runtime;
- record GPU model, CUDA/PyTorch versions, and peak GPU memory.

After that rerun, regenerate the table values and keep the current wording unless
the final timing changes the qualitative conclusion.
