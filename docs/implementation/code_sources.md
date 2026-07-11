---
doc_type: implementation_sources
status: active_draft
---

# Code Sources

This page maps the implementation summaries to the code and experiment records
they were derived from. It is not a full code index; it highlights the files
that determine the paper-facing technical behavior.

## External Method Sources

- `external/IDIR/`: upstream IDIR-style coordinate-network registration code.
  Important files:
  - `external/IDIR/models/models.py`: dense coordinate-network training loop,
    default arguments, sampled coordinate loss.
  - `external/IDIR/networks/networks.py`: SIREN and MLP definitions.
  - `external/IDIR/objectives/regularizers.py`: coordinate-autograd Jacobian,
    hyper-elastic, and bending-energy regularizers.
  - `external/IDIR/objectives/ncc.py`: global NCC implementation.

- `external/SINR/`: upstream SINR-style spline-enhanced INR code. Important
  files:
  - `external/SINR/models/models.py`: SIREN control generator, B-spline
    training loop, DeepAli B-spline expansion path in the original code.
  - `external/SINR/models/transformation.py`: separable cubic B-spline FFD
    transform, cubic basis formulas, and dense warping utilities.
  - `external/SINR/objectives/regularizers.py`: dense finite-difference
    diffusion and bending-energy losses.
  - `external/SINR/conf/`: original Hydra defaults for network,
    transformation, loss, regularization, and training.

## Repo-Local Experiment Implementations

The active paper-facing implementation behavior is in repo-local experiment
scripts, not in `src/`. The `src/` folder is currently a placeholder.

- `scripts/experiments/oasis_alpha_sweep/`
  - `common.py`: OASIS loading, SIREN, DirectCP generator, cubic B-spline
    expansion, analytic B-spline derivatives, IDIR autograd regularity,
    finite-difference metrics, Jacobian/NDV metrics, Dice, HD95, and ASD.
  - `run_one.py`: one-run implementation for IDIR, SINR cps2/cps4, DirectCP
    cps2/cps4, and OASIS multistage DirectCP.
  - `launch_first10_be_2500e.py`: seed-3 first10 alpha-sweep matrix.

- `scripts/experiments/lungct_no_sweep_repro/`
  - `common.py`: pTV-prep 4DCT loading, pTV local CC, isoTV, cubic B-spline
    expansion and derivatives, multistage control grids, TRE, and regularity
    metrics.
  - `run_one.py`: one-run implementation for LungCT IDIR, SINR cps4, DirectCP
    cps4, multistage DirectCP, global NCC + analytic B-spline BE, and global
    NCC + dense forward-difference BE variants.
  - Launch and aggregation scripts in the same folder define run matrices and
    review reports.

## Experiment Records Used For Specs

- `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/`: OASIS seed-3
  first10 IDIR/SINR/DirectCP BE sweep, population summaries, and review report.
- `experiments/2026-06-26_oasis_seed3_next10_alpha_sweep/`: OASIS next10
  extension using the same method conventions.
- `experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/`: canonical pTV-prep
  no-sweep LungCT reproduction for IDIR, SINR cps4, DirectCP cps4, and
  DirectCP 32-16-8-4.
- `experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/`: LungCT
  IDIR and DirectCP 32-16-8-4 regularization sweep.
- `experiments/2026-06-25_lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep/`
  and `experiments/2026-06-26_lungct_case08_be_impl_compare_alpha1e4/`:
  diagnostics comparing analytic B-spline BE and dense forward-difference BE
  training variants.

## Scope Notes

- Treat `external/IDIR/` and `external/SINR/` as source references and upstream
  implementation anchors.
- Treat repo-local scripts as the source of truth for the current paper-facing
  runs, because they encode OASIS foreground rules, seed-3 pair order, LungCT
  protocol tags, pTV-prep handling, and the current metric pipeline.
- Do not copy private absolute paths from run logs into this folder. Use
  relative repo paths and dataset/protocol handles instead.
