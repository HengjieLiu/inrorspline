---
doc_type: paper_claims_crosswalk
status: active_draft
---

# Paper Claims Crosswalk

This file maps paper-facing claims to implementation sources, experiment
records, metrics, and caveats. It is a writing aid, not a results table.

## Claim: DirectCP Isolates The Spline/Local-Control Component

Proposed wording:

```text
On OASIS, direct optimization of the same cubic B-spline control structure
tests how much of SINR's behavior comes from the spline/local-control
parameterization rather than from the INR control generator.
```

Implementation source:

- `scripts/experiments/oasis_alpha_sweep/common.py`
- `scripts/experiments/oasis_alpha_sweep/run_one.py`
- `external/SINR/models/transformation.py`

Experiment source:

- `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/`
- `experiments/2026-06-26_oasis_seed3_next10_alpha_sweep/`

Metric family:

- Dice, HD95, ASD
- native analytic BE/diffusion
- dense forward-difference BE/diffusion
- log-Jacobian spread, folding percentage, NDV

Caveats:

- DirectCP and SINR share B-spline expansion but differ in optimizer parameter
  space: direct trainable controls versus SIREN-generated controls.
- IDIR alpha, SINR alpha, and DirectCP alpha should not be treated as
  numerically interchangeable.

## Claim: IDIR Behavior Depends On Deformation Regime

Proposed wording:

```text
IDIR's weaker OASIS behavior and stronger LungCT behavior are consistent with
a deformation-regime explanation: coordinate-network priors can suit smooth
coherent intrasubject respiratory motion better than localized intersubject
brain deformation.
```

Implementation source:

- `external/IDIR/`
- `scripts/experiments/oasis_alpha_sweep/run_one.py`
- `scripts/experiments/lungct_no_sweep_repro/run_one.py`

Experiment source:

- `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/`
- `experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/`
- `experiments/2026-06-25_motion_regime_seed3_first10_current_methods/`

Metric family:

- OASIS overlap and surface metrics
- LungCT TRE
- displacement magnitude and coherence diagnostics
- regularity metrics across both datasets

Caveats:

- LungCT comparisons must preserve protocol tags. Native IDIR, pTV-prep SINR,
  pTV-prep DirectCP, and pTVReg references do not always optimize the same
  image domain.
- IDIR training BE uses the original-style scale with the extra batch-size
  division.

## Claim: Pareto Regularity Metrics Are Required Beside Accuracy

Proposed wording:

```text
Single Dice or TRE operating points hide alignment-regularity trade-offs, so
the comparison should be framed with Pareto curves across common regularity
metrics.
```

Implementation source:

- `scripts/experiments/oasis_alpha_sweep/common.py`
- `scripts/experiments/oasis_alpha_sweep/aggregate_first10_population.py`
- `scripts/experiments/lungct_no_sweep_repro/common.py`
- `scripts/experiments/lungct_no_sweep_repro/aggregate_reg_sweep_idir_directcp32.py`

Experiment source:

- `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/`
- `experiments/2026-06-25_oasis_seed3_first10_saved_baselines/`
- `experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/`
- `experiments/2026-06-25_lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep/`

Metric family:

- Accuracy: Dice, HD95, ASD, TRE
- Smoothness: diffusion energy, bending energy
- Topology/volume diagnostics: log-Jacobian spread, `J <= 0`, NDV

Caveats:

- Saved dense-field baselines do not expose native model/control
  parameterizations, so only dense finite-difference and Jacobian-style
  regularity metrics are available for those rows.
- Folding and NDV can be sensitive to postprocessing and should not carry the
  regularity argument alone.

## Claim: LungCT Multistage Gains Come From Coarse-To-Fine Capture

Proposed wording:

```text
For LungCT, matching the final local control spacing is not enough. The
32-16-8-4 DirectCP schedule changes the optimization path and improves capture
of large coherent respiratory motion.
```

Implementation source:

- `scripts/experiments/lungct_no_sweep_repro/common.py`
- `scripts/experiments/lungct_no_sweep_repro/run_one.py`
- `docs/implementation/directcp_bspline_optimization.md`

Experiment source:

- `experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/`
- `experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/`
- `experiments/2026-06-26_lungct_case01_ptvreg_fit_modes_lr_sweep/`
- `experiments/2026-06-26_lungct_case08_ptvreg_fit_modes_lr_sweep/`

Metric family:

- Landmark TRE, especially original/native-crop snap TRE for pTVReg-style
  comparisons.
- Foreground log-Jacobian spread, folding percentage, BE, diffusion, and
  displacement magnitude.
- Stage histories and prolongation diagnostics.

Caveats:

- The multistage method changes the optimization path, not just the final
  representation.
- The pTV-prep domain, loss, mask rule, and TRE convention must be stated beside
  every LungCT row.
