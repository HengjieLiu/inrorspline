# Camera-Ready Diagnostic Metric Domain Summary

This note documents what the v002 camera-ready diagnostic figures measure, where
the metrics are evaluated, and where the training objectives do or do not match
the plotted evaluation domain.

The source figure family is
`experiments/paper_artifacts/fit_modes_population_diagnostics/v002_lungmask_panelA/`.
Relative to v001, OASIS is unchanged, DIR-LAB Panel A is replaced with fixed
T00 lung-mask `1−NCC`, and DIR-LAB Panel B is unchanged because it already used
the fixed-mask field-fitting convention.

## Core metric definitions

### Panel A: registration `1−NCC`

Both OASIS and LungCT/DIR-LAB use the same normalized cross-correlation
definition in code:

```text
ncc_loss = -corr(fixed, warped)
NCC      = -ncc_loss
1−NCC    = 1 - NCC = 1 + ncc_loss
```

The code writes this as `global_ncc_value = -global_ncc_loss`, so the plotted
full-image/full-crop diagnostic is `1 - global_ncc_value`. In v002, DIR-LAB
Panel A instead plots `lung_mask_one_minus_ncc`, computed with the same NCC
formula but restricted to fixed T00 lung-mask voxels.

This distinction matters:

- `global_ncc_value` / `global_ncc_loss` is the full-image or full-crop
  diagnostic.
- `lung_mask_one_minus_ncc` is the v002 DIR-LAB camera-ready Panel A metric.
- For DIR-LAB v002, all four methods are evaluated with the same fixed T00
  lung-mask `1−NCC`, even though their training domains are not identical.

### Panel B: supervised field-fitting MSE and rRMSE

Panel B optimizes MSE to a target displacement field. The figure y-axis is not
raw MSE; it is normalized root relative MSE:

```text
rRMSE (%) = 100 × sqrt(masked MSE / (masked target power + 1e-12))
```

The underlying MSE is `field_mse_objective_normalized_grid`. The normalization
uses `field_target_power_masked_normalized_grid`. Both are reduced over the
dataset-specific fixed mask and over displacement components.

## What is plotted?

| Dataset | Panel | Plotted metric | Evaluation domain | Source convention |
|---|---|---|---|---|
| OASIS | A: unsupervised registration | Full-image global `1−NCC` | Full skull-stripped/foreground-heavy OASIS image grid; not explicitly mask-restricted | `global_ncc_value` from `global_ncc_diagnostic.csv` |
| OASIS | B: supervised field fitting | rRMSE (%) from masked MSE | Fixed raw `image > 0` mask | SITReg target displacement in normalized-grid units |
| DIR-LAB/LungCT | A: unsupervised registration | Fixed-mask global `1−NCC` | Fixed T00 lung mask on the canonical 1 mm pTVReg crop | v002 `lung_mask_one_minus_ncc` from paired diagnostic curves |
| DIR-LAB/LungCT | B: supervised field fitting | rRMSE (%) from masked MSE | Fixed T00 lung mask on the canonical 1 mm pTVReg crop | pTVReg target displacement reconstructed on the 1 mm crop in normalized-grid units |

## Do the four methods match within each dataset?

| Dataset | Panel | Are the plotted metrics matched across methods? | Are the training objectives/domains matched across methods? | Camera-ready interpretation |
|---|---|---|---|---|
| OASIS | A | Yes for the plotted diagnostic: all four curves are full-image global `1−NCC`. | Mostly, but not perfectly. IDIR trains sampled NCC on fixed `image>0` coordinates; spline/control-point methods use full-image NCC. | Use as a common full-image registration diagnostic, not as proof that every optimizer saw the exact same sampled training domain. |
| OASIS | B | Yes. All four curves report fixed-`image>0` rRMSE from normalized-grid MSE. | Yes for the supervised objective: MSE-only field fitting on fixed `image>0`; regularization is not part of the objective. | Strong within-dataset representation-capacity comparison. |
| DIR-LAB/LungCT | A | Yes in v002. All four curves report fixed T00 lung-mask `1−NCC`. | No. IDIR trains sampled NCC in the fixed T00 lung mask; INR-BSCP, D-BSCP, and MR-D-BSCP train full-crop global NCC with dense forward-difference bending energy. | The v002 figure fixes the evaluation-domain fairness issue, but the training-domain caveat must remain explicit. |
| DIR-LAB/LungCT | B | Yes. All four curves report fixed T00 lung-mask rRMSE from normalized-grid MSE. | Yes for the supervised objective: MSE-only field fitting on the fixed T00 lung mask; regularization is not part of the objective. | Strong within-dataset representation-capacity comparison on the pTVReg target field. |

## Cross-dataset consistency

The two datasets use the same high-level figure semantics:

- Panel A is an image-registration similarity diagnostic where lower `1−NCC`
  is better.
- Panel B is supervised target-field fitting where lower rRMSE is better.
- Panel B uses normalized-grid displacement units in both datasets.
- Panel B rRMSE is computed from masked MSE divided by masked target power in
  both datasets.

The datasets do not use identical evaluation masks:

- OASIS uses the fixed brain foreground convention, `image > 0`, for Panel B
  and a full-image global NCC diagnostic for Panel A. This is tolerable because
  the OASIS images are skull-stripped/foreground-heavy, but Panel A is still not
  explicitly mask-restricted.
- DIR-LAB uses the fixed T00 lung mask for the camera-ready v002 Panel A metric
  and for Panel B field fitting. This is the anatomically appropriate domain for
  LungCT because full-crop air/background can dominate full-crop NCC.

Therefore, the safe cross-dataset wording is:

> Each dataset is evaluated on its anatomically appropriate fixed-domain
> convention: OASIS brain foreground/full-image registration diagnostics and
> DIR-LAB fixed-T00-lung-mask diagnostics. The comparisons are internally
> method-consistent for the plotted metrics, but OASIS and DIR-LAB should not be
> described as using identical masks.

## Camera-ready wording guidance

Safe claims:

- Panel B is method-consistent within each dataset: all four methods fit the
  same target field using MSE-only objectives and report the same masked rRMSE
  definition.
- v002 DIR-LAB Panel A evaluation is method-consistent: all four methods are
  plotted using fixed T00 lung-mask `1−NCC`.
- The v002 DIR-LAB Panel A fix addresses the previous full-crop NCC diagnostic
  concern for LungCT visualization and endpoint reporting.
- OASIS and DIR-LAB both use global NCC-style image similarity diagnostics, and
  both use normalized-grid displacement MSE/rRMSE for Panel B.

Avoid or qualify:

- Do not claim DIR-LAB Panel A training objective/domain is identical across all
  four methods. It is not: IDIR trains in the lung mask, while the three
  BSCP-style methods train full-crop global NCC in the accepted v002 runs.
- Do not claim OASIS and DIR-LAB use identical evaluation masks. Instead, say
  each uses the anatomically appropriate fixed-domain convention.
- Do not describe Panel B as plotting raw MSE. The plotted value is rRMSE (%),
  computed from masked MSE and masked target power.
- Do not imply that the DIR-LAB v002 lung-mask diagnostic changed the
  optimization objective. It is a diagnostic/evaluation-domain correction; the
  accepted training objectives remain unchanged.

Suggested caption wording:

> Panel A reports `1−NCC` during unsupervised registration. For OASIS, `1−NCC`
> is measured on the full skull-stripped brain image grid. For DIR-LAB, the
> v002 camera-ready diagnostic reports the same NCC formula restricted to the
> fixed T00 lung mask on the canonical 1 mm crop. Panel B reports supervised
> field-fitting rRMSE, computed from masked normalized-grid MSE divided by the
> corresponding masked target-field power; OASIS uses the fixed `image>0` mask
> and DIR-LAB uses the fixed T00 lung mask.

Suggested reviewer-response wording:

> We revised the DIR-LAB Panel A diagnostic to use the fixed T00 lung mask for
> all methods, matching the anatomical domain used by the supervised field-fit
> analysis. This change affects the reported diagnostic curves and endpoints,
> not the optimization objectives. Panel B was already evaluated on the fixed
> mask, so no Panel B rerun was required.

## Verification references

Artifact-level references:

- `camera_ready_source_index_lungmask_panelA.csv`: v002 source-of-truth index.
- `curve_summary.csv`: plotted source metric, point count, extrema, and endpoint
  checks for all 160 trajectories.
- `validation_report.md`: confirms 40/40 DIR-LAB Panel A dual-diagnostic records,
  160/160 source rows, and 251-point DIR-LAB Panel A curves ending at epoch
  2500.
- `diagnostic_comparison.md`: before/after v001-to-v002 endpoint comparison and
  generated figure links.

Code-level references:

- `scripts/experiments/oasis_alpha_sweep/common.py`: OASIS `ncc_loss`.
- `scripts/experiments/oasis_alpha_sweep/run_one.py`: OASIS Panel A full-image
  NCC diagnostics and training-domain metadata.
- `scripts/experiments/oasis_alpha_sweep/run_rank009_greedy_field_fit_one.py`:
  OASIS Panel B fixed-`image>0` field-fit MSE/rRMSE.
- `scripts/experiments/lungct_no_sweep_repro/common.py`: LungCT `ncc_loss`.
- `scripts/experiments/lungct_no_sweep_repro/run_one.py`: LungCT Panel A
  full-crop and fixed-lung-mask paired NCC diagnostics.
- `scripts/experiments/lungct_no_sweep_repro/run_case01_ptvreg_field_fit_one.py`:
  LungCT Panel B pTVReg target-field MSE/rRMSE on the fixed T00 lung mask.
- `scripts/experiments/fit_modes_population/plot_case_diagnostics.py`: figure
  conversion from NCC/MSE histories to plotted `1−NCC` and rRMSE.
- `scripts/experiments/fit_modes_population/render_lungmask_panelA_population_diagnostics.py`:
  v002 replacement logic that changes only DIR-LAB Panel A to
  `lung_mask_one_minus_ncc`.
