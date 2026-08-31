# Technical log

## Known protocol limitations

- This is a historical-P2e raw-window adaptation: historical P2e applied
  `[80,900]` to raw stored values, while the pair-local NIfTI is physical HU
  with documented `HU = stored - 1000`. The port applies the raw window to
  `stored = HU + 1000`; it does not apply it directly to HU.
- The full 1.5-mm shared grid differs from the historical 1-mm pTV landmark
  crop. TRE values from the two settings are not method-only comparisons.
- Internal P2e warping uses `align_corners=True`; the unchanged Learn2Breath
  validation evaluator uses its challenge convention. The preflight records a
  synthetic boundary-sampling difference.
- R231 masks are only a qualitative QC dependency. Their geometry must match
  the CT grids before display and they never enter the objective.
- The initial request was COPD8 only. After its review, the authorized scope is
  a 40-run two-window audit; no profile will be selected from its validation
  Dice or COPD TRE.

## Superseded COPD8 audit result

The first `case_111` run completed its 2,500 updates and exported/reloaded the
fixed-grid field. Its physical trilinear landmark TRE was 27.1577 mm compared
with 26.4270 mm at identity. It is superseded because the historical raw-value
window was incorrectly applied directly to HU, mapping 90.6% of fixed voxels
to zero. No additional COPD or validation case has been launched.

## Corrected COPD8 audit result

The rerun converted input HU to the documented historical stored-value domain
before applying the P2e window. It completed all 2,500 updates and the field
export/reload checks. Physical trilinear TRE improved from **26.4270 mm** at
identity to **1.7129 mm** across 300 landmarks (median 0.9413 mm; p95 8.8858
mm). Whole-grid folding was 0.22037%, and GPU optimisation took 41.73 s
(1.87 GiB peak allocated memory). The executed coronal QC notebook uses this
corrected field and the accepted mask-audit orientation. Its prior
single-case review gate was satisfied before the authorized all-case rollout.

## Authorized all-case two-window rollout

The cohort audit compares physical-HU `p2e_equiv_hu` `[-920,-100]` against
Learn2Breath-lung `[-1000,200]` for each COPD and validation case. Four
persistent GPU workers each own one cohort/profile condition, so cold denotes
the first case in that worker and warm denotes its subsequent nine cases. The
headline folding definition is the unchanged Learn2Breath evaluator for both
cohorts. This is an exploratory preprocessing comparison, not a submission or
a profile-selection experiment.

Preflight completed for all 40 jobs with no failures. The four cold workers
then completed their finite field and metric gates before the controller
released the respective warm queues.

## Completed execution record

All four workers completed ten sequential cases and returned zero. The
four-worker makespan was 3,395.8 s. Worker totals were 3,256.7 s
(COPD/P2e-equivalent), 2,867.3 s (COPD/Learn2Breath-lung), 3,392.9 s
(validation/P2e-equivalent), and 3,386.2 s (validation/Learn2Breath-lung).
The per-case records distinguish synchronized optimizer work from the full
load/preflight/optimise/export/reload/evaluate/regularity pipeline, and retain
the cold first case plus nine warm cases for each worker.

The reported-only results are: COPD mean TRE 1.4683 mm and 1.7598 mm for
`p2e_equiv_hu` and `l2b_lung_hu`, respectively; validation mean lobe Dice
0.9724 and 0.9700, respectively. Challenge whole-grid folding was 0.2444%,
0.7905%, 0.2767%, and 0.5832% in GPU order 0--3. These results do not select a
preprocessing winner or form a challenge submission.

Both all-case notebooks executed in Docker and contain 20 inline panels with
zero cell errors. A first COPD notebook invocation used the parent R231 audit
directory; rerunning with its validated `masks/r231` child completed the
intended QC without changing any registration artifact or metric.
