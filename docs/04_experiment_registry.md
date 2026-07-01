---
doc_type: experiment_registry
status: active
---

# Experiment Registry

Every real run should get an entry before launch. Large artifacts belong under `SCRATCH_ROOT`, not Git.

## Required Fields

```text
Run ID:
Status:
Owner/agent:
Date planned:
Date started:
Date completed:
Server:
GPU model/count:
Git commit:
Branch:
Docker image tag/digest:
Dataset root:
Dataset manifest:
Split config:
Standards preflight:
Protocol tag:
Command:
Output root:
Primary metrics:
Known limitations:
Next decision:
```

## Planned Priority Runs

### P0: Scaffold Only

- Status: complete when root scaffold lands.
- Protocol tag: `repo_scaffold_no_experiments`.
- Notes: no data, Docker, rsync, or package/test scaffold.

### P1: OASIS Seed-3 First10 Main Pareto

- Status: completed; full launch started and completed 2026-06-25 on shenggpu9
  after dry-run and smoke checks passed.
- Methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4.
- Purpose: main brain Pareto result.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`.
- Grid: IDIR alpha `0,0.01,0.1,1,10,100`; spline alpha
  `0,10,100,1000,10000,100000`; 2500 epochs.
- LR: IDIR `1e-4`; SINR cps4/cps2 `1e-4`; DirectCP cps4 `1e-3`.
- Regularizer: BE-trained sweep. IDIR uses coordinate-autograd analytic BE;
  SINR/DirectCP use cubic B-spline analytic BE.
- Metrics: Dice, HD95, ASD, native analytic BE/diffusion, forward-difference
  BE/diffusion, log-Jacobian, folding percentage, NDV; retain per-label and
  per-case rows, and report population means/spread across cases.
- Output root: `/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1`.
- Review root: `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/`.
- Launch environment: Docker container
  `inrorspline_oasis_seed3_first10_run_20260625`, image
  `hengjieliu96/dl_dir:py310sitreg_v1_251008`; dry-run and 24-job smoke passed.

### P1c: OASIS Seed-3 First10 DirectCP cps2 Preliminary Overlay

- Status: completed; add-on launch ran 2026-06-25 11:34-11:57 Pacific after
  P1 completion and DirectCP cps2 smoke passed. Overlay aggregation completed
  2026-06-25 11:58 Pacific.
- Purpose: preliminary check of DirectCP cps2 against the completed first10
  Pareto curves without overwriting P1 reports.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`.
- Method: DirectCP cps2 with cubic B-spline analytic BE/diffusion.
- Grid: alpha `100,1000,10000`; 2500 epochs; seed `1`; eval interval `250`.
- LR: `1e-3`, matching the DirectCP cps4 setting except for `cps=2`.
- Output root: `/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1`.
- Overlay review root:
  `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_prelim_overlay/`.
- Launch environment: Docker container
  `inrorspline_oasis_seed3_first10_run_20260625`, image
  `hengjieliu96/dl_dir:py310sitreg_v1_251008`; requested GPUs `1,2,3`.
- Known limitation: preliminary 3-alpha subset only; do not interpret as a full
  six-alpha DirectCP cps2 Pareto sweep.

### P1d: OASIS Seed-3 First10 Saved Baseline Evaluation

- Status: completed 2026-06-25 16:40 Pacific in Docker after dry-run, rank-0
  smoke, 50/50 saved-field evaluations, and overlay aggregation passed.
- Purpose: add paper-facing OASIS reference dots for saved DL and Greedy
  baselines using the same first10 accuracy and dense-field regularity pipeline
  as P1.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`.
- Methods: `bs1_vxm_v0_2`, `bs2_tm_v0`, `bs3_vfa_v0`, `bs4_sitreg_v0`, and
  canonical `Greedy`.
- Inputs: saved dense voxel-space displacement fields from the local DL
  baseline root and canonical Greedy result root documented in
  `private/data_sources.md`.
- Metrics: Dice, HD95, ASD, dense finite-difference BE/diffusion, log-Jacobian,
  folding percentage, and NDV; retain per-label, per-case, and population rows.
- Output root: `/database/inrorspline/runs/oasis_seed3_first10_saved_baselines_v1`.
- Review root:
  `experiments/2026-06-25_oasis_seed3_first10_saved_baselines/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`, GPU `0`.
- Acceptance: `50/50` summaries and `50/50` `DONE` files present; each run has
  case, regularity, BE, diffusion, history, mask-audit, and summary artifacts;
  aggregation reported `errors=[]`.
- Actual population means: BS1 VoxelMorph Dice/HD95/ASD
  `0.7683/2.4477/0.5497`; BS2 TransMorph `0.7913/2.0084/0.4467`;
  BS3 VFA `0.8198/1.7743/0.3669`; BS4 SITReg `0.8215/1.7159/0.3509`;
  Greedy `0.8037/1.8726/0.4096`.
- Greedy displacement format note: canonical Greedy NIfTI fields are stored as
  5D channel-last arrays and are reoriented by applying the OASIS scalar
  LIA-to-RAS transform to each component plus component signs `-,-,+`; rank-0
  smoke reproduced the saved Greedy warped-label Dice closely.
- Known limitation: saved dense fields do not expose native model/control
  parameterizations, so native analytic IDIR/B-spline regularity columns are
  intentionally unavailable for these baseline rows.

### P1e: OASIS Seed-3 First20 Saved Baseline Extension

- Status: completed 2026-06-25 Pacific after rank `10-19` dry-run confirmed
  all 50 source displacement files, 50 new saved-field evaluations completed,
  and first20 aggregation passed with `100/100` saved baseline rows.
- Purpose: extend the saved DL/Greedy baseline evaluation from seed-3 first10
  to seed-3 first20 without retraining.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-19`; ranks `0-9` reuse P1d
  completed outputs, ranks `10-19` are newly evaluated. Additional original
  sorted indices for ranks `10-19`: `130,100,118,67,79,56,142,199,36,12`.
- Methods: `bs1_vxm_v0_2`, `bs2_tm_v0`, `bs3_vfa_v0`, `bs4_sitreg_v0`, and
  canonical `Greedy`.
- Metrics: same saved dense-field posthoc pipeline as P1d: Dice, HD95, ASD,
  dense finite-difference BE/diffusion, log-Jacobian, folding percentage, and
  NDV using fixed raw image-positive foreground for regularity metrics.
- Output root: `/database/inrorspline/runs/oasis_seed3_first10_saved_baselines_v1`
  extended with rank `10-19` run directories.
- First20 review root:
  `experiments/2026-06-25_oasis_seed3_first20_saved_baselines/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`, GPU `0`.
- Actual first20 population means: BS1 VoxelMorph Dice/HD95/ASD
  `0.7724/2.5298/0.5493`; BS2 TransMorph `0.7954/2.0932/0.4483`;
  BS3 VFA `0.8223/1.8050/0.3638`; BS4 SITReg `0.8245/1.7505/0.3491`;
  Greedy `0.8067/1.9405/0.4101`.
- Acceptance: `100/100` summaries and `100/100` `DONE` files present across
  ranks `0-19`; each run has case, regularity, BE, diffusion, history,
  mask-audit, and summary artifacts; aggregation reported `errors=[]`.
- Known limitation: output-root slug retains the historical `first10` name
  because it is the existing P1d artifact root being extended; review artifacts
  use the explicit `first20` slug.

### P1f: OASIS Rank-9 Registration vs Greedy-Field Fitting LR Sweep

- Status: completed 2026-06-26 after dry-run, smoke, full launch, and
  aggregation passed with `18/18` summaries and `errors=[]`.
- Date planned: 2026-06-26.
- Purpose: illustrate how IDIR, SINR cps4, and DirectCP cps4 behave under
  image-registration optimization versus direct fitting of a strong Greedy
  dense displacement field.
- Dataset: OASIS BrainMR seed-3 shuffled rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`.
- Methods: IDIR original dense SIREN, SINR cps4 with cubic B-spline analytic
  BE, and DirectCP cps4 with cubic B-spline analytic BE.
- Exp1 objective: image registration with the method-native BE regularizer.
  Alphas are fixed to the completed sweep's best operating points:
  IDIR `1`, SINR cps4 `1000`, DirectCP cps4 `1000`.
- Exp2 objective: direct MSE fitting of the canonical Greedy dense displacement
  in normalized-grid coordinates over the fixed-image-positive brain mask, with
  no image loss and no regularization. The brain mask is the raw fixed OASIS
  image after standard reorientation with `fixed_raw > 0`; segmentation masks
  are not training masks.
- LR sweep: `1e-2,1e-3,1e-4` for every method/objective; 2500 epochs; seed `1`;
  eval interval `250`.
- Greedy displacement convention: canonical Greedy NIfTI fields are loaded with
  the same saved-baseline convention: 5D channel-last layout, scalar OASIS
  LIA-to-RAS reorientation per component, and component signs `-,-,+`.
- Output root: `/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1`.
- Review root:
  `experiments/2026-06-26_oasis_rank009_fit_modes_lr_sweep/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2`.
- Primary metrics: per-epoch raw and plot-normalized loss curves, final
  brain-mask Greedy-field MSE, full-volume Greedy-field MSE as context,
  Dice/HD95/ASD, and posthoc deformation regularity metrics.
- Known limitations: single illustrative OASIS pair only; IDIR field-fit samples
  `10,000` fixed-image-positive coordinates per epoch, while spline variants
  expand their cps4 dense field and compute MSE on all fixed-image-positive
  voxels. IDIR BE scale normalization is plot-only; the IDIR image-registration
  training objective remains unchanged.
- Actual best final points: image-registration Dice favored SINR cps4 at
  lr `1e-4` with Dice `0.8296`; Greedy-field brain-mask MSE favored DirectCP
  cps4 at lr `1e-2` with MSE `2.08659e-06`.
- Next decision: compare against a SITReg-field target on the same case before
  deciding whether to add more cases.

### P1g: OASIS Rank-9 Registration vs SITReg-Field Fitting LR Sweep

- Status: completed 2026-06-26 after dry-run, smoke, full launch, and
  aggregation passed with `18/18` combined summaries and `errors=[]`.
- Date planned: 2026-06-26.
- Purpose: repeat the rank-9 field-fitting diagnostic with the saved BS4 SITReg
  dense displacement as the target while reusing the completed rank-9
  image-registration rows from P1f.
- Dataset: OASIS BrainMR seed-3 shuffled rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`.
- Methods: IDIR original dense SIREN, SINR cps4 with cubic B-spline controls,
  and DirectCP cps4.
- Reused objective: image registration with method-native BE regularization
  from `/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1`.
  Alphas remain IDIR `1`, SINR cps4 `1000`, DirectCP cps4 `1000`.
- New objective: direct MSE fitting of the saved `bs4_sitreg_v0` best-Dice
  dense displacement in normalized-grid coordinates over the
  fixed-image-positive brain mask, with no image loss and no regularization.
  The brain mask is the raw fixed OASIS image after standard reorientation with
  `fixed_raw > 0`; segmentation masks are not training masks.
- LR sweep: `1e-2,1e-3,1e-4` for every method; 2500 epochs; seed `1`; eval
  interval `250`.
- SITReg displacement convention: saved NIfTI shape `(3,160,224,192)`,
  channel-first `zyx` repo/RAS layout, no reorientation or component sign flip,
  converted from voxel units to normalized-grid displacement for PyTorch
  fitting.
- Output root:
  `/database/inrorspline/runs/oasis_rank009_sitreg_fit_modes_lr_sweep_v1`.
- Review root:
  `experiments/2026-06-26_oasis_rank009_sitreg_fit_modes_lr_sweep/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `2,3` with
  `--max-concurrent 2`.
- Primary metrics: per-epoch SITReg-field brain-mask MSE, final full-volume
  MSE as context, Dice/HD95/ASD, posthoc deformation regularity metrics, and
  reused image-registration loss curves.
- Known limitations: single illustrative OASIS pair only; IDIR field-fit
  samples `10,000` fixed-image-positive coordinates per epoch, while spline
  variants expand their cps4 dense field and compute MSE on all
  fixed-image-positive voxels.
- Acceptance: `9/9` new SITReg field-fit `DONE` files, `18/18` combined rows,
  `45,000` history rows, and `errors=[]`.
- Actual best final points: reused image-registration Dice favored SINR cps4 at
  lr `1e-4` with Dice `0.8296`; SITReg-field brain-mask MSE favored DirectCP
  cps4 at lr `1e-2` with MSE `7.77649e-06`.
- Next decision: inspect whether SITReg-field fitting changes the relative
  optimization behavior compared with Greedy-field fitting.

### P1h: OASIS Seed-3 First20 DirectCP 8-4-2 Alpha1000 Preliminary

- Status: completed 2026-06-26; final aggregation passed with `20/20` new
  summaries and no errors.
- Purpose: test whether a LungCT-style coarse-to-fine DirectCP path improves
  the OASIS cps2 operating point before expanding to the full six-alpha sweep.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-19`. The first launch covers
  ranks `0-9`; a same-setting extension launch covers ranks `10-19`.
- Method: DirectCP multistage `8,4,2` with exact cubic B-spline dyadic knot
  insertion between stages and image downsampling.
- Objective: same OASIS spline objective as SINR/DirectCP alpha sweeps:
  full-image global NCC plus `alpha *` analytic cubic B-spline BE. No
  foreground mask in the image loss; foreground regularity uses fixed raw
  image-positive mask.

### P7: OASIS Rank-9 SINR CP-Space NTK Conditioning

- Status: completed 2026-06-29 after dry-run, smoke, full first pass, and
  artifact verification.
- Owner/agent: Codex.
- Purpose: test whether SINR's coefficient INR induces a rank-deficient,
  anisotropic, or amplifying CP-space tangent kernel compared with DirectCP's
  identity kernel in the same cps4 B-spline control space.
- Dataset: OASIS BrainMR seed-3 shuffled rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`.
- Method state: SINR cps4 BE alpha `1000`, lr `1e-4`; compare local kernel
  action against the hypothetical DirectCP cps4 update from the same CP-space
  gradient.
- Protocol tag: `oasis_rank9_sinr_cps4_ntk_conditioning_local_gradient_action`.
- Planned diagnostics: `B s` versus `B J_theta J_theta^T s`, dense-action
  cosine, scalar-rescaled residual, CP/dense amplification, PSD/Rayleigh checks,
  random Rayleigh probes, matrix-size/rank/nullity summary.
- Output root:
  `/database/inrorspline/runs/sinr_ntk_conditioning_20260629_v1`.
- Review root:
  `experiments/2026-06-29_sinr_ntk_conditioning/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Known limitations: local NTK/SGD diagnostic only; initial scope is one OASIS
  pair; dense kernels are not materialized; Adam trajectory effects are not
  directly modeled.
- Actual result: `4` action rows and `128` spectrum-probe rows completed with
  `errors=[]`. The SINR CP-space kernel is rank-deficient by construction in
  the cps4 space: `388,161` CP coefficients versus `133,379` SINR parameters,
  giving nullity lower bound `254,782`. Sampled dense update cosine was
  `0.2888/0.2888` at init for data/total gradients and `0.4638/0.4649` at
  final; scalar-rescaled residual remained high (`0.9574` init and about
  `0.885` final). Dense-action amplification was large (`1.21e6` init data and
  `2.38e5` final data).
- Next decision: decide whether to extend to replayed checkpoints, first10
  OASIS cases, or LungCT case01/case08.
- Fixed parameters: alpha `1000`, learning rate `1e-2`, stage downsample
  `4,2,1`, stage epochs `400,800,1300`, seed `1`, eval interval `250`.
- Planned GPUs: first10 on `2,3` with max concurrency `2`; extension ranks
  `10-19` on `0,1` with max concurrency `2`.
- Output root:
  `/database/inrorspline/runs/oasis_seed3_first10_directcp842_alpha1000_prelim_v1`.
- Review root:
  `experiments/2026-06-26_oasis_seed3_first10_directcp842_alpha1000_prelim/`.
- Comparator rows: existing first10 IDIR best-Dice alpha `1`, SINR cps4/cps2
  alpha `1000`, and DirectCP cps4/cps2 alpha `1000`.
- Acceptance: compile/import passes; rank-0 `1,1,1` stage-epoch smoke passes;
  first10 dry-run queues exactly `10` jobs on GPUs `2,3`; extension dry-run
  queues exactly `10` jobs on GPUs `0,1`; full run produces `20/20` summaries
  with dense fields, regularity metrics, BE/diffusion CSVs, and
  stage/prolongation diagnostics; aggregation writes separate first10 and
  first20 comprehensive comparison tables.
- Known limitations: alpha `1000` only and no full six-alpha DirectCP 8-4-2
  sweep in this preliminary step.
- Result: DirectCP `8-4-2` alpha `1000` was the best mean-Dice row in both
  generated tables. First10 Dice/HD95/symmetric ASD/fold%:
  `0.8054` / `2.2429` / `0.4507` / `4.3015`. First20:
  `0.8077` / `2.2589` / `0.4531` / `4.4071`.

### P1i: OASIS Seed-3 First20 DirectCP 8-4-2 Remaining Alpha Sweep

- Status: completed 2026-06-26.
- Date planned: 2026-06-26.
- Purpose: complete the full six-alpha DirectCP `8,4,2` OASIS curve and add it
  to the saved-baseline paper-style first10/first20 Pareto figures without
  overwriting the existing five-sweep package.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-19`, using the documented
  first20 continuation of `pair_test_200`.
- Method: DirectCP multistage `8,4,2` with exact cubic B-spline dyadic knot
  insertion between stages and image downsampling.
- Objective: full-image global NCC plus `alpha *` analytic cubic B-spline BE;
  foreground regularity uses fixed raw image-positive mask.
- New alpha grid: `0,10,100,10000,100000`; alpha `1000` is reused from P1h and
  must not be relaunched.
- Fixed parameters: learning rate `1e-2`, stage downsample `4,2,1`, stage
  epochs `400,800,1300`, seed `1`, eval interval `250`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2,3` with
  `--max-concurrent 4`.
- Remaining-alpha output root:
  `/database/inrorspline/runs/oasis_seed3_first20_directcp842_alpha_sweep_remaining5_v1`.
- Reused alpha1000 root:
  `/database/inrorspline/runs/oasis_seed3_first10_directcp842_alpha1000_prelim_v1`.
- Review root:
  `experiments/2026-06-26_oasis_seed3_first20_directcp842_alpha_sweep/`.
- New figure root:
  `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/pareto_first10_first20_alpha5_saved_baselines_plus_directcp842_paper_v1/`.
- Acceptance: compile/import passes; rank-0 smoke with stage epochs `1,1,1`
  and alpha `0` passes; dry-run queues exactly `100` jobs, ranks `0-19`,
  alphas `0,10,100,10000,100000`, GPUs `0,1,2,3`, and no alpha `1000`; full
  run produces `100/100` remaining-alpha `DONE` files; aggregation validates
  `100` new rows plus `20` reused alpha1000 rows; figure generation writes
  `41` population rows per cohort and the expected PDF/PNG plot families.
- Known limitations: this completes the DirectCP `8,4,2` alpha curve only; the
  previous five-sweep figure package remains the historical baseline.
- Completion note: Docker compile and rank-0 alpha-0 `1,1,1` stage-epoch smoke
  passed; dry-run queued exactly `100` jobs with no alpha `1000`; full launch
  completed `100/100` new jobs with `0` skipped and `0` failures in `6697`
  seconds. Aggregation reused the P1h alpha `1000` root and validated `100`
  remaining rows, `20` reused alpha1000 rows, and `120` combined DirectCP
  `8,4,2` rows with `errors=[]`. The new plus-DirectCP842 figure package wrote
  `360` first10 sweep rows, `720` first20 sweep rows, `41` population rows per
  cohort, and `162` non-empty PDF/PNG files per cohort.
- Result: DirectCP `8,4,2` alpha `1000` remained the best mean-Dice point on
  both cohorts; alpha `10000` had best mean HD95. First10 Dice/HD95/symmetric
  ASD/fold% at alpha `1000`: `0.8054` / `2.2429` / `0.4507` / `4.3015`.
  First20: `0.8077` / `2.2589` / `0.4531` / `4.4071`.

### P1j: Fit-Modes Paper Figure IDIR Matched-Similarity Curves

- Status: completed on 2026-06-26; compile, smoke, full launch, and CSV
  validation passed.
- Date planned: 2026-06-26.
- Purpose: add the missing IDIR image-similarity evaluation curves needed for a
  paper-quality figure comparing registration optimization curves with direct
  target-field fitting curves.
- Dataset/cases: OASIS BrainMR seed-3 rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`; LungCT pTV-preprocessed case `01`; LungCT
  pTV-preprocessed case `08`.
- Methods: IDIR only. Existing SINR/DirectCP histories from P1g and the
  LungCT case01/case08 fit-mode sweeps are reused.
- Training objectives: unchanged IDIR objectives. OASIS uses sampled fixed
  image-positive NCC plus IDIR coordinate-autograd BE with alpha `1`. LungCT
  uses pTV-preprocessed fixed-T00 lung-mask sampled NCC plus IDIR bending with
  `alpha_bending=10`.
- New diagnostic eval curves: OASIS full-image global NCC every `10` epochs;
  LungCT pTV LCC data loss every `10` epochs using the same border-masked
  pTVReg 1mm crop convention as the LungCT SINR/DirectCP runs.
- Fixed parameters: lr `1e-4`, 2500 epochs, seed `1`, normal summary eval
  interval `250`, image-similarity eval interval `10`.
- Output root:
  `/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1`.
- Review root:
  `experiments/2026-06-26_fit_modes_paper_figure/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2`.
- Primary metrics: `image_similarity_eval.csv` curves, final Dice for OASIS,
  final TRE for LungCT, and existing field-fit MSE curves for later notebook
  composition.
- Known limitations: eval curves are diagnostic only and do not affect
  training. The OASIS registration curve is global NCC; the LungCT registration
  curve is pTV LCC data loss for dataset-protocol matching.
- Acceptance: Docker compile passed for modified OASIS and LungCT runners;
  20-epoch smoke produced eval CSVs with epochs `1,10,20`; full run completed
  all three IDIR jobs and each `image_similarity_eval.csv` has `251` rows
  covering epoch `1` plus every `10` epochs through `2500`.
- Final eval values: OASIS full-image global NCC value `0.9750214815`; LungCT
  case01 pTV LCC data loss `2817243.5`; LungCT case08 pTV LCC data loss
  `5032755.5`.
- Next decision: generate the notebook figure combining these IDIR eval curves
  with existing SINR/DirectCP registration histories and SITReg/pTVReg
  field-fit MSE histories.

### P1k: Fit-Modes Paper Figure DirectCP 4-Stage Field-Fit Finest-MSE Diagnostics

- Status: completed 2026-06-26; compile, smoke, two reruns, and notebook
  update passed.
- Date planned: 2026-06-26.
- Purpose: replace the LungCT DirectCP 4-stage target-field fitting curve in the
  paper figure with a true finest-domain diagnostic curve rather than a
  stage-domain coarse-to-fine training curve.
- Dataset/cases: LungCT pTV-preprocessed case `01` and case `08`.
- Method/objective: DirectCP 4-stage cps4 pTVReg field fitting only. Training
  remains MSE-only with the existing schedule `32,16,8,4`, downsample
  `8,4,2,1`, stage epochs `400,400,800,900`, lr `1e-2`.
- New diagnostic curve: `finest_data_loss` equals masked normalized-grid MSE on
  the original 1mm pTVReg crop and fixed T00 lung mask, evaluated posthoc every
  50 epochs during coarse stages and every epoch during the final 1mm stage;
  diagnostic only and does not affect training.
- Output root:
  `/database/inrorspline/runs/fit_modes_paper_curves_directcp4stage_field_mse_diag_v1`.
- Review root:
  `experiments/2026-06-26_fit_modes_paper_figure/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `2,3` if available.
- Acceptance: modified field-fit runner compiles; smoke validates
  `finest_data_loss` and `finest_field_mse_objective_normalized_grid` columns;
  full case01/case08 reruns completed with `2500` history rows and `935` finite
  finest-domain diagnostic points per case; notebook uses these finest-domain
  MSE curves.
- Final diagnostic values: LungCT case01 `finest_data_loss=2.846068e-08`;
  LungCT case08 `finest_data_loss=9.682780e-08`.
- Known limitations: DirectCP 4-stage training still optimizes stage-domain MSE
  at coarse stages. The new curve is posthoc finest-domain diagnostic only.

### P1l: Fit-Modes rRMSE Learning-Rate Trajectory Comparison

- Status: completed 2026-07-01; smoke, four diagnostic reruns, notebook
  execution, and CSV validation passed.
- Owner/agent: Codex.
- Date planned: 2026-07-01.
- Git commit: `1872fe0` on branch `gpu8` with figure/report edits in the
  worktree.
- Docker image tag/digest:
  `hengjieliu96/dl_dir:py310sitreg_v1@sha256:8a159ba5e7f438cf36ea10d74a4a45fac1b0250d7d70f0dba114eefcbe3416da`.
- Purpose: create a field-fit-only rRMSE trajectory figure comparing requested
  learning rates for IDIR, SINR cps4, DirectCP cps4, and LungCT DirectCP
  4-stage.
- Dataset/cases: OASIS BrainMR seed-3 rank `9` SITReg-field and Greedy-field
  targets; LungCT pTV-preprocessed cases `01` and `08` pTVReg-field targets.
- Reused inputs: existing OASIS single-stage field-fit histories and LungCT
  single-stage field-fit histories; existing LungCT DirectCP 4-stage
  `lr=1e-2` finest-domain field-MSE diagnostic.
- New diagnostic runs: LungCT case01/case08 DirectCP 4-stage pTVReg-field
  fitting at `lr=1e-3` and `lr=1e-4`, using the same schedule
  `32,16,8,4`, downsample `8,4,2,1`, stage epochs `400,400,800,900`, seed
  `1`, and 2500 epochs. Training remains stage-domain MSE-only; posthoc
  finest-domain masked normalized-grid MSE is logged every 50 epochs during
  coarse stages and every epoch during the final 1mm stage.
- Output root:
  `/database/inrorspline/runs/fit_modes_paper_curves_directcp4stage_field_mse_diag_v1`.
- Review root:
  `experiments/2026-06-26_fit_modes_paper_figure/`.
- Primary metrics: relative target-field RMSE percent,
  `100 * sqrt(mse / (target_power + 1e-12))`, using the same masked
  normalized-grid component-MSE convention as the original field-fit losses.
- Planned review artifacts:
  `fit_modes_rrmse_lr_trajectory_curves.{png,pdf}`,
  `fit_modes_rrmse_lr_trajectory_curve_summary.csv`, and
  `fit_modes_rrmse_lr_trajectory_selected_runs.csv`.
- Acceptance: two-epoch smoke wrote finite `finest_data_loss` and
  `finest_field_mse_objective_normalized_grid` columns; all four new full
  diagnostic histories have `2500` rows and `935` finite finest-domain
  diagnostic points; final finest-domain history values are within `1%` of the
  corresponding original-sweep final summaries. The notebook executed in
  Docker and wrote the requested rRMSE LR trajectory PNG/PDF/CSVs.
- Final finest-domain history values: case01 lr `1e-3`
  `2.894892e-08`, case01 lr `1e-4` `3.456253e-08`, case08 lr `1e-3`
  `9.883180e-08`, case08 lr `1e-4` `3.594418e-05`.
- Known limitations: no OASIS DirectCP 4-stage runs are added; those panels are
  marked not run. The DirectCP 4-stage coarse-stage rRMSE curves are posthoc
  diagnostics and are not the loss used for backpropagation at coarse stages.

### P1a: OASIS Seed-3 Case-0 Alpha Pre-Run

- Status: completed; aggregation passed with 12/12 runs and no acceptance errors.
- Date planned: 2026-06-25.
- Dataset: OASIS BrainMR seed-3 shuffled rank `0`, original sorted index `158`,
  fixed/moving `(0333, 0115)`.
- Methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4.
- Grid: IDIR alpha `0.01,0.1,1`; spline alpha `100,1000,10000`; 1000 epochs.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Output root: `/database/inrorspline/runs/oasis_seed3_case0_prerun_be_1000e_v1`.
- Review root: `experiments/2026-06-25_oasis_seed3_case0_prerun/`.
- Actual launcher elapsed: 504.5 seconds; large output size: 1.1 GB; review
  artifact size: 4.6 MB.
- Notes: native analytic regularity means IDIR coordinate-autograd Hessian BE
  and Jacobian diffusion; spline methods use cubic B-spline analytic BE and
  diffusion.
- Next decision: user reviews pre-run Pareto/contact-sheet outputs before full
  overnight first10 launch.

### P1b: OASIS Original Sorted Case-0 Alpha Pre-Run

- Status: completed; aggregation passed with 12/12 runs and no acceptance errors.
- Date planned: 2026-06-25.
- Dataset: OASIS BrainMR original sorted `pair_test_200.txt` row `0`,
  fixed/moving `(0004, 0236)`.
- Methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4.
- Grid: IDIR alpha `0.01,0.1,1`; spline alpha `100,1000,10000`; 1000 epochs.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Output root: `/database/inrorspline/runs/oasis_sorted_case0_prerun_be_1000e_v1`.
- Review root: `experiments/2026-06-25_oasis_sorted_case0_prerun/`.
- Actual launcher elapsed: 504.5 seconds; large output size: 1.1 GB; review
  artifact size: 4.6 MB.
- Notes: native analytic regularity means IDIR coordinate-autograd Hessian BE
  and Jacobian diffusion; spline methods use cubic B-spline analytic BE and
  diffusion.
- Next decision: compare original sorted case-0 and seed-3 shuffled case-0
  curves before choosing first10 full-run alpha ranges.

### P2: OASIS Seed-3 First10 Deformation-Regime Stats

- Status: planned.
- Purpose: regenerate deformation-regime statistics on the same first10 protocol.

### P2a: LungCT pTV-Prep No-Sweep Reproduction

- Status: completed; 40/40 jobs completed with return code 0 and aggregation
  passed.
- Date planned: 2026-06-25.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Methods: IDIR BE default with fixed T00 lung-mask sampling, SINR cps4 pTV LCC
  plus isoTV, DirectCP cps4 pTV LCC plus isoTV, DirectCP multistage
  `32,16,8,4` pTV LCC plus isoTV.
- Fixed parameters: IDIR `alpha_bending=10`, `lr=1e-4`, `epochs=2500`;
  SINR cps4 `lr=1e-4`, `epochs=2500`; DirectCP cps4 `lr=1e-3`,
  `epochs=2500`; multistage epochs `200,400,800,1000`, `lr=1e-3`.
- pTV loss settings: `sigma_mm=2.1`, `isoTV weight=0.11`, `csqrt=5e-3`,
  `border_mask_voxels=5`.
- Mask convention: IDIR uses fixed T00 lung mask for sampled-coordinate
  training; pTV SINR/DirectCP losses do not use lung-mask weighting; all
  foreground regularity metrics use the fixed T00 lung mask on the cropped 1mm
  grid.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Output root: `/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1`.
- Review root: `experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/`.
- Primary metrics: original/native-crop identity/snap/subvoxel TRE,
  cropped-1mm identity/snap/subvoxel TRE, fixed-lung-mask foreground
  log-Jacobian std, folding percentage, bending energy, diffusion energy,
  displacement magnitude, runtime, peak GPU memory.
- Actual result, original/native-crop snap/sub TRE: IDIR `1.017/1.133`;
  SINR cps4 `2.904/3.014`; DirectCP cps4 `6.370/6.450`; DirectCP multistage
  `0.922/1.049`. Cropped-1mm diagnostic snap/sub TRE: IDIR `1.236/1.138`;
  SINR cps4 `3.143/3.037`; DirectCP cps4 `6.575/6.495`; DirectCP multistage
  `1.168/1.052`.
- Review artifacts: `summary_runs.csv`, `summary_runs.json`, and
  `review_report.md` under the review root.

### P2b: LungCT pTV-Prep IDIR + DirectCP 32-16-8-4 Regularization Sweep

- Status: completed; 80/80 nonbaseline jobs completed with return code 0 and
  aggregation passed with 100 case rows and 10 mean rows.
- Date planned: 2026-06-25.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Methods: IDIR BE default with fixed T00 lung-mask sampling and DirectCP
  multistage `32,16,8,4` pTV LCC plus isoTV.
- Sweep: reuse completed `1e0` baseline; launch nonbaseline factors
  `0.01,0.1,10,100` around each method's default regularization weight.
- IDIR regularization: base `alpha_bending=10`; swept values
  `0.1,1,100,1000`.
- DirectCP regularization: base `ptv_isotv_weight=0.11`; swept values
  `0.0011,0.011,1.1,11`.
- Fixed parameters: IDIR `lr=1e-4`, `epochs=2500`; DirectCP stage epochs
  `200,400,800,1000`, `lr=1e-3`; pTV settings `sigma_mm=2.1`,
  `csqrt=5e-3`, `border_mask_voxels=5`.
- Primary metric: `original_snap` TRE. Track all four TREs:
  `original_snap`, `original_subvoxel`, `reg1mm_snap`, and
  `reg1mm_subvoxel`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Baseline root: `/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1`.
- Sweep output root: `/database/inrorspline/runs/lungct_ptvprep_reg_sweep_idir_directcp32_v1`.
- Review root: `experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/`.
- Actual result, primary original snap TRE: IDIR factors
  `0.01/0.1/1/10/100` gave `1.317/1.046/1.017/1.085/1.257` mm;
  DirectCP factors `0.01/0.1/1/10/100` gave
  `1.320/1.022/0.922/1.583/6.809` mm.
- Best primary point: default `1e0` for both methods, IDIR
  `alpha_bending=10` with original snap TRE `1.017` mm and DirectCP
  `ptv_isotv_weight=0.11` with original snap TRE `0.922` mm.
- Compute cost, default `1e0`: IDIR `2500` epochs, `2.63 +/- 0.17` min/case,
  `4.14 +/- 0.06` GB peak memory; DirectCP 4-stage `200,400,800,1000`
  stage epochs, `2400` total epochs, `3.90 +/- 0.71` min/case,
  `5.48 +/- 1.07` GB peak memory.
- Compute cost range, default `1e0`: IDIR `2.44-2.92` min/case and
  `4.05-4.25` GB; DirectCP 4-stage `3.03-5.84` min/case and `3.64-7.45` GB.
- Training-curve analysis: 50/50 planned IDIR histories confirm `lr=1e-4`;
  50/50 planned DirectCP histories confirm stage LR `[1e-3,1e-3,1e-3,1e-3]`
  and stage epochs `[200,400,800,1000]`. Objective curves support validating
  IDIR `1500/2000` epochs against `2500`, and DirectCP shortened schedules
  `200,400,300,650` and `200,200,200,400` against the current
  `200,400,800,1000` before changing canonical runs.
- Verification: dry-run queued exactly `80` nonbaseline jobs, smoke jobs passed
  for case `01` at factors `0.01` and `100` for both methods, and aggregation
  generated complete case/mean tables plus 8 method-specific Pareto plots and
  4 combined method-comparison Pareto plots.
- Refined DirectCP addendum: launched DirectCP-only factors `0.2,0.5,2,5` on
  cases `01-10` using GPUs `0,1`; 40/40 jobs completed with return code 0.
  These rows are aggregated in `directcp_refined_review.md` with non-overwriting
  figures under `plots_refined_directcp/`. Refined DirectCP original-snap TRE
  factors `0.2/0.5/1/2/5` gave `0.951/0.917/0.922/0.937/0.978` mm; best in
  this refined grid is `0.5x` (`ptv_isotv_weight=0.055`), marginally better
  than the original `1x` point.
- Refined IDIR addendum: launched IDIR BE factors `1/30,1/3,3` on cases
  `01-10` using GPUs `1,2,3`; 30/30 jobs completed with return code 0. These
  rows are aggregated in `both_refined_roi_no_extremes_review.md` with
  non-overwriting ROI figures under `plots_refined_both_roi_no_extremes/`.
  Refined IDIR original-snap TRE for factors `0.033333/0.333333/1/3/10`
  gave `1.244/1.028/1.017/1.037/1.085` mm; best remains `1x`
  (`alpha_bending=10`).

### P2c: LungCT DirectCP 32-16-8-4 Global NCC + Analytic BE Alpha Sweep

- Status: planned; implementation and launch requested by user.
- Date planned: 2026-06-25.
- Dataset: DIR-Lab 4DCT cases `01` and `08`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Method: DirectCP multistage `32,16,8,4` with exact crop-aligned cubic
  B-spline dyadic knot insertion.
- Objective: full-crop global NCC plus `alpha *` analytic cubic B-spline
  bending energy. No lung-mask or border-mask loss weighting.
- Sweep: alpha values `0,0.1,1,10,100,1000`; absolute weights, not factors.
- Fixed parameters: downsample schedule `8,4,2,1`, stage epochs
  `200,400,800,1000`, `lr=1e-3`, seed `1`.
- Mask convention: fixed T00 lung mask is used only for foreground regularity
  metrics.
- Primary metric: `original_snap` TRE. Track all four TREs:
  `original_snap`, `original_subvoxel`, `reg1mm_snap`, and
  `reg1mm_subvoxel`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Output root: `/database/inrorspline/runs/lungct_directcp32_global_ncc_be_case01_case08_alpha_sweep_v1`.
- Review root:
  `experiments/2026-06-25_lungct_directcp32_global_ncc_be_case01_case08_alpha_sweep/`.
- Planned command:
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_directcp32_global_ncc_be_sweep.py --cases 1,8 --alphas 0,0.1,1,10,100,1000 --gpus 2,3`.
- Known limitations: selected-case diagnostic only; global NCC + analytic BE is
  not pTVReg-matched because pTVReg uses local CC and isoTV.
- Next decision: inspect case-wise TRE/regularity curves before deciding whether
  to expand to all 10 cases or refine alpha locally.

### P2d: LungCT DirectCP 32-16-8-4 Global NCC + Dense FD-BE Alpha Sweep

- Status: completed; initial 12/12 selected-case jobs completed with return
  code 0, extension 6/6 larger-alpha jobs completed with return code 0, and
  aggregation passed with 18 rows.
- Date planned: 2026-06-25.
- Dataset: DIR-Lab 4DCT cases `01` and `08`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Method: DirectCP multistage `32,16,8,4` with exact crop-aligned cubic
  B-spline dyadic knot insertion.
- Objective: full-crop global NCC plus `alpha *` dense forward-difference
  bending energy on the dense displacement field. No lung-mask or border-mask
  loss weighting.
- Sweep: alpha values `0,0.1,1,10,100,1000,10000,100000,1000000`;
  absolute weights, not factors.
- Fixed parameters: downsample schedule `8,4,2,1`, stage epochs
  `200,400,800,1000`, `lr=1e-3`, seed `1`.
- Mask convention: fixed T00 lung mask is used only for foreground regularity
  metrics.
- Primary metric: `original_snap` TRE. Track all four TREs:
  `original_snap`, `original_subvoxel`, `reg1mm_snap`, and
  `reg1mm_subvoxel`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Output root: `/database/inrorspline/runs/lungct_directcp32_global_ncc_fdbe_case01_case08_alpha_sweep_v1`.
- Review root:
  `experiments/2026-06-25_lungct_directcp32_global_ncc_fdbe_case01_case08_alpha_sweep/`.
- Planned command:
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_directcp32_global_ncc_fdbe_sweep.py --cases 1,8 --alphas 0,0.1,1,10,100,1000,10000,100000,1000000 --gpus 2,3`.
- Actual result, primary original snap TRE: case `01` alphas
  `0/0.1/1/10/100/1000/10000/100000/1000000` gave
  `1.254/1.240/1.208/1.093/1.004/0.880/0.755/0.734/0.815` mm;
  case `08` gave
  `7.042/6.975/7.027/6.205/3.017/1.048/1.087/1.110/1.375` mm.
- Best primary point in this extended selected-case grid: case `01` at alpha
  `100000` with original snap TRE `0.734` mm and foreground stdLogJ/J<=0%
  `0.071/0.0000%`; case `08` at alpha `1000` with original snap TRE
  `1.048` mm and foreground stdLogJ/J<=0% `0.355/0.4173%`.
- Verification: compile/import passed, loader audit passed for cases `01,08`,
  launcher dry-runs queued exactly `12` initial jobs and `6` extension jobs over
  GPUs `2,3`, smoke jobs passed for case `01` alphas `0` and `1000`, all
  launched jobs completed with return code 0, and aggregation generated complete
  case tables plus four per-case plots.
- Known limitations: selected-case diagnostic only; dense FD-BE is not the same
  regularizer as analytic cubic B-spline BE, and global NCC + FD-BE is not
  pTVReg-matched because pTVReg uses local CC and isoTV.
- Next decision: refine around case-specific optima, especially alpha
  `10000-1000000` for case `01` and `1000-10000` for case `08`, before any
  all-case expansion.

### P2e: LungCT DirectCP 4-Stage Global NCC + Dense FD-BE All-Case Sweep

- Status: completed; launch, aggregation, and ellipse-style comparison plots
  completed.
- Date planned: 2026-06-25.
  Date completed: 2026-06-25.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Method: DirectCP multistage `32,16,8,4` with exact crop-aligned cubic
  B-spline dyadic knot insertion.
- Objective: full-crop global NCC plus `alpha *` dense forward-difference
  bending energy on the dense displacement field. No lung-mask or border-mask
  loss weighting.
- Sweep: alpha values `0,10,100,1000,10000,100000`; absolute weights, not
  factors.
- Reuse policy: reuse completed case `01` and `08` rows from P2d for the six
  requested alphas; launch missing cases `02-07,09,10` only.
- New jobs: `48` = 8 missing cases times 6 alpha values.
- Fixed parameters: downsample schedule `8,4,2,1`, stage epochs
  `200,400,800,1000`, `lr=1e-3`, seed `1`.
- Mask convention: fixed T00 lung mask is used only for foreground regularity
  metrics.
- Primary metric: `original_snap` TRE. Track all four TREs:
  `original_snap`, `original_subvoxel`, `reg1mm_snap`, and
  `reg1mm_subvoxel`.
- Comparison: generate new non-overwriting all-case ellipse-style Pareto figures
  against IDIR BE and DirectCP 4-stage pTV LCC + isoTV.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Existing selected-case root: `/database/inrorspline/runs/lungct_directcp32_global_ncc_fdbe_case01_case08_alpha_sweep_v1`.
- New missing-case output root: `/database/inrorspline/runs/lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep_v1`.
- Review root:
  `experiments/2026-06-25_lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep/`.
- Launch command:
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_global_ncc_fdbe_all10_sweep.py --gpus 0,1,2,3`.
- Verification: compile/import passed; loader audit passed for cases `01-10`;
  launcher dry-run queued exactly `48` missing-case jobs; smoke job on case
  `02` alpha `100000` with stage epochs `1,1,1,1` passed; full launch
  completed `48/48` jobs with return code 0; aggregation produced `60/60`
  FD-BE rows, `160` comparison rows, and all-case ellipse-style plots.
- Best FD-BE mean point: alpha `10000`, original-snap TRE
  `0.966 +/- 0.151` mm, foreground stdLogJ `0.185452`, foreground
  `J <= 0` `0.0037%`.
- Comparator best points in the same all-case ellipse report: IDIR BE alpha
  `10` gives original-snap TRE `1.017 +/- 0.176` mm, foreground stdLogJ
  `0.142869`, foreground `J <= 0` `0.0000%`; DirectCP 4-stage pTV LCC +
  isoTV weight `0.11` gives original-snap TRE `0.922 +/- 0.152` mm,
  foreground stdLogJ `0.179961`, foreground `J <= 0` `0.0138%`.
- Known limitations: all-case run still uses dense FD-BE rather than analytic
  cubic B-spline BE and is not pTVReg-matched because pTVReg uses local CC and
  isoTV.
- Next decision: decide whether the FD-BE objective is worth refining around
  alpha `1000-100000`, given that its best all-case accuracy is close to but
  still behind the pTV LCC + isoTV DirectCP best point.

### P2f: LungCT Case08 DirectCP 4-Stage FD-BE vs Analytic B-Spline BE

- Status: completed; two-method diagnostic completed for case `08`.
- Date planned: 2026-06-26.
  Date completed: 2026-06-26.
- Dataset: DIR-Lab 4DCT case `08`, canonical regenerated pTVReg-style landmark
  crop resampled to 1mm.
- Methods: DirectCP multistage `32,16,8,4` with exact crop-aligned cubic
  B-spline dyadic knot insertion, comparing full-crop global NCC plus
  `alpha *` dense forward-difference BE against full-crop global NCC plus
  `alpha *` analytic cubic B-spline BE.
- Fixed parameters: alpha `10000`, downsample schedule `8,4,2,1`, stage epochs
  `200,400,800,1000`, `lr=1e-3`, seed `1`.
- Mask convention: fixed T00 lung mask is used only for foreground regularity
  metrics.
- Primary metric: `original_snap` TRE. Track all four TREs and foreground
  regularity metrics.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Output root:
  `/database/inrorspline/runs/lungct_case08_be_impl_compare_alpha1e4_v1`.
- Review root:
  `experiments/2026-06-26_lungct_case08_be_impl_compare_alpha1e4/`.
- Launch: ran the FD-BE job on GPU `2` and analytic B-spline BE job on
  GPU `3` concurrently.
- Verification: compile/import passed; loader audit passed for case `08`;
  minimal smoke passed for both BE variants; full launch completed both jobs
  with status `completed`; comparison rows and review report were generated.
- Result: original-snap TRE tied after snap-to-voxel rounding at `1.087383` mm.
  Analytic B-spline BE was slightly better on original-subvoxel TRE
  (`1.145162` vs `1.146554` mm) and foreground stdLogJ (`0.191389` vs
  `0.192354`), but took longer (`367.985` vs `351.897` seconds). Peak memory
  was identical at `6.492` GB.
- Known limitations: selected-case diagnostic only; alpha `10000` is the same
  numeric weight but not guaranteed to imply matched effective regularization
  scale between dense FD-BE and analytic B-spline BE.

### P3: LungCT 4DCT Matched Pareto

- Status: planned.
- Methods: IDIR, SINR, single-stage DirectCP cps4, multistage DirectCP, pTVReg references.
- Required: `docs/standards/lungct_protocol_tags.md` and explicit protocol
  tags for preprocessing/domain/mask/loss.

### P2g: LungCT Case01 Registration vs pTVReg-Field Fitting LR Sweep

- Status: completed; full 24-job run and aggregation passed 2026-06-26.
- Date planned: 2026-06-26.
- Dataset: DIR-Lab 4DCT case `01`, canonical regenerated pTVReg-style landmark
  crop resampled to 1mm.
- Methods: IDIR, SINR cps4, DirectCP cps4, and DirectCP 4-stage
  `32,16,8,4`.
- Objectives: image registration and direct MSE fitting to the saved pTVReg
  case01 displacement field.
- Learning rates: `1e-2,1e-3,1e-4`; total jobs `24`.
- Scheduling: method batches run serially. Within each method batch, the six
  objective/LR jobs are distributed across GPUs `0,1,2,3`.
- Field-fit MSE objective domain: fixed T00 lung mask in the 1mm crop; full
  crop MSE is diagnostic only.
- Registration weights: IDIR `alpha_bending=10`; pTV LCC+isoTV methods
  `ptv_isotv_weight=0.11`.
- DirectCP 4-stage schedule: `32,16,8,4`, downsample `8,4,2,1`, stage epochs
  `400,400,800,900`.
- pTVReg source field:
  `/database/inrorspline/ptvreg/4DCT/case01_T50_to_T00_deformation.mat`.
  The saved HDF5 field is `Tptv_rsz` with h5py shape `(3,z,y,x)`, equivalent
  to MATLAB `T(x,y,z,component)`, physical-mm `x,y,z` components, fixed-to-moving.
- Required target validation: reproduce saved pTVReg `TRE_phys=0.7671410824`
  and `TREstd_case=0.8917488134` using the pTVReg snap-to-voxel convention.
- Output root:
  `/database/inrorspline/runs/lungct_case01_ptvreg_fit_modes_lr_sweep_v1`.
- Review root:
  `experiments/2026-06-26_lungct_case01_ptvreg_fit_modes_lr_sweep/`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Known limitations: the field-fit target is reconstructed on the repo's
  reg1mm grid from pTVReg's saved native/full displacement output; the original
  pTVReg internal reg1mm displacement is not stored in the `.mat`.
- Restart note: an initial method-pinned launch was stopped and replaced with
  method-batch scheduling because method runtimes are imbalanced.
- Completion note: final review includes full LR curves, final metric-vs-LR
  plots, and best-LR masked-MSE/registration-loss plots with continuous
  DirectCP 4-stage epoch axes.

### P2h: LungCT Case08 Registration vs pTVReg-Field Fitting LR Sweep

- Status: completed; full 24-job run and aggregation passed 2026-06-26.
- Date planned: 2026-06-26.
- Dataset: DIR-Lab 4DCT case `08`, canonical regenerated pTVReg-style landmark
  crop resampled to 1mm.
- Methods, objectives, learning rates, scheduling, masks, and regularization:
  same as P2g Case01.
- DirectCP 4-stage schedule: `32,16,8,4`, downsample `8,4,2,1`, stage epochs
  `400,400,800,900`.
- pTVReg source field:
  `/database/inrorspline/ptvreg/4DCT/case08_T50_to_T00_deformation.mat`.
  The saved HDF5 field is `Tptv_rsz` with h5py shape `(3,128,512,512)`,
  physical-mm `x,y,z` components, fixed-to-moving.
- Required target validation: reproduce saved pTVReg
  `TRE_phys=1.0142198317` and `TREstd_case=1.2922227355` using the pTVReg
  snap-to-voxel convention.
- Output root:
  `/database/inrorspline/runs/lungct_case08_ptvreg_fit_modes_lr_sweep_v1`.
- Review root:
  `experiments/2026-06-26_lungct_case08_ptvreg_fit_modes_lr_sweep/`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Acceptance: smoke and full aggregation must report `24/24` completed rows;
  final visualization must match Case01, including best-LR masked-MSE and
  registration-loss plots with continuous DirectCP 4-stage epoch axes.
- Known limitations: same as Case01; the field-fit target is reconstructed on
  the repo's reg1mm grid from pTVReg's saved native/full displacement output.
- Completion note: final review includes full LR curves, final metric-vs-LR
  plots, and best-LR masked-MSE/registration-loss plots with continuous
  DirectCP 4-stage epoch axes.

### P2i: LungCT Case01/Case08 DirectCP 4-Stage Finest-Domain Loss Diagnostic

- Status: completed; smoke, full reruns, validation, and companion plots
  completed 2026-06-26.
- Date planned: 2026-06-26.
- Purpose: rerun only DirectCP 4-stage image registration at `lr=0.01` for
  cases `01` and `08`, preserving the original stage-wise training objective
  while recording a diagnostic loss on the finest 1mm crop domain at every
  epoch.
- Dataset: DIR-Lab 4DCT cases `01` and `08`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Method: `direct_cp_32_16_8_4_ptv_lcc_isotv`.
- Schedule: `32,16,8,4`, downsample `8,4,2,1`, stage epochs
  `400,400,800,900`, Adam, seed `1`, `lr=0.01`.
- Training objective: unchanged stage-domain pTV LCC + isoTV with
  `ptv_isotv_weight=0.11`, `border_mask_voxels=5`, LCC sigma `2.1 mm`.
- Diagnostic objective: pre-step, no-gradient full 1mm crop pTV LCC + isoTV
  after analytic cubic B-spline expansion of current controls to the finest
  grid; diagnostic only and does not affect optimization.
- Output root:
  `/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_v1`.
- Review root:
  `experiments/2026-06-26_lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08/`.
- Companion plots are written next to the accepted Case01/Case08 sweep plots
  without overwriting `best_lr_registration_loss_spline_methods.png`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Acceptance: smoke passes for both cases; full reruns produce `2500`
  `stage_history.csv` rows with finite `finest_loss`, `finest_data_loss`, and
  `finest_reg_loss`; companion plots render continuous DirectCP 4-stage curves
  on epochs `1-2500`.
- Known limitations: this is a loss-curve diagnostic for two cases only; the
  finest-domain trace is not the training loss used for backprop on coarse
  stages.
- Completion note: case01 elapsed `258.6` sec and case08 elapsed `311.7` sec;
  both histories have `2500` finite diagnostic rows. Companion PNGs were written
  next to the accepted Case01 and Case08 spline-method loss plots with the
  suffix `_finest_domain_directcp4stage_lr0p01.png`.

### P2j: LungCT DirectCP 4-Stage Mid-Ratio Pareto Extension

- Status: completed.
- Date planned: 2026-06-26.
- Date completed: 2026-06-26.
- Purpose: add mid-ratio regularization points to improve the all-10
  TRE-versus-foreground-regularity Pareto visualization.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Methods:
  - `direct_cp_32_16_8_4_ptv_lcc_isotv`
  - `direct_cp_32_16_8_4_global_ncc_fd_be`
- New pTV-LCC+isoTV ratios: `0.3x` and `3x` around anchor
  `ptv_isotv_weight=0.11`, giving weights `0.033` and `0.33`.
- New global-NCC+FD-BE ratios: `0.3x` and `3x` around anchor alpha `10000`,
  giving alphas `3000` and `30000`.
- IDIR addendum ratios: `0.3x` and `3x` around anchor `alpha_bending=10`,
  giving alphas `3` and `30`. Alpha `30` is reused from the existing completed
  IDIR sweep root; alpha `3` is launched as 10 new jobs.
- Fixed parameters: `lr=1e-3`, seed `1`, schedule `32,16,8,4`, downsample
  `8,4,2,1`, stage epochs `200,400,800,1000`.
- Launch ordering: run all pTV-LCC+isoTV jobs first; after `20/20` complete,
  run all global-NCC+FD-BE jobs. pTV-LCC+isoTV used GPUs `0,1,2,3`; after a
  mid-run user request, FD-BE was paused after active children finished and the
  remaining `5` jobs were relaunched on GPUs `0,1` only so GPUs `2,3` stayed
  available.
- pTV-LCC+isoTV output root:
  `/database/inrorspline/runs/lungct_directcp4stage_midratio_ptvlcc_v1`.
- global-NCC+FD-BE output root:
  `/database/inrorspline/runs/lungct_directcp4stage_midratio_fdbe_v1`.
- IDIR addendum output roots:
  `/database/inrorspline/runs/lungct_idir_midratio_alpha3_v1` for new alpha
  `3` jobs, and
  `/database/inrorspline/runs/lungct_ptvprep_reg_sweep_idir_directcp32_v1` for
  reused alpha `30` rows.
- Review root:
  `experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/`.
- Notebook:
  `pareto_visual_review_midratio_extension.ipynb`.
- Aggregation: reuse existing all-10 rows from P2e, add the 40 new rows, and
  recompute/validate pTVReg reference rows from uploaded `.mat` fields.
- Primary metric: `original_snap` TRE. Track `original_subvoxel`,
  `reg1mm_snap`, `reg1mm_subvoxel`, and foreground regularity metrics.
- Mask convention: foreground regularity uses the fixed T00 lung mask on the
  pTV 1mm crop; DirectCP training losses do not use lung-mask weighting.
- Acceptance: dry-runs queue exactly `20` jobs per method; smoke jobs pass for
  both methods; full batches complete sequentially; final executed notebook has
  `210` case rows, `21` mean rows, pTVReg validation errors below `1e-8`, and
  all 12 requested inline/saved Pareto figures.
- Completion note: pTV-LCC+isoTV completed `20/20`; global-NCC+FD-BE completed
  `20/20`; strict aggregation produced `210` case rows and `21` mean rows. The
  executed notebook and 12 PNG Pareto figures are under
  `experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/pareto_visual_review_midratio_extension/`.
- IDIR addendum acceptance: alpha `3` dry-run queues 10 jobs on GPUs
  `0,1,2,3`; smoke passes; full alpha `3` batch completes `10/10`; reused
  alpha `30` validates `10/10`; re-executed notebook has `230` case rows,
  `23` mean rows, and the same 12 inline/saved Pareto figure set updated with
  IDIR ratios `0.3x` and `3x`.
- IDIR addendum completion note: alpha `3` completed `10/10`; reused alpha
  `30` validated `10/10`; strict aggregation produced `230` case rows and `23`
  mean rows; the re-executed notebook contains IDIR ratios
  `0.01,0.1,0.3,1,3,10,100` and 12 inline/saved PNG figures.
- Known limitations: this is a regularization-grid refinement only and does not
  resolve pTVReg protocol differences such as optimizer, linear basis, and
  image pyramid.

### P2k: LungCT IDIR Sampled pTV-LCC + TV 0.11 Pilot

- Status: completed 2026-06-26; implementation, smoke, full launch, and
  aggregation passed.
- Purpose: test whether IDIR remains competitive when its sampled-coordinate
  mechanism uses a pTV-style local-CC data term and an IDIR-specific sampled TV
  analogue instead of sampled global NCC plus bending energy.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Method: `idir_sampled_ptv_lcc_tv`.
- Objective: sampled pTV-LCC on pTV 1mm border-crop samples plus
  `0.11 *` sampled analytic physical-unit TV of the SIREN displacement
  Jacobian. Fixed T00 lung mask is foreground-metric-only, not a training
  sampling mask.
- Fixed parameters: `ptv_isotv_weight=0.11`, `lr=1e-4`, `epochs=2500`,
  seed `1`, SIREN omega `32`, sampled LCC centers `512`, sampled patch offsets
  `64`, `sigma_mm=2.1`, `csqrt=5e-3`, border mask `5` voxels.
- Output root:
  `/database/inrorspline/runs/lungct_idir_sampled_ptvlcc_tv_w0p11_all10_v1`.
- Review root:
  `experiments/2026-06-26_lungct_idir_sampled_ptvlcc_tv_w0p11_all10/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2,3`.
- Primary metric: `original_snap` TRE. Track `original_subvoxel`,
  `reg1mm_snap`, `reg1mm_subvoxel`, fixed-T00-lung foreground regularity,
  elapsed time, and peak GPU memory.
- Acceptance: compile passes; loader audit for cases `01-10` passes; case01
  short smoke at weight `0.11` has finite losses and writes summary metrics;
  dry-run queues exactly `10` jobs; full launch completes `10/10` summaries;
  aggregation writes case/mean tables with no completion errors.
- Result: `10/10` completed summaries and `errors=[]`. Mean original-snap TRE
  `1.400 +/- 0.857` mm; original-subvoxel TRE `1.476 +/- 0.821` mm;
  reg1mm snap/sub TRE `1.579 +/- 0.829` / `1.483 +/- 0.836` mm. Foreground
  stdLogJ `0.171860`, foreground `J <= 0` `0.0477%`, foreground BE
  `0.001448`, foreground diffusion `0.010796`. Mean elapsed `31.7` sec/case
  and peak memory `0.58` GB. Case08 remained the outlier with original-snap TRE
  `3.895` mm.
- Known limitations: IDIR has no control lattice, so this TV term is a sampled
  analytic SIREN-Jacobian analogue of pTV isoTV, not the same control-grid
  regularizer used by pTVReg or DirectCP.

### P4: pTV-Style DirectCP Ablation Ladder

- Status: planned.
- Purpose: isolate pTVReg gap by adding local CC, border mask, isoTV, pyramid, optimizer, and basis factors one at a time.

### P5: Overnight First-Three-Case Update-Alignment Diagnostic

- Status: completed 2026-06-29; full aggregation passed and follow-up gate did
  not pass.
- Date planned: 2026-06-29.
- Purpose: adapt the compact update-alignment and SINR/direct-CP mechanism
  prompts into a self-supervised first-three-case run. The primary diagnostic
  tests whether sampled image-registration updates move toward strong teacher
  displacement fields, separating representational fit evidence from
  image-optimization trajectory evidence.
- Source prompts:
  `docs/codex_prompts/260629_2_smaller_plan.md` and
  `docs/codex_prompts/260629_1_sinr_cp_kernel_experiment_plan.md`.
- Adapted plans:
  `docs/codex_prompts/260629_3_from_260629_2_overnight_update_alignment_run_plan.md`
  and
  `docs/codex_prompts/260629_4_from_260629_1_optional_sinr_cp_kernel_followup_plan.md`.
- Dataset/cases:
  OASIS BrainMR seed-3 shuffled rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`; DIR-Lab 4DCT case `01`; DIR-Lab 4DCT case `08`.
- Teacher fields: saved BS4 SITReg dense displacement for OASIS rank `9`;
  saved pTVReg displacement fields for LungCT case `01` and case `08`,
  reconstructed on the canonical pTV 1mm crop with saved TRE validation.
- Methods: OASIS `idir_original`, `sinr_cps4_be_bspline_analytic`, and
  `direct_cp_cps4_be_bspline_analytic`; LungCT `idir_be_default_lungmask`,
  `sinr_cps4_ptv_lcc_isotv`, `direct_cp_cps4_ptv_lcc_isotv`, and
  `direct_cp_32_16_8_4_ptv_lcc_isotv`.
- Fixed operating points: OASIS IDIR alpha `1` lr `1e-4`, SINR cps4 alpha
  `1000` lr `1e-4`, DirectCP cps4 alpha `1000` lr `1e-3`; LungCT IDIR
  alpha_bending `10` lr `1e-4`, pTV-LCC+isoTV methods weight `0.11`, SINR lr
  `1e-4`, DirectCP lr `1e-3`, multistage schedule `32,16,8,4` with
  `200,400,800,1000` stage epochs.
- Checkpoint logging: opt-in deterministic sampled displacement checkpoints,
  sample seed `1729`, up to `100000` fixed-mask points per case, checkpoints
  `0,1,10,25,50,100,250,500,1000,1500,2000,2500`. Large checkpoint samples
  stay under the run root, not Git.
- Output root:
  `/database/inrorspline/runs/update_alignment_overnight_20260629_v1`.
- Review root:
  `experiments/2026-06-29_update_alignment_overnight/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2,3`.
- Preflight smoke: completed on 2026-06-29 with `11/11` jobs successful,
  `22` checkpoint-alignment rows, `11` case rows, and `3`
  SINR-vs-DirectCP similarity rows. The smoke promising gate did not pass,
  as expected for the 2-epoch sanity run; it verified launcher,
  checkpoint-sample logging, teacher loading, aggregation, plots, and the
  optional follow-up smoke path.
- Full result: completed `11/11` jobs with `0` failures. Aggregation wrote
  `119` checkpoint-alignment rows, `11` case rows, and `3`
  SINR-vs-DirectCP similarity rows. Follow-up gate did not pass:
  similarity values were LungCT case01 `0.0009`, LungCT case08 `0.2794`, and
  OASIS rank9 `0.3876`, so `promising_similarity_count=0`.
- Primary metrics: per-interval update alignment to teacher residual, early
  mean/median alignment, cumulative early alignment, final teacher distance,
  SINR-vs-DirectCP cumulative early update similarity, final OASIS Dice/HD95
  and LungCT reg1mm crop TRE diagnostics.
- Follow-up gate: if all `11/11` small-plan jobs complete, metrics are finite,
  and SINR-vs-DirectCP cumulative early update similarity is at least `0.5` in
  at least two of the three cases, auto-start
  `scripts/experiments/update_alignment/run_sinr_cp_kernel_followup.py`.
  The first follow-up implementation is an overnight-safe sampled dense-update
  proxy; it does not yet form the full autograd CP kernel `J_theta J_theta^T`.
- Known limitations: LungCT final metrics emitted by the current runner are
  cropped 1mm registration-grid TRE diagnostics, while the teacher pTVReg field
  is validated against original/native snap TRE. Do not treat these as a new
  headline pTVReg-comparable LungCT table without the full original-snap metric
  reconstruction. First-three-case evidence is diagnostic, not population-final.
- Next decision: after full small-plan aggregation, decide whether to trust the
  auto-follow-up gate, expand to more cases, or refine checkpoint/teacher-unit
  handling before running the heavier kernel diagnostic.

### P6: Exploratory SINR/DirectCP Mechanism Follow-Up

- Status: active exploratory self-supervised run; smoke completed and launch
  scope adapted.
- Date planned: 2026-06-29.
- Purpose: continue the optional follow-up as exploration despite the failed
  promising gate, with explicit Scientist/Reviewer role rounds and durable
  communication logging. The target artifact is a useful paper table, figure,
  or draft section explaining whether the observed behavior is better explained
  by SINR/DirectCP kernel similarity, representation capacity, or
  optimization-path/capture-range effects.
- Primary review root:
  `experiments/2026-06-29_sinr_cp_kernel_followup/`.
- Primary run root:
  `/database/inrorspline/runs/sinr_cp_mechanism_explore_20260629_v1`.
- Communication log:
  `experiments/2026-06-29_sinr_cp_kernel_followup/scientist_reviewer_log.md`.
- Exploration controller:
  `scripts/experiments/update_alignment/explore_mechanism_followup.py`.
- Sidecar monitor:
  `scripts/experiments/update_alignment/monitor_mechanism_explore.py`, writing
  `experiments/2026-06-29_sinr_cp_kernel_followup/exploration/live_monitor_status.json`
  and appending role-log updates without launching or stopping jobs.
- Initial evidence sources:
  completed P5 update-alignment tables, sampled SINR-vs-DirectCP proxy metrics,
  and prior LungCT case01/case08 pTVReg field-fit versus image-registration LR
  sweeps.
- New schedule-ablation direction: DirectCP `32,16,8,4` on LungCT cases `01`
  and `08`, removing or weakening the image-pyramid downsampling while
  preserving control-grid staging. Primary variants include pTV LCC + isoTV and
  global NCC + dense FD-BE objectives on canonical pTV-preprocessed 1mm crops.
- Smoke evidence: no-pyramid `1,1,1,1` stage-epoch jobs completed on cases
  `01` and `08` with no failures, but took roughly 4-6 minutes/job. Case08
  remained poor after this tiny smoke, so the long launch is a time-budgeted
  diagnostic set with `25,25,25,25` stage epochs, targeted extra case01/case08
  variants, and no all10 expansion yet.
- First-phase result: targeted schedule ablation completed `10/10` with `0`
  failures. Case01 reached `1.076` mm best reg1mm snap TRE without the image
  pyramid, while case08 remained poor under no-pyramid (`6.478` mm best after
  LR probe) and partial `4,2,1,1` pyramid (`11.937` mm best). This strengthens
  the optimization-path/capture-range interpretation but does not yet isolate
  whether the missing factor is the full standard `8,4,2,1` pyramid or a
  longer schedule.
- Second-phase standard-pyramid probe:
  `scripts/experiments/update_alignment/run_standard_pyramid_probe.py`,
  case08-only DirectCP `32,16,8,4` with standard `8,4,2,1` pyramid, pTV/FD-BE,
  LR, and modest schedule-length probes.
- Second-phase result: standard-pyramid probe completed `4/4` with `0`
  failures. Best case08 standard-pyramid short row was pTV LCC + isoTV,
  `8,4,2,1`, `25,25,25,25`, LR `0.01`, with `2.270` mm reg1mm snap TRE.
  LR `0.001` standard-pyramid rows remained poor (`11.496` mm pTV and
  `11.361` mm FD-BE), and the longer LR `0.001` pTV row reached `8.896` mm.
  Interpretation: full standard image-pyramid capture range plus optimizer
  scale substantially rescues case08, while no/partial pyramid does not.
  Remaining gap to the prior full schedule should be described as runtime/LR
  and metric-convention limited, not as a population-final conclusion.
- Active refinement: launched
  `scripts/experiments/update_alignment/run_standard_pyramid_probe.py` with
  `--probe-set refinement --phase-name capture_refinement_probe` on GPUs
  `0,1,2,3`. This tests case08 standard `8,4,2,1` LR `0.01` at
  `100,100,100,100` and `200,200,200,200`, plus partial `4,2,1,1` and
  no-pyramid LR `0.01` controls at `100,100,100,100`.
- Refinement result: completed `4/4` with `0` failures. Case08 reg1mm snap TRE
  converged near `1.21` mm for all LR `0.01` e100/e200 controls: standard
  `8,4,2,1` e100 `1.213`, standard e200 `1.217`, partial `4,2,1,1` e100
  `1.219`, and no-pyramid e100 `1.218`. Interpretation: the image pyramid
  matters under short optimization budgets, but longer LR `0.01` optimization
  removes the pyramid ablation gap in this diagnostic case.
- Requested runtime: user requested a long exploratory run of at least four
  hours, with the controller refreshing aggregation and role logs during the
  run.
- Known limitations: this is not a population-final result and not a
  pTVReg-method-only comparison. LungCT rows must be tagged as pTV-preprocessed
  diagnostic-domain experiments unless original/native TRE reconstruction is
  explicitly supplied by the source artifact.

### P7: Direct Dense-DVF vs IDIR Preliminary Check

- Status: completed.
- Date planned: 2026-06-30.
- Purpose: test whether replacing IDIR's continuous SIREN displacement model
  with a directly optimized dense displacement field is worth a fuller
  ablation. This isolates optimization over dense grid values from the IDIR
  coordinate-network parameterization while keeping the IDIR-style sampled NCC
  loss and bending anchor.
- Method handles:
  - OASIS `direct_dvf_sampled_ncc_fd_be`.
  - LungCT `direct_dvf_lungmask_sampled_ncc_fd_be`.
- Cases:
  - OASIS rank `9`, seed-3 shuffled first10 pair `(fixed 0173, moving 0136)`,
    original pair index `88`.
  - LungCT cases `01` and `08`, canonical pTV-preprocessed diagnostic crop
    domain.
- Fixed settings: seed `1`, LR `1e-4`, `2500` epochs, sampled image-loss
  locations `10000`, no tuning in this precheck.
- Regularization: direct dense DVF has no native continuous analytic bending
  formula, so training BE is tagged as
  `dense_forward_difference_be`. OASIS uses fixed image-positive valid-stencil
  BE; LungCT uses fixed T00 lung-mask valid-stencil BE on the cropped 1mm grid.
- IDIR anchors for interpretation:
  - OASIS IDIR alpha `1`, LR `1e-4`.
  - LungCT IDIR `alpha_bending=10`, LR `1e-4`.
- Launch GPU assignment: OASIS rank9 on GPU `0`; LungCT case01 on GPU `2`;
  LungCT case08 on GPU `3`. Do not use GPU `1` for this launch.
- Run roots:
  - `/database/inrorspline/runs/oasis_rank009_direct_dvf_idir_loss_precheck_v1`
  - `/database/inrorspline/runs/lungct_case01_case08_direct_dvf_idir_loss_precheck_v1`
- Review root:
  `experiments/2026-06-30_direct_dvf_idir_precheck/`.
- Required preflight: compile/import checks, OASIS rank9 validation, LungCT
  loader audit for cases `01,08`, and two-epoch smoke after implementation.
- Planned aggregation: report accuracy/TRE, dense finite-difference BE,
  diffusion, log-Jacobian, folding, NDV, runtime, and peak memory against the
  existing IDIR anchors. LungCT metrics remain pTV-preprocessed reg1mm crop
  diagnostics unless an original/native TRE reconstruction is explicitly added.
- Result: all `3/3` full jobs completed on requested GPUs `0,2,3` with no
  process failures. Aggregation wrote
  `experiments/2026-06-30_direct_dvf_idir_precheck/review_report.md`,
  `direct_dvf_idir_precheck_rows.csv`, and
  `direct_dvf_idir_precheck_rows.json`.
- Headline comparison: direct dense DVF under the matched IDIR-style loss was
  worse than the IDIR anchor on all three preliminary cases. OASIS rank9 Dice
  was `0.7673` versus IDIR `0.8110`, with much higher foreground folding
  (`9.889%` versus `1.560%`). LungCT case01 snap TRE was `3.764` mm versus
  IDIR `1.063` mm, and case08 snap TRE was `15.076` mm versus IDIR `1.331` mm.
- Preliminary interpretation: the direct dense-grid parameterization is
  feasible to run, but this fixed LR/alpha precheck does not motivate a broader
  same-settings expansion before trying stronger smoothness, different LR, or a
  multiscale/capture-range schedule.

### P8: DCTField IDIR-Loss K-Sweep Preliminary Check

- Status: completed.
- Date planned: 2026-06-30.
- Purpose: test a non-neural, non-spline compressed displacement basis as a
  paper-facing baseline for the deformation-search-space story. DCTField
  optimizes low-frequency cosine coefficients and decodes them to a dense
  displacement field.
- Method handles:
  - OASIS `dctfield_sampled_ncc_fd_be`.
  - LungCT `dctfield_lungmask_sampled_ncc_fd_be`.
- Cases:
  - OASIS rank `9`, seed-3 shuffled first10 pair `(fixed 0173, moving 0136)`,
    original pair index `88`.
  - LungCT cases `01` and `08`, canonical pTV-preprocessed diagnostic crop
    domain.
- DCT sizes: `16^3`, `32^3`, `64^3`, `128^3`, with coefficient counts
  `12288`, `98304`, `786432`, and `6291456`.
- Fixed settings: seed `1`, LR `1e-4`, `2500` epochs, sampled image-loss
  locations `10000`, no tuning in this precheck.
- Regularization: training BE is finite-difference BE over the decoded dense
  DCT field, tagged as `dct_decoded_dense_forward_difference_be`. Spectral DCT
  analytic BE is intentionally not used in this first round.
- IDIR/direct-field anchors for interpretation:
  - OASIS IDIR alpha `1`, LR `1e-4`; direct dense-DVF P7.
  - LungCT IDIR `alpha_bending=10`, LR `1e-4`; direct dense-DVF P7.
- Launch queue: GPU `0` runs OASIS K `16,32,64,128` sequentially; GPU `2`
  runs LungCT case01 K `16,32,64,128` sequentially; GPU `3` runs LungCT case08
  K `16,32,64,128` sequentially. Do not use GPU `1`.
- Run roots:
  - `/database/inrorspline/runs/oasis_rank009_dctfield_idir_k_sweep_precheck_v1`
  - `/database/inrorspline/runs/lungct_case01_case08_dctfield_idir_k_sweep_precheck_v1`
- Review root:
  `experiments/2026-06-30_dctfield_idir_k_sweep_precheck/`.
- Required preflight: compile/import checks, DCT decoder sanity checks, OASIS
  rank9 validation, LungCT loader audit for cases `01,08`, and two-epoch K128
  smokes for OASIS rank9 and LungCT case08. If K128 smoke fails with OOM or
  NaN, stop and report rather than silently dropping K128.
- Planned aggregation: compare IDIR anchor, direct dense-DVF P7, and DCTField
  K `16,32,64,128` on accuracy/TRE, dense finite-difference BE, diffusion,
  log-Jacobian, folding, NDV, final loss terms, runtime, and peak memory.
- Result: all `12/12` DCTField full jobs completed on requested GPU queues
  `0,2,3` with no process failures. Aggregation wrote
  `experiments/2026-06-30_dctfield_idir_k_sweep_precheck/review_report.md`,
  `dctfield_idir_k_sweep_rows.csv`, and `dctfield_idir_k_sweep_rows.json`.
- Headline comparison: OASIS DCTField K128 reached the best DCT Dice
  (`0.7781`) but folded heavily (`16.89%` foreground), while K32 was the
  smoother tradeoff (`0.7242` Dice, `0.265%` folding). LungCT case01 was best
  at K32 (`3.652` mm snap TRE, `0.739%` folding), slightly better and smoother
  than direct dense-DVF but far behind IDIR (`1.063` mm). LungCT case08 stayed
  near identity/direct-dense performance across K values, with best DCT K64
  `14.986` mm versus IDIR `1.331` mm.
- Preliminary interpretation: DCTField is useful as a compact spectral
  search-space baseline and supports the representation-prior framing, but this
  fixed LR/alpha single-stage DCT sweep does not explain IDIR's LungCT success
  or rescue the hard case08 capture-range failure.

### P9: Fit-Modes Panel A Global-NCC Curve Refresh

- Status: completed.
- Date planned: 2026-07-01.
- Purpose: rebuild the fit-modes paper figure Panel A on a consistent global
  NCC loss scale for OASIS rank9 and LungCT cases `01`/`08`, so all three
  cases can share the same y-axis while the fitting Panel B uses relative
  target-field error.
- Output root:
  `/database/inrorspline/runs/fit_modes_panelA_global_ncc_v1`.
- Smoke output root:
  `/database/inrorspline/runs/fit_modes_panelA_global_ncc_smoke_v1`.
- Review/artifact root:
  `experiments/paper_artifacts/fitting_figure/v001/`.
- Docker target: `py310sitreg_v1_test1`.
- Common diagnostic: write `global_ncc_diagnostic.csv` with full-image
  OASIS/global-crop LungCT NCC loss at epoch `1` and every `10` epochs. These
  diagnostic rows are tagged `affects_training=false` unless the diagnostic is
  also the optimized data term.
- OASIS rank9 existing reused Panel A curves:
  - IDIR `idir_original_dense_siren_be_autograd`, alpha `1`, lr `1e-4`,
    2500 epochs. Training remains IDIR sampled NCC + BE; diagnostic is
    full-image global NCC every 10 epochs from the completed matched-eval run.
  - SINR cps4 `sinr_cps4_be_bspline_analytic`, alpha `1000`, lr `1e-4`,
    2500 epochs. Optimized loss is full-image global NCC + analytic
    B-spline BE.
  - DirectCP cps4 `direct_cp_cps4_be_bspline_analytic`, alpha `1000`,
    lr `1e-3`, 2500 epochs. Optimized loss is full-image global NCC +
    analytic B-spline BE.
- OASIS rank9 new run:
  - DirectCP 8-4-2
    `direct_cp_8_4_2_be_bspline_analytic_multistage`, alpha `1000`,
    lr `1e-3`, total 2500 epochs.
  - Stage controls: cps `8,4,2`; stage downsample `4,2,1`; stage epochs
    `400,800,1300`.
  - Optimized loss is stage-domain global NCC + analytic B-spline BE; plotted
    diagnostic is full-image global NCC every 10 epochs.
- LungCT case01/case08 new runs:
  - IDIR `idir_be_default_lungmask`, `alpha_bending=3`, lr `1e-4`,
    2500 epochs. Training remains native IDIR sampled NCC + IDIR BE in the
    fixed T00 lung-mask sampling convention; plotted diagnostic is full-crop
    global NCC every 10 epochs.
  - SINR cps4 `sinr_cps4_global_ncc_fd_be`, alpha `10000`, lr `1e-4`,
    2500 epochs. Optimized loss is full-crop global NCC + dense
    forward-difference BE.
  - DirectCP cps4 `direct_cp_cps4_global_ncc_fd_be`, alpha `10000`,
    lr `1e-3`, 2500 epochs. Optimized loss is full-crop global NCC + dense
    forward-difference BE.
  - DirectCP 32-16-8-4 `direct_cp_32_16_8_4_global_ncc_fd_be`,
    alpha `10000`, lr `1e-3`. Stage controls `32,16,8,4`; stage downsample
    `8,4,2,1`; stage epochs `200,400,800,1000` for a 2400-epoch run matching
    the existing Pareto sweep protocol. Optimized loss is full-crop global NCC
    + dense forward-difference BE; plotted diagnostic is full-crop global NCC
    every 10 epochs.
- Standardized artifact outputs planned:
  - `fit_modes_standardized_global_ncc_panelA_curves.png`
  - `fit_modes_standardized_global_ncc_panelA_selected_runs.csv`
  - `fit_modes_standardized_global_ncc_panelA_curve_summary.csv`
  - `fit_modes_standardized_global_ncc_panelA_manifest.json`
- Figure rule: one row by three cases, shared global NCC y-axis where possible,
  and y tick labels plus y-axis label visible on every subplot.
- Required preflight: Docker `py_compile`, two-epoch or three-epoch smokes
  covering OASIS multistage, LungCT IDIR diagnostic, LungCT single-stage
  global-NCC+FD-BE, and LungCT multistage global-NCC+FD-BE.
- Result: preflight compile and all five smoke paths passed. Full launch
  completed `9/9` new jobs with all queues returning status `0`.
- Diagnostic validation: every `global_ncc_diagnostic.csv` has finite losses
  and expected epoch coverage. 2500-epoch runs have `251` rows; LungCT
  2400-epoch multistage runs have `241` rows.
- Final global NCC losses:
  - OASIS DirectCP 8-4-2: `-0.994035`.
  - LungCT case01 IDIR/SINR/DirectCP cps4/DirectCP multistage:
    `-0.946411`, `-0.997760`, `-0.998162`, `-0.998160`.
  - LungCT case08 IDIR/SINR/DirectCP cps4/DirectCP multistage:
    `-0.910503`, `-0.993551`, `-0.992049`, `-0.997240`.
- Standardized artifact generated:
  `experiments/paper_artifacts/fitting_figure/v001/fit_modes_standardized_global_ncc_panelA_curves.png`.

### P10: OASIS DirectCP 8-4-2 Field-Fit rRMSE Panel B Add-On

- Status: completed.
- Date planned: 2026-07-01.
- Date started: 2026-07-01.
- Date completed: 2026-07-01.
- Purpose: add the OASIS DirectCP 8-4-2 multistage trajectory to the
  standardized Panel B target-field rRMSE fitting figure for both OASIS rank9
  SITReg and Greedy targets.
- Dataset: OASIS BrainMR seed-3 shuffled rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`.
- Target fields:
  - SITReg target: saved `bs4_sitreg_v0` best-Dice dense displacement,
    normalized-grid units.
  - Greedy target: canonical Greedy dense displacement, normalized-grid units.
- Method: `direct_cp_8_4_2_be_bspline_analytic_multistage`, recorded as
  `direct_cp_8_4_2_field_mse` for field-fit outputs.
- LR/alpha/schedule: alpha `1000` recorded for operating-point consistency
  but not used by the MSE-only objective; lr `1e-3`; total 2500 epochs; cps
  `8,4,2`; stage downsample `4,2,1`; stage epochs `400,800,1300`.
- Optimized loss: stage-domain target-field MSE over the fixed raw
  image-positive OASIS brain mask, no image loss and no regularization.
- Diagnostic loss: finest-domain masked target-field MSE in normalized-grid
  units at epoch `1`, every `10` global epochs, and final epoch. The Panel B
  rRMSE curve uses `100 * sqrt(mse / (target_power + 1e-12))`.
- Output root:
  `/database/inrorspline/runs/fit_modes_oasis_directcp842_field_fit_v1`.
- Smoke output root:
  `/database/inrorspline/runs/fit_modes_oasis_directcp842_field_fit_smoke_v1`.
- Review/artifact root:
  `experiments/paper_artifacts/fitting_figure/v001/`.
- Docker target: `py310sitreg_v1_test1`.
- Required preflight: Docker `py_compile` for modified OASIS field-fit
  runners and figure generator; three-epoch smoke runs for both SITReg and
  Greedy targets with finest-domain MSE diagnostics every epoch.
- Full commands: run the SITReg and Greedy target fits with
  `--method direct_cp_8_4_2_be_bspline_analytic_multistage --alpha 1000
  --lr 1e-3 --epochs 2500 --stage-full-cps 8,4,2 --stage-downsample 4,2,1
  --stage-epochs 400,800,1300 --finest-mse-diagnostic
  --finest-mse-every 10`.
- Known limitations: this is a single OASIS illustrative pair and field-fit-only
  diagnostic; the field-fit objective uses target-field MSE rather than the
  image-registration global NCC + BE objective used in Panel A.
- Result: Docker `py_compile` passed; both three-epoch smoke jobs passed; both
  full 2500-epoch jobs completed with `DONE` markers.
- Diagnostic validation: both full `stage_history.csv` files contain 2500 rows
  and 251 finite finest-domain MSE diagnostics covering epoch `1` plus every
  `10` epochs through `2500`, with no missing/extra diagnostic epochs.
- Final masked normalized-grid target-field MSE:
  - SITReg target: `1.148402e-06`.
  - Greedy target: `1.004255e-07`.
- Standardized Panel B rRMSE results:
  - SITReg target: start `97.33%`, epoch 10 `76.30%`, final `4.50%`.
  - Greedy target: start `97.58%`, epoch 10 `78.18%`, final `1.23%`.
- Standardized artifact regenerated:
  `experiments/paper_artifacts/fitting_figure/v001/fit_modes_standardized_rrmse_field_fit_curves.png`.

### P11: OASIS DirectCP 8-4-2 Canonical lr=1e-3 Pareto/Table Standardization

- Status: completed 2026-07-01 after smoke, full launch, aggregation,
  candidate artifact regeneration, and comparison audit.
- Purpose: replace the final-paper OASIS DirectCP 8-4-2 `lr=1e-2`
  Pareto/table temporary exception with a canonical `lr=1e-3` candidate while
  leaving current `v001` artifacts intact for historical review.
- Dataset: OASIS BrainMR seed-3 shuffled first20 ranks `0-19`.
- Method: `direct_cp_8_4_2_be_bspline_analytic_multistage`.
- Objective: full-image global NCC plus `alpha *` analytic cubic B-spline BE;
  foreground regularity metrics use the fixed raw image-positive mask.
- Alpha grid: `0,10,100,1000,10000,100000`.
- LR/schedule: `lr=1e-3`; cps `8,4,2`; stage downsample `4,2,1`; stage
  epochs `400,800,1300`; total epochs `2500`; seed `1`; eval interval `250`.
- Launch environment: Docker container `py310sitreg_v1_test1`; use all visible
  GPUs with one job per GPU.
- Smoke output root:
  `/database/inrorspline/runs/oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_smoke_v1`.
- Full output root:
  `/database/inrorspline/runs/oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_v1`.
- Review root:
  `experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization/`.
- Candidate artifact roots:
  `experiments/paper_artifacts/best_result_table/v002_candidate/`,
  `experiments/paper_artifacts/pareto_figure/v002_candidate/`, and
  `experiments/paper_artifacts/comparison/v001_vs_v002_candidate/`.
- Monitoring: poll every 5 minutes; report completed/failed/running/remaining
  jobs, elapsed time, throughput, ETA, newest log update, and failure log
  paths; save snapshots in the review root.
- Required preflight: Docker `py_compile`; one rank/one-alpha smoke with stage
  epochs `1,1,1`; verify finite metrics and expected multistage files.
- Acceptance: full run produces `120/120` summaries and `DONE` markers;
  aggregation has no missing case-alpha combinations; `v002_candidate`
  validation has zero `temporary_exception`, `mismatch`, or blocking LR issues;
  current `v001` artifacts remain unchanged.
- Actual completion: full run completed `120/120` jobs with `0` failures in
  about `2h 7m`; aggregation reported `combined_rows=120`,
  `population_scopes={"first10": 6, "first20": 6}`, and `errors=[]`.
- Candidate artifact audit: best-result table validation has `11`
  `matches_standard` rows plus `4` references; Pareto validation has `11`
  `matches_standard` rows; no `temporary_exception`, `mismatch`, or blocking
  LR issue remains in `v002_candidate`.
