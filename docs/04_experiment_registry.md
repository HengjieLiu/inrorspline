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

- Status: in progress.
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
- Required verification: dry-run queues exactly `80` jobs and smoke jobs pass
  for case `01` at factors `0.01` and `100` for both methods.

### P3: LungCT 4DCT Matched Pareto

- Status: planned.
- Methods: IDIR, SINR, single-stage DirectCP cps4, multistage DirectCP, pTVReg references.
- Required: `docs/standards/lungct_protocol_tags.md` and explicit protocol
  tags for preprocessing/domain/mask/loss.

### P4: pTV-Style DirectCP Ablation Ladder

- Status: planned.
- Purpose: isolate pTVReg gap by adding local CC, border mask, isoTV, pyramid, optimizer, and basis factors one at a time.
