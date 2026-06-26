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

### P1d: OASIS Seed-3 First10 DirectCP cps2 Six-Alpha Completion

- Status: completed; completion launch ran 2026-06-25 15:18-15:41 Pacific
  after a 2-epoch smoke passed, and full overlay aggregation completed
  2026-06-25 15:43 Pacific.
- Purpose: complete DirectCP cps2 from the preliminary alpha subset to the full
  six-alpha spline grid without overwriting P1 or P1c review artifacts.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`.
- Method: DirectCP cps2 with cubic B-spline analytic BE/diffusion.
- Completion grid: alpha `0,10,100000`; combine with P1c alpha
  `100,1000,10000` for the full cps2 grid.
- Fixed settings: 2500 epochs; seed `1`; eval interval `250`; lr `1e-3`.
- Output root:
  `/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_completion_a0_10_100000_v1`.
- Full overlay review root:
  `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_full_overlay/`.
- Launch environment: Docker container
  `inrorspline_oasis_seed3_first10_run_20260625`, image
  `hengjieliu96/dl_dir:py310sitreg_v1_251008`; requested GPUs `1,2,3`.
- Acceptance: completion root `30/30` with `failures=[]`, combined DirectCP
  cps2 `60/60`, six alphas by ten ranks, full overlay report PASS.

### P1e: OASIS Seed-3 Next10 Five-Model Alpha Sweep

- Status: completed; dry-run and 30-job smoke passed on 2026-06-26, full
  launch completed 2026-06-26 04:16 Pacific on shenggpu9 with aggregation PASS.
- Purpose: extend the OASIS seed-3 BE alpha sweep to the next ten shuffled
  ranks, with DirectCP cps2 included as a first-class fifth model.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `10-19`, original sorted
  indices `130,100,118,67,79,56,142,199,36,12`.
- Methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4, DirectCP cps2.
- Grid: IDIR alpha `0,0.01,0.1,1,10,100`; spline alpha
  `0,10,100,1000,10000,100000`; 2500 epochs; seed `1`; eval interval `250`.
- LR: IDIR `1e-4`; SINR cps4/cps2 `1e-4`; DirectCP cps4/cps2 `1e-3`.
- Regularizer: BE-trained sweep. IDIR uses coordinate-autograd analytic BE;
  spline/DirectCP methods use cubic B-spline analytic BE.
- Launch pattern: method phases run sequentially, starting with slower SINR
  cps2, and each phase dynamically distributes its `10 x 6` case/alpha jobs
  across four GPUs.
- Expected jobs: `300`.
- Output root:
  `/database/inrorspline/runs/oasis_seed3_next10_alpha_sweep_be_2500e_5models_v1`.
- Review root:
  `experiments/2026-06-26_oasis_seed3_next10_alpha_sweep/`.
- Launch environment: Docker image `hengjieliu96/dl_dir:py310sitreg_v1_251008`;
  container `inrorspline_oasis_seed3_next10_run_20260626`.
- Acceptance: `300/300` summaries, `300/300` `DONE` markers, `failures=[]`,
  and review report PASS.

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

### P3: LungCT 4DCT Matched Pareto

- Status: planned.
- Methods: IDIR, SINR, single-stage DirectCP cps4, multistage DirectCP, pTVReg references.
- Required: `docs/standards/lungct_protocol_tags.md` and explicit protocol
  tags for preprocessing/domain/mask/loss.

### P4: pTV-Style DirectCP Ablation Ladder

- Status: planned.
- Purpose: isolate pTVReg gap by adding local CC, border mask, isoTV, pyramid, optimizer, and basis factors one at a time.
