# OASIS 8-4-2 to frozen-VFA-feature dense refinement

## Objective and limitations

Implement the user-approved bounded experiment, not a guarantee of matching
VFA. Rank 0 is exploratory test-case hyperparameter tuning. Freeze the selected
feature and raw-image control configurations before ranks 1–9. Shared-GPU
runtime is diagnostic; cuDNN deterministic mode is not global CUDA determinism.
Frozen-feature MSE/NCC is not VFA's native local-attention objective.

## Protocol

- Seed-3 shuffled first10 pair file; rank 0 is fixed 0333, moving 0115,
  sorted index 158. Moving-to-fixed pull sampling; 160x224x192, 1-mm voxels.
- Unchanged camera-ready analytic-BE prefix: cps 8/4/2, downsample 4/2/1,
  lambda 1000, LR .03/.01/.003, max 400/800/1300, min 40/80/130,
  patience 20/35/50, check 25, total/data thresholds 1e-4/5e-5.
- Cache the expanded prefix once per case. Dense Adam parameter is absolute
  voxel-unit xyz displacement, initialized by a detached clone. No cps1 or
  analytic spline regularizer in refinement.
- Strictly load best-Dice VFA checkpoint dice0.8115_epoch0171.pth.tar using
  the native architecture (16-to-8 full-resolution projection). Native raw
  float32 RAS images feed VFA; pair-minmax images feed the prefix/control.
- Warp-then-project follows native moving replicate padding and central
  projected-candidate extraction. Project-then-warp caches that projection.
  Neither prewarps and reapplies the prefix twice. Sampling is trilinear,
  border padded, align_corners=True.
- Loss is feature loss divided by its fixed initial value (floor 1e-8), plus
  lambda times unmasked forward-valid L2 diffusion of the total voxel field.
  Means include components, then average the three spatial-axis means.
- MSE and signed per-channel global NCC are primary. Fallback local NCC uses
  a 9-cubed window with valid boundary counts; fixed-positive support selects
  window centers, not neighborhoods. Unit-MSE L2-normalizes channels per voxel.
  NCC denominator floor is sqrt(1e-12); channel normalization epsilon is 1e-6.
- Segmentation labels and saved VFA fields never enter optimizer loss or epoch
  selection. Evaluate the lowest-total-loss state, including initialization.

## Bounded search

- A: 48 = two orders x MSE/global NCC x lambda 0/.001/.01/.1/1/10 x
  LR .03/.1 voxels.
- B: 24 = two best families x best-positive lambda times 1/3,1,3 x
  LR .01/.05/.2/.3. Each restarts from the prefix, not an earlier fit.
- C: 20 = winning order/LR x local NCC/unit-MSE x full/fixed-positive support
  x best-positive family lambda times .1/.3/1/3/10.
- Controls: four raw global-NCC/diffusion fits using selected LR and positive
  family lambda times .1/.3/1/3. No more than 96 production rank-0 fits.
- Stop expanding after a completed feature phase meets all three tolerances.
  Four one-step smoke fits are preflight only and separately identified.
- Rank by the maximum tolerance-normalized deficit, unless a candidate meets
  all tolerances: then highest Dice, lower ASD, lower HD95. Tolerances versus
  original VFA are Dice -0.001, HD95 +0.02 mm, ASD +0.005 mm.
- Dense budget 2000/min300; checks/window25, patience100; thresholds
  1e-4/5e-5. Halve LR twice on plateaus, retaining Adam state; wait at least
  100 steps after each change, then stop on the next plateau. Seed1, FP32,
  cuDNN benchmark=False/deterministic=True.

## Execution and artifacts

- Use the approved existing container as UID:GID 1009:1020. Pass dataset,
  baseline, VFA source, and checkpoint roots explicitly; private absolute
  paths do not belong in this specification.
- Launcher: scripts/experiments/oasis_alpha_sweep/launch_oasis_vfa_feature_dense.py,
  with --gpus 0,1,2,3 --max-concurrent 4. --stop-after prepare/smoke/search/full
  supports gated/resumable execution; default full.
- One new worker per physical GPU. Log physical/visible device, free memory,
  utilization, source/checkpoint/cache/DVF hashes, UID/GID, settings, histories,
  stopping, peak memory and runtime. Never modify existing jobs.
- Preserve all failed attempts/logs. One OOM retry on the most-free released
  GPU; non-OOM failures or exhausted retries block phase aggregation.
- Prefix/features and all results are external under
  ${ARTIFACT_ROOT}/runs/oasis_seed3_first10_mrdbscp842_dense_vfa_features_v1;
  CSV/JSON/Markdown reports use the corresponding repo_experiments root.
- After freezing, prepare ranks1–9 and run the selected feature/control on
  each. Report per-case and separate ranks1–9/0–9 means and match counts.
  Retain the historical rank0 raw-NCC/BE result as contextual comparison.

## Acceptance

- Unit contracts, legacy tests, and four full-volume one-step smokes pass.
- Native VFA parity before fitting: voxel RMSE <=.001; canonical metric
  deviations <=.0001 Dice, .01 HD95, .001 ASD. Save all discrepancies and
  stop rather than replacing the historical reference on failure.
- Rank0 cached prefix stops at 125/150/175; converted prefix-to-parameter clone
  difference <=1e-6 voxel. Redundant normalized-coordinate round-trip rounding
  is reported separately and is not mistaken for initialization drift.
- Finite correctly shaped fields, exact case/direction/backend/config/cache
  identity, canonical absent-label exclusions, and DONE markers required.
- Technical completion and scientific matching are separate report flags.
  No reporting metric mixtures across different fields and no post-confirmation
  hyperparameter changes.
