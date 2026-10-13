# OASIS first10: DL NCC9/diffusion six-way sweep

User approved implementation and execution after the accompanying audit.

## Protocol

- Dataset: seed3 shuffled pair_test_200 ranks0-9, first subject fixed, second
  moving; pull sampling moving(identity + displacement).
- Representations: cubic cps8/4/2; cubic cps8/4/2/1; cubic cps8/4/2 then direct
  dense normalized XYZ displacement. No residual composition or double warp.
- Resolution modes: pyramid downsample4/2/1[/1] and full1/1/1[/1]. Both loss
  terms use the stage grid. Stage images use existing Gaussian antialiasing
  sigma0.5*factor, radius ceil(3*sigma); no blur at factor1.
- Lambda0.01/0.1/0.3/1/3/10/100, identical at every stage.
- Raw float32 intensities; VFA training NCC_vfa(window_size=9) + lambda times
  unmasked forward-valid component-mean L2 diffusion of total voxel field.
- Normalized parameters and LRs0.03/0.01/0.003/0.001 throughout; fresh Adam
  beta0.9/0.999, eps1e-8, weight_decay0, amsgradFalse at every stage.
- Prefix max/min400/40,800/80,1300/130, requested patience20/35/50, effective
  patience25/35/50. Check/window25, relative total/data thresholds1e-4/5e-5.
- Both final refinements: max/min2000/300, patience100, check/window25,
  same thresholds; halve LR up to twice, >=100 updates between actions, then
  stop on the next plateau. No additional LR sweep.
- Best-total-loss checkpoint, including initialization, at every stage;
  retain last checkpoints/histories. Compare losses only within a stage.
- Correct endpoint-aligned cubic evaluation in pyramid mode; retain exact
  crop-aligned dyadic knot insertion and existing full-resolution cubic basis.
- Seed1, FP32, cuDNN benchmarkFalse/deterministicTrue, TF32 disabled. This does
  not guarantee globally bitwise deterministic CUDA grid_sample backward.

## Execution and acceptance

140 independently optimized prefixes provide420 outputs after280 refinement
branches. Share a prefix only within the same case/lambda/resolution/settings.
Run one added worker per GPU0-3, max4, container UID:GID1009:1020. Preserve
failures and retry CUDA OOM once on a sufficiently free released GPU. Do not
alter existing workloads. GPU timing is diagnostic.

Use distinct run/review roots under ARTIFACT_ROOT named
oasis_seed3_first10_mrdbscp_dl_ncc9_diffusion_6way_v1. Source hashes, input hashes,
commands, GPU state, prefix hashes, effective schedules and best/last losses
are recorded. Existing runs and sources are not overwritten. Smokes have
separate names and cannot count toward420 production outputs.

Run CPU/CUDA mathematical parity tests and six rank0 one-step full-geometry
smokes before production. Aggregate only420 successful finite native-geometry
outputs, with valid checkpoints, metadata, hashes and completion markers.

## Reports

Per-case CSV/JSON and42 configuration means/sample SDs, original audited VFA
and SITReg references, paired metric differences, canonical Dice/HD95/symmetric
ASD, foreground SD(log J), folding/positive-J fraction, diffusion, excluded
labels, stage epochs and runtime. Plot2x2 faceted and combined two-panel
Dice/HD95 vs SD(log J) curves, with marginal +/-1 sample-SD ellipses and lambda
labels; PNG/PDF/SVG. Include a folding companion. Choose a lambda per variant
by mean Dice, then lower ASD/HD95, with metric-specific winners and Pareto
points. All first10 are exploratory tuning, not held-out confirmation.
