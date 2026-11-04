# P71: Bounded Three-Map Composition Extension to the First-10 Depth Comparison

## Objective

Add one fresh bounded three-map composition setting to each P70 schedule:

- cps8
- cps8→4
- cps8→4→2
- cps8→4→2→1

Use seed-3 shuffled OASIS ranks0–9. This adds40 registrations. The accepted P70 160-run dataset remains immutable and is included only as the four existing method rows in a separate combined report.

## Frozen configuration

The new setting is bounded cubic B-spline, three independently optimized maps per active level, retained and composed as `Phi_previous o delta_current`. Each increment starts from zero controls; the whole registration starts from identity. Every map receives a complete per-stage optimizer budget and a fresh Adam optimizer, matching P70's two-map convention. This intentionally increases capacity and update count; it is not an equal-compute comparison.

All active levels use diffusion λ=1, squared local NCC9 (`1 - mean(correlation²)`), unmasked total-field voxel-unit diffusion, P70 preprocessing, endpoint-aligned spline grids, and border sampling. Keep the validated0.99 tanh bound, normalized-coordinate learning rates, per-stage stopping, checkpoint selection, FP32, seed1, deterministic cuDNN request, and TF32 disabled unchanged.

| CPS | Downsample | LR | Max/min epochs | Patience |
|---:|---:|---:|---:|---:|
| 8 | 4 | 0.03 | 400/40 | 20 |
| 4 | 2 | 0.01 | 800/80 | 35 |
| 2 | 1 | 0.003 | 1300/130 | 50 |
| 1 | 1 | 0.001 | 2000/300 | 100 |

Each active map uses the corresponding row's full schedule; P70's adaptive checks and checkpoint rules remain unchanged. No hyperparameter tuning or added stages are allowed.

## Execution and artifacts

Run as UID:GID1009:1020 on available GPUs0–3, one worker per GPU. Keep each case's four schedules on one GPU and run its jobs sequentially. Recheck live memory before each launch; require at least8GiB free or measured smoke peak+2GiB, whichever is larger. Do not modify existing jobs. Preserve all failed attempts and logs.

Use a new artifact key: `oasis_seed3_first10_mrdbscp8421_bspline_bounded3_depth_extension_v1`. Run four one-update full-grid smokes (one per schedule) before production. Save all three maps per stage, histories, checkpoints, final DVFs, audits, source/input hashes, GPU state, stopping records, and runtime diagnostics.

The separate report combines the accepted P70 four-arm data with the40 new registrations and original VFA/SITReg references. It must not rewrite P70 files. It contains four schedule tables with five optimization methods and two reference rows; mean±sample SD across the same ten cases; per-case and paired-reference CSV; stage diagnostics; and P71 shared-prefix diagnostics.

## Acceptance

- Four smokes pass before the40 production fits.
- All40 new finite fields have correct identity, rank, direction, and source metadata.
- All300 map checkpoints (10 cases × 3 maps × (1+2+3+4 schedule levels)) reconstruct exactly, with consistent histories, lowest-loss checkpoint selection, and bounded-map contraction audits.
- All40 independent replay audits reproduce final canonical Dice, HD95, symmetric ASD, foreground SD(log J), folding, and off-grid map diagnostics.
- Combined summary has28 rows; per-case data280 rows; paired VFA/SITReg comparisons400; map-stage diagnostics900.
- Report and source/reference hashes pass. First-ten results remain exploratory; no generalization claim.

## Known comparison caveat

Three-map composition is more expressive and performs three full map schedules at every active level. Any accuracy/regularity gain must be interpreted alongside actual Adam updates and optimizer time, not as a pure bound-only effect.
