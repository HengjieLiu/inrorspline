# P70: First-10 B-Spline Pyramid Depth × Constraint Factorial

## Objective

Run a fresh 4×4 factorial on seed-3 shuffled OASIS ranks0–9. Cross pyramid
depths cps8, cps8→4, cps8→4→2, and cps8→4→2→1 with four update styles:

1. Unbounded cubic B-spline, one map per level.
2. Unbounded cubic B-spline, two independently optimized maps per level,
   retained and composed.
3. Bounded cubic B-spline, one map per level.
4. Bounded cubic B-spline, two independently optimized bounded maps per level,
   retained and composed.

This is 16 settings and 160 fresh registrations. Every setting starts from
identity; prior P69 outputs are comparison/provenance data only.

## Frozen numerical setup

All active levels use diffusion λ=1 and the existing raw-intensity local NCC9
objective `1 - mean(correlation²)`, plus unmasked forward-valid total-field
voxel-unit L2 diffusion. Keep endpoint-aligned cubic B-spline geometry,
existing preprocessing and border sampling, and retained pull composition
`Phi_previous o delta_current`. Start each independent map from zero controls
with fresh Adam. For two-map arms, each map gets the full stage optimizer
schedule, so their update count is larger by design. Bounded maps use the
validated 0.99 scaled-tanh cap with unit derivative at zero. Unbounded maps are
identity-parameterized controls, with no clipping or rejection for folding.

| CPS | Loss downsample | λ | Normalized-coordinate LR | Max/min epochs | Patience |
|---:|---:|---:|---:|---:|---:|
| 8 | 4 | 1 | 0.03 | 400/40 | 20 |
| 4 | 2 | 1 | 0.01 | 800/80 | 35 |
| 2 | 1 | 1 | 0.003 | 1300/130 | 50 |
| 1 | 1 | 1 | 0.001 | 2000/300 | 100 |

Retain checks every25 epochs, median window25 and relative total/data
thresholds1e-4/5e-5. The cps1 stage allows two LR halvings, at least100 epochs
apart, then stops on the next plateau. Select the lowest total-loss state,
including initialization. Seed1, FP32, deterministic cuDNN, benchmark off,
TF32 off. Record residual CUDA nondeterminism caveats.

## Execution and reporting

Use the accepted P69 artifact only as read-only provenance, original VFA/SITReg
references, and a fresh-vs-prior reproduction diagnostic for the full-depth
bounded two-map λ=1 cell. Never count an old output as a factorial cell.

Run one worker per available GPU0–3 as UID:GID1009:1020, after a live memory
check requiring at least8GiB free (raised to measured smoke peak+2GiB if larger).
Assign each case to one GPU and execute all16 settings for that case
sequentially; distribute ranks across GPUs. Do not alter existing GPU jobs.
Run four native-grid one-update smokes, one per update style, before production.

Produce four Markdown tables, one per pyramid depth, each comparing four new
settings plus original VFA and SITReg. Report first-ten mean±sample-SD Dice,
HD95, symmetric ASD, fixed-foreground SD(log J), foreground folding percent,
optimizer time, worker time, and actual Adam updates. Include per-case CSV/JSON,
paired DL-reference differences, stage stop/best-epoch diagnostics, all map
checkpoints/histories, source and input hashes, current GPU state and
assignments, and an explicitly separate P69 reproducibility appendix.
VFA/SITReg timing is n/a. Label this cohort exploratory because its cases
informed earlier tuning.

## Verification and acceptance

Require160 complete finite `[160,224,192,3]` outputs, correct rank/case/direction
metadata, histories/checkpoints consistent with best selected loss, canonical
absent-label handling, and independent replay of every final result and map.
Record shared-map prefix reproducibility across independently restarted depths.
Require byte-identical cps8/cps4 prefixes; for cps2, report full-field
differences, objective deltas, and stopping-epoch differences because CUDA
backward operations can remain nondeterministic despite deterministic-mode
flags. Do not discard otherwise valid independent cells solely for that
cross-run variation.
Bounded increments must pass validated contraction checks; unbounded results
must remain truly unconstrained and report measured folding. Aggregate only
after four smokes, all production cells, all independent replays, paired checks,
and report row-count/hash checks pass. Preserve failures and prior artifacts.

Artifacts use the distinct key
`oasis_seed3_first10_mrdbscp8421_bspline_depth_bound_steps_lambda1_v3`.
