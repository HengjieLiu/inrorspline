# P66: rank0 selectively bounded compositional B-splines

User requests three8/4/2/1 variants: bound only cps1, cps2/1, or cps4/2/1.
Run exactly three fresh identity registrations on seed3 shuffled rank0,
fixed0333/moving0115. Stage weights remain[1,1,5,1]. Keep composition at ALL
stages to isolate which elementary increments receive tanh bounding; this is
not historical knot-insertion prefix optimization followed by clipping.

| Variant | cps8 | cps4 | cps2 | cps1 |
|---|---|---|---|---|
| last1 | unbounded | unbounded | unbounded | bounded |
| last2 | unbounded | unbounded | bounded | bounded |
| last3 | unbounded | bounded | bounded | bounded |

Reuse P64's native cubic expansion, retained Phi_previous o delta_new chain,
border displacement/image sampling, raw squared local NCC9 and unmasked
total-composition evaluation-grid voxel diffusion. One zero-initialized
increment/stage, no reused optimized prefix or accumulated-map resampling.
Unbounded controls c=p; bounded controls c=a*tanh(p/a), a=0.99*cps*b/S.
Both have unit derivative at zero. Only bounded maps receive contraction
acceptance gates. Unbounded elementary/total folding is measured, NOT rejected
or silently repaired. Earlier folds can persist through later bounded maps.

Frozen P64 schedules: downsample4/2/1/1; LR0.03/0.01/0.003/0.001;
max400/800/1300/2000; min40/80/130/300; requested patience20/35/50/100,
effective>=25; window/check25; relative total/data1e-4/5e-5. Fresh Adam,
lowest total-loss state including initialization. Only cps1 gets two LR
halvings,>=100 epochs apart. Seed1, FP32, cuDNN benchmarkFalse/deterministicTrue,
TF32off. No segmentation supervision, analytic regularizer, lambda/LR tuning,
cps16 or first-ten expansion. CUDA grid_sample nondeterminism remains possible.

Separate core/runner, unchanged historical sources/output hashes. Explicit
per-stage bounded flags, applied bound or null, raw parameter and transformed
control checkpoints, field/history hashes and source/device/input provenance.
Tests cover stage masks/names/CLI, bounded parity, true unbounded controls and
checkpoint roundtrip, unit initial gradients, composition and finite folding
diagnostics. One-step-four-stage smoke for EACH variant gates production.

Run GPU3 sequentially as1009:1020, current memory admission8GiB or maximum
smoke peak+2GiB. Existing jobs untouched. Preserve failures and stop for audit.
Require finite native160x224x192x3 DVFs, canonical label metrics, exact reload
and selected-loss consistency, bounded atomic derivative limits, and the same
100k foreground probe locations as P64 for both chain and exported interpolant.
Check independent runs' common initial stages agree within deterministic
tolerance; report any drift rather than assuming field equality.

Artifact key under external ARTIFACT_ROOT runs/repo_experiments:
oasis_seed3_rank000_mrdbscp8421_selective_tanh_lambda1151_v1.
CSV/JSON/Markdown compare all three with P64 fully bounded, P63 historical
unbounded prolonged spline, P62 dense, original VFA and SITReg. Report
Dice/HD95/symmetric ASD, fixed-raw-positive positive-J SDlogJ and folding%,
stage diagnostics, saturation only where bounded, epochs, loss and runtime.
Independently reconstruct/recompute each final candidate before handoff.
Single exploratory case, no across-case SD or unconditional topology guarantee.
