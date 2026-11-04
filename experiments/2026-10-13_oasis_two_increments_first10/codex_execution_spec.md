# P69: two bounded increments, first10, cps2 lambda5 versus1

User requested the P68 two-increment method on all seed3 shuffled first10
OASIS pairs at stage weights[1,1,5,1] and[1,1,1,1]. This is20 outputs: reuse
the independently accepted P68 rank0/4 lambda1151 results unchanged, run18
fresh identity registrations. No same-map continuation reruns or other tuning.

## Frozen method

Retain8 maps: cps8a/8b/4a/4b/2a/2b/1a/1b. Each starts at identity, with
fresh Adam and frozen previous parameters but coordinate gradients through
their sampling. Preserve P68 composition order, endpoint-aligned cubic
expansion, native border-trilinear displacement interpolation,0.99 elementary
bound and c=a*tanh(p/a) normalized latent parameterization. Never collapse
the retained composition, reuse an optimized prefix, or double bounds.

| CPS | Loss downsampling | Lambda | Normalized LR | Max/min epochs | Requested patience |
|---|---:|---:|---:|---:|---:|
|8|4|1|0.03|400/40|20|
|4|2|1|0.01|800/80|35|
|2|1|5 or1|0.003|1300/130|50|
|1|1|1|0.001|2000/300|100|

Each setting repeats for both increments at its level; do not split lambda.
Raw squared local NCC9, unmasked total-field forward-valid component/axis-mean
voxel diffusion in the current evaluation grid. This is an explicit override
to the repo analytic-regularizer default. No intensity renormalization or new
loss scaling, no segmentation loss/reference supervision. Same best-total-loss
selection including identity, median/check25, thresholds1e-4/5e-5, effective
patience>=25. Each cps1 block permits2 LR halvings at least100 epochs apart.
Seed1, FP32, cuDNN benchmarkFalse/deterministicTrue, TF32off.

## Implementation and execution

Separate cohort adapter with explicit stage-alpha vectors and unique names;
reuse immutable P68 worker/numerical optimizer through a scoped configuration
binding. Restore bindings after calls; do not edit historical source files.
Record adapter/source hashes, source revision, input/split/reference hashes,
complete settings, bounds, stages, GPU and selected/stop epochs.

Use existing container as1009:1020, one new worker per GPU0/1/2/3. Recheck
GPU state before each launch; require8GiB or smoke peak+2GiB. Keep each
case's configurations on the same GPU; rank0/4 new1111 fits use their pilot
GPU0/2 where available. Leave existing jobs untouched. Preserve failures,
stop admitting new work after failure, let active work finish, then audit.
No automatic numerical/protocol adjustments or output overwrites.

Before production: unit/regression tests for CLI/weights/names/context
restoration/inherited numerical identity and complete-cohort/sampleSD logic;
two full-size rank1 eight-block one-step smokes, one per vector. Independently
replay every new production run's8 checkpoints, initial and selected losses,
per-label canonical metrics, elementary corners and100k off-grid probes.
Verify the common cps8a/8b/4a/4b fields/stopping/losses across vectors within
historical tolerance1e-4 max/1e-5 RMS voxel and1e-6 loss; exact epochs.

## Reporting and acceptance

External runs/repo_experiments key:
`oasis_seed3_first10_mrdbscp8421_tanh_two_increments_lambda1151_1111_v1`.
Require all20 finite completed fields,18 new independent audits and2 accepted
pilot audits before aggregation. Save all160 block records, histories and
checkpoints (pilot files referenced, not copied/modified).

CSV/JSON/Markdown: per-case results, paired1111-minus1151 changes, stage
diagnostics, runtime, folding/off-grid diagnostics; mean±across-case sampleSD
for Dice,HD95,symmetricASD,positive-J FG SDlogJ and FG folding%. Compare
original VFA/SITReg and P67 all-bounded/last2 references on identical cases.
FG=fixed raw image>0, unchanged interior Jacobian crop and absent-label rules.
Main ranks0–9 table plus nonpilot ranks1/2/3/5/6/7/8/9 separately. All are
exploratory test-case results; lowering cps2 lambda was motivated by pilot
observations. Do not claim unbiased test selection or unconditional zero folds.
