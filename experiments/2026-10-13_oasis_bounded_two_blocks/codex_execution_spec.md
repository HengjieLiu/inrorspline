# P68: two bounded increments versus two blocks on the same map

Approved four-run pilot only: seed3 shuffled OASIS ranks0 (0333←0115) and4
(0084←0108), two independently identity-initialized configurations per case.
No first-ten expansion, lambda/LR search, added affine or further increments.

## Arms

- Two increments: optimize8a,8b,4a,4b,2a,2b,1a,1b. Every new map starts
  with zero latent controls and fresh Adam. Freeze all earlier map parameters;
  keep coordinate gradients through their sampling operations. Retain8 maps.
- Continuation control: one map per cps8/4/2/1, with two optimization blocks
  per map. Block b starts from block a's selected latent, with fresh Adam,
  initial LR and stopping history. Freeze only earlier pyramid-level maps.
  Replace the current map; never compose its previous and continued states.
  Retain4 maps while preserving all8 block checkpoints for audit.

Both retain Phi_previous o delta_new pull composition, native B-spline samples
with border-trilinear displacement interpolation, and border image sampling
from the original moving-image pyramid. No repeated image prewarp, accumulated
field collapse, knot insertion or clipping. Each elementary map uses the same
endpoint-aware0.99 bound and normalized latent c=a*tanh(p/a); no cap doubling.

## Frozen settings, per block

| CPS | Downsample | Lambda | Normalized LR | Max/min epochs | Requested patience |
|---|---:|---:|---:|---:|---:|
|8|4|1|0.03|400/40|20|
|4|2|1|0.01|800/80|35|
|2|1|5|0.003|1300/130|50|
|1|1|1|0.001|2000/300|100|

Same raw-intensity squared local NCC9 plus unmasked total-composed-field
forward-valid component/axis-mean L2 diffusion in current evaluation-grid
voxel units. Lambda is NOT divided between blocks. This explicitly overrides
the repository analytic-regularizer default. No loss normalization, label loss
or reference supervision. Lowest total-loss checkpoint includes initialization.
Median/check25, relative total/data1e-4/5e-5, effective patience>=25; each cps1
block allows2 LR halvings at least100 epochs apart. Other levels allow none.
Seed1, FP32, cuDNN benchmarkFalse/deterministicTrue, TF32off.

Two blocks match allowed schedules, NOT actual adaptive epoch counts, FLOPs
or runtime. Separate pyramid index and global block index prevent assigning
cps1 schedule to the fourth block, which is actually cps4b.

## Execution and verification

Separate core/runner/auditor, immutable historical sources and outputs. Use
1009:1020, one new worker per GPU0/1/2/3; assign rank0 two-map/control to0/1,
rank4 two-map/control to2/3. Existing jobs unchanged. Recheck memory before
each worker, requiring8GiB or smoke allocated peak+2GiB. Preserve failures and
stop for audit; let already-running workers finish, no automatic retries.

Two rank0 full-size one-step-per-block smokes (one per arm), sequentialGPU3,
gate four concurrent production runs. Unit/regression tests cover numerical
legacy parity, initialized Adam continuation, map replacement versus append,
indexing/names/CLI, identity objective continuity, gradients and serialization.

Every block checks input/source hashes, selected checkpoint/control/field
reconstruction, finite native160x224x192x3 serialization, canonical labels,
and full native-cell-corner elementary contraction. At each second block,
initial field/loss must equal the preceding selected state and selected loss
must not worsen. The first cps8 block must match the historical all-bounded
reference within1e-6 loss and1e-4 max/1e-5 RMS voxel tolerance.

Same100k seed1 off-grid fixed-raw-positive points per case measure retained
chain and final exported interpolant independently. No gate forces final
finite-difference/export folding to zero. Separate processes replay all32
block checkpoints, loss terms, per-label/accuracy/regularity metrics, corner
audits and final off-grid probes before report acceptance.

## Artifacts and report

External ARTIFACT_ROOT runs/repo_experiments key:
oasis_seed3_rank000_004_mrdbscp8421_tanh_two_blocks_lambda1151_v1.
Require4 completed fits and4 independent audits before CSV/JSON/Markdown
acceptance. Compare each new arm with the existing one-block all-bounded,
selective last1/2/3, VFA and SITReg on the same cases. Five canonical metrics:
Dice, HD95mm, symmetric ASDmm, fixed-raw-positive positive-J SDlogJ and
J<=0 foreground folding%. Save second-block metric/loss changes, paired
two-map-minus-control differences, bounds/saturation, maps/epochs/runtime/GPU,
source revisions and hashes. Two exploratory cases; no population-SD or
generalization claim. Accuracy improvements are an outcome, not acceptance.
