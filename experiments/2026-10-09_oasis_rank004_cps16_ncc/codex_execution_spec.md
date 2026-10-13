# P47: rank4 cps16 NCC9/diffusion rescue

User approved the audited two-run plan, then explicitly required GPU3 only,
sequential execution. Existing GPU jobs and historical artifacts are immutable.
Artifact key: `oasis_seed3_rank004_cps16842_dense_ncc9_lambda1_v1`.

## Scope and limitations

Exploratory tuning of one difficult test pair, not a first10 or held-out claim.
Adding cps16 adds compute. Pyramid changes NCC physical support and diffusion
sampling; full mode isolates the added control stage but is not equal-compute.
Timing is diagnostic with existing workloads. CUDA sampling backward can remain
nondeterministic. No improvement is guaranteed. No label-based epoch selection.

## Frozen protocol

Seed3 shuffled rank4: fixed0084/moving0108, original sorted index47;
moving-to-fixed registration by fixed-to-moving pull sampling. Raw stored
float32 intensities, LIA-to-RAS, native160x224x192 at1mm, border image sampling;
canonical zero-padded nearest-neighbor segmentation evaluation.

Two independent zero-start registrations, pyramid first, full second:

| Stage | CPS | Pyramid factor | Full factor | Normalized LR | Max/min epochs | Effective patience |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 16 | 8 | 1 | .03 | 400/40 | 25 |
| 1 | 8 | 4 | 1 | .03 | 400/40 | 25 |
| 2 | 4 | 2 | 1 | .01 | 800/80 | 35 |
| 3 | 2 | 1 | 1 | .003 | 1300/130 | 50 |
| 4 | dense, no CPS | 1 | 1 | .001 | 2000/300 | 100 |

Loss at every stage is native squared local NCC_vfa9 (1-mean squared
correlation) plus lambda1 times unmasked forward-valid total-field diffusion.
Use evaluation-grid voxel units inside autograd, component and axis means;
all Adam parameters stay normalized XYZ. No loss normalization, lambda
rescaling, bending energy, feature loss or label supervision.

Reuse unchanged P43 StageObjective/AlignedBSpline/optimize_stage: fresh Adam
betas.9/.999, eps1e-8, no weight decay; checks/window25, relative total/data
1e-4/5e-5. Effective patience is max(requested,window). Dense alone permits
two LR halvings, at least100 updates apart, then stops on next plateau.
Preserve best/last states and history; initialization is eligible as best.
Exact dyadic16-to8-to4-to2 knot insertion; dense is an exact detached clone of
native cps2 expansion. No cps1 controls or double warp. Seed1 FP32,
cuDNN benchmarkFalse/deterministicTrue, TF32off.

## Audit resolutions

- Existing latest full-loss baseline has no image downsampling; compare both
  full and pyramid16-start paths against saved matched8-start references.
- CPS16 is16mm control spacing, not16x image downsampling. Primary first image
  stage is20x28x24 (factor8), Gaussian sigma4/radius12 antialiasing.
- New controls14x18x16 prolong into existing24x32x28,44x60x52,84x116x100 grids.
- Historical final-refinement index3 must not attach dense rules to new cps2.
  Explicit stage objects place dense at index4.
- Leave old numerical modules/runners unchanged. New names contain complete
  CPS/resolution schedules and settings digest; no old optimized cache reuse.
- Preserve endpoint-aligned continuous spline evaluation in pyramid mode;
  never use the historical integer domain-CPS expansion shortcut.

## Verification and execution

New CPU contracts plus existing P43 tests precede sequential GPU3 one-update-
per-stage smokes for both modes. Test16-to8 prolongation on full/coarse grids,
values/gradients, affine diffusion units, zero field, CLI, names, legacy
schedule preservation, incomplete-report rejection and native loss parity.
Only after both smokes succeed, run production pyramid then full onGPU3.

Container UID:GID1009:1020. PhysicalGPU3 appears as cuda:0. Query current
memory/utilization before each worker; require8GiB or measured smoke peak
plus4GiB free, whichever is greater. Existing processes remain untouched.
Preserve failures/logs and stop for review; no automatic retry or protocol
change. Strictly one experiment worker at a time.

Source hashes must match immutable P43 numerical files. Verify input/native-
source/split hashes, CUDA backend, all five finite native stage fields,
model/DVF hashes and completion markers. Knot insertion tolerance:max1e-4
and RMS1e-5 native voxels; dense clone exactly zero difference. Require all35
valid labels and per-label means equal canonical summaries.

## Artifacts and report

Source/spec/tests stay in Git. Generated material stays under
`${ARTIFACT_ROOT}/runs/<key>` and `${ARTIFACT_ROOT}/repo_experiments/<key>`.
Record stage histories, best/last checkpoints, native stage fields, all35
label metrics, source/input hashes, device snapshots and stopping data.

CSV/JSON/Markdown compare both new fits with saved full/pyramid842-dense
lambda1 and original VFA/SITReg:Dice/HD95/symmetricASD, foreground positive-J
SDlogJ, nonpositive-J percentage, diffusion, paired deltas and stage diagnostics.
Focus interpretation on ventricles, caudates and choroid plexus while retaining
every label. Aggregate only after both runs succeed and sources/references
remain unchanged. No single-case SD.
