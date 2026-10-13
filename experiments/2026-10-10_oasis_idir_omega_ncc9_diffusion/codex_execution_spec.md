# IDIR frequency sweep — cancelled full-volume implementation and sampled replan

## Current status: stopped; replacement NOT authorized to launch

Superseding authorization:the user subsequently approved sampled global1-NCC
with analytic diffusion, two workers perGPU1/3, then changed the search to
diffusion-first atomega32 followed byomega at frozenlambda. The active
replacement is documented separately in
`experiments/2026-10-10_oasis_idir_sampled_diffusion_staged/`. This full-volume
implementation remains cancelled; do not resume it.

On2026-10-10 the user clarified that the original IDIR10,000 sampled-coordinate
updates must be retained, changing only the intended loss and omega. The
assistant's conversion to full-volume optimization exceeded that intent.
The full-volume controller and its two workers have been stopped. Its generated
run directory has been cleared into recoverable artifact-store trash. No
cancelled-run checkpoint, smoke field, or result is eligible for the replacement.
Historical experiments and other GPU jobs remain unchanged.

## Corrected sampled proposal — awaiting the loss-definition choice

Keep the historical sampling/optimization workflow:

- Canonical seed3first10 cases, same direction and geometry.
- Random10,000 voxel-center coordinates per update, sampled without replacement
  from the fixed raw-image-positive domain, refreshed every update, as in
  `run_one.py:make_idir_training_coords/train_idir`. Do not change to10,000
  local-window centers or evaluate the full SIREN field during training.
- Same SIREN[3,256,256,256,3], seed1, native omega-dependent initialization,
  no Fourier features or spline prefix. Omegas8/16/32/64/128.
- Same fresh Adam LR1e-4, betas.9/.999, eps1e-8, no weight decay/AMSGrad,
  2500updates, no scheduler or early stopping, terminal2500 primary checkpoint.
- Preserve old pair-minmax image preprocessing and zero-padded image sampling
  unless a separately identified loss-parity change is explicitly chosen.
  Do not silently retain the cancelled worker's raw-intensity/border-sampling
  changes. Audit historical effective CUDA backend before implementation;
  constructor defaults are not proof of the old run's effective backend.
- Lambda keys0.1/1over3/1/3/10 (exact1/3),250fresh fits includingomega32.
  New identity/artifact key must explicitly say sampled and identify the loss.
- GPUs1/3, at most one new worker each. Measure the corrected runner before
  offering a runtime estimate; the cancelled full-volume ETA is inapplicable.

There is a real compatibility choice, not an implementation detail:

1. Closest to the requested old sampling workflow: sampled global signed
   `1-NCC`, with NCC equal to the old IDIR sampled correlation. This is exactly
   `1+old_ncc_loss`, so adding1 alone does not change its optimization gradient.
   Replace BE with sampled analytic first-derivative diffusion. For normalized
   coordinates x/output q and voxel scales s, use
   `R=mean_{sample,component,axis}[(s_component/s_axis)*dq_component/dx_axis]^2`.
   Use derivatives of displacement, not identity-plus-displacement, and keep
   create_graph=True for training. No extra division by10,000 after the mean.
   This preserves10,000 network coordinates per update, but is NOT the native
   full-volume DL NCC9/finite-difference objective. Its training support remains
   fixed foreground, and analytic diffusion is not exactly voxel forward FD.
2. If native VFA local squared NCC9 remains mandatory, sparse intensities cannot
   be treated as a spatial grid.10,000 independent NCC-window centers require
   up to7,290,000 sample-neighbor field evaluations before deduplication.
   A smaller spatial-patch batch, frozen-center local translation, or cached
   neighborhood field changes sampling/gradients or approximates the objective.
   Such a change must be chosen explicitly, with its actual coordinate budget;
   it must not be passed off as the old10,000-point IDIR loop or exact DL parity.

For either approved replacement, evaluate the full dense field after training
for canonical Dice/HD95/symmetric ASD and foreground std(logJ)/folding percent.
Retain all35label records/case, absent-label reasons, terminal weights, field,
warped labels, losses and metadata. Final250fits produce8750label rows; report
complete-cohort mean/sampleSD and read-only VFA/SITReg references.

Before launch: test exact point count/domain and optimizer-step count, same
initialization and sample sequence acrosslambda, coordinate-unit factors,
zero-translation/affine analytic diffusion, loss/gradient contracts, label
serialization and new cache identities. Run a bounded timing/metric smoke only
after approval. No implementation or replacement launch is authorized now.

## Archived full-volume specification below — NOT the active plan

Former execution placement: GPU1 andGPU3 concurrently, one new worker per GPU;
existing jobs/queues and historical outputs unchanged. Container UID:GID1009:1020.

## Frozen scientific protocol

- Canonical OASIS seed3 shuffled ranks0–9; first subject fixed, second moving.
- 5 frequencies8/16/32/64/128 × 5 lambdas0.1/1over3/1/3/10 ×10 cases =250
  fresh registrations. Exact binary float1/3, not0.3 or truncated CLI input.
- Standalone SIREN[3,256,256,256,3],133379 trainable parameters; normalized XYZ
  coordinates and displacement; native omega-dependent initialization, seed1.
  No Fourier encoding, spline initialization, segmentation loss or supervision.
- Raw stored FP32 images with canonical LIA-to-RAS geometry160×224×192.
  Full-volume VFA squared local NCC9(1-correlation), zero-padded local moments,
  FP32 epsilon, moving-image border sampling and align_corners=True.
- Loss=NCC9+lambda*unmasked forward-valid diffusion of the total voxel-unit
  displacement; mean over valid neighbors/components, then mean over3axes.
  No old coordinate-autograd BE, extra batch divisor or initial-loss scaling.
- Fresh Adam LR1e-4, betas.9/.999, eps1e-8, weight_decay0, AMSGradFalse;
  exactly2500 updates, no scheduler/early stopping. Primary output is terminal
  state2500; minimum-total-loss checkpoint including initialization is diagnostic.
- FP32, cuDNN benchmarkFalse/deterministicTrue, both TF32 flagsFalse.
  CUDA grid_sample backward remains potentially nondeterministic.

## Memory-safe full-volume differentiation

Evaluate the whole SIREN field without retaining network graphs, compute the
single full-volume NCC9/diffusion field gradient, then recompute262144-coordinate
chunks and accumulate vector-Jacobian products into network-weight gradients.
One Adam update follows all chunks. The temporary field is not an Adam parameter.
Do not split NCC windows or omit diffusion edges at chunk boundaries. Small-grid
monolithic loss, gradient and Adam-step parity gates the implementation.

## Verification, execution and output

- Unit tests:250unique names, exact thirds, fixed budget, CLI/legacy isolation,
  initialization, component/voxel conversion, objective parity, full support,
  all-frequency two-pass gradients/Adam updates, explicit missing-label records.
- Five rank0 full-volume smokes cover all frequencies and both GPUs; omega32
  uses10updates for warm timing, others1. Native VFA loss/gradient parity,
  checkpoint and DVF roundtrips,35label records and finite metrics must pass.
- Current free memory and utilization/process list recorded per worker; gate
  max(12GiB, measured smoke reserved peak+2GiB). OOM attempts preserved and
  retried once with the identical protocol; no changes to existing GPU jobs.
- Save terminal/best-diagnostic/latest checkpoints, state-indexed loss history,
  normalized XYZ dense DVF, warped segmentation and all35label rows/case.
  Missing fixed/moving/warped labels have explicit reasons and null JSON/blank
  CSV metrics. Present-label numerical failures are fatal, not silently excluded.
- Canonical Dice, HD95mm, pooled-area symmetric ASDmm; foreground positive-J
  natural-log standard deviation and nonpositive-J fraction reported as percent
  using LUMIR crop2 and fixed raw>0 foreground. Diffusion/runtime diagnostic.
- Incremental per-case and per-label CSV/JSON; cohort mean/sampleSD only for
  complete10case groups. Final acceptance needs250runs,8750label records,
  25cohorts, correct identities/hashes and identical initial state withinomega.
  Complete CSV/JSON/Markdown and line-only accuracy-vs-stdlogJ sweep plots,
  with read-only hash-validated original VFA/SITReg first10 references.

Artifact key:`oasis_seed3_first10_idir_siren_omega_ncc9_diffusion_2500_v1` under
`${ARTIFACT_ROOT}/runs/` and `${ARTIFACT_ROOT}/repo_experiments/`. Old omega32
sampled-global-NCC/analytic-BE artifacts are read-only and not reused.

Known limitations: exploratory first10 test tuning; equal2500steps do not
guarantee convergence at every frequency. Omega also changes initialization
and optimization conditioning, not only frequency. Shared-GPU timing is not
a controlled speed comparison. Voxel-grid finite differences do not certify
off-grid invertibility of the continuous SIREN field.
