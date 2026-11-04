# P76: First10 bounded two-map 842 lambda sweep

## Approved scope

Seventy fresh registrations, seed3 shuffled ranks0–9, seven global lambdas
0.01,0.1,0.3,1,3,10,100. Every fit starts from identity. Optimize bounded
increments8a,8b,4a,4b,2a,2b, retaining all six maps. Both maps at every level
use the same lambda; no division between maps. Rerunlambda1 and preserve P70
lambda1 as a separate reproducibility comparison. No other settings/schedules.

## Frozen numerics

Keep P70 native endpoint-aligned cubic expansion,0.99 geometry-validated
cap*tanh(p/cap) controls, unit derivative at zero, normalized XYZ latents,
retained pull composition Phi_previous o delta_new, frozen preceding maps
with coordinate gradients, border-trilinear displacement/image sampling.
No knot insertion, collapse, repeated image prewarp or clipping.

Loss: raw-intensity1-mean(local NCC9 correlation squared), plus lambda times
unmasked forward-valid total-composed-displacement component/axis-mean
diffusion in current evaluation-grid voxel units. This explicitly overrides
the general analytic-spline-regularizer default. No similarity normalization,
mask, labels or target-field supervision in the training objective.

| CPS | Loss downsample | Normalized LR | Max/min updates per map | Requested patience |
|---:|---:|---:|---:|---:|
|8|4|0.03|400/40|20|
|4|2|0.01|800/80|35|
|2|1|0.003|1300/130|50|

Fresh Adam per map; seed1, FP32, deterministic cuDNN, benchmark/TF32off.
Median/check25, relative total/data thresholds1e-4/5e-5, effective patience
at least25. No LR reductions. Select lowest total loss including initialization.
Do not select by accuracy. CUDA backward nondeterminism remains documented.

## Execution and gates

Isolated runner/auditor/queue/report; historical numerical core remains
unchanged. Source/input/checkpoint hashes and exact bounds in provenance.
Three full-size rank0 smokes atlambda0.01/1/100, one Adam update per each of
six maps, serialization/canonical metrics and independent replay before
production. Unit-test CLI/names/matrix, six-map stage indices, NCC/diffusion
value/gradient parity atlambda1, lambda scaling and bounded controls.

Use container1009:1020, GPUs0–3, at most two new workers per GPU/eight total.
Fresh memory checks per admission:8GiB or maximum smoke allocated/reserved
peak+2GiB, whichever is larger. Conservative active-worker reservations and
host RAM/disk preflight. Do not change existing jobs. Record optimizer, worker
and independent-audit time separately; allocated/reserved peaks and GPU state.

Durable queue with exclusive controller lock, resumable terminal state and
preserved attempts/logs. Individual failures deferred; unrelated jobs continue.
One deferred CUDA OOM retry maximum after ordinary work drains. Shared storage,
input/source drift or inadequate resource telemetry pauses admission; never
pretend the queue is complete. No automatic numerical repair or relaxed gates.

Seventy successful fits and independent replays required for acceptance:
correct identities/direction/shape/units, finite native160x224x192x3 DVFs,
six bounded maps, checkpoint/history/loss consistency, canonical absent-label
handling, elementary contraction and100k seed1 off-grid foreground probes.
Measure exported folding even when nonzero. Freshlambda1/P70 differences are
diagnostic, not a bitwise cross-run acceptance condition.

## Reporting and figures

CSV/JSON/Markdown: seven complete ten-case groups, per-case/per-label results,
mean±sample SD Dice/HD95/symmetric ASD, fixed raw-positive positive-J SDlogJ,
foreground folding percent. Paired differences, per-map stopping/best updates,
saturation, map2 objective gains, optimizer/worker/audit time, GPU memory.
Compare saved SITReg/displacement/velocity-SVF results on identical pairs.

Update C/F bounded-two point to a seven-point black dotted/diamond curve;
C's unbounded/bounded-one remainlambda1 points. G remains four-levellambda1,
not this three-level sweep. Preserve tighter means-only F/G y axes and full
SD bottom ranges, widening shared limits only if needed to retain all data.
Produce versioned PNG/PDF/SVG and updated A–G PDF. Preserve prior artifacts.
No claim of zero folding or unbiased generalization: exploratory first10.

External runs/reports key:
ARTIFACT_ROOT/{runs,repo_experiments}/oasis_seed3_first10_bspline_bounded2_842_ncc9_lambda_sweep_v1.
