# P83: GaussianDIR fitting and registration specification

Status: **four v1 endpoints completed**. On 2026-11-06 the user selected both
the released control and corrected paper reconstruction, the explicit frozen
coarse/fine scheme with literal paper caps, and full-box MSE fitting with fixed
populations. [Protocol v1](protocol_v1.json) is the concrete pilot configuration.
The numerical, data and native-resource gates passed before production. All four
endpoints reached2,000 steps without timeout/failure; see the technical log.

## Known limitations

- Only one repeatedly explored OASIS pair; no held-out or population claim.
- Released default is single-scale isotropic/translation; the paper describes
  anisotropic local rigid maps, adaptive density and additive coarse/fine fields.
- The optional upstream rigid branch has confirmed forward/gradient defects;
  the local paper model corrects them. Native validation uses an isolated
  PyTorch 2.4.1/CUDA11.8/PyTorch3D0.7.8 environment; the base remains unchanged.
- The supervised target is a learned method's estimate, not anatomical truth.
- Nonlinear geometry optimization does not certify optimal representation error.
- Paper/code TV, density statistics/caps and metrics differ. Use explicit tags.

Full evidence: [technical report](../../docs/implementation/gaussiandir_paper_code_audit.md).

## Scientific questions and controls

A. How accurately can the chosen GaussianDIR representation reconstruct the
existing SITReg endpoint under a stated optimization/capacity budget? Measure
both field fidelity and preservation of the source registration's accuracy and
regularity. Field-fit error is the primary endpoint for this arm.

B. Starting from identity and using images only, what accuracy/regularity tradeoff
does the same representation achieve relative to the saved SITReg endpoint?
Measure the final registration in the shared pipeline. Do not initialize from
the fitting arm or optimize toward SITReg in this arm.

Reference rows: identity and original saved SITReg. Candidate rows: fitted
GaussianDIR and independently registered GaussianDIR, with full method/protocol
tags. Add a corrected paper variant only as an explicitly separate model.
Do not relabel existing P81/P82 Gaussian dictionaries as GaussianDIR.

## Frozen input identity

- Dataset: `brain`, canonical seed-3 shuffled `pair_test_200`, rank 0.
- Split: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Split SHA-256: `40ff80b23407c37ed8a20092f018fd9588e5df3eb95628069ec42da698aac3a0`.
- Fixed/moving: `0333` / `0115`; reoriented native DHW `160×224×192`.
- Pullback map: fixed coordinate to moving sampling coordinate.
- Target: `rank000_sitreg` in the P82 manifest at
  `${ARTIFACT_ROOT}/runs/oasis_rank000_fieldfit_parameter_scaling_v1/execution_20261106T0644Z/manifest.json`.
- Original field SHA-256:
  `caf5be5c136615b5d833a7c1d44edd8d98de9475847836388b5f014dfad2f878`.
- Canonical physical-RAS-mm field SHA-256:
  `408ce279ca76bddf2bced1b438f2f5d559495fa0684f46390abd06821c9c90c1`.
- Reference source regularization setting: historical lambda 1. This does not
  set the new fit loss or GaussianDIR TV lambda.
- Both field hashes were verified during the audit; images, labels, masks and
  reference metric replay still require a fresh run-specific preflight.

Resolve private paths through the manifest and ignored runtime configuration.
Historical inputs and outputs are immutable; never reuse their run directories
for GaussianDIR artifacts.

## Frozen decisions

| ID | Proposed default | Other scientifically meaningful choice | Status |
| --- | --- | --- | --- |
| D1 model | Released control plus corrected paper reconstruction | Released model only | User selected both |
| D2 supervised domain | Full native box, physical-mm vector MSE | Fixed-image-positive objective matching P81 instead of P82 | User selected full box |
| D3 supervised density | Fixed node population; train geometry and local motion | Adaptive population with calibrated MSE-gradient threshold | User selected fixed population |
| D4 registration loss | Variant-specific signed NCC/TV below; lambda8 each | Secondary 2/4/8/16/32 tradeoff sweep | Frozen anchor; sweep deferred |
| D5 capacity | Control s8; paper s8→s4; K20; actual counts | K10/40 sensitivity, one factor at a time | Frozen anchor |
| D6 fit optimizer | Adam from identity, released group LRs; stability checked by native smoke | LR changes only under a newly recorded protocol | Frozen v1 starting rates |
| D7 fit budget | 2,000 total steps, 1,200-second active limit; stop at whichever comes first | Larger convergence study later | Frozen pilot, no optimum claim |
| D8 code corrections | Separate adapter; control preserves optimizer ordering, corrected model updates before adaptation | Additional matched-component ablations later | Enumerated below |

D1 is the principal scientific choice. A full-paper implementation requires
correcting spatial versus covariance coordinates and detach behavior, selecting
position-gradient density control, and specifying the missing coarse/fine
transition semantics. Do not describe “all three toggles enabled” as a complete
paper reproduction. The written `/8` and `/4` caps also need interpretation;
they are not the released `/s^3` cap and can be much larger.

D3 fixed population does **not** freeze centers or covariance widths. It avoids
silently applying an NCC-tuned threshold to a differently scaled supervised
objective. It must be labeled `fixed_population_fieldfit`, not native adaptive
GaussianDIR. An adaptive fit can follow as an additional arm with its threshold
calibration, actual population history and matching time budget recorded.

## Supervised arm

Render direct GaussianDIR displacement at native voxel centers; convert to
physical endpoints using the verified affines. Optimize

```text
L_fit = mean_x ||u_pred_mm(x) - u_SITReg_mm(x)||_2^2
```

This is vector MSE (sum of three squared components, then mean over voxels),
with mm² units. It differs by a factor of three from elementwise tensor MSE;
freeze the convention before setting LR or adaptive thresholds. No TV,
diffusion, BE, label loss, image loss, global affine prefit or SVF is included
in the primary fit. The target's regularization does not become a fitting term.

For sampled Adam, use uniform full-box samples and 20,000 query centers without
the registration TV-neighbor grouping. The sampled loss estimates the full-box
objective. Evaluate checkpoints with a complete chunked full-box pass. Start
at identity; record loss, gradient norms, scale ranges and node counts. A
time-limited endpoint is an achieved fit, not a capacity optimum. Its LR horizon
must be specified before launch; do not continue a decayed 2,000-step schedule
silently or borrow work from another endpoint's budget.

Sampled error and full-box error are separate fields in the report. Foreground
and complement summaries expose extrapolation behavior. Fit, export and metric
time have separate timers, with the budget's charged operations explicitly listed.

## Unsupervised arm

Use images at native resolution; no downsampling for the released-code anchor.
Use raw fixed `image>0` to select centers and initial nodes. Use 5,000 centers
and their 3 positive-axis neighbors, for nominal 20,000 queries, as in the
release. Neighbors may leave the mask or volume; preserving zero padding and
this behavior must be explicit. A boundary-filtered sampler is a separate variant.

Optimize released signed sampled NCC plus released direction-averaged TV on
normalized displacements. Start with lambda 8, 2,000 steps, 50-step warmup,
K=20, sparsification=8, seed=6 and the parameter-group learning rates in the
technical report. Code-native adaptive thresholds/schedule are the starting
control; any optimizer-order fix is named in the method ID.

Labels never influence optimization, early stopping or checkpoint selection.
Use the predetermined final step for the main registration point. A subsequent
lambda sweep retains every point and presents an exploratory rank-0 curve;
do not call the best test-case point a validated optimum. Different native
losses from SITReg are allowed for a method comparison but prevent attributing
the difference to representation alone.

Evaluate the continuous model across the **whole** native box, even when the
training centers came from foreground. Export the true model field, not a
zero-filled masked field. Report nonfinite predictions as failure; do not hide
them by zeroing or clipping. Keep any stabilization policy in the manifest.

## Common accuracy and regularity evaluation

Use identical geometry, interpolation and evaluation code for identity, SITReg,
the supervised fit and the unsupervised result. Warp intensities trilinearly and
labels with nearest neighbor. Match the existing canonical image padding; record
it in every row. The technical report gives the exact align-corners conversion.

- All 35 labels: Dice, HD95 mm, symmetric and both directional ASD mm; per-label
  values, eligibility counts and missing/empty-warp reasons alongside macro means.
- Image NCC, MAE and MSE over full box and raw fixed foreground.
- Folding `J<=0` percentage, positive-J SD(log J) and valid-positive fraction,
  J mean/min/quantiles; full box and fixed foreground on shared valid stencils.
- Full/foreground diffusion, finite-difference BE, NDV and NDV fraction, using
  the same canonical metric rows/functions for both exported dense fields.
- Fit arm: full/foreground/complement vector RMSE and mean/p50/p95/max endpoint
  error in mm, Jacobian discrepancy and source-warp preservation.
- Initial/final/peak Gaussian counts and all trainable scalar counts; elapsed
  active/evaluation/export time, GPU memory and stopping status. V1 does not
  separately instrument exact per-endpoint end-to-end worker wall time.

Reference metric functions are named in the technical report. The new runner
must freeze their source hashes and serialize units, domains, border crops,
positive-J treatment, reduction conventions and exclusions. Recompute SITReg
with this exact evaluator rather than copying a published number. Compare
historical fields/metrics only after source replay passes.

## Implementation and verification sequence

1. Resolve D1–D8 and write a frozen configuration with method IDs. Update this
   spec/registry before real runs; v1 choices are now frozen.
2. Prepare a compatible pinned dependency environment without replacing the
   active container's packages while historical jobs run. Inspect current
   runtime settings and use the configured artifact-writing UID/GID.
3. Build a dedicated adapter under `scripts/experiments/gaussiandir/`. It must
   respect external outputs, use canonical pair loading, retain model/optimizer/
   RNG/adaptation state, and expose effective counts and loss components.
4. Native tests: PyTorch3D CPU/GPU KNN parity away from ties, stable-weight
   parity on ordinary inputs, constant translation/rotation, covariance rotation,
   finite-difference parameter gradients, identity/voxel/mm/corner conversions,
   clone/prune state and population guards, checkpoint/restart replay.
5. Data preflight: original/canonical target and image hashes; shape/affines;
   identity and saved SITReg image/label warp parity; whole-field export parity;
   shared label and regularity baseline replay. No fitting needed for these.
6. Short native-resource smoke for each selected arm, including an actual
   adaptation event if enabled. Check finite losses/gradients, full-box export,
   actual memory, synchronized timing and all metric outputs. Choose a free GPU
   after inspecting occupancy; do not stop or reuse an existing run.
7. Run the frozen supervised arm, review reconstruction/convergence, then run
   the independent unsupervised anchor. Extend to the declared lambda/capacity
   sensitivities only after the anchor results are interpretable.
8. Produce external CSV/JSON, field/error/Jacobian slices, accuracy–regularity
   plots, generated report and an executed review notebook. Keep only its
   output-free source in the experiment directory. Re-evaluate serialized fields
   independently before recording acceptance.

Runtime backend: requested/effective cuDNN benchmark false, deterministic true;
proposed TF32 false. Record actual behavior, dependency versions, device, image
digest, root commit/dirty source hashes, upstream commit and adapter hashes.
Native gates in steps4–6 passed; receipts and limitations are in `technical_log.md`.

## Selected v1 model and numerical details

There are four independent endpoints: two representations × supervised/image-only
objectives. Each starts from identity and optimization seed6. No source field,
optimizer work or learned parameters pass from fitting to registration.

Released control: one 2,000-step stage, s8, K20, scalar radii and translations.
Use the upstream class and upstream NCC custom backward. Retain translation
gradient density statistics, adaptation at300…1200 every100steps, `/8^3` cap,
and the upstream adaptation-before-Adam behavior (including dropped current
gradients when parameters are replaced). One arithmetic adaptation uses log-space
weights while retaining the determinant epsilon1e-7; numerical parity is tested.
Full-box export, canonical masks, isolated paths and invalid-population/nonfinite
fail-fast checks are explicit protocol adaptations.

Paper reconstruction: anisotropic scales and covariance quaternions, plus local
rigid motion quaternions/translations; correct spatial motion delta and full
center gradients. Optimize coarse s8 for1,000steps, freeze it, add zero-motion
fine s4 evaluated at the original coordinates, and optimize fine for1,000steps.
The result is the sum of fields, with no composition or image pyramid. Stage
LRs each restart the released parameter-group rates with50-step warmup and a
1,000-step cosine horizon. The sampler continues across the transition.

The user selected literal total caps `floor(domain_voxels/8)` then
`floor(domain_voxels/4)`; fine capacity subtracts retained coarse nodes. The
domain is the full box in fitting and raw fixed foreground in registration.
Fixed-population fitting never clones/prunes; these caps are upper bounds only.
Registration uses accumulated mean position-gradient norm, thresholds0.002 and
1e-7, every100 local steps from100 through900 at each stage. New Adam moments
are zero; the current step count is retained. Clones use the supplement's
axis-aligned normal offsets with current scale components. Optimizer updates
precede parameter replacement, preserving the completed step. Invalid populations
(fewer thanK survivors) and nonfinite states fail explicitly; no silent rescue.

The corrected weights use the exact determinant factor without the release's
epsilon, evaluated in log space. Registration uses signed Pearson NCC with
population moments and1e-10 denominator epsilon, and mean-over-centers,
sum-over-three-directions TV per the supplement; lambda8 is the journal anchor.
The control uses direction-mean TV and the upstream NCC. Thus this first pilot
compares complete method variants, not isolated representation effects or
regularization-matched operating points. Subsequent objective-matched or lambda
sensitivity runs must be identified separately.

Every endpoint has a1,200-second active watchdog, including initialization,
optimization, adaptation, telemetry and checkpoint writes. The fixed iteration
schedule is not extended after it decays. If the time limit prevents completion,
report `time_limit`, achieved stage/count and overshoot; do not call that a
completed2,000-step result. Final full-box rendering, replay, metrics and export
are timed separately. No lambda/capacity/seed sweep is part of these four anchors.

## Artifact and publication layout

```text
${ARTIFACT_ROOT}/runs/oasis_rank000_gaussiandir_v1/<execution-id>/
  manifest.json, effective_config.json, input_receipts.json
  supervised/<method-id>/
  unsupervised/<method-id>/<lambda-id>/
  reference/
${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_v1/<execution-id>/
  results.json, results.csv, per_label.csv, figures/, report.md
  review_executed.ipynb
```

Choose a fresh execution ID; refuse output overwrite. Generated data never go
under `external/`, `experiments/` or `docs/`. Do not push or commit unrelated
worktree changes. No new results-summary claim is justified at this stage.
## Approved convergence/resource extension

The user-approved [convergence specification](convergence_spec.md) and
`protocol_convergence_v1.json` define the active follow-up. They preserve the
original pilot above, use sampled signed NCC/TV lambda8 first, and defer the
LNCC/diffusion prototype. Registration execution and generated review artifacts
use the separate `oasis_rank000_gaussiandir_convergence_v1` external key.
