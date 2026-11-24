---
doc_type: implementation_index
status: active_draft
---

# Implementation Notes

This folder is the code-derived technical hub for `inrorspline`. It records how
the current code implements the method families, training objectives,
regularity metrics, and experiment-specific adaptations that matter for paper
writing.

Use these notes when converting code behavior into paper Methods text, figure
captions, protocol caveats, and implementation claims. Keep scientific protocol
definitions in `docs/03_methods.md`; keep run ledgers in
`docs/04_experiment_registry.md`; keep experiment-specific audits and plots in
`experiments/` or `docs/reports/`.

## Files

- [LungCT HU/crop/lambda ablation](lungct_hu_crop_lambda.md): unchanged MR solver,
  common intensity scale, physical mask crops, independent bounded searches and
  provisional anatomy evaluation.

- [Three coronal grids](lungct_grid_compare.md): analytic and raster pull grids,
  projected mapped mesh, common voxel-grid phase and complete GPU cache lifecycle.

- [DIR-Lab case viewer](lungct_case_viewer.md): physical LPS display, original
  field adapters, inverse landmark locations, compact case bundles and the
  request-free browser pipeline.

- [Corrected 1 mm LungCT and CorrField](lungct_corrected_1mm_corrfield.md):
  versioned crop contracts, unchanged MR solver, raw-HU CorrField adapter,
  native TRE replay and ordered historical/new casewise comparisons.

- [Diffusion plus analytic KL-C1](oasis_diffusion_plus_kl_c1.md): fixed
  diffusion lambda1, rank0 gamma pilot and separately authorized first10
  extension, exact gamma0 compatibility/reference reuse, loss components,
  paired geometry and separate resource reporting.

- [Jacobian SD(logJ), folding and NDV explained](jacobian_sdlogj_folding_and_ndv.md):
  canonical versus spline-analytic derivatives, positive-J reductions, folding
  counts versus negative-volume measures, exact code conventions and limitations.

- [OASIS FD/analytic regularizer sweep](oasis_regularizer_fd_analytic_sweep.md):
  matched NCC9 diffusion/BE/KL-C1 arms, continuous pyramid derivatives, six-arm
  bracketing and per-setting cold/warm time and memory measurements.

- [GaussianDIR paper and code audit](gaussiandir_paper_code_audit.md): pinned
  source, paper-revision differences, Gaussian blending, released defaults,
  verified discrepancies, corrected two-scale adapter, SITReg geometry/metrics
  and selected rank-0 fitting/registration protocols.

- [OASIS rank0 parameter scaling](oasis_fieldfit_parameter_scaling.md): full-box
  direct-DVF reconstruction, native versus joint multiresolution, exact scalable
  operators and independent20-minute endpoint budgets.

- [OASIS adaptive local-affine field fitting](oasis_polyaffine_fieldfit.md):
  physical-coordinate dictionaries, tensor-separable native rendering, matched
  D0/SVF capacity matrices, bounded-DVF diagnostics and GPU0 failure-tolerant queue.
- [Matched-prefix cps8421 extension](bspline8421_matched_prefix.md): retained
  accepted842 prefixes, newcps1 optimization, scoped roundoff proof and paired-depth reports.
- [First100 three-family B-spline842 extension](bspline842_first100.md):
  exact-protocol reuse, immutable numerical adapters, eight-slot fault isolation,
  complete-cohort accuracy/regularity and resource reporting.
- [Symmetric SVF Phase B](symmetric_bspline_svf_phase_b.md): native-mm amplitude
  and global RMS-balance ablation, full-objective histories and matched-GPU checks.
- [Symmetric B-spline SVF Phase A](symmetric_bspline_svf.md): two independently
  refined total velocities, two-gradient midpoint NCC, explicit equal-split
  diffusion/LR scaling, inverse construction and branch-energy diagnostics.
- [Bounded two-map lambda sweep](bounded_two_map_lambda_sweep.md): explicit
  global stage weights, retained six-map composition, unchanged bound capacity,
  replay/reproduction semantics and memory/runtime reporting.
- [B-spline SVFs](bspline_svf.md): native scaling/squaring, refined versus
  composed velocities, velocity/displacement diffusion and numerical audits.
- [First10 refined-SVF sweep](bspline_svf_sweep.md): global stage-lambda sweep,
  complete-cohort reporting, independent replay and process time/memory semantics.

- [NCC loss definitions and normalization](ncc_loss_normalization.md):
  current local squared, global signed, and global squared formulas; compatible
  population moments but unequal loss-to-diffusion balance at equal lambda;
  planned FeDDA squared-GNCC follow-up.
- `code_sources.md`: map from implementation claims to repo-local source files
  and experiment records.
- `idir_sinr_architecture_training.md`: IDIR and SINR architecture, training
  losses, and OASIS/LungCT protocol differences.
- `idir_fullres_ncc9_diffusion.md`: cancelled full-voxel SIREN attempt retained
  for audit; replacement must preserve old10,000-point IDIR sampling.
- [Sampled IDIR diffusion-first replacement](../../experiments/2026-10-10_oasis_idir_sampled_diffusion_staged/codex_execution_spec.md):
  original10k sampling, signed global1-NCC, analytic voxel diffusion, adaptive
  lambda bracketing atomega32 followed byomega at a frozenlambda.
- [Fourier-input sampled IDIR](../../experiments/2026-10-10_oasis_idir_fourier_diffusion/codex_execution_spec.md):
  frozen Gaussian input features, explicit frequency convention, original-XYZ
  analytic diffusion, native initialization audit and parameter-count controls.
- `directcp_bspline_optimization.md`: DirectCP optimization, cubic B-spline
  formulas, analytic derivatives, finite-difference comparisons, and
  multistage knot insertion.
- `mr_d_bscp_fast_842.md`: comprehensive code-derived specification of the
  OASIS MR-D-BSCP fast 8-4-2 schedule, implementation conventions, timing
  semantics, and acceleration-audit invariants.
- `vfa_feature_dense_refinement.md`: frozen full-resolution VFA projections,
  dense-voxel refinement, diffusion units, and exploratory selection semantics.
- `dl_ncc9_diffusion_sweep.md`: DL-matched squared local NCC9, voxel-unit
  diffusion with normalized parameters, aligned pyramid geometry and six-way
  spline/dense comparison.
- `paper_claims_crosswalk.md`: paper claims mapped to code paths, experiment
  evidence, metrics, and caveats.
- [VFA and SITReg lambda1 training audit](../../experiments/2026-10-09_oasis_dl_training_lambda_audit/codex_execution_spec.md):
  native NCC9/diffusion, model-specific directional Adam counts, persistent
  data iteration, checkpoint-selection conventions and approval-gated DL sweep.
- [Four-GPU SITReg training adapter](../../experiments/2026-10-09_oasis_sitreg_training_lambda_sweep/codex_execution_spec.md):
  fresh DL lambda runs, numerical reference-reproduction gates, native joint
  updates, validation-selected checkpoints and first10 posthoc metrics.
- [SITReg detailed first100 evaluator](../../experiments/2026-10-09_oasis_sitreg_training_lambda_sweep/first100_evaluation_spec.md):
  non-invasive post-training queue, byte-preserved first10 fields, all35
  per-label metrics with explicit absent-label handling and canonical regularity.
- [Native VFA training and automatic first100 evaluation](../../experiments/2026-10-09_oasis_vfa_training_lambda_sweep/codex_execution_spec.md):
  sequential directional Adam updates, forward validation, per-GPU SITReg
  dependency queue and one explicitly approved GPU0-to-GPU3 OOM fallback.
- [Native FeDDA-1MF training sweep](../../experiments/2026-10-10_oasis_fedda_training_lambda_sweep/codex_execution_spec.md):
  original200-epoch sequential-directional protocol, GPU0 memory preflight,
  VFA-dependent queue and automatic detailed first100 comparison.

## Review Safety

This folder is intended to be safe for anonymous review. Do not add raw data,
checkpoints, dense fields, private host paths, credentials, private usernames,
private hostnames, W&B entities, or identity-breaking metadata.

Use repo-relative paths, dataset handles, protocol tags, and environment
variables when a path or artifact location is needed.
## Native full-resolution loss sweep

See [full-resolution DL-loss dense sweep](fullres_dl_loss_sweep.md) for P44's
MSE/MIND/RC/NGF/Feature-MSE protocol and its distinction from VFA-feature matching.

P55 adds [signed/squared global NCC](../../experiments/2026-10-11_oasis_gncc_dense_sweep/codex_execution_spec.md)
using consistent population moments and ordinary autograd. It retains P43/P44
geometry, diffusion and Adam, with no initial-loss normalization. IDIR is deferred.

P56 uses [signed global NCC for native SITReg training](../../experiments/2026-10-11_oasis_sitreg_gncc_training/codex_execution_spec.md),
with a stateless per-image loss and immutable historical training/evaluation
loops bound to a separate process-local adapter. Joint bidirectional updates,
voxel diffusion, initialization, pair order and validation selection are retained.

P57 uses [the same signed GNCC for original FeDDA-1MF](../../experiments/2026-10-12_oasis_fedda_gncc_training/codex_execution_spec.md),
but retains FeDDA's two sequential directional Adam updates per pair and
forward-only validation,not SITReg's jointly averaged update. Historical LNCC
sources and results remain immutable; the new comparison uses full100-case
cohorts and ARC mean curves above sample-SD ellipses.
