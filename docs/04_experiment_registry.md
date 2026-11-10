---
doc_type: experiment_registry
status: active
---

# Experiment Registry

### P83: OASIS Rank0 GaussianDIR Paper/Code Audit, Pilot and Convergence Study

- Completed paper-loss follow-up: sampled signed NCC+TV lambda8, fresh200/500/2000
  schedules with three same-seed timing repeats, one startup measurement and one
  joint convergence trajectory per variant. Maximum1200s including monitoring
  and usable outputs. Phase wall/CUDA timing and Torch/NVML memory on one idle
  GPU. Spec: `experiments/2026-11-06_oasis_rank000_gaussiandir/convergence_spec.md`.
  Artifact key: `oasis_rank000_gaussiandir_convergence_v1`, execution
  `execution_20261106_v1`; 312 distinct exports/checkpoints and14 exact-field
  aliases scored, all35 labels eligible. Full report, tables, figures and executed
  notebook are under `${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_convergence_v1/execution_20261106_v1/report_v1/`.
- Both trajectories stopped at the1200s ceiling with
  `convergence_unresolved_at_time_limit`, zero operation overshoot and four LR
  reductions. Control:111,500 updates, selected step110,300, Dice0.767119
  (+0.017527 vs the2,000-step anchor). Corrected reconstruction:31,500 updates,
  selected step31,400, Dice0.785612 (+0.018009). Neither met the plateau rule.
- Mean usable-output latencies (± sample SD) for200/500/2,000-step schedules:
  control4.206±1.318 /7.220±1.035 /24.049±0.618s; corrected reconstruction
  10.519±1.418 /17.647±1.488 /53.841±1.342s. The1200-second trajectories
  improve selected Dice over anchors but do not establish convergence. The
  saved SITReg reference is accuracy-only; no SITReg inference time is claimed.
- Convergence peak GPU memory (Torch allocated/reserved; sampled NVML process):
  control217.4/272.0MiB optimization,144.4/272.0MiB output,786MiB process;
  corrected260.7/292.0MiB optimization,184.1/292.0MiB output,804MiB process.
  Posthoc scorer PID peaks are unavailable under the container PID namespace;
  per-evaluation Torch peaks remain reported. Registration ran sequentially on
  an otherwise idle RTX6000 Ada with benchmarkFalse/deterministicTrue/TF32False.
- Deferred by user2026-11-06: two fresh identity-start unsupervised variants
  using P43-style1-LNCC9² plus unmasked total-voxel forward-valid diffusion,
  lambda1.128 uniformly sampled complete windows and5,000 valid edges per axis
  estimate the full-volume loss; exact final dense loss is separately evaluated.
  Existing geometry/population rules,2,000-step budgets and1,200-second watchdog
  remain. Prototype loss/gradient tests and two native smokes passed; no full
  standardized-loss runs. Source spec:
  `experiments/2026-11-06_oasis_rank000_gaussiandir/lncc_diffusion_spec.md`;
  new external artifact key `oasis_rank000_gaussiandir_lncc9_diffusion_v1`.

- Status2026-11-06: user selected released control plus corrected paper model,
  1,000-step coarse/frozen-coarse+1,000-step fine residual with literal total
  domain/8 then domain/4 caps, and full-box MSE with fixed populations for fitting.
  Four independent anchors completed2026-11-06T20:35:53Z in
  `pilot_20261106_v1`, sequential GPU0, supervised before image registration.
  All reached2,000 steps; no failures/timeouts,35 eligible labels per row,
  zero checkpoint-replay error and exact paper-stage transition preservation.
  Registration uses adaptive density and journal lambda8; all endpoints use
  2,000 total steps with a1,200-second active watchdog and explicit timeout status.
- Seed3 shuffled rank0, fixed0333/moving0115. Supervised target is the original
  historical SITReg lambda1 field already used by P81/P82; original and canonical
  field hashes reverified. Full-box physical-mm fit, followed by
  independent image-only registration from identity and common accuracy/regularity.
- Upstream submodule `external/GaussianDIR` pinned at
  `b19987b81d1f3d172d882f1375a27d3b2ade307e`, unchanged. Released defaults are
  single-scale isotropic/translation; full paper and released model differ.
  Optional rigid forward/gradient discrepancies and adaptation defects documented.
- Eleven initial characterization checks and13 mathematical/native/continuation
  tests passed. All four full-volume resource smokes passed, including adaptive
  population events and exact coarse/fine and checkpoint field preservation.
  SITReg historical Dice/HD95/ASD/folds replay exactly; SDlogJ error<8e-11.
  Isolated Python3.10/PyTorch2.4.1+cu118/PyTorch3D0.7.8; base environment unchanged.
- Spec/log: `experiments/2026-11-06_oasis_rank000_gaussiandir/`.
  Durable reference: `docs/implementation/gaussiandir_paper_code_audit.md`.
  Audit source: `scripts/experiments/gaussiandir/audit_upstream.py`.
  External run/review key: `oasis_rank000_gaussiandir_v1`.
- Requested/effective backend: cuDNN benchmarkFalse/deterministicTrue, TF32False;
  source/config/input/runtime snapshots and all artifacts external. Single-case
  exploratory interpretation.
- Fit full-box RMSE: control0.372129mm, corrected0.240210mm. Image-registration
  Dice:0.749593/0.767602 versus SITReg0.798719; FG folds4.09227%/1.74520%
  versus0.000696944%. All accuracy/regularity fields and exclusions are in the
  external `repo_experiments/oasis_rank000_gaussiandir_v1/pilot_20261106_v1/review_v2/`.
  The corrected model improves this registration anchor over the release, but
  does not match SITReg accuracy/folding. Capacity/loss/optimization differ.

### P82: OASIS Rank0 Full-Box Parameter Scaling, Native vs Multiresolution

- Launched2026-11-05 22:43:11PST (2026-11-06T06:43:11Z), execution
  `execution_20261106T0644Z`. Initial queue20running/148pending on GPUs0/1/2/3
  with8/4/4/4workers respectively, independently admitted by available memory.
  All14 unit contracts,10 native operator profiles (including dense-sized
  fixed grids), two native worker smokes and external notebook execution passed.
  Original cps2 replay has exactly zero normalized-field discrepancy and
  physical vector RMSE7.0018e-16mm. This is a reconstruction control only.
  Source/configuration receipts are frozen; full168-point results remain pending.
- Seed3 rank0 fixed0333/moving0115; immutable original SITReg and unbounded842
  lambda1 saved endpoint fields. Full-box physical-mm supervised direct-DVF
  fitting only; no image registration, SVF, labels or regularizer in the loss.
- Seven P81 mathematical families x six caps3072/24576/196608/1572864/
  12582912/20643840 x two targets x two independent execution arms =168 points.
  Native versus joint quarter/half/native model-and-target refinement; matched
  final grid dictionaries and tree leaf counts, actual-parameter curve abscissa.
-1200-second active cap includes growth, conditioning and transfer. Joint
  stages use cumulative120/360/1200seconds. Exact B-spline knot insertion,
  <=60-second Gaussian projection, field-preserving tree splits. No borrowed
  lower-budget work. Full-box float64 LSQR, exact operators, no tail pruning.
- Resource/convergence limits are explicit endpoint outcomes. GPU0/1 max16,
  GPU2/3 max4; memory admission,60-second polling, cross-GPU OOM requeue with
  cumulative timing. Native cps2 replay and mathematical/native smoke gates.
- Spec/config/log/notebook: `experiments/2026-11-05_oasis_rank000_fieldfit_parameter_scaling/`.
  Source: `scripts/experiments/oasis_fieldfit_scaling/`.
  External artifact key:`oasis_rank000_fieldfit_parameter_scaling_v1`.

### P81: OASIS Seed-3 First10 Adaptive Local-Affine Field Fitting

- Execution completed2026-11-05T16:50:14Z (08:50:14PST):1680/1680 jobs,
  2640 endpoint records, no runtime failures/OOM retries,7.53hours elapsed.
  First metadata/coverage/paired-error review: external
  `execution_20261105_v3/reports/completed_review_v1/`; source in experiment folder.
  Scientific completion remains open:575/840 main D0 solves unresolved;
  all1680 main SVF final stages reached their fixed iteration budgets.
  Achieved cap3072 mean foreground RMSE: adaptive-affine D0 1.7087mm,
  adaptive-translation D0 1.7147mm; adaptive-translation zero-init SVF1.8558mm.
  Four source fields per each of ten repeated pairs; no independent-n40 claim.
  All1696 SVF main/replay endpoints pass the existing foreground integration
  sensitivity check; full-box accuracy is not implied. All32 D1 certificates
  satisfy0.9. Final mechanism analyses/convergence disposition remain pending.

- Runtime continuation authorized2026-11-05: GPU0/1 max16 each, GPU2/3 max4
  each; one shared queue, independent memory tiers16/8/4/2/1,60-second polling,
  and automatic CUDA OOM requeue with increased reservation/cooldown.
  Scientific protocolv1.3 and completed results remain unchanged. See
  `experiments/2026-11-05_oasis_adaptive_local_affine_fieldfit/scheduler_v1.md`.
  This supersedes the GPU0-only/max8/no-retry runtime policy below.
  Activated around10:26:55UTC/02:26:55PST after94 CPU tests and four native GPU
  smokes passed. Preserved93 completed jobs and adopted eight active workers;
  no worker was stopped. New controller immediately used GPUs2/3 while GPU1
  was externally occupied. Original scientific-source hashes remain unchanged.

- Launch history: v3 started2026-11-05T09:18:25Z (01:18:25PST), physicalGPU0.
  Initial verified snapshot8live workers/1672pending/0failures/no warning;
  source hashes match.76CPU tests,17native profiles and3worker smokes
  (7endpoints) passed, plus independent foreground-scoped multistep review.
  v2 stopped after conditioning failure with artifacts retained; original,
  v1.1/v1.2 plans preserved. Protocolv1.3; completion status and results are
  summarized in the preceding bullets.
- User-confirmed targets: same shuffled seed3 ranks0–9, original SITReg and
  8→4→2 lambda1 prolonged unbounded DVF, bounded-two-per-level (six retained
  maps), refined single-SVF with velocity diffusion.40 existing fields located.
- Main study: seven D0 basis/adaptation families × three motion-parameter caps
  768/1536/3072 ×40 targets =840 endpoints; fixed-geometry LS references.
  Raw fixed-image-positive ROI, physical pullback endpoint loss, no labels or
  image loss in fitting, no deformation regularizer. Source lambda is separate.
- SVF: same840 target/family/budget configurations as D0, with zero/global_affine_ls initial
  trials retained separately (1680 trial endpoints). D1:24 pilot configurations;
  adaptive bidirectional topology replay adds24 configurations/32 trials.
  Tree paths share budget checkpoints; total1680 queued worker jobs including
  40 global-affine anchors and24 frozen-geometry solver diagnostics.
  Supplementary same-K checkpoints cost separately.
- Preconditions: immutable hash/coordinate/warp audit, mathematical checks,
  native resource profiles and actual production-limit LS/multistep SVF pilot,
  frozen resource/numerical settings. v1.3 shares32³ full-box feature RMS scales
  with0.01 floor across LSQR and optimizers, retaining the foreground objective.
  This changes numerical solution selection in ill-conditioned systems; it
  neither adds regularization nor guarantees stable extrapolation. Divergent
  nonlinear checkpoints fail explicitly and independent work continues.
  All SVF families/replays share the same ROI-fitted12-parameter affine start,
  embedded as velocity without a matrix logarithm. D0/D1 local LS is unchanged.
  Corrected-scaling pilot still shows extreme ROI-exterior extrapolation and
  unresolved higher-budget LS; no global magnitude or stability bound is claimed.
  Both130-step SVF pilots improve fit but fail an exploratory full-box7/8
  sensitivity threshold; frozen-coefficient review under the existing
  foreground criterion passed, with exterior errors/raw failure retained.
  Smooth OASIS fields are a hypothesis; saved reference folding and fitting
  fidelity stay distinct.
- PhysicalGPU0 only, memory-based admission, max8 concurrentjobs; per-job failure
  ledger and continue-on-error, bounded watchdog/admission deadlines. Default
  configured container identity; cuDNN benchmarkFalse/deterministicTrue, TF32off.
  Runtime paths private, all generated outputs external. No historical
  registration or training reruns. Repeatedly explored first10, not held-out.
- Spec/log: `experiments/2026-11-05_oasis_adaptive_local_affine_fieldfit/`.
- Analysis status: full fitting matrix defined; Taylor/derivative-variation,
  transition maps, final paired statistics/figures/notebook remain explicit
  pending work in `analysis_completion_ledger.md`. Queue completion is not
  scientific completion; source-velocity replay remains conditional/unresolved.
- Artifact key: `oasis_seed3_first10_adaptive_local_affine_fieldfit_v1`.

### P80: First100 Matched-Prefix B-spline8421 NCC9/Diffusion Sweep

- Main release accepted2026-10-18 05:11:42UTC:2100/2100 independent outputs,
  zero deferred production failures;52 complete comparison groups and1178
  serialized statistic checks. Original ARC/report release remains immutable.
- Foreground-diffusion companion accepted2026-10-18 20:46:37UTC:5200 finite
  checksum-verified fields,52 complete groups, two new2x3 figures with common
  axes. Ten new/six legacy tests,52-cell smoke,520 source-statistic and728
  independent NumPy checks passed. Max full-volume parity error1.68613e-7;
  all12 earlier figure export hashes unchanged. External key:
  `oasis_seed3_first100_bspline842_8421_foreground_diffusion_arc_v1/release_v1`.
- Foreground-diffusion ARC companion requested2026-10-18: two additional2x3
  figures for842/8421, CPU-only analysis of5200 saved final-displacement fields.
  Both-endpoint fixed raw-positive edges; no lambda weighting; not identical to
  unmasked training diffusion or SVF velocity penalty. Existing outputs immutable.
  Spec/log:`experiments/2026-10-18_oasis_matched_depth_diffusion_arc/`.
- Approved:2100 fresh cps1 extensions of the accepted P79 prefixes,700 each
  displacement / bounded2 / refinedSVF-velocity. Seven global lambdas unchanged;
  bounded2 adds two full-schedule maps,2800 optimizerblocks total.
- Same saved prefix per case/family/lambda; older8421 finals appendix-only.
  Explicit v4 cps2->1 proof gate, original FP32 numerics and sources unchanged.
- Nine native rank0 smokes and independent replay gate production. Eight slots,
  two/GPU0–3, same-prefixGPU affinity,1009:1020, fresh memory admission.
- Case-local errors deferred; oneOOM tail retry; shared infrastructure pauses
  only admissions. Full acceptance requires2100 verified outputs.
- Report paired842/8421 accuracy/regularity, separate new/prefix time/memory,
  full SITReg/VFA curves, two dual-axis2x3 ARC and matched-axis842 companions.
- Spec/log:`experiments/2026-10-16_oasis_bspline8421_matched_prefix/`.
- Artifact key:
  `oasis_seed3_first100_bspline8421_matched_prefix_displacement_bounded2_svf_velocity_ncc9_lambda_sweep_v1`.
- Production launched2026-10-17 05:35:31UTC (Oct16 22:35PDT), execution_v2,
  eight workers/two perGPU. All53 tests, both roundoff witnesses, allnine
  corrected native smokes/audits and all2100 prefix hash/finite checks passed.
- First native smoke batch exposed bounded offgrid-autograd context error after
  optimization. execution_v1 preserved6accepted/3failed smokes; isolatedv2
  enables coordinate gradients only for that diagnostic. No objective change.
- Smoke reserved peaks2.277/4.627/3.922GiB (displacement/bounded2/SVF);
  admission8GiB per child. Storage projection2.192e12bytes including30%margin.

### P79: First100 Three-Family B-spline842 NCC9/Diffusion Sweep

- Accepted2026-10-17 04:15:43UTC (Oct16 21:15PDT): all2100 registrations,
  700 per family; both scoped continuations and independent replays passed.
  All2098 earlier results,29 failed-attempt files and prior reports preserved.
  Final69 complete cohort groups (10/90/100 cases),345 statistic checks and
  13 output checksums verified. Continuation suite89 tests:88 passed/one skipped.
  New optimizer431.39/383.72s; peak reserved3.297/3.262GiB for ranks29/44.
  Accepted report: `transition_continuation_v3/release_1792210489765127750`.
  `readable_arc_v2` is a separately validated title-wrap-only figure companion.
- Five-family ARC companion adds complete SITReg/VFA NCC9 sweeps:31 settings
  on100 paired cases,3100 rows. Two2x3 figures use FGSDlogJ and log10 FGfolding
  ratio, zoomed mean-only tops/full-SD bottoms. All310 source statistics and
  six presentation tests pass. Exact-zero means have a labeled display-only
  floor; no data changes. Key `oasis_seed3_first100_five_family_ncc9_arc_stdlogj_folding_v1`.
- Queue drained2026-10-17 02:47:53UTC:1528 new fits accepted plus570 reused;
  two velocity-SVF lambda.01 failures, ranks29(cps8->4)/44(cps4->2). Partial
  report2098/2100 preserved. One rank91 audit OOM recovered under the existing
  one-retry policy. Both outstanding failures are numerical transition gates.
- Approved scoped continuation after read-only native replay: rank29 retains
  cps8; rank44 retains cps8/4. Only missing stages optimized. Validationv3
  FP32 max.002/RMS1e-5 voxel plus exact prescribed controls and mandatory
  FP64 fallback proof; ideal velocity/field1e-9/1e-8 unchanged, actual-rounded
  field max2e-4/RMS1e-6. No optimizer/objective/integration change. Separate
  transition_continuation_v3 artifacts, originalGPU3/rank29 andGPU2/rank44,
  full remaining-stage one-update smokes and independent replay before fits.
  Preserve2098 accepted results, failure attempts and published reports.
- Approved2026-10-16: seven global lambdas.01/.1/.3/1/3/10/100 on ranks0–99;
  prolonged displacement, bounded-two-map composition and refined single-SVF
  velocity diffusion. All8/4/2, no final dense/cps1 or symmetric stage.
- Reuse430 accepted P43/P48 standalone prefixes plus70 P76 and70 P74 fits;
  run270/630/630 missing registrations,1530 new and2100 final cells total.
- Isolated first100 adapters preserve historical objectives/optimizers.
  Six extreme-lambda native smokes and independent replay gate production;
  every new fit needs independent field/loss/metric verification.
- GPUs0–3, two workers/auditors each,1009:1020; fresh8GiB-or-smoke+2 admission,
  same-GPU audits, durable continue-on-error/OOM-tail-retry queue. Prior outputs
  immutable. Original preflight validated570/570 reuses and200 references;
  four of six smoke/replays passed. Both SVF smokes failed before optimization:
  mixed controller settings were sent to the legacy path-only setup. Production
  never launched. Original manifests, four successes and failure logs retained.
- User approved repair/resume2026-10-16. New `metadata_recovery_v2` execution
  filters worker settings to four explicit paths and validates their types.
  Numerical sources remain unchanged; prior hash receipts are reused only
  after unchanged scientific-contract/source and artifact-stat checks. All six
  smokes will be rerun under the corrected source manifest before1530 new fits.
  Regression suite72 tests,71 passed/one skipped. Recovery revalidated570 reuses
  and200 reference hashes; all six smoke/replays passed in62.1s (maximum measured
  peak3.371GiB, admission8GiB). Production started2026-10-16 17:24:12UTC;
  eight active workers confirmed, two/GPU, no initial failures at that launch.
- Complete-only five-metric tables/2x3 ARC, separate10/90/100 summaries,
  originalVFA/SITReg, paired/label/stage diagnostics and time/memory provenance.
- Key:oasis_seed3_first100_bspline842_displacement_bounded2_svf_velocity_ncc9_lambda_sweep_v1.
  Spec/log:experiments/2026-10-16_oasis_bspline842_first100_sweep/.

### P78: Rank-0 Symmetric SVF Amplitude/Balance Phase B

- Approved 28 fresh cps8->4->2 registrations on rank0: baseline, seven
  amplitude weights, seven balance weights, nine combined selected neighbors,
  three matched-GPU input swaps and one matched-GPU same-order winner repeat.
- Preserve A1 midpoint NCC9 plus 2*(R1+R2), half-scaled actual velocities,
  exact independent SVF refinement and historical schedules. Add native full-
  volume mm-squared amplitude and squared RMS imbalance, no other loss change.
- Four full-size smokes and independent replay gates; all28 complete and
  independently replayed outputs required. Preserve Phase-A sources/results.
  One worker/GPU0-3,1009:1020, live admission8GiB or smoke peak+2GiB.
- Alpha1e-5/3e-5/1e-4/3e-4/1e-3/3e-3/1e-2; beta.01/.03/.1/.3/1/3/10.
  Dice/ASD/HD95 selects tested neighbors; no grid expansion or first-ten runs.
- Accepted2026-10-16 07:27:41UTC: all28 fits/replays and four smokes/replays
  passed,45 tests, no failures/retries. Production queue40.28min, maximum
  allocated/reserved peak across runs/replays4.918GiB; timings diagnostic only.
- Amplitude-only alpha.001 wins: Dice.796250,HD951.980043mm,ASD.430644mm,
  FGSDlogJ.555949,FGfold0%. Shared RMS27.316->2.915mm versus fresh A1.
  Best balance beta1 Dice.795014; combined alpha.001/beta1 Dice.793762.
  Balancing RMS more exactly did not improve the best tested accuracy.
- Same-order winner repeat Dice.796218, swapped reverse export Dice.796538,
  but geometric differences remain: repeat FG EPE p95/max.287850/11.501729mm;
  swap p95/max.289065/4.883435mm. No exact symmetry/retraining claim.
- Selected candidate uses125/150/250 updates,61.44s optimizer,121.37s worker,
  61.06s replay,4.568/4.859GiB allocated/reserved. Full report has all24 primary
  settings, four diagnostics, per-label/stage records and integration sensitivity.
- Key:oasis_seed3_rank000_bspline_two_total_svf_midpoint_ablation_v1/phase_b_v1.
  Spec/log:experiments/2026-10-15_oasis_symmetric_svf_phase_b/.

### P77: Rank-0 Symmetric Refined-SVF Phase A

- Approved four fresh 8->4->2 fits: independent two-SVF midpoint NCC9 with
  diffusion coefficient 2 (equal-split-effective lambda 1), coefficient 1
  scaling control, tied +/-w/2 single-SVF midpoint control, and swapped-input
  coefficient-2 check. No amplitude/balance sweep until the Phase-A review.
- Both total SVFs use exact knot refinement, native seven-step integration,
  twice-branch normalized parameters and historical 0.03/0.01/0.003 Adam LR.
  Separate two-input NCC supplies both gradients; no pyramid-map composition.
- Require four native smokes and four independent production replays, full
  forward/reverse canonical metrics, 8/10-step sensitivity, energy/common-drift
  audit and input-swap check. One new worker/GPU 0-3, UID:GID 1009:1020,
  live memory admission, preserve existing jobs and historical outputs.
- Completed and report accepted 2026-10-16 06:02 UTC (local October 15):
  four smokes/replays and four production fits/replays passed, 31 tests,
  no failures or retries. Production/replay queue wall time 386.02 s;
  peak allocated/reserved memory approximately 4.41/4.73 GiB per worker.
- A1/A2/A3 Dice is 0.785973/0.788333/0.790653, below historical bounded-two
  0.801036 and refined-SVF 0.797502. A1 has zero measured foreground folding,
  not an unconditional topology guarantee. Results are single-case only.
- A1/A2 branch RMS magnitudes are close (2.15%/1.19% relative gaps), but
  common-mode RMS is 27.19/35.92 mm. Midpoint foreground coverage grows from
  approximately 31% to 72-73%/84-86%. This is consistent with a shared-motion
  incentive in unmasked midpoint NCC9; balance alone cannot remove it.
- Input exchange is not numerically identical: original-direction A1 versus
  swapped-A1 reverse export has foreground endpoint difference mean/p95/max
  0.272/0.964/14.785 mm. Symmetric loss/gradient tests pass; the optimization
  discrepancy remains a reported limitation, not a proven GPU-only effect.
  Phase B was left approval-gated at Phase-A completion, then approved as P78.
- Key: oasis_seed3_rank000_bspline_two_total_svf_midpoint_ablation_v1/phase_a.
  Spec/log: experiments/2026-10-15_oasis_symmetric_svf_phase_a/.

### P76: First10 Bounded Two-Map 842 Lambda Sweep

- Approved70 fresh registrations: seed3 shuffled ranks0–9, cps8→4→2, two
  independent bounded maps per level, lambda0.01/0.1/0.3/1/3/10/100 at every
  level/map. Rerunlambda1; accepted P70 is read-only reproduction evidence.
- Preserve P70 NCC9, total-composed-field evaluation-grid voxel diffusion,
  normalized latent LR0.03/0.01/0.003,0.99 tanh bounds, full schedule per map,
  identity initialization and fresh Adam. No cps1/dense/SVF stage or tuning.
- Up to two workers per physical GPU0–3,1009:1020, live memory admission and
  existing jobs unchanged. Three endpoint/center smokes gate production.
  Individual failures are deferred without stopping unrelated tasks; shared
  storage/source failures pause admissions. One deferred OOM retry maximum.
- Require70 independent replays, six maps per fit, canonical five metrics,
  bound/off-grid diagnostics, checkpoint/history consistency, time/memory.
  Update ARC C/F; G retains its four-levellambda1 bounded point. Versioned
  artifacts; no historical source/output replacements.
- Accepted 2026-10-16 03:58:29 UTC (local October 15): all 70 registrations,
  70 independent replays and three full-size smokes/replays passed; no failed
  candidates or retries. Queue wall time 1344.36 s. All 420 map records,
  2450 label records and 101920 history rows verified; 51 regression tests.
- Lambda 1 has the best mean Dice/HD95/ASD: 0.819183/1.764939/0.364375 mm;
  FG SD(log J) 0.603642 and folding 0.000037%. Lambda 3/10/100 have zero
  measured foreground folding, with reduced accuracy. This is exploratory.
- C/F figure update accepted 2026-10-16 04:02:08 UTC: 55 displayed settings,
  550 case rows, 550 independent mean/SD checks and 18 axes verified. G keeps
  its historical four-level point. PNG/PDF/SVG and two-/seven-page PDF books
  saved under arc_panels_v1; source hashes and unchanged A/B/D/E pages checked.
- Key:oasis_seed3_first10_bspline_bounded2_842_ncc9_lambda_sweep_v1.
  Spec/log:experiments/2026-10-15_oasis_bounded2_842_lambda_sweep/.

### P75: First10 NCC9 Method-Family ARC Figures

- Approved aggregation-only presentation: five2×3 figures, mean-only top and
  marginal sample-SD ellipses below, Dice/HD95/symmetric ASD versus foreground
  SD(log J). Shared ranges/ticks/labels and large typography across all figures.
- A: SITReg/VFA/FeDDA NCC9 sweeps. B: P43 pyramid842/8421/dense sweeps.
  C/D: P70 matched unbounded-one/bounded-one/bounded-two composition at842/8421,
  all lambda1, marker-only. E: four P74 refined-SVF sweeps, continuation_v2.
-70 operating points/700 first10 case rows; same seed3 ranks0–9. DL historical
  lambda1 uses uniform visual styling, with provenance disclosed separately.
  No registration, inference, GPU job changes, or metric reevaluation.
- Accepted2026-10-15 20:35UTC: ten tests, all700 independently recomputed
  mean/SD values,70 cohorts,30 axes, source/output hashes, embedded fonts and
  five-page PDF checks passed. All five PNGs visually reviewed without clipping.
  Source artifacts remain unchanged; no GPU work or metric reevaluation.
- Key:oasis_seed3_first10_ncc9_method_family_arc_panels_v1.
  Spec/log:experiments/2026-10-15_oasis_first10_family_arc/.
- Approved lambda1 appendix accepted:11 ordered method/depth rows,110 cases,
  five mean/sample-SD metrics; three tests and110 independent statistic checks
  passed. New lambda1_table_v1 subdirectory; existing figures/reports unchanged.
- Approved F/G extension: three DL sweeps plus displacement/bounded-two/
  velocity-regularized SVF at842 (F) or8421 (G). Six methods,30 points per
  figure; bounded islambda1 only. Same axes/fonts/ellipses; no GPU work.
  Separate panels_FG_v1 release accepted2026-10-15 21:06:45UTC. Four new tests,
  independent600-statistic/12-axis checks, source hashes and visual review pass.
  Two-/seven-page PDF books verified; A–E and lambda1 table remain unchanged.
- Requested F/G revision: remove VFA/FeDDA only, keep SITReg plus displacement,
  bounded-two and velocity-SVF. Separate panels_FG_sitreg_v2 accepted
  2026-10-15 21:28:53UTC. Improved four-method styles; requested tighter top-row
  y scales shared by F/G, unchanged x axes and full-range ellipse rows.
  All40 settings/400 displayed records (350 unique) and400 statistics verified;
  all points visible,21 regression tests passed. Prior artifacts unchanged.

### P74: First10 Refined B-Spline SVF Lambda Sweep

- **Accepted2026-10-15 19:21:02UTC:**280/280 independently validated, all28
  first10 configurations complete. The continuation_v2 report has980 stage
  records,840 integration evaluations,9800 label rows and1400 paired rows;
  all output hashes and11619 preserved preexisting files verified. Ranks2/5
  completed only cps1;278 prior results and failed attempts remain unchanged.
- Queue finished278 accepted/two deferred. Approved continuation_v2 completes
  only cps1 for ranks2/5,8421/displacement/lambda0.01, from their verified own
  cps2 checkpoints. No numerical optimizer/source changes or prefix reruns.
  Proof-gated FP32 max0.002/RMS1e-5 voxel requires ideal and actual-rounded-
  control FP64 checks. Preserve278 results and failures; two native fixtures,
  independent replay and a new complete280-fit report gate acceptance.
- Continuation revision1:62 tests and both native-size one-update fixtures
  passed;11619 files preserved by inventory. Two cps1 workers launched
  2026-10-15 19:11:36UTC onGPU3/rank2 andGPU1/rank5. Final audits/report
  pending. Initial adapter metadata smoke failure is preserved separately.
- Both cps1 continuations independently passed by2026-10-15 19:18:53UTC:
  rank2 stopped1800 updates,rank5 stopped1850. All280 fits are validated;
  full artifact-preservation verification and final report publication pending.
- Latest approved amendment: preserve35 accepted fits and one unresolved
  rank2/cps8421/displacement/lambda0.01 failure, resume244 unattempted fits
  with two processes/GPU0–3. New continue-on-error controller isolates failures,
  persists retry/admission state, watches stalls and reports incomplete coverage.
  Numerical checks remain unchanged. Failed cases are reviewed after the queue;
  no incomplete ten-case means or full-cohort acceptance. See P74 spec/log.
- Continue-on-error resumption verified2026-10-15 13:28UTC:50 regression
  tests passed;1425 existing files inventoried;35 accepted/one deferred/eight
  active/236 waiting. Exactly two new workers on eachGPU0–3; numerical sources
  unchanged. This is live execution status, not final experiment acceptance.
- Approved2026-10-15:280 fresh registrations, two penalties (total velocity or
  integrated total displacement), two schedules842/8421, seven lambdas
  0.01/0.1/0.3/1/3/10/100 and shuffled seed3 ranks0–9. One lambda at every
  active stage; no reuse of accepted pilot fits or cross-cell prefixes.
- Preserve P73 NCC9, native seven-step SVF integration, normalized controls,
  pyramid/adaptive Adam schedule. Separate generalized sources; P73 immutable.
- Four full-size extreme-lambda smokes and unit parity gates before production.
  Independent replay,8/10-step sensitivity and inverse/corner diagnostics for
  all fits.280 outputs/980 stages/840 integration evaluations required.
- All available GPUs0–3, one new process/GPU, UID:GID1009:1020, fresh memory
  admission max(8GiB,smoke peak+2GiB). Concurrent jobs remain untouched.
  Record optimizer/worker/audit time and process peak allocated/reserved memory,
  stage diagnostics and concurrent device load; report mean/SD/min/max.
- Four tables, per-case/per-label companions, original VFA/SITReg references,
  paired comparisons and line-only2x3 accuracy/regularity curves. Exploratory.
- Key:oasis_seed3_first10_bspline_svf_refine_842_8421_ncc9_lambda_sweep_v1.
  Spec/log:experiments/2026-10-15_oasis_bspline_svf_first10_sweep/.
- Status:17 numerical/CLI/report unit tests, four full-size extreme-lambda
  smokes and four independent audits passed. Smoke peak reserved3.995GiB;
  admission8GiB. Persistent280-fit controller launched onGPUs0–3 with
  automatic complete-cohort reporting and separate resource measurements.
- Recovery approved after17 accepted fits: rank3/lambda0.01 velocity arm
  stopped on FP32 knot-transition allclose despite FP64 agreement1.031e-12
  voxel and identical accuracy/folding metrics. Isolated validation-only
  adapter uses voxel-unit limits plus FP64 confirmation; preserves all17
  successes and failure logs, restarts that cell from identity in attempt1,
  and resumes262 unlaunched cells after regression/smoke gates. See P74 log.
- Recovery gates passed:24 tests, fresh four-stage native-size smoke and
  independent replay. All691 preserved files hashed. Four recovery workers
  started2026-10-15 12:30:48UTC;8GiB admission retained. Production/reporting
  remains pending; no completed results are inferred from this launch status.
- User-approved concurrency amendment: two workers/auditors perGPU0–3, eight
  total. Scheduler-only adapter adopts active children, preserves completed
  fits/failures, keeps numerical sources hash-frozen, and enforces per-process
  memory admission and per-GPU slots. Mixed-concurrency timings are diagnostic.
- Eight-job handover confirmed2026-10-15 12:42:30UTC:21/280 accepted, four
  existing workers adopted without interruption and four added, exactly two
  perGPU. Six scheduler tests/live preflight passed; no failures at handover.

### P73: Rank-0 B-Spline SVF Eight-Run Pilot

- User-approved2026-10-14: eight fresh rank0 registrations, refined single SVF
  versus composed stage SVFs, velocity versus total-displacement diffusion,
  cps842 versus8421. NCC9/lambda1, normalized LR and established adaptive Adam.
- Native-grid seven-step scaling/squaring; explicit v/128, endpoint-aligned
  border sampling, unbounded velocity coefficients. Composed velocity penalty
  applies only to the active SVF. No historical numerical sources changed.
- Eight full-size smokes, one worker/GPU0–3, UID:GID1009:1020; independent
  replay,8/10-step sensitivity, inverse and topology diagnostics gate acceptance.
- External key:oasis_seed3_rank000_bspline_svf_842_8421_ncc9_lambda1_v1.
  Spec/log:experiments/2026-10-14_oasis_bspline_svf_rank0/.
- Status: accepted2026-10-15 06:59:57UTC; eight native-size smokes and eight
  fresh registrations, each independently replayed, passed. No OOM/retries or
  numerical failures. Peak reserved5.114GiB; admission8GiB. All28 saved stage
  states and24 integration evaluations verified; six final report hashes pass.
  Initial aggregation failed only on historical runtime strings `n/a`; a
  separately tested report adapter preserved partial outputs and numerical
  source hashes, normalized unavailable timings, and completed without reruns.
- Refined velocity-regularized8421 gives Dice/HD95/ASD
  0.804515/1.990093/0.417839 with foreground folding0.001394%.
  Displacement-regularized8421 improves accuracy but folds0.914–1.407%.
  One scaled initial-map derivative bound exceeds1; this is a reported outcome,
  not a numerical rejection or a diffeomorphism guarantee.

### P72: Test100 Label-Wise Accuracy–Regularity Curves

- Approved aggregation-only release: main19 NCC9 settings (MR-D-BSCP dense,
  VFA, SITReg, FeDDA) plus four saved MR-D-BSCP prefix-only settings.
- Reuse canonical35-label accuracy and whole fixed-foreground SD(log J)/folding;
  no registration, inference or metric reevaluation. Common valid cases per
  label; Right-Vessel currently98, other labels100. Historical DL1 flagged.
- Produce35 label-specific2x3 pages per comparison, PDF books, overview atlases,
  per-case/per-label CSV/JSON, descriptive trend and paired prefix/dense tables.
- Append-only artifact key `oasis_seed3_test100_label_arc`, release `v001`;
  source provenance registry supports future sweeps without overwriting releases.
- Spec/log: `experiments/2026-10-14_oasis_test100_label_arc/`.
- Status: accepted `releases/v001/`. Ten tests and full-cohort scalar smoke
  passed;2300 case rows,80500 label slots,805 summaries. All original19 ARC
  case means/SDs reproduced. Independent8050 summary-value checks match exactly;
  70 PNG/SVG label figures, two35-page PDF books and six PNG/PDF/SVG atlases
  verified; all180 output hashes and versioned registry linkage passed.

### P71: First10 B-Spline Depth Comparison with Bounded Three-Map Composition

- Add one fresh bounded-three-map composition arm to each accepted P70 depth:
  cps8,8→4,8→4→2,and8→4→2→1. Forty new registrations cover seed-3 shuffled
  OASIS ranks0–9; P70's accepted160 cells remain read-only comparison rows.
- Three independent bounded identity increments are retained/composed per
  active CPS level. Each receives the full existing stage schedule and fresh
  Adam. All active diffusion weights remain1; preserve NCC9, total-field
  voxel diffusion, preprocessing, bound, learning rates, and adaptive stopping.
  This adds representational capacity and optimization updates; it is not an
  equal-compute comparison.
- Four one-update native-grid smokes gate production. Run as1009:1020 on
  available GPUs0–3, one worker/GPU, case-centric sequential queues, live
  memory admission of8GiB or measured smoke peak+2GiB. Do not modify existing
  GPU jobs. Preserve failures/logs.
- New runner/auditor adapter and isolated artifact key:
  `oasis_seed3_first10_mrdbscp8421_bspline_bounded3_depth_extension_v1`.
  Produce a separate combined report with four tables, each containing the
  original four P70 settings, the new bounded-three-map setting, VFA, and
  SITReg. Include per-case and paired tables, stopping/runtime, and
  shared-prefix diagnostics. Never rewrite P70 accepted artifacts.
- Spec/log:
  `experiments/2026-10-14_oasis_bspline_bounded3_extension_first10/`.
- Status: accepted after four native-grid smokes,40 fresh registrations, and
  40 independent replay audits; no production failures or OOMs. All300 map
  blocks reconstructed and passed their bounded elementary-map checks. The
  combined report contains four tables with five optimized settings plus VFA
  and SITReg references; report hashes/counts passed and P70 accepted hashes
  remain unchanged. Prefix diagnostics include180 comparisons; report observed
  cross-depth numerical variation rather than rejecting independent cells.
- New artifact key:
  `oasis_seed3_first10_mrdbscp8421_bspline_bounded3_depth_extension_v1`.
  Bounded-three-map mean Dice by schedule is0.7169/0.7941/0.8244/0.8322;
  corresponding HD95 is2.4593/1.9280/1.7698/1.7013mm and ASD is
  0.6667/0.4314/0.3542/0.3330mm. Full-depth foreground SD(logJ)/folding is
  0.7047/0.0209%, versus bounded-two-map0.6509/0.0031%. Three maps modestly
  improve full-depth accuracy over two maps, but increase log-Jacobian spread
  and folding; this is not an equal-compute comparison. First10 remains
  exploratory due prior tuning.
- Combined report:
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first10_mrdbscp8421_bspline_bounded3_depth_extension_v1/`.

### P70: First10 B-Spline Pyramid Depth × Constraint Factorial

- Fresh 4×4 comparison on seed3 shuffled OASIS ranks0–9: schedules8,8→4,
  8→4→2,8→4→2→1 crossed with unbounded/bounded single-map and two-map
  composition. All active diffusion weights are1. Every cell starts fresh
  from identity; the full16 settings per case run sequentially on one GPU.
- Two-map settings give each independent map the full stage schedule and fresh
  Adam. Keep P68/P69 NCC9, total-field evaluation-grid diffusion, bound,
  composition, normalized LR and adaptive optimizer behavior. There are160
  new registrations; accepted P69 is read-only and only a reproduction check.
- Separate runner/auditor and artifact key:
  `oasis_seed3_first10_mrdbscp8421_bspline_depth_bound_steps_lambda1_v3`.
- User approved available GPUs0–3. One worker/GPU, live memory checks,
  UID:GID1009:1020; no existing job is altered.
- Four full-size one-update smokes gate production. Require160 complete fields,
  160 independent replays, same-setting shared-prefix checks, and four depth
  tables with VFA/SITReg references before acceptance.
- Spec/log: `experiments/2026-10-14_oasis_two_increments_842_first10/`.
- Status: accepted v3 run. Four one-update smokes, all160 fresh registrations,
  and all160 independent replay audits passed with zero failed cells. Report
  contains four depth tables and verified hashes. The first integrated report
  attempt exposed an over-strict requirement for bitwise-equal cps2 prefixes;
  the artifact-only aggregator preserves all run outputs and reports this
  CUDA repeatability difference instead of rejecting cells. Across300 cps8/
  cps4 prefix checks, all were byte-identical. Across60 cps2 checks, max RMS
  difference was0.04335 voxel, maximum local difference6.3676 voxels, and max
  objective-component delta3.50e-5; this is not bitwise reproducibility.
- Accepted report:
  `/database/inrorspline/repo_experiments/oasis_seed3_first10_mrdbscp8421_bspline_depth_bound_steps_lambda1_v3/`
  (`report.md`, `summary.csv`, `per_case.csv`, `paired_reference_differences.csv`,
  `stage_diagnostics.csv`, and `shared_prefix_checks.csv`). Full-depth rank0–9
  exploratory means: unbounded one/two-map Dice0.834680/0.833919 and folding
  2.207595/2.174567%; bounded one/two-map Dice0.809085/0.828813 and folding
  0/0.003094%; VFA/SITReg Dice0.819772/0.821546 and folding0.218560/0.000423%.

### P69: First10 Two Bounded Increments, Lambda1151 Versus1111

- User-approved first10 extension of P68's two-map method, cps8/4/2/1,
  lambda[1,1,5,1] versus[1,1,1,1]. Reuse2 accepted rank0/4 lambda1151 outputs;
  run18 fresh identity registrations. No continuation-control reruns/tuning.
- Preserve8 retained bounded maps, NCC9/evaluation-grid total-field diffusion,
  bounds, normalized LR and adaptive schedules. Separate adapter; immutable
  numerical sources and historical outputs. Explicit finite-difference override.
- GPU0/1/2/3 one worker each,1009:1020,8GiB or smoke peak+2GiB admission;
  existing jobs untouched. Keep paired configurations on the same GPU.
- Tests/two rank1 full-size smokes,18 independent eight-block audits,
  common four-block prefix checks,20 finite complete fields before acceptance.
- Report five metrics per case and mean/sampleSD, paired lambda differences,
  original VFA/SITReg and P67 references. Separate8 nonpilot cases; exploratory.
- Artifact key: `oasis_seed3_first10_mrdbscp8421_tanh_two_increments_lambda1151_1111_v1`.
- Spec/log: `experiments/2026-10-13_oasis_two_increments_first10/`.
- Status:47 tests passed (optional native parity rerun with explicit roots),
  both full-size rank1 smokes passed. All18 fresh registrations completed
  acrossGPU0/1/2/3, plus2 reused pilot fits; no failures/retries. All20 audits
  passed, including160 block checkpoints. Accepted2026-10-14 06:55:43UTC.
- All40 common-prefix comparisons are byte-identical. Lambda1151/1111
  mean Dice0.821953/0.828756, HD951.724650/1.678458mm, ASD0.358358/
  0.338309mm; FG SDlogJ0.581739/0.650974 and folding0.000381/0.003143%.
- Lambda1111 wins paired Dice9/10, HD959/10 and ASD10/10, but raises both
  regularity metrics10/10. All retained-chain100k probes have0 nonpositive
  Jacobians; exported interpolants do not. Full means/sampleSD and nonpilot8
  comparisons are reported; all source/reference/acceptance hashes verified.

### P68: Two Bounded Increments Versus Same-Map Two-Block Control

- User-approved four-run pilot: ranks0/4, two bounded increments per cps8/4/2/1
  versus one bounded map per level continued for a second fresh-Adam block.
- Retain8 versus4 maps; both use8 optimization blocks. Frozen lambda[1,1,5,1],
  NCC9/total-field evaluation-grid diffusion, bounds, LR and adaptive schedules.
- Separate adapter and source/hash guards; no prior output modifications.
  Two full-size eight-block smokes gate production onGPU0/1/2/3,1009:1020,
  one worker/GPU, memory8GiB or measured peak+2GiB. Existing jobs unchanged.
- Require32 block checkpoint/loss/field/metric/topology audits and same100k
  off-grid probes. Complete-only per-case CSV/JSON/Markdown, matched historical
  all-bounded/selective/VFA/SITReg references. No tuning or first-ten expansion.
- Artifact key: `oasis_seed3_rank000_004_mrdbscp8421_tanh_two_blocks_lambda1151_v1`.
- Spec/log: `experiments/2026-10-13_oasis_bounded_two_blocks/`.
- Status:43 tests passed/1 skipped; both full-size eight-block smokes passed,
  peak3.458GiB. All4 fresh fits and32 independent block replays passed;
  accepted2026-10-14 06:09:54UTC (Oct13 PDT). No failures/retries/tuning.
- Two increments versus same-map continuation Dice: rank0 0.798510 versus
  0.780452; rank4 0.728966 versus0.687008. Historical one-block all-bounded
  Dice0.774069/0.684704. Extra maps improve all three accuracy metrics on
  both cases beyond the continuation control, but rank4 remains below DL.
- Two-map-arm FG folding0.000743/0.000300%; controls0%. All atomic bounds
  pass and retained-chain probes have0/100000 nonpositive determinants;
  exported interpolants have4/100000 and3/100000. Do not claim zero folding
  for the export. Reports preserve all selective and DL comparisons.

### P67: First10 Last1/Last2/Last3/All4 Bounded Composition

- User-approved first-ten extension of frozen P64/P66: reuse4 accepted rank0
  outputs, run36 fresh identity registrations on ranks1–9; no tuning.
- Same cps8/4/2/1 retained composition, weights[1,1,5,1], NCC9/diffusion,
  normalized latent LR0.03/0.01/0.003/0.001 and adaptive schedules.
- One new worker per available GPU0/1/2/3 approved;1009:1020; current8GiB
  admission or smoke peak+2GiB. Existing jobs and historical sources unchanged.
- Separate adapter, complete-cohort tests and rank1 extreme-bound smokes;
  checkpoint/field/loss/corner/off-grid checks, common-prefix agreement.
- Complete-only40-fit per-case and mean/sampleSD report, plus same-ten
  original VFA/SITReg references; ranks1–9 separate from exploratory rank0.
- Artifact key: `oasis_seed3_first10_mrdbscp8421_selective_tanh_lambda1151_v1`.
- Spec/log: `experiments/2026-10-13_oasis_selective_bounded_first10/`.
- Status:38 tests passed/1 skipped; both full-size rank1 smokes passed.
  All36 fresh fits completed across four GPUs, plus4 reused rank0; no failures.
  Complete40-fit reports passed acceptance and all40 common-prefix field
  comparisons are byte-identical. Independent replay passed for all160 stage
  checkpoints and all40 final per-label/accuracy/regularity results.

### P66: Rank0 Last1/Last2/Last3 Bounded B-Spline Composition

- User requests three8/4/2/1 variants: bound cps1 only, cps2/1, or cps4/2/1.
  Fresh identity increments at every stage, weights[1,1,5,1], unchanged P64
  NCC9/diffusion/normalized-LR/pyramid/adaptive schedule and retained composition.
- Unbounded earlier controls use c=p, bounded controls use scaled tanh with
  validated0.99 cap. Save accurate stage-specific metadata/checkpoints; do not
  apply the bound when serializing unbounded controls or reject expected folds.
- Separate adapter, three one-step smoke fits then three production fits,
  GPU3 sequential1009:1020, memory8GiB or smoke peak+2GiB. Existing jobs and
  prior sources/outputs unchanged. No extra sweeps, cps16 or cohort expansion.
- Compare P64 all-bounded, historical prolonged unbounded/dense, VFA/SITReg;
  same five metrics and100k off-grid points, stage folding/bounds/saturation,
  independent checkpoint/metric audit. Single exploratory case only.
- Artifact key: `oasis_seed3_rank000_mrdbscp8421_selective_tanh_lambda1151_v1`.
- Spec/log: `experiments/2026-10-13_oasis_selective_bounded/`.
- Status: all29 unit/regression tests passed, including unbounded checkpoint
  preservation and native loss parity. All three full-size four-stage smokes
  passed; peak2.678GiB, admission8GiB. Production cleared sequentially onGPU3.
- All three fresh registrations completed without failures/retries; report
  accepted after common-prefix checks. Independent replay passed2026-10-14
  00:08:23UTC: all12 stage reconstructions/losses/35-label metrics/corner audits,
  same100k off-grid probes, unchanged source/reference/report hashes. Common
  prefixes are byte-identical;128-point gradient checks max error<2.789e-9.
  Optimization66.47/64.50/64.73s; worker124.71/125.38/117.30s.
- Last1/last2/last3 Dice0.789160/0.801810/0.792172; HD952.113888/1.975920/
  2.023935mm; ASD0.466397/0.422405/0.448346mm. Foreground SDlogJ0.610573/
  0.668009/0.562964, folding1.021628/1.621929/0.378766%.
- Last2 improves all three rank0 accuracy metrics versus VFA/SITReg but not
  folding. Last3 improves all five metrics versus last1 on this case. More
  final bounded stages do not monotonically reduce measured folding; early
  unbounded maps are already folded. P64 all-bounded retains0% at lower Dice.

### P65: Rank0 Cps16 Extension of Bounded Compositional B-Splines

- Approved one fresh cps16/8/4/2/1 identity-initialized registration, one
  bounded increment each, lambda[1,1,1,5,1]. Preserve P64's downstream stages.
- Loss pyramid8/4/2/1/1, normalized latent LR0.03/0.03/0.01/0.003/0.001;
  NCC9 and evaluation-grid total-field diffusion unchanged. Only final cps1
  permits two LR halvings; no reused prefix or accumulated-map resampling.
- Separate numerical adapter, actual-kernel FP64 cps16 bound,0.99 margin,
  unit initial tanh derivative; five saved constituent maps and final DVF.
- GPU3 sequential smoke/production, UID:GID1009:1020, memory8GiB or
  smoke peak+2GiB. Preserve failures and other jobs; no scope expansion.
- Compare P64 bounded8421, P63 unbounded, P62 dense, original VFA/SITReg;
  five metrics plus stage saturation, bounds, epochs, runtime and shared100k
  off-grid probes. Single-case capacity/pyramid pilot, not a causal guarantee.
- Artifact key: `oasis_seed3_rank000_mrdbscp168421_tanh_composed_lambda11151_v1`.
- Spec/log: `experiments/2026-10-13_oasis_bounded_composed_cps16/`.
- Status: all29 unit/regression tests passed including native loss parity.
  Full-size five-stage smoke passed, peak2.832GiB and71.69s including metrics;
  exact initialization,35 valid labels/stage and all topology checks passed.
  One production fit completed without failures/retries, optimization66.85s,
  complete worker123.18s, peak2.833GiB. Stop275/300/175/150/500, best final495.
  Report accepted and independent replay passed2026-10-13 23:45:32UTC:
  exact five-stage reconstruction, losses,35-label metrics, corner checks,
  same100k off-grid probes and reference/source/report hashes; additional
  128-point FP64 finite-difference gradient max error2.426e-9.
- Rank0 Dice/HD95/ASD0.770576/2.379341/0.532366, FG SDlogJ0.506026,
  folding0%. Relative to bounded P64, Dice-0.003493, HD95+0.154360mm,
  ASD+0.035322mm: adding cps16 did not improve this case. Cps8 saturation
  drops40.726% to35.302%, without an accuracy benefit. All atomic bounds<1;
  retained/export shared100k probes both have zero nonpositive determinants.

### P64: Rank0 Tanh-Bounded Compositional B-Splines

- Approved one fresh identity-initialized cps8/4/2/1 run, one bounded increment
  per stage, lambda[1,1,5,1]. No reuse of folded prefixes or extra sweep.
- Sampling-aware FP64 bounds for native endpoint-aligned cubic samples,
  multiplier0.99; scaled tanh retains normalized latent LR convention.
  Retained pull composition Phi_previous o delta_new, not additive updates.
- Existing NCC9/evaluation-grid total-field diffusion and adaptive schedules;
  source/output isolation, full-cell-corner atomic checks and100k shared
  off-grid probes for retained chain versus interpolated export.
- GPU3, UID:GID1009:1020; tests and one-step-four-stage smoke gate production.
  Existing jobs untouched; at least8GiB free or smoke peak+2GiB.
- Artifact key: `oasis_seed3_rank000_mrdbscp8421_tanh_composed_lambda1151_v1`.
- Spec/log: `experiments/2026-10-13_oasis_bounded_composed/`.
- Status:23 unit/regression tests passed, native SITReg bounds independently
  reproduced. Native-voxel component caps3.174971/1.554847/0.7128/0.3564 for
  cps8/4/2/1. Full-size smoke passed all35 labels/stage, exact identity/native
  loss parity, atomic contraction and100k off-grid checks. Peak2.678GiB,
  memory gate8GiB. Production completed without failures/retries and passed
  independent audit2026-10-13 16:26PDT, including exact four-stage checkpoint
  reconstruction, losses, labels, topology, report hashes and off-grid replay.
- Rank0 Dice/HD95/ASD0.774069/2.224981/0.497044; FG SDlogJ/folding
  0.493470/0%. Folding falls from unbounded P63's3.718802%, but Dice
  drops0.031294, HD95/ASD worsen0.241106/0.085058mm. Accuracy is not matched.
- No nonpositive off-grid probes for either retained chain or interpolated
  export; minimum determinants0.045250/0.045843. Atomic max derivative row
  sums0.977792/0.984813/0.988912/0.989734. No general export guarantee.
- Stop epochs400/200/150/500, best400/200/150/496, optimization63.29s,
  worker109.82s. High coarse coefficient saturation is diagnostic, not proof
  of the cause of accuracy loss. No additional capacity or tuning was run.

### P63: Rank0 Cps1 B-Spline versus Dense after [1,1,5] Prefix

- User requested one all-B-spline8→4→2→1 fit with stage weights[1,1,5,1],
  compared against accepted P62 dense refinement on the same rank0 pair.
- Reuse accepted P61 prefix, exact dyadic cps2→cps1 prolongation; change only
  final representation. Preserve local squared NCC9, full-volume total-field
  voxel finite-difference diffusion, normalized LR0.001 and adaptive2000/300
  schedule. Explicitly no analytic B-spline regularizer or label supervision.
- GPU3, UID:GID1009:1020, seed1/FP32/cuDNN F/T/TF32off. Existing jobs untouched.
  Tests and one-step serialization/metric smoke gate production.
- Same-case five-metric comparison plus prefix and original DL references;
  exploratory n=1, no topology guarantee or across-case SD.
- Artifact key: `oasis_seed3_rank000_mrdbscp8421_lambda1151_vs_dense_v1`.
- Spec/log: `experiments/2026-10-13_oasis_cps2lambda5_cps1/`.
- Status:19 tests passed; full-size one-step smoke passed canonical metrics
  and native loss/gradient parity. Cps2 reconstruction is exact; prolongation
  max/RMS error3.323e-6/1.664e-7 voxel. Initial total loss matches dense exactly.
  Peak1.986GiB, free-memory gate4GiB. Production complete and independently
  audited2026-10-13 15:42PDT: saved controls reconstruct the field exactly,
  both variants' five metrics and35 per-label values reproduce, lowest-loss
  checkpoint selection and all report hashes pass. No failures/retries.
- Cps1 Dice/HD95/ASD0.805363/1.983874/0.411986 versus dense
  0.804582/1.995025/0.411972. Accuracy is close, but folding rises from
  3.695896% to3.718802% and FG SDlogJ from0.799019 to0.802908.
  Cps1 stopped725/best719, optimization58.78s. No folding benefit on this case.

### P62: Cps2 Lambda5 Prefix to Dense Diffusion Sweep

- User requested70 fresh dense refinements on seed3 first10, dense lambda
  0.1/0.2/0.5/1/2/5/10 after accepted P61 prefixes with weights[1,1,5].
- Preserve native local squared NCC9, total-field voxel diffusion, raw inputs,
  normalized dense LR0.001 and2000/300 adaptive schedule with two LR halvings.
  Fresh Adam, seed1, FP32, cuDNN F/T, TF32off; no cps1 controls or label loss.
- Separate runner validates P61 stage weights/checksums and exact dense cloning;
  tests and rank0 one-step0.1/10 smokes gate production. GPUs0/1/3 one worker
  each, memory gated, UID:GID1009:1020;GPU2/existing jobs untouched.
- Report seven weights, unrefined[1,1,5], original VFA/SITReg/FeDDA, five
  accuracy/regularity metrics and sample SD, plus paired differences against
  the previous[1,1,1] dense sweep at six shared weights. Exploratory tuning.
- Artifact key: `oasis_seed3_first10_mrdbscp842_prefix115_dense_lambda_sweep_v1`.
- Spec/log: `experiments/2026-10-13_oasis_cps2lambda5_dense/`.
- Status: all70 fits accepted2026-10-13 15:15:01PDT; no failures/retries.
  All20 unit/legacy tests and both smokes passed, exact reconstruction/clone
  and native loss/gradient parity. Peak1.638GiB; free-memory gate4GiB.
- Dense lambda0.5 wins all three mean accuracy metrics:
  Dice/HD95/ASD0.834955/1.681926/0.331114, FG SDlogJ/folding0.819045/3.879464%.
  Compared with P60 at the same dense weight, Dice changes-0.000817 while
  HD95/ASD improve0.056620/0.003405mm and folding drops0.583381 percentage
  points. All six shared weights reduce folding, but DL regularity is not
  matched at the accuracy-selected point. Independent110-row report and
  60 paired-comparison checks passed. Mean optimization/worker52.38/68.91s.

### P61: Fixed cps8/cps4 Lambda1 to Cps2 Diffusion Sweep

- User approved 60 fresh cps2 fits: seed3 shuffled first10 × lambda
  0.1/0.2/0.5/1/2/5. Reuse identical saved lambda1 cps8/cps4 stages only;
  initialize cps2 by exact dyadic knot insertion from stage1/best.pt.
- No dense/cps1 refinement. Unchanged local squared NCC9, unmasked
  evaluation-grid voxel diffusion; normalized cps2 LR0.003, max/min1300/130,
  patience50, median/check25, no LR halving, lowest-total-loss checkpoint.
- Separate runner preserves all historical sources and outputs. Four new
  unit tests passed; rank0 one-step0.1/5 smokes gate production. GPUs0/1/3,
  one worker each with memory gate; UID:GID1009:1020, GPU2 untouched.
- Report six settings versus original842lambda1 and original VFA/SITReg/
  FeDDA: five accuracy/regularity metrics, mean/sample SD, paired deltas,
  historical lambda1 reproducibility, stopping/runtime and full provenance.
- Artifact key: `oasis_seed3_first10_mrdbscp84_prefixlambda1_cps2_lambda_sweep_v1`.
- Spec/log: `experiments/2026-10-13_oasis_fixed_prefix_cps2/`.
- Status: all60 fits accepted 2026-10-13 14:31:07 PDT; no failures/retries.
  All20 unit/legacy tests and both rank0 smokes passed. Prolongation error
  <5e-6 voxel; native loss/gradient parity exact. Peak1.461GiB; gate4GiB.
- Cps2lambda0.5 wins mean Dice/ASD (0.829577/0.350710 mm); lambda1 wins
  HD95 (1.787126 mm). Lambda0.5 folding is5.217171%, versuslambda1 3.676989%.
  All ten freshlambda1 field hashes and metrics reproduce original842 exactly.
  Mean cps2 optimization/total worker time25.88/39.56s; independent report,
  per-label, sample-SD, paired-delta and fixed-prefix checks passed.

### P60: Fixed Lambda1 MR-D-BSCP Prefix to Dense Diffusion Sweep

- User approved60 fresh dense fits: seed3 shuffled first10 × dense lambda
  0.1/0.2/0.5/1/2/5. Each reuses its identical accepted pyramid cps8/4/2
  prefix withlambda1 at every prefix stage. Historical dense lambda1 is a
  reproducibility reference only, not reused as a new result.
- Retain local squared NCC9, unmasked total-field voxel diffusion, normalized
  LR0.001, seed1, FP32, cuDNN F/T and TF32off; dense2000/300, window/check25,
  patience100, two LR halvings, lowest-total-loss state including initialization.
- Separate runner preserves historical source/output hashes. Tests and rank0
  one-step0.1/5 smokes gate production. GPUs0/1/3, one new worker each, memory
  gated; existing jobs andGPU2 untouched. UID:GID1009:1020.
- Report six weights, prefix and original VFA/SITReg/FeDDA on the same10 cases:
  Dice/HD95/symmetric ASD and foreground SDlogJ/folding, mean/sample SD,
  paired changes, stage/runtime diagnostics and old/newlambda1 differences.
- Status: all 60 fresh fits accepted 2026-10-13 13:23:32 PDT, with zero
  failures/retries. All 21 CPU tests and both native-geometry smokes passed;
  exact prefix reconstruction/clone and NCC/diffusion value/gradient parity.
  Peak tensor allocation was 1.638 GiB; the admission gate remained 4 GiB.
- Dense lambda0.5 wins mean Dice/ASD (0.835772/0.334519 mm); lambda1 wins
  mean HD95 (1.734039 mm) among the dense weights. Lambda5 reduces foreground
  folding to 0.682494%, still above original VFA/SITReg, with lower Dice.
  All ten fresh lambda1 DVF hashes and five metrics match historical outputs.
  Independent report-hash, sample-SD, label-mean, and paired-difference checks
  passed. Mean dense optimization/total worker time was 43.97/61.35 seconds.
- Artifact key:`oasis_seed3_first10_mrdbscp842_prefixlambda1_dense_lambda_sweep_v1`.
- Spec/log:`experiments/2026-10-13_oasis_fixed_prefix_dense/`.

### P59: SITReg squared GNCC after checkpoint-faithful FeDDA recovery

- Status: recovery launched 2026-10-13 01:43:10 PDT (08:43:10 UTC).
  Persistent queue PID 102712; FeDDA controller PID 102802 and workers
  102805/102810/102813/102816 on GPUs 0/1/2/3. All four exact checkpoint/RNG/
  iterator gates passed; finite resumed updates and historical pair IDs verified.
  Automatic SITReg launch requires all four P58 recovered trainings and
  first100 evaluations plus the final report.
- Recover P58 from epochs 103/120/118/120 on GPUs 0/1/2/3 into distinct
  `_recovery_v1` directories; preserve failed sources/artifacts. Freeze repo
  runtime sources and recover an exact historical trailing newline. Validate
  saved Adam/RNG state and persistent shuffled iterator before continuing.
- New SITReg GNCC² at 0.1/exact 1over3/1/3 on GPUs 0/1/2/3, seed42,
  200 epochs, 10,000 joint bidirectional AMSGrad updates, LR1e-4 and native
  diffusion. Only signed GNCC becomes squared GNCC. Native backend retained.
- Full strict native/reference preflight, automatic 400-case detailed evaluation,
  14,000 label records and three-loss comparison/2x3 ARC. No test-based checkpoint
  selection, no reference retraining, no changes to other GPU jobs.
- Spec/log: `experiments/2026-10-13_gncc2_recovery_and_sitreg/`.
- Queue key: `oasis_gncc2_recovery_then_sitreg_v1`.
- SITReg run/report key: `oasis_sitreg_gncc2_diffusion_training_lambda_sweep_v1`.

### P58: FeDDA squared global NCC training lambda sweep

- Recovery note 2026-10-13: original jobs stopped on a hash guard after one
  trailing blank line was removed from the evaluator during a source commit.
  Checkpoints are intact. User approved isolated-source recovery and subsequent
  SITReg GNCC²; see P59. Original failed artifacts remain immutable.

- Status: production launched 2026-10-12 17:05:11 PDT
  (2026-10-13 00:05:11 UTC); controller PID 96452, workers on GPUs 0/1/2/3.
  CPU tests and GPU 3 preflight passed; detailed evaluation/report auto-queued.
- Four fresh FeDDA-1MF trainings at lambda 0.1/exact 1over3/1/3 on GPUs 0/1/2/3,
  followed by automatic best-validation-checkpoint first100 evaluation.
- Only replace signed `1-rho` with per-image `1-rho^2`; population moments,
  FP32 variance-product floor, ordinary autograd, no similarity rescaling.
- Preserve native unmasked voxel L2 diffusion, seed 42, 200 epochs, 50 pairs per
  epoch, two sequential directional Adam AMSGrad updates per pair, 20,000
  updates, constant LR 1e-4, forward validation, initialization/pair hashes.
- Explicit historical backend override: cuDNN benchmark True, deterministic
  False, cuDNN TF32 True, matmul TF32 False. No bitwise reproducibility claim.
- Require 400 new case records and 14,000 per-label rows; compare immutable
  GNCC four-lambda and LNCC five-lambda references on the same 100 pairs.
- Final three-loss report: 13 settings, 1,300 case records, detailed accuracy
  and foreground regularity; primary ARC 2x3 has Dice/HD95/symmetric ASD versus
  foreground std(log J), means above and marginal sample-SD ellipses below.
- Preserve historical sources/results; new adapters and artifact identities.
  No range extension, retraining of references, or extra seeds in this plan.
- Run key: `oasis_fedda_gncc2_diffusion_training_lambda_sweep_v1`;
  report key: `oasis_seed3_first100_fedda_three_ncc_v1`.
- Spec/log: `experiments/2026-10-12_oasis_fedda_gncc2_training/`.
- Formula audit: `docs/implementation/ncc_loss_normalization.md`.
- Preflight: original parity tolerances passed without changes; four fresh
  directional smokes and restored Adam continuations, warm/trained updates,
  and rank 0/10 serialization/35-label metrics passed. Peak reserved 21.2168 GiB;
  production free-memory gate 24 GiB. Historical numerical sources unchanged.

### P57: FeDDA-1MF Signed Global NCC Training Lambda Sweep

- Status:production launched2026-10-12 08:12:09UTC (01:12:09PDT),
  controllerPID95152; four workers on GPUs0/1/2/3 with finite initial updates.
- GNCC0.1/exact1over3/1/3 on GPUs0/1/2/3;original2854079-parameter FeDDA-1MF,
  seed42,200epochs,50pairs/epoch,TWO sequential directional AdamAMSGrad
  updates/pair,20000updates,constantLR1e-4,native voxel L2diffusion.
- Reuse stateless P56 signed populationGNCC; retain original model,preprocessing,
  forward validation,initialization,pair sequence and historical CUDA override.
  Separate new adapters; preserve old sources/results,failed gates and GPU jobs.
- Four best-validation-checkpoint first100 evaluations;400cases/14000labelrows.
  Compare immutableLNCC0.1/1over3/1/3/10;historicalLNCC1 explicitly marked.
- All32 CPU tests passed. Independent CPU qualification verified a complete
  zero-issue GPU3 audit:exact historical and native/adapter field equality,
  unchanged validation/metric gates,four fresh sequential-update/checkpoint
  smokes,trained gradient smoke and ranks0/10 detailed serialization. Peak
  reserved21.2168GiB; production free-memory gate24GiB. Preserved dependency
  and historical-CUDA-variability preflight failures; no tolerance changes.
- Full-cohort CSV/JSON/Markdown and2x3ARC:mean curves top,across-caseSD ellipses
  bottom;Dice/HD95/symmetricASD versus foregroundstdlogJ,folding separate.
- Run key:`oasis_fedda_gncc_diffusion_training_lambda_sweep_v1`;
  report key:`oasis_seed3_first100_fedda_gncc_vs_lncc_v1`.
- Spec/log:`experiments/2026-10-12_oasis_fedda_gncc_training/`.

### P56: SITReg Signed Global NCC Training Lambda Sweep

- Status: completed/accepted2026-10-12; four trainings,400 new case evaluations
  and14000 per-label rows. Launched2026-10-11 18:40:21UTC (11:40:21PDT),
  controller PID93592; four workers on GPUs0/1/2/3.
- FreshGNCC1-rho lambda0.1/exact1over3/1/3 on GPUs0/1/2/3; native joint
  bidirectional diffusion,seed42,200epochs,10000AdamAMSGrad updates,LR1e-4.
- Preserve all historicalLNCC sources/results and explicitly retain native
  backend override. Verify initialization and per-epoch pair-stream hashes.
- Automatic best-validation-checkpoint first100 evaluation; all35 label rows
  percase,3accuracy/2foregroundregularity metrics; compareLNCC0.1/1over3/1/3/10.
- HistoricalLNCC1 remains contextual. New output identities, full preflight
  gates, source hashes, no automatic retraining/range expansion.
- All27 CPU tests and full GPU3 preflight passed. Initial-state hash matches
  historicalLNCC; numerical checkpoint/30-pair validation parity passed unchanged
  tolerances. GNCC fresh/trained updates, serialization and70 label rows passed.
- Peak reserved36.49GiB; launch guard38GiB. Preserved an initial dependency-path
  failure; restored read-only historical isolatedpystrum, no shared installation.
- Artifact key:`oasis_sitreg_gncc_diffusion_training_lambda_sweep_v1`.
- Spec/log:`experiments/2026-10-11_oasis_sitreg_gncc_training/`.
- Validation-selected GNCC lambda1/3 (Dice0.81225398,epoch193); not a boundary.
  User-requested presentation export:`arc_lines_top_v1/` under the report root,
  with mean curves top and across-case SD ellipses bottom. All nine operating
  points use100 matched cases; historicalLNCC1 explicitly marked. This export
  preserves the original accepted report and hash-locked experiment sources.

### P55: Full-Resolution Signed/Squared GNCC Dense Sweep

- Status: production running since2026-10-11 13:55:09UTC (06:55:09PDT);
  controller PID89865, four workers on GPUs0-3. IDIR deferred.
- 140 fresh fits: GNCC1-rho and GNCC1-rho² x lambda.01/.1/.3/1/3/10/100 x first10.
- P43/P44 full-resolution cps8/4/2 then direct dense, normalized LR.03/.01/.003/.001,
  unchanged adaptive Adam and unmasked total voxel-field FD diffusion.
- Population moments, ordinary autograd, documented epsilon; no historical
  StableStd or initial-loss normalization. Common-field calibration is diagnostic.
- GPUs0-3, one added worker/GPU, UID:GID1009:1020; other workloads untouched.
- Gates: unit contracts, three full-geometry smokes, LNCC rank0/lambda1 reproduction.
- All38 tests and three GPU smokes passed. Full LNCC reproduction had exactly
  zero field/metric differences and identical stopping/best epochs. Calibration
  completed30 common-field states; no training weights were rescaled.
- Require140 finite fields/4900 label records; compare70 accepted full-loss LNCC
  and20 VFA/SITReg references. Three-NCC tables plus2x3 ARC: three accuracy columns,
  stdlogJ x-axis, ellipses top and lines-only bottom. Eight-loss context is separate.
- Artifact key:`oasis_seed3_first10_mrdbscp842_dense_fullres_gncc_signed_squared_diffusion_v1`.
- Spec/log:`experiments/2026-10-11_oasis_gncc_dense_sweep/`.

### P54: Fourier-Input IDIR Bandwidth/Diffusion and Capacity Controls

- Status: running since 2026-10-11 05:54:37 UTC (2026-10-10 22:54:37 PDT).
  Controller PID79480; four workers verified on GPUs0/1/2/3. All38 IDIR tests
  and four fresh GPU smokes passed, with exact native-control field reproduction.
- Preserve P53 sampled global1-NCC/analytic voxel diffusion, 10000coordinates,
  Adam1e-4/2500updates/terminal checkpoint. No dense training or localNCC9.
- Initial230 new fits: six Fourier bandwidths x three lambdas x first10 (180),
  two additional native-control lambdas (20), width286 capacity controls (30).
  Read-only P53 omega32/64 lambda1 references; no old outputs overwritten.
- Fixed64 Gaussian sine/cosine pairs plus XYZ, native fan-in initialization;
  audit coordinate-chain-rule derivatives and initial activation/gradient scales.
- All four GPUs0–3, one new worker/GPU, UID:GID1009:1020; other jobs unchanged.
- Full-cohort boundary extension and explicitly gated optional confirmation;
  five case metrics and all35label records, hashes and persistent monitoring.
- Artifact key: `oasis_seed3_first10_idir_fourier_sampled_diffusion_v2`.
  Rejected v1 qualification preserved; pre-training diagnostic autograd was
  moved to a saved-initialization CPU copy after training to restore exact
  native-control reproduction. No tolerance relaxation or objective change.
- Spec/log: `experiments/2026-10-10_oasis_idir_fourier_diffusion/`.

### P53: Sampled IDIR Diffusion-First Then Frozen-Lambda Frequency Search

- Status: completed/accepted, 110 successful fits and3850label records in36.7min.
  Selected lambda1 atomega32 (interior), then omega64 by mean Dice at frozenlambda.
  Historical launch: 2026-10-10 21:54:45 UTC (14:54:45 PDT), after 21
  unit/canonical-evaluator tests and four concurrent 100-update smokes passed.
  Detached controller PID 73655; production slots GPU 1/1/3/3.
- Preserve10,000 fixed-foreground random coordinates/update, original SIREN,
  pair-minmax/zero padding, seed1, Adam1e-4,2500updates, terminal checkpoint.
- Signed sampled global1-NCC plus analytic voxel-coordinate diffusion; NOT
  full-volume/localNCC9 or forward-FD. No extra batch-size divisor for diffusion.
- Atomega32, initiallambda1e-3..1e3 across first10 (70fits). Select highest
  mean Dice, ties ASD/HD95; extend a boundary winner one decade at a time.
  Then freeze selectedlambda and testomega8/16/32/64/128, adding40fresh fits.
  Full2Dsweep postponed. All selection requires complete validated10case cohorts.
- Four concurrent slotsGPU1/1/3/3, UID:GID1009:1020; other jobs unchanged.
  Read-only BE scale calibration and four100-update co-resident smokes gate runs.
- Five canonical case metrics,35labels/case, sample-sequence hashes, full
  checkpoints/fields/history, incremental reports and logged adaptive decisions.
- Artifact key:`oasis_seed3_first10_idir_sampled_global_ncc_analytic_diffusion_staged_v1`.
- Spec/log:`experiments/2026-10-10_oasis_idir_sampled_diffusion_staged/`.

### P52: IDIR SIREN Frequency × Full-Resolution NCC9/Diffusion Lambda — CANCELLED

- Status:cancelled by the user2026-10-10. The full-volume training domain
  exceeded the intended old10,000-point IDIR sampling workflow. Controller
  PID72359 and workers72493/72494 were stopped and verified absent; existing
  other jobs untouched. Active outputs cleared into recoverable artifact-store
  trash (about411MiB); no review directory existed. Superseded by sampled P53.
- Historical launch:2026-10-10 21:22:27UTC (14:22PDT), after36CPU contracts
  and five full-volume GPU1/3 smokes. No accepted production result is claimed.
- 250fresh registrations:omega8/16/32/64/128 ×lambda0.1/1over3/1/3/10 ×
  canonical seed3first10. Old omega32 sampled-global-NCC/BE outputs not reused.
- Standalone native SIREN256×3, seed1, raw-image full-volume NCC_vfa9 and
  total voxel-field forward-valid diffusion; Adam1e-4,2500updates, terminal
  checkpoint primary. No Fourier features, analytic BE or B-spline prefix.
- Exact-chain-rule two-pass coordinate chunking; native loss/gradient and
  monolithic/chunked update parity plus five full-volume smokes gate production.
- GPU1/3 only, one new worker/GPU, UID:GID1009:1020; preserve existing jobs.
- Save five accuracy/regularity case metrics and all35label rows/case, fields,
  checkpoints, histories, source/input hashes; final250case/8750label acceptance.
- Artifact key:`oasis_seed3_first10_idir_siren_omega_ncc9_diffusion_2500_v1`.
- Spec/log:`experiments/2026-10-10_oasis_idir_omega_ncc9_diffusion/`.

> Historical LungCT dataset paths below record the locations used when each run
> executed. Since 2026-07-11, their current pTVReg-equivalent root is
> `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`; see
> `docs/migrations/lungct_dirlab_2026-07-11.md`.

Every real run should get an entry before launch. Generated run and review
artifacts of every size belong under `${ARTIFACT_ROOT}`, not Git.
`experiments/` is source-only and may retain execution specifications, technical
logs, and output-free notebook sources. Historical rows below intentionally
retain their executed paths; they are not evidence that those paths remain
valid after the completed external-artifact-store migration. See
`docs/migrations/artifact_store_2026-09-02.md` and its symbolic manifest
mapping. The former source data tree remains rollback-only through the recorded
30-day retention date and requires separate explicit approval for deletion.

## Required Fields

```text
Run ID:
Status:
Owner/agent:
Date planned:
Date started:
Date completed:
Server:
GPU model/count:
Git commit:
Branch:
Docker image tag/digest:
Dataset root:
Dataset manifest:
Split config:
Standards preflight:
Protocol tag:
Command:
Artifact root: `${ARTIFACT_ROOT}`
Run/output root: `${ARTIFACT_ROOT}/runs/<run-id>`
Review artifact root: `${ARTIFACT_ROOT}/repo_experiments/<experiment-key>`
Artifact manifest/verification:
Primary metrics:
Known limitations:
Next decision:
```

## Planned Priority Runs

### P2ah: OASIS-100 fixed-50 CP-BE evaluation

- Status: completed and accepted on 2026-09-25. Run
  `oasis_cpbe_fixed50_first100_20260925T052403Z_v1` contains 202/202 clean
  timing rows, 200/200 new evaluations, and complete 100-rank historical
  quality anchors. The external Markdown and machine-readable reports passed
  acceptance; paper/Table-1 material remains unchanged pending review.
- Record: `experiments/2026-09-24_oasis_cpbe_fixed50_first100/`.
- New registration arms: fixed 50/50/50 cache+batch+CP-BE with Adam, and fixed
  50/50/50 CP-BE with AMSGrad. Both use spacing 8/4/2, downsampling 4/2/1,
  stage LR .03/.01/.003, alpha1000, seed1, float32 and cuDNN F/T.
- Protocol: seed-3 shuffled ranks0–99; one automatically selected idle physical
  GPU locked for timing; per arm rank0 cold, repeated rank0 warm, ranks1–99
  warm; endpoint case_load_complete through dvf_file_ready.
- Evaluation: 200 new warm fields only, after timing, with adaptive 16-versus-32
  persistent workers across idle GPUs. Keep 32 only for >=10% throughput gain
  with host/GPU memory safety gates.
- Comparison: existing OASIS-100 MR_A_fast_decay and canonical alpha1000
  2500-step analytic-BE quality summaries. Baselines are not rerun. Controlled
  first10 GPU3 F/T timings are context-only, not matched full100 runtime.
- Outputs: `${ARTIFACT_ROOT}/runs/oasis_cpbe_fixed50_first100_<timestamp>`
  and the matching external review root under the experiment key.
- Acceptance: 202 clean timing rows, 200 valid new DVFs/evaluations and exactly
  100 unique historical quality rows per baseline; first100 plus ranks10–99
  summaries, paired case/label differences and complete provenance.


### P2ag: Case04 fixed-50 cache/batch CP-BE presentation animation

- Status: accepted `render_v6` and dense-cadence extension complete. The
  original accepted delivery is `render_v6` under run
  `mrdbscp_progress_case04_fixed50_20260923T021216Z_v2`; `render_v1` through
  `render_v5` are retained non-delivery layout/report audits.
- Record: `experiments/2026-09-22_mrdbscp_fixed50_cached_batch_animation/`.
- Scope: OASIS seed-3 rank3 (Case04, fixed0117/moving0322), fixed50/50/50
  stages, cache+batch+CP-BE, alpha1000 seed1, cuDNN F/T, physical GPU0.
- Timing: by user-authorized amendment, no new benchmark was run. The video
  reuses the accepted Case04 headline warm endpoint `1.893926 s` and measured
  stage spans `0.522098/0.552847/0.634517 s` from P2af. A concurrent untimed
  snapshot run supplied fields only; its `23.531749 s` wall time is excluded.
- Presentation: new non-overwriting 1080p/30fps artifact, compact left image
  grid and right combined loss panel; display similarity is1-NCC and display
  total is `(1-NCC)+1000*BE`, a +1 display shift only. The final 15-second
  phase-aware replay gives 12 seconds to optimization, proportionally by the
  three accepted stage spans, and compresses setup/transitions/output. Visible
  status text distinguishes preparation, optimization and final serialization.
- Outputs: `${ARTIFACT_ROOT}/runs/mrdbscp_progress_case04_fixed50_<timestamp>`
  and matching review root under the experiment key.
- Limitations: single-case teaching artifact; labels/metrics are offline and do
  not guide optimization; only stage totals were measured, so optimization
  steps are evenly positioned within each accepted historical stage span and
  the tiny aggregate preparation residual is explicitly allocated equally;
  CP-BE numerical-equivalence warning retained.
- Acceptance: 150 loss values matched the historical trajectory exactly; final
  DVF file and array hashes matched; 18 reconstructed states passed with zero
  final reconstruction error. The final H.264/yuv420p video is 1920x1080,
  30 fps, 15 s and 450 frames. Independent decoded-frame checks confirmed
  moving loss curves, static held images, and pixel-identical frozen columns.
- Final quality transferred only after exact array-hash matching: Dice
  `0.830247`, HD95 `1.700792 mm`, foreground SDlogJ `0.836015`, foreground
  folding `4.202109%`.
- Dense-cadence extension: one new untimed GPU0 capture saves every optimizer
  step and must reproduce all 150 losses and the final accepted DVF exactly.
  One offline reconstruction is then subsampled into measured field cadences
  of 5, 2 and 1 step(s), producing non-overwriting
  `animation_step5.mp4`, `animation_step2.mp4`, and `animation_step1.mp4`.
  All retain the same accepted warm endpoint and stage clock; transition/setup
  labels are removed so only optimizing stage/step and final DVF-ready status
  are shown.
- Dense-cadence accepted run:
  `mrdbscp_progress_case04_dense_step1_20260923T044548Z`. The capture and
  153-state evaluation were clean; final reconstruction RMS/max error was zero.
  Accepted outputs are `step5_v2/animation_step5.mp4`,
  `step2/animation_step2.mp4`, and `step1/animation_step1.mp4` beneath the
  matching review run. Each is 1920x1080, 30 fps, 15 s and 450 frames; all
  renderer, cadence, decoded-frame, frozen-column and delivery checks passed.

### P2af: OASIS CP-BE accuracy-first overnight acceleration, GPU 0

- Status: running automatically; v3 passed8 CPU contracts,12 GPU schedule
  smokes and a clean35-label metric smoke after two preserved preflight-only
  corrections. Formal reference registration started2026-09-21 09:39UTC.
  Stop by09:00 Pacific local time, with earlier GPU cutoff and report reserve.
  No headline cohort conclusions yet.
- Record: experiments/2026-09-21_oasis_cpbe_accuracy_speed_gpu0/.
- Twelve primary candidates: CP-BE fast reference, final-stage cap, fixed
  budgets, looser adaptive stopping, cosine/step LR, and16/8/4/2 variants.
- Protocol: seed3 shuffled first10, alpha1000 seed1, F/T, six CPU threads,
  sequential physical GPU0, fresh worker cold+10warm, separate10-field metrics.
- Primary Dice/HD95; final full-resolution NCC, FD BE/diffusion, foreground/full
  SDlogJ, folding and positive fractions; per-case/label deltas and explicit gates.
- Continuous contamination monitoring/retries, hard scoped-worker deadline,
  incremental reports and complete provenance. Extensions only after core work.
- Outputs: ${ARTIFACT_ROOT}/runs/oasis_cpbe_accuracy_speed_<timestamp> and
  ${ARTIFACT_ROOT}/repo_experiments/2026-09-21_oasis_cpbe_accuracy_speed_gpu0/<run-id>.
- Limits: tuning, not held-out validation; CP-BE non-equivalence warning retained.
  No paper edits, existing artifact overwrite or commit.

### P2ae: MR-D-BSCP CP-BE oral-presentation progress animation

- Presentation revision completed: final `render_v4/` with geometric white
  pull-coordinate grid contours and loss playback independent of field snapshot
  cadence. CPU-only reconstruction/rendering; previous versions and measured
  numerical results retained. Twenty CPU tests, fifteen render checks and eight
  independent encoded-frame/local-player/frozen-column checks passed. `render_v3/`
  retained; v4 clarifies field-snapshot versus DVF-ready timestamps. Per-video-frame
  mapping, source snapshots, exact command and checksum manifests retained.

- Status: completed and acceptance-checked. User-selected Case 04 (rank 3,
  fixed 0117 / moving 0322) has 49 snapshots, 450 loss rows, a 30-second 1080p
  animation and offline player. Accepted run:
  `mrdbscp_progress_case04_20260920T223559Z_v2`, primary presentation `render_v4/`.
  Preview ID:
  `preview_20260920T221447Z_v1` (ten 1080p panels, overview and offline gallery).
- Verification: ten CPU contracts and thirteen delivery checks passed; source,
  DVF and input hashes, Docker provenance, commands and test logs retained.
- Case-04 continuation verification: fourteen CPU contracts and fifteen final
  delivery checks passed; zero contamination, exact baseline/clock/capture loss
  and field parity, exact final control reconstruction. Initial smoke export
  bug and its corrected fresh run are documented; failed artifacts retained.
- Record: experiments/2026-09-20_mrdbscp_progress_animation/.
- Source: accepted P2ad CP-BE-only headline warm outputs, seed-3 first10 ranks
  0–9. Static preview uses CPU in Docker; no registration or GPU work.
- Next phase: selected-case MR_A_fast_decay+cpbe, alpha1000 seed1, cuDNN F/T,
  sequential GPU0; original adaptive three-stage schedule, separate clean warm
  timing and snapshot runs, offline full-volume Dice and presentation rendering.
- Deliverables: case-selection PNG/HTML gallery first; after selection, frozen
  stage-column animation, real warm clock, per-step loss curves, MP4/HTML,
  source/data provenance and machine/human-readable reports.
- Outputs: ${ARTIFACT_ROOT}/repo_experiments/2026-09-20_mrdbscp_progress_animation/<run-id>;
  future capture under ${ARTIFACT_ROOT}/runs/mrdbscp_progress_<timestamp>.
- Known limits: demonstration, not cohort evidence; prior CP-BE non-equivalence
  warning retained. No paper changes, production-default changes or commit.

### P2ad: MR-D-BSCP cache/batch/CP-BE first10 and component timing, GPU 0

- Status: completed and acceptance-checked; 33 clean headline + 33 clean
  diagnostic registrations, 33 metric records, zero contaminated attempts.
- Diagnostic/headline fields, steps and stop categories match exactly; the
  eleven combination fields are byte-identical to CP-BE alone. The existing
  BE-versus-original non-equivalence warning remains unchanged.
- Record: experiments/2026-09-20_mrdbscp842_cache_batch_cpbe_gpu0/ (UTC date).
- Arms: original MR_A_fast_decay, CP-BE alone, cache+batch+CP-BE; alpha 1000,
  seed 1, seed3 shuffled first10, original adaptive stopping, cuDNN F/T.
- Sequential physical GPU 0: fresh worker per arm, rank0 cold, rank0 warm,
  ranks1-9 warm; 33 endpoint-only and 33 separate component-instrumented jobs.
- Separate evaluation: Dice, HD95, positive-determinant foreground/full SDlogJ,
  folding, field differences, and diagnostic/headline parity. Existing BE
  non-equivalence warning is retained, not an exclusion gate.
- Outputs: ${ARTIFACT_ROOT}/runs/mrdbscp842_be_combo_<timestamp> and matching
  external review root keyed by the source experiment directory.
- Raw fields, logs, source snapshots, reports and JSON/CSV remain external.
- No paper edits, production-default changes, or commit.

### P2ac: MR-D-BSCP 8-4-2 acceleration and BE quality characterization, GPU 0

- Status: completed and acceptance-checked 2026-09-19; external completion
  queue and acceptance manifest verified before P2ad. Source log updated.
- Scope: MR_A_fast_decay only; first10 OASIS, alpha 1000, seed 1,
  cuDNN benchmark=False/deterministic=True, sequential physical GPU 0.
- Source reference: HEAD and all modified source hashes/snapshot are captured
  in the external execution manifest before numerical workers start.
- Protocol: CPU/GPU correctness, fresh GPU-0 reference and A/A, isolated
  caching/layout/history/CP-BE pilots, gated combination, first10 confirmation.
- Record: experiments/2026-09-19_mrdbscp842_acceleration_gpu0/.
- Outputs: ${ARTIFACT_ROOT}/runs/mrdbscp842_acceleration_gpu0_<timestamp>
  and ${ARTIFACT_ROOT}/repo_experiments/2026-09-19_mrdbscp842_acceleration_gpu0/<timestamp>.
- Acceptance: finite normalized XYZ float32 DVFs, gradient/optimizer/stopping
  parity for strict-equivalence promotion, paired fixed-step and adaptive runtime,
  separate Dice/Jacobian metrics and BE-arm HD95,
  fail-closed GPU contamination monitoring, machine-readable decisions and report.
- User amendment: characterize BE registration on all first10 cases even when
  strict FP32/Adam stress tests fail; retain the warning and report empirical
  Dice/HD95 changes separately from numerical equivalence.
- Limits: first10 diagnostic; later solver/precision/framework experiments are
  deferred; no paper changes or commit authorized.

### P2x: COPD P2e-Geometry isoTV `0.055` All10 Screen

- Run ID: `copd_p2e_geometry_w0p055_all10_v1`.
- Status: completed 2026-08-31; isolated candidate screen, not a replacement
  for the completed COPD P2e `w=0.11` reference.
- Owner/agent: Codex.
- Date planned: 2026-08-31.
- Date started: 2026-08-31.
- Date completed: 2026-08-31.
- Server: local project GPU server; no private hostname recorded.
- GPU model/count: GPU `0` only; all ten cases run sequentially.
- Git commit: `8302f7b9d244` plus unrelated dirty worktree; execution-script
  SHA-256 hashes are recorded in the experiment specification.
- Branch: `gpu8`.
- Docker image/container: `hengjieliu96/dl_dir:py310sitreg_v1` /
  `py310sitreg_v1_test1`.
- Dataset root: `${LUNGCT_PTVREG_ISO1MM}`.
- Dataset manifest: canonical pTVReg-style 1-mm COPD cases `01-10`.
- Split config: all COPD cases `01-10`; moving `eBHCT` to fixed `iBHCT`.
- Standards preflight: reviewed `docs/standards/README.md`,
  `docs/standards/fair_experiment_checklist.md`,
  `docs/standards/lungct_protocol_tags.md`,
  `docs/standards/lungct_tre_metric_conventions.md`, and the LungCT protocol
  warnings.
- Protocol tag:
  `ptv_landmark_crop_1mm_no_lungmask+directcp_p2e_geometry_w0p055_candidate`.
- Method/objective: direct cubic B-spline control-point optimization without an
  INR; pTV local CC (`sigma=2.1 mm`) plus physical isoTV `0.055`,
  `csqrt=5e-3`, and a five-voxel border domain.
- Frozen geometry/schedule: controls `64,32,16,8,4`; image downsampling
  `16,8,4,2,1`; stage epochs `400,400,400,400,900`; Adam `lr=1e-3`; seed `1`.
- Command: launcher/runner/aggregator compiled; GPU-`0` was idle; dry-run
  queued exactly ten sequential GPU-`0` jobs with the frozen settings. The
  isolated case-01 `1,1,1,1,1` smoke completed with finite fields/TRE and the
  recorded `0.055` weight. The full all-ten-case sequential GPU-`0` cohort
  completed and was aggregated in the dedicated review folder.
- Output root: `${SCRATCH_ROOT}/runs/copd_p2e_geometry_w0p055_all10_v1`.
- Review root:
  `experiments/2026-08-31_copd_p2e_geometry_w0p055_all10/`.
- Primary metrics: native-crop snap-to-voxel TRE, native-crop subvoxel TRE,
  registration-grid diagnostics, posthoc regularity, elapsed time, and peak
  memory.
- Known limitations: this is a new COPD regularization candidate.  It is not a
  method-only comparison with pTVReg and will not update the frozen P2e or the
  cross-dataset report unless explicitly selected after review.
- Acceptance: smoke must produce finite loss/field/TRE outputs with the exact
  requested settings; full cohort requires `10/10` completed case summaries,
  finite metrics, expected final field grids, and a separately labelled
  aggregate.
- Result: all `10/10` summaries/fields passed the frozen-setting, 300-landmark,
  finite-metric, and final-grid checks. Native snap TRE is `1.0501 +/- 0.7331`
  mm using sample SD across case means, versus P2e `w=0.11`
  `1.0505 +/- 0.8088` mm. Mean paired candidate-minus-P2e change is
  `-0.0004 +/- 0.1325` mm (sample SD across cases).
- Next decision: review the completed candidate casewise TRE before deciding
  whether to add it to any comparison report.

### P2w: ptvreg-python COPD Fixed All10 Trial

- Run ID: `ptvreg_python_copd_fixed_all10_v1`.
- Status: adapter/preflight implemented 2026-08-28; production all-10 launch
  gated after reduced 1mm smoke did not return cleanly and native-grid case08
  showed very high memory use.
- Owner/agent: Codex.
- Date planned: 2026-08-28.
- Date started: preflight started 2026-08-28.
- Date completed: pending.
- Server: local project GPU server; no private hostname recorded.
- GPU model/count: four GPUs requested for production launch, with cases
  parallelized one case per GPU.
- Git commit: `8204459` plus dirty source tree until the adapter changes are
  committed or otherwise snapshotted by source hashes.
- Branch: `gpu8`.
- Docker image/container: `hengjieliu96/dl_dir:py310sitreg_v1` /
  `py310sitreg_v1_test1`.
- Dataset roots: `${LUNGCT_PTVREG_ISO1MM}` for canonical 1mm inputs, plus raw
  `${LUNGCT_DATA}/raw/COPD` for paper-HU native-crop inputs.
- Dataset manifest: canonical pTVReg-style 1 mm COPD cases `01-10`; raw COPD
  case files and native lung masks are used for the native-crop gate.
- Split config: all COPD cases `01-10`; moving `eBHCT` to fixed `iBHCT`.
- Standards preflight: reviewed `docs/standards/README.md`,
  `docs/standards/fair_experiment_checklist.md`,
  `docs/standards/lungct_protocol_tags.md`,
  `docs/standards/lungct_tre_metric_conventions.md`, and LungCT protocol
  warnings.
- Protocol tags:
  `ptvreg_python_admm+canonical_ptv_landmark_crop_1mm_no_lungmask` and
  `ptvreg_python_admm+paper_hu_landmark_crop_1mm_no_lungmask`, plus
  `ptvreg_python_admm+paper_hu_native_landmark_crop_no_lungmask`.
- Method/objective: public `external/ptvreg-python` ADMM registration imported
  through `ptv.register_min.register_pair_min(moving, fixed, opts)`.
- Fixed options: `opt_method=admm`, `grid_spacing=(8,8,8)`, `isoTV=0.12`,
  `metric=lcc`, `metric_param=(2.1,2.1,2.1)`,
  `scale_metric_param=True`, `pix_resolution=(1,1,1)`,
  `spline_order=1`, `k_down=0.7`, `border_mask=5`, `interp_type=1`,
  `dtype=float64`. The native-crop protocol overrides `pix_resolution` per case
  with the raw COPD native spacing.
- Command: preflight tests and loader audit first; synthetic CUDA smoke passed;
  reduced full-volume case01 smoke was attempted and manually stopped after it
  remained inside the external registration call with no completed summary.
  Prepared production all-10 canonical run with
  `scripts/experiments/ptvreg_python_repro/launch_copd_fixed_all10.py --protocol canonical_1mm --gpus 0,1,2,3`;
  then repeat the same fixed all-10 launch with `--protocol paper_hu_iso1mm`.
  Native-crop paper-HU dry-run is prepared with `--protocol paper_hu_native
  --gpus 0,1,2,3`, skipping completed case08 by default.
- Output root:
  `/database/inrorspline/runs/ptvreg_python_copd_fixed_all10_v1`.
- Review root:
  `experiments/2026-08-28_ptvreg_python_copd_fixed_all10/`.
- Primary metrics: native-crop snap TRE, native-crop subvoxel TRE, cropped-1mm
  or native registration-grid snap/subvoxel TRE as appropriate, foreground
  stdLogJ, foreground folding percentage, bending energy, diffusion L2,
  runtime, and peak memory.
- Known limitations: `ptvreg-python` may not be numerically identical to
  MATLAB pTVReg; canonical inputs retain this repo's regenerated pTVReg-style
  crop/interpolation conventions; paper-HU input changes intensity only while
  reusing the same crop/landmark geometry.
- Preflight result: adapter tests passed (`12/12`), canonical loader audit passed
  (`10/10`), four GPUs visible, canonical and paper-HU four-GPU dry-run
  manifests generated, synthetic CUDA registration smoke passed, and paper-HU
  inputs were generated/verified for all ten cases.
- Native-crop gate result: COPD case08 completed with native paper-HU crop shape
  `[100,299,409]`, spacing `[2.5,0.586,0.586]`, headline
  `native_crop_tre_snap_to_voxel=2.8865 mm`, subvoxel TRE `3.0143 mm`,
  identity TRE `26.4569 mm`, elapsed `914.5 s`, and peak GPU memory
  `46.72 GiB`. This is a one-case gate, not a production all-10 result.
- Acceptance: require `10/10` completed cases, finite TRE/regularity, dense
  field shape equal to input shape, no NaNs/infinities, all four TRE
  conventions present, and no production launch if field-conversion tests fail.
- Next decision: after canonical and paper-HU aggregate, decide whether the new
  public Python pTVReg row is a useful contextual baseline or only a protocol
  diagnostic.

### P2v: OASIS Paper Cold/Warm DVF Timing Rerun

- Run ID: `oasis_paper_cold_warm_timing_alpha1000_v1`.
- Status: completed 2026-08-15 after dry-run, smoke, full timing, and
  aggregation.
- Owner/agent: Codex.
- Date planned: 2026-08-15.
- Date started: 2026-08-15.
- Date completed: 2026-08-15.
- Server: local project GPU server; no private hostname recorded.
- GPU model/count: four NVIDIA RTX 6000 Ada GPUs observed inside
  `py310sitreg_v1_test1`.
- Git commit: `68d2c4cc02ae465f2a9b23dfb965ee46ca864f41`, dirty source tree.
- Branch: `gpu8`.
- Docker image/container: `hengjieliu96/dl_dir:py310sitreg_v1` /
  `py310sitreg_v1_test1`.
- Dataset root: `${OASIS_ROOT}` / default container OASIS split root.
- Dataset manifest: OASIS BrainMR seed-3 shuffled `pair_test_200`, ranks `0-9`.
- Split config:
  `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Standards preflight: reviewed `docs/standards/README.md`,
  `docs/standards/fair_experiment_checklist.md`, and
  `docs/standards/oasis_brainmr_protocol.md`.
- Protocol tag:
  `oasis_seed3_first10_dvf_only_loaded_images_to_saved_dense_dvf_cold_warm_timing`.
- Methods: five selected fast schedules
  (`MR_A_fast_decay`, `D2_A_bal_lr10x`, `D4_A_bal_lr10x`,
  `S2_A_bal_lr10x`, `S4_A_bal_lr10x`) and their five 2500-step baselines.
- Timing convention: for each method, run rank `0` once as cold in a fresh
  worker, then rerun rank `0` and ranks `1-9` in the same worker as warm
  measured runs. This is process/GPU warmup timing, not registration warm-start.
- Command: dry-run, smoke, full fast phase, full baseline phase, and final
  aggregation with
  `scripts/experiments/oasis_alpha_sweep/run_oasis_paper_cold_warm_timing.py`.
- Output root:
  `/database/inrorspline/runs/oasis_paper_cold_warm_timing_alpha1000_v1`.
- Review root:
  `experiments/2026-08-15_oasis_paper_cold_warm_timing/`.
- Primary metrics: loaded-images-to-saved-DVF time, script-to-saved-DVF time,
  persistent worker wall time, actual optimizer steps, and peak GPU memory.
- Known limitations: OASIS-only timing; four-GPU execution may include
  GPU-to-GPU noise; runtime jobs are DVF-only and exclude Dice/HD95/ASD/dense
  NCC/regularity evaluation.
- Acceptance: passed. Dry-run planned `10` candidates and `110` measured
  registrations; smoke completed `20/20`; full timing completed `110/110`
  valid rows with zero missing/failed jobs, finite DVF endpoint times, existing
  dense DVF files, and `no_evaluation=true`.
- Result artifacts:
  `experiments/2026-08-15_oasis_paper_cold_warm_timing/paper_cold_warm_timing_report.md`
  and
  `experiments/2026-08-15_oasis_paper_cold_warm_timing/paper_runtime_table_values.csv`.
- Next decision: paste the refreshed compact runtime table into the camera-ready
  LaTeX source.

### P2y: OASIS MR-D-BSCP cuDNN Cold/Warm Ablation

- Run ID: oasis_mrdbscp_cudnn_ablation_20260901_v1.
- Status: completed and technically accepted 2026-09-01 (`20/20` timing,
  `20/20` Dice).
- Owner/agent: Codex.
- Date planned: 2026-09-01.
- Date started: 2026-09-01.
- Date completed: 2026-09-01.
- Server: local project GPU server; no private hostname recorded.
- GPU model/count: physical GPU 2 only; four configurations run sequentially.
- Git commit: recorded in the execution manifest; worktree was already dirty.
- Branch: gpu8.
- Docker image/container: hengjieliu96/dl_dir:py310sitreg_v1 /
  py310sitreg_v1_test1.
- Dataset root: OASIS_ROOT / default container OASIS split root.
- Dataset manifest: OASIS BrainMR seed-3 shuffled pair_test_200, ranks 0-4.
- Split config:
  configs/splits/oasis/pair_test_200_shuffled_seed000003.txt.
- Standards preflight: reviewed docs/standards/README.md,
  docs/standards/fair_experiment_checklist.md, and
  docs/standards/oasis_brainmr_protocol.md.
- Protocol tag:
  oasis_seed3_first5_dvf_only_mrdbscp_cudnn_matrix_cold_warm.
- Method: MR_A_fast_decay / direct MR-D-BSCP 8-4-2, alpha 1000, seed 1,
  using the existing candidate schedule.
- Timing matrix: all combinations of cuDNN benchmark and deterministic flags;
  rank 0 cold and ranks 1-4 warm in one persistent worker per arm.
- Command: syntax check and dry-run first; then the focused sequential GPU-2
  controller with deep DVF-only timing; posthoc Dice evaluation starts only
  after all timing summaries pass.
- Output root:
  `${SCRATCH_ROOT}/runs/oasis_mrdbscp_cudnn_ablation_20260901_v1`.
- Review root:
  experiments/2026-09-01_oasis_mrdbscp_cudnn_ablation/.
- Primary metrics: loaded-images-to-saved-DVF cold/warm time and gap,
  expansion/backward/loss-volume phase timings, setup.device time, actual
  optimizer steps, peak memory, effective backend metadata, and posthoc Dice.
- Result: benchmark=True/deterministic=False had cold 95.94 s, warm mean
  10.88 s, and gap 85.06 s; benchmark=False/deterministic=False had cold
  23.68 s, warm mean 10.25 s, and gap 13.43 s. Disabling benchmark reduced
  the gap by 84.2%; expansion and backward explained 70.19 s of the reference
  gap. Deterministic=True completed successfully; its gaps were 108.71 s with
  benchmark=True and 11.86 s with benchmark=False. All four Dice means were
  0.7838778 (SD 0.0490037).
- Known limitations: five-case diagnostic only; cold and warm use different
  cases; evaluation is separate from timing; no profiler is included unless
  the clean 2x2 result leaves the benchmark attribution inconclusive.
- Acceptance: passed 20/20 timing cases across four arms, one cold plus four
  warm cases per arm, complete timing summaries, 20 finite dense DVFs with
  shape and coordinate-convention checks, and 20/20 posthoc Dice evaluations.
- Next decision: the clean batch attributes most of the observed gap to
  benchmark-dependent first-use work in expansion/backward; no profiler is
  required for this batch because disabling benchmark reduced the gap by more
  than 20%. A profiler remains optional for kernel-level attribution.

### P2z: OASIS MR-D-BSCP Table-1 cuDNN Rerun on GPU 3

- Run ID: `oasis_mrdbscp_cudnn_table1_first10_gpu3_20260916_v1`.
- Status: completed and technically accepted 2026-09-16 (`33/33` timing,
  `30/30` Dice).
- Owner/agent: Codex.
- Date planned: 2026-09-16.
- Date started: 2026-09-16.
- Date completed: 2026-09-16.
- Server: local project GPU server; no private hostname recorded.
- GPU model/count: physical GPU 3 only; three configurations run sequentially.
- Git commit: `0d16f8f4fdb6fa252b9a858971a8d2d545aed7b2`; dirty source tree at launch.
- Branch: `gpu8`.
- Docker image/container: `hengjieliu96/dl_dir:py310sitreg_v1` /
  `py310sitreg_v1_artifact_store`.
- Dataset root: shared-storage OASIS split root, recorded in the execution
  manifest; not the legacy `/database/wavereg/...` path.
- Dataset manifest: OASIS BrainMR seed-3 shuffled `pair_test_200`, ranks `0-9`.
- Split config: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Standards preflight: reviewed the OASIS protocol, fair-experiment checklist,
  standards index, implementation notes, and prior P2y cuDNN ablation.
- Protocol tag:
  `oasis_seed3_first10_dvf_only_mrdbscp_cudnn_gpu3_table1_sequence`.
- Method: `MR_A_fast_decay`, direct MR-D-BSCP 8-4-2, alpha `1000`, seed `1`,
  with the existing decayed-stage learning rates and adaptive stopping.
- Configuration order:
  `benchmark_false_deterministic_true`,
  `benchmark_false_deterministic_false`,
  `benchmark_true_deterministic_false`.
- Timing sequence: one fresh worker per configuration; rank `0` cold, then rank
  `0` rerun plus ranks `1-9` warm, for `33` timing measurements total.
- Timing endpoint: `case_load_complete` through `dvf_file_ready`; DVF-only,
  including optimization and DVF output but excluding evaluation.
- Command source: `scripts/experiments/oasis_alpha_sweep/run_mrdbscp_cudnn_ablation.py`
  and `evaluate_mrdbscp_cudnn_ablation.py`, extended without changing P2y
  defaults.
- Output root:
  `/database/inrorspline/runs/oasis_mrdbscp_cudnn_table1_first10_gpu3_20260916_v1`.
- Review root:
  `/database/inrorspline/repo_experiments/2026-09-16_oasis_mrdbscp_cudnn_table1_first10_gpu3`.
- Primary metrics: cold endpoint, warm mean/SD, cold-minus-warm gap, phase
  timing breakdown, actual steps, peak memory, backend metadata, and sampled
  Dice over 30 unique case/configuration outputs.
- Acceptance: passed. Timing completed `33/33`; all dense DVFs were finite
  float32 normalized-grid XYZ arrays with shape `(160, 224, 192, 3)`; every
  timing JSON recorded matching requested/effective cuDNN flags; Dice completed
  `30/30` finite rows.
- Known limitations: first10 only; cold-gap differences combine backend
  first-use overhead and case-specific runtime; GPU-3 results are directly
  paired only within this rerun and not bitwise comparable to historical GPU-2
  timings.
- Experiment record:
  `experiments/2026-09-16_oasis_mrdbscp_cudnn_table1_first10_gpu3/`.
- Results: paired reference cold/warm/gap `110.20/9.23/100.97` s; disabling
  benchmark with deterministic `True` gave `26.49/9.31/17.19` s and with
  deterministic `False` gave `22.42/9.29/13.14` s. The generated report records
  absolute and percentage changes for both arms.
- Result artifacts:
  `/database/inrorspline/repo_experiments/2026-09-16_oasis_mrdbscp_cudnn_table1_first10_gpu3/`.
- Next decision: determine whether the GPU-3 reference and deterministic arm
  support updating the camera-ready runtime interpretation or only provide a
  reproducibility diagnostic.

### P2aa: OASIS Table-1 Candidate cuDNN Cold/Warm Rerun on GPU 3

- Run ID: `oasis_table1_candidates_cudnn_first10_gpu3_20260916_v1`.
- Status: completed and technically accepted 2026-09-17 (`165/165` timing,
  `150/150` Dice).
- Owner/agent: Codex.
- Date planned: 2026-09-16.
- Date started: 2026-09-16.
- Date completed: 2026-09-17.
- Server: local project GPU server; no private hostname recorded.
- GPU model/count: physical GPU 3 only; all candidates/configurations run
  sequentially.
- Git commit: `0d16f8f4fdb6fa252b9a858971a8d2d545aed7b2`; dirty source tree at
  launch.
- Branch: `gpu8`.
- Docker image/container: `hengjieliu96/dl_dir:py310sitreg_v1` /
  `py310sitreg_v1_artifact_store`.
- Dataset root:
  `/mnt/shengdata1/hengjie/workspace/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1`
  inside the recreated container mount.
- Dataset manifest: OASIS BrainMR seed-3 shuffled `pair_test_200`, ranks
  `0-9`.
- Split config: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Standards preflight: OASIS protocol, fair-experiment checklist, standards
  index, implementation notes, and prior P2z cuDNN rerun reviewed.
- Protocol tag:
  `oasis_seed3_first10_table1_candidates_dvf_only_cudnn_gpu3_sequence`.
- Candidates in Table-1 order: `MR_A_fast_decay`, `D2_A_bal_lr10x`,
  `D4_A_bal_lr10x`, `S2_A_bal_lr10x`, `S4_A_bal_lr10x`.
- Configuration order per candidate:
  `benchmark_false_deterministic_true`,
  `benchmark_false_deterministic_false`,
  `benchmark_true_deterministic_false`.
- Timing sequence: fresh worker per candidate/configuration; rank `0` cold,
  then rank `0` rerun plus ranks `1-9` warm, for `165` measurements total.
- Timing endpoint: `case_load_complete` through `dvf_file_ready`; DVF-only,
  including optimization and DVF output but excluding evaluation.
- Command source:
  `scripts/experiments/oasis_alpha_sweep/run_mrdbscp_cudnn_ablation.py` and
  `evaluate_mrdbscp_cudnn_ablation.py`, generalized from P2z while preserving
  the historical single-candidate defaults.
- Output root:
  `/database/inrorspline/runs/oasis_table1_candidates_cudnn_first10_gpu3_20260916_v1`.
- Review root:
  `/database/inrorspline/repo_experiments/2026-09-16_oasis_table1_candidates_cudnn_first10_gpu3`.
- Primary metrics: cold endpoint, warm mean/SD, cold-minus-warm gap, phase
  timing, setup-device time, actual steps, peak memory, effective backend
  metadata, and paired Dice over `150` unique saved DVFs.
- Acceptance: passed. Dry-run queued `165` jobs; all `15` one-step smoke jobs
  passed; timing completed `165/165`; all selected DVFs were finite float32
  normalized-grid XYZ arrays of shape `(160,224,192,3)`; requested/effective
  backend flags matched in every timing JSON; and separate Dice evaluation
  completed `150/150` rows.
- Known limitations: first10 only; backend timing differences are diagnostic
  and do not by themselves justify changing the camera-ready table. No
  camera-ready PDF or LaTeX source will be modified before user review.
- Experiment record:
  `experiments/2026-09-16_oasis_table1_candidates_cudnn_first10_gpu3/`.
- Results: paired reference cold/warm/gap were `102.52/9.38/93.14` s for
  MR-D-BSCP (8-4-2), `72.95/17.58/55.36` s for D-BSCP (cps2),
  `72.80/20.61/52.19` s for D-BSCP (cps4), `177.71/91.34/86.37` s for
  INR-BSCP (cps2), and `98.67/32.29/66.38` s for INR-BSCP (cps4). The two
  benchmark-disabled arms and all absolute/percentage paired changes are in
  the generated report.
- Result artifacts:
  `/database/inrorspline/repo_experiments/2026-09-16_oasis_table1_candidates_cudnn_first10_gpu3/`.
- Next decision: review the generated full report before deciding whether any
  camera-ready Table-1 values or prose should change.

### P2ab: OASIS Table-1 Base/Fast F/T Rerun on GPU 3

- Run ID: `oasis_table1_base_fast_ft_first10_gpu3_20260916_v1`.
- Status: in progress.
- Owner/agent: Codex.
- Date planned: 2026-09-16.
- GPU: physical GPU 3 only; all ten schedules run sequentially.
- Git branch: `gpu8`; exact launch hash and dirty-tree state will be recorded
  in the external manifest.
- Docker image/container: `hengjieliu96/dl_dir:py310sitreg_v1` /
  `py310sitreg_v1_artifact_store`.
- Dataset root: shared-storage OASIS split root at the documented container
  path; not the legacy `/database/wavereg/...` path.
- Dataset manifest: seed-3 shuffled `pair_test_200`, ranks `0-9`.
- Split config: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Protocol tag:
  `oasis_seed3_first10_table1_base_fast_ft_gpu3_sequence`.
- Candidates: base/fast pairs for MR-D-BSCP (8-4-2), D-BSCP cps2/cps4, and
  INR-BSCP cps2/cps4.
- Backend: `benchmark=False, deterministic=True`.
- Timing: one rank-0 cold run, repeated rank-0 plus ranks 1-9 warm runs per
  candidate; endpoint is `case_load_complete` through `dvf_file_ready` and
  excludes evaluation.
- Monitoring: GPU 3 is polled approximately once per second. Contaminated
  warm measurements are retried in place; contaminated cold measurements
  restart the worker in a fresh process. Retry attempts remain external-only.
- Expected acceptance: `110/110` clean timing rows, `100/100` separate Dice
  rows, valid finite float32 normalized-grid XYZ DVFs of shape
  `(160,224,192,3)`, matching requested/effective backend metadata, and exact
  Git provenance.
- Known limitation: first10 diagnostic only; it cannot replace the OASIS-100
  Dice column or authorize camera-ready edits.
- Experiment record:
  `experiments/2026-09-16_oasis_table1_base_fast_ft_first10_gpu3/`.
- Artifact roots:
  `/database/inrorspline/runs/oasis_table1_base_fast_ft_first10_gpu3_20260916_v1`
  and
  `/database/inrorspline/repo_experiments/2026-09-16_oasis_table1_base_fast_ft_first10_gpu3`.

### P2s: LungCT FD-BE cps4 Alpha Extension for Expanded TRE/SDlogJ Panel

- Run ID: `lungct_fdbe_cps4_alpha1e5_1e6_all10_v1`.
- Status: completed and technically accepted 2026-08-14 (`40/40`).
- Owner/agent: Codex.
- Date planned: 2026-08-14.
- Date started: 2026-08-14.
- Date completed: 2026-08-14.
- Server: local project GPU server; no private hostname recorded.
- GPU model/count: four NVIDIA RTX 6000 Ada GPUs, one alpha lane per GPU.
- Git commit: `68d2c4cc02ae465f2a9b23dfb965ee46ca864f41`, dirty source tree.
- Branch: `gpu8`.
- Docker image/container: `hengjieliu96/dl_dir:py310sitreg_v1` /
  `py310sitreg_v1_test1`.
- Dataset root: `${LUNGCT_PTVREG_ISO1MM}`.
- Dataset manifest: canonical pTVReg 1 mm DIR-Lab cases `01-10`; all ten
  preprocessed case fields must resolve before execution.
- Split config: all DIR-Lab 4DCT cases `01-10`; moving T50 to fixed T00.
- Standards preflight: reviewed `docs/standards/README.md`,
  `docs/standards/fair_experiment_checklist.md`,
  `docs/standards/lungct_protocol_tags.md`, and LungCT protocol warnings.
- Protocol tag:
  `ptv_landmark_crop_1mm_no_lungmask+single_stage_{sinr,directcp}_cps4+full_crop_global_ncc_dense_fd_be+fixed_t00_lung_mask_metric`.
- Methods: `sinr_cps4_global_ncc_fd_be` at LR `1e-4` and
  `direct_cp_cps4_global_ncc_fd_be` at LR `1e-3`, each with cps `4`, seed `1`,
  epochs `2500`, alpha `100000` and `1000000`.
- GPU schedule: GPU `0` INR-BSCP alpha `100000`, GPU `1` INR-BSCP alpha
  `1000000`, GPU `2` D-BSCP alpha `100000`, GPU `3` D-BSCP alpha `1000000`;
  each lane runs case01 through case10 sequentially.
- Command: dry-run, smoke, then full launch with
  `scripts/experiments/lungct_no_sweep_repro/launch_fdbe_cps4_alpha_extension_all10.py`;
  aggregate with
  `scripts/experiments/lungct_no_sweep_repro/aggregate_fdbe_cps4_alpha_extension_all10.py`;
  refresh the figure with
  `experiments/paper_artifacts/pareto_figure/build_current_tre_stdlogj_expanded.py`.
- Output root:
  `/database/inrorspline/runs/lungct_fdbe_cps4_alpha1e5_1e6_all10_v1`.
- Review root:
  `experiments/2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10/`.
- Paper artifact root:
  `experiments/paper_artifacts/pareto_figure/v011_current_tre_stdlogj_fdbe_cps4_alpha_extension/`.
- Primary metrics: native-crop snap TRE, foreground stdLogJ, foreground folding
  percentage, full-crop `1-NCC`, fixed-T00 lung-mask diagnostic `1-NCC`,
  runtime, peak memory, final data loss, and final regularization loss.
- Known limitations: alpha scales are FD-BE-family operating points only and are
  not directly comparable to IDIR, pTVReg isoTV, or Dual-INR loss weights. The
  fixed T00 lung mask is metric-only here; no masked-loss variant is launched.
- Acceptance: dry-run planned exactly `40` jobs; smoke completed `4/4`; full
  launch completed `40/40`; aggregation found zero validation errors and no
  masked-loss method IDs or masked-loss source paths.
- Result: alpha `100000` improved the cps4 FD-BE operating point substantially
  for both INR-BSCP and D-BSCP while keeping foreground folding at `0%`; alpha
  `1000000` was smoother but had worse TRE. Mean native-crop snap TRE/stdLogJ
  were `1.043/0.1193` and `1.218/0.0858` for INR-BSCP, and `1.016/0.1182` and
  `1.209/0.0857` for D-BSCP at alpha `100000` and `1000000`, respectively.
- Result artifacts:
  `experiments/2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10/acceptance_report.md`
  and
  `experiments/paper_artifacts/pareto_figure/v011_current_tre_stdlogj_fdbe_cps4_alpha_extension/`.
- Next decision: decide whether the expanded panel should keep the old alpha
  `10000` cps4 FD-BE points as cautionary context or show only the useful
  alpha-extension sweep segment.

### P2t: LungCT FD-BE cps4 Alpha Landscape for Expanded TRE/SDlogJ Panel

- Run ID: `lungct_fdbe_cps4_alpha_landscape_all10_v1`.
- Status: completed 2026-08-14 after dry-run, smoke, full launch,
  aggregation, casewise report, and v012 figure refresh.
- Owner/agent: Codex.
- Date planned: 2026-08-14.
- Date started: 2026-08-14.
- Date completed: 2026-08-14.
- Server: local project GPU server; no private hostname recorded.
- GPU model/count: four NVIDIA RTX 6000 Ada GPUs expected; alpha lanes share
  GPUs by running INR-BSCP then D-BSCP sequentially.
- Git commit: `68d2c4cc02ae465f2a9b23dfb965ee46ca864f41`, dirty source tree.
- Branch: `gpu8`.
- Docker image/container: `hengjieliu96/dl_dir:py310sitreg_v1` /
  `py310sitreg_v1_test1`.
- Dataset root: `${LUNGCT_PTVREG_ISO1MM}`.
- Dataset manifest: canonical pTVReg 1 mm DIR-Lab cases `01-10`; all ten
  preprocessed case fields must resolve before execution.
- Split config: all DIR-Lab 4DCT cases `01-10`; moving T50 to fixed T00.
- Standards preflight: reviewed `docs/standards/README.md`,
  `docs/standards/fair_experiment_checklist.md`,
  `docs/standards/lungct_protocol_tags.md`,
  `docs/standards/lungct_tre_metric_conventions.md`, and LungCT protocol
  warnings.
- Protocol tag:
  `ptv_landmark_crop_1mm_no_lungmask+single_stage_{sinr,directcp}_cps4+full_crop_global_ncc_dense_fd_be+fixed_t00_lung_mask_metric`.
- Methods: `sinr_cps4_global_ncc_fd_be` at LR `1e-4` and
  `direct_cp_cps4_global_ncc_fd_be` at LR `1e-3`, each with cps `4`, seed `1`,
  epochs `2500`. New/current-source alphas are `10000`, `20000`, `50000`,
  `200000`, and `500000`; accepted P2s rows are reused for `100000` and
  `1000000`.
- GPU schedule: wave 1 assigns alpha lanes `20000`, `50000`, `200000`,
  `500000` to GPUs `0`, `1`, `2`, `3`; each lane runs INR-BSCP cases `01-10`
  then D-BSCP cases `01-10`. Wave 2 reruns alpha `10000` with INR-BSCP on GPU
  `0` and D-BSCP on GPU `1`.
- Command: dry-run, smoke, then full launch with
  `scripts/experiments/lungct_no_sweep_repro/launch_fdbe_cps4_alpha_landscape_all10.py`;
  aggregate with
  `scripts/experiments/lungct_no_sweep_repro/aggregate_fdbe_cps4_alpha_landscape_all10.py`;
  refresh the figure with
  `experiments/paper_artifacts/pareto_figure/build_current_tre_stdlogj_fdbe_landscape.py`.
- Output root:
  `/database/inrorspline/runs/lungct_fdbe_cps4_alpha_landscape_all10_v1`.
- Review root:
  `experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/`.
- Paper artifact root:
  `experiments/paper_artifacts/pareto_figure/v012_current_tre_stdlogj_fdbe_cps4_alpha_landscape/`.
- Primary metrics: native-crop snap TRE, foreground stdLogJ, foreground folding
  percentage, full-crop `1-NCC`, fixed-T00 lung-mask diagnostic `1-NCC`,
  runtime, peak memory, final data loss, and final regularization loss.
- Known limitations: alpha scales are FD-BE-family operating points only and are
  not directly comparable to IDIR, pTVReg isoTV, or Dual-INR loss weights. The
  fixed T00 lung mask is metric-only here; no masked-loss variant is launched.
- Acceptance: passed. Dry-run verified `80` wave-1 jobs and `20` wave-2 jobs;
  smoke verified `8/8` wave-1 jobs and `2/2` wave-2 jobs. Full launch completed
  `100/100` new jobs with `100/100` valid aggregate rows and no masked-loss
  source rows. The v012 FD-BE case table has `140` rows after merging accepted
  `1e5/1e6` rows, and the alpha `1e4` rerun-vs-legacy audit reported `0.0 mm`
  mean TRE delta and `0.0 mm` max case TRE delta for both FD-BE methods.
- Next decision: inspect the seven-point FD-BE cps4 TRE/stdLogJ landscape and
  decide the paper-facing sweep subset.

### P2r: OASIS Fit-Modes Panel A Fixed-Mask Diagnostic Refresh

- Run ID: `oasis_panelA_fixedmask_diag_curves_v1`.
- Status: completed and technically accepted 2026-08-14 (`40/40`).
- Owner/agent: Codex.
- Date planned: 2026-08-14.
- Date started: 2026-08-14.
- Date completed: 2026-08-14.
- Dataset: OASIS BrainMR seed-3 shuffled `pair_test_200` first10.
- Purpose: refresh OASIS Panel A registration diagnostics so the plotted
  `1-NCC` metric is evaluated inside the fixed raw `image > 0` foreground mask,
  matching the OASIS Panel B fixed-domain convention.
- Scope: rerun only OASIS Panel A diagnostics for `10 cases × 4 methods = 40`
  records; preserve optimization objectives and all v002 non-OASIS-Panel-A
  records.
- Methods:
  `idir_original`, `sinr_cps2_be_bspline_analytic`,
  `direct_cp_cps2_be_bspline_analytic`, and
  `direct_cp_8_4_2_be_bspline_analytic_multistage`.
- OASIS settings: IDIR alpha `1`, LR `1e-4`; INR-BSCP/D-BSCP cps `2`, alpha
  `1000`, LRs `1e-4`/`1e-3`; MR-D-BSCP cps `8/4/2`, downsample `4/2/1`,
  stages `400/800/1300`, LR `1e-3`; seed `1`; 2500 epochs total.
- GPU schedule: GPUs `0,1` split ranks `0-4`/`5-9` for INR-BSCP then D-BSCP;
  GPUs `2,3` split ranks `0-4`/`5-9` for IDIR then MR-D-BSCP.
- Docker image/container: `${DOCKER_IMAGE}` / `${CONTAINER_NAME}`.
- Dataset root: `${OASIS_ROOT}`.
- Split config:
  `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Output root: `${OASIS_MASKED_PANELA_RUN_ROOT}`.
- Review root:
  `experiments/2026-08-14_oasis_panelA_fixedmask_diagnostics/`.
- Paper artifact root:
  `experiments/paper_artifacts/fit_modes_population_diagnostics/v003_masked_panelA_both_datasets/`.
- Primary metric: fixed raw `image > 0` masked `1-NCC` at epoch 2500.
- Secondary metric: full-image global `1-NCC` retained in the same diagnostic
  curve CSV as a reproducibility cross-check versus v002.
- Acceptance: `40/40` completed, zero unresolved failures, 251 finite
  fixed-mask diagnostics per curve ending at epoch 2500, exact MR-D-BSCP
  schedule metadata, and a v003 160-row source index with exactly 40 OASIS
  Panel A replacements.
- Result artifacts: `validation_report.md`, `curve_summary.csv`,
  `figure_manifest.json`, 20 individual case figures, two 4-by-5 composites,
  and population overlay/design-trial figures under the v003 paper artifact
  root.
- Known limitations: diagnostic/evaluation-domain correction only; training
  objectives are unchanged.
- Next decision: use v003 as the camera-ready diagnostic candidate and update
  figure captions/reviewer response.

### P2u: Fit-Modes Figure 3 v004 DIR-LAB Regularity-Matched Panel A Refresh

- Run ID: `fit_modes_figure3_v004_dirlab_regularity_matched_panelA`.
- Status: completed and technically accepted 2026-08-14.
- Owner/agent: Codex.
- Date planned: 2026-08-14.
- Date completed: 2026-08-14.
- Purpose: create a no-overwrite Figure 3 candidate that preserves the v003
  fixed-domain metric convention while replacing DIR-LAB Panel A single-stage
  BSCP alphas with regularity-matched settings from the accepted FD-BE cps4
  alpha landscape.
- Scope: no new optimization was required. The renderer exact-compatible reused
  the existing all-10 DIR-LAB alpha-landscape diagnostic curves for
  INR-BSCP cps4 `a=20000` and D-BSCP cps4 `a=50000`.
- Preserved rows: all OASIS v003 records, all Panel B records, DIR-LAB Panel A
  IDIR `a=3`, and DIR-LAB Panel A MR-D-BSCP `a=10000` with
  `400/400/800/900`.
- Replacement rows: `20/20` DIR-LAB Panel A rows, all with 251 finite
  fixed-T00-lung-mask `1-NCC` diagnostics through epoch 2500.
- Primary artifact root:
  `experiments/paper_artifacts/fit_modes_population_diagnostics/v004_dirlab_regularity_matched_panelA/`.
- Source index:
  `camera_ready_source_index_dirlab_regularity_matched_panelA.csv`.
- Axis convention: Panel A `1e-2–1e0`; Panel B `1e-1–1e3`.
- TRE labeling: DIR-LAB Panel A case figures, the DIR-LAB 4-by-5 composite, and
  the representative-case 2-by-4 figure label all four methods using
  `tre_snap_to_voxel.tre_mean_mm` from each run summary.
- Acceptance: passed. The v004 source index has exactly 160 rows, exactly 20
  replacements relative to v003, zero duplicate dataset/case/panel/method
  records, and all plotted endpoints match the source index.
- Known limitations: v004 regularity-matches the plotted DIR-LAB operating
  points by foreground `SD(log J)`, but it does not make the underlying
  optimization/training domains identical across all methods.
- Next decision: use v004 as the camera-ready Figure 3 candidate unless visual
  styling changes are requested.

### P2q: DIR-LAB Fit-Modes Panel A MR-D-BSCP Diagnostic Schedule Refresh

- Run ID: `fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1`.
- Status: completed and technically accepted 2026-08-13 (`10/10`).
- Owner/agent: Codex.
- Date planned: 2026-08-13.
- Date started: 2026-08-13.
- Date completed: 2026-08-13.
- Server: local project GPU server; no private hostname recorded.
- GPU model/count: two NVIDIA RTX 6000 Ada GPUs, locked to GPU indices `0,1`,
  one optimization per GPU with dynamic work stealing.
- Git commit: `17034615581f7fb3b251a1a82d28bd5d68dab1c6`, dirty source tree; exact
  dirty paths and source hashes will be stored in the run provenance.
- Branch: `gpu8`.
- Docker image/container: `${DOCKER_IMAGE}` / `${CONTAINER_NAME}`; the exact
  execution image is retained in private run provenance.
- Dataset root: `${LUNGCT_PTVREG_ISO1MM}`.
- Dataset manifest: canonical pTVReg 1 mm DIR-LAB cases `01-10`; all ten
  preprocessed case fields must resolve before execution.
- Split config: all DIR-LAB 4DCT cases `01-10`; moving T50 to fixed T00.
- Standards preflight: reviewed `docs/standards/README.md`,
  `docs/standards/fair_experiment_checklist.md`,
  `docs/standards/lungct_protocol.md`, and LungCT protocol warnings.
- Protocol tag:
  `ptv_directcp_multistage+ptv_landmark_crop_1mm_no_lungmask+full_crop_global_ncc_dense_fd_be+fixed_t00_lung_mask_metric`.
- Method: `direct_cp_32_16_8_4_global_ncc_fd_be`, control spacing
  `32/16/8/4`, downsampling `8/4/2/1`, stages `400/400/800/900`, LR
  `1e-3`, alpha `10000`, seed `1`, full-crop global-NCC diagnostic every 10
  epochs.
- Command: run
  `scripts/experiments/fit_modes_population/launch_dirlab_panelA_schedule_refresh.py`
  in dry-run, isolated smoke, then full modes; aggregate with the corresponding
  focused aggregator.
- Output root:
  `${RUN_ROOT}/fit_modes_dirlab_panelA_mrdbscp_e400_400_800_900_diag10_v1`.
- Review root:
  `experiments/2026-08-13_fit_modes_dirlab_panelA_schedule_refresh/`.
- Primary metric: full-crop diagnostic `1-NCC` at epoch 2500. Secondary
  metrics: native snap TRE, foreground stdLogJ, foreground folding percentage,
  runtime, and peak memory.
- Comparison: paired against the accepted population records with
  `200/400/800/1000` and cross-checked against the August 13
  diagnostic-disabled `400/400/800/900` outputs.
- Acceptance: `10/10` completed, exact metadata, 2500 history rows, 251 finite
  diagnostic rows per case, finite secondary metrics, and zero unresolved
  failures. Final result: PASS.
- Result: historical versus replacement `1-NCC` was
  `0.003216 +/- 0.001950` versus `0.003215 +/- 0.001950`; paired mean change
  `-0.000000441`, range `[-0.000006, 0.000001]`. All population comparisons
  use ten paired cases.
- Replay cross-check: the ten diagnostic-enabled reruns reproduced the August
  13 diagnostic-disabled displacement-derived TRE, stdLogJ, folding, peak
  memory, and native-crop headline TRE exactly at stored precision. The
  diagnostic added `89.9 s` mean runtime.
- Reports:
  `experiments/2026-08-13_fit_modes_dirlab_panelA_schedule_refresh/acceptance_report.md`
  and
  `experiments/2026-08-13_fit_modes_dirlab_panelA_schedule_refresh/comparison_report.md`.
- Schedule-matched supplementary/reviewer diagnostics completed with a
  versioned 160-record source index, two 4-by-5 dataset composites, 20
  individual case figures, endpoint validation, checksums, and visual review:
  `experiments/paper_artifacts/fit_modes_population_diagnostics/v001_schedule_matched/`.
- Known limitations: this is a 2500-versus-2400 epoch schedule correction, not
  a fixed-budget schedule ablation; it refreshes only MR-D-BSCP Panel A. The
  original 2400-epoch index and figures remain immutable sensitivity evidence.
- Next decision: design the separate compact main-paper figure, then revise
  Methods, caption, and reviewer response. The 4-by-5 composites are explicitly
  supplementary/reviewer diagnostics.

### P2p: OASIS100 Selected Fast-Schedule Validation

- Run ID: `oasis_selected_fast_first100_alpha1000_v1`.
- Status: launching 2026-08-13.
- Owner/agent: Codex.
- Date planned: 2026-08-13.
- Date started: 2026-08-13.
- Dataset: OASIS BrainMR seed-3 shuffled `pair_test_200` first100.
- Purpose: validate the selected accelerated schedules on OASIS100 after the
  unified first10 runtime-reduction screen.
- Selected schedules:
  `MR_A_fast_decay`, `D2_A_bal_lr10x`, `D4_A_bal_lr10x`,
  `S2_A_bal_lr10x`, and `S4_A_bal_lr10x`.
- First10 reuse policy: reuse ranks `0-9` from
  `/database/inrorspline/runs/oasis_unified_acceleration_sweep_alpha1000_v1`;
  do not rerun them in this output root.
- New-run policy: evaluate ranks `10-99` only, producing `450` new
  registrations and a combined `500`-row selected OASIS100 report.
- Execution order: first run `MR_A_fast_decay` data-parallel across rank chunks
  `10-32`, `33-55`, `56-77`, and `78-99` on GPUs `0,1,2,3`; after MR
  completes, run `D2_A_bal_lr10x`, `D4_A_bal_lr10x`, `S2_A_bal_lr10x`, and
  `S4_A_bal_lr10x` model-parallel on GPUs `0,1,2,3`, each over ranks `10-99`.
- Alpha: `1000`.
- Protocol tag:
  `oasis_seed3_first100_selected_fast_alpha1000_reuse_first10`.
- Standards preflight: OASIS BrainMR protocol and fair experiment checklist
  reviewed; use seed-3 shuffled first100 and fixed raw image-positive
  foreground convention.
- Docker image/container: `py310sitreg_v1_test1`,
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root:
  `/database/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1`.
- Split config:
  `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Output root:
  `/database/inrorspline/runs/oasis_selected_fast_first100_alpha1000_v1`.
- Review root:
  `experiments/2026-08-13_oasis_selected_fast_first100/`.
- Commands:
  syntax check with
  `docker exec py310sitreg_v1_test1 python -m py_compile /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/run_one.py /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/launch_oasis_selected_fast_first100.py /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/aggregate_oasis_selected_fast_first100.py`;
  smoke with
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/launch_oasis_selected_fast_first100.py --stage smoke --epoch-override 1 --gpus 0,1,2,3 --max-concurrent 4 --skip-aggregate`;
  full launch with
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/launch_oasis_selected_fast_first100.py --stage all --gpus 0,1,2,3 --max-concurrent 4 --allow-incomplete-aggregate`.
- Primary metrics: OASIS100 Dice, HD95, ASD, foreground regularity metrics,
  actual epochs, peak GPU memory, and reference runtime.
- Known limitations: runtime is reference-only because GPUs were explicitly
  allowed to remain shared/occupied; final cold/warm deployment timing should
  be measured later under isolated persistent-process conditions.
- Next decision: after aggregation, compare selected fast schedules with the
  full-budget OASIS100 counterparts and decide final paper rows.

### P2o: OASIS Unified Runtime-Reduction Sweep

- Run ID: `oasis_unified_acceleration_sweep_alpha1000_v1`.
- Status: launching 2026-08-13.
- Owner/agent: Codex.
- Date planned: 2026-08-13.
- Date started: 2026-08-13.
- Dataset: OASIS BrainMR seed-3 shuffled `pair_test_200` first10.
- Purpose: apply a unified acceleration template before selecting one
  accelerated variant each for MR-D-BSCP `8-4-2`, D-BSCP `cps2`, and SINR
  `cps2` for later first100 validation; then run D-BSCP/SINR `cps4` as
  contextual follow-up.
- Methods:
  `direct_cp_8_4_2_be_bspline_analytic_multistage`,
  `direct_cp_cps2_be_bspline_analytic`,
  `sinr_cps2_be_bspline_analytic`,
  `direct_cp_cps4_be_bspline_analytic`, and
  `sinr_cps4_be_bspline_analytic`.
- Candidate template: method-native baseline LR, fixed epoch budgets
  `60%/40%/25%` with LR multipliers `3x/10x/30x`, and adaptive `A_bal_lr10x`
  / `A_fast_lr30x` schedules; MR-D-BSCP additionally includes decayed-stage LR
  variants and the previous `A_fast_decay_legacy`.
- Alpha: `1000`.
- Protocol tag:
  `oasis_seed3_first10_unified_acceleration_alpha1000_persistent_candidate_workers`.
- Standards preflight: OASIS BrainMR protocol and fair experiment checklist
  reviewed; use seed-3 shuffled first10 and fixed raw image-positive foreground
  convention.
- Docker image/container: `py310sitreg_v1_test1`,
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root:
  `/database/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1`.
- Split config:
  `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Output root:
  `/database/inrorspline/runs/oasis_unified_acceleration_sweep_alpha1000_v1`.
- Review root:
  `experiments/2026-08-13_oasis_unified_acceleration_sweep/`.
- Commands:
  syntax check with
  `docker exec py310sitreg_v1_test1 python -m py_compile /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/run_one.py /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/aggregate_oasis_unified_acceleration_sweep.py`;
  primary launch with
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py --phase primary --gpus 0,1,2,3 --max-concurrent 4 --allow-incomplete-aggregate`;
  cps4 launch with
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py --phase cps4 --gpus 0,1,2,3 --max-concurrent 4 --allow-incomplete-aggregate`.
- Primary metrics: Dice degradation grades versus same-method baseline,
  HD95/ASD degradation, full registration cold/warm runtime,
  loaded-images-to-DVF-file cold/warm runtime, actual epochs, and peak GPU
  memory.
- Known limitations: existing GPU jobs were explicitly allowed by the user, so
  screening runtime may be noisier than final timing; this is first10 screening
  only; first100 validation and final isolated deployment timing are deferred
  until variants are selected.
- Next decision: after aggregation, select accelerated variants for first100
  OASIS validation and controlled paper-facing timing.

### P2n: LungCT Figure 2 MR-D-BSCP 2500-Step Schedule Refresh

- Run ID: `lungct_fig2_mrdbscp_2500_schedule_refresh_v1`.
- Status: planned for launch 2026-08-13.
- Owner/agent: Codex.
- Date planned: 2026-08-13.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Purpose: rerun only the Figure 2 MR-D-BSCP curve points that currently use
  the old `200,400,800,1000` schedule, changing only the schedule to
  `400,400,800,900` while preserving each plotted method/loss definition.
- IDIR-loss method: `direct_cp_32_16_8_4_global_ncc_fd_be`, full-crop global
  NCC plus dense forward-difference BE, alphas
  `1000,3000,10000,30000,100000`.
- pTVReg-loss method: `direct_cp_32_16_8_4_ptv_lcc_isotv`, pTV-LCC plus isoTV,
  weights `0.011,0.033,0.055,0.11,0.33`; reuse completed 2500-step `w=0.11`
  anchor if validation passes.
- Fixed parameters: stage full control points `32,16,8,4`, downsample
  `8,4,2,1`, stage epochs `400,400,800,900`, seed `1`, `lr=1e-3`.
- Protocol tag:
  `ptv_directcp_multistage_fig2_schedule_refresh_2500`.
- Standards preflight: LungCT protocol tags, fair experiment checklist,
  Docker GPU workflow, pTVReg protocol warnings, and previous schedule pilot
  reviewed before launch.
- Docker image/container: `py310sitreg_v1_test1`,
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`.
- Output roots:
  `/database/inrorspline/runs/lungct_fig2_mrdbscp_idirloss_fdbe_e400_400_800_900_v1`
  and
  `/database/inrorspline/runs/lungct_fig2_mrdbscp_ptvloss_e400_400_800_900_v1`.
- Review root:
  `experiments/2026-08-13_lungct_fig2_mrdbscp_2500_schedule_refresh/`.
- Commands:
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_lungct_fig2_mrdbscp_2500_schedule_refresh.py --families idir_fdbe --gpus 0,1,2,3`
  followed by
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_lungct_fig2_mrdbscp_2500_schedule_refresh.py --families ptv --gpus 0,1,2,3`.
- Primary metrics: original/native-crop snap TRE mean/std, foreground fixed
  T00 lung-mask stdLogJ mean/std, foreground `J<=0%`, elapsed time, peak
  memory, and case-by-case TRE.
- Known limitations: schedule-refresh comparison only; does not switch Figure 2
  to the newer corrected fixed-mask NCC + analytic B-spline BE protocol; does
  not update plotting source until the comparison table is reviewed.
- Next decision: after aggregation, decide whether the 2500-step refreshed rows
  should replace the old Figure 2 MR-D-BSCP schedule rows.

### P2m: LungCT 2500-Step Schedule Standardization Pilots

- Run ID: `lungct_2500_schedule_standardization_pilot_v1`.
- Status: planned for launch 2026-08-12.
- Owner/agent: Codex.
- Date planned: 2026-08-12.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Purpose: standardize LungCT multistage MR-D-BSCP pilot schedules to
  `400,400,800,900` epochs (`2500` total) before deciding whether full
  2500-step sweeps are necessary for paper curves.
- Pilot A method: `direct_cp_32_16_8_4_ptv_lcc_isotv`, pTVReg loss,
  `ptv_isotv_weight=0.11`, `lr=1e-3`.
- Pilot B method: `direct_cp_32_16_8_4_masked_ncc_bspline_be`, corrected
  IDIR loss with fixed T00 lung-mask sampled NCC plus analytic cubic B-spline
  BE, `alpha=33333.333333333336`, `lr=1e-3`.
- Fixed schedule: stage full control points `32,16,8,4`, stage downsample
  `8,4,2,1`, stage epochs `400,400,800,900`, seed `1`.
- Protocol tag:
  `ptv_directcp_multistage_schedule_standardization_2500_pilot`.
- Standards preflight: LungCT protocol tags, fair experiment checklist,
  Docker GPU workflow, pTVReg protocol warnings, and prior epoch-budget audit
  reviewed before launch.
- Docker image/container: `py310sitreg_v1_test1`,
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`.
- Output roots:
  `/database/inrorspline/runs/lungct_directcp32_ptvlcc_w0p11_e400_400_800_900_all10_v1`
  and
  `/database/inrorspline/runs/lungct_directcp4stage_masked_ncc_bspline_be_alpha33333_e400_400_800_900_all10_v1`.
- Review roots:
  `experiments/2026-08-12_lungct_directcp32_ptvlcc_w0p11_schedule_standardization/`,
  `experiments/2026-08-12_lungct_directcp4stage_masked_ncc_bspline_be_alpha33333_e400_400_800_900_all10/`,
  and
  `experiments/2026-08-12_lungct_2500_schedule_standardization_pilot/`.
- Commands:
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_lungct_2500_schedule_pilots.py --pilots ptvreg_w0p11 --gpus 0,1,2,3`
  followed by
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_lungct_2500_schedule_pilots.py --pilots masked_alpha33333 --gpus 0,1,2,3`.
- Primary metrics: original/native-crop snap TRE mean/std, foreground fixed
  T00 lung-mask stdLogJ mean/std, foreground `J<=0%`, elapsed time, peak
  memory, and case-by-case TRE.
- Known limitations: pilot only; no full sweep should be launched from this
  entry; comparisons are schedule-standardization checks and not method-only
  claims across unmatched loss/preprocessing families.
- Next decision: after both reports complete, decide whether pTVReg-loss and
  corrected IDIR-loss rows warrant full 2500-step sweep reruns.

### P2l: LungCT Corrected MR-D-BSCP IDIR-Loss Alpha Sweep

- Run ID: `lungct_directcp4stage_masked_ncc_bspline_be_alpha_sweep_v1`.
- Status: stopped after alpha `100000` completed, 2026-08-12.
- Owner/agent: Codex.
- Date planned: 2026-08-12.
  Date started: 2026-08-12.
  Date stopped: 2026-08-12.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Purpose: resolve the apparent regularization-scale mismatch in the corrected
  `MR-D-BSCP (IDIR loss)` protocol by sweeping alpha around and beyond the
  completed `alpha=10000` anchor before choosing finer `/3` or `x3` scales.
- Method: `direct_cp_32_16_8_4_masked_ncc_bspline_be`.
- Objective: fixed T00 lung-mask sampled global NCC plus `alpha *` sampled
  analytic cubic B-spline BE.
- New sweep alphas: `100`, `1000`, `100000`, `1000000`.
- Aggregation anchor: completed P2k `alpha=10000` result, imported into the
  review table rather than re-run.
- Fixed parameters: `lr=1e-3`, seed `1`, sampled batch size `10000`, stage
  full cps `32,16,8,4`, downsample `8,4,2,1`, stage epochs
  `200,400,800,1000`.
- Protocol tag:
  `ptv_directcp_multistage+ptv_landmark_crop_1mm+fixed_t00_lung_mask_sampled_global_ncc_bspline_be_alpha_sweep`.
- Standards preflight: LungCT protocol tags, TRE metric conventions, pTVReg
  protocol warnings, fair experiment checklist, Docker GPU workflow, and P2k
  corrected-method audit reviewed.
- Docker image/container: `py310sitreg_v1_test1`,
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`.
- Output root:
  `/database/inrorspline/runs/lungct_directcp4stage_masked_ncc_bspline_be_alpha_sweep_v1`.
- Review root:
  `experiments/2026-08-12_lungct_directcp4stage_masked_ncc_bspline_be_alpha_sweep/`.
- Command:
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_masked_ncc_bspline_be_alpha_sweep.py --gpus 0,1,2,3`.
- Primary metrics: original/native-crop snap TRE mean/std and foreground
  fixed T00 lung-mask stdLogJ mean/std; retain all four TRE variants and
  foreground regularity metrics.
- Known limitations: scale-finding sweep only; alpha `10000` is reused from
  P2k; pTVReg-style 1mm crop input is not the native full-volume original IDIR
  protocol; sampled fixed-mask NCC introduces stochasticity.
- Verification so far: compile/import passed; dry-run queued exactly `40` new
  jobs across GPUs `0,1,2,3`; patched non-final stage initialization to skip
  full 1mm prolongation diagnostics by default; patched case01
  `stage_epochs=1,1,1,1` smoke passed and wrote final metrics; `alpha=100000`
  completed all `10/10` cases with return code `0`; launcher and in-progress
  `alpha=1000000` workers were stopped by request before any `alpha=1000000`
  summaries completed; aggregation over alpha `10000` and `100000` completed
  with errors `[]`.
- Partial result: corrected alpha `100000` original-snap TRE
  `1.003 +/- 0.154 mm`, foreground stdLogJ
  `0.155840 +/- 0.033335`, foreground `J<=0` `0.0005%`, foreground BE
  `0.001099`, foreground diffusion `0.008170`.
- Interpretation: alpha `100000` improves the corrected protocol strongly over
  alpha `10000` and is smoother than the current full-crop NCC + dense-FD-BE
  reference, but the current reference still has lower TRE
  (`0.966 +/- 0.151 mm`).
- Next decision: if continuing this corrected variant, test local scales around
  `100000`, especially `/3` and `x3`; otherwise retain the current reference
  row for TRE-dominant paper plotting.

### P2k: LungCT Corrected MR-D-BSCP IDIR-Loss Alpha10000 All10

- Run ID: `lungct_directcp4stage_masked_ncc_bspline_be_alpha10000_all10_v1`.
- Status: completed 2026-08-12.
- Owner/agent: Codex.
- Date planned: 2026-08-12.
  Date started: 2026-08-12.
  Date completed: 2026-08-12.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Purpose: correct the plotted `MR-D-BSCP (IDIR loss)` operating point by
  matching the intended fixed T00 lung-mask sampled loss domain and analytic
  cubic B-spline bending-energy regularizer, then compare against the current
  all-case full-crop global-NCC + dense-FD-BE best point.
- Method: `direct_cp_32_16_8_4_masked_ncc_bspline_be`.
- Objective: fixed T00 lung-mask sampled global NCC plus `alpha *` sampled
  analytic cubic B-spline BE.
- Fixed parameters: alpha `10000`, `lr=1e-3`, seed `1`, sampled batch size
  `10000`, stage full cps `32,16,8,4`, downsample `8,4,2,1`, stage epochs
  `200,400,800,1000`.
- Comparison target: existing `direct_cp_32_16_8_4_global_ncc_fd_be`, alpha
  `10000`, full-crop global NCC + dense forward-difference BE, fixed T00 lung
  mask metric-only, current all-case TRE `0.966 +/- 0.151 mm` and foreground
  stdLogJ `0.185452 +/- 0.060879`.
- Protocol tag:
  `ptv_directcp_multistage+ptv_landmark_crop_1mm+fixed_t00_lung_mask_sampled_global_ncc_bspline_be`.
- Standards preflight: LungCT protocol tags, TRE metric conventions, pTVReg
  protocol warnings, fair experiment checklist, and Docker GPU workflow
  reviewed.
- Docker image/container: `py310sitreg_v1_test1`,
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`.
- Output root:
  `/database/inrorspline/runs/lungct_directcp4stage_masked_ncc_bspline_be_alpha10000_all10_v1`.
- Review root:
  `experiments/2026-08-12_lungct_directcp4stage_masked_ncc_bspline_be_alpha10000_all10/`.
- Primary metrics: original/native-crop snap TRE mean/std and foreground
  fixed T00 lung-mask stdLogJ mean/std; retain all four TRE variants and
  foreground regularity metrics.
- Verification: compile/import passed; sampled analytic B-spline derivative
  check matched dense expansion with maximum absolute error `1.788e-7`;
  case01 `1,1,1,1` smoke passed; dry-run queued exactly `10` jobs; full
  all-10 launch completed `10/10` jobs with return code `0`; aggregation
  completed with errors `[]`.
- Actual corrected result at alpha `10000`: original-snap TRE
  `1.037 +/- 0.153 mm`, foreground stdLogJ `0.265395 +/- 0.063547`,
  foreground `J<=0` `0.0349%`, foreground BE `0.006341`, foreground diffusion
  `0.016738`.
- Comparison target remains better at the same alpha: existing
  `direct_cp_32_16_8_4_global_ncc_fd_be` original-snap TRE
  `0.966 +/- 0.151 mm`, foreground stdLogJ `0.185452 +/- 0.060879`,
  foreground `J<=0` `0.0037%`.
- Known limitations: single alpha only; pTVReg-style 1mm crop input, not native
  full-volume original IDIR; stochastic sampled loss.
- Next decision: do not replace the current plotted `MR-D-BSCP (IDIR loss)`
  point with this corrected alpha-10000 point unless a follow-up alpha sweep or
  regularization-scale adjustment improves the corrected protocol.

### P0: Scaffold Only

- Status: complete when root scaffold lands.
- Protocol tag: `repo_scaffold_no_experiments`.
- Notes: no data, Docker, rsync, or package/test scaffold.

### P2c: COPD DirectCP32 pTV-LCC + isoTV Sanity

- Status: completed 2026-07-07 after COPD mask generation, pTV-prep, case `01`
  smoke, all-GPU launch, and aggregation passed with `10/10` completed
  summaries.
- Purpose: generate the COPD replacement curve/row for the lower-right
  TRE-vs-SDlogJ figure sanity check using
  `MR-D-BSCP (pTVReg loss)`.
- Dataset: COPD cases `01-10`, fixed `iBHCT`, moving `eBHCT`, direction
  `eBHCT -> iBHCT`.
- Preprocessing: pTV-style landmark crop, intensity clamp `[80,900] -> [0,1]`,
  crop margin xyz `[10,10,5]`, 1mm target grid
  `round(crop_shape_zyx * native_spacing_zyx)`, masks propagated for QA and
  foreground regularity metrics only.
- Method: `direct_cp_32_16_8_4_ptv_lcc_isotv`; stage full cps `32,16,8,4`;
  downsample `8,4,2,1`; stage epochs `200,400,800,1000`; `lr=1e-3`;
  `seed=1`; pTV LCC sigma `2.1mm`; isoTV weight `0.11`; `csqrt=5e-3`;
  border mask `5`; save dense DVF.
- Output root:
  `/database/inrorspline/runs/copd_directcp32_ptvlcc_w0p11_sanity_v1`.
- Review root:
  `experiments/2026-07-07_copd_directcp32_ptvlcc_w0p11_sanity/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2,3`.
- Primary metrics: native-crop/original-domain snap-to-voxel TRE mean and
  per-case TRE; diagnostic native subvoxel TRE, reg-1mm TREs, foreground
  SDlogJ, folding percentage, BE, diffusion, runtime, and peak memory.
- Actual headline result: native-crop/original snap TRE mean
  `6.8486 +/- 5.4261 mm` across cases `01-10`; per-case TREs
  `6.683, 5.304, 0.824, 16.298, 15.952, 9.044, 0.849, 5.070, 0.635, 7.825`.
- Actual regularity result: mean foreground SDlogJ `0.6267`, mean
  foreground `J<=0` percentage `4.7831`.
- Known limitation: COPD official pTVReg best/fine-tuned settings are closer to
  an `8 -> 4` refinement path; this run intentionally tests the 4DCT-selected
  `32,16,8,4` workflow and `ptv_isotv_weight=0.11`.
- Interpretation: the 4DCT-selected workflow is not robust on COPD as-is;
  high-error cases `04`, `05`, `06`, and `10` motivate a COPD-tuned follow-up
  before using this as a final figure curve.

### P2d: COPD DirectCP64-32-16-8-4 pTV-LCC + isoTV Sanity

- Status: completed 2026-07-07.
- Purpose: test whether adding a coarser `64` stage improves the COPD failures
  from P2c while keeping all other pTVReg-loss settings fixed.
- Dataset: COPD cases `01-10`, fixed `iBHCT`, moving `eBHCT`, direction
  `eBHCT -> iBHCT`, using the completed COPD pTV-prep root.
- Method: `direct_cp_32_16_8_4_ptv_lcc_isotv` runner with custom stages;
  stage full cps `64,32,16,8,4`; downsample `16,8,4,2,1`; stage epochs
  `100,200,400,800,1000`; total epochs `2500`; `lr=1e-3`; `seed=1`;
  pTV LCC sigma `2.1mm`; isoTV weight `0.11`; `csqrt=5e-3`; border mask `5`;
  save dense DVF.
- Output root:
  `/database/inrorspline/runs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_sanity_v1`.
- Review root:
  `experiments/2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_sanity/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2,3`.
- Primary metrics: native-crop/original-domain snap-to-voxel TRE mean and
  per-case TRE; diagnostic native subvoxel TRE, reg-1mm TREs, foreground
  SDlogJ, folding percentage, BE, diffusion, runtime, and peak memory.
- Actual headline result: native-crop/original snap TRE mean
  `2.5026 +/- 2.2543 mm` across cases `01-10`; per-case TREs
  `1.303, 3.335, 0.823, 7.261, 5.873, 0.986, 0.849, 3.164, 0.639, 0.793`.
- Actual regularity result: mean foreground SDlogJ `0.4778`, mean
  foreground `J<=0` percentage `2.1468`.
- Decision use: compared with P2c, the coarser `64` stage substantially reduces
  the COPD failures while keeping total epochs fixed at `2500`; cases `04` and
  `05` remain the hardest.

### P2e: COPD DirectCP64-32-16-8-4 pTV-LCC + isoTV E400x4+900

- Status: completed 2026-07-07.
- Purpose: rerun P2d with the same total `2500` epochs but front-load more
  optimization into the coarser stages using `400,400,400,400,900`.
- Dataset: COPD cases `01-10`, fixed `iBHCT`, moving `eBHCT`, direction
  `eBHCT -> iBHCT`, using the completed COPD pTV-prep root.
- Method: `direct_cp_32_16_8_4_ptv_lcc_isotv` runner with custom stages;
  stage full cps `64,32,16,8,4`; downsample `16,8,4,2,1`; stage epochs
  `400,400,400,400,900`; total epochs `2500`; `lr=1e-3`; `seed=1`;
  pTV LCC sigma `2.1mm`; isoTV weight `0.11`; `csqrt=5e-3`; border mask `5`;
  save dense DVF.
- Output root:
  `/database/inrorspline/runs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity_v1`.
- Review root:
  `experiments/2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2,3`.
- Primary metrics: native-crop/original-domain snap-to-voxel TRE mean and
  per-case TRE; diagnostic native subvoxel TRE, reg-1mm TREs, foreground
  SDlogJ, folding percentage, BE, diffusion, runtime, and peak memory.
- Actual headline result: native-crop/original snap TRE mean
  `1.0505 +/- 0.7673 mm` across cases `01-10`; per-case TREs
  `0.714, 3.348, 0.823, 0.732, 0.740, 0.851, 0.836, 0.820, 0.851, 0.789`.
- Actual regularity result: mean foreground SDlogJ `0.4021`, mean
  foreground `J<=0` percentage `0.7251`.
- Decision use: compared with P2d, front-loading the same total epoch budget
  improves mean TRE from `2.5026` to `1.0505 mm`; cases `04`, `05`, and `08`
  are rescued, while case `02` remains the main outlier.

### P1: OASIS Seed-3 First10 Main Pareto

- Status: completed; full launch started and completed 2026-06-25 on shenggpu9
  after dry-run and smoke checks passed.
- Methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4.
- Purpose: main brain Pareto result.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`.
- Grid: IDIR alpha `0,0.01,0.1,1,10,100`; spline alpha
  `0,10,100,1000,10000,100000`; 2500 epochs.
- LR: IDIR `1e-4`; SINR cps4/cps2 `1e-4`; DirectCP cps4 `1e-3`.
- Regularizer: BE-trained sweep. IDIR uses coordinate-autograd analytic BE;
  SINR/DirectCP use cubic B-spline analytic BE.
- Metrics: Dice, HD95, ASD, native analytic BE/diffusion, forward-difference
  BE/diffusion, log-Jacobian, folding percentage, NDV; retain per-label and
  per-case rows, and report population means/spread across cases.
- Output root: `/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1`.
- Review root: `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/`.
- Launch environment: Docker container
  `inrorspline_oasis_seed3_first10_run_20260625`, image
  `hengjieliu96/dl_dir:py310sitreg_v1_251008`; dry-run and 24-job smoke passed.

### P1c: OASIS Seed-3 First10 DirectCP cps2 Preliminary Overlay

- Status: completed; add-on launch ran 2026-06-25 11:34-11:57 Pacific after
  P1 completion and DirectCP cps2 smoke passed. Overlay aggregation completed
  2026-06-25 11:58 Pacific.
- Purpose: preliminary check of DirectCP cps2 against the completed first10
  Pareto curves without overwriting P1 reports.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`.
- Method: DirectCP cps2 with cubic B-spline analytic BE/diffusion.
- Grid: alpha `100,1000,10000`; 2500 epochs; seed `1`; eval interval `250`.
- LR: `1e-3`, matching the DirectCP cps4 setting except for `cps=2`.
- Output root: `/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1`.
- Overlay review root:
  `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_prelim_overlay/`.
- Launch environment: Docker container
  `inrorspline_oasis_seed3_first10_run_20260625`, image
  `hengjieliu96/dl_dir:py310sitreg_v1_251008`; requested GPUs `1,2,3`.
- Known limitation: preliminary 3-alpha subset only; do not interpret as a full
  six-alpha DirectCP cps2 Pareto sweep.

### P1d: OASIS Seed-3 First10 Saved Baseline Evaluation

- Status: completed 2026-06-25 16:40 Pacific in Docker after dry-run, rank-0
  smoke, 50/50 saved-field evaluations, and overlay aggregation passed.
- Purpose: add paper-facing OASIS reference dots for saved DL and Greedy
  baselines using the same first10 accuracy and dense-field regularity pipeline
  as P1.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`.
- Methods: `bs1_vxm_v0_2`, `bs2_tm_v0`, `bs3_vfa_v0`, `bs4_sitreg_v0`, and
  canonical `Greedy`.
- Inputs: saved dense voxel-space displacement fields from the local DL
  baseline root and canonical Greedy result root documented in
  `private/data_sources.md`.
- Metrics: Dice, HD95, ASD, dense finite-difference BE/diffusion, log-Jacobian,
  folding percentage, and NDV; retain per-label, per-case, and population rows.
- Output root: `/database/inrorspline/runs/oasis_seed3_first10_saved_baselines_v1`.
- Review root:
  `experiments/2026-06-25_oasis_seed3_first10_saved_baselines/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`, GPU `0`.
- Acceptance: `50/50` summaries and `50/50` `DONE` files present; each run has
  case, regularity, BE, diffusion, history, mask-audit, and summary artifacts;
  aggregation reported `errors=[]`.
- Actual population means: BS1 VoxelMorph Dice/HD95/ASD
  `0.7683/2.4477/0.5497`; BS2 TransMorph `0.7913/2.0084/0.4467`;
  BS3 VFA `0.8198/1.7743/0.3669`; BS4 SITReg `0.8215/1.7159/0.3509`;
  Greedy `0.8037/1.8726/0.4096`.
- Greedy displacement format note: canonical Greedy NIfTI fields are stored as
  5D channel-last arrays and are reoriented by applying the OASIS scalar
  LIA-to-RAS transform to each component plus component signs `-,-,+`; rank-0
  smoke reproduced the saved Greedy warped-label Dice closely.
- Known limitation: saved dense fields do not expose native model/control
  parameterizations, so native analytic IDIR/B-spline regularity columns are
  intentionally unavailable for these baseline rows.

### P1e: OASIS Seed-3 First20 Saved Baseline Extension

- Status: completed 2026-06-25 Pacific after rank `10-19` dry-run confirmed
  all 50 source displacement files, 50 new saved-field evaluations completed,
  and first20 aggregation passed with `100/100` saved baseline rows.
- Purpose: extend the saved DL/Greedy baseline evaluation from seed-3 first10
  to seed-3 first20 without retraining.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-19`; ranks `0-9` reuse P1d
  completed outputs, ranks `10-19` are newly evaluated. Additional original
  sorted indices for ranks `10-19`: `130,100,118,67,79,56,142,199,36,12`.
- Methods: `bs1_vxm_v0_2`, `bs2_tm_v0`, `bs3_vfa_v0`, `bs4_sitreg_v0`, and
  canonical `Greedy`.
- Metrics: same saved dense-field posthoc pipeline as P1d: Dice, HD95, ASD,
  dense finite-difference BE/diffusion, log-Jacobian, folding percentage, and
  NDV using fixed raw image-positive foreground for regularity metrics.
- Output root: `/database/inrorspline/runs/oasis_seed3_first10_saved_baselines_v1`
  extended with rank `10-19` run directories.
- First20 review root:
  `experiments/2026-06-25_oasis_seed3_first20_saved_baselines/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`, GPU `0`.
- Actual first20 population means: BS1 VoxelMorph Dice/HD95/ASD
  `0.7724/2.5298/0.5493`; BS2 TransMorph `0.7954/2.0932/0.4483`;
  BS3 VFA `0.8223/1.8050/0.3638`; BS4 SITReg `0.8245/1.7505/0.3491`;
  Greedy `0.8067/1.9405/0.4101`.
- Acceptance: `100/100` summaries and `100/100` `DONE` files present across
  ranks `0-19`; each run has case, regularity, BE, diffusion, history,
  mask-audit, and summary artifacts; aggregation reported `errors=[]`.
- Known limitation: output-root slug retains the historical `first10` name
  because it is the existing P1d artifact root being extended; review artifacts
  use the explicit `first20` slug.

### P1f: OASIS Rank-9 Registration vs Greedy-Field Fitting LR Sweep

- Status: completed 2026-06-26 after dry-run, smoke, full launch, and
  aggregation passed with `18/18` summaries and `errors=[]`.
- Date planned: 2026-06-26.
- Purpose: illustrate how IDIR, SINR cps4, and DirectCP cps4 behave under
  image-registration optimization versus direct fitting of a strong Greedy
  dense displacement field.
- Dataset: OASIS BrainMR seed-3 shuffled rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`.
- Methods: IDIR original dense SIREN, SINR cps4 with cubic B-spline analytic
  BE, and DirectCP cps4 with cubic B-spline analytic BE.
- Exp1 objective: image registration with the method-native BE regularizer.
  Alphas are fixed to the completed sweep's best operating points:
  IDIR `1`, SINR cps4 `1000`, DirectCP cps4 `1000`.
- Exp2 objective: direct MSE fitting of the canonical Greedy dense displacement
  in normalized-grid coordinates over the fixed-image-positive brain mask, with
  no image loss and no regularization. The brain mask is the raw fixed OASIS
  image after standard reorientation with `fixed_raw > 0`; segmentation masks
  are not training masks.
- LR sweep: `1e-2,1e-3,1e-4` for every method/objective; 2500 epochs; seed `1`;
  eval interval `250`.
- Greedy displacement convention: canonical Greedy NIfTI fields are loaded with
  the same saved-baseline convention: 5D channel-last layout, scalar OASIS
  LIA-to-RAS reorientation per component, and component signs `-,-,+`.
- Output root: `/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1`.
- Review root:
  `experiments/2026-06-26_oasis_rank009_fit_modes_lr_sweep/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2`.
- Primary metrics: per-epoch raw and plot-normalized loss curves, final
  brain-mask Greedy-field MSE, full-volume Greedy-field MSE as context,
  Dice/HD95/ASD, and posthoc deformation regularity metrics.
- Known limitations: single illustrative OASIS pair only; IDIR field-fit samples
  `10,000` fixed-image-positive coordinates per epoch, while spline variants
  expand their cps4 dense field and compute MSE on all fixed-image-positive
  voxels. IDIR BE scale normalization is plot-only; the IDIR image-registration
  training objective remains unchanged.
- Actual best final points: image-registration Dice favored SINR cps4 at
  lr `1e-4` with Dice `0.8296`; Greedy-field brain-mask MSE favored DirectCP
  cps4 at lr `1e-2` with MSE `2.08659e-06`.
- Next decision: compare against a SITReg-field target on the same case before
  deciding whether to add more cases.

### P1g: OASIS Rank-9 Registration vs SITReg-Field Fitting LR Sweep

- Status: completed 2026-06-26 after dry-run, smoke, full launch, and
  aggregation passed with `18/18` combined summaries and `errors=[]`.
- Date planned: 2026-06-26.
- Purpose: repeat the rank-9 field-fitting diagnostic with the saved BS4 SITReg
  dense displacement as the target while reusing the completed rank-9
  image-registration rows from P1f.
- Dataset: OASIS BrainMR seed-3 shuffled rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`.
- Methods: IDIR original dense SIREN, SINR cps4 with cubic B-spline controls,
  and DirectCP cps4.
- Reused objective: image registration with method-native BE regularization
  from `/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1`.
  Alphas remain IDIR `1`, SINR cps4 `1000`, DirectCP cps4 `1000`.
- New objective: direct MSE fitting of the saved `bs4_sitreg_v0` best-Dice
  dense displacement in normalized-grid coordinates over the
  fixed-image-positive brain mask, with no image loss and no regularization.
  The brain mask is the raw fixed OASIS image after standard reorientation with
  `fixed_raw > 0`; segmentation masks are not training masks.
- LR sweep: `1e-2,1e-3,1e-4` for every method; 2500 epochs; seed `1`; eval
  interval `250`.
- SITReg displacement convention: saved NIfTI shape `(3,160,224,192)`,
  channel-first `zyx` repo/RAS layout, no reorientation or component sign flip,
  converted from voxel units to normalized-grid displacement for PyTorch
  fitting.
- Output root:
  `/database/inrorspline/runs/oasis_rank009_sitreg_fit_modes_lr_sweep_v1`.
- Review root:
  `experiments/2026-06-26_oasis_rank009_sitreg_fit_modes_lr_sweep/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `2,3` with
  `--max-concurrent 2`.
- Primary metrics: per-epoch SITReg-field brain-mask MSE, final full-volume
  MSE as context, Dice/HD95/ASD, posthoc deformation regularity metrics, and
  reused image-registration loss curves.
- Known limitations: single illustrative OASIS pair only; IDIR field-fit
  samples `10,000` fixed-image-positive coordinates per epoch, while spline
  variants expand their cps4 dense field and compute MSE on all
  fixed-image-positive voxels.
- Acceptance: `9/9` new SITReg field-fit `DONE` files, `18/18` combined rows,
  `45,000` history rows, and `errors=[]`.
- Actual best final points: reused image-registration Dice favored SINR cps4 at
  lr `1e-4` with Dice `0.8296`; SITReg-field brain-mask MSE favored DirectCP
  cps4 at lr `1e-2` with MSE `7.77649e-06`.
- Next decision: inspect whether SITReg-field fitting changes the relative
  optimization behavior compared with Greedy-field fitting.

### P1g2: OASIS Rank-9 Final-CPS Matched Panel A/B Field-Fit Add-On

- Status: completed 2026-07-01; smoke `4/4` and full run `4/4` completed.
- Owner/agent: Codex.
- Date planned: 2026-07-01.
- Server: current workstation/container host using default Docker container
  `py310sitreg_v1_test1`.
- GPU model/count: 4 x NVIDIA RTX 6000 Ada Generation visible in Docker.
- Git commit: `0ceda03` at planning time, with local uncommitted artifact/code
  changes for this add-on.
- Branch: `gpu8`.
- Docker image tag: `hengjieliu96/dl_dir:py310sitreg_v1`.
- Purpose: switch OASIS Panel A/B fitting comparison from single-stage cps4 to
  single-stage cps2 for final-control-spacing matching against OASIS
  DirectCP `8-4-2`, while leaving LungCT cps4/`32-16-8-4` unchanged.
- Dataset: OASIS BrainMR seed-3 shuffled rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`.
- New methods/objectives: `sinr_cps2_be_bspline_analytic` and
  `direct_cp_cps2_be_bspline_analytic` direct field fitting to both saved BS4
  SITReg and canonical Greedy dense displacement targets.
- Reused rows: compatible OASIS cps2 image-registration histories from the
  first10/DirectCP-cps2 add-on roots, plus existing IDIR and DirectCP `8-4-2`
  Panel A/B rows.
- LR/alpha: SINR cps2 `lr=1e-4`, alpha `1000`; DirectCP cps2 `lr=1e-3`,
  alpha `1000`.
- Field-fit loss: masked normalized-grid target-field MSE over fixed raw
  image-positive foreground, no regularizer.
- Epochs/eval: `2500` epochs, eval interval `250`, seed `1`.
- Output root:
  `${DATA_ROOT}/runs/oasis_rank009_final_cps2_field_fit_v1`.
- Review root:
  `experiments/2026-07-01_oasis_panelAB_final_cps2_standardization/`.
- Paper artifact output:
  `experiments/paper_artifacts/fitting_figure/v002_candidate/`.
- Standards preflight: OASIS raw image-positive foreground rule; final-control
  spacing policy in `experiments/paper_artifacts/artifact_standard.json`.
- Known limitations: single illustrative OASIS pair only; field-fit IDIR is not
  rerun because the matching issue concerns spline final control spacing.
- Next decision: compare v001 cps4-vs-v002 cps2 Panel A/B curves and decide
  whether to promote v002 candidate.
- Acceptance: all four new cps2 field-fit histories reached `2500` epochs with
  finite final MSE values; v002 candidate and comparison artifacts were
  generated under `experiments/paper_artifacts/`.

### P1h: OASIS Seed-3 First20 DirectCP 8-4-2 Alpha1000 Preliminary

- Status: completed 2026-06-26; final aggregation passed with `20/20` new
  summaries and no errors.
- Purpose: test whether a LungCT-style coarse-to-fine DirectCP path improves
  the OASIS cps2 operating point before expanding to the full six-alpha sweep.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-19`. The first launch covers
  ranks `0-9`; a same-setting extension launch covers ranks `10-19`.
- Method: DirectCP multistage `8,4,2` with exact cubic B-spline dyadic knot
  insertion between stages and image downsampling.
- Objective: same OASIS spline objective as SINR/DirectCP alpha sweeps:
  full-image global NCC plus `alpha *` analytic cubic B-spline BE. No
  foreground mask in the image loss; foreground regularity uses fixed raw
  image-positive mask.

### P7: OASIS Rank-9 SINR CP-Space NTK Conditioning

- Status: completed 2026-06-29 after dry-run, smoke, full first pass, and
  artifact verification.
- Owner/agent: Codex.
- Purpose: test whether SINR's coefficient INR induces a rank-deficient,
  anisotropic, or amplifying CP-space tangent kernel compared with DirectCP's
  identity kernel in the same cps4 B-spline control space.
- Dataset: OASIS BrainMR seed-3 shuffled rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`.
- Method state: SINR cps4 BE alpha `1000`, lr `1e-4`; compare local kernel
  action against the hypothetical DirectCP cps4 update from the same CP-space
  gradient.
- Protocol tag: `oasis_rank9_sinr_cps4_ntk_conditioning_local_gradient_action`.
- Planned diagnostics: `B s` versus `B J_theta J_theta^T s`, dense-action
  cosine, scalar-rescaled residual, CP/dense amplification, PSD/Rayleigh checks,
  random Rayleigh probes, matrix-size/rank/nullity summary.
- Output root:
  `/database/inrorspline/runs/sinr_ntk_conditioning_20260629_v1`.
- Review root:
  `experiments/2026-06-29_sinr_ntk_conditioning/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Known limitations: local NTK/SGD diagnostic only; initial scope is one OASIS
  pair; dense kernels are not materialized; Adam trajectory effects are not
  directly modeled.
- Actual result: `4` action rows and `128` spectrum-probe rows completed with
  `errors=[]`. The SINR CP-space kernel is rank-deficient by construction in
  the cps4 space: `388,161` CP coefficients versus `133,379` SINR parameters,
  giving nullity lower bound `254,782`. Sampled dense update cosine was
  `0.2888/0.2888` at init for data/total gradients and `0.4638/0.4649` at
  final; scalar-rescaled residual remained high (`0.9574` init and about
  `0.885` final). Dense-action amplification was large (`1.21e6` init data and
  `2.38e5` final data).
- Next decision: decide whether to extend to replayed checkpoints, first10
  OASIS cases, or LungCT case01/case08.
- Fixed parameters: alpha `1000`, learning rate `1e-2`, stage downsample
  `4,2,1`, stage epochs `400,800,1300`, seed `1`, eval interval `250`.
- Planned GPUs: first10 on `2,3` with max concurrency `2`; extension ranks
  `10-19` on `0,1` with max concurrency `2`.
- Output root:
  `/database/inrorspline/runs/oasis_seed3_first10_directcp842_alpha1000_prelim_v1`.
- Review root:
  `experiments/2026-06-26_oasis_seed3_first10_directcp842_alpha1000_prelim/`.
- Comparator rows: existing first10 IDIR best-Dice alpha `1`, SINR cps4/cps2
  alpha `1000`, and DirectCP cps4/cps2 alpha `1000`.
- Acceptance: compile/import passes; rank-0 `1,1,1` stage-epoch smoke passes;
  first10 dry-run queues exactly `10` jobs on GPUs `2,3`; extension dry-run
  queues exactly `10` jobs on GPUs `0,1`; full run produces `20/20` summaries
  with dense fields, regularity metrics, BE/diffusion CSVs, and
  stage/prolongation diagnostics; aggregation writes separate first10 and
  first20 comprehensive comparison tables.
- Known limitations: alpha `1000` only and no full six-alpha DirectCP 8-4-2
  sweep in this preliminary step.
- Result: DirectCP `8-4-2` alpha `1000` was the best mean-Dice row in both
  generated tables. First10 Dice/HD95/symmetric ASD/fold%:
  `0.8054` / `2.2429` / `0.4507` / `4.3015`. First20:
  `0.8077` / `2.2589` / `0.4531` / `4.4071`.

### P1i: OASIS Seed-3 First20 DirectCP 8-4-2 Remaining Alpha Sweep

- Status: completed 2026-06-26.
- Date planned: 2026-06-26.
- Purpose: complete the full six-alpha DirectCP `8,4,2` OASIS curve and add it
  to the saved-baseline paper-style first10/first20 Pareto figures without
  overwriting the existing five-sweep package.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-19`, using the documented
  first20 continuation of `pair_test_200`.
- Method: DirectCP multistage `8,4,2` with exact cubic B-spline dyadic knot
  insertion between stages and image downsampling.
- Objective: full-image global NCC plus `alpha *` analytic cubic B-spline BE;
  foreground regularity uses fixed raw image-positive mask.
- New alpha grid: `0,10,100,10000,100000`; alpha `1000` is reused from P1h and
  must not be relaunched.
- Fixed parameters: learning rate `1e-2`, stage downsample `4,2,1`, stage
  epochs `400,800,1300`, seed `1`, eval interval `250`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2,3` with
  `--max-concurrent 4`.
- Remaining-alpha output root:
  `/database/inrorspline/runs/oasis_seed3_first20_directcp842_alpha_sweep_remaining5_v1`.
- Reused alpha1000 root:
  `/database/inrorspline/runs/oasis_seed3_first10_directcp842_alpha1000_prelim_v1`.
- Review root:
  `experiments/2026-06-26_oasis_seed3_first20_directcp842_alpha_sweep/`.
- New figure root:
  `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/pareto_first10_first20_alpha5_saved_baselines_plus_directcp842_paper_v1/`.
- Acceptance: compile/import passes; rank-0 smoke with stage epochs `1,1,1`
  and alpha `0` passes; dry-run queues exactly `100` jobs, ranks `0-19`,
  alphas `0,10,100,10000,100000`, GPUs `0,1,2,3`, and no alpha `1000`; full
  run produces `100/100` remaining-alpha `DONE` files; aggregation validates
  `100` new rows plus `20` reused alpha1000 rows; figure generation writes
  `41` population rows per cohort and the expected PDF/PNG plot families.
- Known limitations: this completes the DirectCP `8,4,2` alpha curve only; the
  previous five-sweep figure package remains the historical baseline.
- Completion note: Docker compile and rank-0 alpha-0 `1,1,1` stage-epoch smoke
  passed; dry-run queued exactly `100` jobs with no alpha `1000`; full launch
  completed `100/100` new jobs with `0` skipped and `0` failures in `6697`
  seconds. Aggregation reused the P1h alpha `1000` root and validated `100`
  remaining rows, `20` reused alpha1000 rows, and `120` combined DirectCP
  `8,4,2` rows with `errors=[]`. The new plus-DirectCP842 figure package wrote
  `360` first10 sweep rows, `720` first20 sweep rows, `41` population rows per
  cohort, and `162` non-empty PDF/PNG files per cohort.
- Result: DirectCP `8,4,2` alpha `1000` remained the best mean-Dice point on
  both cohorts; alpha `10000` had best mean HD95. First10 Dice/HD95/symmetric
  ASD/fold% at alpha `1000`: `0.8054` / `2.2429` / `0.4507` / `4.3015`.
  First20: `0.8077` / `2.2589` / `0.4531` / `4.4071`.

### P1j: Fit-Modes Paper Figure IDIR Matched-Similarity Curves

- Status: completed on 2026-06-26; compile, smoke, full launch, and CSV
  validation passed.
- Date planned: 2026-06-26.
- Purpose: add the missing IDIR image-similarity evaluation curves needed for a
  paper-quality figure comparing registration optimization curves with direct
  target-field fitting curves.
- Dataset/cases: OASIS BrainMR seed-3 rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`; LungCT pTV-preprocessed case `01`; LungCT
  pTV-preprocessed case `08`.
- Methods: IDIR only. Existing SINR/DirectCP histories from P1g and the
  LungCT case01/case08 fit-mode sweeps are reused.
- Training objectives: unchanged IDIR objectives. OASIS uses sampled fixed
  image-positive NCC plus IDIR coordinate-autograd BE with alpha `1`. LungCT
  uses pTV-preprocessed fixed-T00 lung-mask sampled NCC plus IDIR bending with
  `alpha_bending=10`.
- New diagnostic eval curves: OASIS full-image global NCC every `10` epochs;
  LungCT pTV LCC data loss every `10` epochs using the same border-masked
  pTVReg 1mm crop convention as the LungCT SINR/DirectCP runs.
- Fixed parameters: lr `1e-4`, 2500 epochs, seed `1`, normal summary eval
  interval `250`, image-similarity eval interval `10`.
- Output root:
  `/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1`.
- Review root:
  `experiments/2026-06-26_fit_modes_paper_figure/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2`.
- Primary metrics: `image_similarity_eval.csv` curves, final Dice for OASIS,
  final TRE for LungCT, and existing field-fit MSE curves for later notebook
  composition.
- Known limitations: eval curves are diagnostic only and do not affect
  training. The OASIS registration curve is global NCC; the LungCT registration
  curve is pTV LCC data loss for dataset-protocol matching.
- Acceptance: Docker compile passed for modified OASIS and LungCT runners;
  20-epoch smoke produced eval CSVs with epochs `1,10,20`; full run completed
  all three IDIR jobs and each `image_similarity_eval.csv` has `251` rows
  covering epoch `1` plus every `10` epochs through `2500`.
- Final eval values: OASIS full-image global NCC value `0.9750214815`; LungCT
  case01 pTV LCC data loss `2817243.5`; LungCT case08 pTV LCC data loss
  `5032755.5`.
- Next decision: generate the notebook figure combining these IDIR eval curves
  with existing SINR/DirectCP registration histories and SITReg/pTVReg
  field-fit MSE histories.

### P1k: Fit-Modes Paper Figure DirectCP 4-Stage Field-Fit Finest-MSE Diagnostics

- Status: completed 2026-06-26; compile, smoke, two reruns, and notebook
  update passed.
- Date planned: 2026-06-26.
- Purpose: replace the LungCT DirectCP 4-stage target-field fitting curve in the
  paper figure with a true finest-domain diagnostic curve rather than a
  stage-domain coarse-to-fine training curve.
- Dataset/cases: LungCT pTV-preprocessed case `01` and case `08`.
- Method/objective: DirectCP 4-stage cps4 pTVReg field fitting only. Training
  remains MSE-only with the existing schedule `32,16,8,4`, downsample
  `8,4,2,1`, stage epochs `400,400,800,900`, lr `1e-2`.
- New diagnostic curve: `finest_data_loss` equals masked normalized-grid MSE on
  the original 1mm pTVReg crop and fixed T00 lung mask, evaluated posthoc every
  50 epochs during coarse stages and every epoch during the final 1mm stage;
  diagnostic only and does not affect training.
- Output root:
  `/database/inrorspline/runs/fit_modes_paper_curves_directcp4stage_field_mse_diag_v1`.
- Review root:
  `experiments/2026-06-26_fit_modes_paper_figure/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `2,3` if available.
- Acceptance: modified field-fit runner compiles; smoke validates
  `finest_data_loss` and `finest_field_mse_objective_normalized_grid` columns;
  full case01/case08 reruns completed with `2500` history rows and `935` finite
  finest-domain diagnostic points per case; notebook uses these finest-domain
  MSE curves.
- Final diagnostic values: LungCT case01 `finest_data_loss=2.846068e-08`;
  LungCT case08 `finest_data_loss=9.682780e-08`.
- Known limitations: DirectCP 4-stage training still optimizes stage-domain MSE
  at coarse stages. The new curve is posthoc finest-domain diagnostic only.

### P1l: Fit-Modes rRMSE Learning-Rate Trajectory Comparison

- Status: completed 2026-07-01; smoke, four diagnostic reruns, notebook
  execution, and CSV validation passed.
- Owner/agent: Codex.
- Date planned: 2026-07-01.
- Git commit: `1872fe0` on branch `gpu8` with figure/report edits in the
  worktree.
- Docker image tag/digest:
  `hengjieliu96/dl_dir:py310sitreg_v1@sha256:8a159ba5e7f438cf36ea10d74a4a45fac1b0250d7d70f0dba114eefcbe3416da`.
- Purpose: create a field-fit-only rRMSE trajectory figure comparing requested
  learning rates for IDIR, SINR cps4, DirectCP cps4, and LungCT DirectCP
  4-stage.
- Dataset/cases: OASIS BrainMR seed-3 rank `9` SITReg-field and Greedy-field
  targets; LungCT pTV-preprocessed cases `01` and `08` pTVReg-field targets.
- Reused inputs: existing OASIS single-stage field-fit histories and LungCT
  single-stage field-fit histories; existing LungCT DirectCP 4-stage
  `lr=1e-2` finest-domain field-MSE diagnostic.
- New diagnostic runs: LungCT case01/case08 DirectCP 4-stage pTVReg-field
  fitting at `lr=1e-3` and `lr=1e-4`, using the same schedule
  `32,16,8,4`, downsample `8,4,2,1`, stage epochs `400,400,800,900`, seed
  `1`, and 2500 epochs. Training remains stage-domain MSE-only; posthoc
  finest-domain masked normalized-grid MSE is logged every 50 epochs during
  coarse stages and every epoch during the final 1mm stage.
- Output root:
  `/database/inrorspline/runs/fit_modes_paper_curves_directcp4stage_field_mse_diag_v1`.
- Review root:
  `experiments/2026-06-26_fit_modes_paper_figure/`.
- Primary metrics: relative target-field RMSE percent,
  `100 * sqrt(mse / (target_power + 1e-12))`, using the same masked
  normalized-grid component-MSE convention as the original field-fit losses.
- Planned review artifacts:
  `fit_modes_rrmse_lr_trajectory_curves.{png,pdf}`,
  `fit_modes_rrmse_lr_trajectory_curve_summary.csv`, and
  `fit_modes_rrmse_lr_trajectory_selected_runs.csv`.
- Acceptance: two-epoch smoke wrote finite `finest_data_loss` and
  `finest_field_mse_objective_normalized_grid` columns; all four new full
  diagnostic histories have `2500` rows and `935` finite finest-domain
  diagnostic points; final finest-domain history values are within `1%` of the
  corresponding original-sweep final summaries. The notebook executed in
  Docker and wrote the requested rRMSE LR trajectory PNG/PDF/CSVs.
- Final finest-domain history values: case01 lr `1e-3`
  `2.894892e-08`, case01 lr `1e-4` `3.456253e-08`, case08 lr `1e-3`
  `9.883180e-08`, case08 lr `1e-4` `3.594418e-05`.
- Known limitations: no OASIS DirectCP 4-stage runs are added; those panels are
  marked not run. The DirectCP 4-stage coarse-stage rRMSE curves are posthoc
  diagnostics and are not the loss used for backpropagation at coarse stages.

### P1a: OASIS Seed-3 Case-0 Alpha Pre-Run

- Status: completed; aggregation passed with 12/12 runs and no acceptance errors.
- Date planned: 2026-06-25.
- Dataset: OASIS BrainMR seed-3 shuffled rank `0`, original sorted index `158`,
  fixed/moving `(0333, 0115)`.
- Methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4.
- Grid: IDIR alpha `0.01,0.1,1`; spline alpha `100,1000,10000`; 1000 epochs.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Output root: `/database/inrorspline/runs/oasis_seed3_case0_prerun_be_1000e_v1`.
- Review root: `experiments/2026-06-25_oasis_seed3_case0_prerun/`.
- Actual launcher elapsed: 504.5 seconds; large output size: 1.1 GB; review
  artifact size: 4.6 MB.
- Notes: native analytic regularity means IDIR coordinate-autograd Hessian BE
  and Jacobian diffusion; spline methods use cubic B-spline analytic BE and
  diffusion.
- Next decision: user reviews pre-run Pareto/contact-sheet outputs before full
  overnight first10 launch.

### P1b: OASIS Original Sorted Case-0 Alpha Pre-Run

- Status: completed; aggregation passed with 12/12 runs and no acceptance errors.
- Date planned: 2026-06-25.
- Dataset: OASIS BrainMR original sorted `pair_test_200.txt` row `0`,
  fixed/moving `(0004, 0236)`.
- Methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4.
- Grid: IDIR alpha `0.01,0.1,1`; spline alpha `100,1000,10000`; 1000 epochs.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Output root: `/database/inrorspline/runs/oasis_sorted_case0_prerun_be_1000e_v1`.
- Review root: `experiments/2026-06-25_oasis_sorted_case0_prerun/`.
- Actual launcher elapsed: 504.5 seconds; large output size: 1.1 GB; review
  artifact size: 4.6 MB.
- Notes: native analytic regularity means IDIR coordinate-autograd Hessian BE
  and Jacobian diffusion; spline methods use cubic B-spline analytic BE and
  diffusion.
- Next decision: compare original sorted case-0 and seed-3 shuffled case-0
  curves before choosing first10 full-run alpha ranges.

### P2: OASIS Seed-3 First10 Deformation-Regime Stats

- Status: planned.
- Purpose: regenerate deformation-regime statistics on the same first10 protocol.

### P2a: LungCT pTV-Prep No-Sweep Reproduction

- Status: completed; 40/40 jobs completed with return code 0 and aggregation
  passed.
- Date planned: 2026-06-25.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Methods: IDIR BE default with fixed T00 lung-mask sampling, SINR cps4 pTV LCC
  plus isoTV, DirectCP cps4 pTV LCC plus isoTV, DirectCP multistage
  `32,16,8,4` pTV LCC plus isoTV.
- Fixed parameters: IDIR `alpha_bending=10`, `lr=1e-4`, `epochs=2500`;
  SINR cps4 `lr=1e-4`, `epochs=2500`; DirectCP cps4 `lr=1e-3`,
  `epochs=2500`; multistage epochs `200,400,800,1000`, `lr=1e-3`.
- pTV loss settings: `sigma_mm=2.1`, `isoTV weight=0.11`, `csqrt=5e-3`,
  `border_mask_voxels=5`.
- Mask convention: IDIR uses fixed T00 lung mask for sampled-coordinate
  training; pTV SINR/DirectCP losses do not use lung-mask weighting; all
  foreground regularity metrics use the fixed T00 lung mask on the cropped 1mm
  grid.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Output root: `/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1`.
- Review root: `experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/`.
- Primary metrics: original/native-crop identity/snap/subvoxel TRE,
  cropped-1mm identity/snap/subvoxel TRE, fixed-lung-mask foreground
  log-Jacobian std, folding percentage, bending energy, diffusion energy,
  displacement magnitude, runtime, peak GPU memory.
- Actual result, original/native-crop snap/sub TRE: IDIR `1.017/1.133`;
  SINR cps4 `2.904/3.014`; DirectCP cps4 `6.370/6.450`; DirectCP multistage
  `0.922/1.049`. Cropped-1mm diagnostic snap/sub TRE: IDIR `1.236/1.138`;
  SINR cps4 `3.143/3.037`; DirectCP cps4 `6.575/6.495`; DirectCP multistage
  `1.168/1.052`.
- Review artifacts: `summary_runs.csv`, `summary_runs.json`, and
  `review_report.md` under the review root.

### P2b: LungCT pTV-Prep IDIR + DirectCP 32-16-8-4 Regularization Sweep

- Status: completed; 80/80 nonbaseline jobs completed with return code 0 and
  aggregation passed with 100 case rows and 10 mean rows.
- Date planned: 2026-06-25.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Methods: IDIR BE default with fixed T00 lung-mask sampling and DirectCP
  multistage `32,16,8,4` pTV LCC plus isoTV.
- Sweep: reuse completed `1e0` baseline; launch nonbaseline factors
  `0.01,0.1,10,100` around each method's default regularization weight.
- IDIR regularization: base `alpha_bending=10`; swept values
  `0.1,1,100,1000`.
- DirectCP regularization: base `ptv_isotv_weight=0.11`; swept values
  `0.0011,0.011,1.1,11`.
- Fixed parameters: IDIR `lr=1e-4`, `epochs=2500`; DirectCP stage epochs
  `200,400,800,1000`, `lr=1e-3`; pTV settings `sigma_mm=2.1`,
  `csqrt=5e-3`, `border_mask_voxels=5`.
- Primary metric: `original_snap` TRE. Track all four TREs:
  `original_snap`, `original_subvoxel`, `reg1mm_snap`, and
  `reg1mm_subvoxel`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Baseline root: `/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1`.
- Sweep output root: `/database/inrorspline/runs/lungct_ptvprep_reg_sweep_idir_directcp32_v1`.
- Review root: `experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/`.
- Actual result, primary original snap TRE: IDIR factors
  `0.01/0.1/1/10/100` gave `1.317/1.046/1.017/1.085/1.257` mm;
  DirectCP factors `0.01/0.1/1/10/100` gave
  `1.320/1.022/0.922/1.583/6.809` mm.
- Best primary point: default `1e0` for both methods, IDIR
  `alpha_bending=10` with original snap TRE `1.017` mm and DirectCP
  `ptv_isotv_weight=0.11` with original snap TRE `0.922` mm.
- Compute cost, default `1e0`: IDIR `2500` epochs, `2.63 +/- 0.17` min/case,
  `4.14 +/- 0.06` GB peak memory; DirectCP 4-stage `200,400,800,1000`
  stage epochs, `2400` total epochs, `3.90 +/- 0.71` min/case,
  `5.48 +/- 1.07` GB peak memory.
- Compute cost range, default `1e0`: IDIR `2.44-2.92` min/case and
  `4.05-4.25` GB; DirectCP 4-stage `3.03-5.84` min/case and `3.64-7.45` GB.
- Training-curve analysis: 50/50 planned IDIR histories confirm `lr=1e-4`;
  50/50 planned DirectCP histories confirm stage LR `[1e-3,1e-3,1e-3,1e-3]`
  and stage epochs `[200,400,800,1000]`. Objective curves support validating
  IDIR `1500/2000` epochs against `2500`, and DirectCP shortened schedules
  `200,400,300,650` and `200,200,200,400` against the current
  `200,400,800,1000` before changing canonical runs.
- Verification: dry-run queued exactly `80` nonbaseline jobs, smoke jobs passed
  for case `01` at factors `0.01` and `100` for both methods, and aggregation
  generated complete case/mean tables plus 8 method-specific Pareto plots and
  4 combined method-comparison Pareto plots.
- Refined DirectCP addendum: launched DirectCP-only factors `0.2,0.5,2,5` on
  cases `01-10` using GPUs `0,1`; 40/40 jobs completed with return code 0.
  These rows are aggregated in `directcp_refined_review.md` with non-overwriting
  figures under `plots_refined_directcp/`. Refined DirectCP original-snap TRE
  factors `0.2/0.5/1/2/5` gave `0.951/0.917/0.922/0.937/0.978` mm; best in
  this refined grid is `0.5x` (`ptv_isotv_weight=0.055`), marginally better
  than the original `1x` point.
- Refined IDIR addendum: launched IDIR BE factors `1/30,1/3,3` on cases
  `01-10` using GPUs `1,2,3`; 30/30 jobs completed with return code 0. These
  rows are aggregated in `both_refined_roi_no_extremes_review.md` with
  non-overwriting ROI figures under `plots_refined_both_roi_no_extremes/`.
  Refined IDIR original-snap TRE for factors `0.033333/0.333333/1/3/10`
  gave `1.244/1.028/1.017/1.037/1.085` mm; best remains `1x`
  (`alpha_bending=10`).

### P2c: LungCT DirectCP 32-16-8-4 Global NCC + Analytic BE Alpha Sweep

- Status: planned; implementation and launch requested by user.
- Date planned: 2026-06-25.
- Dataset: DIR-Lab 4DCT cases `01` and `08`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Method: DirectCP multistage `32,16,8,4` with exact crop-aligned cubic
  B-spline dyadic knot insertion.
- Objective: full-crop global NCC plus `alpha *` analytic cubic B-spline
  bending energy. No lung-mask or border-mask loss weighting.
- Sweep: alpha values `0,0.1,1,10,100,1000`; absolute weights, not factors.
- Fixed parameters: downsample schedule `8,4,2,1`, stage epochs
  `200,400,800,1000`, `lr=1e-3`, seed `1`.
- Mask convention: fixed T00 lung mask is used only for foreground regularity
  metrics.
- Primary metric: `original_snap` TRE. Track all four TREs:
  `original_snap`, `original_subvoxel`, `reg1mm_snap`, and
  `reg1mm_subvoxel`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Output root: `/database/inrorspline/runs/lungct_directcp32_global_ncc_be_case01_case08_alpha_sweep_v1`.
- Review root:
  `experiments/2026-06-25_lungct_directcp32_global_ncc_be_case01_case08_alpha_sweep/`.
- Planned command:
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_directcp32_global_ncc_be_sweep.py --cases 1,8 --alphas 0,0.1,1,10,100,1000 --gpus 2,3`.
- Known limitations: selected-case diagnostic only; global NCC + analytic BE is
  not pTVReg-matched because pTVReg uses local CC and isoTV.
- Next decision: inspect case-wise TRE/regularity curves before deciding whether
  to expand to all 10 cases or refine alpha locally.

### P2d: LungCT DirectCP 32-16-8-4 Global NCC + Dense FD-BE Alpha Sweep

- Status: completed; initial 12/12 selected-case jobs completed with return
  code 0, extension 6/6 larger-alpha jobs completed with return code 0, and
  aggregation passed with 18 rows.
- Date planned: 2026-06-25.
- Dataset: DIR-Lab 4DCT cases `01` and `08`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Method: DirectCP multistage `32,16,8,4` with exact crop-aligned cubic
  B-spline dyadic knot insertion.
- Objective: full-crop global NCC plus `alpha *` dense forward-difference
  bending energy on the dense displacement field. No lung-mask or border-mask
  loss weighting.
- Sweep: alpha values `0,0.1,1,10,100,1000,10000,100000,1000000`;
  absolute weights, not factors.
- Fixed parameters: downsample schedule `8,4,2,1`, stage epochs
  `200,400,800,1000`, `lr=1e-3`, seed `1`.
- Mask convention: fixed T00 lung mask is used only for foreground regularity
  metrics.
- Primary metric: `original_snap` TRE. Track all four TREs:
  `original_snap`, `original_subvoxel`, `reg1mm_snap`, and
  `reg1mm_subvoxel`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Output root: `/database/inrorspline/runs/lungct_directcp32_global_ncc_fdbe_case01_case08_alpha_sweep_v1`.
- Review root:
  `experiments/2026-06-25_lungct_directcp32_global_ncc_fdbe_case01_case08_alpha_sweep/`.
- Planned command:
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_directcp32_global_ncc_fdbe_sweep.py --cases 1,8 --alphas 0,0.1,1,10,100,1000,10000,100000,1000000 --gpus 2,3`.
- Actual result, primary original snap TRE: case `01` alphas
  `0/0.1/1/10/100/1000/10000/100000/1000000` gave
  `1.254/1.240/1.208/1.093/1.004/0.880/0.755/0.734/0.815` mm;
  case `08` gave
  `7.042/6.975/7.027/6.205/3.017/1.048/1.087/1.110/1.375` mm.
- Best primary point in this extended selected-case grid: case `01` at alpha
  `100000` with original snap TRE `0.734` mm and foreground stdLogJ/J<=0%
  `0.071/0.0000%`; case `08` at alpha `1000` with original snap TRE
  `1.048` mm and foreground stdLogJ/J<=0% `0.355/0.4173%`.
- Verification: compile/import passed, loader audit passed for cases `01,08`,
  launcher dry-runs queued exactly `12` initial jobs and `6` extension jobs over
  GPUs `2,3`, smoke jobs passed for case `01` alphas `0` and `1000`, all
  launched jobs completed with return code 0, and aggregation generated complete
  case tables plus four per-case plots.
- Known limitations: selected-case diagnostic only; dense FD-BE is not the same
  regularizer as analytic cubic B-spline BE, and global NCC + FD-BE is not
  pTVReg-matched because pTVReg uses local CC and isoTV.
- Next decision: refine around case-specific optima, especially alpha
  `10000-1000000` for case `01` and `1000-10000` for case `08`, before any
  all-case expansion.

### P2e: LungCT DirectCP 4-Stage Global NCC + Dense FD-BE All-Case Sweep

- Status: completed; launch, aggregation, and ellipse-style comparison plots
  completed.
- Date planned: 2026-06-25.
  Date completed: 2026-06-25.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Method: DirectCP multistage `32,16,8,4` with exact crop-aligned cubic
  B-spline dyadic knot insertion.
- Objective: full-crop global NCC plus `alpha *` dense forward-difference
  bending energy on the dense displacement field. No lung-mask or border-mask
  loss weighting.
- Sweep: alpha values `0,10,100,1000,10000,100000`; absolute weights, not
  factors.
- Reuse policy: reuse completed case `01` and `08` rows from P2d for the six
  requested alphas; launch missing cases `02-07,09,10` only.
- New jobs: `48` = 8 missing cases times 6 alpha values.
- Fixed parameters: downsample schedule `8,4,2,1`, stage epochs
  `200,400,800,1000`, `lr=1e-3`, seed `1`.
- Mask convention: fixed T00 lung mask is used only for foreground regularity
  metrics.
- Primary metric: `original_snap` TRE. Track all four TREs:
  `original_snap`, `original_subvoxel`, `reg1mm_snap`, and
  `reg1mm_subvoxel`.
- Comparison: generate new non-overwriting all-case ellipse-style Pareto figures
  against IDIR BE and DirectCP 4-stage pTV LCC + isoTV.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Existing selected-case root: `/database/inrorspline/runs/lungct_directcp32_global_ncc_fdbe_case01_case08_alpha_sweep_v1`.
- New missing-case output root: `/database/inrorspline/runs/lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep_v1`.
- Review root:
  `experiments/2026-06-25_lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep/`.
- Launch command:
  `docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_global_ncc_fdbe_all10_sweep.py --gpus 0,1,2,3`.
- Verification: compile/import passed; loader audit passed for cases `01-10`;
  launcher dry-run queued exactly `48` missing-case jobs; smoke job on case
  `02` alpha `100000` with stage epochs `1,1,1,1` passed; full launch
  completed `48/48` jobs with return code 0; aggregation produced `60/60`
  FD-BE rows, `160` comparison rows, and all-case ellipse-style plots.
- Best FD-BE mean point: alpha `10000`, original-snap TRE
  `0.966 +/- 0.151` mm, foreground stdLogJ `0.185452`, foreground
  `J <= 0` `0.0037%`.
- Comparator best points in the same all-case ellipse report: IDIR BE alpha
  `10` gives original-snap TRE `1.017 +/- 0.176` mm, foreground stdLogJ
  `0.142869`, foreground `J <= 0` `0.0000%`; DirectCP 4-stage pTV LCC +
  isoTV weight `0.11` gives original-snap TRE `0.922 +/- 0.152` mm,
  foreground stdLogJ `0.179961`, foreground `J <= 0` `0.0138%`.
- Known limitations: all-case run still uses dense FD-BE rather than analytic
  cubic B-spline BE and is not pTVReg-matched because pTVReg uses local CC and
  isoTV.
- Next decision: decide whether the FD-BE objective is worth refining around
  alpha `1000-100000`, given that its best all-case accuracy is close to but
  still behind the pTV LCC + isoTV DirectCP best point.

### P2f: LungCT Case08 DirectCP 4-Stage FD-BE vs Analytic B-Spline BE

- Status: completed; two-method diagnostic completed for case `08`.
- Date planned: 2026-06-26.
  Date completed: 2026-06-26.
- Dataset: DIR-Lab 4DCT case `08`, canonical regenerated pTVReg-style landmark
  crop resampled to 1mm.
- Methods: DirectCP multistage `32,16,8,4` with exact crop-aligned cubic
  B-spline dyadic knot insertion, comparing full-crop global NCC plus
  `alpha *` dense forward-difference BE against full-crop global NCC plus
  `alpha *` analytic cubic B-spline BE.
- Fixed parameters: alpha `10000`, downsample schedule `8,4,2,1`, stage epochs
  `200,400,800,1000`, `lr=1e-3`, seed `1`.
- Mask convention: fixed T00 lung mask is used only for foreground regularity
  metrics.
- Primary metric: `original_snap` TRE. Track all four TREs and foreground
  regularity metrics.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Dataset root: `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Output root:
  `/database/inrorspline/runs/lungct_case08_be_impl_compare_alpha1e4_v1`.
- Review root:
  `experiments/2026-06-26_lungct_case08_be_impl_compare_alpha1e4/`.
- Launch: ran the FD-BE job on GPU `2` and analytic B-spline BE job on
  GPU `3` concurrently.
- Verification: compile/import passed; loader audit passed for case `08`;
  minimal smoke passed for both BE variants; full launch completed both jobs
  with status `completed`; comparison rows and review report were generated.
- Result: original-snap TRE tied after snap-to-voxel rounding at `1.087383` mm.
  Analytic B-spline BE was slightly better on original-subvoxel TRE
  (`1.145162` vs `1.146554` mm) and foreground stdLogJ (`0.191389` vs
  `0.192354`), but took longer (`367.985` vs `351.897` seconds). Peak memory
  was identical at `6.492` GB.
- Known limitations: selected-case diagnostic only; alpha `10000` is the same
  numeric weight but not guaranteed to imply matched effective regularization
  scale between dense FD-BE and analytic B-spline BE.

### P3: LungCT 4DCT Matched Pareto

- Status: planned.
- Methods: IDIR, SINR, single-stage DirectCP cps4, multistage DirectCP, pTVReg references.
- Required: `docs/standards/lungct_protocol_tags.md` and explicit protocol
  tags for preprocessing/domain/mask/loss.

### P2g: LungCT Case01 Registration vs pTVReg-Field Fitting LR Sweep

- Status: completed; full 24-job run and aggregation passed 2026-06-26.
- Date planned: 2026-06-26.
- Dataset: DIR-Lab 4DCT case `01`, canonical regenerated pTVReg-style landmark
  crop resampled to 1mm.
- Methods: IDIR, SINR cps4, DirectCP cps4, and DirectCP 4-stage
  `32,16,8,4`.
- Objectives: image registration and direct MSE fitting to the saved pTVReg
  case01 displacement field.
- Learning rates: `1e-2,1e-3,1e-4`; total jobs `24`.
- Scheduling: method batches run serially. Within each method batch, the six
  objective/LR jobs are distributed across GPUs `0,1,2,3`.
- Field-fit MSE objective domain: fixed T00 lung mask in the 1mm crop; full
  crop MSE is diagnostic only.
- Registration weights: IDIR `alpha_bending=10`; pTV LCC+isoTV methods
  `ptv_isotv_weight=0.11`.
- DirectCP 4-stage schedule: `32,16,8,4`, downsample `8,4,2,1`, stage epochs
  `400,400,800,900`.
- pTVReg source field:
  `/database/inrorspline/ptvreg/4DCT/case01_T50_to_T00_deformation.mat`.
  The saved HDF5 field is `Tptv_rsz` with h5py shape `(3,z,y,x)`, equivalent
  to MATLAB `T(x,y,z,component)`, physical-mm `x,y,z` components, fixed-to-moving.
- Required target validation: reproduce saved pTVReg `TRE_phys=0.7671410824`
  and `TREstd_case=0.8917488134` using the pTVReg snap-to-voxel convention.
- Output root:
  `/database/inrorspline/runs/lungct_case01_ptvreg_fit_modes_lr_sweep_v1`.
- Review root:
  `experiments/2026-06-26_lungct_case01_ptvreg_fit_modes_lr_sweep/`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Known limitations: the field-fit target is reconstructed on the repo's
  reg1mm grid from pTVReg's saved native/full displacement output; the original
  pTVReg internal reg1mm displacement is not stored in the `.mat`.
- Restart note: an initial method-pinned launch was stopped and replaced with
  method-batch scheduling because method runtimes are imbalanced.
- Completion note: final review includes full LR curves, final metric-vs-LR
  plots, and best-LR masked-MSE/registration-loss plots with continuous
  DirectCP 4-stage epoch axes.

### P2h: LungCT Case08 Registration vs pTVReg-Field Fitting LR Sweep

- Status: completed; full 24-job run and aggregation passed 2026-06-26.
- Date planned: 2026-06-26.
- Dataset: DIR-Lab 4DCT case `08`, canonical regenerated pTVReg-style landmark
  crop resampled to 1mm.
- Methods, objectives, learning rates, scheduling, masks, and regularization:
  same as P2g Case01.
- DirectCP 4-stage schedule: `32,16,8,4`, downsample `8,4,2,1`, stage epochs
  `400,400,800,900`.
- pTVReg source field:
  `/database/inrorspline/ptvreg/4DCT/case08_T50_to_T00_deformation.mat`.
  The saved HDF5 field is `Tptv_rsz` with h5py shape `(3,128,512,512)`,
  physical-mm `x,y,z` components, fixed-to-moving.
- Required target validation: reproduce saved pTVReg
  `TRE_phys=1.0142198317` and `TREstd_case=1.2922227355` using the pTVReg
  snap-to-voxel convention.
- Output root:
  `/database/inrorspline/runs/lungct_case08_ptvreg_fit_modes_lr_sweep_v1`.
- Review root:
  `experiments/2026-06-26_lungct_case08_ptvreg_fit_modes_lr_sweep/`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Acceptance: smoke and full aggregation must report `24/24` completed rows;
  final visualization must match Case01, including best-LR masked-MSE and
  registration-loss plots with continuous DirectCP 4-stage epoch axes.
- Known limitations: same as Case01; the field-fit target is reconstructed on
  the repo's reg1mm grid from pTVReg's saved native/full displacement output.
- Completion note: final review includes full LR curves, final metric-vs-LR
  plots, and best-LR masked-MSE/registration-loss plots with continuous
  DirectCP 4-stage epoch axes.

### P2i: LungCT Case01/Case08 DirectCP 4-Stage Finest-Domain Loss Diagnostic

- Status: completed; smoke, full reruns, validation, and companion plots
  completed 2026-06-26.
- Date planned: 2026-06-26.
- Purpose: rerun only DirectCP 4-stage image registration at `lr=0.01` for
  cases `01` and `08`, preserving the original stage-wise training objective
  while recording a diagnostic loss on the finest 1mm crop domain at every
  epoch.
- Dataset: DIR-Lab 4DCT cases `01` and `08`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Method: `direct_cp_32_16_8_4_ptv_lcc_isotv`.
- Schedule: `32,16,8,4`, downsample `8,4,2,1`, stage epochs
  `400,400,800,900`, Adam, seed `1`, `lr=0.01`.
- Training objective: unchanged stage-domain pTV LCC + isoTV with
  `ptv_isotv_weight=0.11`, `border_mask_voxels=5`, LCC sigma `2.1 mm`.
- Diagnostic objective: pre-step, no-gradient full 1mm crop pTV LCC + isoTV
  after analytic cubic B-spline expansion of current controls to the finest
  grid; diagnostic only and does not affect optimization.
- Output root:
  `/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_v1`.
- Review root:
  `experiments/2026-06-26_lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08/`.
- Companion plots are written next to the accepted Case01/Case08 sweep plots
  without overwriting `best_lr_registration_loss_spline_methods.png`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.
- Acceptance: smoke passes for both cases; full reruns produce `2500`
  `stage_history.csv` rows with finite `finest_loss`, `finest_data_loss`, and
  `finest_reg_loss`; companion plots render continuous DirectCP 4-stage curves
  on epochs `1-2500`.
- Known limitations: this is a loss-curve diagnostic for two cases only; the
  finest-domain trace is not the training loss used for backprop on coarse
  stages.
- Completion note: case01 elapsed `258.6` sec and case08 elapsed `311.7` sec;
  both histories have `2500` finite diagnostic rows. Companion PNGs were written
  next to the accepted Case01 and Case08 spline-method loss plots with the
  suffix `_finest_domain_directcp4stage_lr0p01.png`.

### P2j: LungCT DirectCP 4-Stage Mid-Ratio Pareto Extension

- Status: completed.
- Date planned: 2026-06-26.
- Date completed: 2026-06-26.
- Purpose: add mid-ratio regularization points to improve the all-10
  TRE-versus-foreground-regularity Pareto visualization.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Methods:
  - `direct_cp_32_16_8_4_ptv_lcc_isotv`
  - `direct_cp_32_16_8_4_global_ncc_fd_be`
- New pTV-LCC+isoTV ratios: `0.3x` and `3x` around anchor
  `ptv_isotv_weight=0.11`, giving weights `0.033` and `0.33`.
- New global-NCC+FD-BE ratios: `0.3x` and `3x` around anchor alpha `10000`,
  giving alphas `3000` and `30000`.
- IDIR addendum ratios: `0.3x` and `3x` around anchor `alpha_bending=10`,
  giving alphas `3` and `30`. Alpha `30` is reused from the existing completed
  IDIR sweep root; alpha `3` is launched as 10 new jobs.
- Fixed parameters: `lr=1e-3`, seed `1`, schedule `32,16,8,4`, downsample
  `8,4,2,1`, stage epochs `200,400,800,1000`.
- Launch ordering: run all pTV-LCC+isoTV jobs first; after `20/20` complete,
  run all global-NCC+FD-BE jobs. pTV-LCC+isoTV used GPUs `0,1,2,3`; after a
  mid-run user request, FD-BE was paused after active children finished and the
  remaining `5` jobs were relaunched on GPUs `0,1` only so GPUs `2,3` stayed
  available.
- pTV-LCC+isoTV output root:
  `/database/inrorspline/runs/lungct_directcp4stage_midratio_ptvlcc_v1`.
- global-NCC+FD-BE output root:
  `/database/inrorspline/runs/lungct_directcp4stage_midratio_fdbe_v1`.
- IDIR addendum output roots:
  `/database/inrorspline/runs/lungct_idir_midratio_alpha3_v1` for new alpha
  `3` jobs, and
  `/database/inrorspline/runs/lungct_ptvprep_reg_sweep_idir_directcp32_v1` for
  reused alpha `30` rows.
- Review root:
  `experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/`.
- Notebook:
  `pareto_visual_review_midratio_extension.ipynb`.
- Aggregation: reuse existing all-10 rows from P2e, add the 40 new rows, and
  recompute/validate pTVReg reference rows from uploaded `.mat` fields.
- Primary metric: `original_snap` TRE. Track `original_subvoxel`,
  `reg1mm_snap`, `reg1mm_subvoxel`, and foreground regularity metrics.
- Mask convention: foreground regularity uses the fixed T00 lung mask on the
  pTV 1mm crop; DirectCP training losses do not use lung-mask weighting.
- Acceptance: dry-runs queue exactly `20` jobs per method; smoke jobs pass for
  both methods; full batches complete sequentially; final executed notebook has
  `210` case rows, `21` mean rows, pTVReg validation errors below `1e-8`, and
  all 12 requested inline/saved Pareto figures.
- Completion note: pTV-LCC+isoTV completed `20/20`; global-NCC+FD-BE completed
  `20/20`; strict aggregation produced `210` case rows and `21` mean rows. The
  executed notebook and 12 PNG Pareto figures are under
  `experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/pareto_visual_review_midratio_extension/`.
- IDIR addendum acceptance: alpha `3` dry-run queues 10 jobs on GPUs
  `0,1,2,3`; smoke passes; full alpha `3` batch completes `10/10`; reused
  alpha `30` validates `10/10`; re-executed notebook has `230` case rows,
  `23` mean rows, and the same 12 inline/saved Pareto figure set updated with
  IDIR ratios `0.3x` and `3x`.
- IDIR addendum completion note: alpha `3` completed `10/10`; reused alpha
  `30` validated `10/10`; strict aggregation produced `230` case rows and `23`
  mean rows; the re-executed notebook contains IDIR ratios
  `0.01,0.1,0.3,1,3,10,100` and 12 inline/saved PNG figures.
- Known limitations: this is a regularization-grid refinement only and does not
  resolve pTVReg protocol differences such as optimizer, linear basis, and
  image pyramid.

### P2k: LungCT IDIR Sampled pTV-LCC + TV 0.11 Pilot

- Status: completed 2026-06-26; implementation, smoke, full launch, and
  aggregation passed.
- Purpose: test whether IDIR remains competitive when its sampled-coordinate
  mechanism uses a pTV-style local-CC data term and an IDIR-specific sampled TV
  analogue instead of sampled global NCC plus bending energy.
- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Method: `idir_sampled_ptv_lcc_tv`.
- Objective: sampled pTV-LCC on pTV 1mm border-crop samples plus
  `0.11 *` sampled analytic physical-unit TV of the SIREN displacement
  Jacobian. Fixed T00 lung mask is foreground-metric-only, not a training
  sampling mask.
- Fixed parameters: `ptv_isotv_weight=0.11`, `lr=1e-4`, `epochs=2500`,
  seed `1`, SIREN omega `32`, sampled LCC centers `512`, sampled patch offsets
  `64`, `sigma_mm=2.1`, `csqrt=5e-3`, border mask `5` voxels.
- Output root:
  `/database/inrorspline/runs/lungct_idir_sampled_ptvlcc_tv_w0p11_all10_v1`.
- Review root:
  `experiments/2026-06-26_lungct_idir_sampled_ptvlcc_tv_w0p11_all10/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2,3`.
- Primary metric: `original_snap` TRE. Track `original_subvoxel`,
  `reg1mm_snap`, `reg1mm_subvoxel`, fixed-T00-lung foreground regularity,
  elapsed time, and peak GPU memory.
- Acceptance: compile passes; loader audit for cases `01-10` passes; case01
  short smoke at weight `0.11` has finite losses and writes summary metrics;
  dry-run queues exactly `10` jobs; full launch completes `10/10` summaries;
  aggregation writes case/mean tables with no completion errors.
- Result: `10/10` completed summaries and `errors=[]`. Mean original-snap TRE
  `1.400 +/- 0.857` mm; original-subvoxel TRE `1.476 +/- 0.821` mm;
  reg1mm snap/sub TRE `1.579 +/- 0.829` / `1.483 +/- 0.836` mm. Foreground
  stdLogJ `0.171860`, foreground `J <= 0` `0.0477%`, foreground BE
  `0.001448`, foreground diffusion `0.010796`. Mean elapsed `31.7` sec/case
  and peak memory `0.58` GB. Case08 remained the outlier with original-snap TRE
  `3.895` mm.
- Known limitations: IDIR has no control lattice, so this TV term is a sampled
  analytic SIREN-Jacobian analogue of pTV isoTV, not the same control-grid
  regularizer used by pTVReg or DirectCP.

### P4: pTV-Style DirectCP Ablation Ladder

- Status: planned.
- Purpose: isolate pTVReg gap by adding local CC, border mask, isoTV, pyramid, optimizer, and basis factors one at a time.

### P5: Overnight First-Three-Case Update-Alignment Diagnostic

- Status: completed 2026-06-29; full aggregation passed and follow-up gate did
  not pass.
- Date planned: 2026-06-29.
- Purpose: adapt the compact update-alignment and SINR/direct-CP mechanism
  prompts into a self-supervised first-three-case run. The primary diagnostic
  tests whether sampled image-registration updates move toward strong teacher
  displacement fields, separating representational fit evidence from
  image-optimization trajectory evidence.
- Source prompts:
  `docs/codex_prompts/260629_2_smaller_plan.md` and
  `docs/codex_prompts/260629_1_sinr_cp_kernel_experiment_plan.md`.
- Adapted plans:
  `docs/codex_prompts/260629_3_from_260629_2_overnight_update_alignment_run_plan.md`
  and
  `docs/codex_prompts/260629_4_from_260629_1_optional_sinr_cp_kernel_followup_plan.md`.
- Dataset/cases:
  OASIS BrainMR seed-3 shuffled rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`; DIR-Lab 4DCT case `01`; DIR-Lab 4DCT case `08`.
- Teacher fields: saved BS4 SITReg dense displacement for OASIS rank `9`;
  saved pTVReg displacement fields for LungCT case `01` and case `08`,
  reconstructed on the canonical pTV 1mm crop with saved TRE validation.
- Methods: OASIS `idir_original`, `sinr_cps4_be_bspline_analytic`, and
  `direct_cp_cps4_be_bspline_analytic`; LungCT `idir_be_default_lungmask`,
  `sinr_cps4_ptv_lcc_isotv`, `direct_cp_cps4_ptv_lcc_isotv`, and
  `direct_cp_32_16_8_4_ptv_lcc_isotv`.
- Fixed operating points: OASIS IDIR alpha `1` lr `1e-4`, SINR cps4 alpha
  `1000` lr `1e-4`, DirectCP cps4 alpha `1000` lr `1e-3`; LungCT IDIR
  alpha_bending `10` lr `1e-4`, pTV-LCC+isoTV methods weight `0.11`, SINR lr
  `1e-4`, DirectCP lr `1e-3`, multistage schedule `32,16,8,4` with
  `200,400,800,1000` stage epochs.
- Checkpoint logging: opt-in deterministic sampled displacement checkpoints,
  sample seed `1729`, up to `100000` fixed-mask points per case, checkpoints
  `0,1,10,25,50,100,250,500,1000,1500,2000,2500`. Large checkpoint samples
  stay under the run root, not Git.
- Output root:
  `/database/inrorspline/runs/update_alignment_overnight_20260629_v1`.
- Review root:
  `experiments/2026-06-29_update_alignment_overnight/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2,3`.
- Preflight smoke: completed on 2026-06-29 with `11/11` jobs successful,
  `22` checkpoint-alignment rows, `11` case rows, and `3`
  SINR-vs-DirectCP similarity rows. The smoke promising gate did not pass,
  as expected for the 2-epoch sanity run; it verified launcher,
  checkpoint-sample logging, teacher loading, aggregation, plots, and the
  optional follow-up smoke path.
- Full result: completed `11/11` jobs with `0` failures. Aggregation wrote
  `119` checkpoint-alignment rows, `11` case rows, and `3`
  SINR-vs-DirectCP similarity rows. Follow-up gate did not pass:
  similarity values were LungCT case01 `0.0009`, LungCT case08 `0.2794`, and
  OASIS rank9 `0.3876`, so `promising_similarity_count=0`.
- Primary metrics: per-interval update alignment to teacher residual, early
  mean/median alignment, cumulative early alignment, final teacher distance,
  SINR-vs-DirectCP cumulative early update similarity, final OASIS Dice/HD95
  and LungCT reg1mm crop TRE diagnostics.
- Follow-up gate: if all `11/11` small-plan jobs complete, metrics are finite,
  and SINR-vs-DirectCP cumulative early update similarity is at least `0.5` in
  at least two of the three cases, auto-start
  `scripts/experiments/update_alignment/run_sinr_cp_kernel_followup.py`.
  The first follow-up implementation is an overnight-safe sampled dense-update
  proxy; it does not yet form the full autograd CP kernel `J_theta J_theta^T`.
- Known limitations: LungCT final metrics emitted by the current runner are
  cropped 1mm registration-grid TRE diagnostics, while the teacher pTVReg field
  is validated against original/native snap TRE. Do not treat these as a new
  headline pTVReg-comparable LungCT table without the full original-snap metric
  reconstruction. First-three-case evidence is diagnostic, not population-final.
- Next decision: after full small-plan aggregation, decide whether to trust the
  auto-follow-up gate, expand to more cases, or refine checkpoint/teacher-unit
  handling before running the heavier kernel diagnostic.

### P6: Exploratory SINR/DirectCP Mechanism Follow-Up

- Status: active exploratory self-supervised run; smoke completed and launch
  scope adapted.
- Date planned: 2026-06-29.
- Purpose: continue the optional follow-up as exploration despite the failed
  promising gate, with explicit Scientist/Reviewer role rounds and durable
  communication logging. The target artifact is a useful paper table, figure,
  or draft section explaining whether the observed behavior is better explained
  by SINR/DirectCP kernel similarity, representation capacity, or
  optimization-path/capture-range effects.
- Primary review root:
  `experiments/2026-06-29_sinr_cp_kernel_followup/`.
- Primary run root:
  `/database/inrorspline/runs/sinr_cp_mechanism_explore_20260629_v1`.
- Communication log:
  `experiments/2026-06-29_sinr_cp_kernel_followup/scientist_reviewer_log.md`.
- Exploration controller:
  `scripts/experiments/update_alignment/explore_mechanism_followup.py`.
- Sidecar monitor:
  `scripts/experiments/update_alignment/monitor_mechanism_explore.py`, writing
  `experiments/2026-06-29_sinr_cp_kernel_followup/exploration/live_monitor_status.json`
  and appending role-log updates without launching or stopping jobs.
- Initial evidence sources:
  completed P5 update-alignment tables, sampled SINR-vs-DirectCP proxy metrics,
  and prior LungCT case01/case08 pTVReg field-fit versus image-registration LR
  sweeps.
- New schedule-ablation direction: DirectCP `32,16,8,4` on LungCT cases `01`
  and `08`, removing or weakening the image-pyramid downsampling while
  preserving control-grid staging. Primary variants include pTV LCC + isoTV and
  global NCC + dense FD-BE objectives on canonical pTV-preprocessed 1mm crops.
- Smoke evidence: no-pyramid `1,1,1,1` stage-epoch jobs completed on cases
  `01` and `08` with no failures, but took roughly 4-6 minutes/job. Case08
  remained poor after this tiny smoke, so the long launch is a time-budgeted
  diagnostic set with `25,25,25,25` stage epochs, targeted extra case01/case08
  variants, and no all10 expansion yet.
- First-phase result: targeted schedule ablation completed `10/10` with `0`
  failures. Case01 reached `1.076` mm best reg1mm snap TRE without the image
  pyramid, while case08 remained poor under no-pyramid (`6.478` mm best after
  LR probe) and partial `4,2,1,1` pyramid (`11.937` mm best). This strengthens
  the optimization-path/capture-range interpretation but does not yet isolate
  whether the missing factor is the full standard `8,4,2,1` pyramid or a
  longer schedule.
- Second-phase standard-pyramid probe:
  `scripts/experiments/update_alignment/run_standard_pyramid_probe.py`,
  case08-only DirectCP `32,16,8,4` with standard `8,4,2,1` pyramid, pTV/FD-BE,
  LR, and modest schedule-length probes.
- Second-phase result: standard-pyramid probe completed `4/4` with `0`
  failures. Best case08 standard-pyramid short row was pTV LCC + isoTV,
  `8,4,2,1`, `25,25,25,25`, LR `0.01`, with `2.270` mm reg1mm snap TRE.
  LR `0.001` standard-pyramid rows remained poor (`11.496` mm pTV and
  `11.361` mm FD-BE), and the longer LR `0.001` pTV row reached `8.896` mm.
  Interpretation: full standard image-pyramid capture range plus optimizer
  scale substantially rescues case08, while no/partial pyramid does not.
  Remaining gap to the prior full schedule should be described as runtime/LR
  and metric-convention limited, not as a population-final conclusion.
- Active refinement: launched
  `scripts/experiments/update_alignment/run_standard_pyramid_probe.py` with
  `--probe-set refinement --phase-name capture_refinement_probe` on GPUs
  `0,1,2,3`. This tests case08 standard `8,4,2,1` LR `0.01` at
  `100,100,100,100` and `200,200,200,200`, plus partial `4,2,1,1` and
  no-pyramid LR `0.01` controls at `100,100,100,100`.
- Refinement result: completed `4/4` with `0` failures. Case08 reg1mm snap TRE
  converged near `1.21` mm for all LR `0.01` e100/e200 controls: standard
  `8,4,2,1` e100 `1.213`, standard e200 `1.217`, partial `4,2,1,1` e100
  `1.219`, and no-pyramid e100 `1.218`. Interpretation: the image pyramid
  matters under short optimization budgets, but longer LR `0.01` optimization
  removes the pyramid ablation gap in this diagnostic case.
- Requested runtime: user requested a long exploratory run of at least four
  hours, with the controller refreshing aggregation and role logs during the
  run.
- Known limitations: this is not a population-final result and not a
  pTVReg-method-only comparison. LungCT rows must be tagged as pTV-preprocessed
  diagnostic-domain experiments unless original/native TRE reconstruction is
  explicitly supplied by the source artifact.

### P7: Direct Dense-DVF vs IDIR Preliminary Check

- Status: completed.
- Date planned: 2026-06-30.
- Purpose: test whether replacing IDIR's continuous SIREN displacement model
  with a directly optimized dense displacement field is worth a fuller
  ablation. This isolates optimization over dense grid values from the IDIR
  coordinate-network parameterization while keeping the IDIR-style sampled NCC
  loss and bending anchor.
- Method handles:
  - OASIS `direct_dvf_sampled_ncc_fd_be`.
  - LungCT `direct_dvf_lungmask_sampled_ncc_fd_be`.
- Cases:
  - OASIS rank `9`, seed-3 shuffled first10 pair `(fixed 0173, moving 0136)`,
    original pair index `88`.
  - LungCT cases `01` and `08`, canonical pTV-preprocessed diagnostic crop
    domain.
- Fixed settings: seed `1`, LR `1e-4`, `2500` epochs, sampled image-loss
  locations `10000`, no tuning in this precheck.
- Regularization: direct dense DVF has no native continuous analytic bending
  formula, so training BE is tagged as
  `dense_forward_difference_be`. OASIS uses fixed image-positive valid-stencil
  BE; LungCT uses fixed T00 lung-mask valid-stencil BE on the cropped 1mm grid.
- IDIR anchors for interpretation:
  - OASIS IDIR alpha `1`, LR `1e-4`.
  - LungCT IDIR `alpha_bending=10`, LR `1e-4`.
- Launch GPU assignment: OASIS rank9 on GPU `0`; LungCT case01 on GPU `2`;
  LungCT case08 on GPU `3`. Do not use GPU `1` for this launch.
- Run roots:
  - `/database/inrorspline/runs/oasis_rank009_direct_dvf_idir_loss_precheck_v1`
  - `/database/inrorspline/runs/lungct_case01_case08_direct_dvf_idir_loss_precheck_v1`
- Review root:
  `experiments/2026-06-30_direct_dvf_idir_precheck/`.
- Required preflight: compile/import checks, OASIS rank9 validation, LungCT
  loader audit for cases `01,08`, and two-epoch smoke after implementation.
- Planned aggregation: report accuracy/TRE, dense finite-difference BE,
  diffusion, log-Jacobian, folding, NDV, runtime, and peak memory against the
  existing IDIR anchors. LungCT metrics remain pTV-preprocessed reg1mm crop
  diagnostics unless an original/native TRE reconstruction is explicitly added.
- Result: all `3/3` full jobs completed on requested GPUs `0,2,3` with no
  process failures. Aggregation wrote
  `experiments/2026-06-30_direct_dvf_idir_precheck/review_report.md`,
  `direct_dvf_idir_precheck_rows.csv`, and
  `direct_dvf_idir_precheck_rows.json`.
- Headline comparison: direct dense DVF under the matched IDIR-style loss was
  worse than the IDIR anchor on all three preliminary cases. OASIS rank9 Dice
  was `0.7673` versus IDIR `0.8110`, with much higher foreground folding
  (`9.889%` versus `1.560%`). LungCT case01 snap TRE was `3.764` mm versus
  IDIR `1.063` mm, and case08 snap TRE was `15.076` mm versus IDIR `1.331` mm.
- Preliminary interpretation: the direct dense-grid parameterization is
  feasible to run, but this fixed LR/alpha precheck does not motivate a broader
  same-settings expansion before trying stronger smoothness, different LR, or a
  multiscale/capture-range schedule.

### P8: DCTField IDIR-Loss K-Sweep Preliminary Check

- Status: completed.
- Date planned: 2026-06-30.
- Purpose: test a non-neural, non-spline compressed displacement basis as a
  paper-facing baseline for the deformation-search-space story. DCTField
  optimizes low-frequency cosine coefficients and decodes them to a dense
  displacement field.
- Method handles:
  - OASIS `dctfield_sampled_ncc_fd_be`.
  - LungCT `dctfield_lungmask_sampled_ncc_fd_be`.
- Cases:
  - OASIS rank `9`, seed-3 shuffled first10 pair `(fixed 0173, moving 0136)`,
    original pair index `88`.
  - LungCT cases `01` and `08`, canonical pTV-preprocessed diagnostic crop
    domain.
- DCT sizes: `16^3`, `32^3`, `64^3`, `128^3`, with coefficient counts
  `12288`, `98304`, `786432`, and `6291456`.
- Fixed settings: seed `1`, LR `1e-4`, `2500` epochs, sampled image-loss
  locations `10000`, no tuning in this precheck.
- Regularization: training BE is finite-difference BE over the decoded dense
  DCT field, tagged as `dct_decoded_dense_forward_difference_be`. Spectral DCT
  analytic BE is intentionally not used in this first round.
- IDIR/direct-field anchors for interpretation:
  - OASIS IDIR alpha `1`, LR `1e-4`; direct dense-DVF P7.
  - LungCT IDIR `alpha_bending=10`, LR `1e-4`; direct dense-DVF P7.
- Launch queue: GPU `0` runs OASIS K `16,32,64,128` sequentially; GPU `2`
  runs LungCT case01 K `16,32,64,128` sequentially; GPU `3` runs LungCT case08
  K `16,32,64,128` sequentially. Do not use GPU `1`.
- Run roots:
  - `/database/inrorspline/runs/oasis_rank009_dctfield_idir_k_sweep_precheck_v1`
  - `/database/inrorspline/runs/lungct_case01_case08_dctfield_idir_k_sweep_precheck_v1`
- Review root:
  `experiments/2026-06-30_dctfield_idir_k_sweep_precheck/`.
- Required preflight: compile/import checks, DCT decoder sanity checks, OASIS
  rank9 validation, LungCT loader audit for cases `01,08`, and two-epoch K128
  smokes for OASIS rank9 and LungCT case08. If K128 smoke fails with OOM or
  NaN, stop and report rather than silently dropping K128.
- Planned aggregation: compare IDIR anchor, direct dense-DVF P7, and DCTField
  K `16,32,64,128` on accuracy/TRE, dense finite-difference BE, diffusion,
  log-Jacobian, folding, NDV, final loss terms, runtime, and peak memory.
- Result: all `12/12` DCTField full jobs completed on requested GPU queues
  `0,2,3` with no process failures. Aggregation wrote
  `experiments/2026-06-30_dctfield_idir_k_sweep_precheck/review_report.md`,
  `dctfield_idir_k_sweep_rows.csv`, and `dctfield_idir_k_sweep_rows.json`.
- Headline comparison: OASIS DCTField K128 reached the best DCT Dice
  (`0.7781`) but folded heavily (`16.89%` foreground), while K32 was the
  smoother tradeoff (`0.7242` Dice, `0.265%` folding). LungCT case01 was best
  at K32 (`3.652` mm snap TRE, `0.739%` folding), slightly better and smoother
  than direct dense-DVF but far behind IDIR (`1.063` mm). LungCT case08 stayed
  near identity/direct-dense performance across K values, with best DCT K64
  `14.986` mm versus IDIR `1.331` mm.
- Preliminary interpretation: DCTField is useful as a compact spectral
  search-space baseline and supports the representation-prior framing, but this
  fixed LR/alpha single-stage DCT sweep does not explain IDIR's LungCT success
  or rescue the hard case08 capture-range failure.

### P9: Fit-Modes Panel A Global-NCC Curve Refresh

- Status: completed.
- Date planned: 2026-07-01.
- Purpose: rebuild the fit-modes paper figure Panel A on a consistent global
  NCC loss scale for OASIS rank9 and LungCT cases `01`/`08`, so all three
  cases can share the same y-axis while the fitting Panel B uses relative
  target-field error.
- Output root:
  `/database/inrorspline/runs/fit_modes_panelA_global_ncc_v1`.
- Smoke output root:
  `/database/inrorspline/runs/fit_modes_panelA_global_ncc_smoke_v1`.
- Review/artifact root:
  `experiments/paper_artifacts/fitting_figure/v001/`.
- Docker target: `py310sitreg_v1_test1`.
- Common diagnostic: write `global_ncc_diagnostic.csv` with full-image
  OASIS/global-crop LungCT NCC loss at epoch `1` and every `10` epochs. These
  diagnostic rows are tagged `affects_training=false` unless the diagnostic is
  also the optimized data term.
- OASIS rank9 existing reused Panel A curves:
  - IDIR `idir_original_dense_siren_be_autograd`, alpha `1`, lr `1e-4`,
    2500 epochs. Training remains IDIR sampled NCC + BE; diagnostic is
    full-image global NCC every 10 epochs from the completed matched-eval run.
  - SINR cps4 `sinr_cps4_be_bspline_analytic`, alpha `1000`, lr `1e-4`,
    2500 epochs. Optimized loss is full-image global NCC + analytic
    B-spline BE.
  - DirectCP cps4 `direct_cp_cps4_be_bspline_analytic`, alpha `1000`,
    lr `1e-3`, 2500 epochs. Optimized loss is full-image global NCC +
    analytic B-spline BE.
- OASIS rank9 new run:
  - DirectCP 8-4-2
    `direct_cp_8_4_2_be_bspline_analytic_multistage`, alpha `1000`,
    lr `1e-3`, total 2500 epochs.
  - Stage controls: cps `8,4,2`; stage downsample `4,2,1`; stage epochs
    `400,800,1300`.
  - Optimized loss is stage-domain global NCC + analytic B-spline BE; plotted
    diagnostic is full-image global NCC every 10 epochs.
- LungCT case01/case08 new runs:
  - IDIR `idir_be_default_lungmask`, `alpha_bending=3`, lr `1e-4`,
    2500 epochs. Training remains native IDIR sampled NCC + IDIR BE in the
    fixed T00 lung-mask sampling convention; plotted diagnostic is full-crop
    global NCC every 10 epochs.
  - SINR cps4 `sinr_cps4_global_ncc_fd_be`, alpha `10000`, lr `1e-4`,
    2500 epochs. Optimized loss is full-crop global NCC + dense
    forward-difference BE.
  - DirectCP cps4 `direct_cp_cps4_global_ncc_fd_be`, alpha `10000`,
    lr `1e-3`, 2500 epochs. Optimized loss is full-crop global NCC + dense
    forward-difference BE.
  - DirectCP 32-16-8-4 `direct_cp_32_16_8_4_global_ncc_fd_be`,
    alpha `10000`, lr `1e-3`. Stage controls `32,16,8,4`; stage downsample
    `8,4,2,1`; stage epochs `200,400,800,1000` for a 2400-epoch run matching
    the existing Pareto sweep protocol. Optimized loss is full-crop global NCC
    + dense forward-difference BE; plotted diagnostic is full-crop global NCC
    every 10 epochs.
- Standardized artifact outputs planned:
  - `fit_modes_standardized_global_ncc_panelA_curves.png`
  - `fit_modes_standardized_global_ncc_panelA_selected_runs.csv`
  - `fit_modes_standardized_global_ncc_panelA_curve_summary.csv`
  - `fit_modes_standardized_global_ncc_panelA_manifest.json`
- Figure rule: one row by three cases, shared global NCC y-axis where possible,
  and y tick labels plus y-axis label visible on every subplot.
- Required preflight: Docker `py_compile`, two-epoch or three-epoch smokes
  covering OASIS multistage, LungCT IDIR diagnostic, LungCT single-stage
  global-NCC+FD-BE, and LungCT multistage global-NCC+FD-BE.
- Result: preflight compile and all five smoke paths passed. Full launch
  completed `9/9` new jobs with all queues returning status `0`.
- Diagnostic validation: every `global_ncc_diagnostic.csv` has finite losses
  and expected epoch coverage. 2500-epoch runs have `251` rows; LungCT
  2400-epoch multistage runs have `241` rows.
- Final global NCC losses:
  - OASIS DirectCP 8-4-2: `-0.994035`.
  - LungCT case01 IDIR/SINR/DirectCP cps4/DirectCP multistage:
    `-0.946411`, `-0.997760`, `-0.998162`, `-0.998160`.
  - LungCT case08 IDIR/SINR/DirectCP cps4/DirectCP multistage:
    `-0.910503`, `-0.993551`, `-0.992049`, `-0.997240`.
- Standardized artifact generated:
  `experiments/paper_artifacts/fitting_figure/v001/fit_modes_standardized_global_ncc_panelA_curves.png`.

### P10: OASIS DirectCP 8-4-2 Field-Fit rRMSE Panel B Add-On

- Status: completed.
- Date planned: 2026-07-01.
- Date started: 2026-07-01.
- Date completed: 2026-07-01.
- Purpose: add the OASIS DirectCP 8-4-2 multistage trajectory to the
  standardized Panel B target-field rRMSE fitting figure for both OASIS rank9
  SITReg and Greedy targets.
- Dataset: OASIS BrainMR seed-3 shuffled rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`.
- Target fields:
  - SITReg target: saved `bs4_sitreg_v0` best-Dice dense displacement,
    normalized-grid units.
  - Greedy target: canonical Greedy dense displacement, normalized-grid units.
- Method: `direct_cp_8_4_2_be_bspline_analytic_multistage`, recorded as
  `direct_cp_8_4_2_field_mse` for field-fit outputs.
- LR/alpha/schedule: alpha `1000` recorded for operating-point consistency
  but not used by the MSE-only objective; lr `1e-3`; total 2500 epochs; cps
  `8,4,2`; stage downsample `4,2,1`; stage epochs `400,800,1300`.
- Optimized loss: stage-domain target-field MSE over the fixed raw
  image-positive OASIS brain mask, no image loss and no regularization.
- Diagnostic loss: finest-domain masked target-field MSE in normalized-grid
  units at epoch `1`, every `10` global epochs, and final epoch. The Panel B
  rRMSE curve uses `100 * sqrt(mse / (target_power + 1e-12))`.
- Output root:
  `/database/inrorspline/runs/fit_modes_oasis_directcp842_field_fit_v1`.
- Smoke output root:
  `/database/inrorspline/runs/fit_modes_oasis_directcp842_field_fit_smoke_v1`.
- Review/artifact root:
  `experiments/paper_artifacts/fitting_figure/v001/`.
- Docker target: `py310sitreg_v1_test1`.
- Required preflight: Docker `py_compile` for modified OASIS field-fit
  runners and figure generator; three-epoch smoke runs for both SITReg and
  Greedy targets with finest-domain MSE diagnostics every epoch.
- Full commands: run the SITReg and Greedy target fits with
  `--method direct_cp_8_4_2_be_bspline_analytic_multistage --alpha 1000
  --lr 1e-3 --epochs 2500 --stage-full-cps 8,4,2 --stage-downsample 4,2,1
  --stage-epochs 400,800,1300 --finest-mse-diagnostic
  --finest-mse-every 10`.
- Known limitations: this is a single OASIS illustrative pair and field-fit-only
  diagnostic; the field-fit objective uses target-field MSE rather than the
  image-registration global NCC + BE objective used in Panel A.
- Result: Docker `py_compile` passed; both three-epoch smoke jobs passed; both
  full 2500-epoch jobs completed with `DONE` markers.
- Diagnostic validation: both full `stage_history.csv` files contain 2500 rows
  and 251 finite finest-domain MSE diagnostics covering epoch `1` plus every
  `10` epochs through `2500`, with no missing/extra diagnostic epochs.
- Final masked normalized-grid target-field MSE:
  - SITReg target: `1.148402e-06`.
  - Greedy target: `1.004255e-07`.
- Standardized Panel B rRMSE results:
  - SITReg target: start `97.33%`, epoch 10 `76.30%`, final `4.50%`.
  - Greedy target: start `97.58%`, epoch 10 `78.18%`, final `1.23%`.
- Standardized artifact regenerated:
  `experiments/paper_artifacts/fitting_figure/v001/fit_modes_standardized_rrmse_field_fit_curves.png`.

### P11: OASIS DirectCP 8-4-2 Canonical lr=1e-3 Pareto/Table Standardization

- Status: completed 2026-07-01 after smoke, full launch, aggregation,
  candidate artifact regeneration, and comparison audit.
- Purpose: replace the final-paper OASIS DirectCP 8-4-2 `lr=1e-2`
  Pareto/table temporary exception with a canonical `lr=1e-3` candidate while
  leaving current `v001` artifacts intact for historical review.
- Dataset: OASIS BrainMR seed-3 shuffled first20 ranks `0-19`.
- Method: `direct_cp_8_4_2_be_bspline_analytic_multistage`.
- Objective: full-image global NCC plus `alpha *` analytic cubic B-spline BE;
  foreground regularity metrics use the fixed raw image-positive mask.
- Alpha grid: `0,10,100,1000,10000,100000`.
- LR/schedule: `lr=1e-3`; cps `8,4,2`; stage downsample `4,2,1`; stage
  epochs `400,800,1300`; total epochs `2500`; seed `1`; eval interval `250`.
- Launch environment: Docker container `py310sitreg_v1_test1`; use all visible
  GPUs with one job per GPU.
- Smoke output root:
  `/database/inrorspline/runs/oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_smoke_v1`.
- Full output root:
  `/database/inrorspline/runs/oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_v1`.
- Review root:
  `experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization/`.
- Candidate artifact roots:
  `experiments/paper_artifacts/best_result_table/v002_candidate/`,
  `experiments/paper_artifacts/pareto_figure/v002_candidate/`, and
  `experiments/paper_artifacts/comparison/v001_vs_v002_candidate/`.
- Monitoring: poll every 5 minutes; report completed/failed/running/remaining
  jobs, elapsed time, throughput, ETA, newest log update, and failure log
  paths; save snapshots in the review root.
- Required preflight: Docker `py_compile`; one rank/one-alpha smoke with stage
  epochs `1,1,1`; verify finite metrics and expected multistage files.
- Acceptance: full run produces `120/120` summaries and `DONE` markers;
  aggregation has no missing case-alpha combinations; `v002_candidate`
  validation has zero `temporary_exception`, `mismatch`, or blocking LR issues;
  current `v001` artifacts remain unchanged.
- Actual completion: full run completed `120/120` jobs with `0` failures in
  about `2h 7m`; aggregation reported `combined_rows=120`,
  `population_scopes={"first10": 6, "first20": 6}`, and `errors=[]`.
- Candidate artifact audit: best-result table validation has `11`
  `matches_standard` rows plus `4` references; Pareto validation has `11`
  `matches_standard` rows; no `temporary_exception`, `mismatch`, or blocking
  LR issue remains in `v002_candidate`.

### P12: OASIS Seed-3 First100 Paper-Figure Extension

- Status: production launch started 2026-07-04 after Docker compile, dry-run,
  and smoke verification passed; not yet marked complete.
- Purpose: extend the OASIS seed-3 first20 publication-figure results to
  first100 by reusing ranks `0-19` and running ranks `20-99`.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-99`; foreground regularity
  metrics use `fixed raw image > 0`; segmentation is used only for Dice,
  HD95, and ASD.
- Common sweep settings: epochs `2500`, eval interval `250`, seed `1`.
- Reference evaluation first: `bs1_vxm_v0_2`, `bs2_tm_v0`, `bs3_vfa_v0`,
  `bs4_sitreg_v0`, and canonical `greedy` saved fields.
- Sweep matrix:
  - `direct_cp_8_4_2_be_bspline_analytic_multistage`, alpha
    `0,10,100,1000,10000,100000`, lr `1e-3`, cps `8,4,2`, downsample
    `4,2,1`, stage epochs `400,800,1300`.
  - `direct_cp_cps2_be_bspline_analytic`, alpha
    `0,10,100,1000,10000,100000`, lr `1e-3`.
  - `sinr_cps2_be_bspline_analytic`, alpha
    `0,10,100,1000,10000,100000`, lr `1e-4`.
  - `idir_original_dense_siren_be_autograd`, alpha
    `0,0.01,0.1,1,10,100`, lr `1e-4`.
  - `direct_cp_cps4_be_bspline_analytic`, alpha
    `0,10,100,1000,10000,100000`, lr `1e-3`.
  - `sinr_cps4_be_bspline_analytic`, alpha
    `0,10,100,1000,10000,100000`, lr `1e-4`.
- New job count: `3280` jobs = `400` saved baseline/reference evaluations,
  `480` MR-D-BSCP 8-4-2 jobs, and `2400` remaining single-stage sweep jobs.
- Launch environment: Docker container `py310sitreg_v1_test1`; all visible
  GPUs, one job per GPU by default.
- Output roots:
  - `/database/inrorspline/runs/oasis_seed3_ranks020_099_saved_baselines_v1`
  - `/database/inrorspline/runs/oasis_seed3_ranks020_099_alpha_sweep_be_2500e_5models_v1`
  - `/database/inrorspline/runs/oasis_seed3_ranks020_099_directcp842_lr1e3_alpha_sweep_v1`
- Review root:
  `experiments/2026-07-04_oasis_seed3_first100_extension/`.
- Main progress log:
  `experiments/2026-07-04_oasis_seed3_first100_extension/progress_log.md`.
- Tracker figure:
  `experiments/2026-07-04_oasis_seed3_first100_extension/tracker/oasis_first100_progress_tracker.png`.
- Hourly watchdog:
  `experiments/2026-07-04_oasis_seed3_first100_extension/hourly_watchdog.md`.
- Final first100 paper plots:
  `experiments/2026-07-04_oasis_seed3_first100_extension/first100_artifacts/oasis100/`.
- Metric convention update: if fixed, moving, or warped voxels for an OASIS
  label are absent, Dice/HD95/ASD are recorded as `NaN`; case/population means
  use finite-only `nanmean`.
- Implementation entry point:
  `scripts/experiments/oasis_alpha_sweep/launch_oasis_first100_extension.py`.
- Launch command:
  `python scripts/experiments/oasis_alpha_sweep/launch_oasis_first100_extension.py --stage all --gpus auto --max-concurrent 0`.
- Launch log:
  `experiments/2026-07-04_oasis_seed3_first100_extension/launch_all_20260703_215334.log`.
- Acceptance: planned method/rank/alpha/reference combinations report `DONE`,
  failures are `0`, first100 aggregation has no missing or duplicate
  combinations, final figures use only complete first100 rows, and the tracker
  and progress log agree on counts.

### P13: OASIS10/DIR-LAB10 Population Fit-Modes Figure

- Status: completed and accepted 2026-08-02; schedule-matched diagnostic refresh
  accepted 2026-08-13. Original aggregate: 160/160 terminal-
  success records, including 136 new completions and 24 exact-compatible
  reuses; 0 failures; all 16 dataset/panel/method groups have N=10.
- Purpose: replace the illustrative one-OASIS/two-DIR-LAB representation-
  capacity evidence with paired population curves for a locked OASIS N=10
  subset and all DIR-LAB N=10 cases.
- Dataset: OASIS seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`; DIR-LAB 4DCT cases `01-10`, moving
  `T50` to fixed `T00` on the canonical pTVReg-equivalent isotropic 1 mm crop.
- Matrix: `20 cases × 2 objectives × 4 methods = 160` records. The current
  source-backed figure's 24 exact-compatible selected-case records are eligible
  for provenance-preserving reuse; 136 records are expected to run anew.
- OASIS Panel A: IDIR `alpha=1, lr=1e-4`; INR-BSCP cps2
  `alpha=1000, lr=1e-4`; D-BSCP cps2 `alpha=1000, lr=1e-3`; MR-D-BSCP
  `8-4-2`, `lr=1e-3`, stage epochs `400,800,1300`; total 2500 epochs.
- Historical DIR-LAB Panel A index: IDIR `alpha_BE=3, lr=1e-4`; INR-BSCP cps4
  `alpha=10000, lr=1e-4`; D-BSCP cps4 `alpha=10000, lr=1e-3`;
  MR-D-BSCP `32-16-8-4`, `lr=1e-3`, stage epochs `200,400,800,1000`.
- Versioned schedule-matched diagnostic index: the same 150 nonreplacement
  records plus ten accepted DIR-LAB Panel A MR-D-BSCP records at
  `400,400,800,900` (2500 epochs). The original index is not overwritten.
- Panel B: MSE-only normalized-grid target-field fitting. OASIS uses SITReg
  targets and the fixed `image>0` mask; DIR-LAB uses reconstructed pTVReg
  targets and the fixed-T00 lung mask. Multistage schedules are OASIS
  `400,800,1300` and DIR-LAB `400,400,800,900`.
- Seed: `1`; single-stage epochs: `2500`; Panel A global-NCC diagnostic interval:
  `10`; OASIS multistage finest-MSE interval: `10`; DIR-LAB multistage
  finest-MSE interval: `1`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `${DOCKER_IMAGE}`; GPUs `0,1,2,3`, one active optimizer
  per GPU with dynamic work stealing.
- Output root: `${RUN_ROOT}` (completed-run variant
  `fit_modes_population_n10_v1`).
- Smoke output root: `${SMOKE_RUN_ROOT}` (completed-run variant
  `fit_modes_population_n10_smoke_v1`).
- Review root:
  `experiments/2026-08-02_fit_modes_population_n10/`.
- Split config:
  `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Standards preflight: OASIS raw image-positive foreground and shuffled-subset
  rules reviewed; LungCT protocol warnings and fixed-T00 mask rule reviewed.
- Primary endpoints: Panel A `1-NCC`; Panel B target-normalized rRMSE (%).
  Every index record also retains target power where applicable, diagnostic
  NCC, runtime, peak memory, and source provenance.
- Monitoring: scheduler poll no longer than 60 seconds; persistent
  `monitor_status.md/json`; provisional ETA from compatible historical runs,
  then observed median runtime by dataset/panel/method.
- Acceptance: `10/10` SITReg and `10/10` pTVReg targets validate; dry-run
  resolves 160 unique records; isolated 16-job smoke passes; all 160 full
  records are complete or exact-compatible reuse with no unresolved failures;
  all endpoint/mask/unit/schedule/schema checks pass before figure generation.
- Accepted outputs: `acceptance_report.md/json`, `optimization_index.csv/json`,
  and `population_statistics.csv` in the review root. Scheduler wall time was
  2 h 20 min 26 s on four GPUs. After acceptance, 20 source-backed case
  diagnostic figures were generated and arranged in `diagnostic_comparison.md`
  as two five-by-two dataset grids. All 160 plotted endpoints agree with the
  accepted index; axis limits are shared within each dataset and recorded in
  `diagnostic_manifest.json`. A compact final population-statistics figure is
  still a separate paper-design step.
- Schedule-matched diagnostic outputs:
  `experiments/paper_artifacts/fit_modes_population_diagnostics/v001_schedule_matched/`.
  The versioned source index contains exactly 160 records with 10 replacements
  and 150 canonically unchanged rows. All 160 endpoints validate; all ten new
  DIR-LAB curves contain 251 finite diagnostics through epoch 2500. Two 4-by-5
  dataset composites and 20 individual case PNGs passed checksum and visual
  review gates on globally locked panel axes.
- Known limitation: OASIS N=10 resolves the single-case cherry-picking concern
  but does not literally repeat the previous N=100 testing cohort.
- Resolved camera-ready issue: the schedule-matched supplementary diagnostics
  use `400,400,800,900` for DIR-LAB MR-D-BSCP in both panels. The historical
  `200,400,800,1000` rows remain provenance/sensitivity evidence.
- Implementation entry point:
  `scripts/experiments/fit_modes_population/launch_population.py`.
- Diagnostic entry point:
  `scripts/experiments/fit_modes_population/plot_case_diagnostics.py`.
- Schedule-matched diagnostic entry point:
  `scripts/experiments/fit_modes_population/render_schedule_matched_diagnostics.py`.

### P14: DUAL-INR-DIR Author Reproduction and Semantic-Loss Ablation

- Status: complete 2026-08-02; implementation/data, paired smoke, full-length
  cases01/08 pilot, 20-record production, common-regularity, aggregation, and
  visual-review gates all passed.
  The author-format ten-case artifact is published and independently validated;
  the focused preprocessing/runner/orchestration/reporting suite passes `28/28`.
  A0 reproduced the published author mean (`0.9789 mm` observed versus
  `0.98 mm` reported). Schedule-controlled A1 produced `0.9834 mm`, a paired
  descriptive delta of `+0.0045 +/- 0.0139 mm` (sample SD).
- Purpose: reproduce the released DUAL-INR-DIR author configuration on all ten
  DIR-LAB 4DCT cases and isolate the semantic lobe-mask loss without changing
  the author data, crop, sampling, architecture, optimizer, or realized
  casewise schedule.
- Source: clean `external/DUAL-INR-DIR` submodule at commit
  `f7c8e833435ec9c6b7ebec6c9a8098b566709146`.
- Dataset and direction: DIR-LAB 4DCT cases `01-10`; moving T50/end-exhale to
  fixed T00/end-inhale (`T50 -> T00`); one locked seed, `42`, matching the
  upstream runner default.
- Protocol handle: `dual_author_native_lungbbox_totalseg`. This author-code
  protocol uses native spacing, per-image min-max normalization, a two-voxel-
  padded crop around the union of fixed and moving five-lobe TotalSegmentator
  masks, and fixed/T00 lobe-union coordinate sampling. It is neither native
  full-volume IDIR nor pTVReg landmark-crop/iso1mm preprocessing.
- A0 `a0_author_full_loss`: released-code sampled NCC plus Laplacian/curvature
  weight `0.01`, negative-determinant weight `0.01`, and five-lobe semantic
  mask weight `0.1`.
- A1 `a1_author_no_semantic_mask_loss`: the scientific factor changes only
  `loss.mask_weight` from `0.1` to `0.0`. A1 retains mask-derived cropping and
  fixed-mask sampling and must not be labeled no-mask. As a schedule-control
  intervention, A1 replays the realized blur-transition steps from A0 for the
  same case and seed instead of rerunning the adaptive trigger.
- Common author settings: dual SIREN layers `3-256-256-3`, coarse/fine
  frequencies `10/30`, `30,000` sampled points per step, blur stages `4,2,0`,
  AdamW, cosine schedule, initial/end LR `1e-3/1e-5`, and `3000` total steps.
  A0 uses the author adaptive relative-convergence transitions; A1 uses the
  exact casewise A0 boundaries.
- Matrix: `10 cases x 2 variants x 1 seed = 20` production records.
- Pilot: run full-length A0 first on cases `01` and `08` using the available
  GPU pool with at most one optimizer per GPU
  after source/data/config/dry-run and short-smoke gates, lock the realized A0
  boundaries, then run paired A1. For production, finish A0 across all cases
  and freeze its transition manifest before launching A1. Use all visible GPUs,
  at most one optimizer per GPU, only after all four full pilot records and
  visual checks pass.
- Required environment variables: `${REPO_ROOT}`, `${DUAL_AUTHOR_DATA_ROOT}`,
  `${DUAL_AUTHOR_DATA_MANIFEST}`, `${DUAL_AUTHOR_PILOT_ROOT}`, and
  `${DUAL_AUTHOR_RUN_ROOT}`. Treat the prepared data artifact as logically
  read-only and forbid output roots beneath it; the existing default
  container's `/database` mapping itself is read-write. Large artifacts remain
  outside Git.
- Headline metric: native/crop-domain snap-to-voxel TRE in physical mm. Keep
  subvoxel TRE, decoded-dense versus direct-INR agreement, common posthoc
  regularity metrics, runtime, and peak memory as required diagnostics.
- Technical acceptance: exact clean source commit; all ten cases pass phase,
  orientation, shape/spacing, 300-landmark, five-lobe, checksum, and
  TotalSegmentator-provenance checks. Converter-derived identity TRE for
  cases01-10 must match `[3.892406, 4.337842, 6.942993, 9.830147, 7.476853,
  10.891024, 11.026220, 14.994681, 7.918264, 7.301382]` mm within `1e-5 mm`
  per case; the rounded paper Table 1 no-registration row is a release
  discrepancy, not the data gate. Dry-run resolves 20 unique records; pilots
  complete `4/4`; production aggregates `20/20`, N=10 per variant, with 3000
  finite history rows and no unresolved, missing, or duplicate record.
- Accepted prepared-data artifact: version
  `dual_author_native_lungbbox_totalseg_v001`, with 84 verified checksum
  entries. Manifest SHA-256 is
  `40e40c98219ac68e0b323f4a50d9b1687c447c03c42093cf180f2453fe402cb2`,
  preprocessing-contract SHA-256 is
  `12cc0b9cd667d2d4d133d828162759e5cec245687f7e899d773a924fa0ce4c24`,
  and checksum-manifest SHA-256 is
  `b8b8d751def598b8c70b34cdfc293baa3e0c197649d5a3e814ec7046cd3ab0c3`.
  The exact executed converter is embedded read-only in the artifact at
  SHA-256 `1f33439328f5d586d8086b5d01297b3067fe7d63f4fb8ee0cf99193e01179ba3`.
- Isolation acceptance: the scientific config difference is only
  `loss.mask_weight`; allowed execution differences are transition mode/source
  metadata and output identity. Each A1 run must use exactly its paired A0
  stage boundaries. Dense and direct-INR pilot snap TRE differ by at most
  `0.05 mm`; every full run visits blur stages `4,2,0` and activates the fine
  branch at stage index `1`.
- Fidelity review: compare A0 with the published author mean `0.98 mm`; use the
  preregistered `0.82-1.12 mm` consistency interval based on the published
  `0.97 +/- 0.15 mm` multiseed summary. Outside the interval, preserve and
  report the completed experiment but mark fidelity failed and do not tune or
  reseed under this experiment ID.
- Interpretation limits: seed `42` alone does not reproduce the ten-seed
  robustness analysis; A1 isolates only the semantic loss term; neither row is
  a clean method-only comparison with IDIR or pTVReg; analysis is descriptive
  and does not support equivalence or superiority claims.
- Completed result: A0/A1 dense snap TRE is `0.9789 +/- 0.1727` versus
  `0.9834 +/- 0.1669 mm` across cases (mean +/- sample SD). Mean five-lobe
  Dice is `0.96788` versus `0.96463`; the paired A1-minus-A0 Dice delta is
  `-0.00325`, negative in all ten cases. Foreground SDlogJ is
  `0.1401 +/- 0.0361` versus `0.1389 +/- 0.0353`; neither variant has a
  nonpositive foreground Jacobian voxel under the shared posthoc convention.
  All ten A1 stage traces exactly match their paired A0 traces.
- Locked plan and execution specification:
  `experiments/2026-08-02_dual_inr_dirlab_author_repro/`.

### P15: Controlled DUAL Native/pTVReg Full-Resolution and Image-Pyramid Baselines

- Status: complete 2026-08-03; 40/40 optimization records and 40/40 common
  regularity records passed strict technical acceptance with zero failures or
  retries.
- Purpose: retain the completed author reproduction while adding a controlled,
  camera-ready DUAL baseline under two explicitly tagged preprocessing
  protocols and isolating an MR-D-BSCP-shaped image-pyramid schedule within
  each protocol.
- Dataset: DIR-LAB 4DCT cases `01-10`, moving T50/end-exhale to fixed
  T00/end-inhale, seed `42`.
- Matrix: `N0,N1,P0,P1 x 10 cases = 40` records. N uses native full volumes,
  native intensities, no crop/resampling, and fixed-T00 R231 mask sampling. P
  uses the canonical pTVReg-style landmark crop at approximately 1 mm,
  `[80,900] -> [0,1]` CT scaling, no lung mask in optimization, and a
  five-voxel border domain.
- Common objective/model: pinned DUAL SIREN `3-256-256-3`, omega `10/30`,
  30,000 samples per step, sampled global NCC plus released Laplacian and
  negative-Jacobian weights `0.01/0.01`, fine regularization scale `0.1`, and
  no semantic-mask loss.
- Schedule: fixed stage lengths `400,400,800,900` (2500 total), fine branch at
  step 400. N0/P0 use image factors `1,1,1,1`; N1/P1 use independently built,
  antialiased levels `8,4,2,1`. Model weights persist; fresh AdamW and
  stage-local warmup-cosine state are used identically in every arm.
- Execution: all four visible RTX 6000 Ada GPUs, one optimizer per GPU,
  dynamic work queue. All 40 records are attempted. Technical failures are
  isolated and retried while unrelated records continue; TRE and regularity
  values never stop a record or the queue.
- Headline endpoint: native snap-to-voxel TRE in mm. P fields are converted
  from the 1 mm registration crop to the native crop before the headline
  metric. Registration-grid snap/subvoxel TRE, direct/dense diagnostics,
  runtime, peak memory, and shared posthoc regularity are retained.
- Interpretation: only `N0 <-> N1` and `P0 <-> P1` isolate the schedule. N/P
  changes preprocessing and optimization domain. N1/P1 are MR-inspired image
  pyramids, not the literal MR-D-BSCP control lattice, knot insertion,
  objective, or tuned weights. Native anisotropy and fixed normalized-grid
  regularization scale remain explicit caveats.
- Completed headline/native snap TRE (mean +/- sample SD, mm): N0
  `1.0738 +/- 0.2041`, N1 `1.0649 +/- 0.2091`, P0
  `1.0410 +/- 0.2160`, and P1 `1.0461 +/- 0.2211`. The paired
  multiresolution-minus-full-resolution deltas are `-0.0089 +/- 0.0134 mm`
  for N and `+0.0051 +/- 0.0147 mm` for P, a small and
  direction-inconsistent schedule effect under the frozen recipe.
- Common foreground SDlogJ is `0.1282`, `0.1286`, `0.1135`, and `0.1138`
  for N0, N1, P0, and P1, respectively; all four have zero nonpositive
  foreground Jacobian percentage. Voxel-coordinate diffusion/bending values
  remain protocol-tagged and are not directly compared across N/P.
- Contextual iso-1-mm references are pTVReg `0.9191 +/- 0.1607 mm` and
  MR-D-BSCP `0.9175 +/- 0.1561 mm`. MR-D-BSCP uses one fixed cohort-selected
  `w=0.055` for all cases, never a per-case oracle; both references retain
  distinct preprocessing tags and regularity tables.
- Review root and locked specification:
  `experiments/2026-08-02_dual_inr_dirlab_controlled_multires/`.

### P16: COPD DUAL-INR Six-Arm Author/Controlled Baseline

- Status: production launched 2026-08-03. The all-case controlled dry-run and
  all four controlled case-01 technical smokes passed; the N/P 40-record
  queue is active. Five-lobe author preparation is atomic and its validation,
  A smoke, and A0-to-A1 production pipeline are queued to begin only after
  publication of the complete all-case artifact.
- Purpose: extend the audited DUAL author reproduction and controlled
  native/pTV baselines to COPD without selecting hyperparameters on COPD
  outcomes. The resulting camera-ready table separates author fidelity,
  semantic-loss ablation, preprocessing, and the MR-shaped image schedule.
- Dataset and direction: COPD cases 01-10; moving eBHCT to fixed iBHCT
  (eBH -> iBH); seed 42.
- Matrix: A0,A1,N0,N1,P0,P1 x 10 = 60 new records. A0 is the released
  five-lobe objective; A1 removes only the semantic loss and replays its paired
  A0 transition trace. N0/N1 use native fixed-iBH R231 sampling; P0/P1 use the
  existing legacy-Python pTV-style 1 mm artifact and no lung mask in
  optimization. N0/P0 use factors 1,1,1,1; N1/P1 use 8,4,2,1.
- Frozen common settings: DUAL SIREN 3-256-256-3, frequencies 10/30,
  scales .1/.05, 30,000 points/step, NCC, Laplacian and negative-Jacobian
  weights .01/.01, fine scale .1, and no N/P semantic term. The author
  arms retain the released adaptive blur schedule 4,2,0 and 3,000 steps;
  controlled N/P arms use 400,400,800,900 steps with fresh optimizer and
  warmup-cosine state per stage.
- Prepared-data requirement: all ten fixed/moving TotalSegmentator masks must
  contain labels 10..14; R231 is retained only as a fixed-iBH posthoc metric
  mask for A, never as an A optimizer input.
- References: import the official pTVReg COPD best casewise TRE only; regularity
  is N/A because no official dense field exists. Use one completed cohort,
  MR-D-BSCP P2e, as the lowest-population-TRE 1 mm reference, never per-case
  selection.
- Output and review roots are versioned external scratch artifacts and
  experiments/2026-08-03_dual_inr_copd_six_arm/, respectively.
- Known limitation: the P artifact is pTV-style but its legacy crop bounds do
  not exactly equal official pTVReg MATLAB bounds. Consequently P/MR are
  matched to each other, while pTVReg remains a contextual, tagged reference.

### P17: OASIS DirectCP 8-4-2 Runtime Optimization Sweep

- Status: completed 2026-08-04; smoke and first10 aggregation passed with
  `140/140` first10 summaries and `errors=[]`.
- Purpose: improve practical runtime for OASIS DirectCP `8-4-2` while retaining
  most of the current best alpha-1000 accuracy, so the reviewer-facing runtime
  argument can report optimization time and total time.
- Dataset: OASIS BrainMR seed-3 shuffled first10 ranks `0-9`; smoke ranks
  `0,4,8`.
- Method: `direct_cp_8_4_2_be_bspline_analytic_multistage`; alpha `1000`;
  cps schedule `8,4,2`; image downsample schedule `4,2,1`; seed `1`.
- Candidate matrix: shorter fixed schedules `250,500,800`, `150,300,500`, and
  `100,200,300`; higher LRs from `3e-3` through selected `1e-1`; and adaptive
  plateau-stopping candidates with recorded per-stage minimums, patience, and
  loss-improvement thresholds.
- Baseline for decision: current DirectCP `8-4-2` first10 alpha-1000 Dice
  `0.8070`, HD95 `2.2020`, training `194.8 s`, total `229.7 s`.
- Promotion criteria: smoke candidates must avoid nonfinite failures and stay
  within `0.05` Dice and `0.5 mm` HD95 of matched DirectCP `8-4-2` baseline.
  First10 tiers compare with DirectCP cps2, SINR cps2, and the user-accepted
  `0.02` Dice / `0.2 mm` HD95 degradation window.
- Output root:
  `/database/inrorspline/runs/oasis_directcp842_runtime_sweep_alpha1000_v1`.
- Review root:
  `experiments/2026-08-04_oasis_directcp842_runtime_sweep/`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`, planned GPUs `0,1,2,3`.
- Known limitation: first10 runtime/accuracy screening only; any candidate used
  in paper-facing claims should be explicitly tagged unless extended to the
  intended final OASIS population.

### P18: OASIS DirectCP 8-4-2 Single-GPU Serial Runtime Control

- Status: completed 2026-08-04; both serial rank-0 jobs completed with `DONE`
  and no active GPU processes remained afterward.
- Purpose: check whether the P17 runtime reduction is underestimated or
  distorted by running four jobs concurrently. Run the original MR-D-BSCP /
  DirectCP `8-4-2` reference and the `A_fast_decay` candidate serially on one
  GPU, leaving the other visible GPUs idle.
- Dataset: OASIS BrainMR seed-3 shuffled rank `0` only, fixed `0333`, moving
  `0115`, original sorted index `158`.
- Methods:
  - Original MR-D-BSCP / DirectCP `8-4-2`: alpha `1000`, lr `1e-3`, cps
    `8,4,2`, downsample `4,2,1`, stage epochs `400,800,1300`.
  - `A_fast_decay`: alpha `1000`, base lr `1e-2`, stage lr
    `3e-2,1e-2,3e-3`, cps `8,4,2`, downsample `4,2,1`, max stage epochs
    `400,800,1300`, adaptive minimums `75,150,250`, patience `50,75,100`.
- Execution: Docker container `py310sitreg_v1_test1`, GPU `0` only, one job at
  a time. Confirm no compute processes before launch.
- Output root:
  `/database/inrorspline/runs/oasis_directcp842_single_gpu_control_alpha1000_v1`.
- Review root:
  `experiments/2026-08-04_oasis_directcp842_single_gpu_control/`.
- Primary metrics: training elapsed seconds, total elapsed seconds, wall-clock
  command seconds, actual epochs, Dice, and HD95.
- Actual result on rank `0`: original MR-D-BSCP / DirectCP `8-4-2` ran
  `2500` epochs with train/total/wall `186.69/216.20/226.14 s`; `A_fast_decay`
  ran `600` epochs with train/total/wall `151.08/181.89/191.18 s`. Speedups:
  `1.24x` train, `1.19x` total script, `1.18x` command wall. Dice/HD95 changed
  from `0.76074/2.65234` to `0.75901/2.66425`.
- Interpretation: the modest runtime reduction is not mainly a four-GPU
  concurrency artifact; fixed per-stage and post-training costs prevent epoch
  reduction from translating linearly into wall time.
- Known limitation: single case timing control only; if results differ
  materially from P17, extend to first10 serial or one-GPU queued runs before
  making paper-facing runtime claims.

### P19: OASIS DirectCP 8-4-2 Detailed Timing Breakdown

- Status: completed 2026-08-04; smoke, rank-0, and first10 serial timing all
  passed with `20/20` first10 jobs complete and timing JSONs present.
- Purpose: decompose DirectCP `8-4-2` wall time into reconciled setup,
  per-stage fixed overhead, optimizer-loop step-proportional work,
  post-training evaluation, and output-writing categories. This follows P18,
  where large epoch reduction produced only modest wall-clock reduction.
- Dataset: OASIS BrainMR seed-3 shuffled protocol. Validate on rank `0` first
  (fixed `0333`, moving `0115`, original sorted index `158`), then run ranks
  `0-9` serially if rank-0 timing reconciliation passes.
- Methods:
  - Original MR-D-BSCP / DirectCP `8-4-2`: alpha `1000`, lr `1e-3`, cps
    `8,4,2`, downsample `4,2,1`, stage epochs `400,800,1300`, no early stop.
  - `A_fast_decay`: alpha `1000`, base lr `1e-2`, stage lr
    `3e-2,1e-2,3e-3`, cps `8,4,2`, downsample `4,2,1`, max stage epochs
    `400,800,1300`, adaptive minimums `75,150,250`, patience `50,75,100`.
- Timing mode: `run_one.py --timing-detail deep` for the multistage DirectCP
  path only. Existing non-timing experiments keep the default `off` behavior.
- Execution order: tiny `1,1,1` epoch smoke, serial rank-0 pair on GPU `0`,
  analyzer reconciliation, then first10 serial pair on GPU `0` if accounting
  passes. GPUs `1-3` are intentionally unused by the launcher.
- Output root:
  `/database/inrorspline/runs/oasis_directcp842_timing_breakdown_alpha1000_v1`.
- Review root:
  `experiments/2026-08-04_oasis_directcp842_timing_breakdown/`.
- Expected artifacts: per-run `timing.json`, phase-specific
  `command_timing_<phase>.csv`, `timing_reconciliation.csv`,
  `timing_stage_breakdown.csv`, `timing_method_comparison.csv`,
  `timing_stage_linear_fits.csv`, and `timing_report.md`.
- Acceptance checks: `py_compile` modified/new scripts passed; smoke,
  rank-0, and first10 all wrote `DONE`, `summary.json`, and `timing.json`.
  First10 analyzer reported residual failures `0`, parent intervals with large
  self-unaccounted time `0`, external uninstrumented wall mean `4.29 s`, and
  no non-target GPU compute-app observations.
- First10 result: original MR-D-BSCP / DirectCP `8-4-2` averaged wall/script/
  training/optimizer-loop `247.76/243.47/206.12/110.59 s` with Dice/HD95
  `0.80697/2.2020` over `2500` epochs. `A_fast_decay` averaged
  `206.62/202.32/161.79/63.69 s` with Dice/HD95 `0.80581/2.2147` over
  `577.5` epochs.
- First10 interpretation: `A_fast_decay` saved `41.14 s` wall per case
  (`1.207x` speedup) and `46.89 s` optimizer-loop time while losing only
  `0.00116` Dice and increasing HD95 by `0.0127 mm`. The modest wall speedup
  despite a `1922.5` epoch reduction is explained by fixed/io known cost of
  about `132.10 s` for original and `137.80 s` for `A_fast_decay`.
- Known limitation: detailed timing instrumentation, especially deep CUDA
  event collection, may add small overhead. Comparisons are therefore made
  between methods under the same timing mode.

### P20: OASIS DirectCP 8-4-2 DVF-Only Runtime Timing

- Status: completed 2026-08-04; smoke and full rank-0 serial timing passed.
- Purpose: measure the practical runtime needed to go from OASIS input images to
  a dense DirectCP `8-4-2` DVF, without post-training evaluation or full
  metric/output packaging. This follows P19, where fixed evaluation/output costs
  complicated interpretation of epoch-count savings.
- Dataset: OASIS BrainMR seed-3 shuffled rank `0` only, fixed `0333`, moving
  `0115`, original sorted index `158`.
- Methods:
  - Original MR-D-BSCP / DirectCP `8-4-2`: alpha `1000`, lr `1e-3`, cps
    `8,4,2`, downsample `4,2,1`, stage epochs `400,800,1300`, no early stop.
  - `A_fast_decay`: alpha `1000`, base lr `1e-2`, stage lr
    `3e-2,1e-2,3e-3`, cps `8,4,2`, downsample `4,2,1`, max stage epochs
    `400,800,1300`, adaptive minimums `75,150,250`, patience `50,75,100`.
- Timing mode: `run_one.py --output-mode dvf_only --timing-detail deep` for the
  multistage DirectCP path only. DVF endpoints are marked at final dense
  expansion in memory and after dense `.npy` write on disk.
- Execution order: tiny `1,1,1` epoch smoke for both methods, analyzer check
  that evaluation/output events are absent from the DVF endpoint, then full
  serial rank-0 original and `A_fast_decay` on GPU `0`.
- Output root:
  `/database/inrorspline/runs/oasis_directcp842_dvf_only_timing_alpha1000_v1`.
- Review root:
  `experiments/2026-08-04_oasis_directcp842_dvf_only_timing/`.
- Expected artifacts: `dvf_only_timing_report.md`,
  `dvf_only_method_comparison.csv`, `dvf_only_stage_breakdown.csv`,
  `dvf_only_grouped_breakdown.csv`, `dvf_only_reconciliation.csv`, and
  `command_timing_dvf_only_rank0.csv`.
- Acceptance checks: `py_compile` passed for `run_one.py`,
  `run_directcp842_dvf_only_timing.py`, and
  `analyze_directcp842_dvf_only_timing.py` after redirecting pycache to `/tmp`.
  Smoke and full rank-0 runs wrote `DONE`, `summary.json`, `timing.json`, and
  dense DVF `.npy` for both methods. Rank-0 analyzer reported reconciliation
  residual failures `0`, parent intervals with large self-unaccounted time `0`,
  evaluation/full-output events `0`, and no non-target GPU compute-app
  observations.
- Rank-0 result: original MR-D-BSCP / DirectCP `8-4-2` ran `2500` epochs and
  reached DVF-in-memory/DVF-file/external-wall in `234.18/234.26/238.22 s`.
  `A_fast_decay` ran `600` epochs and reached DVF-in-memory/DVF-file/external
  wall in `175.22/175.28/178.99 s`.
- Rank-0 interpretation: `A_fast_decay` saved `58.96 s` to in-memory DVF and
  `58.98 s` to DVF file (`59.24 s` command wall). Most of the saving came from
  optimizer-loop time (`52.20 s`), with additional non-loop training reduction
  (`8.41 s`). Dense DVF file write itself was only about `0.03-0.04 s`.
- Known limitation: this is a rank-0 timing characterization only. It is meant
  to isolate input-image-to-DVF runtime, not to report Dice/HD95/regularity.

### P21: OASIS DirectCP 8-4-2 Production DVF Timing And Transition Benchmark

- Status: completed 2026-08-04 after production smoke, diagnostic smoke,
  CPU/GPU transition benchmark, full rank-0 production pair, and analyzer
  reconciliation passed.
- Purpose: convert the DirectCP `8-4-2` DVF-only path from conservative
  diagnostic timing to production timing by skipping intermediate dense
  expansion/preservation checks unless explicitly requested. Separately
  benchmark CPU/GPU runtime and numerical agreement for dense expansion, pure
  knot insertion, and preservation checks.
- Dataset: OASIS BrainMR seed-3 shuffled rank `0` only, fixed `0333`, moving
  `0115`, original sorted index `158`.
- Methods:
  - Original MR-D-BSCP / DirectCP `8-4-2`: alpha `1000`, lr `1e-3`, cps
    `8,4,2`, downsample `4,2,1`, stage epochs `400,800,1300`, no early stop.
  - `A_fast_decay`: alpha `1000`, base lr `1e-2`, stage lr
    `3e-2,1e-2,3e-3`, cps `8,4,2`, downsample `4,2,1`, max stage epochs
    `400,800,1300`, adaptive minimums `75,150,250`, patience `50,75,100`.
- Timing mode: `run_one.py --output-mode dvf_only --timing-detail deep` with
  production defaults `--prolongation-preserve-check` off and
  `--save-intermediate-dense` off. Diagnostic mode can reproduce conservative
  checks by passing `--prolongation-preserve-check`.
- Output root:
  `/database/inrorspline/runs/oasis_directcp842_production_dvf_timing_alpha1000_v1`.
- Review root:
  `experiments/2026-08-04_oasis_directcp842_production_dvf_timing/`.
- Expected artifacts: production `dvf_only_timing_report.md`,
  `dvf_only_method_comparison.csv`, `dvf_only_stage_breakdown.csv`,
  `dvf_only_previous_conservative_comparison.csv`,
  `transition_kernel_benchmark.csv`, and
  `transition_kernel_benchmark_report.md`.
- Actual rank-0 production results: original MR-D-BSCP / DirectCP `8-4-2`
  ran `2500` epochs and reached full-script DVF-in-memory/DVF-file/external
  wall in `174.72/174.78/177.85 s`; loaded-images-to-DVF-memory/file was
  `149.83/149.89 s`. `A_fast_decay` ran `600` epochs and reached full-script
  DVF-in-memory/DVF-file/external wall in `129.29/129.36/132.00 s`;
  loaded-images-to-DVF-memory/file was `101.27/101.33 s`.
- Actual savings: `A_fast_decay` saved `45.42 s` to full-script in-memory DVF,
  `45.42 s` to full-script DVF file, and `45.84 s` command wall relative to
  original. Loaded-images-to-DVF-memory/file savings were `48.56/48.56 s`.
  Optimizer-loop savings were `49.56 s`; non-loop training was essentially
  unchanged (`-0.22 s`) after diagnostic dense expansions were removed.
- Production-vs-conservative timing comparison: skipping intermediate
  diagnostic dense expansion/preservation checks reduced original full-script
  DVF-in-memory time from `234.18 s` to `174.72 s` and `A_fast_decay` from
  `175.22 s` to `129.29 s`.
- CPU/GPU transition benchmark result: dense expansion and knot insertion
  matched exactly within tolerance (`max_abs_diff=0`, tolerance `1e-5`) for all
  synthetic rank-0 schedule-shape tests. GPU medians were faster than CPU for
  dense expansion (`33.50x`, `55.22x`, `78.05x` for cps `8,4,2`) and pure knot
  insertion (`2.26x` for `8->4`, `15.87x` for `4->2`). These kernels are
  already PyTorch tensor operations and already run on CUDA in normal GPU
  DirectCP runs; the main avoidable overhead was diagnostic dense
  expansion/preservation checking.
- Acceptance: `2/2` production rank-0 jobs completed; `2/2` dense DVF files,
  `summary.json`, `timing.json`, and `DONE` files were present; evaluation and
  full-output event counts were `0`; reconciliation residual failures were
  `0`; parent self-unaccounted warnings were `0`; non-target GPU compute-app
  observations were `0`.
- Known limitation: this run isolates runtime; no Dice/HD95/regularity metrics
  are computed in production DVF-only mode.

### P22: OASIS DirectCP 8-4-2 Persistent-Process DVF Timing First10

- Status: completed 2026-08-04 after smoke, full first10 persistent workers,
  DVF-only analyzer, and persistent-process analyzer passed.
- Purpose: measure practical persistent-process input-image-to-DVF runtime by
  running OASIS seed-3 shuffled ranks `0-9` sequentially inside one long-lived
  Python worker subprocess per candidate. Rank `0` is the cold first case;
  ranks `1-9` estimate warm persistent-process throughput after CUDA/PyTorch
  and cuDNN first-use overhead has been paid.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`.
- Methods:
  - Original MR-D-BSCP / DirectCP `8-4-2`: alpha `1000`, lr `1e-3`, cps
    `8,4,2`, downsample `4,2,1`, stage epochs `400,800,1300`, no early stop.
  - `A_fast_decay`: alpha `1000`, base lr `1e-2`, stage lr
    `3e-2,1e-2,3e-3`, cps `8,4,2`, downsample `4,2,1`, max stage epochs
    `400,800,1300`, adaptive minimums `75,150,250`, patience `50,75,100`.
- Timing mode: production `run_one.py --output-mode dvf_only
  --timing-detail deep` with diagnostic preservation checks and intermediate
  dense saves off.
- Process model: one worker subprocess for original and one separate worker
  subprocess for `A_fast_decay`; this avoids giving the second method the first
  method's warm CUDA state.
- Output root:
  `/database/inrorspline/runs/oasis_directcp842_persistent_dvf_timing_alpha1000_v1`.
- Review root:
  `experiments/2026-08-04_oasis_directcp842_persistent_process_dvf_timing/`.
- Expected artifacts: `persistent_process_timing_report.md`,
  `persistent_process_summary.csv`, `persistent_process_case_timing.csv`,
  `persistent_process_savings.csv`, `dvf_only_method_comparison.csv`,
  `dvf_only_stage_breakdown.csv`, and `command_timing_dvf_only_first10.csv`.
- Actual cold rank-0 results: original MR-D-BSCP / DirectCP `8-4-2`
  full-script DVF-in-memory/DVF-file/wall `177.41/177.48/177.49 s`;
  `A_fast_decay` `144.55/144.63/144.64 s`.
- Actual warm ranks `1-9` results: original full-script
  DVF-in-memory/DVF-file/wall mean `62.48/62.55/62.56 s` with wall standard
  deviation `0.17 s`; `A_fast_decay` `15.07/15.18/15.19 s` with wall
  standard deviation `0.69 s`.
- Actual amortized ranks `0-9` results: original full-script
  DVF-in-memory/DVF-file/wall mean `73.97/74.05/74.06 s`; `A_fast_decay`
  `28.02/28.12/28.14 s`.
- Actual speed comparison: on warm persistent-process cases, `A_fast_decay`
  saved `47.40 s` to in-memory DVF, `47.37 s` to DVF file, and `47.37 s`
  wall per case, with mean epochs `575` vs original `2500`. On amortized
  first10 including cold rank0, `A_fast_decay` saved `45.95 s` to in-memory
  DVF and `45.92 s` wall per case.
- Acceptance: smoke and full first10 completed; `20/20` first10 case rows had
  dense DVF, `summary.json`, `timing.json`, and `DONE`; evaluation/full-output
  event count was `0`; reconciliation residual failures were `0`; parent
  self-unaccounted warnings were `0`; non-target GPU compute-app observations
  were `0`.
- Known limitation: `cold_minus_warm_mean` estimates first-use overhead plus
  rank-0-specific case/runtime differences; it is not a pure hardware
  constant. This run isolates runtime only and intentionally skips evaluation
  metrics.

### P23: OASIS SINR cps4/cps2 Persistent-Process DVF Timing First10

- Status: completed 2026-08-04 after SINR smoke, full first10 persistent
  workers, DVF-only analyzer, and persistent-process analyzer passed.
- Purpose: add SINR cps4 and SINR cps2 to the same persistent-process
  input-image-to-DVF timing surface as DirectCP `8-4-2`, so cold rank-0,
  warm ranks `1-9`, and amortized first10 runtime can be compared in one
  report.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`.
- Methods:
  - `sinr_cps4_be_bspline_analytic`: alpha `1000`, cps `4`, lr `1e-4`,
    `2500` epochs, seed `1`, no early stopping.
  - `sinr_cps2_be_bspline_analytic`: alpha `1000`, cps `2`, lr `1e-4`,
    `2500` epochs, seed `1`, no early stopping.
- Timing mode: production `run_one.py --output-mode dvf_only
  --timing-detail deep`. DVF-only mode skips image/segmentation evaluation,
  Dice, HD95/ASD, dense NCC, regularity metrics, model/control checkpoint
  saves, and full packaging.
- Process model: one worker subprocess for SINR cps4 and one separate worker
  subprocess for SINR cps2; rank `0` is the cold first case in each worker,
  and ranks `1-9` are warm persistent-process cases.
- Output root:
  `/database/inrorspline/runs/oasis_directcp842_persistent_dvf_timing_alpha1000_v1`.
- Review root:
  `experiments/2026-08-04_oasis_directcp842_persistent_process_dvf_timing/`.
- Expected artifacts: updated `persistent_process_timing_report.md`,
  `persistent_process_summary.csv`, `persistent_process_case_timing.csv`,
  `persistent_process_pairwise_runtime_comparison.csv`,
  `persistent_process_cold_warm_breakdown.csv`,
  `dvf_only_method_comparison.csv`, `dvf_only_stage_breakdown.csv`, and
  `command_timing_dvf_only_first10.csv`.
- Actual SINR runtime:
  - `sinr_cps4` cold rank0 wall `182.81 s`; warm ranks `1-9` wall
    `107.49 +/- 0.09 s`; amortized ranks `0-9` wall
    `115.02 +/- 23.82 s`.
  - `sinr_cps2` cold rank0 wall `356.54 s`; warm ranks `1-9` wall
    `271.45 +/- 0.15 s`; amortized ranks `0-9` wall
    `279.96 +/- 26.91 s`.
- Warm persistent-process comparison: `sinr_cps4` was `44.93 s` slower than
  original DirectCP `8-4-2` and `92.30 s` slower than `A_fast_decay` by
  external wall; `sinr_cps2` was `208.89 s` slower than original DirectCP
  `8-4-2` and `256.26 s` slower than `A_fast_decay`.
- Acceptance: combined report has `40/40` case rows, `80` stage rows, dense
  DVF/timing/DONE present for all command rows, SINR epochs all `2500`,
  evaluation/full-output event count `0`, reconciliation residual failures
  `0`, parent self-unaccounted warnings `0`, and non-target GPU compute-app
  observations `0`.
- Known limitation: this extension isolates runtime only and intentionally
  skips evaluation metrics. Existing DirectCP P22 outputs are reused rather
  than rerun.

### P24: OASIS cps4 Runtime Sweep For D-BSCP And SINR

- Status: completed 2026-08-06.
- Purpose: adapt the MR-D-BSCP runtime-optimization workflow to single-stage
  `D-BSCP cps4` and `SINR cps4`, screening faster fixed-epoch and adaptive
  learning-rate variants before any controlled sequential persistent-process
  timing follow-up.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`.
- Methods:
  - `direct_cp_cps4_be_bspline_analytic` (`D-BSCP cps4`): alpha `1000`;
    rerun baseline `lr=1e-3`, `2500` epochs; candidates
    `D_F1000_lr3e-3`, `D_F600_lr1e-2`, `D_F400_lr3e-2`,
    `D_A_lr1e-2`, and `D_A_lr3e-2`.
  - `sinr_cps4_be_bspline_analytic` (`SINR cps4`): alpha `1000`; rerun
    baseline `lr=1e-4`, `2500` epochs; candidates `S_F1500_lr3e-4`,
    `S_F1000_lr1e-3`, `S_F600_lr3e-3`, `S_A_lr1e-3`, and
    `S_A_lr3e-3`.
- Adaptive early stopping: single-stage total training loss only, no extra
  evaluation pass; stop only after the candidate-specific `min_epochs`, after
  `patience` epochs without at least `min_delta=1e-4` improvement.
- Screening mode: full evaluation/output mode because Dice, HD95, ASD, and
  regularity metrics are required to decide whether a candidate is promising.
- Promising gate: same-method first10 Dice drop `<=0.02`, HD95 increase
  `<=0.2 mm`, and either full-pipeline runtime improvement `>=25%` or clear
  epoch reduction with no accuracy loss. Existing DirectCP cps2 and SINR cps2
  rows are comparison anchors, not rerun in this sweep.
- Output root:
  `/database/inrorspline/runs/oasis_cps4_single_stage_runtime_sweep_alpha1000_v1`.
- Review root:
  `experiments/2026-08-05_oasis_cps4_single_stage_runtime_sweep/`.
- Screening artifacts: `method_ordered_report.md`, `candidate_summary.csv`,
  `case_results.csv`, `accuracy_runtime_tradeoff.csv`,
  `promising_candidates.csv`, `job_manifest.csv/json`, and
  `launch_summary.json`.
- Screening completion: dry-run manifest expected `120` jobs; rank-0 smoke
  completed `12/12`; full first10 screening completed `120/120` with zero
  launcher failures and zero aggregation errors.
- Promising screening candidates:
  - `D_A_lr1e-2`: Dice `0.7946`, HD95 `2.3653`, total elapsed `105.4 s`,
    full-pipeline saving `35.9%`, mean epochs `476.9`.
  - `D_F600_lr1e-2`: Dice `0.7949`, HD95 `2.3795`, total elapsed `117.7 s`,
    full-pipeline saving `28.4%`, mean epochs `600.0`.
  - `D_F1000_lr3e-3`: Dice `0.7958`, HD95 `2.3343`, total elapsed
    `120.0 s`, full-pipeline saving `27.0%`, mean epochs `1000.0`.
  - `S_F1000_lr1e-3`: Dice `0.7866`, HD95 `2.2503`, total elapsed
    `131.0 s`, full-pipeline saving `31.7%`, mean epochs `1000.0`.
  - `S_A_lr1e-3`: Dice `0.7813`, HD95 `2.2331`, total elapsed `134.4 s`,
    full-pipeline saving `29.9%`, mean epochs `846.4`.
- Rejected fast/high-LR candidates: `D_A_lr3e-2`, `D_F400_lr3e-2`,
  `S_A_lr3e-3`, and `S_F600_lr3e-3` failed the HD95/Dice tolerance despite
  runtime savings. `S_F1500_lr3e-4` preserved accuracy but did not meet the
  runtime gate.
- cps2 anchor comparison: none of the promising cps4 candidates beat their
  same-family cps2 reference on both Dice and HD95; cps2 rows remain anchors
  and were not rerun here.
- Follow-up persistent DVF-only timing: ran baselines plus the five promising
  candidates as one persistent worker per candidate on GPU `0`, ranks `0-9`
  serially. Smoke completed `14/14`; full timing completed `70/70`; dense DVF,
  `summary.json`, `timing.json`, and `DONE` existed for every run. Validation
  reported evaluation/full-output events `0`, non-target GPU observations `0`,
  reconciliation residual failures `0`, and parent self-unaccounted warnings
  `0`.
- Warm persistent-process DVF-file time from loaded images, ranks `1-9`:
  D-BSCP baseline `80.80 s`; `D_F1000_lr3e-3` `31.83 s`; `D_F600_lr1e-2`
  `19.13 s`; `D_A_lr1e-2` `15.12 s`. SINR baseline `106.92 s`;
  `S_F1000_lr1e-3` `42.73 s`; `S_A_lr1e-3` `37.04 s`.
- Persistent timing artifacts: `persistent_cps4_dvf_timing_report.md`,
  `persistent_cps4_summary.csv`, `persistent_cps4_case_timing.csv`,
  `persistent_cps4_runtime_vs_baseline.csv`, and
  `persistent_cps4_screen_and_timing.csv`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested GPUs `0,1,2,3`.
- Known limitation: this is a first10 screening sweep, not a final
  full-population result. Screening runtime is full evaluation/output runtime;
  persistent timing is DVF-only and intentionally excludes evaluation and full
  packaging.

### P25: OASIS cps2 Runtime Sweep For D-BSCP And SINR

- Status: completed 2026-08-07; implementation, smoke, screening, persistent
  timing, clean D-BSCP baseline timing rerun, and final aggregation handled by
  Codex.
- Purpose: repeat the completed cps4 runtime-optimization workflow for
  single-stage `D-BSCP cps2` and `SINR cps2`, screening faster fixed-epoch and
  adaptive variants before controlled persistent-process DVF-only timing.
- Dataset: OASIS BrainMR seed-3 shuffled ranks `0-9`, original sorted indices
  `158,62,14,69,47,137,110,6,116,88`.
- Methods:
  - `direct_cp_cps2_be_bspline_analytic` (`D-BSCP cps2`): alpha `1000`;
    rerun baseline `lr=1e-3`, `2500` epochs; candidates
    `D2_F1000_lr3e-3`, `D2_F600_lr1e-2`, `D2_F400_lr3e-2`,
    `D2_A_lr1e-2`, and `D2_A_lr3e-2`.
  - `sinr_cps2_be_bspline_analytic` (`SINR cps2`): alpha `1000`; rerun
    baseline `lr=1e-4`, `2500` epochs; candidates `S2_F1500_lr3e-4`,
    `S2_F1000_lr1e-3`, `S2_F600_lr3e-3`, `S2_A_lr1e-3`, and
    `S2_A_lr3e-3`.
- Adaptive early stopping: single-stage total training loss only, no extra
  evaluation pass; stop only after the candidate-specific `min_epochs`, after
  `patience` epochs without at least `min_delta=1e-4` improvement.
- Screening mode: full evaluation/output mode because Dice, HD95, ASD, and
  regularity metrics are required to decide whether a candidate is promising.
- Promising gate: same-method first10 Dice drop `<=0.02`, HD95 increase
  `<=0.2 mm`, and either full-pipeline runtime improvement `>=25%` or clear
  epoch reduction with no accuracy loss. Existing cps4 and MR-D-BSCP rows are
  contextual anchors, not the primary gate.
- Screening output root:
  `/database/inrorspline/runs/oasis_cps2_single_stage_runtime_sweep_alpha1000_v1`.
- Persistent DVF-only output root:
  `/database/inrorspline/runs/oasis_cps2_single_stage_persistent_dvf_timing_alpha1000_v1`.
- Review root:
  `experiments/2026-08-06_oasis_cps2_single_stage_runtime_sweep/`.
- Expected screening artifacts: `method_ordered_report.md`,
  `candidate_summary.csv`, `case_results.csv`,
  `accuracy_runtime_tradeoff.csv`, `promising_candidates.csv`,
  `job_manifest.csv/json`, and `launch_summary.json`.
- Expected persistent timing artifacts: `persistent_cps2_dvf_timing_report.md`,
  `persistent_cps2_summary.csv`, `persistent_cps2_case_timing.csv`,
  `persistent_cps2_runtime_vs_baseline.csv`, and
  `persistent_cps2_screen_and_timing.csv`.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`; requested screening GPUs `0,1,2,3`
  and persistent timing GPU `0`.
- Completion summary:
  - Screening completed `120/120` full-evaluation jobs.
  - Promising D-BSCP cps2 candidates: `D2_A_lr3e-2`, `D2_A_lr1e-2`,
    `D2_F400_lr3e-2`, and `D2_F600_lr1e-2`.
  - Promising SINR cps2 candidates: `S2_F1000_lr1e-3`, `S2_A_lr1e-3`,
    and `S2_F1500_lr3e-4`.
  - Persistent DVF-only timing completed `90/90` rows with dense DVFs present,
    failed subprocesses `0`, DVF-only evaluation/full-output events `0`,
    reconciliation residual failures `0`, parent self-unaccounted warnings `0`,
    and non-target GPU compute-app observations `0`.
  - Warm ranks `1-9` external wall: D-BSCP cps2 baseline `86.00 s/case`,
    `D2_A_lr3e-2` `8.82 s/case`; SINR cps2 baseline `273.91 s/case`,
    `S2_F1000_lr1e-3` `109.39 s/case`.
- Known limitation: this is a first10 screening sweep. Screening runtime is
  full evaluation/output runtime; persistent timing is DVF-only and
  intentionally excludes evaluation and full packaging.

### P26: OASIS MR-SINR 8/4/2 Revision Pilot

- Status: completed 2026-08-07; technical pilot passed, but composed MR-SINR
  was not promoted based on the recorded pilot decision.
- Purpose: test whether the intended composed-field MR-SINR schedule is
  technically stable and worth promoting to seed-3 first10/first100 runs, with
  a continued/re-query SINR ablation for comparison.
- Dataset: OASIS BrainMR.
- Split config:
  `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Pilot case: seed-3 rank `0`, pair `(0333, 0115)`, original sorted
  `pair_test_200` index `158`.
- Methods:
  - `mr_sinr842_composed`: independent SINR residual stages at BSCP
    `8,4,2`, training each finer stage on the original moving image warped by
    the frozen composed previous stages, with final pull-field composition
    `phi842(x)=phi8(phi4(phi2(x)))`.
  - `mr_sinr842_requery`: one continued SINR generator re-queried at BSCP
    `8,4,2`, optimizing an absolute field at each stage without composition.
- Optimization: alpha `1000`, total `2500` epochs, stage epochs
  `400,800,1300`, downsample factors `4,2,1`, SINR omega `50.0`, default SINR
  LR `1e-4`, full-image global NCC plus analytic cubic B-spline BE per stage.
- Foreground and metrics: fixed raw image-positive foreground,
  `fixed_image > 0`, for foreground-scoped regularity; segmentation labels are
  only for Dice/HD95/ASD.
- Output root:
  `/database/inrorspline/runs/oasis_mr_sinr842_revision_pilot_v1`.
- Review root:
  `experiments/2026-08-07_oasis_mr_sinr842_revision/`.
- Expected artifacts: normal per-run `run_one.py` outputs plus
  `pilot_comparison_summary.csv/json`, `pilot_decision.json`,
  `pilot_report.md`, combined metric tables, composed-vs-sequential warp
  diagnostics, and re-query transition-jump diagnostics.
- Launch environment: Docker container `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`, GPU selected at launch.
- Known limitation: rank-0 one-case pilot only. The composed final deformation
  is not a single B-spline field, so final BE/diffusion regularity is dense
  posthoc regularity; per-stage training regularization remains analytic
  B-spline BE.

### P27: LungCT DUAL-INR pTV 1 mm Fixed-Lung-Mask Revision

- Status: completed 2026-08-07. Smoke passed; full-length cases `01` and `08`
  completed for both loss branches in the pilot; full ten-case scale-up
  completed with `20/20` accepted records and `20/20` posthoc regularity
  records.
- Purpose: add a controlled DIR-Lab 4DCT DUAL-INR revision on the canonical
  pTVReg iso1mm image domain using the fixed T00 lung mask as both optimization
  sampling mask and posthoc metric mask.
- Dataset: DIR-Lab 4DCT only.
- Cases: pilot `01` and `08`; full scale-up cases `01-10` completed after
  pilot inspection.
- Direction: T50 moving image to T00 fixed image.
- Protocol tag:
  `ptv_landmark_crop_1mm_fixed_t00_lungmask_dual_inr_revision`.
- Variants:
  - `PL1D`: DUAL-native regularization branch, loss identifier
    `dual_native_reg`.
  - `PL1B`: IDIR-type coordinate-autograd bending-energy branch, loss
    identifier `idir_type_be`.
- Common model/settings: released DUAL SIREN `3-256-256-3`, omega `10/30`,
  semantic/lobe/segmentation loss disabled with `mask_weight=0`, stage lengths
  `400,400,800,900`, image factors `8,4,2,1`, total steps `2500`, seed `42`
  unless explicitly overridden.
- IDIR-type BE convention: preserve the audited IDIR coordinate-autograd
  bending-energy batch-size scaling; default pilot weight `10.0`.
- Output root:
  `${SCRATCH_ROOT}/runs/lungct_dual_inr_1mm_lungmask_revision_v1` or the
  equivalent container path supplied by
  `INRORSPLINE_DUAL_LUNGMASK_REVISION_RUN_ROOT`.
- Review root:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/`.
- Expected artifacts: resolved job manifest, smoke/pilot/full trackers,
  `optimization_index.csv/json`, `regularity_metrics.csv/json`,
  `population_statistics.csv/json`, `acceptance_report.md/json`, and
  `pilot_readout.md` or full-cohort readout.
- Pilot result: native snap TRE mean was `0.9282 +/- 0.2310 mm` for `PL1D`
  and `0.9481 +/- 0.1793 mm` for `PL1B`; foreground SDlogJ was
  `0.125992 +/- 0.041732` for `PL1D` and `0.149517 +/- 0.048923` for
  `PL1B`; both branches had foreground `J<=0` percentage `0.0`.
- Full result: native snap TRE mean was `0.9815 +/- 0.1686 mm` for `PL1D` and
  `0.9865 +/- 0.1689 mm` for `PL1B`; reg-grid subvoxel TRE was
  `1.1174 +/- 0.1289 mm` for `PL1D` and `1.1068 +/- 0.1296 mm` for `PL1B`.
  Against native snap TRE references, PL1D beat pTVReg in `3/10` cases and
  MR-D-BSCP pTV-loss `w=0.055` in `3/10` cases; PL1B beat pTVReg in `0/10`
  cases and MR-D-BSCP in `2/10` cases.
- Runtime: `PL1D` pilot training wall time averaged `352.2 s`; `PL1B`
  averaged `569.5 s`, reflecting the added IDIR-type coordinate-autograd BE
  cost.
- Full runtime: `PL1D` total wall time averaged `443.7 +/- 35.9 s`; `PL1B`
  averaged `638.4 +/- 38.1 s`.
- Acceptance: technical only; require expected artifacts, finite metrics,
  protocol `PL`, fixed T00 lung-mask sampling and metric rules, exact stages,
  and completed posthoc regularity. TRE and regularity magnitude are not
  completion gates.
- Known limitation: pilot records are protocol/implementation checks, not a
  selected full-cohort result. Do not compare this row as method-only evidence
  against native IDIR, pTVReg, DirectCP/SINR, or older DUAL rows unless crop,
  mask, metric, interpolation, loss, and regularization conventions are
  explicitly matched or tagged.

### P28: LungCT DUAL-INR pTV 1 mm Fixed-Lung-Mask Author-Schedule Revision

- Status: completed 2026-08-07 after P27 full-cohort review; full `PL1A`
  cohort completed with `10/10` accepted records and `10/10` posthoc
  regularity records.
- Purpose: test whether P27's weaker aggregate TRE is caused by the controlled
  MR-style image pyramid by running the same `PL` pTVReg 1 mm fixed T00
  lung-mask protocol with the released DUAL-INR original blur schedule.
- Dataset: DIR-Lab 4DCT only.
- Cases: full cases `01-10`.
- Direction: T50 moving image to T00 fixed image.
- Protocol tag:
  `ptv_landmark_crop_1mm_fixed_t00_lungmask_dual_inr_author_schedule_revision`.
- Variant:
  - `PL1A`: protocol `PL`, loss identifier `dual_native_reg`, original DUAL
    blur sigmas `4,2,0`, adaptive convergence transitions, total steps `3000`,
    semantic/lobe/segmentation loss disabled with `mask_weight=0`.
- Model/settings: released DUAL SIREN `3-256-256-3`, omega `10/30`, native
  DUAL Laplacian/negative-Jacobian regularization with weights `0.01/0.01`,
  fine regularizer scale `0.1`, seed `42` unless explicitly overridden.
- Output root:
  `${SCRATCH_ROOT}/runs/lungct_dual_inr_1mm_lungmask_author_schedule_v1` or
  the equivalent container path supplied at launch.
- Review root:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_author_schedule/`.
- Acceptance: technical only; require expected artifacts, finite metrics,
  protocol `PL`, fixed T00 lung-mask sampling and metric rules, exact blur
  sigmas `[4,2,0]`, all three blur stages visited, and completed posthoc
  regularity. TRE and regularity magnitude are not completion gates.
- Result: native snap TRE mean was `0.9758 +/- 0.1752 mm`; reg-grid subvoxel
  TRE was `1.1144 +/- 0.1280 mm`; foreground SDlogJ was
  `0.143408 +/- 0.037411`; foreground `J<=0` percentage was `0.0`.
- Comparison: PL1A beat pTVReg native snap TRE in `3/10` cases and MR-D-BSCP
  pTV-loss `w=0.055` in `3/10` cases. It beat PL1D in `5/10` cases, improving
  PL native-reg aggregate TRE only slightly (`0.9758` vs `0.9815 mm`) and not
  closing the gap to pTVReg/MR-D-BSCP.
- Runtime: PL1A total wall time averaged `359.2 +/- 10.8 s`.
- Known limitation: this is not the author A1 row, because A1 used author
  five-lobe crop/sampling; this row isolates schedule while retaining the P27
  fixed-lung-mask pTVReg 1 mm protocol.

### P29: LungCT DUAL-INR Camera-Ready TRE/stdlogJ Curve Candidates

- Status: Stage 0 implementation, Stage 1 PL1C default, compact Stage 2
  sweep, and Stage 3 PL1A local refinement completed 2026-08-11.
- Purpose: add a small number of defensible DUAL-INR-family curve candidates
  to the camera-ready DIR-Lab TRE versus stdlogJ figure without running an
  unnecessarily broad hyperparameter sweep.
- Dataset: DIR-Lab 4DCT only.
- Cases: all cases `01-10` for every curve point used for figure selection.
- Direction: T50 moving image to T00 fixed image.
- Protocol tag:
  `ptv_landmark_crop_1mm_fixed_t00_lungmask_dual_inr_camera_ready_curve`.
- Image/mask contract: pTVReg 1 mm isotropic image domain; unmasked image
  intensities; fixed T00 1 mm lung mask for sampling and metric masks; no
  moving-mask optimization; no semantic/lobe/segmentation loss.
- Candidate variants:
  - `PL1D`: image pyramid `[8,4,2,1]`, native DUAL regularization.
  - `PL1B`: image pyramid `[8,4,2,1]`, IDIR-type bending energy.
  - `PL1A`: original blur `[4,2,0]`, native DUAL regularization.
  - `PL1C`: original blur `[4,2,0]`, sampled pTVReg-style LCC plus sampled
    physical-unit isoTV regularization.
- Compact sweep plan:
  - Native DUAL and BE variants: multipliers `[0.01,0.1,1,10,100]`.
  - `PL1C`: pTVReg-style weights `[0.0055,0.017,0.055,0.17,0.55]`.
  - Existing `PL1D`, `PL1B`, and `PL1A` multiplier-1 rows may be reused only
    if protocol, schedule, loss, weights, and posthoc regularity match.
- Stage 3 refinement plan: add `PL1A` local multipliers `[1/3,3]` on all
  cases `01-10`, using the same PL1A protocol as Stage 2. For final
  paper-style overlays, omit the coarse `PL1A m=100` point and compare the
  refined PL1A curve against existing LungCT paper rows.
- Stage 0 output:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage0_implementation_and_manifest_report.md`.
- Stage 1 result: `PL1C` weight `0.055` completed `10/10` accepted records
  and `10/10` posthoc regularity records; native snap TRE
  `0.9909 +/- 0.1727 mm`, registration-grid subvoxel TRE
  `1.1011 +/- 0.1303 mm`, foreground SDlogJ `0.2099 +/- 0.0599`,
  foreground `J<=0` percentage `0.000119 +/- 0.000294`, total wall time
  `119.0 +/- 5.9 s`.
- Stage 1 output root:
  `/database/inrorspline/runs/lungct_dual_inr_camera_ready_curve_stage1_pl1c_default_v1`.
- Stage 1 review root:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/`.
- Stage 1 report:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage1_pl1c_default_report.md`.
- Stage 2 output root:
  `/database/inrorspline/runs/lungct_dual_inr_camera_ready_curve_stage2_compact_v1`.
- Stage 3 output root:
  `/database/inrorspline/runs/lungct_dual_inr_camera_ready_curve_stage3_pl1a_refinement_v1`.
- Stage 2 review outputs:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage2_compact_all_case_sweep_report.md`
  and
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage2_coarse_tre_stdlogj_curve.png`.
- Stage 3 planned review outputs:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage3_pl1a_refinement_report.md`
  and final paper-style overlays under
  `experiments/paper_artifacts/pareto_figure/`.
- Stage 3 result: `PL1A m=1/3` completed `10/10` accepted records with
  native snap TRE `0.977 +/- 0.165 mm` and foreground stdLogJ
  `0.1660 +/- 0.0431`; `PL1A m=3` completed `10/10` accepted records with
  native snap TRE `1.006 +/- 0.182 mm` and foreground stdLogJ
  `0.1313 +/- 0.0350`. Both had `10/10` posthoc regularity records and zero
  process retries.
- Final PL1A paper-style overlay candidates omit the coarse `PL1A m=100`
  point and include Stage 3 `m=1/3` and `m=3`:
  `experiments/paper_artifacts/pareto_figure/v006_candidate_pl1a_refined_full_context/`
  for the full previous LungCT context, and
  `experiments/paper_artifacts/pareto_figure/v007_candidate_pl1a_refined_mrdbscp_ptvreg_focus/`
  for the focused MR-D-BSCP/PL1A/pTVReg comparison.
- Stage 2 execution policy: one variant per GPU, with `PL1D`, `PL1B`,
  `PL1A`, and `PL1C` pinned to four GPUs; every non-default independent job is
  attempted, and default points are imported from P27/P28/P29 Stage 1 when
  their metadata matches.
- Stage 1 command: use
  `scripts.experiments.lungct_dual_inr_controlled.launch_lungmask_revision`
  with `--stage full --variants PL1C`, all four GPUs, pTV root
  `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`, native raw root
  `/database/datasets/lungCT/DIR-Lab/raw/4DCT`, native mask root
  `/database/datasets/lungCT/DIR-Lab/processed/lungmask_R231_native`.
- Primary metrics: all-case native snap TRE, registration-grid subvoxel TRE,
  foreground stdlogJ, foreground nonpositive-Jacobian percentage, per-case TRE,
  runtime, and peak memory.
- Acceptance: technical only; require expected artifacts, finite metrics,
  protocol `PL`, fixed T00 lung-mask sampling and metric rules, exact schedule
  identity, exact loss/weight metadata, and completed posthoc regularity. TRE
  and stdlogJ magnitude are selection metrics, not completion gates.
- Known limitation: `PL1C` uses sampled pTVReg-style LCC/isoTV on a DUAL SIREN
  displacement and original DUAL blur schedule; it is a loss-control curve
  candidate, not a reproduced pTVReg or MR-D-BSCP architecture.

### P30: LungCT Case01/Case08 Masked-NCC FD-BE Quick A/B

- Status: completed 2026-08-14; all six masked jobs finished and the matched
  masked-vs-unmasked aggregate was generated.
- Purpose: test whether replacing full-crop global NCC with fixed T00
  lung-mask global NCC changes the LungCT case01/case08 behavior for the
  FD-BE family, while keeping the dense FD-BE regularizer matched to the
  previous unmasked runs.
- Dataset: DIR-Lab 4DCT only.
- Cases: selected cases `01` and `08`.
- Direction: T50/end-exhale moving image to T00/end-inhale fixed image.
- Data root:
  `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`.
- Output root:
  `/database/inrorspline/runs/lungct_case01_case08_masked_ncc_fdbe_quick_v1`.
- Review root:
  `experiments/2026-08-14_lungct_case01_case08_masked_ncc_fdbe_quick/`.
- Comparison source:
  `/database/inrorspline/runs/dirlab_case01_case08_panelA_lungmask_diag_curves_v1`
  and
  `experiments/2026-08-14_dirlab_case01_case08_lungmask_diagnostics/lungmask_before_after_curves_summary.csv`.
- New method IDs:
  - `sinr_cps4_masked_ncc_fd_be`
  - `direct_cp_cps4_masked_ncc_fd_be`
  - `direct_cp_32_16_8_4_masked_ncc_fd_be`
- Hyperparameters:
  - SINR cps4: alpha `10000`, lr `1e-4`, epochs `2500`, seed `1`.
  - D-BSCP cps4: alpha `10000`, lr `1e-3`, epochs `2500`, seed `1`.
  - MR-D-BSCP: alpha `10000`, lr `1e-3`, cps `32,16,8,4`,
    downsample `8,4,2,1`, epochs `400,400,800,900`, seed `1`.
- GPU assignment: one method queue per GPU, with SINR on GPU `1`, D-BSCP on
  GPU `2`, and MR-D-BSCP on GPU `3`; each queue runs case01 then case08.
- Protocol tag pattern:
  `ptv_landmark_crop_1mm_with_lungmask+...+fixed_t00_lung_mask_global_ncc_dense_fd_be`.
- Acceptance: py-compile passes, three one-epoch smoke runs pass, launcher
  dry-run reports exactly six jobs, GPU preflight returns within timeout and
  assigned GPUs have the configured free-memory margin, all six full jobs
  complete, and the aggregate side-by-side report is generated.
- Verification/results:
  - Py-compile passed for the modified runner, launcher, and aggregator.
  - One-epoch Docker smoke runs passed for SINR cps4, D-BSCP cps4, and
    MR-D-BSCP `32-16-8-4` on case01.
  - Launcher dry-run reported six planned jobs; full launch completed all six
    jobs with no failed records.
  - Aggregate outputs:
    `experiments/2026-08-14_lungct_case01_case08_masked_ncc_fdbe_quick/masked_vs_unmasked_summary.csv`,
    `experiments/2026-08-14_lungct_case01_case08_masked_ncc_fdbe_quick/masked_vs_unmasked_long.csv`,
    and
    `experiments/2026-08-14_lungct_case01_case08_masked_ncc_fdbe_quick/masked_vs_unmasked_report.md`.
  - Main quick-check observation: fixed-T00 lung-mask NCC improves the
    fixed-mask `1-NCC` diagnostic for all six rows, but usually worsens TRE and
    foreground regularity at the reused unmasked hyperparameters. The only TRE
    exception is case08 MR-D-BSCP, which changes from `1.087` mm to `1.054` mm
    while stdLogJ increases from `0.1922` to `0.3518`.
- Known limitation: this isolates the NCC data-loss domain within the FD-BE
  family. It is not a native-IDIR equivalence claim because native IDIR differs
  in image domain, model, sampling, and analytic/autograd bending-energy
  implementation.

### P31: Learn2Breath-style P2e 1.5 mm Two-Window All-Case Audit

- Status: complete. All 40 preflights and registrations passed, including the
  four cold production-shaped gates before warm release; all four workers
  returned zero.
- Purpose: port the selected historical COPD DirectCP / MR-D-BSCP P2e
  mechanics to the shared `224 x 192 x 224`, 1.5-mm challenge-like pair-local
  grid, and report the locked CT-only two-window audit across COPD and
  Learn2Breath validation cohorts without selecting a window.
- Review artifact: `experiments/2026-08-30_learn2breath_copd10_pairlocal_coarse_registration/`.
- Method: historical pTV local-CC plus physical isoTV with CP stages
  `64,32,16,8,4`, image stages `16,8,4,2,1`, epochs
  `400,400,400,400,900`, Adam `1e-3`, and seed `1`. The proposed 32x extension
  is not part of this run.
- Protocol tag: `challenge_like_pairlocal_1p5mm_fullgrid+historical_p2e_raw80_900_from_HUplus1000+ct_only`.
- Direction: phase-2 EXP moving to phase-1 INSP fixed.
- Field contract: fixed-grid `float32 [3,X,Y,Z]`, XYZ voxel components,
  `moving(fixed + u)` sampling semantics.
- Metrics: 300-landmark physical trilinear COPD TRE and per-landmark CSV;
  validation per-lobe Dice and mean Dice; challenge whole-grid/fixed-lobe
  folding, runtime, and cohort-appropriate QC-only coronal audits.
- Acceptance: preflight verifies geometry/affines, HU occupancy, landmark
  IDs/bounds and baseline TRE, R231 grid alignment, zero/known-shift field
  handling, and output reload. The COPD8 inline 2x9 audit was reviewed before
  the all-case rollout; each cohort/profile cold result must have finite fields
  and metrics before its warm queue continues.
- Correction: historical P2e applied `[80,900]` to raw DIR-Lab stored values.
  The pair-local manifest declares `HU = stored - 1000`; consequently the
  valid port uses `stored = HU + 1000` before the historical window. The first
  run instead windowed HU directly, mapped 90.6% of fixed voxels to zero, and
  produced `27.1577 mm` physical trilinear TRE versus `26.4270 mm` at identity.
  Those artifacts are retained as superseded diagnostics only. No lobe label,
  R231 mask, or landmark enters the objective.
- Corrected COPD8 result: after applying the historical window to
  `stored = HU + 1000`, physical trilinear TRE was `1.7129 mm` mean (median
  `0.9413`, p95 `8.8858`) versus `26.4270 mm` at identity across 300
  landmarks. Whole-grid folding was `0.22037%`; the executed QC notebook
  follows the accepted mask-audit coronal orientation. This historical
  single-case result is retained as provenance and was superseded for uniform
  timing/output metadata by the authorized all-case rerun.
- Rollout: execute each cohort under both `p2e_equiv_hu` `[-920,-100]` and
  `l2b_lung_hu` `[-1000,200]` profiles. GPU 0 owns COPD/P2e-equivalent, GPU 1
  COPD/Learn2Breath-lung, GPU 2 validation/P2e-equivalent, and GPU 3
  validation/Learn2Breath-lung. Each persistent worker records one cold and
  nine warm runs. Record synchronized optimizer-only time, full
  load-to-evaluation time, worker totals, and four-worker makespan. Report both
  profiles without selecting a winner; all lobe labels remain evaluation/QC-only.
- Completed results: COPD mean TRE was `1.4683 mm` with `0.2444%` challenge
  whole-grid folding for `p2e_equiv_hu`, and `1.7598 mm` with `0.7905%`
  folding for `l2b_lung_hu`. Validation mean lobe Dice was `0.9724` with
  `0.2767%` whole-grid folding for `p2e_equiv_hu`, and `0.9700` with `0.5832%`
  folding for `l2b_lung_hu`; per-lobe Dice remains in the external per-case
  record. These are reported-only values, not a profile-selection result.
- Runtime: cold full-case times were `394.7`, `411.7`, `446.7`, and `446.6 s`
  in GPU order 0--3; warm full-case means +/- SD were `311.7 +/- 121.4`,
  `268.5 +/- 96.0`, `327.1 +/- 132.8`, and `326.3 +/- 134.1 s`. Persistent
  worker totals were `3256.7`, `2867.3`, `3392.9`, and `3386.2 s`; measured
  four-GPU makespan was `3395.8 s`.
- QC: both all-case Docker notebooks executed with 20 inline 2x9 panels and
  no cell errors. COPD uses R231 masks and validation uses challenge-sampler
  lobe masks in the accepted coronal orientation.

### P32: Learn2Breath-style FireANTs Greedy CT-only Two-Window Audit

- Status: all-40 preflight passed and a recovery-environment production relaunch
  is underway. P31 P2e records are locked, read-only provenance and will not
  be rerun.
- Purpose: run the CT-only FireANTs Greedy local-NCC method under the same two
  physical-HU profile names and shared 1.5-mm grids as P31, then provide
  detailed case-by-case COPD TRE/folding and validation per-lobe Dice/folding
  tables for both methods and both profiles.
- Dataset / direction: challenge-style COPD `case_104` through `case_113` and
  Learn2Breath validation `NLST_0001` through `NLST_0010`; phase-2 EXP moving
  to phase-1 INSP fixed.
- Protocol tag:
  `challenge_like_pairlocal_1p5mm_fullgrid+fireants_greedy_local_ncc_ct_only+two_physical_hu_windows`.
- FireANTs configuration: scalar CT only; Greedy local NCC kernel `5`, scales
  `16,8,4,2,1`, iterations `200,200,200,200,200`, Adam LR `0.2`, compositive
  deformation, warp/gradient smoothing `0.25/0.5`, tolerance `1e-4`, ten
  tolerance iterations, identity affine. No lobe label, R231 mask, or landmark
  enters the registration input or loss.
- Intensity profiles: `p2e_equiv_hu` clips physical HU `[-920,-100]` then
  applies `(HU+920)/820`; `l2b_lung_hu` clips `[-1000,200]` then applies
  `(HU+1000)/1200`. FireANTs' legacy direct `[80,900]` HU normalizer is
  excluded.
- Field / metric contract: export fixed-grid `float32 [3,X,Y,Z]`, XYZ voxel
  components with `moving(fixed + u)` semantics. FireANTs sign/axis conversion
  is selected only by native normalized warped-CT agreement, then validated by
  NIfTI reload and challenge sampling. COPD reports physical trilinear TRE;
  validation reports five per-lobe Dice values and mean Dice. Learn2Breath
  challenge whole-grid folding is headline for both cohorts; validation also
  reports fixed-lobe folding.
- GPU protocol: four persistent workers, GPU 0 COPD/P2e-equivalent, GPU 1
  COPD/Learn2Breath-lung, GPU 2 validation/P2e-equivalent, GPU 3
  validation/Learn2Breath-lung. Each worker has one cold plus nine warm cases;
  all four cold fields and metrics must pass before warm release. Record
  optimizer-only, full-case, worker-case, worker-total, peak memory, and
  four-GPU-makespan timing.
- Reporting: merge 40 new FireANTs rows with 40 immutable P31 rows into two
  detailed 40-row cohort tables, preserve every validation lobe Dice value,
  and label P2e CUDA 11.8 versus FireANTs CUDA 12.1 environments. Do not use
  these timings as a direct cross-container method-runtime ranking, select a
  winner, or create a submission.
- QC: execute all-case inline `2 x 9` accepted-orientation coronal notebooks.
  COPD displays CT/R231 QC masks; validation displays CT/discrete lobe masks.
  FireANTs uses the challenge sampler; P2e retains established historical
  sampling for COPD and challenge sampling for validation.

### P33: LungCT MR-D-BSCP 1 mm Acceleration and Cold/Warm Timing

- Run ID: `2026-09-01_lungct_mrdbscp_acceleration_best1mm`.
- Status: completed and technically accepted 2026-09-01 (`120/120` case
  registrations; six variants across 4DCT and COPD cohorts).
- Owner/agent: Codex.
- Date planned: 2026-09-01.
- Date completed: 2026-09-01.
- Dataset: best 1-mm 4DCT cases `01-10` and canonical 1-mm COPD cases `01-10`.
- Protocol: DirectCP cubic B-spline controls with pTV local CC (`sigma=2.1`),
  isoTV weight `0.055`, `csqrt=5e-3`, five-voxel border, seed `1`.
- Variants: the six frozen Learn2Breath schedules
  `original_corrected`, `frontloaded_fixed`, balanced/fast cumulative, and
  balanced/fast ordinal. Ordinal variants are independently cold-started for
  this timing campaign, as explicitly selected for direct cold/warm comparison.
- 4DCT schedule: existing anchor controls `32,16,8,4`, downsampling
  `8,4,2,1`, and original `[400,400,800,900]` 2500-epoch rerun. Accelerated
  schedules use the documented 64+32-to-32 four-stage projection.
- COPD schedule: exact five-stage L2B controls `64,32,16,8,4`, downsampling
  `16,8,4,2,1`, and variant-specific epoch/LR/stopping schedules.
- Timing: `loaded_images_to_dvf_file_sec` from loaded inputs to saved dense DVF;
  each variant/dataset cohort has case 1 cold and cases 2-10 warm in one
  persistent worker. Queue time is excluded; controls and optimizer state are
  reset for every case.
- Materialization: per-epoch stage-grid expansion is retained; non-final
  full-grid expansion is disabled and the expected trace is final-stage-only.
- Scale/result: 120 case registrations completed across GPUs `0,1,2,3`, with
  casewise TRE, ten-case mean/sample-SD TRE, and cold/warm/all-ten timing
  tables.
- Execution spec: `experiments/2026-09-01_lungct_mrdbscp_acceleration_best1mm/`.
- Preflight/compatibility record: both exact-schedule smokes completed with
  finite cohort metrics, and the immutable production manifest's 40 rows all
  passed geometry, provenance, landmark/lobe, and evaluator gates. The original
  `learn2breath26-fireants:e7a613c-cu121` image hangs on physical GPU 0 for a
  basic full-grid CUDA allocation, including in a fresh container. GPU 0 itself
  is healthy in `fireants-devhj:v1_0`; a recovery smoke using the identical
  current FireANTs source mounted into that CUDA-12.1 base completed with a
  finite field, zero folding, and COPD TRE improvement. The relaunch uses this
  recovery runtime only for GPU 0, tagged
  `fireants_current_source_on_devhj_v1_cuda12.1`; GPUs 1--3 retain the original
  image that passed their prior smokes, tagged
  `learn2breath26_fireants_e7a613c_cuda12.1`. It retains the method source and
  protocol but introduces a base-runtime asymmetry, recorded per case and
  explicitly distinguished from P31's CUDA-11.8 environment.

### P34: GPU-1 LungCT MR-D-BSCP Adaptive and Fixed-50 Study

- Run ID prefix: `lungct_mrdbscp_adaptive_fixed50_gpu1_`.
- Status: completed and technically accepted 2026-09-29. The 20-case loader
  audit, four monitored one-step smokes, 44/44 production registrations, and
  strict aggregate validation all passed.
- Owner/agent: Codex.
- Date planned: 2026-09-29.
- Dataset: canonical pTVReg-style isotropic-1-mm 4DCT cases 01-10 and COPD
  cases 01-10.
- Purpose: compare the best prior adaptive schedule for each cohort with the
  matched-LR fixed-50-per-stage acceleration stress test while preserving the
  pTV local-CC plus isoTV DirectCP backend.
- Scale: four fresh persistent GPU-1 arms and 44 primary registrations. Each
  arm runs case01 cold, case01 warm repeat, then cases02-10 warm. The 40 warm
  fields form the quality cohort; four cold fields are retained for QA.
- GPU contract: physical GPU 1 only, three-sample idle gate, approximately
  1-Hz plus case-boundary NVML monitoring, no foreign-process intervention,
  and complete-arm restart after contamination.
- Timing: synchronized loaded-input to serialized-DVF endpoint, plus
  optimizer-only, in-memory, peak-memory, stage-step, and case01 cold/warm
  diagnostics.
- Comparison policy: fixed-50 is compared directly with same-dataset adaptive.
  P33 is historical quality/reproducibility context only; no matched speedup
  against the old 2,500-step timings is permitted.
- Accepted timing: warm input-to-serialized-DVF mean +/- sample SD was
  `57.097 +/- 33.319 s` adaptive versus `48.690 +/- 30.222 s` fixed-50 for
  4DCT (`1.17x`), and `51.332 +/- 46.893 s` versus
  `36.228 +/- 37.620 s` for COPD (`1.42x`).
- Subsequent P35/P36 diagnosis established that these process-warm timings
  include large new-shape cuDNN benchmarking costs and are not steady-state
  per-case runtime. P34 remains the accepted T/F shape-first record; use P36
  for the backend-fixed 1-cold plus 10-warm timing.
- Accepted headline quality: warm native-crop snap TRE was
  `0.9209 +/- 0.1587 mm` adaptive versus `0.9226 +/- 0.1571 mm` fixed-50 for
  4DCT, and `1.0054 +/- 0.6673 mm` versus `1.0469 +/- 0.6970 mm` for COPD.
- Integrity: no production retries or contaminated accepted samples; maximum
  accepted-arm monitor gaps were 1.199-1.200 seconds. All 44 DVFs were finite
  and all landmark metrics accounted for 300 landmarks per case.
- Execution spec:
  `experiments/2026-09-29_lungct_mrdbscp_adaptive_fixed50_gpu1/`.

### P35: GPU-1 LungCT Adjacent Shape-Pair Cache Diagnostic

- Run ID prefix: `lungct_mrdbscp_shape_pair_cache_diagnostic_gpu1_`.
- Status: completed and technically accepted 2026-09-29. The 80-row dry run,
  20-case loader audit, four monitored GPU-1 smokes, 80/80 production fields,
  and 40/40 adjacent pairs passed.
- Owner/agent: Codex.
- Date planned: 2026-09-29.
- Purpose: diagnose the P34 shape-dependent timing pattern by running every
  4DCT and COPD case twice consecutively within each adaptive/fixed-50 process.
- Scale: four physical-GPU-1 arms, 20 registrations per arm, 80 total fields,
  and 40 adjacent identical-matrix pairs.
- Interpretation: fixed-50 is primary because paired optimizer steps are
  identical. Adaptive paired timings must report step-count changes. Adjacent
  repetition diagnoses established per-shape/process state but does not alone
  identify a cache or kernel-selection mechanism.
- Accepted timing: shape-first versus immediate-repeat mean was
  `68.525/8.670 s` for 4DCT adaptive (`7.90x`) and `76.902/3.674 s` for 4DCT
  fixed-50 (`20.93x`). COPD was `70.924/11.157 s` adaptive (`6.36x`) and
  `62.669/4.318 s` fixed-50 (`14.51x`). Fixed-50 paired step counts were
  identical; adaptive steps changed in 3/10 4DCT and 7/10 COPD pairs.
- Relationship: voxel count correlated weakly with shape-first time
  (`r=0.258-0.338`), while adaptive and fixed-50 shape-first case patterns
  were nearly identical (`r=0.991` 4DCT, `r=0.999` COPD). Immediate-repeat
  peak allocation was about 2.87x lower. This establishes a same-shape
  process/backend-state effect, not a simple matrix-size law.
- Integrity: zero accepted-arm bad monitor samples; maximum accepted gaps were
  1.199-1.202 seconds. Six retained parse-only attempts preceded the accepted
  4DCT adaptive production attempt; no CUDA registration ran in those failed
  attempts, and no accepted production arm was contaminated.
- Next decision: controlled LungCT fixed-50 A/B of cuDNN benchmark/determinism
  flags, followed by per-stage/operator tracing only if the backend toggle
  does not remove the first-shape cost. Existing OASIS evidence makes
  `benchmark=False` the leading remedy, but this diagnostic alone does not
  identify cuDNN as the sole mechanism.
- Execution spec:
  `experiments/2026-09-29_lungct_shape_pair_cache_diagnostic_gpu1/`.

### P36: GPU-1 LungCT cuDNN F/T Cold/Warm Rerun

- Run ID prefix: `lungct_mrdbscp_cudnn_ft_gpu1_`.
- Status: completed and technically accepted 2026-09-29. The 44-row dry run,
  20-case loader audit, focused CPU contracts, four monitored F/T smokes, and
  44/44 production fields passed with zero validation errors.
- Owner/agent: Codex.
- Date planned: 2026-09-29.
- Purpose: rerun P34's four 1-cold plus 10-warm arms with
  `cudnn.benchmark=False, deterministic=True`, then compare casewise and
  aggregate timing with the original P34 T/F warm cohort and P35 T/F adjacent
  same-shape repeats.
- Scale: four physical-GPU-1 arms and 44 new registrations. Schedules,
  objective, data, timing endpoint, seed, and process layout remain matched to
  P34; only the requested cuDNN flags change.
- Interpretation: fixed-50 is primary because it executes identical work.
  Adaptive step mismatches are explicit. P35 immediate repeats estimate the
  already-tuned T/F floor; comparison across separate accepted runs remains a
  controlled diagnostic rather than a universal hardware claim.
- Acceptance: all requested/effective flags must be F/T; 44 finite fields,
  40 unique warm results, clean GPU-1 monitoring, full provenance, and valid
  P34/P35 reference joins are required.
- Accepted timing: new F/T warm mean was `7.589 +/- 1.551 s` adaptive and
  `3.958 +/- 0.449 s` fixed-50 for 4DCT, and `12.482 +/- 2.182 s` adaptive and
  `4.689 +/- 0.577 s` fixed-50 for COPD.
- Fixed-50 backend result: versus P34 T/F warm timing, F/T was 12.30x faster
  for 4DCT and 7.73x faster for COPD with identical optimizer work. It was
  only 1.08x/1.09x the P35 immediate-repeat floor. Case-01 cold time fell from
  57.701 to 4.670 seconds and 101.004 to 5.615 seconds, respectively.
- Interpretation: this confirms that new-shape cuDNN benchmarking dominated
  P34's slow process-warm cases. Adaptive ratios are supporting evidence only
  because steps differed in 6/10 4DCT and 7/10 COPD comparisons versus P34.
- Integrity: every production arm was accepted on attempt 00 with zero bad
  monitor samples and maximum gaps of 1.192-1.199 seconds. All 44 summaries
  recorded effective F/T backend flags.
- Quality: fixed-50 F/T minus P34 T/F warm native-crop snap TRE was -0.0022 mm
  for 4DCT and -0.0120 mm for COPD; no material cohort regression was
  observed, although global bitwise determinism is not claimed.
- Policy outcome: F/T is now the repository default for future unspecified GPU
  registration experiments. Requested/effective flags are mandatory
  provenance, and any alternative must be an explicit protocol override.
- Execution spec:
  `experiments/2026-09-29_lungct_mrdbscp_cudnn_ft_gpu1/`.

### P37: OASIS Rank-0 MR-D-BSCP 8-4-2-1 Lambda Sweep

- Run ID: `oasis_seed3_rank000_mrdbscp8421_cps1_lambda_sweep_v1`.
- Status: completed and technically accepted 2026-10-06; 5/5 production
  registrations, finite DVFs, strict metadata validation, and deterministic
  prefix validation passed.
- Dataset: shuffled OASIS test rank 0, fixed `0333`, moving `0115`, original
  test index `158`, moving-to-fixed.
- Protocol: camera-ready adaptive cps `8,4,2` prefix with lambda `1000`, plus
  a full-resolution cps1 stage at LR `1e-3`. The cps1 lambda sweep is
  `10,100,1000,10000,100000`; cuDNN flags are F/T and seed is `1`.
- Execution: one added worker per GPU 0-3 without altering existing jobs;
  lambda 100000 queued to the first released GPU. The accepted run required
  no OOM retry.
- Result: lambda 10 won Dice and symmetric ASD within the sweep
  (`0.762269`, `0.589742`); lambda 10000 won HD95 (`2.545912`). Lambda 10 is
  the frozen overall selection. It improves Dice/ASD but worsens HD95 versus
  the original 8-4-2 result and does not beat Greedy, VFA, or SITReg.
- Integrity: all five cps8/4/2 prefixes stopped at `125/150/175` epochs with
  exactly zero terminal loss deltas across candidates.
- Execution spec:
  `experiments/2026-10-06_oasis_mrdbscp8421_lambda_sweep/`.

### P38: OASIS Rank-0 MR-D-BSCP 8-4-2 to Dense-Field Lambda Sweep

- Run ID: `oasis_seed3_rank000_mrdbscp842_dense_lambda_sweep_v1`.
- Status: completed and technically accepted 2026-10-06; 5/5 registrations,
  finite DVFs, dense-final metadata validation, exact initialization checks,
  and deterministic prefix validation passed.
- Dataset: shuffled OASIS test rank 0, fixed `0333`, moving `0115`, original
  test index `158`, moving-to-fixed.
- Protocol: unchanged camera-ready adaptive cps8/4/2 prefix at lambda 1000,
  then direct full-resolution dense-voxel Adam refinement at LR `1e-3` using
  global NCC and unmasked forward-valid finite-difference bending energy.
  Dense lambda is swept over `10,100,1000,10000,100000`; seed is `1` and
  cuDNN flags are F/T.
- Result: lambda 100 wins Dice (`0.761925`) and is the frozen overall
  selection. Lambda 10000 wins HD95 (`2.472854`); lambda 10 wins symmetric
  ASD (`0.578115`).
- Paired result: at selected lambda 100, dense minus matched cps1 B-spline is
  `+0.003449` Dice, `-0.027524` HD95, and `-0.012393` ASD.
- Integrity: every cps2-to-dense initialization difference is exactly zero;
  all cps8/4/2 prefixes stopped at `125/150/175` with exactly zero terminal
  loss deltas across candidates. No final cps1 control representation exists.
- Execution spec:
  `experiments/2026-10-06_oasis_mrdbscp842_dense_lambda_sweep/`.

### P39: OASIS Rank-0 VFA/SITReg Independent B-Spline Approximation

- Run ID: `oasis_seed3_rank000_vfa_sitreg_bspline_cps8421_fullvoxel_mse_v1`.
- Status: completed and technically accepted 2026-10-06. All 24 independent
  candidates, eight selected fits, external reports, curves, and strict
  acceptance checks passed.
- Dataset: shuffled OASIS test rank 0, fixed `0333`, moving `0115`, original
  test index `158`, moving-to-fixed.
- Targets: immutable canonical best-Dice VFA and SITReg displacement fields,
  loaded as saved channel-first `zyx` voxel fields and converted to
  normalized-grid `xyz` for cubic B-spline expansion.
- Matrix: independent zero-initialized direct cubic B-spline fits at cps
  `8,4,2,1`, each with Adam learning rates `1e-2,3e-3,1e-3` (24 fits).
- Objective: full-volume voxel/mm MSE over all `160x224x192x3` residual
  values. No image loss, training mask, or regularizer is used. The fixed raw
  image-positive mask is secondary-metric-only.
- Schedule: maximum/minimum epochs `5000/500`, checks every 50 epochs, median
  window 50, patience 200, relative MSE threshold `1e-6`, seed 1, and cuDNN
  F/T. The lowest-loss checkpoint is evaluated and retained.
- Execution: schedule longest jobs first across physical GPUs 0-3 with at
  most one new worker per GPU; existing GPU jobs are not changed. Preserve
  failed attempts; retry one CUDA OOM only after another sweep worker exits,
  on the GPU with the most free memory.
- Selection/recovery: choose the lowest full voxel MSE per target/cps, then
  normalized-grid MSE and lower LR. A nested-space violation above 0.1%
  triggers one best-checkpoint continuation at one-tenth LR for up to 5000
  additional epochs; acceptance is forbidden if ordering remains invalid.
- Outputs: `${ARTIFACT_ROOT}/runs/oasis_seed3_rank000_vfa_sitreg_bspline_cps8421_fullvoxel_mse_v1`
  and the matching external review root.
- Primary metrics: voxel MSE/RMSE/rRMSE and residual endpoint-error summaries;
  secondary metrics include normalized-grid and fixed-positive-mask errors,
  component MSE, and canonical Dice/HD95/symmetric ASD changes from each
  source field.
- Result: selected VFA voxel MSE at cps `8/4/2/1` is
  `0.1146656573/0.0288353432/0.0053879051/1.6147702e-7`; selected SITReg MSE
  is `0.1258704364/0.0267495271/0.0037603469/1.2670101e-7`. The selected
  RMSE sequences are `0.338623/0.169810/0.073402/0.000402` and
  `0.354782/0.163553/0.061322/0.000356`, respectively.
- Selection: LR `1e-3` won every VFA cps and SITReg cps8/4/2; LR `1e-2` won
  SITReg cps1. Nested-space ordering passed directly, so no continuation was
  launched. All 24 candidates completed on attempt 0 with no OOM.
- Execution spec:
  `experiments/2026-10-06_oasis_rank000_vfa_sitreg_bspline_fit/`.

### P40: OASIS 8-4-2 to Frozen-VFA-Feature Dense Refinement

- Run ID: `oasis_seed3_first10_mrdbscp842_dense_vfa_features_v1`.
- Status: completed and technically accepted 2026-10-07 02:01 UTC (Oct6 PDT).
  All 96 rank0 fits and 18 frozen confirmation fits completed. No feature
  candidate met all three rank0 tolerances; ranks1–9 mean also missed HD95.
  Native VFA parity passed and prefix terminal scalar losses match P38 exactly.
  The accepted external reports are authoritative; no P40 workers remain.
- Dataset: seed-3 shuffled rank0 (fixed0333/moving0115) for exploratory
  selection; freeze one feature and one control configuration for ranks1–9.
- Method: unchanged camera-ready analytic-BE cps8/4/2 prefix, then direct
  dense voxel Adam with frozen VFA full-resolution projected-feature MSE/NCC
  and unmasked voxel-unit L2 diffusion. No cps1 or spline BE in refinement.
- Search: 48 primary + up to24 local tuning + up to20 fallback + four raw-NCC
  diffusion controls, capped at96 production rank0 fits. Stop expansion once
  one field meets Dice -0.001, HD95 +0.02mm and ASD +0.005mm versus VFA.
- Native checkpoint/preprocessing parity, strict source/cache hashes, dense
  initialization checks and one-step canonical-metric smokes precede search.
- Execution: one new worker per GPU0–3, UID:GID1009:1020; seed1, FP32, cuDNN
  F/T. Existing workloads remain unchanged. One preserved OOM retry permitted.
- Outputs: external run/review directories named by the run ID, including
  every candidate, frozen selection, first10 casewise/mean comparisons and
  separate technical-completion/scientific-match acceptance flags.
- Limitations: rank0 test-case tuning; feature objectives are not VFA attention;
  timing remains diagnostic under concurrent workloads. Matching is not assured.
- Execution spec:
  `experiments/2026-10-06_oasis_mrdbscp842_vfa_feature_dense/`.

### P41: First10 Raw-NCC/Diffusion Lambda Boundary Extension

- Run ID: `oasis_seed3_first10_mrdbscp842_dense_raw_ncc_diffusion_lambda_extension_v1`.
- Status: completed and technically accepted 2026-10-07 19:49 PDT
  (2026-10-08 02:49 UTC). All 30 production fits and one full-volume smoke
  passed; 20 CPU contracts passed, with zero failed attempts or OOM retries.
- Matrix: 30 fresh fits, seed3 shuffled ranks0–9 x lambda100/300/1000; reuse
  P40's immutable lambda30 results and verified cached camera-ready8/4/2 prefixes.
- Only lambda changes: full-image normalized global NCC, total voxel-unit L2
  diffusion, LR0.3, max/min2000/300, patience100, check/window25, two LR halvings,
  lowest-total-loss checkpoint, seed1, FP32, cuDNN F/T, border sampling.
- Execution: UID:GID1009:1020, GPUs0–3, one new worker/GPU; no existing job changes.
- Reports: all casewise metrics, paired VFA/lambda30 deltas, ranks0–9/1–9 means,
  folding/diffusion and stopping/runtime diagnostics; select mean Dice then ASD/HD95.
- Result: first10 Dice/HD95/ASD means are `0.826171/1.892240/0.370711`
  at lambda100, `0.800950/1.986010/0.436688` at lambda300, and
  `0.748155/2.251212/0.595842` at lambda1000. Prior lambda30 remains
  `0.830631/1.929211/0.366981`: best Dice/ASD and overall Dice-first selection.
- Lambda100 wins HD95 among the four raw-NCC settings and lowers mean
  foreground folding from2.925658% to1.304788%. It matches all three VFA
  tolerances on6/10 cases versus4/10 at lambda30, but its cohort mean still
  fails the HD95 tolerance. No raw-NCC lambda meets all three mean tolerances.
- Integrity: all 30 fields are finite and correctly shaped; cached-prefix and
  source hashes passed. Independent recomputation verified all14 mean rows,
  exact prior lambda30 mean preservation and unchanged prior artifact hashes.
- Limitation: all first10 now exploratory tuning, not held-out confirmation.
  This separately approved extension does not rewrite or expand P40's artifacts.
- Execution spec:
  `experiments/2026-10-07_oasis_raw_ncc_diffusion_lambda_extension/`.

### P43: First10 Six-Way DL NCC9/Diffusion Sweep

- Status: production running since 2026-10-08 14:04 UTC, after 39 passing
  OASIS unit tests (16 new) and six successful full-geometry GPU smokes;
  aggregation pending. Scheduler and four distinct GPU workers verified alive.
- Variant: `oasis_seed3_first10_mrdbscp_dl_ncc9_diffusion_6way_v1`.
- Matrix: cps8/4/2, cps8/4/2/1 and cps8/4/2-to-dense, each with pyramid
  versus full-resolution loss evaluation; lambda0.01/0.1/0.3/1/3/10/100 over
  seed3 shuffled ranks0-9. 140 fresh shared prefixes yield420 outputs.
- Exact training NCC_vfa(window9,1-mean squared correlation) plus unmasked
  total-field voxel-unit finite-difference diffusion; raw stored intensities,
  border image sampling. Normalized parameters at every stage with
  LR0.03/0.01/0.003/0.001; lambda applied at every stage.
- Approved audit corrections: endpoint-aligned continuous spline evaluation
  in the pyramid, state-indexed loss histories, best-total-loss checkpoint
  including initialization at every stage, explicit backend/source/cache
  invariants. Historical runners and artifacts remain immutable.
- Execution: UID:GID1009:1020, GPUs0-3, one added worker/GPU, existing jobs
  unchanged; seed1, FP32, cuDNN F/T, TF32off. Native parity and six full-geometry
  smokes gate production. Reports require420 successful finite outputs.
- Original VFA/SITReg references; Dice/HD95/ASD and foreground SD(log J),
  folding/diffusion, marginal sample-SD ellipses, paired and mean reports.
  All ten cases are exploratory; timing and pipeline budgets are not controlled.
- Audit/spec/log: `experiments/2026-10-08_oasis_dl_ncc9_diffusion_sixway/`.

### P44: Full-Resolution Native DL-Loss Dense Sweep

- Status: complete and technically accepted2026-10-09 07:50UTC
  (00:50PDT); all350 new fits succeeded without production retries/failures.
  Preflight included49 passing tests and five native-parity/full-volume smokes.
- Variant: `oasis_seed3_first10_mrdbscp842_dense_fullres_dl_loss_sweep_v1`.
- 350 fresh registrations: MSE/MIND/RC/NGF/Feature-MSE × seven weights × first10.
  Full-resolution cps8→4→2→dense, normalized LR.03/.01/.003/.001 throughout.
- DL-centered effective lambda .005/.04/.1/.2/.2 respectively, multiplied by
  .01, .1, 1/3, 1, 3, 10, 100; native similarity scales1/25/10/1/1 preserved.
- Frozen FEDD-A Feature-MSE encodes warped images; not prior VFA features.
- Five full-volume native-parity smokes gate four-GPU UID:GID1009:1020 execution;
  no existing jobs/outputs changed. Complete-only350 output acceptance.
- Runtime dependencies isolated in artifacts; initial missing-dependency failures
  preserved. Active run/review subdirectory: `verified_environment_v1`.
- References:70 immutable P43 full-resolution dense NCC fits plus20 VFA/SITReg.
  Canonical accuracy, foreground regularity, sample-SD ARC/folding plots.
- Requested six-loss line-only Dice/HD95/symmetric-ASD ARC replot uses all42
  lambda cohorts plus two original-DL points, with foreground SD(log J) on x;
  no ellipses, error bars or smoothing. Outputs: `verified_environment_v1/lines_only_v1`.
- All first10 exploratory; heterogeneous DL anchors and shared-workload timings
  limit interpretation. Spec/log: `experiments/2026-10-08_oasis_fullres_dl_loss_sweep/`.

### P42: First10 Raw-NCC vs VFA/SITReg ARC Report

- Status: completed and technically accepted; no registration optimization.
- Report ID: `oasis_seed3_first10_raw_ncc_vfa_sitreg_arc_v1`.
- Scope: canonical first10 original VFA/SITReg metrics and common foreground
  Jacobian statistics; compare accepted raw-NCC lambda30/100/300/1000 cohorts.
- Main plot: three accuracy metrics versus foreground SD(log J), means and
  axis-aligned +/-1 sample-SD ellipses across10 cases, not confidence intervals.
  Companion folding trade-off, CSV/JSON and per-case report are included.
- Original fields and previous outputs are read-only. All first10 remain
  exploratory tuning; no GPU jobs are stopped or modified.
- Results: original SITReg first10 Dice/HD95/ASD mean +/- sample SD is
  `0.821546 +/- 0.024232`, `1.715898 +/- 0.273759`,
  `0.350896 +/- 0.075233`. Mean foreground SD(log J) is0.569473 and
  nonpositive-Jacobian percentage is0.000423%.
- Verified20 baseline evaluations and40 reused raw-NCC rows, exact VFA
  agreement with P40, complete paired cohorts and independently recomputed
  all12 mean/SD rows. Four CPU contracts and rank0 smoke passed. PNG/SVG/PDF
  figures and casewise CSV/JSON are in the distinct report directory.
- Execution spec: `experiments/2026-10-07_oasis_raw_ncc_first10_arc/`.

### P45: VFA and SITReg Training Lambda Audit

- Status:audit and sequential GPU3 memory smokes complete; full training
  and proposed eight-run lambda sweep NOT authorized yet.
- Recover lambda1 protocol; propose fresh lambda0.1/1over3/3/10 for each
  model,200 epochs, LR1e-4, Adam AMSGrad, batch1, seed42, NCC9 and diffusion.
  Preserve model-specific bidirectional updates and historical cuDNN backend.
- OASIS300 training subjects/30 validation pairs/200 test pairs, disjoint.
- Six-pair full-volume forward/backward/Adam smokes per network, VFA then
  SITReg on GPU3 only, UID:GID1009:1020. No saved training weights or changes
  to existing jobs/data/reference checkpoints.
- Artifact key:`oasis_vfa_sitreg_training_lambda_audit_v1`.
- Measured allocated/reserved peaks:VFA36.42/39.55GiB; SITReg startup
  33.34/35.94GiB and trained phase25.79/28.71GiB. No OOM. Both checkpoints
  loaded strictly, finite gradients and Adam state verified, source weights
  unchanged. Missing pystrum resolved in isolated experiment dependencies.
- Spec/log:`experiments/2026-10-09_oasis_dl_training_lambda_audit/`.

### P46: Four-GPU SITReg Training Lambda Sweep

- Status:production launched2026-10-09 14:29:43UTC (07:29:43PDT), after
  six CPU tests, numerical/metric reference reproduction, full30-pair
  validation reproduction and four native Adam/checkpoint-roundtrip smokes.
- Four fresh SITReg runs at lambda0.1/1over3/3/10 on GPUs0/1/2/3 concurrently.
  No VFA or new lambda1 training. Preserve P45's native OASIS300/30 data,
  NCC9/diffusion, seed42, Adam AMSGrad LR1e-4, batch1,200 epochs and50 joint
  bidirectional updates/epoch; persistent data iterator and historical backend.
- Strict checkpoint/saved-field parity and native update/serialization gates
  precede launch. Tight memory on GPUs0/1 explicitly documented; per-worker
  allocator caps, preserved failures and no automatic protocol changes.
- Best validation-Dice and latest checkpoints; first10 canonical test export
  after training. Existing jobs and historical artifacts remain unchanged.
- Artifact key:`oasis_sitreg_ncc9_diffusion_training_lambda_sweep_v1`.
- Spec/log:`experiments/2026-10-09_oasis_sitreg_training_lambda_sweep/`.

### P47: Rank4 CPS16 NCC9/Diffusion Rescue

- Status:complete and technically accepted2026-10-09 21:16:58UTC after24 CPU
  tests, two GPU smokes and two production runs; no failures/retries. All
  workers ran sequentially onGPU3 only, existing jobs unchanged.
- Two fresh16/8/4/2/dense registrations, rank4 fixed0084/moving0108,
  lambda1 throughout. Pyramid factors8/4/2/1/1 first, full1/1/1/1/1 second.
- Preserve P43 numerical helpers, normalized LR.03/.03/.01/.003/.001,
  evaluation-grid voxel diffusion, native NCC9 and best-loss adaptive Adam.
- Native stage fields, all35 label metrics and paired accuracy/regularity
  differences against saved full/pyramid842-dense and original VFA/SITReg.
- GPU3 UID:GID1009:1020, CPU/smoke gates, fresh memory checks and explicit
  source/input/reference hashes; complete-only two-fit report.
- Exploratory one-case rescue, extra compute, no first10 improvement claim.
- Pyramid Dice/HD95/ASD0.754233/2.728347/0.588810, folding3.692091%;
  full0.755956/2.762280/0.601316, folding3.496063%. Small/mixed improvements
  versus matched8-start baselines, not a rescue of the rank4 DL accuracy gap.
- Artifact key:`oasis_seed3_rank004_cps16842_dense_ncc9_lambda1_v1`.
- Spec/log:`experiments/2026-10-09_oasis_rank004_cps16_ncc/`.

### P48: OASIS First100 Pyramid 8→4→2→Dense NCC9/Diffusion Sweep

- Status: accepted2026-10-10 08:37:26UTC after CPU-only report recovery.
  All400 dense candidates,400 matching prefixes and200 reference evaluations
  passed finite/hash/identity/pairing checks;23 tests and both serial/spawn
  metric parity checks passed. Original launch/numerical provenance and GPU
  outputs remained unchanged. CSV/JSON/Markdown agree, including sample SD
  and leave-rank4-out summaries; acceptance was written last.
- Scope: shuffled seed-3 OASIS ranks0–99, pyramid8→4→2→dense, λ0.1/0.3/1/3
  shared across stages; compare with the matching original best-Dice VFA and
  SITReg fields using the canonical evaluator.
- 400 distinct dense candidates; reuse the already accepted P43 first10
  pyramid cps8/4/2 prefixes and dense outputs only when source/protocol/case/
  output hashes validate exactly. Run new fits for ranks10–99.
- Optimizer/loss/backend frozen to P43: native NCC_vfa9 plus unmasked total
  voxel-field L2 diffusion; normalized LR0.03/0.01/0.003/0.001; seed1,
  cuDNN F/T, TF32 off, P43 adaptive checks and best-loss checkpoints.
- Placement: four added workers using physical GPU slots0/2/3/3. Existing jobs
  remain unchanged. Require4GiB free per worker, including an8GiB reservation
  for concurrent GPU3 workers; memory rechecked before each launch.
- Original VFA/SITReg fields exist for all200 shuffled test pairs; first100
  rows are evaluated/reused with the same canonical metrics. Report per-case
  and mean/sample-SD Dice/HD95/ASD, foreground regularity, paired baselines and
  leave-rank4-out sensitivity. Runtime remains diagnostic under active jobs.
- Run/report artifact key:
  `oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1`.
- Spec/log: `experiments/2026-10-09_oasis_first100_ncc9_dense_sweep/`.

### P49: SITReg Training Sweep Detailed First100 Evaluation

- Status: all four evaluations complete,400 cases and14000 label rows.
  Revalidated2026-10-10 for the requested P48/SITReg combined report: all
  checkpoint/field hashes, canonical metric sources, case identities,
  label-to-case reconstruction and recomputed mean/sample-SD checks passed.
  No training or inference rerun was needed. Combined ten-setting export:
  `oasis_seed3_first100_mrdbscp_vfa_sitreg_sweeps_comparison_v1`.
- Evaluate P46 lambda0.1/1over3/3/10 best-validation checkpoints on seed3
  shuffled ranks0–99. Start completed lambda10; others wait for clean training
  plus first10 completion. Same released GPU slots0/1/2/3, existing jobs unchanged.
- Preserve P46 sources/checkpoints and reuse saved first10 field bytes exactly.
  Infer ranks10–99 with native raw-intensity preprocessing and historical
  benchmarkTrue/deterministicFalse backend override; UID:GID1009:1020.
- Save all35 label rows/case for Dice/HD95/pooled-area symmetric ASD and both
  directional ASDs, presence counts and missing-label exclusions. Per-case
  canonical foreground/full Jacobian statistics and diffusion, fields, warped
  labels, source/input/checkpoint hashes and complete-only mean/sample SD reports.
- CPU contracts and reused-rank0/fresh-rank10 GPU3 smoke gate the queue.
  Both70-label-row smoke cases passed; rank0 metrics exactly reproduced after
  restoring C-contiguous field layout. Explicit UTF-8 report serialization.
  Two preliminary failed smokes preserved. Inference peak14.38/16.28GiB
  allocated/reserved; queue requires24GiB free before each inference worker.
  No training changes, VFA runs, lambda1 rerun or test-based checkpoint selection.
- Artifact key: `oasis_seed3_first100_sitreg_training_lambda_eval_v1`.
- Spec: `experiments/2026-10-09_oasis_sitreg_training_lambda_sweep/first100_evaluation_spec.md`.

### P50: VFA Training Lambda Sweep With Automatic Detailed First100 Evaluation

- Status: launched2026-10-09 23:18PDT; controller PID66434. GPU0/2/3 started;
  GPU1 waits for its P49 evaluation. Nine CPU tests and native VFA preflight
  passed: exact historical field reproduction, full30 validation parity,
  four fresh lambda pair updates, serialization and35 per-label smoke rows.
  GPU0 lambda10 hit CUDA OOM during its first backward pass; failed attempt
  preserved and fresh retry queued afterGPU3 lambda3 training/evaluation.
  GPU2 lambda1/3 and GPU3 lambda3 confirmed training with finite losses.
- Final mapping:GPU0 lambda10 (explicit tight-memory attempt), GPU1 lambda0.1,
  GPU2 lambda1/3, GPU3 lambda3. Each waits for its P49 SITReg evaluation exit.
  If GPU0 lambda10 OOMs, preserve it and append one retry afterGPU3 lambda3
  training/evaluation. No existing jobs altered; no precision/batch/loss changes.
- Native bs3_vfa_v0, seed42, batch1, FP32, NCC9+voxel diffusion,200 epochs,
 50 pairs/epoch with TWO sequential Adam updates/pair, LR1e-4 AMSGrad,
 persistent iterator and native30-pair validation-Dice checkpoint selection.
- Explicit historical backend override benchmarkTrue/deterministicFalse,
  cuDNN TF32True/matmulFalse. Other training slots require42GiB free; GPU0
  memory override approved with allocator cap and one documented fallback.
- Strict reference field/metrics, full30 validation, native directional-update
  and detailed-label/serialization smokes gate production. Separate sources
  leave P46/P49 and all original fields/checkpoints unchanged.
- Automatic first100 evaluation saves3500 per-label rows/run for Dice, HD95,
  pooled-area symmetric ASD, directional ASDs and exclusions, plus canonical
  accuracy/regularity summaries, fields and full provenance. UID:GID1009:1020.
- Artifact key:`oasis_vfa_ncc9_diffusion_training_lambda_sweep_v1`.
- Spec/log:`experiments/2026-10-09_oasis_vfa_training_lambda_sweep/`.

### P51: FeDDA-1MF Training Lambda Sweep With Detailed First100 Evaluation

- Status:user-approved2026-10-10; launched02:40PDT after12 CPU tests and
  independently certified exact-field/validation/memory audit. GPU0 lambda3
  training; other three lambdas dependency-queued. Original200-epoch FeDDA-1MF,
  not the later101-epoch onepass checkpoint used for Feature-MSE.
- Placement:GPU0 lambda3 then10,including evaluation between them;GPU1
  lambda0.1 afterP50 VFA0.1 evaluation;GPU2 lambda1/3 afterP50 VFA1/3
  evaluation. GPU3 queue unchanged. Preserve all existing jobs and sources.
- Native NCC_vfa9+voxel L2 diffusion,seed42,fresh FP32 batch1,LR1e-4 AMSGrad,
  200 epochs,50 pairs/epoch,TWO sequential directional updates/pair,20000
  updates. Historical benchmarkTrue/deterministicFalse,cudnn TF32True,
  matmul TF32False override. Best native30-pair validation-Dice checkpoint.
- Strict original lambda1 checkpoint/field and validation parity,per-label
  serialization smoke and full-shape training memory smoke gate production.
  Require max(24GiB,measured reserved peak+2GiB) free memory per new worker.
  Measured maximum reserved21.22461GiB; gate24GiB. Two initial reference
  preflight failures are preserved; historical benchmark mode is not
  cross-process repeatable. Exact reproduction demonstrates adapter correctness,
  not a guarantee that every fresh process passes tight metric tolerances.
- Auto first100 evaluation:3500 per-label rows per lambda,canonical Dice/
  HD95/symmetric ASD and foreground SDlogJ/folding;CSV/JSON/Markdown and
  line-only sweep comparison against historical FeDDA1,VFA1,SITReg1.
- Run key:`oasis_fedda_ncc9_diffusion_training_lambda_sweep_v1`.
- Spec/log:`experiments/2026-10-10_oasis_fedda_training_lambda_sweep/`.
