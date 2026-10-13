---
doc_type: results_summary
status: active_draft
---

# Results Summary

This repo carries audited generated results together with planning reports.

## Strong Existing Evidence

- OASIS case000 suggests DirectCP cps4 with matched spline-style regularization can match or exceed SINR cps4, while original IDIR is weaker.
- LungCT/4DCT shows IDIR can be competitive when motion is smooth, coherent, and intrasubject.
- LungCT single-stage DirectCP cps4 fails, while multistage DirectCP improves substantially, supporting a capture-range/coarse-to-fine explanation.
- Deformation-stat reports support a brain-vs-lung regime difference: OASIS has lower displacement amplitude but more local heterogeneity; 4DCT has larger but more coherent motion.

## Not Yet Paper-Final

- Main OASIS Pareto must be rerun on seed-3 shuffled first10.
- OASIS deformation-regime stats need the same seed-3 first10 protocol.
- 4DCT matched Pareto remains incomplete.
- Baseline displacement regularity metrics need same-pipeline computation.

## Controlled DUAL DIR-Lab Baseline (2026-08-03)

- The controlled N0/N1/P0/P1 cohort completed 40/40 records on DIR-Lab 4DCT
  cases 01-10 with zero failures or retries. All runs used the same frozen DUAL
  architecture, loss, optimizer recipe, seed, and 2,500-step stage budget.
- Headline/native snap TRE (mean +/- sample SD, mm) was
  `1.0738 +/- 0.2041` for N0, `1.0649 +/- 0.2091` for N1,
  `1.0410 +/- 0.2160` for P0, and `1.0461 +/- 0.2211` for P1.
- The paired MR-shaped image-pyramid effect was small and direction-
  inconsistent: N1-N0 `-0.0089 +/- 0.0134 mm`; P1-P0
  `+0.0051 +/- 0.0147 mm`. This supports using these rows as controlled
  baselines, not claiming a robust improvement from the image schedule.
- All four controlled variants had zero nonpositive foreground Jacobian
  percentage under the shared posthoc convention. Foreground SDlogJ was
  `0.1282`, `0.1286`, `0.1135`, and `0.1138` for N0, N1, P0, and P1.
- Contextual iso-1-mm references were pTVReg `0.9191 +/- 0.1607 mm` and
  MR-D-BSCP `0.9175 +/- 0.1561 mm`. The MR-D-BSCP row uses the single
  cohort-selected `w=0.055` setting for all ten cases, never casewise minima.
- N uses native anisotropic full volumes with fixed-T00 lung-mask sampling; P
  uses a pTVReg-style isotropic-1-mm landmark crop with no lung mask in
  optimization. Only N0-N1 and P0-P1 are controlled schedule contrasts; N-P
  and external-reference comparisons retain explicit preprocessing caveats.
- Full casewise TRE, regularity, runtime, reference provenance, and acceptance
  are in
  `experiments/2026-08-02_dual_inr_dirlab_controlled_multires/camera_ready_readout.md`.

## GPU-1 LungCT Adaptive versus Fixed-50 (2026-09-29)

- The canonical pTVReg-style isotropic-1-mm 4DCT/COPD study completed and
  passed technical acceptance: 44/44 clean primary measurements, 44 finite
  case-specific fields, 40 unique warm quality fields, four cold QA fields,
  300-landmark accounting throughout, and no production retry or accepted
  GPU-contamination sample.
- Warm synchronized input-to-serialized-DVF timing was
  `57.097 +/- 33.319 s` for 4DCT adaptive and `48.690 +/- 30.222 s` for 4DCT
  fixed-50 (`1.17x` speedup). COPD was `51.332 +/- 46.893 s` adaptive and
  `36.228 +/- 37.620 s` fixed-50 (`1.42x`). These timings include final field
  materialization, CPU transfer, and serialization but exclude metric
  evaluation.
- Headline warm native-crop snap TRE was `0.9209 +/- 0.1587 mm` adaptive and
  `0.9226 +/- 0.1571 mm` fixed-50 for 4DCT (fixed-50 minus adaptive
  `+0.0017 mm`). COPD was `1.0054 +/- 0.6673 mm` and
  `1.0469 +/- 0.6970 mm`, respectively (`+0.0415 mm`). These are descriptive
  results; technical acceptance was not a quality-noninferiority test.
- Adaptive/fixed-50 mean optimizer steps were `625/200` for 4DCT and
  `837.5/250` for COPD. Foreground SDlogJ was `0.2254/0.2262` for 4DCT and
  `0.4878/0.4888` for COPD; foreground folding was `0.0406%/0.0399%` and
  `1.4640%/1.4763%`, respectively.
- P33 remains historical quality/reproducibility context only because its
  timing layout and GPU assignment differ. No matched speedup versus the old
  2,500-step runs is claimed. Full casewise timing, all four TRE conventions,
  regularity, loss, memory, stage, provenance, and acceptance records are in
  `${ARTIFACT_ROOT}/repo_experiments/2026-09-29_lungct_mrdbscp_adaptive_fixed50_gpu1/lungct_mrdbscp_adaptive_fixed50_gpu1_20260929T170843Z/`.
- P35/P36 subsequently showed that P34's `benchmark=True` process-warm cohort
  includes large new-shape autotuning costs. Retain these values as the T/F
  shape-first record, not as steady-state per-case runtime.

## GPU-1 LungCT Adjacent Shape-Pair Diagnostic (2026-09-29)

- The separate cache diagnostic passed technical acceptance with 80/80 finite
  production fields and 40/40 adjacent first/repeat pairs. All four accepted
  arm monitors had zero bad samples and maximum gaps of 1.199-1.202 seconds.
- Fixed-50 provides the clean result because both observations use identical
  step counts. Shape-first versus immediate-repeat synchronized timing was
  `76.902 +/- 39.292 s` versus `3.674 +/- 0.717 s` for 4DCT (`20.93x` ratio
  of means), and `62.669 +/- 60.096 s` versus `4.318 +/- 0.487 s` for COPD
  (`14.51x`).
- Adaptive shape-first/repeat timing was `68.525 +/- 32.267 s` versus
  `8.670 +/- 1.502 s` for 4DCT (`7.90x`) and `70.924 +/- 61.399 s` versus
  `11.157 +/- 2.064 s` for COPD (`6.36x`). These ratios are descriptive
  because stopping steps changed in 3/10 and 7/10 pairs, respectively.
- Full-volume voxel count was only weakly correlated with shape-first time
  (`r=0.258-0.338`). The casewise shape-first pattern instead matched nearly
  perfectly across adaptive and fixed-50 (`r=0.991` for 4DCT and `r=0.999`
  for COPD), and immediate-repeat peak allocation was about 2.87x lower.
- The experiment establishes a dominant persistent same-shape backend-state
  effect, not a simple matrix-size law. The Lung runner uses
  `cudnn.benchmark=True`; together with the prior controlled OASIS backend
  ablation, cuDNN shape-specific autotuning is the leading explanation, but a
  Lung-specific backend-flag A/B is still required before treating that
  mechanism as proven.
- Same-seed fixed-50 repetitions were not numerically identical with the
  current `benchmark=True, deterministic=False` setting. Maximum absolute
  native-crop subvoxel TRE drift was `0.0011 mm` for 4DCT and `0.0497 mm` for
  COPD, so the backend fix must validate field/quality reproducibility as well
  as timing.
- Full casewise matrices, timing, step counts, quality diagnostics, monitoring,
  provenance, and acceptance are in
  `${ARTIFACT_ROOT}/repo_experiments/2026-09-29_lungct_shape_pair_cache_diagnostic_gpu1/lungct_mrdbscp_shape_pair_cache_diagnostic_gpu1_20260929T211725Z/`.

## GPU-1 LungCT cuDNN F/T Cold/Warm Rerun (2026-09-29)

- The matched backend-fixed rerun passed with 44/44 finite fields, 40/40 warm
  cohort results, effective `benchmark=False, deterministic=True` on every
  row, clean GPU-1 monitoring, and valid P34/P35 reference joins.
- Fixed-50 supplies the controlled result. F/T warm timing was
  `3.958 +/- 0.449 s` for 4DCT versus P34 T/F `48.690 +/- 30.222 s`
  (`12.30x` faster), and `4.689 +/- 0.577 s` for COPD versus
  `36.228 +/- 37.620 s` (`7.73x` faster), with identical step counts.
- The F/T warm means were only 1.08x and 1.09x the P35 T/F immediate-repeat
  floors (`3.674` and `4.318 s`). Case-01 cold fixed-50 time fell from
  `57.701` to `4.670 s` for 4DCT and `101.004` to `5.615 s` for COPD.
- Adaptive F/T warm timing was `7.589 +/- 1.551 s` for 4DCT and
  `12.482 +/- 2.182 s` for COPD, but cross-run ratios remain descriptive
  because stopping steps changed casewise.
- The evidence confirms that shape-specific cuDNN benchmarking dominated the
  slow P34 process-warm cases. Future canonical LungCT timing should use F/T
  unless another backend is explicitly declared.
- Fixed-50 native-crop snap TRE changed by only `-0.0022 mm` for 4DCT and
  `-0.0120 mm` for COPD versus P34; no material aggregate quality regression
  was observed. Complete casewise timing and acceptance are in
  `${ARTIFACT_ROOT}/repo_experiments/2026-09-29_lungct_mrdbscp_cudnn_ft_gpu1/lungct_mrdbscp_cudnn_ft_gpu1_20260930T032535Z/`.

## OASIS Rank-0 MR-D-BSCP 8-4-2-1 Lambda Sweep (2026-10-06)

- Five cps1 lambda candidates completed with the camera-ready adaptive
  cps8/4/2 prefix frozen at lambda 1000. All five finite DVFs and canonical
  Dice/HD95/symmetric-ASD outputs passed strict validation.
- The cps8/4/2 prefix was bitwise-consistent at the recorded scalar level:
  every candidate stopped at `125/150/175` epochs with zero terminal loss,
  data-loss, and regularizer-loss deltas.
- Cps1 lambda 10 won Dice (`0.762269`) and symmetric ASD (`0.589742`), while
  lambda 10000 won HD95 (`2.545912`). The predefined Dice-first rule selects
  lambda 10.
- Versus the original 8-4-2 result, selected lambda 10 changes Dice from
  `0.760745` to `0.762269` and ASD from `0.592500` to `0.589742`, but HD95
  worsens from `2.652342` to `2.741775`. Greedy, VFA, and SITReg remain better
  than the selected 8-4-2-1 result on all three metrics for this one case.
- Complete metrics, stage stopping diagnostics, validation, and provenance
  are under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_rank000_mrdbscp8421_cps1_lambda_sweep_v1/`.

## OASIS Rank-0 MR-D-BSCP 8-4-2 to Dense-Field Sweep (2026-10-06)

- Five direct dense-refinement candidates completed after the unchanged
  camera-ready cps8/4/2 prefix. All finite DVFs, canonical metrics, exact
  cps2-to-dense initialization checks, and dense-final representation checks
  passed.
- Dense lambda 100 won Dice (`0.761925`) and is selected by the predefined
  Dice-first rule. Lambda 10000 won HD95 (`2.472854`), while lambda 10 won
  symmetric ASD (`0.578115`).
- At selected lambda 100, dense refinement improved over matched cps1
  B-spline refinement by `+0.003449` Dice, `-0.027524` HD95, and `-0.012393`
  ASD. It also exceeds the original 8-4-2 Dice/ASD slightly, while its HD95
  is `0.007117` higher.
- Greedy, VFA, and SITReg remain better than the selected dense candidate on
  all three metrics for this one case.
- Every independent cps8/4/2 prefix stopped at `125/150/175` epochs with zero
  terminal scalar deltas. Dense stopping epochs were 150, 150, 150, 275, and
  900 for lambda 10, 100, 1000, 10000, and 100000, respectively.
- Complete metrics, matched B-spline deltas, stopping diagnostics, GPU
  provenance, and validation are under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_rank000_mrdbscp842_dense_lambda_sweep_v1/`.

## OASIS Rank-0 VFA/SITReg B-Spline Approximation (2026-10-06)

- The 24-fit independent cubic B-spline benchmark completed and passed strict
  acceptance. It fits the immutable VFA and SITReg rank-0 fields at cps
  `8/4/2/1` with three Adam learning rates each using full-volume voxel/mm MSE
  and no image loss, mask, or regularizer.
- Selected VFA voxel MSE decreases from `0.114666` at cps8 to `0.028835` at
  cps4, `0.005388` at cps2, and `1.61e-7` at cps1. Selected SITReg MSE is
  `0.125870`, `0.026750`, `0.003760`, and `1.27e-7`, respectively. The
  predefined 0.1%-tolerance nested-space ordering passed for both targets.
- The selected cps1 approximations are effectively exact at the field level:
  RMSE is `0.000402` mm for VFA and `0.000356` mm for SITReg. Their canonical
  Dice changes from the saved targets are `+0.000013` and `-0.000002`; HD95
  is unchanged for both, and ASD changes are `-0.000040` and `+0.000005` mm.
- LR `1e-3` won all VFA rows and SITReg cps8/4/2; LR `1e-2` won SITReg cps1.
  All 24 fits completed on first attempts, no monotonicity continuation was
  needed, and runtime remains diagnostic because existing GPU workloads were
  retained.
- Complete candidate/selected tables, masked and component metrics,
  registration deltas, histories, checkpoints, curves, validation, and
  provenance are under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_rank000_vfa_sitreg_bspline_cps8421_fullvoxel_mse_v1/`.

## Camera-Ready Figure Artifacts

- Final Figure 2 for camera-ready:
  `experiments/paper_artifacts/pareto_figure/v013_final_dirlab4dct_axes_labels/`.
  Use `pareto_four_panel_final_dirlab4dct_no_annotations.png` or
  `pareto_four_panel_final_dirlab4dct_no_annotations.pdf` for the manuscript;
  `pareto_four_panel_final_dirlab4dct_annotated_review.png` is review-only.
- Figure 2 combines unchanged OASIS100 source rows from
  `v003_candidate_oasis100` with the DIR-LAB 4DCT source table from
  `v012_current_tre_stdlogj_fdbe_cps4_alpha_landscape`. The final source
  manifest validates both source matches and excludes masked-loss rows.
- All four Figure 2 panels use `Foreground SDlogJ` as the x-axis label. OASIS
  foreground means fixed raw image-positive foreground; DIR-LAB foreground
  means the fixed T00 lung/lobe foreground used by each row's documented metric
  protocol.

## OASIS First10 Raw-NCC/Diffusion Lambda Extension (2026-10-07)

- P41 completed all 30 requested fits: first10 ranks times lambda100/300/1000.
  Only lambda changed from P40's raw-NCC control; verified prefix caches,
  voxel units, normalized global NCC, diffusion, LR0.3, stopping schedule,
  border sampling and best-loss checkpoint selection remained fixed.
- First10 mean Dice/HD95/symmetric ASD at lambda30/100/300/1000:
  `0.830631/1.929211/0.366981`, `0.826171/1.892240/0.370711`,
  `0.800950/1.986010/0.436688`, `0.748155/2.251212/0.595842`.
- Lambda30 remains best for mean Dice/ASD and the overall Dice-first selection.
  Lambda100 improves mean HD95 by0.036971 mm while losing0.004460 Dice and
  increasing ASD by0.003730 mm versus30. Mean foreground folding falls from
  2.925658% to1.304788%; stronger lambdas further reduce folding but worsen
  registration accuracy substantially.
- Lambda100 meets all three VFA tolerances on6/10 cases versus4/10 for30.
  No raw-NCC setting meets all three mean tolerances; lambda100's HD95 gap
  versus VFA remains0.117899 mm (allowed0.02 mm).
- All prior fields/reports remain unchanged. All first10 cases, including
  ranks1–9, now participate in exploratory tuning; this is not a held-out
  validation result. Ranks1–9 are still separately tabulated descriptively.
- Complete per-case metrics, paired differences, means, folding/diffusion,
  stopping diagnostics and provenance are under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first10_mrdbscp842_dense_raw_ncc_diffusion_lambda_extension_v1/`.

## OASIS First10 Original SITReg/VFA ARC Comparison

- P42 re-evaluated20 original saved best-Dice VFA/SITReg fields with the same
  canonical accuracy and foreground Jacobian code as P41. VFA metrics match
  the previous report exactly; none of these baseline fields is a spline fit.
- SITReg first10 mean +/- sample SD: Dice `0.821546 +/- 0.024232`,
  HD95 `1.715898 +/- 0.273759` mm, symmetric ASD
  `0.350896 +/- 0.075233` mm. VFA means are
  `0.819772/1.774341/0.366851` respectively.
- Raw-NCC lambda30 has higher mean Dice than SITReg but worse HD95/ASD and
  much more foreground folding: `2.925658%` versus `0.000423%`. These are
  different alignment/regularity operating points, not a uniform method win.
- ARC figures compare raw-NCC lambda30/100/300/1000 against original VFA and
  SITReg using foreground SD(log J), with a separate folding companion.
  Ellipses denote marginal +/-1 across-case sample SD, not confidence regions.
- Verified60 paired case rows and12 independently reproduced mean/SD rows.
  Reports and PNG/SVG/PDF figures are under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first10_raw_ncc_vfa_sitreg_arc_v1/`.

## OASIS First10 Native DL-Loss Dense Sweep (2026-10-09)

- P44 completed350/350 new registrations with no production failures/retries;
  report accepted2026-10-09 00:50PDT. Includes70 read-only P43 full-resolution
  dense NCC fits and20 original VFA/SITReg rows: six seven-weight sweeps.
- Dice-first selected Feature-MSE lambda0.2 has mean Dice/HD95/ASD
  `0.836136/1.790880/0.337951`; selected MIND lambda0.04 has
  `0.829749/1.675762/0.335328`. Corresponding mean foreground folding is
  4.077982% and3.759377%, versus VFA0.218560% and SITReg0.000423%.
  Strong accuracy does not imply matched deformation regularity.
- User-requested line-only ARC export compares Dice, HD95 and symmetric ASD
  versus foreground positive-J SD(log J), with all six sweep curves and
  original VFA/SITReg reference points, without ellipses or error bars.
- Complete reports and the `lines_only_v1/` presentation export are under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first10_mrdbscp842_dense_fullres_dl_loss_sweep_v1/verified_environment_v1/`.
  All first10 cases remain exploratory tuning; this is not an unbiased test claim.

## OASIS Rank4 CPS16 NCC9/Diffusion Rescue (2026-10-09)

- P47 completed two approved16/8/4/2/dense lambda1 registrations sequentially
  onGPU3, after24 CPU tests and two five-stage native-geometry smokes. Preserved
  all prior outputs/numerical helpers and existing jobs; all35 labels valid.
- Full-resolution loss gives Dice/HD95/ASD0.755956/2.762280/0.601316:
  +0.001619/-0.016224/-0.009677 versus the matched8-start pipeline. Foreground
  folding increases3.402029%-to3.496063%; SDlogJ0.850896-to0.855863.
- Pyramid factors8/4/2/1/1 give0.754233/2.728347/0.588810:
  -0.001178/+0.004001/-0.013499 versus the saved4/2/1/1 image pyramid.
  Foreground folding falls3.854271%-to3.692091%; SDlogJ0.836562-to0.828301.
- Neither resolves this case's DL gap:VFA0.760943/2.216616/0.534639 and
  SITReg0.768742/2.134475/0.510531. Left-caudate Dice remains about0.45 versus
  SITReg0.71717. Adding a coarser stage yields small/mixed gains under this
  frozen schedule, not evidence of a general first10 improvement.
- New cps2-stage folding3.913655%/3.638515% decreases during dense refinement
  to3.692091%/3.496063%, so dense refinement is not the sole folding source.
- Complete CSV/JSON/Markdown, per-label differences, best/stopping epochs,
  histories, checkpoints and native stage fields are under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_rank004_cps16842_dense_ncc9_lambda1_v1/`;
  source spec/log in `experiments/2026-10-09_oasis_rank004_cps16_ncc/`.

## OASIS First100 Pyramid NCC9/Diffusion Dense Sweep (2026-10-10)

- P48 accepted400 dense candidates (100 pairs × λ0.1/0.3/1/3),400 matching
  prefixes and200 original VFA/SITReg evaluations. Forty dense first10 outputs
  are reused;360 are new. A report provenance-path failure was repaired with
  CPU-only resumable scoring, without changing numerical code or GPU outputs.
- Lambda1 is best among the four weights on all three mean accuracy metrics:
  Dice `0.833679 ± 0.026726`, HD95 `1.820084 ± 0.508698` mm and ASD
  `0.343654 ± 0.100760` mm (across-case sample SD). Original VFA is
  `0.821588/1.884796/0.370851`; SITReg is `0.824610/1.815750/0.355635`.
  SITReg retains slightly lower HD95; no statistical superiority claim is made.
- Lambda1 foreground SDlogJ/folding is `0.754557/3.145283%`, versus
  VFA `0.613349/0.188933%` and SITReg `0.566181/0.000502%`.
  Lambda3 brings SDlogJ to `0.574104` but folding remains `1.085931%`.
  Similar SDlogJ therefore does not establish similar folding behavior.
- Excluding rank4 leaves lambda1 selected, with mean Dice/HD95/ASD
  `0.834469/1.810951/0.341041`; the accuracy/regularity tradeoff is not solely
  explained by this case. The first10 were exploratory tuning, so this is a
  broader descriptive comparison, not an untouched confirmatory test.
- Complete CSV/JSON/Markdown, paired per-case differences, all mean/sample-SD
  values, leave-rank4-out sensitivity and checksummed acceptance are under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1/`.

## SITReg Training Sweep First100 Comparison (2026-10-10)

- All four P49 checkpoints have complete canonical100-case evaluations and
  3500 label rows each. Recomputed cohort statistics agree with the saved
  summaries;400 field hashes and all checkpoint hashes passed validation.
- Among the new seed42 SITReg runs (λ0.1,1/3,3,10), λ1/3 has the best
  mean Dice/HD95/ASD: `0.825681/1.820617/0.349815`, with foreground
  SDlogJ `0.679408` and folding `0.013091%`. The original historical λ1
  checkpoint has slightly lower HD95 `1.815750`, but lower Dice and higher ASD.
- New λ3/10 have zero measured foreground folding across these100 cases,
  at lower mean Dice `0.810004/0.767804`. Zero measured folding is not a
  proof of globally invertible continuous deformations.
- P48 MR-D-BSCP λ1 retains higher Dice `0.833679` and lower ASD `0.343654`
  than SITReg λ1/3, with nearly equal HD95 but much higher foreground folding
  `3.145283%`. Similar accuracy does not imply matched regularity.
- The combined10-setting table,1000 per-case rows and checksum provenance
  are saved under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first100_mrdbscp_vfa_sitreg_sweeps_comparison_v1/`.
  Original SITReg λ1 is historical, not a newly retrained seed42 sweep point.

## Source Reports

- `docs/reports/current_results_audit_and_plan.md`
- `docs/reports/reorganized_paper_points.md`
- `docs/reports/protocol_warnings/lungct_idir_ptvreg_protocol_bias.md`
- `docs/reports/protocol_warnings/ptvreg_technique_audit_directcp_multistage.md`
