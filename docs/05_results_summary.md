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

## Source Reports

- `docs/reports/current_results_audit_and_plan.md`
- `docs/reports/reorganized_paper_points.md`
- `docs/reports/protocol_warnings/lungct_idir_ptvreg_protocol_bias.md`
- `docs/reports/protocol_warnings/ptvreg_technique_audit_directcp_multistage.md`
