# Current Results Audit And Paper Plan

## Executive Status

The current evidence is promising but not yet paper-final.

What is already strong:

- The OASIS shuffled evaluation protocol is defined and balanced for the first 10 pairs.
- Saved displacement files exist for all OASIS DL baselines and Greedy for first 10, first 20, and all 200 shuffled pairs.
- Task9 already shows, on OASIS case000, that DirectCP cps4 can match or beat SINR cps4 under matched spline-style regularization, while original IDIR is weaker.
- Task4 already shows that IDIR can work well on 4DCT lung, while the four-stage knot-insertion DirectCP method, currently `32-16-8-4`, with pTV-style loss is close to pTVReg.
- Task4 also shows that single-stage DirectCP cps4 fails on 4DCT; this should be reported as a negative control, not treated as the main DirectCP method for lung.
- Deformation-stat reports already support the key explanation: 4DCT motion is larger but more coherent; OASIS motion is lower amplitude but locally heterogeneous.

What is not yet final:

- The main OASIS Pareto result must be rerun on the seed-3 shuffled first 10 cases, not only case000 or old sorted first10.
- The 4DCT matched Pareto sweep for IDIR/SINR/DirectCP is not yet complete.
- OASIS deformation-regime statistics should be regenerated on the seed-3 shuffled first10 for consistency.
- The final paper must standardize metric computation for learned baselines, Greedy, INR/spline methods, and postprocessed fields.

## Absolute Path Manifest

Use this manifest when porting the plan into a clean repository.

Current project root:

- `/home/hengjie/DL_projects/wavereg`

Current planning reports:

- Reorganized paper points: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/paper_planning/01_reorganized_paper_points.md`
- Current-results audit and plan: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/paper_planning/02_current_results_audit_and_plan.md`

Literature anchors:

- SINR paper: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/literature/2024_SINR_Spline_enhanced_Implicit_Neural_Representation_for_Multimodal_Registration.pdf`
- IDIR paper: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/literature/2022_IDIR_Implicit_Neural_Representations_for_Deformable_Image_Registration.pdf`
- ARC/Pareto trade-off paper: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/literature/Evaluation of Deformable Image Registration Under Alignment-Regularity Trade-Off.pdf`

OASIS protocol files:

- All-200 OASIS metric summary: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task2_inr_benchmark/doc/oasis_summary_test200.csv`
- Shuffled-pair README: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/oasis_test200_shuffle/README.md`
- Seed-3 shuffled pair list: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/oasis_test200_shuffle/pair_test_200_shuffled_seed000003.txt`
- Shuffle-generation script: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/scripts/create_oasis_test200_shuffle.py`

OASIS saved baseline deformation roots:

- Pretrained DL displacement root: `/data/hengjie/wavereg/wipmidl_0407/OASIS_v1/round1_MIDL`
- Greedy displacement root matching the summary arrays: `/data/hengjie/wavereg/datasets/OASIS/OASIS_official/oasis_organized/test_results/Greedy`
- Greedy variant root, NCC444: `/data/hengjie/wavereg/datasets/OASIS/OASIS_official/oasis_organized/test_results/Greedy_v1_2_ncc444`
- Greedy variant root, level4: `/data/hengjie/wavereg/datasets/OASIS/OASIS_official/oasis_organized/test_results/Greedy_v1_3_level4`
- Greedy variant root, NCC444 level4: `/data/hengjie/wavereg/datasets/OASIS/OASIS_official/oasis_organized/test_results/Greedy_v1_4_ncc444_level4`

Task9 OASIS evidence reports:

- IDIR/SINR/DirectCP case000 Pareto: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/pareto_idir_original_case000/reports/pareto_idir_original_case000_report.md`
- Regularizer campaign, case000 refined: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/pareto_regularizer_campaign/reports/stage2_case000_refined/pareto_report.md`
- LR scale pilot, case000: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/lr_scale_pilot_case000/reports/lr_scale_pilot_report.md`
- DirectCP old sorted first10 feasibility report: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/directcp_lr1e2_first10/reports/directcp_lr1e2_first10_report.md`
- B-spline fit probe: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/pareto_bspline_fit/reports/bspline_fit_probe_report.md`
- SVF posthoc README: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/svf_fit_posthoc/README.md`
- SVF case000 report: `/data/hengjie/wavereg/task9_inr_work_pareto/svf_fit_case000/svf_fit_report.md`

Task4 4DCT evidence reports:

- DirectCP/IDIR/pTV-prep iso1mm 4DCT report: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/lungct_32_16_8_4_directcp_idir_ptvprep_iso1mm_4dct_report.md`
- DirectCP 32-16-8-4 reduced-schedule LR sweep: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/lungct_directcp_32_16_8_4_ptvloss_reduced900_lr_sweep_report.md`
- pTV-prep visual audit README: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/lungct_ptvprep_visual_audit_4dct/README.md`
- Motion-regime characterization: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/motion_regime_characterization/motion_regime_characterization_report.md`
- OASIS-vs-4DCT foreground deformation stats: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/oasis_vs_lungct_1mm_deformation_stats/oasis_vs_lungct_1mm_foreground_deformation_stats_all10_report.md`
- OASIS-vs-4DCT lungmask deformation stats: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/oasis_vs_lungct_1mm_deformation_stats/oasis_vs_lungct_1mm_lungmask_deformation_stats_all10_report.md`

## Audited Data Status

### OASIS Shuffle

The current OASIS protocol should use:

- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/oasis_test200_shuffle/pair_test_200_shuffled_seed000003.txt`
- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/oasis_test200_shuffle/README.md`
- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task2_inr_benchmark/doc/oasis_summary_test200.csv`

The seed-3 shuffled first10 original sorted test indices are:

| Shuffled rank | Original test index |
|---:|---:|
| 0 | 158 |
| 1 | 62 |
| 2 | 14 |
| 3 | 69 |
| 4 | 47 |
| 5 | 137 |
| 6 | 110 |
| 7 | 6 |
| 8 | 116 |
| 9 | 88 |

The first 10 pairs contain 20 unique image IDs. The first 20 pairs satisfy the duplicate constraint; the maximum appearance of one image is 2, which is below the requested limit of 4.

### OASIS Saved Displacement Availability

Verified available for first 10, first 20, and all 200 shuffled pairs:

- `bs1_vxm_v0_2`
- `bs2_tm_v0`
- `bs3_vfa_v0`
- `bs4_sitreg_v0`
- `fedd_a-conv_1mf`
- `fedd_a-conv_2mfcorr`
- `fedd_a-conv_3corr`
- `fedd_b-convic_1mf`
- `fedd_b-convic_3corr`
- `fedd_c-vfa`
- `Greedy`
- Greedy variants `Greedy_v1_2_ncc444`, `Greedy_v1_3_level4`, `Greedy_v1_4_ncc444_level4`

The pretrained DL displacement root is:

- `/data/hengjie/wavereg/wipmidl_0407/OASIS_v1/round1_MIDL`

The Greedy displacement root matching the saved summary arrays is:

- `/data/hengjie/wavereg/datasets/OASIS/OASIS_official/oasis_organized/test_results/Greedy`

### OASIS Shuffled First10 Baseline Metrics

These are from the existing metric arrays using the seed-3 shuffled first10.

| Method | Dice mean | Dice std | HD95 mean |
|---|---:|---:|---:|
| pre | 0.5694 | 0.0616 | 3.8718 |
| greedy | 0.8029 | 0.0348 | 1.8639 |
| bs1_vxm_v0_2 | 0.7683 | 0.0362 | 2.4477 |
| bs2_tm_v0 | 0.7913 | 0.0287 | 2.0084 |
| bs3_vfa_v0 | 0.8198 | 0.0255 | 1.7743 |
| bs4_sitreg_v0 | 0.8215 | 0.0230 | 1.7159 |
| fedd_a-conv_1mf | 0.8219 | 0.0246 | 1.7667 |
| fedd_a-conv_2mfcorr | 0.8273 | 0.0239 | 1.7037 |
| fedd_a-conv_3corr | 0.8290 | 0.0227 | 1.6827 |
| fedd_b-convic_1mf | 0.8224 | 0.0239 | 1.6986 |
| fedd_b-convic_3corr | 0.8238 | 0.0226 | 1.6776 |
| fedd_c-vfa | 0.8214 | 0.0231 | 1.7838 |

These dots are useful for the OASIS Pareto figure, but they still need same-pipeline deformation regularity metrics from their saved displacement fields.

### HD95 Exclusion Audit

Known OASIS HD95 issue cases from the previous audit:

- Original sorted test indices: 49, 109, 135, 171, 172, 187, 188.
- Cause: label 34 missing or extremely tiny, producing `nan` or `inf` surface-distance entries.
- The seed-3 first20 does not include these cases.

Recommendation:

- Keep Dice summaries over all selected labels/cases where labels are valid.
- For HD95/ASD, state the exclusion rule explicitly.
- Do not silently drop cases from all metrics just because a surface-distance label is pathological.

## Task9 OASIS Result Audit

### What Task9 Already Supports

The strongest current OASIS evidence is from:

- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/pareto_idir_original_case000/reports/pareto_idir_original_case000_report.md`
- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/pareto_regularizer_campaign/reports/stage2_case000_refined/pareto_report.md`
- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/lr_scale_pilot_case000/reports/lr_scale_pilot_report.md`

Main observations:

- Original IDIR is weaker on OASIS case000.
- SINR cps4 improves over IDIR.
- DirectCP cps4 with matched B-spline regularization can match or exceed SINR cps4.
- Best SINR/DirectCP alpha is near `1e3` for the current NCC plus BE/diffusion scale.
- LR choices are mostly consistent with the pilot:
  - IDIR: `1e-4`.
  - SINR: `1e-4`.
  - DirectCP: `1e-2` or `1e-3`, depending on BE/diffusion and stability.

Representative case000 results:

| Method / setting | Dice | HD95 | Diffusion FG | BE FG | J<=0 FG % | logJ std |
|---|---:|---:|---:|---:|---:|---:|
| DirectCP cps4, BE analytic, alpha1000 | 0.8345 | 2.1185 | 0.1540 | 0.0916 | 3.9054 | 0.8152 |
| SINR cps4, BE analytic, alpha1000 | 0.8304 | 2.2418 | - | - | - | - |
| IDIR original, alpha0.1 | 0.8136 | - | - | - | - | - |
| IDIR original, alpha0.01 | 0.8056 | - | - | - | - | - |
| IDIR original, alpha0 | 0.7902 | - | - | - | - | - |

This is enough to justify the final full OASIS run, but it is not enough for the paper by itself because it is primarily case000.

### Important Task9 Caveats

Current OASIS task9 results are mixed across protocols:

- Some are case000 only.
- Some are old sorted first10.
- The paper should use seed-3 shuffled first10 as the main reported protocol.

The B-spline fitting probe is useful but should be secondary:

- Fitting dense DL fields into coarse B-spline grids loses Dice, especially for cps4.
- This means we should not overclaim that all good dense fields are trivially B-spline fields.
- The cleaner claim is narrower: within the SINR-style low-dimensional spline setting, direct control-point optimization is a strong and necessary baseline.
- Source report: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/pareto_bspline_fit/reports/bspline_fit_probe_report.md`

The SVF fitting result should also stay secondary:

- It helps demonstrate folding/NDV sensitivity to postprocessing.
- It should not be used as the main regularity definition.
- Source README: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/svf_fit_posthoc/README.md`
- Source case000 report: `/data/hengjie/wavereg/task9_inr_work_pareto/svf_fit_case000/svf_fit_report.md`

## Task4 4DCT Result Audit

### What Task4 Already Supports

The 4DCT experiments are a key addition because they prevent the paper from becoming a one-dataset criticism of INR.

The main source reports are:

- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/lungct_32_16_8_4_directcp_idir_ptvprep_iso1mm_4dct_report.md`
- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/lungct_directcp_32_16_8_4_ptvloss_reduced900_lr_sweep_report.md`
- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/lungct_ptvprep_visual_audit_4dct/README.md`

Main all10 results from the current reports:

| Method / setting | Snap TRE | Sub TRE | FG logJ std | J<=0 FG % | BE | Diffusion | Time sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| pTVReg official artifact | 0.918 | - | - | - | - | - | - |
| PyTorch pTV-style reproduced | 1.030 | 1.133 | - | - | - | - | 51.7 |
| IDIR reproduced | 1.127 | 1.215 | 0.1288 | 0.0000 | - | - | 218.4 |
| IDIR iso1mm + lungmask sampled NCC | 1.132 | 1.212 | 0.1398 | 0.0000 | 0.0046 | 0.0133 | - |
| SINR cps4 lr1e-4 pTV-prep | 2.383 | 2.463 | 0.2518 | 0.2594 | - | - | 189.1 |
| DirectCP cps4 lr1e-3 single-stage negative control | 6.142 | 6.199 | 0.4376 | 1.4234 | - | - | - |
| DirectCP 16-8-4 border NCC | 1.176 | 1.241 | 0.2147 | 0.0410 | 0.0407 | 0.0163 | - |
| DirectCP 16-8-4 pTV LCC+isoTV | 1.661 | 1.749 | 0.1593 | 0.0878 | 0.0143 | 0.0138 | - |
| DirectCP four-stage knot insertion 32-16-8-4 border NCC | 1.177 | 1.242 | - | - | - | - | - |
| DirectCP four-stage knot insertion 32-16-8-4 pTV LCC+isoTV | 1.040 | 1.134 | 0.1471 | 0.0057 | 0.0119 | 0.0120 | 293.2 |
| DirectCP four-stage knot insertion 32-16-8-4 pTV reduced lr1e-2 | 1.035 | 1.135 | 0.1498 | 0.0075 | 0.0127 | 0.0125 | 265.0 |
| DirectCP four-stage knot insertion 32-16-8-4 pTV reduced lr1e-3 | 1.045 | 1.139 | 0.1468 | 0.0079 | 0.0119 | 0.0121 | 243.9 |

Interpretation:

- IDIR working on lung is real and should be discussed positively.
- The key DirectCP method for lung is four-stage knot-insertion control-point optimization, currently `32-16-8-4`.
- DirectCP single-stage cps4 is not enough for lung and should be reported as a not-working negative control.
- SINR cps4 single-stage is weak under the current lung setup.
- pTVReg remains the best reference if the official artifact is accepted as the target.

The intended 4DCT message is therefore not "DirectCP cps4 works on lung." It is: "single-stage local control fails on lung, but adding a coarse-to-fine/knot-insertion deformation prior makes direct control-point optimization competitive."

### Why IDIR Can Work On Lung

The best explanation is deformation-regime fit:

- 4DCT is intrasubject.
- Respiratory motion has large amplitude but strong spatial coherence.
- A coordinate MLP can represent smooth global deformations well.
- OASIS is intersubject.
- Brain anatomical differences are more localized, heterogeneous, and label-dependent.
- Local control is more important there, so spline-like methods have an advantage.

This should be stated as an interpretation of the evidence, not as a universal theorem.

## Deformation-Stats Audit

The deformation-stat reports give one of the strongest paper messages, but their protocol needs a final consistency pass.

Source reports:

- Motion-regime characterization: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/motion_regime_characterization/motion_regime_characterization_report.md`
- Foreground deformation stats: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/oasis_vs_lungct_1mm_deformation_stats/oasis_vs_lungct_1mm_foreground_deformation_stats_all10_report.md`
- Lungmask deformation stats: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/oasis_vs_lungct_1mm_deformation_stats/oasis_vs_lungct_1mm_lungmask_deformation_stats_all10_report.md`

### Current Main Finding

Using method-estimated fields:

| Metric | OASIS | 4DCT | 4DCT/OASIS |
|---|---:|---:|---:|
| Mean displacement magnitude | 3.799 mm | 8.917 mm | 2.35 |
| P95 displacement magnitude | 7.244 mm | 18.696 mm | 2.58 |
| Voxels > 10 mm | 0.718% | 36.258% | 50.50 |
| Diffusion energy | 1.4041 | 0.0917 | 0.07 |
| Bending energy | 0.3926 | 0.0071 | 0.02 |
| Displacement TV | 1.0544 | 0.2601 | 0.25 |
| logJ std | 2.9266 | 0.2399 | 0.08 |
| J<=0 | 4.4937% | 0.0203% | 0.0045 |
| Correlation length | 10.94 mm | >=64.00 mm | 5.85 |

Autocorrelation also supports the story:

| Distance | OASIS autocorr | 4DCT autocorr |
|---:|---:|---:|
| 2 mm | 0.949 | 0.999 |
| 4 mm | 0.831 | 0.998 |
| 8 mm | 0.597 | 0.993 |
| 16 mm | 0.328 | 0.981 |
| 32 mm | 0.139 | 0.947 |
| 64 mm | 0.033 | 0.848 |

Coarse-grid explained energy:

| Spacing | OASIS | 4DCT |
|---:|---:|---:|
| 8 mm | 0.823 | 0.999 |
| 16 mm | 0.520 | 0.995 |
| 32 mm | 0.205 | 0.982 |
| 64 mm | -0.010 | 0.944 |

### Caveat

The OASIS source for these deformation stats is currently old sorted first10 DirectCP cps4 diffusion alpha1000 lr1e-2 epoch500. The 4DCT source is DirectCP 32-16-8-4 pTV LCC+isoTV reduced lr1e-2 all10.

For the final paper:

- Rerun the OASIS deformation-stat comparison on the seed-3 shuffled first10.
- State that these are estimated deformation fields, not ground-truth motion fields.
- Use the analysis to characterize the task/method operating regime, not as absolute anatomy truth.

## Highest-Priority Experiments

### Priority A: OASIS Seed-3 First10 Main Pareto

Run the full main method set on the seed-3 shuffled first10:

- IDIR original.
- SINR cps4.
- SINR cps2.
- DirectCP cps4.

Recommended OASIS alpha plan:

- For SINR/DirectCP common NCC plus BE/diffusion scale:
  - `0, 1, 10, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000`
  - If compute is tight: `0, 10, 100, 300, 1000, 3000, 10000, 100000`
- For IDIR original:
  - Keep its own alpha grid and report separately because scales differ.
  - Existing case000 grid was `0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000`.

Recommended OASIS LR:

- IDIR: `1e-4`.
- SINR: `1e-4`.
- DirectCP: keep both `1e-2` and `1e-3` in the pilot, then choose one per regularizer if stability differs.

Regularizer plan:

- Main paper should probably show one primary regularizer axis, either BE analytic or diffusion analytic.
- Include forward-difference versions as validation/supplement unless the findings differ materially.
- Report logJ std and folding/NDV for all saved fields.

Outputs needed:

- Per-case metrics.
- Mean and spread across 10 cases.
- Saved displacements for all operating points.
- Same-pipeline deformation stats for DL and Greedy dots.

### Priority B: 4DCT Matched Pareto

Run a matched 4DCT Pareto sweep, but allow lung-specific method variants to be clearly labeled.

Minimum set:

- IDIR original/lungmask sampled NCC alpha sweep.
- SINR cps4 and cps2 under the same preprocessing/loss scale as DirectCP where possible.
- DirectCP cps4 single-stage as the direct OASIS analog and expected failure/negative control.
- DirectCP four-stage knot insertion `32-16-8-4` as the main lung DirectCP method.
- pTVReg official/reproduced as reference dots.

Recommended 4DCT interpretation:

- Do not force the conclusion that DirectCP single-stage should work everywhere.
- Use the single-stage failure and four-stage knot-insertion success to argue that lung needs coarse-to-fine capture of coherent large motion.
- Use IDIR's good result to show that coordinate MLPs are not inherently bad; they can match coherent motion regimes.

Outputs needed:

- TRE versus regularity Pareto.
- Per-case pTVReg official iso1mm result if available.
- Regularity metrics computed over consistent lung foreground/mask.
- Runtime/memory.

### Priority C: Consistency And Supplement

Run or organize:

- OASIS seed-3 first10 deformation-regime stats.
- Optional OASIS first20 after the first10 story is stable.
- SVF fitting/postprocessing supplement.
- B-spline fitting probe as supplement, with restrained interpretation.
- Visual examples:
  - One OASIS case showing local anatomical deformation differences.
  - One 4DCT case showing coherent respiratory field.
  - Jacobian/logJ or displacement magnitude overlays if readable.

## Standardization Checklist

Before final results are generated, lock down:

- Pair order and case IDs.
- Fixed/moving direction.
- Voxel spacing and physical-unit convention.
- Whether fields are reported in voxel or mm units.
- Interpolation convention.
- `align_corners` convention if PyTorch grids are used.
- Foreground and lungmask definitions.
- Label set used for OASIS Dice/HD95/ASD.
- HD95/ASD exclusion rule for missing/tiny labels.
- Dense deformation sampling resolution for regularity metrics.
- Boundary handling for finite differences.
- Analytical B-spline regularizer formula and normalization.
- Whether energy metrics are averaged over foreground, full image, or valid finite-difference domain.
- Alpha grids and LR grids.
- Epoch count and early stopping policy.
- Runtime/memory measurement.

## Recommended Main Tables And Figures

### Table 1: Dataset And Motion Regime

Columns:

- Dataset.
- Pair type: intersubject or intrasubject.
- Primary metric.
- Mean displacement magnitude.
- P95 displacement.
- Correlation length.
- Coarse-grid explained energy.

### Figure 1: Representation And Prior Schematic

Show:

- IDIR: continuous coordinate MLP, global smooth implicit prior.
- SINR: INR plus spline-enhanced local control.
- DirectCP: direct spline control, no neural implicit part.
- DL/Greedy/pTVReg: reference methods.

### Figure 2: OASIS Pareto

Show:

- Dice or HD95 versus BE/diffusion.
- Curves for IDIR, SINR cps4, SINR cps2, DirectCP cps4.
- Mean plus spread.
- DL and Greedy dots.

### Figure 3: 4DCT Pareto

Show:

- TRE versus regularity.
- IDIR, SINR, single-stage DirectCP cps4 negative control, four-stage knot-insertion DirectCP `32-16-8-4`, and pTVReg.
- Mean plus spread.

### Figure 4: Deformation-Regime Comparison

Show:

- OASIS versus 4DCT displacement magnitude.
- Regularity burden.
- Autocorrelation/coarse-grid energy.

### Supplement

Include:

- Full alpha tables.
- Regularizer variants.
- SVF fitting analysis.
- HD95 exclusion audit.
- Per-case DL/Greedy displacement availability.

## Paper Risks And How To Handle Them

Risk 1: The story sounds like an attack on SINR.

Response:

- Phrase the contribution as component disentanglement.
- Credit SINR for identifying a useful hybrid representation.
- Show that the spline component is an important mechanism.

Risk 2: IDIR works on lung, which seems to weaken the claim.

Response:

- Treat this as one of the paper's strongest findings.
- It supports deformation-regime dependence.
- It prevents the paper from making an overbroad anti-INR conclusion.

Risk 3: B-spline is technically grid-controlled, so "off-grid" framing could be challenged.

Response:

- Acknowledge this directly.
- State that both B-spline and INR define continuous transformations, but their control priors differ.
- The lesson is that continuous evaluation alone is not the decisive property.

Risk 4: Folding/NDV can be improved by SVF fitting.

Response:

- Make SVF fitting a minor sensitivity analysis.
- Use diffusion, BE, and logJ std as main regularity endpoints.

Risk 5: OASIS and 4DCT use different task metrics.

Response:

- Do not compare Dice and TRE directly.
- Compare method behavior within each dataset.
- Use cross-dataset deformation stats only for explaining motion regimes.

## Suggested Final Claim

A concise final claim for the paper:

> Our results suggest that the practical advantage of continuous registration models is not explained by off-grid representation alone. On OASIS intersubject brain registration, local spline control recovers much of the SINR gain over dense INR, while original IDIR underperforms. On 4DCT intrasubject lung registration, IDIR becomes competitive because the deformation is large but coherent. Alignment-regularity Pareto evaluation reveals this dependence more clearly than single operating points or folding-only summaries.
