# Reorganized Paper Points

## Working Thesis

This paper should not read as "SINR is invalid" or "IDIR does not work." The stronger and fairer claim is:

> Continuous off-grid deformation representations are not sufficient by themselves. In deformable registration, the decisive factor is the transformation prior induced by the parameterization, optimization, and regularization. B-splines, coordinate MLPs, and hybrid INR-spline models impose different priors, and their success depends on the motion regime.

This framing fits an off-grid workshop because both B-splines and INRs represent continuous differentiable transformations. The important observation is that "off-grid" is not a single methodological category with one expected behavior. A B-spline is controlled on a grid, but its evaluated deformation field is continuous; an INR is also continuous, but its coordinate-network prior is different. The paper can use this contrast to ask what the off-grid representation actually buys.

## Source Files To Carry Forward

Use these absolute paths when moving this plan into a cleaner repository.

Project root for the current evidence:

- `/home/hengjie/DL_projects/wavereg`

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
- Motion-regime characterization: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/motion_regime_characterization/motion_regime_characterization_report.md`
- OASIS-vs-4DCT foreground deformation stats: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/oasis_vs_lungct_1mm_deformation_stats/oasis_vs_lungct_1mm_foreground_deformation_stats_all10_report.md`
- OASIS-vs-4DCT lungmask deformation stats: `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/oasis_vs_lungct_1mm_deformation_stats/oasis_vs_lungct_1mm_lungmask_deformation_stats_all10_report.md`

## Central Message

The paper should make three connected arguments.

1. SINR's gains on OASIS should be disentangled from the INR component.

   SINR combines an INR with a spline-enhanced deformation representation. The current evidence suggests that a large part of the benefit may come from the spline/local-control component rather than the neural implicit representation itself. A direct B-spline control-point optimization baseline is therefore not a strawman; it is the critical control experiment.

2. IDIR's mixed behavior is informative, not just a failure.

   IDIR is expected to be weaker on OASIS intersubject brain registration because the deformation is local, heterogeneous, and anatomically irregular. Its surprisingly good behavior on 4DCT lung registration is important: respiratory lung motion is larger in amplitude but much more spatially coherent and organ-scale. That suggests the registration method should be chosen based on deformation characteristics, not only on whether the representation is neural or continuous.

3. Regularity should be evaluated as a Pareto trade-off, not as a single best Dice/TRE number.

   A method can look good at one operating point but fail when the alignment-regularity trade-off is sampled. The ARC-style Pareto evaluation is a strong validation tool because it compares methods as families of solutions. This is especially important here because folding and NDV can be changed by postprocessing, including SVF fitting, while diffusion energy, bending energy, and log-Jacobian spread better expose the deformation behavior.

## Intended Narrative

The story can be organized as:

1. Off-grid registration methods are often discussed through representation class: INR, spline, dense field, or learned network.
2. But in deformable registration, the prior over transformations may matter more than the fact that a function can be evaluated continuously.
3. SINR improves over IDIR on brain registration, but SINR changes more than "adding INR": it introduces local spline structure.
4. A direct control-point B-spline model isolates that spline structure.
5. The comparison should be done through Pareto fronts because alignment and deformation regularity cannot be collapsed into one number.
6. OASIS and 4DCT provide complementary motion regimes:
   - OASIS: intersubject brain, lower displacement magnitude, high local heterogeneity.
   - 4DCT: intrasubject lung, larger displacement magnitude, high spatial coherence.
7. IDIR's lung result and brain weakness together support the broader claim: method success is deformation-regime dependent.

## What To Standardize

The paper should standardize the following so the comparison feels careful rather than adversarial.

### Datasets

Use two primary datasets:

- OASIS brain: seed-3 shuffled test pairs, first 10 pairs for the main version, first 20 if time allows.
- 4DCT lung: 10 pairs.

OASIS pair definition should be tied to:

- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task2_inr_benchmark/doc/oasis_summary_test200.csv`
- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/oasis_test200_shuffle/README.md`
- `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task9_inr_work_pareto/oasis_test200_shuffle/pair_test_200_shuffled_seed000003.txt`

### Methods

The common method set should be:

- IDIR: original implementation style and original loss as much as possible.
- SINR cps4.
- SINR cps2.
- DirectCP cps4.

The dataset-specific references should be:

- OASIS: pretrained DL baselines bs1-bs4, six FedD variants, and Greedy, shown as dots on the Pareto plot.
- 4DCT: pTVReg, ideally official iso-1mm per-case results if available, plus reproduced pTV-style baseline.
- 4DCT main DirectCP method: four-stage knot-insertion control-point optimization, currently the `32-16-8-4` DirectCP setting in `/home/hengjie/DL_projects/wavereg/codex_auto_exp/task4_lungct/doc/lungct_32_16_8_4_directcp_idir_ptvprep_iso1mm_4dct_report.md`.
- 4DCT negative control: single-stage DirectCP cps4. This is expected to fail on lung and should be reported as evidence that large coherent respiratory motion needs coarse-to-fine capture, not just local cps4 control.

### Losses And Regularizers

For OASIS SINR and DirectCP, standardize the image loss and regularizer scale:

- Same NCC definition.
- Same foreground/mask convention.
- Same dense evaluation resolution for reported deformation metrics.
- Same alpha sweep grid for comparable spline-based methods.

For regularizers, treat these as controlled variants:

- Bending energy, analytical from B-spline.
- Bending energy, forward finite difference.
- Diffusion energy, analytical from B-spline.
- Diffusion energy, forward finite difference.

IDIR can use its original loss and alpha scale, but the paper should explicitly state that IDIR's alpha is not numerically comparable to the spline alpha because the regularizer and sampling are different.

### Metrics

For OASIS:

- Dice mean.
- HD95 and ASD, with label/pathology exclusions explicitly documented.
- NCC or image similarity for optimization sanity checking.
- Diffusion energy.
- Bending energy.
- Standard deviation of log-Jacobian.
- Folding percentage / `J <= 0`.
- NDV, but not as the sole regularity endpoint.
- Runtime and peak memory if space allows.

For 4DCT:

- Landmark TRE, snap and sub-voxel if available.
- Same deformation statistics as above.
- Runtime and memory.

## What To Demonstrate

### Main Demonstration 1: SINR Versus Direct B-spline Control

The key OASIS experiment should ask:

> If we keep the spline structure and regularization but remove the INR part, how much of SINR's Pareto behavior remains?

A strong result would be:

- DirectCP cps4 matches or improves SINR cps4 at the same regularity level.
- SINR cps2 and cps4 show whether finer local control explains performance.
- IDIR is below the spline-based methods on OASIS, especially at comparable regularity.

The language should be careful:

- Say: "On OASIS, matching SINR's spline parameterization with direct control-point optimization recovers much of the gain."
- Avoid: "SINR only works because of B-splines."

### Main Demonstration 2: Pareto Fronts Instead Of Single Points

The paper should present ARC-style Pareto curves:

- OASIS: Dice or HD95 versus BE/diffusion/logJ std.
- 4DCT: TRE versus BE/diffusion/logJ std.

For the main figure, choose one primary regularity axis. A good candidate is bending energy or diffusion energy, with log-Jacobian std as a secondary panel. Folding/NDV can be included but should not dominate because SVF fitting/postprocessing can bias them.

The plot should show:

- Mean curve across 10 cases.
- Spread band or error bars.
- Case-level faint lines if readable.
- DL/Greedy/pTVReg references as dots.

### Main Demonstration 3: Motion-Regime Dependence

The OASIS versus 4DCT deformation-stat comparison should support the explanation:

- 4DCT lung: larger displacement magnitude, more coherent deformation.
- OASIS brain: lower displacement magnitude, much higher local heterogeneity and regularity burden.

This explains why IDIR can be competitive on lung without explicit multiresolution spline modeling while still struggling on brain.

Suggested wording:

> IDIR's lung result should not be treated as a contradiction. It is evidence that a coordinate-network prior can be appropriate when the motion is smooth, coherent, and intrasubject. Its OASIS weakness then reveals a limitation under localized intersubject anatomical deformation.

For DirectCP on 4DCT, the key result is not the single-stage cps4 setting. The paper should emphasize the four-stage knot-insertion DirectCP model, currently `32-16-8-4`, as the main spline/control-point method for lung. The single-stage cps4 result should be presented as a failure mode: it demonstrates that matching the final local control resolution is not enough for large respiratory motion unless the optimization also has a coarse-to-fine path.

### Minor Demonstration: SVF Fitting Sensitivity

SVF fitting should be presented as a cautionary analysis, not as a main claim:

- Folding and NDV can improve after projection/postprocessing.
- Alignment may remain similar.
- But diffusion, bending energy, and log-Jacobian spread may not tell the same story.

The message is:

> Topology-style summary metrics are useful, but they are not robust enough to carry the whole regularity comparison by themselves.

## Proposed Paper Shape

### Title Direction

Recommended title shortlist:

- "Is It the INR, the Spline, or the Prior? Disentangling Progress in Continuous Medical Image Registration"
- "Is It the INR or the Spline, or Is It the Prior? Disentangling Progress in Continuous Medical Image Registration"
- "Is It the INR or the Spline? Disentangling Representation and Prior in Continuous Medical Image Registration"
- "Beyond INR Versus Spline: Disentangling Deformation Priors in Continuous Medical Image Registration"
- "Is It the Representation or the Prior? Disentangling Progress in Continuous Medical Image Registration"
- "Is It the INR or the Prior? A Controlled Pareto Study of Continuous Deformable Registration"
- "Is It the INR, the Spline, or the Prior? Disentangling Continuous Deformation Models Through Pareto Evaluation"
- "Is It the INR or the Deformation Prior? A Controlled Study Across Brain and Lung Registration"
- "Is It the INR or the Representation Prior? A Pareto Study of Implicit, Spline, and Hybrid Registration"
- "Off-Grid Is Not Enough: Disentangling Representation and Deformation Priors in Image Registration"
- "Continuous Representations, Different Priors: A Pareto Study of INR and Spline Registration"
- "When Does Implicit Registration Help? A Deformation-Regime Study of INR and Spline Priors"
- "Beyond the Coordinate Network: What Makes Implicit Registration Work?"
- "The Prior Behind the Representation: Reassessing INR-Based Deformable Registration"

More workshop-sharp options:

- "Off-Grid, On-Prior: Disentangling Continuous Deformation Models for Registration"
- "Is It the INR or the Spline? A Pareto Analysis of SINR for Deformable Registration"
- "Not All Continuous Deformations Are Equal: INR, Splines, and Registration Priors"
- "Continuous But Different: How Transformation Priors Shape Off-Grid Registration"
- "What Is Learned by Going Off-Grid? A Controlled Study of INR and Spline Registration"

Spline-first options:

- "Splines Before INRs? Disentangling Hybrid Implicit Registration"
- "Spline Priors, Not Just INRs: A Pareto Study of Continuous Deformable Registration"
- "Spline Control Explains the Gain: A Controlled Study of SINR for Registration"
- "Spline-Enhanced or INR-Driven? Reassessing SINR Through Pareto Evaluation"
- "Spline Locality Matters: Continuous Deformation Priors for Brain and Lung Registration"
- "Spline Priors Across Motion Regimes: Revisiting Implicit Neural Registration"
- "Spline Control in Off-Grid Registration: Representation, Regularity, and Motion Regime"
- "Splines, INRs, and the Shape of Deformation Priors"

INR-first options:

- "Is It the INR or the Prior? A Controlled Pareto Study of Continuous Deformable Registration"
- "Is It the INR or the Deformation Prior? A Controlled Study Across Brain and Lung Registration"
- "Is It the INR or the Representation Prior? A Pareto Study of Implicit, Spline, and Hybrid Registration"
- "INR or Spline? Disentangling the Source of Improvement in SINR"
- "INRs Are Not Enough: The Role of Spline Priors in Deformable Registration"
- "INR Registration Across Motion Regimes: When Coordinate Networks Help and When They Struggle"
- "INR-Based Registration Revisited: Continuous Representation Versus Deformation Prior"
- "INRs, Splines, and Pareto Fronts: A Controlled Study of Deformable Registration"
- "INR Meets Spline: Disentangling Hybrid Priors in Continuous Registration"
- "INR Is a Prior Too: Motion-Regime Dependence in Continuous Registration"
- "INR Without Illusion: A Pareto Analysis of Spline-Enhanced Registration"

Broader INR/encoding options if hash or NGP-style baselines are included:

- "Is It the INR or the Encoding? A Pareto Study of Coordinate, Hash, and Spline Registration Priors"
- "Is It the Coordinate Network, the Hash Grid, or the Spline? Disentangling Continuous Registration Priors"
- "Implicit, Hashed, or Spline? A Controlled Pareto Study of Continuous Deformation Models"
- "Coordinate, Hash, and Spline Priors for Continuous Deformable Registration"
- "Beyond Coordinate Networks: Hash, Spline, and Implicit Priors for Registration"

Motion-regime variants:

- "Is It the INR or the Prior? Motion-Regime Dependence in Continuous Deformable Registration"
- "Is It the INR or the Motion Prior? A Controlled Pareto Study Across Brain and Lung Registration"
- "Is It the Representation or the Motion Regime? Disentangling Continuous Priors for Deformable Registration"
- "Continuous Priors Across Motion Regimes: A Pareto Study of Brain and Lung Registration"
- "When Coordinate Networks Work: Motion-Regime Dependence in Continuous Registration"

Disentangling-progress variants inspired by the broader "trend-driven architecture versus domain-specific strategy" framing:

- "Is It the INR, the Spline, or the Prior? Disentangling Progress in Continuous Medical Image Registration"
- "Is It the INR or the Spline, or Is It the Prior? Disentangling Progress in Continuous Medical Image Registration"
- "Disentangling Progress in Continuous Medical Image Registration: Beyond Implicit Representations Toward Deformation Priors"
- "Beyond Trend-Driven Implicit Representations: A Pareto Study of Deformation Priors in Medical Image Registration"
- "Rethinking Implicit Registration: From Trend-Driven Representations to Motion-Specific Deformation Priors"
- "Disentangling Off-Grid Registration: Continuous Representations, Spline Priors, and Motion Regimes"

More careful academic options:

- "Disentangling Implicit and Spline Priors in Continuous Deformable Image Registration"
- "A Pareto Evaluation of Implicit Neural and Spline-Based Deformation Models"
- "Representation, Regularity, and Motion Regime in Continuous Deformable Registration"
- "Evaluating Continuous Deformation Priors Across Brain and Lung Registration"
- "Alignment-Regularity Trade-offs in Implicit Neural and B-spline Registration"

More provocative options, but still fair:

- "SINR Without the Mystery: A Controlled Study of Spline Priors in Implicit Registration"
- "The Spline in SINR: Disentangling Hybrid Implicit Registration"
- "When the Spline Explains the Gain: A Pareto Study of SINR and Direct Control Points"
- "Implicit Neural Registration Is a Prior, Not a Magic Coordinate System"
- "The Off-Grid Illusion? Continuous Representations and Deformation Priors in Registration"

Current favorites:

- Current lead: "Is It the INR, the Spline, or the Prior? Disentangling Progress in Continuous Medical Image Registration"
- Best overall/broad: "Is It the INR or the Prior? A Controlled Pareto Study of Continuous Deformable Registration"
- Best if motion-regime dependence becomes central: "Is It the INR or the Deformation Prior? A Controlled Study Across Brain and Lung Registration"
- Best if hash/NGP baselines are included: "Is It the INR or the Encoding? A Pareto Study of Coordinate, Hash, and Spline Registration Priors"
- Best for the off-grid workshop: "Off-Grid, On-Prior: Disentangling Continuous Deformation Models for Registration"
- Best if we want a careful academic tone: "Representation, Regularity, and Motion Regime in Continuous Deformable Registration"

### Abstract Message

The abstract should state:

- INR-based registration methods are attractive because they define continuous deformation fields.
- But continuous evaluation does not identify the effective transformation prior.
- We compare IDIR, SINR, and direct B-spline control under matched alignment-regularity Pareto evaluation.
- On intersubject brain registration, spline/local-control structure explains much of the SINR improvement over IDIR.
- On intrasubject lung registration, IDIR can be competitive because the motion is more globally coherent.
- These results suggest off-grid registration should be evaluated by deformation-regime fit and Pareto behavior, not representation label alone.

### Main Figures

Figure 1: Conceptual diagram.

- Axis 1: continuous/off-grid evaluation.
- Axis 2: transformation prior/locality.
- Place IDIR, SINR, DirectCP, DL baselines, Greedy/pTVReg.

Figure 2: OASIS Pareto.

- Dice or HD95 versus regularity.
- Curves for IDIR, SINR cps4, SINR cps2, DirectCP cps4.
- DL/Greedy dots.
- Mean and spread across 10 shuffled cases.

Figure 3: 4DCT Pareto/reference comparison.

- TRE versus regularity.
- IDIR, SINR, DirectCP, pTVReg.
- Make four-stage knot-insertion DirectCP, currently `32-16-8-4`, the main DirectCP curve for lung.
- Show single-stage DirectCP cps4 as a negative-control point/curve, not as the expected working method.

Figure 4: Deformation-regime characterization.

- Displacement magnitude and P95.
- Diffusion/BE/logJ std.
- Correlation length or low-pass/coarse-grid explained energy.

Supplement:

- Full alpha tables.
- Regularizer variants.
- HD95 label exclusion audit.
- SVF fitting/postprocessing analysis.

## Bottom-Line Contribution

The clean contribution is:

> We show that the practical success of continuous registration models depends less on being "off-grid" in the abstract and more on the deformation prior induced by the representation. On brain registration, SINR's spline component and local control explain much of its advantage over dense INR. On lung registration, IDIR's coordinate-network prior can work well because the target motion is coherent and intrasubject. Pareto evaluation exposes these differences more reliably than single-point metrics or folding alone.

For the 4DCT spline baseline, the final claim should specifically refer to four-stage knot-insertion DirectCP rather than single-stage cps4 DirectCP. The failed single-stage cps4 result is useful because it shows that the lung result depends on a coarse-to-fine deformation prior, not only on the final control spacing.
