---
doc_type: results_summary
status: active_draft
---

# Results Summary

This repo carries audited generated results together with planning reports.

## First100 three-family B-spline842 sweep (2026-10-16 local, P79)

- Accepted2100 registrations:700 each for prolonged displacement, bounded
  two-map composition, and refined single-SVF/velocity diffusion. Seven global
  lambdas .01/.1/.3/1/3/10/100, seed3 shuffled ranks0–99, squared local NCC9.
  No dense, cps1, or symmetric stage.570 accepted earlier results reused;
  1530 new cells completed with independent replay. Original VFA/SITReg each
  contribute100 reference fields. Tables separate first10/additional90/all100.
- Best mean-Dice settings within each tested family: displacementlambda1
  Dice.827703,HD951.888310mm,ASD.360354mm,FGSDlogJ.754939,fold3.291952%;
  bounded-twolambda1 .822440/1.924015/.373693,.603710,9.920904e-5%;
  refined-SVFlambda.3 .825833/1.953872/.369430,.728630,.012774%.
  These are exploratory test-set selections, not prespecified winners or
  regularity-matched comparisons. Full tables retain all seven weights.
- Original SITReg .824610/1.815750/.355635,.566181,5.021921e-4%;
  VFA .821588/1.884796/.370851,.613349,.188933%. Displacement can exceed
  mean Dice while folding much more; bounded-two sharply reduces folding
  with lower mean Dice than SITReg. No family dominates all five metrics.
  Bounded maps or exponentiation do not imply zero measured exported folding.
- Two weak-lambda SVF cells (ranks29/44) required approved proof-gated
  transition policyv3. Retain their own completed stages and optimize only
  missing stages; objectives, FP32 optimization and seven-step integration
  unchanged. Exact stage/field/loss/metric replay passed. All2098 previously
  accepted results, original failures and partial report remain unchanged.
- Continuation suite89 tests (88 pass,one optional skip), two full-size
  missing-stage smokes/replays,345 serialized statistic checks and13 accepted
  output hashes passed. Rank29/44 new optimizer times431.39/383.72s, worker
  477.27/424.18s, replay37.90/36.83s, peak reserved3.297/3.262GiB. Original
  prefix optimizer time26.10/36.79s is separate, not rerun. Shared-load timing
  remains diagnostic; historical unavailable resource values remain missing.
- Accepted artifact family
  `oasis_seed3_first100_bspline842_displacement_bounded2_svf_velocity_ncc9_lambda_sweep_v1`,
  report `transition_continuation_v3/release_1792210489765127750`.
  Includes CSV/JSON/Markdown, per-label/stage/paired/resource/integration data,
  and2x3 ARC. `readable_arc_v2` wraps titles without changing data or original
  accepted outputs. First10 informed tuning and additional90 were previously
  inspected: no untouched confirmatory-test claim.

## Rank-0 symmetric SVF amplitude/balance ablation (2026-10-16, P78)

- Accepted28 fresh fits and independent replays: baseline, seven amplitude-only,
  seven balance-only, nine combined neighbors, three matched-GPU input swaps,
  one matched-GPU same-order winner repeat. Four full-size smokes/replays and
  45 tests passed. No failures, retries, grid expansion or first-ten evaluation.
  Only rank0 fixed0333/moving0115, independently refined total SVFs at8/4/2.
- Preserve A1 NCC9 +2*(R1+R2). Add native-full-volume amplitude alpha*(E1+E2)
  and global-RMS balance beta*(sqrt(E1+eps)-sqrt(E2+eps))^2; no other loss change.
  Select weights using original-direction Dice/ASD/HD95, never epochs by labels.
- Best tested: amplitude alpha.001,beta0. Dice.796250,HD951.980043mm,
  ASD.430644mm,FGSDlogJ.555949,fold0%. Fresh A1 is .786141/2.156043/.473178,
  FGSDlogJ.540750,fold0%. Thus accuracy improves while foreground SDlogJ rises
  slightly; zero measured folding is not an invertibility guarantee.
- Shared velocity RMS drops27.316->2.915mm; branch RMS3.007/3.472mm. Midpoint
  positive coverage falls from71.9/73.4% to38.9/39.6% (original approximately31%).
  Higher amplitude .003/.01 lowers Dice to.789940/.774736, consistent with
  suppressing useful motion as well as shared motion. This is an observed
  ablation, not a proof of a unique optimization mechanism.
- Best balance-only beta1: Dice.795014,HD952.034708,ASD.444027,FGSDlogJ.558556,
  fold4.646296e-05%. RMS18.731/18.733mm is closely balanced but shared RMS
  remains18.552mm. Strong balance also reduces common motion indirectly in
  these finite-budget runs; it still does not directly penalize equal shared
  velocities. Combined alpha.001/beta1: Dice.793762,HD951.989542,ASD.434541,
  FGSDlogJ.554815,fold4.646296e-05%; RMS3.220/3.221mm,shared2.877mm. Better
  RMS balance did not improve best Dice within the approved combined grid.
- Historical842 refined-SVF velocity lambda1 remains slightly better on all
  three accuracy metrics (.797502/1.960509/.425200), with higher FGSDlogJ.596835
  and fold4.646296e-05%. Bounded-two lambda1 has higher Dice.801036 and lower
  ASD.418641, but HD951.985539 is slightly higher than the amplitude winner.
- Geometric repeatability is limited despite stable aggregate metrics. Winner
  same-order repeat Dice.796218 versus.796250; input swap reverse-export Dice
  .796538. Foreground repeat EPE mean/p95/max .078744/.287850/11.501729mm;
  swap .077392/.289065/4.883435mm. Do not claim exact input symmetry or identical
  retraining fields, and do not attribute discrepancies solely to CUDA without
  further evidence. Independent replay of each saved checkpoint is exact.
- Winner within-fit forward/reverse off-grid cycle means .051508/.050119mm,
  versus fresh-A1 .117697/.105758mm. Export-interpolant/retained-map errors and
  7/8/10-step sensitivity remain separately recorded, not substituted for the
  seven-step primary output. No zero-inverse-error guarantee.
- Queue40.28min; winner optimizer/worker/replay61.44/121.37/61.06s,525 joint
  updates (125/150/250),peak allocated/reserved4.568/4.859GiB. Maximum measured
  peak across all runs/replays4.918GiB. Shared-load timings are diagnostic.
- CSV/JSON/Markdown accepted after140 candidate metric checks,84 stage rows,
  168 integration rows and source/output checks. Artifact key
  `oasis_seed3_rank000_bspline_two_total_svf_midpoint_ablation_v1/phase_b_v1`.
  Historical sources and results are immutable. Single exploratory test case,
  no across-case SD/generalization claim; no follow-up experiments launched.

## Rank-0 symmetric refined-SVF Phase A (2026-10-15, P77)

- Four fresh 8→4→2 fits and independent replays accepted: two independent
  total SVFs with coefficient 2 on their summed diffusion (A1), coefficient 1
  scaling control (A2), tied +/-w/2 midpoint control (A3), and swapped-input
  A1. No amplitude or balance penalty, no extra stages or tuning. Equal-split
  effective lambda is 1 for A1 and 0.5 for A2; this is not exact matching for
  arbitrary finite nonlinear transformations.
- Rank-0 Dice/HD95/ASD: A1 0.785973/2.160976/0.473968 mm, A2
  0.788333/2.120749/0.447276 mm, A3 0.790653/2.116758/0.458248 mm.
  All are below the historical lambda-1 bounded-two and refined-SVF accuracy
  (Dice 0.801036 and 0.797502). A1/A2/A3 foreground SD(log J) is
  0.542075/0.668229/0.567224; folding is 0/0.002462537/0.000464630%.
- Similar branch energies do not imply a useful midpoint. A1 branch RMS
  magnitudes are 27.062/27.649 mm, native diffusion 0.056072/0.056512,
  but common-mode RMS is 27.195 mm and velocity cosine is +0.9765. A2
  has still larger common-mode RMS 35.921 mm despite only 1.19% RMS imbalance.
  The tied control has zero common mode by construction and branch RMS
  1.755 mm. Large common motion appears already at cps8.
- Raw-positive coverage is originally about 31% per image, versus 72-73%
  at the A1 midpoint and 84-86% for A2. In the inherited NCC9 arithmetic,
  all-zero windows have correlation zero and loss one. Joint coordinate
  deformation changes spatial weighting of informative and empty windows.
  Observed coverage and common motion are consistent with this objective-level
  incentive, not a failed checkpoint replay. No mask or correction was added.
  A balance-only penalty cannot suppress equally large shared velocities;
  amplitude control is the recommended next ablation, subject to approval.
- Swapping inputs does not reproduce an identical optimized field: foreground
  endpoint difference mean/p95/max is 0.272/0.964/14.785 mm, with different
  adaptive stopping epochs. CPU symmetry/gradient tests pass, but this does
  not establish numerical exchange equivariance of the CUDA optimization.
  Its cause is not proven to be CUDA nondeterminism alone. Separately, A1
  forward/reverse off-grid inverse-cycle mean errors are 0.118/0.106 mm.
- All four full-size smokes/replays, 31 tests, 30 reported metric checks and
  source/output hashes passed. Production/replay queue took 386.02 s; optimizer
  times span 55.5-274.3 s, peak allocated/reserved memory about 4.41/4.73 GiB.
  Existing jobs were untouched; shared-load timing is diagnostic only.
- Artifact key: `oasis_seed3_rank000_bspline_two_total_svf_midpoint_ablation_v1`,
  subdirectory `phase_a`. CSV/JSON/Markdown, paired differences, stage energies,
  7/8/10-step sensitivity, inverse/export diagnostics and acceptance are saved
  externally. Phase B had not started at this acceptance; see P78 for follow-up.
  Single exploratory case, no sample SD
  or generalization claim; numerical acceptance does not imply success on
  accuracy or exact symmetry.

## First10 bounded two-map 8→4→2 lambda sweep (2026-10-15, P76)

- Accepted 70 fresh registrations and independent replays: ten seed-3 pairs
  at each global diffusion lambda 0.01/0.1/0.3/1/3/10/100. Six bounded maps,
  unchanged NCC9/P70 objective and schedules, no cps1 or dense stage. Both
  maps at every level receive the full selected lambda.
- Lambda 1 is best on all three mean accuracy metrics: Dice
  0.819183 ± 0.033157, HD95 1.764939 ± 0.309214 mm, symmetric ASD
  0.364375 ± 0.088790 mm. FG SD(log J) is 0.603642 ± 0.034641 and
  folding is 0.000037 ± 0.000048%. All dispersions are sample SD, n=10.
- SITReg lambda 1 has slightly better accuracy and lower SD(log J):
  0.821546/1.715898/0.350896 and 0.569473, respectively; its measured
  folding is 0.000423%. Thus the new lambda-1 point has lower measured
  folding, not uniformly better accuracy or regularity.
- Increasing lambda above 1 reduces deformation variation but loses accuracy.
  At lambda 3, Dice/HD95/ASD are 0.806586/1.848633/0.401794, SD(log J)
  is 0.477905 and measured foreground folding is zero. Lambda 10/100 also
  have zero measured foreground folding but larger accuracy losses. These
  observations do not establish unconditional invertibility.
- Queue completed without failures in 22.4 minutes, with up to two workers
  per GPU 0–3. Lambda-1 optimizer/worker/replay mean times are
  67.66/96.41/23.50 s. Production allocated/reserved peaks are about
  2.49/3.10 GiB. Existing concurrent workloads make timing diagnostic only.
- Three full-size smokes, 51 regression tests, 420 saved maps, 2450 label
  records and 101920 history rows verified. Per-case data, paired differences,
  fresh-versus-P70 lambda-1 reproduction and runtime/memory diagnostics saved.
- ARC C/F now show the seven-point bounded-two sweep; G retains its four-level
  lambda-1 point. All 550 displayed metric mean/SD values independently
  verified, 18 axes checked, figures visually reviewed. Existing axis ranges
  suffice, including the tighter F/G means-only rows. Historical A/B/D/E and
  all prior artifacts remain unchanged.
- Artifact key: `oasis_seed3_first10_bspline_bounded2_842_ncc9_lambda_sweep_v1`.
  CSV/JSON/Markdown and acceptance are external; updated PNG/PDF/SVG and A–G
  PDF book are in `arc_panels_v1`. This is exploratory first-ten test tuning,
  not an unbiased generalization result.

## First10 NCC9 method-family ARC figures (2026-10-15, P75)

- Accepted aggregation-only release: five2×3 figures with identical axis
  limits/ticks/labels, mean-only top and marginal sample-SD ellipses below.
  Columns are Dice/HD95/symmetric ASD versus fixed-foreground positive-J SDlogJ.
- A compares three DL NCC9 sweeps; B compares three pyramid displacement
  representations; C/D compare matched P70 unbounded-one/bounded-one/bounded-two
  retained composition at842/8421 andlambda1; E compares four refined-SVF
  sweeps from complete P74 continuation_v2. C/D are markers, not invented curves.
-70 operating points/700 canonical first10 case records; all700 mean/SD values
  independently reproduced. Ten tests,30 axis checks, source/output hashes,
  PDF/PNG/SVG structure and visual review passed. No new registration or metric
  reevaluation. DLlambda1 styling is uniform; historical provenance remains in
  README/data. Folding is tabulated separately, not inferred from SDlogJ.
- Key:oasis_seed3_first10_ncc9_method_family_arc_panels_v1. Five PNG/PDF/SVGs,
  five-page PDF, CSV/JSON data and acceptance are external. This is exploratory
  presentation of existing evidence, not a new generalization claim.
- Lambda1 appendix provides11 rows: three DL references and842/8421 pairs
  for prolonged displacement, bounded-one, bounded-two and velocity-penalized
  refined SVFs. All110 cases and110 mean/SD statistics independently verified;
  Markdown/CSV/JSON saved under lambda1_table_v1 without changing prior outputs.
- F/G extension accepted: each compares three DL sweeps with displacement,
  bounded-two and velocity-regularized SVF at842/8421. Sixty displayed settings,
  600 records (450 unique),600 independent statistic checks and12 shared axes
  verified. Individual PNG/PDF/SVG, F–G and A–G PDF books are in panels_FG_v1;
  original A–E pages and lambda1 table preserved. No new numerical experiments.
- F/G presentation revision accepted under panels_FG_sitreg_v2: SITReg is the
  only DL curve; displacement, bounded-two and velocity-SVF retained. Distinct
  solid/dashed/dash-dot curves and black diamond; single-row four-method legend.
  Top y ranges tightened to Dice0.62–0.85, HD951.6–5.0mm and ASD0.30–1.12mm,
  shared across F/G with all20 points visible per figure. Bottom SD ranges and
  all x axes unchanged. Independent400-statistic/12-axis/hash/PDF checks and
  visual review passed;21 regression tests. New two-/seven-page PDF books,
  PNG/PDF/SVG and CSV/JSON; original A–G and lambda1 table remain unchanged.

## First10 refined B-spline SVF sweep completed (2026-10-15, P74)

- Accepted280 registrations: ten seed3-shuffled cases,842/8421 schedules,
  velocity/displacement diffusion, and seven global stage weights
  0.01/0.1/0.3/1/3/10/100. All28 configuration summaries use ten unique cases.
- Two low-lambda displacement8421 runs required a versioned numerical
  transition proof and continuation of only cps1 from their own verified cps2
  controls. FP32 objective, optimizer and seven-step integration are unchanged;
  their full four-stage histories and7/8/10 diagnostics independently replayed.
  All278 other accepted fits and every failure artifact remain unchanged.
- The complete continuation_v2 release includes four sweep tables, accuracy/
  regularity ARC lines, per-case/per-label data, stage/integration diagnostics,
  resource summaries and mixed-concurrency/continuation timing disclosures.
  All11619 preserved files and final report hashes verified. High measured
  folding at weak regularization remains reported, not filtered or repaired.
- Artifact key:oasis_seed3_first10_bspline_svf_refine_842_8421_ncc9_lambda_sweep_v1;
  final reports:ARTIFACT_ROOT/repo_experiments/<key>/continuation_v2/.
  This is exploratory first10 evidence, not an unbiased generalization estimate.

## Test100 label-wise accuracy–regularity curves (2026-10-14, P72)

- Aggregation-only accepted release `oasis_seed3_test100_label_arc/releases/v001`:
  main19 NCC9/diffusion settings plus four saved MR-D-BSCP8→4→2 prefixes.
  All2300 fields are present;80500 saved label slots and805 label summaries
  are retained. No registration, inference or field-metric evaluation ran.
- Official label34 Right-Vessel excludes ranks35/58 (98 common cases);
  the other34 labels use100. Both axes share the label-specific case subset.
  X remains whole fixed-raw-positive foreground SD(log J), not within-label
  regularity. Folding is separately tabulated. Historical DLlambda1 is marked.
- MR dense best mean-Dice lambda differs by structure:0.3 for14 labels,
  1 for10,3 for11. This is descriptive test-set analysis, not per-label field
  selection. At matched lambda1/3, refinement improves mean Dice for35/35
  labels, HD95 for30/35 and31/35, and ASD for34/35 and35/35, respectively.
- Two35-page2x3 PDF books,70 individual PNG/SVG pages and six overview atlases
  compare main methods and prefix addition. Ellipses show marginal sample SD,
  not confidence regions. Full tables, exclusions, lambda trends, measured-point
  Pareto flags and paired prefix differences accompany the release.
- Ten tests passed, all8050 independently reconstructed mean/SD values match
  exactly, original19 case-level ARC summaries match and180 output hashes pass.
  Source package: `experiments/2026-10-14_oasis_test100_label_arc/`.

## Bounded three-map composition by pyramid depth, first10 (2026-10-14, P71)

- Added40 fresh bounded-three-map registrations to the accepted P70 four-setting
  depth comparison. Schedules are8,8→4,8→4→2,and8→4→2→1; every active stage
  uses λ=1. The existing NCC9/diffusion objective and bounded-control scheme
  remain fixed. Each of the three maps at a level gets a full fresh-Adam stage
  schedule, so this is a capacity/update-budget extension, not an equal-compute
  comparison.
- Bounded-three-map mean±sampleSD Dice/HD95/ASD: cps8
  0.7169±0.0414 /2.4593±0.3407 /0.6667±0.1241mm; cps8→4
  0.7941±0.0292 /1.9280±0.2888 /0.4314±0.0833mm; cps8→4→2
  0.8244±0.0294 /1.7698±0.3035 /0.3542±0.0801mm; cps8→4→2→1
  0.8322±0.0288 /1.7013±0.3248 /0.3330±0.0816mm.
- Full-depth bounded-three-map versus bounded-two-map: Dice0.8322 vs0.8288,
  HD951.7013 vs1.7015mm, ASD0.3330 vs0.3397mm; foreground SD(logJ)0.7047 vs
  0.6509 and folding0.0209 vs0.0031%. It modestly improves accuracy while
  increasing log-Jacobian spread and measured folding. The slight nonzero fold
  rate is compatible with contraction-checked elementary maps: that check is
  not a guarantee that the exported composed field has zero finite-difference
  folds.
- Full-depth bounded-three-map mean accuracy exceeds original VFA/SITReg:
  Dice0.8322 vs0.8198/0.8215, HD951.7013 vs1.7743/1.7159mm, ASD0.3330 vs
  0.3669/0.3509mm. Its FG SD(logJ)0.7047 is higher than VFA/SITReg0.6208/0.5695;
  folding0.0209% is below VFA0.2186% but above SITReg0.0004%. Across-case SDs
  for the three accuracy metrics remain slightly higher than both DL references.
- Four full-grid smokes and all40 independent replays passed. All300 map
  checkpoints reconstructed and passed elementary bound checks. The combined
  report has28 summary rows,280 per-case rows,400 paired reference differences,
  and900 map-stage rows. P70 report hashes were verified unchanged. First10 is
  exploratory due previous tuning.
- Artifact key:
  `oasis_seed3_first10_mrdbscp8421_bspline_bounded3_depth_extension_v1`;
  report is under `${ARTIFACT_ROOT}/repo_experiments/<artifact-key>/`.
  Detailed four-table report and audits are recorded in the matching experiment
  technical log.

## Two bounded increments, first10 lambda1151/1111 (2026-10-13, P69)

- Completed20 case/vector results:18 fresh identity registrations plus2
  read-only accepted P68 lambda1151 pilots. Cps8/4/2/1 with two bounded
  increments at each level,8 retained maps; only cps2 lambda changes5→1.
  No source changes, extra configurations, failures or retries during execution.
- Lambda1151 mean±sampleSD: Dice0.821953±0.037352, HD951.724650±0.320593mm,
  symmetric ASD0.358358±0.104480mm, FG SDlogJ0.581739±0.030237,
  folding0.000381±0.000225%. Lambda1111:0.828756±0.031502 /
  1.678458±0.316117mm /0.338309±0.086565mm /0.650974±0.037057 /
  0.003143±0.001225%.
- Lower cps2 lambda improves mean Dice0.006803, HD950.046191mm and
  ASD0.020049mm. Paired accuracy wins are9/10 Dice,9/10 HD95,10/10 ASD.
  Rank1 Dice and rank6 HD95 are the exceptions. FG SDlogJ and folding
  increase in all10 cases; mean changes+0.069235 and+0.002762 percentage points.
- Original VFA/SITReg mean Dice0.819772/0.821546, HD951.774341/1.715898mm,
  ASD0.366851/0.350896mm, FG SDlogJ0.620839/0.569473, folding0.218560/
  0.000423%. Lambda1151 exceeds VFA mean accuracy and has SITReg-like
  regularity, with slightly worse HD95/ASD than SITReg. Lambda1111 improves
  all three mean accuracy metrics over both DL references, but is less
  regular than1151 and still has higher across-case accuracy SDs than DL.
- Rank4 remains difficult:1111 Dice0.750952 versus VFA0.760943/SITReg0.768742.
  Do not drop it from the main table. The separately reported nonpilot8
  (excluding previously selected ranks0/4) have1151/1111 mean Dice0.836507/
  0.840620, HD951.603709/1.557724mm, ASD0.316866/0.304859mm. These are
  diagnostic exploratory subsets, not an unbiased held-out benchmark.
- Historical one-map all-bounded mean Dice0.802831, HD951.964730mm,
  ASD0.427594mm, SDlogJ0.470407 and folding0%. Two maps substantially
  recover accuracy but no longer yield exactly zero exported-field folding.
- All47 tests and two full-size smokes pass.20 accepted independent audits
  cover160 block reconstructions, losses, per-label metrics and elementary
  corner checks. All40 shared cps8/cps4 block comparisons are byte-identical.
  Every retained-chain100k probe set has0 nonpositive Jacobians; maximum
  exported-interpolant probe fractions are0.011%/0.045% for1151/1111.
  These diagnostics do not imply unconditional export topology guarantees.
- Mean optimization151.91/153.56s, complete worker230.65/225.40s,
  peak3.458GiB; shared-load timings are diagnostic. Historical inputs,
  sources and accepted outputs remain unchanged; final hashes rechecked.
- External key `oasis_seed3_first10_mrdbscp8421_tanh_two_increments_lambda1151_1111_v1`;
  complete CSV/JSON/Markdown accepted2026-10-14 06:55:43UTC (Oct13 PDT).
  [Spec/log](../experiments/2026-10-13_oasis_two_increments_first10/technical_log.md).

## Two bounded increments per level, ranks0/4 (2026-10-13, P68)

- Four fresh identity registrations: two bounded maps per cps8/4/2/1 versus
  one map per level with two fresh-Adam continuation blocks. Both have eight
  schedule blocks, lambda[1,1,5,1], unchanged NCC9/total-field diffusion,
  bounds and LR schedules. Retained map counts are8 versus4. Allowed budgets
  are matched, not actual adaptive updates, FLOPs or runtime.
- Rank0 two-map-arm Dice/HD95/ASD0.798510/1.999331/0.429354mm, FG SDlogJ
  0.617426 and folding0.000743%. Same-map control0.780452/2.193289/0.484875,
  SDlogJ0.492586, folding0%. Historical one-block all-bounded Dice0.774069.
  Two maps improve all three accuracy metrics over original VFA; versus
  SITReg, Dice is0.000209 lower, HD95 is0.061988mm lower and ASD0.006290mm
  higher. These are metrics of one field, not separately selected outputs.
- Rank4 two-map-arm0.728966/2.417488/0.619302mm, SDlogJ0.621042 and
  folding0.000300%. Same-map control0.687008/3.181210/0.878093, SDlogJ
  0.499194 and folding0%. Historical all-bounded Dice0.684704; VFA/SITReg
  Dice0.760943/0.768742. The difficult-case DL accuracy gap remains.
- Extra maps improve Dice over continuation by0.018058/0.041959, HD95 by
  0.193958/0.763721mm and ASD by0.055521/0.258791mm on ranks0/4. This
  supports adding bounded composition capacity beyond extra optimization
  alone, while also changing the parameterization and optimization path.
- At both cases the second cps2 increment (lambda5) lowers objective but
  worsens Dice (-0.005023/-0.010309), HD95 and ASD. The second cps8/cps4/
  cps1 increments improve all three accuracy metrics. No lambda retuning
  or additional capacity was run to address this observation.
- Every elementary corner bound passes. Same100k foreground off-grid probes
  give0 nonpositive retained-chain Jacobians; interpolated exports have4/3
  nonpositive probes. Canonical folding percentages above use a different
  finite-difference evaluation. Very low folding is not exactly zero and
  chain validity is not an unconditional exported-interpolant guarantee.
- All43 tests passed/1 skipped, both full-size smokes passed, and all32
  checkpoints/losses/fields/per-label metrics/corner audits plus4 final
  off-grid diagnostics were independently replayed. No failures/retries.
  Two-map optimization138.58/203.47s, peak3.458GiB; controls128.21/126.77s,
  peak3.150GiB. Shared workloads remain active; timings are diagnostic.
- External key `oasis_seed3_rank000_004_mrdbscp8421_tanh_two_blocks_lambda1151_v1`;
  per-case CSV/JSON/Markdown accepted. These are two selected exploratory
  cases, not first-ten results or a population/generalization claim.
  [Spec/log](../experiments/2026-10-13_oasis_bounded_two_blocks/technical_log.md).

## Selectively bounded stages, first10 (2026-10-13, P67)

- Frozen P64/P66 four variants, cps8/4/2/1 retained composition and weights
  [1,1,5,1]. All36 fresh ranks1–9 registrations completed;4 accepted rank0
  outputs reused unchanged. No failures/retries or hyperparameter tuning.
- Last1/last2/last3/all4 mean Dice0.813035/0.821291/0.812331/0.802831;
  HD951.870758/1.813396/1.923744/1.964730mm; symmetric ASD0.397113/
  0.364367/0.390775/0.427594mm.
- Same-ten original VFA/SITReg mean Dice0.819772/0.821546, HD951.774341/
  1.715898mm, ASD0.366851/0.350896mm. Last2 has the strongest mean
  accuracy of the four new variants and wins Dice in8/10 cases; last1 wins
  ranks1/2. Rank0's accuracy ranking is not universal.
- Last2 spatial FG SDlogJ0.618354 nearly matches VFA0.620839, but folding
  is1.203616% versus VFA0.218560% and SITReg0.000423%. Positive-J spread
  must not substitute for the separate folding endpoint. Last1/last3 fold
  0.747005/0.386588%; all4 has measured0% in all10 cases at lower accuracy.
- Last2 across-case sample Dice SD0.031516 is above VFA0.026892 and
  SITReg0.024232. Rank4 remains difficult: last2 Dice0.746120 versus
  all4 Dice0.684704. Reports retain the case and show ranks1–9 separately.
- All40 common-prefix comparisons are byte-identical; bounded elementary
  corner norms remain<1. These checks do not imply unconditional topology
  guarantees for exported interpolants. Runtime is shared-load diagnostic.
- Independent replay passed all160 checkpoint reconstructions and all40
  final per-label/accuracy/regularity metric comparisons without discrepancies.
- External key `oasis_seed3_first10_mrdbscp8421_selective_tanh_lambda1151_v1`;
  full per-case/mean/sample-SD CSV/JSON/Markdown and stage diagnostics.
  [Spec/log](../experiments/2026-10-13_oasis_selective_bounded_first10/technical_log.md).

## Selectively bounded stages, rank0 (2026-10-13, P66)

- Three fresh8/4/2/1 compositional fits, weights[1,1,5,1], bounding cps1 only,
  cps2/1, or cps4/2/1. Other settings match P64; unbounded coefficients are
  direct normalized parameters, not a reused historical prolonged prefix.
- Last1: Dice0.789160,HD952.113888mm,ASD0.466397mm,FGSDlogJ0.610573,
  folding1.021628%. Last2:0.801810/1.975920/0.422405/0.668009/1.621929%.
  Last3:0.792172/2.023935/0.448346/0.562964/0.378766%.
- Last2 has the best new accuracy and exceeds original same-case VFA/SITReg
  on Dice/HD95/ASD. It does not match DL folding:1.621929% versus0.255035%/
  0.000697%. Last3 improves all five metrics over last1 and has lowest new
  folding, still above both DL references. P64 all-bounded has0% with lower
  Dice0.774069. These remain single-case accuracy/regularity tradeoffs.
- Shared unbounded cps8 already folds0.543802%; shared last1/last2 cps4
  folds2.225994%. At cps2 lambda5, unbounded controls reduce folding further
  but lose Dice relative to bounded controls. Later bounding cannot establish
  whole-chain invertibility. Positive-J SDlogJ excludes folds and must not
  substitute for the separate folding endpoint.
- All29 tests and three full-size smokes passed; no failed/retried fits.
  Optimization66.47/64.50/64.73s, peak2.678GiB; shared-load timing diagnostic.
- Independent audit passed for all12 saved stages, canonical metrics and
  100k off-grid probes. Shared initial stages have identical fields/histories;
  numerical sources and historical outputs remain unchanged.
- External key `oasis_seed3_rank000_mrdbscp8421_selective_tanh_lambda1151_v1`;
  [spec/log](../experiments/2026-10-13_oasis_selective_bounded/technical_log.md).

## Bounded cps16 extension, rank0 (2026-10-13, P65)

- Fresh16/8/4/2/1 retained bounded composition, lambda[1,1,1,5,1], one
  increment per stage. Existing P64 downstream loss/optimizer schedules remain
  unchanged; new cps16 uses downsample8, LR0.03 and400/40 adaptive budget.
- Dice/HD95/ASD0.770576/2.379341/0.532366, foreground SDlogJ0.506026 and
  folding0%. Relative to bounded8421 P64: Dice-0.003493, HD95+0.154360mm,
  ASD+0.035322mm, SDlogJ+0.012556. No accuracy improvement on this case.
- Cps8 coefficient saturation>95% falls40.726% to35.302%; new cps16
  saturation23.934%. More coarse motion capacity did not resolve the gap.
  This also changes coarse image support and optimization path, so saturation
  is not causally isolated. No additional tuning or scope expansion was run.
- Final loss improves0.754680 to0.750544 and NCC9 data loss0.731728 to
  0.725302 while label accuracy worsens; objective improvement alone is not
  evidence that the segmentation-metric gap has closed.
- All five elementary contraction audits pass. Same100k off-grid foreground
  points give zero nonpositive Jacobians for both chain and export; minima
  0.043210/0.046213. No unconditional export or generalization guarantee.
- Optimization66.85s, complete worker123.18s, peak2.833GiB onGPU3;
  stop275/300/175/150/500, final best495. Shared-load timing is diagnostic.
- All29 tests, full-size smoke and independent checkpoint/loss/35-label/
  topology replay passed. Historical numerical sources and outputs unchanged.
- External key `oasis_seed3_rank000_mrdbscp168421_tanh_composed_lambda11151_v1`;
  [spec/log](../experiments/2026-10-13_oasis_bounded_composed_cps16/technical_log.md).

## Bounded compositional B-splines, rank0 (2026-10-13, P64)

- One fresh identity-initialized bounded increment per cps8/4/2/1, weights
  [1,1,5,1], same NCC9/diffusion loss and nominal normalized LR schedule.
  This is a retained composition of four elementary maps, not clipping or
  reusing P63's prefix. Sampler-specific FP64 bounds use0.99 safety margin.
- Rank0 Dice/HD95/ASD0.774069/2.224981/0.497044; FG SDlogJ0.493470 and
  canonical foreground folding0%. Unbounded P63 is0.805363/1.983874/0.411986,
  SDlogJ0.802908, folding3.718802%. Thus Dice falls0.031294 and HD95/ASD
  increase0.241106/0.085058mm: folding is reduced, accuracy is not matched.
- All elementary-map maximum derivative row sums are below1
  (0.977792/0.984813/0.988912/0.989734). Retained-chain and interpolated-export
  probes each have0 nonpositive Jacobians among100,000 shared off-grid
  foreground locations; respective minimum determinants0.045250/0.045843.
  These measurements do not assert a universal exported-field guarantee.
- Coefficient fractions above95% of their bounds are40.726%/28.417%/14.054%/
  16.320% across cps8/4/2/1. Restricted motion capacity and saturation are
  plausible contributors, not causally isolated explanations from one run.
- Stopping epochs400/200/150/500; best400/200/150/496. Total optimization
  63.29s, complete worker109.82s, peak2.678GiB, physical GPU3. Timing is
  diagnostic under shared workloads. No additional increments or tuning.
- External artifact key:
  `oasis_seed3_rank000_mrdbscp8421_tanh_composed_lambda1151_v1`;
  spec/log: `experiments/2026-10-13_oasis_bounded_composed/`.

## Rank0 cps1 B-spline versus dense after [1,1,5] prefix (2026-10-13, P63)

- Same saved prefix, final lambda1, local squared NCC9, expanded-total-field
  finite-difference voxel diffusion, normalized LR0.001 and adaptive schedule.
  Final representation alone switches from dense voxels to cubic cps1 controls.
- Rank0 (fixed0333/moving0115) cps1 versus dense Dice/HD95/ASD:
  0.805363/1.983874/0.411986 versus0.804582/1.995025/0.411972.
  Dice improves0.000782 and HD95 improves0.011151mm; ASD differs+0.000014mm.
- Folding is not improved: cps1 foreground folding3.718802% versus dense
  3.695896% (+0.022906 percentage points); foreground SDlogJ rises from
  0.799019 to0.802908. Original same-case VFA/SITReg folding is
  0.255035%/0.000697%. These are individual-case values, not cohort means.
- Thus cps1 matches dense accuracy on this case but does not solve folding.
  Neither the cubic representation nor diffusion enforces Jacobian positivity.
- Cps1 stopped725/best719 versus dense675/best662; final-stage optimization
  58.78s versus54.02s under different shared-GPU conditions, diagnostic only.
- Artifact key: `oasis_seed3_rank000_mrdbscp8421_lambda1151_vs_dense_v1`;
  spec/log: `experiments/2026-10-13_oasis_cps2lambda5_cps1/`.

## Cps2 lambda5 prefix to dense refinement (2026-10-13, P62)

- All70 fresh fits passed acceptance after frozen [1,1,5] cps8/4/2 prefixes;
  seven dense weights0.1/0.2/0.5/1/2/5/10, unchanged NCC9/diffusion and LR0.001.
- Dense lambda0.5 wins all three mean accuracy metrics: Dice
  0.834955 ± 0.029999, HD951.681926 ± 0.358336mm, symmetric ASD
  0.331114 ± 0.096987mm. Foreground SDlogJ/folding is
  0.819045 ± 0.053222 / 3.879464 ± 0.713840% (sample SD, first10).
- Against P60's [1,1,1] prefix at the same dense lambda0.5, Dice is0.000817
  lower, HD95/ASD improve0.056620/0.003405mm and folding drops0.583381
  percentage points. All six shared weights reduce folding, but the change
  is not a uniform win across all accuracy/regularity metrics.
- The selected point has better mean accuracy than original VFA/SITReg, but
  higher folding and larger accuracy SDs. The regular prefix's0.497181%
  folding rises to3.879464% under selected dense refinement. Dense lambda10
  gives0.024494% folding at lower Dice0.773654. This remains an accuracy/
  regularity tradeoff, not matched DL regularity.
- No failures/retries; mean dense optimization/total worker52.38/68.91s.
  Independent statistics, per-label, prefix and paired-comparison checks passed.
  Exploratory test-case tuning; no statistical superiority claim.
- Accepted artifact key:
  `oasis_seed3_first10_mrdbscp842_prefix115_dense_lambda_sweep_v1`;
  spec/log: `experiments/2026-10-13_oasis_cps2lambda5_dense/`.

## Fixed cps8/cps4 lambda1 to cps2 diffusion sweep (2026-10-13, P61)

- All60 fresh cps2 fits passed acceptance, with the same per-case saved
  cps8/cps4 lambda1 prefix and no dense refinement. Six cps2 weights
  0.1/0.2/0.5/1/2/5; unchanged NCC9/diffusion and normalized LR0.003.
- Lambda0.5 selects the best mean Dice (0.829577 ± 0.029780) and symmetric
  ASD (0.350710 ± 0.099560 mm). Lambda1 has the best mean HD95
  (1.787126 ± 0.380951 mm). Mean ± across-case sample SD, same ten pairs.
- Compared withlambda1, lambda0.5 gains0.003281 Dice and lowers ASD0.005604mm,
  but increases HD950.019762mm and foreground folding1.540182 percentage
  points (3.676989% to5.217171%). Stronger diffusion reduces folding but
  trades away accuracy; lambda5 Dice/folding is0.797722/0.497181%.
- All ten freshlambda1 fields and five metrics exactly reproduce original842.
  No failures/retries; mean cps2 optimization25.88s, total worker39.56s.
  Shared-load runtime is diagnostic and first-ten tuning is exploratory.
- Accepted reports use external artifact key
  `oasis_seed3_first10_mrdbscp84_prefixlambda1_cps2_lambda_sweep_v1`;
  see `experiments/2026-10-13_oasis_fixed_prefix_cps2/technical_log.md`.

## Fixed-lambda1 prefix to dense diffusion sweep (2026-10-13, P60)

- All 60 fresh first-ten fits passed acceptance: six dense weights after the
  same per-case cps8/4/2 lambda1 prefix, unchanged local squared NCC9 and
  total-field voxel diffusion, normalized dense LR 0.001.
- Dense lambda0.5 maximizes mean Dice (0.835772 ± 0.028822) and minimizes
  mean symmetric ASD (0.334519 ± 0.096449 mm). Dense lambda1 has the lowest
  mean HD95 among the sweep (1.734039 ± 0.401562 mm).
- Accuracy gains are not regularity equivalence: lambda0.5 has foreground
  SDlogJ 0.830600 ± 0.054856 and folding 4.462845 ± 0.785205%. Even lambda5
  has folding 0.682494 ± 0.204216%, above original VFA (0.218560%) and
  SITReg (0.000423%), while its mean Dice drops to 0.804058.
- Fresh lambda1 reproduces all ten historical DVF hashes and all five
  reported metrics exactly. No failures/retries; mean dense optimization
  43.97 s and total worker time 61.35 s, under shared workloads.
- Statistics use across-case sample SD and the same ten reference pairs.
  This is exploratory test-case tuning. The accepted CSV/JSON/Markdown
  reports are under external artifact key
  `oasis_seed3_first10_mrdbscp842_prefixlambda1_dense_lambda_sweep_v1`;
  see `experiments/2026-10-13_oasis_fixed_prefix_dense/technical_log.md`.

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

## OASIS Rank0 B-Spline SVF Pilot (2026-10-15 UTC)

- P73 accepted8/8 fresh fits after8 native-size smokes, independent replay of
  every fit,28 stage-state checks and7/8/10-step reintegration diagnostics.
  A report-only historical-runtime formatting recovery required no reruns or
  numerical source changes. Prior registration artifacts remain read-only.
- Refined single-SVF8421 with velocity diffusion gives Dice/HD95/ASD
  `0.804515/1.990093/0.417839`, better than original VFA and SITReg on all
  three accuracy metrics for this case. FG SDlogJ is0.606455; folding is
  0.001394%, versus VFA0.255035% and SITReg0.000697%. This is near-zero
  measured folding, not zero or a global diffeomorphism guarantee.
- Refined displacement-regularized8421 reaches0.810485/1.883917/0.392670
  but folds1.406991%; composed displacement-regularized8421 reaches
  0.810081/1.901635/0.388712 but folds0.913555%. Composed active-velocity
  penalties give worse HD95/ASD than the refined-total-velocity arms.
  Equal lambda on velocity/displacement does not impose equal regularity.
- Increasing integration from7 to10 steps does not resolve residual folding.
  One of28 scaled initial maps exceeds the sufficient derivative bound1;
  all tested initial-map corner determinants remain positive. Retained-map
  off-grid topology and sampled-export topology are separately recorded.
- The two full comparison tables, paired differences, histories, times,
  inverse residuals, source hashes and acceptance record are under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_rank000_bspline_svf_842_8421_ncc9_lambda1_v1/`.
  This is one exploratory case, with no across-case SD or generalization claim.

## Source Reports

- `docs/reports/current_results_audit_and_plan.md`
- `docs/reports/reorganized_paper_points.md`
- `docs/reports/protocol_warnings/lungct_idir_ptvreg_protocol_bias.md`
- `docs/reports/protocol_warnings/ptvreg_technique_audit_directcp_multistage.md`
