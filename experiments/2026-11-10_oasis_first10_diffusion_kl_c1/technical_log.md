# Technical log

## Known limitations

Exploratory tuning on the same shuffled first10 used for earlier comparisons;
not a held-out confirmation. Gamma0 metrics/resources are audited historical
P84 records, not fresh timing controls. Cold/warm rank0 repeats are not extra
subjects. Central and analytic sampled Jacobians and digital NDV are different
geometric checks, not invertibility proofs. No within-stage LR reduction is
added to the existing stage-rate/plateau protocol.

## 2026-11-10 authorization and implementation

- User authorized gamma0/1e-4/1e-3/1e-2/0.1/1 on shuffled ranks0..9,
  with diffusion lambda1 and analytic KL-C1. Both diffusion backends retained.
- New scope adapter imports P85's exact objective/optimizer and P84's loader,
  geometry, evaluation and telemetry. Historical numerical files are immutable.
- 110 new production observations plus22 historical references;120 primary
  warm case endpoints in12 settings. Four native-volume preflight observations.
- Execution `oasis_first10_diff1_kl_20261110T213230Z_v1` initialized21:32:20UTC;
  preflight accepted21:33:35UTC on2026-11-10. All26 tests passed, including
  exact zero-gamma parity and first10 adapter value/gradient identity. Both
  complete historical11-observation cohorts passed input/source/field/model
  audits. Four native-volume endpoint smokes passed independent replay/scoring.
- Peak smoke reserved memory2.535GiB; measured artifact projection19.61GB
  including30% margin, within admitted capacity. Same pinned Docker image,
  UID:GID and four RTX6000 Ada GPUs as P84/P85; no competing GPU jobs.
- Production finished21:58:57UTC:110 new registration/scoring observations in
  ten blocks, plus22 audited historical gamma0 records and4 smokes. All136
  scored, with all120 primary warm case endpoints retaining35 labels. Every
  production stage met the plateau rule without hitting its update cap.

## Independent audit and interpretation

- External analysis completed22:00:09UTC. It verified26 frozen source hashes,
  all input/field/model/metric identities,53554 per-step component states,
  no worker/GPU contamination or failure, and exact cold/warm rank0 fields
  for all12 settings. The ten positive-gamma rank0 fields also match P85's
  independently executed pilot byte for byte. The complete posthoc receipt,
  case/label tables, paired deltas, resources and plots are external under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_first10_diffusion1_analytic_kl_c1_gamma_v1/oasis_first10_diff1_kl_20261110T213230Z_v1/`.
- FD gamma1e-4 has the largest tested mean Dice0.826404 versus gamma0
  0.826296, a paired +0.0108 percentage points:5/10 case wins and a
  percentile paired case-bootstrap95% CI[-0.0819,+0.1042]pp. HD95/ASD
  worsen slightly. Analytic diffusion's largest mean Dice remains gamma0
  0.826191. The tiny FD gain is exploratory and not a general accuracy result.
- Gamma0.01 is an accuracy/geometry compromise:FD/analytic Dice
  0.825192/0.824987 (loss0.110/0.120pp versus matched gamma0), mean NDV
  -79.9/-79.6%, analytic SDlogJ -3.85/-3.82%, analytic sampled folds
  4.709→2.042% /4.568→2.004%. HD95 improves0.026/0.041mm and ASD changes
  -0.00142/+0.00019mm. FD paired Dice CI[-0.1868,-0.0363]pp; analytic
  CI[-0.2845,+0.0502]pp. Both lose Dice in8/10 pairs.
- Gamma0.1 markedly suppresses geometry failure:analytic SDlogJ falls
  about36%, sampled analytic folding to0.054% and NDV about98%. FD/analytic
  Dice loses0.814/0.884pp and ASD worsens0.0180/0.0198mm. Gamma1 loses
  about5.87pp Dice in all10 matched pairs. Sampled central/analytic folding
  reaches0% but NDV remains positive; no invertibility guarantee follows.
- Analytic diffusion generally has slightly lower analytic SDlogJ, folding
  and NDV than FD diffusion at matched gamma, but does not consistently improve
  Dice/HD95/ASD. At gamma0.1 its Dice is0.080pp below FD with an exploratory
  paired CI[-0.120,-0.036]pp; elsewhere Dice CIs mostly span zero.
- Positive-gamma warm registration means across gamma:FD27.97-30.56s,
  analytic24.71-25.93s. Historical gamma0 means22.23/22.65s. Cold values,
  per-case warm time ranges, startup/load/output clocks, updates/stage costs,
  Torch allocated/reserved and200ms process GPU/RSS are in the external report.
  Warm max Torch allocated/reserved:FD2.154/2.479GiB;
  analytic1.930/2.516GiB. Registration process peaks2.967/3.004GiB;
  independent scoring peaks5.527GiB. Scoring uses ten unique fields per
  eleven-observation block because cold/warm rank0 fields are identical.
- Means/sample SD summarize ten case metrics; percentile bootstrap resamples
  paired cases20000 times with a fixed seed. These are exploratory intervals
  after choosing gamma on the same first10, with no multiplicity correction.
  Neither the first10 optimum nor the CIs establish held-out generalization.
