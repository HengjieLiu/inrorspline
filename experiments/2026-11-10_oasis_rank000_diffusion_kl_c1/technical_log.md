# Technical log

## Known limitations

Rank0 exploratory tuning only; no population or held-out conclusion. Diffusion
backend and analytic-KL backend are explicit. Gamma0 is a validated historical
reference, including historical time/memory. Cold/warm is not a case-level
replication sample. Analytic/central Jacobians and NDV remain complementary,
sampled geometric checks, not invertibility guarantees.

## 2026-11-10 authorization and implementation

- User requested NCC9 + diffusion(lambda1) + gamma*KL-C1 for gamma0/1e-4/1e-3/
  1e-2/1e-1/1 on first case fixed0333/moving0115 only.
- Carry forward FD and analytic diffusion; choose analytic KL for geometric
  consistency, with epsilon0.05/kappa120 unchanged.
- New adapter reuses the frozen geometry, loader, stage stopping, telemetry
  and evaluation paths; historical P84 source files and artifacts stay unchanged.
- Separate component accounting and gamma0 bypass/parity checks are required.
- Historical numerical files and P84 artifacts were not edited. The new source
  and execution spec were frozen in the external run snapshot before launch.

## Completed execution

- Execution `oasis_rank000_diff1_kl_20261110T201710Z_v1`, started20:17:10UTC
  and accepted20:23:19UTC on2026-11-10. Healthy pinned-image experiment runtime
  from P84; four RTX6000 Ada GPUs, one job perGPU, separate same-GPU scoring.
  The original default container was preserved after its earlier CUDA failure.
- Preflight:19 existing tests plus5 combined-loss tests, historical source/
  input/field/model audit, and4 native-volume endpoint-gamma smokes. Exact
  gamma0 scalar/gradient/optimizer parity passed for both diffusion backends.
- Production:10 positive-gamma cold/warm blocks=20 new outputs, plus4 reused
  gamma0 records. All14 fresh blocks including smokes registered/scored;
  no failures. All12 primary endpoints retain all35 labels. All production
  stages met the configured plateau rule without hitting update caps. The
  P84 stage learning rates0.03/0.01/0.003 were preserved; there were no
  within-stage learning-rate reductions.
- Posthoc source `scripts/reports/analyze_oasis_diffusion_kl_rank0.py` verified
  20 frozen source hashes, all field/model hashes, input identity,10084
  component-history states and exact cold/warm field agreement for all12
  settings. Cold/warm agreement is deterministic QA, not case replication.

## Interpretation and resource boundaries

- FD diffusion's maximum tested Dice is0.802083 at gamma1e-4 (+0.112pp versus
  gamma0); HD95 is slightly worse and ASD effectively unchanged. Analytic
  diffusion's maximum remains gamma0 (0.802291); no universal accuracy gain.
- At gamma1e-3, FD Dice changes-0.028pp, HD95/ASD improve and NDV drops30%,
  but analytic SDlogJ barely changes. Analytic diffusion loses0.461pp Dice
  at this gamma. The two diffusion backends do not give identical trade-offs.
- At gamma0.01, FD/analytic Dice losses are0.340/0.510pp, NDV falls79/78%,
  analytic folding falls54/53%, but analytic SDlogJ falls only about2.7%.
- At gamma0.1, Dice losses are1.108/1.250pp; analytic SDlogJ falls35-36%,
  analytic folding is0.0655/0.0652%, and NDV is0.000368/0.000343. ASD gets
  worse. At gamma1, Dice loses about8pp. Both sampled folding measures are
  zero while NDV remains positive; sampled Jacobian sign and digital NDV are
  complementary checks. No invertibility guarantee is implied.
- Positive-gamma FD registration is31.53-35.31s cold /29.62-32.25s warm;
  analytic diffusion27.89-31.04s cold /25.38-27.61s warm. Cold is a fresh
  worker's first fit, not a guarantee of an uncached filesystem. Startup/load
  and scoring are separately reported. Historical gamma0 warm values are
  21.91s FD /23.74s analytic; this is not a controlled timing rebenchmark.
- Warm Torch allocated/reserved peaks:FD2.154/2.479GiB;
  analytic1.930/2.516GiB. Sampled registration process peaks2.967/3.004GiB.
  Separate scoring process peaks5.527GiB. Do not equate allocated, reserved,
  process GPU memory or host RSS. Sample interval200ms; no contamination.
- All cold/warm times/memory, startup/load/output times, per-label accuracy,
  full/FG central and analytic Jacobian summaries, FD/analytic diffusion and
  BE in both unit systems, KL-C1, NDV, histories and receipts are external:
  `${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_diffusion1_analytic_kl_c1_gamma_v1/oasis_rank000_diff1_kl_20261110T201710Z_v1/`.
  Main report/curves and `detailed_review_v1/report.md` are the review surfaces.
- First10 is not launched. Suggested later candidates0/1e-3/1e-2/0.1 cover
  the observed accuracy/geometry trade-off;1e-4 can assess FD's small Dice
  gain and0.003/0.03 could refine the transition. Further work awaits the
  user's decision rather than an automatic grid extension.
