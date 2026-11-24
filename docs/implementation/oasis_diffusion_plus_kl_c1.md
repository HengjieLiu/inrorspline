# Diffusion plus analytic KL-C1

The isolated `scripts/experiments/oasis_diffusion_kl_rank0/` adapter tests
NCC9 + diffusion(lambda1) + gamma*analytic_KL_C1 on fixed0333/moving0115 only.
The fixed gamma grid is0/1e-4/1e-3/1e-2/1e-1/1, retaining both FD and analytic
diffusion. The rank0 adapter itself has no cohort queue or gamma-range search;
P86 adds the separately authorized first10 queue described below.

The new objective derives from P84's frozen spline/NCC9 objective. Analytic KL
uses the exact total-field spline Jacobian in current-grid voxel coordinates,
crop2, epsilon0.05/kappa120. Analytic diffusion shares its first derivative fields
with KL; FD diffusion retains its established valid-stencil computation. At
gamma0 the KL branch is bypassed completely. Component-aware histories expose
data, diffusion, raw/weighted KL, coefficients, actual/selected updates and
stopping reasons. The best checkpoint still minimizes the total stage objective.

The worker is a process-local adapter to the established P84 loader, geometry,
optimizer conventions, cold/warm sequencing and independent replay/evaluator.
It hard-rejects rank>0. Historical source files and artifacts remain unchanged.
Reference auditing reuses only accepted rank0 cold/warm records for diffusion1,
checking scientific source/protocol/input identity and current field/model hashes.
Historical baseline resource measurements retain their provenance.

Execution uses all available GPUs, one child/GPU; registration and same-GPU
scoring are separate waves. Four native-volume smokes cover gamma1e-4/1 across
the two diffusion backends. The fixed production matrix has10 new cold/warm
blocks (20 registrations), plus two historical gamma0 reference pairs. Reports
retain all accuracy/regularity fields and separate registration/scoring resources;
single-case data do not justify case-population confidence intervals.

See the [execution specification](../../experiments/2026-11-10_oasis_rank000_diffusion_kl_c1/codex_execution_spec.md)
and [Jacobian/NDV definitions](jacobian_sdlogj_folding_and_ndv.md).

## Separately authorized first10 extension

P86 adds `scripts/experiments/oasis_diffusion_kl_first10/` after the user's
explicit cohort authorization. It imports the unchanged P85 objective and
optimizer, and changes only scope/provenance/orchestration: cold rank0 then
warm ranks0..9 in each positive-gamma worker. Both complete P84 diffusion1
cohorts are audited/reused for gamma0. P85's rank0 guard and artifacts remain
unchanged; the new adapter independently limits ranks to0..9.

Ten fresh blocks give110 registrations;22 historical observations yield120
primary warm endpoints across12 settings. Same-GPU independent scoring and
the original metric conventions remain fixed. The analysis reports case-level
means/sample SD, matching-baseline and matching-gamma backend paired changes,
exploratory paired case-bootstrap intervals, label eligibility/sensitivity,
and separate cold/warm/scoring resource scopes. Historical gamma0 timing is
observational, not a controlled loss-overhead rebenchmark.

See the [first10 execution specification](../../experiments/2026-11-10_oasis_first10_diffusion_kl_c1/codex_execution_spec.md).
