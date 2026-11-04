# P77 Phase-A technical log

## Limitations

Single exploratory case; no generalization or across-case SD. The two-field
model has extra capacity and midpoint similarity differs from endpoint NCC.
Factor-two diffusion matching holds exactly for equal/opposite velocities,
not for arbitrary nonlinear map compositions. Seven-step sampling is not an
unconditional topology/inverse-consistency guarantee. Equal branch RMS does
not prove a geometric midpoint; diffusion does not penalize constant drift.
Shared GPU timings are diagnostic; CUDA grid-sample backward can vary.

## Preflight

- User approved Phase A only: A1/A2/A3 plus swapped A1, then energy report.
- Existing NCC9 detaches fixed image; add isolated two-gradient NCC instead.
- Optimize twice-branch coefficients with historical parameter LR to avoid
  doubling relative step size. Actual branch velocities are explicitly half.
- Preserve both branches as a joint checkpoint and prolong each independently.
- Current GPUs 0/1/2 are idle; GPU3 has an existing 3.2 GiB workload. No jobs
  changed. Container image identity and live admission will be recorded.

## Implementation and smoke gates

- Added isolated symmetric numerical objective, joint Adam, diagnostics,
  worker/replay and one-worker-per-GPU controller; historical sources unchanged.
- All 28 tests passed: 12 new symmetric tests plus existing SVF and precision-
  transition regressions. Two-input FP64 gradient check, both FP32 image
  gradients, input exchange, composition order and independent refinement pass.
- Four full-size smokes and independent replays passed in 59.52 s. Each smoke
  performs one update at each of the three levels, full-size gradient checks,
  forward/reverse serialization, metrics, energy and 100000-point diagnostics.
  Peak allocated/reserved maximum 4.734 GiB; production admission stays 8 GiB.
- Production launched on GPUs 0/1/2/3 for A1/A2/A3/A1_swap, respectively.
  No existing jobs changed. Phase B remains unlaunched and approval-gated.

## Completion and interpretation

- All four production fits and independent replays passed without failures,
  OOMs or retries. Queue wall time 386.018 s, accepted 2026-10-16 06:02 UTC
  (October 15 local). Three reporting tests also passed: 31 total tests.
- Report acceptance verifies 6 comparison rows, 12 stage rows, 24 integration
  sensitivity rows, 30 canonical statistics and unchanged numerical sources.
  Numerical source, checkpoints and accepted report outputs remain immutable.
- A1/A2/A3 Dice 0.785973/0.788333/0.790653 does not reach historical
  bounded-two 0.801036 or SVF 0.797502. A1 folding is zero on the canonical
  foreground metric, not a general diffeomorphism guarantee.
- A1/A2 RMS branch gaps are only 2.1464%/1.1860%, but shared-motion RMS is
  27.194954/35.920834 mm. A1/A2 velocity cosines +0.976533/+0.981687 show
  largely aligned velocities, unlike tied A3 (-1, shared RMS zero).
- Midpoint raw-positive fractions rise from approximately 31% originally
  to 71.778/73.296% in A1 and 83.970/85.559% in A2. All-zero NCC9 windows
  have loss one; the unmasked midpoint objective allows common deformation
  to reweight empty/informative regions. This explains a plausible incentive,
  not a proven unique cause. Balance alone will not penalize equal large
  common motion. Recommend amplitude-only control before balance-only tuning.
- A1 input-swap foreground endpoint difference mean/p95/max is
  0.272466/0.963502/14.785166 mm; component RMS 0.529317 mm. Stops differ:
  A1 275/550/375, swapped A1 350/600/525. This is not a failed replay or
  within-fit inverse consistency check. Symmetric CPU objective tests do not
  prove exchange-identical GPU optimization. CUDA sampling nondeterminism and
  adaptive trajectory divergence are possible contributors, not established
  sole causes. A stronger symmetry claim requires a separate audit.
- A1 forward/reverse all-point cycle means 0.117632/0.105585 mm, p95
  0.230974/0.203380 mm; in-bounds fractions 99.106/99.545%. A2 cycle
  means 0.168956/0.150978 mm; A3 0.042579/0.040494 mm. Seven-step fields
  remain primary; eight/ten-step diagnostics do not replace them.
- Optimizer times A1/A2/A3/swap 118.720/274.273/55.499/245.757 s;
  joint updates 1200/1875/500/1475. Maximum production allocated/reserved
  memory approximately 4.413/4.730 GiB, excluding unrelated jobs/context.
  These shared-load timings are not controlled speed benchmarks.
- CSV/JSON/Markdown saved under the external experiment key's `phase_a`
  report directory. No amplitude/balance sweep or first-ten expansion launched.
