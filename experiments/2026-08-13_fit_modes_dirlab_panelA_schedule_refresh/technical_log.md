# Technical Log

## Known limitations and boundaries

- This refresh changes only DIR-LAB Panel A MR-D-BSCP. It does not rerun the
  remaining methods, Panel B, or OASIS.
- The accepted historical comparator has 2400 epochs
  (`200/400/800/1000`), whereas the replacement has 2500
  (`400/400/800/900`); this is an intentional camera-ready schedule correction,
  not a one-variable fixed-budget stage-allocation ablation.
- The August 13 2500-epoch cohort has the target schedule but no full-crop
  diagnostic trajectory. It is a replay cross-check only.
- The source tree is dirty because related experiment infrastructure is under
  active development. Launch-time dirty paths and exact runner/common/launcher
  hashes are recorded in `provenance.json`.
- Main-index replacement and figure/manuscript regeneration are deliberately
  deferred to camera-ready step 4.

## Timeline

### 2026-08-13 — implementation and preflight

- Locked cases `01-10`, implementation, pTVReg 1 mm crop, T50-to-T00 direction,
  full-crop global NCC, fixed-T00 lung evaluation mask, dense FD bending energy,
  `32/16/8/4`, `8/4/2/1`, `400/400/800/900`, LR `1e-3`, alpha `10000`, seed 1,
  and diagnostic interval 10.
- Confirmed the existing August 13 exact-schedule outputs store
  `global_ncc_diagnostic_interval=0` and lack diagnostic CSVs, so exact reuse is
  prohibited.
- Implemented the focused launcher, monitor/manifests, and acceptance/comparison
  aggregator.
- The first smoke preflight stopped before launch because its initial free-GPU
  threshold treated the server's approximately 1.1 GiB display/driver baseline
  as contention. A direct compute-process query was empty and utilization was
  0% on GPUs 0 and 1. The preflight threshold was calibrated to reject more
  than 2 GiB allocated memory or more than 20% utilization; no job had launched
  under the rejected preflight.
- Both smoke optimizations reached output finalization, but the launcher's
  completion checker then rejected the intentionally repeated smoke epoch list
  `1,1,1,1` because it reused a uniqueness-enforcing parser intended for cases
  and GPUs. The completion checker was changed to use an order-preserving
  integer-sequence parser. Existing completed smoke outputs are retained and
  are eligible only for exact-compatible `--resume` validation.

## Completion record

The full launch ran from approximately 18:56 to 19:25 UTC on 2026-08-13. All
ten records completed on their first attempt in about 30 minutes of cohort wall
time. Strict aggregation completed at 19:28 UTC.

| Item | Result |
|---|---|
| Dry-run | PASS: 10 unique full records, schedule `400/400/800/900`, diagnostic interval 10 |
| Smoke case02/case08 | PASS: exact metadata, 4/4 histories and diagnostics, finite endpoints and secondary metrics |
| Full jobs | PASS: 10/10 completed, 0 reused, 0 failed |
| Acceptance | PASS: exact protocol; 2500 history rows and 251 finite diagnostics per case |
| Historical comparison | PASS: paired N=10 for endpoint, runtime, memory, TRE, stdLogJ, and folding |
| August 13 cross-check | PASS: 10/10 protocol matches; displacement-derived metrics reproduced exactly |
| Camera-ready checklist update | Complete for schedule rerun and focused reaggregation; figures/manuscript remain unchecked |
| Registry finalization | Complete |

## Result summary

- Historical versus replacement `1-NCC`: `0.003216 ± 0.001950` versus
  `0.003215 ± 0.001950`; paired mean change `-0.000000441`, range
  `[-0.000006, 0.000001]`.
- Mean runtime: `305.0` versus `299.8` seconds; paired mean change `-5.2`
  seconds. Mean peak memory: `5.483` versus `5.392` GiB.
- Common registration-crop snap TRE: `1.2123` versus `1.2133` mm; paired mean
  change `+0.0010` mm. Foreground stdLogJ changed by `-0.000062` on average;
  foreground nonpositive-Jacobian percentage changed by `+0.000029` points.
- Against the August 13 diagnostic-disabled exact-schedule runs, stored TRE,
  stdLogJ, folding, peak memory, and native-crop headline TRE are identical.
  The full-crop trajectory diagnostic added a mean `89.9` seconds runtime.
- These results support treating the schedule correction as technically valid
  and empirically endpoint-neutral for this cohort. The historical 2400-epoch
  cohort remains immutable sensitivity evidence.
- A post-acceptance `--dry-run --resume` check resolved all ten records as
  exact-compatible. Because dry-run manifests are live planning views, that
  check changed `job_manifest.*`/`monitor_status.*` to a `completed_existing`
  view; `launcher_state.json` preserves the actual full-launch outcome of ten
  newly completed records, and `execution_assignments.csv` preserves the
  observed per-case GPU/PID assignments and runtimes.
- That check also exposed Docker's Git repository-ownership guard: the initially
  generated commit/branch fields were blank even though source hashes were
  present. The provenance was repaired with the launch commit/branch and
  relevant dirty paths, and the launcher now runs read-only Git metadata queries
  as the mounted repository owner when Docker starts the launcher as root.
  `provenance.json` intentionally retains the pre-fix launcher hash that actually
  launched the optimization; the Git fix is orchestration-only and occurred
  after all ten runs completed.
