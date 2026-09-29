# P2ad technical log

Status: completed and acceptance-checked. All 33 headline registrations, 33
diagnostic registrations and 33 headline metric records are accepted.

## Known limitations

First10 diagnostic only. CP-BE previously failed strict FP32 trajectory/field
equivalence; the user explicitly requests empirical comparison despite this.
One cold sample per arm is descriptive. Sequential arm order is not a randomized
repeat study. Detailed instrumentation perturbs runtime and allocations: only
endpoint-only measurements are headline timing/memory results. Host and CUDA
spans are not disjoint work. Sampling cannot detect every very short foreign job.

## Implementation

Reuse persistent Docker workers and fail-closed PID/NVML monitoring. Extend
timing with an opt-in component context; noninstrumented operation order remains
unchanged. Diagnostic expansion and CP-BE reproduce the exact original operations
with nested intervals; CPU and GPU smoke parity are required before full runs.
The old P2ac acceptance and completed queue were verified; its evidence is retained.

CPU preflight passed all 13 contracts, including exact instrumented/uninstrumented
field/BE/gradient results and five-step Adam updates for each requested feature
set, plus nested host accounting and identity/affine/nonpositive SDlogJ cases.
The 66-job manifest is written externally. Data/environment validation confirms
the canonical first10, expected 200 pairs and 84 image/segmentation files, and
available surface-distance backend. Six sequential GPU smoke jobs were then run.

All six smokes subsequently passed: exact GPU DVF equality between endpoint and
component modes, identical steps/stop categories, required phases present, no
evaluation phases in timing, and expected cold factor construction/cache accounting.
Formal source hashes are frozen before the first headline worker. Headline runs
precede all diagnostics and evaluation; monitoring remains active throughout.

## Untimed evaluation watchdog recovery

The existing evaluator computes a float64 full-volume Jacobian twice when the
optional regional regularity metrics are requested. A separate CPU-only probe
confirmed that one existing Jacobian calculation takes roughly two minutes in
this environment. Neither the metric formulas nor the evaluator were changed.
This makes the historical one-hour whole-command watchdog unsuitable for this
33-record HD95/regularity pass.

The narrowly scoped host recovery entrypoint
`scripts/experiments/resume_mrdbscp842_be_combo_evaluation.py` verifies both
controller and evaluator identities, starts an overlapping GPU-0 monitor, and
replaces only the host controller. The already-running evaluator is not stopped
or restarted; no registration is repeated, and every frozen training/evaluation
source hash remains unchanged. The default recovery deadline is four hours.
Original and replacement monitor records are combined for the evaluation
verdict. Controller identity, exact adoption command, recovery source snapshot,
progress, and monitor evidence are retained under the external run root's
`evaluation_watchdog_recovery/` directory. The old controller's exit status 143
is the recorded orchestration replacement, not a failed registration.

After evaluation, the recovery runs the original final contracts and report
builder and writes `all_DONE.json` only upon success. It does not declare
completion merely because registration timing has finished.

## Completion and post-run presentation/orchestration amendment

The recovery completed successfully with no contaminated attempts. All 33
diagnostic/headline comparisons have identical fields, actual steps and stop
categories. All eleven combination fields are byte-identical to BE alone.
Evaluation retains 33 records from 20 unique case/field computations and 13
explicit identical-field cache reuses. GPU 0 was released after evaluation.
The original final contracts passed all 13 tests; accepted fields, backend flags,
source hashes, ordered sequences, label accounting and cache counts passed.

Only after `all_DONE.json`, the host-only evaluation watchdog became configurable
(four-hour default for this dedicated workflow; historical defaults preserved).
An additional contract checks timeout propagation without leaking the host
option into worker commands. The report renderer was refined to keep comparison
tables contiguous, show full-domain positive-Jacobian fractions, expose per-label
voxel counts and per-iteration component columns, and independently reconstruct
Dice from saved counts. These are post-run amendments, not training changes.
The first report is retained under the external review history; updated renderer
hash/provenance and `postrun_source_checks.json` distinguish the later sources
from the frozen worker snapshots. No timings or metrics are replaced.

Accepted run suffix: `mrdbscp842_be_combo_20260920T022500Z_v1`. See the external
`report.md`, `results.json`, `acceptance.json`, component CSVs, label comparisons,
evaluation records and `all_DONE.json` beneath the symbolic roots in the execution
specification. This is still first10 diagnostic evidence, not a publication-cohort
replacement or a proof of BE numerical/clinical equivalence. No paper or commit
was changed.
