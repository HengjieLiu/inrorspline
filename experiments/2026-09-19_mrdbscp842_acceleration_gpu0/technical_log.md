---
doc_type: technical_log
status: completed
---

# MR-D-BSCP GPU-0 acceleration log

Completion: the external queue reached final_report and the acceptance manifest
passed, including all planned registrations, evaluations, field scans and final
source checks. The source text below retains chronological implementation notes.
The subsequent P2ad study uses new roots and does not modify these artifacts.

## Known limitations

First10 OASIS diagnostic; not a publication-cohort replacement. Historical GPU-3
deep instrumentation is contextual only. Reference StableStd has a modified
backward; coordinate prolongation is lattice-aligned; worsening losses can trigger
the original plateau rule. Preserve these behaviors. GPU monitoring is sampled,
and query failures invalidate timing. No speedup claim before paired confirmation.

## Initial audit

The supplied handoff checksum and bundled source-spec checksum match. Small CPU
checks reproduced FP64 Gram equivalence and naive float32 constant-field negative
energy. The baseline repository derivative implementation forces float32.
GPU 0 was idle at preflight. Runtime container has torch 2.6.0+cu118,
CUDA 11.8, cuDNN 90100 and Triton 3.2.0; native CUDA build tooling is absent.

## Execution

Implementation and preflight in progress. Generated evidence and live progress are
written to the external run/review roots specified in codex_execution_spec.md.

Initial CPU and GPU checks pass for kernel caching, batched expansion, fixed NCC
statistics and loss buffering. The QR/difference BE agrees with the independent
FP64 sampled oracle, but fails the strict FP32 compatibility gate on near-affine
gradients and repeated regularizer-only Adam nullspace stress. The user explicitly
requested registration characterization anyway: retain the warning, run BE, and
quantify Dice and HD95 rather than infer quality degradation from numerical tests.

Seven CPU contract tests pass: original CLI defaults, patching the actual module
used by the runner (avoiding duplicate Python imports), endpoint-only timing,
the original first stopping checkpoints, buffered row identity, fail-closed
contention verdicts, and detectable axis/crop errors.

Preflight uses the initial nvidia-smi polling monitor. Its subprocess cadence was
slower than intended on this host, so formal registration phases use direct
read-only NVML with the installed header ABI verified, matching GPU UUID, and
host PID/start-time verification retained. Five initial direct queries averaged
about 0.19 s. Source snapshots and per-worker hashes distinguish these versions.

The initial operator harness used the single-stage control-grid helper, which
gives one fewer control per axis than the MR nested convention. This was found
and corrected before formal full registration. Both the initial generic-grid
checks/microbenchmarks and the corrected exact nested-grid versions are retained;
only the latter are used as MR-stage operator evidence. Actual registration has
always used the original nested-grid builder. A regression test now fixes all
three expected grid dimensions. The contract suite now contains eight tests.

Later verification adds a ninth contract test: deliberately removing the factor
two from mixed derivative terms must be detected. Final verification reruns the
expanded suite; earlier test records remain available.

All six one-step registration smokes completed, including the user-authorized BE
arm. Full reference repeats and first10 measurements are underway; no end-to-end
speed or registration-quality conclusion is asserted from the microbenchmarks.

The timing engine (run_one.py/common.py/backends.py) is frozen for paired
registration. Reporting, validation and post-timing diagnostics are developed
while those runs proceed. Each phase retains the controller/source snapshot and
each worker records launch-time source hashes. Later worker launches additionally
retain source copies; changes to evaluator/report helpers do not alter the timed
optimization engine. Final validation checks training-engine hash consistency.

The first BE characterization has completed and is retained externally, including
all per-case and per-label Dice/HD95 values. Interpretation explicitly separates
case means from structure-specific outliers, full-DVF differences and stopping
changes. A raw-fixed-image-positive foreground/background error localization is
added as a diagnostic, without replacing the preregistered full-field gate.

A sequential completion queue is provided for the remaining confirmation and
reporting phases. It waits for the exact predecessor controller PID/start time,
never overlaps its jobs, and records each command and failure externally. It
does not promote BE into the strict-equivalence combination. Source/contract
checks and the full DVF acceptance scan run after timed registrations, avoiding
CPU data-processing contention during measurements.
