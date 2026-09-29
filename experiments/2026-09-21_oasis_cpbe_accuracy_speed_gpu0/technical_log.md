# Accuracy-first overnight experiment technical log

## Known limitations

- First10 is a tuning screen; no held-out or publication-cohort claim.
- CP-BE has prior strict numerical non-equivalence; fresh CP-BE is the reference.
- Intermediate Dice plateau does not imply stable HD95/folding/smoothness.
- Cold n=1 per candidate; cohort warm SD mixes case variability and timer noise.
- Finite-difference posthoc BE is not the analytic CP-BE training loss.
- Deadline can leave later candidates incomplete; never promote partial cohorts.

## Approved plan / implementation

- User authorized overnight GPU0 execution and later removed the >5% weekly
  quota condition. Extensions require completed main work and enough time only.
- Twelve prespecified main candidates cover fixed/adaptive stopping, LR decay
  and exact nested16/8/4/2. Preserve historical defaults and all prior outputs.
- Registration accuracy uses Dice/HD95 first, with explicit similarity/regularity
  guards and per-case/per-label reporting. Segmentation is offline-only.
- Reused worker/backend/Docker/NVML framework with new opt-in schedule actions.
  New deadline-aware controller and incremental machine/human reports.
- Physical GPU0 was idle at planning. Shared data mount RO, artifacts RW,
  existing default image/container verified; no Docker recreation required.
- Deadline interpreted as09:00 local Pacific civil time; GPU cutoff15:50UTC,
  no-new-work cutoff15:40UTC. All operational paths remain external/private.

## Verification / execution

- Initial dry-run confirmed12 candidates,132 headline registrations,120 quality
  jobs. Six new CPU contracts and fourteen legacy contracts passed.
- Initial v1 preflight was stopped deliberately before any formal run after
  identifying a strong-reference list that would retain earlier-stage scheduled
  Adam states. Corrected to integer-only stage bookkeeping and added a weakref
  release regression. Preserve all v1 smoke artifacts; use a fresh v2 run.
- Correctness-only candidate smokes now share one worker, avoiding repeated CUDA
  initialization. Formal timing still starts a fresh worker for EVERY candidate.
- A separate CPU-only completion queue automatically produces a four-panel
  accuracy/speed figure, checksum/source/backend/endpoint validation, and checks
  confirmation DVFs against headline field hashes after the controller exits.
- No user review/approval gate remains; automatically proceed after checks.
- v2 passed all ten three-stage smokes but exposed a legacy smoke-only hardcoded
  three-element check interval in the unified launcher's epoch override. No
  headline measurements had started. Generalized the interval to the actual
  stage count (unchanged for historical three-stage schedules), added a contract
  covering every candidate's effective smoke arrays, and retained v2 unchanged.
  Restart under v3 with newly frozen sources; no failed/preflight rows enter
  headline comparisons.
- Eight CPU contracts now pass, including all candidate smoke arrays and
  optimizer lifetime. The v3 dry-run again lists132 headline registrations and
  120 quality jobs; v3 controller and independent completion queue have started
  automatically. Runtime IDs and exact commands are in external provenance.
- v3 preflight completed:12/12 schedule smokes (including both four-stage
  variants),35 valid labels in the clean independent metric smoke, matching
  Dice/HD95 label accounting, verified200 pairs and84 image/segmentation files.
  Formal reference cold registration started at09:39UTC on2026-09-21.
- Independent CPU-only delivery stage-summary audit passed on12 retained smoke
  records /38 stage rows. Final reports will include per-stage actual steps,
  stop counts, effective LR endpoints, pre-update losses, and carefully labelled
  loop/post-loop wall spans; no instrumentation was added to timed training.
