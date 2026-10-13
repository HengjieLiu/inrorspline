# Technical log

## Approved report recovery (2026-10-10)

- All400 dense candidates and400 matching prefixes are present (720 newly
  completed production workers plus80 accepted first10 outputs). Two smoke
  workers are not production cases. No registration rerun is required.
- Root cause: the report module resolved provenance files under
  `experiments/oasis_alpha_sweep` instead of `scripts/experiments/oasis_alpha_sweep`.
  This failed after reference scoring, whose results were held only in RAM.
- An interim process-pool recovery also failed because fork workers inherited
  initialized PyTorch interop threads. The temporary threaded scorer saved
  batches but lacked per-case atomic persistence, locking and cache provenance.
- Approved repair: correct the report path; add report-only finalization with
  an exclusive lock, spawn workers with one-time thread initialization, source/
  split/DVF-bound atomic caches, serial/spawn canonical parity for both methods,
  resumable output-hash validation receipts, and CSV/JSON/Markdown cross-checks.
- Preserve original launch provenance, all numerical code, fields, checkpoints,
  histories and existing GPU jobs. Record recovery hashes separately. Acceptance
  is written last, only after400 candidates and200 reference rows pass checks.
- ETA is based on freshly computed reference rows, excluding imported caches and
  accepted first10 rows. Final output hashing is reported as a separate phase.
- Recovery checks:3 new report/cache tests and4 existing first100 contracts
  passed; all16 NCC9 numerical/compatibility tests passed after setting the
  native DL/VFA source paths, including native loss/gradient parity. Actual
  VFA/SITReg rank10 serial-versus-spawn canonical metrics both passed. Handoff
  preserved70 interim rows; adding accepted first10 SITReg rows and freshly
  verified SITReg rank10 gave81 starting rows for the parallel scoring phase.
- Completed2026-10-10 08:37:26UTC (01:37:26PDT). All200 references and800
  production outputs passed; each DVF/checkpoint was rehashed once and each
  field checked for finite native geometry. All400 prefix/dense pairings match.
  Five final report files passed cross-format/recomputed-summary validation;
  `ACCEPTED.json` records their SHA256 values. No GPU fits were rerun.
- Lambda1 has the best mean Dice/HD95/ASD among the four sweep points:
  0.833679/1.820084/0.343654. SITReg HD95 remains slightly better at1.815750.
  Lambda1 foreground folding is3.145283%, versus VFA0.188933% and
  SITReg0.000502%; higher accuracy does not imply matched regularity.

## Known limitations before launch

- The first10 rows were previously used for exploratory tuning; the first100
  summary is therefore a broader test-cohort confirmation, not a pristine
  untouched validation set.
- GPU workloads are already active. The proposed placement adds one job each
  on GPUs0 and2 and two on GPU3; runtime is diagnostic and workers are admitted
  only when free-memory reservations pass.
- P43 first10 candidates are reused only after exact source-hash, protocol,
  pair, stage, output-hash and completion-marker validation.
- CUDA grid-sample backward may remain nondeterministic despite deterministic
  cuDNN settings.

## Execution notes

- CPU contract tests passed:4/4. The launcher dry run confirmed100 shuffled
  pairs, all200 saved VFA/SITReg files,400 candidate configs,800 worker outputs,
  and the GPU slot sequence0/2/3/3.
- Full-volume rank0 smoke passed at2026-10-09 22:04UTC (15:04PDT): cps8/4/2
  and dense workers both completed; dense initialization clone was exact,
  canonical metrics finite, native loss parity passed, and completion markers
  validated. The prefix smoke produced finite Dice0.670983/HD953.118229/ASD
  0.877101; dense one-step smoke produced0.682984/3.085008/0.845322.
- Eighty accepted P43 outputs (40 prefix plus40 matching dense fields) passed
  complete-marker, source/protocol/pair, checkpoint, DVF and SHA256 validation
  before reuse. They remain read-only in the original run root.
- Production launched2026-10-09 22:05:51UTC (15:05:51PDT). Initial workers:
  rank10 λ0.1 onGPU0, λ0.3 onGPU2, λ1 andλ3 onGPU3. Post-launch free-memory
  snapshot at22:06:57UTC: GPU0 4327MiB, GPU2 11592MiB, GPU3 14916MiB; all
  workers held4GiB reservations and GPU3's two slots remain within its8GiB
  combined reservation. Four existing SITReg jobs continue unchanged.
- At22:08UTC, all four rank10 prefixes completed and the four corresponding
  dense refinements were active. No failures or OOM retries at that checkpoint.
