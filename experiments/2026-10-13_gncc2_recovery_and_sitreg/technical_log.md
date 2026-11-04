# Technical log

## 2026-10-13: approved recovery and dependent SITReg sweep

- Audited the source-drift interruption. Recovered exact evaluator hash by
  appending one trailing newline in a proposed isolated execution copy.
- Saved FeDDA checkpoints at 103/120/118/120 epochs contain finite weights,
  complete optimizer states, best-checkpoint metadata and CPU/CUDA RNG states.
- Native dataset uses shuffled moving subjects and Python-random fixed pairing;
  native loader persists across epochs. A resume cannot simply recreate a fresh
  shuffled iterator from the checkpoint RNG. Added exact metadata-only replay
  and saved RNG equality gates before continuation.
- New source files only; historical numerical runners and failed outputs are
  preserved. Snapshot verification redirects only historical repository paths,
  not data/reference/model dependency paths.
- New SITReg squared adapter retains the signed adapter's entire protocol
  except the similarity formula/identity. Full reference/smoke gates remain.
- Implemented durable all-four FeDDA completion barrier before SITReg preflight,
  four fresh trainings and automatic detailed first100 evaluations.
- Verification and launch evidence will be appended after checks complete.

## Verified launch: 01:43:10 PDT / 08:43:10 UTC

- 29 unique CPU tests passed: existing FeDDA GNCC²/SITReg signed regression
  tests plus five recovery/SITReg squared/barrier/report tests. `git diff
  --check` passed. Synthetic report tests are not real registration results.
- Frozen 425 source/config/test/metadata files. Base revision
  `1a2f0df4d76627b087db58255419789e99043ec8`, with separately hashed new
  recovery/SITReg source files; no Git commit was made.
- Every historical FeDDA gate/baseline/evidence hash and SITReg baseline source
  hash passes in the frozen environment. Working-tree evaluator unchanged.
- All four GPU recovery checks passed: strict native initialization identity,
  saved finite parameters/Adam state, exact completed pair hashes, exact CPU,
  Python and NumPy RNG replay, CUDA RNG restoration and best-checkpoint lineage.
- Queue PID 102712; FeDDA recovery controller PID 102802. GPU 0/1/2/3 workers
  102805/102810/102813/102816 resumed epochs 104/121/119/121 respectively.
  Initial forward/reverse updates are finite. First 44/46/23/42 new pair IDs
  verified against completed LNCC control histories; original latest checkpoint
  hashes remain unchanged. Per-epoch pair hash guards stay active.
- Current launch free memory: 48,515 MiB on GPUs 0/1/3 and approximately
  44,286 MiB on GPU 2. Existing GPU 2 workload retained without intervention.
- FeDDA detailed evaluations and three-loss report are automatic. SITReg is
  durably queued behind all four FeDDA train/eval completions and report; its
  strict preflight is not yet run. Any failed gate blocks later production.
- Historical timing for context: signed FeDDA lambda1 training 5.54 h and
  first100 evaluation 0.43 h; signed SITReg lambda1 training 9.25 h and evaluation
  0.35 h. Initial estimate: 3–4 h for recovered FeDDA train/eval/report, followed
  by approximately 9–11 h for SITReg preflight/train/eval/report, workload-dependent.
