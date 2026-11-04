# P68 technical log

## Known limitations

- Ranks0 and4 were deliberately selected exploratory cases, not a population.
- Both arms double allowed optimization blocks; actual epochs/runtime differ.
- Additional maps change capacity and optimization path, not only saturation.
- Elementary contraction does not guarantee every exported FD metric has0% folds.
- CUDA sampling backward may remain nondeterministic; shared-load time is diagnostic.

## Implementation

- User approved the four-run pilot: two bounded maps per level versus a
  same-map two-block continuation control on ranks0/4.
- Historical source modules and outputs stay unchanged. New optimizer keeps
  original Adam/plateau semantics and only adds selected-latent initialization.
- New bookkeeping separates4 pyramid levels from8 optimization blocks.
  Control uses retained block indices[1,3,5,7]; proposed method uses[0..7].
- Lambda[1,1,5,1] and all per-level settings repeat unchanged for block b.
- First memory check: GPUs0/1/3 about47.4GiB free, GPU2 about43.2GiB free.
  Existing GPU2 workload retained. Admission is rechecked at launch.
- Test and full-size smoke gates precede production. External reports only.

## Verification

- All28 bounded-composition unit/regression tests passed, including the five
  new two-block tests. Fresh-zero optimizer matches the immutable numerical
  predecessor; continued initialization preserves the chosen latent and loss,
  starts Adam step counters afresh and keeps control caps unchanged.
- Translation-chain tests distinguish8 composed maps from4 continued maps;
  identity second-map insertion preserves the first-map field/objective.
- Native DL loss/gradient parity, full native serialization and elementary
  bounds are checked by each full-size smoke before production.

## Smoke and production launch

- Both rank0 full-size eight-block smokes passed. Eight-map peak allocated
  memory3.45766GiB; admission remains8GiB. Same-map second-block continuation
  reproduces the preceding field/objective, and new increments start at identity.
- Additional loss/schedule suite:15 passed/1 skipped. Combined43 passed and
  one skipped. No test failures or numerical-source modifications.
- Four fresh production fits launched as1009:1020: rank0 two/control onGPU0/1,
  rank4 two/control onGPU2/3. Existing GPU2 job retained; no retries/tuning.
- Independent per-block metric/checkpoint/topology replay gates report acceptance.

## Completion and independent acceptance

- All4 production fits completed with no failures/retries. Four independent
  audit processes replayed all32 selected block checkpoints: exact dense-field
  reconstruction, initial/selected objectives, all35-label metrics, five final
  metrics, full elementary native-cell-corner bounds and final100k probes.
- Reports accepted2026-10-14 06:09:54UTC (2026-10-13 PDT). Final verification
  rechecked acceptance-file hashes, current source hashes, immutable reference
  hashes, all4 DONE/summary records and audit-to-summary hashes.
- First cps8 blocks agree across arms and with historical all-bounded runs
  under frozen tolerances. Both arms preserve block-a to block-b initial
  field/loss continuity, with fresh Adam state. All sources remain unchanged.

| Rank | Arm | Dice | HD95 mm | ASD mm | FG SDlogJ | FG folding % | Optimization s | Worker s | Peak GiB | GPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|Two increments/level|0.798510|1.999331|0.429354|0.617426|0.000743|138.580|231.329|3.458|0|
|0|Same-map continuation|0.780452|2.193289|0.484875|0.492586|0.000000|128.212|225.243|3.150|1|
|4|Two increments/level|0.728966|2.417488|0.619302|0.621042|0.000300|203.470|275.702|3.458|2|
|4|Same-map continuation|0.687008|3.181210|0.878093|0.499194|0.000000|126.765|199.599|3.150|3|

- Two increments improve all three accuracy metrics beyond same-map
  continuation in both cases: Dice+0.018058/+0.041959; HD95-0.193958/
  -0.763721mm; ASD-0.055521/-0.258791mm. This supports added compositional
  capacity, but does not isolate capacity from the changed optimization path.
- Rank0 exceeds VFA on all three accuracy metrics and is close to SITReg
  (Dice-0.000209, HD95-0.061988mm, ASD+0.006290mm). Rank4 improves clearly
  over historical all-bounded Dice0.684704 but still trails VFA0.760943 and
  SITReg0.768742. No generalization or first-ten claim is made.
- Second cps2 increments at lambda5 lower total objective but worsen Dice
  by0.005023/0.010309 on ranks0/4. Second cps8/cps4/cps1 increments improve
  all three accuracy metrics. Do not confuse lower objective with label gains.
- All elementary bounds remain<1. Retained8-map chains have0/100000
  nonpositive off-grid determinants, minimumJ0.013809/0.022624. Interpolated
  final exports have4/100000 and3/100000, minimumJ-0.070874/-0.217291.
  These differ from canonical finite-difference foreground folding above.
  Four-map continuation probes/export have0/100000 in both cases.
- Existing workloads were left intact; runtime is diagnostic. No added
  increments, LR/lambda tuning, or first-ten expansion after this pilot.

## Artifacts and reproduction

External run/report key:
`oasis_seed3_rank000_004_mrdbscp8421_tanh_two_blocks_lambda1151_v1`.
Reports: comparison.csv (16 rows), blocks.csv (32), second_block_changes.csv
(16), two_minus_control.csv (2), results.json, report.md and ACCEPTED.json.
All histories, eight block checkpoints per run, DVFs, manifests and independent
audits are retained in the external run tree. Historical references are read-only.

Inside the recorded Docker image as1009:1020, with ARTIFACT_ROOT set:

```bash
python -B -m unittest discover -s tests -p 'test_oasis*bounded*.py'
python -B -m unittest discover -s tests -p 'test_oasis_dl_ncc_diffusion.py'
python -u -B scripts/experiments/oasis_alpha_sweep/run_oasis_bounded_two_blocks.py launch \
  --artifact-root /database/inrorspline --gpus 0,1,2,3 \
  --stage-alpha 1 1 5 1 --blocks-per-level 2 --bound-multiplier 0.99
```

The completed launcher returns the existing accepted report; it does not
overwrite or rerun accepted candidates. Source revisions/hashes, Docker image,
GPU state and full numerical schedules are stored in launch_manifest.json.
