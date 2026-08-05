# Technical Log: OASIS DirectCP 8-4-2 Runtime Sweep

## Known Limitations

- This is an optimization/runtime sweep on first10 OASIS only, not a new final
  population claim unless a candidate is later extended.
- MATLAB pTVReg completion is no longer a dependency for launch; the prior
  pTVReg run was intentionally stopped by the user before this DirectCP work.
- The run uses the current working tree because the adaptive stopping and
  runtime launcher are being added immediately before launch.
- The primary runtime comparison should report optimization/training time.
  Total time includes dense warp expansion, label metrics, regularity metrics,
  serialization, and launcher overhead.

## Execution Notes

- Planned GPU policy: use all visible GPUs in `py310sitreg_v1_test1`, one job
  per GPU.
- Preflight policy: confirm Docker, CUDA, case resolution, and a one-rank
  minimal DirectCP command before launching the smoke matrix.
- Smoke policy: run candidates on ranks `0,4,8`; aggregate in the smoke review
  subroot and use `promoted_candidates.txt` for the first10 launch.
- First10 policy: run only smoke-promoted candidates on ranks `0-9`, reusing
  completed smoke jobs from the same external run root when possible.

## 2026-08-04 Setup

- Host branch: `gpu8`.
- Host commit before runtime edits: `b9d6c52`.
- Docker container: `py310sitreg_v1_test1`.
- Docker PyTorch version probe: `2.6.0+cu118`.
- GPU preflight: four visible RTX 6000 Ada GPUs, each at `5 MiB` memory and
  `0%` utilization before the DirectCP preflight.
- Case validation: seed-3 shuffled rank `0` resolves to original sorted index
  `158`, fixed `0333`, moving `0115`.
- Tiny adaptive preflight command completed on GPU `0` with stage epochs
  `1,1,1`, alpha `1000`, lr `1e-2`, and adaptive stopping flags enabled.
  Output root:
  `<external-run-root>/runs/oasis_directcp842_runtime_preflight_v1`.
- Tiny preflight result: `requested_epochs=3`, `actual_epochs=3`,
  `stage_actual_epochs=[1,1,1]`, train `130.94 s`, total `159.52 s`. The high
  total despite three optimizer steps confirms that dense evaluation,
  regularity metrics, and serialization are a substantial fixed tail.

## 2026-08-04 Smoke Sweep

- Smoke launch: all `16` runtime candidates on ranks `0,4,8`, GPUs `0-3`, one
  job per GPU.
- Smoke aggregation: `48/48` summaries, `errors=[]`.
- Dropped by smoke rule: `F600_lr1e-1` and `F950_lr1e-1`; both lost more than
  `0.05` Dice and worsened HD95 by more than `0.5 mm` against the matched
  DirectCP `8-4-2` baseline subset.
- Promoted to first10: `F600_decay`, `F600_lr3e-2`, `A_bal_lr3e-2`,
  `A_bal_lr1e-2`, `A_fast_decay`, `F600_lr1e-2`, `F950_decay`,
  `F1550_lr3e-3`, `F950_lr1e-2`, `A_safe_lr1e-2`, `F950_lr3e-2`,
  `A_safe_lr3e-2`, `F1550_lr3e-2`, and `F1550_lr1e-2`.
- Smoke fastest stable candidate: `F600_decay`, Dice `0.7701`, HD95
  `2.6675`, train `141.38 s`, total `176.06 s`.
- Note: smoke acceptance here means stability/promotion only. The final
  DirectCP cps2/SINR cps2/user-tolerance tiers are decided on the first10
  aggregate.

## 2026-08-04 First10 Sweep

- First10 launch: `14` smoke-promoted candidates on ranks `0-9`, GPUs `0-3`,
  one job per GPU. Existing smoke outputs for ranks `0,4,8` were reused.
- First10 aggregation: `140/140` summaries, `errors=[]`.
- Matched baseline for DirectCP `8-4-2`, alpha `1000`, lr `1e-3`, fixed
  `400,800,1300` schedule on ranks `0-9`: Dice `0.80697`, HD95 `2.20200`,
  train `194.79 s`, total `229.75 s`.
- Baseline training-vs-total gap: total exceeds training by `34.96 s`
  (`15.3%` of total). The runtime-sweep candidates have similar absolute
  fixed tails of about `32.6-35.8 s`, so the practical speedup comes mostly
  from reducing optimization/training time.
- Fastest first10 candidate: `A_fast_decay`, Dice `0.80581` (`-0.00116` vs
  baseline), HD95 `2.21472` (`+0.01272 mm`), train `149.51 s`, total
  `183.22 s`, mean actual epochs `577.5`. Speedup vs matched baseline:
  `1.30x` train and `1.25x` total.
- Best fixed short candidate: `F600_decay`, Dice `0.80630` (`-0.00067`),
  HD95 `2.20893` (`+0.00693 mm`), train `151.82 s`, total `186.20 s`, mean
  actual epochs `600`. Speedup vs matched baseline: `1.28x` train and
  `1.23x` total.
- Highest Dice among candidates: `F1550_lr3e-3`, Dice `0.80713`
  (`+0.00016`), HD95 `2.21244` (`+0.01044 mm`), train `172.12 s`, total
  `205.80 s`. This is slower than the short/adaptive candidates but still
  faster than baseline.
- All `14` first10 candidates were tier `A` under the configured criteria and
  all remained within the user-specified tolerance against the matched
  DirectCP `8-4-2` baseline. All except `F600_lr3e-2` beat both DirectCP cps2
  and SINR cps2 in Dice/HD95; `F600_lr3e-2` still beat DirectCP cps2 and stayed
  within tolerance but did not beat SINR cps2 HD95.
- Recommended candidates for extension: use `A_fast_decay` when prioritizing
  total runtime; use `F600_decay` if a fixed deterministic schedule is
  preferred with nearly identical runtime and slightly better Dice/HD95.
