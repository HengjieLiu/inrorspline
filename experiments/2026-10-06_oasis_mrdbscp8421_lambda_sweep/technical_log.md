# Technical log

## 2026-10-06 preflight

- Added backward-compatible per-stage lambda support to the OASIS runner.
- Added the explicit MR-D-BSCP 8-4-2-1 method identity and cps1 metadata.
- Added the four-GPU controller, strict validator, and report generator.
- Focused container tests passed: stage-alpha fallback/validation, nested
  8-4-2-1 geometry, run-name uniqueness, and complete launch arguments.
- Dry run resolved shuffled rank 0 to fixed `0333`, moving `0115`, original
  test index `158`, and emitted all five expected commands.
- The generic dataset default was stale for this container; the launcher now
  freezes the verified mounted OASIS path. External outputs must run as the
  artifact-store owner (`1009:1009`) because the NFS export root-squashes the
  container root user.

Production status and accepted metrics will be appended after aggregation.

## 2026-10-06 production and acceptance

- The first launch stopped before training because UID/GID `1009:1009` could
  write the NFS artifact store but lacked access to `/dev/nvidia*`. Production
  uses `1009:1020`, preserving artifact ownership and CUDA access.
- The next attempts completed optimization but exposed a PyTorch flattened
  `quantile` size limit while computing the diagnostic cps1 control p95. The
  runner now uses a GPU nearest-rank `kthvalue` only above 16 million values.
  A four-stage one-step smoke then passed DVF serialization and all metrics.
- The accepted attempts completed 5/5 with no OOM or retry. Every saved DVF
  is finite with shape `[160,224,192,3]`. The deterministic cps8/4/2 prefix
  stopped at `[125,150,175]` for every candidate; terminal prefix loss deltas
  are exactly zero across GPUs.

| cps1 lambda | Dice | HD95 | Symmetric ASD | Stage epochs |
|---:|---:|---:|---:|---|
| 10 | 0.762269 | 2.741775 | 0.589742 | 125,150,175,175 |
| 100 | 0.758476 | 2.686983 | 0.597420 | 125,150,175,150 |
| 1000 | 0.758851 | 2.606502 | 0.593440 | 125,150,175,150 |
| 10000 | 0.752330 | 2.545912 | 0.606033 | 125,150,175,250 |
| 100000 | 0.727667 | 2.598633 | 0.670282 | 125,150,175,650 |

- Lambda 10 is selected by the frozen Dice, then ASD, then HD95 rule. It
  slightly improves Dice and ASD over the original 8-4-2 result, while HD95
  is worse. Greedy, VFA, and SITReg remain better on all three metrics for
  this case.
- Accepted generated report:
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_rank000_mrdbscp8421_cps1_lambda_sweep_v1/report.md`.
