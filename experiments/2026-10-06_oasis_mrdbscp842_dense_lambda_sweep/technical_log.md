# Technical log

## 2026-10-06 implementation and verification

- Added method identity
  `direct_cp_8_4_2_then_dense_fd_be_multistage` and explicit dense-stage CLI
  controls without changing either accepted B-spline method.
- The cps2 field is expanded once at full resolution and detach/cloned into a
  fresh Adam parameter. The final stage uses full-image global NCC and
  unmasked forward-valid finite-difference bending energy with component-mean
  reduction.
- Added dense-stage representation, objective, initialization, stopping,
  parameter, device, seed, case, backend, and source-revision provenance.
- Focused compatibility assertions passed for the previous 8-4-2-1 path.
  Dense run-name/command assertions and a one-step optimizer unit smoke also
  passed.
- A full one-step 8-4-2-to-dense smoke passed DVF serialization, model-state
  persistence, canonical Dice/HD95/ASD calculation, finite shape validation,
  exact-zero initialization validation, and absence of a final B-spline
  control artifact.

## 2026-10-06 production and acceptance

- All 5/5 production registrations completed without OOM or retry. Lambda
  10/100/10000/1000 started on GPU 0/1/2/3; queued lambda 100000 ran on GPU 3,
  the first released device. Existing GPU jobs were not modified.
- All saved DVFs are finite with shape `[160,224,192,3]`. Every final summary
  identifies `dense_voxel` and contains no final cps1/control representation.
- Every cps2-to-dense initialization audit is exactly zero. The independent
  cps8/4/2 prefixes all stopped at `125/150/175`; terminal loss, data-loss,
  and regularizer-loss deltas are exactly zero across candidates.

| Dense lambda | Dice | HD95 | Symmetric ASD | Stage epochs |
|---:|---:|---:|---:|---|
| 10 | 0.761308 | 2.620805 | 0.578115 | 125,150,175,150 |
| 100 | 0.761925 | 2.659459 | 0.585027 | 125,150,175,150 |
| 1000 | 0.761210 | 2.610768 | 0.588087 | 125,150,175,150 |
| 10000 | 0.755493 | 2.472854 | 0.602692 | 125,150,175,275 |
| 100000 | 0.728933 | 2.542220 | 0.661515 | 125,150,175,900 |

- Lambda 100 is selected by the frozen Dice, then ASD, then HD95 rule.
  Lambda 10000 wins HD95 and lambda 10 wins symmetric ASD.
- Against the matched cps1 B-spline candidate, dense lambda 100 changes
  Dice/HD95/ASD by `+0.003449/-0.027524/-0.012393` (lower is better for the
  last two metrics).
- Accepted generated report:
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_rank000_mrdbscp842_dense_lambda_sweep_v1/report.md`.
