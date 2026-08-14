---
doc_type: technical_log
status: active
date_started: 2026-08-12
---

# LungCT Corrected MR-D-BSCP IDIR-Loss Alpha Sweep Log

## Known Limitations

- This is a scale-finding sweep for the corrected fixed-mask analytic-BE
  protocol, not yet the final camera-ready operating-point selection.
- The input image protocol remains pTVReg-style 1mm landmark crop. It is not
  the native full-volume original IDIR protocol.
- The `alpha=10000` point is imported from the completed anchor run rather
  than re-run, provided its protocol metadata remains identical.
- The loss uses sampled fixed T00 lung-mask global NCC, so stochastic variation
  is possible even with fixed seed.

## Protocol

```text
method = direct_cp_32_16_8_4_masked_ncc_bspline_be
cases = 01-10
new_alphas = 100,1000,100000,1000000
anchor_alpha = 10000
lr = 1e-3
seed = 1
batch_size = 10000
stage_full_cps = 32,16,8,4
stage_downsample = 8,4,2,1
stage_epochs = 200,400,800,1000
```

## Verification Log

- Passed: compile/import check for the sweep launcher and aggregator inside
  `py310sitreg_v1_test1`.
- Passed: launcher dry-run queued exactly `40` new jobs over GPUs `0,1,2,3`.
- Fixed before clean relaunch: non-final low-resolution stages no longer
  expand a full 1mm dense field for prolongation preservation diagnostics by
  default. Exact knot-insertion initialization is unchanged; diagnostic
  preservation rows now record `diagnostics_skipped=true` unless
  `--prolongation-diagnostics` is explicitly enabled.
- Passed: patched case01 smoke with `stage_epochs=1,1,1,1`; prolongation
  diagnostics were skipped for stages 1-3 and final metrics were written.
- Stopped by request after `alpha=100000` completed all 10 cases. The launcher
  and four in-progress `alpha=1000000` workers were terminated before any
  `alpha=1000000` summaries completed.
- Passed: aggregation over `alpha=10000` anchor plus completed
  `alpha=100000` sweep wrote `20` case rows, `2` mean rows, and completion
  errors `[]`.

## Result

| Alpha | TRE mean +/- SD | FG stdLogJ mean +/- SD | FG J<=0 % | FG BE | FG diffusion |
| --- | --- | --- | --- | --- | --- |
| 10000 | 1.037 +/- 0.153 | 0.265395 +/- 0.063547 | 0.0349 | 0.006341 | 0.016738 |
| 100000 | 1.003 +/- 0.154 | 0.155840 +/- 0.033335 | 0.0005 | 0.001099 | 0.008170 |

Current full-crop NCC + dense FD-BE reference at alpha `10000` remains better
on TRE (`0.966 +/- 0.151`) but is less smooth by foreground stdLogJ
(`0.185452 +/- 0.060879`) than corrected alpha `100000`.
