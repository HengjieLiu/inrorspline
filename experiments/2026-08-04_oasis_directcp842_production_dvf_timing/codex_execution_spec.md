# Codex Execution Spec: DirectCP Production DVF Timing

## Objective

Measure rank-0 OASIS input-image-to-DVF runtime after removing intermediate
dense diagnostic work from the default DirectCP `8-4-2` path. Separately
benchmark CPU/GPU runtime and agreement for dense expansion, pure knot
insertion, and preservation checks.

## Production Default

Production DirectCP DVF-only runs use:

```text
--output-mode dvf_only
--timing-detail deep
--prolongation-preserve-check false
--save-intermediate-dense false
```

Only final stage dense expansion is mandatory, because that is the requested
dense DVF output. Intermediate dense expansion and preservation metrics are
diagnostics and require explicit flags.

## Case And Methods

- OASIS seed-3 shuffled rank `0`: fixed `0333`, moving `0115`, original sorted
  index `158`.
- Original MR-D-BSCP / DirectCP `8-4-2`: alpha `1000`, lr `1e-3`, cps
  `8,4,2`, downsample `4,2,1`, epochs `400,800,1300`.
- `A_fast_decay`: alpha `1000`, base lr `1e-2`, stage lr
  `3e-2,1e-2,3e-3`, adaptive minimums `75,150,250`, patience `50,75,100`.

## Execution

1. `py_compile` modified and new scripts.
2. Run production `1,1,1` smoke for both methods.
3. Run diagnostic `1,1,1` smoke with `--prolongation-preserve-check`.
4. Run CPU/GPU transition-kernel benchmark.
5. Run full production rank-0 pair on GPU `0`.
6. Generate production DVF-only report comparing against previous conservative
   DVF-only timing.

## Roots

- Output root:
  `<external-run-root>/runs/oasis_directcp842_production_dvf_timing_alpha1000_v1`
- Review root:
  `experiments/2026-08-04_oasis_directcp842_production_dvf_timing/`
