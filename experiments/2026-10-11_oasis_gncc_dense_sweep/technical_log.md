# P55 technical log

## Limitations

- First10 test-case tuning is exploratory, not held-out confirmation.
- Equal lambda is not assumed to imply equal effective regularization across losses.
- Calibration uses already available exploratory cases/states, not independent validation.
- Shared GPU workloads make runtime diagnostic. CUDA grid_sample backward may vary.
- Historical initial-normalized global-NCC refinements are not interchangeable controls.
- IDIR remains deferred. No bandwidth, learning-rate or pyramid sweep is included.

## 2026-10-11

- User approved two GNCC sweeps and a three-NCC table plus2x3 ARC figure.
- Implementation/preflight in progress; previous numerical helpers left unchanged.
- Preflight:12 new GNCC/report tests,16 prior LNCC/geometry tests (including native
  VFA NCC/diffusion value-and-gradient parity), and10 prior DL-loss tests passed.
  Dry-run matrix verified140 production candidates and3 distinct smoke controls.
- Approved controller started with automatic smoke, LNCC reproduction, calibration,
  production and final-report gates. Current launch GPU free memory was approximately
  38.6/47.4/43.3/47.4GiB; existing GPU0/2 jobs were not changed.
- All three GPU smokes passed through four-stage optimization, DVF serialization,
  canonical metrics and per-label reconstruction. Peak reserved CUDA memory was
  approximately1.63GiB for either GNCC loss and1.86GiB for LNCC.
- Full rank0/lambda1 LNCC reproduction started at2026-10-11 13:51:38UTC.
  Production remains gated on reproduction and diagnostic calibration.
- Reproduction passed: stopping epochs325/250/200/525 and best epochs290/235/197/525
  exactly matched the accepted reference; max voxel-field difference and all three
  accuracy-metric differences were exactly zero.
- Calibration passed for10 cases x3 common states. Median LNCC/global gradient-RMS
  ratios were2.590334 for signed GNCC and1.302883 for squared GNCC, with p10-p90
  ranges1.127579-3.735925 and0.644040-1.892669 respectively. These are diagnostics,
  not universal equivalences; the approved raw lambda grid remains unchanged.
- Production started2026-10-11 13:55:09UTC (06:55:09PDT), controller PID89865.
  Four initial workers ran on GPUs0/1/2/3, one added worker perGPU. Final aggregation
  and the requested three-NCC tables/2x3 figures are queued after all140 pass.
