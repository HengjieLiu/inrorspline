# P82 technical log

## Known limitations

- Twenty minutes is an achieved-accuracy budget, not a convergence guarantee.
- Exact high-rank sigmoid trees may time out before reaching their target size.
- Full-box fitting differs from P81 foreground-only fitting; old errors are not
  reused as new points.
- Gaussian grid changes require approximate initialization by LS projection;
  the final basis and all evaluation operators remain exact.
- Concurrent GPU workloads make runtime descriptive, not isolated benchmarking.
- Operator-boundary deadline checks may finish an in-flight kernel after the
  deadline; record measured active time and do not hide overshoot.

## Implementation

2026-11-05: isolated source/config/spec created after user approval. Exact grid
and tree operators, GPU LSQR, compact topology, nested-knot transfer, cumulative
timer and immutable checkpoint pointers implemented. Initial11 unit contracts
passed in the configured Docker environment. Production not yet launched.

2026-11-05 22:43:11PST: launched frozen production execution
`execution_20261106T0644Z`; initial20 workers (GPU0:8,GPU1:4,GPU2:4,GPU3:4)
and148pending. Independent memory admission retained all requested maxima.
Preflight:14 unit tests,10 native operator profiles,2 worker smokes passed.
The three grid families passed forward/adjoint resource profiles at the maximum
dense-sized cap. Cps2 replay normalized error0, physical RMSE7.0018e-16mm.
First smoke B-spline joint fit converged at all three stages; the short20-second
adaptive-affine smoke reached3072parameters and correctly recorded a timeout.

Report validation fixed empty-log-axis initialization and removed a pandas
dependency from the review notebook. The externally executed notebook succeeds
in the existing Docker kernel without environment changes. Earlier staging
execution `execution_20261106T0634Z` remains preserved; no production fits used
it. The production source/configuration hashes were frozen after these fixes.

The controller automatically refreshes CSV/JSON and curves during execution,
and generates final error maps and an executed notebook after all168 identities
have terminal dispositions. Scientific convergence and queue completion remain
separate. No production conclusions have yet been drawn.

22:47PST monitoring: controller healthy;4 converged endpoints,20running,
144pending at the latest scheduler snapshot. The four smallest B-spline fits
converged, with native/joint RMSE differences below1e-9mm on both targets.
This is a matched-final-basis solver consistency check, not a result for all
families or budgets. Four GPU2 startup OOMs were recorded; all automatically
requeued with retained timing and smaller rank tiles, including two reassigned
to GPU1. Retry workers are progressing; no terminal failures at this snapshot.
The live report has completed points; final curves and notebook remain pending.
