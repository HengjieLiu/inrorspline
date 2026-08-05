# DirectCP 8-4-2 CPU/GPU Transition Kernel Benchmark

- Image shape: `(160, 224, 192)`
- Schedule: `8,4,2`
- Warmups/repeats: `1/2`
- CPU threads: `6`
- Tolerance: `1e-05` max absolute difference

Expansion and knot insertion are PyTorch tensor operations. In normal GPU DirectCP runs, tensors are CUDA tensors, so these operations already run on GPU. This benchmark quantifies CPU/GPU agreement and the isolated runtime of each component.

| operation | cps/transition | CPU median | GPU median | CPU/GPU speedup | max diff | within tol |
|---|---|---:|---:|---:|---:|---|
| dense_expand | 8 | 0.084276s | 0.002516s | 33.50x | 0 | True |
| dense_expand | 4 | 0.074103s | 0.001342s | 55.22x | 0 | True |
| dense_expand | 2 | 0.088280s | 0.001131s | 78.05x | 0 | True |
| pure_knot_insertion | 8->4 | 0.001381s | 0.000612s | 2.26x | 0 | True |
| pure_knot_insertion | 4->2 | 0.006088s | 0.000384s | 15.87x | 0 | True |
| preservation_check | 8->4 | 0.408180s | 0.140045s | 2.91x | 0 | True |
| preservation_check | 4->2 | 0.425434s | 0.003391s | 125.44x | 0 | True |

Interpretation:

- `dense_expand` is the B-spline control-grid to dense-field expansion.
- `pure_knot_insertion` is control-grid refinement only, with no dense preservation check.
- `preservation_check` includes pure knot insertion, inserted-control dense expansion, and dense difference metrics against the previous field.
- Production DVF timing should avoid `preservation_check` unless diagnostics are explicitly requested.
