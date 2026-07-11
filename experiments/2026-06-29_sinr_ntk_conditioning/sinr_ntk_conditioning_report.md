# SINR NTK Conditioning Report

Status: `completed`

This report compares the local dense update action `B s` for DirectCP against `B J_theta J_theta^T s` for the SINR control generator. Dense kernels are not materialized.

## Action Metrics

| State | Gradient | D_U | R_U | A_U | A_C | lambda_s | PSD |
|---|---|---:|---:|---:|---:|---:|---|
| init | data | 0.2888 | 0.9574 | 1.208e+06 | 1.658e+06 | 2.173e+05 | pass |
| init | total | 0.2888 | 0.9574 | 1.207e+06 | 1.657e+06 | 2.172e+05 | pass |
| final | data | 0.4638 | 0.8859 | 2.381e+05 | 3.891e+05 | 6.985e+04 | pass |
| final | total | 0.4649 | 0.8854 | 2.352e+05 | 3.973e+05 | 7.083e+04 | pass |

## Interpretation

- The SINR CP-space kernel is not acting like a scalar-rescaled DirectCP update on this case: the best scalar-residual remains `0.957` at init and `0.885` at final.
- The task-gradient dense-action amplification is very large (`2.38e+05` to `1.21e+06` for data gradients), which supports a severe local scale mismatch.
- Data-loss and total-loss rows are nearly identical in this setting, so the observed kernel action is driven mainly by the image term rather than the BE term.
- Raw amplification should not be read as a one-to-one learning-rate ratio because Adam, finite training, and NTK normalization conventions change effective step sizes.

## Matrix Size / Rank

- CP coefficients: `388161`
- SINR parameter count: `133379`
- Rank upper bound: `133379`
- Nullity lower bound: `254782`

## Effective Condition Probe

The full CP-space condition number is infinite because the SINR CP-space kernel
has a nullspace of dimension at least `254782`. As a practical finite proxy, a
64-dimensional randomized range-finder Ritz probe was run in parameter space on
`J_theta^T J_theta`, whose nonzero eigenvalues match those of `J_theta
J_theta^T`.

| State | Ritz dim | min Ritz | median Ritz | max Ritz | max/min proxy | p95/p05 proxy |
|---|---:|---:|---:|---:|---:|---:|
| init | 64 | 65511.7 | 127598.4 | 13794139.0 | 210.6 | 2.95 |
| final | 64 | 35530.7 | 48797.0 | 2670927.2 | 75.2 | 2.69 |

These are leading-subspace proxies, not exact smallest-nonzero condition
numbers. They are useful because the maximum Ritz values match the independently
estimated top-eigenvalue scale from the action run.

## Interpretation Guardrails

- This is a local NTK/SGD diagnostic, not a global optimizer proof.
- The dense metrics are sampled over the fixed raw-image-positive OASIS foreground.
- The prior cumulative-update proxy is not reused as evidence for these rows.
