# Supervisor / Executor Log

## Round 1

**Supervisor:** The experiment is useful only if it cleanly separates the true
SINR CP-space NTK action from the earlier cumulative-update proxy. Start with
OASIS rank9 where the rank-deficiency claim is exact and the saved final SINR
state already exists. Do not materialize dense kernels.

**Executor:** Create a fresh review root, update the registry, implement the
runner, then run dry-run and smoke before the full first pass.

## Round 2

**Supervisor:** Dry-run passed and smoke completed with finite metrics and a
passing PSD check. The smoke result is not a trivial scalar rescale at init:
sampled dense cosine is low and amplification is large. Proceed to the planned
OASIS rank9 full first pass, but keep interpretation local to the tangent
kernel.

**Executor:** Launch OASIS rank9 init/final, data/total gradients, with 32
random spectrum probes and 10 power iterations. Monitor runtime because the
smoke took about 52 seconds for a reduced kernel workload.

## Round 3

**Supervisor:** Full pass completed cleanly. The result is stronger than a
conditioning-only footnote: SINR's CP-space kernel is rank-deficient by
construction and the measured dense action is not a scalar-rescaled DirectCP
action. Be careful not to equate raw amplification directly with the empirical
learning-rate ratio because Adam and NTK normalization alter effective scale.

**Executor:** Regenerated the report with interpretation guardrails, verified
tables and plots, checked `py_compile`, and confirmed no
`run_sinr_ntk_conditioning.py` process remains active.

## Round 4

**Supervisor:** The full CP-space condition number is infinite, but the user
asked whether a finite effective condition can be probed. Do not run an exact
smallest-nonzero ARPACK solve first; it is likely to be unstable or slow for a
large rank-deficient kernel. Start with a randomized parameter-subspace Ritz
probe and label it clearly as a proxy.

**Executor:** Add `run_sinr_ntk_param_spectrum.py` to estimate the spectrum of
`J_theta^T J_theta` in a randomized parameter subspace for init/final SINR
states.

## Round 5

**Supervisor:** The first raw random-subspace Ritz probe missed the known
top-eigenvalue scale, so it is not an acceptable condition proxy. Upgrade to a
range-finder Ritz probe using `Q=orth(H Omega)`.

**Executor:** Patched the spectrum probe, reran the 64D init/final estimate, and
recorded leading-subspace condition proxies of `210.6` at init and `75.2` at
final. The top Ritz values match the prior power-iteration scale.
