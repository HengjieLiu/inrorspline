# SINR NTK Conditioning Technical Log

## Known Limitations

- This is a local NTK/SGD diagnostic, not a global optimizer proof.
- The primary run is OASIS rank9 only; LungCT extension is a follow-up unless
  explicitly launched later.
- Adam, regularization, finite training, and nonlinear image warping can change
  the realized trajectory relative to the local tangent-kernel prediction.
- The saved final SINR checkpoint is loaded from repo-generated experiment
  output using `torch.load(..., weights_only=False)` because PyTorch 2.6 changed
  the default checkpoint loading behavior.
- Dense NTK matrices are intentionally not materialized.

## Execution Notes

- 2026-06-29: User approved full OASIS first-pass execution with
  supervisor/executor role separation.
- 2026-06-29: Supervisor requires a dry-run and OASIS smoke before the full
  init/final data/total pass.
- 2026-06-29: Implemented
  `scripts/experiments/update_alignment/run_sinr_ntk_conditioning.py`.
- 2026-06-29: Docker `py_compile` passed.
- 2026-06-29: Dry-run completed and wrote plan JSONs.
- 2026-06-29: OASIS rank9 smoke completed with `1` action row, `2`
  spectrum-probe rows, `errors=[]`, and passing PSD check.
- 2026-06-29: Full OASIS rank9 first pass completed with `4` action rows,
  `128` spectrum-probe rows, `errors=[]`, and passing artifact verification.

## Key Results

- Matrix-size/rank: `388,161` cps4 CP coefficients, `133,379` SINR parameters,
  rank upper bound `133,379`, nullity lower bound `254,782`.
- Init data gradient: `D_U=0.2888`, `R_U=0.9574`, `A_U=1.208e6`.
- Init total gradient: `D_U=0.2888`, `R_U=0.9574`, `A_U=1.207e6`.
- Final data gradient: `D_U=0.4638`, `R_U=0.8859`, `A_U=2.381e5`.
- Final total gradient: `D_U=0.4649`, `R_U=0.8854`, `A_U=2.352e5`.
- Random Rayleigh probes are far smaller than task-gradient Rayleigh values,
  indicating that the actual image-gradient direction is strongly aligned with
  amplified kernel directions.

## Current Interpretation

The OASIS rank9 result supports the claim that the SINR coefficient generator
induces a rank-deficient and strongly anisotropic CP-space tangent kernel. On
the actual image-registration gradient, `B J_theta J_theta^T s` is not well
explained by a scalar-rescaled `B s` DirectCP update. This is mechanism evidence
for INR-induced conditioning/scale mismatch, with the caveat that raw
amplification is not a one-to-one learning-rate ratio under Adam.

## Effective Condition Probe

- 2026-06-29: User asked whether an effective finite condition number could be
  estimated in the nonzero subspace.
- Supervisor decision: exact full-space condition number is already infinite
  from nullity. Exact smallest-nonzero eigenvalue estimation is risky for a
  large rank-deficient operator, so the first practical extension is a
  randomized parameter-subspace Ritz probe for `J_theta^T J_theta`.
- First 64D raw random-subspace probe did not capture the known top eigenvalue
  scale, so it was rejected as an effective condition proxy.
- Replaced it with a randomized range-finder Ritz probe using `Q=orth(H Omega)`
  for `H=J_theta^T J_theta`.
- 64D range-finder result: init max/min Ritz proxy `210.6`, final max/min Ritz
  proxy `75.2`. The top Ritz values (`1.379e7` init and `2.671e6` final) match
  the action-run power-iteration scale, so this is a useful leading-subspace
  condition proxy.
- This is still not the exact global nonzero-subspace condition number; it is a
  practical finite estimate for the leading supported subspace.
