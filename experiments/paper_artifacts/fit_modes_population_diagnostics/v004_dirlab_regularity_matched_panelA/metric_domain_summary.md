# v004 Camera-Ready Diagnostic Metric Domain Summary

v004 is the camera-ready Figure 3 candidate. It preserves the v003 fixed-domain metric convention and additionally regularity-matches the DIR-LAB Panel A single-stage BSCP operating points.

| Dataset | Panel | Plotted metric | Evaluation domain | v004 change |
|---|---|---|---|---|
| OASIS | A: unsupervised registration | `1−NCC` | Fixed raw `image > 0` mask | unchanged from v003 |
| OASIS | B: supervised field fitting | rRMSE (%) from masked MSE | Fixed raw `image > 0` mask | unchanged from v003 |
| DIR-LAB/LungCT | A: unsupervised registration | `1−NCC` | Fixed T00 lung mask on canonical 1 mm crop | INR-BSCP `a=20000`; D-BSCP `a=50000` |
| DIR-LAB/LungCT | B: supervised field fitting | rRMSE (%) from masked MSE | Fixed T00 lung mask on canonical 1 mm crop | unchanged from v003 |

Panel A uses global NCC: `ncc_loss = -corr(fixed, warped)`, plotted as `1 + ncc_loss`. The plotted DIR-LAB Panel A metric is evaluated inside the fixed T00 lung mask for all four methods.

For DIR-LAB Panel A, the single-stage BSCP alphas are selected from the Pareto sweep to approximately match foreground `SD(log J)` to IDIR and MR-D-BSCP, rather than keeping identical nominal alpha values across parameterizations.

Safe claim: v004 compares DIR-LAB Panel A methods at approximately matched deformation regularity and a common fixed-lung evaluation domain. Caveat: the optimization/training domains are still not identical across all methods.
