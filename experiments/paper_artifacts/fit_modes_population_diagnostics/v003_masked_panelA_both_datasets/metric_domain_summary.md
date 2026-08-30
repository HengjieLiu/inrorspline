# v003 Camera-Ready Diagnostic Metric Domain Summary

v003 is the camera-ready diagnostic candidate where both datasets use anatomically appropriate fixed-domain masks for Panel A and Panel B.

| Dataset | Panel | Plotted metric | Evaluation domain |
|---|---|---|---|
| OASIS | A: unsupervised registration | `1−NCC` | Fixed raw `image > 0` mask |
| OASIS | B: supervised field fitting | rRMSE (%) from masked MSE | Fixed raw `image > 0` mask |
| DIR-LAB/LungCT | A: unsupervised registration | `1−NCC` | Fixed T00 lung mask on canonical 1 mm crop |
| DIR-LAB/LungCT | B: supervised field fitting | rRMSE (%) from masked MSE | Fixed T00 lung mask on canonical 1 mm crop |

Panel A uses the same global NCC formula in both datasets: `ncc_loss = -corr(fixed, warped)`, and the plotted value is `1 + ncc_loss`. The difference is the fixed-domain mask: OASIS uses fixed raw `image > 0`; DIR-LAB uses the fixed T00 lung mask.

Panel B is unchanged from v002. It plots `100 × sqrt(masked MSE / (masked target power + 1e-12))`, using SITReg normalized-grid targets for OASIS and pTVReg normalized-grid targets for DIR-LAB.

Training objectives are not changed by v003. The refresh changes the diagnostic/evaluation domain for OASIS Panel A only.

Safe claim: the plotted Panel A metric is now method-consistent within each dataset and uses each dataset's fixed anatomical domain. Safe caveat: OASIS and DIR-LAB use different anatomical masks, and Panel A training domains are still not identical across all methods.
