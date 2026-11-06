# P81 analysis completion ledger

Update2026-11-05: fitting execution completed at08:50:14PST with all1680 jobs
and2640 endpoint records, zero runtime failures/OOM retries. The first review
script verifies complete unique planned coverage and reproduces all CSV
foreground errors from endpoint metadata; it records source-separated summaries,
matched comparisons and ten-pair averages across the four correlated sources.
Artifacts: `execution_20261105_v3/reports/completed_review_v1/` outside Git.
This completes queue execution and a first descriptive review, not the full
scientific ledger.575/840 main D0 solves are unresolved; all1680 main SVF
final stages reached fixed iteration limits. Convergence disposition and the
remaining Taylor/transition, figure and notebook deliverables remain open.
The startup status below is retained as historical context.

This ledger distinguishes the complete fitting matrix from the work still
needed for a scientific report. Current revision: v1.3, launched2026-11-05
01:18:25PST (09:18:25Z) after scoped validation. Initial verified snapshot:
8live GPU0 workers,1672pending,0failures; full study remains in progress.
The fitting job definitions cover the requested full matrix; that does not
mean those jobs or the analyses below have completed.

| Deliverable | Implemented/evidence status | Completion requirement |
| --- | --- | --- |
| Forty immutable reference endpoints and seeded cohort | P0 hashes, finite values, geometry/coordinate round trips and rank0 source metric replay passed; v3 launch binds and checks the input/source/configuration receipts | Retain immutable inputs and drift checks during execution |
| Full fitting matrix | D0 and SVF each840 configurations; zero/global_affine_ls SVF starts, D1/replay,40 anchors and24 solver diagnostics queued in1680 worker jobs | Execute, retain failures/missing cells and report actual coverage |
| Native endpoint and derivative/label metrics | Implemented in the worker for saved/reloaded endpoints; reference Jacobian metrics use the same evaluation stencil | Evaluate each completed endpoint and summarize coverage; a missing endpoint is not a zero-error observation |
| Multistep SVF stability and integration scope | Both130-step pilots improved foreground fit; raw pilot failed its exploratory full-box7/8 RMSE threshold of0.05mm. Independent frozen-coefficient review passed the existing worker foreground criterion, bound in the launch receipt | Preserve raw failure/full-box errors and foreground scope; this is not a guarantee for every queued trial |
| P0 local Taylor residuals at8/16/32mm and derivative variation | Planned descriptive analysis; not yet implemented or computed | Implement a fixed physical-window analysis across the reference fields; document window/domain/stencil and boundary coverage before comparison |
| Local affine/blending derivative decomposition | Implemented sampled diagnostic for Gaussian/tree families | Report sample/domain scope; do not treat it as a full spatial transition analysis |
| Neighboring affine disagreement and transition-region residual/fold maps | Not yet implemented | Define neighborhood/transition masks independent of favorable results, compute spatial diagnostics and retain failures |
| Split preservation | Runtime compares up to512 fit-sample analytic values before/after each split; mathematical unit tests cover exact zero-contrast inheritance and generic/separable rendering parity | Report this evidence accurately; native dense/renderer comparison is not run at every production transition |
| Translation-tree same-K supplements | Saved at K64/128/256; D0 receives checkpoint LS, SVF saves the coarse growth-stage state | Tag optimization stage and unequal refinement; SVF supplements are not convergence-matched native endpoints |
| Coverage report | Implemented CSV coverage and source-stratified mean/SD for selected foreground error/time measures | Keep `study_complete=false` until the final analyses are delivered |
| Final paired comparisons and figures | Paired differences, median/IQR, win/tie/loss, parameter/time/same-K/tree plots, qualitative overlays and Chinese scientific synthesis remain post-fit work | Use pair-level comparisons, actual parameter counts and failure denominators; distinguish fitting from initialization/search cost |
| Review notebook | Output-free notebook source and externally executed review notebook remain pending | Execute in the intended environment with inline tables/figures; store the executed copy under ARTIFACT_ROOT |
| Native source-SVF checkpoint replay and velocity LS | Conditional diagnostic; checkpoint availability metadata is not a successful velocity reconstruction | Resolve dependencies, reproduce the recorded seven-squaring endpoint, then run velocity analysis or document unresolved status |
| Dense-SVF, two-stage D1 or small constrained-QP references | Conditional diagnostics from the plan, not baseline queued fitting jobs | Apply only the declared trigger and report scope/cost; do not imply they were already run |

The v3 conditioner changes numerical parameter coordinates, not the
foreground objective. A corrected-conditioning cap768 D0 pilot reached its
LS stopping criterion while still producing an approximately14,322mm field
outside the fitting ROI. This is a warning about unconstrained extrapolation,
not a reason to assert that the objective includes or bounds the background.
Full-box RMS scaling is not a full-box field constraint. The cap1536/3072 D0
pilots did not converge and must not be called capacity optima. All SVF
families/replays now use zero/global_affine_ls starts: the same ROI-fitted
12-parameter affine displacement is embedded exactly and interpreted as
initial velocity, without a matrix logarithm. D0/D1 local LS remains unchanged.
Both130-step pilots (100 quarter-grid,25 half-grid,5 native) improved their
fit MSE, but the raw pilot failed its exploratory full-box7/8-squaring
RMSE threshold. Independent frozen-coefficient replay passed the
already-existing worker foreground criterion; no fit, renderer or protocol
configuration changes are made. This is not a claim of bounded final
extrapolation or universal stability.

| SVF start | Fit MSE, initial→final (mm²) | Foreground7/8 RMSE / existing threshold (mm) | Full-box7/8 RMSE / maximum vector difference (mm) |
| --- | --- | --- | --- |
| zero | 19.2210→6.7665 | 0.001197 / 0.13006 | 0.17782 / 46.4123 |
| global_affine_ls | 16.4307→6.7258 | 0.001219 / 0.12967 | 0.21153 / 53.1270 |

The worker threshold is `max(0.001,0.05*sqrt(fit_MSE_mm2))`, applied to the
full raw-positive foreground mask. The much larger full-box errors occur
outside the fit ROI and remain mandatory diagnostics. Passing the foreground
criterion cannot be described as passing a full-box integration-accuracy
criterion. The raw `stability/full_box_v2/FAILED.json` remains intact; any
acceptance review must state this distinction rather than overwrite it.

Scientific completion requires an explicit disposition for every row: done
with evidence, failed with evidence, unresolved with a reason, or conditional
and not triggered. Queue termination alone cannot close this ledger.
