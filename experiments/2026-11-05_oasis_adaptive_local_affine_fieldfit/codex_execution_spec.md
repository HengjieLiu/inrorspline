# P81 execution specification

Current runtime: the [scheduling addendum](scheduler_v1.md) supersedes the
historical GPU0-only policy below. Numerical protocolv1.3 is unchanged.

Status: v3 launched2026-11-05T09:18:25Z (01:18:25PST), physicalGPU0.
Initial verified snapshot:8live workers,1672pending,0failures,no warning.
76CPU tests,17native profiles and3worker smokes (7endpoints) passed; independent
multistep review passed the existing foreground criterion. The raw full-box
sensitivity failure and all superseded v2 artifacts remain preserved.
Date: 2026-11-05 (America/Los_Angeles).

## Scope and decisions

Implement and execute the [current protocol](revised_plan_v1.3.md).
The user confirmed the existing seed-3 first10, original SITReg and three
8→4→2 spline references at lambda1. The latest instruction requires identical
840-configuration D0/SVF coverage, GPU0 only, memory-based parallel admission
up to eight jobs, and record-and-continue handling for individual failures.
D1 remains a smaller mechanism branch. Lambda belongs to reference registration.

Keep source registration scripts and historical numerical results immutable.
Do not launch inference/training or reuse old registration launchers to fit
fields. No rsync, container recreation, commit, or push is part of this task.

## Implementation boundaries

Implementation source locations:

- `scripts/experiments/oasis_polyaffine_fieldfit/`: source selection, geometry,
  basis/adjoint, tree, solvers, renderers, certificate, metrics, resumable runner.
- `tests/test_oasis_polyaffine*.py`: mathematical, coordinate and queue gates.
- `protocol_v1.3.json`: prespecified fitting settings, frozen with source/input
  hashes after resource and actual multistep stability validation. Preserve
  v1.2 unchanged as the superseded numerical protocol.

Inspect/reuse pure utilities from `scripts/experiments/oasis_alpha_sweep/`:

| Existing source | Use and caution |
| --- | --- |
| `common.py` | Scalar reorientation, spline geometry, metric conventions; override all output roots; distinguish array axes from grid components and world axes |
| `evaluate_saved_baselines_first10.py` | SITReg channel-first zyx loader/conversion; resolve source paths from selected report rows, not legacy defaults |
| `run_oasis_saved_field_bspline_fit_one.py` | Prior field-fitting coordinate/provenance reference; its full-volume voxel loss/Adam protocol is not P81's ROI/mm LS protocol |
| `dl_ncc_diffusion.py` | Endpoint-aligned spline expansion; fitting does not use its NCC objective |
| `bounded_composed_bspline.py` | Pull composition/field sampling; avoid assuming old B-spline caps apply to affine/Gaussian weights |
| `bspline_svf.py` | Native-grid scaling/squaring and retained/refined distinction; confirm import closure and padding |

## Ordered gates

1. Resolve sources exactly as v1.3 §3; validate acceptance/report hashes and
   all40 field hashes, pairs, geometry, finite values. Save external inventory.
2. Implement and test canonical adapter; reproduce historical warp/metrics and
   round trips, with separate image/label/field padding settings.
3. Implement shared D0 basis/linear operator and split transfer; pass small
   float64 reproduction, adjoint, derivative and preservation checks.
4. Run highest-budget native D0/V resource probes for all seven families and
   three middle-budget D1 probes; freeze sample hashes, lattice counts,
   prespecified stopping rules, serialization and conservative reservations.
   These are numerical/resource smokes, not converged capacity pilot results.
5. Pass the actual production-limit LS and multistep SVF stability gate. The
   rank000_bounded2 B-spline1536 case must run the checkpoint LS limit and its
   permitted extension if needed, then both zero/global_affine_ls SVF starts for at least
   100 quarter-grid,25 half-grid and5 native steps. Record coefficient and
   full-box field magnitudes, finite checks and loss improvement; include
   other-cap and nearly unsupported-column regressions. Bind explicit gates
   and source/config/input hashes in `STABILITY_PASSED.json`. Renew affected
   CPU, native resource and serialized-worker checks after numerical edits.
6. Queue full D0 and SVF matrices:840 configurations each; retain two V initial
   states (zero and global_affine_ls), so V contributes1680 initialization trial endpoints.
   Tree paths share three budget checkpoints; dictionaries use independent fits.
7. Queue24 D1 pilot configurations and24 new replay configurations/32 trials,
   plus24 frozen-geometry solver diagnostic jobs, with explicit parent-checkpoint
   dependencies.40 affine anchors are included. Total1680 worker jobs.
8. Recompute serialized results, generate external Chinese report/notebook,
   record phase coverage, limitations and code/config/input provenance.

LSQR, nonlinear fitting and sampled optimizer diagnostics share deterministic
32³ full-declared-box feature RMS scaling with a0.01 minimum scale. These
points condition coefficients only; the raw-positive foreground endpoint loss,
fit samples and sample weights remain unchanged. This adds no background loss,
ridge or deformation penalty. In underdetermined/ill-conditioned systems,
scaling and stopping alter numerical solution selection and may alter
extrapolation. The floor does not bound coefficients or prove SVF stability.
Save conditioner geometry, RMS/scale summaries and hashes. Keep v2 outputs
separate; do not merge them as numerically identical v3 results.

At nonlinear checkpoints, abort an attempt if training or validation MSE
exceeds its corresponding `min(1e6,max(100,100*initial_MSE))` mm² threshold.
This is a recorded failure condition, not a changed loss or output clipping;
independent jobs continue. Retaining the best earlier state cannot convert a
divergent path into a successful attempt.

SVF initialization uses the same12-parameter global-affine displacement LS on
the same foreground samples for every family and frozen replay. Embed that
field exactly through `basis.coefficients_for_affine`, then interpret it as
initial velocity, without a matrix logarithm. Optimize the full selected
basis thereafter. This replaces local-LS SVF starts after observed large
out-of-ROI extrapolation; D0/D1 local-LS fitting remains unchanged. A converged
cap768 D0 pilot still produced approximately14,322mm outside the ROI, and
cap1536/3072 pilots were unresolved; no full-box bound is claimed.

The130-step SVF pilots improved foreground fit but failed the raw pilot's
exploratory0.05mm full-box7/8 integration-sensitivity threshold. The separate
frozen-coefficient review uses the pre-existing worker foreground criterion,
`max(0.001,0.05*sqrt(native_fit_MSE_mm2))`; its foreground RMSE is about0.0012mm
for each initialization. Preserve raw failure and full-box RMSE0.178/0.212mm,
including exterior maximum vector differences46.4/53.1mm. No fitting,
renderer or protocol change is authorized by this review. Any accepted
receipt must state its foreground scope and retained full-box limitation.

Runtime uses the running default container and its configured identity, as
verified in ignored `private/data_sources.md`. Repo is read-only inside it.
Use `OASIS_ROOT` and `ARTIFACT_ROOT` from its environment; explicit source and
output arguments override stale launcher defaults. Set cuDNN F/T and TF32 off.
All CUDA workers set CUDA_VISIBLE_DEVICES=0. User authorized at most eight
jobs on physicalGPU0; profiles determine a smaller admissible count when needed.
Reserve measured peak+2GiB plus2GiB device headroom; keep24GiB host RAM and
300GiB artifact space available. Each job has a12-hour wall limit and60-minute
no-activity watchdog; terminate only its identity-verified process group.
Record exceptions/OOM/timeouts/dependency failures, preserve attempts, continue
other jobs. No automatic infinite retries. If no workers run and admission is
impossible for60minutes, defer remaining jobs explicitly instead of hanging.

## Output and review contract

Run key: `oasis_seed3_first10_adaptive_local_affine_fieldfit_v1`.
Run artifacts: `${ARTIFACT_ROOT}/runs/<key>/<unique-run-id>/`.
Active corrected run ID: `execution_20261105_v3`; v2 remains retained.
Review artifacts: `${ARTIFACT_ROOT}/repo_experiments/<key>/<unique-run-id>/`.
Use unique IDs and preserve failed attempts and all historical input bytes.
Capture exact image identity, versions, GPU settings, source revision/dirty
diff or source hashes, manifest selectors, actual units and timing semantics.

Review tables, metrics, generated audit reports and executed notebook copies
are external artifacts. Before a later commit/source-only audit run
`python -B scripts/experiments/check_tracked_notebook_outputs.py` in the intended
container environment. Scientific code and queue tests, source/input acceptance,
native forward/backward profiles, actual multistep stability and serialized
smoke evaluation must pass before production. Executed uncommitted sources
are archived externally with hashes.

Track remaining scientific deliverables in the
[analysis completion ledger](analysis_completion_ledger.md). The current
reporter provides coverage and selected mean/SD summaries. Physical-window
Taylor/derivative variation, neighboring-affine/transition maps, complete
paired statistics, figures and an executed review notebook remain explicit
post-fit work. Conditional source-SVF velocity replay is unresolved; other
conditional diagnostic branches are not part of the baseline queued matrix.

## Runtime-policy override — 2026-11-05

The user-authorized [multi-GPU scheduling continuation](scheduler_v1.md)
supersedes GPU0-only/max8/no-retry rules in this original execution specification.
New limits: GPU0/1 max16 each; GPU2/3 max4 each, independent memory admission,
60-second polling and automatic CUDA OOM requeue. Scientificv1.3 remains frozen.
