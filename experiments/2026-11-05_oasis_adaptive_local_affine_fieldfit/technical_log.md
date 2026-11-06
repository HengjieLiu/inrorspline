# P81 technical log

## Known limitations

- The initial sections are historical planning/launch records. v2 was stopped
  after an optimization-conditioning failure; all attempts remain preserved.
  v3 launched2026-11-05 01:18:25PST after independent foreground-scoped
  multistep review; the raw full-box sensitivity failure remains preserved.
- The earlier67 CPU tests,17 native resource probes and serialized worker
  smokes did not detect instability during actual longer LS/SVF optimization.
  They cannot establish stability of the corrected numerical protocol.
- Corrected full-box conditioning does not constrain extrapolation: a cap768
  D0 pilot met LS stopping criteria but still produced approximately14,322mm
  outside the ROI; cap1536/3072 pilot LS remained unresolved. The SVF starts
  are revised to zero/global_affine_ls for all families/replays and passed the
  scoped multistep review. D0/D1 local LS remains unchanged.
- Both corrected SVF pilots completed130 updates and improved foreground MSE,
  but the raw pilot failed the exploratory0.05mm full-box7/8 integration RMSE
  check. Large exterior differences remain recorded; independent review using
  the existing worker foreground criterion passed. Full-matrix findings remain pending.
- File/header availability alone was not a coordinate proof. The later P0
  section records completed all40 hashes, finite scans and round trips plus
  source warp/metric parity. Converged full-matrix findings remain pending.
- The established first10 have been used repeatedly for exploratory tuning.
  This study is descriptive, including its non-pilot ranks2–9.
- The selected references have different regularization and transformation
  constructions; a shared lambda1 is not a matched regularity condition.
- Native source-velocity checkpoint reconstruction and conditional dense-SVF
  diagnostics remain separate work. Current field adapters do not need legacy
  training imports. New-directory writes and v2 native profiles passed historically.

## 2026-11-05 — request and resolved choices

The user requested a new experiment folder, an unchanged Markdown copy of the
attached plan, and a local-environment/data-aware revision. They clarified that
test10 means the long-used seeded subset and questioned using an extreme
rotation counterexample to motivate SVF for smooth OASIS fields.

After repository discovery, the user explicitly selected:

1. 8→4→2 at lambda1: original SITReg, prolonged unbounded DVF, two bounded
   increments per level, refined single total SVF with velocity diffusion.
2. Direct DVF main capacity/adaptation study, with a smaller predeclared D1/SVF
   follow-up. No unresolved user preference blocks this revised document.

The submitted plan is preserved byte-for-byte as `original_plan_v1.0.md`.
Attachment and saved copy SHA-256 both equal
`8559292adc68b0fbd416e7476863b0f46f6154f9d546a84ba023e5e6c346706b`.

## Local evidence inspected

- Read root instructions, project map/brief/methods, implementation index,
  registry/results/decision log, OASIS/fair-experiment standards, Docker and
  anonymous-release workflows, and ignored local data-source map.
- Initial Git worktree was clean at
  `12c6d14ba5f3d63cdd50f16e17581fed1cca2925`.
- Confirmed running default container, configured identity, read-only repo/raw
  mounts and read-write artifact bind. Current private runtime differs from
  historical host defaults. No container/job changes were made.
- Read-only Python in that container imported PyTorch2.6.0+cu118,
  NumPy2.2.3, SciPy1.15.2 and NiBabel5.3.2. OASIS and artifact roots exist.
- Parsed exact source report selectors: ten P43 unbounded candidates, ten P43
  historical SITReg references, ten P76 bounded-two candidates and ten P74
  continuation-release refined velocity-SVF candidates. Selected IDs match
  ranks0–9. Report acceptance markers are present and indicate completion.
- All40 selected fields exist and their headers are readable: SITReg
  `(3,160,224,192)`, the three spline families `(160,224,192,3)`.
- All40 required image/label paths exist (two images and two segmentations per
  pair). The rank0 fixed/moving image headers are LIA, `(160,192,224)`, 1 mm
  isotropic with equal affines. Remaining affine equality is a P0 check.
- Source loaders, earlier field-fit source, composition/integration utilities,
  and implementation notes establish the different stored vector conventions.
  See revised plan for the required physical-coordinate conversion.
- Existing first10 selected unbounded rows report foreground folding between
  approximately2.64% and4.80%. SITReg, bounded-two and SVF have much smaller,
  sometimes nonzero reported folding. These are historical source metrics read
  during audit, not recomputed P81 results or smoothness measurements.

Relevant repository evidence: P39 prior rank0 field fitting,
P75 first10 family ARC definitions, P76 bounded-two lambda sweep,
P74 refined-SVF sweep/continuation, and P79 first100 representation notes.
The new source selection fixes P43/P76/P74 provenance rather than choosing
independent results by whichever has the best metric.

An all-field checksum read was started immediately before a user interruption.
Its completion output was not recovered; it is not used as verification evidence.
The audit process was subsequently observed to have exited. No fitting process
was launched. P0 must write durable external hash receipts when implemented.

## Plan changes

The revised protocol fixes the seeded split and raw-positive foreground,
documents six-map bounded composition and single-SVF semantics, separates
source from fitting regularization, and moves all generated outputs outside Git.
It preserves the840-endpoint D0 matrix and reduces the initial D1/V comparison
to48 endpoints on eight pilot targets, with24 separately counted reverse-tree
replay endpoints. Resource/numerical calibration is explicitly not frozen yet.

The smooth-field hypothesis now has measurable local Taylor/derivative/blend
diagnostics. The rotation example is only a counterexample/unit check. No claim
of invertibility, zero folding, or expected affine superiority is made.

## Verification record

Confirmed original-file checksum equality, seeded pair identities and subset
hash, source selection counts, field/header availability and container imports.
Final checks passed: local Markdown links, new-document whitespace, exact
first10 IDs/subset hash,20 unique subject IDs, and one P81 registry entry.
`git diff --check` passed; the original attachment and saved plan compare equal.
The default-container notebook-output checker reported:
`No Git-tracked notebooks contain cell outputs.`

A read-only protocol review confirmed matrix counts and caught an ambiguity
in D1 wording: the contraction bound is on displacement Du, not I+Du. The
revised plan now states this explicitly and distinguishes endpoint counts
from actual solver/search costs. No source fitting code, data copies, GPU
experiments, staging, commits, or pushes are part of this setup.

## 2026-11-05 — authorized full DVF/SVF execution

The user superseded the reduced-SVF choice: match D0's40 targets/seven families/
three budgets with a full840-configuration SVF matrix, run all CUDA work on
physicalGPU0, admit parallel jobs according to measured memory up to eight,
record per-job failures and continue. This authorizes implementation, necessary
checks and launch without another approval round. v1.2 is now authoritative;
the original andv1.1 remain unchanged historical records.

Implemented isolated source adapters, seven basis families, exact split
inheritance, matrix-free scaled LSQR, physical-mm native renderers, D1 all-cell
Du certificate, differentiable SVF integration, common serialized evaluation,
diagnostics, source-bound launcher and durable controller. On diagonal OASIS
geometry, exact tensor-separable basis contractions avoid the full native
voxel-by-feature matrix; generic/fast forward and adjoint parity are tested.

P0 durable evidence under the new external execution root:

- All40 field hashes match the selected sources; all40 are finite and pass
  physical-coordinate round trips, maximum1.49e-8 normalized units.
- All ten pairs' image/segmentation affines, masks and disjoint shared sample
  indices pass. Source inputs remain read-only.
- Four rank0 source warp audits have no label mismatch. All12 historical
  Dice/HD95/symmetric-ASD values reproduce exactly through the physical adapter.
- `INPUTS_ACCEPTED.json` binds the manifest, source adapter and metric replay.

Native GPU0 resource checks passed17/17, with independently verified receipts:
all seven D0 and SVF families at the largest budget, and three D1 families at
the middle budget. Reservations including2GiB allowance are2816–6912MiB/job;
controller adds2GiB device headroom. Native SVF one-update fixtures took
approximately1.90–3.05s, including initial/final render; this is not throughput
or an ETA. Coarse stages, sharedGPU execution and LS convergence change cost.
These are numerical/resource smokes, not completed converged research results.

Queue tests include actual subprocess failure followed by independent success,
GPU0/max8 enforcement, memory accounting, PID reuse/orphan recovery, malformed
progress, deadline handling, input/source drift and completion artifact checks.
No source numerical changes are permitted after launch without a new revision.
Full native worker fit/save/reload/evaluate smokes precede the final launch.

External run family:
`${ARTIFACT_ROOT}/runs/oasis_seed3_first10_adaptive_local_affine_fieldfit_v1/execution_20261105_v2/`.
The queue contains1680 workers, including both SVF initializations, D1/replay,
40affine anchors and24solver diagnostics; translation same-K checkpoints share
their growing paths. Source/config/input hashes and an executed-source archive
will bind the launched queue. Scientific study completion is separate from
queue termination; missing conditional diagnostics/final analysis stay explicit.

## Launch and initial observation

- Launched2026-11-05T08:50:28Z =2026-11-05 00:50:28PST.
- Final CPU gate:67tests, exit0, source hashes unchanged; prior validation
  receipt/log preserved.17/17 native profiles and independent receipt audit pass.
- Updated worker smoke passes D0 growing-tree, adaptiveSVF growing-tree and
  bounded-spline runs, with seven native serialized/reloaded metric endpoints.
  Earlier smoke artifacts remain preserved. Smoke objectives are deliberately
  short and do not constitute converged capacity results.
- `launch_manifest.json` freezes all1680jobs, dependencies, memory profiles,
  source/config/input hashes and gate hashes. `executed_sources.tar.gz` archives
  the executed uncommitted implementation and numeric configuration.
- `queue_state.json` initially reports8running,1672pending, no warning/failures.
  Worker progress confirms real loading/fitting/evaluation onGPU0. PhysicalGPU0
  was observed at100% utilization; otherGPU allocations were not modified.
- Persistent `controller.log`, per-attempt `worker.log`, failure ledger,
  endpoint pointers and final coverage report remain under the external run root.
  No repository staging, commits, pushes, rsync or container changes occurred.

## v2 conditioning failure and v1.3 recovery

After launch, real optimization exposed a defect not covered by the short
resource checks: foreground-only feature RMS scaling could greatly amplify
columns with almost no foreground support. Observed LS coefficients reached
approximately1e11, producing large out-of-ROI fields and divergent subsequent
SVF optimization. Keeping an earlier best state could conceal a divergent
trajectory if treated as normal completion. These are numerical failures,
not evidence that the representation cannot fit smooth OASIS deformation.

The v2 controller and its eight workers were stopped; outputs, logs and failed
attempts remain intact under `execution_20261105_v2`. Original, v1.1 and v1.2
plans/configuration remain unchanged. Do not combine v2 endpoints with the
corrected run as if their numerical protocols were identical.

The current revision is [v1.3](revised_plan_v1.3.md), with explicit
`protocol_v1.3.json`. It uses one shared conditioner in LSQR, nonlinear Adam
and sampled Adam/L-BFGS diagnostics: deterministic32³ endpoint-including
samples of the declared physical box, feature RMS floored at0.01. The helper
records domain/sample geometry, RMS/scale summaries, floored columns and
scale hashes. The actual foreground samples, endpoint MSE and fitting weights
are unchanged; no background loss, ridge, coefficient penalty or clipping is
introduced. Nevertheless, preconditioning and finite tolerances affect
implicit scaled-norm solution selection in ill-conditioned or underdetermined
systems and can change unconstrained extrapolation. Warm starts may retain
different nullspace components. This revision is not only a speed change.

At nonlinear checkpoints, training or validation MSE exceeding its own
`min(1e6,max(100,100*initial_MSE))` mm² threshold fails the attempt explicitly.
Independent jobs continue and the failed attempt remains recorded. This is
an abort diagnostic, not an objective modification or a convergence proof.

The corrected launch requires a hash-bound `STABILITY_PASSED.json` from actual
production-limit LS and multistep zero/LS SVF on rank000_bounded2 B-spline1536:
at least100 quarter-grid,25 half-grid and5 native steps for each start,
coefficient/full-box magnitude and finite checks, and recorded loss behavior.
Other-budget and almost-unsupported-column regressions supplement this gate.
Affected unit, native resource and serialized worker checks must be renewed
after source changes. The sampled optimizer conditioner regression passed
with all seven diagnostic CPU tests in the configured default container.

Intended corrected run:
`${ARTIFACT_ROOT}/runs/oasis_seed3_first10_adaptive_local_affine_fieldfit_v1/execution_20261105_v3/`.
Status at this entry: validation pending; no claim that v3 has passed stability
checks or launched. Full D0/SVF coverage, two SVF initializations, D1/replay/
solver diagnostics, physicalGPU0/max8 admission and failure-and-continue
semantics remain unchanged. The conditioner floor does not bound coefficients
or guarantee stable SVF integration, and a finite pilot does not prove all
1680 worker jobs will succeed.

## v1.3 initialization revision and explicit analysis coverage

The full-box conditioner alone does not make an unregularized foreground-only
LS solution well behaved outside its fit domain. The corrected cap768 D0
pilot met its stopping condition but had an approximately14,322mm field
outside the ROI; cap1536/3072 pilot solves did not converge. Preserve the
pilot records and solver status. These results do not justify describing the
floor as a coefficient/field bound or calling the unresolved fits optima.

Before v3 launch, all seven SVF families and frozen-tree replays were changed
to two explicitly named starts: `zero` and `global_affine_ls`. The latter
fits a12-parameter global-affine displacement using the same foreground
sampled endpoint objective, embeds that field exactly into the chosen basis,
and treats it as initial velocity. It does not take a matrix logarithm or
claim identical affine endpoints before/after integration. Full-basis SVF
optimization follows; D0/D1 local LS is unchanged. The initialization helper
is `initialization.py`. The revised protocol checksum is
`0e644e69d1e5fd888ba0f4a8dec7cf588df9a1ebbf6e5b0cd6003f0a59d39338`.

Earlier candidate receipts stay preserved; a new hash-bound multistep gate
must verify this policy before full launch. No stabilization claim is made.

The source review also separated complete fitting-job definitions from
incomplete scientific analyses. See the
[analysis completion ledger](analysis_completion_ledger.md). P0 local Taylor
8/16/32mm and derivative-variation analysis, neighboring-affine disagreement,
transition residual/fold maps, final paired statistics/figures and executed
review notebook are pending. Native source-SVF checkpoint/velocity replay
is conditional and unresolved; dense-SVF/two-stage-D1/QP references remain
conditional rather than silently counted as baseline jobs.

Production split checks use up to512 analytic fit-sample points plus separate
mathematical/rendering unit tests, not a native dense comparison at each
transition. Translation-tree same-K SVF supplements capture coarse growth
states without main-budget native refinement and must not be interpreted as
convergence-matched endpoints. Current `report.py` is a coverage/selected
mean-SD reporter; it explicitly keeps scientific completion false.

## Multistep pilot: optimizer improvement versus integration scope

The frozen `stability/full_box_v2/` pilot ran100 quarter-grid,25 half-grid and
5 native updates per SVF initialization. Both trajectories stayed finite and
improved the native sampled endpoint MSE: zero19.2210357→6.7664812mm²;
global_affine_ls16.4306618→6.7258125mm². Native validation MSE was6.7859831
and6.7244066mm². This is limited pilot evidence, not convergence of the full
study or stable behavior throughout the unconstrained exterior.

The raw pilot remained **failed** solely because its exploratory full-box
seven-versus-eight-squaring RMSE cutoff was0.05mm: the measured values were
0.1778195 and0.2115294mm. Full-box maximum vector differences were46.4123 and
53.1270mm. These large discrepancies lie outside the fit ROI and must remain
visible in review outputs. They are not removed by foreground reporting.

The existing worker integration-sensitivity diagnostic instead uses the full
raw-positive foreground mask and compares its vector RMSE with
`max(0.001,0.05*sqrt(native_fit_MSE_mm2))`. Independent frozen-coefficient
measurements gave foreground RMSE0.00119737 and0.00121917mm, below the
existing thresholds0.13006 and0.12967mm. Foreground maximum differences were
0.025208 and0.020617mm, respectively. The different scopes explain why the
foreground diagnostic passes while the raw full-box check fails.

An independent `stability_review.py` acceptance audit is being added to replay
the frozen coefficients and bind the applicable foreground criterion, source
hashes and unchanged protocol to a separate review receipt. It changes no fit,
renderer, initialization or numerical configuration. Preserve raw
`FAILED.json`, raw summaries, coefficient files and all full-box measurements.
Any resulting acceptance must be described as finite/improving pilot behavior
with foreground integration sensitivity passing the existing criterion; it
does not establish full-box integration accuracy. At this entry, review and
launch remain pending. Protocol SHA256 remains
`0e644e69d1e5fd888ba0f4a8dec7cf588df9a1ebbf6e5b0cd6003f0a59d39338`.

## v3 accepted for scoped execution and launched

The final independent frozen-coefficient stability review passed using the
existing foreground integration-sensitivity criterion. The raw full-box
pilot failure, large exterior differences and unresolved high-budget D0
solves remain recorded; this is not full-box accuracy acceptance or a claim
that all future jobs will succeed. Final gates passed76CPU tests,17native
resource profiles and3serialized-worker smokes covering7endpoints. The
configuration and scientific source hashes match their bound receipts.

The full v3 queue launched2026-11-05T09:18:25Z, or01:18:25PST. The controller
identity was verified live; the initial snapshot showed1680jobs total,
8live workers,1672pending,0failures and no warning. PhysicalGPU0/max8 and
memory-based admission remain enforced; independent per-job failures will
be preserved and the queue continues. The active external run is
`${ARTIFACT_ROOT}/runs/oasis_seed3_first10_adaptive_local_affine_fieldfit_v1/execution_20261105_v3/`.

The original/v1.1/v1.2 plans and protocol files remain unchanged, and all v2
attempts remain preserved. Source/config snapshots and launch/gate receipts
bind v3 execution. The [analysis completion ledger](analysis_completion_ledger.md)
still lists pending mechanism analyses, final paired statistics/figures and
the executed review notebook. Queue launch is the current accomplishment;
scientific study completion is not claimed.

## 2026-11-05 — four-GPU scheduler continuation

The final user instruction authorizes GPU0/1 at maximum16 jobs each and GPU2/3
at maximum4 each. A shared pending queue considers each case on any device
whose measured free memory and reserved peak growth permit it. Admission uses
per-GPU tiers16/8/4/2/1 and checks memory before each batch and individual spawn.
Occupied devices are polled every60 seconds without terminal occupancy deferral.

The new runtime layer is separate from the frozen scientific implementation.
All original scientific source hashes, numerical configuration and the original
launch manifest are unchanged. CPU regression suite94/94 passed; serialized
native-volume SVF smokes passed on all four GPUs (peak reserved1494MiB each).
The source-bound receipts/archive and handoff snapshot are external beneath
`execution_20261105_v3/scheduler_revisions/multigpu_v1/`.

At the controller handoff around10:26:55UTC/02:26:55PST,93 jobs were complete
and eight workers were active. Only the previous controller was stopped; all
eight worker processes and existing outputs were retained. The first new
snapshot admitted four jobs on GPU2 and one on GPU3, while GPU1's external
occupancy left insufficient memory. Source hashes remained exact and the
controller reported no warning.

CUDA OOM attempts now receive an immutable failure event, increased reservation
and automatic requeue after60 seconds, backing off up to15 minutes for repeats.
Retries use new attempt directories. Published tree-budget checkpoints retain
their hashes; retries verify geometry and restore exact saved coefficients at
those boundaries. Other failures continue to be recorded without blocking
independent work. Timing comparisons must retain hardware/concurrency and
failed-attempt provenance; these shared-worker times are not exclusive timings.

## 2026-11-05 — completed queue and first results review

The queue finished at16:50:14UTC/08:50:14PST. All1680 jobs and2640 planned
endpoint pointers are complete; no runtime failures, OOM retries or pending
workers remain. Total elapsed time from the v3 launch is7.5303hours.

`review_completed_run.py` independently checks endpoint identity/coverage and
reproduces all2640 foreground RMSE entries in the existing coverage CSV from
their saved metadata. It adds source-separated and balanced descriptive
summaries, paired target comparisons and ten-pair averages across sources.
Generated artifacts remain in `reports/completed_review_v1/` under the external
execution root; original fields and original generated reports are unchanged.

The achieved cap3072 mean foreground vector RMSE is1.7087mm for adaptive-affine
D0 and1.7147mm for adaptive-translation D0. Adaptive-translation zero-start SVF
achieves1.8558mm versus2.0430mm for adaptive-affine SVF. Both adaptive tree types
improve over uniform counterparts for all40 reference fields at this cap.
Those40 fields share ten repeatedly explored subject pairs; they are not40
independent cases. The first results are descriptive, not a capacity proof.

575/840 main D0 solves remain unconverged (including all280 cap3072 rows), and
all1680 main SVF final stages used their fixed iteration budgets.24 LS solver
diagnostics and8 reverse-topology D0 rows are also unresolved, giving607
solver_unresolved records overall. All1696 SVF main/replay records pass the
declared foreground7/8-step sensitivity threshold; this does not certify
whole-box integration accuracy. All32 D1 bounds are below0.9, maximum0.899991989.
Final convergence disposition, local Taylor/transition analyses and review
figures/notebook remain pending; no additional fitting jobs were started.
