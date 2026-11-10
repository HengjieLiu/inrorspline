# P83 technical log

## Approved paper-loss convergence/resource extension

User approved the [convergence specification](convergence_spec.md). A separate
runner preserves the original pilot. Known limits: one exploratory pair, unequal
loss reductions/capacity, reconstructed two-scale details, sampled operational
stopping rather than a global optimum, and NVML sampling lower bounds. New tests
and native smokes gate the timing matrix. LNCC/diffusion is deferred.

Prelaunch:24 tests passed in15.79s (original13, deferred-loss5, convergence6).
Native40+8-step smokes passed for both variants, including population events,
corrected coarse/fine transition, joint unfreezing, frozen populations, full DVF
and warped image, exact serialized replay, and NVML attribution without competing
compute processes. Smoke registration times4.02/8.30s are functional checks only.
Sequential production queue launched on the inspected empty GPU with source
snapshots and input/runtime/backend receipts, execution `execution_20261106_v1`.
The queue checks occupancy and source hashes before every worker. No prior
results, unrelated jobs, submodule source or base runtime were changed.

## 2026-11-06: requested LNCC9/diffusion follow-up

Deferred after prototype tests and both native smokes; no full runs launched.
User initially requested independent unsupervised1-LNCC² registration with diffusion
lambda1. The existing P43 objective fixes9³ windows, zero-padded moments with
divisor729, FP32 variance-product floor, border image warping and unmasked
forward-valid total-voxel diffusion. Both GaussianDIR variants will use this
common objective from identity. Complete-window and valid-edge sampling give
an unbiased full-volume estimator; numerical tests compare values and gradients
against the untouched dense P43 implementation. The exact dense final objective
is recorded separately. Known limitations and frozen sampling/budget choices are
in [the follow-up specification](lncc_diffusion_spec.md). Original pilot artifacts
and `protocol_v1.json` are retained unchanged. The later paper-loss-first decision
supersedes execution of this prototype in the current phase.

## Known limitations and current status

Native preflight and all four independent2,000-step v1 anchors completed.
The released code is not a complete implementation of the full paper model.
The corrected model is explicitly a reconstruction of unspecified two-scale
details. Its NCC/TV reductions and adaptation rules also differ from the control.
The full-box MSE arm uses fixed populations, and neither achieved nonlinear fit
is a certified optimum. Single-case results will be exploratory. No timing
advantage over the historical saved SITReg model can be inferred.

## 2026-11-06: source acquisition and review

- Read the project map, brief, methods, implementation index, current relevant
  registry/results records, decision log, OASIS/fairness standards and workflows.
  Reviewed the P81/P82 target conventions and canonical metric implementations.
- Read all three local GaussianDIR paper versions. Distinguished journal 2025,
  supplement and arXiv v2 2024 despite the local filenames' 2026 prefix.
- Added `https://github.com/MedAPT-group/GaussianDIR` as the submodule
  `external/GaussianDIR`, pinned to
  `b19987b81d1f3d172d882f1375a27d3b2ade307e`. No upstream edits or commit/push.
  `git submodule add` stages its gitlink and `.gitmodules`; unrelated changes
  remain outside this task's index additions.
- Reviewed network, registration, sampling/loading, loss, metric, configuration,
  inference and checkpoint paths. Authored the reusable technical report in
  `docs/implementation/gaussiandir_paper_code_audit.md`.
- Confirmed single-scale isotropic/translation defaults, translation-gradient
  adaptation, `/s^3` effective cap, and paper/code loss-reduction differences.
- Confirmed optional rigid-coordinate and gradient errors, NaN-pruning defect,
  unguarded weight underflow and adaptation-time optimizer-gradient loss.

## Verification

- Configured Docker runtime inspected read-only; Python 3.10.16 and PyTorch
  2.6.0+cu118. PyTorch3D, Hydra, OmegaConf, W&B and SimpleITK absent; no install.
- Ran `scripts/experiments/gaussiandir/audit_upstream.py` in that container with
  CUDA hidden. All 11 synthetic characterization assertions passed. They
  confirm documented behavior, including defects, not implementation correctness.
- Synthetic harness executes selected upstream AST definitions without importing
  the CUDA-only constructor; exact tiny `torch.cdist` lookup replaces PyTorch3D.
  No native KNN/kernel, patient image, model-fitting or runtime benchmark claim.
- External receipt:
  `${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_v1/audit_20261106/synthetic_audit.json`.
- Read P82's existing manifest and rehashed original rank-0 SITReg and canonical
  mm fields. Both match recorded hashes; pair is fixed0333/moving0115,
  shape160×224×192, source lambda1. Input data were not moved or modified.
- Shared image/label metric replay, native imports, GPU smoke and production
  runs remain pending. Detailed discrepancy values are in the technical report.
- Final source checks: unstaged/staged whitespace checks passed; Python AST and
  local Markdown links passed; new source docs contain no private host paths.
  The tracked-notebook checker reports no Git-tracked cell outputs. The upstream
  submodule worktree is clean. No whole-repository test suite was needed for
  this documentation and isolated audit change.

## 2026-11-06: selected implementation and native preflight

- User selected both variants, coarse1,000/fine1,000 frozen-coarse residual
  stages with literal total caps domain/8 then domain/4, and full-box physical-mm
  MSE with fixed populations. The frozen settings are in `protocol_v1.json`.
- Implemented upstream-class control with equivalent stable density arithmetic;
  corrected anisotropic local-rigid model with center gradients; explicit
  two-scale transitions; complete model/Adam/RNG/sampler checkpoints; common
  evaluator; immutable input checks; source snapshots and external-only outputs.
- Isolated torch2.4.1+cu118/PyTorch3D0.7.8 environment prepared under ARTIFACT_ROOT.
  Native CUDA KNN passed; base torch2.6.0+cu118 unchanged. See `runtime.md` for
  image/package/archive pins, invocation and external installation receipts.
- Thirteen mathematical/native/continuation tests passed, including independent
  covariance/local-motion equations, all parameter gradients, native KNN parity,
  released forward/backward parity, coordinate conversions, Adam clone/prune
  state, exact zero-residual transition and checkpoint continuation.
- Rehashed raw images/labels, original and canonical fields, and the historical
  metric report. Original-to-canonical mm discrepancy <=9.54e-7. Identity and
  SITReg were evaluated with all shared accuracy/regularity metrics.
- Historical SITReg Dice/HD95/ASD and full/foreground fold rates replay exactly;
  SDlogJ differences <8e-11. Native versus canonical interpolation has image
  range-normalized RMSE <2.8e-7 and four nearest-label tie differences out of
  6,881,280 voxels. Canonically exported SITReg labels agree exactly with direct
  source labels. Receipts: external `preflight_v1/preflight.json`.
- Four native-volume smokes passed (12 control steps or12+12 paper steps),
  with finite full-box exports, all35 eligible labels and zero checkpoint-replay
  discrepancy. Both paper transitions preserve the field exactly. Control
  cloning/pruning and corrected fine-stage pruning actually executed. Numerical
  tests separately force corrected cloning and verify optimizer-state retention.
- Smokes are resource/function checks, not scientific endpoints. Results are
  under `${ARTIFACT_ROOT}/runs/oasis_rank000_gaussiandir_v1/smoke_v1/`.
  Reference/preflight outputs are under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_v1/preflight_v1/`.
- Subsequent source changes only added diagnostics/cap metadata, wired the
  already selected threshold values through configuration, and handled empty
  sampler-state restoration. The13 tests passed again. Final runner smoke passed
  and reproduced the earlier corrected-registration field checksum exactly.
  Gated sequential launch began. GPU0 is selected after occupancy
  inspection; unrelated jobs on other GPUs remain untouched.

## Pilot launch

Execution key `pilot_20261106_v1` under the external run root. Launch the two
supervised endpoints first, then two independent image-registration endpoints.
The queue checks reference and all smoke receipts,8GiB free GPU memory, and
source immutability before each child. Every endpoint snapshots exact source
bytes and records config/input hashes, image/runtime, backend settings, command,
loss/gradient/scale diagnostics, actual populations, timer and memory data.

## 2026-11-06: completed pilot

Queue completed20:35:53UTC. All four endpoints reached2,000/2,000 steps, with no
timeout, failure or OOM. Checkpoint replay errors are zero; both corrected
coarse/fine transitions preserve the field exactly. All35 labels are eligible
for identity, SITReg and all candidates. Requested/effective cuDNN benchmarkFalse,
deterministicTrue and both TF32 flagsFalse were retained.

| Endpoint | Final nodes | Stored parameters | Active seconds | Full-box vector RMSE to SITReg mm |
| --- | ---: | ---: | ---: | ---: |
| Released fit |11,799|82,593|22.56|0.372129|
| Corrected fit |112,614|1,914,438|62.53|0.240210|
| Released registration |4,203|29,421|25.20|4.962127|
| Corrected registration |30,749|522,733|55.25|4.275409|

These timings exclude final render/export/replay and evaluation, each separately
recorded. Exact per-endpoint end-to-end worker wall time was not instrumented.
Other host jobs remained active; no SITReg speed comparison is justified.

Fitting foreground RMSE is0.655807/0.420567mm, but Dice0.763342/0.780445 remains
below SITReg0.798719 and foreground folding0.106958%/0.151887% is higher than
SITReg0.000696944%. The corrected fit uses substantially more parameters.
Unregularized displacement MSE does not enforce overlap or Jacobian preservation.

Image-registration Dice is0.749593/0.767602, HD952.572788/2.201555mm,
ASD0.619833/0.554596mm, foreground folding4.092272%/1.745195%, and foreground
SDlogJ0.747576/0.572438. Corrected registration improves these foreground/accuracy
metrics over the control, but both remain worse than SITReg accuracy/folding.
The corrected field has higher whole-box NDV fraction2.32064% versus1.80532%
and voxel BE1.07142 versus0.86816, despite better foreground NDV/BE. Retain both
foreground and full-box views of extrapolation; do not claim uniform dominance.

Full results are external under
`${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_v1/pilot_20261106_v1/review_v2/`:
`report.md`, `all_metrics.csv`, `per_label.csv`, `comparison.json`, source/hash
receipt, convergence/accuracy–regularity/field-error/Jacobian figures and the
executed review notebook. Reviewv2 clarifies metric applicability and adds
interpretation of full-box versus foreground behavior; underlying run/metric
files are unchanged. Posthoc report source is separately frozen in that review;
the four training source snapshots are retained unchanged.

Final review verification:6 comparison rows with202 scalar columns,210 per-label
rows, and3 inline figures. The external notebook executed all3 code cells with
zero error outputs; the repository source remains output-free. Its first render
attempt exposed a missing pandas dependency in the default notebook kernel;
the source now renders tables using standard-library CSV/HTML and IPython,
without changing the runtime or experimental data. Plot layout was inspected.
Python AST, local Markdown links and whitespace checks passed; the tracked
notebook checker reports no cell outputs, and the upstream submodule is clean.

To repeat the audit, use a fresh output filename (existing receipts are never
overwritten):

```bash
docker exec -e CUDA_VISIBLE_DEVICES='' py310sitreg_v1_artifact_store \
  python -B /workspace/repo/scripts/experiments/gaussiandir/audit_upstream.py \
  --output "$ARTIFACT_ROOT/repo_experiments/oasis_rank000_gaussiandir_v1/audit_repeat/synthetic_audit.json"
```

The configured container provides `ARTIFACT_ROOT`, and the script checks the
output boundary.

## 2026-11-06: paper-loss convergence and resource follow-up

The approved convergence protocol completed as execution
`execution_20261106_v1` under artifact key
`oasis_rank000_gaussiandir_convergence_v1`. It used the sampled signed NCC + TV
paper loss at journal lambda8, the released control and corrected reconstruction,
fresh200/500/2000-step schedules with three same-seed timing repeats, one fresh
process run per variant, and one charged convergence trajectory per variant.

- Both trajectories stopped at the1200-second ceiling with
  `convergence_unresolved_at_time_limit`, four LR reductions, and zero operation
  overshoot. The control completed111,500 total updates and selected step110,300;
  the corrected reconstruction completed31,500 and selected step31,400. The
  selected Dice gains over the fresh2,000-step anchors are+0.017527/+0.018009.
  Probe-field change and both fixed monitoring banks did not jointly satisfy the
  plateau rule, so no convergence claim is made.
- Three-run mean usable-output latencies (sample SD) in seconds for200/500/2000
  steps are4.206±1.318 /7.220±1.035 /24.049±0.618 for control and
  10.519±1.418 /17.647±1.488 /53.841±1.342 for corrected reconstruction.
  Fresh-process latencies are33.478s and64.342s respectively. The selected
  long-trajectory outputs were generated at1199.177s and1198.516s including
  required monitoring, with no SITReg inference timing claim.
- Convergence Torch optimization peaks are217.4/272.0MiB alloc/reserved
  (control) and260.7/292.0MiB (corrected); output peaks are144.4/272.0MiB and
  184.1/292.0MiB. Sampled process peaks are786MiB and804MiB. Timing runs were
  sequential on an otherwise idle RTX6000 Ada; benchmark=False,
  deterministic=True, and TF32=False.
- Posthoc evaluation finished all312 distinct checkpoints/exports with14 exact
  field-hash aliases and no scoring failures. The report has328 aggregated
  metrics rows,11,480 per-label rows,2,624 energy rows, and a descriptive
  accuracy-speed frontier. The external notebook executed with zero errors and
  four inline plots. Resource attribution documents that the container PID
  namespace prevents per-process NVML peaks for concurrent supplemental scorers;
  registration NVML attribution and per-evaluation Torch peaks are intact.
- Report finalization exposed a malformed generated PID-map suffix and three
  report-builder recovery/aggregation issues. They were corrected in the source;
  the external sidecar was canonicalized and the final report was regenerated
  from completed evaluations. The finalizer now supports resumed report output
  and records failed stages explicitly.

Review deliverables:
`${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_convergence_v1/execution_20261106_v1/report_v1/`.
The LNCC²/diffusion prototype remains deferred; no dense-gradient benchmark,
dataset expansion or lambda sweep was launched.
