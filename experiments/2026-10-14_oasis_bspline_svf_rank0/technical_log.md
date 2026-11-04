# P73 technical log

## Limitations

Single exploratory rank0, finite-step integration, shared-load runtime, and
non-bitwise CUDA backward. Velocity and displacement lambda1 are not equivalent
constraints; refined and composed fields have different representations.

## Preimplementation audit

Read Qiu et al. MIDL2021, VoxelMorph torch VecInt/networks, local TransMorph
B-spline and probabilistic integration. The B-spline TransMorph/MIDIR sampler
mixes endpoint identity coordinates with align_corners=False. CPU zero-flow
constant-image error was0.875 at boundaries; the endpoint/border sampler gave0.
Constant x-translation integrated with the endpoint/border sampler remained1;
the native TransMorph sampler ranged0.0175057–1. The local probabilistic
TransMorph implementation squares its predicted small-step field without an
explicit initial division; its output convention must not be copied as a
unit-time SVF. The new runner explicitly divides by128.

Implementation preflight: FP32 normalized/voxel paths differ by at most7.80e-9
in displacement and1.19e-4 in isolated gradient entries (gradient RMS2.73e-6).
Different coordinate arithmetic can straddle trilinear derivative knots. The
strict recurrence/gradient parity test therefore uses FP64 consistently
constructed identity grids; FP32 objective gradients are independently checked
against finite differences. This is not a change to production FP32 geometry.

CPU normalized/voxel recurrence parity: forward max5.64e-17, gradient
max2.44e-14, derivative-at-zero error1.24e-14. No files or registrations were
changed during that audit. The approved implementation follows the current
endpoint-aligned geometry and keeps historical sources immutable.

## Production preflight

Nine CPU test groups passed. Eight full-size one-update-per-stage smokes and
eight independent replay audits passed, including exact exported fields,
selected losses, per-label canonical metrics, inverse diagnostics and composed
100k-point off-grid checks. Peak reserved5.1133GiB, leaving admission at8GiB.
Production begins with refine/velocity onGPU0, refine/displacement onGPU1,
compose/velocity onGPU2, compose/displacement onGPU3. Each runs842 then a
fresh8421 registration. Current GPU load is recorded separately at each launch.

## Report-only recovery

All eight production fits and independent replay audits completed without
numerical failures or OOM. Initial Markdown aggregation failed on historical
DL runtime strings `n/a`. A separate aggregator normalizes unavailable timings
to null and retains available P43 pipeline time. It preserves partial report
files under failed_aggregation_v1 and leaves all executed numerical sources,
launch manifests, histories, metrics and checkpoints unchanged. No fit reruns.

## Acceptance and results

Accepted2026-10-15 06:59:57UTC. All8 production fields,8 independent replay
audits,28 selected stage states and24 integration evaluations are complete.
No OOM, optimization failure or numerical-audit failure occurred. The final
six report checksums, source/input/reference hashes,24 comparison rows,
64 paired differences and all case/direction records passed validation.
Final standard-library unittest discovery passed10 tests (nine numerical/CLI
groups plus the report-runtime regression). Pytest is not installed in this
container; no dependency installation was needed. Scoped git diff check passed.
The artifact report is `repo_experiments/oasis_seed3_rank000_bspline_svf_842_8421_ncc9_lambda1_v1/report.md`
under ARTIFACT_ROOT. CSV/JSON retain all historical comparisons and diagnostics.

| Schedule | Formulation / regularization | Dice | HD95 mm | ASD mm | FG SDlogJ | FG folding % | Optimizer s | Worker s | Updates by stage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 842 | Refine / velocity | 0.798388 | 1.973455 | 0.427857 | 0.598585 | 0.000093 | 32.97 | 84.05 | 125/125/200 |
| 842 | Refine / displacement | 0.806226 | 1.915795 | 0.402086 | 0.813166 | 0.205645 | 128.43 | 163.51 | 150/300/1000 |
| 842 | Compose / velocity | 0.794187 | 2.429919 | 0.485614 | 0.722930 | 0.010268 | 35.45 | 82.91 | 125/125/225 |
| 842 | Compose / displacement | 0.804230 | 1.959683 | 0.406399 | 0.848904 | 0.149007 | 88.94 | 147.05 | 150/250/525 |
| 8421 | Refine / velocity | 0.804515 | 1.990093 | 0.417839 | 0.606455 | 0.001394 | 89.50 | 141.74 | 125/125/200/500 |
| 8421 | Refine / displacement | 0.810485 | 1.883917 | 0.392670 | 0.863943 | 1.406991 | 431.64 | 646.72 | 150/325/1050/2000 |
| 8421 | Compose / velocity | 0.797099 | 2.408488 | 0.474442 | 0.798148 | 0.044233 | 103.41 | 158.96 | 125/150/225/500 |
| 8421 | Compose / displacement | 0.810081 | 1.901635 | 0.388712 | 0.910278 | 0.913555 | 229.13 | 269.51 | 150/225/550/1100 |

VFA reference:0.796820/2.048563/0.440015, FG SDlogJ0.661324,
folding0.255035%. SITReg:0.798719/2.061319/0.423064, SDlogJ0.625268,
folding0.000697%. Refined velocity8421 is better on all three accuracy metrics
than both references, with lower SDlogJ but roughly twice SITReg's very small
folding percentage. This is not evidence of a first10 or population benefit.

Production peak reserved memory ranges3.174–5.114GiB. Worker time includes
serialization, canonical metrics and topology/integration diagnostics, but not
the separate independent replay process. Existing jobs remained untouched.

### Topology and integration sensitivity

- All28 saved scaled initial maps had positive determinants at tested cell
  corners.27/28 passed the sufficient derivative infinity-norm bound below1.
  The refined-displacement8421 cps1 map had bound1.120489 and minimum corner
  determinant0.407192. Failure of this sufficient bound is not itself proof
  of folding; it is recorded rather than silently rejected.
- Reintegrating at10 versus7 steps changes mean displacement magnitude by
  0.001986–0.005063mm; maximum pointwise changes range0.124711–1.332003mm.
  It does not eliminate folding. Refined-displacement8421 folding changes
  1.406991%-to1.354628%, and refined-velocity8421 changes
  0.001394%-to0.001347%. Seven-step outputs remain primary.
- Seven-step inverse-after-forward mean residuals range0.077327–0.477804mm
  on100000 fixed-raw-positive voxel centers. Refined-velocity8421 is0.097292mm
  (10 steps0.091084mm). These are numerical inverse-consistency residuals,
  not proof of an exact inverse.
- On identical100000 off-grid foreground points, retained/exported
  nonpositive-J percentages are0/0.058 for composed-velocity842,
  0/0.121 for composed-velocity8421,0.184/0.440 for composed-displacement842,
  and0.935/1.438 for composed-displacement8421. Sampled export and retained
  composition differ; neither replaces the canonical foreground folding metric.
- Displacement regularization admits considerably larger velocities: final
  refined8421 velocity diffusion2.698920 versus0.050392 in the velocity arm,
  although displacement diffusion is0.048063 versus0.045020. Equal lambda is
  not an equal-strength constraint. No causal guarantee follows from this
  single-case observation.

Independent fresh842 and8421 prefixes can stop at different updates because
CUDA grid-sampling backward is not bitwise deterministic despite deterministic
cuDNN settings. No prefix matching or reuse was imposed in this fresh-run pilot.
