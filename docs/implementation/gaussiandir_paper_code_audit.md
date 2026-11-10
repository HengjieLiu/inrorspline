---
doc_type: implementation_audit
status: reviewed_source_and_native_adapter_validated
review_date: 2026-11-06
upstream_commit: b19987b81d1f3d172d882f1375a27d3b2ade307e
---

# GaussianDIR: paper, implementation, and OASIS experiment reference

This is the durable technical reference for GaussianDIR. It distinguishes the
published method, the released implementation, verified implementation issues,
and selected local adaptations. Read this before revisiting the PDFs; revisit the
primary sources when changing a disputed convention or updating the submodule.

**Finding:** the released default is a single-scale, adaptive, isotropic Gaussian
translation model. It is not the paper's full anisotropic, locally rigid,
two-scale model. The optional rigid branch also has forward/gradient discrepancies.
Neither variant guarantees diffeomorphism. Section9 documents the subsequently
validated adapter; pilot results and limitations are tracked in P83 separately
from the paper's reported results.

Experiment: [P83 rank-0 fitting and registration](../../experiments/2026-11-06_oasis_rank000_gaussiandir/README.md).

## 1. Source identity and revision precedence

The three user-provided PDFs are under `docs/literature/GaussianDIR/`. Their
filename prefix says 2026, but their actual publication versions are:

| Source | Bibliographic identity | Use |
| --- | --- | --- |
| `2026_PhiRO_JiheLi_Gaussian primitives for deformable image registration.pdf` | Li et al., *Physics and Imaging in Radiation Oncology* 35 (2025), 100821; six pages; [DOI](https://doi.org/10.1016/j.phro.2025.100821) | Primary published method/results |
| Same stem with `_supplement.pdf` | Five-page journal supplement, July 2025 PDF metadata | Adaptive density, multiscale, losses, detailed results/ablations |
| Same stem with `_arxiv.pdf` | Ten-page [arXiv:2406.03394v2](https://arxiv.org/abs/2406.03394v2), 16 October 2024 | Earlier revision; retain differences explicitly |

SHA-256, in the above order:

```text
d22e60327ff12b9b3be672059aa9e4220b7a3e03e2773d88b9eebad0f3b3d274
c01dd06d0e4d3ded0de9335423fc533abebedf1157ef8cac2eaab00263f475e6
478fb3619e522a1d246685802a919a6576ba41c4b51a5c38e937e1259aaea1f9
```

Code: [MedAPT-group/GaussianDIR](https://github.com/MedAPT-group/GaussianDIR),
tracked at `external/GaussianDIR`, commit
`b19987b81d1f3d172d882f1375a27d3b2ade307e` (the inspected repository has one
commit). All code line references below refer to that commit, not a moving branch.
The upstream worktree remains unmodified. Local literature PDFs were already
present and untracked; this task does not stage or relocate them.

## 2. Scientific model in the paper

GaussianDIR optimizes a deformation independently for each image pair. There is
no pretrained neural network. It transfers the idea of movable anisotropic
Gaussian primitives from 3D Gaussian splatting to a deformation representation;
it does not use image-rendering opacity or depth-ordered alpha compositing.

Let `x` be a normalized fixed-image coordinate. Each primitive has center
`mu_i`, positive scales `s_i`, covariance quaternion `q_i`, local-motion
quaternion `r_i`, and translation `t_i`. With the corresponding rotations
`Q_i` and `R_i`:

```text
Sigma_i = Q_i diag(s_i^2) Q_i^T
a_i(x) = exp(-0.5 (x-mu_i)^T Sigma_i^-1 (x-mu_i)) / sqrt(det(Sigma_i))
w_i(x) = a_i(x) / sum_{j in N_K(x)} a_j(x)
d_i(x) = (R_i - I)(x-mu_i) + t_i
u(x) = sum_{i in N_K(x)} w_i(x) d_i(x)
phi(x) = x + u(x)
I_warped(x) = I_moving(phi(x))
```

The Gaussian constant `(2*pi)^(-3/2)` cancels on normalization. Neighbors are
chosen by Euclidean distance between query and center, not Mahalanobis distance.
The determinant factor matters when widths differ: this is normalized Gaussian
density blending, not simply softmax of squared distance at a common bandwidth.
Reference: journal §2.2.1–2.2.2, equations 2–6; arXiv §III.B–C.

Although the paper's verbal mapping description is ambiguous, its image-warp
equation and the code define a **fixed-to-moving pullback**: evaluate on the
fixed lattice, sample the moving image at `x+u(x)`. Do not invert the saved SITReg
field on the basis of the phrase “moving to fixed registration.”

The local maps are rigid in normalized cube coordinates. On a non-cubic image,
conjugating a normalized-coordinate rotation by unequal axis scale factors does
not generally give a physical-mm rigid motion. Reparameterizing rotations in mm
would therefore be a distinct variant, even with isotropic voxel spacing.

### Published optimization

- Identity local motion, grid-initialized centers in `[-1,1]^3`.
- Adam, 2,000 iterations, learning-rate warmup followed by cosine decay.
- 20,000 sampled query locations per iteration; center/neighbor groups permit
  local finite differences. The arXiv text explicitly divides the query budget
  by `D+1` to obtain the number of centers.
- Signed negative NCC, with means and variances over the mini-batch; this is
  neither squared global NCC nor sliding-window NCC9.
- Neighbor TV penalizes Euclidean displacement differences, not squared
  diffusion. Journal supplement S2 writes a sum over directions and an average
  over centers. Units and normalization must be recorded alongside lambda.
- Journal §2.3.3 specifies lambda **8**; arXiv §IV.C specifies **15**.
- Supplement A.1: clone for position-gradient norm above `0.002`, prune below
  `1e-7`, every `0.05*M` steps. Clones receive Gaussian position offsets;
  other parameters are copied.
- Supplement A.2: coarse plus fine displacement, `u=u_coarse+u_fine`, without
  an image pyramid. It describes a transition at half the iterations and caps
  `|Omega|/8` then `|Omega|/4`. It does not fully specify coarse-field freezing,
  fine initialization, optimizer-state transfer, or every scale's width/LR.
  These unresolved details prevent claiming an exact reconstruction from prose.

The additive field is a direct DVF. Neither averaging rigid maps nor adding
coarse and fine displacements ensures an invertible map. KNN support also
changes discretely: within a fixed neighbor set the field is smooth, but global
continuity is not guaranteed at neighbor-set boundaries. Analytic derivatives
holding KNN membership fixed can miss such changes.

### Published evidence and its limits

Journal Table 1 reports OASIS DSC `81.3 ± 2.3%`, HD95 `1.89 ± 0.50 mm`, and
NJD `1.091 ± 0.24%` on **49 pairs from the last 50 Neurite OASIS subjects**.
That nonzero folding result is relevant to our accuracy–regularity question.
This is not our seed-3 shuffled `pair_test_200` split or a SITReg comparison.

The arXiv additionally reports 380 OASIS pairs from 20 subjects: DSC
`81.6 ± 2.1%`, HD95 `1.86 ± 0.39`, NJD `1.106 ± 0.22%`. Do not mix this
cohort with the journal's 49-pair table. Journal ACDC results are DSC
`81.2 ± 5.3%`, HD95 `3.43 ± 1.26 mm`, NJD `0.347 ± 0.32%`.

Supplement Table S2 gives fixed/isotropic/translation DSC `79.6%`, free
isotropic/translation `81.0%`, free anisotropic/translation `81.2%`, and free
anisotropic/rigid `81.3%`; random and grid initialization have similar reported
means. These are the authors' ablations, not independently verified effects.
The free-center change is larger than the last two additions in that table.

The journal advertises DIR-Lab TRE `1.00 ± 1.11 mm` at about 2.5 seconds;
supplement Table S1 reports `0.97 ± 1.07 mm` at 5.1 seconds, with results
averaged over five random seeds. These are different reported operating points,
not interchangeable timing claims. Hardware is RTX 4090. Our LungCT protocol
warnings still apply; no local lung comparison is planned here.

The IXI comparison is GaussianDIR `75.6%` versus OASIS-trained, weakly supervised
TransMorph `69.3%` DSC. It is evidence for that particular transfer comparison,
not a general ranking against all learned registration methods or SITReg.

## 3. Released code map

| File / symbol | Responsibility and important behavior |
| --- | --- |
| `networks.py:11` `GaussianWarp` | Primitive parameters, initialization, KNN weighting, direct field, adaptation |
| `networks.py:45` `reinitialize` | Replaces constructor grid/cap using image shape, mask, sparsification |
| `networks.py:110` `cal_nn_weight` | PyTorch3D KNN, Gaussian density normalization |
| `networks.py:133` `forward` | Translation or optional rigid local displacements |
| `networks.py:166` `densify_and_prune` | Population changes and Adam parameter/state replacement |
| `registrators.py:24` | Dataset/model construction, seed, device, optimizer |
| `registrators.py:102` `ODE` | Just `coords + network(coords)`; despite its name there is no ODE integration |
| `registrators.py:109–187` | TV, training step, adaptation timing, LR schedule, evaluation/checkpoint |
| `datasets.py:122–143` | Shuffled masked centers and their positive-axis neighbors |
| `datasets.py:163–186` | `grid_sample`, `align_corners=False`, default zero padding; nearest labels |
| `loaders.py:80–122` | OASIS NIfTI files, external ROI, `pairs_val.csv`, indices limited to 0–19 |
| `losses.py:9–45` | Signed global NCC and custom standard-deviation backward |
| `metrics/` | Dice, HD95, folding, landmark TRE; not our complete metric set |
| `train.py`, `inference.py` | Hydra CLI; override `save_dir` with a relative `outputs/...` path |

### Effective defaults, not just YAML names

| Quantity | Effective released default |
| --- | --- |
| Primitive model | Free centers, one learned scalar radius, translation; 7 stored trainable scalars/node |
| Shape / local motion toggles | `anisotropy.scaling=false`, `anisotropy.quaternion=false`, `with_quaternion=false` |
| Full optional model | 17 stored scalars/node: 3 center + 3 scale + 4 covariance quaternion + 4 motion quaternion + 3 translation; 15 intrinsic DoF because of quaternion scale redundancy |
| Neighbors | `K=20` |
| Initialization | Cell-centered grid; nominal `[10,10,10]` is replaced by `int(image_shape/8-1)` and masked |
| Width | Half the diagonal between neighboring grid nodes, in normalized coordinates; log-radius/log-scales optimized |
| Population cap | Nominal `5000` is overwritten by `floor(mask_voxels / 8^3)` |
| Adaptation | Enabled; accumulated translation-gradient norm; events at steps 300,400,…,1200; no events after 60% of run |
| Thresholds | Clone `>=0.002`; prune `<=1e-7`; average gradient norms since previous event (first window is longer) |
| Query batch | Usually 5,000 centers plus 15,000 positive-axis neighbors; final shuffled batch may be smaller |
| Image loss | NCC on **all** center and neighbor query values |
| TV | Mean over centers **and directions**, on normalized-coordinate displacement differences |
| TV weight | `8` |
| Steps / warmup | `2000 / 50`; cosine schedule denominator `max_steps+1-warmup_steps` |
| Parameter-group LR | centers `.01`; translation `.002`; log-radius/log-scales `.02`; both quaternion groups `.001` |
| Optimizer | Adam; YAML base LR `.0001` is superseded by explicit parameter-group LRs; other Adam settings are library defaults |
| RNG | `torch.manual_seed(6)`; split shuffle seed 3 is a separate concept |
| Inference chunks / validation | 50,000 queries / every 500 steps |
| Integration / stages | Direct displacement; one model; no coarse/fine stage switch |

For our `160×224×192` box, sparsification 8 gives an unmasked `19×27×23`
initial grid (11,799 nodes) and a full-box cap of 13,440. A foreground mask
reduces both counts, which must be measured. `node_shape` and `max_densify_num`
CLI overrides alone do not control the effective counts after `reinitialize`.
The cap is not the supplement's written `|Omega|/8`; interpreting that formula
literally as a voxel count is 64 times the code's `|Omega|/512`.

KNN is recomputed for each query batch. Clones are offset with axis-aligned
`normal(0,s)` samples; the covariance rotation is not applied to clone offsets.
New Adam moments are zero, but the existing optimizer step counter is retained.
Positions/scales are unconstrained except for exponential scale activation.
There is no guard ensuring at least K nodes remain after pruning.

## 4. Paper–code correspondence and defects

| Item | Audit finding | Consequence |
| --- | --- | --- |
| Normalized Gaussian blending | Matches the main equation up to determinant stabilization `+1e-7` | Preserve the determinant term when implementing |
| Full primitive model | Optional in code; disabled by default | A default run is the simpler ablation, not the full published model |
| Multiscale field | Absent from training and model code | Enabling anisotropy/rotation does not recover the complete paper |
| Density statistic | `translation.grad`, not `node_position.grad` (`networks.py:252–255`) | Different adaptation mechanism and threshold scale |
| Density schedule/cap | Stops at 60%, starts at 300, uses one `/s^3` cap | Not the prose two-stage policy |
| TV reduction | Code averages 3 directions; supplement sums them per center | For the same center interpretation, `TV_code=TV_written/3`; lambda 8 is not directly transferable |
| NCC reduction | Population covariance divided by sample standard deviations | Ignoring eps, NCC is `(n-1)/n` times Pearson; 0.99995 factor at n=20,000 |
| Rigid forward coordinates | Anisotropy rotates `local_coords` into covariance axes; `forward` reuses them in motion equation | Optional full branch differs from equation 5 |
| Rigid gradients | `local_coords.detach()` before motion rotation | Omits direct center derivative; also omits covariance-frame derivative when enabled |
| Adaptation before optimizer | Replaces Parameters after backward, before `optimizer.step()` | On actual clone/prune events, current gradients are lost and the step skips replaced parameters |
| NaN pruning | Applies `.isnan()` to an already boolean `nan_nodes` mask | Does not identify scale NaNs through that clause |
| Weight normalization | Denominator is unguarded sum of exponentials | All tails can underflow to zero, yielding NaNs |
| OASIS pipeline | External ROI and 20-row validation loader | Does not implement our pair/domain protocol |
| Output / metrics | Only evaluates mask, fills outside with zero | Artificial boundary jumps and incomplete full-box deformation; unsuitable for shared regularity comparison |

### Rigid branch discrepancy in detail

At `networks.py:121`, the row vector `x-mu` becomes `(x-mu)Q`, equivalent
to `Q^T(x-mu)` in column notation, for the Gaussian exponent. That transformed
vector is returned and reused at line 139. Consequently, with both covariance
orientation and motion rotation enabled, the code blends

```text
(R-I) Q^T (x-mu) + t
```

instead of the paper's `(R-I)(x-mu)+t`. The result should instead preserve the
original spatial delta for the motion and use a separate covariance-frame delta
only for the exponent. Removing the detach is a separate gradient correction.
These changes do not affect translation-only forward outputs, but they change
the full rigid model and its optimization. Any corrected version must have a
distinct method tag and keep the upstream gitlink untouched.

### Synthetic evidence

[Audit source](../../scripts/experiments/gaussiandir/audit_upstream.py) executes
selected, unmodified upstream AST definitions on tiny CPU tensors. It bypasses
the CUDA constructor and substitutes exact `torch.cdist` KNN only for these small
examples. It is not a native PyTorch3D/CUDA integration test.

On 2026-11-06 all **11 characterization checks** passed in the configured Docker
container. Passing means the documented behavior was observed, including defects.
The external receipt is
`${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_v1/audit_20261106/synthetic_audit.json`.

- Constant translation blends with maximum error `2.78e-17`.
- Anisotropic/rigid forward differs from the paper formula by `0.6` in the
  constructed normalized-coordinate example.
- Rigid center gradient: autograd `-0.1643066`, finite difference `0.7757892`.
- Density accumulation ignores nonzero center gradients when translation
  gradients are zero.
- Tiny scales produce nonfinite weights; boolean `isnan` gives false for both
  true and false input mask entries.
- Pruning before Adam leaves all replaced gradients absent and causes exactly
  zero survivor update in the constructed event.
- Perfect 8-sample correlation yields NCC loss `-0.875`, not `-1`.
- Three unit neighbor differences produce TV `1` rather than the written sum `3`.
- A K=1 counterexample shows a finite jump across a neighbor boundary. This
  proves no general continuity guarantee; it is not a measured K=20 failure.
- `align_corners` coordinate/displacement conversion passes to machine precision.

## 5. Geometry and metric interoperability with SITReg

Use the canonical rank-0 pair: fixed `0333`, moving `0115`, first row of
`configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`. Source is the
**original historical SITReg lambda-1 endpoint**, already selected by P81/P82;
do not substitute a later lambda sweep's checkpoint or the unbounded B-spline
field. The P82 manifest was read and both SITReg field hashes were rechecked:

```text
original disp_0333_0115.nii.gz:
caf5be5c136615b5d833a7c1d44edd8d98de9475847836388b5f014dfad2f878
canonical field_mm.npy:
408ce279ca76bddf2bced1b438f2f5d559495fa0684f46390abd06821c9c90c1
```

Manifest locator:
`${ARTIFACT_ROOT}/runs/oasis_rank000_fieldfit_parameter_scaling_v1/execution_20261106T0644Z/manifest.json`,
target `rank000_sitreg`. It resolves the historical paths without embedding
private host locations here. Expected shape is `160×224×192`.

The original field is channel-first ZYX voxel displacement. The canonical
P81 field is channel-last world-RAS mm on the reoriented fixed DHW array. Its
image/segmentation reorientation and affine update are in
`scripts/experiments/oasis_polyaffine_fieldfit/data.py:reorient`.
Do not confuse PyTorch grid XYZ, array DHW, and world RAS/LPS.

For one array axis of size `N`, voxel index `i`, voxel displacement `v`:

```text
g_true(i)  = 2*i/(N-1) - 1            # saved-field adapter convention
g_false(i) = 2*(i+0.5)/N - 1          # GaussianDIR convention
g_false   = g_true*(N-1)/N
u_false   = 2*v/N
u_true    = 2*v/(N-1) = u_false*N/(N-1)
```

Apply this in XYZ order, using dimensions `[W,H,D]`. For mm targets use
the fixed/moving affine matrices to convert physical endpoints, not an assumed
spacing or a channel permutation alone. With different grids, transform the
fixed physical point plus mm displacement into the moving index grid first.
The current canonical target preparation requires matched atlas geometry.

Use raw fixed `image > 0` for foreground summaries and any foreground sampling.
The upstream `aligned_ROI` file is not proven equivalent. For full-box output,
evaluate GaussianDIR at **every native center**, including outside a foreground
training mask; never use `masked_gather` zero-fill as the scientific field.

### Shared evaluation contract

Both candidate arms and the saved SITReg reference must pass through the same
metric implementation, domains, interpolation, units and missing-label rules.

| Family | Required outputs |
| --- | --- |
| Label accuracy | Per-label and macro Dice, physical-mm HD95, symmetric ASD and both directional ASDs; all 35 canonical labels with explicit absence/empty-warp status and eligible counts |
| Image alignment | Signed global NCC, MSE, MAE, full box and fixed foreground; identify normalization and padding |
| Regularity | Full/foreground folding percentage `J<=0`, positive-J SD(log J) with positive coverage, mean/min/quantiles of J, diffusion, finite-difference bending energy, NDV and NDV fraction |
| Supervised reconstruction | Vector RMSE, mean/p50/p95/max endpoint error in mm; full box, foreground and complement; Jacobian/warped-image discrepancy from target |
| Resources | Actual initial/final/peak node and stored parameter counts, optimizer/evaluation/export/total time, peak allocated/reserved GPU memory, stopping reason |

Reuse `scripts/experiments/oasis_alpha_sweep/common.py:regularity_metrics`,
`jacobian_metrics_voxel`, `diffusion_loss_voxel`, `ndv_metrics_voxel`, and
`compute_be_metric_rows` after canonical field conversion. Existing LUMIR
Jacobian evaluation crops two voxels and uses positive-J population log spread;
foreground diffusion requires both edge endpoints in the mask. NDV uses its
own one-voxel interior convention. Record each metric's own valid support.
For dense-field BE use the same finite-difference metric row for both methods,
not spline-native BE for one and finite differences for the other.

Reuse the detailed label conventions in the SITReg first100 evaluator and P81
metric source after checking their parity with the selected historical reference.
Never replace absent-label distances with zero or silently drop empty warped
structures. Baseline self-replay and identity/translation analytic checks are
required before interpreting differences. Gaussian analytic derivatives, if
added, are supplementary and explicitly conditional on neighbor membership.

Upstream metrics cannot substitute for this contract: Dice uses labels present
in the fixed segmentation and only the fixed ROI; HD95 hardcodes unit spacing;
folding uses `<0` and returns a fraction on the interior of a zero-filled field;
there is no ASD/BE/diffusion/SDlogJ/NDV suite. Its landmark path rounds predicted
coordinates. Its sign-flipped `ODE(forward=False)` is not an inverse map.

## 6. Environment and reproducibility

The initial read-only inventory of the configured container found Python `3.10.16`,
PyTorch `2.6.0+cu118`, nibabel and scipy. PyTorch3D, Hydra, OmegaConf, W&B and
SimpleITK were absent. No installation was done during that initial audit.
The subsequent isolated native runtime and adapter are documented in section9.
The README's suggested Python 3.12 / CUDA 12.1 / PyTorch 2.4.1 PyTorch3D
binary does not match this container. The README references `requirements.txt`,
which is absent from this commit. No root license file is present in the
inspected tree; record upstream provenance when considering redistribution.

The constructor allocates statistics on `cuda` unconditionally, despite the
registrator's apparent CPU fallback. Native execution needs a compatible
PyTorch3D build or a separately validated environment. The synthetic audit
provides no evidence about CUDA kernel parity, runtime or native peak memory.

For eventual runs, keep current historical jobs/environment stable, use the
configured artifact-writing user, pin dependencies/image digest, and record
requested/effective cuDNN `benchmark=False`, `deterministic=True`, plus TF32
policy. Those flags do not prove deterministic PyTorch3D KNN tie handling.
Use synchronized GPU timing; upstream wall-clock accounting excludes periodic
validation and does not explicitly synchronize CUDA.

The upstream checkpoint stores only `state_dict`; it does not save optimizer,
RNG, population statistics, iteration or scheduler state. `resume=true` is not
exact training continuation. A local runner must save these if restart is needed.
Its CLI also overwrites `save_dir`, so an ordinary save-dir override is
insufficient to enforce the external artifact policy. Use a dedicated wrapper
with explicit external paths and W&B disabled.

## 7. Decisions before the two experiments

These are the initial audit recommendations. The selected v1 implementation in
section9 and the linked execution specification supersede unresolved choices.

| Decision | Recommendation | Why / alternative |
| --- | --- | --- |
| Method identity | Start with a labeled released-code translation/isotropic control; add a separately tagged corrected paper model if desired | Full model needs motion-coordinate/gradient fixes and an explicit reconstruction of missing multiscale behavior |
| Supervised target | Original rank-0 SITReg lambda-1 field, immutable hashes above | Matches recent fitting studies; not ground truth |
| Fit objective/domain | Full-native-box vector MSE in physical mm, no image/label/TV loss | Matches P82; report foreground additionally. Foreground-only would instead match P81's objective |
| Fit density policy | First diagnostic: fixed population but learn centers, widths and translations; then a separately labeled adaptive fit | NCC thresholds do not retain meaning after changing to mm-MSE. Fixed population is an ablation, not full adaptive GaussianDIR |
| Registration objective/domain | Released sampled signed NCC + code TV, centers from raw fixed `image>0`; labels only for evaluation | This is an OASIS protocol adaptation; keep center-neighbor loss support explicit |
| TV weight | Anchor 8; exploratory 2/4/8/16/32 only after anchor preflight | Accuracy–regularity tradeoff matters; rank-0 tuning is exploratory, not held-out selection |
| Primitive density / neighbors | Start `s=8`, `K=20`; record actual counts | Secondary s=4 or K=10/40 changes capacity, optimization, memory and support, so vary separately |
| Budget / LR | Registration: 2,000 steps, warmup 50, released group LRs. Fit: report the 2,000-step checkpoint plus a proposed independent 1,200-second cap | MSE scale changes suitable LRs and gradient thresholds; require a short stability/convergence pilot before freezing |
| Seeds | Optimization seed 6 first, optional 0/1/2 sensitivity later | Split seed remains 3; one seed does not characterize adaptation variance |
| Unsupervised initialization | Identity, independently of the supervised fit and SITReg | SITReg initialization would turn it into a learned-baseline refinement |
| Numerical corrections | Local wrapper/adapter, unmodified upstream; enumerate every correction | Stabilized weight computation, finite checks and explicit population guards need parity tests; optimizer-order fixes change behavior |

P81/P82's fixed Gaussian dictionaries, global-affine roots and LSQR are not
GaussianDIR. GaussianDIR learns geometry and uses KNN support, so its achieved
nonlinear fit is not a certified best representational approximation. Count all
stored trainable scalars, not just translation coefficients, for capacity plots.
Single-pair fitting and registration answer different questions: approximation
of an existing map versus image-based estimation. Neither can establish a
population-level accuracy or speed advantage over SITReg.

## 8. Reading and maintenance checklist

When upstream changes, refresh the commit and file hashes, recheck defaults and
the discrepancy table, rerun the synthetic audit, and then perform native
dependency/KNN/gradient/coordinate/metric smoke tests before real runs. Do not
carry this audit's conclusions to another upstream revision without review.
The experiment specification lists remaining implementation and acceptance work.

## 9. Selected P83 reconstruction and implementation map

The user selected both the released control and corrected anisotropic/local-rigid
paper reconstruction. The exact settings are in
[protocol_v1.json](../../experiments/2026-11-06_oasis_rank000_gaussiandir/protocol_v1.json)
and [the execution specification](../../experiments/2026-11-06_oasis_rank000_gaussiandir/codex_execution_spec.md).
The upstream submodule remains pristine. These local additions do not establish
what the authors used for underspecified details.

- **Control:** use the pinned upstream class, isotropic radii/translations, K20,
  s8, upstream NCC custom backward and direction-averaged TV. Log-space weights
  retain the release's determinant epsilon1e-7 and pass ordinary-input
  forward/gradient parity. Translation-gradient density statistics, `/8³` cap,
  steps300…1200 every100, and adaptation-before-Adam gradient discard remain.
- **Paper reconstruction:** use spatial deltas for rigid motion, covariance
  deltas only for Mahalanobis weights, full center gradients, positive scales
  through exponentials and scalar-first normalized quaternion rotations.
  Log-density normalization omits the release-only determinant epsilon. The
  field is directly represented; no integration or invertibility guarantee.
  Local rotations are orthogonal in the release's normalized XYZ coordinates.
  On this rectangular image box, unequal axis scales conjugate these rotations
  when converting to physical mm, so they are not strictly rigid physical-mm
  maps. Covariance and motion parameters retain the release coordinate system.
- **Two scales:** coarse s8 for1,000steps; freeze coarse; add a zero-motion s4
  residual evaluated at original coordinates for1,000steps. Both stages use
  independent50-step warmup/cosine schedules with released starting LRs. No
  composition, target downsampling or image pyramid. The transition must preserve
  every tested field value exactly. User-selected TOTAL caps are domain/8 then
  domain/4; fine capacity subtracts retained coarse count.
- **Paper registration:** signed Pearson population NCC plus direction-summed
  TV at lambda8; position-gradient density statistics; clone/prune thresholds
  .002/1e-7; events100…900 in each stage; optimizer step before adaptation;
  surviving Adam moments retained and cloned moments zero. Clone offsets use
  the supplement's axis-aligned normal. Invalid populations and nonfinite values
  fail explicitly. Thresholds are not recalibrated on the observed result.
- **Both fits:** full-box physical-RAS-mm vector MSE, fixed populations with
  learnable geometry and motion, no regularizer. Both image-registration runs
  independently start from identity, use raw fixed image-positive centers and
  3 positive-axis neighbors, and never optimize using labels or SITReg fields.
- **Budget:** seed6,2,000 total steps,1,200-second active watchdog. Initialization,
  optimization, adaptation, diagnostics and checkpointing are charged; final
  full-box rendering and common evaluation are separately timed. These are
  achieved nonlinear fits, not certified best approximation errors.

| Local source | Responsibility |
| --- | --- |
| [model.py](../../scripts/experiments/gaussiandir/model.py) | Stable released adapter, corrected equations, native KNN, population/Adam mutation, multiscale freeze and serialization |
| [data.py](../../scripts/experiments/gaussiandir/data.py) | Immutable raw/canonical target verification, split/mask/affine checks, explicit corner/voxel/mm conversions, resumable shuffled sampling |
| [run.py](../../scripts/experiments/gaussiandir/run.py) | Independent objectives, schedules, checkpoints, source snapshots, finite checks, full-box export and serialized replay |
| [evaluate.py](../../scripts/experiments/gaussiandir/evaluate.py) | Shared35-label Dice/HD95/ASD, image errors, full/foreground diffusion/BE/NDV/Jacobian metrics and field/Jacobian fidelity |
| [preflight.py](../../scripts/experiments/gaussiandir/preflight.py) | Historical metric replay and independent native/canonical image/label warp checks |
| [pilot.py](../../scripts/experiments/gaussiandir/pilot.py) | Smoke/reference gates, memory admission, source-drift check and sequential four-endpoint launch |
| [report.py](../../scripts/experiments/gaussiandir/report.py) | Complete scalar/per-label exports, loss traces, field/warp/Jacobian figures and external Markdown comparison |
| [tests](../../tests/test_gaussiandir.py) |13 mathematical, native-kernel, coordinate, adaptation and continuation contracts |

The native runtime uses an isolated Python3.10/PyTorch2.4.1+cu118/PyTorch3D0.7.8
environment and official matching binary; the base torch2.6 installation remains
unchanged. Exact archive/image pins and commands are in
[runtime.md](../../experiments/2026-11-06_oasis_rank000_gaussiandir/runtime.md).
All13 tests and four native-volume smokes passed. Historical SITReg macro Dice,
HD95, ASD and full/foreground folding replay exactly; SDlogJ error is<8e-11.
These checks validate the adapter/protocol, not the paper's reported results.

The six final rows share a full-box evaluator, including identity and saved
SITReg. Foreground uses raw fixed `image>0`; no masked zero-filled field export.
There are explicit differences in representation, parameter count, loss
normalization and adaptive optimization between the two GaussianDIR variants.
Their comparison cannot attribute effects to anisotropy, rigid motion or scale
individually. A single repeatedly explored pair cannot establish population
accuracy or speed advantages over SITReg. See the experiment log for external
execution/review paths and endpoint completion status.
## 10. Paper-loss convergence and resource study

The approved follow-up is specified in
[convergence_spec.md](../../experiments/2026-11-06_oasis_rank000_gaussiandir/convergence_spec.md).
`study.py` preserves the original2000-step sampled signed-NCC/TV lambda8 anchor
and adds separately tagged200/500-step schedules and fixed-population refinement.
`GaussianModel.optimizer(joint=True)` explicitly unfreezes both corrected scales;
checkpoint payloads preserve per-parameter trainability. Default optimizer scope
and the original pilot runner remain stage-local.

`convergence.py` defines independent fixed monitoring banks and the loss/probe-only
plateau controller. It uses two banks of16 per-batch losses, not whole-volume
NCC as a replacement objective. Fresh refinement Adam starts at10% native LRs,
with50-step warmup, up to four halvings, and a post-reduction stability window.
Best-monitor selection and terminal state are both retained; time to establish
stopping is charged in full. A20-minute cap without a plateau remains unresolved.

`resources.py` measures exclusive synchronized wall/CUDA phases and resets peak
Torch allocation/reservation at each phase. NVML samples process memory every
100ms, so context/non-Torch allocations are represented but short peaks may be
missed. `study.py` excludes labels/SITReg fields from registration GPU inputs,
generates both full-box DVF and usable warped image, and separates snapshot,
serialization, reference/label metrics and posthoc rendering from latency.
`study_queue.py` checks an empty GPU and freezes sources before sequential runs.

The release's2000-step loop has no automatic early stopping. Its wall timer starts
after model/data setup, excludes periodic validation/final save and lacks explicit
CUDA synchronization. Published DIR-Lab2.5s and supplement5.1s on RTX4090 refer
to distinct operating points, not this OASIS pair or full-pipeline latency.
LNCC9²/diffusion remains a tested, deferred prototype; no dense-gradient benchmark
or standardized-loss production result belongs to this follow-up.
