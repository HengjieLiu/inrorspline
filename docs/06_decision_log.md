---
doc_type: decision_log
status: active
---

# Decision Log

## 2026-11-06: GaussianDIR Paper Loss First, Explicit Convergence and Latency

The user approved sampled signed NCC/TV lambda8 before standardized LNCC²/diffusion.
Retain2000-step anchors, three same-seed timing repeats of200/500/2000 schedules,
one startup run and one1200-second continuation per variant. Continuation freezes
populations, jointly unfreezes both corrected scales, and uses two fixed loss banks
plus physical field-change checks for operational stopping. Unresolved time limits
remain explicit. Separate optimization, monitoring, usable-output, setup and
analysis/storage clocks; record phase Torch and sampled process GPU memory.
The deferred loss prototype and all earlier results remain preserved. See
[the execution specification](../experiments/2026-11-06_oasis_rank000_gaussiandir/convergence_spec.md).

## 2026-11-06: Separate GaussianDIR Release Control From Paper Reconstruction

Use both an unchanged-upstream isotropic/translation control with an explicitly
documented stable-arithmetic adapter, and a local corrected anisotropic/rigid
paper reconstruction. Do not call released defaults the complete paper model.
The user selected1,000 coarse steps, frozen coarse plus1,000 fine residual steps,
s8→s4 initialization, and literal TOTAL population caps domain/8→domain/4.
The two-scale transition is an explicit reconstruction of unspecified details.

Supervised fitting uses full-box physical-mm vector MSE, no regularizer and
fixed populations with learnable geometry/motion. NCC-calibrated clone/prune
thresholds are not transferred to MSE. Image registration independently starts
from identity, uses images only and adaptive density, and is evaluated against
the same original saved SITReg field with a shared full-box metric pipeline.
Variant-specific loss reductions and adaptation rules remain visible; no isolated
representation-effect claim follows from their comparison. Exact configuration
and provenance are in [P83](../experiments/2026-11-06_oasis_rank000_gaussiandir/README.md).

## 2026-06-25: Promote `inrorspline` To Real Repo Root

Decision:

- `${REPO_ROOT}` is the real Git repo root.
- `starting_kit/` remains as source material for now.
- Root scaffold is created without `pyproject.toml` or a Python package/test scaffold.

Rationale:

- The immediate need is governance, documentation, protocol safety, and reproducible workflow structure before implementation or GPU runs.

## 2026-06-25: Preserve Original IDIR Bending-Energy Scale

Decision:

- Keep `idir_original` training BE on the public original IDIR scale.
- Do not reinterpret raw IDIR alpha values as directly comparable to SINR or
  DirectCP alpha values.
- Treat any normalized IDIR BE implementation as a clearly tagged future method
  variant, not as a replacement for `idir_original`.

Rationale:

- The original IDIR BE implementation averages second-derivative terms and then
  divides the result again by the sampled coordinate batch size.
- With the default IDIR coordinate batch size of 10,000, this downscales the
  reported training BE by 10,000 relative to the pre-divide mean BE.
- Preserving this behavior maintains comparability with original IDIR runs and
  explains why IDIR alpha scales can differ sharply from SINR/DirectCP alpha
  scales.
- Cross-method comparisons should rely on `alpha * training_reg_loss` and common
  posthoc regularity metrics rather than raw alpha labels alone.

## 2026-07-11: Consolidate LungCT Under DIR-Lab

Decision:

- Use `/data/$USER/datasets/lungCT/DIR-Lab` as the sole LungCT dataset root.
- Keep raw `4DCT` and `COPD` data under `raw/`.
- Keep native R231 masks and pTVReg-style 1 mm data as separately named
  preprocessing protocols: `lungmask_R231_native` and `pTVReg_iso1mm`.
- Store migration/checksum records outside the dataset tree under
  `/data/$USER/datasets/manifests/` and do not maintain compatibility symlinks.

Rationale:

- The former top-level raw trees were byte-identical duplicates.
- Named protocol directories make preprocessing assumptions explicit and keep
  raw data separate from derived artifacts.
- External integrity records allow the dataset root to remain clean without
  discarding provenance.

## 2026-08-15: Freeze v013 As Camera-Ready Figure 2

Decision:

- Use `experiments/paper_artifacts/pareto_figure/v013_final_dirlab4dct_axes_labels/`
  as the final camera-ready Figure 2 artifact family.
- The camera-ready files are
  `pareto_four_panel_final_dirlab4dct_no_annotations.png` and
  `pareto_four_panel_final_dirlab4dct_no_annotations.pdf`; the annotated PNG is
  review-only.
- The four-panel figure uses `Foreground SDlogJ` consistently across OASIS and
  DIR-LAB panels. The DIR-LAB panels are ordered as all methods then selected
  methods, with shared bottom-row axis limits and ticks.

Rationale:

- The final source manifest validates that OASIS rows are unchanged from the
  paper-ready `v003_candidate_oasis100` source table and DIR-LAB rows match the
  `v012_current_tre_stdlogj_fdbe_cps4_alpha_landscape` source table.
- No masked-loss method IDs or masked-loss source paths enter the final source
  rows.
- The final styling keeps pTVReg-loss methods marked with a dagger and defines
  `† pTVReg loss` as a bottom legend entry.

## 2026-09-16: Retain Canonical Off-Grid Submission PDFs In Git

Decision:

- Retain exactly two submitted-paper records under
  `docs/offgrid_submission/submitted_files/`: the initial anonymized submission
  and the named camera-ready submission.
- Treat them as a narrow archival override to the external-only generated-PDF
  policy, with checksums and replacement rules maintained next to the files.
- Exclude the entire archive directory from anonymous-review and
  anonymous-release bundles because the camera-ready record identifies authors.

Rationale:

- The two PDFs are canonical historical submission records, not reproducible
  experiment output or a manuscript-build destination.
- Tracking their exact bytes and documented roles preserves paper provenance
  while keeping ordinary generated PDFs in ARTIFACT_ROOT.

## 2026-09-29: Default GPU Registration To cuDNN Benchmark-Off Deterministic

Decision:

- Future GPU registration experiments default to
  `torch.backends.cudnn.benchmark=False` and
  `torch.backends.cudnn.deterministic=True` when not otherwise specified.
- Record requested and effective backend flags with library/device provenance
  for every timed run.
- An alternative backend is allowed only as an explicit protocol override in
  the execution specification, registry, command/manifest, and report.
- Preserve historical backend behavior explicitly. In particular, P34/P35
  remain `benchmark=True, deterministic=False` records rather than inheriting
  the new default on reproduction.

Rationale:

- The controlled P36 fixed-50 rerun reduced P34 warm time from
  `48.690 +/- 30.222` to `3.958 +/- 0.449` seconds for 4DCT and from
  `36.228 +/- 37.620` to `4.689 +/- 0.577` seconds for COPD, with identical
  optimizer work.
- New F/T warm means were only 1.08x/1.09x the independent P35 T/F
  immediate-repeat floors, showing that shape-specific cuDNN autotuning
  dominated P34's slow process-warm cases.
- Case-01 fixed-50 cold time also fell from 57.701 to 4.670 seconds for 4DCT
  and from 101.004 to 5.615 seconds for COPD.
- Fixed-50 native-crop snap TRE changed by only -0.0022 mm for 4DCT and
  -0.0120 mm for COPD versus P34; no material aggregate quality regression was
  observed. `deterministic=True` still does not claim global bitwise
  determinism outside cuDNN.

## Pending Decisions

- Final public dataset handles that avoid private host paths.
- Whether to add a Python package scaffold, and which tooling to use.
- Which OASIS alpha/LR grid to lock for the first seed-3 run.
