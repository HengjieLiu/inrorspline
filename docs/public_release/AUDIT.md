# Public release audit

Historical initial audit, 2026-09-29. Subsequent implementation/validation is in
`PROGRESS.md` and `progress.json`; the no-modification statement below describes
the initial audit phase only.

2026-09-30 superseding decision: the user confirms SINR permission is pending.
Any earlier confirmed-permission interpretation below is historical, not current
publication clearance. See `PUBLICATION_READINESS.md`.

Audit sources: the user's attached discussion, explicit
follow-up answers, local source/worktrees, canonical camera-ready PDF, and
external result records. No optimization, preprocessing, Docker build, or
public-repository modification was performed.

## Findings that affect correctness

### A01 — The publication and older scaffold describe different cohorts

The accepted paper evaluates 100 OASIS pairs, ten 4DCT cases, and a total
2,500-step budget for the main method comparison. Older project briefs and
standards prominently describe first10 screening. Do not export these as the
paper protocol. The first100 extension combines ranks 0--19 with 20--99.

Evidence: `docs/offgrid_submission/submitted_files/camera_ready_off-grid_30.pdf`,
`experiments/2026-07-04_oasis_seed3_first100_extension/codex_execution_spec.md`,
and the checked-in shuffled split file. OASIS loader semantics are in
`scripts/experiments/oasis_alpha_sweep/common.py`.

### A02 — Freeze paper Figure 2 separately from acceleration updates

The existing public `figures/fig2.png` is byte-identical to the canonical v013
camera-ready PNG. SHA-256:
`eb0db83bbd1f96d34c3afc9c25576e56c7944c07cc04ce74ece6f615d23e3e16`.

Actual stored source directory:
`${ARTIFACT_ROOT}/repo_experiments/migration_2026-09-02/experiments/paper_artifacts/pareto_figure/v013_final_dirlab4dct_axes_labels/`.
The initially inferred non-migration path does not exist. Inputs there include
`oasis100_pareto_source_rows.csv`, `source_rows.csv`, and `source_manifest.json`.
The manifest confirms OASIS v003 and LungCT v012 source matches, with no
masked-loss rows. Curate public columns and portable provenance rather than
copying private paths or unnecessary fields.

Renderer source:
`experiments/paper_artifacts/pareto_figure/build_final_dirlab4dct_axes_labels.py`.
It imports other paper modules and private artifact routing, so it needs a
portable dependency extraction. Preserve method selection, means/SDs, labels,
axis limits, ellipses, and pTV-loss markers.

Figure 1 is an overview illustration in the paper. The existing public PNG
has SHA-256 `39cd5f9dbf33135ea6985e7e166ba1c360d98f01a9770febf603d83b493afd63`.
Editable source has not been established. Do not promise code generation of it.

### A03 — There are multiple incompatible meanings of "updated table"

1. The camera-ready PDF's Table 1 is authoritative for the published values.
   An older `.tex`/Markdown insert has different timing values; it cannot be
   substituted for the PDF's final table without checking its source.
2. September 16 paired five-method base/fast F/T experiment completed externally
   even though its repository technical log still has unchecked TODOs.
   Its report records 110 timing rows and 100 unique first10 quality outputs.
   MR-D-BSCP warm time is 57.725 to 9.154 s. The final user choice is this
   experiment's **N=10 paired timing**, with **historical OASIS100 accuracy
   separately listed**. This supersedes an earlier first10-accuracy choice.
3. A newer September 24 OASIS100 study also exists. Acceptance passed with 202
   timing rows, 200 new quality rows, and two 100-case historical quality anchors.
   Fixed-50/cache/batch: 1.918 s warm and Dice 0.80904; AMSGrad: 1.987 s and
   Dice 0.80902. The user explicitly did not select this study for the release
   update; keep it outside the main table and selected accelerated code path.

September 16 source:
`${ARTIFACT_ROOT}/repo_experiments/2026-09-16_oasis_table1_base_fast_ft_first10_gpu3/`
(`timing_summary.csv`, `dice_summary.csv`, case rows, execution manifest, report).
The endpoint is loaded images on the selected device to serialized DVF;
device setup occurs outside that interval. Cold has one observation per
candidate; warm has ten. Do not call this whole-process end-to-end timing or
attach a repeated-cold confidence claim.

Historical OASIS100 accuracy source, verified present:
`${ARTIFACT_ROOT}/repo_experiments/migration_2026-09-02/experiments/2026-08-13_oasis_selected_fast_first100/fast_vs_2500_baselines.csv`.
Use its own method/schedule identifiers and source provenance when exporting the
separate accuracy block; do not relabel these as September 16 measurements.

September 24 source:
`${ARTIFACT_ROOT}/repo_experiments/2026-09-24_oasis_cpbe_fixed50_first100/oasis_cpbe_fixed50_first100_20260925T052403Z_v1/`.
Its local technical log records three important caveats: CP-BE failed strict
repeated-Adam equivalence to dense analytic BE; the first10 influenced candidate
selection (report ranks 10--99 sensitivity); historical baseline timing is
first10/GPU3, whereas new timing is first100/GPU0. A simple ratio is not a matched
speedup measurement. These caveats remain necessary even when diagnostic
experiment code is excluded.

### A04 — COPD crop semantics contradict the generic documentation

`scripts/data/preprocessing/lungct/generate_4dct_ptvprep.py:204` selects corrected
MATLAB-origin bounds. `generate_copd_ptvprep.py:80` selects `legacy_bounds`,
with metadata `crop_convention=plan_expected_legacy_python_bounds`.
Both reuse `ptvreg_landmark_crop_1mm_trilinear_v1`. Meanwhile the generic TRE
standard says the canonical 4DCT/COPD handle uses corrected bounds.

Tracked issue: **CROP-001**, linked task **R04**, status **pending fix**.
The user requested retaining this issue in private documentation on 2026-09-29.

Cause: the legacy crop function uses the original one-based landmark values
directly in Python crop bounds. The corrected function computes the MATLAB
bounds and subtracts one from each endpoint. Both use the same landmark union,
margin, and subsequent resampling recipe.

Verified saved metadata under
`${LUNGCT_DATA}/processed/pTVReg_iso1mm/{4DCT,COPD}/caseXX/preprocess_audit.json`:

- All ten 4DCT records report legacy-minus-corrected start/end offsets of +1
  in each axis; the corrected crop is the selected 4DCT input.
- COPD cases 02--10 use legacy bounds with the same +1 offset at both ends
  of every axis, retaining crop size.
- COPD case01 hits the lower z boundary. Corrected versus used legacy bounds,
  expressed as zero-based inclusive native indices, are:

| Axis | Corrected | Used legacy |
| --- | --- | --- |
| z | 0--108 | 0--109 |
| y | 114--434 | 115--435 |
| x | 70--435 | 71--436 |

Thus case01 has 109 versus 110 native z slices, not a pure three-axis
translation. Its native spacing is ZYX `[2.5,0.625,0.625]` mm. The offsets
refer to native voxels before 1 mm resampling. Bounds touching a volume edge
must be tested explicitly rather than assuming crop shape always stays equal.

The landmark transformation separately subtracts one and then the actual crop
origin. A legacy crop does not by itself demonstrate an incorrect TRE when
images, masks, landmarks, and field mapping all use the same geometry. Changed
crop content and interpolation can still affect optimization; numerical result
equivalence has not been established by this metadata audit.

Future fix scope:

- Give corrected and legacy preprocessing distinct, explicit identifiers and
  record the selected crop convention in every dataset/result manifest.
- Provide an explicit corrected COPD path while retaining the legacy path
  needed for historical results; do not silently change reproduction defaults.
- Reconcile `docs/standards/lungct_tre_metric_conventions.md`, which currently
  describes both datasets as corrected, and the COPD metadata's stale
  `mask_resampling` wording that calls its crop corrected.
- Validate interior/boundary crop cases, image/mask alignment, all 300 landmarks,
  native-to-crop-to-registration coordinate round trips, and legacy artifact
  parity. Registration reruns are not required to document or test geometry.
- Preserve existing prepared data, saved fields, and reported results. Any
  future corrected dataset gets a separate version and provenance; old result
  numbers must not be relabeled as outputs of corrected preprocessing.

This update records the fix only. No preprocessing code, datasets, or metric
values were changed.

### A05 — Masks and preprocessing have additional dependencies

Both 1 mm preprocessors require `--mask-root` and load both phases. R231 mask
generation uses `lungmask` and SimpleITK; the mask model weights are an additional
runtime artifact. The public guide must cover obtaining/caching the pinned
weights or a separately validated image-only preparation path. Fixed-mask
sampling, paired masks for QA, and evaluation foreground have different roles.

Raw landmark-based crop, CT intensity encoding, nearest mask interpolation,
trilinear image interpolation, and `align_corners=True` must be documented.
The Python resampling is not byte-equivalent to official MATLAB pTVReg.

### A06 — Three submodules do not yet imply three faithful wrappers

Inspected gitlinks are clean and currently pinned to:

| Dependency | Upstream | Candidate SHA |
| --- | --- | --- |
| IDIR | `https://github.com/MIAGroupUT/IDIR` | `9ab3ccfbbf09dc8fc65fa848332e7d9a0b3b81b3` |
| SINR | `https://github.com/vasl12/SINR` | `1a524ca7ae453b55310595fe957245088a108233` |
| Dual-INR | `https://github.com/IPMI-ICNS-UKE/DUAL-INR-DIR` | `f7c8e833435ec9c6b7ebec6c9a8098b566709146` |

Dual-INR has an explicit pinned import adapter in
`scripts/experiments/lungct_dual_inr_author/external_api.py`. OASIS and core
LungCT code define local SIREN, NCC, derivatives, and spline operations in
their own `common.py` modules. Extracting wrappers needs a behavior/provenance
crosswalk, with small numerical checks for initialization, forward fields,
losses/gradients, coordinates, and stage transfer. Do not silently replace the
paper's adaptation with upstream defaults. Historical result-source hashes
must be checked; current clean gitlinks alone do not prove historical identity.

Submodules pin a commit through the superproject's gitlink, independently of
upstream branch changes. See [Git submodule documentation](https://git-scm.com/docs/gitsubmodules).

### A07 — A single environment is a deliverable, not an established fact

There is no implemented Dockerfile in the inspected `docker/`; its READMEs are
placeholders. Actual experimental evidence includes Python 3.10 containers and
PyTorch `2.6.0+cu118`, CUDA runtime `11.8`, cuDNN `90100`.

Pinned upstream metadata conflicts: SINR requires `torch==2.2.1` and
`numpy==1.26.4`; Dual-INR declares `torch>=2.11.0` and `numpy>=2.2.6`.
IDIR's environment file lists mostly development tooling. Installing all three
dependency files together is not a workable release recipe.

Audit the imported runtime subset and produce a tested lock. Source imports or
installation without upstream dependency resolution need explicit compatible
constraints and validation, not a blanket bypass. Keep optional segmentation,
tracking, development, and external-reference tools out unless used.

Pin the public base by digest ([Docker build guidance](https://docs.docker.com/build/building/best-practices/)).
The host still needs a compatible NVIDIA driver and configured container
runtime ([NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

### A08 — New LungCT results exist, but publication is still pending

September 29 P34--P36 contain 4DCT/COPD adaptive/fixed50/backend results in
`docs/05_results_summary.md` and experiment records. Completion does not
override the user's request to leave Lung acceleration pending. COPD extension
rows need explicit source, method coverage, crop, and metric acceptance;
missing method cells must say unavailable rather than prompting automatic reruns.

### A09 — Public assets and private governance require selective export

The public checkout already tracks `figures/fig1.png`, `fig2.png`, and
`fig3.png`. `poster/offgrid_poster_v3.pdf` exists locally and is untracked
(approximately 4.3 MB); SHA-256
`e31a88bfcfbaf037ed44c8e8c3164224a66eee9feafc5428107050583736217b`.
Its README does not embed the figures or link the poster. The arXiv badge text
uses a different identifier from its destination. Verify paper/citation links
and replace the Oct 1 placeholder with actual release status when implementing.

The private repository's old `public/release_manifest.md` includes AGENTS and
internal planning reports. It is an anonymous scaffold, not a suitable public
export list. Use the selective new manifest; keep the existing public history.
Private source-only rules remain intact. No user-owned dirty files were edited.

## Resolved choices and remaining implementation inputs

- Q04 resolved: September 16 five-method base/fast paired timing, with historical
  OASIS100 accuracy separately listed. Do not merge the two into a claim that
  the updated backend was evaluated on all 100 cases. September 24 is excluded.
- Implementation follow-up: finalize project license/citation metadata, exact
  COPD row allowlist, Figure 1 editable-source availability, and tested Docker
  compatibility. The initial SINR interpretation was superseded by the user's
  explicit pending-permission clarification on 2026-09-30.
