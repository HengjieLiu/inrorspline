---
doc_type: technical_log
status: complete
experiment_id: dual_inr_dirlab_author_repro_v1
date_planned: 2026-08-02
---

# Technical Log

## Known limitations and warnings

- The plan is approved and locked. A provenance-bound adapter, launcher,
  evaluator, visual-review surface, and common posthoc regularity evaluator
  have been added. No smoke, pilot, or production optimization had been
  launched at the time of this ledger update.
- The author protocol crops the native-spacing images to a two-voxel-padded
  bounding box around the union of fixed and moving five-lobe masks. It is not
  native full-volume IDIR preprocessing and not pTVReg landmark-crop/iso1mm
  preprocessing.
- A1 disables only the semantic loss weight. Both variants still depend on
  TotalSegmentator masks for the crop and fixed/T00 coordinate-sampling domain;
  A1 is not a no-mask experiment.
- The locked released code implements the semantic term as summed per-lobe
  mean-squared error. The DUAL paper describes a binary-cross-entropy semantic
  loss. A0 means the released-code full loss at the locked commit, and the
  paper/code mismatch must be disclosed.
- Mask generation can materially change the crop, sampling domain, and A0
  objective. TotalSegmentator version/model provenance and segmentation
  checksums are therefore an acceptance requirement, not optional metadata.
- Blur-stage transitions are convergence-triggered within a 3000-step budget.
  A0 stage timing can differ by case even with the same seed. A1 intentionally
  replaces the adaptive trigger with a fixed replay of the same case/seed A0
  transition steps. Every accepted pair must reach stages `4,2,0` and have
  exactly equal realized boundaries; this mechanism change is a schedule-
  control intervention, not a second scientific factor.
- Canonical identity TRE after the upstream converter is `[3.892406,
  4.337842, 6.942993, 9.830147, 7.476853, 10.891024, 11.026220, 14.994681,
  7.918264, 7.301382]` mm for cases01-10. The data gate uses `1e-5 mm`
  absolute tolerance per case. The paper Table 1 rounded no-registration row
  differs and is recorded as a release discrepancy, not used as the data gate.
- The existing default container maps `/database` read-write. The prepared
  author-format data artifact is nevertheless logically read-only: future
  runners must reject output paths beneath it and must not mutate its contents.
- This N=10, one-seed cohort does not reproduce the paper's ten-seed robustness
  experiment. It measures case coverage under one locked initialization.
- The author-code result and pTVReg/IDIR rows are benchmark-context results, not
  clean method-only comparisons, because their domains, masks, preprocessing,
  losses, and optimization paths differ.
- Snap-to-voxel TRE in the native/crop coordinate system is the headline
  metric. Subvoxel and decoded-dense versus direct-INR TRE are diagnostics.
- The preregistered A0 fidelity interval is an audit gate, not an equivalence
  test. Falling outside it must be reported without post-hoc tuning.

## Locked decisions

- Cases: DIR-LAB 4DCT `01-10`.
- Direction: moving T50/end-exhale to fixed T00/end-inhale.
- Variants: A0 author full loss and A1 semantic mask weight zero.
- Seed: `42`, matching the upstream runner default.
- Source commit: `f7c8e833435ec9c6b7ebec6c9a8098b566709146`.
- Full-length pilot: cases `01` and `08`, A0 first and then A1 with the paired
  A0 transition boundaries. Per the user's explicit all-GPU authorization,
  use the available GPU pool with at most one optimizer per GPU.
- Cohort launch: all visible GPUs only after every source/data/config/pilot gate
  passes; at most one optimizer per GPU. Complete all A0 cases and freeze the
  transition manifest before launching A1.
- Output paths: environment-variable roots only; outputs are forbidden beneath
  the prepared data root and large artifacts remain outside Git.

## Planned run ledger

| Stage | Status | Required result |
|---|---|---|
| source and data audit | passed | exact source commit; 10 valid cases; 84-entry checksummed mask provenance |
| config-isolation audit | passed | semantic diff is only mask weight; allowed execution diff is A1 boundary replay |
| short case01 smoke | passed | 2/2 finite three-step records; exact A0 boundary replay; complete outputs |
| full cases01/08 A0 pilot | passed | 2/2 accepted 3000-step records and locked transition traces |
| full cases01/08 A1 replay pilot | passed | 2/2 accepted records with exact paired boundaries and visual audit |
| all-GPU A0 cohort | passed | 10/10 accepted A0 records and frozen transition manifest |
| all-GPU A1 replay cohort | passed | 10/10 accepted A1 records with exact paired boundaries |
| aggregate and fidelity audit | passed | paired tables, regularity/runtime metrics, and fidelity pass |

## Decisions during execution

Record future deviations, failures, resumes, and acceptance decisions here.
Do not silently revise `locked_plan.yaml` after a run begins.

### 2026-08-02 pre-launch gates

- Prepared-data version: `dual_author_native_lungbbox_totalseg_v001`.
- Independent validation: pass for cases `01-10`; phases T00/T50, labels
  `10-14`, image/segmentation geometry, 300 landmarks per phase, exact union
  crop containment, permissions, and all 84 checksums passed.
- Data bindings: dataset manifest
  `40e40c98219ac68e0b323f4a50d9b1687c447c03c42093cf180f2453fe402cb2`,
  preprocessing contract
  `12cc0b9cd667d2d4d133d828162759e5cec245687f7e899d773a924fa0ce4c24`,
  checksum manifest
  `b8b8d751def598b8c70b34cdfc293baa3e0c197649d5a3e814ec7046cd3ab0c3`.
- Exact executed converter SHA-256:
  `1f33439328f5d586d8086b5d01297b3067fe7d63f4fb8ee0cf99193e01179ba3`.
- Focused Docker verification: `28 passed`, with three benign SimpleITK SWIG
  warnings. Runner, launcher, aggregator, and posthoc CLI help gates passed;
  `git diff --check` passed.

### 2026-08-02 paired short smoke

- Launcher status: complete; A0 and A1 both independently revalidated as
  complete records.
- Both histories contain exactly three finite rows. A0 resolves semantic-mask
  weight `0.1`; A1 resolves `0.0`; both have the short-smoke boundary `[0]`.
- Dense versus direct-INR snap TRE difference is exactly `0.0 mm` for both
  variants. Case01 dense snap TRE is `3.026004 mm` for A0 and `3.047322 mm`
  for A1; these short-run values are execution diagnostics, not results.
- Training time was `28.24 s` for A0 and `31.31 s` for A1; peak reserved CUDA
  memory was `4.953 GiB` for each. This short run was later confirmed to be
  dominated by startup overhead; the measured full-pilot times below supersede
  the initial runtime estimate.

### 2026-08-02 full-length cases01/08 pilot

- Launcher status: complete; all four A0/A1 records have exactly 3000 finite
  history rows and independently validated output contracts.
- A0 realized stage starts were `[0,332,564]` for case01 and `[0,308,482]`
  for case08. Each paired A1 record replayed its exact casewise boundaries;
  both A1 histories have zero semantic-mask weighted loss throughout.
- A0/A1 dense snap TRE was `0.760360/0.767666 mm` for case01 and
  `1.088984/1.103322 mm` for case08. A0 agrees closely with the published
  per-case values `0.76` and `1.10 mm`.
- Dense versus direct-INR snap TRE difference was exactly `0.0 mm` for all
  four records. A0/A1 mean five-lobe Dice was `0.975088/0.973685` for case01
  and `0.960274/0.955736` for case08.
- Training wall time was approximately `340-346 s` per record; peak reserved
  CUDA memory was `8.54 GiB` for case01 and `9.82 GiB` for case08.
- Paired review figures and their provenance manifest are in `pilot_review/`.
  Visual inspection found coherent displacements, aligned lobe boundaries,
  and no obvious crop, sign, orientation, or schedule defect. Production is
  released without modifying the frozen executable code snapshot.

### 2026-08-02 production and final audit

- Production launcher status: complete; `20/20` records are
  `validated_done`, with exactly 3000 finite history rows per record, no
  unresolved/missing/duplicate records, and no in-place resume.
- All ten A1 traces exactly replay their paired A0 starts. A0 starts range from
  `[0,278,559]` to `[0,478,661]`, confirming why casewise replay was required.
- A0 dense snap TRE is `0.97894 +/- 0.17273 mm` and A1 is
  `0.98343 +/- 0.16688 mm` (across-case mean +/- sample SD). The paired
  A1-minus-A0 delta is `+0.00449 +/- 0.01385 mm`; seven cases are positive
  and three negative. Median paired delta is `+0.00889 mm` with IQR
  `0.01735 mm`.
- A0 fidelity status: pass. The observed mean is effectively the published
  `0.98 mm`, lies inside the locked `0.82-1.12 mm` interval, and differs from
  the rounded published per-case row by mean absolute `0.02356 mm` (maximum
  absolute `0.05964 mm`).
- Mean five-lobe Dice is `0.96788` for A0 and `0.96463` for A1. The paired
  delta is `-0.00325 +/- 0.00125`, negative in all ten cases; this is expected
  objective-sensitive evidence, not an independent anatomical endpoint.
- Common fixed-lobe-foreground regularity: A0/A1 diffusion
  `0.0130/0.0128`, bending `0.0046/0.0045`, SDlogJ `0.1401/0.1389`, and
  nonpositive-Jacobian fraction `0/0%`. Diffusion and bending remain
  voxel-coordinate quantities under the documented conversion.
- Mean training time is `358.93 s` for A0 and `350.30 s` for A1. Mean peak
  reserved CUDA memory is `9.11 GiB` for both.
- Final focused Docker verification: `30 passed` with three benign SimpleITK
  SWIG warnings. `git diff --check` passed.
- Machine-readable aggregation is in `results/`; common regularity records are
  in `regularity/`; paired visual review is in `pilot_review/`.

### 2026-08-02 contextual 1 mm references

- The camera-ready readout now includes casewise pTVReg and MR-D-BSCP
  pTV-style 1 mm reference TREs under the same native-landmark snap convention,
  plus common posthoc regularity mean +/- sample SD. They are explicitly tagged
  as protocol context rather than method-only comparisons.
- pTVReg uses the validated uploaded fields underlying the manuscript
  `0.9191 +/- 0.1525 mm` row (population SD). The official repository's saved
  headline artifact is a distinct `0.9176 +/- 0.1510 mm` row and is not
  silently substituted.
- Per the requested lowest-TRE rule, MR-D-BSCP uses the one global refined-sweep
  setting `w=0.055` for all cases: `0.9175 +/- 0.1481 mm` (population SD). This
  differs from the manuscript-quoted pTVReg-matched `w=0.11` operating point,
  `0.9218 +/- 0.1523 mm`, and must remain labeled as a posthoc lowest-TRE sweep
  reference rather than a preregistered estimate if adopted camera-ready.
- DUAL regularity is measured on the native author crop and fixed T00
  five-lobe union. The pTVReg/MR-D-BSCP regularity references are measured on
  the pTV 1 mm crop with a fixed T00 lung foreground used only for evaluation.
  Voxel-coordinate diffusion/bending therefore remain contextual across these
  preprocessing domains.
- Reference sources are the validated combined pTVReg case table and the
  refined DirectCP/MR-D-BSCP sweep case and mean tables listed in the readout.

### Review-artifact portability

- Before commit, machine-local run, data, review, and repository roots in the
  compact JSON/CSV/Markdown review bundle were replaced with symbolic roots
  such as `${RUN_ROOT}` and `${REPO_ROOT}`. Numeric results and provenance
  hashes were not recomputed or altered; the recorded generator hashes still
  identify the exact source snapshot used for the completed run.
