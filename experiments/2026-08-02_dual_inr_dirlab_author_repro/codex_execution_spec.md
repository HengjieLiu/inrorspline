---
doc_type: codex_execution_spec
status: planned
experiment_id: dual_inr_dirlab_author_repro_v1
date_planned: 2026-08-02
---

# DUAL-INR-DIR Author Reproduction Execution Specification

## Objective

Reproduce the released DUAL-INR-DIR dual-branch configuration on all ten
DIR-LAB 4DCT cases, then isolate the semantic lobe-mask loss by changing only
`loss.mask_weight` from `0.1` (A0) to `0.0` (A1). To prevent that loss change
from also changing the adaptive multiscale schedule, A1 replays the exact blur-
transition steps realized by A0 for the same case and seed.

This specification authorizes future implementation and execution only after
the listed gates pass. It does not launch a run, and this planning change does
not modify baseline code.

## Locked scope

- Source: clean `external/DUAL-INR-DIR` submodule at commit
  `f7c8e833435ec9c6b7ebec6c9a8098b566709146`.
- Dataset: DIR-LAB 4DCT cases `01-10`, fixed T00/end-inhale and moving
  T50/end-exhale (`T50 -> T00`).
- Protocol: `dual_author_native_lungbbox_totalseg`.
- Variants: A0 author-code full loss and A1 with semantic mask weight zero.
- Seed: `42` only, matching the upstream runner default.
- Matrix: `10 cases x 2 variants = 20` production records.
- Full-length pilot: cases `01` and `08` for both variants (`4` records).
- No COPD, pTVReg preprocessing, IDIR-style native-full adaptation,
  multiresolution adaptation, alpha sweep, or multi-seed robustness campaign.

The machine-readable source of truth is `locked_plan.yaml`. Any change to its
dataset, source commit, model, loss weights, schedule, seed, or acceptance
thresholds requires a new versioned plan and experiment ID.

## Variant isolation

A0 follows the released author code at the locked commit:

```text
sampled NCC
+ 0.01 * Laplacian/curvature loss
+ 0.01 * negative-determinant penalty
+ 0.10 * five-lobe semantic mask loss
```

A1 changes the scientific loss factor only as follows:

```text
loss.mask_weight: 0.10 -> 0.00
```

The execution mechanism also changes in one controlled way: A0 uses the
released adaptive relative-convergence transition rule, while A1 uses fixed
case-specific transition steps copied from the completed A0 run. The realized
stage boundaries must be identical within every A0/A1 case pair. This is a
schedule-control intervention, not an additional scientific factor.

A1 still loads fixed and moving lobe masks, crops around their union, and
samples training coordinates from the fixed/T00 lobe union. It must never be
described as no-mask or intensity-only registration.

At this source commit the semantic implementation is a per-lobe summed MSE,
not BCE. Reports must describe the executed code, while separately noting the
paper/code wording difference.

## Environment and data contract

Resolve paths only through:

```text
REPO_ROOT
DUAL_AUTHOR_DATA_ROOT
DUAL_AUTHOR_DATA_MANIFEST
DUAL_AUTHOR_PILOT_ROOT
DUAL_AUTHOR_RUN_ROOT
```

Treat `DUAL_AUTHOR_DATA_ROOT` as a logically read-only prepared artifact. The
existing default container exposes `/database` through a read-write mapping,
so this plan does not claim mount-level read-only enforcement. A future runner
must reject pilot or production output paths beneath the data root and must not
mutate its contents. The root must contain author-format converted NIfTI cases
with phase `00` and `05` images, corresponding segmentations, metadata, and 300
extreme-phase landmarks. The manifest must record checksums, conversion
provenance, TotalSegmentator version/model provenance, image shapes/spacings,
mask labels, and phase orientation.

Large models, dense fields, histories, and logs remain under the external
pilot/run roots. Only compact manifests, aggregate tables, audit reports, and
review figures may be written under this experiment directory.

## Required validation order

1. **Source gate:** verify the exact submodule commit and a clean submodule.
2. **Data gate:** validate all ten cases, phase direction, shapes/spacings,
   300 landmarks per phase, five nonempty lobe channels, mask-derived bounding
   boxes, checksums, and segmentation provenance. Converter-derived identity
   TRE must match
   `3.892406,4.337842,6.942993,9.830147,7.476853,10.891024,11.026220,14.994681,7.918264,7.301382`
   mm for cases01-10 within `1e-5 mm` per case. These canonical raw-landmark
   values intentionally replace the paper Table 1 rounded no-registration row.
3. **Static/config gate:** compile future experiment wrappers in the declared
   Docker image, inspect CLI help, resolve configs, and prove that the A0/A1
   scientific config diff contains only the semantic weight. The only other
   allowed execution diff is adaptive A0 transitions versus fixed A0-boundary
   replay in A1.
4. **Manifest gate:** dry-run exactly 20 unique production records and four
   unique full-length pilot records, with no duplicate variant/case/seed key.
5. **Short smoke:** run three steps for A0 and A1 on case01; require finite
   loss components and correctly named outputs. These outputs are disposable.
6. **Full-length pilot:** run A0 first for cases01 and 08 on one selected GPU,
   validate and lock each A0 transition trace, then run paired A1 with those
   exact boundaries. All four runs must complete 3000 finite steps, visit sigma
   stages `4,2,0` in order, activate the fine branch at stage index 1, match
   A0/A1 boundaries casewise, write every required artifact, and keep dense-
   versus-INR snap TRE within `0.05 mm`.
7. **Pilot review gate:** inspect case01/08 image orientation, crop/mask
   overlays, displacement magnitude, landmark movement, stage curves, and
   A0/A1 config-diff report. Any unresolved failure blocks the cohort.
8. **Production launch:** only after all prior gates pass, use all visible GPUs
   with at most one active optimizer per GPU. Complete A0 first, validate and
   freeze the ten-case transition manifest, then launch A1 using the matching
   casewise boundaries. Exact-compatible pilot records may be reused through
   explicit provenance; launch or resume remaining records without overwriting
   accepted pilot artifacts.
9. **Aggregation gate:** require 20/20 accepted records, N=10 for both
   variants, no missing/duplicate keys, 3000 finite history rows per record,
   identical paired stage boundaries, and complete TRE, runtime, memory,
   dense-field, and posthoc-regularity outputs.
10. **Fidelity gate:** compare A0 with the published author mean (`0.98 mm`).
    The preregistered consistency interval is `0.82-1.12 mm`, based on the
    published `0.97 +/- 0.15 mm` multiseed summary. Outside this interval,
    preserve and report all outputs but mark author-result fidelity failed;
    do not tune or reseed under this experiment ID.

## Reporting contract

Headline TRE is native/crop-domain snap-to-voxel TRE in physical millimeters.
Also retain subvoxel TRE and direct-INR versus decoded-dense TRE agreement.
Aggregate A0 and A1 separately and provide a paired per-case `A1 - A0` table,
mean/SD, median/IQR, runtime, peak memory, and common posthoc diffusion,
bending, SDlogJ, and folding metrics.

The analysis is descriptive. Do not claim statistical equivalence,
superiority, architecture isolation, or a method-only comparison with IDIR or
pTVReg. A scientific outcome is never converted into a job failure; technical
acceptance and author-result fidelity are reported separately.

## Failure behavior

- Stop before full-cohort launch on any source, data, config-isolation,
  manifest, short-smoke, or full-pilot gate failure.
- Preserve successful pilot and production outputs; never overwrite or relabel
  an incompatible record.
- Report the exact case, variant, seed, GPU, log, and failed gate.
- Do not change weights, seed, A0 stage criteria, replayed A1 boundaries, masks,
  or preprocessing to rescue a result. Any such change requires a new plan.
- No camera-ready result may be generated from a partial cohort.

## Planned compact deliverables

- resolved job manifest, A0/A1 scientific config-diff audit, and casewise
  transition-replay audit;
- data/orientation/mask validation report;
- pilot acceptance report and visual review;
- 20-record optimization index;
- per-case and population metric tables;
- author-fidelity assessment;
- provenance manifest with source/data/output checksums.
