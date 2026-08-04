---
doc_type: codex_execution_spec
status: authorized_for_execution
experiment_id: dual_inr_dirlab_controlled_multires_v1
date_planned: 2026-08-02
---

# Controlled DUAL-INR DIR-LAB Execution Specification

## Outcome and scope

Execute all four locked variants for DIR-LAB 4DCT cases `01-10`, seed `42`
(`40` records). N0/N1 use native full-volume IDIR-style preprocessing; P0/P1
use the canonical pTVReg-style landmark crop at 1 mm isotropic spacing. The
`0` variants are four-stage full-resolution schedule controls and the `1`
variants use a true `[8,4,2,1]` image pyramid. Poor TRE or regularity never
stops a record, blocks the queue, or triggers tuning.

`locked_plan.yaml` is the machine-readable source of truth. A change to source
commit, preprocessing, sampling domain, architecture, weights, stage timing,
pyramid construction, seed, or optimizer rule requires a new experiment ID.

## Frozen protocols

N0 and N1 load raw native T00/T50 volumes without cropping, resampling, or
pTVReg intensity conversion. Optimization samples 30,000 coordinates per step
from the fixed/T00 R231 native lung mask only. The moving mask is not used.
Headline and subvoxel TRE remain in native physical coordinates.

P0 and P1 use the canonical pTVReg-style 1 mm data: the fixed/moving landmark-
union bounding box with xyz margin `[10,10,5]`, CT clipped to `[80,900]` and
scaled to `[0,1]`. Optimization uses the full fixed crop except a five-voxel
border on every face. No fixed or moving lung mask influences optimization.
The displacement is evaluated on the registration grid and remapped to the
native crop; native-remapped snap TRE is headline, while registration-grid and
native-remapped snap/subvoxel TREs are retained as diagnostics.

N-to-P differences are intentional protocol differences. They must not be
presented as a preprocessing-controlled method comparison.

## Frozen model and objective

Use the DUAL-INR code at commit
`f7c8e833435ec9c6b7ebec6c9a8098b566709146`, with additive coarse/fine SIREN
branches `3-256-256-3`, frequencies `10/30`, and output scales `0.1/0.05`.
The fine contribution is disabled for global steps `0-399` and enabled at
step `400`; model weights persist thereafter.

Every variant optimizes the same sampled global NCC plus `0.01` Laplacian and
`0.01` negative-Jacobian penalties. Fine-branch regularization is scaled by
`0.1`. The semantic mask weight is exactly zero and the semantic term must not
be evaluated. Thus N sampling-mask use is a coordinate-domain restriction, not
a semantic loss; P has no optimization-mask use at all.

These are reused, frozen working hyperparameters. Cases 01/08 validate only
finite execution, coordinate transforms, memory, and outputs. They cannot
select weights, stages, or a winning variant. Any later tuning is a separately
identified sensitivity study.

## Exact four-stage schedule

| Stage | Global steps | Length | N0/P0 factor | N1/P1 factor | Fine branch |
|---|---:|---:|---:|---:|---|
| 0 | 0-399 | 400 | 1 | 8 | off |
| 1 | 400-799 | 400 | 1 | 4 | on |
| 2 | 800-1599 | 800 | 1 | 2 | on |
| 3 | 1600-2499 | 900 | 1 | 1 | on |

Build every image level independently from the original full-resolution fixed
or moving tensor; do not recursively downsample. For factor `f`, anti-alias
with a separable Gaussian of sigma `0.5*f` voxels, kernel radius
`ceil(3*sigma)`, and replicate padding, then trilinearly resample each axis to
`ceil(full_size/f)` using aligned corners. Factor-one stages retain the full
image. Record full and stage shapes in `stage_trace.csv`.

The coordinate sampling lattice never changes with the image level. At every
stage, draw full-resolution fixed-lattice indices from the same locked N or P
domain, normalize them against the full-resolution lattice to `[-1,1]`, and
query the current fixed/moving level at those normalized locations. Do not
downsample the R231 mask or redefine the P border in stage voxels.

At each stage boundary, preserve model weights but instantiate a fresh AdamW
optimizer and fresh stage-local warmup-cosine scheduler. Use LR
`1e-3 -> 1e-5`, weight decay `0.01`, betas `(0.9,0.999)`, epsilon `1e-8`, and
20 warmup steps. Optimizer moments do not cross stage boundaries. N0/P0 use
the same resets, stage lengths, and fine activation as N1/P1; only the image
factors differ within each causal pair.

## Implementation and validation order

1. Verify the clean pinned submodule, Docker image/container, four visible
   GPUs, and logically read-only data roots. Reject any output under a data
   root.
2. Validate all ten raw native and pTVReg-preprocessed cases: phase direction,
   geometry, 300 landmarks per phase, native fixed-mask nonemptiness, pTV crop
   metadata, intensity contract, and native-remapping round trip.
3. Add the controlled adapter without modifying the pinned external source or
   the completed author-reproduction executable snapshot. Unit-test independent
   pyramid shapes, anti-aliasing, aligned-corner coordinates, full-resolution
   sampling domains, fine activation, fresh optimizer state, and P native
   remapping.
4. Dry-run a unique 40-record manifest. Resolved configs may differ within
   N0/N1 or P0/P1 only in variant identity, pyramid factors, and output path.
5. Run disposable case01 technical smokes that cross all four stage boundaries.
   Repair technical faults; do not inspect TRE as an acceptance threshold.
6. Run full cases01/08 technical records for all variants. Exact-compatible
   accepted records may be reused in the 40-record cohort. Finite values,
   coordinate correctness, memory, stage traces, and artifact completeness are
   the only gates.
7. Dispatch the full matrix over four visible GPUs with one optimizer per GPU
   and a dynamic queue. Attempt every record. A failed record is logged and
   retried while unrelated jobs continue; never change the frozen scientific
   config to rescue it. A low-quality scientific result is a completed result.
8. Aggregate only after checking unique variant/case/seed keys, exactly 2500
   finite rows, exact stage boundaries and factors, complete fields, and valid
   evaluation transforms for each accepted record.

## Metrics and reporting

Report casewise and population TRE using mean/sample SD and median/IQR. Native-
remapped landmark snap TRE is the cross-protocol headline. Retain snap and
subvoxel diagnostics in both registration and native domains where applicable.
Also report runtime, peak GPU memory, and common posthoc diffusion, bending,
SDlogJ, and nonpositive-Jacobian fraction.

For N, common regularity uses the fixed-T00 R231 native lung foreground only
as a metric domain. For P, it uses the fixed-T00 lung foreground mapped to the
pTVReg 1 mm crop, also metric-only. Never imply that the P evaluation mask was
used in optimization. Native anisotropy and normalized-coordinate derivatives
mean a fixed regularization weight has different physical meaning across N and
P; voxel-coordinate diffusion/bending remain tagged contextual values.

The causal schedule contrasts are N0 versus N1 and P0 versus P1. N1/P1 are
“MR-D-BSCP-shaped” or “MR-inspired” image schedules only: DUAL-INR does not
gain MR-D-BSCP control-point spacings, knot insertion, objective, or its tuned
regularization weights.

## Night-run failure behavior

- There is no scientific early stopping and no TRE/regularity rejection gate.
- Preserve each attempt's resolved config, stdout/stderr, partial history, GPU,
  timestamps, and failure reason. Never overwrite an accepted record.
- Retry technical failures while the dispatcher continues unrelated jobs.
- Do not modify preprocessing, weights, seed, model, schedule, or evaluation to
  improve a result under this experiment ID.
- If an external blocker prevents one record from completing, finish all other
  records and report the exact unresolved key and evidence; do not substitute
  another case or silently aggregate a partial cohort as complete.

## Required compact deliverables

- source/data/config/manifest audit;
- 40-record optimization index and attempt ledger;
- per-case TRE, regularity, runtime, and memory tables;
- paired N0-N1 and P0-P1 delta tables;
- stage/coordinate/remapping audit and representative visual review;
- provenance manifest linking configs, code, input manifests, and output hashes.
