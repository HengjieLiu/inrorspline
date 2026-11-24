# MR-D-BSCP HU/crop/lambda campaign

Approved2026-11-19; execution key lungct_hu_crop_lambda_20261120T075001Z.
All20 DIR-Lab4DCT/COPD cases, nine preparations and180 independent searches.
First4DCT01/08,COPD01/08; emit their report before automatically releasing
remaining cases02/03/04/05/06/07/09/10. No approval pause or pTVReg execution.

## Frozen scientific protocol

- H0[-944,-124], H1[-1000,1000], H2[-1000,2000]HU. Subtract1024 from raw;
  clamp native images before resize, common gain1/820 and offset944/820.
  Preserve H0 bitwise against original img_thr. No subsequent clipping.
- C0 corrected landmark-union crop, native XYZ margins10/10/5. C1 bilateral
  R231 union across both phases, XYZ20/20/10mm padding, native ceil then clip.
  C2 full native FOV. Same pair bounds; masks remain metric/crop-only.
- Native-to-zero-based conversion, np.rint(native_shape*spacing), trilinear
  align_corners=True, nearest masks. Original/native snap TRE over300 pairs;
  keep subvoxel and registration-grid diagnostics, endpoint geometry and units.
- Reuse unmodified numerical MR train_multistage_spline implementation:
  controls64/32/16/8/4, downsample16/8/4/2/1,400/400/400/400/900 iterations,
  Adam1e-3, seed1, FP32, zero controls, exact knots, pTV-LCCsigma2.1mm,
  csqrt0.005, border5. Fresh process/optimizer/candidate; no warm starts.
- Backend benchmarkFalse/deterministicTrue. Record TF32, requested/effective
  flags, software, image, source/config/input hashes and shared-GPU context.
  These backend flags do not guarantee bitwise determinism for every CUDA
  operation; repeated original-CLI smokes show this limitation. The unchanged
  numerical solver, not bitwise identity across reruns, is the reuse contract.
- Lambda0.055*2^k, k=-8..8. Center then independent outward branches. Strict
  increase threshold1e-6mm against prior point; reset on tie/improvement/failure.
  Two successive increases stop that branch; exponent cap is a separate status.
  Failed candidate is not infinity. Best tested score ties prefer nearest0
  exponent, then higher weight. All candidate outcomes retained.

## Execution and failure policy

Source scripts/experiments/lungct_hu_crop_lambda; preparation helpers under
scripts/data/preprocessing/lungct. Freeze all executed sources externally before
production. Raw/source mounts RO; new artifact mounts RW in pinned Docker.
Prepared/runs/review roots use ARTIFACT_ROOT and execution key. Never overwrite
historical artifacts, stage, commit, publish or stop existing jobs/containers.

All four memory-qualified GPUs, max one campaign registration/GPU. Admission
free>=1.2*predicted incremental peak+2GiB; unknown shapes first profile on the
most-free GPU. Small jobs may share otherwise occupied GPUs if memory allows.
Memory smokes use two iterations per stage, are clearly nonproduction, and do
not contribute scientific TRE. Production always2500 steps. Failures produce
immutable receipts/logs/partial fields; no retry or algorithm/resolution fallback.
Preparation failure skips affected candidates and leaves other work running.
Center jobs precede branch expansion within each batch; persist controller state
atomically and validate receipts/source identity on resume. Completed first-batch
report is a durable automatic transition checkpoint, not an approval gate.

## Outputs and checks

Save each canonical FP32 ZYX3 pull DVF and native-copy, all landmark predictions,
four TRE endpoints, geometry, loss terms, resource statistics and provenance.
Reporting: fixed0.055 and best-tested nine-condition casewise tables, selected
lambdas, curves, stop reasons and failures; separate4DCT/COPD cohort mean/sample
SD of ten case means. Never hide missing cases or pool unequal available cohorts.
Tuning uses the reported300 landmarks and must be labelled. Capped best is not
a proven optimum. Retain high TRE and deformation pathology as results.

Additional independently inferred anatomy labels are evaluation-only, frozen
and reviewed for coverage/segmentation quality. Report lung/bone/soft-tissue
alignment and folding/Jacobian diagnostics on shared valid crop coverage;
additional FOV coverage separately. Annotation failure does not stop DIR.

Preflight: synthetic clip/crop/round-trip/TRE/stop/failure tests; all20 geometries
and300 points; H0C0 parity; zero/translation fields; frozen-source import and
real reduced MR memory smokes. Gate CUDA/dependencies/storage before release.
At most3060 production candidates,612 in first batch; smokes listed separately.
