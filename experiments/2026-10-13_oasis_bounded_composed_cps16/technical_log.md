# P65 cps16 extension technical log

## Known limitations

One exploratory rank0 fit. Adding cps16 changes motion capacity, optimization
budget and the coarsest image/NCC physical support, not an isolated causal
test of coefficient saturation. P64 cps8 hit400 epochs as well as40.726%
coefficient saturation. Bounds apply to elementary map interpolation; sampled
export and canonical finite differences are separately measured. CUDA
grid_sample backward may remain nondeterministic; shared-load timing is diagnostic.

## Approved audit and preflight

- Fresh cps16/8/4/2/1, lambda[1,1,1,5,1], one zero-initialized increment each.
- New16 controls14x18x16x3; existing24x32x28,44x60x52,84x116x100,
  164x228x196 control shapes preserved. No prolongation between increments.
- Independent read-only FP64 audit found cps16 bound0.40190721937748236,
  cap6.366210354939321 native voxels/component. SITReg half-voxel result
  0.40175329749498867 matches its lookup; not substituted into our geometry.
- Loose chain per-axis motion ceiling rises5.799017689 to12.165228044 voxels.
- Explicit five-stage scheduler prevents cps2 inheriting index3 LR reductions.
- Initial current GPU3 snapshot:48515MiB free,4MiB used,utilization0%; no jobs
  changed. Admission is rechecked by launcher.
- Verified container image hengjieliu96/dl_dir:py310sitreg_v1, image ID
  sha256:dab700c415830dcf07c16ef7f2901deed95bb104eda388514c3ce74a43354ba8;
  separate writable artifact bind and read-only OASIS shared-storage mount.
- All29 tests passed: six new numerical/CLI tests, seven unchanged P64 tests,
  and16 unchanged NCC9/diffusion tests. The native-source regression was
  rerun with both source roots configured and passed without skips.
  No test skips remain.
- Full-size smoke completed71.69s, peak2.83241GiB. All five identity checks
  exactly0 voxel,35 valid labels/stage, native loss/gradient parity passed,
  all atomic corner bounds<1. Retained/export off-grid minimum determinants
  0.163430/0.164909, zero nonpositive probes. Probe point hash matches P64.
  Memory gate remains8GiB. Production cleared.

## Production outcome

- One fresh registration completed2026-10-13 23:44:30UTC without failures or
  retries. All five stages passed corner, serialization, loss and metric checks.
- Dice0.770575918, HD952.379340995mm, symmetric ASD0.532365927mm,
  fixed-foreground positive-J SDlogJ0.506025791, folding0%.
- Versus bounded P64: Dice-0.003493027, HD95+0.154360297mm,
  ASD+0.035321727mm, SDlogJ+0.012555775, folding unchanged at0%.
  Thus prepending cps16 did not improve accuracy on this case.
- Coefficient saturation>95% is23.934%/35.302%/24.262%/14.063%/16.266%
  for cps16/8/4/2/1. Cps8 saturation decreases from P64's40.726% to35.302%
  without an accuracy benefit; this does not establish a unique causal mechanism.
- Best/stop epochs275/275,300/300,175/175,150/150,495/500. Final cps1
  used exactly two LR halvings; earlier stages used none. Optimization66.849s,
  complete worker123.178s, peak2.833GiB; timing remains diagnostic.
- Atomic maximum derivative row sums0.818346/0.951523/0.980303/0.987692/
  0.986339. All54,178,296 one-sided cell corners per elementary map passed.
- At the same100,000 off-grid points as P64, retained-chain/export minima
  0.043209718/0.046212820 with zero nonpositive probes. These are measured
  diagnostics, not a universal exported-field topology guarantee.
- Six-row comparison CSV/JSON/Markdown and five-row stage report accepted.
- The final native-grid objective falls0.754680157 (P64) to0.750544369,
  and NCC9 data loss falls0.731728196 to0.725302100, despite worse canonical
  label accuracy. Diffusion rises0.022951951 to0.025242247. Extra coarse
  capacity does not guarantee a better segmentation-metric solution.

## Independent acceptance

- Independent audit passed2026-10-13 23:45:32UTC. All five checkpoint
  latent/control/field reconstructions are exact; selected objective terms,
  full-cell-corner audits and five canonical metrics reproduce.
- All35 labels/stage are valid. Dice, HD95 and both directional/symmetric ASD
  per-label results replay within1e-12. Fixed raw-positive foreground and
  existing absent-label convention are unchanged.
- The100k off-grid diagnostic replays exactly, including the identical P64
  point hash. Additional128-point FP64 central differences agree with chain
  coordinate autograd to max error2.425201e-9 (normalized step1e-7).
- Numerical source, historical reference and accepted report hashes remain
  unchanged. Independent audit source hash and findings are preserved in the
  external run root's independent_audit.json. No failures/retries or new jobs
  beyond the approved smoke, production and verification were run.

## Reproduction

Use the verified existing container as UID:GID1009:1020. Set ARTIFACT_ROOT,
SITREG_ROOT, DL_NCC_CODE_ROOT and VFA_ROOT from local configuration. Export
INRORSPLINE_CONTAINER_IMAGE/INRORSPLINE_CONTAINER_IMAGE_ID from the verified
container. Use Python-B, CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=3,
MPLCONFIGDIR=/tmp and PYTORCH_KERNEL_CACHE_PATH=/tmp. Existing jobs are not
stopped or reprioritized. The launcher admits one GPU3 worker at a time.

```bash
python -B -m unittest discover -s tests -p test_oasis_bounded_composed_cps16.py -v
python -B -m unittest discover -s tests -p test_oasis_bounded_composed.py -v
python -B -m unittest discover -s tests -p test_oasis_dl_ncc_diffusion.py -v
python -u -B scripts/experiments/oasis_alpha_sweep/run_oasis_bounded_composed_cps16.py launch \
  --artifact-root "$ARTIFACT_ROOT" --sitreg-root "$SITREG_ROOT" --gpu 3 \
  --case-rank 0 --stage-alpha 1 1 1 5 1 --bound-multiplier 0.99 \
  --increments-per-stage 1 --stop-after smoke
python -u -B scripts/experiments/oasis_alpha_sweep/run_oasis_bounded_composed_cps16.py launch \
  --artifact-root "$ARTIFACT_ROOT" --sitreg-root "$SITREG_ROOT" --gpu 3 \
  --case-rank 0 --stage-alpha 1 1 1 5 1 --bound-multiplier 0.99 \
  --increments-per-stage 1
python -u -B scripts/experiments/oasis_alpha_sweep/audit_oasis_bounded_composed_cps16.py \
  --artifact-root "$ARTIFACT_ROOT"
```

All runs, metric CSVs, checkpoints, histories and generated reports remain
under the distinct external P65 artifact key. The launcher refuses to overwrite
incomplete attempts and returns without changes for an accepted experiment.
The independent audit also refuses to overwrite an existing audit record.
