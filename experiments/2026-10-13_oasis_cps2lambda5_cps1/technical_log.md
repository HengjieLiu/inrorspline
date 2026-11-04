# Rank0 [1,1,5,1] cps1 comparison technical log

## Known limitations

Single exploratory test pair; no sample SD or first10 claim. Final parameter
geometry changes Adam's field-space updates even with identical nominal LR.
No Jacobian-positivity guarantee. CUDA grid_sample backward may be
nondeterministic. Shared-load runtimes are diagnostic only.

## Preflight

- User requested all-B-spline8→4→2→1 at [1,1,5,1] versus saved dense version.
- Source P62 rank0 dense: Dice0.804581580, HD951.995024962mm,
  ASD0.411972004mm, FG SDlogJ0.799018895, folding3.695896205%.
- Same saved prefix, full-res NCC9 and finite-difference total-field diffusion;
  do not replace the regularizer with analytic B-spline diffusion.
- GPU3 idle at inspection,48515MiB free. GPU0/2 have unrelated active jobs.
  Single GPU3 worker, UID:GID1009:1020, existing jobs untouched.
- Existing image hengjieliu96/dl_dir:py310sitreg_v1; verified image digest
  sha256:dab700c415830dcf07c16ef7f2901deed95bb104eda388514c3ce74a43354ba8.
- All3 new unit tests and16 unchanged NCC9/diffusion regression tests passed.
- Full-size one-step smoke passed:35 valid labels, finite serialized field,
  cps2 reconstruction exact, cps2→cps1 max/RMS3.322959e-6/1.663603e-7 voxel.
  Initial total/data loss exactly equal to saved dense; diffusion differs
  1.86e-9 from FP32 prolongation. On the identical expanded field, objectives
  agree exactly. Native NCC9/diffusion value and gradient errors are all0.
- Smoke peak1.986350GiB, free-memory gate4GiB; worker25.38s including metrics.
- Production completed on GPU3 without failures/retries: stopped725, best719,
  two LR halvings, selected loss0.707847953, diffusion0.0451655611.
  Optimization58.778s, worker82.371s, peak1.986350GiB.

## Results and independent acceptance

- Cps1 Dice/HD95/ASD0.805363430/1.983874347/0.411986484;
  FG SDlogJ0.802907954, folding3.718802445%.
- Cps1 minus dense: Dice+0.000781851, HD95-0.011150616mm,
  ASD+0.000014480mm, FG SDlogJ+0.003889059, folding+0.022906240pp
  (+0.619775% relative). Matching accuracy does not reduce folding here.
- Independent audit2026-10-13 22:42:03UTC recomputed both saved fields'
  accuracy and regularity, checked all35 per-label values and mean reduction,
  verified fixed_raw>0 mask, exact cps1 checkpoint-to-DVF reconstruction,
  best-history epoch/loss consistency, accepted CSV/JSON/Markdown hashes,
  and unchanged historical source hashes. All checks passed.
- Report directory has comparison.csv, results.json, report.md, ACCEPTED.json.
  The run directory retains one-step smoke, production logs/histories,
  best/last control checkpoints, finite dense field and independent_audit.json.
- No cohort claim: these are rank0 only. Original rank0 VFA/SITReg folding
  0.255035%/0.000697% remains far lower; cps1 does not enforce positive J.

## Reproduction

In the verified container as UID:GID1009:1020, with image tag/digest exported
as INRORSPLINE_CONTAINER_IMAGE and INRORSPLINE_CONTAINER_IMAGE_ID:

```bash
python -B -m unittest discover -s tests -p test_oasis_cps2lambda5_cps1.py -v
python -B -m unittest discover -s tests -p test_oasis_dl_ncc_diffusion.py -v
python -u -B scripts/experiments/oasis_alpha_sweep/cps2lambda5_cps1.py launch \
  --artifact-root /database/inrorspline --gpu 3 --stop-after smoke
python -u -B scripts/experiments/oasis_alpha_sweep/cps2lambda5_cps1.py launch \
  --artifact-root /database/inrorspline --gpu 3
```

The launch command does not overwrite accepted results; it returns if complete.
