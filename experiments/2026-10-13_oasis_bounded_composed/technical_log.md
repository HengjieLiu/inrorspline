# Bounded compositional pilot log

## Known limitations

One exploratory case, one bounded increment per level: total motion capacity
is restricted. This changes parameterization and composition, not only tanh.
The native atomic trilinear fields and their retained composition differ from
the final exported field's interpolant. Canonical finite-difference folding is
not a certificate over continuous coordinates. Shared-load timing is diagnostic;
CUDA grid_sample backward can remain nondeterministic.

## Preflight

- Approved fresh identity cps8/4/2/1 with alpha[1,1,5,1], one increment each.
- Local SITReg revision546d6c760aa0cb456a46fad2f1627f6ef069531f; source audit
  confirms componentwise tanh,0.99 bound multiplier, and composition. Its
  even-stride kernel uses half-voxel phases unlike our endpoint-aligned kernel.
- GPU3 utilization0%,48086MiB free at initial inspection; existing jobs untouched.
- Image hengjieliu96/dl_dir:py310sitreg_v1, digest
  sha256:dab700c415830dcf07c16ef7f2901deed95bb104eda388514c3ce74a43354ba8.
- All7 new unit tests and16 unchanged NCC9/diffusion regression tests passed.
- The independent all-phase/all-corner/all-sign FP64 calculation reproduces
  local SITReg native algorithm/lookup within1e-12. Our endpoint-aligned
  control-unit bounds at cps8/4/2/1 are0.40088018474881926,
  0.39263803680981596,0.36,0.3600000000000001. With0.99 margin, native-voxel
  component caps are3.1749710632106485,1.5548466257668712,0.7128,0.3564.
- Adversarial coefficient patterns instantiated through the production spline
  expander attain derivative row norm0.99 within FP32 tolerance; no negative
  one-sided cell-corner determinants. Tanh derivative at zero is exactly1.
- Full-size one-step-four-stage smoke passed: all new-stage identity checks
  exactly0 voxel, all35 labels valid per stage, exact native NCC9/diffusion
  value/gradient parity. Every atomic corner bound<1. No nonpositive canonical
  or100k off-grid Jacobians; retained/export minimum probes0.239157/0.245759.
- Smoke40.61s including metrics/topology, peak2.67824GiB; memory gate8GiB.
  GPU3 production cleared. A harmless PyTorch kernel-cache warning in smoke
  is addressed by PYTORCH_KERNEL_CACHE_PATH=/tmp for production; no numerical
  source/settings change.

## Production and results

- One full fresh fit completed without failures or retries; no extra capacity
  or lambda/LR search. Stage stopping400/200/150/500; best400/200/150/496.
  Cps8 reached its approved maximum; other stages stopped adaptively, with
  cps1 applying both LR halvings. Total optimization63.292s, worker109.819s,
  peak2.67824GiB. Timings are diagnostic only.
- Final Dice0.774068946, HD952.224980697mm, symmetric ASD0.497044200mm,
  fixed-foreground positive-J SDlogJ0.493470017, folding0%.
- Against unbounded P63, Dice-0.031294485, HD95+0.241106351mm,
  ASD+0.085057716mm, SDlogJ-0.309437938, folding-3.718802445 percentage points.
  This reduces folding but does not preserve previous accuracy.
- Max atomic displacement-Jacobian row sums across all cell corners:
  0.977792225/0.984813288/0.988911904/0.989733569 (cps8/4/2/1).
  All elementary-map corner determinants positive. Coefficient fractions
  above95% of caps40.726%/28.417%/14.054%/16.320%; saturation/restricted
  capacity are possible explanations, not isolated causes from this pilot.
- The retained chain and interpolated export have0/100,000 nonpositive
  off-grid Jacobians, with minima0.045249890/0.045843159. The canonical
  foreground folding metric is also0%; no universal downstream claim.

## Independent acceptance

- Audit completed2026-10-13 23:26:18UTC. All four native fields reconstruct
  exactly from their saved latent/control checkpoints. Selected losses and
  full native-cell-corner derivative checks reproduce exactly.
- All35 labels are valid at every stage; per-label reductions, final accuracy
  and regularity recompute within1e-12, using fixed_raw>0 foreground.
- Seed1 off-grid probes replay exactly. An additional512-point FP64 coordinate
  finite-difference check matches the retained-chain autograd Jacobian with
  max absolute error2.158879e-9. Source/reference/report hashes unchanged.
- External report contains comparison.csv, stages.csv, results.json,
  report.md and ACCEPTED.json. Run directory retains all four best/last
  checkpoints, fields, histories, per-stage metric CSVs, corner/probe audits,
  smoke outputs and independent_audit.json.

## Reproduction

Use the verified existing container, UID:GID1009:1020, physical GPU3. Export
INRORSPLINE_CONTAINER_IMAGE and INRORSPLINE_CONTAINER_IMAGE_ID from its verified
tag/digest. Set SITREG_ROOT to the local SITReg checkout and ARTIFACT_ROOT to
the external artifact bind. Use Python-B, MPLCONFIGDIR in/tmp, and
PYTORCH_KERNEL_CACHE_PATH=/tmp.

```bash
python -B -m unittest discover -s tests -p test_oasis_bounded_composed.py -v
python -B -m unittest discover -s tests -p test_oasis_dl_ncc_diffusion.py -v
python -u -B scripts/experiments/oasis_alpha_sweep/run_oasis_bounded_composed.py launch \
  --artifact-root "$ARTIFACT_ROOT" --sitreg-root "$SITREG_ROOT" --gpu 3 \
  --case-rank 0 --stage-alpha 1 1 5 1 --bound-multiplier 0.99 \
  --increments-per-stage 1 --stop-after smoke
python -u -B scripts/experiments/oasis_alpha_sweep/run_oasis_bounded_composed.py launch \
  --artifact-root "$ARTIFACT_ROOT" --sitreg-root "$SITREG_ROOT" --gpu 3 \
  --case-rank 0 --stage-alpha 1 1 5 1 --bound-multiplier 0.99 \
  --increments-per-stage 1
```

Accepted outputs are immutable; rerunning the launch command returns without
overwriting them. Failed attempts require audit rather than automatic restart.
