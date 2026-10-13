# FeDDA squared global NCC planning log

## Known limitations

- Equal lambda does not calibrate similarity gradients or achieved regularity.
- This is a single-seed study on a previously inspected test cohort, not a
  new unbiased benchmark. Across-case SD is not training reproducibility.
- Historical LNCC lambda 1 is not a fresh matched-seed control.
- The native CUDA backend can vary across processes. Keep explicit historical
  flags, fixed parity tolerances, and failed-preflight evidence.
- Training diffusion is full-volume and unmasked; foreground diffusion is a
  posthoc restricted diagnostic. std(log J) and folding measure different
  properties, and zero measured folding does not establish invertibility.
- Current GPU headroom and squared-GNCC memory consumption are not measured
  by this planning task. Runtime remains diagnostic under shared workloads.

## 2026-10-12 documentation and planning

The user requested a durable explanation of the three NCC losses and a plan
for four FeDDA squared-GNCC trainings followed by a three-loss 2x3 ARC plot.
Only documentation was added. No implementation, GPU smoke, training, or
evaluation was launched, and no historical source or artifact was modified.

The read-only P57 `ACCEPTED.json` records four signed-GNCC trainings, 400 new
case evaluations, 14,000 new label records, and 500 LNCC reference cases.
Its `results.json` expected SHA256 is
`77e446099ad479e6cab1d550ca77eb564168543ab31542390384cab88402e434`.
The new experiment must revalidate the accepted references before use; this
planning check is not a new full-cohort numerical certification.

Planning base revision: `880c9898184f95057bfbb3968ade03d121031d93`.
The worktree contains pre-existing uncommitted implementation and documentation;
the revision alone does not identify all audited source bytes.

| Audited source | SHA256 |
| --- | --- |
| `scripts/experiments/oasis_alpha_sweep/fedda_gncc_training.py` | `884a78d153fd52b893cd4c783761a1e47486275cf6200ed37ac895e188f54b76` |
| `scripts/experiments/oasis_alpha_sweep/fedda_training.py` | `fd501562e4418cabbdb9662a77b9ebc401875a23a20703522caeb15b79877126` |
| `scripts/experiments/oasis_alpha_sweep/sitreg_gncc_training.py` | `8e6955d5acacaed81ddeb5b4afcca6ba6e66b7542ac0a9067137a10b5338f15c` |
| `scripts/experiments/oasis_alpha_sweep/gncc_dense.py` | `006af4711892bb3f45ecd574e2bb4f5070ce85b60267c69e5804fa2cc0bba62a` |
| `${WAVEREG_ROOT}/code/utils_losses_ncc.py` | `1b1de653079ace929c72a3496cc37f91fdeeb230a109c87ea54526244a727f88` |

## 2026-10-12 implementation approval

The user approved implementation and execution. Added separate squared-GNCC
adapter, training/evaluation entry points, four-GPU controller, three-loss
reporter, and tests; no historical numerical source was changed. The loss
squares per-image population correlation before batch averaging and logs
detached signed-correlation diagnostics for each directional update.

The first CPU suite caught a leftover signed-GNCC run-name prefix; it was
corrected before any GPU or production run. The expanded preflight restores
and continues all four saved Adam states, in addition to the historical
reference-parity, fresh-update, warm-memory, trained-update, and metric gates.

Live initial GPU free memory was approximately 38.6/47.4/47.4/47.4 GiB on
GPUs 0/1/2/3. This is a launch-session observation, not a reservation. The
existing container image ID is
`sha256:dab700c415830dcf07c16ef7f2901deed95bb104eda388514c3ce74a43354ba8`.
GPU 3 preflight must pass before production workers start.

## 2026-10-12 preflight passed

The 31-test combined FeDDA suite passed after the naming correction; a further
full-report synthetic-fixture test passed with the other 11 new-loss tests.
All 41 accepted-reference files and 900 existing GNCC/LNCC case records passed
read-only validation. Historical numerical-source hashes remain unchanged.

GPU 3 preflight passed all original tolerances on the first GPU attempt:
reference DVF maximum difference 0.0186961 voxel and RMS 0.000160294 voxel;
Dice difference 0.000037441, HD95 difference 0, symmetric-ASD difference
0.0000802171 mm. Native validation Dice was 0.8141065024 versus historical
0.8141095882. This is numerical parity under the explicit historical backend,
not bitwise equality. No gates or tolerances were relaxed.

All four fresh two-direction update/checkpoint smokes, all four restored Adam
continuations to step 4, warm-state step 6, trained-checkpoint gradients, and
rank 0/10 serialized-field metrics with 35 labels each passed. Measured peak
reserved memory was 21.2168 GiB; the validated production gate is 24 GiB.
Evidence: `${ARTIFACT_ROOT}/runs/oasis_fedda_gncc2_training_preflight_v1/`.

## 2026-10-12 production launch

Launched at 17:05:11 PDT (2026-10-13 00:05:11 UTC), controller PID 96452.
Workers: GPU 0 lambda 0.1 PID 96460; GPU 1 lambda 1/3 PID 96465;
GPU 2 lambda 1 PID 96468; GPU 3 lambda 3 PID 96471. The detached controller
owns sequential training-to-first100 evaluation dependencies per GPU, then
the three-loss report after all four candidates succeed. No unrelated GPU
process was stopped, reprioritized, or changed.

The six existing foreground-diffusion/ARC tests also passed. Total coverage is
32 FeDDA tests plus six foreground metric/plot regressions. Historical signed
GNCC elapsed training times were 8.06/5.47/5.54/6.28 hours, with detailed
evaluation taking 20.6-26.1 minutes. A provisional 6-9 hours for the new sweep
including evaluation is based on those observations, not a guaranteed deadline.

Startup verification confirmed all four workers actively training in epoch 1
with finite losses and no failure markers, effective `1-rho^2` metadata,
UID:GID 1009:1020, and identical initial-state hash
`3d53895f2ddc0fcc339870c7635eac2328a567c6dbb78eaf5985a7c90729d6e4`.
At verification they had completed 42/82/62/82 Adam updates on GPUs 0/1/2/3.
