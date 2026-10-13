# P47 execution log

## Limitations

Exploratory rank4 rescue; extra stage adds compute. Pyramid NCC support and
diffusion sampling differ from full resolution. Existing GPU3 workloads remain
active, so runtime is diagnostic. CUDA sampling backward can be nondeterministic.
Best epochs are selected by total loss, never segmentation metrics.

## 2026-10-09 implementation

- User approved two new registrations and explicitly required GPU3 sequentially.
- Audited current numerical hashes against saved P43 rank4 records; geometry,
  objective and original worker hashes match exactly.
- Added isolated protocol/worker/sequential launcher; existing numerical source
  and historical outputs remain unchanged.
- Initial container GPU3 snapshot:18534MiB free,0% instantaneous utilization.
  Query current state again at every launch; do not modify existing jobs.
- All24 CPU tests passed (8 new contracts plus16 existing P43 tests), including
  native NCC/diffusion value/gradient parity. No tests skipped. Runtime9.834s.
- Both sequential GPU3 smokes passed all five stages with35 valid labels.
  Native NCC/diffusion value and gradient differences were exactly zero;
  max knot-insertion difference2.49222e-6 native voxels; dense clones exact.
  Peak allocated CUDA memory1.63784GiB in each smoke. Both fields serialized
  exactly; physicalGPU3/logicalcuda0, FP32, cuDNN F/T and TF32off verified.
- Synthetic temporary report roundtrip also passed:6 comparison rows,
  210 label metric rows and280 paired label deltas.
- Production completed and accepted2026-10-09 21:16:58UTC (14:16:58PDT).
  Pyramid ran21:09:19-21:12:23UTC, full21:12:24-21:16:56UTC. All four
  smoke/production processes exited0, strictly sequential, with no retries.
- Both production runs have35 valid labels, finite native stage fields and
  exact dense initialization. Maximum production knot-insertion difference
  5.69224e-6 native voxels. Initial/final checksums of16 reference files match;
  historical numerical source files remain unchanged.
- Pyramid Dice/HD95/ASD:0.754233/2.728347/0.588810; FG SDlogJ0.828301,
  folding3.692091%. Relative to matched8-start pyramid:Dice-0.001178,
  HD95+0.004001mm, ASD-0.013499mm, folding-0.162180percentage points.
- Full Dice/HD95/ASD:0.755956/2.762280/0.601316; FG SDlogJ0.855863,
  folding3.496063%. Relative to matched8-start full:Dice+0.001619,
  HD95-0.016224mm, ASD-0.009677mm, folding+0.094033percentage points.
- Stopping epochs16/8/4/2/dense:275/125/125/225/500 (pyramid),
  200/225/225/200/525 (full). All spline stages stopped on adaptive plateau;
  dense stopped after two LR halvings. Best epochs:248/118/124/220/464 and
  180/218/197/200/513, respectively.
- Training-only times100.986/191.129s; worker wall times176.372/265.760s.
  Stage-wise full-resolution metrics and serialization contribute overhead.
  These shared-workload timings are diagnostic, not a speed comparison.
- The extra stage does not rescue rank4 under the fixed objective/schedule.
  Left-caudate Dice remains0.456332 (pyramid)/0.453870 (full), versus original
  SITReg0.717170. Full right-ventricle HD95 worsens13.304135-to14.317821mm;
  pyramid right-choroid-plexus HD95 worsens10.770330-to11.916375mm.
- Folding is already3.913655%/3.638515% at cps2 and decreases during dense
  refinement to3.692091%/3.496063%; the dense stage is not the sole source.
- Accepted CSV/JSON/Markdown, all35 label differences and ten native stage
  fields are under the separate approved artifact key. No first10 aggregate
  or additional registration was produced.
