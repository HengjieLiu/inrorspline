# P70 technical log

## Scope

Fresh 4×4 factorial over seed-3 shuffled OASIS ranks0–9:

- Pyramid schedules:8;8→4;8→4→2;8→4→2→1.
- Update styles: unbounded one map; unbounded two-map composition; bounded
  one map; bounded two-map composition.
- Every active diffusion λ=1. Every cell starts from identity. Each map in a
  two-map arm gets the full stage schedule and a fresh Adam optimizer.
- P69 is read-only and may only provide provenance, VFA/SITReg references, and
  a separate reproducibility diagnostic. It does not replace any new cell.

## Preflight and implementation

- Added a separate P70 factorial runner and independent auditor. P68/P69
  numerical sources and accepted outputs remain unchanged. Two initial
  launcher attempts (artifact keys `_v1` and `_v2`) each started only the rank1
  unbounded-one-map smoke. The four one-update stages and native-loss parity
  passed; the launcher then treated the comma character in `0,1,2,3` as a GPU
  ID because `--gpus` had not been split into a list. Both smoke outputs and
  logs are preserved; no production cell started. The v2 retry logic alone did
  not fix that parser error.
- The v1 smoke was validated: Dice0.756410, HD952.080208mm, ASD0.536346mm;
  peak CUDA allocation2.703GiB, four Adam updates, and exact native NCC9/
  diffusion value and gradient parity. The v2 smoke also completed. Smoke
  accuracy is not a production result.
- Fixed GPU-list parsing and assigned the clean attempt artifact key `_v3`;
  this preserves both partial manifests and avoids rewriting their source
  fingerprints.
- Case-centric placement: one physical GPU per rank; all16 settings for that
  rank execute sequentially on the assigned device. User approved GPUs0–3.
- Container UID:GID is fixed at1009:1020. GPU admission is rechecked at each
  worker launch; no existing process is stopped, reprioritized, or modified.
- Acceptance requires four native-grid one-update smokes,160 fresh production
  cells,160 independent replays, exact canonical label/metric checks, and
  reported cross-depth shared-prefix reproducibility diagnostics.

## Execution record

## Execution and acceptance

- Clean v3 execution used GPUs0–3, UID:GID1009:1020, and left existing jobs
  untouched. The four native-grid one-update smokes passed. All160 fresh
  registrations completed with completion markers and finite DVFs; all160
  independent checkpoint/map replays passed. The run progress/failure record
  contains zero failures. GPU memory remained adequate.
- The original integrated report gate stopped after replay because it required
  byte-level-equivalent prefixes at every pyramid depth. Inspection found cps8
  and cps4 prefixes byte-identical, while cps2 has non-bitwise variation
  between independently restarted depths. A separate artifact-only aggregator
  was added; it preserves the original runner/auditor hashes and all numerical
  outputs, validates every saved run again, and emits cps2 differences as
  diagnostics rather than selecting or dropping cells.
- Across360 adjacent-depth/map prefix comparisons, cps8/cps4 were exact in
  300/300. The60 cps2 comparisons had max RMS difference0.04335 voxel, maximum
  pointwise difference6.3676 voxels, and maximum objective-component delta
  3.50e-5. Exact best epochs matched34/60; actual update counts matched56/60
  (largest difference50 epochs). Thus the global RMS is small, but localized
  differences can be several voxels and should not be described as identical.
- The first report-generation attempt also exposed a reference-row accounting
  bug in the original report code. The post-hoc aggregator writes references
  once per case/depth and retains both paired VFA and SITReg comparisons.
- Accepted report:
  `/database/inrorspline/repo_experiments/oasis_seed3_first10_mrdbscp8421_bspline_depth_bound_steps_lambda1_v3/`
  with four Markdown depth tables, per-case CSV, paired-difference CSV,
  stage diagnostics, shared-prefix diagnostics, JSON results, and verified
  output hashes. Counts:24 summary rows,240 per-case rows,320 paired reference
  differences,600 stage rows,360 prefix checks, and10 P69 reproduction
  diagnostics.
- Full-depth means (rank0–9 exploratory): unbounded one-map Dice/HD95/ASD
  0.834680/1.735355/0.334958, SD(logJ)/folding 0.789931/2.207595%; unbounded
  two-map 0.833919/1.726313/0.335452, 0.793823/2.174567%; bounded one-map
  0.809085/1.886869/0.405835, 0.536066/0%; bounded two-map
  0.828813/1.701463/0.339701, 0.650939/0.003094%. VFA was
  0.819772/1.774341/0.366851, 0.620839/0.218560%; SITReg was
  0.821546/1.715898/0.350896, 0.569473/0.000423%.
- Interpretation: unbounded one/two-map arms have high accuracy but much more
  folding and log-Jacobian spread. Bounded two-map has the best full-depth
  HD95 among these six rows and very low folding, while bounded one-map trades
  away more accuracy. See the four schedule tables for all depths and paired
  per-case comparisons.
