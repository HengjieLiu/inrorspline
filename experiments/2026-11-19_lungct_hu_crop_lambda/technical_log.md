# Technical log

## Known limitations

- Best lambda is selected using the same300 landmarks whose TRE is reported.
- Two increases is a local heuristic; capped searches do not prove optimality.
- Bone/soft-tissue automatic labels require coverage/QC and are not manual truth.
- Full-FOV84M-voxel cases may exceed available memory. Failures remain visible;
  no silent resolution, objective, iteration or crop change is permitted.
- Other GPU jobs may share the device; wall times are not isolated benchmarks.
- Historical fields retain earlier backend/crop protocols; fresh centers are used.
- The requested cuDNN deterministic setting does not make all CUDA operations
  bitwise reproducible. Repeated original-CLI smokes also differed; this campaign
  does not claim exact field reproducibility across reruns.

## Execution

- Approved protocol/registry recorded before preparation and registration.
  First4DCT01/08,COPD01/08, then automatic remainder. Source implementation,
  validation and memory profiling follow before production release.

## Preflight results

- Eighteen targeted tests passed: common HU scaling, exact H0, physical mask
  margins, both sweep directions, ties/failures/caps, known native translations,
  anatomy metrics and the automatic first-batch transition/report logic.
- All nine preparations for each first-batch case passed, including exact H0/C0
  image parity against the accepted corrected campaign and300-point round trips.
- Reduced MR smokes use two iterations per stage, not scientific results. An
  initial strict bitwise adapter/CLI check failed and is retained externally.
  Repeating the original CLI on the same device also changed its field:
  maximum0.08390 nominal voxels, mean absolute difference0.00002098 voxels.
  Adapter versus original maximum was0.11353 and mean0.00003339 voxels;
  99th percentile0.00009020 voxels. All three native snap mean TREs were exactly
  3.8033387455547123mm; subvoxel mean differences were below0.000001mm.
  Solver training dictionaries match, and the numerical implementation is
  unmodified. Retain this as a reproducibility limitation, not a bitwise pass.
- One preflight shell redirection failed before the reference command launched;
  its parent directory was created and the reference launched successfully.
- CUDA/dependencies verified in a fresh pinned-image runtime. Projected complete
  scientific fields are2.25TiB maximum across this cohort; a conservative
  largest-case envelope is4.00TiB, against30.79TiB currently free. Preparation,
  annotations and metadata have additional headroom. Existing jobs are retained.

## Production release

- Frozen85 source files and the configuration before launch; hashes and runtime
  inspections retained externally. Source mounts are read-only; raw data remain
  read-only. Separate pinned, restartable services run registration, independent
  CPU anatomy inference and CPU posthoc evaluation. Previous containers remain.
- The first memory profiles passed, including COPD01/08 full FOV. Production
  center candidates started on all four GPUs, subject to memory admission.
  First-batch checkpoint/report and automatic continuation are persisted in the
  controller. Scientific comparisons await completed sweeps.

## Completion2026-11-20

- First-batch report sealed at10:45UTC; remaining16 cases released automatically
  after preparation at10:51UTC. Registration finished at18:02UTC and anatomy
  supplement at18:03UTC, about9h48 after campaign startup.
- All180 searches terminal:1245 accepted, one retained CUDA OOM atCOPD06 H1/C0
  k=-2. Other processes grew on the shared GPU after admission; no automatic
  retry or parameter change. Its branch continued with a reset comparison anchor.
- All60 geometry memory profiles passed. Three lower branches reached-8:
  4DCT08 H1/C1,COPD07 H1/C2,COPD10 H1/C1. Remaining357 branches stopped after
  two successive increases. Every case/preparation has accepted center and best
  values; all fixed/tuned cohort TRE denominators are10.
- Forty automatic anatomy volumes and1245 anatomy metric records completed.
  Geometry checks passed; visual label QC remains pending and the metrics retain
  their provisional automatic-label tag. Sealed registration reports are retained
  alongside separate anatomy supplements. No historical results were changed.
- Post-run audit source added separately from the immutable executable snapshot;
  it replays all native landmark endpoints and verifies metadata/field headers,
  full optimization histories, anatomy receipts and sealed report hashes. Dense
  fields were already scanned, hashed and exactly reloaded by their workers;
  the post-run audit does not rehash terabytes of fields.

## Completion replay and anatomy tradeoff summary

- Independent audit passed1245 accepted candidates/373500 landmark rows with
  exactly zero native snap/subvoxel replay error; source/configuration and
  sealed reports unchanged. All optimization histories and field headers passed.
- Added posthoc summary source, no new registrations/segmentations. All360
  center/TRE-selected case rows match sealed tables; CSVs retain sample SDs,
  counts, paired changes and label/coverage limitations. Figure inspected.
- External analysisv2 corrects the beneficial direction of source-inside/coverage
  diagnostic fractions in v1. Raw values and all TRE/Dice summaries are unchanged;
  v1 remains retained. Deformation folding metrics keep their original domain
  and are not relabelled as common-ROI anatomy metrics.
- Two representative central-slice QC overlays inspected (4DCT01 fixed,
  COPD08 moving); full phase/structure/slice QC is still pending. This limited
  inspection is not a complete validation of the automatic anatomy labels.
- Model settings confirmed in pinned runtime: TotalSegmentator2.16.0 `total`,
  non-fast1.5mm, independent full-native raw-HU inputs; lungs reuse paired R231
  masks. Common anatomy region includes on average72.8%/76.2% of segmented
  fixed bone in4DCT/COPD. No whole-thorax or per-bone rigidity conclusion follows.

## Bone-component reporting extension

- User requested separate ribs, vertebrae and sternum Dice while retaining
  combined bone. Reused the accepted per-candidate group metrics; no new DIR,
  segmentation, parameter selection or field changes. Each group is a union
  of its visible labels, not individual rib/vertebral-level correspondence.
- Extended source reporting with fixed/TRE-selected casewise component values,
  cohort means/sample SD/counts, correct mixed metric units and measured union
  Dice retained independently. Added component CSVs, paired HU/crop changes,
  C2-minus-C1 comparisons and eight-panel component tradeoff figure.
- Generated `anatomy_components_v1/` and `hu_crop_anatomy_tradeoff_v3/` externally.
  Older sealed reports/sources/configuration retained unchanged. Visual QC of
  automatic labels remains provisional; expanded plot inspected.
- Nine targeted tests passed, including independently retained union Dice and
  explicit missing-sternum denominator. Data validation checked all4980 Dice
  values across1245 accepted registrations, all360 selected rows and144 component
  cohort endpoints. Previous selected/aggregate CSVs are byte-identical and
  previous sealed report hashes match. No staging or commits.
