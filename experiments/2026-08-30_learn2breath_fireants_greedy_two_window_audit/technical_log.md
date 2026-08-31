# Technical log

## Protocol limitations recorded before launch

- The audit is CT-only. COPD landmarks, R231 masks, and validation lobe masks
  are absent from FireANTs registration inputs and loss construction. They are
  used only by geometry gates, evaluation, and visual QC.
- `p2e_equiv_hu` is intentionally a historical-P2e physical-HU adaptation,
  not a Learn2Breath challenge preprocessing requirement. `l2b_lung_hu` is
  also reported only; neither profile is a selected winner.
- FireANTs materializes a native field, then selects the public field
  sign/axis candidate solely by agreement between FireANTs' native normalized
  warped CT and the challenge-sampler warped normalized CT. No semantic asset
  participates in candidate selection.
- COPD R231 displays use the challenge sampler for FireANTs and historical
  P2e sampling for P2e. Validation uses the challenge sampler for both methods
  so lobe visuals match Dice.
- P2e results were previously produced in a CUDA 11.8 development container;
  FireANTs is executed in a separately recorded CUDA 12.1 runtime. Runtime tables
  retain environments and are not a direct method-runtime ranking.
- Results are an audit only: no challenge package, submission, or
  method/window selection is generated.

## 2026-08-30 implementation and preflight record

- The adapter builds FireANTs' constant float32 geometry cache on CPU and
  transfers it with the normalized CT tensor to CUDA. In the active container,
  GPU-side `torch.inverse` hangs during the otherwise constant image-cache
  setup. This construction-only compatibility workaround leaves the CT values,
  geometry, FireANTs loss, optimizer, deformation type, scale/iteration
  schedule, and field contract unchanged.
- Exact-schedule smoke gates passed: one COPD smoke produced a finite
  challenge-compatible field, physical TRE, and whole-grid folding; one
  validation smoke produced all five finite lobe Dice values and zero
  whole-grid/fixed-lobe folding. Both exited cleanly after removing a
  post-summary CUDA cache-cleanup call that could hang in this environment.
- The immutable production manifest contains all 40 requested jobs and all 40
  non-optimizing preflight rows passed. The original
  `learn2breath26-fireants:e7a613c-cu121` image hangs on physical GPU 0 during
  a basic full-grid CUDA allocation, including in a fresh container, although
  the same GPU allocates correctly in `fireants-devhj:v1_0`. A recovery smoke
  ran the *same current FireANTs source tree* mounted read-only at `/fireants`
  on that CUDA-12.1 base image: it completed case 104 with a finite field,
  `0%` whole-grid folding, and mean TRE `10.5510 mm` versus baseline
`26.3293 mm`. Independent checks show that this recovery base has separate
single-GPU hangs on GPUs 1--3, while the original image passed its prior
smokes on those devices. The relaunch therefore uses the recovery runtime only
on GPU 0 (`fireants_current_source_on_devhj_v1_cuda12.1`) and the original
runtime on GPUs 1--3 (`learn2breath26_fireants_e7a613c_cuda12.1`). Every case
summary records its exact runtime. This preserves the selected method source
and protocol but introduces a base-environment asymmetry (notably SimpleITK),
which is retained as a protocol caveat and prevents cross-runtime timing
interpretation.
