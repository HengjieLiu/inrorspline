# Technical Log

## Known limitations

- This is a five-case diagnostic, not a replacement for the paper's OASIS-100
  accuracy result.
- Cold rank 0 and warm ranks 1-4 are different image pairs, so the observed
  difference combines process warmup with case-to-case variation.
- The phase timers localize first-use cost to operation categories but do not
  identify individual CUDA library kernels without profiling.
- The repository's MR-D-BSCP runner has no torch.compile call; this ablation
  therefore tests the cuDNN flags first.
- GPU 2 must be idle before the real launch. No unrelated process will be
  stopped automatically.

## Planned provenance

- Run ID: oasis_mrdbscp_cudnn_ablation_20260901_v1.
- Container: py310sitreg_v1_test1.
- Image: hengjieliu96/dl_dir:py310sitreg_v1.
- Raw output root:
  `/workspace/scratch/runs/oasis_mrdbscp_cudnn_ablation_20260901_v1`.
- Review root:
  experiments/2026-09-01_oasis_mrdbscp_cudnn_ablation/.

## Execution notes

Completed 2026-09-01 after container syntax checks, a four-arm dry-run, GPU-2
idle preflight, four sequential timing workers, timing aggregation, and a
separate saved-DVF Dice pass. GPU 2 was an NVIDIA RTX 6000 Ada Generation;
PyTorch was 2.6.0+cu118, CUDA runtime 11.8, cuDNN 90100. The host commit was
8302f7b9d2441cb044bbb8c02d7e9622736cf8a7 on branch gpu8, with a pre-existing
dirty worktree.

## Timing result

| Configuration | Cold (s) | Warm mean (s) | Gap (s) | Expand gap (s) | Backward gap (s) |
|---|---:|---:|---:|---:|---:|
| benchmark=True, deterministic=False | 95.94 | 10.88 | 85.06 | 43.66 | 26.53 |
| benchmark=False, deterministic=False | 23.68 | 10.25 | 13.43 | 0.43 | 1.03 |
| benchmark=True, deterministic=True | 119.32 | 10.61 | 108.71 | 63.70 | 27.91 |
| benchmark=False, deterministic=True | 22.41 | 10.55 | 11.86 | 0.71 | 0.50 |

The primary benchmark comparison reduced the cold/warm gap from 85.06 s to
13.43 s, an 84.2% reduction. With benchmark=True, deterministic=True
increased the gap by 23.65 s; with benchmark=False it changed the gap by
-1.58 s. In the reference arm, expansion plus backward accounted for 70.19 s
of the 85.06 s gap. The 25.11 s cold setup.device cost was measured separately
before the loaded-images endpoint.

All four arms completed 5/5 cases, with one cold rank-0 case and four warm
cases in one persistent worker. All 20 timing JSON files recorded requested
and effective cuDNN settings. All 20 dense DVFs passed shape (160, 224, 192,
3), float32, finite-value, and normalized-grid XYZ checks. The four Dice means
were identical at 0.7838778 (SD 0.0490037), with evaluation time excluded from
the timing endpoint. The benchmark effect exceeded the 20% profiler trigger,
so no profiler follow-up was launched.

The first controller attempt reached the completed timing aggregation but its
posthoc evaluator omitted the alpha argument when reconstructing run paths.
The evaluator was fixed and rerun from the saved DVFs; its case-table writer
was also corrected so the final artifact contains all 20 rows. No timing run
was repeated.
