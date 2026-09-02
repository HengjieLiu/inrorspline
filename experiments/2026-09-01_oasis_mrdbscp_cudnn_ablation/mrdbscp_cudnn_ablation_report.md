# MR-D-BSCP cuDNN Cold/Warm Ablation

- Candidate: MR_A_fast_decay; protocol: seed-3 shuffled OASIS ranks 0-4.
- GPU: physical GPU 2; four configurations were run sequentially.
- Timing endpoint: loaded images resident on the selected device through saved dense DVF .npy; evaluation is excluded.
- Cold/warm convention: rank 0 is cold; ranks 1-4 are warm cases in the same worker.

## Timing summary

| Configuration | Cold (s) | Warm mean (s) | Warm SD (s) | Gap (s) | setup.device cold/warm (s) | Expand gap (s) | Backward gap (s) | Dice mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benchmark=True, deterministic=False | 95.94 | 10.88 | 1.32 | 85.06 | 25.11/0.00 | 43.66 | 26.53 | 0.7839 |
| benchmark=False, deterministic=False | 23.68 | 10.25 | 0.34 | 13.43 | 26.19/0.00 | 0.43 | 1.03 | 0.7839 |
| benchmark=True, deterministic=True | 119.32 | 10.61 | 0.41 | 108.71 | 24.01/0.00 | 63.70 | 27.91 | 0.7839 |
| benchmark=False, deterministic=True | 22.41 | 10.55 | 0.58 | 11.86 | 23.02/0.00 | 0.71 | 0.50 | 0.7839 |

## Detailed timing breakdown

| Configuration | Loss-volume setup gap (s) | Regularizer gap (s) | Warp/NCC gap (s) | Logging gap (s) | Training residual gap (s) | Optimizer steps cold/warm | Peak memory cold/warm (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| benchmark=True, deterministic=False | 8.68 | 3.65 | 0.41 | -0.43 | 0.83 | 450.0/437.5 | 3.86/1.94 |
| benchmark=False, deterministic=False | 5.07 | 3.21 | 0.03 | -0.10 | 1.63 | 450.0/437.5 | 1.93/1.93 |
| benchmark=True, deterministic=True | 7.25 | 8.47 | 0.28 | 0.00 | 1.08 | 450.0/437.5 | 3.86/1.94 |
| benchmark=False, deterministic=True | 4.77 | 3.72 | 0.01 | -0.19 | 0.98 | 450.0/437.5 | 1.93/1.93 |

## Interpretation

All gap values are cold minus the mean of warm cases. The expand phase is dense B-spline expansion; backward_step includes zeroing gradients, autograd backward, and the optimizer update.
The setup.device block is reported separately because it occurs before the loaded-images timing endpoint. DVF output preparation and writing are also reported separately.
- Benchmark effect with deterministic=False: the cold/warm gap changed from 85.06 s to 13.43 s (84.2% reduction when benchmark was disabled).
- Deterministic effect with benchmark=True: changing deterministic=False to True changed the gap by +23.65 s.
- Deterministic effect with benchmark=False: changing deterministic=False to True changed the gap by -1.58 s.

## Environment

- PyTorch: 2.6.0+cu118; CUDA runtime: 11.8; cuDNN: 90100.
- GPU query: 2, NVIDIA RTX 6000 Ada Generation, 570.86.15, 5, 49140.
- Git commit: 8302f7b9d2441cb044bbb8c02d7e9622736cf8a7; dirty tree at launch: True.

## Dice evaluation

Dice was evaluated after timing aggregation from the 20 saved DVFs. Evaluation time is not included in the timing table.
