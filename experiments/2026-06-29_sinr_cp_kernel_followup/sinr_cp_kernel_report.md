# SINR CP-Kernel Follow-Up Report

Status: `completed`

This run is a sampled dense-update proxy for the CP-kernel diagnostic. It reuses the deterministic checkpoint samples from the update-alignment run and compares SINR cumulative updates against the paired single-stage DirectCP cumulative updates.

## Metrics

| Dataset | Case | Checkpoint | D_U proxy | R_U proxy | alpha | A_U proxy |
|---|---|---|---:|---:|---:|---:|
| lungct | lungct_case01 | init_step | 0.2484 | 0.9687 | 20.24 | 81.5016 |
| lungct | lungct_case01 | epoch50 | -0.0678 | 0.9977 | -0.1528 | 2.2523 |
| lungct | lungct_case01 | epoch250 | 0.0009 | 1.0000 | 0.001516 | 1.7512 |
| lungct | lungct_case08 | init_step | 0.0052 | 1.0000 | 0.4857 | 93.5016 |
| lungct | lungct_case08 | epoch50 | 0.1600 | 0.9871 | 0.4513 | 2.8202 |
| lungct | lungct_case08 | epoch250 | 0.2794 | 0.9602 | 0.5261 | 1.8829 |
| oasis | oasis_rank009 | init_step | -0.0479 | 0.9989 | -4.234 | 88.3574 |
| oasis | oasis_rank009 | epoch50 | 0.3268 | 0.9451 | 0.9779 | 2.9923 |
| oasis | oasis_rank009 | epoch250 | 0.3876 | 0.9218 | 0.8998 | 2.3218 |

## Notes

- `D_U proxy` is cosine similarity between sampled SINR and DirectCP cumulative dense updates.
- `R_U proxy` is the relative residual after fitting one scalar from DirectCP to SINR.
- `A_U proxy` is the sampled dense-update norm ratio, SINR over DirectCP.
- This does not estimate `K = J_theta J_theta^T`; it is the overnight-safe continuation artifact that identifies whether the full autograd kernel follow-up is worth running.
