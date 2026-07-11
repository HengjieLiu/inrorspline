# SINR CP-Kernel Follow-Up Report

Status: `completed`

This run is a sampled dense-update proxy for the CP-kernel diagnostic. It reuses the deterministic checkpoint samples from the update-alignment run and compares SINR cumulative updates against the paired single-stage DirectCP cumulative updates.

## Metrics

| Dataset | Case | Checkpoint | D_U proxy | R_U proxy | alpha | A_U proxy |
|---|---|---|---:|---:|---:|---:|
| lungct | lungct_case01 | epoch1 | 0.2574 | 0.9663 | 20.92 | 81.2689 |
| lungct | lungct_case01 | epoch2 | 0.2121 | 0.9772 | 7.886 | 37.1743 |
| lungct | lungct_case08 | epoch1 | -0.0032 | 1.0000 | -0.303 | 93.4133 |
| lungct | lungct_case08 | epoch2 | -0.0291 | 0.9996 | -1.429 | 49.0993 |
| oasis | oasis_rank009 | epoch1 | -0.0553 | 0.9985 | -4.884 | 88.3539 |
| oasis | oasis_rank009 | epoch2 | -0.0160 | 0.9999 | -0.6759 | 42.3614 |

## Notes

- `D_U proxy` is cosine similarity between sampled SINR and DirectCP cumulative dense updates.
- `R_U proxy` is the relative residual after fitting one scalar from DirectCP to SINR.
- `A_U proxy` is the sampled dense-update norm ratio, SINR over DirectCP.
- This does not estimate `K = J_theta J_theta^T`; it is the overnight-safe continuation artifact that identifies whether the full autograd kernel follow-up is worth running.
