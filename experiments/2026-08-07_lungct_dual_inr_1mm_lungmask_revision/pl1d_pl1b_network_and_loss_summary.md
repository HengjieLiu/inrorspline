# PL1D / PL1B Network And Loss Summary

## One-line Summary

PL1D and PL1B both use the same released two-branch DUAL SIREN model.  The
`8,4,2,1` entries are image pyramid resolutions, not four separate SIRENs and
not four network levels.  The same model parameters persist through all stages.

## Network Schedule

Both variants use one DUAL-INR model with:

- a coarse SIREN branch
- a fine SIREN branch
- released DUAL architecture: `layers = [3, 256, 256, 3]`
- `coarse_omega0 = 10`
- `fine_omega0 = 30`
- `coarse_scale = 0.1`
- `fine_scale = 0.05`
- `fine_start_stage = 1`

The deformation is:

```text
u(x) = u_coarse(x)                         before fine branch activation
u(x) = u_coarse(x) + u_fine(x)             after fine branch activation
```

The stage schedule is:

| Stage | Steps | Image factor | Image used for LNCC | Active network branches |
|---:|---:|---:|---|---|
| 0 | 0-399 | 8 | 8x downsampled image | coarse SIREN only |
| 1 | 400-799 | 4 | 4x downsampled image | coarse + fine SIREN |
| 2 | 800-1599 | 2 | 2x downsampled image | coarse + fine SIREN |
| 3 | 1600-2499 | 1 | full 1mm image | coarse + fine SIREN |

Interpretation:

- The coarse branch starts at the coarsest image level, factor 8.
- The fine branch turns on at stage 1, factor 4, because `fine_start_stage = 1`.
- After stage 1 starts, both branches are optimized together on factor 4, then
  factor 2, then factor 1.
- The optimizer and scheduler are reset at each stage, but the model weights are
  not reset.
- No new SIREN is created for any stage.

## Relation To Original DUAL-INR

The released/original DUAL-INR training schedule uses three blur stages:

```text
blur_sigmas = [4, 2, 0]
```

The PL1D/PL1B controlled lung-mask revision instead uses four explicit image
pyramid stages:

```text
image_factors = [8, 4, 2, 1]
stage_lengths = [400, 400, 800, 900]
```

So PL1D/PL1B keep the released two-branch DUAL SIREN architecture, but replace
the original three-level blur schedule with the controlled four-level image
pyramid schedule.

## Shared Similarity Term

Both PL1D and PL1B optimize sampled local normalized cross-correlation on the
current image pyramid level.  Let:

- `x` be sampled coordinates from the fixed T00 lung mask domain
- `F_s` be the fixed image at stage `s`
- `M_s` be the moving image at stage `s`
- `u(x)` be the DUAL displacement
- `LNCC` be the repository's DUAL-INR sampled local NCC loss

The shared similarity term is:

```text
L_sim = LNCC(M_s(x + u(x)), F_s(x))
```

Sampling and evaluation domain for PL1D/PL1B:

- protocol: `PL`
- sampling mask: fixed T00 1mm lung mask
- metric mask: fixed T00 1mm lung mask
- sampled points per step: `30000`
- semantic mask loss weight: `0.0`

## PL1D Loss

PL1D uses the DUAL native regularization branch:

```text
L_PL1D = L_sim
       + 0.01 * R_jac(u_coarse)
       + 0.01 * R_lap(u_coarse)
       + 0.1 * [0.01 * R_jac(u_fine) + 0.01 * R_lap(u_fine)]
```

Equivalently:

```text
L_PL1D = L_sim
       + 0.01 * R_jac(u_coarse)
       + 0.01 * R_lap(u_coarse)
       + 0.001 * R_jac(u_fine)
       + 0.001 * R_lap(u_fine)
```

Configured weights:

| Component | Weight |
|---|---:|
| LNCC similarity | 1.0 |
| coarse negative-Jacobian regularizer | 0.01 |
| coarse Laplacian regularizer | 0.01 |
| fine regularizer scale | 0.1 |
| effective fine negative-Jacobian regularizer | 0.001 |
| effective fine Laplacian regularizer | 0.001 |
| semantic mask loss | 0.0 |

The fine-branch regularization contributes only after the fine branch is active.

## PL1B Loss

PL1B uses the IDIR-type bending-energy branch on the total DUAL displacement:

```text
u(x) = u_coarse(x) + u_fine(x)             after fine branch activation
```

The loss is:

```text
L_PL1B = L_sim + 10.0 * BE(u)
```

where:

```text
BE(u) = mean over sampled coordinates and displacement components of
        squared second spatial derivatives of u
```

Configured weights:

| Component | Weight |
|---|---:|
| LNCC similarity | 1.0 |
| IDIR-type bending energy on total DUAL displacement | 10.0 |
| DUAL native negative-Jacobian regularizer | 0.0 |
| DUAL native Laplacian regularizer | 0.0 |
| fine regularizer scale | 0.0 |
| semantic mask loss | 0.0 |

## PL1D vs PL1B

| Variant | Architecture | Image schedule | Regularization branch |
|---|---|---|---|
| PL1D | two-branch DUAL SIREN | `[8,4,2,1]` | DUAL native Laplacian + negative-Jacobian branch regularizers |
| PL1B | two-branch DUAL SIREN | `[8,4,2,1]` | IDIR-type bending energy on total DUAL displacement |

Neither PL1D nor PL1B is an IDIR or SINR architecture.  They do not use a
B-spline control grid, a SINR control-point size, or a different number of
SIRENs.  They use the same two DUAL SIREN branches and differ only in the
regularization loss branch.

## Code Anchors

- Controlled factors and stage lengths:
  `scripts/experiments/lungct_dual_inr_controlled/adapter.py`
- Variant definitions:
  `PL1D -> dual_native_reg`, `PL1B -> idir_type_be`
- Fine branch activation:
  `fine_start_stage = 1`
- Original released DUAL blur schedule:
  `external/DUAL-INR-DIR/src/inrdir/configs.py`
