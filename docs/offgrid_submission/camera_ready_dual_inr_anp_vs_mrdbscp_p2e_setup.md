# Camera-ready setup note: A/N/P DUAL-INR arms vs MR-D-BSCP P2e

This note explains the protocol and model differences between the DUAL-INR
`A/N/P` `0/1` arms and the COPD `MR-D-BSCP P2e` reference. It is a setup
companion for the camera-ready evidence, not a new result claim or a replacement
for the numeric readouts.

## Source readouts

- [COPD DUAL-INR six-arm readout](../../experiments/2026-08-03_dual_inr_copd_six_arm/camera_ready_readout.md)
- [DIR-LAB 4DCT controlled DUAL-INR readout](../../experiments/2026-08-02_dual_inr_dirlab_controlled_multires/camera_ready_readout.md)
- [DIR-LAB 4DCT author A0/A1 readout](../../experiments/2026-08-02_dual_inr_dirlab_author_repro/camera_ready_readout.md)
- [COPD P2e execution spec](../../experiments/2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity/codex_execution_spec.md)
- [COPD P2e review report](../../experiments/2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity/review_report.md)

## At a glance

| Arm/reference | Dataset/use | Preprocessing | Model | Loss | Schedule | Optimizer/LR | Knot insertion |
|---|---|---|---|---|---|---|---|
| `A0` | 4DCT and COPD author-style reproduction | Released DUAL native author crop; TotalSegmentator five-lobe union; fixed-lobe sampling | DUAL-INR/SIREN | NCC + spatial regularization + semantic lobe loss | Author adaptive blur `[4,2,0]`, 3000 steps | AdamW, warmup-cosine `1e-3 -> 1e-5`; scheduler reset on blur transitions | No |
| `A1` | A0 semantic-loss ablation | Same A0 data, crop, sampling, seed, and paired transition trace | DUAL-INR/SIREN | NCC + spatial regularization only; semantic weight `0` | Replays paired A0 realized transitions | AdamW, warmup-cosine `1e-3 -> 1e-5`; scheduler reset on blur transitions | No |
| `N0` | Controlled native full-volume baseline | Native full volume; fixed lung-mask sampling | DUAL-INR/SIREN | NCC + spatial regularization | Full-resolution control `[1,1,1,1]`; steps `400,400,800,900` | AdamW, warmup-cosine `1e-3 -> 1e-5`; reset per stage | No |
| `N1` | Controlled native multiresolution probe | Same as N0 | DUAL-INR/SIREN | NCC + spatial regularization | MR-D-BSCP-shaped image pyramid `[8,4,2,1]`; steps `400,400,800,900` | AdamW, warmup-cosine `1e-3 -> 1e-5`; reset per stage | No |
| `P0` | Controlled pTV-style 1 mm baseline | pTV-style 1 mm landmark crop; no lung mask in optimization; 5-voxel border excluded | DUAL-INR/SIREN | NCC + spatial regularization | Full-resolution control `[1,1,1,1]`; steps `400,400,800,900` | AdamW, warmup-cosine `1e-3 -> 1e-5`; reset per stage | No |
| `P1` | Controlled pTV-style 1 mm multiresolution probe | Same as P0 | DUAL-INR/SIREN | NCC + spatial regularization | MR-D-BSCP-shaped image pyramid `[8,4,2,1]`; steps `400,400,800,900` | AdamW, warmup-cosine `1e-3 -> 1e-5`; reset per stage | No |
| `MR-D-BSCP P2e` | COPD 1 mm contextual reference | Legacy pTV-style 1 mm landmark crop; no lung mask in optimization; 5-voxel border excluded | DirectCP cubic B-spline field, no INR | pTV-style local CC + isoTV | cps `64,32,16,8,4`; downsample `16,8,4,2,1`; epochs `400,400,400,400,900` | Adam, LR `1e-3`, seed `1` | Yes, exact crop-aligned dyadic cubic B-spline knot insertion |

## Naming

- `A` means the released author-style DUAL-INR reproduction family.
- `N` means the controlled native full-volume family.
- `P` means the controlled pTV-style 1 mm preprocessing family.
- For `A`, `0/1` separates the released semantic-loss objective from the no-semantic-loss ablation.
- For `N/P`, `0/1` separates full-resolution training from the MR-D-BSCP-shaped image-pyramid schedule.
- `P2e` is an internal COPD reference nickname, not a method name from the literature. In paper-facing text, spell it out as: `MR-D-BSCP, pTV-style 1 mm crop, cps 64-32-16-8-4, downsample 16-8-4-2-1, epochs 400-400-400-400-900`.

## DUAL-INR model and controlled N/P schedule

All `A/N/P` arms use the DUAL-INR/SIREN model. The controlled `N/P` `0/1` suffix does not mean INR versus non-INR.

The controlled `N/P` model is one persistent DUAL SIREN trained across stages:

- Per branch architecture: `3 -> 256 -> 256 -> 3`.
- Coarse branch: omega `10`, scale `0.1`.
- Fine branch: omega `30`, scale `0.05`, activated at global step `400`.
- Before fine activation, `u(x) = u_coarse(x)`.
- After fine activation, `u(x) = u_coarse(x) + u_fine(x)`.
- Model parameters persist across stages; AdamW and the warmup-cosine scheduler are reset at each stage.

For `N0/P0`, the image factors are `[1,1,1,1]`. For `N1/P1`, the image factors are `[8,4,2,1]`. These are image-pyramid stages only. There is no B-spline control-point schedule, no stage-wise deformation composition, and no knot insertion in `N1/P1`.

## A0/A1 author setup

`A0` uses the pinned released DUAL-INR-DIR behavior: native author-format data, the released author crop and sampling conventions, sampled NCC, spatial regularization, and the semantic lobe loss. The author schedule is the released adaptive blur schedule `[4,2,0]` over 3000 steps, optimized with AdamW and warmup-cosine LR `1e-3 -> 1e-5` with warmup `20`. The scheduler resets on blur transitions.

`A1` is paired to `A0`: it keeps the same data, crop, sampling, model, seed, and realized transition trace, but sets the semantic lobe-loss weight to zero. This isolates the semantic-loss contribution under the author-code protocol.

## MR-D-BSCP P2e setup

`MR-D-BSCP P2e` is the completed COPD DirectCP reference selected as a whole cohort, not per case.

- Dataset: DIR-Lab COPD cases `01-10`.
- Direction: moving `eBH` to fixed `iBH`.
- Preprocessing: legacy pTV-style 1 mm landmark crop.
- Optimization domain: no lung mask in optimization; 5-voxel border excluded.
- Model: direct cubic B-spline control-point displacement field, no INR.
- Full-resolution cps schedule: `64,32,16,8,4`.
- Image/downsample pyramid: `16,8,4,2,1`.
- Stage epochs: `400,400,400,400,900`; total `2500`.
- Optimizer: Adam with LR `1e-3` at every stage; seed `1`.
- Loss: pTV-style local cross-correlation plus isoTV.
- LCC sigma: `2.1 mm`; isoTV weight: `0.11`; isoTV `csqrt`: `5e-3`.
- Prolongation: exact crop-aligned cubic B-spline dyadic knot insertion between stages.
- Final field: the final cps-4 B-spline field expanded on the full 1 mm crop, not a composition of five separate deformation fields.

The paired `full_cps/downsample` values make the effective cps on each stage image approximately constant:

| Stage | Full cps | Downsample | Effective stage cps | Epochs |
|---:|---:|---:|---:|---:|
| 0 | 64 | 16 | 4 | 400 |
| 1 | 32 | 8 | 4 | 400 |
| 2 | 16 | 4 | 4 | 400 |
| 3 | 8 | 2 | 4 | 400 |
| 4 | 4 | 1 | 4 | 900 |

## Interpretation limits

- `N1/P1` are MR-D-BSCP-shaped image-pyramid DUAL-INR baselines, not MR-D-BSCP implementations.
- `MR-D-BSCP P2e` is a non-INR DirectCP cubic B-spline reference with explicit knot insertion.
- `N0` versus `N1` and `P0` versus `P1` isolate the image-pyramid schedule within a preprocessing family.
- `A`, `N`, `P`, pTVReg, and MR-D-BSCP rows differ in crop, resampling, sampling domain, loss, and evaluation convention. Cross-preprocessing rows must remain protocol-tagged and should not be presented as pure model-only comparisons.
- The August 2 DIR-LAB 4DCT readouts use their own pTVReg and MR-D-BSCP references. The `P2e` nickname specifically refers to the COPD five-stage reference described here.
