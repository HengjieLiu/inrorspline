# OASIS MR-SINR 8/4/2 Pilot Summary

- Complete: `True`.
- Technical pass: `True`.
- Promote composed candidate: `False`.
- Decision rule: Promote only if both runs pass diagnostics and composed is not clearly worse: Dice >= requery-0.005, HD95 <= requery+0.10, logJ std <= requery+0.05, and J<=0 pct <= requery+0.5.

## Method Rows

| Method | Dice | HD95 | ASD | NCC | BE | Diffusion | logJ std fg | J<=0 pct fg | Runtime s | Diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mr_sinr842_composed | 0.7182 | 55.6173 | 9.8988 | 0.9743 | 0.126112 | 0.210154 | 0.9488 | 1.3168 | 340.1 | final_composed_vs_sequential_image_max_abs=0.884570 |
| mr_sinr842_requery | 0.7459 | 2.7300 | 0.6218 | 0.9897 | 0.104947 | 0.190986 | 1.0057 | 5.9520 | 322.8 | max_requery_transition_jump_mag=0.057730 |

## Delta

- Composed - requery Dice: `-0.0277`.
- Composed - requery HD95: `52.8873`.
- Composed - requery foreground log-J std: `-0.0569`.
- Composed - requery foreground J<=0 pct: `-4.6352`.
