# Implement Leow-Style Symmetric KL Jacobian Regularization with C1 Quadratic Continuation

**Status:** Codex-ready implementation specification / independent research adaptation  
**Fixed initial settings:** `epsilon = 0.05`, `kappa = 120.0`  
**Scope:** differentiable Jacobian *loss*, not a new deformation parameterization, and not a full reimplementation of the 2007 registration solver.

> **重要区分：** Leow et al. (2007) 提出了原始 symmetric-KL Jacobian regularizer。本文档的 **C1 quadratic continuation 是我们针对 `J <= 0` 数值优化问题提出的工程修改，不是原论文的方法**。实验报告必须注明这个区别。

## 0. Codex task: required deliverables

Integrate an optional **Leow-style symmetric KL Jacobian regularizer with C1 quadratic continuation** into the existing registration repository. **Inspect the repository first** to locate: the registration objective, final composed deformation, Jacobian determinant implementation, coordinate conventions, loss registry/config, and existing unit tests. Follow the repository's patterns rather than creating a parallel framework.

Implementation requirements:

1. Add a differentiable, configurable loss with **`epsilon=0.05`, `kappa=120.0`** as defaults; expose its overall weight as a **separate** parameter `lambda_kl` (do not confuse it with `kappa`).
2. Compute the loss from the determinant of the **final composed transform**, `J = det(D phi)`. Reuse a validated Jacobian implementation where possible; **do not compute `det(grad(u))` in place of `det(I + grad(u))`**.
3. The original Leow integrand must be used **exactly whenever `J >= epsilon`**. For `J < epsilon`, use the C1 quadratic continuation defined in Section 3.
4. The training loss must remain **finite and differentiable for finite `J`, including negative and zero values**; **do not simply clamp the entire loss with a zero-gradient plateau**. Preserve gradients through `J` and to control points / network parameters.
5. Add unit tests for analytic values, C1 value/gradient matching, negative-J gradients, `torch.autograd.gradcheck`, mask normalization, identity, and a deformation-parameter gradient test.
6. Add logged diagnostics for min(J), fraction(J <= 0), fraction(J < epsilon), and the KL-C1 loss **without altering any evaluation metric's pre-existing definition**.
7. Keep existing experiments and losses unchanged by default. Make activation opt-in or set `lambda_kl=0` until selected by the experiment configuration.
8. Add a short implementation note identifying the exact code locations modified, how to enable the loss, test results, and any Jacobian-coordinate assumptions. **Do not launch full training / GPU experiments unless specifically requested.**

---

## 1. Original paper and the actual contribution

**Reference:**

Leow AD, Yanovsky I, Chiang M-C, et al. *Statistical Properties of Jacobian Maps and the Realization of Unbiased Large-Deformation Nonlinear Image Registration*. **IEEE Transactions on Medical Imaging**. 2007;26(6):822–832. DOI: **10.1109/TMI.2007.892646**. [PubMed](https://pubmed.ncbi.nlm.nih.gov/17679333/) · [DOI](https://doi.org/10.1109/TMI.2007.892646).

**Relevant original equations:** Eq. (11) defines the symmetrized KL regularizer; Eq. (12) integrates it into the image-registration objective; Eq. (13) develops its Euler–Lagrange variation. The paper also separately proposes *equivalent perturbation* for symmetric image matching (Eqs. 17–19); **that second contribution is out of scope here**.

Let `phi: Omega -> Omega` be a differentiable, one-to-one and onto deformation with positive Jacobian determinant:

\[
J_\phi(x)=\det D\phi(x)>0.
\]

The original symmetric KL regularization is:

\[
\boxed{R_{\mathrm{Leow}}(\phi)=\int_\Omega \underbrace{(J_\phi(x)-1)\log J_\phi(x)}_{f(J_\phi(x))}\,dx.}
\]

The scalar loss density is:

\[
f(J)=(J-1)\log J,\quad J>0.
\]

Its first and second derivatives are:

\[
\boxed{f'(J)=1+\log J-\frac{1}{J}},\qquad
\boxed{f''(J)=\frac{1}{J}+\frac{1}{J^2}}.
\]

Properties: `f(1)=0`, `f(J)>=0` for `J>0`; `f'(J)<0` for `0<J<1`, `f'(1)=0`, and `f'(J)>0` for `J>1`. The scalar density is convex on the positive real axis. It penalizes unnecessary local volume distortion and does **not** penalize volume-preserving shear (`det(D phi)=1`). Therefore this loss is a supplement, not a replacement for spatial smoothness / geometric constraints.

**Why “symmetric”?** This is an integrated **inverse-mapping symmetry**, not pointwise equality of `f(J)` and `f(1/J)`. If `psi(J) = (J-1) log J`, then

\[
\psi(J)=J\,\psi(1/J).
\]

Using `J_{phi^{-1}}(phi(x))=1/J_phi(x)` and the change of variables `dy=J_phi(x) dx`, it follows that:

\[
R_{\mathrm{Leow}}(\phi)=R_{\mathrm{Leow}}(\phi^{-1}).
\]

**Important scope:** the symmetry requires a legitimate positive-J mapping and compatible forward/inverse domains and integration measures. Masking an arbitrary ROI does not automatically preserve that identity. In the original paper, the proposed regularizer was experimentally used with a numerical solver inspired by viscous-fluid registration; it was **not** a B-spline/SVF-specific parameterization or the C1 modification described here.

---

## 2. Why a modified numerical loss is required

The original `f(J)` is undefined for `J<=0` and its slope diverges toward negative infinity as `J->0+`. For **direct DVF / direct B-spline** optimization, numerical steps may produce `J<=0` even from a valid initialization.

A naive stabilized loss,

```python
J_safe = J.clamp_min(epsilon)
loss = ((J_safe - 1.0) * torch.log(J_safe)).mean()
```

avoids NaNs but **has zero loss gradient with respect to `J` for `J<epsilon`**, making negative-J corrections depend entirely on other losses. We want a continuation with a nonzero derivative that is continuously joined to the original function.

Our chosen approach is **C1 quadratic continuation**, not hard-clamped KL and not KL plus a second folding loss.

---

## 3. Proposed C1 quadratic continuation (our modification)

Let `epsilon` be the positive joining threshold and `kappa` be the curvature of the continuation branch. Define:

\[
\boxed{
\widetilde f_{\epsilon,\kappa}(J)=
\begin{cases}
 (J-1)\log J, & J\ge\epsilon,\\[5pt]
 f(\epsilon)+f'(\epsilon)(J-\epsilon)+\dfrac{\kappa}{2}(J-\epsilon)^2, & J<\epsilon.
\end{cases}}
\]

Here

\[
f(\epsilon)=(\epsilon-1)\log\epsilon,\qquad
f'(\epsilon)=1+\log\epsilon-1/\epsilon.
\]

**Derivative used by autograd / mathematical validation:**

\[
\boxed{
\widetilde f'_{\epsilon,\kappa}(J)=
\begin{cases}
1+\log J-1/J,&J\ge\epsilon,\\[3pt]
f'(\epsilon)+\kappa(J-\epsilon),&J<\epsilon.
\end{cases}}
\]

At `J=epsilon`, left and right values and derivatives agree exactly. The loss is thus `C1` on the whole real line, including `J=0` and negative values. For `0<epsilon<1` and `kappa>0`, the continuation is decreasing as `J` rises toward `epsilon`, so gradient descent is supplied with a scalar incentive to raise dangerously small `J` values (the full vector-field optimization can still be coupled across voxels).

The second derivative jumps unless:

\[
\kappa=f''(\epsilon)=1/\epsilon+1/\epsilon^2.
\]

Thus it is generally **not C2**. Full `C2` matching is not an objective here.

### Fixed initial parameters — use these exact defaults

| Parameter | Selected value | Interpretation |
|---|---:|---|
| `epsilon` | **0.05** | When `J<0.05`, replace raw KL with the continuation. `J=0.05` means **5% of the original local volume remains**, i.e. **20-fold volume compression** (95% volume loss), **not** 20-fold compression along each spatial axis. |
| `kappa` | **120.0** | Constant curvature `d²f_tilde/dJ²=120` on the continued branch only; larger values make the loss and negative derivative grow faster as J decreases below epsilon. |
| `lambda_kl` | **Not fixed here** | Separate multiplier on this regularizer in the whole registration objective. Must be swept/calibrated against the image-similarity and existing smoothness terms. |

At `epsilon=0.05`,

\[
\begin{aligned}
f(\epsilon)&=2.8459456599,\\
f'(\epsilon)&=-21.9957322736,\\
f''(\epsilon)&=420,\\
\kappa&=120<420.
\end{aligned}
\]

We deliberately choose a **softer curvature than the original KL at the connection point**, while retaining its exact value and slope there. The continuation lies below the original function immediately to the left of `epsilon` on the positive axis (and the original function has no definition at `J<=0`). This is a practical numerical surrogate, **not the unique or theoretically compelled choice**.

**Numeric reference values (for unit tests):**

| `J` | `f_tilde(J)` | `d f_tilde / dJ` | Comment |
|---:|---:|---:|---|
| −0.10 | 7.495305501 | −39.995732274 | Negative determinant: finite loss and active gradient |
| 0 | 4.095732274 | −27.995732274 | No log singularity |
| 0.01 | 3.821774951 | −26.795732274 | Continued branch |
| 0.05 | 2.845945660 | −21.995732274 | C1 joining point |
| 0.50 | 0.346573590 | −1.693147181 | Exact original KL |
| 1.00 | 0 | 0 | Identity: global minimum |
| 2.00 | 0.693147181 | 1.193147181 | Exact original KL |

### How epsilon and kappa change the shape

![Epsilon sweep against original KL](leow_kl_figures/epsilon_effect.png)

- **Epsilon sweep (`kappa=120`):** Compare `epsilon=0.02`, `0.05`, and `0.10` against raw KL. Smaller `epsilon` extends the domain of *exact agreement* with the raw KL and moves the join closer to zero. However, its joining slope `f'(epsilon)` becomes substantially more negative, so this is **not automatically more numerically stable**.

![Kappa sweep against original KL](leow_kl_figures/kappa_effect.png)

- **Kappa sweep (`epsilon=0.05`):** Compare `kappa=40`, `120`, and `420`. All curves share the exact same value and first derivative at the join and are identical to the original KL for `J>=epsilon`; kappa only changes the curvature and recovery gradient below the threshold. `kappa=420` matches the original *second derivative at the join*; indefinitely increasing kappa does **not** monotonically make the continuation closer to the original KL everywhere.

![Derivative sweeps](leow_kl_figures/derivative_effects.png)

- **Gradient perspective:** epsilon fixes the joining point and joining slope; kappa controls the growth rate of the negative derivative below epsilon. A larger kappa may cause stronger and less stable updates if regularization is heavily weighted.

![Original versus selected C1](leow_kl_figures/original_vs_c1.png)

**Mathematical caveats:**

- `f_tilde` is a **finite penalty**, not a hard invertibility constraint. `J<=0` can still occur at an optimum.
- Exact Leow inverse symmetry is retained only when both a forward determinant and its reciprocal fall in the raw branch (locally `epsilon <= J <= 1/epsilon`), assuming valid inverse mappings and domains. The global C1 extended loss is **not** strictly inverse-symmetric in general.
- It still only depends on local volume, not full deformation gradient, shear or true anatomical correspondence.
- Raw Leow diverges at `J->0+`; a quadratic polynomial cannot reproduce that singularity globally. The goal is robust optimization, not asymptotic equality in the invalid region.

---

## 4. PyTorch reference implementation

A tested standalone reference is included as [`reference_kl_c1.py`](reference_kl_c1.py). **Codex should adapt it to the existing loss style and shapes, rather than copy in a new Jacobian solver without checking conventions.**

```python
import math
import torch


def leow_kl_c1_density(J: torch.Tensor,
                       epsilon: float = 0.05,
                       kappa: float = 120.0) -> torch.Tensor:
    """Per-voxel Leow KL with C1 quadratic extension for J < epsilon."""
    assert 0.0 < epsilon < 1.0
    assert kappa > 0.0

    # Preserve float64 for gradcheck; promote half precision to float32.
    j = J.float() if J.dtype in (torch.float16, torch.bfloat16) else J

    # Both branches must be numerically valid even though torch.where selects one.
    j_safe = j.clamp_min(epsilon)
    raw_branch = (j_safe - 1.0) * torch.log(j_safe)

    f_eps = (epsilon - 1.0) * math.log(epsilon)
    fp_eps = 1.0 + math.log(epsilon) - 1.0 / epsilon
    delta = j - epsilon
    c1_branch = f_eps + fp_eps * delta + 0.5 * kappa * delta.square()

    return torch.where(j >= epsilon, raw_branch, c1_branch)


def leow_kl_c1_loss(J: torch.Tensor,
                    epsilon: float = 0.05,
                    kappa: float = 120.0,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
    density = leow_kl_c1_density(J, epsilon=epsilon, kappa=kappa)
    if mask is None:
        return density.mean()
    # This reference expects mask.shape == J.shape; adapt explicit
    # batch/channel normalization to the existing repository.
    if mask.shape != density.shape:
        raise ValueError('Mask must match determinant tensor shape')
    weights = mask.to(device=density.device, dtype=density.dtype)
    if bool((weights.sum() <= 0).item()):
        raise ValueError('Mask must contain positive total weight')
    return (density * weights).sum() / weights.sum()
```

### Autograd and differentiability details

- **Do not detach `J`**, and do not convert the loss to Python floats in the optimization graph.
- `j_safe = clamp_min(j,epsilon)` appears **only inside the safe evaluation of the raw branch**. It does **not** cause a gradient plateau for `J<epsilon` because `torch.where` selects the continuation there.
- Never evaluate `log(J)` unguarded in an inactive `torch.where` branch: a non-selected NaN can still cause problems during backward.
- For half/bfloat16, compute the penalty in float32 (prefer float32 for the determinant calculation itself as well); preserve float64 during gradchecks.
- The scalar mathematical derivative is not the deformation-parameter derivative. PyTorch must also backpropagate through `J=det(Dphi)` and the underlying sampled/composed deformation.
- A mean over voxels is a normalized approximation to the spatial integral. Keep normalization consistent across baselines, resolutions and masks, and tune the separate `lambda_kl` accordingly.

### Minimal integration pattern

```python
# After obtaining the FINAL composed transform phi:
J = jacobian_determinant_of_final_transform(phi)  # existing trusted utility
loss_jac = leow_kl_c1_loss(J, epsilon=0.05, kappa=120.0, mask=brain_mask)
loss_total = loss_similarity + lambda_smooth * loss_smooth + lambda_kl * loss_jac
```

For a displacement field `phi(x)=x+u(x)`, use `J=det(I + Du)` if differentiating displacement; if differentiating the absolute coordinate transform, use `J=det(Dphi)` directly. Match the existing code's axis order (`D,H,W` vs `x,y,z`) and spatial coordinate units (voxels, millimeters, or normalized grid coordinates). **The identity deformation must produce `J==1`**, which is a required test.

**Multi-resolution / composition:** evaluate this loss on the final composed deformation for each optimization step at the appropriate resolution, not merely on individual incremental displacements. In an SVF method, apply it to the final integrated `exp(v)` deformation, not to `det(I + Dv)` (which is not the final transform). Do not silently apply additional per-stage penalties without documenting them.

---

## 5. Required tests and acceptance checks

The bundle includes a CPU-only reference test suite: [`test_reference_kl_c1.py`](test_reference_kl_c1.py). It passed `8` tests against the standalone reference, including a simple differentiable affine-deformation test. After integration, **adapt and rerun tests on the actual repository code**, including its Jacobian computation.

| Test | Expected result |
|---|---|
| `J=1` identity | Loss = 0, derivative = 0; integration of identity deformation should yield this |
| `J>=epsilon` | Exact equality with original `(J-1)*log(J)` (within floating-point tolerance) |
| `J=epsilon +/- delta` | Values and first derivatives converge to `f(epsilon)` and `f'(epsilon)` |
| `J=0`, negative J | Finite values; derivatives finite and strictly negative |
| Negative J ordering | More negative J yields greater per-voxel penalty |
| `gradcheck` in float64 | Autograd matches finite differences away from branch transition |
| Mask reduction | Correct masked sum / mask weight; no silent all-zero masks |
| Computational dtype | No half-precision NaNs in expected input range |
| Parameter gradients | Final loss backpropagates through a small deformation, determinant and B-spline/control-point parameters |
| Known deformations | Translation: `J=1`; diagonal affine scaling: `J=product(scales)`; identity and simple shear: `J=1` |
| No regression | Original registration loss/config unchanged with `lambda_kl=0` |

**Logging must keep raw determinant values**; don't clamp them for folding statistics. Suggested fields per image and per resolution: `kl_c1_loss`, `J_min`, `J_p01`, `J_median`, `J_p99`, `frac_J_le_0`, `frac_J_lt_epsilon`, `grad_norm_from_jacobian_loss` (optional). Do not confuse training loss with a standard `std(logJ)` evaluation definition; report the actual evaluation convention separately, particularly when negative determinants are present.

### Specific failure modes to inspect

1. **Wrong `J` tensor:** determinant of `Du` instead of `I+Du`, wrong coordinate normalization, wrong physical spacing or channel order.
2. **No repair gradient:** implementing only a hard clamp, or detaching `J`, causes zero gradients in the invalid region.
3. **Incorrect stage:** loss computed on the B-spline increment rather than the final composed displacement.
4. **Instability:** large `lambda_kl` or sharp determinant gradients overwhelm the image matching term; tune the overall weight and learning rate, not just `epsilon` and `kappa`.
5. **Evaluation leakage:** changing J before calculating folding statistics or other established metrics.
6. **Misleading claims:** this is *not* a hard diffeomorphism guarantee, and its extended branch is *not* exactly inversion-invariant.

---

## 6. Suggested experimental protocol (only after implementation is approved)

Keep the same dataset, image pair ordering, image similarity, optimizer, multiresolution settings, initial transformation, parameterization and Jacobian numerical definition. Start with one or two representative 3D pairs and compare:

| Variant | Jacobian regularization |
|---|---|
| Existing baseline | Existing smoothness only |
| New method | Existing smoothness + C1-Leow (`eps=0.05`, `kappa=120`, weight tuned) |
| Optional ablation A | Existing smoothness + hard-clamped original KL, clearly labeled |
| Optional ablation B | Existing smoothness + original KL **only when positive-J feasibility is assured** |

Tune `lambda_kl` independently from `epsilon` and `kappa` (the latter two are **fixed for the initial implementation**). At matched image-alignment accuracy, compare Dice / TRE or relevant label metrics, SD(logJ) with documented evaluation definition, minimum/quantiles of J, nonpositive-J fraction, and objective convergence. Lower SD(logJ) alone does **not** demonstrate correct anatomical correspondence or volume preservation in a structure that naturally changes volume.

---

## 7. Final Codex checklist

- [ ] Locate and summarize existing registration deformation/Jacobian/loss conventions before editing.
- [ ] Add `leow_kl_c1` with `epsilon=0.05`, `kappa=120.0` and separate configurable `lambda_kl`.
- [ ] Use raw KL exactly at `J >= epsilon` and C1 quadratic continuation at `J < epsilon`.
- [ ] Keep entire training path differentiable for negative J; avoid NaN in inactive branches.
- [ ] Add direct scalar loss tests and full deformation-parameter backprop test.
- [ ] Preserve all legacy losses and metrics unless a new variant is explicitly selected.
- [ ] Log invalid-J rates and diagnostics using raw J, not the continuation's surrogate values.
- [ ] Record any limitations of masks, inverse symmetry, discrete Jacobian evaluation, and folding guarantees.
- [ ] Report changed file paths, exact run/test commands, outcomes and remaining concerns; do not run a full training sweep by default.

### Bundled resources

- `reference_kl_c1.py` — independent PyTorch reference, **not** original Leow source.
- `test_reference_kl_c1.py` — reference unit tests.
- `make_plots.py` — analytical figure regeneration, no training/data required.
- `leow_kl_figures/` — four figures comparing raw KL with the continuation and parameter sweeps.
