# SINR vs Direct B-Spline Control-Point Optimization: Kernel Diagnostic Experiment Plan

## Purpose

This document describes a compact experiment plan for testing whether the observed behavior of SINR is mainly caused by the B-spline parameterization, while the INR coefficient generator worsens optimization conditioning.

The goal is to produce a small, paper-ready diagnostic experiment, not a complete theoretical characterization of SINR. The experiment should be easy to run inside the existing registration codebase and should reuse the already implemented SINR, direct control-point B-spline optimization, IDIR, and image-similarity losses.

The main result we want to support is:

> SINR and direct B-spline control-point optimization induce similar dense update directions after scalar rescaling, suggesting that the dominant useful inductive bias comes from the B-spline parameterization. However, the coefficient INR introduces an additional kernel, \(K_{\mathrm{CP\text{-}INR}} = J_\theta J_\theta^\top\), which amplifies and anisotropically rescales the control-point gradient. This worsens optimization conditioning and explains why SINR requires a much smaller learning rate than direct control-point optimization.

---

## Background and empirical motivation

Current observations:

1. **Task-dependent registration performance**
   - Inter-subject brain MR after affine alignment: SINR performs better than IDIR.
   - Intra-subject lung inhale-exhale registration: IDIR performs better than SINR when no multiresolution strategy is used.
   - SINR can get trapped in local minima for large lung motion unless multiresolution is used.
   - IDIR surprisingly works well for lung even without multiresolution.

2. **Teacher-DVF fitting result**
   - A SOTA registration method was used to produce displacement fields, treated as teacher or pseudo-GT fields.
   - SINR and direct B-spline control-point optimization fit these teacher DVFs very well for both brain and lung.
   - Therefore, SINR failure on lung registration is unlikely to be caused by lack of spline expressivity.
   - This suggests the main issue is optimization under the image-similarity loss.

3. **Learning-rate observation**
   - SINR only works reliably around LR \(10^{-4}\).
   - SINR becomes unstable at LR \(10^{-3}\) and \(10^{-2}\).
   - Direct B-spline control-point optimization works well up to LR \(10^{-2}\) and is much faster.
   - This suggests the spline parameterization is not the source of LR instability. The INR coefficient generator is likely responsible.

---

## Main hypotheses

### Hypothesis 1: The dominant useful bias is the B-spline parameterization

SINR and direct control-point optimization share the same B-spline decoder from control-point coefficients to dense displacement field. If their dense update directions are similar, then the main registration behavior of SINR can be attributed to the B-spline parameterization, rather than the neural network used to generate the coefficients.

Operational test:

\[
BK_{\mathrm{CP\text{-}INR}}s \approx \alpha Bs,
\]

where:

- \(B\): B-spline interpolation operator,
- \(s\): control-point-space image gradient,
- \(K_{\mathrm{CP\text{-}INR}} = J_\theta J_\theta^\top\): SINR coefficient INR kernel,
- \(\alpha\): best scalar rescaling.

If the approximation holds on actual registration gradients, SINR is behaving like direct B-spline CP optimization plus learning-rate rescaling.

### Hypothesis 2: The INR coefficient generator worsens optimization conditioning

Direct CP optimization has an identity kernel in coefficient space:

\[
K_{\mathrm{direct\ CP}} = I.
\]

SINR has:

\[
K_{\mathrm{CP\text{-}INR}} = J_\theta J_\theta^\top.
\]

If \(K_{\mathrm{CP\text{-}INR}}\) is ill-conditioned or strongly amplifies the actual control-point gradient, SINR should require a much smaller learning rate than direct CP optimization.

Operational test:

\[
A_C = \frac{\|K_{\mathrm{CP\text{-}INR}}s\|}{\|s\|}
\]

and

\[
\kappa_{\mathrm{eff}}(K_{\mathrm{CP\text{-}INR}}) \gg 1.
\]

If \(A_C\approx 100\), this directly explains why SINR works at LR \(10^{-4}\) while direct CP works at LR \(10^{-2}\):

\[
10^{-4}\|K_{\mathrm{CP\text{-}INR}}s\| \approx 10^{-2}\|s\|.
\]

---

## Problem formulation

Let:

- \(F\): fixed image,
- \(M\): moving image,
- \(\mathcal{L}_{img}(F, M \circ \phi)\): image-similarity loss,
- \(U\): dense displacement field,
- \(C\): B-spline control-point coefficients,
- \(Q\): coordinates of B-spline control points,
- \(B\): B-spline interpolation operator,
- \(h_\theta\): SINR coefficient INR,
- \(C_\theta = h_\theta(Q)\): SINR-predicted B-spline coefficients.

The dense displacement is:

\[
U = BC.
\]

For direct CP optimization, \(C\) is optimized directly. For SINR, \(C\) is produced by an INR:

\[
C = h_\theta(Q).
\]

Let:

\[
G = \frac{\partial \mathcal{L}_{img}}{\partial U}
\]

be the dense displacement-space gradient. The corresponding control-point-space gradient is:

\[
s = \frac{\partial \mathcal{L}_{img}}{\partial C} = B^\top G.
\]

### Direct CP update

For direct control-point optimization:

\[
\Delta C_{\mathrm{CP}} = -\eta s.
\]

The induced dense update is:

\[
\Delta U_{\mathrm{CP}} = -\eta Bs = -\eta BB^\top G.
\]

### SINR update

For SINR:

\[
J_\theta = \frac{\partial \mathrm{vec}(C_\theta)}{\partial \theta}.
\]

Under infinitesimal vanilla SGD:

\[
\Delta C_{\mathrm{SINR}} \approx -\eta J_\theta J_\theta^\top s.
\]

Define:

\[
K_{\mathrm{CP\text{-}INR}} = J_\theta J_\theta^\top.
\]

Then:

\[
\Delta U_{\mathrm{SINR}} \approx -\eta B K_{\mathrm{CP\text{-}INR}} s = -\eta B J_\theta J_\theta^\top B^\top G.
\]

Therefore, SINR combines:

1. the B-spline preconditioning/update operator \(BB^\top\), and
2. the coefficient INR kernel \(J_\theta J_\theta^\top\).

The central experiment is to determine whether \(J_\theta J_\theta^\top\) mainly rescales the B-spline update or changes it substantially.

---

## Minimal metrics

Only three core metrics are needed for the paper.

### Metric 1: Dense update-direction agreement

Compute:

\[
D_U = \cos(BK_{\mathrm{CP\text{-}INR}}s, Bs)
\]

where:

\[
\cos(a,b) = \frac{\langle a,b\rangle}{\|a\|\|b\|}.
\]

Interpretation:

- \(D_U \approx 1\): SINR and direct CP induce nearly the same dense update direction.
- \(D_U \ll 1\): the coefficient INR changes the direction substantially.

This is the main metric for testing whether the B-spline parameterization dominates the SINR update direction.

### Metric 2: Scalar-rescaled residual

Find the best scalar \(\alpha\):

\[
\alpha = \frac{\langle BK_{\mathrm{CP\text{-}INR}}s, Bs\rangle}{\langle Bs, Bs\rangle}.
\]

Then compute:

\[
R_U = \frac{\|BK_{\mathrm{CP\text{-}INR}}s - \alpha Bs\|}{\|BK_{\mathrm{CP\text{-}INR}}s\|}.
\]

Interpretation:

- \(R_U \approx 0\): SINR is approximately direct CP after scalar LR rescaling.
- Large \(R_U\): SINR adds a distinct neural-kernel update bias.

This metric is stronger than cosine alone because it asks whether SINR can be explained by direct CP plus a scalar learning-rate adjustment.

### Metric 3: Gradient amplification

Coefficient-space amplification:

\[
A_C = \frac{\|K_{\mathrm{CP\text{-}INR}}s\|}{\|s\|}.
\]

Dense-space amplification:

\[
A_U = \frac{\|BK_{\mathrm{CP\text{-}INR}}s\|}{\|Bs\|}.
\]

Interpretation:

- \(A_C \gg 1\) or \(A_U \gg 1\): SINR amplifies the CP update.
- If the amplification is roughly \(100\), this explains a two-order-of-magnitude smaller stable LR for SINR.

---

## Optional but useful metric: effective condition number

The full condition number

\[
\kappa(K) = \frac{\lambda_{\max}}{\lambda_{\min}}
\]

may be unstable because \(K = JJ^\top\) can have near-zero eigenvalues.

Use a robust effective condition number instead:

\[
\kappa_{\mathrm{eff}} = \frac{\lambda_{95}}{\lambda_{5}},
\]

or:

\[
\kappa_{\mathrm{eff}} = \frac{\lambda_{\max}}{\lambda_{\mathrm{median}}}.
\]

Only include eigenvalues above a threshold such as:

\[
\lambda_i > 10^{-6}\lambda_{\max}.
\]

Direct CP has identity coefficient-space kernel:

\[
K_{\mathrm{direct\ CP}} = I.
\]

Therefore:

\[
\kappa_{\mathrm{direct\ CP}} = 1.
\]

If SINR has a highly skewed spectrum, this supports the claim that the INR coefficient generator worsens optimization conditioning.

---

## When to measure the kernel

For the main paper, measure only two checkpoints:

1. **Initialization**: \(t=0\)
2. **Early training**: \(t=10\), \(t=50\), or another early checkpoint before major convergence

Recommended minimal choice:

\[
t \in \{0, 50\}.
\]

Rationale:

- \(t=0\): measures architecture-induced conditioning before task-specific training.
- early training: captures the optimization phase where LR instability and local-minimum behavior are most relevant.

Final checkpoint \(t=T\) is optional and better suited for supplementary material. The final kernel describes the geometry after SINR has already succeeded or failed, so it is less useful for explaining early optimization behavior.

---

## Main experiment design

### Experiment 1: LR sweep confirmation

Purpose: connect the kernel diagnostics to the observed LR sensitivity.

Run or reuse existing runs for:

- SINR with LR \(10^{-2}\), \(10^{-3}\), \(10^{-4}\)
- direct CP with LR \(10^{-2}\), \(10^{-3}\), \(10^{-4}\)

Use the same:

- image pairs,
- preprocessing,
- B-spline grid resolution,
- image-similarity loss,
- regularization if any,
- optimizer family if possible,
- number of iterations,
- initialization.

Record:

- image loss curve,
- final registration metric,
- runtime or iterations to convergence,
- Jacobian folding percentage if already available,
- whether the run is stable or diverges.

Expected result:

- direct CP remains stable at LR \(10^{-2}\), \(10^{-3}\), and \(10^{-4}\), with faster convergence at larger LR;
- SINR only stable at LR \(10^{-4}\);
- SINR unstable or divergent at LR \(10^{-3}\) and \(10^{-2}\).

Figure:

- line plot of image loss versus iteration for SINR and direct CP across learning rates.

### Experiment 2: Update-kernel diagnostic on actual registration gradients

Purpose: show whether SINR follows the same dense update direction as direct CP, and whether it amplifies the gradient.

For each selected case and checkpoint:

1. Run SINR to checkpoint \(t\).
2. Extract current coefficient INR \(h_{\theta_t}\).
3. Compute current control points:

   \[
   C_t = h_{\theta_t}(Q).
   \]

4. Treat \(C_t\) as a leaf tensor and compute the CP-space image gradient:

   \[
   s_t = \frac{\partial \mathcal{L}_{img}}{\partial C_t}.
   \]

5. Compute:

   \[
   K_t s_t = J_tJ_t^\top s_t.
   \]

6. Compute dense CP update proxy:

   \[
   B s_t.
   \]

7. Compute dense SINR update proxy:

   \[
   B K_t s_t.
   \]

8. Report:

   \[
   D_U = \cos(BK_ts_t, Bs_t),
   \]

   \[
   R_U = \frac{\|BK_ts_t - \alpha Bs_t\|}{\|BK_ts_t\|},
   \]

   \[
   A_C = \frac{\|K_ts_t\|}{\|s_t\|},
   \]

   \[
   A_U = \frac{\|BK_ts_t\|}{\|Bs_t\|}.
   \]

Case selection:

- 3 to 5 representative brain pairs.
- 3 to 5 representative lung inhale-exhale pairs.
- Include successful and difficult cases if possible.

Expected result:

- \(D_U\) high, e.g. \(>0.9\) or \(>0.95\): SINR dense update direction is close to direct CP.
- \(R_U\) low: SINR update is largely explained by scalar-rescaled direct CP.
- \(A_C\) or \(A_U\) large: SINR amplifies gradients, explaining smaller stable LR.

Interpretation:

- High update agreement supports spline-dominance.
- Large amplification supports INR-induced LR instability.

### Experiment 3: CP-INR kernel spectrum

Purpose: show that the coefficient INR kernel is ill-conditioned compared with direct CP.

For selected checkpoints:

1. Estimate the largest eigenvalue \(\lambda_{\max}(K_t)\) using power iteration with matrix-vector products.
2. Estimate a robust effective condition number using either:
   - sampled empirical kernel spectrum, or
   - Lanczos / randomized eigensolver, or
   - approximate percentile eigenvalue ratio.
3. Compare against direct CP, whose coefficient-space kernel is identity.

Expected result:

- direct CP spectrum is flat by definition: all eigenvalues equal to 1.
- SINR spectrum is skewed or heavy-tailed.
- \(\kappa_{\mathrm{eff}}(K_t) \gg 1\).

Figure:

- normalized eigenvalue spectrum of \(K_t\) for SINR.
- horizontal reference line for direct CP identity spectrum.

Main-text interpretation:

> The coefficient INR induces an anisotropic optimization metric in control-point space, while direct CP optimization is isotropic.

---

## Implementation details

### Required code hooks

Codex should locate or implement access to the following components in the existing codebase:

1. **SINR coefficient network**
   - Function or module: \(h_\theta(Q)\)
   - Input: control-point coordinates \(Q\)
   - Output: B-spline control-point coefficients \(C\)

2. **Control-point coordinate grid**
   - Tensor \(Q\), shape approximately:

     ```text
     [num_control_points, coord_dim]
     ```

   - coordinate normalization must match training.

3. **B-spline interpolation layer**
   - Function: `bspline_interpolate(C)`
   - Input: control-point coefficients \(C\)
   - Output: dense displacement field \(U=BC\)

4. **Image-similarity loss**
   - Same loss used in training.
   - Should take fixed image, moving image, and dense displacement field.

5. **Warping operation**
   - Same spatial transformer used in training.

6. **Current checkpoint loader**
   - Load SINR at initialization and early checkpoint.

---

## Computing the control-point gradient \(s\)

Do not compute \(G = \partial \mathcal{L}/\partial U\) explicitly. Compute \(s\) directly by treating the current control points as leaf variables.

Pseudocode:

```python
# Current SINR-predicted control points
with torch.no_grad():
    C_current = sinr_model(control_point_coords)  # shape: [Ncp, disp_dim] or grid shape

# Treat C as an optimizable leaf variable
C_leaf = C_current.detach().clone().requires_grad_(True)

# Same B-spline decoder used in registration
U = bspline_interpolate(C_leaf)

# Same image loss used during training
loss = image_similarity_loss(fixed_image, moving_image, U)

# CP-space gradient
(s,) = torch.autograd.grad(loss, C_leaf)
s_flat = s.reshape(-1).detach()
```

This gives:

\[
s = \frac{\partial \mathcal{L}}{\partial C} = B^\top G.
\]

---

## Computing \(K_ts = J_tJ_t^\top s\) without forming \(J_t\) or \(K_t\)

Use one vector-Jacobian product and one Jacobian-vector product.

Pseudocode using `torch.func`:

```python
import torch
from torch.func import functional_call, vjp, jvp


def make_coeff_function(model, q):
    """
    Returns f(params) = flattened SINR control-point coefficients.
    """
    params = {
        k: v.detach().clone().requires_grad_(True)
        for k, v in model.named_parameters()
    }
    buffers = dict(model.named_buffers())

    def f(params):
        C = functional_call(model, (params, buffers), (q,))
        return C.reshape(-1)

    return f, params


def K_matvec(f, params, v):
    """
    Computes K v = J J^T v, where J = d vec(C) / d theta.
    v should have shape [num_control_point_coefficients].
    """
    _, vjp_fn = vjp(f, params)

    # Parameter-space vector: J^T v
    (g_params,) = vjp_fn(v)

    # Coefficient-space vector: J (J^T v)
    _, Kv = jvp(f, (params,), (g_params,))

    return Kv.detach()
```

Usage:

```python
f, params = make_coeff_function(sinr_model, control_point_coords)
Ks_flat = K_matvec(f, params, s_flat)
```

Reshape:

```python
Ks = Ks_flat.reshape_as(C_current)
```

---

## Computing dense update proxies

Direct CP dense update proxy:

```python
Bs = bspline_interpolate(s.detach())
```

SINR dense update proxy:

```python
BKs = bspline_interpolate(Ks.detach())
```

Then flatten and compute metrics:

```python
def cosine(a, b, eps=1e-12):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return torch.dot(a, b) / ((a.norm() + eps) * (b.norm() + eps))

DU = cosine(BKs, Bs)

alpha = torch.dot(BKs.reshape(-1), Bs.reshape(-1)) / (
    torch.dot(Bs.reshape(-1), Bs.reshape(-1)) + 1e-12
)

RU = (BKs - alpha * Bs).norm() / (BKs.norm() + 1e-12)

AC = Ks_flat.norm() / (s_flat.norm() + 1e-12)
AU = BKs.norm() / (Bs.norm() + 1e-12)
```

Save:

```text
D_U, R_U, A_C, A_U, alpha
```

---

## Estimating \(\lambda_{\max}(K_t)\)

Use power iteration with `K_matvec`.

```python
def estimate_lambda_max(f, params, num_iter=30, eps=1e-12):
    y = f(params).detach()
    v = torch.randn_like(y)
    v = v / (v.norm() + eps)

    for _ in range(num_iter):
        Kv = K_matvec(f, params, v)
        v = Kv / (Kv.norm() + eps)

    Kv = K_matvec(f, params, v)
    lam = torch.dot(v, Kv)
    return lam.item()
```

This estimates the worst-case gradient amplification direction of the SINR coefficient INR.

Important:

- \(\lambda_{\max}\) is a worst-case quantity.
- The actual task gradient \(s\) may not align with the top eigenvector.
- Therefore always report \(A_C=\|Ks\|/\|s\|\) together with \(\lambda_{\max}\).

---

## Estimating a robust effective condition number

Full eigen-decomposition may be too expensive. Use one of the following strategies.

### Option A: Sampled empirical kernel

Sample \(m\) control-point coefficient indices, e.g. \(m=256\), \(512\), or \(1024\).

Construct an empirical kernel on those sampled outputs:

\[
K_{ij} = \langle \nabla_\theta C_i, \nabla_\theta C_j \rangle.
\]

Then compute eigenvalues of the sampled matrix.

Use:

\[
\kappa_{\mathrm{eff}} = \frac{\lambda_{95}}{\lambda_5}
\]

or:

\[
\kappa_{\mathrm{eff}} = \frac{\lambda_{\max}}{\lambda_{\mathrm{median}}}.
\]

Normalize the sampled kernel before plotting:

\[
\bar K = \frac{mK}{\mathrm{tr}(K)}.
\]

This makes spectra more comparable across checkpoints and cases.

### Option B: Lanczos or randomized eigensolver

If the codebase already has access to `scipy.sparse.linalg.eigsh` or a PyTorch Lanczos utility, use `K_matvec` as a linear operator and estimate top and middle eigenvalues.

### Option C: Only report \(\lambda_{\max}\), \(A_C\), and \(A_U\)

If effective condition number is too difficult or unstable, omit \(\kappa_{\mathrm{eff}}\) from the main paper and use amplification metrics instead. The amplification metrics are more directly linked to LR sensitivity.

---

## Optional: actual optimizer-step diagnostic

The formula

\[
\Delta C_{\mathrm{SINR}} \approx -\eta J_tJ_t^\top s_t
\]

is exact only for infinitesimal vanilla SGD. If the implementation uses Adam, momentum, weight decay, or gradient clipping, the actual update may differ.

Therefore, optionally compute the realized coefficient update:

\[
\Delta C_{\mathrm{actual}} = h_{\theta_{t+1}}(Q) - h_{\theta_t}(Q).
\]

Procedure:

1. Clone the current SINR model and optimizer state.
2. Compute one training loss.
3. Perform one optimizer step.
4. Evaluate control points before and after.
5. Compute:

   \[
   \cos(\Delta C_{\mathrm{actual}}, -s_t),
   \]

   \[
   \cos(B\Delta C_{\mathrm{actual}}, -Bs_t),
   \]

   \[
   \frac{\|\Delta C_{\mathrm{actual}}\|}{\|s_t\|},
   \]

   \[
   \frac{\|B\Delta C_{\mathrm{actual}}\|}{\|Bs_t\|}.
   \]

This is a practical validation that the theoretical kernel analysis matches the actual optimizer behavior.

If Adam is used, main text can say:

> We use \(J_\theta J_\theta^\top\) as a local SGD kernel diagnostic and additionally verify the finite coefficient update induced by the actual optimizer.

---

## Optional: teacher-residual alignment

If SOTA teacher DVFs are available, compute whether CP and SINR updates point toward the teacher residual.

Let:

\[
u^*\]

be the teacher DVF and:

\[
u_t\]

be the current dense displacement. Define residual:

\[
r_t = u^* - u_t.
\]

Direct CP alignment:

\[
\mathrm{Align}_{\mathrm{CP}} = \cos(-Bs_t, r_t).
\]

SINR alignment:

\[
\mathrm{Align}_{\mathrm{SINR}} = \cos(-BK_ts_t, r_t).
\]

This is useful for explaining brain/lung differences, but it is optional for the compact paper experiment.

Expected:

- Brain: SINR and CP updates may align well with teacher residual.
- Lung full resolution: SINR may have weaker or unstable residual alignment.
- Lung with multiresolution: alignment may improve.

---

## Recommended logging schema

Save one CSV or JSONL file per experiment group.

Suggested columns:

```text
case_id
case_type                  # brain or lung
image_pair_id
method                     # SINR or direct_cp
checkpoint_iter            # 0, 50, etc.
lr
seed
loss_value
D_U                        # cos(BKs, Bs)
R_U                        # scalar-rescaled dense residual
A_C                        # ||Ks|| / ||s||
A_U                        # ||BKs|| / ||Bs||
alpha                      # best scalar matching BKs to Bs
lambda_max_K               # optional
kappa_eff_K                # optional
num_cp_coefficients
num_sampled_coefficients   # if sampled spectrum is used
optimizer                  # SGD, Adam, etc.
actual_step_D_C            # optional actual optimizer-step cosine in CP space
actual_step_D_U            # optional actual optimizer-step cosine in dense space
actual_step_A_C            # optional actual optimizer-step amplification
actual_step_A_U            # optional actual optimizer-step amplification
teacher_align_cp           # optional
teacher_align_sinr         # optional
runtime_sec
notes
```

For LR sweep logs:

```text
case_id
case_type
method
lr
seed
iteration
loss_value
metric_name
metric_value
is_stable
folding_percent
runtime_sec
```

---

## Recommended figures

### Figure 1: LR sensitivity

Line plot:

- x-axis: iteration
- y-axis: image loss or validation metric
- curves: SINR LR \(10^{-2}\), \(10^{-3}\), \(10^{-4}\); direct CP LR \(10^{-2}\), \(10^{-3}\), \(10^{-4}\)

Message:

> Direct CP is stable at larger learning rates, whereas SINR requires a much smaller LR.

### Figure 2: Update agreement and amplification

Bar plot or violin plot across cases:

- \(D_U = \cos(BKs, Bs)\)
- \(R_U\)
- \(A_C\) or \(A_U\)

Suggested layout:

- separate bars for brain and lung,
- separate panels for initialization and early checkpoint.

Message:

> SINR and direct CP updates are directionally similar after the B-spline decoder, but SINR strongly amplifies the gradient.

### Figure 3: CP-INR kernel spectrum

Plot normalized eigenvalues for SINR's \(K_{\mathrm{CP\text{-}INR}}\) using sampled empirical kernel or approximate spectrum.

Include direct CP reference:

- direct CP: flat identity spectrum.
- SINR: skewed/heavy-tailed spectrum.

Message:

> The INR coefficient generator induces a poorly conditioned optimization metric in CP space.

### Optional Figure 4: Dense update visualization

For one brain case and one lung case, show:

- direct CP dense update \(Bs\),
- SINR dense update \(BKs\),
- difference after scalar rescaling \(BKs - \alpha Bs\).

Message:

> Visual update fields are similar after rescaling, supporting B-spline dominance.

---

## Expected results and interpretation

### Expected outcome A: High update agreement and large amplification

Possible observation:

\[
D_U > 0.9,
\]

\[
R_U \text{ small},
\]

\[
A_C \text{ or } A_U \approx 10\text{--}100+.
\]

Interpretation:

> SINR follows nearly the same dense update direction as direct CP but with much larger effective step size. Therefore, the dominant update direction is determined by the B-spline parameterization, while the coefficient INR worsens learning-rate conditioning.

This is the ideal result.

### Expected outcome B: SINR kernel has high effective condition number

Possible observation:

\[
\kappa_{\mathrm{eff}}(K_{\mathrm{CP\text{-}INR}}) \gg 1.
\]

Interpretation:

> Direct CP is isotropic in coefficient space, whereas SINR is anisotropic because of the coefficient INR kernel. This explains why direct CP can use a larger LR and converge faster.

### Expected outcome C: Dense update agreement is high but CP-space agreement is lower

This can happen because the B-spline decoder smooths differences between CP-space updates.

Interpretation:

> Although the coefficient INR changes the CP-space update, the B-spline decoder suppresses many of those differences in dense displacement space. Therefore, the final effective deformation update is still dominated by the spline parameterization.

### Expected outcome D: Lung and brain have similar kernel metrics

This is not a problem.

Interpretation:

> The LR instability is architecture/parameterization-driven and therefore similar across tasks. The brain/lung performance difference likely comes from the interaction between this update operator and task-specific image gradients/capture range, rather than from different kernel spectra alone.

---

## Unexpected outcomes and workarounds

### Case 1: \(D_U\) is not high

Observation:

\[
D_U \ll 1
\]

or \(R_U\) is large.

Interpretation:

> The coefficient INR is not merely rescaling the spline update; it introduces a distinct neural-kernel bias.

Workaround:

- Do not claim B-spline completely dominates.
- Reframe the result:

  > SINR combines a B-spline output bias with an INR-induced coefficient-space kernel. Direct CP removes this additional kernel and therefore provides a cleaner and better-conditioned spline optimizer.

- Compare whether \(D_U\) is higher than CP-space cosine. If dense cosine is still moderate after B-spline interpolation, the spline decoder still partially dominates.

### Case 2: Amplification \(A_C\) is not large

Observation:

\[
A_C \approx 1.
\]

Potential explanations:

- The actual task gradient \(s\) does not align with amplified eigen-directions of \(K\).
- Adam or optimizer state is the main cause of LR sensitivity, not raw \(J J^\top\).
- LR instability comes from finite-step nonlinearity, not the infinitesimal kernel.

Workarounds:

- Report \(\lambda_{\max}(K)\) and effective condition number. The worst-case amplification may still be large.
- Run the actual optimizer-step diagnostic:

  \[
  \Delta C_{\mathrm{actual}} = h_{\theta_{t+1}}(Q)-h_{\theta_t}(Q).
  \]

- Compare \(\|\Delta C_{\mathrm{actual}}\|\) across SINR and direct CP at the same nominal LR.
- If actual SINR step is much larger, the LR explanation still holds.

### Case 3: Eigenvalue computation is too expensive

Workarounds:

1. Use only matrix-vector products and estimate \(\lambda_{\max}\) with power iteration.
2. Use sampled empirical kernel with 256 to 1024 sampled coefficient dimensions.
3. Skip \(\kappa_{\mathrm{eff}}\) and report only:

   \[
   D_U, R_U, A_C, A_U.
   \]

These are enough for the main claim.

### Case 4: \(K\) has near-zero eigenvalues

This is expected.

Do not report raw \(\lambda_{\max}/\lambda_{\min}\). Instead report:

\[
\lambda_{95}/\lambda_5
\]

or:

\[
\lambda_{\max}/\lambda_{\mathrm{median}}.
\]

Filter eigenvalues:

\[
\lambda_i > 10^{-6}\lambda_{\max}.
\]

### Case 5: Adam makes \(J J^\top\) insufficient

If Adam is used, the exact local kernel is closer to:

\[
J_\theta D_t J_\theta^\top,
\]

where \(D_t\) is Adam's diagonal parameter-space preconditioner.

Workaround:

- Still use \(J J^\top\) as a theoretical diagnostic.
- Add the actual finite optimizer-step diagnostic.
- Main text can say:

  > Because the training uses Adam, \(J J^\top\) is not the exact optimizer kernel. We therefore also measure the realized coefficient update after one optimizer step and find the same trend.

### Case 6: SINR output layer is zero-initialized or near-degenerate

If the final layer or output is initialized in a special way, \(J\) at \(t=0\) may be degenerate or not representative.

Workaround:

- Report both \(t=0\) and \(t=50\).
- If \(t=0\) is degenerate, emphasize early nonzero checkpoint.

### Case 7: Boundary artifacts dominate dense metrics

B-spline interpolation and warping can behave differently near image boundaries.

Workaround:

- Compute dense metrics inside a cropped valid region.
- Example: ignore a border of 5 to 10 voxels or one control-point spacing.

### Case 8: \(BKs\) and \(Bs\) have different physical units or scaling

Workaround:

- Ensure control-point coefficients and dense displacement are in the same units used during registration.
- Use cosine and residual metrics, which are less sensitive to global scaling.
- Report amplification separately.

---

## Sanity checks

Before running all experiments, run these checks on one small case.

### Check 1: Direct CP identity behavior

For direct CP, \(K=I\). Therefore:

\[
D_U = 1,
\]

\[
R_U = 0,
\]

\[
A_C = 1,
\]

\[
A_U = 1.
\]

This sanity check verifies the metric code.

### Check 2: Positive semidefinite kernel

For random vector \(v\):

\[
v^\top K v \geq 0.
\]

Numerically:

```python
v = torch.randn_like(s_flat)
Kv = K_matvec(f, params, v)
assert torch.dot(v, Kv) > -tolerance
```

Small negative values can occur due to numerical error, but large negative values indicate a bug.

### Check 3: Symmetry

For random \(v,w\):

\[
v^\top K w \approx w^\top K v.
\]

### Check 4: Finite-difference sanity check

For a small random parameter perturbation \(p\):

\[
Jp \approx \frac{h_{\theta+\epsilon p}(Q) - h_\theta(Q)}{\epsilon}.
\]

This checks the JVP direction.

### Check 5: Tiny-step loss decrease

For a very small CP update:

\[
C' = C - \epsilon s,
\]

loss should decrease approximately. For a very small SINR-kernel update:

\[
C' = C - \epsilon Ks,
\]

loss should also usually decrease if \(Ks\) is a descent direction.

---

## Compact main-text wording

Possible paragraph for the paper:

> To disentangle the B-spline deformation model from the INR coefficient generator, we compared the infinitesimal update induced by SINR with that of direct control-point optimization. Direct CP optimization gives \(\Delta U_{\mathrm{CP}}\propto -BB^\top G\), whereas SINR gives \(\Delta U_{\mathrm{SINR}}\propto -BJ_\theta J_\theta^\top B^\top G\), where \(J_\theta\) is the Jacobian of the coefficient INR with respect to its parameters. We therefore measured the agreement between \(BB^\top G\) and \(BJ_\theta J_\theta^\top B^\top G\) on the actual image-matching gradients. The two updates were highly aligned after scalar rescaling, indicating that the dominant update direction is determined by the B-spline parameterization. However, the coefficient INR kernel had a highly skewed spectrum and amplified the CP-space gradient, explaining the substantially smaller stable learning rate required by SINR compared with direct CP optimization.

Possible conclusion:

> These results suggest that SINR's useful inductive bias primarily comes from spline-space registration, while the neural coefficient generator introduces an ill-conditioned optimization metric. Direct CP retains the same B-spline deformation bias but removes this additional kernel, explaining its faster and more stable optimization.

---

## Minimal paper claim to avoid overstatement

Avoid claiming:

> \(BB^\top\) globally dominates \(BJ_\theta J_\theta^\top B^\top\).

That is a strong operator-level claim and hard to prove.

Instead claim:

> On the actual image-matching gradients encountered during registration, the SINR update \(BJ_\theta J_\theta^\top B^\top G\) is close to a scalar-rescaled version of the direct CP update \(BB^\top G\), while the coefficient INR kernel strongly amplifies and anisotropically rescales gradients.

This is precise, empirical, and directly supported by the proposed metrics.

---

## Codex action checklist

1. Locate SINR coefficient network \(h_\theta\) and control-point coordinates \(Q\).
2. Locate B-spline interpolation function used to decode CP coefficients to dense DVF.
3. Implement CP-gradient extraction by treating current SINR output \(C=h_\theta(Q)\) as a leaf variable.
4. Implement `K_matvec(f, params, v)` using VJP + JVP.
5. Implement metrics:
   - \(D_U = \cos(BKs, Bs)\),
   - \(R_U = \|BKs - \alpha Bs\|/\|BKs\|\),
   - \(A_C = \|Ks\|/\|s\|\),
   - \(A_U = \|BKs\|/\|Bs\|\),
   - optional \(\lambda_{\max}\),
   - optional \(\kappa_{\mathrm{eff}}\).
6. Add a CLI/script to run diagnostics at checkpoint \(t=0\) and early checkpoint \(t=50\).
7. Run on a small set of representative brain and lung cases.
8. Save CSV/JSONL logs.
9. Generate figures:
   - LR sweep curves,
   - update agreement/amplification bars,
   - optional kernel spectrum.
10. Write a short result summary using the interpretation rules above.

---

## Suggested file/script names

Possible additions to the codebase:

```text
experiments/kernel_diagnostics/run_sinr_cp_kernel_diagnostic.py
experiments/kernel_diagnostics/plot_kernel_diagnostics.py
experiments/kernel_diagnostics/plot_lr_sweep.py
experiments/kernel_diagnostics/README.md
```

Suggested output directory:

```text
results/kernel_diagnostics/
```

Suggested outputs:

```text
results/kernel_diagnostics/kernel_metrics.csv
results/kernel_diagnostics/lr_sweep_metrics.csv
results/kernel_diagnostics/fig_lr_sweep.png
results/kernel_diagnostics/fig_update_agreement.png
results/kernel_diagnostics/fig_kernel_spectrum.png
```

---

## Final concise objective for Codex

Implement and run a compact diagnostic experiment that tests whether SINR's behavior is mainly explained by the B-spline update operator while the INR coefficient generator worsens optimization conditioning. The main computation is to compare the direct CP update proxy \(Bs\) with the SINR update proxy \(BKs\), where \(s=\partial\mathcal{L}/\partial C\) and \(K=J_\theta J_\theta^\top\). Report update-direction agreement, scalar-rescaled residual, gradient amplification, and optionally the kernel spectrum/effective condition number. Use these results to explain why direct CP is stable at LR \(10^{-2}\) while SINR requires LR \(10^{-4}\).
