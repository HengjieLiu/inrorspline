# Codex Experiment Plan: Optimization-Kernel / Update-Alignment Analysis for IDIR, SINR, and Direct B-Spline Control Points

## 0. Purpose of this experiment

This experiment is intended as a compact mechanistic add-on to the main paper. The main paper focuses on matched validation of IDIR, SINR, and direct control-point optimization across two registration tasks:

1. **Inter-subject brain MR registration**
2. **Intra-subject lung inhale-exhale registration**

Empirically, we observe:

- On brain MR, SINR performs better than IDIR.
- On lung inhale-exhale, IDIR performs better than SINR at full resolution.
- SINR can get trapped in local minima for large lung motion unless multiresolution is used.
- Direct B-spline control-point optimization is comparable to, or better than, SINR in supervised DVF fitting.
- In supervised fitting to a strong SOTA teacher DVF, SINR/direct control-point optimization fit both brain and lung DVFs well.

The key scientific question is:

> If SINR/direct B-spline control-point optimization can fit the lung teacher DVF well, why does SINR fail during unsupervised image registration?

The hypothesis tested here is:

> The performance gap is not primarily caused by final DVF expressivity. Instead, IDIR, SINR, and direct B-spline control-point optimization induce different **displacement update directions** during image-based optimization. SINR's spline parameterization acts as a B-spline gradient aggregation/preconditioning operator. This is beneficial for small residual inter-subject brain motion, but can reduce the effective capture range for large inhale-exhale lung motion at full resolution.

This experiment should produce one compact paper figure/table showing:

1. **Supervised DVF fitting error** to a teacher field.
2. **Registration performance**.
3. **Early update alignment** between each method's actual displacement update and the remaining teacher residual deformation.
4. Optional: **SINR/direct-CP update similarity**, showing whether SINR's behavior is dominated by the spline parameterization.

The goal is not to develop a new theory-heavy section. The desired paper footprint is approximately **half a page to one page**.

---

## 1. Theoretical formulation to support the experiment

### 1.1 Dense displacement notation

Let:

- \(F\) be the fixed image.
- \(M\) be the moving image.
- \(U_t \in \mathbb{R}^{N \times d}\) be the dense displacement field at optimization iteration \(t\), evaluated on the fixed image grid.
- \(d\) is the spatial dimension, usually 2 or 3.
- \(N\) is the number of voxels/pixels in the evaluation mask.
- \(\mathcal{L}_{img}(F, M \circ (\mathrm{Id} + U_t))\) is the image similarity loss.
- \(G_t = \partial \mathcal{L}_{img} / \partial U_t\) is the dense image-matching gradient.

For a general parameterized displacement field \(U_\theta\), a small gradient step in parameter space induces a first-order update in displacement space:

\[
\Delta U_t
\approx
-\eta J_\theta J_\theta^\top G_t,
\]

where

\[
J_\theta = \frac{\partial U_\theta}{\partial \theta}.
\]

Therefore, the parameterization does not only define what fields can be represented. It also defines an operator \(J_\theta J_\theta^\top\) that maps image gradients into dense displacement updates.

This is the key conceptual distinction:

- **Supervised DVF fitting** tests whether the model can represent a target displacement.
- **Unsupervised image registration** tests whether the parameterization-induced optimization trajectory can reach a good displacement from identity.

---

### 1.2 IDIR update geometry

In IDIR, the dense displacement is directly represented by a coordinate INR:

\[
U = f_\theta(X),
\]

where \(X\) are image-grid coordinates.

The induced update is approximately:

\[
\Delta U_{\text{IDIR}}
\approx
-\eta K_{\text{INR}} G_t,
\]

where

\[
K_{\text{INR}} = J_f J_f^\top.
\]

Thus, IDIR follows the update geometry induced by the dense coordinate network.

---

### 1.3 Direct B-spline control-point update geometry

For direct B-spline control-point optimization,

\[
U = BC,
\]

where:

- \(C \in \mathbb{R}^{K \times d}\) are B-spline control-point displacements.
- \(B \in \mathbb{R}^{N \times K}\) is the B-spline interpolation matrix mapping control-point displacements to dense-grid displacements.

The gradient with respect to the control points is:

\[
\nabla_C \mathcal{L}
=
B^\top G_t.
\]

A gradient step on \(C\) gives:

\[
\Delta C
=
-\eta B^\top G_t.
\]

The induced dense-field update is:

\[
\Delta U_{\text{CP}}
=
B \Delta C
=
-\eta BB^\top G_t.
\]

Therefore, direct B-spline control-point optimization updates the dense field by applying the spline-induced operator \(BB^\top\) to the image-matching gradient. This operator aggregates local image forces over the support of the spline basis and redistributes the result as a smooth dense update.

Interpretation:

- In brain MR after affine alignment, the residual motion is usually smaller and more local. B-spline aggregation can suppress noisy or unstable local image forces.
- In lung inhale-exhale registration, the motion is larger and the correct correspondence can be outside the local capture range of the full-resolution image similarity. Local forces may be inconsistent or point to nearby wrong structures. B-spline aggregation can average or cancel these forces, producing a poor early update direction.

---

### 1.4 SINR update geometry

In SINR, the B-spline control points are generated by an INR:

\[
C = h_\theta(Q),
\]

where \(Q\) are the control-point coordinates. The dense displacement is then:

\[
U = B h_\theta(Q).
\]

The induced first-order update is approximately:

\[
\Delta U_{\text{SINR}}
\approx
-\eta B K_{\text{CP-INR}} B^\top G_t,
\]

where \(K_{\text{CP-INR}}\) is the kernel induced by the control-point INR.

Therefore, SINR combines:

1. The B-spline interpolation operator \(B\).
2. The control-point INR update kernel \(K_{\text{CP-INR}}\).
3. The B-spline image-force aggregation \(B^\top G_t\).

If SINR behaves similarly to direct B-spline control-point optimization, this suggests that the dominant bias is the spline parameterization, not the INR coefficient generator.

---

## 2. Core diagnostic metric: update alignment

### 2.1 Teacher field

Use an existing strong SOTA registration method to produce a teacher displacement field:

\[
U^\*.
\]

This teacher field is a pseudo-ground truth. It does not need to be perfect. It is used to define a plausible target direction for analyzing optimization behavior.

Important:

- \(U^\*\), IDIR fields, SINR fields, and direct-CP fields must all be converted to the same grid, same coordinate convention, and same physical units.
- Use the same forward/backward convention across methods.
- If the codebase uses normalized grid-sample coordinates internally, convert all fields to either voxel units or physical units before computing metrics.
- Use the same foreground mask for all methods.

---

### 2.2 Residual teacher deformation

For method \(m\), at checkpoint \(t\), define the remaining teacher residual:

\[
R_t^m = U^\* - U_t^m.
\]

This is the teacher displacement that remains unexplained by the current method's field.

---

### 2.3 Actual method update

Save dense displacement fields at checkpoints during registration. Define the actual field update as:

\[
\Delta U_t^m = U_{t+s}^m - U_t^m,
\]

where \(s\) is the checkpoint interval.

This metric uses the **actual optimizer update**. Therefore, it remains meaningful even if the method uses Adam, learning-rate schedules, regularization, or other non-idealities not captured exactly by the simple gradient-descent derivation.

Recommended checkpoint intervals:

- If registration runs for approximately 1,000 to 5,000 iterations: save every 25 to 100 iterations.
- If registration runs for fewer iterations: save every 5 to 20 iterations.
- Keep at least 10 checkpoints in the early phase.
- Do not save every iteration if dense 3D fields are too large.

---

### 2.4 Update alignment metric

Compute the cosine similarity between the actual update and the teacher residual:

\[
A_t^m
=
\frac{
\langle \Delta U_t^m, R_t^m \rangle_\Omega
}{
\|\Delta U_t^m\|_\Omega \|R_t^m\|_\Omega + \epsilon
}.
\]

Here \(\Omega\) is the foreground mask, and the inner product is:

\[
\langle A, B \rangle_\Omega
=
\sum_{x \in \Omega}
A(x)^\top B(x).
\]

Interpretation:

- \(A_t^m > 0\): the update moves toward the teacher deformation.
- \(A_t^m \approx 0\): the update is unrelated to the teacher deformation.
- \(A_t^m < 0\): the update moves away from the teacher deformation.

Report the early-stage average:

\[
A_{\text{early}}^m
=
\frac{1}{|\mathcal{T}_{early}|}
\sum_{t \in \mathcal{T}_{early}} A_t^m.
\]

Recommended early phase:

- Default: first 20% of optimization checkpoints.
- Alternative: first 500 iterations if all methods use the same number of iterations.
- For multiresolution methods, report either:
  - alignment during the first coarse level, and/or
  - alignment after converting all fields to full-resolution physical coordinates.

---

### 2.5 Optional cumulative update alignment

If per-checkpoint updates are noisy, compute cumulative early update alignment:

\[
A_{\text{cum-early}}^m
=
\frac{
\langle U_{T_e}^m - U_0^m,\; U^\* - U_0^m \rangle_\Omega
}{
\|U_{T_e}^m - U_0^m\|_\Omega
\|U^\* - U_0^m\|_\Omega
+
\epsilon
},
\]

where \(T_e\) is the end of the early phase.

This is less noisy and may be useful for a compact table.

---

## 3. Optional diagnostic metric: SINR/direct-CP update similarity

To test whether SINR behaves like direct B-spline control-point optimization, compute:

\[
S_t =
\frac{
\langle \Delta U_t^{\text{SINR}}, \Delta U_t^{\text{CP}} \rangle_\Omega
}{
\|\Delta U_t^{\text{SINR}}\|_\Omega
\|\Delta U_t^{\text{CP}}\|_\Omega
+
\epsilon
}.
\]

Because SINR and direct CP may not have identical learning-rate schedules or optimization speed, use one of the following alignment strategies:

1. **Same checkpoint index:** compare updates at the same iteration.
2. **Same normalized progress:** compare updates at the same fraction of total optimization.
3. **Early cumulative update:** compare \(U_{T_e}-U_0\) for both methods.

Expected interpretation:

- High SINR/direct-CP update similarity suggests the spline parameterization dominates SINR's optimization behavior.
- Low similarity suggests the INR coefficient generator or optimizer settings substantially change the update direction.

This metric is optional but highly useful because the existing supervised fitting result already suggests that direct CP and SINR behave similarly.

---

## 4. Required experiments

### Experiment A: supervised teacher-DVF fitting

This is either already available or should be rerun under matched settings.

#### Goal

Verify that spline-based methods can represent the teacher fields on both datasets.

#### Methods

Run supervised fitting to the teacher field \(U^\*\):

1. IDIR-style dense INR.
2. SINR-style spline-enhanced INR.
3. Direct B-spline control-point optimization.
4. Optional: dense grid with matched smoothness regularization.

#### Loss

Use MSE or relative MSE:

\[
\mathcal{L}_{fit}
=
\frac{1}{|\Omega|}
\sum_{x \in \Omega}
\|U_\theta(x) - U^\*(x)\|^2.
\]

Also report relative RMSE:

\[
\text{relRMSE}
=
\frac{\|U_\theta - U^\*\|_\Omega}
{\|U^\*\|_\Omega + \epsilon}.
\]

#### Outputs

For each dataset and method:

- final fitting MSE,
- relative RMSE,
- optional endpoint error statistics:
  - mean EPE,
  - median EPE,
  - 95th percentile EPE,
- optional Jacobian/folding statistics.

#### Expected result

- SINR and direct CP should fit both brain and lung teacher DVFs well.
- If SINR/direct CP fit lung well but fail in unsupervised lung registration, this supports the claim that lung failure is not primarily due to limited spline expressivity.

---

### Experiment B: unsupervised registration with update-alignment logging

#### Goal

Test whether each method's early image-based update points toward the teacher residual deformation.

#### Methods

Run matched registration experiments for:

1. IDIR.
2. SINR.
3. Direct B-spline control-point optimization.
4. Optional but important: SINR + multiresolution for lung.
5. Optional: IDIR + matched regularization or IDIR + spline projection.

#### During registration, log

At every checkpoint:

- iteration number,
- current image similarity loss,
- total loss,
- dense displacement field \(U_t\),
- optional:
  - regularization loss,
  - Jacobian/folding statistics,
  - Dice/TRE if available,
  - current learning rate,
  - current resolution level for multiresolution.

The dense field should be saved on the fixed image grid, in common units.

#### After registration, compute

For each pair, method, and checkpoint interval:

- actual update:
  \[
  \Delta U_t^m = U_{t+s}^m - U_t^m
  \]
- residual to teacher:
  \[
  R_t^m = U^\* - U_t^m
  \]
- update alignment:
  \[
  A_t^m
  \]
- early average:
  \[
  A_{\text{early}}^m
  \]
- cumulative early alignment:
  \[
  A_{\text{cum-early}}^m
  \]
- optional final alignment:
  \[
  A_{\text{late}}^m
  \]

#### Outputs

For each dataset and method:

- final registration metric:
  - Dice if segmentation labels are available,
  - TRE if landmarks are available,
  - image similarity metric,
  - final teacher-field distance if relevant,
- early update alignment,
- cumulative early update alignment,
- optional alignment curve over iterations.

#### Expected result

Brain MR:

- SINR/direct CP should have higher early update alignment than IDIR.
- This should match their better final registration performance.
- SINR and direct CP may have similar update trajectories.

Lung inhale-exhale:

- IDIR should have higher early update alignment than SINR/direct CP at full resolution.
- SINR/direct CP may show low, unstable, or near-zero early alignment despite good supervised DVF fitting.
- SINR + multiresolution should improve early alignment.

This would support:

> SINR has sufficient representation capacity for lung-like fields, but its full-resolution image-based update direction is poorly aligned with the large respiratory residual motion. Multiresolution improves the update direction by increasing the capture range of the image loss.

---

### Experiment C: SINR/direct-CP update similarity

#### Goal

Determine whether SINR's behavior is mainly caused by the spline parameterization.

#### Methods

Use saved checkpoint fields from Experiment B.

Compute:

\[
S_t =
\cos(\Delta U_t^{\text{SINR}}, \Delta U_t^{\text{CP}}).
\]

Also compute cumulative early similarity:

\[
S_{\text{cum-early}}
=
\cos(
U_{T_e}^{\text{SINR}} - U_0^{\text{SINR}},
U_{T_e}^{\text{CP}} - U_0^{\text{CP}}
).
\]

#### Expected result

- If SINR and direct CP have high update similarity, then the spline basis is likely the dominant source of SINR's optimization bias.
- If they differ substantially, inspect:
  - SINR control-point INR architecture,
  - learning-rate differences,
  - initialization,
  - regularization,
  - grid spacing,
  - optimizer.

---

### Experiment D: optional gradient-operator diagnostic

This experiment is optional. Do it only if extracting dense image gradients is easy.

#### Goal

Directly test whether the B-spline operator \(BB^\top\) changes image-force alignment.

#### Required tensors

At selected checkpoints, compute or save:

\[
G_t = \partial \mathcal{L}_{img}/\partial U_t.
\]

Then compute:

1. Raw dense gradient update:
   \[
   \Delta U_{\text{raw}} = -G_t.
   \]
2. B-spline-filtered gradient update:
   \[
   \Delta U_{\text{spline}} = -BB^\top G_t.
   \]

Then compute alignment with the teacher residual:

\[
\cos(-G_t, R_t)
\]

and

\[
\cos(-BB^\top G_t, R_t).
\]

#### Expected result

Brain MR:

- \( -BB^\top G_t \) should align better with the teacher residual than the raw dense gradient.

Lung:

- At full resolution, \( -BB^\top G_t \) may align poorly because local image forces are inconsistent under large motion.
- At coarse resolution, alignment should improve.

#### Implementation note

This can be expensive in 3D. It is not required for the main paper. The actual-update alignment metric is enough because it already captures what the optimizer did.

---

## 5. Implementation details for Codex

### 5.1 Files and modules to locate in the codebase

Codex should identify the existing modules/scripts for:

- IDIR registration training.
- SINR registration training.
- Direct control-point optimization.
- Supervised DVF fitting.
- Warping and displacement-field evaluation.
- Dataset loaders for brain MR and lung inhale-exhale.
- SOTA teacher DVF loading.
- Evaluation metrics: Dice, TRE, image similarity, Jacobian/folding.
- Multiresolution registration if already implemented.

Do not rewrite the whole training pipeline. Add a small logging/evaluation layer around the existing code.

---

### 5.2 New script/config names

Suggested additions:

```text
experiments/
  update_alignment/
    run_registration_with_field_logging.py
    compute_update_alignment.py
    plot_update_alignment.py
    summarize_update_alignment.py
    configs/
      brain_idir.yaml
      brain_sinr.yaml
      brain_direct_cp.yaml
      lung_idir.yaml
      lung_sinr.yaml
      lung_direct_cp.yaml
      lung_sinr_multires.yaml
```

If the repository has an existing experiment system, follow the existing style instead of creating a new folder.

---

### 5.3 Field checkpoint format

For each pair and method, save:

```text
results/update_alignment/
  {dataset}/
    {pair_id}/
      {method}/
        config.yaml
        metrics.csv
        fields/
          iter_000000.npz
          iter_000100.npz
          iter_000200.npz
          ...
        final_field.npz
```

Each field file should contain:

```python
{
    "u": dense_displacement,        # shape: [D, H, W, dim] or [H, W, dim]
    "iteration": iteration_number,
    "spacing": voxel_spacing,
    "units": "voxel" or "physical",
    "direction": "fixed_to_moving" or "moving_to_fixed",
    "resolution_level": optional_int,
}
```

Use `.pt` files if the codebase is PyTorch-heavy. Use `.npz` if cross-script analysis is easier.

Important: fields must be comparable across methods. If one method outputs normalized coordinates and another outputs voxel displacements, convert them before saving or record metadata and convert during analysis.

---

### 5.4 CSV schema for registration logging

Create or extend a `metrics.csv` file with columns:

```text
dataset
pair_id
method
iteration
resolution_level
image_loss
regularization_loss
total_loss
dice
tre
jacobian_negative_percent
field_norm_mean
field_norm_p95
field_path
```

Fill unavailable metrics with empty values or `NaN`.

---

### 5.5 CSV schema for update-alignment output

Create:

```text
results/update_alignment/update_alignment_summary.csv
```

with columns:

```text
dataset
pair_id
method
teacher_method
iteration_start
iteration_end
early_phase
alignment
cumulative_alignment
update_norm
residual_norm
final_registration_metric
final_fit_relrmse
notes
```

Also create a per-iteration file:

```text
results/update_alignment/update_alignment_timeseries.csv
```

with columns:

```text
dataset
pair_id
method
teacher_method
iteration_start
iteration_end
alignment
update_norm
residual_norm
image_loss_start
image_loss_end
resolution_level_start
resolution_level_end
```

For SINR/direct-CP similarity, create:

```text
results/update_alignment/sinr_cp_similarity.csv
```

with columns:

```text
dataset
pair_id
iteration_start
iteration_end
similarity
sinr_update_norm
cp_update_norm
notes
```

---

### 5.6 Foreground mask

Use a foreground mask for all inner products.

Recommended:

- Brain: brain mask or fixed-image foreground mask.
- Lung: lung mask or union of fixed/moving lung masks.
- If no mask is available, compute a simple intensity-based foreground mask but record this in the output.

Do not let large image backgrounds dominate the cosine similarities.

---

### 5.7 Coordinate and sign checks

Before computing update alignment, run sanity checks.

#### Check 1: teacher field sign

Warp the moving image using \(U^\*\). Confirm that the image similarity to the fixed image improves compared with identity.

If not, the teacher field may have the opposite direction. Try \(-U^\*\) or invert/convert according to the repository convention.

#### Check 2: units

Compute field magnitude statistics for each method and teacher:

```text
mean ||U||
median ||U||
p95 ||U||
max ||U||
```

If one method has magnitudes approximately 2x, 10x, or 1/image-size times another, there is probably a units mismatch.

#### Check 3: grid orientation

Visualize one slice of:

- teacher field magnitude,
- IDIR final field magnitude,
- SINR final field magnitude.

The anatomy-correlated motion pattern should appear in the same location and direction.

#### Check 4: zero-update instability

If \(\|\Delta U_t\|\) is extremely small, cosine similarity is unstable. Skip intervals where:

\[
\|\Delta U_t\|_\Omega < \tau.
\]

Suggested threshold:

```python
tau = 1e-6 * sqrt(number_of_masked_voxels)
```

or a dataset-specific small threshold based on field units.

---

## 6. Pseudocode

### 6.1 Field logging during registration

```python
def run_registration_with_logging(method, pair, config):
    model = build_model(method, config)
    optimizer = build_optimizer(model, config)

    fixed, moving, mask = load_pair(pair)
    teacher = load_teacher_field(pair)

    field_logs = []
    metrics_logs = []

    for iteration in range(config.num_iterations + 1):
        optimizer.zero_grad()

        u = model.evaluate_dense_field(fixed_grid=True)
        warped = warp(moving, u)

        image_loss = image_similarity(fixed, warped, mask=mask)
        reg_loss = regularization(u, model=model, config=config)
        total_loss = image_loss + reg_loss

        if iteration % config.log_field_every == 0:
            field_path = save_dense_field(
                u=u.detach(),
                iteration=iteration,
                pair_id=pair.id,
                method=method.name,
                metadata={
                    "spacing": pair.spacing,
                    "units": config.field_units,
                    "direction": config.field_direction,
                    "resolution_level": current_resolution_level(),
                },
            )

            metrics_logs.append({
                "dataset": pair.dataset,
                "pair_id": pair.id,
                "method": method.name,
                "iteration": iteration,
                "image_loss": float(image_loss.detach()),
                "regularization_loss": float(reg_loss.detach()),
                "total_loss": float(total_loss.detach()),
                "field_path": field_path,
            })

        if iteration == config.num_iterations:
            break

        total_loss.backward()
        optimizer.step()

    save_metrics_csv(metrics_logs)
```

---

### 6.2 Update alignment computation

```python
def masked_inner_product(a, b, mask):
    # a, b: [..., dim]
    # mask: [...]
    return ((a * b).sum(dim=-1) * mask).sum()

def masked_norm(a, mask, eps=1e-12):
    return torch.sqrt(masked_inner_product(a, a, mask) + eps)

def cosine_field(a, b, mask, eps=1e-12):
    numerator = masked_inner_product(a, b, mask)
    denominator = masked_norm(a, mask, eps) * masked_norm(b, mask, eps) + eps
    return numerator / denominator

def compute_update_alignment(field_paths, teacher_field, mask):
    rows = []

    fields = [load_field(p) for p in sorted(field_paths)]

    for f0, f1 in zip(fields[:-1], fields[1:]):
        u0 = convert_to_common_units_and_grid(f0.u, f0.metadata)
        u1 = convert_to_common_units_and_grid(f1.u, f1.metadata)
        teacher = convert_to_common_units_and_grid(teacher_field.u, teacher_field.metadata)

        delta_u = u1 - u0
        residual = teacher - u0

        update_norm = masked_norm(delta_u, mask)
        residual_norm = masked_norm(residual, mask)

        if update_norm < MIN_UPDATE_NORM:
            alignment = float("nan")
            note = "skipped_small_update"
        else:
            alignment = cosine_field(delta_u, residual, mask).item()
            note = ""

        rows.append({
            "iteration_start": f0.iteration,
            "iteration_end": f1.iteration,
            "alignment": alignment,
            "update_norm": update_norm.item(),
            "residual_norm": residual_norm.item(),
            "notes": note,
        })

    return rows
```

---

### 6.3 Cumulative early alignment

```python
def compute_cumulative_early_alignment(fields, teacher, mask, early_fraction=0.2):
    n = len(fields)
    early_idx = max(1, int((n - 1) * early_fraction))

    u0 = fields[0].u
    ue = fields[early_idx].u

    cumulative_update = ue - u0
    initial_residual = teacher - u0

    return cosine_field(cumulative_update, initial_residual, mask).item()
```

---

### 6.4 SINR/direct-CP update similarity

```python
def compute_update_similarity(sinr_fields, cp_fields, mask):
    rows = []

    # Assumption: same number of checkpoints or already resampled to same normalized progress.
    for s0, s1, c0, c1 in zip(
        sinr_fields[:-1],
        sinr_fields[1:],
        cp_fields[:-1],
        cp_fields[1:],
    ):
        delta_sinr = s1.u - s0.u
        delta_cp = c1.u - c0.u

        similarity = cosine_field(delta_sinr, delta_cp, mask).item()

        rows.append({
            "iteration_start_sinr": s0.iteration,
            "iteration_end_sinr": s1.iteration,
            "iteration_start_cp": c0.iteration,
            "iteration_end_cp": c1.iteration,
            "similarity": similarity,
            "sinr_update_norm": masked_norm(delta_sinr, mask).item(),
            "cp_update_norm": masked_norm(delta_cp, mask).item(),
        })

    return rows
```

---

## 7. Statistical summary

Use pairwise aggregation because the same image pairs are registered by all methods.

For each dataset and method, report:

```text
mean ± standard deviation
median [IQR]
paired bootstrap confidence interval
```

Recommended primary table columns:

```text
Dataset
Method
Teacher fit relRMSE ↓
Final registration metric ↓/↑
Early update alignment ↑
Cumulative early alignment ↑
```

For final registration metric, choose the dataset-appropriate primary metric:

- Brain: Dice, TRE, or target registration error if landmarks/labels exist.
- Lung: landmark TRE if DIR-Lab-style landmarks exist, or Dice if lung/lobe masks exist.

If possible, use paired statistical tests or bootstrap confidence intervals:

- IDIR vs SINR
- SINR vs direct CP
- SINR full-res vs SINR multiresolution on lung

Do not overemphasize p-values if the sample size is small. The mechanistic pattern is more important.

---

## 8. Expected results

### 8.1 Brain MR

Expected pattern:

| Method | Teacher fitting | Registration | Early update alignment |
|---|---|---|---|
| IDIR | good | worse than SINR | lower |
| SINR | good or best | best | higher |
| Direct CP | good | similar to SINR | similar to SINR |

Expected interpretation:

> For affine-prealigned inter-subject brain MR, the residual motion is relatively small and local. B-spline aggregation suppresses noisy local image forces and yields updates that are better aligned with the teacher residual. This explains why SINR/direct CP can outperform dense IDIR.

---

### 8.2 Lung inhale-exhale

Expected pattern:

| Method | Teacher fitting | Registration | Early update alignment |
|---|---|---|---|
| IDIR | good | best full-res | higher |
| SINR full-res | good | worse / local minima | low or unstable |
| Direct CP full-res | good | similar to SINR | low or unstable |
| SINR multiresolution | good | improved | higher than SINR full-res |

Expected interpretation:

> The spline model can represent the respiratory teacher field, but full-resolution image-based optimization produces poorly aligned updates. The problem is therefore not spline expressivity; it is the interaction between large motion, local image-loss capture range, and spline-induced gradient aggregation. Multiresolution improves the update alignment, supporting this explanation.

---

### 8.3 SINR/direct-CP similarity

Expected pattern:

- SINR/direct-CP update similarity is high, especially early.
- This supports that SINR's behavior is dominated by the spline parameterization.

Possible interpretation:

> Since direct CP removes the INR coefficient generator but retains the spline interpolation, similar behavior between SINR and direct CP indicates that the spline parameterization is the main source of the observed task-dependent optimization bias.

---

## 9. Unexpected outcomes and workarounds

### 9.1 Unexpected: SINR has high update alignment on lung but still poor registration

Possible reasons:

1. The teacher field is not the same solution preferred by the image loss.
2. SINR initially moves in the right direction but later becomes trapped.
3. Update magnitude is too small despite good direction.
4. Regularization or optimizer settings prevent sufficient progress.

Workarounds:

- Report both alignment and update norm.
- Add a metric for progress toward teacher:
  \[
  \|U^\* - U_t\| - \|U^\* - U_{t+s}\|.
  \]
- Plot alignment over the full optimization, not only early.
- Compare cumulative update magnitude:
  \[
  \|U_{T_e} - U_0\|.
  \]
- Check whether SINR makes small correct updates but too slowly.
- Test a higher learning rate or different control-point grid spacing as an appendix/debug result.

---

### 9.2 Unexpected: IDIR has lower update alignment on lung but better registration

Possible reasons:

1. The teacher field does not represent the IDIR-preferred solution.
2. IDIR achieves better image matching through a different but valid deformation.
3. The teacher has sign/grid mismatch.
4. Landmarks/labels reward IDIR even if teacher alignment is lower.

Workarounds:

- Verify teacher sign and unit conventions.
- Use multiple teacher fields if available.
- Compute alignment to final successful IDIR field as an auxiliary diagnostic.
- Compare alignment to landmark-derived sparse displacement if landmarks exist.
- Report this honestly as "teacher-alignment diagnostic" rather than true ground-truth alignment.

---

### 9.3 Unexpected: direct CP does not behave like SINR

Possible reasons:

1. SINR INR coefficient generator imposes a strong additional bias.
2. Different optimizer or learning rate.
3. Different initialization.
4. Direct CP has too many/few degrees of freedom compared with SINR.
5. Direct CP uses different regularization.

Workarounds:

- Match the B-spline grid spacing exactly.
- Match initialization: zero coefficients if possible.
- Match optimizer type and learning rate scale as much as practical.
- Disable extra regularization for a diagnostic run.
- Add SINR with a very wide/high-capacity coefficient network to test whether it approaches direct CP.
- Add direct CP with coefficient smoothing regularization if SINR appears smoother.

---

### 9.4 Unexpected: supervised fitting shows SINR/direct CP cannot fit lung teacher well

This would weaken the current hypothesis but is still informative.

Possible reasons:

1. B-spline grid spacing is too coarse.
2. Teacher field contains sliding or high-frequency components.
3. Field units/grid are mismatched.
4. Fitting optimization failed.

Workarounds:

- Fit with finer B-spline grid spacing.
- Report spline projection error as a function of grid spacing.
- Separate fitting error inside lung parenchyma vs near sliding boundaries.
- Increase fitting iterations or adjust learning rate.
- Use direct least-squares B-spline projection if available.

If fitting remains poor, revise the interpretation:

> SINR's lung limitation may involve both reduced representation of high-frequency/sliding components and image-based optimization difficulty.

But based on existing results, this is not expected.

---

### 9.5 Unexpected: alignment curves are noisy

Possible reasons:

1. Checkpoint interval is too short.
2. Adam/momentum produces oscillatory updates.
3. Image loss gradients are noisy.
4. Deformation update norm is very small.

Workarounds:

- Use cumulative early alignment.
- Increase checkpoint interval.
- Smooth the alignment curve with a small moving average for visualization only.
- Report median alignment over early intervals.
- Skip intervals with update norm below threshold.
- Use field difference over larger windows:
  \[
  \Delta U_t = U_{t+q} - U_t
  \]
  with \(q > s\).

---

### 9.6 Unexpected: multiresolution fields are hard to compare

Possible reasons:

1. Fields are represented at different resolutions.
2. Each pyramid level has different coordinate normalization.
3. Control-point grids change by level.

Workarounds:

- Always evaluate and save dense fields on the **full fixed-image grid** in physical or voxel units.
- Add a utility:
  ```python
  evaluate_dense_field_on_fullres_grid(model, fixed_fullres_grid)
  ```
- For each checkpoint, record `resolution_level`.
- Report alignment separately by resolution level if global iteration comparison is confusing.

---

### 9.7 Unexpected: storing dense fields is too expensive

Workarounds:

- Save fields every 100 or 200 iterations instead of every 10.
- Save compressed `.npz`.
- Save `float16` fields for alignment if precision is sufficient.
- Compute alignment online during training and save only scalar metrics.
- Save sparse sampled fields over a fixed mask if memory is severe.
- For the main paper, use a representative subset of pairs.

Recommended minimum:

- 5 to 10 brain pairs.
- 5 to 10 lung pairs.
- At least 10 early checkpoints per method.

---

### 9.8 Unexpected: results vary strongly by image pair

Workarounds:

- Use paired plots: each pair has connected points for IDIR vs SINR.
- Report median and IQR in addition to mean.
- Stratify lung pairs by teacher displacement magnitude:
  - small motion,
  - medium motion,
  - large motion.
- Test whether SINR failure correlates with:
  \[
  P_{95} \|U^\*\|
  \]
  or with the ratio of motion magnitude to image feature scale.

This could become an additional insight:

> SINR's full-resolution failure appears primarily in high-motion lung pairs.

---

## 10. Minimal version for the main paper

If time is limited, run only the following:

1. Existing supervised teacher-DVF fitting result:
   - IDIR, SINR, direct CP.
2. Registration with field logging:
   - IDIR, SINR, direct CP.
   - Brain and lung.
3. Compute:
   - early update alignment,
   - cumulative early update alignment,
   - final registration metric.
4. Optional but strongly recommended:
   - SINR multiresolution on lung.

Main table:

```text
Dataset | Method | Fit relRMSE ↓ | Final registration metric ↓/↑ | Early update alignment ↑
```

Main figure:

- One line plot of update alignment over normalized optimization progress:
  - left: brain,
  - right: lung.
- Curves: IDIR, SINR, direct CP, optional SINR+multires.

This is sufficient to support the paper claim.

---

## 11. Stronger optional extension

If the minimum experiment works well and there is time, add:

1. SINR/direct-CP update similarity.
2. Field-magnitude statistics of the teacher DVF:
   - mean,
   - median,
   - p95,
   - max.
3. Low-frequency energy or smoothness statistics.
4. Lung motion stratification by teacher displacement magnitude.
5. Gradient-operator diagnostic:
   - raw gradient alignment,
   - B-spline-filtered gradient alignment.

These can go into supplementary material.

---

## 12. Suggested paper text after results are available

### Method paragraph

> To distinguish representational capacity from optimization behavior, we analyzed the displacement update induced by each parameterization. Let \(U_\theta\) be the dense displacement field and \(G_t=\partial \mathcal{L}_{img}/\partial U_t\) the dense image-matching gradient. A small gradient step in parameter space induces \(\Delta U_t \approx -\eta J_\theta J_\theta^\top G_t\), where \(J_\theta=\partial U_\theta/\partial \theta\). Thus, each parameterization defines an operator that maps image forces to displacement updates. For direct B-spline control-point optimization, \(U=BC\), yielding \(\Delta U_t=-\eta BB^\top G_t\). Therefore, the B-spline representation aggregates image forces over the support of the spline basis before updating the dense field. SINR has the form \(U=Bh_\theta(Q)\), giving \(\Delta U_t\approx-\eta B K_{\text{CP-INR}}B^\top G_t\), whereas IDIR uses \(U=f_\theta(X)\) and is governed by the dense INR kernel.

### Experiment paragraph

> We measured whether each method's actual image-based update moved toward a teacher displacement field \(U^\*\) obtained from a strong conventional registration method. At checkpoint \(t\), we computed the residual \(R_t=U^\*-U_t\) and the actual update \(\Delta U_t=U_{t+s}-U_t\). We report the update alignment \(A_t=\langle \Delta U_t,R_t\rangle/(\|\Delta U_t\|\|R_t\|)\), averaged over the early optimization stage. This diagnostic separates expressivity from optimization: a method may fit \(U^\*\) under supervised MSE while still having poor update alignment during image-based registration from identity.

### Results paragraph template

> SINR and direct control-point optimization achieved low supervised fitting error to the teacher DVF on both datasets, indicating that the spline parameterization could represent the target fields. However, update alignment revealed a task-dependent optimization behavior. On brain MR, SINR/direct CP produced early updates that were more aligned with the teacher residual than IDIR, consistent with their better registration accuracy. On lung inhale-exhale registration, full-resolution SINR/direct CP had poor early alignment despite good supervised fitting, whereas IDIR showed higher alignment and better final registration. Multiresolution improved SINR's early alignment on lung. These results suggest that SINR's lung failure is not primarily due to limited spline expressivity, but to the interaction between spline-induced gradient aggregation and the reduced capture range of full-resolution image matching under large respiratory motion.

---

## 13. Codex execution checklist

### Phase 1: inspect and reuse existing code

- [ ] Locate IDIR registration script.
- [ ] Locate SINR registration script.
- [ ] Locate direct CP optimization script.
- [ ] Locate supervised DVF fitting script.
- [ ] Locate teacher DVF loader.
- [ ] Locate evaluation metrics.
- [ ] Locate image warping utilities.
- [ ] Identify field units and direction conventions.

### Phase 2: add field logging

- [ ] Add dense field evaluation at checkpoints.
- [ ] Save fields with metadata.
- [ ] Save registration metrics with field paths.
- [ ] Ensure fields are saved in common units or convertible metadata is saved.

### Phase 3: add alignment analysis

- [ ] Implement field loading and conversion.
- [ ] Implement foreground mask loading.
- [ ] Implement masked field cosine similarity.
- [ ] Implement per-interval update alignment.
- [ ] Implement cumulative early update alignment.
- [ ] Implement SINR/direct-CP update similarity.
- [ ] Save summary CSVs.

### Phase 4: run experiments

- [ ] Brain IDIR.
- [ ] Brain SINR.
- [ ] Brain direct CP.
- [ ] Lung IDIR.
- [ ] Lung SINR.
- [ ] Lung direct CP.
- [ ] Lung SINR multiresolution if available.
- [ ] Supervised teacher-DVF fitting results for all methods.

### Phase 5: summarize

- [ ] Make table with fit error, final registration, and early alignment.
- [ ] Plot update alignment over iterations for brain and lung.
- [ ] Plot SINR/direct-CP similarity if useful.
- [ ] Save all plots as PDF/PNG.
- [ ] Write a short results summary matching the paper text above.

---

## 14. Success criteria

This experiment is successful if it can support the following three statements:

1. **Spline expressivity is not the main bottleneck.**
   SINR/direct CP can fit teacher DVFs on both brain and lung under supervised MSE.

2. **Optimization behavior is task-dependent.**
   The method with better registration performance has better early update alignment with the teacher residual.

3. **SINR behavior is largely spline-driven.**
   SINR and direct CP show similar fitting behavior and/or similar update directions, suggesting that the spline parameterization is the dominant inductive bias.

The strongest expected conclusion is:

> The difference between IDIR and SINR arises less from what deformation fields they can ultimately represent and more from how their parameterizations transform image-matching gradients into displacement updates. SINR's B-spline-induced update geometry is beneficial for small residual brain MR deformation, but can reduce full-resolution capture range for large respiratory lung motion unless multiresolution is used.
