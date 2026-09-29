# Codex Task: Regenerate Simplified Presentation Versions of ARC Plots

## Goal

I am preparing a 6-minute presentation and need simplified, presentation-quality versions of the registration-performance plots from my continuous DIR project.

Please **find and reuse the existing Python plotting code and saved numerical data in this workspace** that generated the current OASIS and DIR-LAB Figure 2 plots.

Do **not** manually transcribe values from the paper/PDF and do not recompute registration experiments.

The goal is purely to:

1. select a subset of the existing curves,
2. rename them,
3. recolor them consistently with my method schematic,
4. improve presentation readability,
5. export both static figures and PowerPoint animation layers.

The scientific data must remain unchanged.

---

# 1. Methods to retain

There should be exactly four main method curves, always in this conceptual order:

1. **INR-Dense**
2. **INR-BSCP**
3. **D-BSCP**
4. **MR-D-BSCP**

This order is important because they will appear sequentially during my presentation.

---

# 2. Method colors

I want the plot colors to match the four method blocks in my presentation schematic.

Use the **outline/stroke color of the corresponding blocks**, not the pale block background.

Preferred approach:

- First search the existing workspace/source code for the color constants used to draw the four method blocks.
- If those colors are available, reuse them exactly.

If the original constants cannot be located, use these approximate fallback colors:

```python
COLORS = {
    "INR-Dense":   "#4A90E2",   # blue
    "INR-BSCP":    "#9A6FE3",   # purple
    "D-BSCP":      "#F06A2B",   # orange
    "MR-D-BSCP":   "#D39A20",   # gold / mustard
}
```

The important requirement is:

- INR-Dense = blue
- INR-BSCP = purple
- D-BSCP = orange
- MR-D-BSCP = gold

Do not reuse the current arbitrary matplotlib color sequence.

All versions of the same method—curve, markers, and uncertainty ellipses—must use the same color.

---

# 3. OASIS plot

## Keep only this panel

Keep only:

**Dice vs foreground SD(log J), OASIS N=100**

Completely omit the OASIS HD95 panel.

## Exact curves to retain

From the current data:

```text
INR-Dense
INR-BSCP (cps2)
D-BSCP (cps2)
MR-D-BSCP (8-4-2)
```

Relabel them for presentation as:

```text
INR-Dense
INR-BSCP
D-BSCP
MR-D-BSCP
```

### Important

Remove the OASIS:

```text
INR-BSCP (cps4)
D-BSCP (cps4)
```

curves entirely.

Do not average cps2 and cps4.

Use the existing **cps2 data only** for INR-BSCP and D-BSCP.

For MR-D-BSCP use the existing **8-4-2** data but do not display the schedule in the legend.

---

# 4. DIR-LAB lung CT plot

Keep:

**TRE vs foreground SD(log J), DIR-LAB 4DCT N=10**

## Exact curves to retain

Use:

```text
INR-Dense
INR-BSCP (cps4)
D-BSCP (cps4)
MR-D-BSCP (32-16-8-4) with the pTVReg loss
```

Relabel them simply as:

```text
INR-Dense
INR-BSCP
D-BSCP
MR-D-BSCP
```

### Important

For INR-BSCP and D-BSCP, keep the existing cps4 numerical data, but remove `(cps4)` from the displayed label.

For MR-D-BSCP:

- use the **orange/original pTVReg-loss version**, i.e. the curve currently labeled something like:

```text
MR-D-BSCP (32-16-8-4)†
```

- do not use the other MR-D-BSCP curve for the main presentation figure.
- rename this selected curve simply:

```text
MR-D-BSCP
```

Do not display:

```text
32-16-8-4
†
pTVReg loss
```

in the main legend.

Keep the **pTVReg reference star** available for the final comparison figure, labeled simply:

```text
pTVReg
```

---

# 5. Presentation-friendly text

Please simplify the titles and labels.

## OASIS

Preferred title:

```text
OASIS Brain MRI (N=100)
```

Y-axis:

```text
Mean Dice ↑
```

X-axis:

```text
Foreground SD(log J)
```

## Lung

Preferred title:

```text
DIR-LAB Lung 4DCT (N=10)
```

Y-axis:

```text
TRE (mm) ↓
```

X-axis:

```text
Foreground SD(log J)
```

Avoid long manuscript-style titles such as:

```text
TRE vs Foreground SDlogJ (DIR-LAB 4DCT N=10), selected
```

This is for PowerPoint, not for the manuscript.

---

# 6. Overall styling

These figures will be displayed during a Zoom presentation, so readability is more important than manuscript-style compactness.

Use approximately:

```python
title_fontsize = 20-22
axis_label_fontsize = 18-20
tick_fontsize = 15-17
legend_fontsize = 14-16

linewidth = 2.5-3.0
markersize = 6-8
```

Use a clean white background.

Keep subtle grid lines, but make them light.

Example:

```python
ax.grid(alpha=0.18)
```

Do not use seaborn.

Do not introduce a new plotting style globally that could unexpectedly change the data presentation.

---

# 7. Legend

Use only these four main labels:

```text
INR-Dense
INR-BSCP
D-BSCP
MR-D-BSCP
```

For the final lung comparison, also show:

```text
pTVReg
```

as the reference star.

No labels should contain:

```text
cps2
cps4
8-4-2
32-16-8-4
pTVReg loss
†
```

Prefer a compact legend in a relatively empty region of the plot.

Do not allow the legend to obscure an important portion of the curves.

---

# 8. Ellipses

The original figures use ellipses to indicate across-case variability.

Preserve the existing ellipse centers and widths/heights exactly.

Do not recalculate them differently.

For the final figure:

- ellipse face color = corresponding method color
- ellipse edge color = corresponding method color
- low opacity

Suggested:

```python
face_alpha = 0.12
edge_alpha = 0.20
```

The ellipse should support the curve rather than dominate it.

---

# 9. Required static figures

I need **four primary standalone figures**:

```text
01_oasis_no_ellipses
02_oasis_with_ellipses
03_lung_no_ellipses
04_lung_with_ellipses
```

### `01_oasis_no_ellipses`

Contains:

- four selected OASIS curves
- no ellipses
- four simplified method labels

### `02_oasis_with_ellipses`

Contains:

- same four curves
- corresponding ellipses
- same axis limits and layout as no-ellipse version

### `03_lung_no_ellipses`

Contains:

- four selected lung curves
- no ellipses
- four simplified method labels

### `04_lung_with_ellipses`

Contains:

- same four curves
- corresponding ellipses
- pTVReg star
- same axis limits/layout as no-ellipse version

Export each as both:

```text
PNG
SVG
```

PNG should be high-resolution.

---

# 10. PowerPoint animation assets

In addition to the four standalone figures, create animation-ready layers.

The intended sequence is:

```text
INR-Dense
    ↓
INR-BSCP
    ↓
D-BSCP
    ↓
MR-D-BSCP
    ↓
ellipses / reference context
```

## Preferred implementation: transparent overlays

For **each dataset**, export:

### Base

```text
*_anim_base.png
```

Contains only:

- axes
- title
- axis labels
- ticks
- grid
- no method curves
- no ellipses
- no legend method entries

### Layer 1

```text
*_anim_01_inr_dense.png
```

Transparent background.

Contains only:

- INR-Dense curve
- INR-Dense markers
- its legend entry if custom legend entries are built as overlays

### Layer 2

```text
*_anim_02_inr_bscp.png
```

Contains only INR-BSCP.

### Layer 3

```text
*_anim_03_d_bscp.png
```

Contains only D-BSCP.

### Layer 4

```text
*_anim_04_mr_d_bscp.png
```

Contains only MR-D-BSCP.

### Layer 5

```text
*_anim_05_uncertainty.png
```

Contains:

- corresponding ellipses for all four curves

For lung also include:

- pTVReg star + label

either in layer 5 or in a separate optional layer:

```text
lung_anim_06_ptvreg.png
```

---

# 11. Critical alignment requirement for overlays

All overlay PNG files must have:

- identical figure dimensions
- identical DPI
- identical axes position
- identical x/y limits
- identical margins
- identical pixel dimensions

Do **not** use:

```python
bbox_inches="tight"
```

independently for transparent layers, because it may produce different bounding boxes and cause misalignment in PowerPoint.

Instead use a fixed figure geometry, for example:

```python
figsize = (8.0, 5.33)
dpi = 300
```

which would produce approximately:

```text
2400 × 1599 px
```

or choose another fixed high-resolution size.

Use explicit margins, e.g. through:

```python
fig.subplots_adjust(...)
```

and keep those settings exactly identical for every exported layer.

Transparent overlays should use:

```python
transparent=True
```

with no visible figure or axes background.

The base image contains the axes.

---

# 12. Legend animation

If feasible, do **not** use matplotlib's automatically recomputed legend for each transparent overlay, because the legend box can move or resize.

Instead use a fixed manual legend region.

For example, reserve a small area and place four rows at fixed coordinates:

```text
● INR-Dense
● INR-BSCP
● D-BSCP
● MR-D-BSCP
```

Each method overlay should contain its own corresponding legend row.

That way when the curve appears in PowerPoint, its legend entry appears simultaneously.

Alternatively, if this is cumbersome, omit legends from the overlay layers and generate a separate transparent legend layer for each method.

The final static plots can use a normal combined legend.

---

# 13. Also generate cumulative fallback images

Even though transparent overlays are preferred, please also generate cumulative versions because they provide a very robust PowerPoint fallback.

For OASIS:

```text
oasis_step_01.png
    INR-Dense

oasis_step_02.png
    INR-Dense
    + INR-BSCP

oasis_step_03.png
    previous
    + D-BSCP

oasis_step_04.png
    previous
    + MR-D-BSCP

oasis_step_05_final.png
    all four
    + ellipses
```

For lung:

```text
lung_step_01.png
lung_step_02.png
lung_step_03.png
lung_step_04.png
lung_step_05_final.png
```

`lung_step_05_final` should additionally contain the pTVReg reference star.

Every cumulative step should retain exactly the same axes, canvas size, and layout.

---

# 14. OASIS external reference methods

The manuscript plot also contains:

```text
VoxelMorph
TransMorph
VFA
SITReg
Greedy
```

For the **main animated presentation sequence**, do not include these reference points because I want the audience to follow the four continuous parameterizations first.

However, please additionally export one optional context version:

```text
oasis_with_ellipses_and_references.png
oasis_with_ellipses_and_references.svg
```

In this version, include the external reference operating points.

Make them visually secondary:

- neutral gray / muted colors
- diamond markers
- smaller than the four main method curves

Do not allow their colors to compete with the four method colors.

This should make it visually clear that MR-D-BSCP is the best among the tested continuous parameterizations without implying that it is globally the strongest registration method.

---

# 15. Do not alter axes between animation stages

This is important.

The axis limits must be determined from the **complete final dataset first**, then reused for every intermediate animation stage.

Do not autoscale separately when only INR-Dense is visible.

Otherwise the axes will jump when the next method appears.

Use the existing published plot limits unless there is a strong visual reason not to.

Approximately the existing ranges are:

### OASIS

```text
x ≈ 0.15 to 1.55
y ≈ 0.65 to 0.855
```

### DIR-LAB

```text
x ≈ 0.06 to 0.50
y ≈ 0.60 to 2.05
```

But preferably inspect the existing plotting code and retain its exact limits.

---

# 16. Data integrity / QA

Before exporting, verify programmatically that the plotted arrays are identical to the original Figure 2 arrays.

For each retained curve, print a small summary such as:

```text
method
number of operating points
x min / max
y min / max
```

Confirm that only display attributes changed.

Specifically verify:

### OASIS

```text
INR-BSCP == original INR-BSCP(cps2)
D-BSCP == original D-BSCP(cps2)
MR-D-BSCP == original MR-D-BSCP(8-4-2)
```

### Lung

```text
INR-BSCP == original INR-BSCP(cps4)
D-BSCP == original D-BSCP(cps4)
MR-D-BSCP == original MR-D-BSCP(32-16-8-4) with pTVReg loss
```

Do not silently use another variant with a similar name.

---

# 17. Suggested script structure

Please create one reusable plotting script rather than separate manually edited scripts.

Something like:

```text
presentation_arc_plots.py
```

Suggested functions:

```python
load_original_plot_data()

get_oasis_presentation_data()

get_lung_presentation_data()

draw_axes_base(...)

draw_method_curve(...)

draw_method_ellipses(...)

draw_reference_points(...)

export_static_versions(...)

export_animation_layers(...)

export_cumulative_steps(...)
```

Keep the data selection logic separate from styling.

At the top of the file define:

```python
METHOD_ORDER
METHOD_COLORS
DISPLAY_LABELS
FIGSIZE
DPI
```

so I can modify presentation styling later easily.

---

# 18. Output directory

Create something like:

```text
presentation_arc_assets/
│
├── static/
│   ├── 01_oasis_no_ellipses.png
│   ├── 01_oasis_no_ellipses.svg
│   ├── 02_oasis_with_ellipses.png
│   ├── 02_oasis_with_ellipses.svg
│   ├── 03_lung_no_ellipses.png
│   ├── 03_lung_no_ellipses.svg
│   ├── 04_lung_with_ellipses.png
│   ├── 04_lung_with_ellipses.svg
│   └── oasis_with_ellipses_and_references.*
│
├── overlays/
│   ├── oasis_anim_base.png
│   ├── oasis_anim_01_inr_dense.png
│   ├── oasis_anim_02_inr_bscp.png
│   ├── oasis_anim_03_d_bscp.png
│   ├── oasis_anim_04_mr_d_bscp.png
│   ├── oasis_anim_05_uncertainty.png
│   │
│   ├── lung_anim_base.png
│   ├── lung_anim_01_inr_dense.png
│   ├── lung_anim_02_inr_bscp.png
│   ├── lung_anim_03_d_bscp.png
│   ├── lung_anim_04_mr_d_bscp.png
│   └── lung_anim_05_uncertainty_ptvreg.png
│
├── cumulative/
│   ├── oasis_step_01.png
│   ├── oasis_step_02.png
│   ├── oasis_step_03.png
│   ├── oasis_step_04.png
│   ├── oasis_step_05_final.png
│   ├── lung_step_01.png
│   ├── lung_step_02.png
│   ├── lung_step_03.png
│   ├── lung_step_04.png
│   └── lung_step_05_final.png
│
└── presentation_arc_plots.py
```

---

# 19. Before finishing

Please produce a quick contact sheet or montage showing:

- OASIS no ellipse
- OASIS final
- lung no ellipse
- lung final

so I can review colors, font sizes, axis limits, and legend positions without opening each image individually.

Also report:

1. which original plotting/data files were used;
2. the exact data-series keys selected for each of the four methods;
3. the final hex colors used;
4. confirmation that the numerical data were not changed.

Do not overwrite the original manuscript figure or plotting script.

---

# 20. PowerPoint usage after generation

Use the **transparent overlay version** as the preferred presentation workflow.

Stack the base plus all transparent layers perfectly on top of one another.

Recommended animation sequence:

1. INR-Dense → **Fade**, On Click, ~0.25 s
2. INR-BSCP → **Fade**, On Click, ~0.25 s
3. D-BSCP → **Fade**, On Click, ~0.25 s
4. MR-D-BSCP → **Fade**, On Click, ~0.25 s
5. uncertainty ellipses + reference → **Fade**, On Click, ~0.3 s

The intended scientific progression is:

```text
neural dense
    ↓
neural spline
    ↓
remove the neural network
    ↓
add multiresolution
```

The cumulative images are a fallback in case transparent overlays are inconvenient or become misaligned in PowerPoint.

---

# Resolved implementation record

The implementation is `presentation_arc_plots.py` in this directory. It uses
the archived camera-ready v013 source rows beneath
`${ARTIFACT_ROOT}/repo_experiments/migration_2026-09-02/` and writes generated
assets only beneath
`${ARTIFACT_ROOT}/repo_experiments/2026-09-22_slides_for_midl_yrs/presentation_arc_assets/`.

The exact camera-ready Figure 1 block-outline colors are:

```python
METHOD_COLORS = {
    "INR-Dense": "#4E95D9",
    "INR-BSCP": "#A783DC",
    "D-BSCP": "#E97132",
    "MR-D-BSCP": "#E0AD46",
}
```

Run in the configured project container:

```bash
docker exec --user 1009:1009 \
  -e MPLCONFIGDIR=/tmp/matplotlib_presentation_arc \
  py310sitreg_v1_artifact_store \
  python /homebase/code_sync/inrorspline/experiments/2026-09-22_slides_for_midl_yrs/presentation_arc_plots.py
```

Use `--validate-only` for source-key, count, numeric-finiteness, and
source-array identity checks without writing figures. The full render writes a
manifest and review report alongside the static figures, animation overlays,
cumulative fallbacks, and contact sheet. It never modifies the submitted PDF,
the camera-ready builder, or the archived source CSVs.
