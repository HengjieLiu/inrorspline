#!/usr/bin/env python3
"""Build v001 vs v002_candidate Panel A/B comparison artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_DIR = REPO_ROOT / "experiments/paper_artifacts/fitting_figure/v001"
NEW_DIR = REPO_ROOT / "experiments/paper_artifacts/fitting_figure/v002_candidate"
OUT_DIR = (
    REPO_ROOT
    / "experiments/paper_artifacts/comparison/v001_vs_v002_candidate_panelAB_final_cps2"
)

OLD_PAPER_FIG = OLD_DIR / "panelAB_composite/fit_modes_panelAB_composite_sitreg.png"
NEW_PAPER_FIG = NEW_DIR / "panelAB_composite/fit_modes_panelAB_composite_sitreg.png"
OLD_STANDARD_FIG = OLD_DIR / "fit_modes_standardized_panelAB_curves.png"
NEW_STANDARD_FIG = NEW_DIR / "fit_modes_standardized_panelAB_curves.png"


def relpath(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def load_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def merge_by_display(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    keys = ["panel", "plot_case", "display_method"]
    merged = old.merge(new, on=keys, suffixes=("_v001", "_v002"), how="outer", indicator=True)
    return merged.sort_values(keys).reset_index(drop=True)


def build_selected_run_diff() -> pd.DataFrame:
    old = load_csv(OLD_DIR, "fit_modes_standardized_panelAB_selected_runs.csv")
    new = load_csv(NEW_DIR, "fit_modes_standardized_panelAB_selected_runs.csv")
    merged = merge_by_display(old, new)
    rows = []
    for _, r in merged.iterrows():
        old_method = r.get("method_id_v001")
        new_method = r.get("method_id_v002")
        old_plot = r.get("plot_method_id_v001")
        new_plot = r.get("plot_method_id_v002")
        rows.append(
            {
                "panel": r["panel"],
                "plot_case": r["plot_case"],
                "display_method": r["display_method"],
                "old_method_id": old_method,
                "new_method_id": new_method,
                "old_plot_method_id": old_plot,
                "new_plot_method_id": new_plot,
                "old_lr": r.get("lr_v001"),
                "new_lr": r.get("lr_v002"),
                "old_points": r.get("points_v001"),
                "new_points": r.get("points_v002"),
                "old_source": r.get("source_v001"),
                "new_source": r.get("source_v002"),
                "source_changed": old_method != new_method or old_plot != new_plot,
                "status": "matched" if r["_merge"] == "both" else r["_merge"],
            }
        )
    return pd.DataFrame(rows)


def build_endpoint_diff() -> pd.DataFrame:
    old = load_csv(OLD_DIR, "fit_modes_standardized_panelAB_curve_summary.csv")
    new = load_csv(NEW_DIR, "fit_modes_standardized_panelAB_curve_summary.csv")
    merged = merge_by_display(old, new)
    rows = []
    for _, r in merged.iterrows():
        old_end = r.get("value_end_v001")
        new_end = r.get("value_end_v002")
        old_start = r.get("value_start_v001")
        new_start = r.get("value_start_v002")
        old_epoch10 = r.get("value_epoch10_v001")
        new_epoch10 = r.get("value_epoch10_v002")
        old_epoch100 = r.get("value_epoch100_v001")
        new_epoch100 = r.get("value_epoch100_v002")
        delta_end = new_end - old_end if pd.notna(old_end) and pd.notna(new_end) else pd.NA
        pct_delta_end = (
            100.0 * delta_end / old_end
            if pd.notna(delta_end) and pd.notna(old_end) and old_end != 0
            else pd.NA
        )
        rows.append(
            {
                "panel": r["panel"],
                "plot_case": r["plot_case"],
                "display_method": r["display_method"],
                "metric": r.get("metric_v002", r.get("metric_v001")),
                "endpoint_label": "final 1-NCC" if r["panel"] == "A" else "final rRMSE (%)",
                "old_method_id": r.get("method_id_v001"),
                "new_method_id": r.get("method_id_v002"),
                "old_plot_method_id": r.get("plot_method_id_v001"),
                "new_plot_method_id": r.get("plot_method_id_v002"),
                "old_lr": r.get("lr_v001"),
                "new_lr": r.get("lr_v002"),
                "old_value_start": old_start,
                "new_value_start": new_start,
                "delta_value_start": (
                    new_start - old_start if pd.notna(old_start) and pd.notna(new_start) else pd.NA
                ),
                "old_value_epoch10": old_epoch10,
                "new_value_epoch10": new_epoch10,
                "delta_value_epoch10": (
                    new_epoch10 - old_epoch10
                    if pd.notna(old_epoch10) and pd.notna(new_epoch10)
                    else pd.NA
                ),
                "old_value_epoch100": old_epoch100,
                "new_value_epoch100": new_epoch100,
                "delta_value_epoch100": (
                    new_epoch100 - old_epoch100
                    if pd.notna(old_epoch100) and pd.notna(new_epoch100)
                    else pd.NA
                ),
                "old_value_end": old_end,
                "new_value_end": new_end,
                "delta_value_end": delta_end,
                "pct_delta_value_end": pct_delta_end,
                "source_changed": r.get("method_id_v001") != r.get("method_id_v002")
                or r.get("plot_method_id_v001") != r.get("plot_method_id_v002"),
                "status": "matched" if r["_merge"] == "both" else r["_merge"],
            }
        )
    return pd.DataFrame(rows)


def build_delta_summary(endpoint_diff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in endpoint_diff.iterrows():
        if r["plot_case"] == "OASIS rank-9" and r["display_method"] == "INR-BSCP":
            protocol_change = "OASIS INR-BSCP: cps4 -> cps2"
        elif r["plot_case"] == "OASIS rank-9" and r["display_method"] == "D-BSCP":
            protocol_change = "OASIS D-BSCP: cps4 -> cps2"
        elif r["plot_case"] == "OASIS rank-9" and r["display_method"] == "MR-D-BSCP":
            protocol_change = "OASIS MR-D-BSCP: unchanged 8-4-2"
        elif r["plot_case"].startswith("LungCT"):
            protocol_change = "LungCT: unchanged cps4 / 32-16-8-4"
        else:
            protocol_change = "unchanged"
        rows.append(
            {
                "panel": r["panel"],
                "plot_case": r["plot_case"],
                "display_method": r["display_method"],
                "protocol_change": protocol_change,
                "endpoint_label": r["endpoint_label"],
                "old_value_end": r["old_value_end"],
                "new_value_end": r["new_value_end"],
                "delta_value_end": r["delta_value_end"],
                "pct_delta_value_end": r["pct_delta_value_end"],
            }
        )
    return pd.DataFrame(rows)


def fmt_float(value: object) -> str:
    if pd.isna(value):
        return ""
    value_f = float(value)
    if abs(value_f) >= 1:
        return f"{value_f:.3f}"
    return f"{value_f:.5f}"


def write_markdown(summary: pd.DataFrame, endpoint_diff: pd.DataFrame) -> None:
    oasis = summary[summary["plot_case"] == "OASIS rank-9"].copy()
    oasis = oasis[
        oasis["display_method"].isin(["INR-BSCP", "D-BSCP", "MR-D-BSCP"])
    ].sort_values(["panel", "display_method"])

    md = [
        "# Panel A/B v001 vs v002_candidate Delta Summary",
        "",
        "v002_candidate changes the OASIS single-stage spline Panel A/B selections to final-cps matched cps2 variants. v001 is unchanged.",
        "",
        "Protocol changes:",
        "- OASIS INR-BSCP: cps4 -> cps2.",
        "- OASIS D-BSCP: cps4 -> cps2.",
        "- OASIS MR-D-BSCP: unchanged 8-4-2.",
        "- LungCT: unchanged cps4 single-stage and 32-16-8-4 multistage.",
        "",
        "Endpoint changes for OASIS:",
        "",
        "| Panel | Method | Endpoint | v001 | v002_candidate | Delta | Delta % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in oasis.iterrows():
        pct = "" if pd.isna(r["pct_delta_value_end"]) else f"{float(r['pct_delta_value_end']):.1f}%"
        md.append(
            "| {panel} | {method} | {endpoint} | {old} | {new} | {delta} | {pct} |".format(
                panel=r["panel"],
                method=r["display_method"],
                endpoint=r["endpoint_label"],
                old=fmt_float(r["old_value_end"]),
                new=fmt_float(r["new_value_end"]),
                delta=fmt_float(r["delta_value_end"]),
                pct=pct,
            )
        )

    lung_changed = endpoint_diff[
        endpoint_diff["plot_case"].astype(str).str.startswith("LungCT")
        & endpoint_diff["source_changed"].astype(bool)
    ]
    md.extend(
        [
            "",
            "LungCT source-change check:",
            f"- Changed LungCT source rows: {len(lung_changed)}.",
            "",
            "Files to compare:",
            f"- Old paper-layout figure: `{relpath(OLD_PAPER_FIG)}`",
            f"- New paper-layout figure: `{relpath(NEW_PAPER_FIG)}`",
            f"- Old standard generator figure: `{relpath(OLD_STANDARD_FIG)}`",
            f"- New standard generator figure: `{relpath(NEW_STANDARD_FIG)}`",
        ]
    )
    (OUT_DIR / "panelAB_v001_vs_v002_delta_summary.md").write_text("\n".join(md) + "\n")


def write_comparison_guide(summary: pd.DataFrame, endpoint_diff: pd.DataFrame) -> None:
    changed = endpoint_diff[endpoint_diff["source_changed"].astype(bool)]
    oasis_changed = changed[changed["plot_case"] == "OASIS rank-9"]
    lung_changed = changed[changed["plot_case"].astype(str).str.startswith("LungCT")]
    md = [
        "# Panel A/B Final-CPS Matched Comparison Guide",
        "",
        "Purpose: compare v001 against v002_candidate after switching OASIS Panel A/B single-stage spline methods from cps4 to cps2.",
        "",
        "Primary paper-layout comparison:",
        f"- Old: `{relpath(OLD_PAPER_FIG)}`",
        f"- New: `{relpath(NEW_PAPER_FIG)}`",
        "",
        "Standard generator comparison:",
        f"- Old: `{relpath(OLD_STANDARD_FIG)}`",
        f"- New: `{relpath(NEW_STANDARD_FIG)}`",
        "",
        "Side-by-side review image:",
        f"- `{relpath(OUT_DIR / 'panelAB_v001_vs_v002_side_by_side.png')}`",
        "",
        "What changed:",
        "- OASIS INR-BSCP uses cps2 in v002_candidate instead of cps4 in v001.",
        "- OASIS D-BSCP uses cps2 in v002_candidate instead of cps4 in v001.",
        "- OASIS MR-D-BSCP stays the same 8-4-2 run.",
        "- LungCT selections stay unchanged: single-stage cps4 and multistage 32-16-8-4.",
        "",
        "Audit counts:",
        f"- Source-changed rows total: {len(changed)}.",
        f"- Source-changed OASIS rows: {len(oasis_changed)}.",
        f"- Source-changed LungCT rows: {len(lung_changed)}.",
        "",
        "Interpretation:",
        "v002_candidate is the final-cps matched OASIS Panel A/B review candidate. It should be compared against v001 mainly in the OASIS column; LungCT panels should be visually and numerically unchanged except for any regeneration-level formatting differences.",
    ]
    (OUT_DIR / "panelAB_comparison_guide.md").write_text("\n".join(md) + "\n")


def build_side_by_side() -> None:
    old_img = Image.open(OLD_PAPER_FIG).convert("RGB")
    new_img = Image.open(NEW_PAPER_FIG).convert("RGB")
    target_h = max(old_img.height, new_img.height)

    def pad_to_height(img: Image.Image, height: int) -> Image.Image:
        if img.height == height:
            return img
        canvas = Image.new("RGB", (img.width, height), "white")
        canvas.paste(img, (0, 0))
        return canvas

    old_img = pad_to_height(old_img, target_h)
    new_img = pad_to_height(new_img, target_h)
    margin = 40
    title_h = 80
    gap = 30
    canvas = Image.new(
        "RGB",
        (old_img.width + new_img.width + gap + 2 * margin, target_h + title_h + margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    old_title = "v001: OASIS single-stage cps4"
    new_title = "v002_candidate: OASIS single-stage cps2"
    draw.text((margin, 20), old_title, fill="black", font=font)
    draw.text((margin + old_img.width + gap, 20), new_title, fill="black", font=font)
    canvas.paste(old_img, (margin, title_h))
    canvas.paste(new_img, (margin + old_img.width + gap, title_h))
    canvas.save(OUT_DIR / "panelAB_v001_vs_v002_side_by_side.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_diff = build_selected_run_diff()
    endpoint_diff = build_endpoint_diff()
    summary = build_delta_summary(endpoint_diff)

    selected_diff.to_csv(OUT_DIR / "panelAB_selected_run_diff.csv", index=False)
    endpoint_diff.to_csv(OUT_DIR / "panelAB_curve_endpoint_diff.csv", index=False)
    summary.to_csv(OUT_DIR / "panelAB_v001_vs_v002_delta_summary.csv", index=False)
    write_markdown(summary, endpoint_diff)
    write_comparison_guide(summary, endpoint_diff)
    build_side_by_side()

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "old_version": "v001",
        "new_version": "v002_candidate",
        "old_dir": relpath(OLD_DIR),
        "new_dir": relpath(NEW_DIR),
        "outputs": [
            "panelAB_v001_vs_v002_delta_summary.csv",
            "panelAB_v001_vs_v002_delta_summary.md",
            "panelAB_selected_run_diff.csv",
            "panelAB_curve_endpoint_diff.csv",
            "panelAB_comparison_guide.md",
            "panelAB_v001_vs_v002_side_by_side.png",
        ],
    }
    (OUT_DIR / "panelAB_v001_vs_v002_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
