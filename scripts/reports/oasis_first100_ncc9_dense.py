#!/usr/bin/env python
"""Aggregate first-100 pyramid 8→4→2→dense NCC9/diffusion results."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent / "experiments/oasis_alpha_sweep"
sys.path.insert(0, str(SWEEP))
sys.path.insert(0, str(HERE))

import common as c  # noqa: E402
import dl_ncc_diffusion as d  # noqa: E402
import oasis_ncc9_first100 as cohort  # noqa: E402
import oasis_dl_ncc_diffusion_first10_arc as first10  # noqa: E402
from evaluate_saved_baselines_first10 import DEFAULT_DL_ROOT  # noqa: E402
from run_oasis_saved_field_bspline_fit_one import (  # noqa: E402
    load_target,
    sha256_file,
    target_path,
)
from run_oasis_vfa_feature_dense_one import canonical_metrics  # noqa: E402

METRICS = first10.METRICS
ACCURACY = ("dice", "hd95", "asd")
BASELINE_REFERENCE_ROOT = Path(
    "/database/inrorspline/repo_experiments/oasis_seed3_first10_raw_ncc_vfa_sitreg_arc_v1"
)


def _regularity(field_voxel, batch):
    xyz = field_voxel.detach().cpu().numpy()
    zyx = xyz[..., ::-1].transpose(3, 0, 1, 2)[None].copy()
    foreground = batch["fixed_mask"].detach().cpu().numpy().astype(bool)
    result = c.jacobian_metrics_voxel(zyx, foreground)
    tensor = field_voxel.permute(3, 0, 1, 2)[None]
    result["diffusion"] = float(c.forward_valid_diffusion(tensor, None))
    return result


def evaluate_references(*, oasis_root, pair_file, dl_root, reference_root, output_root):
    """Reuse audited rank0–9 metrics and canonically evaluate ranks10–99."""
    # Resolve provenance before doing expensive work, not after the last case.
    for name in ("common.py", "run_oasis_vfa_feature_dense_one.py"):
        if not (SWEEP / name).is_file():
            raise FileNotFoundError(SWEEP / name)
    pairs = c.parse_pair_file(pair_file)
    if len(pairs) != 200:
        raise RuntimeError(f"Expected 200 shuffled test pairs, got {len(pairs)}")
    device = torch.device("cpu")
    torch.set_num_threads(6)
    old_reference = reference_root / "baseline_cases"
    rows, input_hashes = [], {}
    for method, target in (("VFA", "vfa"), ("SITReg", "sitreg")):
        target_args = argparse.Namespace(target=target, dl_root=dl_root)
        for rank, (fixed_id, moving_id) in enumerate(pairs[:100]):
            path = target_path(target_args, fixed_id, moving_id)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {method} rank{rank:03d} reference field: {path}")
            source_hash = sha256_file(path)
            input_hashes[str(path)] = source_hash
            if rank < 10:
                saved = c.read_json(old_reference / f"{target}_rank{rank:03d}.json")
                row = dict(saved["row"])
                if (row["fixed_id"], row["moving_id"], row["source_sha256"]) != (
                    fixed_id, moving_id, source_hash
                ):
                    raise RuntimeError(f"Accepted first10 {method} reference changed at rank{rank}")
                if row["direction"] != "moving_to_fixed":
                    raise RuntimeError("Reference registration direction mismatch")
            else:
                batch = c.load_oasis_case(
                    rank=rank, device=device, oasis_root=oasis_root, pair_file=pair_file
                )
                if (str(batch["fixed_id"]), str(batch["moving_id"])) != (fixed_id, moving_id):
                    raise RuntimeError(f"OASIS pair identity mismatch at rank{rank}")
                normalized, audit = load_target(
                    target_args, fixed_id=fixed_id, moving_id=moving_id, device=device
                )
                if audit["source_sha256"] != source_hash:
                    raise RuntimeError("Reference field changed while loading")
                field = normalized * d.voxel_scale(c.IMAGE_SHAPE, device=device)
                metrics, _ = canonical_metrics(field, batch)
                regularity = _regularity(field, batch)
                row = {
                    **metrics,
                    **regularity,
                    "case_rank": rank,
                    "fixed_id": fixed_id,
                    "moving_id": moving_id,
                    "direction": "moving_to_fixed",
                    "source_path": str(path),
                    "source_sha256": source_hash,
                }
            row.update(
                resolution="reference",
                representation=method,
                alpha=None,
                group=method,
                source_kind="original_saved_DL",
                gpu="",
                prefix_gpu="",
                training_elapsed_sec="",
                pipeline_training_elapsed_sec="",
            )
            if not all(math.isfinite(float(row[key])) for key in METRICS):
                raise FloatingPointError(f"Nonfinite {method} reference at rank{rank}")
            rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    c.write_json(output_root / "reference_manifest.json", {
        "pair_file": str(pair_file),
        "pair_file_sha256": sha256_file(pair_file),
        "reference_fields": input_hashes,
        "reference_count": len(rows),
        "source_methods": ["VFA", "SITReg"],
        "metric_code_sha256": sha256_file(SWEEP / "common.py"),
        "evaluator_code_sha256": sha256_file(SWEEP / "run_oasis_vfa_feature_dense_one.py"),
    })
    return rows, input_hashes


def make_rows(results, references):
    candidates = []
    for result in results:
        if result["spec"]["representation"] != "842_dense":
            continue
        row = first10.flat_row(result)
        row["variant"] = cohort.VARIANT
        row["source_kind"] = (
            "reused_accepted_first10" if row["case_rank"] < 10 else "new_fit"
        )
        candidates.append(row)
    expected = {
        (rank, alpha)
        for rank in range(100)
        for alpha in cohort.LAMBDAS
    }
    observed = {(row["case_rank"], row["alpha"]) for row in candidates}
    if len(candidates) != 400 or observed != expected:
        raise RuntimeError(f"Expected 400 unique dense outputs, got {len(candidates)}")
    all_rows = candidates + references
    for row in all_rows:
        for method in ("VFA", "SITReg"):
            baseline = next(
                ref for ref in references
                if ref["representation"] == method and ref["case_rank"] == row["case_rank"]
            )
            for metric in ACCURACY:
                row[f"delta_{method.lower()}_{metric}"] = row[metric] - baseline[metric]
    return candidates, all_rows


def summarize(rows, ranks=range(100)):
    ranks = set(ranks)
    groups = [("pyramid", "842_dense", alpha) for alpha in cohort.LAMBDAS]
    groups += [("reference", method, None) for method in ("VFA", "SITReg")]
    means = []
    for resolution, representation, alpha in groups:
        selected = [
            row for row in rows
            if row["case_rank"] in ranks
            and (row["resolution"], row["representation"], row["alpha"])
            == (resolution, representation, alpha)
        ]
        expected_ranks = sorted(ranks)
        if sorted(row["case_rank"] for row in selected) != expected_ranks:
            raise RuntimeError(f"Incomplete/duplicate group {resolution}/{representation}/{alpha}")
        output = {
            "resolution": resolution,
            "representation": representation,
            "alpha": alpha,
            "n": len(selected),
            "group": selected[0]["group"],
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Nonfinite metric {metric}")
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        means.append(output)
    return means


def write_report(results, references, output_root, provenance, *, mark_accepted=True):
    candidates, rows = make_rows(results, references)
    means = summarize(rows)
    for row in means:
        for method in ("VFA", "SITReg"):
            baseline = next(item for item in means if item["representation"] == method)
            for metric in ACCURACY:
                row[f"delta_{method.lower()}_{metric}_mean"] = (
                    row[f"{metric}_mean"] - baseline[f"{metric}_mean"]
                )
    selected = min(
        (row for row in means if row["representation"] == "842_dense"),
        key=lambda row: (-row["dice_mean"], row["asd_mean"], row["hd95_mean"]),
    )
    leave4out = summarize(rows, ranks=[rank for rank in range(100) if rank != 4])
    c.write_csv(output_root / "case_metrics.csv", rows)
    c.write_csv(output_root / "mean_std.csv", means)
    c.write_csv(output_root / "leave_rank004_out_mean_std.csv", leave4out)
    c.write_json(output_root / "results.json", {
        "complete": True,
        "variant": cohort.VARIANT,
        "cohort": "seed-3 shuffled OASIS ranks0-99",
        "candidate_count": len(candidates),
        "reference_count": len(references),
        "means": means,
        "selected_by_mean_dice_then_asd_hd95": selected,
        "leave_rank004_out_means": leave4out,
        "case_metrics": rows,
        "candidates": results,
        "provenance": provenance,
    })
    lines = [
        "# OASIS first-100 8→4→2→dense pyramid NCC9/diffusion sweep",
        "",
        "All 400 dense candidates and 200 original learned-reference evaluations are complete.",
        "The 40 first10 dense candidates are reused from the accepted P43 sweep because their numerical source hashes, protocol, pair identities, and settings match exactly; the remaining 360 are new fits.",
        "Means are across-case means; SD is sample SD (ddof=1) across the stated cohort, not repeated-seed variability.",
        "Loss: NCC_vfa window9 (1−mean squared local correlation) plus λ times unmasked total-field finite-difference diffusion. λ is shared across cps8/4/2 and dense stages.",
        "Pyramid loss factors are 4/2/1/1; normalized-parameter learning rates are 0.03/0.01/0.003/0.001.",
        "The run overlaps existing GPU workloads; runtime is diagnostic, not a controlled speed benchmark.",
        "",
        "| Method / λ | Dice ↑ | HD95 (mm) ↓ | ASD (mm) ↓ | FG SD(log J) ↓ | FG folding (J≤0, %) ↓ | Diffusion |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in means:
        label = row["representation"] if row["alpha"] is None else f"λ={row['alpha']:g}"
        values = [f"{row[f'{metric}_mean']:.6f} ± {row[f'{metric}_std']:.6f}" for metric in METRICS]
        lines.append(f"| {label} (n={row['n']}) | " + " | ".join(values) + " |")
    lines += [
        "",
        f"Dice-first sweep selection: λ={selected['alpha']:g}; Dice {selected['dice_mean']:.6f}, HD95 {selected['hd95_mean']:.6f} mm, ASD {selected['asd_mean']:.6f} mm.",
        "",
        "## Leave-rank-4-out sensitivity",
        "",
        "| Method / λ | Dice ↑ | HD95 (mm) ↓ | ASD (mm) ↓ | FG SD(log J) | FG folding (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in leave4out:
        label = row["representation"] if row["alpha"] is None else f"λ={row['alpha']:g}"
        vals = [f"{row[f'{metric}_mean']:.6f} ± {row[f'{metric}_std']:.6f}" for metric in METRICS[:5]]
        lines.append(f"| {label} (n={row['n']}) | " + " | ".join(vals) + " |")
    lines += [
        "",
        "Per-case metrics and paired differences versus both references are in `case_metrics.csv`; full numerical rows and provenance are in `results.json`.",
    ]
    (output_root / "report.md").write_text("\n".join(lines) + "\n")
    if not mark_accepted:
        return
    c.write_json(output_root / "ACCEPTED.json", {
        "technical_completion": True,
        "candidate_count": len(candidates),
        "new_candidate_count": 360,
        "reused_accepted_first10_count": 40,
        "reference_count": len(references),
        "configuration_count": len(cohort.LAMBDAS),
        "completed_at": c.now_text(),
    })
