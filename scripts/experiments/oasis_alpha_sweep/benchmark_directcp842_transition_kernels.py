#!/usr/bin/env python
"""Benchmark DirectCP 8-4-2 expansion and knot-insertion kernels on CPU/GPU."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common as c  # noqa: E402


DEFAULT_REVIEW_ROOT = c.REPO_ROOT / "experiments/2026-08-04_oasis_directcp842_production_dvf_timing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--allow-tolerance-fail", action="store_true")
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def deterministic_control(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    total = int(np.prod(tuple(int(v) for v in shape))) * 3
    base = torch.linspace(-1.0, 1.0, total, dtype=torch.float32, device=device)
    values = 0.015 * torch.sin(17.0 * base) + 0.005 * torch.cos(31.0 * base)
    return values.reshape(-1, 3).contiguous()


def timed(fn: Callable[[], torch.Tensor], device: torch.device, warmups: int, repeats: int) -> Tuple[List[float], torch.Tensor]:
    output = fn()
    sync(device)
    for _ in range(max(0, int(warmups))):
        output = fn()
        sync(device)
    times: List[float] = []
    for _ in range(max(1, int(repeats))):
        sync(device)
        start = time.perf_counter()
        output = fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return times, output.detach()


def summarize_times(times: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(times), dtype=np.float64)
    return {
        "median_sec": float(np.median(arr)),
        "min_sec": float(np.min(arr)),
        "p95_sec": float(np.percentile(arr, 95)),
    }


def diff_stats(cpu: torch.Tensor, gpu: torch.Tensor) -> Dict[str, float]:
    diff = cpu.detach().cpu().float() - gpu.detach().cpu().float()
    return {
        "max_abs_diff": float(diff.abs().max()),
        "mean_abs_diff": float(diff.abs().mean()),
        "rmse_diff": float(torch.sqrt(torch.mean(diff * diff))),
    }


def benchmark_pair(
    *,
    name: str,
    metadata: Dict[str, object],
    cpu_fn: Callable[[], torch.Tensor],
    gpu_fn: Callable[[], torch.Tensor],
    cpu: torch.device,
    gpu: torch.device,
    warmups: int,
    repeats: int,
    tolerance: float,
) -> Dict[str, object]:
    cpu_times, cpu_out = timed(cpu_fn, cpu, warmups, repeats)
    gpu_times, gpu_out = timed(gpu_fn, gpu, warmups, repeats)
    cpu_summary = summarize_times(cpu_times)
    gpu_summary = summarize_times(gpu_times)
    diffs = diff_stats(cpu_out, gpu_out)
    speedup = cpu_summary["median_sec"] / gpu_summary["median_sec"] if gpu_summary["median_sec"] > 0 else math.nan
    return {
        "operation": name,
        **metadata,
        "cpu_median_sec": cpu_summary["median_sec"],
        "cpu_min_sec": cpu_summary["min_sec"],
        "cpu_p95_sec": cpu_summary["p95_sec"],
        "gpu_median_sec": gpu_summary["median_sec"],
        "gpu_min_sec": gpu_summary["min_sec"],
        "gpu_p95_sec": gpu_summary["p95_sec"],
        "cpu_over_gpu_speedup": speedup,
        **diffs,
        "within_tolerance": bool(diffs["max_abs_diff"] <= float(tolerance)),
        "tolerance": float(tolerance),
    }


def report_markdown(args: argparse.Namespace, rows: Sequence[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# DirectCP 8-4-2 CPU/GPU Transition Kernel Benchmark")
    lines.append("")
    lines.append(f"- Image shape: `{c.IMAGE_SHAPE}`")
    lines.append("- Schedule: `8,4,2`")
    lines.append(f"- Warmups/repeats: `{args.warmups}/{args.repeats}`")
    lines.append(f"- CPU threads: `{args.cpu_threads}`")
    lines.append(f"- Tolerance: `{args.tolerance}` max absolute difference")
    lines.append("")
    lines.append("Expansion and knot insertion are PyTorch tensor operations. In normal GPU DirectCP runs, tensors are CUDA tensors, so these operations already run on GPU. This benchmark quantifies CPU/GPU agreement and the isolated runtime of each component.")
    lines.append("")
    lines.append("| operation | cps/transition | CPU median | GPU median | CPU/GPU speedup | max diff | within tol |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for row in rows:
        label = row.get("cps") or row.get("transition") or ""
        lines.append(
            f"| {row.get('operation')}"
            f" | {label}"
            f" | {row.get('cpu_median_sec'):.6f}s"
            f" | {row.get('gpu_median_sec'):.6f}s"
            f" | {row.get('cpu_over_gpu_speedup'):.2f}x"
            f" | {row.get('max_abs_diff'):.3g}"
            f" | {row.get('within_tolerance')}"
            " |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- `dense_expand` is the B-spline control-grid to dense-field expansion.")
    lines.append("- `pure_knot_insertion` is control-grid refinement only, with no dense preservation check.")
    lines.append("- `preservation_check` includes pure knot insertion, inserted-control dense expansion, and dense difference metrics against the previous field.")
    lines.append("- Production DVF timing should avoid `preservation_check` unless diagnostics are explicitly requested.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    cpu = torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CPU/GPU benchmark.")
    torch.cuda.set_device(int(args.gpu))
    gpu = torch.device("cuda")
    shapes = c.build_nested_cp_shapes(c.IMAGE_SHAPE, base_spacing=8, spacings=[8, 4, 2])
    c.assert_nested_schedule(shapes, [8, 4, 2])

    rows: List[Dict[str, object]] = []
    for cps in [8, 4, 2]:
        shape = shapes[cps]
        control_cpu = deterministic_control(shape, cpu)
        control_gpu = control_cpu.to(gpu)
        rows.append(
            benchmark_pair(
                name="dense_expand",
                metadata={"cps": int(cps), "control_shape_zyx": list(shape)},
                cpu_fn=lambda control=control_cpu, shape=shape, cps=cps: c.expand_on_shape(
                    control, shape, int(cps), c.IMAGE_SHAPE, 1
                ),
                gpu_fn=lambda control=control_gpu, shape=shape, cps=cps: c.expand_on_shape(
                    control, shape, int(cps), c.IMAGE_SHAPE, 1
                ),
                cpu=cpu,
                gpu=gpu,
                warmups=int(args.warmups),
                repeats=int(args.repeats),
                tolerance=float(args.tolerance),
            )
        )

    for coarse, fine in [(8, 4), (4, 2)]:
        coarse_shape = shapes[coarse]
        fine_shape = shapes[fine]
        control_cpu = deterministic_control(coarse_shape, cpu)
        grid_cpu = c.flat_to_grid(control_cpu, coarse_shape)
        grid_gpu = grid_cpu.to(gpu)
        rows.append(
            benchmark_pair(
                name="pure_knot_insertion",
                metadata={"transition": f"{coarse}->{fine}", "source_shape_zyx": list(coarse_shape), "target_shape_zyx": list(fine_shape)},
                cpu_fn=lambda grid=grid_cpu, fine_shape=fine_shape: c.cubic_bspline_dyadic_knot_insert_3d(
                    grid, fine_shape
                ),
                gpu_fn=lambda grid=grid_gpu, fine_shape=fine_shape: c.cubic_bspline_dyadic_knot_insert_3d(
                    grid, fine_shape
                ),
                cpu=cpu,
                gpu=gpu,
                warmups=int(args.warmups),
                repeats=int(args.repeats),
                tolerance=float(args.tolerance),
            )
        )

    for coarse, fine in [(8, 4), (4, 2)]:
        coarse_shape = shapes[coarse]
        fine_shape = shapes[fine]
        control_cpu = deterministic_control(coarse_shape, cpu)
        grid_cpu = c.flat_to_grid(control_cpu, coarse_shape)
        prev_dense_cpu = c.expand_on_shape(control_cpu, coarse_shape, int(coarse), c.IMAGE_SHAPE, 1)
        grid_gpu = grid_cpu.to(gpu)
        prev_dense_gpu = prev_dense_cpu.to(gpu)

        def cpu_preservation(grid=grid_cpu, fine_shape=fine_shape, fine=fine, prev_dense=prev_dense_cpu) -> torch.Tensor:
            inserted = c.cubic_bspline_dyadic_knot_insert_3d(grid, fine_shape)
            dense = c.expand_on_shape(c.grid_to_flat(inserted), fine_shape, int(fine), c.IMAGE_SHAPE, 1)
            _ = c.field_difference_metrics(dense - prev_dense, prefix="preserve_")
            return dense

        def gpu_preservation(grid=grid_gpu, fine_shape=fine_shape, fine=fine, prev_dense=prev_dense_gpu) -> torch.Tensor:
            inserted = c.cubic_bspline_dyadic_knot_insert_3d(grid, fine_shape)
            dense = c.expand_on_shape(c.grid_to_flat(inserted), fine_shape, int(fine), c.IMAGE_SHAPE, 1)
            _ = c.field_difference_metrics(dense - prev_dense, prefix="preserve_")
            return dense

        rows.append(
            benchmark_pair(
                name="preservation_check",
                metadata={"transition": f"{coarse}->{fine}", "source_shape_zyx": list(coarse_shape), "target_shape_zyx": list(fine_shape)},
                cpu_fn=cpu_preservation,
                gpu_fn=gpu_preservation,
                cpu=cpu,
                gpu=gpu,
                warmups=int(args.warmups),
                repeats=int(args.repeats),
                tolerance=float(args.tolerance),
            )
        )

    args.review_root.mkdir(parents=True, exist_ok=True)
    c.write_csv(Path(args.review_root) / "transition_kernel_benchmark.csv", rows)
    report = report_markdown(args, rows)
    (Path(args.review_root) / "transition_kernel_benchmark_report.md").write_text(report + "\n")
    c.write_json(
        Path(args.review_root) / "transition_kernel_benchmark_summary.json",
        {
            "completed_at": c.now_text(),
            "row_count": len(rows),
            "all_within_tolerance": all(bool(row.get("within_tolerance")) for row in rows),
            "tolerance": float(args.tolerance),
        },
    )
    print(report, flush=True)
    if not args.allow_tolerance_fail and not all(bool(row.get("within_tolerance")) for row in rows):
        raise RuntimeError("CPU/GPU benchmark exceeded tolerance.")


if __name__ == "__main__":
    main()
