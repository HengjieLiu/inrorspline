#!/usr/bin/env python
"""Run one repo-local OASIS alpha-sweep job."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import common_dctfield as dctfield  # noqa: E402
import common as c  # noqa: E402


METHODS = [
    "idir_original",
    "direct_dvf_sampled_ncc_fd_be",
    "dctfield_sampled_ncc_fd_be",
    "sinr_cps4_be_bspline_analytic",
    "sinr_cps2_be_bspline_analytic",
    "direct_cp_cps4_be_bspline_analytic",
    "direct_cp_cps2_be_bspline_analytic",
    "direct_cp_8_4_2_be_bspline_analytic_multistage",
    "direct_cp_8_4_2_1_be_bspline_analytic_multistage",
    "direct_cp_8_4_2_then_dense_fd_be_multistage",
    "mr_sinr842_composed",
    "mr_sinr842_requery",
]


def parse_bool_option(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--case-rank", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--idir-omega", type=float, default=32.0)
    parser.add_argument("--spline-omega", type=float, default=50.0)
    parser.add_argument("--coords-batch-size", type=int, default=10000)
    parser.add_argument("--dct-size", type=int, default=32)
    parser.add_argument("--eval-chunk-size", type=int, default=200000)
    parser.add_argument("--image-sim-eval-interval", type=int, default=0)
    parser.add_argument("--image-sim-eval-chunk-size", type=int, default=None)
    parser.add_argument("--global-ncc-diagnostic-interval", type=int, default=0)
    parser.add_argument("--global-ncc-diagnostic-chunk-size", type=int, default=None)
    parser.add_argument("--autograd-be-chunk-size", type=int, default=20000)
    parser.add_argument("--stage-full-cps", type=str, default="8,4,2")
    parser.add_argument("--stage-downsample", type=str, default="4,2,1")
    parser.add_argument("--stage-epochs", type=str, default="400,800,1300")
    parser.add_argument("--stage-lr", type=str, default="")
    parser.add_argument(
        "--stage-alpha",
        type=str,
        default="",
        help="Comma-separated per-stage regularization weights; defaults to --alpha for every stage.",
    )
    parser.add_argument("--stage-early-stop-mode", choices=["none", "adaptive"], default="none")
    parser.add_argument("--stage-min-epochs", type=str, default="")
    parser.add_argument("--stage-patience", type=str, default="100")
    parser.add_argument("--stage-check-interval", type=str, default="25")
    parser.add_argument("--stage-rel-delta-total", type=str, default="5e-5")
    parser.add_argument("--stage-rel-delta-data", type=str, default="2e-5")
    parser.add_argument("--stage-min-delta", type=str, default="")
    parser.add_argument("--dense-stage-alpha", type=float, default=None)
    parser.add_argument("--dense-stage-lr", type=float, default=1e-3)
    parser.add_argument("--dense-stage-epochs", type=int, default=1300)
    parser.add_argument("--dense-stage-early-stop-mode", choices=["none", "adaptive"], default="adaptive")
    parser.add_argument("--dense-stage-min-epochs", type=int, default=130)
    parser.add_argument("--dense-stage-patience", type=int, default=50)
    parser.add_argument("--dense-stage-check-interval", type=int, default=25)
    parser.add_argument("--dense-stage-rel-delta-total", type=float, default=1e-4)
    parser.add_argument("--dense-stage-rel-delta-data", type=float, default=5e-5)
    parser.add_argument("--runtime-candidate", type=str, default="")
    parser.add_argument("--timing-detail", choices=["off", "coarse", "deep", "endpoints", "components"], default="off")
    parser.add_argument("--loss-history-mode", choices=["cpu", "gpu_buffer"], default="cpu")
    parser.add_argument("--cudnn-benchmark", type=parse_bool_option, default=True)
    parser.add_argument("--cudnn-deterministic", type=parse_bool_option, default=False)
    parser.add_argument("--output-mode", choices=["full", "dvf_only"], default="full")
    parser.add_argument("--prolongation-preserve-check", action="store_true")
    parser.add_argument("--save-intermediate-dense", action="store_true")
    parser.add_argument("--antialias-sigma-scale", type=float, default=0.5)
    parser.add_argument("--antialias-radius-sigma", type=float, default=3.0)
    parser.add_argument("--out-root", type=Path, default=c.DEFAULT_RUN_ROOT)
    parser.add_argument("--oasis-root", type=Path, default=c.DEFAULT_OASIS_ROOT)
    parser.add_argument("--pair-file", type=Path, default=c.DEFAULT_PAIR_FILE)
    parser.add_argument("--expected-original-case-index", type=int, default=None)
    parser.add_argument("--expected-fixed-id", type=str, default=None)
    parser.add_argument("--expected-moving-id", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--variant", type=str, default="oasis_seed3_alpha_prerun_repo_local")
    parser.add_argument("--source-revision", type=str, default="")
    parser.add_argument("--save-final-dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-sample-count", type=int, default=100000)
    parser.add_argument("--checkpoint-sample-seed", type=int, default=1729)
    parser.add_argument("--checkpoint-epochs", type=str, default="0,1,10,25,50,100,250,500,1000,1500,2000,2500")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-case-only", action="store_true")
    return parser.parse_args()


class CheckpointSampler:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
        device: torch.device,
        run_dir: Path,
    ) -> None:
        self.enabled = bool(args.sample_checkpoints)
        self.run_dir = Path(run_dir)
        self.rows: List[Dict[str, object]] = []
        self.disp_norm_xyz: List[np.ndarray] = []
        self.disp_vox_xyz: List[np.ndarray] = []
        self.epochs = {
            int(token)
            for token in str(args.checkpoint_epochs).replace(" ", "").split(",")
            if token
        }
        self.shape = tuple(int(v) for v in c.IMAGE_SHAPE)
        self.device = device
        self.flat_indices_np = np.asarray([], dtype=np.int64)
        self.flat_indices_t = torch.empty(0, dtype=torch.long, device=device)
        self.points_zyx = np.zeros((0, 3), dtype=np.int64)
        self.points_norm_xyz = np.zeros((0, 3), dtype=np.float32)
        if not self.enabled:
            return
        mask = batch["fixed_mask"]
        assert isinstance(mask, torch.Tensor)
        eligible = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1).astype(bool))
        if eligible.size == 0:
            raise RuntimeError("Cannot sample checkpoints from an empty OASIS foreground mask.")
        count = min(int(args.checkpoint_sample_count), int(eligible.size))
        rng = np.random.default_rng(int(args.checkpoint_sample_seed))
        chosen = eligible if count >= eligible.size else rng.choice(eligible, size=count, replace=False)
        self.flat_indices_np = np.sort(chosen.astype(np.int64, copy=False))
        self.flat_indices_t = torch.from_numpy(self.flat_indices_np).to(device=device, dtype=torch.long)
        z, y, x = np.unravel_index(self.flat_indices_np, self.shape)
        self.points_zyx = np.stack((z, y, x), axis=1).astype(np.int64, copy=False)
        d, h, w = self.shape
        self.points_norm_xyz = np.stack(
            (
                2.0 * x.astype(np.float32) / max(float(w - 1), 1.0) - 1.0,
                2.0 * y.astype(np.float32) / max(float(h - 1), 1.0) - 1.0,
                2.0 * z.astype(np.float32) / max(float(d - 1), 1.0) - 1.0,
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        self.coords_t = torch.from_numpy(self.points_norm_xyz).to(device=device, dtype=torch.float32)
        self.scale_xyz = torch.tensor(
            [(w - 1) / 2.0, (h - 1) / 2.0, (d - 1) / 2.0],
            dtype=torch.float32,
            device=device,
        )
        c.write_json(
            self.run_dir / "checkpoint_sample_metadata.json",
            {
                "enabled": True,
                "dataset": "oasis",
                "sample_seed": int(args.checkpoint_sample_seed),
                "requested_sample_count": int(args.checkpoint_sample_count),
                "actual_sample_count": int(self.flat_indices_np.size),
                "sample_domain": "fixed_image_positive",
                "field_units_saved": ["normalized_grid_xyz", "voxel_xyz"],
                "field_direction": "fixed_to_moving",
                "component_order": "x,y,z",
                "grid_shape_zyx": list(self.shape),
                "checkpoint_epochs": sorted(self.epochs),
            },
        )

    def should_record(self, epoch: int) -> bool:
        return self.enabled and int(epoch) in self.epochs

    def record_samples(
        self,
        *,
        epoch: int,
        disp_norm_xyz: torch.Tensor,
        row_extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.should_record(int(epoch)):
            return
        sample_norm = disp_norm_xyz.detach().to(dtype=torch.float32)
        sample_vox = sample_norm * self.scale_xyz.view(1, 3)
        self.disp_norm_xyz.append(sample_norm.cpu().numpy().astype(np.float32, copy=False))
        self.disp_vox_xyz.append(sample_vox.cpu().numpy().astype(np.float32, copy=False))
        row = {
            "checkpoint_index": len(self.rows),
            "epoch": int(epoch),
            "sample_count": int(sample_norm.shape[0]),
            "norm_abs_max": float(sample_norm.abs().max().detach().cpu()),
            "voxel_mag_mean": float(torch.linalg.norm(sample_vox, dim=1).mean().detach().cpu()),
            "voxel_mag_p95": float(torch.quantile(torch.linalg.norm(sample_vox, dim=1), 0.95).detach().cpu()),
        }
        if row_extra:
            row.update(row_extra)
        self.rows.append(row)

    def record_idir(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        row_extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.should_record(int(epoch)):
            return
        model.eval()
        with torch.no_grad():
            disp = model(self.coords_t)
        self.record_samples(epoch=epoch, disp_norm_xyz=disp, row_extra=row_extra)

    def record_dense(
        self,
        *,
        epoch: int,
        dense_disp_norm_xyz: torch.Tensor,
        row_extra: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self.should_record(int(epoch)):
            return
        disp = dense_disp_norm_xyz.reshape(-1, 3)[self.flat_indices_t]
        self.record_samples(epoch=epoch, disp_norm_xyz=disp, row_extra=row_extra)

    def save(self) -> None:
        if not self.enabled:
            return
        c.write_csv(self.run_dir / "checkpoint_sample_rows.csv", self.rows)
        if self.rows:
            np.savez_compressed(
                self.run_dir / "checkpoint_samples.npz",
                epochs=np.asarray([int(row["epoch"]) for row in self.rows], dtype=np.int64),
                flat_indices=self.flat_indices_np,
                points_zyx=self.points_zyx,
                points_norm_xyz=self.points_norm_xyz,
                disp_norm_xyz=np.stack(self.disp_norm_xyz, axis=0),
                disp_vox_xyz=np.stack(self.disp_vox_xyz, axis=0),
            )


def _json_clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return str(tuple(value.shape))
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    return value


@contextmanager
def _null_section() -> Iterator[None]:
    yield


class TimingRecorder:
    """Hierarchical wall-clock timing with optional CUDA synchronization."""

    def __init__(self, detail: str = "off", device: Optional[torch.device] = None) -> None:
        self.detail = str(detail)
        self.enabled = self.detail != "off"
        self.device = device
        self.root_id = "script"
        self.root_start = time.perf_counter()
        self.root_end: Optional[float] = None
        self.stack: List[str] = [self.root_id]
        self.events: List[Dict[str, object]] = []
        self._counter = 0
        self._mark_counter = 0
        self.notes: List[Dict[str, object]] = []
        self.marks: List[Dict[str, object]] = []
        self.stage_phase_summaries: List[Dict[str, object]] = []
        self.components = None
        if self.detail == "components":
            from scripts.experiments.mrdbscp842_acceleration.components import ACTIVE
            self.components = ACTIVE.get()
            if self.components is None:
                raise RuntimeError("Component timing requires the main instrumentation context")
            self.components.origin = self.root_start

    def set_device(self, device: torch.device) -> None:
        self.device = device
        if self.components is not None:
            self.components.device = device

    def sync(self) -> None:
        if not self.enabled:
            return
        if self.device is not None and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def start(
        self,
        name: str,
        *,
        category: str = "",
        metadata: Optional[Dict[str, object]] = None,
        sync: bool = True,
    ) -> Optional[Dict[str, object]]:
        if not self.enabled or (self.detail == "endpoints" and name != "setup.device"):
            return None
        if self.components is not None:
            sync = False
        if sync:
            self.sync()
        now = time.perf_counter()
        self._counter += 1
        token: Dict[str, object] = {
            "id": f"event_{self._counter:05d}",
            "name": str(name),
            "parent_id": self.stack[-1],
            "parent_name": self.stack[-1],
            "category": str(category),
            "metadata": _json_clean(metadata or {}),
            "start_perf": now,
            "start_offset_sec": now - self.root_start,
            "depth": max(0, len(self.stack) - 1),
            "sync": bool(sync),
        }
        if self.components is not None:
            token["component_token"] = self.components.start(str(name))
        self.stack.append(str(token["id"]))
        return token

    def stop(self, token: Optional[Dict[str, object]], *, sync: Optional[bool] = None) -> float:
        if not self.enabled or token is None:
            return 0.0
        do_sync = bool(token.get("sync")) if sync is None else bool(sync)
        if do_sync:
            self.sync()
        now = time.perf_counter()
        if self.components is not None:
            self.components.stop(token["component_token"])
        event_id = str(token["id"])
        if not self.stack or self.stack[-1] != event_id:
            raise RuntimeError(f"Timing stack mismatch while stopping {event_id}.")
        self.stack.pop()
        elapsed = now - float(token["start_perf"])
        parent_id = str(token["parent_id"])
        parent_name = "script" if parent_id == self.root_id else parent_id
        event = {
            "id": event_id,
            "name": str(token["name"]),
            "parent_id": parent_id,
            "parent_name": parent_name,
            "category": str(token.get("category", "")),
            "metadata": token.get("metadata", {}),
            "start_offset_sec": float(token["start_offset_sec"]),
            "end_offset_sec": now - self.root_start,
            "elapsed_sec": float(elapsed),
            "depth": int(token["depth"]),
        }
        self.events.append(event)
        return float(elapsed)

    @contextmanager
    def section(
        self,
        name: str,
        *,
        category: str = "",
        metadata: Optional[Dict[str, object]] = None,
        sync: bool = True,
    ) -> Iterator[None]:
        token = self.start(name, category=category, metadata=metadata, sync=sync)
        try:
            yield
        finally:
            self.stop(token)

    def add_note(self, name: str, payload: Dict[str, object]) -> None:
        if self.enabled:
            self.notes.append({"name": str(name), "payload": _json_clean(payload)})

    def mark(
        self,
        name: str,
        *,
        metadata: Optional[Dict[str, object]] = None,
        sync: bool = True,
    ) -> Optional[Dict[str, object]]:
        if not self.enabled:
            return None
        if sync:
            self.sync()
        now = time.perf_counter()
        self._mark_counter += 1
        parent_id = self.stack[-1]
        parent_name = "script" if parent_id == self.root_id else parent_id
        row = {
            "id": f"mark_{self._mark_counter:05d}",
            "name": str(name),
            "parent_id": parent_id,
            "parent_name": parent_name,
            "offset_sec": float(now - self.root_start),
            "metadata": _json_clean(metadata or {}),
            "sync": bool(sync),
        }
        self.marks.append(row)
        if self.components is not None:
            self.components.marks[str(name)] = now-self.components.origin
        return row

    def add_stage_phase_summary(self, summary: Dict[str, object]) -> None:
        if self.enabled:
            self.stage_phase_summaries.append(_json_clean(summary))

    def finalize(self) -> None:
        if self.enabled:
            self.sync()
            self.root_end = time.perf_counter()

    def reconciliation(self) -> List[Dict[str, object]]:
        if not self.enabled:
            return []
        root_elapsed = (self.root_end or time.perf_counter()) - self.root_start
        elapsed_by_id = {str(event["id"]): float(event["elapsed_sec"]) for event in self.events}
        elapsed_by_id[self.root_id] = float(root_elapsed)
        names_by_id = {str(event["id"]): str(event["name"]) for event in self.events}
        names_by_id[self.root_id] = "script"
        children_by_parent: Dict[str, float] = {}
        counts_by_parent: Dict[str, int] = {}
        for event in self.events:
            parent_id = str(event["parent_id"])
            children_by_parent[parent_id] = children_by_parent.get(parent_id, 0.0) + float(event["elapsed_sec"])
            counts_by_parent[parent_id] = counts_by_parent.get(parent_id, 0) + 1
        rows: List[Dict[str, object]] = []
        for node_id, elapsed in elapsed_by_id.items():
            children = children_by_parent.get(node_id, 0.0)
            child_count = int(counts_by_parent.get(node_id, 0))
            self_unaccounted = elapsed - children
            tolerance_sec = max(0.5, 0.01 * max(abs(elapsed), 1e-9))
            rows.append(
                {
                    "id": node_id,
                    "name": names_by_id.get(node_id, node_id),
                    "elapsed_sec": float(elapsed),
                    "child_elapsed_sec": float(children),
                    "self_unaccounted_sec": float(self_unaccounted),
                    "abs_self_unaccounted_sec": float(abs(self_unaccounted)),
                    "child_count": child_count,
                    "tolerance_sec": float(tolerance_sec),
                    "within_tolerance": bool(child_count == 0 or abs(self_unaccounted) <= tolerance_sec),
                }
            )
        return sorted(rows, key=lambda row: (0 if row["id"] == self.root_id else 1, str(row["id"])))

    def payload(self, *, args: argparse.Namespace, run_dir: Path, run_name: str) -> Dict[str, object]:
        root_elapsed = (self.root_end or time.perf_counter()) - self.root_start
        return {
            "enabled": bool(self.enabled),
            "detail": self.detail,
            "created_at": c.now_text(),
            "run_name": str(run_name),
            "run_dir": str(run_dir),
            "method": str(args.method),
            "runtime_candidate": str(args.runtime_candidate) or None,
            "output_mode": str(getattr(args, "output_mode", "full")),
            "case_rank": int(args.case_rank),
            "cudnn_requested": {
                "benchmark": bool(getattr(args, "cudnn_benchmark", True)),
                "deterministic": bool(getattr(args, "cudnn_deterministic", False)),
            },
            "cuda_backend": c.cuda_backend_metadata(self.device),
            "script_total_sec": float(root_elapsed),
            "events": sorted(self.events, key=lambda row: float(row["start_offset_sec"])),
            "marks": sorted(self.marks, key=lambda row: float(row["offset_sec"])),
            "reconciliation": self.reconciliation(),
            "stage_phase_summaries": list(self.stage_phase_summaries),
            "notes": list(self.notes),
            **({"components":self.components.payload()} if self.components is not None else {}),
        }


class CudaPhaseAccumulator:
    """Summarize optimizer-loop CUDA phase events without per-epoch log rows."""

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = bool(enabled and device.type == "cuda" and torch.cuda.is_available())
        self.device = device
        self.events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self.wall_sec: Dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            start_wall = time.perf_counter()
            try:
                yield
            finally:
                self.wall_sec[name] = self.wall_sec.get(name, 0.0) + (time.perf_counter() - start_wall)
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.setdefault(str(name), []).append((start, end))

    def summary(self) -> Dict[str, float]:
        if self.enabled:
            torch.cuda.synchronize(self.device)
            return {
                str(name): float(sum(start.elapsed_time(end) for start, end in pairs) / 1000.0)
                for name, pairs in sorted(self.events.items())
            }
        return {str(name): float(value) for name, value in sorted(self.wall_sec.items())}


def timed_evaluate_from_dense_coords(
    dense_coords: torch.Tensor,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    timer: Optional[TimingRecorder] = None,
) -> Dict[str, object]:
    if timer is None or not timer.enabled:
        return c.evaluate_from_dense_coords(dense_coords, batch)

    fixed = batch["fixed"]
    moving = batch["moving"]
    moving_seg = batch["moving_seg"]
    fixed_seg = batch["fixed_seg"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed, torch.Tensor)
    assert isinstance(moving, torch.Tensor)
    assert isinstance(moving_seg, torch.Tensor)
    assert isinstance(fixed_seg, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)

    with torch.no_grad():
        with timer.section("post_train_evaluation.image_warp", category="fixed_post_training"):
            warped = c.warp_full(moving, dense_coords, mode="bilinear")
        with timer.section("post_train_evaluation.seg_warp", category="fixed_post_training"):
            warped_seg = c.warp_full(moving_seg, dense_coords, mode="nearest")
        with timer.section("post_train_evaluation.dice", category="fixed_post_training", sync=False):
            dice = c.dice_per_label(warped_seg, fixed_seg, moving_seg)
        with timer.section("post_train_evaluation.hd95_asd", category="fixed_post_training", sync=False):
            surface = c.surface_metrics_per_label(warped_seg, fixed_seg, moving_seg)
        with timer.section("post_train_evaluation.dense_ncc", category="fixed_post_training"):
            image_loss = float(c.ncc_loss(fixed[None, None], warped[None, None]).detach().cpu())

    metrics: Dict[str, object] = {
        "dense_coords": dense_coords.detach(),
        "warped": warped.detach(),
        "warped_seg": warped_seg.detach(),
        "dice": dice,
        "image_loss": image_loss,
        "dice_mean": c.safe_nanmean(dice),
        "foreground_voxels": int(fixed_mask.sum().detach().cpu()),
    }
    metrics.update(surface)
    return metrics


def method_key(args: argparse.Namespace) -> str:
    if args.method == "idir_original":
        return "idir_original_dense_siren_be_autograd"
    return str(args.method)


def method_cps(method: str) -> Optional[int]:
    if method in {"mr_sinr842_composed", "mr_sinr842_requery"}:
        return 2
    if method == "direct_cp_8_4_2_be_bspline_analytic_multistage":
        return 2
    if method == "direct_cp_8_4_2_1_be_bspline_analytic_multistage":
        return 1
    if method == "direct_cp_8_4_2_then_dense_fd_be_multistage":
        return None
    if method in {"sinr_cps2_be_bspline_analytic", "direct_cp_cps2_be_bspline_analytic"}:
        return 2
    if method in {"sinr_cps4_be_bspline_analytic", "direct_cp_cps4_be_bspline_analytic"}:
        return 4
    return None


def default_lr(method: str) -> float:
    if method.startswith("direct_cp_"):
        return 1e-3
    return 1e-4


def build_run_name(args: argparse.Namespace, lr: float) -> str:
    key = method_key(args)
    if args.run_name:
        return args.run_name
    if args.method in {
        "direct_cp_8_4_2_be_bspline_analytic_multistage",
        "direct_cp_8_4_2_1_be_bspline_analytic_multistage",
        "direct_cp_8_4_2_then_dense_fd_be_multistage",
        "mr_sinr842_composed",
        "mr_sinr842_requery",
    }:
        early_stop_suffix = ""
        if str(args.stage_early_stop_mode) != "none":
            early_stop_suffix = (
                f"_es{args.stage_early_stop_mode}"
                f"_min{args.stage_min_epochs.replace(',', '-')}"
                f"_pat{args.stage_patience.replace(',', '-')}"
            )
        stage_alpha_suffix = ""
        if str(args.stage_alpha).strip():
            stage_alpha_suffix = f"_sa{args.stage_alpha.replace(',', '-').replace('.', 'p')}"
        dense_suffix = ""
        if args.method == "direct_cp_8_4_2_then_dense_fd_be_multistage":
            dense_alpha = float(args.alpha if args.dense_stage_alpha is None else args.dense_stage_alpha)
            dense_suffix = (
                f"_dense_e{int(args.dense_stage_epochs)}"
                f"_a{c.format_float(dense_alpha)}_lr{c.format_float(float(args.dense_stage_lr))}"
                f"_es{args.dense_stage_early_stop_mode}"
                f"_min{int(args.dense_stage_min_epochs)}"
                f"_pat{int(args.dense_stage_patience)}"
                f"_ci{int(args.dense_stage_check_interval)}"
                f"_rt{c.format_float(float(args.dense_stage_rel_delta_total))}"
                f"_rd{c.format_float(float(args.dense_stage_rel_delta_data))}"
            )
        return (
            f"{key}_rank{args.case_rank:03d}_sched{args.stage_full_cps.replace(',', '-')}"
            f"_ds{args.stage_downsample.replace(',', '-')}_e{args.stage_epochs.replace(',', '-')}"
            f"_a{c.format_float(args.alpha)}{stage_alpha_suffix}_lr{c.format_float(lr)}"
            f"{dense_suffix}{early_stop_suffix}"
        )
    if args.method == "dctfield_sampled_ncc_fd_be":
        return (
            f"{key}_rank{args.case_rank:03d}_k{int(args.dct_size)}"
            f"_a{c.format_float(args.alpha)}_e{args.epochs}_lr{c.format_float(lr)}"
        )
    return (
        f"{key}_rank{args.case_rank:03d}_a{c.format_float(args.alpha)}"
        f"_e{args.epochs}_lr{c.format_float(lr)}"
    )


def validate_expected_case(args: argparse.Namespace, *, original_index: int, fixed_id: str, moving_id: str) -> None:
    errors: List[str] = []
    if args.expected_original_case_index is not None and int(args.expected_original_case_index) != int(original_index):
        errors.append(
            f"original_case_index {original_index} != expected {int(args.expected_original_case_index)}"
        )
    if args.expected_fixed_id is not None and str(args.expected_fixed_id) != str(fixed_id):
        errors.append(f"fixed_id {fixed_id} != expected {args.expected_fixed_id}")
    if args.expected_moving_id is not None and str(args.expected_moving_id) != str(moving_id):
        errors.append(f"moving_id {moving_id} != expected {args.expected_moving_id}")
    if errors:
        raise RuntimeError("Case validation failed: " + "; ".join(errors))


def idir_gradient(input_coords: torch.Tensor, output: torch.Tensor, *, create_graph: bool = True) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        [input_coords],
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]


def idir_jacobian_matrix(input_coords: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    jac = torch.zeros(input_coords.shape[0], 3, 3, dtype=output.dtype, device=output.device)
    for i in range(3):
        jac[:, i, :] = idir_gradient(input_coords, output[:, i], create_graph=True)
    return jac


def idir_bending_energy(input_coords: torch.Tensor, output_rel: torch.Tensor, batch_size: int) -> torch.Tensor:
    jac = idir_jacobian_matrix(input_coords, output_rel)
    dx_xyz = torch.zeros(input_coords.shape[0], 3, 3, dtype=output_rel.dtype, device=output_rel.device)
    dy_xyz = torch.zeros_like(dx_xyz)
    dz_xyz = torch.zeros_like(dx_xyz)
    for i in range(3):
        dx_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 0], create_graph=True)
        dy_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 1], create_graph=True)
        dz_xyz[:, i, :] = idir_gradient(input_coords, jac[:, i, 2], create_graph=True)

    dx_xyz = dx_xyz.square()
    dy_xyz = dy_xyz.square()
    dz_xyz = dz_xyz.square()
    loss = torch.mean(dx_xyz[:, :, 0]) + torch.mean(dy_xyz[:, :, 1]) + torch.mean(dz_xyz[:, :, 2])
    loss = loss + 2 * torch.mean(dx_xyz[:, :, 1]) + 2 * torch.mean(dx_xyz[:, :, 2]) + torch.mean(dy_xyz[:, :, 2])
    return loss / float(batch_size)


def make_idir_training_coords(
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
) -> torch.Tensor:
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    selected = coords_all[fixed_mask.detach().reshape(-1).bool()]
    if selected.numel() == 0:
        raise RuntimeError("OASIS fixed image-positive mask is empty.")
    return selected.contiguous()


def evaluate_dense_model(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    chunk_size: int,
) -> Dict[str, object]:
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    chunks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], int(chunk_size)):
            coords = coords_all[start : start + int(chunk_size)]
            chunks.append(coords + model(coords))
    dense_coords = torch.cat(chunks, dim=0).reshape(*c.IMAGE_SHAPE, 3)
    return c.evaluate_from_dense_coords(dense_coords, batch)


def should_image_sim_eval(args: argparse.Namespace, epoch: int) -> bool:
    interval = int(args.image_sim_eval_interval)
    if interval <= 0:
        return False
    return int(epoch) == 1 or int(epoch) % interval == 0 or int(epoch) == int(args.epochs)


def should_global_ncc_diagnostic(args: argparse.Namespace, epoch: int, total_epochs: int) -> bool:
    interval = int(args.global_ncc_diagnostic_interval)
    if interval <= 0:
        return False
    return int(epoch) == 1 or int(epoch) % interval == 0 or int(epoch) == int(total_epochs)


def global_ncc_diagnostic_row(
    *,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    dense_coords: torch.Tensor,
    fixed_mask: torch.Tensor | None = None,
    epoch: int,
    global_epoch: int,
    eval_domain: str,
    device: torch.device,
) -> Dict[str, float | str | bool | int]:
    eval_start = time.time()
    with torch.no_grad():
        warped = c.warp_full(moving, dense_coords, mode="bilinear")
        loss = c.ncc_loss(fixed[None, None], warped[None, None])
    if device.type == "cuda":
        torch.cuda.synchronize()
    value = float(loss.detach().cpu())
    fixed_positive_fields = fixed_image_positive_ncc_fields(fixed, warped, fixed_mask)
    del warped, loss
    return {
        "epoch": float(epoch),
        "global_epoch": float(global_epoch),
        "global_ncc_loss": value,
        "global_ncc_value": -value,
        "diagnostic_name": "full_image_global_ncc",
        "eval_domain": eval_domain,
        "affects_training": False,
        "eval_elapsed_sec": time.time() - eval_start,
        **fixed_positive_fields,
    }


def fixed_image_positive_ncc_fields(
    fixed: torch.Tensor,
    warped: torch.Tensor,
    fixed_mask: torch.Tensor | None,
) -> Dict[str, float | str | int]:
    if fixed_mask is None:
        return {}
    with torch.no_grad():
        mask = fixed_mask.detach().bool()
        voxels = int(mask.sum().detach().cpu())
        if voxels < 2:
            raise RuntimeError("Cannot compute fixed-image-positive NCC on an empty/singleton mask.")
        fixed_masked = fixed.detach()[mask]
        warped_masked = warped.detach()[mask]
        loss = c.ncc_loss(fixed_masked, warped_masked)
        value = float(loss.detach().cpu())
    return {
        "fixed_image_positive_ncc_loss": value,
        "fixed_image_positive_ncc_value": -value,
        "fixed_image_positive_one_minus_ncc": 1.0 + value,
        "fixed_image_positive_mask_voxels": voxels,
        "fixed_image_positive_diagnostic_name": "fixed_image_positive_global_ncc",
        "fixed_image_positive_eval_domain": "fixed_raw_image_gt_0",
    }


def evaluate_idir_full_image_ncc(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    chunk_size: int,
    epoch: int,
) -> Dict[str, float | str | bool | int]:
    fixed = batch["fixed"]
    moving = batch["moving"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed, torch.Tensor)
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    eval_start = time.time()
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    chunks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords_all.shape[0], int(chunk_size)):
            coords = coords_all[start : start + int(chunk_size)]
            chunks.append(coords + model(coords))
        dense_coords = torch.cat(chunks, dim=0).reshape(*c.IMAGE_SHAPE, 3)
        warped = c.warp_full(moving, dense_coords, mode="bilinear")
        loss = c.ncc_loss(fixed[None, None], warped[None, None])
    if device.type == "cuda":
        torch.cuda.synchronize()
    value = float(loss.detach().cpu())
    fixed_positive_fields = fixed_image_positive_ncc_fields(fixed, warped, fixed_mask)
    del coords_all, chunks, dense_coords, warped, loss
    return {
        "epoch": float(epoch),
        "global_epoch": float(epoch),
        "similarity_loss": value,
        "similarity_value": -value,
        "similarity_name": "full_image_global_ncc",
        "eval_domain": "full_image",
        "affects_training": False,
        "eval_elapsed_sec": time.time() - eval_start,
        **fixed_positive_fields,
    }


def train_idir(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    model = c.build_siren(args.hidden_dim, args.hidden_layers, args.idir_omega).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if sampler is not None:
        sampler.record_idir(epoch=0, model=model, row_extra={"method": args.method, "checkpoint_source": "model_init"})
    coords_train = make_idir_training_coords(batch, device)
    n_train = int(coords_train.shape[0])
    batch_size = min(int(args.coords_batch_size), n_train)
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    history: List[Dict[str, float]] = []
    image_similarity_eval: List[Dict[str, float | str | bool]] = []
    image_sim_chunk_size = int(args.image_sim_eval_chunk_size or args.eval_chunk_size)
    global_ncc_diagnostic: List[Dict[str, float | str | bool]] = []
    global_ncc_chunk_size = int(args.global_ncc_diagnostic_chunk_size or args.eval_chunk_size)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        indices = torch.randperm(n_train, device=device)[:batch_size]
        coords = coords_train[indices].detach().requires_grad_(True)
        disp = model(coords)
        moved_coords = coords + disp
        warped = c.sample_volume(moving, moved_coords, mode="bilinear")
        fixed_samples = c.sample_volume(fixed_image, coords, mode="bilinear")
        data_loss = c.ncc_loss(fixed_samples, warped)
        reg_loss = idir_bending_energy(coords, disp, batch_size=batch_size)
        loss = data_loss + float(args.alpha) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        entry = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        history.append(entry)
        if should_image_sim_eval(args, epoch):
            sim_row = evaluate_idir_full_image_ncc(model, batch, device, image_sim_chunk_size, epoch)
            image_similarity_eval.append(sim_row)
            print(json.dumps({"image_similarity_eval": sim_row}), flush=True)
        if should_global_ncc_diagnostic(args, epoch, int(args.epochs)):
            diag_row = evaluate_idir_full_image_ncc(model, batch, device, global_ncc_chunk_size, epoch)
            row = {
                "epoch": diag_row["epoch"],
                "global_epoch": diag_row["global_epoch"],
                "global_ncc_loss": diag_row["similarity_loss"],
                "global_ncc_value": diag_row["similarity_value"],
                "diagnostic_name": "full_image_global_ncc",
                "eval_domain": "full_image",
                "affects_training": False,
                "eval_elapsed_sec": diag_row["eval_elapsed_sec"],
            }
            for key, value in diag_row.items():
                if str(key).startswith("fixed_image_positive_"):
                    row[key] = value
            global_ncc_diagnostic.append(row)
            print(json.dumps({"global_ncc_diagnostic": global_ncc_diagnostic[-1]}), flush=True)
        if sampler is not None:
            sampler.record_idir(
                epoch=epoch,
                model=model,
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "loss": entry["loss"],
                    "data_loss": entry["data_loss"],
                    "reg_loss": entry["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    metrics = evaluate_dense_model(model, batch, device, args.eval_chunk_size)
    metrics["history"] = history
    metrics["image_similarity_eval"] = image_similarity_eval
    metrics["global_ncc_diagnostic"] = global_ncc_diagnostic
    metrics["model"] = model
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["train_coords"] = n_train
    metrics["coords_batch_size"] = batch_size
    metrics["param_count"] = c.count_trainable_params(model)
    return metrics


def train_direct_dvf(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    fixed_mask_bool = fixed_mask.detach().bool()

    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    flat_mask = fixed_mask_bool.reshape(-1)
    train_indices = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
    n_train = int(train_indices.shape[0])
    batch_size = min(int(args.coords_batch_size), n_train)
    if batch_size <= 0:
        raise RuntimeError("Empty direct-DVF training coordinate set.")

    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    dense_disp = torch.nn.Parameter(torch.zeros((*c.IMAGE_SHAPE, 3), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([dense_disp], lr=lr)
    history: List[Dict[str, float]] = []
    if sampler is not None:
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=dense_disp.detach(),
            row_extra={"method": args.method, "checkpoint_source": "dense_dvf_init"},
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        local = torch.randperm(n_train, device=device)[:batch_size]
        flat_idx = train_indices[local]
        coords = coords_all[flat_idx]
        disp = dense_disp.reshape(-1, 3)[flat_idx]
        warped = c.sample_volume(moving, coords + disp, mode="bilinear")
        fixed_samples = c.sample_volume(fixed_image, coords, mode="bilinear")
        data_loss = c.ncc_loss(fixed_samples, warped)
        reg_loss = c.forward_valid_be(
            c.dhw3_to_ncdhw(dense_disp),
            fixed_mask_bool,
            component_reduction="mean",
        )
        loss = data_loss + float(args.alpha) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        entry = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        history.append(entry)
        if sampler is not None:
            sampler.record_dense(
                epoch=epoch,
                dense_disp_norm_xyz=dense_disp.detach(),
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "loss": entry["loss"],
                    "data_loss": entry["data_loss"],
                    "reg_loss": entry["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0

    final_dense_disp = dense_disp.detach()
    with torch.no_grad():
        metrics = c.evaluate_from_dense_coords(id_dense + final_dense_disp, batch)
    metrics["history"] = history
    metrics["dense_disp"] = final_dense_disp
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["train_coords"] = n_train
    metrics["coords_batch_size"] = batch_size
    metrics["param_count"] = int(final_dense_disp.numel())
    return metrics


def train_dctfield(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    moving = batch["moving"]
    fixed_image = batch["fixed"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    fixed_mask_bool = fixed_mask.detach().bool()

    dct_size = dctfield.normalize_dct_size(int(args.dct_size), c.IMAGE_SHAPE)
    decoder = dctfield.DCTFieldDecoder(c.IMAGE_SHAPE, dct_size, device=device, dtype=torch.float32)
    coeff = torch.nn.Parameter(torch.zeros((3, *dct_size), dtype=torch.float32, device=device))
    coords_all = c.make_coordinate_tensor(c.IMAGE_SHAPE, device)
    flat_mask = fixed_mask_bool.reshape(-1)
    train_indices = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
    n_train = int(train_indices.shape[0])
    batch_size = min(int(args.coords_batch_size), n_train)
    if batch_size <= 0:
        raise RuntimeError("Empty DCTField training coordinate set.")

    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    optimizer = torch.optim.Adam([coeff], lr=lr)
    history: List[Dict[str, float]] = []
    if sampler is not None:
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=torch.zeros((*c.IMAGE_SHAPE, 3), dtype=torch.float32, device=device),
            row_extra={"method": args.method, "checkpoint_source": "dct_coeff_init"},
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        dense_disp = decoder.decode(coeff)
        local = torch.randperm(n_train, device=device)[:batch_size]
        flat_idx = train_indices[local]
        coords = coords_all[flat_idx]
        disp = dense_disp.reshape(-1, 3)[flat_idx]
        warped = c.sample_volume(moving, coords + disp, mode="bilinear")
        fixed_samples = c.sample_volume(fixed_image, coords, mode="bilinear")
        data_loss = c.ncc_loss(fixed_samples, warped)
        reg_loss = c.forward_valid_be(
            c.dhw3_to_ncdhw(dense_disp),
            fixed_mask_bool,
            component_reduction="mean",
        )
        loss = data_loss + float(args.alpha) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        entry = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        history.append(entry)
        if sampler is not None:
            sampler.record_dense(
                epoch=epoch,
                dense_disp_norm_xyz=dense_disp.detach(),
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "loss": entry["loss"],
                    "data_loss": entry["data_loss"],
                    "reg_loss": entry["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0

    with torch.no_grad():
        final_dense_disp = decoder.decode(coeff).detach()
        metrics = c.evaluate_from_dense_coords(id_dense + final_dense_disp, batch)
    metrics["history"] = history
    metrics["dense_disp"] = final_dense_disp
    metrics["dct_coeff"] = coeff.detach()
    metrics["dct_size_zyx"] = list(dct_size)
    metrics["dct_basis"] = decoder.basis_name
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["train_coords"] = n_train
    metrics["coords_batch_size"] = batch_size
    metrics["param_count"] = int(coeff.numel())
    return metrics


def build_spline_generator(
    args: argparse.Namespace,
    control_shape: Tuple[int, int, int],
) -> torch.nn.Module:
    if args.method.startswith("direct_cp_"):
        return c.DirectCPGenerator(control_shape)
    return c.NetworkGenerator(c.build_siren(args.hidden_dim, args.hidden_layers, args.spline_omega))


def sample_dense_vector_field(field_dhw3: torch.Tensor, coords_dhw3: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [c.warp_full(field_dhw3[..., component], coords_dhw3, mode="bilinear") for component in range(3)],
        dim=-1,
    )


def compose_pull_coords(
    outer_coords: torch.Tensor,
    inner_disp: torch.Tensor,
    id_dense: torch.Tensor,
) -> torch.Tensor:
    inner_coords = id_dense + inner_disp
    return sample_dense_vector_field(outer_coords, inner_coords)


def scalar_difference_metrics(diff: torch.Tensor, prefix: str) -> Dict[str, float]:
    flat = diff.detach().reshape(-1)
    abs_flat = flat.abs()
    return {
        f"{prefix}mean_abs": float(abs_flat.mean().detach().cpu()),
        f"{prefix}max_abs": float(abs_flat.max().detach().cpu()),
        f"{prefix}rmse": float(torch.sqrt(torch.mean(flat.square())).detach().cpu()),
        f"{prefix}p95_abs": float(torch.quantile(abs_flat, 0.95).detach().cpu()),
    }


def train_mr_sinr_stage(
    args: argparse.Namespace,
    *,
    fixed_image: torch.Tensor,
    moving_for_stage: torch.Tensor,
    generator: torch.nn.Module,
    control_coords: torch.Tensor,
    control_shape: Tuple[int, int, int],
    full_cps: int,
    downsample: int,
    epochs: int,
    lr: float,
    stage_idx: int,
    global_epoch_offset: int,
    total_epochs: int,
    stage_role: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object], List[Dict[str, object]]]:
    fixed_l, blur_sigma, blur_kernel = c.make_loss_volume(
        fixed_image,
        int(downsample),
        float(args.antialias_sigma_scale),
        float(args.antialias_radius_sigma),
    )
    moving_l, _sigma_m, _kernel_m = c.make_loss_volume(
        moving_for_stage,
        int(downsample),
        float(args.antialias_sigma_scale),
        float(args.antialias_radius_sigma),
    )
    eval_shape = tuple(int(v) for v in fixed_l.shape)
    domain_cps = c.domain_cps_from_full(int(full_cps), int(downsample))
    id_grid = c.make_coordinate_grid(eval_shape, fixed_image.device).detach()
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(lr))
    rows: List[Dict[str, object]] = []
    best_loss = float("inf")
    best_data_loss = float("nan")
    best_reg_loss = float("nan")
    best_epoch = 0
    stage_start = time.time()
    print(
        json.dumps(
            {
                "stage_start": int(stage_idx),
                "stage_role": str(stage_role),
                "full_cps": int(full_cps),
                "downsample": int(downsample),
                "domain_cps": int(domain_cps),
                "control_shape_zyx": list(control_shape),
                "eval_shape_zyx": list(eval_shape),
                "epochs": int(epochs),
                "lr": float(lr),
                "alpha": float(args.alpha),
                "image_loss": "sinr_ncc",
                "regularizer": "analytic_cubic_bspline_be",
            }
        ),
        flush=True,
    )
    for epoch in range(1, int(epochs) + 1):
        generator.train()
        output = generator(control_coords, control_shape)
        control_grid = output.reshape(*control_shape, 3)
        dense_disp = c.expand_on_shape(output, control_shape, int(full_cps), eval_shape, int(downsample))
        warped = c.warp_full(moving_l, id_grid + dense_disp, mode="bilinear")
        data_loss = c.ncc_loss(fixed_l[None, None], warped[None, None])
        reg_loss = c.analytic_bspline_sampled_be(
            control_grid,
            eval_shape,
            int(domain_cps),
            component_reduction="mean",
        )
        loss = data_loss + float(args.alpha) * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "stage_idx": float(stage_idx),
            "epoch": float(epoch),
            "global_epoch": float(int(global_epoch_offset) + int(epoch)),
            "full_cps": float(full_cps),
            "downsample": float(downsample),
            "domain_cps": float(domain_cps),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "alpha": float(args.alpha),
            "lr": float(lr),
            "stage_role": str(stage_role),
        }
        rows.append(row)
        current_loss = float(row["loss"])
        if best_epoch == 0 or current_loss < best_loss:
            best_loss = current_loss
            best_data_loss = float(row["data_loss"])
            best_reg_loss = float(row["reg_loss"])
            best_epoch = int(epoch)
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(epochs):
            print(json.dumps(row), flush=True)
    actual_epochs = int(rows[-1]["epoch"]) if rows else 0
    generator.eval()
    with torch.no_grad():
        final_output = generator(control_coords, control_shape)
        final_control = final_output.reshape(*control_shape, 3).detach()
        final_dense = c.expand_bspline_controls(
            final_output,
            control_shape,
            c.IMAGE_SHAPE,
            int(full_cps),
        ).detach()
    summary = {
        "stage_idx": int(stage_idx),
        "stage_role": str(stage_role),
        "full_cps": int(full_cps),
        "downsample": int(downsample),
        "domain_cps": int(domain_cps),
        "control_shape_zyx": list(int(v) for v in control_shape),
        "eval_shape_zyx": list(eval_shape),
        "n_control": int(np.prod(control_shape)),
        "epochs": int(actual_epochs),
        "requested_epochs": int(epochs),
        "actual_epochs": int(actual_epochs),
        "lr": float(lr),
        "alpha": float(args.alpha),
        "early_stop_mode": "none",
        "min_epochs": int(epochs),
        "patience": 0,
        "check_interval": 0,
        "rel_delta_total": 0.0,
        "rel_delta_data": 0.0,
        "stop_reason": "requested_epochs_complete",
        "loss_mode": "sinr_ncc_bspline_be",
        "antialias_sigma": float(blur_sigma),
        "antialias_kernel_size": int(blur_kernel),
        "elapsed_sec": float(time.time() - stage_start),
        "control_abs_max_norm": float(final_control.abs().max().detach().cpu()),
        "control_abs_p95_norm": float(torch.quantile(final_control.abs().reshape(-1), 0.95).detach().cpu()),
        "best_loss": float(best_loss),
        "best_loss_epoch": int(best_epoch),
        "best_data_loss": float(best_data_loss),
        "best_reg_loss": float(best_reg_loss),
    }
    return final_dense, final_control, summary, rows


def train_mr_sinr842_composed(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    stage_cps = c.parse_int_list(str(args.stage_full_cps))
    stage_downsample = c.parse_stage_ints(str(args.stage_downsample), len(stage_cps))
    stage_epochs = c.parse_stage_ints(str(args.stage_epochs), len(stage_cps))
    stage_lr = c.parse_float_list(str(args.stage_lr), len(stage_cps), float(lr))
    if int(sum(stage_epochs)) != int(args.epochs):
        raise ValueError(f"stage_epochs sum {sum(stage_epochs)} != --epochs {args.epochs}")
    if stage_cps != [8, 4, 2]:
        raise ValueError(f"mr_sinr842_composed expects stage_full_cps [8, 4, 2], got {stage_cps}")

    moving = batch["moving"]
    fixed_image = batch["fixed"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    if sampler is not None and sampler.enabled:
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=torch.zeros((*c.IMAGE_SHAPE, 3), dtype=torch.float32, device=device),
            row_extra={"method": args.method, "checkpoint_source": "composed_identity_init"},
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    stage_summaries: List[Dict[str, object]] = []
    history: List[Dict[str, object]] = []
    stage_controls: List[torch.Tensor] = []
    stage_control_shapes: List[Tuple[int, int, int]] = []
    stage_state_dicts: List[Dict[str, torch.Tensor]] = []
    stage_dense_disps: List[torch.Tensor] = []
    stage_coords: List[torch.Tensor] = []
    composed_coords_by_stage: List[torch.Tensor] = []
    composition_rows: List[Dict[str, object]] = []
    prev_composed_coords = id_dense
    global_epoch_offset = 0
    moving_for_stage = moving

    for stage_idx, (full_cps, downsample, epochs, stage_lr_value) in enumerate(
        zip(stage_cps, stage_downsample, stage_epochs, stage_lr)
    ):
        control_coords, control_shape = c.bspline_control_coords(c.IMAGE_SHAPE, int(full_cps), device)
        generator = c.NetworkGenerator(c.build_siren(args.hidden_dim, args.hidden_layers, args.spline_omega)).to(device)
        dense_stage, control_stage, summary, rows = train_mr_sinr_stage(
            args,
            fixed_image=fixed_image,
            moving_for_stage=moving_for_stage.detach(),
            generator=generator,
            control_coords=control_coords,
            control_shape=control_shape,
            full_cps=int(full_cps),
            downsample=int(downsample),
            epochs=int(epochs),
            lr=float(stage_lr_value),
            stage_idx=int(stage_idx),
            global_epoch_offset=int(global_epoch_offset),
            total_epochs=int(args.epochs),
            stage_role="composed_residual",
        )
        with torch.no_grad():
            if stage_idx == 0:
                composed_coords = id_dense + dense_stage
            else:
                composed_coords = compose_pull_coords(prev_composed_coords, dense_stage, id_dense).detach()
            moving_for_stage = c.warp_full(moving, composed_coords, mode="bilinear").detach()
        if sampler is not None and sampler.should_record(global_epoch_offset + int(summary["actual_epochs"])):
            sampler.record_dense(
                epoch=global_epoch_offset + int(summary["actual_epochs"]),
                dense_disp_norm_xyz=(composed_coords - id_dense).detach(),
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "stage_end_composed_field",
                    "stage_idx": int(stage_idx),
                    "stage_epoch": int(summary["actual_epochs"]),
                    "full_cps": int(full_cps),
                    "downsample": int(downsample),
                },
            )
        stage_dense_disps.append(dense_stage.detach())
        stage_coords.append((id_dense + dense_stage).detach())
        composed_coords_by_stage.append(composed_coords.detach())
        stage_controls.append(control_stage.detach().cpu())
        stage_control_shapes.append(control_shape)
        stage_state_dicts.append({k: v.detach().cpu() for k, v in generator.state_dict().items()})
        stage_summaries.append(summary)
        history.extend(rows)
        global_epoch_offset += int(summary["actual_epochs"])
        prev_composed_coords = composed_coords.detach()

    with torch.no_grad():
        seq_warped = moving
        for stage_idx, coords in enumerate(stage_coords):
            seq_warped = c.warp_full(seq_warped, coords, mode="bilinear")
            composed_warped = c.warp_full(moving, composed_coords_by_stage[stage_idx], mode="bilinear")
            row: Dict[str, object] = {
                "diagnostic": "composed_vs_sequential_warp",
                "stage_idx": int(stage_idx),
                "stage_count": int(stage_idx + 1),
            }
            row.update(scalar_difference_metrics(composed_warped - seq_warped, "image_"))
            composition_rows.append(row)
    final_coords = composed_coords_by_stage[-1].detach()
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    metrics = timed_evaluate_from_dense_coords(final_coords, batch)
    metrics["history"] = [
        {
            **row,
            "epoch": float(row["global_epoch"]),
            "stage_epoch": float(row["epoch"]),
        }
        for row in history
    ]
    metrics["stage_history"] = history
    metrics["stage_summaries"] = stage_summaries
    metrics["composition_diagnostics"] = composition_rows
    metrics["prolongation_diagnostics"] = []
    metrics["stage_generator_state_dicts"] = stage_state_dicts
    metrics["stage_control_disps"] = stage_controls
    metrics["stage_control_shapes"] = [list(shape) for shape in stage_control_shapes]
    metrics["dense_disp"] = (final_coords - id_dense).detach()
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["param_count"] = int(sum(sum(t.numel() for t in state.values()) for state in stage_state_dicts))
    metrics["cps"] = 2
    metrics["stage_full_cps"] = stage_cps
    metrics["stage_downsample"] = stage_downsample
    metrics["stage_epochs"] = stage_epochs
    metrics["stage_requested_epochs"] = stage_epochs
    metrics["stage_actual_epochs"] = [int(row["actual_epochs"]) for row in stage_summaries]
    metrics["stage_lr"] = stage_lr
    metrics["stage_early_stop_mode"] = "none"
    metrics["stage_min_epochs"] = stage_epochs
    metrics["stage_patience"] = [0 for _ in stage_cps]
    metrics["stage_check_interval"] = [0 for _ in stage_cps]
    metrics["stage_rel_delta_total"] = [0.0 for _ in stage_cps]
    metrics["stage_rel_delta_data"] = [0.0 for _ in stage_cps]
    metrics["stage_stop_reasons"] = [str(row.get("stop_reason")) for row in stage_summaries]
    metrics["requested_epochs"] = int(args.epochs)
    metrics["actual_epochs"] = int(sum(metrics["stage_actual_epochs"]))
    metrics["prolongation_preserve_check"] = False
    metrics["save_intermediate_dense"] = False
    metrics["nested_control_shapes_zyx"] = {
        str(cps): shape for cps, shape in zip(stage_cps, metrics["stage_control_shapes"])
    }
    metrics["prolongation"] = "none_independent_residual_sinr_stages"
    metrics["multistage_strategy"] = "independent_residual_sinr_stages_with_pull_field_composition"
    metrics["field_composition"] = "phi842(x)=phi8(phi4(phi2(x))) using grid_sample align_corners=True"
    metrics["final_field_is_single_bspline"] = False
    return metrics


def train_mr_sinr842_requery(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
) -> Dict[str, object]:
    stage_cps = c.parse_int_list(str(args.stage_full_cps))
    stage_downsample = c.parse_stage_ints(str(args.stage_downsample), len(stage_cps))
    stage_epochs = c.parse_stage_ints(str(args.stage_epochs), len(stage_cps))
    stage_lr = c.parse_float_list(str(args.stage_lr), len(stage_cps), float(lr))
    if int(sum(stage_epochs)) != int(args.epochs):
        raise ValueError(f"stage_epochs sum {sum(stage_epochs)} != --epochs {args.epochs}")
    if stage_cps != [8, 4, 2]:
        raise ValueError(f"mr_sinr842_requery expects stage_full_cps [8, 4, 2], got {stage_cps}")

    moving = batch["moving"]
    fixed_image = batch["fixed"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    generator = c.NetworkGenerator(c.build_siren(args.hidden_dim, args.hidden_layers, args.spline_omega)).to(device)
    if sampler is not None and sampler.enabled:
        sampler.record_dense(
            epoch=0,
            dense_disp_norm_xyz=torch.zeros((*c.IMAGE_SHAPE, 3), dtype=torch.float32, device=device),
            row_extra={"method": args.method, "checkpoint_source": "requery_identity_init"},
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    stage_summaries: List[Dict[str, object]] = []
    transition_rows: List[Dict[str, object]] = []
    history: List[Dict[str, object]] = []
    final_dense: Optional[torch.Tensor] = None
    final_control: Optional[torch.Tensor] = None
    final_control_shape: Optional[Tuple[int, int, int]] = None
    global_epoch_offset = 0
    prev_dense: Optional[torch.Tensor] = None

    for stage_idx, (full_cps, downsample, epochs, stage_lr_value) in enumerate(
        zip(stage_cps, stage_downsample, stage_epochs, stage_lr)
    ):
        control_coords, control_shape = c.bspline_control_coords(c.IMAGE_SHAPE, int(full_cps), device)
        if prev_dense is not None:
            generator.eval()
            with torch.no_grad():
                start_output = generator(control_coords, control_shape)
                start_dense = c.expand_bspline_controls(
                    start_output,
                    control_shape,
                    c.IMAGE_SHAPE,
                    int(full_cps),
                ).detach()
            row: Dict[str, object] = {
                "diagnostic": "requery_stage_transition_jump",
                "stage_idx": int(stage_idx),
                "from_full_cps": int(stage_cps[stage_idx - 1]),
                "to_full_cps": int(full_cps),
            }
            row.update(c.field_difference_metrics(start_dense - prev_dense, prefix="jump_"))
            transition_rows.append(row)
            print(json.dumps({"stage_transition_jump": row}, allow_nan=True), flush=True)
        dense_stage, control_stage, summary, rows = train_mr_sinr_stage(
            args,
            fixed_image=fixed_image,
            moving_for_stage=moving,
            generator=generator,
            control_coords=control_coords,
            control_shape=control_shape,
            full_cps=int(full_cps),
            downsample=int(downsample),
            epochs=int(epochs),
            lr=float(stage_lr_value),
            stage_idx=int(stage_idx),
            global_epoch_offset=int(global_epoch_offset),
            total_epochs=int(args.epochs),
            stage_role="continued_requery_absolute",
        )
        if sampler is not None and sampler.should_record(global_epoch_offset + int(summary["actual_epochs"])):
            sampler.record_dense(
                epoch=global_epoch_offset + int(summary["actual_epochs"]),
                dense_disp_norm_xyz=dense_stage.detach(),
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "stage_end_absolute_field",
                    "stage_idx": int(stage_idx),
                    "stage_epoch": int(summary["actual_epochs"]),
                    "full_cps": int(full_cps),
                    "downsample": int(downsample),
                },
            )
        final_dense = dense_stage.detach()
        final_control = control_stage.detach()
        final_control_shape = control_shape
        prev_dense = dense_stage.detach()
        stage_summaries.append(summary)
        history.extend(rows)
        global_epoch_offset += int(summary["actual_epochs"])

    if final_dense is None or final_control is None or final_control_shape is None:
        raise RuntimeError("MR-SINR requery produced no final field.")
    final_coords = id_dense + final_dense
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    metrics = timed_evaluate_from_dense_coords(final_coords, batch)
    metrics["history"] = [
        {
            **row,
            "epoch": float(row["global_epoch"]),
            "stage_epoch": float(row["epoch"]),
        }
        for row in history
    ]
    metrics["stage_history"] = history
    metrics["stage_summaries"] = stage_summaries
    metrics["stage_transition_diagnostics"] = transition_rows
    metrics["prolongation_diagnostics"] = []
    metrics["generator"] = generator
    metrics["control_shape"] = final_control_shape
    metrics["control_coords"] = torch.empty(0, 3, dtype=torch.float32, device=device)
    metrics["control_disp"] = final_control
    metrics["dense_disp"] = final_dense.detach()
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["param_count"] = c.count_trainable_params(generator)
    metrics["cps"] = 2
    metrics["stage_full_cps"] = stage_cps
    metrics["stage_downsample"] = stage_downsample
    metrics["stage_epochs"] = stage_epochs
    metrics["stage_requested_epochs"] = stage_epochs
    metrics["stage_actual_epochs"] = [int(row["actual_epochs"]) for row in stage_summaries]
    metrics["stage_lr"] = stage_lr
    metrics["stage_early_stop_mode"] = "none"
    metrics["stage_min_epochs"] = stage_epochs
    metrics["stage_patience"] = [0 for _ in stage_cps]
    metrics["stage_check_interval"] = [0 for _ in stage_cps]
    metrics["stage_rel_delta_total"] = [0.0 for _ in stage_cps]
    metrics["stage_rel_delta_data"] = [0.0 for _ in stage_cps]
    metrics["stage_stop_reasons"] = [str(row.get("stop_reason")) for row in stage_summaries]
    metrics["requested_epochs"] = int(args.epochs)
    metrics["actual_epochs"] = int(sum(metrics["stage_actual_epochs"]))
    metrics["prolongation_preserve_check"] = False
    metrics["save_intermediate_dense"] = False
    metrics["nested_control_shapes_zyx"] = {
        str(cps): list(c.bspline_control_shape(c.IMAGE_SHAPE, int(cps))) for cps in stage_cps
    }
    metrics["prolongation"] = "none_continued_sinr_requery"
    metrics["multistage_strategy"] = "single_continued_sinr_generator_requeried_at_8_4_2_absolute_fields"
    metrics["field_composition"] = "none_absolute_final_stage_field"
    metrics["final_field_is_single_bspline"] = True
    return metrics


def save_full_mode_dvf_endpoint(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    dense_disp_norm: torch.Tensor,
    run_dir: Path,
    *,
    timer: Optional[TimingRecorder] = None,
    total_start: Optional[float] = None,
    case_load_complete_sec: Optional[float] = None,
) -> Dict[str, object]:
    """Save the final DVF before full evaluation so the input-to-DVF endpoint is timed."""
    fixed_id = str(batch["fixed_id"])
    moving_id = str(batch["moving_id"])
    path = Path(run_dir) / f"disp_norm_{fixed_id}_{moving_id}.npy"
    if timer is not None:
        timer.mark(
            "dvf_in_memory_ready",
            metadata={
                "definition": "full-mode endpoint immediately after final dense displacement exists, before evaluation",
                "output_mode": "full",
            },
            sync=True,
        )
    prep_token = timer.start(
        "method_execution.full_mode_dvf_file_prep",
        category="dvf_file_output",
        metadata={"definition": "detach final dense displacement and move to CPU float32 numpy before evaluation"},
    ) if timer is not None else None
    dense_np = dense_disp_norm.detach().cpu().numpy().astype(np.float32, copy=False)
    if timer is not None:
        timer.stop(prep_token)
    write_token = timer.start(
        "method_execution.full_mode_dvf_file_write",
        category="dvf_file_output",
        metadata={"path": str(path), "definition": "pre-evaluation dense DVF .npy write in full output mode"},
    ) if timer is not None else None
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, dense_np)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    full_script_to_dvf_file_sec = float(time.time() - float(total_start)) if total_start is not None else float("nan")
    loaded_images_to_dvf_file_sec = (
        full_script_to_dvf_file_sec - float(case_load_complete_sec)
        if case_load_complete_sec is not None and math.isfinite(full_script_to_dvf_file_sec)
        else float("nan")
    )
    if timer is not None:
        timer.stop(write_token)
        timer.mark(
            "dvf_file_ready",
            metadata={
                "path": str(path),
                "definition": "full-mode endpoint immediately after dense DVF .npy write, before evaluation",
                "output_mode": "full",
            },
            sync=False,
        )
    return {
        "pre_evaluation_dvf_file_saved": True,
        "dvf_file_path": str(path),
        "dense_output_units": "normalized_grid_xyz_displacement",
        "dense_output_shape_zyx3": list(dense_np.shape),
        "dense_output_dtype": str(dense_np.dtype),
        "dense_output_file_format": "numpy_npy",
        "full_script_to_dvf_file_sec": full_script_to_dvf_file_sec,
        "loaded_images_to_dvf_file_sec": loaded_images_to_dvf_file_sec,
    }


def train_spline(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
    timer: Optional[TimingRecorder] = None,
    run_dir: Optional[Path] = None,
    total_start: Optional[float] = None,
    case_load_complete_sec: Optional[float] = None,
) -> Dict[str, object]:
    cps = method_cps(args.method)
    if cps is None:
        raise ValueError(f"No cps for method {args.method}")
    single_stage_early_stop_mode = str(args.stage_early_stop_mode)
    if str(args.stage_min_epochs).strip():
        single_stage_min_epochs = c.parse_stage_ints(str(args.stage_min_epochs), 1)[0]
    else:
        single_stage_min_epochs = int(args.epochs)
    single_stage_patience = c.parse_stage_ints(str(args.stage_patience), 1)[0]
    single_stage_check_interval = c.parse_stage_ints(str(args.stage_check_interval), 1)[0]
    if str(args.stage_min_delta).strip():
        single_stage_min_delta = c.parse_float_list(str(args.stage_min_delta), 1, 0.0)[0]
    else:
        single_stage_min_delta = 0.0

    dense_grid_token = timer.start(
        "training.initial_dense_grid",
        category="fixed_stage_overhead",
        metadata={"shape": list(c.IMAGE_SHAPE), "cps": int(cps)},
    ) if timer is not None else None
    control_coords, control_shape = c.bspline_control_coords(c.IMAGE_SHAPE, cps, device)
    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    if timer is not None:
        timer.stop(dense_grid_token)

    moving = batch["moving"]
    fixed_image = batch["fixed"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)

    train_token = timer.start(
        "training",
        category="training",
        metadata={
            "stage_full_cps": [int(cps)],
            "stage_downsample": [1],
            "stage_requested_epochs": [int(args.epochs)],
        },
    ) if timer is not None else None
    stage_token = timer.start(
        "stage_0",
        category="training_stage",
        metadata={
            "stage_idx": 0,
            "full_cps": int(cps),
            "downsample": 1,
            "requested_epochs": int(args.epochs),
            "lr": float(lr),
            "single_stage_sinr_or_directcp": bool(not args.method.startswith("direct_cp_")),
        },
    ) if timer is not None else None
    setup_token = timer.start(
        "stage_0.stage_setup",
        category="fixed_stage_overhead",
        metadata={"stage_idx": 0, "cps": int(cps)},
    ) if timer is not None else None
    generator = build_spline_generator(args, control_shape).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    if timer is not None:
        stage_setup_sec = timer.stop(setup_token)
    else:
        stage_setup_sec = 0.0

    history: List[Dict[str, float]] = []
    global_ncc_diagnostic: List[Dict[str, float | str | bool]] = []
    best_loss = float("inf")
    best_data_loss = float("nan")
    best_reg_loss = float("nan")
    best_epoch = 0
    early_stop_epoch: Optional[int] = None
    early_stop_reason = "requested_epochs_complete"
    if sampler is not None and sampler.enabled:
        generator.eval()
        with timer.section("training.initial_checkpoint_sample", category="fixed_stage_overhead") if timer is not None else _null_section():
            with torch.no_grad():
                init_output = generator(control_coords, control_shape)
                init_dense_disp = c.expand_bspline_controls(init_output, control_shape, c.IMAGE_SHAPE, cps)
            sampler.record_dense(
                epoch=0,
                dense_disp_norm_xyz=init_dense_disp,
                row_extra={"method": args.method, "checkpoint_source": "model_init"},
            )

    pre_loop_token = timer.start(
        "stage_0.pre_loop_cuda_sync_and_memory_reset",
        category="fixed_stage_overhead",
        metadata={
            "stage_idx": 0,
            "definition": "CUDA synchronization and peak-memory-stat reset before epoch timing starts",
        },
    ) if timer is not None else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if timer is not None:
        pre_loop_sync_sec = timer.stop(pre_loop_token)
    else:
        pre_loop_sync_sec = 0.0
    train_start = time.time()
    loop_token = timer.start(
        "stage_0.optimizer_loop",
        category="step_proportional",
        metadata={"stage_idx": 0, "requested_epochs": int(args.epochs)},
    ) if timer is not None else None
    phase_acc = CudaPhaseAccumulator(
        bool(timer is not None and timer.enabled and timer.detail == "deep"),
        device,
    )
    loop_logging_wall_sec = 0.0
    for epoch in range(1, int(args.epochs) + 1):
        generator.train()
        with phase_acc.phase("generator_forward"):
            output = generator(control_coords, control_shape)
        with phase_acc.phase("control_grid"):
            control_disp = output.reshape(*control_shape, 3)
        with phase_acc.phase("expand"):
            dense_disp = c.expand_bspline_controls(output, control_shape, c.IMAGE_SHAPE, cps)
        with phase_acc.phase("warp_ncc"):
            warped = c.warp_full(moving, id_dense + dense_disp, mode="bilinear")
            data_loss = c.ncc_loss(fixed_image[None, None], warped[None, None])
        with phase_acc.phase("regularizer"):
            reg_loss = c.analytic_bspline_sampled_be(
                control_disp,
                c.IMAGE_SHAPE,
                cps,
                component_reduction="mean",
            )
        loss = data_loss + float(args.alpha) * reg_loss
        with phase_acc.phase("backward_step"):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        logging_start = time.perf_counter()
        entry = {
            "epoch": float(epoch),
            "global_epoch": float(epoch),
            "stage_idx": 0.0,
            "stage_epoch": float(epoch),
            "full_cps": float(cps),
            "downsample": 1.0,
            "domain_cps": float(cps),
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "alpha": float(args.alpha),
            "lr": float(lr),
        }
        history.append(entry)
        current_loss = float(entry["loss"])
        if best_epoch == 0 or (best_loss - current_loss) >= float(single_stage_min_delta):
            best_loss = current_loss
            best_data_loss = float(entry["data_loss"])
            best_reg_loss = float(entry["reg_loss"])
            best_epoch = int(epoch)
        if should_global_ncc_diagnostic(args, epoch, int(args.epochs)):
            fixed_positive_fields = fixed_image_positive_ncc_fields(fixed_image, warped, fixed_mask)
            global_ncc_diagnostic.append(
                {
                    "epoch": float(epoch),
                    "global_epoch": float(epoch),
                    "global_ncc_loss": float(data_loss.detach().cpu()),
                    "global_ncc_value": float((-data_loss).detach().cpu()),
                    "diagnostic_name": "full_image_global_ncc",
                    "eval_domain": "full_image",
                    "affects_training": False,
                    "eval_elapsed_sec": 0.0,
                    "stage_idx": 0.0,
                    "stage_epoch": float(epoch),
                    "full_cps": float(cps),
                    "downsample": 1.0,
                    "domain_cps": float(cps),
                    **fixed_positive_fields,
                }
            )
            print(json.dumps({"global_ncc_diagnostic": global_ncc_diagnostic[-1]}), flush=True)
        if sampler is not None and sampler.should_record(epoch):
            generator.eval()
            with torch.no_grad():
                sample_output = generator(control_coords, control_shape)
                sample_dense = c.expand_bspline_controls(sample_output, control_shape, c.IMAGE_SHAPE, cps)
            sampler.record_dense(
                epoch=epoch,
                dense_disp_norm_xyz=sample_dense,
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "stage_idx": 0,
                    "stage_epoch": int(epoch),
                    "full_cps": int(cps),
                    "downsample": 1,
                    "loss": entry["loss"],
                    "data_loss": entry["data_loss"],
                    "reg_loss": entry["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            print(json.dumps(entry), flush=True)
        loop_logging_wall_sec += time.perf_counter() - logging_start
        if str(single_stage_early_stop_mode) == "adaptive":
            can_stop = (
                int(epoch) >= int(single_stage_min_epochs)
                and int(epoch) % max(1, int(single_stage_check_interval)) == 0
                and (int(epoch) - int(best_epoch)) >= max(1, int(single_stage_patience))
            )
            if can_stop:
                early_stop_epoch = int(epoch)
                early_stop_reason = (
                    "adaptive_plateau:"
                    f"epoch={int(epoch)};best_epoch={int(best_epoch)};"
                    f"patience={int(single_stage_patience)};"
                    f"min_delta={float(single_stage_min_delta):.6g};"
                    f"best_loss={float(best_loss):.8g};"
                    f"current_loss={float(current_loss):.8g}"
                )
                print(
                    json.dumps(
                        {
                            "single_stage_stop": 0,
                            "epoch": int(epoch),
                            "requested_epochs": int(args.epochs),
                            "stop_reason": early_stop_reason,
                        }
                    ),
                    flush=True,
                )
                break
    if timer is not None:
        optimizer_loop_sec = timer.stop(loop_token)
    else:
        optimizer_loop_sec = 0.0
    actual_epochs = int(history[-1]["epoch"]) if history else 0
    phase_summary = phase_acc.summary()
    phase_summary["logging_early_stop_wall_sec"] = float(loop_logging_wall_sec)
    phase_summary["actual_epochs"] = float(actual_epochs)
    phase_summary["stage_idx"] = 0.0
    phase_summary["full_cps"] = float(cps)
    phase_summary["downsample"] = 1.0
    if timer is not None:
        timer.add_stage_phase_summary(
            {
                "stage_idx": 0,
                "full_cps": int(cps),
                "downsample": 1,
                "requested_epochs": int(args.epochs),
                "actual_epochs": int(actual_epochs),
                "optimizer_loop_sec": float(optimizer_loop_sec),
                "phase_totals_sec": phase_summary,
            }
        )

    final_dense_token = timer.start(
        "stage_0.final_output_dense_expand",
        category="final_output_dense_expand",
        metadata={"stage_idx": 0, "full_cps": int(cps)},
    ) if timer is not None else None
    generator.eval()
    with torch.no_grad():
        final_output = generator(control_coords, control_shape)
        final_control_disp = final_output.reshape(*control_shape, 3).detach()
        final_dense_disp = c.expand_bspline_controls(final_output, control_shape, c.IMAGE_SHAPE, cps).detach()
    if timer is not None:
        final_output_dense_expand_sec = timer.stop(final_dense_token)
        if str(getattr(args, "output_mode", "full")) == "dvf_only":
            timer.mark(
                "dvf_in_memory_ready",
                metadata={
                    "stage_idx": 0,
                    "full_cps": int(cps),
                    "definition": "immediately after single-stage final dense displacement expansion",
                },
                sync=False,
            )
    else:
        final_output_dense_expand_sec = 0.0

    stats_token = timer.start(
        "stage_0.control_summary_stats",
        category="fixed_stage_overhead",
        metadata={"stage_idx": 0},
    ) if timer is not None else None
    ctrl_abs = final_control_disp.detach().abs().reshape(-1)
    control_abs_max_norm = float(ctrl_abs.max().detach().cpu())
    control_abs_p95_norm = float(torch.quantile(ctrl_abs, 0.95).detach().cpu())
    if timer is not None:
        control_summary_stats_sec = timer.stop(stats_token)
    else:
        control_summary_stats_sec = 0.0
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    stage_timing = {
        "loss_volume_setup_sec": 0.0,
        "stage_setup_sec": float(stage_setup_sec + pre_loop_sync_sec),
        "model_optimizer_setup_sec": float(stage_setup_sec),
        "pre_loop_cuda_sync_and_memory_reset_sec": float(pre_loop_sync_sec),
        "optimizer_loop_sec": float(optimizer_loop_sec),
        "intermediate_dense_expand_sec": 0.0,
        "final_output_dense_expand_sec": float(final_output_dense_expand_sec),
        "final_dense_expand_sec": float(final_output_dense_expand_sec),
        "dense_expand_role": "final_output",
        "control_summary_stats_sec": float(control_summary_stats_sec),
    }
    if timer is not None:
        stage_timing["stage_total_sec"] = timer.stop(stage_token)
        timer.stop(train_token)

    dvf_endpoint_info: Dict[str, object] = {}
    if str(getattr(args, "output_mode", "full")) == "full" and run_dir is not None:
        dvf_endpoint_info = save_full_mode_dvf_endpoint(
            args,
            batch,
            final_dense_disp,
            run_dir,
            timer=timer,
            total_start=total_start,
            case_load_complete_sec=case_load_complete_sec,
        )

    if str(getattr(args, "output_mode", "full")) == "dvf_only":
        metrics: Dict[str, object] = {
            "output_mode": "dvf_only",
            "no_evaluation": True,
            "evaluation_skipped": [
                "image_warp_evaluation",
                "segmentation_warp",
                "dice",
                "hd95_asd",
                "dense_ncc",
                "regularity_metrics",
                "model_control_checkpoint_saves",
                "full_csv_packaging",
            ],
        }
    else:
        eval_start = time.time()
        metrics = timed_evaluate_from_dense_coords(id_dense + final_dense_disp, batch, timer=timer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics["post_dvf_evaluation_sec"] = float(time.time() - eval_start)
    metrics.update(dvf_endpoint_info)
    stage_summary = {
        "stage_idx": 0,
        "full_cps": int(cps),
        "downsample": 1,
        "domain_cps": int(cps),
        "control_shape_zyx": list(int(v) for v in control_shape),
        "eval_shape_zyx": list(int(v) for v in c.IMAGE_SHAPE),
        "n_control": int(np.prod(tuple(int(v) for v in control_shape))),
        "epochs": int(actual_epochs),
        "requested_epochs": int(args.epochs),
        "actual_epochs": int(actual_epochs),
        "lr": float(lr),
        "alpha": float(args.alpha),
        "early_stop_mode": str(single_stage_early_stop_mode),
        "min_epochs": int(single_stage_min_epochs),
        "patience": int(single_stage_patience) if str(single_stage_early_stop_mode) == "adaptive" else 0,
        "check_interval": int(single_stage_check_interval),
        "rel_delta_total": 0.0,
        "rel_delta_data": 0.0,
        "min_delta": float(single_stage_min_delta),
        "stop_reason": str(early_stop_reason),
        "early_stop_epoch": int(early_stop_epoch) if early_stop_epoch is not None else None,
        "best_loss": float(best_loss),
        "best_loss_epoch": int(best_epoch),
        "best_data_loss": float(best_data_loss),
        "best_reg_loss": float(best_reg_loss),
        "loss_mode": "sinr_ncc_bspline_be",
        "antialias_sigma": 0.0,
        "antialias_kernel_size": 0,
        "elapsed_sec": float(training_elapsed_sec),
        "timing_breakdown": stage_timing,
        "timing_phase_totals_sec": phase_summary,
        "control_abs_max_norm": control_abs_max_norm,
        "control_abs_p95_norm": control_abs_p95_norm,
    }
    metrics["history"] = history
    metrics["stage_history"] = history
    metrics["global_ncc_diagnostic"] = global_ncc_diagnostic
    metrics["stage_summaries"] = [stage_summary]
    metrics["prolongation_diagnostics"] = []
    metrics["generator"] = generator
    metrics["control_shape"] = control_shape
    metrics["control_coords"] = control_coords.detach()
    metrics["control_disp"] = final_control_disp
    metrics["dense_disp"] = final_dense_disp
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    metrics["param_count"] = c.count_trainable_params(generator)
    metrics["cps"] = cps
    metrics["stage_full_cps"] = [int(cps)]
    metrics["stage_downsample"] = [1]
    metrics["stage_epochs"] = [int(args.epochs)]
    metrics["stage_requested_epochs"] = [int(args.epochs)]
    metrics["stage_actual_epochs"] = [int(actual_epochs)]
    metrics["stage_lr"] = [float(lr)]
    metrics["stage_early_stop_mode"] = str(single_stage_early_stop_mode)
    metrics["stage_min_epochs"] = [int(single_stage_min_epochs)]
    metrics["stage_patience"] = [int(single_stage_patience) if str(single_stage_early_stop_mode) == "adaptive" else 0]
    metrics["stage_check_interval"] = [int(single_stage_check_interval)]
    metrics["stage_rel_delta_total"] = [0.0]
    metrics["stage_rel_delta_data"] = [0.0]
    metrics["stage_min_delta"] = [float(single_stage_min_delta)]
    metrics["stage_stop_reasons"] = [str(early_stop_reason)]
    metrics["early_stop_epoch"] = int(early_stop_epoch) if early_stop_epoch is not None else None
    metrics["early_stop_reason"] = str(early_stop_reason)
    metrics["best_loss"] = float(best_loss)
    metrics["best_loss_epoch"] = int(best_epoch)
    metrics["best_data_loss"] = float(best_data_loss)
    metrics["best_reg_loss"] = float(best_reg_loss)
    metrics["prolongation_preserve_check"] = False
    metrics["save_intermediate_dense"] = False
    metrics["nested_control_shapes_zyx"] = {str(int(cps)): list(int(v) for v in control_shape)}
    metrics["prolongation"] = "none_single_stage"
    metrics["actual_epochs"] = int(actual_epochs)
    metrics["requested_epochs"] = int(args.epochs)
    return metrics


def prolongate_control_grid(
    prev_control: torch.Tensor,
    prev_shape: Sequence[int],
    target_shape: Sequence[int],
    target_cps: int,
    image_shape: Sequence[int],
    stage_idx: int,
    *,
    prev_dense_full: Optional[torch.Tensor] = None,
    preserve_check: bool = False,
    timer: Optional[TimingRecorder] = None,
) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    knot_token = timer.start(
        f"stage_{stage_idx}.prolongation.pure_knot_insertion",
        category="pure_knot_insertion",
        metadata={"stage_idx": int(stage_idx), "target_cps": int(target_cps)},
    ) if timer is not None else None
    with torch.no_grad():
        prev_grid = c.flat_to_grid(prev_control.detach(), prev_shape)
        exact_grid = c.cubic_bspline_dyadic_knot_insert_3d(prev_grid, target_shape)
    pure_knot_sec = timer.stop(knot_token) if timer is not None else 0.0
    row: Dict[str, object] = {
        "stage_idx": int(stage_idx),
        "target_cps": int(target_cps),
        "method": "exact_knot",
        "source_shape_zyx": list(int(v) for v in prev_shape),
        "target_shape_zyx": list(int(v) for v in target_shape),
        "preserve_check_enabled": bool(preserve_check),
        "timing_pure_knot_insertion_sec": float(pure_knot_sec),
        "timing_prolongation_preserve_dense_expand_sec": 0.0,
        "timing_prolongation_preserve_metrics_sec": 0.0,
    }
    if preserve_check:
        if prev_dense_full is None:
            raise RuntimeError("prolongation preserve check requires previous-stage dense field.")
        dense_token = timer.start(
            f"stage_{stage_idx}.prolongation.preserve_dense_expand",
            category="prolongation_preserve_dense_expand",
            metadata={"stage_idx": int(stage_idx), "target_cps": int(target_cps)},
        ) if timer is not None else None
        with torch.no_grad():
            dense = c.expand_on_shape(c.grid_to_flat(exact_grid), target_shape, target_cps, image_shape, 1)
        row["timing_prolongation_preserve_dense_expand_sec"] = (
            timer.stop(dense_token) if timer is not None else 0.0
        )
        metrics_token = timer.start(
            f"stage_{stage_idx}.prolongation.preserve_metrics",
            category="prolongation_preserve_metrics",
            metadata={"stage_idx": int(stage_idx)},
        ) if timer is not None else None
        row.update(c.field_difference_metrics(dense - prev_dense_full, prefix="preserve_"))
        row["timing_prolongation_preserve_metrics_sec"] = (
            timer.stop(metrics_token) if timer is not None else 0.0
        )
    rows: List[Dict[str, object]] = [row]
    return c.grid_to_flat(exact_grid).detach(), rows


def stage_stop_decision(
    rows: Sequence[Dict[str, object]],
    *,
    min_epochs: int,
    patience: int,
    check_interval: int,
    rel_delta_total: float,
    rel_delta_data: float,
) -> Tuple[bool, str]:
    if not rows:
        return False, ""
    epoch = int(float(rows[-1]["epoch"]))
    if epoch < int(min_epochs) or epoch % int(check_interval) != 0:
        return False, ""
    window = max(1, int(check_interval))
    patience = max(window, int(patience))
    if len(rows) < patience + window:
        return False, ""

    def median_metric(metric: str, start: int, stop: int) -> float:
        values = [float(row[metric]) for row in rows[start:stop]]
        return float(np.median(np.asarray(values, dtype=np.float64)))

    recent_start = len(rows) - window
    prev_stop = len(rows) - patience
    prev_start = max(0, prev_stop - window)
    recent_loss = median_metric("loss", recent_start, len(rows))
    prev_loss = median_metric("loss", prev_start, prev_stop)
    recent_data = median_metric("data_loss", recent_start, len(rows))
    prev_data = median_metric("data_loss", prev_start, prev_stop)

    total_improvement = prev_loss - recent_loss
    data_improvement = prev_data - recent_data
    rel_total = total_improvement / max(abs(prev_loss), 1e-8)
    rel_data = data_improvement / max(abs(prev_data), 1e-8)
    if rel_total < float(rel_delta_total) and rel_data < float(rel_delta_data):
        return (
            True,
            "adaptive_plateau:"
            f"epoch={epoch};patience={patience};window={window};"
            f"rel_total={rel_total:.6g};rel_data={rel_data:.6g};"
            f"threshold_total={float(rel_delta_total):.6g};"
            f"threshold_data={float(rel_delta_data):.6g}",
        )
    return False, ""


def optimize_multistage_stage(
    args: argparse.Namespace,
    fixed_image: torch.Tensor,
    moving: torch.Tensor,
    fixed_mask: torch.Tensor,
    control: torch.nn.Parameter,
    control_shape: Sequence[int],
    full_cps: int,
    downsample: int,
    epochs: int,
    lr: float,
    alpha: float,
    stage_idx: int,
    global_epoch_offset: int,
    total_epochs: int,
    global_ncc_diagnostic: List[Dict[str, float | str | bool]],
    early_stop_mode: str,
    min_epochs: int,
    patience: int,
    check_interval: int,
    rel_delta_total: float,
    rel_delta_data: float,
    sampler: Optional[CheckpointSampler] = None,
    timer: Optional[TimingRecorder] = None,
    is_final_stage: bool = False,
    require_stage_dense: bool = True,
) -> Tuple[Optional[torch.Tensor], Dict[str, object], List[Dict[str, object]]]:
    stage_timing: Dict[str, float] = {}
    stage_token = timer.start(
        f"stage_{stage_idx}",
        category="training_stage",
        metadata={
            "stage_idx": int(stage_idx),
            "full_cps": int(full_cps),
            "downsample": int(downsample),
            "requested_epochs": int(epochs),
            "lr": float(lr),
            "alpha": float(alpha),
        },
    ) if timer is not None else None
    loss_token = timer.start(
        f"stage_{stage_idx}.loss_volume_setup",
        category="fixed_stage_overhead",
        metadata={"stage_idx": int(stage_idx), "downsample": int(downsample)},
    ) if timer is not None else None
    fixed_l, blur_sigma, blur_kernel = c.make_loss_volume(
        fixed_image,
        int(downsample),
        float(args.antialias_sigma_scale),
        float(args.antialias_radius_sigma),
    )
    moving_l, _sigma_m, _kernel_m = c.make_loss_volume(
        moving,
        int(downsample),
        float(args.antialias_sigma_scale),
        float(args.antialias_radius_sigma),
    )
    if timer is not None:
        stage_timing["loss_volume_setup_sec"] = timer.stop(loss_token)

    setup_token = timer.start(
        f"stage_{stage_idx}.stage_setup",
        category="fixed_stage_overhead",
        metadata={"stage_idx": int(stage_idx)},
    ) if timer is not None else None
    eval_shape = tuple(int(v) for v in fixed_l.shape)
    domain_cps = c.domain_cps_from_full(int(full_cps), int(downsample))
    id_grid = c.make_coordinate_grid(eval_shape, fixed_image.device).detach()
    components = timer.components if timer is not None else None
    with components.scope("optimizer_creation", cuda=False) if components is not None else _null_section():
        optimizer = torch.optim.Adam([control], lr=float(lr))
    if timer is not None:
        stage_timing["stage_setup_sec"] = timer.stop(setup_token)

    rows: List[Dict[str, object]] = []
    loss_buffer = None
    if getattr(args, "loss_history_mode", "cpu") == "gpu_buffer":
        from scripts.experiments.mrdbscp842_acceleration.backends import LossBuffer
        loss_buffer = LossBuffer(int(epochs), fixed_image.device)
    snapshot = getattr(args, "_control_snapshot_callback", None)
    if snapshot is not None:
        snapshot(stage_idx, 0, control, control_shape, eval_shape, domain_cps, False)
    stop_reason = "requested_epochs_complete"
    stage_start = time.time()
    loop_token = timer.start(
        f"stage_{stage_idx}.optimizer_loop",
        category="step_proportional",
        metadata={"stage_idx": int(stage_idx), "requested_epochs": int(epochs)},
    ) if timer is not None else None
    phase_acc = CudaPhaseAccumulator(
        bool(timer is not None and timer.enabled and timer.detail == "deep"),
        fixed_image.device,
    )
    loop_logging_early_stop_wall_sec = 0.0
    print(
        json.dumps(
            {
                "stage_start": int(stage_idx),
                "full_cps": int(full_cps),
                "downsample": int(downsample),
                "domain_cps": int(domain_cps),
                "control_shape_zyx": list(int(v) for v in control_shape),
                "eval_shape_zyx": list(eval_shape),
                "epochs": int(epochs),
                "lr": float(lr),
                "alpha": float(alpha),
                "early_stop_mode": str(early_stop_mode),
                "min_epochs": int(min_epochs),
                "patience": int(patience),
                "check_interval": int(check_interval),
                "rel_delta_total": float(rel_delta_total),
                "rel_delta_data": float(rel_delta_data),
                "image_loss": "sinr_ncc",
                "regularizer": "analytic_cubic_bspline_be",
            }
        ),
        flush=True,
    )
    for epoch in range(1, int(epochs) + 1):
        iteration_token = None
        if components is not None:
            components.epoch = epoch
            iteration_token = components.start("iteration")
            dense_disp, data_loss, reg_loss, loss = components.training_step(
                c, control, control_shape, full_cps, eval_shape, downsample,
                moving_l, fixed_l, id_grid, domain_cps, alpha, optimizer)
        else:
            with phase_acc.phase("control_grid"):
                control_grid = c.flat_to_grid(control, control_shape)
            with phase_acc.phase("expand"):
                dense_disp = c.expand_on_shape(control, control_shape, int(full_cps), eval_shape, int(downsample))
            with phase_acc.phase("warp_ncc"):
                warped = c.warp_full(moving_l, id_grid + dense_disp, mode="bilinear")
                data_loss = c.ncc_loss(fixed_l[None, None], warped[None, None])
            with phase_acc.phase("regularizer"):
                reg_loss = c.analytic_bspline_sampled_be(
                    control_grid, eval_shape, int(domain_cps), component_reduction="mean")
            loss = data_loss + float(alpha) * reg_loss
            with phase_acc.phase("backward_step"):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        logging_start = time.perf_counter()
        host_token = components.start("host_logging_and_checks", cuda=False) if components is not None else None
        scalar_token = components.start("scalar_transfers", cuda=False) if components is not None else None
        row = {
            "stage_idx": float(stage_idx),
            "epoch": float(epoch),
            "global_epoch": float(int(global_epoch_offset) + int(epoch)),
            "full_cps": float(full_cps),
            "downsample": float(downsample),
            "domain_cps": float(domain_cps),
            "loss": float(loss.detach().cpu()) if loss_buffer is None else None,
            "data_loss": float(data_loss.detach().cpu()) if loss_buffer is None else None,
            "reg_loss": float(reg_loss.detach().cpu()) if loss_buffer is None else None,
            "alpha": float(alpha),
            "lr": float(lr),
        }
        rows.append(row)
        if components is not None:
            components.stop(scalar_token)
        if loss_buffer is not None:
            loss_buffer.append(epoch - 1, loss, data_loss, reg_loss)
            check_due = (str(early_stop_mode) == "adaptive" and epoch >= int(min_epochs)
                         and epoch % int(check_interval) == 0)
            if (check_due or epoch == 1 or epoch % int(args.eval_interval) == 0
                    or epoch == int(epochs)
                    or (sampler is not None and sampler.should_record(int(row["global_epoch"])))
                    or should_global_ncc_diagnostic(args, int(row["global_epoch"]), int(total_epochs))):
                loss_buffer.flush(rows)
        if snapshot is not None:
            snapshot(stage_idx, epoch, control, control_shape, eval_shape, domain_cps, False)
        if should_global_ncc_diagnostic(args, int(row["global_epoch"]), int(total_epochs)):
            with torch.no_grad():
                dense_full_diag = c.expand_on_shape(control, control_shape, int(full_cps), c.IMAGE_SHAPE, 1)
                dense_coords_diag = c.make_coordinate_grid(c.IMAGE_SHAPE, fixed_image.device).detach() + dense_full_diag
            diag_row = global_ncc_diagnostic_row(
                fixed=fixed_image,
                moving=moving,
                dense_coords=dense_coords_diag,
                fixed_mask=fixed_mask,
                epoch=int(epoch),
                global_epoch=int(row["global_epoch"]),
                eval_domain="full_image",
                device=fixed_image.device,
            )
            diag_row["stage_idx"] = float(stage_idx)
            diag_row["stage_epoch"] = float(epoch)
            diag_row["full_cps"] = float(full_cps)
            diag_row["downsample"] = float(downsample)
            diag_row["domain_cps"] = float(domain_cps)
            global_ncc_diagnostic.append(diag_row)
            print(json.dumps({"global_ncc_diagnostic": diag_row}), flush=True)
        if sampler is not None and sampler.should_record(int(row["global_epoch"])):
            with torch.no_grad():
                sample_dense_full = c.expand_on_shape(control, control_shape, int(full_cps), c.IMAGE_SHAPE, 1)
            sampler.record_dense(
                epoch=int(row["global_epoch"]),
                dense_disp_norm_xyz=sample_dense_full,
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "stage_idx": int(stage_idx),
                    "stage_epoch": int(epoch),
                    "full_cps": int(full_cps),
                    "downsample": int(downsample),
                    "loss": row["loss"],
                    "data_loss": row["data_loss"],
                    "reg_loss": row["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == int(epochs):
            print(json.dumps(row), flush=True)
        should_break = False
        if str(early_stop_mode) == "adaptive":
            stop_token = components.start("early_stop_check", cuda=False) if components is not None else None
            should_stop, reason = stage_stop_decision(
                rows,
                min_epochs=int(min_epochs),
                patience=int(patience),
                check_interval=int(check_interval),
                rel_delta_total=float(rel_delta_total),
                rel_delta_data=float(rel_delta_data),
            )
            if components is not None:
                components.stop(stop_token)
            if should_stop:
                stop_reason = reason
                print(
                    json.dumps(
                        {
                            "stage_stop": int(stage_idx),
                            "epoch": int(epoch),
                            "requested_epochs": int(epochs),
                            "stop_reason": stop_reason,
                        }
                    ),
                    flush=True,
                )
                should_break = True
        loop_logging_early_stop_wall_sec += time.perf_counter() - logging_start
        if components is not None:
            components.stop(host_token)
            components.stop(iteration_token)
        if should_break:
            break
    if components is not None:
        components.epoch = 0
    if loss_buffer is not None:
        loss_buffer.flush(rows)
    if snapshot is not None:
        snapshot(stage_idx, len(rows), control, control_shape, eval_shape, domain_cps, True)
    if timer is not None:
        stage_timing["optimizer_loop_sec"] = timer.stop(loop_token)
    phase_summary = phase_acc.summary()
    phase_summary["logging_early_stop_wall_sec"] = float(loop_logging_early_stop_wall_sec)
    phase_summary["actual_epochs"] = float(int(rows[-1]["epoch"]) if rows else 0)
    phase_summary["stage_idx"] = float(stage_idx)
    phase_summary["full_cps"] = float(full_cps)
    phase_summary["downsample"] = float(downsample)
    if timer is not None:
        timer.add_stage_phase_summary(
            {
                "stage_idx": int(stage_idx),
                "full_cps": int(full_cps),
                "downsample": int(downsample),
                "requested_epochs": int(epochs),
                "actual_epochs": int(rows[-1]["epoch"]) if rows else 0,
                "optimizer_loop_sec": float(stage_timing.get("optimizer_loop_sec", 0.0)),
                "phase_totals_sec": phase_summary,
            }
        )

    dense_full: Optional[torch.Tensor] = None
    stage_timing["intermediate_dense_expand_sec"] = 0.0
    stage_timing["final_output_dense_expand_sec"] = 0.0
    stage_timing["final_dense_expand_sec"] = 0.0
    stage_timing["dense_expand_role"] = "skipped_intermediate"
    if bool(require_stage_dense):
        dense_name = "final_output_dense_expand" if is_final_stage else "intermediate_dense_expand"
        dense_category = "final_output_dense_expand" if is_final_stage else "intermediate_dense_diagnostic"
        dense_token = timer.start(
            f"stage_{stage_idx}.{dense_name}",
            category=dense_category,
            metadata={"stage_idx": int(stage_idx), "full_cps": int(full_cps)},
        ) if timer is not None else None
        with torch.no_grad():
            dense_full = c.expand_on_shape(control, control_shape, int(full_cps), c.IMAGE_SHAPE, 1)
        dense_elapsed = timer.stop(dense_token) if timer is not None else 0.0
        if is_final_stage:
            stage_timing["final_output_dense_expand_sec"] = float(dense_elapsed)
            stage_timing["dense_expand_role"] = "final_output"
        else:
            stage_timing["intermediate_dense_expand_sec"] = float(dense_elapsed)
            stage_timing["dense_expand_role"] = "intermediate_diagnostic_or_save"
        stage_timing["final_dense_expand_sec"] = float(dense_elapsed)
        if is_final_stage and str(getattr(args, "output_mode", "full")) == "dvf_only" and timer is not None:
            timer.mark(
                "dvf_in_memory_ready",
                metadata={
                    "stage_idx": int(stage_idx),
                    "full_cps": int(full_cps),
                    "definition": "immediately after final-stage dense displacement expansion",
                },
                sync=False,
            )
    stats_token = timer.start(
        f"stage_{stage_idx}.control_summary_stats",
        category="fixed_stage_overhead",
        metadata={"stage_idx": int(stage_idx)},
    ) if timer is not None else None
    ctrl_abs = control.detach().abs().reshape(-1)
    control_abs_max_norm = float(ctrl_abs.max().detach().cpu())
    if int(ctrl_abs.numel()) <= 16_000_000:
        control_abs_p95 = torch.quantile(ctrl_abs, 0.95)
        control_abs_p95_definition = "torch_quantile_linear"
    else:
        # torch.quantile rejects very large tensors because its flattened sort
        # index is limited. The cps1 grid crosses that limit, so use the
        # standard nearest-rank percentile without copying ~22M values to CPU.
        kth = max(1, min(int(ctrl_abs.numel()), int(math.ceil(0.95 * int(ctrl_abs.numel())))))
        control_abs_p95 = torch.kthvalue(ctrl_abs, kth).values
        control_abs_p95_definition = "nearest_rank_kthvalue"
    control_abs_p95_norm = float(control_abs_p95.detach().cpu())
    if timer is not None:
        stage_timing["control_summary_stats_sec"] = timer.stop(stats_token)
    actual_epochs = int(rows[-1]["epoch"]) if rows else 0
    summary = {
        "stage_idx": int(stage_idx),
        "representation": "bspline_control",
        "parameterization": "cubic_bspline_control_displacement",
        "bspline_order": 3,
        "regularizer": "analytic_cubic_bspline_be",
        "full_cps": int(full_cps),
        "downsample": int(downsample),
        "domain_cps": int(domain_cps),
        "control_shape_zyx": list(int(v) for v in control_shape),
        "eval_shape_zyx": list(eval_shape),
        "n_control": int(np.prod(tuple(int(v) for v in control_shape))),
        "epochs": int(actual_epochs),
        "requested_epochs": int(epochs),
        "actual_epochs": int(actual_epochs),
        "lr": float(lr),
        "alpha": float(alpha),
        "early_stop_mode": str(early_stop_mode),
        "min_epochs": int(min_epochs),
        "patience": int(patience),
        "check_interval": int(check_interval),
        "rel_delta_total": float(rel_delta_total),
        "rel_delta_data": float(rel_delta_data),
        "stop_reason": str(stop_reason),
        "loss_mode": "sinr_ncc_bspline_be",
        "antialias_sigma": float(blur_sigma),
        "antialias_kernel_size": int(blur_kernel),
        "elapsed_sec": float(time.time() - stage_start),
        "timing_breakdown": stage_timing,
        "timing_phase_totals_sec": phase_summary,
        "control_abs_max_norm": control_abs_max_norm,
        "control_abs_p95_norm": control_abs_p95_norm,
        "control_abs_p95_definition": control_abs_p95_definition,
    }
    if timer is not None:
        stage_timing["stage_total_sec"] = timer.stop(stage_token)
    return dense_full.detach() if dense_full is not None else None, summary, rows


def optimize_dense_stage(
    args: argparse.Namespace,
    fixed_image: torch.Tensor,
    moving: torch.Tensor,
    initial_dense: torch.Tensor,
    stage_idx: int,
    global_epoch_offset: int,
    total_epochs: int,
    sampler: Optional[CheckpointSampler] = None,
    timer: Optional[TimingRecorder] = None,
) -> Tuple[torch.Tensor, Dict[str, object], List[Dict[str, object]]]:
    """Optimize an absolute, full-resolution dense displacement initialized from cps2."""
    alpha = float(args.alpha if args.dense_stage_alpha is None else args.dense_stage_alpha)
    lr = float(args.dense_stage_lr)
    epochs = int(args.dense_stage_epochs)
    min_epochs = int(args.dense_stage_min_epochs)
    patience = int(args.dense_stage_patience)
    check_interval = int(args.dense_stage_check_interval)
    rel_delta_total = float(args.dense_stage_rel_delta_total)
    rel_delta_data = float(args.dense_stage_rel_delta_data)
    early_stop_mode = str(args.dense_stage_early_stop_mode)

    if tuple(initial_dense.shape) != (*c.IMAGE_SHAPE, 3):
        raise ValueError(
            f"Dense initialization shape {tuple(initial_dense.shape)} != expected {(*c.IMAGE_SHAPE, 3)}"
        )
    if not torch.isfinite(initial_dense).all():
        raise ValueError("Dense initialization contains non-finite values.")

    stage_timing: Dict[str, float] = {}
    stage_token = timer.start(
        f"stage_{stage_idx}",
        category="training_stage",
        metadata={
            "stage_idx": int(stage_idx),
            "representation": "dense_voxel",
            "requested_epochs": epochs,
            "lr": lr,
            "alpha": alpha,
        },
    ) if timer is not None else None
    setup_token = timer.start(
        f"stage_{stage_idx}.stage_setup",
        category="fixed_stage_overhead",
        metadata={"stage_idx": int(stage_idx), "representation": "dense_voxel"},
    ) if timer is not None else None
    dense = torch.nn.Parameter(initial_dense.detach().clone())
    initialization_audit = c.field_difference_metrics(
        dense.detach() - initial_dense.detach(), prefix="dense_init_"
    )
    if any(float(value) != 0.0 for value in initialization_audit.values()):
        raise RuntimeError(f"Dense initialization was not exact: {initialization_audit}")
    id_grid = c.make_coordinate_grid(c.IMAGE_SHAPE, fixed_image.device).detach()
    optimizer = torch.optim.Adam([dense], lr=lr)
    if timer is not None:
        stage_timing["stage_setup_sec"] = timer.stop(setup_token)

    print(
        json.dumps(
            {
                "stage_start": int(stage_idx),
                "representation": "dense_voxel",
                "dense_shape_zyx3": list(dense.shape),
                "epochs": epochs,
                "lr": lr,
                "alpha": alpha,
                "early_stop_mode": early_stop_mode,
                "min_epochs": min_epochs,
                "patience": patience,
                "check_interval": check_interval,
                "rel_delta_total": rel_delta_total,
                "rel_delta_data": rel_delta_data,
                "image_loss": "full_image_global_ncc",
                "regularizer": "unmasked_forward_valid_dense_fd_be",
                "regularizer_component_reduction": "mean",
                "initialization": "expanded_cps2_absolute_field_detach_clone",
                **initialization_audit,
            }
        ),
        flush=True,
    )

    rows: List[Dict[str, object]] = []
    stop_reason = "requested_epochs_complete"
    stage_start = time.time()
    loop_token = timer.start(
        f"stage_{stage_idx}.optimizer_loop",
        category="step_proportional",
        metadata={"stage_idx": int(stage_idx), "requested_epochs": epochs},
    ) if timer is not None else None
    for epoch in range(1, epochs + 1):
        warped = c.warp_full(moving, id_grid + dense, mode="bilinear")
        data_loss = c.ncc_loss(fixed_image[None, None], warped[None, None])
        reg_loss = c.forward_valid_be(
            dense.permute(3, 0, 1, 2).unsqueeze(0),
            None,
            component_reduction="mean",
        )
        loss = data_loss + alpha * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        row: Dict[str, object] = {
            "stage_idx": float(stage_idx),
            "representation": "dense_voxel",
            "epoch": float(epoch),
            "global_epoch": float(int(global_epoch_offset) + epoch),
            "full_cps": None,
            "downsample": 1.0,
            "domain_cps": None,
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "alpha": alpha,
            "lr": lr,
        }
        rows.append(row)
        if sampler is not None and sampler.should_record(int(row["global_epoch"])):
            sampler.record_dense(
                epoch=int(row["global_epoch"]),
                dense_disp_norm_xyz=dense,
                row_extra={
                    "method": args.method,
                    "checkpoint_source": "post_optimizer_step",
                    "stage_idx": int(stage_idx),
                    "stage_epoch": int(epoch),
                    "representation": "dense_voxel",
                    "loss": row["loss"],
                    "data_loss": row["data_loss"],
                    "reg_loss": row["reg_loss"],
                },
            )
        if epoch == 1 or epoch % int(args.eval_interval) == 0 or epoch == epochs:
            print(json.dumps(row), flush=True)
        if early_stop_mode == "adaptive":
            should_stop, reason = stage_stop_decision(
                rows,
                min_epochs=min_epochs,
                patience=patience,
                check_interval=check_interval,
                rel_delta_total=rel_delta_total,
                rel_delta_data=rel_delta_data,
            )
            if should_stop:
                stop_reason = reason
                print(
                    json.dumps(
                        {
                            "stage_stop": int(stage_idx),
                            "representation": "dense_voxel",
                            "epoch": int(epoch),
                            "requested_epochs": epochs,
                            "stop_reason": stop_reason,
                        }
                    ),
                    flush=True,
                )
                break

    if timer is not None:
        stage_timing["optimizer_loop_sec"] = timer.stop(loop_token)
    final_dense = dense.detach()
    if not torch.isfinite(final_dense).all():
        raise RuntimeError("Dense optimization produced non-finite displacement values.")
    if timer is not None:
        timer.mark(
            "dvf_in_memory_ready",
            metadata={
                "stage_idx": int(stage_idx),
                "representation": "dense_voxel",
                "definition": "immediately after final dense-voxel optimizer stage",
            },
            sync=False,
        )
    actual_epochs = int(rows[-1]["epoch"]) if rows else 0
    summary: Dict[str, object] = {
        "stage_idx": int(stage_idx),
        "representation": "dense_voxel",
        "parameterization": "direct_full_resolution_voxel_displacement",
        "bspline_order": None,
        "full_cps": None,
        "downsample": 1,
        "domain_cps": None,
        "control_shape_zyx": None,
        "eval_shape_zyx": list(c.IMAGE_SHAPE),
        "dense_shape_zyx3": list(final_dense.shape),
        "n_control": None,
        "param_count": int(final_dense.numel()),
        "epochs": actual_epochs,
        "requested_epochs": epochs,
        "actual_epochs": actual_epochs,
        "lr": lr,
        "alpha": alpha,
        "early_stop_mode": early_stop_mode,
        "min_epochs": min_epochs,
        "patience": patience,
        "check_interval": check_interval,
        "rel_delta_total": rel_delta_total,
        "rel_delta_data": rel_delta_data,
        "stop_reason": stop_reason,
        "loss_mode": "full_image_global_ncc_dense_fd_be",
        "image_loss": "full_image_global_ncc",
        "regularizer": "unmasked_forward_valid_dense_fd_be",
        "regularizer_component_reduction": "mean",
        "initialization": "expanded_cps2_absolute_field_detach_clone",
        "elapsed_sec": float(time.time() - stage_start),
        "timing_breakdown": stage_timing,
        **initialization_audit,
    }
    if timer is not None:
        stage_timing["stage_total_sec"] = timer.stop(stage_token)
        timer.add_stage_phase_summary(
            {
                "stage_idx": int(stage_idx),
                "representation": "dense_voxel",
                "requested_epochs": epochs,
                "actual_epochs": actual_epochs,
                "optimizer_loop_sec": float(stage_timing.get("optimizer_loop_sec", 0.0)),
            }
        )
    return final_dense, summary, rows


def train_multistage_direct_cp(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    device: torch.device,
    lr: float,
    sampler: Optional[CheckpointSampler] = None,
    timer: Optional[TimingRecorder] = None,
    run_dir: Optional[Path] = None,
    total_start: Optional[float] = None,
    case_load_complete_sec: Optional[float] = None,
) -> Dict[str, object]:
    is_dense_hybrid = args.method == "direct_cp_8_4_2_then_dense_fd_be_multistage"
    stage_cps = c.parse_int_list(str(args.stage_full_cps))
    stage_downsample = c.parse_stage_ints(str(args.stage_downsample), len(stage_cps))
    stage_epochs = c.parse_stage_ints(str(args.stage_epochs), len(stage_cps))
    stage_lr = c.parse_float_list(str(args.stage_lr), len(stage_cps), float(lr))
    stage_alpha = c.parse_float_list(str(args.stage_alpha), len(stage_cps), float(args.alpha))
    stage_early_stop_mode = str(args.stage_early_stop_mode)
    if str(args.stage_min_epochs).strip():
        stage_min_epochs = c.parse_stage_ints(str(args.stage_min_epochs), len(stage_cps))
    else:
        stage_min_epochs = list(stage_epochs)
    stage_patience = c.parse_stage_ints(str(args.stage_patience), len(stage_cps))
    stage_check_interval = c.parse_stage_ints(str(args.stage_check_interval), len(stage_cps))
    stage_rel_delta_total = c.parse_float_list(str(args.stage_rel_delta_total), len(stage_cps), 5e-5)
    stage_rel_delta_data = c.parse_float_list(str(args.stage_rel_delta_data), len(stage_cps), 2e-5)
    if is_dense_hybrid and stage_cps != [8, 4, 2]:
        raise ValueError(
            "direct_cp_8_4_2_then_dense_fd_be_multistage requires the frozen prefix "
            f"--stage-full-cps 8,4,2; got {stage_cps}."
        )
    base_spacing = int(stage_cps[0])
    nested_shapes = c.build_nested_cp_shapes(c.IMAGE_SHAPE, base_spacing=base_spacing, spacings=stage_cps)
    c.assert_nested_schedule(nested_shapes, stage_cps)

    moving = batch["moving"]
    fixed_image = batch["fixed"]
    fixed_mask = batch["fixed_mask"]
    assert isinstance(moving, torch.Tensor)
    assert isinstance(fixed_image, torch.Tensor)
    assert isinstance(fixed_mask, torch.Tensor)
    dense_grid_token = timer.start(
        "training.initial_dense_grid",
        category="fixed_stage_overhead",
        metadata={"shape": list(c.IMAGE_SHAPE)},
    ) if timer is not None else None
    id_dense = c.make_coordinate_grid(c.IMAGE_SHAPE, device).detach()
    if timer is not None:
        timer.stop(dense_grid_token)
    stage_summaries: List[Dict[str, object]] = []
    history: List[Dict[str, object]] = []
    global_ncc_diagnostic: List[Dict[str, float | str | bool]] = []
    prolong_rows: List[Dict[str, object]] = []
    actual_stage_epochs: List[int] = []
    prev_control: Optional[torch.Tensor] = None
    prev_shape: Optional[Tuple[int, int, int]] = None
    prev_dense_full: Optional[torch.Tensor] = None
    final_dense: Optional[torch.Tensor] = None
    final_control: Optional[torch.Tensor] = None
    final_control_shape: Optional[Tuple[int, int, int]] = None
    final_cps: Optional[int] = None

    train_token = timer.start(
        "training",
        category="training",
        metadata={
            "stage_full_cps": stage_cps,
            "stage_downsample": stage_downsample,
            "stage_requested_epochs": stage_epochs,
            "stage_alpha": stage_alpha,
        },
    ) if timer is not None else None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.time()
    if sampler is not None and sampler.enabled:
        with timer.section("training.initial_checkpoint_sample", category="fixed_stage_overhead") if timer is not None else _null_section():
            sampler.record_dense(
                epoch=0,
                dense_disp_norm_xyz=torch.zeros((*c.IMAGE_SHAPE, 3), dtype=torch.float32, device=device),
                row_extra={"method": args.method, "checkpoint_source": "model_init"},
            )
    global_epoch_offset = 0
    requested_total_epochs = int(sum(stage_epochs)) + (
        int(args.dense_stage_epochs) if is_dense_hybrid else 0
    )
    for stage_idx, (full_cps, downsample, epochs, stage_lr_value, stage_alpha_value) in enumerate(
        zip(stage_cps, stage_downsample, stage_epochs, stage_lr, stage_alpha)
    ):
        control_shape = tuple(int(v) for v in nested_shapes[int(full_cps)])
        components = timer.components if timer is not None else None
        if components is not None:
            components.stage, components.epoch = stage_idx, 0
        with components.scope("control_initialization") if components is not None else _null_section():
            control = torch.nn.Parameter(torch.zeros(int(np.prod(control_shape)), 3, device=device, dtype=torch.float32))
        if prev_control is not None and prev_shape is not None:
            prolong_token = timer.start(
                f"stage_{stage_idx}.prolongation",
                category="fixed_stage_overhead",
                metadata={"stage_idx": int(stage_idx), "target_cps": int(full_cps)},
            ) if timer is not None else None
            init_flat, rows = prolongate_control_grid(
                prev_control,
                prev_shape,
                control_shape,
                int(full_cps),
                c.IMAGE_SHAPE,
                int(stage_idx),
                prev_dense_full=prev_dense_full,
                preserve_check=bool(args.prolongation_preserve_check),
                timer=timer,
            )
            with torch.no_grad():
                control.copy_(init_flat)
            prolong_elapsed = timer.stop(prolong_token) if timer is not None else 0.0
            for row in rows:
                row["timing_prolongation_sec"] = float(prolong_elapsed)
            prolong_rows.extend(rows)
            print(json.dumps({"prolongation": rows}, allow_nan=True), flush=True)
        is_prefix_last_stage = stage_idx == len(stage_cps) - 1
        is_final_stage = is_prefix_last_stage and not is_dense_hybrid
        require_stage_dense = bool(
            is_final_stage
            or (is_dense_hybrid and is_prefix_last_stage)
            or args.prolongation_preserve_check
            or args.save_intermediate_dense
        )
        dense_full, summary, rows = optimize_multistage_stage(
            args,
            fixed_image,
            moving,
            fixed_mask,
            control,
            control_shape,
            int(full_cps),
            int(downsample),
            int(epochs),
            float(stage_lr_value),
            float(stage_alpha_value),
            int(stage_idx),
            int(global_epoch_offset),
            int(requested_total_epochs),
            global_ncc_diagnostic,
            stage_early_stop_mode,
            int(stage_min_epochs[stage_idx]),
            int(stage_patience[stage_idx]),
            int(stage_check_interval[stage_idx]),
            float(stage_rel_delta_total[stage_idx]),
            float(stage_rel_delta_data[stage_idx]),
            sampler=sampler,
            timer=timer,
            is_final_stage=is_final_stage,
            require_stage_dense=require_stage_dense,
        )
        if bool(args.save_intermediate_dense) and not is_final_stage:
            if dense_full is None:
                raise RuntimeError("Intermediate dense save requested but no stage dense field was computed.")
            if run_dir is None:
                raise RuntimeError("Intermediate dense save requested without a run directory.")
            save_token = timer.start(
                f"stage_{stage_idx}.intermediate_dense_save",
                category="intermediate_dense_diagnostic",
                metadata={"stage_idx": int(stage_idx), "full_cps": int(full_cps)},
            ) if timer is not None else None
            fixed_id = str(batch["fixed_id"])
            moving_id = str(batch["moving_id"])
            dense_path = (
                Path(run_dir)
                / "intermediate_dense"
                / f"disp_norm_stage{stage_idx + 1}_cps{int(full_cps)}_{fixed_id}_{moving_id}.npy"
            )
            dense_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(dense_path, dense_full.detach().cpu().numpy().astype(np.float32, copy=False))
            save_elapsed = timer.stop(save_token) if timer is not None else 0.0
            summary["intermediate_dense_path"] = str(dense_path)
            timing_breakdown = summary.get("timing_breakdown")
            if isinstance(timing_breakdown, dict):
                timing_breakdown["intermediate_dense_save_sec"] = float(save_elapsed)
        actual_epochs = int(summary.get("actual_epochs", epochs))
        actual_stage_epochs.append(actual_epochs)
        global_epoch_offset += actual_epochs
        prev_control = control.detach()
        prev_shape = control_shape
        prev_dense_full = dense_full if bool(args.prolongation_preserve_check) else None
        if is_final_stage:
            final_dense = dense_full
        final_control = control.detach()
        final_control_shape = control_shape
        final_cps = int(full_cps)
        stage_summaries.append(summary)
        history.extend(rows)
        if components is not None:
            with components.scope("instrumentation_stage_resolve", cuda=False):
                components.resolve()
    prefix_final_control = final_control
    prefix_final_control_shape = final_control_shape
    prefix_final_cps = final_cps
    dense_initialization_audit: Dict[str, float] = {}
    if is_dense_hybrid:
        if dense_full is None or prefix_final_cps != 2:
            raise RuntimeError("Dense hybrid requires a full-resolution displacement expanded from cps2.")
        dense_stage_idx = len(stage_cps)
        final_dense, dense_summary, dense_rows = optimize_dense_stage(
            args,
            fixed_image,
            moving,
            dense_full,
            dense_stage_idx,
            global_epoch_offset,
            requested_total_epochs,
            sampler=sampler,
            timer=timer,
        )
        dense_initialization_audit = {
            key: float(value)
            for key, value in dense_summary.items()
            if str(key).startswith("dense_init_")
        }
        stage_summaries.append(dense_summary)
        history.extend(dense_rows)
        dense_actual_epochs = int(dense_summary.get("actual_epochs", args.dense_stage_epochs))
        actual_stage_epochs.append(dense_actual_epochs)
        global_epoch_offset += dense_actual_epochs
    torch.cuda.synchronize()
    training_elapsed_sec = time.time() - train_start
    training_peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    if timer is not None:
        timer.stop(train_token)
    if final_dense is None:
        raise RuntimeError("Multistage training produced no final dense field.")
    if not is_dense_hybrid and (
        final_control is None or final_control_shape is None or final_cps is None
    ):
        raise RuntimeError("B-spline multistage training produced no final control field.")
    is_dvf_only = str(getattr(args, "output_mode", "full")) == "dvf_only"
    dvf_endpoint_info: Dict[str, object] = {}
    if not is_dvf_only and run_dir is not None:
        dvf_endpoint_info = save_full_mode_dvf_endpoint(
            args,
            batch,
            final_dense.detach(),
            run_dir,
            timer=timer,
            total_start=total_start,
            case_load_complete_sec=case_load_complete_sec,
        )
    if is_dvf_only:
        metrics: Dict[str, object] = {
            "output_mode": "dvf_only",
            "evaluation_skipped": [
                "image_warp_evaluation",
                "segmentation_warp",
                "dice",
                "hd95_asd",
                "dense_ncc",
                "regularity_metrics",
                "model_control_checkpoint_saves",
                "full_csv_packaging",
            ],
        }
    else:
        eval_token = timer.start(
            "post_train_evaluation",
            category="fixed_post_training",
            metadata={"stage_count": len(stage_summaries)},
        ) if timer is not None else None
        eval_start = time.time()
        metrics = timed_evaluate_from_dense_coords(id_dense + final_dense.detach(), batch, timer=timer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics["post_dvf_evaluation_sec"] = float(time.time() - eval_start)
        if timer is not None:
            timer.stop(eval_token)
        if not is_dense_hybrid:
            assert final_control is not None and final_control_shape is not None
            final_control_grid = c.flat_to_grid(final_control, final_control_shape).detach()
            metrics["control_coords"] = torch.empty(0, 3, dtype=torch.float32)
            metrics["control_disp"] = final_control_grid
    metrics.update(dvf_endpoint_info)
    metrics["history"] = [
        {
            **row,
            "epoch": float(row["global_epoch"]),
            "stage_epoch": float(row["epoch"]),
        }
        for row in history
    ]
    metrics["stage_history"] = history
    metrics["global_ncc_diagnostic"] = global_ncc_diagnostic
    metrics["stage_summaries"] = stage_summaries
    metrics["prolongation_diagnostics"] = prolong_rows
    metrics["control_shape"] = None if is_dense_hybrid else final_control_shape
    metrics["prefix_final_control_shape"] = prefix_final_control_shape
    metrics["dense_disp"] = final_dense.detach()
    metrics["training_elapsed_sec"] = training_elapsed_sec
    metrics["training_peak_mem_gb"] = training_peak_mem_gb
    if is_dense_hybrid:
        metrics["param_count"] = int(final_dense.numel())
        metrics["cps"] = None
    else:
        assert final_control is not None and final_cps is not None
        metrics["param_count"] = int(final_control.numel())
        metrics["cps"] = int(final_cps)
    metrics["prefix_final_cps"] = prefix_final_cps
    metrics["stage_full_cps"] = stage_cps + ([None] if is_dense_hybrid else [])
    metrics["stage_downsample"] = stage_downsample + ([1] if is_dense_hybrid else [])
    metrics["stage_epochs"] = stage_epochs + ([int(args.dense_stage_epochs)] if is_dense_hybrid else [])
    metrics["stage_requested_epochs"] = list(metrics["stage_epochs"])
    metrics["stage_actual_epochs"] = actual_stage_epochs
    metrics["stage_lr"] = stage_lr + ([float(args.dense_stage_lr)] if is_dense_hybrid else [])
    dense_alpha = float(args.alpha if args.dense_stage_alpha is None else args.dense_stage_alpha)
    metrics["stage_alpha"] = stage_alpha + ([dense_alpha] if is_dense_hybrid else [])
    metrics["stage_early_stop_mode"] = stage_early_stop_mode
    metrics["stage_min_epochs"] = stage_min_epochs + (
        [int(args.dense_stage_min_epochs)] if is_dense_hybrid else []
    )
    metrics["stage_patience"] = stage_patience + (
        [int(args.dense_stage_patience)] if is_dense_hybrid else []
    )
    metrics["stage_check_interval"] = stage_check_interval + (
        [int(args.dense_stage_check_interval)] if is_dense_hybrid else []
    )
    metrics["stage_rel_delta_total"] = stage_rel_delta_total + (
        [float(args.dense_stage_rel_delta_total)] if is_dense_hybrid else []
    )
    metrics["stage_rel_delta_data"] = stage_rel_delta_data + (
        [float(args.dense_stage_rel_delta_data)] if is_dense_hybrid else []
    )
    metrics["stage_stop_reasons"] = [str(row.get("stop_reason")) for row in stage_summaries]
    metrics["requested_epochs"] = requested_total_epochs
    metrics["actual_epochs"] = int(sum(actual_stage_epochs))
    metrics["prolongation_preserve_check"] = bool(args.prolongation_preserve_check)
    metrics["save_intermediate_dense"] = bool(args.save_intermediate_dense)
    metrics["nested_control_shapes_zyx"] = {str(k): list(v) for k, v in nested_shapes.items()}
    metrics["prolongation"] = "exact crop-aligned cubic B-spline dyadic knot insertion"
    metrics["stage_representation"] = ["bspline_control"] * len(stage_cps) + (
        ["dense_voxel"] if is_dense_hybrid else []
    )
    metrics["stage_regularizer"] = ["analytic_cubic_bspline_be"] * len(stage_cps) + (
        ["unmasked_forward_valid_dense_fd_be"] if is_dense_hybrid else []
    )
    metrics["final_field_representation"] = (
        "dense_voxel_displacement" if is_dense_hybrid else "cubic_bspline_control_displacement"
    )
    metrics["final_field_is_single_bspline"] = not is_dense_hybrid
    if is_dense_hybrid:
        metrics["dense_initialization"] = "expanded_cps2_absolute_field_detach_clone"
        metrics["dense_initialization_audit"] = dense_initialization_audit
        metrics["dense_stage_uses_cps1"] = False
        metrics["dense_stage_uses_analytic_bspline_be"] = False
        metrics["prefix_param_count"] = (
            int(prefix_final_control.numel()) if prefix_final_control is not None else None
        )
    return metrics


def write_outputs(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    metrics: Dict[str, object],
    run_dir: Path,
    run_name: str,
    lr: float,
    total_elapsed_sec: float,
    timer: Optional[TimingRecorder] = None,
    total_start: Optional[float] = None,
) -> None:
    write_token = timer.start(
        "write_outputs",
        category="output_writing",
        metadata={"run_name": run_name},
    ) if timer is not None else None
    prep_token = timer.start("write_outputs.prepare_dense_disp", category="output_writing") if timer is not None else None
    fixed_id = str(batch["fixed_id"])
    moving_id = str(batch["moving_id"])
    dense_coords = metrics["dense_coords"]
    assert isinstance(dense_coords, torch.Tensor)
    id_grid = c.make_coordinate_grid(c.IMAGE_SHAPE, dense_coords.device)
    dense_disp_norm = (dense_coords - id_grid).detach()
    fixed_mask = batch["fixed_mask"]
    assert isinstance(fixed_mask, torch.Tensor)
    control_disp = metrics.get("control_disp")
    if timer is not None:
        timer.stop(prep_token)

    reg_token = timer.start("write_outputs.regularity_metrics", category="output_writing") if timer is not None else None
    if isinstance(control_disp, torch.Tensor):
        reg_metrics, be_rows, diffusion_rows = c.regularity_metrics(
            dense_disp_norm,
            fixed_mask.detach().bool(),
            control_disp_norm=control_disp,
            cps=int(metrics["cps"]),
        )
    else:
        reg_metrics, be_rows, diffusion_rows = c.regularity_metrics(dense_disp_norm, fixed_mask.detach().bool())
        model = metrics.get("model")
        if isinstance(model, torch.nn.Module):
            idir_be_rows = c.idir_autograd_be_metric_rows(
                model,
                fixed_mask.detach().bool(),
                dense_coords.device,
                int(args.autograd_be_chunk_size),
            )
            idir_diffusion_rows = c.idir_autograd_diffusion_metric_rows(
                model,
                fixed_mask.detach().bool(),
                dense_coords.device,
                int(args.autograd_be_chunk_size),
            )
            be_rows.extend(idir_be_rows)
            diffusion_rows.extend(idir_diffusion_rows)
            reg_metrics.update(c.idir_autograd_be_rows_to_columns(idir_be_rows))
            reg_metrics.update(c.idir_autograd_diffusion_rows_to_columns(idir_diffusion_rows))
            reg_metrics["idir_autograd_be_chunk_size"] = int(args.autograd_be_chunk_size)
            reg_metrics["idir_autograd_diffusion_chunk_size"] = int(args.autograd_be_chunk_size)
            c.add_native_analytic_be_columns(reg_metrics)
            c.add_native_analytic_diffusion_columns(reg_metrics)
    if timer is not None:
        timer.stop(reg_token)

    if args.save_final_dense and not bool(metrics.get("pre_evaluation_dvf_file_saved")):
        with timer.section("write_outputs.save_final_dense", category="output_writing") if timer is not None else _null_section():
            c.save_dense_disp(run_dir / f"disp_norm_{fixed_id}_{moving_id}.npy", dense_coords)

    state_token = timer.start("write_outputs.save_model_control", category="output_writing") if timer is not None else None
    state: Dict[str, object] = {
        "args": vars(args),
        "run_name": run_name,
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "training_elapsed_sec": metrics["training_elapsed_sec"],
        "training_peak_mem_gb": metrics["training_peak_mem_gb"],
    }
    if "model" in metrics:
        state["model_state_dict"] = metrics["model"].state_dict()  # type: ignore[union-attr]
    if "generator" in metrics:
        state["generator_state_dict"] = metrics["generator"].state_dict()  # type: ignore[union-attr]
        state["control_shape"] = metrics["control_shape"]
    if args.method in {
        "direct_dvf_sampled_ncc_fd_be",
        "direct_cp_8_4_2_then_dense_fd_be_multistage",
    } and "dense_disp" in metrics:
        state["dense_disp_norm"] = metrics["dense_disp"].detach().cpu()  # type: ignore[union-attr]
    if args.method == "dctfield_sampled_ncc_fd_be" and "dct_coeff" in metrics:
        state["dct_coeff"] = metrics["dct_coeff"].detach().cpu()  # type: ignore[union-attr]
        state["dct_size_zyx"] = metrics.get("dct_size_zyx")
        state["dct_basis"] = metrics.get("dct_basis")
    if "control_disp" in metrics and "generator" not in metrics:
        state["control_disp"] = metrics["control_disp"].detach().cpu()  # type: ignore[union-attr]
        state["control_shape"] = metrics["control_shape"]
    for key in (
        "stage_summaries",
        "composition_diagnostics",
        "stage_transition_diagnostics",
        "prolongation_diagnostics",
        "stage_full_cps",
        "stage_downsample",
        "stage_epochs",
        "stage_requested_epochs",
        "stage_actual_epochs",
        "stage_lr",
        "stage_alpha",
        "stage_early_stop_mode",
        "stage_min_epochs",
        "stage_patience",
        "stage_check_interval",
        "stage_rel_delta_total",
        "stage_rel_delta_data",
        "stage_min_delta",
        "stage_stop_reasons",
        "stage_representation",
        "stage_regularizer",
        "requested_epochs",
        "actual_epochs",
        "early_stop_epoch",
        "early_stop_reason",
        "best_loss",
        "best_loss_epoch",
        "best_data_loss",
        "best_reg_loss",
        "multistage_strategy",
        "field_composition",
        "final_field_is_single_bspline",
        "final_field_representation",
        "prefix_final_cps",
        "prefix_final_control_shape",
        "prefix_param_count",
        "dense_initialization",
        "dense_initialization_audit",
        "dense_stage_uses_cps1",
        "dense_stage_uses_analytic_bspline_be",
    ):
        if key in metrics:
            state[key] = metrics[key]
    if "stage_generator_state_dicts" in metrics:
        state["stage_generator_state_dicts"] = metrics["stage_generator_state_dicts"]
        state["stage_control_disps"] = metrics.get("stage_control_disps")
        state["stage_control_shapes"] = metrics.get("stage_control_shapes")
    torch.save(state, run_dir / "model_state.pt")
    if "control_disp" in metrics:
        torch.save(
            {
                "control_disp": metrics["control_disp"].detach().cpu(),  # type: ignore[union-attr]
                "control_coords": metrics["control_coords"].detach().cpu(),  # type: ignore[union-attr]
                "control_shape": metrics["control_shape"],
                "cps": metrics["cps"],
                "image_shape": c.IMAGE_SHAPE,
                "be_type": "be_bspline_analytic",
                "image_loss": "sinr_ncc",
                "method_key": method_key(args),
                "alpha": float(args.alpha),
                "stage_alpha": metrics.get("stage_alpha"),
                "grid_order": "pytorch",
                "bspline_mode": "disp",
            },
            run_dir / "control_output.pt",
        )
    if timer is not None:
        timer.stop(state_token)

    tables_token = timer.start("write_outputs.write_tables", category="output_writing") if timer is not None else None
    c.write_csv(run_dir / "history.csv", metrics["history"])  # type: ignore[arg-type]
    if "image_similarity_eval" in metrics and metrics["image_similarity_eval"]:
        c.write_csv(run_dir / "image_similarity_eval.csv", metrics["image_similarity_eval"])  # type: ignore[arg-type]
    if "global_ncc_diagnostic" in metrics and metrics["global_ncc_diagnostic"]:
        c.write_csv(run_dir / "global_ncc_diagnostic.csv", metrics["global_ncc_diagnostic"])  # type: ignore[arg-type]
    if "stage_history" in metrics:
        c.write_csv(run_dir / "stage_history.csv", metrics["stage_history"])  # type: ignore[arg-type]
    if "stage_summaries" in metrics:
        c.write_csv(run_dir / "stage_summaries.csv", metrics["stage_summaries"])  # type: ignore[arg-type]
    if "composition_diagnostics" in metrics:
        c.write_csv(run_dir / "composition_diagnostics.csv", metrics["composition_diagnostics"])  # type: ignore[arg-type]
        c.write_json(run_dir / "composition_diagnostics.json", metrics["composition_diagnostics"])
    if "stage_transition_diagnostics" in metrics:
        c.write_csv(run_dir / "stage_transition_diagnostics.csv", metrics["stage_transition_diagnostics"])  # type: ignore[arg-type]
        c.write_json(run_dir / "stage_transition_diagnostics.json", metrics["stage_transition_diagnostics"])
    if "prolongation_diagnostics" in metrics:
        c.write_csv(run_dir / "prolongation_metrics.csv", metrics["prolongation_diagnostics"])  # type: ignore[arg-type]
    c.write_csv(run_dir / "case_metrics.csv", [c.accuracy_row(metrics, fixed_id, moving_id)])
    c.write_csv(run_dir / "regularity_metrics.csv", [reg_metrics])
    c.write_csv(run_dir / "be_metrics.csv", be_rows)
    c.write_csv(run_dir / "diffusion_metrics.csv", diffusion_rows)
    c.write_json(run_dir / "mask_audit.json", batch["mask_audit"])
    if timer is not None:
        timer.stop(tables_token)

    summary_token = timer.start("write_outputs.write_summary_done", category="output_writing") if timer is not None else None
    last = metrics["history"][-1]  # type: ignore[index]
    is_idir = args.method == "idir_original"
    is_direct_dvf = args.method == "direct_dvf_sampled_ncc_fd_be"
    is_dense_hybrid = args.method == "direct_cp_8_4_2_then_dense_fd_be_multistage"
    is_dctfield = args.method == "dctfield_sampled_ncc_fd_be"
    is_mr_sinr_composed = args.method == "mr_sinr842_composed"
    is_mr_sinr_requery = args.method == "mr_sinr842_requery"
    is_bspline = not is_idir and not is_direct_dvf and not is_dctfield and not is_dense_hybrid
    activation = (
        "dctfield_coefficients"
        if is_dctfield
        else "direct_dense_grid_displacement_initialized_from_cps2"
        if is_dense_hybrid
        else "direct_dense_grid_displacement"
        if is_direct_dvf
        else "direct_control_points"
        if args.method.startswith("direct_cp_")
        else "Siren"
    )
    omega = float(args.idir_omega) if is_idir else (
        None if (is_direct_dvf or is_dctfield or is_dense_hybrid) else float(args.spline_omega)
    )
    be_type = (
        "be_native_analytic"
        if is_idir
        else "dct_decoded_dense_forward_difference_be"
        if is_dctfield
        else "dense_forward_difference_be"
        if (is_direct_dvf or is_dense_hybrid or is_mr_sinr_composed)
        else "be_bspline_analytic"
    )
    diffusion_type = (
        "diffusion_native_analytic"
        if is_idir
        else "dct_decoded_dense_forward_difference"
        if is_dctfield
        else "dense_forward_difference"
        if (is_direct_dvf or is_dense_hybrid or is_mr_sinr_composed)
        else "diffusion_bspline_analytic"
    )
    native_be_formula = (
        "idir_coordinate_autograd_hessian"
        if is_idir
        else "not_used_dct_spectral_be_possible"
        if is_dctfield
        else "not_available_dense_grid_dvf"
        if (is_direct_dvf or is_dense_hybrid)
        else "not_single_bspline_final_field; residual_stage_training_uses_bspline_analytic"
        if is_mr_sinr_composed
        else "bspline_analytic"
    )
    native_diffusion_formula = (
        "idir_coordinate_autograd_jacobian"
        if is_idir
        else "not_used_dct_spectral_diffusion_possible"
        if is_dctfield
        else "not_available_dense_grid_dvf"
        if (is_direct_dvf or is_dense_hybrid)
        else "not_single_bspline_final_field; residual_stage_controls_have_bspline_analytic_derivatives"
        if is_mr_sinr_composed
        else "bspline_analytic"
    )
    be_training_reduction = (
        "original IDIR coordinate-autograd analytic BE; sampled spatial means then divided by batch size"
        if is_idir
        else "dense valid-stencil finite-difference BE over decoded DCT field and fixed image-positive mask; spatial mean plus component mean"
        if is_dctfield
        else "dense valid-stencil finite-difference BE over the unmasked full voxel field; spatial mean plus component mean"
        if is_dense_hybrid
        else "dense valid-stencil finite-difference BE over fixed image-positive mask; spatial mean plus component mean"
        if is_direct_dvf
        else "final composed dense field uses posthoc dense finite-difference BE; each residual SINR stage trains with analytic cubic B-spline BE"
        if is_mr_sinr_composed
        else "analytic cubic B-spline BE sampled at voxel centers; spatial mean plus component mean"
    )
    image_loss = (
        "full_image_global_ncc"
        if is_dense_hybrid
        else "idir_global_ncc_sampled"
        if (is_idir or is_direct_dvf or is_dctfield)
        else "sinr_ncc"
    )
    sample_domain = (
        "fixed_image_positive"
        if is_idir
        else "fixed_image_positive_grid_points"
        if is_dctfield
        else "fixed_image_positive_grid_points"
        if is_direct_dvf
        else "none_full_image_ncc"
    )
    summary: Dict[str, object] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "variant": str(args.variant),
        "method": args.method,
        "method_key": method_key(args),
        "runtime_candidate": str(args.runtime_candidate) or None,
        "case_rank": int(args.case_rank),
        "original_case_index": int(batch["original_case_index"]),
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "pair_file": str(batch["pair_file"]),
        "sorted_pair_file": str(batch["sorted_pair_file"]),
        "epochs": int(args.epochs),
        "requested_epochs": int(metrics.get("requested_epochs", args.epochs)),
        "actual_epochs": int(metrics.get("actual_epochs", args.epochs)),
        "eval_interval": int(args.eval_interval),
        "image_similarity_eval_interval": int(args.image_sim_eval_interval),
        "image_similarity_eval_name": (
            "full_image_global_ncc"
            if is_idir and int(args.image_sim_eval_interval) > 0
            else None
        ),
        "image_similarity_eval_affects_training": False if is_idir and int(args.image_sim_eval_interval) > 0 else None,
        "global_ncc_diagnostic_interval": int(args.global_ncc_diagnostic_interval),
        "global_ncc_diagnostic_name": (
            "full_image_global_ncc"
            if int(args.global_ncc_diagnostic_interval) > 0
            else None
        ),
        "fixed_image_positive_ncc_diagnostic_name": (
            "fixed_image_positive_global_ncc"
            if int(args.global_ncc_diagnostic_interval) > 0
            else None
        ),
        "global_ncc_diagnostic_affects_training": (
            False if int(args.global_ncc_diagnostic_interval) > 0 else None
        ),
        "fixed_image_positive_ncc_diagnostic_affects_training": (
            False if int(args.global_ncc_diagnostic_interval) > 0 else None
        ),
        "global_ncc_diagnostic_tail": metrics.get("global_ncc_diagnostic", [])[-10:],
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "activation": activation,
        "omega": omega,
        "lr": float(lr),
        "alpha": float(args.alpha),
        "gpu_requested": str(args.gpu),
        "source_revision": str(args.source_revision) or None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_backend": c.cuda_backend_metadata(),
        "be_type": be_type,
        "diffusion_type": diffusion_type,
        "regularizer": be_type,
        "native_analytic_be_formula": native_be_formula,
        "native_analytic_diffusion_formula": native_diffusion_formula,
        "be_training_reduction": be_training_reduction,
        "image_loss": image_loss,
        "sample_domain": sample_domain,
        "foreground_metric_mask": "fixed_image_positive",
        "segmentation_mask_role": "dice_hd95_asd_only",
        "cps": metrics.get("cps"),
        "bspline_order": 3 if is_bspline else None,
        "bspline_mode": "disp" if is_bspline else None,
        "grid_order": "pytorch",
        "seed": int(args.seed),
        "coords_batch_size": metrics.get("coords_batch_size"),
        "train_coords": metrics.get("train_coords"),
        "param_count": metrics.get("param_count"),
        "dct_size_zyx": metrics.get("dct_size_zyx"),
        "dct_basis": metrics.get("dct_basis"),
        "stage_full_cps": metrics.get("stage_full_cps"),
        "stage_downsample": metrics.get("stage_downsample"),
        "stage_epochs": metrics.get("stage_epochs"),
        "stage_requested_epochs": metrics.get("stage_requested_epochs"),
        "stage_actual_epochs": metrics.get("stage_actual_epochs"),
        "stage_lr": metrics.get("stage_lr"),
        "stage_alpha": metrics.get("stage_alpha"),
        "stage_early_stop_mode": metrics.get("stage_early_stop_mode"),
        "stage_min_epochs": metrics.get("stage_min_epochs"),
        "stage_patience": metrics.get("stage_patience"),
        "stage_check_interval": metrics.get("stage_check_interval"),
        "stage_rel_delta_total": metrics.get("stage_rel_delta_total"),
        "stage_rel_delta_data": metrics.get("stage_rel_delta_data"),
        "stage_min_delta": metrics.get("stage_min_delta"),
        "stage_stop_reasons": metrics.get("stage_stop_reasons"),
        "stage_representation": metrics.get("stage_representation"),
        "stage_regularizer": metrics.get("stage_regularizer"),
        "early_stop_epoch": metrics.get("early_stop_epoch"),
        "early_stop_reason": metrics.get("early_stop_reason"),
        "best_loss": metrics.get("best_loss"),
        "best_loss_epoch": metrics.get("best_loss_epoch"),
        "best_data_loss": metrics.get("best_data_loss"),
        "best_reg_loss": metrics.get("best_reg_loss"),
        "stage_summaries": metrics.get("stage_summaries"),
        "composition_diagnostics": metrics.get("composition_diagnostics"),
        "stage_transition_diagnostics": metrics.get("stage_transition_diagnostics"),
        "prolongation_diagnostics": metrics.get("prolongation_diagnostics"),
        "prolongation_preserve_check": metrics.get("prolongation_preserve_check"),
        "save_intermediate_dense": metrics.get("save_intermediate_dense"),
        "nested_control_shapes_zyx": metrics.get("nested_control_shapes_zyx"),
        "prolongation": metrics.get("prolongation"),
        "multistage_strategy": metrics.get("multistage_strategy"),
        "field_composition": metrics.get("field_composition"),
        "final_field_is_single_bspline": metrics.get("final_field_is_single_bspline"),
        "final_field_representation": metrics.get("final_field_representation"),
        "prefix_final_cps": metrics.get("prefix_final_cps"),
        "prefix_final_control_shape": metrics.get("prefix_final_control_shape"),
        "prefix_param_count": metrics.get("prefix_param_count"),
        "dense_initialization": metrics.get("dense_initialization"),
        "dense_initialization_audit": metrics.get("dense_initialization_audit"),
        "dense_stage_uses_cps1": metrics.get("dense_stage_uses_cps1"),
        "dense_stage_uses_analytic_bspline_be": metrics.get("dense_stage_uses_analytic_bspline_be"),
        "mr_sinr_variant": (
            "composed_residual" if is_mr_sinr_composed else "continued_requery_absolute" if is_mr_sinr_requery else None
        ),
        "started_at": metrics.get("started_at"),
        "completed_at": c.now_text(),
        "training_elapsed_sec": float(metrics["training_elapsed_sec"]),
        "training_peak_mem_gb": float(metrics["training_peak_mem_gb"]),
        "total_elapsed_sec": float(total_elapsed_sec),
        "full_registration_time_sec": (
            float(time.time() - float(total_start)) if total_start is not None else float(total_elapsed_sec)
        ),
        "full_script_to_dvf_file_sec": metrics.get("full_script_to_dvf_file_sec"),
        "loaded_images_to_dvf_file_sec": metrics.get("loaded_images_to_dvf_file_sec"),
        "post_dvf_evaluation_sec": metrics.get("post_dvf_evaluation_sec"),
        "pre_evaluation_dvf_file_saved": metrics.get("pre_evaluation_dvf_file_saved"),
        "dvf_file_path": metrics.get("dvf_file_path"),
        "final_loss": last["loss"],
        "final_data_loss": last["data_loss"],
        "final_reg_loss": last["reg_loss"],
        "dense_image_loss": metrics["image_loss"],
        "dice_mean": metrics["dice_mean"],
        "hd95_mean": metrics["hd95_mean"],
        "asd_backend": metrics["asd_backend"],
        "asd_warped_to_fixed_mean": metrics["asd_warped_to_fixed_mean"],
        "asd_fixed_to_warped_mean": metrics["asd_fixed_to_warped_mean"],
        "asd_symmetric_mean": metrics["asd_symmetric_mean"],
    }
    summary.update(batch["mask_audit"])  # type: ignore[arg-type]
    summary.update(c.accuracy_row(metrics, fixed_id, moving_id))
    summary.update(reg_metrics)
    c.write_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text("done\n")
    print(json.dumps(summary, indent=2, allow_nan=True, default=str), flush=True)
    if timer is not None:
        timer.stop(summary_token)
        timer.stop(write_token)


def write_dvf_only_file(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    metrics: Dict[str, object],
    run_dir: Path,
    timer: Optional[TimingRecorder] = None,
) -> Dict[str, object]:
    fixed_id = str(batch["fixed_id"])
    moving_id = str(batch["moving_id"])
    dense_disp = metrics["dense_disp"]
    assert isinstance(dense_disp, torch.Tensor)
    path = run_dir / f"disp_norm_{fixed_id}_{moving_id}.npy"

    prep_token = timer.start(
        "method_execution.dvf_file_prep",
        category="dvf_file_output",
        metadata={
            "definition": "detach final dense displacement and move to a CPU float32 numpy array",
        },
    ) if timer is not None else None
    dense_np = dense_disp.detach().cpu().numpy().astype(np.float32, copy=False)
    if timer is not None:
        timer.stop(prep_token)

    write_token = timer.start(
        "method_execution.dvf_file_write",
        category="dvf_file_output",
        metadata={
            "path": str(path),
            "definition": "numpy .npy serialization of normalized-grid xyz dense displacement",
        },
    ) if timer is not None else None
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, dense_np)
    if timer is not None:
        timer.stop(write_token)
        timer.mark(
            "dvf_file_ready",
            metadata={
                "path": str(path),
                "definition": "immediately after dense DVF .npy write completes",
            },
            sync=False,
        )
    return {
        "dvf_file_path": str(path),
        "dense_output_units": "normalized_grid_xyz_displacement",
        "dense_output_shape_zyx3": list(dense_np.shape),
        "dense_output_dtype": str(dense_np.dtype),
        "dense_output_file_format": "numpy_npy",
    }


def write_dvf_only_metadata(
    args: argparse.Namespace,
    batch: Dict[str, torch.Tensor | str | int | Dict[str, object]],
    metrics: Dict[str, object],
    run_dir: Path,
    run_name: str,
    lr: float,
    total_elapsed_sec: float,
    dvf_info: Dict[str, object],
    sampler: CheckpointSampler,
    timer: Optional[TimingRecorder] = None,
) -> None:
    token = timer.start(
        "post_dvf_metadata_write",
        category="post_dvf_metadata_write",
        metadata={
            "definition": "minimal metadata written after DVF endpoint; excluded from DVF-ready headline times",
        },
    ) if timer is not None else None
    fixed_id = str(batch["fixed_id"])
    moving_id = str(batch["moving_id"])
    history = metrics.get("history", [])
    last = history[-1] if isinstance(history, list) and history else {}
    summary: Dict[str, object] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "variant": str(args.variant),
        "method": args.method,
        "method_key": method_key(args),
        "runtime_candidate": str(args.runtime_candidate) or None,
        "output_mode": "dvf_only",
        "no_evaluation": True,
        "evaluation_skipped": metrics.get("evaluation_skipped", []),
        "dvf_endpoint_definition": "DVF in memory is marked immediately after final-stage dense displacement expansion.",
        "dvf_file_endpoint_definition": "DVF file is marked immediately after normalized-grid dense displacement .npy write.",
        "case_rank": int(args.case_rank),
        "original_case_index": int(batch["original_case_index"]),
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "pair_file": str(batch["pair_file"]),
        "sorted_pair_file": str(batch["sorted_pair_file"]),
        "epochs": int(args.epochs),
        "requested_epochs": int(metrics.get("requested_epochs", args.epochs)),
        "actual_epochs": int(metrics.get("actual_epochs", args.epochs)),
        "eval_interval": int(args.eval_interval),
        "lr": float(lr),
        "alpha": float(args.alpha),
        "gpu_requested": str(args.gpu),
        "source_revision": str(args.source_revision) or None,
        "seed": int(args.seed),
        "cps": metrics.get("cps"),
        "bspline_order": 3,
        "bspline_mode": "disp",
        "grid_order": "pytorch",
        "stage_full_cps": metrics.get("stage_full_cps"),
        "stage_downsample": metrics.get("stage_downsample"),
        "stage_epochs": metrics.get("stage_epochs"),
        "stage_requested_epochs": metrics.get("stage_requested_epochs"),
        "stage_actual_epochs": metrics.get("stage_actual_epochs"),
        "stage_lr": metrics.get("stage_lr"),
        "stage_alpha": metrics.get("stage_alpha"),
        "stage_early_stop_mode": metrics.get("stage_early_stop_mode"),
        "stage_min_epochs": metrics.get("stage_min_epochs"),
        "stage_patience": metrics.get("stage_patience"),
        "stage_check_interval": metrics.get("stage_check_interval"),
        "stage_rel_delta_total": metrics.get("stage_rel_delta_total"),
        "stage_rel_delta_data": metrics.get("stage_rel_delta_data"),
        "stage_min_delta": metrics.get("stage_min_delta"),
        "stage_stop_reasons": metrics.get("stage_stop_reasons"),
        "early_stop_epoch": metrics.get("early_stop_epoch"),
        "early_stop_reason": metrics.get("early_stop_reason"),
        "best_loss": metrics.get("best_loss"),
        "best_loss_epoch": metrics.get("best_loss_epoch"),
        "best_data_loss": metrics.get("best_data_loss"),
        "best_reg_loss": metrics.get("best_reg_loss"),
        "stage_summaries": metrics.get("stage_summaries"),
        "prolongation_diagnostics": metrics.get("prolongation_diagnostics"),
        "prolongation_preserve_check": metrics.get("prolongation_preserve_check"),
        "save_intermediate_dense": metrics.get("save_intermediate_dense"),
        "nested_control_shapes_zyx": metrics.get("nested_control_shapes_zyx"),
        "prolongation": metrics.get("prolongation"),
        "control_shape": metrics.get("control_shape"),
        "param_count": metrics.get("param_count"),
        "started_at": metrics.get("started_at"),
        "completed_at": c.now_text(),
        "training_elapsed_sec": float(metrics["training_elapsed_sec"]),
        "training_peak_mem_gb": float(metrics["training_peak_mem_gb"]),
        "total_elapsed_sec": float(total_elapsed_sec),
        "final_loss": last.get("loss") if isinstance(last, dict) else None,
        "final_data_loss": last.get("data_loss") if isinstance(last, dict) else None,
        "final_reg_loss": last.get("reg_loss") if isinstance(last, dict) else None,
        "image_loss": "sinr_ncc_training_only",
        "regularizer": "be_bspline_analytic",
        "foreground_metric_mask": "fixed_image_positive",
        "segmentation_mask_role": "loaded_but_not_evaluated_in_dvf_only_mode",
        "timing_detail": str(args.timing_detail),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_backend": c.cuda_backend_metadata(),
        **dvf_info,
    }
    summary.update(batch["mask_audit"])  # type: ignore[arg-type]
    if sampler.enabled:
        with timer.section("post_dvf_metadata_write.sampler_save", category="post_dvf_metadata_write") if timer is not None else _null_section():
            sampler.save()
    c.write_json(run_dir / "summary.json", summary)
    (run_dir / "DONE").write_text("done\n")
    print(json.dumps(summary, indent=2, allow_nan=True, default=str), flush=True)
    if timer is not None:
        timer.stop(token)


def main() -> None:
    args = parse_args()
    if str(args.timing_detail) == "components":
        from scripts.experiments.mrdbscp842_acceleration.components import instrument
        with instrument(c):
            _main(args)
    else:
        _main(args)


def _main(args: argparse.Namespace) -> None:
    timer = TimingRecorder(str(args.timing_detail))
    lr = float(args.lr if args.lr is not None else default_lr(args.method))
    if str(args.output_mode) == "dvf_only":
        supported_dvf_only = {
            "direct_cp_8_4_2_be_bspline_analytic_multistage",
            "direct_cp_8_4_2_1_be_bspline_analytic_multistage",
            "direct_cp_cps4_be_bspline_analytic",
            "direct_cp_cps2_be_bspline_analytic",
            "sinr_cps4_be_bspline_analytic",
            "sinr_cps2_be_bspline_analytic",
        }
        if args.method not in supported_dvf_only:
            raise ValueError(
                "--output-mode dvf_only is only implemented for DirectCP 8-4-2 multistage, "
                "DirectCP cps4/cps2, and SINR cps4/cps2."
            )
        if not bool(args.save_final_dense):
            raise ValueError("--output-mode dvf_only requires --save-final-dense.")
    if args.validate_case_only:
        case = c.resolve_case(rank=args.case_rank, pair_file=args.pair_file, oasis_root=args.oasis_root)
        print(json.dumps(case.__dict__, indent=2), flush=True)
        validate_expected_case(
            args,
            original_index=case.original_case_index,
            fixed_id=case.fixed_id,
            moving_id=case.moving_id,
        )
        return

    run_name = build_run_name(args, lr)
    key = method_key(args)
    run_dir = Path(args.out_root) / key / run_name
    done_path = run_dir / "DONE"
    if done_path.exists() and not args.overwrite:
        print(json.dumps({"skip_existing": str(run_dir)}), flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    started_at = c.now_text()
    setup_token = timer.start("setup", category="setup")
    with timer.section("setup.device", category="setup"):
        device = c.set_cuda_device(
            args.gpu,
            cudnn_benchmark=bool(args.cudnn_benchmark),
            cudnn_deterministic=bool(args.cudnn_deterministic),
        )
        timer.set_device(device)
    with timer.section("setup.seed", category="setup"):
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
    with timer.section("setup.case_load", category="setup"):
        batch = c.load_oasis_case(
            rank=int(args.case_rank),
            device=device,
            oasis_root=Path(args.oasis_root),
            pair_file=Path(args.pair_file),
        )
    case_load_complete_sec = float(time.time() - total_start)
    timer.mark(
        "case_load_complete",
        metadata={
            "definition": "loaded-images start point; fixed/moving images and labels are resident on the selected device",
        },
    )
    with timer.section("setup.case_validate_print", category="setup"):
        validate_expected_case(
            args,
            original_index=int(batch["original_case_index"]),
            fixed_id=str(batch["fixed_id"]),
            moving_id=str(batch["moving_id"]),
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "method": args.method,
                    "method_key": key,
                    "gpu": args.gpu,
                    "case_rank": batch["case_rank"],
                    "original_case_index": batch["original_case_index"],
                    "fixed_id": batch["fixed_id"],
                    "moving_id": batch["moving_id"],
                    "mask_audit": batch["mask_audit"],
                    "alpha": args.alpha,
                    "stage_alpha": str(args.stage_alpha) or None,
                    "dense_stage_alpha": (
                        float(args.alpha)
                        if args.dense_stage_alpha is None
                        else float(args.dense_stage_alpha)
                    ) if args.method == "direct_cp_8_4_2_then_dense_fd_be_multistage" else None,
                    "dense_stage_lr": (
                        float(args.dense_stage_lr)
                        if args.method == "direct_cp_8_4_2_then_dense_fd_be_multistage"
                        else None
                    ),
                    "lr": lr,
                    "epochs": args.epochs,
                    "stage_early_stop_mode": args.stage_early_stop_mode,
                    "runtime_candidate": str(args.runtime_candidate) or None,
                    "timing_detail": str(args.timing_detail),
                    "output_mode": str(args.output_mode),
                    "prolongation_preserve_check": bool(args.prolongation_preserve_check),
                    "save_intermediate_dense": bool(args.save_intermediate_dense),
                },
                allow_nan=True,
            ),
            flush=True,
        )
    with timer.section("setup.sampler_init", category="setup"):
        sampler = CheckpointSampler(args=args, batch=batch, device=device, run_dir=run_dir)
    timer.stop(setup_token)

    method_token = timer.start(
        "method_execution",
        category="method_execution",
        metadata={"method": args.method, "runtime_candidate": str(args.runtime_candidate) or None},
    )
    active_timer = timer if timer.enabled else None
    if args.method == "idir_original":
        metrics = train_idir(args, batch, device, lr, sampler=sampler)
    elif args.method == "direct_dvf_sampled_ncc_fd_be":
        metrics = train_direct_dvf(args, batch, device, lr, sampler=sampler)
    elif args.method == "dctfield_sampled_ncc_fd_be":
        metrics = train_dctfield(args, batch, device, lr, sampler=sampler)
    elif args.method in {
        "direct_cp_8_4_2_be_bspline_analytic_multistage",
        "direct_cp_8_4_2_1_be_bspline_analytic_multistage",
        "direct_cp_8_4_2_then_dense_fd_be_multistage",
    }:
        metrics = train_multistage_direct_cp(
            args,
            batch,
            device,
            lr,
            sampler=sampler,
            timer=active_timer,
            run_dir=run_dir,
            total_start=total_start,
            case_load_complete_sec=case_load_complete_sec,
        )
    elif args.method == "mr_sinr842_composed":
        metrics = train_mr_sinr842_composed(args, batch, device, lr, sampler=sampler)
    elif args.method == "mr_sinr842_requery":
        metrics = train_mr_sinr842_requery(args, batch, device, lr, sampler=sampler)
    else:
        metrics = train_spline(
            args,
            batch,
            device,
            lr,
            sampler=sampler,
            timer=active_timer,
            run_dir=run_dir,
            total_start=total_start,
            case_load_complete_sec=case_load_complete_sec,
        )
    metrics["started_at"] = started_at
    metrics["timing_detail"] = str(args.timing_detail)
    if str(args.output_mode) == "dvf_only":
        dvf_info = write_dvf_only_file(args, batch, metrics, run_dir, timer=active_timer)
        timer.stop(method_token)
        total_elapsed_sec = time.time() - total_start
        write_dvf_only_metadata(
            args,
            batch,
            metrics,
            run_dir,
            run_name,
            lr,
            total_elapsed_sec,
            dvf_info,
            sampler,
            timer=active_timer,
        )
    else:
        timer.stop(method_token)
        with timer.section("sampler_save", category="output_writing"):
            sampler.save()
        total_elapsed_sec = time.time() - total_start
        write_outputs(
            args,
            batch,
            metrics,
            run_dir,
            run_name,
            lr,
            total_elapsed_sec,
            timer=active_timer,
            total_start=total_start,
        )
    if timer.enabled:
        timer.finalize()
        c.write_json(run_dir / "timing.json", timer.payload(args=args, run_dir=run_dir, run_name=run_name))


if __name__ == "__main__":
    main()
