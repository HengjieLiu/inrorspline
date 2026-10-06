#!/usr/bin/env python3
"""Run the GPU-1 LungCT adaptive versus fixed-50 timing study.

The host controller is standard-library only.  Dependency-sensitive loading,
registration, smoke tests, and aggregation run inside the configured Docker
container.  Every accepted arm is produced by one fresh persistent Python
worker so that case 01 can be measured cold and then repeated warm before the
remaining nine warm cases.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import artifact_paths  # noqa: E402
from mrdbscp842_acceleration.nvml_monitor import NVML  # noqa: E402


EXPERIMENT_ID = "2026-09-29_lungct_mrdbscp_adaptive_fixed50_gpu1"
RUN_PREFIX = "lungct_mrdbscp_adaptive_fixed50_gpu1_"
METHOD = "direct_cp_32_16_8_4_ptv_lcc_isotv"
DATASETS = ("4dct", "copd")
VARIANTS = ("adaptive", "fixed50")
CASES = tuple(range(1, 11))
PHYSICAL_GPU = 1
EXPECTED_PRIMARY_ROWS = 44
EXPECTED_WARM_ROWS = 40
PRIMARY_ROWS_PER_ARM = 11
WORKER_SCRIPT_NAME = "launch_mrdbscp_adaptive_fixed50_gpu1.py"
AGGREGATOR_SCRIPT_NAME = "aggregate_mrdbscp_adaptive_fixed50_gpu1.py"
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("docs/standards/lungct_protocol_tags.md"),
    Path("docs/standards/lungct_tre_metric_conventions.md"),
    Path("scripts/experiments/artifact_paths.py"),
    Path("scripts/experiments/lungct_no_sweep_repro/common.py"),
    Path("scripts/experiments/lungct_no_sweep_repro/run_one.py"),
    Path("scripts/experiments/lungct_no_sweep_repro/launch_mrdbscp_adaptive_fixed50_gpu1.py"),
    Path("scripts/experiments/lungct_no_sweep_repro/aggregate_mrdbscp_adaptive_fixed50_gpu1.py"),
    Path("tests/test_lungct_mrdbscp_adaptive_fixed50_gpu1.py"),
    Path("experiments/2026-09-29_lungct_mrdbscp_adaptive_fixed50_gpu1/codex_execution_spec.md"),
    Path("experiments/2026-09-29_lungct_mrdbscp_adaptive_fixed50_gpu1/technical_log.md"),
)


class StudyError(RuntimeError):
    """Raised when the frozen study contract cannot be satisfied."""


@dataclass(frozen=True)
class VariantSpec:
    dataset: str
    variant: str
    stage_full_cps: tuple[int, ...]
    stage_downsample: tuple[int, ...]
    stage_epochs: tuple[int, ...]
    stage_lr: tuple[float, ...]
    early_stop_mode: str
    min_epochs: tuple[int, ...]
    patience: tuple[int, ...]
    total_thresholds: tuple[float, ...]
    data_thresholds: tuple[float, ...]
    schedule_source: str


def variant_spec(dataset: str, variant: str) -> VariantSpec:
    dataset = str(dataset).lower()
    variant = str(variant).lower()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant!r}")

    if dataset == "4dct":
        geometry = {
            "stage_full_cps": (32, 16, 8, 4),
            "stage_downsample": (8, 4, 2, 1),
            "stage_lr": (0.01, 0.003, 0.003, 0.003),
        }
        if variant == "adaptive":
            return VariantSpec(
                dataset=dataset,
                variant=variant,
                **geometry,
                stage_epochs=(1400, 400, 400, 400),
                early_stop_mode="adaptive",
                min_epochs=(210, 60, 60, 60),
                patience=(50, 25, 25, 25),
                total_thresholds=(5e-5,) * 4,
                data_thresholds=(2e-5,) * 4,
                schedule_source="P33 best-quality 4DCT balanced-cumulative projection",
            )
        return VariantSpec(
            dataset=dataset,
            variant=variant,
            **geometry,
            stage_epochs=(50, 50, 50, 50),
            early_stop_mode="disabled",
            min_epochs=(0, 0, 0, 0),
            patience=(0, 0, 0, 0),
            total_thresholds=(0.0,) * 4,
            data_thresholds=(0.0,) * 4,
            schedule_source="fixed-50-per-stage analogue with matched adaptive LR",
        )

    geometry = {
        "stage_full_cps": (64, 32, 16, 8, 4),
        "stage_downsample": (16, 8, 4, 2, 1),
        "stage_lr": (0.03, 0.03, 0.01, 0.01, 0.003),
    }
    if variant == "adaptive":
        return VariantSpec(
            dataset=dataset,
            variant=variant,
            **geometry,
            stage_epochs=(800, 600, 400, 400, 400),
            early_stop_mode="adaptive",
            min_epochs=(120, 90, 60, 60, 60),
            patience=(50, 38, 25, 25, 25),
            total_thresholds=(5e-5,) * 5,
            data_thresholds=(2e-5,) * 5,
            schedule_source="P33 best-quality COPD balanced-ordinal schedule",
        )
    return VariantSpec(
        dataset=dataset,
        variant=variant,
        **geometry,
        stage_epochs=(50, 50, 50, 50, 50),
        early_stop_mode="disabled",
        min_epochs=(0, 0, 0, 0, 0),
        patience=(0, 0, 0, 0, 0),
        total_thresholds=(0.0,) * 5,
        data_thresholds=(0.0,) * 5,
        schedule_source="fixed-50-per-stage analogue with matched adaptive LR",
    )


def case_sequence(smoke: bool = False) -> list[dict[str, Any]]:
    if smoke:
        return [
            {
                "case_id": 1,
                "sequence_index": 0,
                "timing_condition": "smoke",
                "case_occurrence": "smoke",
            }
        ]
    rows = [
        {
            "case_id": 1,
            "sequence_index": 0,
            "timing_condition": "cold",
            "case_occurrence": "cold",
        },
        {
            "case_id": 1,
            "sequence_index": 1,
            "timing_condition": "warm",
            "case_occurrence": "warm_repeat",
        },
    ]
    rows.extend(
        {
            "case_id": case_id,
            "sequence_index": case_id,
            "timing_condition": "warm",
            "case_occurrence": "warm_cohort",
        }
        for case_id in range(2, 11)
    )
    return rows


def run_name(dataset: str, variant: str, row: dict[str, Any]) -> str:
    case_id = int(row["case_id"])
    occurrence = str(row["case_occurrence"])
    return f"{EXPERIMENT_ID}_{dataset}_{variant}_case{case_id:02d}_{occurrence}"


def expected_jobs(smoke: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            spec = variant_spec(dataset, variant)
            for item in case_sequence(smoke=smoke):
                row = {
                    "experiment_id": EXPERIMENT_ID,
                    "dataset": dataset,
                    "variant": variant,
                    "physical_gpu": PHYSICAL_GPU,
                    "run_name": run_name(dataset, variant, item),
                    **item,
                    "stage_full_cps": list(spec.stage_full_cps),
                    "stage_downsample": list(spec.stage_downsample),
                    "stage_epochs": [1] * len(spec.stage_epochs) if smoke else list(spec.stage_epochs),
                    "stage_lr": list(spec.stage_lr),
                    "early_stop_mode": "disabled" if smoke else spec.early_stop_mode,
                }
                rows.append(row)
    return rows


def csv_list(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in values)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def dump_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_identity(pid: int) -> dict[str, Any]:
    proc = Path(f"/proc/{int(pid)}")
    stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    argv = (proc / "cmdline").read_bytes().split(b"\0")
    return {
        "pid": int(pid),
        "parent": int(stat[1]),
        "start_ticks": int(stat[19]),
        "argv": [item.decode(errors="replace") for item in argv if item],
    }


def find_worker_identity(token: str, container_pid: int) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            identity = process_identity(int(proc.name))
            if token not in identity["argv"] or not Path(identity["argv"][0]).name.startswith("python"):
                continue
            status = Path(f"/proc/{identity['pid']}/status").read_text()
            namespace_pids = next(
                (line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:")),
                [],
            )
            if namespace_pids and int(namespace_pids[-1]) == int(container_pid):
                matches.append(identity)
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
    if len(matches) != 1:
        raise StudyError(f"Cannot uniquely map worker token {token!r}: {matches}")
    return matches[0]


class GPUMonitor:
    def __init__(self, gpu: int, path: Path) -> None:
        uuid_text = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            timeout=30,
        ).strip()
        self.gpu = int(gpu)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.nvml = NVML(self.gpu, uuid_text)
        self.owner: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    def sample(self, reason: str = "periodic") -> dict[str, Any]:
        now = time.time()
        try:
            apps = self.nvml.processes()
            allowed: set[int] = set()
            if self.owner is not None:
                try:
                    current = process_identity(int(self.owner["pid"]))
                except FileNotFoundError:
                    current = None
                if current is not None:
                    if int(current["start_ticks"]) != int(self.owner["start_ticks"]):
                        raise StudyError("Study worker PID was reused")
                    allowed.add(int(self.owner["pid"]))
                for app in apps:
                    pid = int(app["pid"])
                    original_pid = pid
                    for _ in range(64):
                        if pid in allowed:
                            allowed.add(original_pid)
                            break
                        if pid <= 1:
                            break
                        pid = int(process_identity(pid)["parent"])
            record = {
                "unix": now,
                "gpu": self.gpu,
                "reason": reason,
                "apps": apps,
                "allowed_pids": sorted(allowed),
                "foreign": [app for app in apps if int(app["pid"]) not in allowed],
                "error": None,
            }
        except Exception as exc:  # fail closed and preserve the query error
            record = {
                "unix": now,
                "gpu": self.gpu,
                "reason": reason,
                "apps": [],
                "allowed_pids": [],
                "foreign": [],
                "error": repr(exc),
            }
        self.records.append(record)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def wait_clear(self, poll_seconds: float, wait_timeout_seconds: float) -> None:
        clean = 0
        started = time.monotonic()
        while clean < 3:
            record = self.sample(reason="idle_gate")
            clean = clean + 1 if not record["error"] and not record["foreign"] else 0
            if clean == 0:
                print(json.dumps({"status": "waiting_for_gpu1", "sample": record}), flush=True)
            if wait_timeout_seconds > 0 and time.monotonic() - started > wait_timeout_seconds:
                raise TimeoutError("GPU 1 did not remain clear for three consecutive checks")
            if clean < 3:
                time.sleep(poll_seconds)

    def interval(self, start: float, end: float) -> dict[str, Any]:
        rows = [row for row in self.records if start <= float(row["unix"]) <= end]
        points = sorted([start, end] + [float(row["unix"]) for row in rows])
        max_gap = max((b - a for a, b in zip(points, points[1:])), default=end - start)
        bad = [row for row in rows if row["error"] or row["foreign"]]
        return {
            "clean": bool(rows) and not bad and max_gap <= 5.0,
            "start_unix": start,
            "end_unix": end,
            "sample_count": len(rows),
            "max_sample_gap_sec": max_gap,
            "bad_samples": bad,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dry-run", "preflight", "all", "report"), default="dry-run")
    parser.add_argument("--container", default="py310sitreg_v1_artifact_store")
    parser.add_argument("--repo-container", default="/homebase/code_sync/inrorspline")
    parser.add_argument("--artifact-host", type=Path, default=None)
    parser.add_argument("--artifact-container", type=Path, default=Path("/database/inrorspline"))
    parser.add_argument("--data-root", type=Path, default=Path("/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--review-root", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--max-arm-attempts", type=int, default=10)
    parser.add_argument("--worker-timeout-seconds", type=float, default=14400.0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--preflight-worker", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-dataset", choices=DATASETS, default="")
    parser.add_argument("--worker-variant", choices=VARIANTS, default="")
    parser.add_argument("--attempt-root", type=Path, default=None)
    parser.add_argument("--job-token", default="")
    args = parser.parse_args(argv)
    if int(args.gpu) != PHYSICAL_GPU:
        parser.error(f"This locked experiment requires physical GPU {PHYSICAL_GPU}")
    if not str(args.run_id).startswith(RUN_PREFIX) or "/" in str(args.run_id):
        parser.error(f"--run-id must begin with {RUN_PREFIX!r} and contain no slash")
    if args.worker and (not args.worker_dataset or not args.worker_variant or args.attempt_root is None or not args.job_token):
        parser.error("internal --worker mode requires dataset, variant, attempt root, and job token")
    return args


def resolve_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.worker or args.preflight_worker:
        return args
    configured_text = str(args.artifact_host) if args.artifact_host is not None else os.environ.get("ARTIFACT_ROOT", "")
    if not configured_text:
        raise StudyError("--artifact-host or ARTIFACT_ROOT is required")
    args.artifact_host = Path(configured_text).resolve()
    environment = {"ARTIFACT_ROOT": str(args.artifact_host)}
    args.out_root = artifact_paths.resolve_run_root(
        args.run_id,
        args.out_root,
        environ=environment,
        repo_root=REPO_ROOT,
    )
    args.review_root = artifact_paths.resolve_review_root(
        f"{EXPERIMENT_ID}/{args.run_id}",
        args.review_root,
        environ=environment,
        repo_root=REPO_ROOT,
    )
    return args


def container_path(args: argparse.Namespace, host_path: Path) -> Path:
    if args.artifact_host is None:
        raise StudyError("artifact host is unresolved")
    try:
        relative = Path(host_path).resolve().relative_to(Path(args.artifact_host).resolve())
    except ValueError as exc:
        raise StudyError(f"Path is outside artifact root: {host_path}") from exc
    return Path(args.artifact_container) / relative


def run_one_argv(
    args: argparse.Namespace,
    dataset: str,
    variant: str,
    row: dict[str, Any],
    *,
    smoke: bool,
) -> list[str]:
    spec = variant_spec(dataset, variant)
    stage_epochs = (1,) * len(spec.stage_epochs) if smoke else spec.stage_epochs
    condition = "unspecified" if smoke else str(row["timing_condition"])
    argv = [
        "run_one.py",
        "--dataset", dataset,
        "--method", METHOD,
        "--case-id", str(int(row["case_id"])),
        "--data-root", str(args.data_root),
        "--out-root", str(Path(args.attempt_root) / "outputs"),
        "--run-name", run_name(dataset, variant, row),
        "--gpu", str(PHYSICAL_GPU),
        "--cudnn-benchmark",
        "--no-cudnn-deterministic",
        "--seed", "1",
        "--eval-interval", "1" if smoke else "250",
        "--lr", "1e-3",
        "--stage-full-cps", csv_list(spec.stage_full_cps),
        "--stage-downsample", csv_list(spec.stage_downsample),
        "--stage-epochs", csv_list(stage_epochs),
        "--stage-lr", csv_list(spec.stage_lr),
        "--ptv-lcc-sigma-mm", "2.1",
        "--ptv-isotv-weight", "0.055",
        "--ptv-csqrt", "0.005",
        "--border-mask-voxels", "5",
        "--timing-mode", "input_to_dvf",
        "--timing-condition", condition,
        "--timing-sequence-index", str(int(row["sequence_index"])),
        "--timing-worker-id", f"{dataset}_{variant}_gpu1",
        "--variant-id", variant,
        "--no-prolongation-diagnostics",
        "--save-dense-dvf",
    ]
    if not smoke and spec.early_stop_mode == "adaptive":
        argv.extend(
            [
                "--stage-early-stop-mode", "adaptive",
                "--stage-min-epochs", csv_list(spec.min_epochs),
                "--stage-patience", csv_list(spec.patience),
                "--stage-check-interval", "25",
                "--stage-window", "25",
                "--stage-total-thresholds", csv_list(spec.total_thresholds),
                "--stage-data-thresholds", csv_list(spec.data_thresholds),
            ]
        )
    return argv


def worker_main(args: argparse.Namespace) -> int:
    if str(os.environ.get("CUDA_VISIBLE_DEVICES", "")) != str(PHYSICAL_GPU):
        raise StudyError("Worker must start with CUDA_VISIBLE_DEVICES=1")
    attempt_root = Path(args.attempt_root)
    attempt_root.mkdir(parents=True, exist_ok=False)
    ready = {
        "container_pid": os.getpid(),
        "token": str(args.job_token),
        "dataset": str(args.worker_dataset),
        "variant": str(args.worker_variant),
        "smoke": bool(args.smoke),
    }
    dump_json(attempt_root / "ready.json", ready)
    ack_path = attempt_root / "host_ack.json"
    started_wait = time.monotonic()
    while not ack_path.exists():
        if time.monotonic() - started_wait > 120:
            raise TimeoutError("Host did not acknowledge worker PID mapping")
        time.sleep(0.1)

    # Imports happen only after the host has mapped the worker and made it the
    # allowed NVML owner.  This prevents an unowned CUDA context at startup.
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common as c  # type: ignore  # noqa: E402
    import run_one  # type: ignore  # noqa: E402

    rows = case_sequence(smoke=bool(args.smoke))
    progress: list[dict[str, Any]] = []
    old_argv = list(sys.argv)
    try:
        for row in rows:
            command = run_one_argv(
                args,
                str(args.worker_dataset),
                str(args.worker_variant),
                row,
                smoke=bool(args.smoke),
            )
            case_started = time.time()
            sys.argv = command
            run_one.main()
            output = Path(args.attempt_root) / "outputs" / METHOD / run_name(
                str(args.worker_dataset), str(args.worker_variant), row
            )
            summary_path = output / "summary.json"
            summary = c.read_json(summary_path)
            if not isinstance(summary, dict) or summary.get("status") != "completed":
                raise StudyError(f"Incomplete summary: {summary_path}")
            result = {
                **row,
                "dataset": str(args.worker_dataset),
                "variant": str(args.worker_variant),
                "run_name": output.name,
                "summary_path": str(summary_path),
                "dense_dvf_path": str(output / "disp_norm_xyz_reg1mm.npy"),
                "timing_path": str(output / "timing.json"),
                "worker_case_wall_sec": time.time() - case_started,
                "status": "completed",
            }
            progress.append(result)
            dump_json(attempt_root / "progress.json", progress)
            print(json.dumps({"case_completed": result}, sort_keys=True), flush=True)
    finally:
        sys.argv = old_argv
    dump_json(
        attempt_root / "worker_result.json",
        {
            "status": "completed",
            "dataset": str(args.worker_dataset),
            "variant": str(args.worker_variant),
            "smoke": bool(args.smoke),
            "row_count": len(progress),
            "rows": progress,
        },
    )
    return 0


def preflight_worker_main(args: argparse.Namespace) -> int:
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common as c  # type: ignore  # noqa: E402

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for case_id in CASES:
            case = c.load_case(args.data_root, case_id, dataset=dataset)
            rows.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "shape_zyx": list(case.shape_zyx),
                    "fixed_phase": case.fixed_label,
                    "moving_phase": case.moving_label,
                    "landmark_count": int(case.landmarks_fixed.shape[0]),
                    "fixed_mask_voxels": int(case.fixed_mask.sum()),
                    "preprocess_audit": case.preprocess_audit,
                }
            )
    errors = [
        f"{row['dataset']}/case{int(row['case_id']):02d}: landmark_count={row['landmark_count']}"
        for row in rows
        if int(row["landmark_count"]) != 300
    ]
    payload = {
        "status": "passed" if not errors else "failed",
        "dataset_root": str(args.data_root),
        "case_count": len(rows),
        "errors": errors,
        "rows": rows,
    }
    if args.attempt_root is None:
        raise StudyError("preflight worker requires --attempt-root")
    dump_json(Path(args.attempt_root) / "loader_audit.json", payload)
    if errors:
        raise StudyError(str(errors))
    return 0


def immutable_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": str(args.run_id),
        "physical_gpu": PHYSICAL_GPU,
        "method": METHOD,
        "datasets": list(DATASETS),
        "variants": {f"{dataset}_{variant}": asdict(variant_spec(dataset, variant)) for dataset in DATASETS for variant in VARIANTS},
        "primary_jobs": expected_jobs(smoke=False),
        "expected_primary_rows": EXPECTED_PRIMARY_ROWS,
        "expected_warm_rows": EXPECTED_WARM_ROWS,
        "objective": {
            "optimizer": "Adam",
            "seed": 1,
            "ptv_lcc_sigma_mm": 2.1,
            "ptv_isotv_weight": 0.055,
            "ptv_csqrt": 0.005,
            "border_mask_voxels": 5,
        },
        "timing_endpoint": "case_load_complete_to_dvf_file_ready",
        "cold_warm_sequence": "case01_cold,case01_warm_repeat,cases02-10_warm",
        "historical_anchor_policy": "P33 quality context only; no matched speedup claim",
    }


def manifest_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_root = Path(args.out_root)
        self.review_root = Path(args.review_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.review_root.mkdir(parents=True, exist_ok=True)
        self.monitor = GPUMonitor(PHYSICAL_GPU, self.run_root / "monitor" / "gpu1_samples.jsonl")
        self.accepted_path = self.run_root / "accepted_arms.json"
        self.accepted: dict[str, Any] = read_json(self.accepted_path, {}) or {}

    def ensure_manifest(self) -> None:
        payload = immutable_manifest(self.args)
        digest = manifest_digest(payload)
        normalized_payload = json.loads(json.dumps(payload, sort_keys=True))
        existing = read_json(self.run_root / "study_manifest.json")
        if existing is not None:
            if existing.get("manifest_sha256") != digest or existing.get("manifest") != normalized_payload:
                raise StudyError("Existing run ID has a different immutable study manifest")
            return
        dump_json(
            self.run_root / "study_manifest.json",
            {"manifest_sha256": digest, "manifest": payload},
        )
        dump_json(self.review_root / "job_manifest.json", payload)

    def provenance(self) -> None:
        destination = self.run_root / "provenance" / utc_stamp()
        destination.mkdir(parents=True, exist_ok=False)
        hashes: dict[str, str] = {}
        for relative in SOURCE_PATHS:
            source = REPO_ROOT / relative
            if not source.is_file():
                raise StudyError(f"Required source is missing: {relative}")
            target = destination / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[str(relative)] = sha256_file(source)
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, text=True)
        dirty_patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT)
        (destination / "dirty.patch").write_bytes(dirty_patch)
        docker = json.loads(subprocess.check_output(["docker", "inspect", self.args.container], text=True))
        gpu = subprocess.check_output(["nvidia-smi", "-i", str(PHYSICAL_GPU), "-q"], text=True)
        payload = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(self.args).items()},
            "git_head": git_head,
            "git_status": git_status,
            "source_sha256": hashes,
            "docker_inspect": docker,
            "gpu_query": gpu,
            "host_uid": os.getuid(),
            "host_gid": os.getgid(),
        }
        dump_json(destination / "manifest.json", payload)
        dump_json(self.run_root / "latest_provenance.json", {"path": str(destination)})

    def docker_command(
        self,
        dataset: str,
        variant: str,
        attempt_root: Path,
        token: str,
        *,
        smoke: bool,
    ) -> list[str]:
        script = Path(self.args.repo_container) / "scripts/experiments/lungct_no_sweep_repro" / WORKER_SCRIPT_NAME
        command = [
            "docker", "exec",
            "--workdir", str(self.args.repo_container),
            "--env", "CUDA_VISIBLE_DEVICES=1",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "OMP_NUM_THREADS=6",
            "--env", "MKL_NUM_THREADS=6",
            "--env", "OPENBLAS_NUM_THREADS=6",
            str(self.args.container),
            "setpriv",
            "--reuid=1009",
            "--regid=1009",
            "--groups=1009,1020",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "python", "-B", str(script),
            "--worker",
            "--run-id", str(self.args.run_id),
            "--gpu", str(PHYSICAL_GPU),
            "--worker-dataset", dataset,
            "--worker-variant", variant,
            "--attempt-root", str(container_path(self.args, attempt_root)),
            "--job-token", token,
            "--data-root", str(self.args.data_root),
        ]
        if smoke:
            command.append("--smoke")
        return command

    def terminate_owned_worker(self, identity: dict[str, Any], sig: signal.Signals = signal.SIGTERM) -> None:
        try:
            current = process_identity(int(identity["pid"]))
        except FileNotFoundError:
            return
        if int(current["start_ticks"]) != int(identity["start_ticks"]):
            raise StudyError("Refusing to signal a reused PID")
        os.kill(int(identity["pid"]), sig)

    def run_attempt(self, dataset: str, variant: str, attempt_root: Path, *, smoke: bool) -> dict[str, Any]:
        self.monitor.owner = None
        self.monitor.wait_clear(float(self.args.poll_seconds), float(self.args.wait_timeout_seconds))
        token = uuid.uuid4().hex
        command = self.docker_command(dataset, variant, attempt_root, token, smoke=smoke)
        attempt_root.parent.mkdir(parents=True, exist_ok=True)
        log_path = attempt_root.parent / f"{attempt_root.name}.docker.log"
        log_handle = log_path.open("w")
        launched = time.time()
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
        try:
            ready_path = attempt_root / "ready.json"
            deadline = time.monotonic() + 120
            while not ready_path.exists():
                if process.poll() is not None:
                    raise StudyError(f"Worker exited before ready with code {process.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Worker did not create ready.json")
                time.sleep(0.1)
            ready = read_json(ready_path)
            identity = find_worker_identity(token, int(ready["container_pid"]))
            self.monitor.owner = identity
            dump_json(attempt_root / "host_ack.json", identity)
            started = time.time()
            last_sample = 0.0
            observed_progress = 0
            contaminated = False
            contamination_sample: dict[str, Any] | None = None
            timed_out = False
            while process.poll() is None:
                now = time.time()
                if now - last_sample >= float(self.args.poll_seconds):
                    sample = self.monitor.sample(reason="periodic")
                    last_sample = now
                    if sample["error"] or sample["foreign"]:
                        contaminated = True
                        contamination_sample = sample
                        self.terminate_owned_worker(identity)
                        break
                progress = read_json(attempt_root / "progress.json", []) or []
                if isinstance(progress, list) and len(progress) > observed_progress:
                    observed_progress = len(progress)
                    sample = self.monitor.sample(reason=f"case_boundary_{observed_progress:02d}")
                    last_sample = now
                    if sample["error"] or sample["foreign"]:
                        contaminated = True
                        contamination_sample = sample
                        self.terminate_owned_worker(identity)
                        break
                if now - started > float(self.args.worker_timeout_seconds):
                    timed_out = True
                    self.terminate_owned_worker(identity)
                    break
                time.sleep(min(0.2, float(self.args.poll_seconds)))
            try:
                returncode = process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.terminate_owned_worker(identity, signal.SIGKILL)
                returncode = process.wait(timeout=120)
            end = time.time()
            # The worker PID is expected to be gone at this point, so clear the
            # ownership mapping before the mandatory arm-end NVML sample.
            self.monitor.owner = None
            self.monitor.sample(reason="arm_end")
            verdict = self.monitor.interval(started, end)
            result = read_json(attempt_root / "worker_result.json", {}) or {}
            expected_count = 1 if smoke else PRIMARY_ROWS_PER_ARM
            accepted = (
                not contaminated
                and not timed_out
                and returncode == 0
                and bool(verdict["clean"])
                and result.get("status") == "completed"
                and int(result.get("row_count", -1)) == expected_count
            )
            payload = {
                "accepted": accepted,
                "dataset": dataset,
                "variant": variant,
                "smoke": smoke,
                "attempt_root": str(attempt_root),
                "command": command,
                "worker_identity": identity,
                "returncode": returncode,
                "contaminated": contaminated,
                "contamination_sample": contamination_sample,
                "timed_out": timed_out,
                "monitor": verdict,
                "worker_result": result,
                "launched_unix": launched,
                "ended_unix": end,
                "log_path": str(log_path),
            }
            dump_json(attempt_root / "attempt_verdict.json", payload)
            return payload
        finally:
            self.monitor.owner = None
            log_handle.close()

    def accepted_arm_valid(self, key: str, *, smoke: bool) -> bool:
        record = self.accepted.get(key)
        if not isinstance(record, dict) or not record.get("accepted"):
            return False
        root = Path(str(record.get("attempt_root", "")))
        result = read_json(root / "worker_result.json", {}) or {}
        return (
            result.get("status") == "completed"
            and int(result.get("row_count", -1)) == (1 if smoke else PRIMARY_ROWS_PER_ARM)
            and bool(read_json(root / "attempt_verdict.json", {}).get("accepted"))
        )

    def run_arm(self, dataset: str, variant: str, *, smoke: bool) -> dict[str, Any]:
        namespace = "smoke" if smoke else "primary"
        key = f"{namespace}:{dataset}:{variant}"
        if self.accepted_arm_valid(key, smoke=smoke):
            print(json.dumps({"resume": "accepted_arm", "key": key}), flush=True)
            return self.accepted[key]
        parent = self.run_root / "attempts" / namespace / f"{dataset}_{variant}"
        for attempt_index in range(int(self.args.max_arm_attempts)):
            attempt_root = parent / f"attempt_{attempt_index:02d}"
            if attempt_root.exists():
                continue
            verdict = self.run_attempt(dataset, variant, attempt_root, smoke=smoke)
            if verdict["accepted"]:
                self.accepted[key] = verdict
                dump_json(self.accepted_path, self.accepted)
                return verdict
            print(json.dumps({"retrying_arm": key, "attempt": attempt_index, "verdict": verdict}), flush=True)
        raise StudyError(f"Arm {key} exceeded {self.args.max_arm_attempts} attempts")

    def run_loader_audit(self) -> None:
        preflight_root = self.run_root / "preflight"
        audit_path = preflight_root / "loader_audit.json"
        if audit_path.exists() and read_json(audit_path, {}).get("status") == "passed":
            return
        preflight_root.mkdir(parents=True, exist_ok=True)
        script = Path(self.args.repo_container) / "scripts/experiments/lungct_no_sweep_repro" / WORKER_SCRIPT_NAME
        command = [
            "docker", "exec", "--user", "1009:1009", "--workdir", str(self.args.repo_container),
            "--env", "CUDA_VISIBLE_DEVICES=", "--env", "PYTHONDONTWRITEBYTECODE=1",
            str(self.args.container), "python", "-B", str(script),
            "--preflight-worker", "--run-id", str(self.args.run_id), "--gpu", str(PHYSICAL_GPU),
            "--data-root", str(self.args.data_root),
            "--attempt-root", str(container_path(self.args, preflight_root)),
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=1800)
        dump_json(
            preflight_root / "loader_audit_command.json",
            {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
        )
        if result.returncode != 0 or not audit_path.exists():
            raise StudyError(f"Loader audit failed: {result.stderr[-2000:]}")

    def preflight(self) -> None:
        self.run_loader_audit()
        for dataset in DATASETS:
            for variant in VARIANTS:
                self.run_arm(dataset, variant, smoke=True)
        dump_json(
            self.run_root / "preflight_DONE.json",
            {
                "status": "passed",
                "utc": datetime.now(timezone.utc).isoformat(),
                "loader_cases": 20,
                "smoke_arms": 4,
                "physical_gpu": PHYSICAL_GPU,
            },
        )

    def primary(self) -> None:
        preflight = read_json(self.run_root / "preflight_DONE.json", {}) or {}
        if preflight.get("status") != "passed":
            raise StudyError("Run --phase preflight before --phase all")
        for dataset in DATASETS:
            for variant in VARIANTS:
                self.run_arm(dataset, variant, smoke=False)

    def aggregate(self) -> None:
        script = Path(self.args.repo_container) / "scripts/experiments/lungct_no_sweep_repro" / AGGREGATOR_SCRIPT_NAME
        command = [
            "docker", "exec", "--user", "1009:1009", "--workdir", str(self.args.repo_container),
            "--env", "CUDA_VISIBLE_DEVICES=", "--env", "PYTHONDONTWRITEBYTECODE=1",
            str(self.args.container), "python", "-B", str(script),
            "--run-root", str(container_path(self.args, self.run_root)),
            "--review-root", str(container_path(self.args, self.review_root)),
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=7200)
        dump_json(
            self.run_root / "aggregate_command.json",
            {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
        )
        if result.returncode != 0:
            raise StudyError(f"Aggregation failed: {result.stderr[-4000:]}")


def container_is_running(name: str) -> bool:
    status = subprocess.check_output(["docker", "inspect", "--format", "{{.State.Running}}", name], text=True).strip()
    return status == "true"


def ensure_container(args: argparse.Namespace) -> None:
    if not container_is_running(args.container):
        subprocess.run(["docker", "start", args.container], check=True, cwd=REPO_ROOT)
    inspect = json.loads(subprocess.check_output(["docker", "inspect", args.container], text=True))[0]
    mounts = {item["Destination"]: item for item in inspect.get("Mounts", [])}
    artifact_mount = mounts.get(str(args.artifact_container))
    repo_mount = mounts.get(str(args.repo_container).split("/code_sync/inrorspline")[0])
    if artifact_mount is None or not bool(artifact_mount.get("RW")):
        raise StudyError(f"Container lacks writable artifact mount at {args.artifact_container}")
    if repo_mount is None:
        raise StudyError(f"Container cannot see repository path {args.repo_container}")


def write_dry_run(controller: Controller) -> None:
    manifest = immutable_manifest(controller.args)
    controller.ensure_manifest()
    output = controller.review_root / "dry_run_jobs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = manifest["primary_jobs"]
    fieldnames = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    print(json.dumps({"dry_run": True, "primary_rows": len(rows), "run_root": str(controller.run_root), "review_root": str(controller.review_root)}, indent=2), flush=True)


def main() -> None:
    args = resolve_artifact_paths(parse_args())
    if args.worker:
        raise SystemExit(worker_main(args))
    if args.preflight_worker:
        raise SystemExit(preflight_worker_main(args))
    ensure_container(args)
    controller = Controller(args)
    controller.ensure_manifest()
    if args.phase == "dry-run":
        write_dry_run(controller)
        return
    if args.phase == "report":
        controller.aggregate()
        return
    controller.provenance()
    if args.phase == "preflight":
        controller.preflight()
        return
    controller.primary()
    controller.aggregate()


if __name__ == "__main__":
    main()
