#!/usr/bin/env python
"""Launch fixed-parameter COPD ``ptvreg-python`` runs across GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import run_copd_case as runner


DEFAULT_REVIEW_ROOT = runner.REPO_ROOT / "experiments" / "2026-08-28_ptvreg_python_copd_fixed_all10"
DEFAULT_LOG_ROOT = runner.REPO_ROOT / "logs" / "ptvreg_python_copd_fixed_all10_v1"
RUNNER = Path(__file__).resolve().parent / "run_copd_case.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=runner.PROTOCOL_CHOICES, default="canonical_1mm")
    parser.add_argument("--cases", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--data-root", type=Path, default=runner.c.DEFAULT_PTVPREP_ROOT)
    parser.add_argument("--raw-data-root", type=Path, default=None)
    parser.add_argument("--paper-hu-root", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=runner.DEFAULT_RUN_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-dense-dvf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-mask-fallback", action="store_true")

    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--admm-max-outer", type=int, default=10)
    parser.add_argument("--admm-lbfgs-iters", type=int, default=30)
    parser.add_argument("--nlvl", type=int, default=None)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def is_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return False
    return payload.get("status") == "completed"


def build_opts_namespace(args: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace(
        dtype=args.dtype,
        opt_method="admm",
        grid_spacing="8,8,8",
        isoTV=0.12,
        metric="lcc",
        metric_param="2.1,2.1,2.1",
        scale_metric_param=True,
        pix_resolution="1,1,1",
        spline_order=1,
        k_down=0.7,
        border_mask=5,
        interp_type=1,
        max_iters=int(args.max_iters),
        cp_refinements=0,
        csqrt=5e-3,
        verbose=bool(args.verbose),
        admm_max_outer=int(args.admm_max_outer),
        admm_lbfgs_iters=int(args.admm_lbfgs_iters),
        admm_mu=10.0,
        admm_tau=1.3,
        admm_tol=1e-2,
        admm_rho=None,
        nlvl=args.nlvl,
    )
    return ns


def build_job(args: argparse.Namespace, case_id: int) -> Dict[str, object]:
    opts = runner.build_opts(build_opts_namespace(args), "cuda")
    name = runner.run_name_for_case(str(args.protocol), int(case_id), opts)
    tag = runner.protocol_tag(str(args.protocol))
    raw_root = args.raw_data_root or runner.default_raw_dirlab_root(args.data_root)
    paper_hu_root = runner.paper_hu_root_for_protocol(str(args.protocol), args.paper_hu_root)
    mask_root = args.mask_root or runner.default_native_mask_root(Path(raw_root))
    summary_path = Path(args.out_root) / tag / name / "summary.json"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--protocol",
        str(args.protocol),
        "--case-id",
        str(int(case_id)),
        "--data-root",
        str(args.data_root),
        "--paper-hu-root",
        str(paper_hu_root),
        "--out-root",
        str(args.out_root),
        "--max-iters",
        str(int(args.max_iters)),
        "--admm-max-outer",
        str(int(args.admm_max_outer)),
        "--admm-lbfgs-iters",
        str(int(args.admm_lbfgs_iters)),
        "--dtype",
        str(args.dtype),
    ]
    if args.raw_data_root is not None:
        cmd.extend(["--raw-data-root", str(args.raw_data_root)])
    if str(args.protocol) == "paper_hu_native":
        cmd.extend(["--mask-root", str(mask_root)])
        if args.allow_mask_fallback:
            cmd.append("--allow-mask-fallback")
    if args.nlvl is not None:
        cmd.extend(["--nlvl", str(int(args.nlvl))])
    if args.verbose:
        cmd.append("--verbose")
    if not args.save_dense_dvf:
        cmd.append("--no-save-dense-dvf")
    if args.force:
        cmd.append("--overwrite")
    return {
        "case_id": int(case_id),
        "protocol": str(args.protocol),
        "protocol_tag": tag,
        "run_name": name,
        "summary_path": str(summary_path),
        "log_path": str(Path(args.log_root) / str(args.protocol) / f"{name}.log"),
        "paper_hu_root": str(paper_hu_root),
        "raw_data_root": str(raw_root),
        "mask_root": str(mask_root) if str(args.protocol) == "paper_hu_native" else None,
        "cmd": cmd,
    }


def build_jobs(args: argparse.Namespace) -> List[Dict[str, object]]:
    cases = runner.c.parse_int_list(args.cases)
    return [build_job(args, int(case_id)) for case_id in cases]


def write_manifest(args: argparse.Namespace, jobs: List[Dict[str, object]], queued: List[Dict[str, object]]) -> None:
    review_root = Path(args.review_root)
    review_root.mkdir(parents=True, exist_ok=True)
    rows = []
    queued_paths = {job["summary_path"] for job in queued}
    for idx, job in enumerate(jobs):
        rows.append(
            {
                "case_id": job["case_id"],
                "protocol": job["protocol"],
                "protocol_tag": job["protocol_tag"],
                "run_name": job["run_name"],
                "summary_path": job["summary_path"],
                "log_path": job["log_path"],
                "paper_hu_root": job.get("paper_hu_root"),
                "raw_data_root": job.get("raw_data_root"),
                "mask_root": job.get("mask_root"),
                "queued": job["summary_path"] in queued_paths,
                "command": " ".join(str(v) for v in job["cmd"]),
            }
        )
    fixed_options = runner.opts_for_json(runner.build_opts(build_opts_namespace(args), "cuda"))
    if str(args.protocol) == "paper_hu_native":
        fixed_options["pix_resolution"] = "case_native_spacing_zyx_mm"
        fixed_options["effective_pix_resolution_source"] = "raw COPD case spacing"
    payload = {
        "created_wall": runner.c.now_text(),
        "protocol": str(args.protocol),
        "requested_jobs": len(jobs),
        "queued_jobs": len(queued),
        "skipped_completed": len(jobs) - len(queued),
        "gpus": [str(v) for v in runner.c.parse_int_list(args.gpus)],
        "data_root": str(args.data_root),
        "raw_data_root": str(args.raw_data_root) if args.raw_data_root is not None else None,
        "paper_hu_root": str(runner.paper_hu_root_for_protocol(str(args.protocol), args.paper_hu_root)),
        "mask_root": str(args.mask_root) if args.mask_root is not None else None,
        "out_root": str(args.out_root),
        "log_root": str(args.log_root),
        "fixed_options": fixed_options,
        "jobs": rows,
    }
    runner.c.write_json(review_root / f"job_manifest_{args.protocol}.json", payload)
    runner.c.write_csv(review_root / f"job_manifest_{args.protocol}.csv", rows)


def maybe_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.cases = "1"
    args.max_iters = min(int(args.max_iters), 1)
    args.admm_max_outer = min(int(args.admm_max_outer), 1)
    args.admm_lbfgs_iters = min(int(args.admm_lbfgs_iters), 1)
    args.nlvl = 1 if args.nlvl is None else min(int(args.nlvl), 1)
    args.save_dense_dvf = False


def main() -> None:
    args = parse_args()
    maybe_smoke_defaults(args)
    gpus = [str(v) for v in runner.c.parse_int_list(args.gpus)]
    if not gpus:
        raise SystemExit("At least one GPU is required.")
    jobs = build_jobs(args)
    queued = [job for job in jobs if args.force or not is_completed(Path(job["summary_path"]))]
    write_manifest(args, jobs, queued)
    manifest = {
        "requested_jobs": len(jobs),
        "queued_jobs": len(queued),
        "skipped_completed": len(jobs) - len(queued),
        "protocol": str(args.protocol),
        "gpus": gpus,
        "out_root": str(args.out_root),
        "review_root": str(args.review_root),
        "log_root": str(args.log_root),
        "smoke": bool(args.smoke),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        for idx, job in enumerate(queued):
            gpu = gpus[idx % len(gpus)]
            print(
                json.dumps(
                    {
                        "dry_run_job": job["run_name"],
                        "case_id": job["case_id"],
                        "assigned_gpu": gpu,
                        "cmd": list(job["cmd"]) + ["--gpu", gpu],
                    },
                    indent=2,
                ),
                flush=True,
            )
        return

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.log_root).mkdir(parents=True, exist_ok=True)
    active: List[Dict[str, object]] = []
    pending = list(queued)
    failures: List[Dict[str, object]] = []
    while pending or active:
        while pending and len(active) < len(gpus):
            used = {str(item["gpu"]) for item in active}
            free = [gpu for gpu in gpus if gpu not in used]
            if not free:
                break
            gpu = free[0]
            job = pending.pop(0)
            cmd = list(job["cmd"]) + ["--gpu", gpu]
            log_path = Path(str(job["log_path"]))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w")
            proc = subprocess.Popen(
                cmd,
                cwd=str(runner.REPO_ROOT),
                env=os.environ.copy(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append({"proc": proc, "log_file": log_file, "gpu": gpu, "job": job, "start": time.time()})
            print(
                json.dumps(
                    {
                        "launched": job["run_name"],
                        "case_id": job["case_id"],
                        "gpu": gpu,
                        "pid": proc.pid,
                        "remaining": len(pending),
                        "log": str(log_path),
                    }
                ),
                flush=True,
            )

        time.sleep(30)
        still_active: List[Dict[str, object]] = []
        for item in active:
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            code = proc.poll()
            if code is None:
                still_active.append(item)
                continue
            item["log_file"].close()
            job = item["job"]
            elapsed = time.time() - float(item["start"])
            payload = {
                "finished": job["run_name"],
                "returncode": code,
                "elapsed_sec": elapsed,
                "gpu": item["gpu"],
                "case_id": job["case_id"],
                "log": job["log_path"],
            }
            print(json.dumps(payload), flush=True)
            if code != 0:
                failures.append(payload)
        active = still_active
    if failures:
        raise SystemExit(json.dumps({"failed_jobs": failures}, indent=2))


if __name__ == "__main__":
    main()
