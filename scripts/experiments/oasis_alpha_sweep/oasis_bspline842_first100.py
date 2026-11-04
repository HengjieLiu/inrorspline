"""P79: explicit cohort/reuse contract; no changes to historical numerics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import common as c
from run_oasis_bspline_svf import atomic_json, sha

VARIANT = "oasis_seed3_first100_bspline842_displacement_bounded2_svf_velocity_ncc9_lambda_sweep_v1"
LAMBDAS = (.01, .1, .3, 1., 3., 10., 100.)
FAMILIES = ("displacement", "bounded2", "svf_velocity")
PATH_SETTINGS = ("oasis_root", "pair_file", "dl_code_root", "vfa_root")
EXECUTIONS = ("original", "metadata_recovery_v2")
METRICS = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
LABELS = {
    "displacement": "B-spline displacement · 8→4→2",
    "bounded2": "Bounded 2 maps/level · 8→4→2",
    "svf_velocity": "Refined SVF / velocity · 8→4→2",
}
P43 = "oasis_seed3_first10_mrdbscp_dl_ncc9_diffusion_6way_v1"
P48 = "oasis_seed3_first100_mrdbscp842_dense_ncc9_diffusion_pyramid_lambda_sweep_v1"
P76 = "oasis_seed3_first10_bspline_bounded2_842_ncc9_lambda_sweep_v1"
P74 = "oasis_seed3_first10_bspline_svf_refine_842_8421_ncc9_lambda_sweep_v1"
HERE = Path(__file__).resolve().parent
PROTOCOL = {
    "version": 1, "ranks": list(range(100)), "lambdas": list(LAMBDAS),
    "families": list(FAMILIES), "cps": [8, 4, 2], "downsampling": [4, 2, 1],
    "stage_alpha": "same selected lambda at all levels and both bounded maps",
    "lr_normalized": [.03, .01, .003], "max_updates_per_map": [400, 800, 1300],
    "min_updates_per_map": [40, 80, 130], "requested_patience": [20, 35, 50],
    "median_window": 25, "check_interval": 25, "relative_total": 1e-4,
    "relative_data": 5e-5, "lr_reductions": 0, "seed": 1,
    "loss": "raw squared local NCC9 plus unmasked evaluation-grid voxel diffusion",
    "svf_regularize": "total_velocity", "svf_integration_steps": 7,
    "svf_diagnostic_steps": [8, 10], "bounded_maps_per_level": 2,
    "bound_multiplier": .99, "fresh_missing_fits": 1530, "reused_fits": 570,
    "total_fits": 2100, "cudnn_benchmark": False, "cudnn_deterministic": True,
    "tf32": False, "foreground": "fixed_raw_image_positive",
    "reuse": "accepted same-protocol outputs; no warm starts for missing cells",
}


@dataclass(frozen=True)
class Spec:
    case_rank: int
    family: str
    alpha: float
    smoke: bool = False

    def validate(self):
        if type(self.case_rank) is not int or self.case_rank not in range(100):
            raise ValueError("Require shuffled rank0..99")
        if self.family not in FAMILIES or not math.isfinite(self.alpha) or self.alpha not in LAMBDAS:
            raise ValueError("Require an approved family and lambda")
        if type(self.smoke) is not bool:
            raise ValueError("smoke must be boolean")

    def name(self):
        self.validate()
        digest = hashlib.sha256(json.dumps({"spec": asdict(self), "protocol": PROTOCOL}, sort_keys=True).encode()).hexdigest()[:12]
        return ("smoke_" if self.smoke else "") + f"rank{self.case_rank:03d}_842_{self.family}_lambda{c.format_float(self.alpha)}_{digest}"


def matrix(smoke=False, missing=False):
    if smoke:
        return [Spec(10, f, a, True) for a in (.01, 100.) for f in FAMILIES]
    return [Spec(rank, family, alpha) for rank in range(100) for alpha in LAMBDAS for family in FAMILIES
            if not missing or (rank >= 10 and (family != "displacement" or alpha in (.01, 10., 100.)))]


def source_hashes(sitreg_root):
    import run_oasis_dl_ncc_diffusion_one as displacement
    import run_oasis_bounded2_842_sweep as bounded
    import run_oasis_bspline_svf_sweep_recovery as svf
    files = [HERE / name for name in ("oasis_bspline842_first100.py", "run_oasis_bspline842_first100.py",
             "queue_oasis_bspline842_first100.py", "queue_oasis_svf_continue_on_error.py",
             "queue_oasis_svf_eight_workers.py", "oasis_ncc9_first100.py")]
    files += [c.REPO_ROOT / "scripts/reports/report_oasis_bspline842_first100.py",
              c.REPO_ROOT / "tests/test_oasis_bspline842_first100.py"]
    return {"displacement": displacement.source_hashes(), "bounded2": bounded.sources(Path(sitreg_root)),
            "svf": svf.source_hashes(), "svf_recovery": svf.recovery_hashes(),
            "extension": {str(p.relative_to(c.REPO_ROOT)): sha(p) for p in files}}


def worker_settings(settings):
    """Legacy SVF setup accepts paths only, never controller configuration."""
    result = {}
    for key in PATH_SETTINGS:
        value = settings[key]
        if not isinstance(value, (str, os.PathLike)) or not str(value):
            raise ValueError(f"Expected a nonempty filesystem path for {key}")
        result[key] = str(Path(value))
    return result


def run_root(root, execution="original"):
    if execution not in EXECUTIONS:
        raise ValueError("Unknown execution revision")
    base = Path(root) / "runs" / VARIANT
    return base if execution == "original" else base / execution


def assert_manifest(m):
    if m["variant"] != VARIANT or m["protocol"] != PROTOCOL:
        raise RuntimeError("Wrong first100 experiment protocol")
    if m["settings"] != worker_settings(m["settings"]):
        raise RuntimeError("Worker settings must contain only the four data/code paths")
    if m["source_hashes"] != source_hashes(m["sitreg_root"]):
        raise RuntimeError("Frozen experiment sources changed")
    if sha(m["settings"]["pair_file"]) != m["pair_file_sha256"]:
        raise RuntimeError("Split changed")
    for path, digest in m["reference_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError(f"Historical report changed: {path}")


def read_accepted(root, name):
    folder = root / "repo_experiments" / name
    accepted = c.read_json(folder / "ACCEPTED.json")
    if not any(accepted.get(k) is True for k in ("passed", "complete", "technical_completion")):
        raise RuntimeError(f"Unaccepted source: {folder}")
    for name, digest in accepted.get("output_sha256", accepted.get("report_sha256", {})).items():
        if sha(folder / name) != digest:
            raise RuntimeError(f"Accepted report hash mismatch: {folder / name}")
    return c.read_json(folder / "results.json"), folder


def existing(root):
    data, refs = [], {}
    for name in (P43, P48, P76, P74 + "/continuation_v2"):
        report, folder = read_accepted(root, name)
        data.append(report)
        for filename in ("results.json", "ACCEPTED.json"):
            refs[str(folder / filename)] = sha(folder / filename)
    selected = {}
    for report in data[:2]:
        for row in report["candidates"]:
            s = row["spec"]
            if s["representation"] != "842" or s["resolution"] != "pyramid":
                continue
            spec = Spec(s["case_rank"], "displacement", s["alpha"])
            if spec.name() in selected and selected[spec.name()]["row"]["dvf_sha256"] != row["dvf_sha256"]:
                raise RuntimeError("Disagreeing duplicate reused prefix")
            selected[spec.name()] = {"spec": asdict(spec), "row": row}
    for row in data[2]["candidates"]:
        spec = Spec(row["case_rank"], "bounded2", row["alpha"])
        selected[spec.name()] = {"spec": asdict(spec), "row": row}
    for row in data[3]["candidates"]:
        s = row["spec"]
        if s["schedule"] == "842" and s["regularize"] == "velocity":
            spec = Spec(s["case_rank"], "svf_velocity", s["alpha"])
            selected[spec.name()] = {"spec": asdict(spec), "row": row}
    expected = {s.name() for s in matrix()} - {s.name() for s in matrix(missing=True)}
    if set(selected) != expected or len(selected) != 570:
        raise RuntimeError("Reusable matrix is not exactly430 displacement +70 bounded +70 SVF")
    references = [x for x in data[1]["case_metrics"] if x["representation"] in ("VFA", "SITReg")]
    if {(x["representation"], x["case_rank"]) for x in references} != {(m, i) for m in ("VFA", "SITReg") for i in range(100)} or len(references) != 200:
        raise RuntimeError("Require200 canonical reference records")
    return selected, references, refs


def fingerprint(path):
    st = Path(path).stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def check_history(path, info, alpha):
    with Path(path).open() as stream:
        history = list(csv.DictReader(stream))
    if [int(x["epoch"]) for x in history] != list(range(info["actual_epochs"] + 1)):
        raise RuntimeError("History update sequence mismatch")
    for x in history:
        values = [float(x[k]) for k in ("loss", "data_loss", "reg_loss")]
        if not all(math.isfinite(v) for v in values) or not math.isclose(values[0], values[1] + alpha*values[2], rel_tol=1e-6, abs_tol=1e-7):
            raise RuntimeError("History weighted objective mismatch")
    best = min(history, key=lambda x: float(x["loss"]))
    if int(best["epoch"]) != info["best_epoch"] or not math.isclose(float(best["loss"]), info["best_loss"], rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError("History best-state mismatch")
    return len(history)


def validate_identity(row, spec, manifest):
    case = manifest["cases"][str(spec.case_rank)]
    if any(row[k] != case[k] for k in ("case_rank", "fixed_id", "moving_id", "input_hashes")):
        raise RuntimeError("Run case/input mismatch")
    if row["direction"] != "moving_to_fixed" or row["pair_file_sha256"] != manifest["pair_file_sha256"]:
        raise RuntimeError("Direction/split mismatch")
    if not all(math.isfinite(float(row[k])) for k in METRICS):
        raise RuntimeError("Nonfinite metric")
    if row["cuda_backend"]["cudnn_benchmark"] or not row["cuda_backend"]["cudnn_deterministic"] or row["tf32_matmul"] or row["tf32_cudnn"]:
        raise RuntimeError("Backend mismatch")
    if spec.family == "displacement":
        if row["spec"] != {"case_rank": spec.case_rank, "resolution": "pyramid", "alpha": spec.alpha,
                           "representation": "842", "seed": 1, "smoke": spec.smoke}:
            raise RuntimeError("Wrong displacement variant")
        import dl_ncc_diffusion as d
        if row["protocol"] != d.PROTOCOL or [x["cps"] for x in row["stages"]] != [8,4,2] or any(x["alpha"] != spec.alpha for x in row["stages"]):
            raise RuntimeError("Displacement protocol/stages mismatch")
    elif spec.family == "bounded2":
        if row["cps_schedule"] != [8,4,2] or row["maps_per_level"] != 2 or not row["bounded"] or row["bound_multiplier"] != .99 or row["stage_alpha"] != [spec.alpha]*3:
            raise RuntimeError("Wrong bounded variant")
        if len(row["blocks"]) != 6 or any(not b["atomic_corner_audit"]["bound_below_one"] for b in row["blocks"]):
            raise RuntimeError("Bounded contraction verification missing")
    else:
        s = row["spec"]
        if (s["schedule"], s["regularize"], s["formulation"], s["integration_steps"], s["alpha"], s["smoke"]) != ("842", "velocity", "refine", 7, spec.alpha, spec.smoke):
            raise RuntimeError("Wrong SVF variant")
        if row["stage_alpha"] != [spec.alpha]*3 or len(row["blocks"]) != 3:
            raise RuntimeError("Wrong SVF stages")


def artifact_receipt(row, spec, manifest):
    """Hash/finite validation is read-only; receipts live only in the new family."""
    validate_identity(row, spec, manifest)
    path = Path(row["run_dir"])
    if not (path / "DONE").is_file():
        raise RuntimeError("Missing DONE")
    disk_row = c.read_json(path / "summary.json")
    if any(disk_row[k] != row[k] for k in (*METRICS, "input_hashes", "dvf_sha256", "source_hashes")):
        raise RuntimeError("Accepted report differs from saved summary")
    checks = {}
    def check(p, expected=None, field=False):
        p = Path(p); digest = sha(p)
        if expected is not None and digest != expected:
            raise RuntimeError(f"Artifact checksum mismatch: {p}")
        if field:
            arr = np.load(p, mmap_mode="r")
            if arr.shape != (160,224,192,3) or arr.dtype != np.float32 or not np.isfinite(arr).all():
                raise RuntimeError(f"Nonfinite/wrong saved field: {p}")
        checks[str(p)] = {**fingerprint(p), "sha256": digest}
    check(path / "DONE"); check(path / "summary.json"); check(row["dvf_path"], row["dvf_sha256"], True)
    if spec.family == "displacement":
        for i, info in enumerate(row["stages"]):
            folder = path / f"stage{i}"
            check(folder / "best.pt", row["model_sha256"] if i == 2 else None)
            check(folder / "last.pt"); check(folder / "history.csv")
            check_history(folder / "history.csv", info, spec.alpha)
        check(path / "case_metrics.csv")
    else:
        audit = c.read_json(path / "independent_audit.json")
        if not audit["passed"] or audit["summary_sha256"] != sha(path / "summary.json"):
            raise RuntimeError("Missing/stale independent replay")
        check(path / "independent_audit.json"); check(path / "final_metrics/case_metrics.csv")
        for block in row["blocks"]:
            folder = Path(block["model_path"]).parent
            check(block["model_path"], block["model_sha256"])
            check(block["dvf_path"], block["dvf_sha256"], True)
            check(folder / "last.pt", block.get("last_model_sha256"))
            check(folder / "history.csv", block.get("history_sha256"))
            check_history(folder / "history.csv", block["optimizer"], spec.alpha)
        for diag in row.get("integration_diagnostics", []):
            if "dvf_path" in diag:
                check(diag["dvf_path"], diag["dvf_sha256"], True)
    return {"passed": True, "spec": asdict(spec), "files": checks, "checked_at": c.now_text()}


def verify_receipt(receipt):
    if not receipt.get("passed"):
        raise RuntimeError("Unpassed receipt")
    for path, item in receipt["files"].items():
        if fingerprint(path) != {k: item[k] for k in ("size", "mtime_ns")}:
            raise RuntimeError(f"Previously validated artifact changed: {path}")


def receipt_origin(root, manifest):
    """Reuse validated hash receipts, not old smoke outputs or source approval."""
    origin = run_root(root)
    path = origin / "launch_manifest.json"
    old = c.read_json(path)
    gate = c.read_json(origin / "REUSE_VERIFIED.json")
    if (not gate.get("passed") or gate.get("count") != 570 or
            gate.get("reference_count") != 200 or gate.get("manifest_sha256") != sha(path)):
        raise RuntimeError("Original reuse verification is missing or stale")
    for key in ("variant", "protocol", "cases", "reused", "references", "reference_hashes",
                "pair_file_sha256", "bounds", "sitreg_root", "sitreg_revision", "recovery"):
        if old[key] != manifest[key]:
            raise RuntimeError(f"Recovery changed the scientific/reuse contract: {key}")
    if worker_settings(old["settings"]) != manifest["settings"]:
        raise RuntimeError("Recovery changed dataset/code paths")
    for key in ("displacement", "bounded2", "svf", "svf_recovery"):
        if old["source_hashes"][key] != manifest["source_hashes"][key]:
            raise RuntimeError(f"Historical numerical sources changed: {key}")
    return origin, {"manifest_path": str(path), "manifest_sha256": sha(path),
                    "verification_sha256": sha(origin / "REUSE_VERIFIED.json"),
                    "policy": "stat-bound previously hashed artifacts; numerical sources unchanged"}


def prepare(root, image, image_id, execution="original"):
    out = run_root(root, execution)
    out.mkdir(parents=True, exist_ok=True)
    mp = out / "launch_manifest.json"
    if mp.exists():
        m = c.read_json(mp); assert_manifest(m)
    else:
        selected, references, hashes = existing(root)
        parent_path = root / "runs" / P76 / "launch_manifest.json"
        parent = c.read_json(parent_path)
        hashes[str(parent_path)] = sha(parent_path)
        cases = {}
        for item in selected.values():
            row = item["row"]
            case = {k: row[k] for k in ("case_rank", "fixed_id", "moving_id", "input_hashes")}
            case["original_case_index"] = row.get("original_case_index")
            key = str(case["case_rank"])
            if key in cases and any(cases[key][k] != case[k] for k in ("fixed_id", "moving_id", "input_hashes")):
                raise RuntimeError("Different source inputs for same case")
            if key not in cases or case["original_case_index"] is not None:
                cases[key] = case
        pairs = c.parse_pair_file(Path(parent["settings"]["pair_file"]))
        if len(cases) != 100 or any(tuple(pairs[i]) != (cases[str(i)]["fixed_id"], cases[str(i)]["moving_id"]) for i in range(100)):
            raise RuntimeError("First100 case manifest mismatch")
        free = shutil.disk_usage(out).free / 2**30
        if free < 800:
            raise RuntimeError("Require800GiB free before launch")
        m = {"variant": VARIANT, "protocol": PROTOCOL, "cases": cases, "settings": worker_settings(parent["settings"]),
             "execution_revision": execution,
             "sitreg_root": parent["sitreg_root"], "sitreg_revision": parent["sitreg_revision"],
             "bounds": parent["bounds"], "source_hashes": source_hashes(parent["sitreg_root"]),
             "source_revision": subprocess.check_output(["git","rev-parse","HEAD"], cwd=c.REPO_ROOT, text=True).strip(),
             "source_dirty": bool(subprocess.check_output(["git","status","--porcelain"], cwd=c.REPO_ROOT, text=True)),
             "reference_hashes": hashes, "pair_file_sha256": sha(parent["settings"]["pair_file"]),
             "references": references, "reused": selected, "gpus": ["0","1","2","3"],
             "container_image": image, "container_image_id": image_id, "uid": os.getuid(), "gid": os.getgid(),
             "created_at": c.now_text(), "initial_free_gib": free}
        import run_oasis_bspline_svf_sweep_recovery as svf
        m["recovery"] = {"source_hashes": svf.recovery_hashes(),
                         "transition_policy": svf.transition.POLICY,
                         "note": "unchanged audited P74 FP64-gated validation; all new fits start at identity"}
        if execution != "original":
            _, m["reuse_receipt_origin"] = receipt_origin(root, m)
        atomic_json(mp, m)
    origin = None
    if execution != "original":
        origin, provenance = receipt_origin(root, m)
        if provenance != m["reuse_receipt_origin"]:
            raise RuntimeError("Original reuse provenance changed")
    input_files = {str(Path(m["settings"]["oasis_root"])/"test"/name): digest
                   for case in m["cases"].values() for name,digest in case["input_hashes"].items()}
    for path, digest in input_files.items():
        if sha(path) != digest:
            raise RuntimeError(f"Input changed: {path}")
    receipts = out / "reuse_receipts"; receipts.mkdir(exist_ok=True)
    for i, (name, item) in enumerate(m["reused"].items()):
        rp = receipts / f"{name}.json"
        if rp.exists():
            verify_receipt(c.read_json(rp))
        elif origin is not None:
            old_path = origin / "reuse_receipts" / rp.name
            receipt = c.read_json(old_path)
            if receipt["spec"] != item["spec"]:
                raise RuntimeError("Original receipt belongs to a different registration")
            verify_receipt(receipt)
            atomic_json(rp, {**receipt, "imported_from": str(old_path),
                            "origin_sha256": sha(old_path), "rechecked_at": c.now_text()})
        else:
            atomic_json(rp, artifact_receipt(item["row"], Spec(**item["spec"]), m))
        if (i+1) % 10 == 0:
            print(json.dumps({"reuse_validated": i+1, "total": 570}), flush=True)
    for row in m["references"]:
        if sha(row["source_path"]) != row["source_sha256"]:
            raise RuntimeError("Reference field changed")
    import bounded_composed_bspline as bounded
    if any(bounded.compute_bound(cps) != m["bounds"][str(cps)] for cps in (8,4,2)):
        raise RuntimeError("Validated bounds changed")
    assert_manifest(m)
    atomic_json(out / "REUSE_VERIFIED.json", {"passed": True, "count": 570, "reference_count": 200,
                "manifest_sha256": sha(mp), "checked_at": c.now_text()})
    return m


def validate(path, manifest, spec):
    row = c.read_json(path / "summary.json")
    validate_identity(row, spec, manifest)
    audit = c.read_json(path / "independent_audit.json")
    receipt = c.read_json(path / "extension_receipt.json")
    verify_receipt(receipt)
    if not (path / "DONE").is_file() or not audit["passed"] or audit["summary_sha256"] != sha(path / "summary.json") or receipt["spec"] != asdict(spec):
        raise RuntimeError("Incomplete or stale audited result")
    return row
