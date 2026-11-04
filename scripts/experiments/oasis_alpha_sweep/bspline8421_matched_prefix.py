"""P80 matched accepted 842 prefixes -> fresh cps1, with immutable sources."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import oasis_bspline842_first100 as old

c, sha, atomic_json = old.c, old.sha, old.atomic_json
HERE = Path(__file__).resolve().parent
VARIANT = "oasis_seed3_first100_bspline8421_matched_prefix_displacement_bounded2_svf_velocity_ncc9_lambda_sweep_v1"
FAMILIES, LAMBDAS, METRICS = old.FAMILIES, old.LAMBDAS, old.METRICS
PREFIX_RELEASE = "transition_continuation_v3/release_1792210489765127750"
DL_REPORT = "oasis_seed3_first100_four_method_sweeps_arc_v1"
PROTOCOL = {
    "version": 1, "schedule": [8, 4, 2, 1], "retained_levels": [8, 4, 2],
    "lambdas": list(LAMBDAS), "ranks": list(range(100)), "families": list(FAMILIES),
    "stage_alpha": "same lambda at every stage and map", "native_shape": [160,224,192,3],
    "loss": "raw 1-mean(local NCC9 correlation squared)",
    "regularizer": "unmasked forward-valid evaluation-grid voxel diffusion component/axis mean",
    "targets": {"displacement": "total_displacement", "bounded2": "total_displacement", "svf_velocity": "total_velocity"},
    "cps1_lr": .001, "cps1_max_min": [2000,300], "patience": 100, "median_window": 25,
    "check_interval": 25, "relative_total_data": [1e-4,5e-5], "lr_halvings": 2,
    "min_between_actions": 100, "best": "minimum total loss including initialization",
    "bounded_maps_per_level": 2, "bound_multiplier": .99, "svf_steps": 7,
    "svf_diagnostic_steps": [7,8,10], "seed": 1, "fp32": True,
    "cudnn_benchmark": False, "cudnn_deterministic": True, "tf32": False,
    "initialization": "exact accepted same-family/case/lambda selected cps2 prefix",
    "svf_transition_policy": 4, "fresh_outputs": 2100, "new_optimizer_blocks": 2800,
}


@dataclass(frozen=True)
class Spec:
    case_rank: int
    family: str
    alpha: float
    smoke: bool = False

    def validate(self):
        old.Spec(self.case_rank, self.family, self.alpha, self.smoke).validate()

    def key(self):
        self.validate()
        return f"rank{self.case_rank:03d}_{self.family}_lambda{c.format_float(self.alpha)}"

    def name(self):
        digest = hashlib.sha256(json.dumps([asdict(self),PROTOCOL], sort_keys=True).encode()).hexdigest()[:12]
        weights = "_".join([c.format_float(self.alpha)]*4)
        return ("smoke_" if self.smoke else "") + f"{self.key()}_8421_a{weights}_{digest}"


def matrix(smoke=False):
    return [Spec(rank, family, alpha, smoke) for rank in ([0] if smoke else range(100))
            for alpha in ((.01,1.,100.) if smoke else LAMBDAS) for family in FAMILIES]


def sources(sitreg_root):
    names = ("bspline8421_matched_prefix.py", "svf_transition_validation_v4.py",
             "run_oasis_bspline8421_matched_prefix.py", "queue_oasis_bspline8421_matched_prefix.py")
    paths = [HERE/n for n in names] + [c.REPO_ROOT/"scripts/reports/report_oasis_bspline8421_matched_prefix.py",
             c.REPO_ROOT/"tests/test_oasis_bspline8421_matched_prefix.py"]
    return {"historical": old.source_hashes(sitreg_root),
            "new": {str(p.relative_to(c.REPO_ROOT)): sha(p) for p in paths}}


def assert_manifest(m, references=False):
    if m["variant"] != VARIANT or m["protocol"] != PROTOCOL:
        raise RuntimeError("Wrong matched-prefix protocol")
    if m["source_hashes"] != sources(m["sitreg_root"]):
        raise RuntimeError("Frozen numerical/controller/report source changed")
    if sha(m["settings"]["pair_file"]) != m["pair_file_sha256"]:
        raise RuntimeError("Pair order changed")
    if references:
        for path, digest in m["reference_hashes"].items():
            if sha(path) != digest:
                raise RuntimeError(f"Accepted reference changed: {path}")


def prepare(root,run_root=None):
    import bounded_two_blocks as b
    root = Path(root)
    out = Path(run_root) if run_root is not None else root/"runs"/VARIANT/"execution_v1"
    path = out/"launch_manifest.json"
    if path.exists():
        m = c.read_json(path); assert_manifest(m, True); return m
    source = root/"repo_experiments"/old.VARIANT/PREFIX_RELEASE
    accepted = c.read_json(source/"ACCEPTED.json")
    if not accepted.get("complete") or accepted["candidate_count"] != 2100 or accepted["failures"]:
        raise RuntimeError("Need accepted complete 2100-prefix release")
    refs = {str(source/"ACCEPTED.json"): sha(source/"ACCEPTED.json")}
    for name, digest in accepted["output_sha256"].items():
        if sha(source/name) != digest:
            raise RuntimeError(f"Prefix report drift: {name}")
        refs[str(source/name)] = digest
    parent_path = root/"runs"/old.VARIANT/"metadata_recovery_v2/launch_manifest.json"
    parent = c.read_json(parent_path)
    refs[str(parent_path)] = sha(parent_path)
    index = {}
    rows = c.read_json(source/"results.json")["per_case"]
    cases = parent["cases"]
    checked_inputs = set()
    for case in cases.values():
        for name, digest in case["input_hashes"].items():
            p = Path(parent["settings"]["oasis_root"])/"test"/name
            if str(p) not in checked_inputs and sha(p) != digest:
                raise RuntimeError(f"Changed raw input: {name}")
            checked_inputs.add(str(p))
    for r in rows:
        if r["family"] not in FAMILIES:
            continue
        spec = Spec(r["case_rank"], r["family"], r["alpha"])
        p = Path(r["run_dir"])/"summary.json"
        row = c.read_json(p)
        old.validate_identity(row, old.Spec(spec.case_rank, spec.family, spec.alpha), parent)
        if not (p.parent/"DONE").is_file() or row["dvf_sha256"] != r["dvf_sha256"]:
            raise RuntimeError(f"Invalid accepted prefix: {spec.key()}")
        if any(not math.isclose(row[k], r[k], abs_tol=1e-12, rel_tol=0) for k in METRICS):
            raise RuntimeError("Accepted prefix metric mismatch")
        models = ([{"model_path":row["model_path"], "model_sha256":row["model_sha256"]}]
                  if spec.family == "displacement" else row["blocks"])
        if any(not Path(x["model_path"]).is_file() for x in models):
            raise RuntimeError(f"Missing prefix model: {spec.key()}")
        index[spec.key()] = {"summary_path":str(p), "summary_sha256":sha(p), "gpu":str(row["gpu"]),
                           "accepted_row":r, "model_hashes":{x["model_path"]:x["model_sha256"] for x in models}}
    if set(index) != {s.key() for s in matrix()}:
        raise RuntimeError("Prefix matrix incomplete or duplicated")
    dl = root/"repo_experiments"/DL_REPORT
    marker = c.read_json(dl/"plot_manifest.json")
    if not marker["complete"]:
        raise RuntimeError("DL sweep source incomplete")
    refs[str(dl/"plot_manifest.json")] = sha(dl/"plot_manifest.json")
    for name,digest in marker["output_sha256"].items():
        if sha(dl/name) != digest:
            raise RuntimeError("DL source hash mismatch")
        refs[str(dl/name)] = digest
    bounds = {str(r):b.compute_bound(r) for r in (8,4,2,1)}
    if bounds != parent["bounds"]:
        raise RuntimeError("Bound geometry changed")
    m = {k:parent[k] for k in ("cases","settings","pair_file_sha256","sitreg_root","container_image","container_image_id")}
    m.update(variant=VARIANT, protocol=PROTOCOL, prefixes=index, bounds=bounds,
             prefix_report=str(source), dl_report=str(dl), reference_hashes=refs,
             source_hashes=sources(parent["sitreg_root"]),
             source_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=c.REPO_ROOT,text=True).strip(),
             source_dirty=subprocess.check_output(["git","status","--short"],cwd=c.REPO_ROOT,text=True),
             created_at=c.now_text(), uid=os.getuid(), gid=os.getgid())
    out.mkdir(parents=True,exist_ok=True); atomic_json(path,m)
    return m


def load_prefix(m, spec, verify_files=True):
    entry = m["prefixes"][spec.key()]
    p = Path(entry["summary_path"])
    if sha(p) != entry["summary_sha256"] or not (p.parent/"DONE").is_file():
        raise RuntimeError("Prefix summary/completion changed")
    row = c.read_json(p)
    if verify_files:
        for path,digest in {**entry["model_hashes"],row["dvf_path"]:row["dvf_sha256"]}.items():
            if sha(path) != digest:
                raise RuntimeError(f"Prefix artifact changed: {path}")
        arr = np.load(row["dvf_path"],mmap_mode="r")
        if arr.shape != (160,224,192,3) or arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise RuntimeError("Invalid prefix field")
    return row


def validate(out, m, spec):
    out = Path(out)
    if not (out/"DONE").is_file() or not (out/"ACCEPTED.json").is_file():
        raise RuntimeError("Missing worker/audit completion")
    row, receipt = c.read_json(out/"summary.json"), c.read_json(out/"ACCEPTED.json")
    if row["spec"] != asdict(spec) or row["protocol"] != PROTOCOL or row["source_hashes"] != m["source_hashes"]:
        raise RuntimeError("Wrong accepted cell")
    case = m["cases"][str(spec.case_rank)]
    if any(row[k] != case[k] for k in ("case_rank","fixed_id","moving_id","input_hashes")):
        raise RuntimeError("Case mismatch")
    if row["direction"] != "moving_to_fixed" or row["stage_alpha"] != [spec.alpha]*4:
        raise RuntimeError("Direction/weights mismatch")
    if row["prefix_summary_sha256"] != m["prefixes"][spec.key()]["summary_sha256"]:
        raise RuntimeError("Wrong matched prefix")
    if not receipt["passed"] or any(not math.isfinite(row[k]) for k in METRICS):
        raise RuntimeError("Invalid completion")
    for path,digest in receipt["files"].items():
        if sha(path) != digest:
            raise RuntimeError(f"Accepted output changed: {path}")
    return row
