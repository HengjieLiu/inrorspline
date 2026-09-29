"""Final acceptance audit; no CUDA initialization. Run after report regeneration."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .report import dump


def validate(root, review, scan_fields=False):
    rows = json.loads((root/"primary_rows.json").read_text())
    decisions = json.loads((review/"decisions.json").read_text())
    evaluated = {r["id"]:r for r in json.loads((review/"evaluations.json").read_text())}
    checks = []
    def check(name, ok, details=None):
        checks.append(dict(name=name,passed=bool(ok),details=details))
    check("unique_primary_ids",len({r["id"] for r in rows}) == len(rows))
    check("all_primary_clean",all(r["clean"] and r["monitor"]["clean"] for r in rows))
    counts = Counter(r["block"].split("_",1)[0] for r in rows)
    check("BE_first10_both_modes_40",counts["characterize"] == 40,counts["characterize"])
    check("five_isolated_pilots_180",counts["pilot"] == 180,counts["pilot"])
    check("confirmation_200",counts["confirm"] == 200,counts["confirm"])
    check("cold_6",counts["cold"] == 6,counts["cold"])
    check("reference_AA_6",sum(r["block"] == "reference_AA" for r in rows) == 6)
    check("reference_first10_11",sum(r["block"] == "reference_first10" for r in rows) == 11)
    need_eval = [r for r in rows if not r["block"].startswith(("smoke_","reference_diagnostic","reference_profile"))]
    check("all_requested_evaluations",all(r["id"] in evaluated for r in need_eval),
          dict(expected=len(need_eval),actual=len(evaluated)))
    failures = []
    engine_hashes = {p:set() for p in ("scripts/experiments/oasis_alpha_sweep/run_one.py",
                    "scripts/experiments/oasis_alpha_sweep/common.py",
                    "scripts/experiments/mrdbscp842_acceleration/backends.py")}
    for row in rows:
        timing = json.loads(Path(row["timing"]).read_text())
        marks = {m["name"]:m["offset_sec"] for m in timing["marks"]}
        summary = json.loads(Path(row["summary"]).read_text())
        ok = (timing["cudnn_requested"] == dict(benchmark=False,deterministic=True)
              and timing["cuda_backend"]["cudnn_benchmark"] is False
              and timing["cuda_backend"]["cudnn_deterministic"] is True
              and abs(row["elapsed_sec"]-(marks["dvf_file_ready"]-marks["case_load_complete"])) < 1e-9
              and summary["dense_output_units"] == "normalized_grid_xyz_displacement"
              and summary["grid_order"] == "pytorch"
              and row["dvf_valid"] and summary["stage_actual_epochs"] == row["steps"])
        field = np.load(row["dvf"],mmap_mode="r",allow_pickle=False)
        ok &= field.shape == (160,224,192,3) and field.dtype == np.float32
        if scan_fields:
            ok &= bool(np.isfinite(field).all())
        if not ok:
            failures.append(row["id"])
        if not row["block"].startswith("smoke_"):
            worker = root/"workers"/row["worker"]
            if not (worker/"host_ack.json").exists() or not (worker/"source_sha256.json").exists():
                failures.append(row["id"]+": missing PID/source provenance")
            else:
                hashes = json.loads((worker/"source_sha256.json").read_text())
                for p,values in engine_hashes.items():
                    values.add(hashes[p])
    check("timing_flags_DVF_and_provenance",not failures,failures)
    check("paired_training_engine_hashes_constant",all(len(v)==1 for v in engine_hashes.values()),
          {p:sorted(v) for p,v in engine_hashes.items()})
    be = [r for r in rows if r["block"].startswith("characterize_cpbe_")]
    check("BE_HD95_all_40",all("surface_metrics" in evaluated.get(r["id"],{}) for r in be))
    check("BE_HD95_missing_labels_valid",all(evaluated.get(r["id"],{}).get("hd95_nan_pattern_matches_dice",False) for r in be))
    check("BE_numerical_warning_retained",decisions["numerical_gates"]["cpbe"] is False)
    check("real_control_snapshots_15",len(list((root/"jobs").glob("reference_diagnostic_*/controls/*.npz"))) == 15)
    check("real_control_checks_present",(root/"real_control_checks.json").exists())
    check("source_provenance_present",bool(list(root.glob("provenance/*/manifest.json"))))
    check("data_manifest_present",(root/"data_environment.json").exists())
    source_check = root/"final_source_checks.json"
    source_result = json.loads(source_check.read_text()) if source_check.exists() else {}
    check("final_source_and_contract_checks",source_result.get("complete",False)
          and source_result.get("tests_run",0) >= 9 and not source_result.get("cuda_initialized",True))
    check("field_region_diagnostics_20",(review/"field_region_errors.json").exists()
          and len(json.loads((review/"field_region_errors.json").read_text())["rows"]) == 20)
    result = dict(complete=all(c["passed"] for c in checks),checks=checks,
                  primary_count=len(rows),evaluation_count=len(evaluated),scan_fields=scan_fields,
                  note="Completion is separate from numerical/quality/speed promotion; negative candidates are valid results.")
    dump(review/"acceptance.json",result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--review",required=True,type=Path)
    p.add_argument("--scan-fields",action="store_true")
    a = p.parse_args()
    result = validate(a.root,a.review,a.scan_fields)
    print(json.dumps(result,indent=2))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
