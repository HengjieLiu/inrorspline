#!/usr/bin/env python
"""Independent saved-data/PDF verification for the F/G presentation extension."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with path.open() as f: return list(csv.DictReader(f))


def verify(root):
    read = lambda name: json.loads((root / name).read_text())
    manifest, rendered, result = read("manifest.json"), read("RENDERED.json"), read("results.json")
    for name, digest in rendered["output_sha256"].items():
        assert sha(root / name) == digest, name
    for name, digest in manifest["source_sha256"].items():
        assert sha(name) == digest, name
    cases, summary = csv_rows(root / "per_case.csv"), csv_rows(root / "summary.csv")
    original = csv_rows(Path(manifest["source"]) / "per_case.csv")
    source_index = {(r["panel"], r["series"], float(r["alpha"]), int(r["case_rank"])): r for r in original}
    assert len(cases) == len(result["per_case"]) == 600
    assert len(summary) == len(result["summary"]) == 60
    metrics = ("dice", "hd95", "asd", "log_jdet_std_fg", "jdet_le0_pct_fg")
    key = lambda r: (r["panel"], r["series"], float(r["alpha"]))
    group_keys, source_keys = set(), set()
    for record in summary:
        assert key(record) not in group_keys
        group_keys.add(key(record))
        selected = [r for r in cases if key(r) == key(record)]
        assert sorted(int(r["case_rank"]) for r in selected) == list(range(10))
        assert int(record["n"]) == 10
        for metric in metrics:
            values = np.array([float(r[metric]) for r in selected])
            assert np.isfinite(values).all()
            assert abs(values.mean() - float(record[metric+"_mean"])) < 1e-12
            assert abs(values.std(ddof=1) - float(record[metric+"_sd"])) < 1e-12
    for record, json_record in zip(cases, result["per_case"]):
        source_key = (record["source_panel"], record["source_series"], float(record["alpha"]), int(record["case_rank"]))
        source_keys.add(source_key)
        reference = source_index[source_key]
        assert [record["fixed_id"], record["moving_id"]] == manifest["pairs"][int(record["case_rank"])]
        assert record["direction"] == "moving_to_fixed"
        assert key(record) == key(json_record)
        for metric in metrics:
            assert float(record[metric]) == float(reference[metric]) == json_record[metric]
    assert len(source_keys) == 450
    for panel in "FG":
        assert sum(r["panel"] == panel for r in summary) == 30
        for series in ("sitreg", "vfa", "fedda", "displacement", "bounded2", "svf"):
            weights = sorted(float(r["alpha"]) for r in summary if r["panel"] == panel and r["series"] == series)
            expected = [.1,1/3,1.,3.,10.] if series in ("sitreg","vfa","fedda") else [1.] if series == "bounded2" else [.01,.1,.3,1.,3.,10.,100.]
            assert weights == expected
    assert len(manifest["rendered_axes"]) == 12
    for axis in manifest["rendered_axes"]:
        for stem, spec in (("x", manifest["axes"]["x"]), ("y", manifest["axes"][axis["metric"]])):
            assert axis[stem+"lim"] == spec["limits"]
            assert axis[stem+"ticks"] == spec["ticks"]
            assert axis[stem+"label"] == spec["label"]
        assert axis["point_count"] == 30
        assert axis["ellipse_count"] == (30 if axis["row"] else 0)
    for panel in "FG":
        png = (root / f"panel_{panel}.png").read_bytes()
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (4000,2200)
        xml = ET.parse(root / f"panel_{panel}.svg")
        assert sum(bool(re.fullmatch(r"axes_\d+", e.get("id", ""))) for e in xml.iter()) == 6
    for filename, count in (("panel_F.pdf",1), ("panel_G.pdf",1), ("arc_panels_F_to_G.pdf",2), ("arc_panels_A_to_G.pdf",7)):
        pdf = (root / filename).read_bytes()
        assert len(re.findall(rb"/Type\s*/Page\b", pdf)) == count, filename
        assert b"/FontFile2" in pdf
    return dict(complete=True, plotted_settings=60, displayed_case_rows=600,
                unique_source_case_rows=450, independently_recomputed_statistics=600,
                axes_checked=12, expected_pdf_pages=[1,1,2,7],
                original_figures_and_lambda1_table_unchanged=True,
                exact_source_and_csv_json_metric_agreement=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--confirm-visual-review", action="store_true")
    args = parser.parse_args()
    validation = verify(args.root)
    if args.confirm_visual_review:
        validation.update(visual_review_complete=True,
                          visual_review="F/G PNGs inspected: six distinct methods, readable labels/legends, correct marker-only bounded point; no clipping.",
                          pdf_merge="Host /usr/bin/pdfunite: original A–E PDF plus new F–G PDF; original pages not rerendered.",
                          accepted_at=datetime.now(timezone.utc).isoformat(), verifier_sha256=sha(__file__),
                          output_sha256={p.name: sha(p) for p in sorted(args.root.iterdir()) if p.is_file()})
        with (args.root / "ACCEPTED.json").open("x") as f:
            json.dump(validation, f, indent=2); f.write("\n")
    print(json.dumps({k:v for k,v in validation.items() if k != "output_sha256"}, indent=2))
