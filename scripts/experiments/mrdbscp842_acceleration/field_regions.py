"""CPU-only localization of BE/reference DVF differences; raw fixed image > 0.

Diagnostic only: does not replace the preregistered full-domain equivalence gate.
Run in Docker; no Torch import or CUDA context is required.
"""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from .report import dump, paired_references


def analyze(root, output):
    rows = json.loads((root/"primary_rows.json").read_text())
    pairs = paired_references(rows)
    environment = json.loads((root/"data_environment.json").read_text())
    cases = {c["rank"]:c for c in environment["cases"]}
    masks = {}
    results = []
    for row in rows:
        if not row["block"].startswith("characterize_cpbe_") or row["variant"] != "cpbe":
            continue
        rank = row["rank"]
        if rank not in masks:
            case = cases[rank]
            source = Path(case["sorted_pair_file"]).parent/"test"/f"img{case['fixed_id']}.nii.gz"
            raw = nib.load(str(source)).get_fdata(dtype=np.float32)
            masks[rank] = np.ascontiguousarray(raw.transpose(0,2,1)[::-1,:,::-1] > 0)
        mask = masks[rank]
        reference = pairs[row["id"]]
        a = np.load(row["dvf"],allow_pickle=False)
        b = np.load(reference["dvf"],mmap_mode="r",allow_pickle=False)
        delta = (a.astype(np.float64)-b)*np.array([191/2,223/2,159/2])
        squared = np.sum(delta*delta,axis=-1)
        index = tuple(map(int,np.unravel_index(np.argmax(squared),squared.shape)))
        result = dict(id=row["id"],reference_id=reference["id"],rank=rank,mode=row["mode"],
                      max_location_zyx=index,max_is_fixed_foreground=bool(mask[index]))
        for name,domain in (("foreground",mask),("background",~mask)):
            values = np.sqrt(squared[domain])
            result[name] = dict(voxels=int(values.size),vector_rms_voxel=float(np.sqrt(np.mean(values*values))),
                                mean_vector_error_voxel=float(np.mean(values)),max_vector_error_voxel=float(values.max()),
                                p50_vector_error_voxel=float(np.quantile(values,.5)),
                                p95_vector_error_voxel=float(np.quantile(values,.95)),
                                p99_vector_error_voxel=float(np.quantile(values,.99)))
        results.append(result)
    payload = dict(foreground_rule="reoriented raw fixed image > 0; no segmentation mask",rows=results,
                   interpretation="Localization only; original full-volume gates remain unchanged")
    dump(output,payload)
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    print(json.dumps(analyze(a.root,a.output),indent=2))


if __name__ == "__main__":
    main()
