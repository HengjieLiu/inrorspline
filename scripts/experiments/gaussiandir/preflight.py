"""Historical metric replay and independent native/canonical warp agreement."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.oasis_alpha_sweep import common
from .data import atomic_json, flat_coordinates, load_inputs, mm_to_false, sample_volume, sha256


def run(config, references):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    inputs=load_inputs(config)
    target=inputs["target"]
    historical_path=Path(target["source_config_path"])
    if sha256(historical_path)!=target["source_report_sha256"]:
        raise ValueError("Historical report changed")
    rows=[row for row in json.loads(historical_path.read_text())["case_metrics"]
          if row.get("case_rank")==0 and row.get("representation")=="SITReg"]
    if len(rows)!=1 or rows[0]["source_sha256"]!=config["source_sha256"]:
        raise ValueError("Historical source identity mismatch")
    history=rows[0]
    replay=json.loads((references/"sitreg/metrics.json").read_text())
    comparisons={}
    for old,new in (("dice","dice_mean"),("hd95","hd95_mean"),("asd","asd_symmetric_mean")):
        comparisons[old]=abs(history[old]-replay["accuracy"][new])
    for key in ("jdet_le0_pct_full","jdet_le0_pct_fg","log_jdet_std_full","log_jdet_std_fg"):
        comparisons[key]=abs(history[key]-replay["regularity"][key])
    if max(comparisons.values())>1e-7:
        raise ValueError(f"Historical metric replay failed: {comparisons}")
    shape=inputs["mask"].shape
    grid=common.make_coordinate_grid(shape,torch.device("cuda"))
    # Direct source conversion avoids the native False -> True round trip.
    original=torch.as_tensor(inputs["field_mm"],device="cuda").flip(-1)
    true=original*2/original.new_tensor(np.array(shape[::-1])-1)
    warps={}
    for key,flow in (("identity",torch.zeros_like(original)),
                     ("sitreg",mm_to_false(torch.as_tensor(inputs["field_mm"],device="cuda"),shape,inputs["affine"]))):
        native_image=np.empty(shape,np.float32)
        native_labels=np.empty(shape,np.float32)
        with torch.no_grad():
            moving=torch.as_tensor(inputs["moving_image"],device="cuda")
            seg=torch.as_tensor(inputs["moving_seg"],device="cuda",dtype=torch.float32)
            for start in range(0,int(np.prod(shape)),20000):
                ids=torch.arange(start,min(start+20000,int(np.prod(shape))),device="cuda")
                query=flat_coordinates(ids,shape)+flow.reshape(-1,3)[ids]
                native_image.reshape(-1)[start:start+len(ids)]=sample_volume(moving,query).cpu().numpy()
                native_labels.reshape(-1)[start:start+len(ids)]=sample_volume(seg,query,mode="nearest").cpu().numpy()
            direct=grid+(true if key=="sitreg" else 0)
            direct_image=common.warp_full(moving,direct).cpu().numpy()
            direct_labels=common.warp_full(seg,direct,mode="nearest").cpu().numpy()
        exported_image=np.load(references/key/"warped_image.npy")
        exported_labels=np.load(references/key/"warped_seg.npy")
        scale=max(float(np.ptp(inputs["moving_image"])),1e-10)
        warps[key]=dict(native_vs_direct_image_range_rmse=float(np.sqrt(np.mean((native_image-direct_image)**2))/scale),
                        native_vs_direct_label_mismatch_fraction=float(np.mean(native_labels!=direct_labels)),
                        export_vs_direct_image_range_rmse=float(np.sqrt(np.mean((exported_image-direct_image)**2))/scale),
                        export_vs_direct_label_mismatch_fraction=float(np.mean(exported_labels!=direct_labels)))
        if warps[key]["export_vs_direct_label_mismatch_fraction"]>1e-6 or warps[key]["export_vs_direct_image_range_rmse"]>1e-6:
            raise ValueError(f"Canonical exported warp mismatch: {warps[key]}")
        if warps[key]["native_vs_direct_label_mismatch_fraction"]>1e-5 or warps[key]["native_vs_direct_image_range_rmse"]>1e-5:
            raise ValueError(f"Native/canonical warp mismatch: {warps[key]}")
    receipt=dict(status="passed",historical_metric_absolute_differences=comparisons,warps=warps,
                 input_receipts=inputs["receipts"],historical_report_sha256=sha256(historical_path))
    atomic_json(references/"preflight.json",receipt)
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--references",type=Path,required=True)
    args=parser.parse_args()
    run(json.loads(args.config.read_text()),args.references)
