"""Shared full-box posthoc evaluation for both P83 variants and saved SITReg."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.experiments.oasis_alpha_sweep import common as canonical
from scripts.experiments.oasis_polyaffine_fieldfit import metrics as fidelity
from .data import atomic_json, external, false_to_mm, false_to_true, load_inputs, mm_to_false, sha256


def clean(value):
    if isinstance(value, dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def image_statistics(fixed, warped, mask):
    result = {}
    for name, domain in (("full",np.ones_like(mask)),("foreground",mask)):
        a, b = np.asarray(fixed[domain], np.float64), np.asarray(warped[domain], np.float64)
        delta = a-b
        ac, bc = a-a.mean(), b-b.mean()
        denominator = np.sqrt(np.mean(ac*ac)*np.mean(bc*bc))
        result[name] = dict(count=len(a), mse=float(np.mean(delta*delta)), mae=float(np.mean(np.abs(delta))),
                            ncc=float(np.mean(ac*bc)/(denominator+1e-10)))
    return result


@torch.no_grad()
def evaluate(field_false, inputs, device="cuda"):
    started = time.monotonic()
    shape = tuple(inputs["mask"].shape)
    field_false = torch.as_tensor(np.asarray(field_false), dtype=torch.float32, device=device)
    if tuple(field_false.shape) != (*shape,3) or not torch.isfinite(field_false).all():
        raise ValueError("Invalid full-box field")
    displacement = false_to_true(field_false, shape)
    grid = canonical.make_coordinate_grid(shape, torch.device(device))
    batch = {key:torch.as_tensor(inputs[name], dtype=torch.float32, device=device)
             for key,name in (("fixed","fixed_image"),("moving","moving_image"),
                              ("fixed_seg","fixed_seg"),("moving_seg","moving_seg"))}
    metrics = canonical.evaluate_from_dense_coords(grid+displacement, batch)
    warped_image = canonical.warp_full(batch["moving"], grid+displacement).cpu().numpy()
    warped_seg = metrics.pop("warped_seg").cpu().numpy().astype(np.int16)
    metrics.pop("dense_coords")
    mask = inputs["mask"].astype(bool)
    regularity, be, diffusion = canonical.regularity_metrics(displacement, torch.as_tensor(mask,device=device))
    voxels = canonical.dhw3_norm_to_zyx_voxel_ncdhw(displacement)
    determinants = canonical.jacobian_determinant_zyx(voxels.astype(np.float64))
    cropmask = mask[2:-2,2:-2,2:-2]
    regularity["jdet_distribution"] = {
        name:dict(zip(("min","p01","median","p99","max"),map(float,np.percentile(values,[0,1,50,99,100]))))
        for name,values in (("full",determinants.ravel()),("foreground",determinants[cropmask]))}
    field_mm = false_to_mm(field_false, shape, inputs["affine"]).cpu().numpy()
    regions = dict(full=np.ones_like(mask),foreground=mask,complement=~mask)
    errors = fidelity.field_errors(field_mm, inputs["field_mm"], regions)
    for name, region in regions.items():
        magnitude = np.linalg.norm((field_mm-inputs["field_mm"])[region].astype(np.float64),axis=-1)
        errors[name]["median_epe_mm"] = float(np.median(magnitude)) if len(magnitude) else None
    _, derivative_regions, _ = fidelity.region_masks(mask, inputs["affine"])
    geometry_error = fidelity.geometry_metrics(field_mm,inputs["field_mm"],inputs["affine"],derivative_regions)
    reference_false = mm_to_false(torch.as_tensor(inputs["field_mm"],device=device),shape,inputs["affine"])
    reference_warp = canonical.warp_full(batch["moving"],grid+false_to_true(reference_false,shape)).cpu().numpy()
    source_warp_error = image_statistics(reference_warp,warped_image,mask)
    label_rows = []
    for i,label in enumerate(range(1,36)):
        counts = dict(fixed=int((inputs["fixed_seg"]==label).sum()),
                      moving=int((inputs["moving_seg"]==label).sum()),warped=int((warped_seg==label).sum()))
        missing = [key for key,value in counts.items() if not value]
        values = {key:metrics[key][i] for key in ("dice","hd95","asd_symmetric","asd_fixed_to_warped","asd_warped_to_fixed")}
        if not missing and not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError(f"Unexpected nonfinite metrics for present label {label}")
        label_rows.append(dict(label=label,voxel_counts=counts,eligible=not missing,
                               exclusion_reason="absent_in_"+"_and_".join(missing) if missing else None,**values))
    return clean(dict(accuracy=metrics,per_label=label_rows,
        eligible_labels=sum(row["eligible"] for row in label_rows),
        image_alignment=image_statistics(inputs["fixed_image"],warped_image,mask),
        source_warp_error=source_warp_error,regularity=regularity,bending_energy_rows=be,diffusion_rows=diffusion,
        field_error=errors,jacobian_fidelity=geometry_error,
        metric_seconds=time.monotonic()-started,
        conventions=dict(field="full_box_model_evaluation_no_zero_fill",warp="canonical_align_corners_true_after_conversion; zero_padding",
                         labels="35; absent_fixed_moving_or_warped_excluded_with_counts",spacing_mm=[1.,1.,1.],
                         jacobian="LUMIR_crop2; J<=0_percent; positive_J_population_SDlogJ_and_coverage",
                         fidelity_jacobian="separate_P81_crop1_eroded_FG; not_primary_regularity",
                         diffusion="voxel_forward_valid_mean_components_mean_axes; both_mask_edge_endpoints",
                         bending="voxel_forward_valid_second_differences_mean_components_mixed_terms_weight2",
                         ndv="canonical_ten_direction_volume_fraction_crop1"))), warped_image, warped_seg


def evaluate_path(field, config, out, device="cuda"):
    out = external(out)
    out.mkdir(parents=True,exist_ok=False)
    inputs = load_inputs(config)
    if field == "identity":
        prediction = np.zeros_like(inputs["field_mm"])
        receipt = dict(kind="identity")
    elif field == "sitreg":
        prediction = mm_to_false(torch.from_numpy(inputs["field_mm"]),inputs["mask"].shape,inputs["affine"]).numpy()
        receipt = dict(kind="sitreg",sha256=config["canonical_sha256"])
    else:
        prediction = np.load(field)
        receipt = dict(kind="candidate",path=str(field),sha256=sha256(field))
    result, image, labels = evaluate(prediction,inputs,device)
    result["input_receipts"], result["field_receipt"] = inputs["receipts"],receipt
    np.save(out/"warped_image.npy",image)
    np.save(out/"warped_seg.npy",labels)
    atomic_json(out/"metrics.json",result)
    print(json.dumps(dict(out=str(out),accuracy={k:result["accuracy"][k] for k in
                     ("dice_mean","hd95_mean","asd_symmetric_mean")},
                     folding_fg=result["regularity"]["jdet_le0_pct_fg"],
                     sdlogj_fg=result["regularity"]["log_jdet_std_fg"])),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--field",required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    evaluate_path(args.field,json.loads(args.config.read_text()),args.out)


if __name__ == "__main__":
    main()
