"""P81 serialized physical-field fidelity, geometry and resampling metrics.

This is a common posthoc evaluator, never a model/label selection objective.
All image and label warps use explicit zero padding, align_corners=True.
Central physical derivatives share a fixed interior/eroded-mask domain.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

try:
    from surface_distance import metrics as surface_distance_metrics
except ImportError:
    surface_distance_metrics = None


LABELS = tuple(range(1, 36))


def _array(value):
    return np.load(value, mmap_mode="r") if isinstance(value, (str, Path)) else np.asarray(value)


def _finite_chunks(array, depth):
    return all(np.isfinite(array[start:start + depth]).all() for start in range(0, len(array), depth))


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _quantiles(values):
    if not len(values):
        return dict(count=0, minimum=None, p01=None, median=None, p99=None, maximum=None)
    p = np.percentile(values, [0, 1, 50, 99, 100])
    return dict(count=len(values), **dict(zip(("minimum", "p01", "median", "p99", "maximum"), map(float, p))))


def region_masks(mask, affine, ring_mm=10.):
    mask = np.asarray(mask, dtype=bool)
    linear = np.asarray(affine, dtype=np.float64)[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    gram = linear.T @ linear
    if not np.allclose(gram, np.diag(spacing ** 2), atol=1e-8, rtol=1e-8):
        raise ValueError("P81 ring and surface metrics require an orthogonal image grid; sheared geometry is unsupported")
    if min(mask.shape) < 3 or not mask.any():
        raise ValueError("Require a nonempty foreground in a 3D grid of minimum dimension3")
    ring = (~mask) & (ndi.distance_transform_edt(~mask, sampling=spacing) <= ring_mm)
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1, 1:-1] = True
    structure = ndi.generate_binary_structure(3, 1)
    return ({"foreground": mask, "full": np.ones_like(mask), "ring_10mm": ring},
            {"foreground_eroded": ndi.binary_erosion(mask, structure=structure, border_value=0) & interior,
             "full_interior": interior,
             "ring_eroded": ndi.binary_erosion(ring, structure=structure, border_value=0) & interior},
            spacing)


def field_errors(prediction, reference, regions, chunk_depth=8):
    shape = reference.shape[:3]
    epe = np.empty(shape, dtype=np.float64)
    target_power = np.empty(shape, dtype=np.float64)
    for start in range(0, shape[0], chunk_depth):
        stop = min(start + chunk_depth, shape[0])
        target = np.asarray(reference[start:stop], dtype=np.float64)
        error = np.asarray(prediction[start:stop], dtype=np.float64) - target
        epe[start:stop] = np.sqrt(np.einsum("...i,...i->...", error, error))
        target_power[start:stop] = np.einsum("...i,...i->...", target, target)
    output = {}
    for name, mask in regions.items():
        error = epe[mask]
        if not error.size:
            output[name] = dict(count=0, mean_epe_mm=None, rmse_mm=None,
                                p95_epe_mm=None, max_epe_mm=None, relative_rmse_pct=None,
                                relative_rmse_status="empty_domain")
            continue
        power = float(target_power[mask].mean())
        mse = float(np.mean(error ** 2))
        output[name] = dict(count=int(error.size), mean_epe_mm=float(error.mean()),
                            rmse_mm=float(np.sqrt(mse)), p95_epe_mm=float(np.percentile(error, 95)),
                            max_epe_mm=float(error.max()), target_displacement_rms_mm=float(np.sqrt(power)),
                            relative_rmse_pct=100 * float(np.sqrt(mse / power)) if power > 1e-12 else None,
                            relative_rmse_status="defined" if power > 1e-12 else "target_rms_below_1e-6_mm")
    return output


def _jacobian_chunk(field, start, stop, inverse_linear):
    """Dphi=I+Du at voxel centers strictly inside the one-cell boundary."""
    derivatives = (
        (np.asarray(field[start + 1:stop + 1, 1:-1, 1:-1], dtype=np.float64)
         - np.asarray(field[start - 1:stop - 1, 1:-1, 1:-1], dtype=np.float64)) / 2,
        (np.asarray(field[start:stop, 2:, 1:-1], dtype=np.float64)
         - np.asarray(field[start:stop, :-2, 1:-1], dtype=np.float64)) / 2,
        (np.asarray(field[start:stop, 1:-1, 2:], dtype=np.float64)
         - np.asarray(field[start:stop, 1:-1, :-2], dtype=np.float64)) / 2,
    )
    jacobian = np.stack(derivatives, -1) @ inverse_linear
    jacobian += np.eye(3, dtype=np.float64)
    return jacobian


def geometry_metrics(prediction, reference, affine, regions, chunk_depth=8):
    shape = reference.shape[:3]
    inverse_linear = np.linalg.inv(np.asarray(affine, dtype=np.float64)[:3, :3])
    # Only scalar volumes persist; two N*3*3 Jacobian slabs are discarded each step.
    fit_det = np.full(shape, np.nan, dtype=np.float64)
    reference_det = np.full(shape, np.nan, dtype=np.float64)
    jacobian_error_sq = np.full(shape, np.nan, dtype=np.float64)
    for start in range(1, shape[0] - 1, chunk_depth):
        stop = min(start + chunk_depth, shape[0] - 1)
        fit = _jacobian_chunk(prediction, start, stop, inverse_linear)
        target = _jacobian_chunk(reference, start, stop, inverse_linear)
        fit_det[start:stop, 1:-1, 1:-1] = np.linalg.det(fit)
        reference_det[start:stop, 1:-1, 1:-1] = np.linalg.det(target)
        difference = fit - target
        jacobian_error_sq[start:stop, 1:-1, 1:-1] = np.einsum("...ij,...ij->...", difference, difference)
    result = {}
    for name, mask in regions.items():
        fit, target = fit_det[mask], reference_det[mask]
        if not len(fit):
            result[name] = dict(count=0, status="empty_derivative_domain")
            continue
        fit_positive, target_positive = fit > 0, target > 0
        common_positive = fit_positive & target_positive
        fit_folds, target_folds = ~fit_positive, ~target_positive
        union_folds = fit_folds | target_folds
        row = dict(count=int(len(fit)), status="evaluated", reference={}, prediction={},
                   jacobian_matrix_rmse=float(np.sqrt(jacobian_error_sq[mask].mean())),
                   determinant_mae=float(np.abs(fit - target).mean()),
                   determinant_rmse=float(np.sqrt(np.mean((fit - target) ** 2))),
                   common_positive_count=int(common_positive.sum()),
                   common_positive_fraction=float(common_positive.mean()),
                   log_jacobian_rmse_common_positive=(float(np.sqrt(np.mean(
                       (np.log(fit[common_positive]) - np.log(target[common_positive])) ** 2)))
                       if common_positive.any() else None),
                   newly_folded_fraction=float((fit_folds & ~target_folds).mean()),
                   removed_reference_fold_fraction=float((target_folds & ~fit_folds).mean()),
                   fold_location_agreement_fraction=float((fit_folds == target_folds).mean()),
                   fold_location_jaccard=(float((fit_folds & target_folds).sum() / union_folds.sum())
                                          if union_folds.any() else None))
        for label, values, positive in (("reference", target, target_positive), ("prediction", fit, fit_positive)):
            row[label] = dict(**_quantiles(values), folding_fraction=float((~positive).mean()),
                              folding_percent=float((~positive).mean() * 100),
                              positive_fraction=float(positive.mean()),
                              std_log_jacobian_positive=float(np.std(np.log(values[positive]))) if positive.any() else None)
        result[name] = row
    return result


def warp_volume(volume, displacement_mm, fixed_affine, moving_affine=None, *,
                nearest=False, chunk_depth=8):
    """Full pull warp, preserving PyTorch nearest ties and partial zero padding."""
    volume = np.asarray(volume)
    shape = displacement_mm.shape[:3]
    h_f = np.asarray(fixed_affine, dtype=np.float64)
    h_m = h_f if moving_affine is None else np.asarray(moving_affine, dtype=np.float64)
    inverse = np.linalg.inv(h_m[:3, :3])
    source = torch.from_numpy(np.array(volume, dtype=np.float32, copy=True, order="C"))[None, None]
    warped = np.empty(shape, dtype=np.int16 if nearest else np.float32)
    in_bounds = np.empty(shape, dtype=bool)
    source_last = np.asarray(volume.shape, dtype=np.float64) - 1
    for start in range(0, shape[0], chunk_depth):
        stop = min(start + chunk_depth, shape[0])
        index = np.stack(np.meshgrid(np.arange(start, stop), np.arange(shape[1]),
                                    np.arange(shape[2]), indexing="ij"), -1)
        fixed_mm = index @ h_f[:3, :3].T + h_f[:3, 3]
        endpoint = fixed_mm + np.asarray(displacement_mm[start:stop], dtype=np.float64)
        moving_index = (endpoint - h_m[:3, 3]) @ inverse.T
        in_bounds[start:stop] = ((moving_index >= 0) & (moving_index <= source_last)).all(-1)
        normalized = (moving_index / source_last * 2 - 1)[..., ::-1]
        grid = torch.from_numpy(np.array(normalized, dtype=np.float32, copy=True, order="C"))[None]
        with torch.no_grad():
            values = F.grid_sample(source, grid, mode="nearest" if nearest else "bilinear",
                                   padding_mode="zeros", align_corners=True)[0, 0].numpy()
        warped[start:stop] = values
    return warped, in_bounds


def _image_metrics(fit, reference, fixed_validity, regions, intensity_range):
    result = {}
    for name, region in regions.items():
        domain = region & fixed_validity
        count = int(domain.sum())
        region_count = int(region.sum())
        if not count:
            result[name] = dict(count=0, reference_inbounds_coverage=0. if region_count else None,
                                mae=None, rmse=None, nrmse=None, normalized_mae=None)
            continue
        residual = np.asarray(fit[domain], np.float64) - reference[domain]
        mse = float(np.mean(residual ** 2))
        mae = float(np.abs(residual).mean())
        result[name] = dict(count=count, reference_inbounds_coverage=count / region_count,
                            mae=mae, rmse=float(np.sqrt(mse)),
                            nrmse=float(np.sqrt(mse) / intensity_range) if intensity_range > 1e-12 else None,
                            normalized_mae=mae / intensity_range if intensity_range > 1e-12 else None)
    return result


def _label_metric(fixed, predicted, moving, label, spacing, *, surfaces=True):
    true, pred, original = fixed == label, predicted == label, moving == label
    presence = dict(fixed=bool(true.any()), warped=bool(pred.any()), moving=bool(original.any()))
    base = dict(presence=presence, fixed_voxels=int(true.sum()), warped_voxels=int(pred.sum()),
                valid=all(presence.values()))
    if not base["valid"]:
        return dict(**base, dice=None, hd95_mm=None, asd_symmetric_mm=None,
                    asd_fixed_to_warped_mm=None, asd_warped_to_fixed_mm=None,
                    reason="label_absent_in_" + ",".join(key for key, value in presence.items() if not value))
    dice = float(2 * np.count_nonzero(true & pred) / (np.count_nonzero(true) + np.count_nonzero(pred) + 1e-5))
    if not surfaces:
        return dict(**base, dice=dice, hd95_mm=None, asd_symmetric_mm=None,
                    asd_fixed_to_warped_mm=None, asd_warped_to_fixed_mm=None, reason="surface_metrics_disabled")
    if surface_distance_metrics is None:
        raise RuntimeError("surface_distance is required for P81 HD95/ASD evaluation")
    if np.array_equal(true, pred):
        hd95 = symmetric = gt_to_pred = pred_to_gt = 0.
    else:
        distances = surface_distance_metrics.compute_surface_distances(mask_gt=true, mask_pred=pred,
                                                                        spacing_mm=tuple(map(float, spacing)))
        hd95 = float(surface_distance_metrics.compute_robust_hausdorff(distances, 95.))
        gt_to_pred, pred_to_gt = map(float, surface_distance_metrics.compute_average_surface_distance(distances))
        gt_area, pred_area = distances["surfel_areas_gt"], distances["surfel_areas_pred"]
        denominator = float(gt_area.sum() + pred_area.sum())
        symmetric = float((np.sum(distances["distances_gt_to_pred"] * gt_area)
                           + np.sum(distances["distances_pred_to_gt"] * pred_area)) / denominator)
        if not all(np.isfinite(v) for v in (hd95, symmetric, gt_to_pred, pred_to_gt)):
            raise ValueError(f"Nonfinite surface metric for present label{label}")
    return dict(**base, dice=dice, hd95_mm=hd95, asd_symmetric_mm=symmetric,
                asd_fixed_to_warped_mm=gt_to_pred, asd_warped_to_fixed_mm=pred_to_gt, reason=None)


def label_metrics(fit, reference, fixed, moving, spacing, *, surfaces=True,
                  reference_label_metrics=None):
    cached = {} if reference_label_metrics is None else {int(row["label"]): row["reference"]
                                                        for row in reference_label_metrics}
    rows = []
    for label in LABELS:
        target = cached.get(label) or _label_metric(fixed, reference, moving, label, spacing, surfaces=surfaces)
        predicted = _label_metric(fixed, fit, moving, label, spacing, surfaces=surfaces)
        fit_mask, ref_mask = fit == label, reference == label
        denominator = int(fit_mask.sum() + ref_mask.sum())
        agreement = float(2 * np.count_nonzero(fit_mask & ref_mask) / denominator) if denominator else None
        paired = target["valid"] and predicted["valid"]
        deltas = {metric: (predicted[metric] - target[metric]
                           if paired and predicted[metric] is not None and target[metric] is not None else None)
                  for metric in ("dice", "hd95_mm", "asd_symmetric_mm")}
        rows.append(dict(label=label, reference=target, prediction=predicted,
                         paired_valid=paired, warped_label_dice=agreement, delta_prediction_minus_reference=deltas))
    paired_rows = [r for r in rows if r["paired_valid"]]
    summary = {which: {metric: _mean(r[which][metric] for r in rows if r[which]["valid"])
                       for metric in ("dice", "hd95_mm", "asd_symmetric_mm")}
               for which in ("reference", "prediction")}
    summary["paired"] = {which: {metric: _mean(r[which][metric] for r in paired_rows)
                                 for metric in ("dice", "hd95_mm", "asd_symmetric_mm")}
                         for which in ("reference", "prediction")}
    summary["paired"]["delta_prediction_minus_reference"] = {
        metric: _mean(r["delta_prediction_minus_reference"][metric] for r in paired_rows)
        for metric in ("dice", "hd95_mm", "asd_symmetric_mm")}
    return dict(available=True, surface_metrics_enabled=surfaces, label_count=35,
                reference_valid_labels=sum(r["reference"]["valid"] for r in rows),
                prediction_valid_labels=sum(r["prediction"]["valid"] for r in rows),
                paired_valid_labels=len(paired_rows),
                excluded_reference_labels=[r["label"] for r in rows if not r["reference"]["valid"]],
                excluded_prediction_labels=[r["label"] for r in rows if not r["prediction"]["valid"]],
                disappeared_after_fit_labels=[r["label"] for r in rows
                    if r["reference"]["valid"] and not r["prediction"]["presence"]["warped"]],
                cohort_means_comparable=(all(r["reference"]["valid"] == r["prediction"]["valid"] for r in rows)),
                summary=summary, per_label=rows)


def evaluate_fields(prediction_mm, target, *, chunk_depth=8, surface_metrics=True,
                    reference_label_metrics=None):
    """Evaluate a reloaded mm displacement against data.load_target() output.

Optional *_image_path / *_seg_path keys can instead contain NumPy arrays.
The returned dictionary is strict-JSON compatible (undefined metrics are None).
"""
    started = time.monotonic()
    if chunk_depth < 1:
        raise ValueError("Positive derivative/warp slab depth required")
    prediction, reference = _array(prediction_mm), _array(target["field"])
    mask, affine = _array(target["mask"]).astype(bool), np.asarray(target["affine"], np.float64)
    if (prediction.ndim != 4 or prediction.shape[-1] != 3 or prediction.shape != reference.shape
            or prediction.shape[:3] != mask.shape or affine.shape != (4, 4)):
        raise ValueError("Prediction, reference, mask and affine geometry mismatch")
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("Invalid physical affine")
    if not _finite_chunks(prediction, chunk_depth) or not _finite_chunks(reference, chunk_depth):
        raise ValueError("Nonfinite field cannot be silently removed from metric domains")
    regions, derivative_regions, spacing = region_masks(mask, affine)
    result = dict(version=1, shape=list(mask.shape), spacing_mm=spacing.tolist(),
        field_error=field_errors(prediction, reference, regions, chunk_depth),
        geometry=geometry_metrics(prediction, reference, affine, derivative_regions, chunk_depth),
        conventions=dict(vector_units="physical_world_mm", endpoint_rmse="sqrt(mean(sum_3_components(error_squared)))",
                         derivative="centered_one_voxel_physical_Dphi_on_common_interior_and_cross_eroded_ROI",
                         derivative_boundary_voxels_excluded=1,
                         derivative_vs_historical="not_identical_to_legacy_two_voxel_boundary_trim",
                         folding="det_Dphi_le_zero; sampled_not_certificate", log_jacobian="positive_only_with_coverage",
                         ring="outside_raw_positive_foreground_within_10mm_EDT_on_orthogonal_grid",
                         image_interpolation="trilinear", label_interpolation="nearest",
                         padding_mode="zeros", align_corners=True,
                         image_domain="reference_inbounds_fixed_domain; predicted_outbounds_retained_with_zero_padding",
                         image_normalization="raw_moving_image_max_minus_min",
                         label_domain="full_fixed_grid", absent_label="exclude_if_fixed_moving_or_warped_absent; report_pairwise_common_labels",
                         asd="pooled_surfel_area_weighted_symmetric_distance", label_selection=False),
        warping=dict(available=False, reason="moving_image_not_provided"),
        labels=dict(available=False, reason="fixed_or_moving_segmentation_not_provided"))
    moving_image_value = target.get("moving_image_path", target.get("moving_image"))
    moving_seg_value = target.get("moving_seg_path", target.get("moving_seg"))
    fixed_seg_value = target.get("fixed_seg_path", target.get("fixed_seg"))
    moving_affine = target.get("moving_affine", affine)
    reference_validity = prediction_validity = None
    if moving_image_value is not None:
        moving_image = _array(moving_image_value)
        fit_image, prediction_validity = warp_volume(moving_image, prediction, affine, moving_affine,
                                                    chunk_depth=chunk_depth)
        ref_image, reference_validity = warp_volume(moving_image, reference, affine, moving_affine,
                                                    chunk_depth=chunk_depth)
        intensity_range = float(moving_image.max() - moving_image.min())
        out_of_bounds = {name: dict(reference_fraction=float((~reference_validity[domain]).mean()) if domain.any() else None,
                                    prediction_fraction=float((~prediction_validity[domain]).mean()) if domain.any() else None)
                         for name, domain in regions.items()}
        result["warping"] = dict(available=True, image_range=intensity_range,
                                  image=_image_metrics(fit_image, ref_image, reference_validity, regions, intensity_range),
                                  out_of_bounds=out_of_bounds)
        del fit_image, ref_image
    if moving_seg_value is not None and fixed_seg_value is not None:
        moving_seg, fixed_seg = _array(moving_seg_value), _array(fixed_seg_value)
        if fixed_seg.shape != mask.shape:
            raise ValueError("Fixed segmentation geometry differs from evaluated grid")
        fit_seg, fit_valid = warp_volume(moving_seg, prediction, affine, moving_affine,
                                         nearest=True, chunk_depth=chunk_depth)
        ref_seg, ref_valid = warp_volume(moving_seg, reference, affine, moving_affine,
                                         nearest=True, chunk_depth=chunk_depth)
        result["labels"] = label_metrics(fit_seg, ref_seg, fixed_seg, moving_seg, spacing,
                                          surfaces=surface_metrics, reference_label_metrics=reference_label_metrics)
        result["labels"]["warped_voxel_agreement"] = {
            name: dict(count=int((domain & ref_valid).sum()),
                       fraction=float((fit_seg[domain & ref_valid] == ref_seg[domain & ref_valid]).mean())
                                if (domain & ref_valid).any() else None)
            for name, domain in regions.items()}
        if not result["warping"]["available"]:
            result["warping"]["out_of_bounds"] = {
                name: dict(reference_fraction=float((~ref_valid[domain]).mean()) if domain.any() else None,
                           prediction_fraction=float((~fit_valid[domain]).mean()) if domain.any() else None)
                for name, domain in regions.items()}
    result["evaluation_seconds"] = time.monotonic() - started
    return result
