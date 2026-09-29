"""Opt-in schedule experiments and untimed quality evaluation.

Historical registration defaults remain untouched. Context-local Adam schedules
alter the optimizer's LR BEFORE each update and restore the original class.
"""
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import worker
from .backends import c, run_one as run
from .overnight_candidates import lr_multiplier


@contextmanager
def scheduled_optimizer(spec, smoke=False):
    original = torch.optim.Adam
    base_rates = list(map(float, spec["overrides"]["stage_lr"].split(",")))
    budgets = [1]*len(base_rates) if smoke else list(map(int,spec["overrides"]["stage_epochs"].split(",")))
    created_stages = []  # Integers only: do NOT retain prior-stage GPU optimizer states.
    class ScheduledAdam(original):
        def __init__(self, params, **kwargs):
            self.stage = len(created_stages)
            self.updates = 0
            self.initial_lr = kwargs["lr"]
            if spec["optimizer"] == "amsgrad":
                kwargs["amsgrad"] = True
            super().__init__(params, **kwargs)
            created_stages.append(self.stage)

        def step(self, closure=None):
            self.updates += 1
            for group in self.param_groups:
                group["lr"] = self.initial_lr * lr_multiplier(spec["lr_schedule"],self.updates,
                    budgets[self.stage],spec["lr_floor_ratio"])
            return super().step(closure)
    try:
        if spec["lr_schedule"] != "constant" or spec["optimizer"] != "adam":
            torch.optim.Adam = ScheduledAdam
        yield created_stages
    finally:
        torch.optim.Adam = original


def registration(command, backend, args):
    spec = command["candidate"]
    history = []
    metadata = run.write_dvf_only_metadata
    def capture_metadata(*a, **kw):
        # Called AFTER dvf_file_ready: retained history is not added to runtime.
        metrics = kw.get("metrics", a[2] if len(a)>2 else None)
        history.extend(dict(row) for row in metrics["history"])
        return metadata(*a, **kw)
    run.write_dvf_only_metadata = capture_metadata
    try:
        with scheduled_optimizer(spec, command.get("smoke",False)):
            row = worker.registration(dict(command,candidate_overrides=spec["overrides"]),backend,args)
    finally:
        run.write_dvf_only_metadata = metadata
    budgets = list(map(int,spec["overrides"]["stage_epochs"].split(",")))
    if command.get("smoke"):
        budgets = [1]*len(budgets)
    rates = list(map(float,spec["overrides"]["stage_lr"].split(",")))
    for h in history:
        stage, epoch = int(h["stage_idx"]), int(h["stage_epoch"])
        h["effective_lr"] = rates[stage]*lr_multiplier(spec["lr_schedule"],epoch,budgets[stage],spec["lr_floor_ratio"])
    path = Path(command["output"])/"history.json"
    worker.write_json(path, history)
    row.update(candidate=spec, history=str(path))
    return row


def evaluate(command,args):
    """One field per command, isolated from timing; fixed backend for all arms."""
    item = command["item"]
    started = time.perf_counter()
    device = c.set_cuda_device(str(args.gpu),cudnn_benchmark=False,cudnn_deterministic=True)
    field = np.load(item["dvf"])
    if field.dtype != np.float32 or field.shape != (*c.IMAGE_SHAPE,3) or not np.isfinite(field).all():
        raise ValueError("Invalid DVF")
    batch = c.load_oasis_case(rank=int(item["rank"]),device=device,oasis_root=Path(args.oasis_root),pair_file=Path(args.pair_file))
    dense = torch.from_numpy(field).to(device)
    with torch.no_grad():
        coords = c.make_coordinate_grid(c.IMAGE_SHAPE,device)+dense
        warped = c.warp_full(batch["moving"],coords)
        ncc = -float(c.ncc_loss(batch["fixed"][None,None],warped[None,None]).cpu())
        del warped
        labels = c.warp_full(batch["moving_seg"],coords,mode="nearest")
        dice = np.array(c.dice_per_label(labels,batch["fixed_seg"],batch["moving_seg"]))
        vox = c.dhw3_norm_to_zyx_voxel_ncdhw(field)
        mask = batch["fixed_mask"].cpu().bool().numpy()
        regularity = c.jacobian_metrics_voxel(vox,mask)
        regularity.update(diffusion_l2_full=c.diffusion_loss_voxel(vox),
                          diffusion_l2_fg=c.diffusion_loss_voxel(vox,mask))
        be_rows = c.compute_be_metric_rows(dense,batch["fixed_mask"].bool(),image_shape=c.IMAGE_SHAPE)
        regularity.update(c.be_rows_to_columns(be_rows))
        counts = {}
        for name, value in (("warped",labels),("fixed",batch["fixed_seg"]),("moving",batch["moving_seg"])):
            counts[name] = np.bincount(value.cpu().numpy().astype(np.int16).ravel(),minlength=36)[1:36].tolist()
        surface = c.surface_metrics_per_label(labels,batch["fixed_seg"],batch["moving_seg"])
    result = dict(registration_id=item["id"],rank=item["rank"],variant=item["variant"],
        dvf_sha256=hashlib.sha256(memoryview(field)).hexdigest(),
        dice=dice.tolist(),mean_dice=float(np.nanmean(dice)),valid_labels=int(np.isfinite(dice).sum()),
        surface_metrics=surface,hd95_nan_pattern_matches_dice=bool(np.array_equal(np.isnan(surface["hd95"]),np.isnan(dice))),
        label_voxel_counts=counts,ncc_full=ncc,regularity=regularity,
        metric_seconds=time.perf_counter()-started,backend=c.cuda_backend_metadata(),
        method="35 labels, missing labels excluded; area-weighted HD95 1mm; raw fixed>0; float64 J, crop2; positive-J natural log population SD; FD BE separately tagged")
    if not result["hd95_nan_pattern_matches_dice"] or result["valid_labels"] == 0:
        raise ValueError("Invalid label accounting")
    gc.collect()
    return result
