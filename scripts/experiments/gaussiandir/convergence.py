"""Frozen schedule and stopping definitions; no labels or SITReg target access."""
from __future__ import annotations

import copy
import math

import torch

from .data import false_to_mm, flat_coordinates, sample_volume
from .model import tv_loss


def schedule(base, budget, *, smoke=False):
    config = copy.deepcopy(base)
    if budget not in (200, 500, 2000) and not smoke:
        raise ValueError("Unapproved schedule")
    config["released_control"].update(stage_steps=[budget],
        adaptation_start_step=round(.15*budget), adaptation_end_step=round(.6*budget),
        adaptation_interval=max(1, round(.05*budget)))
    half = budget//2
    config["paper_reconstruction"].update(stage_steps=[half, half],
        adaptation_start_local_step=max(1, round(.1*half)),
        adaptation_interval=max(1, round(.1*half)))
    config["warmup_steps"] = {200:5, 500:13, 2000:50}.get(budget, 2)
    # Keep the pilot's diagnostics cadence so the anchor uses identical arithmetic.
    return config


def relative(a, b):
    # Signed negative NCC requires an absolute denominator.
    return abs(a-b)/max(abs(a), 1e-8)


class Plateau:
    """Loss-only LR controller. State is plain serializable data."""
    def __init__(self, settings):
        self.settings = settings
        self.rows = []
        self.reductions = []
        self.best = math.inf
        self.best_step = None

    def observe(self, step, scores, probe_mm):
        if len(scores) != 2 or not all(math.isfinite(x) for x in scores):
            raise FloatingPointError("Invalid monitoring loss")
        selected = scores[0] < self.best
        if selected:
            self.best, self.best_step = scores[0], step
        row = dict(step=step, scores=list(scores), probe_mm=probe_mm.cpu().clone(), best=self.best)
        self.rows.append(row)
        settings, window = self.settings, self.settings["window_steps"]
        old = next((r for r in reversed(self.rows) if r["step"] <= step-window), None)
        reduced = plateau = False
        diagnostic = {}
        if old is not None:
            recent = [r for r in self.rows if r["step"] >= old["step"]]
            changes = [relative(old["scores"][i], scores[i]) for i in range(2)]
            improvements = [max(0., old["scores"][i]-min(r["scores"][i] for r in recent)) /
                            max(abs(old["scores"][i]), 1e-8) for i in range(2)]
            best_improvement = max(0., old["best"]-self.best)/max(abs(old["best"]), 1e-8)
            rms = float((row["probe_mm"]-old["probe_mm"]).square().sum(-1).mean().sqrt())
            diagnostic = dict(relative_change=changes, relative_improvement=improvements,
                              best_relative_improvement=best_improvement, field_rms_change_mm=rms)
            # Test plateau BEFORE any new reduction: require a full window after a reduction.
            plateau = (len(self.reductions) >= settings["minimum_reductions"] and
                       step-self.reductions[-1] >= window and
                       max(changes+improvements) < settings["relative_tolerance"] and
                       rms < settings["field_rms_tolerance_mm"])
            if (not plateau and best_improvement < settings["relative_tolerance"] and
                    len(self.reductions) < settings["max_reductions"] and
                    step-(self.reductions[-1] if self.reductions else 0) >= window):
                self.reductions.append(step)
                reduced = True
        return dict(selected=selected, reduced=reduced, plateau=plateau,
                    reductions=len(self.reductions), best_step=self.best_step, **diagnostic)

    def multiplier(self, step):
        return (self.settings["lr_factor"] * min(1., step/self.settings["warmup_steps"]) *
                .5**len(self.reductions))

    def state(self):
        return dict(settings=self.settings, rows=self.rows, reductions=self.reductions,
                    best=self.best, best_step=self.best_step)

    @classmethod
    def restore(cls, state):
        obj = cls(state["settings"])
        for key in ("rows", "reductions", "best", "best_step"):
            setattr(obj, key, state[key])
        return obj


def out_of_time(charged, reserve, operation_estimate, limit):
    return charged + reserve + operation_estimate >= limit


class MonitoringBanks:
    def __init__(self, mask, config, settings):
        self.shape = tuple(mask.shape)
        self.affine = None
        device = mask.device
        ids = mask.flatten().nonzero().flatten()
        unit = torch.eye(3, device=device)[None]*(2/torch.tensor(self.shape[::-1], device=device))
        self.banks = []
        # Local generators neither consume nor reset the training RNG.
        for bank in range(2):
            generator = torch.Generator(device=device).manual_seed(settings["monitor_seed"]+bank)
            batches = []
            for _ in range(settings["monitor_batches"]):
                chosen = ids[torch.randint(len(ids), (config["query_batch"]//4,),
                                           device=device, generator=generator)]
                center = flat_coordinates(chosen, self.shape)
                batches.append(torch.cat((center, (center[:, None]+unit).reshape(-1, 3))))
            self.banks.append(batches)
        generator = torch.Generator(device=device).manual_seed(settings["probe_seed"])
        chosen = ids[torch.randint(len(ids), (settings["probe_points"],), device=device, generator=generator)]
        self.probe = flat_coordinates(chosen, self.shape)

    @torch.no_grad()
    def evaluate(self, model, inputs, ncc):
        scores, components = [], []
        for bank in self.banks:
            values = []
            for coordinates in bank:
                flow = model(coordinates)
                similarity = ncc(sample_volume(inputs["moving"], coordinates+flow),
                                 sample_volume(inputs["fixed"], coordinates))
                reg = tv_loss(flow, model.variant)
                values.append(torch.stack((similarity+8.*reg, similarity, reg)))
            mean = torch.stack(values).mean(0).cpu().tolist()
            scores.append(mean[0])
            components.append(dict(total=mean[0], ncc=mean[1], tv=mean[2]))
        probe = false_to_mm(model(self.probe), self.shape, inputs["affine"]).cpu()
        return scores, probe, components
