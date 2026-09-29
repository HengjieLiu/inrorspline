"""Preregistered accuracy/speed candidates; no data, CUDA or side effects."""
from copy import deepcopy
import math

EXPERIMENT = "2026-09-21_oasis_cpbe_accuracy_speed_gpu0"
# Engineering screen, not clinical/noninferiority claims. Dice units are 0..1.
GATES = dict(dice_mean_drop=.002, dice_worst_drop=.01, hd95_mean_increase_mm=.10,
             hd95_worst_increase_mm=.50, ncc_mean_drop=.001,
             folding_fg_increase_pp=.50, sdlogj_fg_ratio=1.10,
             fd_be_fg_ratio=1.10, diffusion_fg_ratio=1.10)


def candidate(name, steps=(400,800,1300), *, cps=(8,4,2), down=(4,2,1),
              lr=(.03,.01,.003), adaptive=False, minimum=(40,80,130),
              patience=(20,35,50), interval=25, threshold_factor=1.,
              lr_schedule="constant", features=("cpbe",), optimizer="adam"):
    n = len(cps)
    if any(len(v) != n for v in (steps,down,lr,minimum,patience)):
        raise ValueError("All stage arrays must have the same length")
    if any(a != 2*b for a,b in zip(cps,cps[1:])) or any(a%b for a,b in zip(cps,down)):
        raise ValueError("Require exact dyadic knot insertion and integer domain cps")
    comma = lambda v: ",".join(map(str,v))
    return dict(name=name, features=list(features), lr_schedule=lr_schedule,
        optimizer=optimizer, lr_floor_ratio=.2,
        overrides=dict(name=name, epochs=sum(steps), stage_epochs=comma(steps),
            stage_full_cps=comma(cps), stage_downsample=comma(down), stage_lr=comma(lr),
            early_stop_mode="adaptive" if adaptive else "none", stage_min_epochs=comma(minimum),
            stage_patience=comma(patience), stage_check_interval=comma([interval]*n),
            stage_rel_delta_total=comma([1e-4*threshold_factor]*n),
            stage_rel_delta_data=comma([5e-5*threshold_factor]*n)))


def main_candidates():
    four = dict(cps=(16,8,4,2), down=(8,4,2,1), lr=(.03,.03,.01,.003),
                minimum=(30,40,50,60), patience=(20,20,20,20))
    return [
        candidate("reference_cpbe", adaptive=True),
        candidate("last_stage_50", (400,800,50), adaptive=True, minimum=(40,80,50)),
        candidate("fixed_50_50_50", (50,50,50)),
        candidate("fixed_50_50_100", (50,50,100)),
        candidate("fixed_75_75_100", (75,75,100)),
        candidate("threshold_2x", adaptive=True, threshold_factor=2),
        candidate("loose_5x", adaptive=True, minimum=(50,50,50), patience=(25,25,25), threshold_factor=5),
        candidate("short_window_5x", adaptive=True, minimum=(50,50,50), patience=(20,20,20), interval=10, threshold_factor=5),
        candidate("cosine_50_50_100", (50,50,100), lr_schedule="cosine"),
        candidate("step_decay_50_50_100", (50,50,100), lr_schedule="step"),
        candidate("mr16_fixed_25_50_50_100", (25,50,50,100), **four),
        candidate("mr16_adaptive", (200,400,800,1300), adaptive=True, interval=10, threshold_factor=5, **four),
    ]


def optional_candidates(base):
    result = []
    for suffix, features, optimizer in (
        ("cached_batch", ["cache","batch","cpbe"], "adam"),
        ("amsgrad", ["cpbe"], "amsgrad")):
        value = deepcopy(base)
        value.update(name=base["name"]+"_"+suffix, features=features, optimizer=optimizer)
        value["overrides"]["name"] = value["name"]
        result.append(value)
    return result


def lr_multiplier(policy, step, total, floor=.2):
    progress = (step-1)/max(total-1,1)
    if policy == "constant":
        return 1.
    if policy == "cosine":
        return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))
    if policy == "step":
        return 1. if progress < .5 else .5 if progress < .8 else .25
    raise ValueError(policy)
