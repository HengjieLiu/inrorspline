"""Explicit presentation defaults shared by preview and future capture tools."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Presentation:
    profile: str = "adaptive_cpbe"
    candidate: str = "MR_A_fast_decay"
    features: tuple = ("cpbe",)
    full_cps: tuple = (8, 4, 2)
    downsampling: tuple = (4, 2, 1)
    learning_rates: tuple = (.03, .01, .003)
    max_steps: tuple = (400, 800, 1300)
    early_stop_mode: str = "adaptive"
    alpha: float = 1000.
    seed: int = 1
    physical_gpu: int = 0
    benchmark: bool = False
    deterministic: bool = True
    capture_interval: int = 10
    axis: int = 2
    slice_index: int = 96
    grid_spacing: int = 8
    width: int = 1920
    height: int = 1080
    fps: int = 30
    duration_seconds: float = 30.
    initial_hold_seconds: float = 1.
    stage_hold_seconds: float = 1.
    final_hold_seconds: float = 3.

    def to_dict(self):
        return asdict(self)


DEFAULT = Presentation()

FIXED50_CACHED_BATCH = Presentation(
    profile="fixed50_cached_batch",
    candidate="fixed_50_50_50_cached_batch",
    features=("cache", "batch", "cpbe"),
    max_steps=(50, 50, 50),
    early_stop_mode="none",
)

FIXED50_CACHED_BATCH_DENSE_CAPTURE = Presentation(
    profile="fixed50_cached_batch_dense_capture",
    candidate="fixed_50_50_50_cached_batch",
    features=("cache", "batch", "cpbe"),
    max_steps=(50, 50, 50),
    early_stop_mode="none",
    capture_interval=1,
)

PROFILES = {value.profile: value for value in (
    DEFAULT, FIXED50_CACHED_BATCH, FIXED50_CACHED_BATCH_DENSE_CAPTURE)}


def presentation(name):
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown presentation profile {name!r}") from error


def candidate_overrides(value):
    """Explicit worker overrides; historical DEFAULT remains unchanged."""
    comma = lambda items: ",".join(map(str, items))
    return dict(
        name=value.candidate,
        epochs=sum(value.max_steps),
        stage_epochs=comma(value.max_steps),
        stage_full_cps=comma(value.full_cps),
        stage_downsample=comma(value.downsampling),
        stage_lr=comma(value.learning_rates),
        early_stop_mode=value.early_stop_mode,
        stage_min_epochs="40,80,130",
        stage_patience="20,35,50",
        stage_check_interval="25,25,25",
        stage_rel_delta_total="0.0001,0.0001,0.0001",
        stage_rel_delta_data="0.00005,0.00005,0.00005",
    )


def checkpoint_steps(actual_steps, interval=10):
    """Stage-local, post-update states; initialization and final state exactly once."""
    if actual_steps < 0 or interval <= 0:
        raise ValueError("Nonnegative step count and positive interval required")
    return sorted({0, actual_steps, *range(interval, actual_steps + 1, interval)})
