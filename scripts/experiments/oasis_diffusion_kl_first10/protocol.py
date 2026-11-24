"""Explicitly authorized first10 scope; numerical settings inherit the pilot."""
from scripts.experiments.oasis_diffusion_kl_rank0.protocol import BACKENDS, GAMMAS, setting
from scripts.experiments.oasis_diffusion_kl_rank0.protocol import PROTOCOL as PILOT

KEY = "oasis_first10_diffusion1_analytic_kl_c1_gamma_v1"
PROTOCOL = dict(PILOT, version=KEY, case_ranks=list(range(10)), first10_extension=True,
                primary_condition="warm", automatic_gamma_extension=False)


def validate_job(job):
    if job["phase"] not in ("smoke", "cohort") or job["weight"] != 1.:
        raise ValueError("Only diffusion1 first10 cohort/smoke jobs are authorized")
    backend = job["arm"].removeprefix("diffusion_")
    setting(backend, job["gamma"])
    if job["gamma"] == 0:
        raise ValueError("Gamma0 uses audited existing cohort records")
