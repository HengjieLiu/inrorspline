"""Small fixed follow-up matrix; no population expansion or adaptive gamma search."""
from scripts.experiments.oasis_regularizer_sweep.protocol import PROTOCOL as BASE

GAMMAS=(0.,1e-4,1e-3,1e-2,1e-1,1.)
BACKENDS=("fd","analytic")
KEY="oasis_rank000_diffusion1_analytic_kl_c1_gamma_v1"
PROTOCOL=dict(BASE,version=KEY,case_ranks=[0],lambda_diffusion=1.,kl_backend="analytic",
              gammas=list(GAMMAS),diffusion_backends=list(BACKENDS),
              objective="NCC9 + (1*diffusion + gamma*analytic_KL_C1)",first10_extension=False)


def setting(backend,gamma):
    if backend not in BACKENDS or gamma not in GAMMAS: raise ValueError("Unapproved setting")
    return f"diffusion_{backend}_gamma{gamma:g}".replace(".","p").replace("-","m").replace("+","")
