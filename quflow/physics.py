import numpy as np

from .laplacian import solve_poisson
from .geometry import norm_Linf, norm_L2, inner_L2

# Define common functions (e.g. used as loggers)

def energy_euler(W):
    """
    Energy of the state with vorticity matrix `W`
    for the 2-D Euler equations.
    """
    P = solve_poisson(W)
    return -inner_L2(W, P)

def enstrophy(W):
    """
    Enstrophy of the vorticity matrix `W`.
    """
    return inner_L2(W, W)