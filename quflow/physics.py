import numpy as np

from .laplacian import solve_poisson, laplace
from .geometry import norm_Linf, norm_L2, inner_L2
from .integrators import commutator

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


def sectional_curvature(F, G):
    DeltaF = laplace(F)
    DeltaG = laplace(G)
    FGcomm = commutator(F, G) 
    DeltaFGcomm = commutator(DeltaF, G)
    DeltaGFcomm = commutator(DeltaG, F)
    DeltaFFcomm = commutator(DeltaF, F)
    DeltaGGcomm = commutator(DeltaG, G)

    C = -inner_L2(DeltaFGcomm + DeltaGFcomm, solve_poisson(DeltaFGcomm + DeltaGFcomm))/4.0

    C -= inner_L2(FGcomm, DeltaFGcomm - DeltaGFcomm)/2.0

    C += inner_L2(FGcomm, laplace(FGcomm))*(3.0/4.0)

    C += inner_L2(DeltaFFcomm, solve_poisson(DeltaGGcomm))

    return C
