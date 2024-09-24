import numpy as np

from .laplacian import solve_poisson, laplace
from .geometry import norm_Linf, norm_L2, inner_L2
from .integrators import commutator

# Define inner products

def inner_Hm1(W1, W2):
    P2 = solve_poisson(W2)
    return -inner_L2(W1, P2)

def norm_Hm1(W):
    return np.sqrt(inner_Hm1(W, W))

def inner_H1(P1, P2):
    W2 = laplace(P2)
    return -inner_L2(P1, W2)

def norm_H1(P):
    return np.sqrt(inner_H1(P, P))


# Define common functions (e.g. used as loggers)

def energy_euler(W):
    """
    Energy of the state with vorticity matrix `W`
    for the 2-D Euler equations.
    """
    P = solve_poisson(W)
    return -inner_L2(W, P)/2.0

def enstrophy(W):
    """
    Enstrophy of the vorticity matrix `W`.
    """
    return inner_L2(W, W)/2.0


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
