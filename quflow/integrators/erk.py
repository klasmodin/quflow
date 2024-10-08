import numpy as np
import scipy.linalg
import inspect
import numba as nb

from ..laplacian import solve_poisson, solve_heat
from ..laplacian import select_skewherm as select_skewherm_laplacian
from ..geometry import norm_Linf, bracket
from .isospectral import commutator
from .isospectral import update_stats


# -------------------------------------------------
# CLASSICAL (EXPLICIT, NON-ISOSPECTRAL) INTEGRATORS
# -------------------------------------------------

def euler(W: np.ndarray,
          dt: float,
          steps: int = 100,
          hamiltonian=solve_poisson,
          forcing=None,
          stats: dict = None,
          **kwargs) -> np.ndarray:
    """
    Time-stepping by Euler's explicit first order method.

    Parameters
    ----------
    W: ndarray
        Initial vorticity (overwritten and returned).
    dt: float
        Time step length.
    steps: int
        Number of steps to take.
    hamiltonian: function(W)
        The Hamiltonian returning a stream matrix.
    forcing: None or function(P, W)
        Extra force function (to allow non-isospectral perturbations).
    stats: None or dict
        Dictionary with statistics
    **kwargs: dict
        Extra keyword arguments

    Returns
    -------
    W: ndarray
    """
    if forcing is None:
        rhs = bracket
    else:
        def rhs(P, W):
            return bracket(P, W) + forcing(P, W)

    for k in range(steps):
        P = hamiltonian(W)
        VF = rhs(P, W)
        W += dt*VF

    if stats is not None:
        update_stats(stats, steps=steps)

    return W


def heun(W, dt, steps=100, hamiltonian=solve_poisson, forcing=None):
    """
    Time-stepping by Heun's second order method.

    Parameters
    ----------
    W: ndarray
        Initial vorticity (overwritten and returned).
    dt: float
        Time step length.
    steps: int
        Number of steps to take.
    hamiltonian: function(W)
        The Hamiltonian returning a stream matrix.
    forcing: function(P, W)
        Extra force function (to allow non-isospectral perturbations).

    Returns
    -------
    W: ndarray
    """
    if forcing is None:
        rhs = bracket
    else:
        def rhs(P, W):
            return bracket(P, W) + forcing(P, W)

    for k in range(steps):

        # Evaluate RHS at W
        P = hamiltonian(W)
        F0 = rhs(P, W)

        # Compute Heun predictor
        Wprime = W + dt*F0

        # Evaluate RHS at predictor WP
        P = hamiltonian(Wprime)
        F = rhs(P, Wprime)

        # Compute averaged RHS
        F += F0
        F *= dt/2.0

        # Update W
        W += F

    return W


def rk4(W, dt, steps=100, hamiltonian=solve_poisson, forcing=None):
    """
    Time-stepping by the classical Runge-Kutta fourth order method.

    Parameters
    ----------
    W: ndarray
        Initial vorticity (overwritten and returned).
    dt: float
        Time step length.
    steps: int
        Number of steps to take.
    hamiltonian: function(W)
        The Hamiltonian returning a stream matrix.
    forcing: function(P, W) or None (default)
        Extra force function (to allow non-isospectral perturbations).

    Returns
    -------
    W: ndarray
    """
    if forcing is None:
        rhs = bracket
    else:
        def rhs(P, W):
            return bracket(P, W) + forcing(P, W)

    for k in range(steps):
        P = hamiltonian(W)
        K1 = rhs(P, W)

        Wprime = W + (dt/2.0)*K1
        P = hamiltonian(Wprime)
        K2 = rhs(P, Wprime)

        Wprime = W + (dt/2.0)*K2
        P = hamiltonian(Wprime)
        K3 = rhs(P, Wprime)

        Wprime = W + dt*K3
        P = hamiltonian(Wprime)
        K4 = rhs(P, Wprime)

        W += (dt/6.0)*(K1+2*K2+2*K3+K4)

    return W


# Default explicit integrator
explicit = heun

