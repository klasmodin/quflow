import numpy as np
from .laplacian.direct import solve_poisson
from .transforms import as_shr
from .utils import elm2ind
from numba import njit

# ----------------
# GLOBAL VARIABLES
# ----------------


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------

def commutator(W, P):
    """
    Efficient computations of commutator for skew-Hermitian matrices.
    Warnings: only works for skew-Hermitian.

    Parameters
    ----------
    W: ndarray
    P: ndarray

    Returns
    -------
    ndarray
    """
    VF = W@P
    VF -= VF.conj().T
    return VF

# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------


def euler(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson):
    """
    Time-stepping by Euler's explicit first order method.

    Parameters
    ----------
    W: ndarray
        Initial vorticity (overwritten and returned).
    stepsize: float
        Time step length.
    steps: int
        Number of steps to take.
    hamiltonian: function
        The Hamiltonian returning a stream matrix.

    Returns
    -------
    W: ndarray
    """
    for k in range(steps):

        P = hamiltonian(W)
        VF = commutator(P, W)
        W += stepsize*VF

    return W


def heun(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson):
    """
    Time-stepping by Heun's second order method.

    Parameters
    ----------
    W: ndarray
        Initial vorticity (overwritten and returned).
    stepsize: float
        Time step length.
    steps: int
        Number of steps to take.
    hamiltonian: function
        The Hamiltonian returning a stream matrix.

    Returns
    -------
    W: ndarray
    """
    for k in range(steps):

        # Evaluate RHS at W
        P = hamiltonian(W)
        F0 = commutator(W, P)

        # Compute Heun predictor
        Wprime = W + stepsize*F0

        # Evaluate RHS at predictor WP
        P = hamiltonian(Wprime)
        F = commutator(Wprime, P)

        # Compute averaged RHS
        F += F0
        F *= stepsize/2.0

        # Update W
        W += F

    return W


def rk4(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson):
    """
    Time-stepping by the classical Runge-Kutta fourth order method.

    Parameters
    ----------
    W: ndarray
        Initial vorticity (overwritten and returned).
    stepsize: float
        Time step length.
    steps: int
        Number of steps to take.
    hamiltonian: function
        The Hamiltonian returning a stream matrix.

    Returns
    -------
    W: ndarray
    """
    for k in range(steps):
        P = hamiltonian(W)
        K1 = commutator(P, W)

        Wprime = W + (stepsize/2.0)*K1
        P = hamiltonian(Wprime)
        K2 = commutator(P, Wprime)

        Wprime = W + (stepsize/2.0)*K2
        P = hamiltonian(Wprime)
        K3 = commutator(P, Wprime)

        Wprime = W + stepsize*K3
        P = hamiltonian(Wprime)
        K4 = commutator(P, Wprime)

        W += (stepsize/6.0)*(K1+2*K2+2*K3+K4)

    return W


def scale_decomposition(W, P=None, hamiltonian=solve_poisson):
    """
    Perform canonical scale separation.

    Parameters
    ----------
    W: ndarray
        Vorticity matrix.
    P: ndarray (optional)
        Stream matrix. Computed if not given.
    hamiltonian: function
        The Hamiltonian returning a stream matrix.

    Returns
    -------
    (Ws, Wr): tuple of ndarray
    """
    if P is None:
        P = hamiltonian(W)

    D, E = np.linalg.eig(P)
    EWE = E.conj().T@W@E
    D2 = np.diag(np.diag(EWE))
    Ws = E@D2@E.conj().T
    Wr = W - Ws

    return Ws, Wr


def energy_spectrum(data):
    """
    Return energy spectrum for `data` in either W, omegar, omegac, or fun format.

    Parameters
    ----------
    data: ndarray

    Returns
    -------
    energy: ndarray
    """
    omegar = as_shr(data)
    N = round(np.sqrt(omegar.shape[0]))
    energy = np.ones(N-1, dtype=float)
    for el in range(1, N):
        energy[el-1] = (omegar[elm2ind(-el, el):elm2ind(el, el)+1]**2).sum()/(el*(el+1))
    return energy
