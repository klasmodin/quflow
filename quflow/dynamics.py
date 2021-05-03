import numpy as np
from .utils import elm2ind
from numba import njit
import scipy.sparse.linalg

# ----------------
# GLOBAL VARIABLES
# ----------------


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def step_with_heun(W0, stepsize=0.1, steps=1, out=None):
    if out is None:
        W1 = np.zeros_like(W0)
    else:
        W1 = out

    for k in range(steps):
        # -------------
        # Heun's method
        # -------------

        # Calculate stream matrix P from vorticity matrix W
        P0 = solve_poisson(W0)


def scale_decomposition(W, P=None):
    """
    Perform canonical scale separation.

    Parameters
    ----------
    W: ndarray
        Vorticity
    P: ndarray (optional)
        Stream matrix. Computed if not given.

    Returns
    -------
    (Ws, Wr): tuple of ndarray
    """
    if P is None:
        P = solve_poisson(W)

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
    from .transforms import as_shr
    from .utils import elm2ind
    omegar = as_shr(data)
    N = round(np.sqrt(omegar.shape[0]))
    energy = np.ones(N-1, dtype=float)
    for el in range(1, N):
        energy[el-1] = (omegar[elm2ind(-el, el):elm2ind(el, el)+1]**2).sum()/(el*(el+1))
    return energy
