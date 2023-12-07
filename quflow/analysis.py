import numpy as np

from .laplacian import solve_poisson
from .transforms import as_shr
from .utils import elm2ind


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


def energy_spectrum(data, beta=0):
    """
    Return energy spectrum for `data` in either W, omegar, omegac, or fun format.

    Parameters
    ----------
    data: ndarray
    beta: (default 0) use H^s norm for s = 1-beta/2

    Returns
    -------
    el, energy: ndarray, ndarray
    """
    omegar = as_shr(data)
    N = round(np.sqrt(omegar.shape[0]))
    energy = np.ones(N-1, dtype=float)
    for el in range(1, N):
        energy[el-1] = (omegar[elm2ind(el, -el):elm2ind(el, el)+1]**2).sum()/(el*(el+1))**(1-beta/2)
    return np.arange(1, N), energy


def enstrophy_spectrum(data):
    """
    Return enstrophy spectrum for `data` in either W, omegar, omegac, or fun format.

    Parameters
    ----------
    data: ndarray

    Returns
    -------
    el, enstrophy: ndarray, ndarray
    """
    omegar = as_shr(data)
    N = round(np.sqrt(omegar.shape[0]))
    enstrophy = np.ones(N-1, dtype=float)
    for el in range(1, N):
        enstrophy[el-1] = (omegar[elm2ind(el, -el):elm2ind(el, el)+1]**2).sum()
    return np.arange(1, N), enstrophy
