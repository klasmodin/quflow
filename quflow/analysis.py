import numpy as np

from .laplacian import solve_poisson
from .transforms import as_shr, mat2shr
from .utils import elm2ind, ind2elm


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


def random_shr(lmax=127, s=1.0, gamma=0.0, seed=None, **kwargs):
    """
    Generate random vector of real spherical harmonic coefficients.
    The Euclidean norm (corresponding to L^2 norm) is 1.

    Parameters
    ----------
    N: int
        Corresponding size of matrix (elmax = N-1).
    s: float
        Sobolev H^s smoothness.
    gamma: float or None,
        Ratio between total angular momentum and square root of enstrophy.
        A value of 0 means vanishing angular momentum.
        Must fulfill 0 <= gamma < 1.
    seed: int or None
        Using random seed.

    Returns
    -------
    omega: ndarray
    """
    N = lmax+1
    if seed is not None:
        np.random.seed(seed)
    omega = np.random.randn(N**2)
    omega[0] = 0.0

    if s != 0.0:
        els = ind2elm(np.arange(N**2))[0]
        omega[1:] /= (els[1:]*(els[1:]+1))**(s/2)
    
    if gamma == 0.0:
        omega[1:4] = 0.0
    elif gamma is not None:
        ens = (omega[4:]**2).sum()
        # angmomsq / (angmomsq + ens) = gamma**2
        # angmomsq = (angmomsq + ens)*gamma**2
        # angmomsq = ens*gamma**2/(1-gamma**2)
        angmom = np.sqrt(ens/(1-gamma**2))*gamma
        omega[1:4] *= angmom/np.linalg.norm(omega[1:4])
    
    # Normalize
    omega /= np.linalg.norm(omega)

    return omega


def gamma_ratio(data):
    """
    Compute ratio between total angular momentum and square root of entrophy.

    Parameters
    ----------
    data: ndarray
        If complex square matrix, treat as mat.
        If 1-d real array, treat as shr.

    Returns
    -------
    gamma: float
    """

    if data.ndim == 2:
        omega = mat2shr(data)
    elif data.ndim == 1:
        omega = data

    gamma = np.linalg.norm(omega[1:4])/np.linalg.norm(omega)

    return gamma