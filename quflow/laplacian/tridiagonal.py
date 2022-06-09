import numpy as np
from scipy.linalg import solveh_banded
from ..utils import mat2diagh, diagh2mat
from numba import njit, prange

# ----------------
# GLOBAL VARIABLES
# ----------------

_tridiagonal_laplacian_cache = dict()
_tridiagonal_heat_cache = dict()
_tridiagonal_viscdamp_cache = dict()
_parallel = True


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------

def compute_tridiagonal_laplacian(N, bc=False):
    """
    Compute tridiagonal laplacian.

    Parameters
    ----------
    N: int
    bc: bool (optional)
        Whether boundary conditions should be added to make the laplacian non-singular.
        Notice that this bc is different from the one used for sparse laplacians.

    Returns
    -------
    lap: array, shape (N//2+1, 2, N)
        Outer index: system for diagonal m and N-m.
        Middle index: which diagonal, stored according to 'lower form' of 'scipy.linalg.solveh_banded'.
        Inner index: entries for diagonal m and N-m.
    """

    lap = np.zeros((N//2+1, 2, N), dtype=np.float64)
    i_full = np.arange(N)
    for m in range(N//2+1):

        # Global diagonal m (of length N-m)
        i = i_full[:N - m]
        lap[m, 0, 0:N - m] = -((N - 1)*(2*i + 1 + m) - 2*i*(i + m))
        i = i_full[1:N - m]
        lap[m, 1, 0:N - m - 1] = np.sqrt(((i + m)*(N - i - m))*(i*(N - i)))

        # Global diagonal N-m (of length m)
        i = i_full[:m]
        lap[m, 0, N - m:] = -((N - 1)*(2*i + 1 + N - m) - 2*i*(i + N - m))
        i = i_full[1:m]
        lap[m, 1, N - m:-1] = np.sqrt(((i + N - m)*(m - i))*(i*(N - i)))

    if bc:
        lap[0, 0, 0] -= 0.5

    return lap


def dot_tridiagonal(lap, P):
    """
    Dot product for tridiagonal operator.

    Parameters
    ----------
    lap: ndarray(shape(N//2+1, 2, N), dtype=float)
        Tridiagonal operator (typically laplacian).
    P: ndarray(shape=(N,N), dtype=complex)
        Input matrix.

    Returns
    -------
    W: ndarray(shape=(N,N), dtype=complex)
        Output matrix.
    """
    N = P.shape[0]

    Pdiagh = mat2diagh(P)

    Wdiagh = lap[:, 0, :]*Pdiagh
    Wdiagh[:, 1:] += lap[:, 1, :-1]*Pdiagh[:, :-1]
    Wdiagh[:, :-1] += lap[:, 1, :-1]*Pdiagh[:, 1:]

    W = diagh2mat(Wdiagh)

    return W


@njit(parallel=True)
def solve_tridiagonal_(lap, W, P, vtmp, ytmp):
    """
    Highly optimized function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` operator).

    Parameters
    ----------
    lap: ndarray(shape=(N//2+1, 2, N), dtype=float)
        Direct laplacian.
    W: ndarray(shape=(N, N), dtype=complex)
        Input matrix.
    P: ndarray(shape=(N, N), dtype=complex)
        Output matrix.
    vtmp: ndarray(shape=(N//2+1, N), dtype=float)
        Temporary float memory needed.
    ytmp: ndarray(shape=(N//2+1, N), dtype=complex)
        Temporary complex memory needed.
    """
    N = W.shape[0]

    for m in prange(N//2+1):
        n = N
        start_ind = lap.shape[1]-n*(n+1)//2
        end_ind = start_ind + n

        a = lap[m, 1, :-1]
        b = lap[m, 0, :]
        y = ytmp[m, :]
        v = vtmp[m, :]

        vk = b[0]
        v[0] = vk
        fk = W[0, m]
        yk = fk
        y[0] = yk

        for k in range(1, n):
            lk = a[k]/vk
            fk = W[k, m+k]
            yk = fk - lk*yk
            y[k] = yk
            vk = b[k] - lk*a[k]
            v[k] = vk

        pk = y[n-1]/v[n-1]
        P[n-1, m+n-1] = pk
        if m != 0:
            P[m+n-1, n-1] = -np.conj(pk)

        for k in range(n-2, -1, -1):
            pk = (y[k]-a[k+1]*pk)/v[k]
            P[k, m+k] = pk
            if m != 0:
                P[m+k, k] = -np.conj(pk)

    trP = np.trace(P)/N
    for k in prange(N):
        P[k, k] -= trP


def solve_tridiagonal(lap, W):
    """
    Highly optimized function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` direct matrix).

    Parameters
    ----------
    lap: ndarray(shape=(N//2+1, 2, N), dtype=float)
        Tridiagonal laplacian.
    W: ndarray(shape=(N, N), dtype=complex)
        Input matrix.

    Returns
    -------
    P: ndarray(shape=(N, N), dtype=complex)
        Output matrix.
    """
    N = W.shape[0]

    Wdiagh = mat2diagh(W)
    Pdiagh = np.zeros_like(Wdiagh)

    # For each double-tridiagonal, solve a tridiagonal system
    for m in range(N//2+1):
        # We need -lap to get positive definiteness, needed for solveh_banded
        Pdiagh[m, :] = solveh_banded(-lap[m, :, :],  -Wdiagh[m, :], lower=True)

    # Make sure we stay in su(N) (corresponds to vanishing mean bc)
    trP = Pdiagh[0, :].sum()/N
    Pdiagh[0, :] -= trP

    # Convert back to matrix
    P = diagh2mat(Pdiagh)

    return P


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def laplacian(N, bc=False):
    """
    Return quantized laplacian (as a direct laplacian).

    Parameters
    ----------
    N: int
    bc: bool (optional)
        Whether to include boundary conditions.

    Returns
    -------
    lap : ndarray(shape=(2, N*(N+1)/2), dtype=flaot)
    """
    global _tridiagonal_laplacian_cache

    if (N, bc) not in _tridiagonal_laplacian_cache:
        lap = compute_tridiagonal_laplacian(N, bc=bc)
        _tridiagonal_laplacian_cache[(N, bc)] = lap

    return _tridiagonal_laplacian_cache[(N, bc)]


def laplace(P):
    """
    Return quantized laplacian applied to stream function `P`.

    Parameters
    ----------
    P: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    W: ndarray(shape=(N, N), dtype=complex)
    """
    N = P.shape[0]
    lap = laplacian(N)

    # Apply dot product
    W = dot_tridiagonal(lap, P)

    return W


def solve_poisson(W):
    """
    Return stream matrix `P` for `W`.

    Parameters
    ----------
    W: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    P: ndarray(shape=(N, N), dtype=complex)
    """
    N = W.shape[0]
    lap = laplacian(N, bc=True)
    P = solve_tridiagonal(lap, W)

    return P


def solve_heat(h_times_nu, W0):
    """
    Solve quantized heat equation using backward Euler method.

    Parameters
    ----------
    h_times_nu: float
        Stepsize (in qtime) times viscosity.
    W0: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    Wh: ndarray(shape=(N, N), dtype=complex)
    """
    global _tridiagonal_heat_cache

    N = W0.shape[0]

    if (N, h_times_nu) not in _tridiagonal_heat_cache:
        # Get tridiagonal laplacian
        lap = laplacian(N, bc=False)

        # Get tridiagonal operator for backward Euler
        heat = -h_times_nu*lap
        heat[:, 0, :] += 1.0

        # Store in cache
        _tridiagonal_heat_cache[(N, h_times_nu)] = heat
    else:
        heat = _tridiagonal_heat_cache[(N, h_times_nu)]

    Wh = solve_tridiagonal(heat, W0)

    return Wh


def solve_viscdamp(h, W0, nu=1e-4, alpha=0.01, force=None, theta=1):
    """
    Solve quantized viscosity and damping equation

        W' - nu * ∆ W + alpha * W = F

    for one time-step using the theta scheme.

    Parameters
    ----------
    h: float
        Time-step.
    W0: ndarray(shape=(N, N), dtype=complex)
        Initial vorticity matrix.
    nu: float
        Viscosity.
    alpha: float
        Damping.
    force: None or ndarray(shape=(N, N), dtype=complex)
        External forcing F applied.
    theta: float
        Weighting in the theta method (Crank-Nicolson for theta=0.5).

    Returns
    -------
    Wh: ndarray(shape=(N, N), dtype=complex)
    """
    global _tridiagonal_viscdamp_cache

    N = W0.shape[0]

    if (N, h, nu, alpha) not in _tridiagonal_viscdamp_cache:
        # Get tridiagonal laplacian
        lap = laplacian(N, bc=False)

        # Get tridiagonal operator for theta method
        viscdamp = -(h*nu*theta)*lap
        viscdamp[:, 0, :] += 1.0+h*alpha*theta

        # Store in cache
        _tridiagonal_viscdamp_cache[(N, h, nu, alpha)] = viscdamp
    else:
        viscdamp = _tridiagonal_viscdamp_cache[(N, h, nu, alpha)]

    # Prepare right hand side in Crank-Nicolson
    if theta == 1:
        Wrhs = W0.copy()
    else:
        Wrhs = (1.0-alpha*h*(1-theta))*W0
        Wrhs += (nu*h*(1-theta))*laplace(W0)
    if force is not None:
        Wrhs += h*force

    # Solve linear subsystems
    Wh = solve_tridiagonal(viscdamp, Wrhs)

    return Wh
