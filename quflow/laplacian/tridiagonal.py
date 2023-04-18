import numpy as np
from scipy.linalg import solveh_banded
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

@njit
def mat2diagh(W):
    """
    Return lower diagonal format for hermitian matrix W.

    Parameters
    ----------
    W: ndarray, shape=(N, N)

    Returns
    -------
    ndarray, shape=(N//2+1, N)
    """
    W = np.ascontiguousarray(W)
    N = W.shape[0]
    d = np.zeros((N//2+1, N), dtype=W.dtype)
    for m in range(N//2+1):
        # Extract m:th lower diagonal
        dm = W.ravel()[N*m:(N-m)*(N+1)+N*m:N+1]

        # Extract (N-m):th lower diagonal
        dNm = W.ravel()[N*(N-m):m*(N+1)+N*(N-m):N+1]

        # Insert in d matrix
        d[m, :N-m] = dm
        d[m, N-m:] = dNm

    return d


@njit
def diagh2mat(dlow):
    """
    Return hermitian matrix W from lower diagonal format.

    Parameters
    ----------
    dlow: ndarray, shape=(N//2+1, N)

    Returns
    -------
    ndarray, shape=(N, N)
    """
    N = dlow.shape[-1]
    assert dlow.shape[-2] == N//2+1, "Seems dlow is out of shape!"
    W = np.zeros((N, N), dtype=dlow.dtype)

    for m in range(N//2+1):
        # Extract m:th lower diagonal
        dlm = W.ravel()[N*m:(N-m)*(N+1)+N*m:N+1]

        # Extract (N-m):th lower diagonal
        dlNm = W.ravel()[N*(N-m):m*(N+1)+N*(N-m):N+1]

        # Extract m:th upper diagonal
        dum = W.ravel()[m:(N-m)*(N+1)+m:N+1]

        # Extract (N-m):th upper diagonal
        duNm = W.ravel()[N-m:m*(N+1)+N-m:N+1]

        # Insert in W matrix
        dum[:] = -dlow[m, :N-m].conj()
        duNm[:] = -dlow[m, N-m:].conj()
        dlm[:] = dlow[m, :N-m]
        dlNm[:] = dlow[m, N-m:]

    return W


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
def solve_tridiagonal_numba(lap, W):
    """
    Function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` array). Uses NUMBA to accelerate the
    tridiagonal solver calculations.

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
    for m in prange(N//2+1):
        a = lap[m, 1, :-1]
        b = lap[m, 0, :].copy()
        d = Wdiagh[m, :]
        x = Pdiagh[m, :]

        # Forward sweep
        for i in range(1, N):
            w = a[i-1]/b[i-1]
            b[i] -= w*a[i-1]
            d[i] -= w*d[i-1]

        # Backward sweep
        x[N-1] = d[N-1]/b[N-1]
        for i in range(N-2, -1, -1):
            x[i] = (d[i] - a[i]*x[i+1])/b[i]

    # Make sure the trace of P vanishes (corresponds to bc for laplacian)
    trP = Pdiagh[0, :].sum()/N
    Pdiagh[0, :] -= trP

    # Convert back to matrix
    P = diagh2mat(Pdiagh)

    return P


def solve_tridiagonal_lapack(lap, W):
    """
    Function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` array). Uses LAPACK as backend for
    tridiagonal systems.

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


# Set default tridiagonal solver
solve_tridiagonal = solve_tridiagonal_numba


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def laplacian(N, bc=False):
    """
    Return quantized laplacian (as a tridiagonal laplacian).

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

    Wh = solve_tridiagonal(-heat, -W0)  # Temporary sign fix since matrix should be pos def.

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
