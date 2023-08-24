import numpy as np
from numba import njit, prange

# ----------------
# GLOBAL VARIABLES
# ----------------

_cpu_laplacian_cache = dict()
_cpu_buffer_cache = dict()
_cpu_heat_cache = dict()
_cpu_helmholtz_cache = dict()
_cpu_viscdamp_cache = dict()


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------


@njit
def mk2ij(m, k):
    if m >= 0:
        i = k
        j = k + m
    else:
        i = k - m
        j = k
    return i, j


@njit
def ij2mk(i, j):
    m = j-i
    if m >= 0:
        k = i
    else:
        k = j
    return m, k


@njit
def compute_cpu_laplacian_(N, bc=False, dtype=np.float64):
    """
    Compute tridiagonal laplacian.

    Parameters
    ----------
    N: int
    lap: int or real array, shape (N, N, 2)
        If int, then N = lap and the actual array is created.
        Two outer indices: corresponding to matrix entries in W.
        Inner index: 0: 0th diagonal, 1: 1:st and -1:st diagonal.
    bc: bool (optional)
        Whether boundary conditions should be added to make the laplacian non-singular.
        Notice that this bc is different from the one used for sparse laplacians.
    dtype: float type

    Returns
    -------
    lap: real array (N, N, 2)
    """
    lap = np.zeros((N, N, 2), dtype=dtype)

    # Loops to set elements of lap
    for m in range(-N+1, N):
        absm = np.abs(m)
        for k in range(N-absm):
            i, j = mk2ij(m, k)
            lap[i, j, 0] = -((N - 1)*(2*k + 1 + absm) - 2*k*(k + absm))
            lap[i, j, 1] = np.sqrt(((k + absm)*(N - k - absm))*(k*(N - k)))

    if bc:
        lap[0, 0, 0] += 0.5

    return lap


@njit(parallel=False)
def dot_cpu_generic_(lap, P, W):
    N = P.shape[0]
    for i in prange(N):
        for j in range(N):
            W[i, j] = lap[i, j, 0]*P[i, j]
            if i < N-1 and j < N-1:
                W[i, j] += lap[i+1, j+1, 1]*P[i+1, j+1]
            if i > 0 and j > 0:
                W[i, j] += lap[i, j, 1]*P[i-1, j-1]
    return W


@njit(parallel=False)
def dot_cpu_skewh2_(lap, P, W):
    N = P.shape[0]
    for i in prange(N):
        for j in range(i):
            W[i, j] = lap[i, j, 0]*P[i, j]
            if i < N-1 and j < N-1:
                W[i, j] += lap[i+1, j+1, 1]*P[i+1, j+1]
            if i > 0 and j > 0:
                W[i, j] += lap[i, j, 1]*P[i-1, j-1]
            W[j, i] = -np.conj(W[i,j])
    return W


@njit(parallel=True)
def dot_cpu_nonskewh_(lap, P, W):
    """
    Dot product for tridiagonal operator.

    Parameters
    ----------
    lap: ndarray(shape(N, N, 2), dtype=float)
        Tridiagonal operator (typically laplacian).
    P: ndarray(shape=(N,N), dtype=complex)
        Input matrix.
    W: ndarray(shape=(N,N), dtype=complex)
        Output matrix.
    """
    N = P.shape[0]

    for m in prange(-N+1, N):
        absm = np.abs(m)
        if N-absm == 1:
            i, j = mk2ij(m, 0)
            W[i, j] = lap[i, j, 0]*P[i, j]
        else:
            i, j = mk2ij(m, 0)
            W[i, j] = lap[i, j, 0]*P[i, j] + lap[i+1, j+1, 1]*P[i+1, j+1]
            for k in range(1, N-absm-1):
                i, j = mk2ij(m, k)
                W[i, j] = lap[i, j, 0]*P[i, j] + lap[i+1, j+1, 1]*P[i+1, j+1] + lap[i, j, 1]*P[i-1, j-1]
            i, j = mk2ij(m, N-absm-1)
            W[i, j] = lap[i, j, 0]*P[i, j] + lap[i, j, 1]*P[i-1, j-1]

    return W


@njit(parallel=False)
def dot_cpu_skewh_(lap, P, W):
    """
    Dot product for tridiagonal operator, for skew-Hermitian P.

    Parameters
    ----------
    lap: ndarray(shape(N, N, 2), dtype=float)
        Tridiagonal operator (typically laplacian).
    P: ndarray(shape=(N,N), dtype=complex)
        Input matrix.
    W: ndarray(shape=(N,N), dtype=complex)
        Output matrix.
    """
    N = P.shape[0]

    for m in prange(N):
        if N-m == 1:
            i, j = mk2ij(m, 0)
            W[i, j] = lap[i, j, 0]*P[i, j]
        else:
            i, j = mk2ij(m, 0)
            W[i, j] = lap[i, j, 0]*P[i, j] + lap[i+1, j+1, 1]*P[i+1, j+1]
            for k in range(1, N-m-1):
                i, j = mk2ij(m, k)
                W[i, j] = lap[i, j, 0]*P[i, j] + lap[i+1, j+1, 1]*P[i+1, j+1] + lap[i, j, 1]*P[i-1, j-1]
            i, j = mk2ij(m, N-m-1)
            W[i, j] = lap[i, j, 0]*P[i, j] + lap[i, j, 1]*P[i-1, j-1]

    # Make skewh from upper triangular part
    for i in range(N):
        for j in range(i):
            W[i, j] = -np.conj(W[j, i])

    return W


# Set default dot product
# dot_cpu_ = dot_cpu_skewh_
dot_cpu_ = dot_cpu_generic_


@njit(parallel=True)
def solve_cpu_nonskewh_(lap, W, P, buffer_float, buffer_complex):
    """
    Function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` array). Uses NUMBA to accelerate the
    Thomas algorithm for tridiagonal solver calculations.

    Parameters
    ----------
    lap: ndarray(shape=(N, N, 2), dtype=float)
        Tridiagonal laplacian.
    W: ndarray(shape=(N, N), dtype=complex)
        Input matrix.
    P: ndarray(shape=(N, N), dtype=complex)
        Output matrix.
    buffer_float: ndarray(shape=(N, N), dtype=float)
        Float buffer.
    buffer_complex: ndarray(shape=(N, N), dtype=complex)
        Complex buffer.
    """
    N = W.shape[0]

    # For each m-diagonal in W, solve a tridiagonal system with Thomas algorithm
    for m in prange(-N+1, N):

        absm = np.abs(m)

        # Initialize buffers
        i, j = mk2ij(m, 0)
        buffer_float[i, j] = lap[i, j, 0]
        buffer_complex[i, j] = W[i, j]

        # Forward sweep
        for k in range(1, N-absm):
            i, j = mk2ij(m, k)
            im, jm = i-1, j-1

            w = lap[i, j, 1]/buffer_float[im, jm]
            buffer_float[i, j] = lap[i, j, 0] - w*lap[i, j, 1]
            buffer_complex[i, j] = W[i, j] - w*buffer_complex[im, jm]

        # Backward sweep
        i, j = mk2ij(m, N-absm-1)
        P[i, j] = buffer_complex[i, j]/buffer_float[i, j]
        # P[j, i] = np.conj(P[i, j])
        for k in range(N-absm-2, -1, -1):
            i, j = mk2ij(m, k)
            ip, jp = i+1, j+1
            P[i, j] = (buffer_complex[i, j] - lap[ip, jp, 1]*P[ip, jp])/buffer_float[i, j]
            # P[j, i] = np.conj(P[i, j])

    # Make sure the trace of P vanishes (corresponds to bc for laplacian)
    trP = P[0, 0]
    for k in range(1, N):
        trP += P[k, k]
    trP /= N
    for k in range(N):
        P[k, k] -= trP

    return P


@njit(parallel=True)
def solve_cpu_skewh_(lap, W, P, buffer_float, buffer_complex):
    """
    Function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` array). Uses NUMBA to accelerate the
    Thomas algorithm for tridiagonal solver calculations.

    Parameters
    ----------
    lap: ndarray(shape=(N, N, 2), dtype=float)
        Tridiagonal laplacian.
    W: ndarray(shape=(N, N), dtype=complex)
        Input matrix.
    P: ndarray(shape=(N, N), dtype=complex)
        Output matrix.
    buffer_float: ndarray(shape=(N, N), dtype=float)
        Float buffer.
    buffer_complex: ndarray(shape=(N, N), dtype=complex)
        Complex buffer.
    """
    N = W.shape[0]

    # For each m-diagonal in W, solve a tridiagonal system with Thomas algorithm
    for m in prange(N):

        # Initialize buffers
        i, j = mk2ij(m, 0)
        buffer_float[i, j] = lap[i, j, 0]
        buffer_complex[i, j] = W[i, j]

        # Forward sweep
        for k in range(1, N-m):
            i, j = mk2ij(m, k)
            im, jm = i-1, j-1

            w = lap[i, j, 1]/buffer_float[im, jm]
            buffer_float[i, j] = lap[i, j, 0] - w*lap[i, j, 1]
            buffer_complex[i, j] = W[i, j] - w*buffer_complex[im, jm]

        # Backward sweep
        i, j = mk2ij(m, N-m-1)
        P[i, j] = buffer_complex[i, j]/buffer_float[i, j]
        if m != 0:
            P[j, i] = -np.conj(P[i, j])
        for k in range(N-m-2, -1, -1):
            i, j = mk2ij(m, k)
            ip, jp = i+1, j+1
            P[i, j] = (buffer_complex[i, j] - lap[ip, jp, 1]*P[ip, jp])/buffer_float[i, j]
            if m != 0:
                P[j, i] = -np.conj(P[i, j])

    # Make sure the trace of P vanishes (corresponds to bc for laplacian)
    trP = P[0, 0]
    for k in range(1, N):
        trP += P[k, k]
    trP /= N
    for k in range(N):
        P[k, k] -= trP

    return P


# Set default solver
solve_cpu_ = solve_cpu_skewh_


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def select_skewherm(flag):
    """
    Select whether matrices for Laplacian are skew Hermitian.

    Parameters
    ----------
    flag: bool

    Returns
    -------
    previous flag (bool)
    """
    global solve_cpu_
    global dot_cpu_
    oldflag = False
    if solve_cpu_ is solve_cpu_skewh_:
        oldflag = True

    if flag:
        solve_cpu_ = solve_cpu_skewh_
        if dot_cpu_ is dot_cpu_nonskewh_:
            dot_cpu_ = dot_cpu_skewh_
    else:
        solve_cpu_ = solve_cpu_nonskewh_
        if dot_cpu_ is dot_cpu_skewh_:
            dot_cpu_ = dot_cpu_nonskewh_

    return oldflag


def allocate_buffer(W):
    global _cpu_buffer_cache
    N = W.shape[0]

    buffer_complex = W.copy()
    buffer_float = W.real.copy()
    P_out = W.copy()
    _cpu_buffer_cache[N] = {'complex': buffer_complex, 'float': buffer_float, 'P': P_out}


def laplacian(N, bc=False, dtype=np.float64):
    """
    Return quantized laplacian (as a tridiagonal laplacian).

    Parameters
    ----------
    N: int
    bc: bool (optional)
        Whether to include boundary conditions.
    dtype: float type

    Returns
    -------
    lap : ndarray(shape=(2, N*(N+1)/2), dtype=flaot)
    """
    global _cpu_laplacian_cache

    if (N, bc, dtype) not in _cpu_laplacian_cache:
        lap = compute_cpu_laplacian_(N, bc=bc, dtype=dtype)
        _cpu_laplacian_cache[(N, bc, dtype)] = lap

    return _cpu_laplacian_cache[(N, bc, dtype)]


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
    lap = laplacian(N, dtype=type(P[0, 0].real))

    # Apply dot product
    W = np.zeros_like(P)
    dot_cpu_(lap, P, W)

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

    if N not in _cpu_buffer_cache or W.dtype != _cpu_buffer_cache[N]['complex'].dtype:
        allocate_buffer(W)

    P = _cpu_buffer_cache[N]['P']
    solve_cpu_(lap, W, P,
               _cpu_buffer_cache[N]['float'],
               _cpu_buffer_cache[N]['complex'])

    return P


def solve_heat(h_times_nu, W0):
    """
    Solve heat equation
        W' =  ν Δ W,  W(0) = W0
    using one step of backward Euler method, i.e.,
        ( 1 - h ν Δ ) W = W0

    Parameters
    ----------
    h_times_nu: float
        Stepsize (in qtime) times viscosity ν.
    W0: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    Wh: ndarray(shape=(N, N), dtype=complex)
    """
    global _cpu_heat_cache

    N = W0.shape[0]

    if (N, h_times_nu) not in _cpu_heat_cache:
        # Get cpu laplacian
        lap = laplacian(N, bc=False, dtype=type(W0[0, 0].real))

        # Allocate buffer
        allocate_buffer(W0)

        # Get cpu operator for backward Euler
        heat = lap.copy()
        heat[:, :, 0] = 1.0
        heat[:, :, 1] = 0.0
        heat -= h_times_nu*lap

        # Store in cache
        _cpu_heat_cache[(N, h_times_nu)] = heat
    else:
        heat = _cpu_heat_cache[(N, h_times_nu)]

    Wh = _cpu_buffer_cache[N]['P']
    solve_cpu_(heat, W0, Wh,
               _cpu_buffer_cache[N]['float'],
               _cpu_buffer_cache[N]['complex'])

    return Wh


def solve_helmholtz(W, alpha=1.0):
    """
    Solve the inhomogeneous Helmholtz equation
        ( 1 - alpha * Δ ) P = W

    Parameters
    ----------
    alpha: float
    W: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    P: ndarray(shape=(N, N), dtype=complex)
    """
    global _cpu_helmholtz_cache

    N = W.shape[0]

    if (N, alpha) not in _cpu_helmholtz_cache:

        # Get cpu laplacian
        lap = laplacian(N, bc=False, dtype=type(W[0, 0].real))

        # Allocate buffer
        allocate_buffer(W)

        # Get cpu operator for backward Euler
        helmholtz = lap.copy()
        helmholtz[:, :, 0] = 1.0
        helmholtz[:, :, 1] = 0.0
        helmholtz -= alpha*lap

        # Store in cache
        _cpu_helmholtz_cache[(N, alpha)] = helmholtz
    else:
        helmholtz = _cpu_helmholtz_cache[(N, alpha)]

    P = _cpu_buffer_cache[N]['P']
    solve_cpu_(helmholtz, W, P,
               _cpu_buffer_cache[N]['float'],
               _cpu_buffer_cache[N]['complex'])

    return P


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
    global _cpu_viscdamp_cache

    N = W0.shape[0]

    if (N, h, nu, alpha) not in _cpu_viscdamp_cache:

        # Get cpu laplacian
        lap = laplacian(N, bc=False, dtype=type(W0[0, 0].real))

        # Allocate buffer
        allocate_buffer(W0)

        # Get cpu operator for backward Euler
        viscdamp = lap.copy()
        viscdamp[:, :, 0] = 1.0+h*alpha*theta
        viscdamp[:, :, 1] = 0.0
        viscdamp -= (h*nu*theta)*lap

        # Store in cache
        _cpu_viscdamp_cache[(N, h, nu, alpha)] = viscdamp
    else:
        viscdamp = _cpu_viscdamp_cache[(N, h, nu, alpha)]

    Wh = _cpu_buffer_cache[N]['P']

    # Prepare right hand side in Crank-Nicolson
    if theta == 1:
        Wrhs = W0.copy()
    else:
        Wrhs = (1.0-alpha*h*(1-theta))*W0
        Wrhs += (nu*h*(1-theta))*laplace(W0)
    if force is not None:
        Wrhs += h*force

    solve_cpu_(viscdamp, Wrhs, Wh,
               _cpu_buffer_cache[N]['float'],
               _cpu_buffer_cache[N]['complex'])

    return Wh
