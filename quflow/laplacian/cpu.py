import numpy as np
from numba import njit, prange
import scipy.sparse as sp
from scipy.sparse import isspmatrix_dia

# ----------------
# GLOBAL VARIABLES
# ----------------

_cpu_laplacian_cache = dict()
_cpu_buffer_cache = dict()
_cpu_heat_cache = dict()
_cpu_helmholtz_cache = dict()
_cpu_viscdamp_cache = dict()
_cpu_cache = dict()


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------

def _get_cache(shape, dtype, *args):
    global _cpu_cache
    key = (shape, dtype, *args)
    if key not in _cpu_cache:
        if "real" in args:
            _cpu_cache[key] = np.zeros(shape, dtype=dtype).real
        else:
            _cpu_cache[key] = np.zeros(shape, dtype=dtype)
    return _cpu_cache[key]

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


@njit(error_model='numpy', fastmath=True)
def dot_cpu_generic_(lap, P, W):
    N = P.shape[0]
    for i in range(N):
        for j in range(N):
            W[i, j] = lap[i, j, 0]*P[i, j]
            if i < N-1 and j < N-1:
                W[i, j] += lap[i+1, j+1, 1]*P[i+1, j+1]
            if i > 0 and j > 0:
                W[i, j] += lap[i, j, 1]*P[i-1, j-1]
    return W


@njit(parallel=False, error_model='numpy', fastmath=True)
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


@njit(parallel=True, error_model='numpy', fastmath=True)
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


@njit(parallel=False, error_model='numpy', fastmath=True)
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


@njit(parallel=True, error_model='numpy', fastmath=True)
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


@njit(parallel=True, error_model='numpy', fastmath=True)
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


@njit(parallel=True, error_model='numpy', fastmath=True)
def solve_cpu_generic_(lap, W, P, buffer_float, buffer_complex):
    """

    Parameters
    ----------
    lap: ndarray(shape=(N**2, 2), dtype=float)
        Tridiagonal laplacian.
    W: ndarray(shape=(N**2,), dtype=complex)
        Input matrix.
    P: ndarray(shape=(N**2,), dtype=complex)
        Output matrix.
    buffer_float: ndarray(shape=(N**2,), dtype=float)
        Float buffer.
    buffer_complex: ndarray(shape=(N**2,), dtype=complex)
        Complex buffer.
    """
    N = W.shape[-1]
    lap_flat = lap.reshape((N**2, 2))
    W_flat = W.ravel()
    P_flat = P.ravel()
    buffer_float_flat = buffer_float.ravel()
    buffer_complex_flat = buffer_complex.ravel()

    # Set array stride. This should be the grid-stride on the GPU.
    stride = N + 1

    # For each m-diagonal in W, solve a tridiagonal system with Thomas algorithm.
    # Notice that m=N corresponds to the first lower diagonal. For example, if N=6,
    # we have the following m-values with respect to the rows and columns of the
    # original matrix W:
    #
    # 0 1 2 3 4 5
    # 6 0 1 2 3 4
    # 5 6 0 1 2 3
    # 4 5 6 0 1 2
    # 3 4 5 6 0 1
    # 2 3 4 5 6 0
    #
    for m in prange(N+1):

        # First element: k = m
        val_float = lap_flat[m, 0]
        val_complex = W_flat[m]
        buffer_float_flat[m] = val_float
        buffer_complex_flat[m] = val_complex

        # Forward sweep according to grid-stride
        for k in range(m + stride, N**2, stride):
            w = lap_flat[k, 1]/val_float
            val_float = lap_flat[k, 0] - w*lap_flat[k, 1]
            val_complex = W_flat[k] - w*val_complex
            buffer_float_flat[k] = val_float
            buffer_complex_flat[k] = val_complex
            k_last = k

        # Backward sweep according to grid-stride
        val_complex = buffer_complex_flat[k_last]/buffer_float_flat[k_last]
        val_float = lap_flat[k_last, 1]
        P_flat[k_last] = val_complex
        for k in range(k_last - stride, -1, -stride):
            P_flat[k] = (buffer_complex_flat[k] - val_float*val_complex)/buffer_float_flat[k]
            val_complex = P_flat[k]
            val_float = lap_flat[k, 1]

    # Make sure the trace of P vanishes (corresponds to bc for laplacian)
    trP = P_flat[::N+1].sum()
    trP /= N
    P_flat[::N+1] -= trP

    return P


# Set default solver
solve_cpu_ = solve_cpu_skewh_
# solve_cpu_ = solve_cpu_generic_


@njit(error_model='numpy', fastmath=True)
def dot_cpu_m_diag_(lap, P_m_diag, W_m_diag,):
    """
    Dot product for tridiagonal operator applied to m diagonal.

    Parameters
    ----------
    lap: ndarray(shape(N, N, 2), dtype=float)
        Tridiagonal operator (typically laplacian).
    P_m_diag: ndarray(shape=(N-m,), dtype=complex)
        Input diagonal.
    W_m_diag: ndarray(shape=(N-m,), dtype=complex)
        Output diagonal.
    """
    N = lap.shape[-2]
    m = N - W_m_diag.shape[-1]
    absm = np.abs(m)

    if N-absm == 1:
        i, j = mk2ij(m, 0)
        W_m_diag[0] = lap[i, j, 0]*P_m_diag[0]
    else:
        i, j = mk2ij(m, 0)
        W_m_diag[0] = lap[i, j, 0]*P_m_diag[0] + lap[i+1, j+1, 1]*P_m_diag[1]
        for k in range(1, N-absm-1):
            i, j = mk2ij(m, k)
            W_m_diag[k] = lap[i, j, 0]*P_m_diag[k] + lap[i+1, j+1, 1]*P_m_diag[k+1] + lap[i, j, 1]*P_m_diag[k-1]
        k = N-absm-1
        i, j = mk2ij(m, k)
        W_m_diag[k] = lap[i, j, 0]*P_m_diag[k] + lap[i, j, 1]*P_m_diag[k-1]

    return W_m_diag


@njit(error_model='numpy', fastmath=True)
def solve_cpu_m_diag_(lap, W_m_diag, P_m_diag):
    """
    Solve Poisson equation for only m diagonal.

    Parameters
    ----------
    lap: ndarray(shape=(N**2, 2), dtype=float)
        Tridiagonal laplacian.
    W_m_diag: ndarray(shape=(N-m,), dtype=complex)
        Input diagonal.
    P_m_diag: ndarray(shape=(N-m,), dtype=complex)
        Output diagonal.
    buffer_float: ndarray(shape=(N**2,), dtype=float)
        Float buffer.
    buffer_complex: ndarray(shape=(N**2,), dtype=complex)
        Complex buffer.
    """
    N = lap.shape[-2]
    m = N - W_m_diag.shape[-1]
    absm = np.abs(m)
    # buffer_float_flat = buffer_float.ravel()
    buffer_float_flat = np.zeros(N-absm, dtype=lap.dtype)
    # buffer_complex_flat = buffer_complex.ravel()
    buffer_complex_flat = np.zeros(N-absm, dtype=W_m_diag.dtype)

    # Initialize buffers
    i, j = mk2ij(m, 0)
    # buffer_float[i, j] = lap[i, j, 0]
    buffer_float_flat[0] = lap[i, j, 0]
    # buffer_complex[i, j] = W_m_diag[0]
    buffer_complex_flat[0] = W_m_diag[0]

    # Forward sweep
    for k in range(1, N-absm):
        i, j = mk2ij(m, k)
        # im, jm = i-1, j-1

        # w = lap[i, j, 1]/buffer_float[im, jm]
        w = lap[i, j, 1]/buffer_float_flat[k-1]
        # buffer_float[i, j] = lap[i, j, 0] - w*lap[i, j, 1]
        buffer_float_flat[k] = lap[i, j, 0] - w*lap[i, j, 1]
        # buffer_complex[i, j] = W[i, j] - w*buffer_complex[im, jm]
        buffer_complex_flat[k] = W_m_diag[k] - w*buffer_complex_flat[k-1]

    # Backward sweep
    i, j = mk2ij(m, N-absm-1)
    # P[i, j] = buffer_complex[i, j]/buffer_float[i, j]
    k = N-absm-1
    P_m_diag[k] = buffer_complex_flat[k]/buffer_float_flat[k]
    for k in range(N-absm-2, -1, -1):
        i, j = mk2ij(m, k)
        ip, jp = i+1, j+1
        # P[i, j] = (buffer_complex[i, j] - lap[ip, jp, 1]*P[ip, jp])/buffer_float[i, j]
        P_m_diag[k] = (buffer_complex_flat[k] - lap[ip, jp, 1]*P_m_diag[k+1])/buffer_float_flat[k]

    if m == 0:
        # Make sure the trace of P vanishes (corresponds to bc for laplacian)
        trP = P_m_diag[0]
        for k in range(1, N):
            trP += P_m_diag[k]
        trP /= N
        for k in range(N):
            P_m_diag[k] -= trP

    return P_m_diag


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def select_skewherm(flag):
    """
    Select whether matrices for Laplacian are skew Hermitian.

    Parameters
    ----------
    flag: bool
    gpu_like: bool

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
    lap : ndarray(shape=(N, N, 2), dtype=float)
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
    N = P.shape[-1]

    # Apply dot product
    if isspmatrix_dia(P):
        # Sparse diamatrix

        if hasattr(P, "el"):
            # This is a hack to keep track of "pure el" dia_matrix objects.
            # It makes the calculations of laplace and solve_poisson much faster.
            # In the future, one should probably make a subclass
            # of dia_matrix instead.
            el = P.el
            W = P.copy()
            W *= -el*(el+1)
            W.el = el  # The new dia_matrix will have the same el.
        else:
            lap = laplacian(N, dtype=np.float32 if P.dtype == np.complex64 else np.float64)
            W = P.copy()
            for P_diag_m, W_diag_m, m in zip(P.data, W.data, P.offsets):
                if m < 0:
                    dot_cpu_m_diag_(lap, P_diag_m[:N+m], W_diag_m[:N+m])
                else:
                    dot_cpu_m_diag_(lap, P_diag_m[m:], W_diag_m[m:])
    else:
        # Full matrix
        lap = laplacian(N, dtype=type(P[0, 0].real))
        W = np.zeros_like(P)
        dot_cpu_(lap, P, W)

    return W


def select_first(W):
    zeroind = (0,)*(W.ndim-2) + (Ellipsis,)
    return np.ascontiguousarray(W[zeroind])


def select_sum(W):
    return W.sum(axis=tuple(range(W.ndim-2)))


def solve_poisson(W, reduce=select_first):
    """
    Return stream matrix `P` for `W`.

    Parameters
    ----------
    W: ndarray(shape=(N, N) or (k, N, N), dtype=complex)
    reduce: callable(W)

    Returns
    -------
    P: ndarray(shape=(N, N), dtype=complex)
    """
    if W.ndim >= 3:
        W = reduce(W)
    W_shape = W.shape
    N = W_shape[-1]
    

    # if N not in _cpu_buffer_cache or W.dtype != _cpu_buffer_cache[N]['complex'].dtype:
    #     allocate_buffer(W)

    if isspmatrix_dia(W):
        # Sparse diamatrix
        if hasattr(W, "el") and W.el > 0:
            # This is a hack to keep track of "pure el" dia_matrix objects.
            # It makes the calculations of laplace and solve_poisson much faster.
            # In the future, one should probably make a subclass
            # of dia_matrix instead.
            el = W.el
            P = W.copy()
            P /= -el*(el+1)
            P.el = el  # The new dia_matrix will have the same el.
        else:
            lap = laplacian(N, bc=True, dtype=np.float32 if W.dtype == np.complex64 else np.float64)
            P = W.copy()
            for W_diag_m, P_diag_m, m in zip(W.data, P.data, W.offsets):
                if m < 0:
                    solve_cpu_m_diag_(lap, W_diag_m[:N+m], P_diag_m[:N+m])
                else:
                    solve_cpu_m_diag_(lap, W_diag_m[m:], P_diag_m[m:])
    else:
        lap = laplacian(N, bc=True, dtype=type(W[0, 0].real))
        P = _get_cache(W_shape, W.dtype)
        solve_cpu_(lap, W, P,
                # _cpu_buffer_cache[N]['float'],
                _get_cache(W_shape, W.dtype, "real"),
                # _cpu_buffer_cache[N]['complex']
                _get_cache(W_shape, W.dtype, "complex"),
                )

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
