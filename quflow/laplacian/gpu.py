import numpy as np
from numba import njit, prange

# ----------------
# GLOBAL VARIABLES
# ----------------

_gpu_laplacian_cache = dict()
_gpu_buffer_cache = dict()
_gpu_heat_cache = dict()
_gpu_helmholtz_cache = dict()
_gpu_viscdamp_cache = dict()


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
def compute_gpu_laplacian_(N, bc=False, dtype=np.float64):
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
def dot_gpu_generic_(lap, P, W):
    N = P.shape[0]
    for i in prange(N):
        for j in range(N):
            W[i, j] = lap[i, j, 0]*P[i, j]
            if i < N-1 and j < N-1:
                W[i, j] += lap[i+1, j+1, 1]*P[i+1, j+1]
            if i > 0 and j > 0:
                W[i, j] += lap[i, j, 1]*P[i-1, j-1]
    return W


# Set default dot product
dot_gpu_ = dot_gpu_generic_


@njit(parallel=True, error_model='numpy', fastmath=True)
def solve_gpu_generic_(lap, W, P, buffer_float, buffer_complex):
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

    lap_flat = lap.reshape((N**2, 2))
    W_flat = W.ravel()
    P_flat = P.ravel()
    buffer_float_flat = buffer_float.ravel()
    buffer_complex_flat = buffer_complex.ravel()

    # assert lap_flat.flags.c_contiguous
    # assert W_flat.flags.c_contiguous
    # assert P_flat.flags.c_contiguous

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

        # Initialize buffers
        # i, j = mk2ij(m, 0)

        # First element: k = m
        buffer_float_flat[m] = lap_flat[m, 0]
        buffer_complex_flat[m] = W_flat[m]

        # Forward sweep according to grid-stride
        for k in range(m + stride, N**2, stride):
            # i, j = mk2ij(m, k)
            # im, jm = i-1, j-1

            w = lap_flat[k, 1]/buffer_float_flat[k-stride]
            buffer_float_flat[k] = lap_flat[k, 0] - w*lap_flat[k, 1]
            buffer_complex_flat[k] = W_flat[k] - w*buffer_complex_flat[k-stride]
            k_last = k

        # Backward sweep according to grid-stride
        # i, j = mk2ij(m, N-m-1)
        P_flat[k_last] = buffer_complex_flat[k_last]/buffer_float_flat[k_last]
        for k in range(k_last - stride, -1, -stride):
            # i, j = mk2ij(m, k)
            # ip, jp = i+1, j+1
            P_flat[k] = (buffer_complex_flat[k] - lap_flat[k+stride, 1]*P_flat[k+stride])/buffer_float_flat[k]

    # Make sure the trace of P vanishes (corresponds to bc for laplacian)
    trP = P_flat[::N+1].sum()
    trP /= N
    P_flat[::N+1] -= trP

    # trP = P[0, 0]
    # for k in range(1, N):
    #     trP += P[k, k]
    # trP /= N
    # for k in range(N):
    #     P[k, k] -= trP

    return P


@njit(parallel=True, error_model='numpy', fastmath=True)
def solve_gpu_generic2_(lap, W, P, buffer_float, buffer_complex):
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


@njit(parallel=True, error_model='numpy', fastmath=True)
def solve_gpu_generic3_(lap, W, P, buffer_float, buffer_complex):
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

    lap_flat = lap.reshape((N**2, 2))
    W_flat = W.ravel()
    P_flat = P.ravel()
    buffer_float_flat = buffer_float.ravel()
    buffer_complex_flat = buffer_complex.ravel()

    lap_rec = lap.ravel()[:2*N**2-2].reshape((N-1, N+1, 2))
    W_rec = W_flat[:N**2-1].reshape((N-1, N+1))
    P_rec = P_flat[:N**2-1].reshape((N-1, N+1))
    buffer_float_rec = buffer_float_flat[:N**2-1].reshape((N-1, N+1))
    buffer_complex_rec = buffer_complex_flat[:N**2-1].reshape((N-1, N+1))

    # Set array stride. This should be the grid-stride on the GPU.
    # stride = N + 1

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
    # which gets maps to the rectangular (N-1)x(N+1) matrix
    #
    # 0 1 2 3 4 5 6
    # 0 1 2 3 4 5 6
    # 0 1 2 3 4 5 6
    # 0 1 2 3 4 5 6
    # 0 1 2 3 4 5 6
    # 0
    #
    # Notice the extra element 0, which is not in the rectangular matrix
    # and must therefore be treated individually.
    #
    for m in prange(N+1):

        # First forward sweep
        buffer_float_rec[0, m] = lap_rec[0, m, 0]
        buffer_complex_rec[0, m] = W_rec[0, m]

        # Forward sweep according to grid-stride
        for k in range(1, N-1):
            w = lap_rec[k, m, 1]/buffer_float_rec[k-1, m]
            buffer_float_rec[k, m] = lap_rec[k, m, 0] - w*lap_rec[k, m, 1]
            buffer_complex_rec[k, m] = W_rec[k, m] - w*buffer_complex_rec[k-1, m]

        # Set last element
        if m == 0:
            # Last forward sweep
            w = lap_flat[-1, 1]/buffer_float_rec[N-2, m]
            buffer_float_flat[-1] = lap_flat[-1, 0] - w*lap_flat[-1, 1]
            buffer_complex_flat[-1] = W_flat[-1] - w*buffer_complex_rec[N-2, m]

            # First backward sweep
            P_flat[-1] = buffer_complex_flat[-1]/buffer_float_flat[-1]
            P_rec[-1, m] = (buffer_complex_rec[-1, m] - lap_flat[-1, 1]*P_flat[-1])/buffer_float_rec[-1, m]
        else:
            # Backward sweep according to grid-stride
            P_rec[-1, m] = buffer_complex_rec[-1, m]/buffer_float_rec[-1, m]

        for k in range(N-3, -1, -1):
            P_rec[k, m] = (buffer_complex_rec[k, m] - lap_rec[k+1, m, 1]*P_rec[k+1, m])/buffer_float_rec[k, m]

    # Make sure the trace of P vanishes (corresponds to bc for laplacian)
    trP = P_flat[::N+1].sum()
    trP /= N
    P_flat[::N+1] -= trP

    return P


@njit(parallel=True, error_model='numpy', fastmath=True)
def solve_gpu_generic4_(lap, W, P, buffer_float, buffer_complex):
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

    lap_flat = lap.reshape((N**2, 2))
    W_flat = W.ravel()
    P_flat = P.ravel()
    buffer_float_flat = buffer_float.ravel()
    buffer_complex_flat = buffer_complex.ravel()

    W_rec = W_flat[:N**2-1].reshape((N-1, N+1))
    P_rec = P_flat[:N**2-1].reshape((N-1, N+1))
    buffer_float_rec = buffer_float_flat[:N**2-1].reshape((N-1, N+1))
    buffer_complex_rec = buffer_complex_flat[:N**2-1].reshape((N-1, N+1))
    lap_rec = lap_flat.ravel()[:2*N**2-2].reshape((N-1, N+1, 2))

    # Set array stride. This should be the grid-stride on the GPU.
    # stride = N + 1

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
    # which gets maps to the rectangular (N-1)x(N+1) matrix
    #
    # 0 1 2 3 4 5 6
    # 0 1 2 3 4 5 6
    # 0 1 2 3 4 5 6
    # 0 1 2 3 4 5 6
    # 0 1 2 3 4 5 6
    # 0
    #
    # Notice the extra element 0, which is not in the rectangular matrix
    # and must therefore be treated individually.
    #
    # for m in prange(N+1):

    # First forward sweep
    buffer_float_rec[0, :] = lap_rec[0, :, 0]
    buffer_complex_rec[0, :] = W_rec[0, :]

    # Forward sweep according to grid-stride
    for k in range(1, N-1):
        w = lap_rec[k, :, 1]/buffer_float_rec[k-1, :]
        buffer_float_rec[k, :] = lap_rec[k, :, 0] - w*lap_rec[k, :, 1]
        buffer_complex_rec[k, :] = W_rec[k, :] - w*buffer_complex_rec[k-1, :]

    # Set last element

    # Last forward sweep
    w0 = lap_flat[-1, 1]/buffer_float_rec[N-2, 0]
    buffer_float_flat[-1] = lap_flat[-1, 0] - w0*lap_flat[-1, 1]
    buffer_complex_flat[-1] = W_flat[-1] - w0*buffer_complex_rec[N-2, 0]

    # First backward sweep
    P_flat[-1] = buffer_complex_flat[-1]/buffer_float_flat[-1]
    P_rec[-1, 0] = (buffer_complex_rec[-1, 0] - lap_flat[-1, 1]*P_flat[-1])/buffer_float_rec[-1, 0]

    # Backward sweep according to grid-stride
    P_rec[-1, 1:] = buffer_complex_rec[-1, 1:]/buffer_float_rec[-1, 1:]

    for k in range(N-3, -1, -1):
        P_rec[k, :] = (buffer_complex_rec[k, :] - lap_rec[k+1, :, 1]*P_rec[k+1, :])/buffer_float_rec[k, :]

    # Make sure the trace of P vanishes (corresponds to bc for laplacian)
    trP = P_flat[::N+1].sum()
    trP /= N
    P_flat[::N+1] -= trP

    return P


# Set default solver
solve_gpu_ = solve_gpu_generic2_


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
    global solve_gpu_
    global dot_gpu_

    pass


def allocate_buffer(W):
    global _gpu_buffer_cache
    N = W.shape[0]

    buffer_complex = W.copy()
    buffer_float = W.real.copy()
    P_out = W.copy()
    _gpu_buffer_cache[(N, W.dtype)] = {'complex': buffer_complex, 'float': buffer_float, 'P': P_out}


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
    global _gpu_laplacian_cache

    if (N, bc, dtype) not in _gpu_laplacian_cache:
        lap = compute_gpu_laplacian_(N, bc=bc, dtype=dtype)
        _gpu_laplacian_cache[(N, bc, dtype)] = lap

    return _gpu_laplacian_cache[(N, bc, dtype)]


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
    W = P.copy()
    dot_gpu_(lap, P, W)

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
    lap = laplacian(N, bc=True, dtype=type(W[0, 0].real))

    if (N, W.dtype) not in _gpu_buffer_cache or W.dtype != _gpu_buffer_cache[(N, W.dtype)]['complex'].dtype:
        allocate_buffer(W)

    P = _gpu_buffer_cache[(N, W.dtype)]['P']
    solve_gpu_(lap, W, P,
               _gpu_buffer_cache[(N, W.dtype)]['float'],
               _gpu_buffer_cache[(N, W.dtype)]['complex'])

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
    global _gpu_heat_cache

    N = W0.shape[0]

    if (N, h_times_nu) not in _gpu_heat_cache:
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
        _gpu_heat_cache[(N, h_times_nu)] = heat
    else:
        heat = _gpu_heat_cache[(N, h_times_nu)]

    Wh = _gpu_buffer_cache[(N, W0.dtype)]['P']
    solve_gpu_(heat, W0, Wh,
               _gpu_buffer_cache[(N, W0.dtype)]['float'],
               _gpu_buffer_cache[(N, W0.dtype)]['complex'])

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
    global _gpu_helmholtz_cache

    N = W.shape[0]

    if (N, alpha) not in _gpu_helmholtz_cache:

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
        _gpu_helmholtz_cache[(N, alpha)] = helmholtz
    else:
        helmholtz = _gpu_helmholtz_cache[(N, alpha)]

    P = _gpu_buffer_cache[(N, W.dtype)]['P']
    solve_gpu_(helmholtz, W, P,
               _gpu_buffer_cache[(N, W.dtype)]['float'],
               _gpu_buffer_cache[(N, W.dtype)]['complex'])

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
    global _gpu_viscdamp_cache

    N = W0.shape[0]

    if (N, h, nu, alpha) not in _gpu_viscdamp_cache:

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
        _gpu_viscdamp_cache[(N, h, nu, alpha)] = viscdamp
    else:
        viscdamp = _gpu_viscdamp_cache[(N, h, nu, alpha)]

    Wh = _gpu_buffer_cache[(N, W0.dtype)]['P']

    # Prepare right hand side in Crank-Nicolson
    if theta == 1:
        Wrhs = W0.copy()
    else:
        Wrhs = (1.0-alpha*h*(1-theta))*W0
        Wrhs += (nu*h*(1-theta))*laplace(W0)
    if force is not None:
        Wrhs += h*force

    solve_gpu_(viscdamp, Wrhs, Wh,
               _gpu_buffer_cache[(N, W0.dtype)]['float'],
               _gpu_buffer_cache[(N, W0.dtype)]['complex'])

    return Wh
