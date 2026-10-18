r"""
Quantized Laplacian on the CPU, in a tridiagonal representation.

This is the default Laplacian backend: :mod:`quflow.laplacian` re-exports the
public functions of this module, so for example ``quflow.solve_poisson`` is
:func:`solve_poisson` below.

The quantized (Hoppe-Yau) Laplacian acts on complex :math:`N\times N`
matrices as

.. math::

    \Delta_N P = \sum_{a=1}^{3} [S_a, [S_a, P]]
               = \frac{1}{\hbar^2}\sum_{a=1}^{3} [X_a, [X_a, P]],

where :math:`S_a` are the so(3) generators of
:func:`quflow.geometry.so3_generators` and :math:`X_a = \hbar S_a` are the
matrices of :func:`quflow.geometry.cartesian_generators`. The quantized
spherical harmonics are eigenvectors,
:math:`\Delta_N T_{\ell m} = -\ell(\ell+1)\, T_{\ell m}`.

:math:`\Delta_N` maps each diagonal of `P` (the entries :math:`P_{ij}` with
:math:`j - i = m`, for :math:`-N < m < N`) to itself, and acts on it as a
symmetric tridiagonal matrix. This module stores :math:`\Delta_N` as an array
`lap` of shape ``(N, N, 2)`` (see :func:`laplacian`): ``lap[i, j, 0]`` is the
coefficient of :math:`P_{ij}` in :math:`(\Delta_N P)_{ij}`, and
``lap[i, j, 1]`` is the coupling between the entries ``(i, j)`` and
``(i-1, j-1)``. Both applying and inverting :math:`\Delta_N` therefore cost
:math:`O(N^2)` operations; the solvers use the Thomas algorithm on each
diagonal, in numba-compiled kernels.

On the main diagonal (:math:`m = 0`) the Laplacian is singular: its kernel is
spanned by the identity matrix, i.e. the :math:`\ell = 0` mode (the trace).
:func:`solve_poisson` returns the unique trace-free :math:`P` with
:math:`\Delta_N P = W - \tfrac{\operatorname{tr} W}{N} I`.

Notes
-----
For speed, several functions return internal arrays that are overwritten by
later calls (see the individual functions). Copy the result if it must be kept.

By default the solvers assume skew-Hermitian matrices; see :func:`select_skewherm`.
"""
import numpy as np
from numba import njit, prange
import scipy.sparse as sp
from scipy.sparse import isspmatrix_dia
from ..geometry import cartesian_generators

# ----------------
# GLOBAL VARIABLES
# ----------------

_cpu_laplacian_cache = dict()
_cpu_buffer_cache = dict()
_cpu_heat_cache = dict()
_cpu_helmholtz_cache = dict()
_cpu_globalqg_cache = dict()
_cpu_viscdamp_cache = dict()
_cpu_cache = dict()


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------

def _get_cache(shape, dtype, *args):
    """
    Return a cached, zero-initialized work array.

    The same array is returned at every call with the same arguments.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array.
    dtype : dtype
        Data type of the array.
    *args
        Extra keys, to keep several arrays of the same shape and dtype. If
        ``"real"`` is among them, the array is the real part (a view) of a
        complex array of the given dtype.

    Returns
    -------
    ndarray
    """
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
    """
    Convert diagonal coordinates ``(m, k)`` to matrix indices ``(i, j)``.

    Here ``m = j - i`` is the diagonal and ``k = min(i, j)`` is the position
    along it. Inverse of :func:`ij2mk`.

    Parameters
    ----------
    m : int
        Diagonal, ``-N < m < N``.
    k : int
        Position along the diagonal, ``0 <= k < N - |m|``.

    Returns
    -------
    i, j : int
        Row and column index.
    """
    if m >= 0:
        i = k
        j = k + m
    else:
        i = k - m
        j = k
    return i, j


@njit
def ij2mk(i, j):
    """
    Convert matrix indices ``(i, j)`` to diagonal coordinates ``(m, k)``.

    Here ``m = j - i`` is the diagonal and ``k = min(i, j)`` is the position
    along it. Inverse of :func:`mk2ij`.

    Parameters
    ----------
    i, j : int
        Row and column index.

    Returns
    -------
    m, k : int
        Diagonal and position along it.
    """
    m = j-i
    if m >= 0:
        k = i
    else:
        k = j
    return m, k


@njit
def _compute_cpu_laplacian(N, bc=False, dtype=np.float64):
    r"""
    Compute tridiagonal laplacian.

    With ``(m, k)`` the diagonal coordinates of entry ``(i, j)`` (see
    :func:`mk2ij`) the coefficients are

    .. math::

        \text{lap}[i, j, 0] &= -\big((N-1)(2k + 1 + |m|) - 2k(k + |m|)\big), \\
        \text{lap}[i, j, 1] &= \sqrt{(k + |m|)(N - k - |m|)\,k\,(N - k)}.

    Parameters
    ----------
    N: int
        Quantization matrix size.
    bc: bool (optional)
        Whether boundary conditions should be added to make the laplacian non-singular.
        Notice that this bc is different from the one used for sparse laplacians.
        It subtracts 0.5 from ``lap[0, 0, 0]``, which makes the operator on
        the main diagonal (singular without it) definite.
    dtype: float type

    Returns
    -------
    lap: real array (N, N, 2)
        Two outer indices: corresponding to matrix entries in W.
        Inner index: 0: 0th diagonal, 1: 1:st and -1:st diagonal.
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
        # Old bc
        # lap[0, 0, 0] += 0.5

        # Better bc (makes matrix definite)
        lap[0, 0, 0] -= 0.5

        # Experimental bc (not working yet)
        # lap[-1, -1, 1] = 0.0

    return lap


@njit(parallel=False, error_model='numpy', fastmath=True)
def _dot_cpu_generic(lap, P, W):
    """
    Apply a tridiagonal operator to an arbitrary matrix: ``W = lap(P)``.

    Default kernel of :func:`laplace`.

    Parameters
    ----------
    lap : ndarray, shape (N, N, 2), float
        Tridiagonal operator (typically the Laplacian).
    P : ndarray, shape (N, N), complex
        Input matrix.
    W : ndarray, shape (N, N), complex
        Output matrix (overwritten).

    Returns
    -------
    W : ndarray
        The output array.
    """
    N = P.shape[0]
    for i in prange(N):
        for j in range(N):
            W[i, j] = lap[i, j, 0]*P[i, j]
            if i < N-1 and j < N-1:
                W[i, j] += lap[i+1, j+1, 1]*P[i+1, j+1]
            if i > 0 and j > 0:
                W[i, j] += lap[i, j, 1]*P[i-1, j-1]
    return W


@njit(parallel=False, error_model='numpy', fastmath=True)
def _dot_cpu_skewh2(lap, P, W):
    """
    Apply a tridiagonal operator to a skew-Hermitian matrix (unused variant).

    Computes the strict lower triangle of ``W = lap(P)`` and fills the upper
    triangle by skew-Hermitian symmetry. Note that the diagonal of `W` is not
    computed. Use :func:`_dot_cpu_skewh` instead.

    Parameters
    ----------
    lap : ndarray, shape (N, N, 2), float
        Tridiagonal operator.
    P : ndarray, shape (N, N), complex
        Input skew-Hermitian matrix.
    W : ndarray, shape (N, N), complex
        Output matrix (overwritten, except its diagonal).

    Returns
    -------
    W : ndarray
        The output array.
    """
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


@njit(parallel=False, error_model='numpy', fastmath=True)
def _dot_cpu_nonskewh(lap, P, W):
    """
    Dot product for tridiagonal operator.

    Computes ``W = lap(P)`` diagonal by diagonal, for an arbitrary matrix `P`.

    Parameters
    ----------
    lap: ndarray(shape(N, N, 2), dtype=float)
        Tridiagonal operator (typically laplacian).
    P: ndarray(shape=(N,N), dtype=complex)
        Input matrix.
    W: ndarray(shape=(N,N), dtype=complex)
        Output matrix (overwritten).

    Returns
    -------
    W : ndarray
        The output array.
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


@njit(parallel=True, error_model='numpy', fastmath=True)
def _dot_cpu_skewh(lap, P, W):
    """
    Dot product for tridiagonal operator, for skew-Hermitian P.

    Computes the diagonals ``m >= 0`` of ``W = lap(P)`` from the upper
    triangle of `P` (the lower triangle is not read), and fills the lower
    triangle of `W` by skew-Hermitian symmetry.

    Parameters
    ----------
    lap: ndarray(shape(N, N, 2), dtype=float)
        Tridiagonal operator (typically laplacian).
    P: ndarray(shape=(N,N), dtype=complex)
        Input matrix.
    W: ndarray(shape=(N,N), dtype=complex)
        Output matrix (overwritten).

    Returns
    -------
    W : ndarray
        The output array.
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
    for i in prange(N):
        for j in range(i):
            W[i, j] = -np.conj(W[j, i])

    return W


# Set default dot product
# _dot_cpu = _dot_cpu_skewh
_dot_cpu = _dot_cpu_generic


@njit(parallel=True, error_model='numpy', fastmath=True)
def _solve_cpu_nonskewh(lap, W, P, buffer_float, buffer_complex):
    """
    Function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` array). Uses NUMBA to accelerate the
    Thomas algorithm for tridiagonal solver calculations.

    Solves all ``2N - 1`` diagonals, for an arbitrary matrix `W`.

    Parameters
    ----------
    lap: ndarray(shape=(N, N, 2), dtype=float)
        Tridiagonal laplacian.
    W: ndarray(shape=(N, N), dtype=complex)
        Input matrix.
    P: ndarray(shape=(N, N), dtype=complex)
        Output matrix (overwritten).
    buffer_float: ndarray(shape=(N, N), dtype=float)
        Float buffer.
    buffer_complex: ndarray(shape=(N, N), dtype=complex)
        Complex buffer.

    Returns
    -------
    P : ndarray
        The output array.

    Notes
    -----
    On the main diagonal (``m = 0``), the mean of the diagonal of `W`
    (``trace(W)/N``) is subtracted from the right-hand side, and the trace of
    the solution is removed afterwards. With `lap` from
    ``laplacian(N, bc=True)`` this gives the trace-free solution of the
    Poisson equation. The same is done for any other operator `lap`.
    """
    N = W.shape[0]

    # For each m-diagonal in W, solve a tridiagonal system with Thomas algorithm
    for m in prange(-N+1, N):

        absm = np.abs(m)
        
        # Initialize buffers
        i, j = mk2ij(m, 0)
        buffer_float[i, j] = lap[i, j, 0]
        buffer_complex[i, j] = W[i, j]
        if m==0:
            # Compute circulation (scaled trace) of W matrix
            trW = W[0, 0]
            for k in range(1, N):
                trW += W[k, k]
            trW /= N        
            buffer_complex[i, j] -= trW

        # Forward sweep
        for k in range(1, N-absm):
            i, j = mk2ij(m, k)
            im, jm = i-1, j-1

            w = lap[i, j, 1]/buffer_float[im, jm]
            buffer_float[i, j] = lap[i, j, 0] - w*lap[i, j, 1]
            buffer_complex[i, j] = W[i, j] - w*buffer_complex[im, jm]
            if m==0:
                buffer_complex[i, j] -= trW

        # Backward sweep
        i, j = mk2ij(m, N-absm-1)
        P[i, j] = buffer_complex[i, j]/buffer_float[i, j]
        # P[j, i] = np.conj(P[i, j])
        for k in range(N-absm-2, -1, -1):
            i, j = mk2ij(m, k)
            ip, jp = i+1, j+1
            P[i, j] = (buffer_complex[i, j] - lap[ip, jp, 1]*P[ip, jp])/buffer_float[i, j]
            # P[j, i] = np.conj(P[i, j])

        if m==0:
            # Make sure the trace of P vanishes (corresponds to bc for laplacian)
            trP = P[0, 0]
            for k in range(1, N):
                trP += P[k, k]
            trP /= N
            for k in range(N):
                P[k, k] -= trP

    # # Make sure the trace of P vanishes (corresponds to bc for laplacian)
    # trP = P[0, 0]
    # for k in range(1, N):
    #     trP += P[k, k]
    # trP /= N
    # for k in range(N):
    #     P[k, k] -= trP

    return P


@njit(parallel=True, error_model='numpy', fastmath=True)
def _solve_cpu_skewh(lap, W, P, buffer_float, buffer_complex):
    """
    Function for solving the quantized
    Poisson equation (or more generally the equation defined by
    the `lap` array). Uses NUMBA to accelerate the
    Thomas algorithm for tridiagonal solver calculations.

    Version for skew-Hermitian `W` (the default solver): only the diagonals
    ``m >= 0`` are solved, reading the upper triangle of `W` (the lower
    triangle is not read), and the lower triangle of `P` is filled by
    skew-Hermitian symmetry.

    Parameters
    ----------
    lap: ndarray(shape=(N, N, 2), dtype=float)
        Tridiagonal laplacian.
    W: ndarray(shape=(N, N), dtype=complex)
        Input matrix.
    P: ndarray(shape=(N, N), dtype=complex)
        Output matrix (overwritten).
    buffer_float: ndarray(shape=(N, N), dtype=float)
        Float buffer.
    buffer_complex: ndarray(shape=(N, N), dtype=complex)
        Complex buffer.

    Returns
    -------
    P : ndarray
        The output array.

    Notes
    -----
    On the main diagonal (``m = 0``), the mean of the diagonal of `W`
    (``trace(W)/N``) is subtracted from the right-hand side, and the trace of
    the solution is removed afterwards. With `lap` from
    ``laplacian(N, bc=True)`` this gives the trace-free solution of the
    Poisson equation. The same is done for any other operator `lap`.
    """
    N = W.shape[0]

    # For each m-diagonal in W, solve a tridiagonal system with Thomas algorithm
    for m in prange(N):

        # Initialize buffers
        i, j = mk2ij(m, 0)
        buffer_float[i, j] = lap[i, j, 0]
        buffer_complex[i, j] = W[i, j]
        if m==0:
            # Compute circulation (scaled trace) of W matrix
            trW = W[0, 0]
            for k in range(1, N):
                trW += W[k, k]
            trW /= N        
            buffer_complex[i, j] -= trW

        # Forward sweep
        for k in range(1, N-m):
            i, j = mk2ij(m, k)
            im, jm = i-1, j-1

            w = lap[i, j, 1]/buffer_float[im, jm]
            buffer_float[i, j] = lap[i, j, 0] - w*lap[i, j, 1]
            buffer_complex[i, j] = W[i, j] - w*buffer_complex[im, jm]
            if m==0:
                buffer_complex[i, j] -= trW

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
        
        if m==0:
            # Experimental bc
            # P[-1, -1] = 0.0

            # Make sure the trace of P vanishes (corresponds to bc for laplacian)
            trP = P[0, 0]
            for k in range(1, N):
                trP += P[k, k]
            trP /= N
            for k in range(N):
                P[k, k] -= trP

    # # Make sure the trace of P vanishes (corresponds to bc for laplacian)
    # trP = P[0, 0]
    # for k in range(1, N):
    #     trP += P[k, k]
    # trP /= N
    # for k in range(N):
    #     P[k, k] -= trP

    return P


@njit(parallel=True, error_model='numpy', fastmath=True)
def _solve_cpu_generic(lap, W, P, buffer_float, buffer_complex):
    """
    Solve the tridiagonal system for an arbitrary matrix, in a flat layout.

    Alternative to :func:`_solve_cpu_nonskewh` (not used by default). The
    arrays are traversed in their flattened form with stride ``N + 1``, so
    that each of the ``N + 1`` sweeps covers an upper diagonal ``m``
    followed by the lower diagonal ``m - N`` (the two are decoupled since the
    coupling coefficient vanishes at the start of a diagonal). This mirrors a
    grid-stride loop on a GPU. Trace handling on the main diagonal is as in
    :func:`_solve_cpu_skewh`. Gives wrong results for ``N = 2``.

    Parameters
    ----------
    lap: ndarray(shape=(N, N, 2), dtype=float)
        Tridiagonal laplacian.
    W: ndarray(shape=(N, N), dtype=complex)
        Input matrix (C-contiguous).
    P: ndarray(shape=(N, N), dtype=complex)
        Output matrix (overwritten; C-contiguous).
    buffer_float: ndarray(shape=(N, N), dtype=float)
        Float buffer.
    buffer_complex: ndarray(shape=(N, N), dtype=complex)
        Complex buffer.

    Returns
    -------
    P : ndarray
        The output array.
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
        if m == 0:
            # Compute scaled trace
            trW = W_flat[0]
            for k in range(stride, N**2, stride):
                trW += W_flat[k]
            trW /= N

            # Correct diagonal
            val_complex -= trW

        buffer_float_flat[m] = val_float
        buffer_complex_flat[m] = val_complex            

        # Forward sweep according to grid-stride
        for k in range(m + stride, N**2, stride):
            w = lap_flat[k, 1]/val_float
            val_float = lap_flat[k, 0] - w*lap_flat[k, 1]
            val_complex = W_flat[k] - w*val_complex
            if m == 0:
                # Correct diagonal
                val_complex -= trW
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

        if m == 0:
            # Make sure the trace of P vanishes (corresponds to bc for laplacian)
            trP = P_flat[::stride].sum()
            trP /= N
            P_flat[::stride] -= trP

    return P


# Set default solver
_solve_cpu = _solve_cpu_skewh
# _solve_cpu = _solve_cpu_generic


@njit(error_model='numpy', fastmath=True)
def _dot_cpu_m_diag(lap, P_m_diag, W_m_diag,):
    """
    Dot product for tridiagonal operator applied to m diagonal.

    Used by :func:`laplace` for sparse ``dia_matrix`` input. The diagonal
    ``|m|`` is determined from the length of the input, ``N - |m|``; since
    the coefficients depend only on ``|m|``, the same call serves the
    diagonals ``m`` and ``-m``.

    Parameters
    ----------
    lap: ndarray(shape(N, N, 2), dtype=float)
        Tridiagonal operator (typically laplacian).
    P_m_diag: ndarray(shape=(N-m,), dtype=complex)
        Input diagonal.
    W_m_diag: ndarray(shape=(N-m,), dtype=complex)
        Output diagonal (overwritten).

    Returns
    -------
    W_m_diag : ndarray
        The output array.
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
def _solve_cpu_m_diag(lap, W_m_diag, P_m_diag):
    """
    Solve Poisson equation for only m diagonal.

    Used by :func:`solve_poisson` for sparse ``dia_matrix`` input. The
    diagonal ``|m|`` is determined from the length of the input, ``N - |m|``.

    Parameters
    ----------
    lap: ndarray(shape=(N, N, 2), dtype=float)
        Tridiagonal laplacian (with ``bc=True`` for the Poisson equation).
    W_m_diag: ndarray(shape=(N-m,), dtype=complex)
        Input diagonal.
    P_m_diag: ndarray(shape=(N-m,), dtype=complex)
        Output diagonal (overwritten).

    Returns
    -------
    P_m_diag : ndarray
        The output array.

    Notes
    -----
    For the main diagonal (``m = 0``) the trace of the solution is removed,
    but, unlike the dense solvers, the trace of the right-hand side is not
    subtracted first. For input with nonzero trace, the result therefore
    differs from :func:`_solve_cpu_skewh`.
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

    In skew-Hermitian mode (the default), the solvers use
    :func:`_solve_cpu_skewh`, which reads only the upper triangle of the
    right-hand side. Otherwise :func:`_solve_cpu_nonskewh` is used, which
    works for arbitrary matrices. The setting is global for this module.
    It is also set by :func:`quflow.integrators.isospectral.select_skewherm`.

    :func:`laplace` uses a kernel that is correct for arbitrary matrices,
    and is not changed by this setting.

    Parameters
    ----------
    flag: bool
        ``True`` for skew-Hermitian mode, ``False`` for arbitrary matrices.

    Returns
    -------
    previous flag (bool)
    """
    global _solve_cpu
    global _dot_cpu
    oldflag = False
    if _solve_cpu is _solve_cpu_skewh:
        oldflag = True

    if flag:
        _solve_cpu = _solve_cpu_skewh
        if _dot_cpu is _dot_cpu_nonskewh:
            _dot_cpu = _dot_cpu_skewh
    else:
        _solve_cpu = _solve_cpu_nonskewh
        if _dot_cpu is _dot_cpu_skewh:
            _dot_cpu = _dot_cpu_nonskewh

    return oldflag


def allocate_buffer(W):
    """
    Allocate the work and output buffers for matrices like `W`.

    The buffers (keys ``'complex'``, ``'float'`` and ``'P'``) are stored in
    ``_cpu_buffer_cache[N]``, replacing earlier ones for the same `N`, and are
    used by :func:`solve_heat`, :func:`solve_helmholtz`,
    :func:`solve_globalqg` and :func:`solve_viscdamp`.

    Parameters
    ----------
    W : ndarray, shape (N, N), complex
        Template for shape and dtype (its values are irrelevant).

    Returns
    -------
    None
    """
    global _cpu_buffer_cache
    N = W.shape[0]

    buffer_complex = W.copy()
    buffer_float = W.real.copy()
    P_out = W.copy()
    _cpu_buffer_cache[N] = {'complex': buffer_complex, 'float': buffer_float, 'P': P_out}


def laplacian(N, bc=False, dtype=np.float64):
    """
    Return quantized laplacian (as a tridiagonal laplacian).

    The array is computed by :func:`_compute_cpu_laplacian` and cached; see
    the module docstring for its layout.

    Parameters
    ----------
    N: int
        Size of the matrices.
    bc: bool (optional)
        Whether to include boundary conditions, which make the operator on the
        main diagonal non-singular (used for solving). Default is False.
    dtype: float type
        Default is ``np.float64``.

    Returns
    -------
    lap : ndarray(shape=(N, N, 2), dtype=float)
        The cached array (shared between calls; do not modify it).
    """
    global _cpu_laplacian_cache

    if (N, bc, dtype) not in _cpu_laplacian_cache:
        lap = _compute_cpu_laplacian(N, bc=bc, dtype=dtype)
        _cpu_laplacian_cache[(N, bc, dtype)] = lap

    return _cpu_laplacian_cache[(N, bc, dtype)]


def laplace(P):
    r"""
    Return quantized laplacian applied to stream function `P`.

    Computes :math:`W = \Delta_N P` (see the module docstring). Works for
    arbitrary (not necessarily skew-Hermitian) matrices.

    Parameters
    ----------
    P: ndarray(shape=(N, N), dtype=complex) or scipy.sparse.dia_matrix
        Matrix to apply the Laplacian to. A sparse ``dia_matrix`` is handled
        diagonal by diagonal. If it has an attribute ``el`` (as set by
        :func:`quflow.elmr2mat` and :func:`quflow.elmc2mat` for matrices
        that contain only one wavenumber), the result is simply
        ``-el*(el+1)*P``.

    Returns
    -------
    W: ndarray(shape=(N, N), dtype=complex) or scipy.sparse.dia_matrix
        A new array, of the same kind as `P`.

    Examples
    --------
    >>> import quflow as qf
    >>> T = qf.elmr2mat(2, 1, N=8)
    >>> bool(abs(qf.laplacian.laplace(T.toarray()) + 6*T.toarray()).max() < 1e-12)
    True
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
                    _dot_cpu_m_diag(lap, P_diag_m[:N+m], W_diag_m[:N+m])
                else:
                    _dot_cpu_m_diag(lap, P_diag_m[m:], W_diag_m[m:])
    else:
        # Full matrix
        lap = laplacian(N, dtype=type(P[0, 0].real))
        W = np.zeros_like(P)
        _dot_cpu(lap, P, W)

    return W


def select_first(W):
    """
    Return the first matrix ``W[0, ..., 0, :, :]`` of a stack, as a contiguous array.

    Default `reduce` function of :func:`solve_poisson`.

    Parameters
    ----------
    W : ndarray, shape (..., N, N)

    Returns
    -------
    ndarray, shape (N, N)
    """
    zeroind = (0,)*(W.ndim-2) + (Ellipsis,)
    return np.ascontiguousarray(W[zeroind])


def select_sum(W):
    """
    Return the sum of the matrices in a stack.

    Alternative `reduce` function for :func:`solve_poisson`.

    Parameters
    ----------
    W : ndarray, shape (..., N, N)

    Returns
    -------
    ndarray, shape (N, N)
    """
    return W.sum(axis=tuple(range(W.ndim-2)))


def solve_poisson(W, reduce=select_first):
    r"""
    Solve the Poisson equation
    :math:`\Delta_N P = W`,
    i.e. return the stream matrix `P` for `W`.

    The trace of `W` (its :math:`\ell = 0` component, which lies outside the
    range of :math:`\Delta_N`) is ignored: the result is the unique trace-free
    `P` with :math:`\Delta_N P = W - \tfrac{\operatorname{tr} W}{N} I`.
    The cost is :math:`O(N^2)`.

    Parameters
    ----------
    W: ndarray(shape=(N, N) or (k, N, N), dtype=complex) or scipy.sparse.dia_matrix
        Vorticity matrix. In skew-Hermitian mode (the default, see
        :func:`select_skewherm`) it must be skew-Hermitian, and only its upper
        triangle is read. A stack of matrices is first reduced to one matrix
        by `reduce`. For a sparse ``dia_matrix``, see :func:`laplace`; the
        sparse path does not subtract the trace of `W` first, so for sparse
        input with nonzero trace the result differs from the dense one.
    reduce: callable(W)
        Function reducing a stack of matrices to one matrix. Default is
        :func:`select_first` (use the first matrix); see also :func:`select_sum`.

    Returns
    -------
    P: ndarray(shape=(N, N), dtype=complex)
        Stream matrix. For dense input, this is an internal buffer (one per
        shape and dtype) that is overwritten by the next call with the same
        shape and dtype. Copy it if it must be kept.

    Examples
    --------
    >>> import numpy as np
    >>> import quflow as qf
    >>> W = qf.shr2mat(np.random.default_rng(0).standard_normal(64), N=8)
    >>> P = qf.solve_poisson(W).copy()
    >>> bool(abs(qf.laplacian.laplace(P) - (W - np.trace(W)/8*np.eye(8))).max() < 1e-12)
    True
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
                    _solve_cpu_m_diag(lap, W_diag_m[:N+m], P_diag_m[:N+m])
                else:
                    _solve_cpu_m_diag(lap, W_diag_m[m:], P_diag_m[m:])
    else:
        lap = laplacian(N, bc=True, dtype=type(W[0, 0].real))
        P = _get_cache(W_shape, W.dtype)
        _solve_cpu(lap, W, P,
                # _cpu_buffer_cache[N]['float'],
                _get_cache(W_shape, W.dtype, "real"),
                # _cpu_buffer_cache[N]['complex']
                _get_cache(W_shape, W.dtype, "complex"),
                )

    return P


def solve_heat(h_times_nu, W0):
    r"""
    Solve heat equation
    :math:`\dot W = \nu \Delta_N W`, :math:`W(0) = W_0`,
    using one step of backward Euler method, i.e.,
    :math:`(1 - h \nu \Delta_N) W_h = W_0`.

    Parameters
    ----------
    h_times_nu: float
        Stepsize (in qtime) times viscosity ν.
    W0: ndarray(shape=(N, N), dtype=complex)
        Initial matrix (not modified). In skew-Hermitian mode (the default)
        only its upper triangle is read.

    Returns
    -------
    Wh: ndarray(shape=(N, N), dtype=complex)
        Solution after one step. The result is an internal buffer, shared by :func:`solve_heat`,
    :func:`solve_helmholtz`, :func:`solve_globalqg` and :func:`solve_viscdamp`
    for each `N`: it is overwritten by the next call to any of them. Copy it
    if it must be kept.

    Notes
    -----
    The solver removes the trace (the :math:`\ell = 0` component) of the
    right-hand side and of the solution (see :func:`_solve_cpu_skewh`), so the
    result is trace-free even if `W0` is not.

    The operator and the output buffer are cached per `N` and parameter
    values, but not per dtype: after a first call with single precision
    input, later calls with double precision input are computed in single
    precision (and vice versa).
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
    _solve_cpu(heat, W0, Wh,
               _cpu_buffer_cache[N]['float'],
               _cpu_buffer_cache[N]['complex'])

    return Wh


def solve_helmholtz(W, alpha=1.0):
    r"""
    Solve the inhomogeneous Helmholtz equation
    :math:`(1 - \alpha \Delta_N) P = W`.

    Parameters
    ----------
    alpha: float
        Coefficient :math:`\alpha`. Default is 1.0.
    W: ndarray(shape=(N, N), dtype=complex)
        Right-hand side (not modified). In skew-Hermitian mode (the default)
        only its upper triangle is read.

    Returns
    -------
    P: ndarray(shape=(N, N), dtype=complex)
        Solution. The result is an internal buffer, shared by :func:`solve_heat`,
    :func:`solve_helmholtz`, :func:`solve_globalqg` and :func:`solve_viscdamp`
    for each `N`: it is overwritten by the next call to any of them. Copy it
    if it must be kept.

    Notes
    -----
    The solver removes the trace (the :math:`\ell = 0` component) of the
    right-hand side and of the solution (see :func:`_solve_cpu_skewh`), so the
    result is trace-free even if `W` is not.

    The operator and the output buffer are cached per `N` and parameter
    values, but not per dtype: after a first call with single precision
    input, later calls with double precision input are computed in single
    precision (and vice versa).
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
    _solve_cpu(helmholtz, W, P,
               _cpu_buffer_cache[N]['float'],
               _cpu_buffer_cache[N]['complex'])

    return P


def solve_globalqg(W, gamma=1.0):
    r"""
    Solve the inhomogeneous Helmholtz equation
    of the global quasi-geostrophic model for the stream matrix.

    The equation solved is

    .. math::

        \Delta_N P + \frac{\gamma}{2}\big(Z^2 P + P Z^2\big) = W,

    where :math:`Z = X_3` is the third matrix of
    :func:`quflow.geometry.cartesian_generators` (the quantization of the
    coordinate function :math:`x_3`). Since :math:`Z` is diagonal, the
    operator keeps the tridiagonal structure.

    Parameters
    ----------
    gamma: float
        Coefficient :math:`\gamma`. Default is 1.0.
    W: ndarray(shape=(N, N), dtype=complex)
        Right-hand side (not modified). In skew-Hermitian mode (the default)
        only its upper triangle is read.

    Returns
    -------
    P: ndarray(shape=(N, N), dtype=complex)
        Solution. The result is an internal buffer, shared by :func:`solve_heat`,
    :func:`solve_helmholtz`, :func:`solve_globalqg` and :func:`solve_viscdamp`
    for each `N`: it is overwritten by the next call to any of them. Copy it
    if it must be kept.

    Notes
    -----
    The solver removes the trace of the right-hand side and of the solution
    (see :func:`_solve_cpu_skewh`). Since this operator is not singular on
    the main diagonal, the returned `P` does not satisfy the equation on
    the diagonal; the off-diagonal entries are exact.

    The operator and the output buffer are cached per `N` and parameter
    values, but not per dtype: after a first call with single precision
    input, later calls with double precision input are computed in single
    precision (and vice versa).
    """
    global _cpu_globalqg_cache

    N = W.shape[0]

    if (N, gamma) not in _cpu_globalqg_cache:

        # Get cpu laplacian
        lap = laplacian(N, bc=False, dtype=type(W[0, 0].real))

        # Allocate buffer
        allocate_buffer(W)

        # Get cpu operator
        globalqg = lap.copy()
        zvec = np.diag(cartesian_generators(W.shape[-1], dtype=W.dtype)[-1]).imag
        # globalqg[:, :, :] = 0.0
        globalqg[:, :, 0] -= (gamma/2.0)*zvec**2
        globalqg[:, :, 0] -= (gamma/2.0)*zvec[:, np.newaxis]**2
        # for i in range(W.shape[-1]):
        #     globalqg[i, :, 0] *= zvec
        # for i in range(W.shape[-1]):
        #     globalqg[:, i, 0] *= zvec
        # globalqg += lap

        # Store in cache
        _cpu_globalqg_cache[(N, gamma)] = globalqg
    else:
        globalqg = _cpu_globalqg_cache[(N, gamma)]

    P = _cpu_buffer_cache[N]['P']
    _solve_cpu(globalqg, W, P,
               _cpu_buffer_cache[N]['float'],
               _cpu_buffer_cache[N]['complex'])

    return P


def solve_viscdamp(h, W0, nu=1e-4, alpha=0.01, force=None, theta=1):
    r"""
    Solve quantized viscosity and damping equation
    :math:`\dot W - \nu \Delta_N W + \alpha W = F`
    for one time-step using the theta scheme:

    .. math::

        (1 + h\alpha\theta) W_h - h\nu\theta\,\Delta_N W_h
        = (1 - h\alpha(1-\theta)) W_0 + h\nu(1-\theta)\,\Delta_N W_0 + hF.

    Parameters
    ----------
    h: float
        Time-step.
    W0: ndarray(shape=(N, N), dtype=complex)
        Initial vorticity matrix (not modified).
    nu: float
        Viscosity. Default is 1e-4.
    alpha: float
        Damping. Default is 0.01.
    force: None or ndarray(shape=(N, N), dtype=complex)
        External forcing F applied.
    theta: float
        Weighting in the theta method (Crank-Nicolson for theta=0.5,
        backward Euler for theta=1, the default).

    Returns
    -------
    Wh: ndarray(shape=(N, N), dtype=complex)
        Solution after one step. The result is an internal buffer, shared by :func:`solve_heat`,
    :func:`solve_helmholtz`, :func:`solve_globalqg` and :func:`solve_viscdamp`
    for each `N`: it is overwritten by the next call to any of them. Copy it
    if it must be kept.

    Notes
    -----
    The solver removes the trace (the :math:`\ell = 0` component) of the
    right-hand side and of the solution (see :func:`_solve_cpu_skewh`), so the
    result is trace-free even if `W0` is not.

    The operator is cached per ``(N, h, nu, alpha)``, without `theta`: a
    call with a different `theta` but the same other parameters reuses the
    operator of the earlier `theta`, which gives wrong results.

    The operator and the output buffer are cached per `N` and parameter
    values, but not per dtype: after a first call with single precision
    input, later calls with double precision input are computed in single
    precision (and vice versa).
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

    _solve_cpu(viscdamp, Wrhs, Wh,
               _cpu_buffer_cache[N]['float'],
               _cpu_buffer_cache[N]['complex'])

    return Wh
