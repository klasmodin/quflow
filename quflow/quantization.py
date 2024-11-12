import numpy as np
from .utils import elm2ind, complex_dtype, real_dtype
from .laplacian.direct import compute_direct_laplacian
from numba import njit, prange
import os
from scipy.linalg import eigh_tridiagonal
from .io import save_basis, load_basis
from scipy.sparse import dia_matrix

# ----------------
# GLOBAL VARIABLES
# ----------------

_save_computed_basis_default = False
_basis_cache = dict()


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------

@njit(error_model='numpy', fastmath=True)
def basis_break_index(absm, N):
    """
    Computes the m break indices for the basis.

    Parameters
    ----------
    n: int or np.array of int
    N: int

    Returns
    -------
    int or np.array of int
    """
    absm -= 1
    ind = absm + 2*absm**2 - 6*absm*N + 6*N**2
    ind *= 1 + absm
    return ind // 6


@njit(error_model='numpy', fastmath=True)
def adjust_basis_orientation_(w2, m, tol=1e-16):
    """
    Adjust (inline) the sign of the eigenbasis `w2` so that it corresponds
    to standard spherical harmonics.
    """
    for i in range(w2.shape[1]):
        val = w2[-1, i]
        if val < 0:
            w2[:, i] *= (-1)*(-1 if m % 2 == 1 else 1)
        elif val == 0.0:
            for j in range(2, w2.shape[0]):
                if np.abs(w2[-j, i]) > tol and np.abs(w2[-j-1, i]) > tol:
                    prev_sign = np.sign(w2[-j-1, i])
                    this_sign = np.sign(w2[-j, i])
                    if this_sign*prev_sign == -1:
                        w2[:, i] *= this_sign*(-1 if m % 2 == 1 else 1)*(-1 if j % 2 == 0 else 1)
                    else:
                        w2[:, i] *= this_sign*(-1 if m % 2 == 1 else 1)
                    break
        else:
            w2[:, i] *= (-1 if m % 2 == 1 else 1)


def compute_basis(N, dtype=np.float64):
    """
    Compute quantization basis.

    Parameters
    ----------
    N: int

    Returns
    -------
    basis: ndarray
    """

    # basis_break_indices = np.hstack((0, (np.arange(N, 0, -1)**2).cumsum()))
    # basis = np.zeros(basis_break_indices[-1], dtype=float)
    basis = np.zeros(basis_break_index(N, N), dtype=dtype)

    # Compute direct laplacian
    lap = compute_direct_laplacian(N, bc=False, dtype=dtype)

    for m in range(N):

        # Compute eigen decomposition
        n = N - m
        start_ind = N*(N+1)//2 - n*(n+1)//2
        end_ind = start_ind + n
        v2, w2 = eigh_tridiagonal(lap[1, start_ind:end_ind], lap[0, start_ind+1:end_ind])

        # Rescale to get correct L2 scaling
        w2 *= np.sqrt(N)

        # Reverse the order
        w2 = w2[:, ::-1]

        # The eigenvectors are only defined up to sign.
        # Therefore, we must adjust the sign so that it corresponds with
        # the quantization basis of Hoppe (i.e. with the spherical harmonics).
        adjust_basis_orientation_(w2, m)

        # Assign basis
        bind0 = basis_break_index(m, N)  # basis_break_indices[m]
        # bind1 = basis_break_index(m+1, N)  # basis_break_indices[m+1]
        bind1 = bind0 + (N-m)**2
        basis[bind0:bind1] = w2.ravel()

    return basis


@njit
def assign_lower_diag_(diag_m, m, W_out):
    N = W_out.shape[0]
    for i in range(N-m):
        W_out[i+m, i] = diag_m[i]


@njit
def assign_upper_diag_(diag_m, m, W_out):
    N = W_out.shape[0]
    for i in range(N-m):
        W_out[i, i+m] = diag_m[i]


@njit
def shr2mat_serial_(omega, basis, W_out):
    """
    Low-level implementation of `shr2mat`.

    Parameters
    ----------
    omega: ndarray, dtype=float, shape=(N**2,)
    basis: ndarray, dtype=float, shape=(np.sum(np.arange(N)**2),)
    W_out: ndarray, dtype=complex, shape=(N,N)
    """
    N = W_out.shape[-1]
    Nmax = N

    # Find out maximum el
    elmax = N - 1
    if omega.shape[0] < N**2:
        elmax = int(np.sqrt(omega.shape[0])) - 1
        Nmax = elmax + 1

    # basis_break_indices = np.zeros((N+1,), dtype=np.int32)
    # basis_break_indices[1:] = (np.arange(N, 0, -1, dtype=np.int32)**2).cumsum()

    for m in range(Nmax):
        bind0 = basis_break_index(m, N)  # basis_break_indices[m]
        # bind1 = basis_break_index(m + 1, N)  # basis_break_indices[m+1]
        bind1 = bind0 + (N-m)**2
        basis_m_mat = basis[bind0:bind1].reshape((N-m, N-m)).astype(W_out.dtype)

        if m == 0:  # Diagonal
            omega_zero_ind = elm2ind(np.arange(0, Nmax), 0)
            diag = basis_m_mat[:,:Nmax]@omega[omega_zero_ind].astype(W_out.dtype)
            assign_lower_diag_(diag, 0, W_out)
        else:
            # Lower diagonal
            omega_minus_m_ind = elm2ind(np.arange(m, Nmax), -m)
            omega_plus_m_ind = elm2ind(np.arange(m, Nmax), m)
            omega_complex = (1./np.sqrt(2))*(omega[omega_plus_m_ind]-1j*omega[omega_minus_m_ind])
            sgn = 1 if m % 2 == 0 else -1
            diag_m = basis_m_mat[:,:Nmax-m]@omega_complex
            diag_m *= sgn
            assign_lower_diag_(diag_m.conj(), m, W_out)

            # Upper diagonal
            assign_upper_diag_(diag_m, m, W_out)

    W_out *= 1.0j


@njit(parallel=True, error_model='numpy', fastmath=True)
def shr2mat_parallel_(omega, basis, W_out):
    """
    Low-level parallel implementation of `shr2mat`.

    Parameters
    ----------
    omega: ndarray, dtype=float, shape=(N**2,)
    basis: ndarray, dtype=float, shape=(np.sum(np.arange(N)**2),)
    W_out: ndarray, dtype=complex, shape=(N,N)
    """
    N = W_out.shape[-1]

    # Find out maximum el
    elmax = N - 1
    if omega.shape[0] < N**2:
        elmax = int(np.sqrt(omega.shape[0])) - 1
    Nmax = elmax + 1

    c1dsq2 = 1./np.sqrt(2)

    # elmin = 0
    # for el in range(elmax+1):
    #     for m in range(-m, m+1):
    #         if omega[elm2ind(el, m)] != 0.0:
    #             break
    #     elmin += 1

    for m in prange(Nmax):
        bind0 = basis_break_index(m, N)
        bind1 = bind0 + (N-m)**2
        basis_m_mat = basis[bind0:bind1].reshape((N-m, N-m))

        if m == 0:  # Diagonal
            diag = np.zeros(N, dtype=W_out.dtype)
            for el in range(0, Nmax):
                omega_zero_ind = elm2ind(el, 0)
                omega_elm = omega[omega_zero_ind]
                if omega_elm != 0.0:
                    diag += basis_m_mat[:, el] * omega_elm
            assign_lower_diag_(diag, 0, W_out)
        else:
            # Lower diagonal
            diag_m = np.zeros(N-m, dtype=W_out.dtype)
            for el in range(m, Nmax):
                omega_minus_m_ind = elm2ind(el, -m)
                omega_plus_m_ind = elm2ind(el, m)

                omega_complex = c1dsq2 * (omega[omega_plus_m_ind] - 1j*omega[omega_minus_m_ind])
                if omega_complex != 0.0j:
                    diag_m += basis_m_mat[:, el-m] * omega_complex
            sgn = 1 if m % 2 == 0 else -1
            diag_m *= sgn
            assign_lower_diag_(diag_m.conj(), m, W_out)

            # Upper diagonal
            assign_upper_diag_(diag_m, m, W_out)

    W_out *= 1.0j


# Default choice
shr2mat_ = shr2mat_parallel_


@njit
def mat2shr_serial_(W, basis, omega_out):
    N = W.shape[-1]
    # basis_break_indices = np.zeros((N+1,), dtype=np.int32)
    # basis_break_indices[1:] = (np.arange(N, 0, -1, dtype=np.int32)**2).cumsum()

    # Find out maximum el
    elmax = N - 1
    Nmax = N
    if omega_out.shape[-1] < N**2:
        elmax = round(np.sqrt(omega_out.shape[-1])) - 1
        Nmax = elmax + 1

    for m in range(Nmax):
        bind0 = basis_break_index(m, N)  # basis_break_indices[m]
        # bind1 = basis_break_index(m+1, N) # basis_break_indices[m+1]
        bind1 = bind0 + (N-m)**2
        basis_m_mat = basis[bind0:bind1].reshape((N-m, N-m)).astype(W.dtype)

        if m == 0:  # Diagonal
            omega_zero_ind = elm2ind(np.arange(0, Nmax), 0)
            diag_m = np.diag(W, 0)  # np.diagonal is more efficient than np.diag, but doesn't work with njit
            omega_out[omega_zero_ind] = ((diag_m@basis_m_mat[:,:Nmax])/1.0j).real

        else:
            # Lower diagonal
            omega_pos_m_ind = elm2ind(np.arange(m, Nmax), m)
            diag_m = np.diag(W, -m)  # np.diagonal is more efficient than np.diag, but doesn't work with njit
            omega_partial_complex = diag_m@basis_m_mat[:,:Nmax-m]
            sgn = 1 if m % 2 == 0 else -1
            omega_out[omega_pos_m_ind] = np.sqrt(2)*sgn*omega_partial_complex.imag

            # Upper diagonal
            omega_neg_m_ind = elm2ind(np.arange(m, Nmax), -m)
            omega_out[omega_neg_m_ind] = -np.sqrt(2)*sgn*omega_partial_complex.real

    omega_out /= N


@njit(parallel=True, error_model='numpy', fastmath=True)
def mat2shr_parallel_(W, basis, omega_out):
    N = W.shape[-1]

    # Find out maximum el
    elmax = N - 1
    if omega_out.shape[-1] < N**2:
        elmax = int(np.sqrt(omega_out.shape[-1])) - 1
    Nmax = elmax + 1

    sqrt2 = np.sqrt(2.0)

    for m in prange(Nmax):
        bind0 = basis_break_index(m, N)
        bind1 = bind0 + (N-m)**2
        basis_m_mat = basis[bind0:bind1].reshape((N-m, N-m))

        if m == 0:  # Diagonal
            diag = np.zeros(N, dtype=W.dtype)
            for k in range(N):
                diag[k] = W[k, k]
            for el in range(Nmax):
                omega_zero_ind = elm2ind(el, 0)
                tmp = (diag*basis_m_mat[:, el]).sum()
                tmp /= 1.0j
                omega_out[omega_zero_ind] = np.real(tmp)
        else:
            sgn = 1 if m % 2 == 0 else -1
            diag_m = np.zeros(N-m, dtype=W.dtype)
            for k in range(N-m):
                diag_m[k] = W[k+m, k]
            for el in range(m, Nmax):
                omega_pos_m_ind = elm2ind(el, m)
                omega_partial_complex = (diag_m*basis_m_mat[:, el-m]).sum()

                # Lower diagonal
                omega_out[omega_pos_m_ind] = sqrt2*sgn*np.imag(omega_partial_complex)

                # Upper diagonal
                omega_neg_m_ind = elm2ind(el, -m)
                omega_out[omega_neg_m_ind] = -sqrt2*sgn*np.real(omega_partial_complex)

    omega_out /= N


# Default choice
mat2shr_ = mat2shr_parallel_


@njit #(parallel=True, error_model='numpy')
def shc2mat_(omega, basis, W_out):
    """
    Low-level implementation of `shc2mat`.

    Parameters
    ----------
    omega: ndarray, shape (N*(N+1)/2,)
    basis: ndarray, shape (np.sum(np.arange(N)**2),)
    W_out: ndarray, shape (N,N)
    """
    N = W_out.shape[-1]
    # basis_break_indices = np.zeros((N+1,), dtype=np.int32)
    # basis_break_indices[1:] = (np.arange(N, 0, -1, dtype=np.int32)**2).cumsum()

    for m in range(N):
        bind0 = basis_break_index(m, N)  # basis_break_indices[m]
        # bind1 = basis_break_index(m + 1, N)  # basis_break_indices[m+1]
        bind1 = bind0 + (N-m)**2
        basis_m_mat = basis[bind0:bind1].reshape((N-m, N-m)).astype(W_out.dtype)

        # Lower diagonal
        omega_m_ind = elm2ind(np.arange(m, N), m)
        diag_m = basis_m_mat@omega[omega_m_ind]
        assign_lower_diag_(diag_m, m, W_out)

        # Upper diagonal
        if m != 0:
            omega_m_ind = elm2ind(np.arange(m, N), -m)
            sgn = 1 if m % 2 == 0 else -1
            diag_m = sgn*basis_m_mat@omega[omega_m_ind]
            assign_upper_diag_(diag_m, m, W_out)

    W_out *= 1.0j


@njit #(parallel=True, error_model='numpy')
def mat2shc_(W, basis, omega_out):
    N = W.shape[0]
    # basis_break_indices = np.zeros((N+1,), dtype=np.int32)
    # basis_break_indices[1:] = (np.arange(N, 0, -1, dtype=np.int32)**2).cumsum()

    for m in range(N):
        bind0 = basis_break_index(m, N)  # basis_break_indices[m]
        # bind1 = basis_break_index(m + 1, N)  # basis_break_indices[m+1]
        bind1 = bind0 + (N-m)**2
        basis_m_mat = basis[bind0:bind1].reshape((N-m, N-m)).astype(W.dtype)

        # Lower diagonal
        omega_m_ind = elm2ind(np.arange(m, N), m)
        diag_m = np.diag(W, -m)  # np.diagonal is more efficient than np.diag, but doesn't work with njit
        omega_out[omega_m_ind] = diag_m@basis_m_mat

        # Upper diagonal
        if m != 0:
            omega_m_ind = elm2ind(np.arange(m, N), -m)
            diag_m = np.diag(W, m)
            sgn = 1 if m % 2 == 0 else -1
            omega_out[omega_m_ind] = sgn*diag_m@basis_m_mat

    omega_out /= 1.0j*N


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def get_basis(N, allow_compute=True, dtype=np.double):
    """
    Return a quantization basis for band limit N.
    The basis is obtained as follows:
    - First look in memory cache.
    - Second look in storage cache.
    - Third compute basis from scratch.

    Parameters
    ----------
    N: int
    allow_compute: bool, optional
        Whether to allow computation of basis if not found elsewhere.
        Default is `True`.

    Returns
    -------
    basis: ndarray
    """
    global _basis_cache

    basis = None

    # First look in the cache and quickly return if found
    if (N, dtype) in _basis_cache:
        return _basis_cache[(N, dtype)]

    # Next look for a precomputed saved basis
    basis = load_basis(N)

    # Finally, if no precomputed basis is to be found, compute it
    if basis is None and allow_compute:
        basis = compute_basis(N, dtype=dtype)
        if 'QUFLOW_SAVE_COMPUTED_BASIS' in os.environ:
            save_computed_basis = False if os.environ['QUFLOW_SAVE_COMPUTED_BASIS'] \
                                           in ("0", "false", "False", "FALSE") else True
        else:
            save_computed_basis = _save_computed_basis_default
        if save_computed_basis:
            save_basis(basis)

    # Save basis to cache
    if basis is not None:
        _basis_cache[(N, dtype)] = basis

    return basis


def shr2mat(omega, N=-1):
    """
    Convert real spherical harmonics to matrix.

    Parameters
    ----------
    omega: ndarray(shape=(N**2,), dtype=float)
    N : int (optional)
        Size of matrix (automatic if not specified).

    Returns
    -------
    W : ndarray(shape=(N, N), dtype=complex)
    """
    
    assert np.isrealobj(omega), "omega must be a real array."

    # Process input depending on N
    if N == -1:
        N = round(np.sqrt(omega.shape[0]))
    # else:
    #     if omega.shape[0] < N**2:
    #         omega = np.hstack((omega, np.zeros(N**2-omega.shape[0], dtype=omega.dtype)))
    #     else:
    #         omega = omega[:N**2]

    W_out = np.zeros((N, N), dtype=complex_dtype(omega.dtype))
    basis = get_basis(N, omega.dtype)
    shr2mat_(omega, basis, W_out)

    return W_out


def mat2shr(W, elmax=-1):
    """
    Convert NxN complex matrix to real spherical harmonics.

    Parameters
    ----------
    W: ndarray(shape=(N, N), dtype=complex)
    elmax: int (optional)
        Maximum value of el

    Returns
    -------
    omega: ndarray(shape=(N**2,), dtype=float)
    """

    assert np.iscomplexobj(W), "W must be a complex array."

    N = W.shape[-1]

    Nmax = N
    if elmax > 0:
        Nmax = (elmax+1)**2

    omega = np.zeros(Nmax**2, dtype=real_dtype(W.dtype))
    basis = get_basis(N, dtype=omega.dtype)
    mat2shr_(W, basis, omega)

    return omega


def shc2mat(omega, N=-1):
    """
    Convert complex spherical harmonics to matrix.

    Parameters
    ----------
    omega: complex ndarray, shape (N**2,)
    N : (optional) size of matrix (automatic if not specified)

    Returns
    -------
    W : complex ndarray, shape (N, N)
    """

    # Process input depending on N
    if N == -1:
        N = round(np.sqrt(omega.shape[0]))
    else:
        if omega.shape[0] < N**2:
            omega = np.hstack((omega, np.zeros(N**2-omega.shape[0])))
        else:
            omega = omega[:N**2]

    W_out = np.zeros((N, N), dtype=omega.dtype)
    basis = get_basis(N, dtype=real_dtype(W_out.dtype))
    shc2mat_(omega, basis, W_out)

    return W_out


def mat2shc(W):
    """
    Convert NxN complex matrix to complex spherical harmonics.

    Parameters
    ----------
    W: complex ndarray, shape (N, N)

    Returns
    -------
    omega: complex ndarray, shape (N**2,)
    """
    N = W.shape[0]
    omega = np.zeros(N**2, dtype=W.dtype)
    basis = get_basis(N, dtype=real_dtype(W.dtype))
    mat2shc_(W, basis, omega)

    return omega


def elmr2mat(el, m, N, dtype=np.cdouble):
    """
    Return real T_elm matrix in the sparse format `diamatrix`.
    This gives a basis of u(N).
    The matrix is normalized with respect to `geometry.normL2`.

    Parameters
    ----------
    el: int
    m: int
    N: int

    Returns
    -------
    T_elm: diamatrix, shape (N, N)
    """
    basis = get_basis(N=N, dtype=real_dtype(dtype))
    # basis_break_indices = np.zeros((N+1,), dtype=np.int32)
    # basis_break_indices[1:] = (np.arange(N, 0, -1, dtype=np.int32)**2).cumsum()

    absm = np.abs(m)

    bind0 = basis_break_index(absm, N)  # basis_break_indices[absm]
    bind1 = basis_break_index(absm + 1, N)  # basis_break_indices[absm+1]
    basis_m_mat = basis[bind0:bind1].reshape((N-absm, N-absm)).astype(complex_dtype(dtype))

    if m == 0:  # Diagonal
        diag = 1.0j*basis_m_mat[:, el]
        T_elm = dia_matrix((diag, 0), shape=(N, N))
    else:
        # Lower diagonal
        sgn = 1 if m % 2 == 0 else -1
        diag_m = basis_m_mat[:, el-absm]
        diag_m *= sgn if m < 0 else 1.0j*sgn
        diag_m /= np.sqrt(2)

        data = np.zeros((2, N), dtype=diag_m.dtype)
        data[0,:N-absm] = -diag_m.conj()
        data[1,absm:] = diag_m

        T_elm = dia_matrix((data, np.array([-absm, absm])), shape=(N, N))

    # This is a hack to keep track of "pure el" dia_matrix objects.
    # It makes the calculations of laplace and solve_poisson much faster.
    # In the future, one should probably make a subclass
    # of dia_matrix instead.
    T_elm.el = el 

    return T_elm


def elmc2mat(el, m, N, dtype=np.cdouble):
    """
    Return complex T_elm matrix in the sparse format `diamatrix`.
    This gives a basis of gl(N, C).
    The matrix is normalized with respect to `geometry.normL2`.

    Parameters
    ----------
    el: int
    m: int
    N: int

    Returns
    -------
    T_elm: diamatrix, shape (N, N)
    """
    basis = get_basis(N=N, dtype=real_dtype(dtype))
    # basis_break_indices = np.zeros((N+1,), dtype=np.int32)
    # basis_break_indices[1:] = (np.arange(N, 0, -1, dtype=np.int32)**2).cumsum()

    absm = np.abs(m)

    bind0 = basis_break_index(absm, N)  # basis_break_indices[absm]
    bind1 = basis_break_index(absm + 1, N)  # basis_break_indices[absm+1]
    basis_m_mat = basis[bind0:bind1].reshape((N-absm, N-absm)).astype(complex_dtype(dtype))

    data = np.zeros(N, dtype=basis_m_mat.dtype)
    if m >= 0:
        data[:N-absm] = basis_m_mat[:, el-absm]
    else:
        data[absm:] = basis_m_mat[:, el-absm]
    data *= 1.0j if m % 2 == 0 or m >= 0 else -1.0j
    
    T_elm = dia_matrix((data, -m), shape=(N, N))

    # This is a hack to keep track of "pure el" dia_matrix objects.
    # It makes the calculations of laplace and solve_poisson much faster.
    # In the future, one should probably make a subclass
    # of dia_matrix instead.
    T_elm.el = el 

    return T_elm
