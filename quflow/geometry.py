import numpy as np
from numba import njit
from scipy.linalg import expm
from scipy.sparse import isspmatrix_dia, dia_matrix


@njit(error_model='numpy', fastmath=True)
def hbar(N):
    return 2.0/np.sqrt(N**2-1)


@njit(error_model='numpy', fastmath=True)
def mult_dia_core(a_data, a_offsets, b_data, b_offsets, N):
    c_data = np.zeros((a_data.shape[0]*b_data.shape[0], N), dtype=a_data.dtype)
    c_offsets = np.zeros(c_data.shape[0], dtype=a_offsets.dtype)
    c_offsets.fill(N+1)
    kmax = 0
    for m in range(a_data.shape[0]):
        a_offset = a_offsets[m]
        for n in range(b_data.shape[0]):
            b_offset = b_offsets[n]
            c_offset = a_offset + b_offset
            if np.abs(c_offset) < N:
                k = 0
                while c_offsets[k] != c_offset and c_offsets[k] != N+1:
                    k += 1
                if k > kmax:
                    kmax = k
                for i in range(max(0, -a_offset, -c_offset), min(N-c_offset, N-a_offset, N)):
                    c_data[k, c_offset+i] += a_data[m, a_offset+i]*b_data[n, c_offset+i]
                c_offsets[k] = c_offset
    return c_data[:kmax+1, :], c_offsets[:kmax+1]


def matmul_dia(a, b):
    c_data, c_offsets = mult_dia_core(a.data, a.offsets, b.data, b.offsets, N=a.shape[-1])
    return dia_matrix((c_data, c_offsets), shape=a.shape)


# @njit(error_model='numpy', fastmath=True)
def bracket(P, W):
    if isspmatrix_dia(P) and isspmatrix_dia(W):
        A = matmul_dia(P, W) 
        A -= matmul_dia(W, P)
    else:
        A = P@W
        A -= W@P
    A /= hbar(P.shape[-1])
    return A


# @njit(error_model='numpy', fastmath=True)
def norm_L2(W):
    """
    Scaled Frobenius norm of `W`, corresponding to L2.

    Parameters
    ----------
    W, array

    Returns
    -------
    float
    """
    if isspmatrix_dia(W):
        return np.sqrt((W.data*W.data.conj()).sum().real/W.shape[-1])
    sqN = np.sqrt(W.shape[-1])
    return np.linalg.norm(W, ord='fro')/sqN


# @njit(error_model='numpy', fastmath=True)
def inner_L2(P, W):
    N = W.shape[-1]
    if isspmatrix_dia(P) and isspmatrix_dia(W) and np.array_equal(W.offsets, P.offsets):
        return (P.data*W.data.conj()).sum().real/N
    return (P*W.conj()).sum().real/N


# @njit(error_model='numpy', fastmath=True)
def norm_Linf(W):
    """
    Spectral norm of `W`, corresponding to L-infinity.

    Parameters
    ----------
    W, array

    Returns
    -------
    float
    """
    return np.linalg.norm(W, ord=2)


@njit(error_model='numpy', fastmath=True)
def norm_L1(W):
    """
    Scaled nuclear norm of `W`, corresponding to L1.

    Parameters
    ----------
    W: array

    Returns
    -------
    float
    """
    sW = np.abs(np.linalg.eigvals(W))
    sW /= W.shape[-1]
    return sW.sum()


@njit(error_model='numpy', fastmath=True)
def integral(W):
    """
    Compute the integral of `W`, which
    is given by tr(W)/N.

    Parameters
    ----------
    W: array

    Returns
    -------
    float
    """
    trW = np.trace(W)
    trW /= W.shape[-1]
    return np.real(-1j*trW)


def so3_generators(N, dtype=np.complex128):
    """
    Return a basis S1, S2, S3 for the representationn of so(3) in u(N).

    Parameters
    ----------
    N: int
    dtype: array type

    Returns
    -------
    S1, S2, S3: tuple of ndarray
    """
    s = (N-1)/2
    S3 = 1j*np.diag(np.arange(-s, s+1))
    S1 = 1j*np.diag(np.sqrt(s*(s+1)-np.arange(-s, s)*np.arange(-s+1, s+1)), 1)/2 + \
        1j*np.diag(np.sqrt(s*(s+1)-np.arange(-s, s)*np.arange(-s+1, s+1)), -1)/2
    S2 = np.diag(np.sqrt(s*(s+1)-np.arange(-s, s)*np.arange(-s+1, s+1)), 1)/2 - \
        np.diag(np.sqrt(s*(s+1)-np.arange(-s, s)*np.arange(-s+1, s+1)), -1)/2
    return S1.astype(dtype), S2.astype(dtype), S3.astype(dtype)


def rotate(xi, W):
    """
    Apply axis-angle (Rodrigues) rotation to vorticity matrix.

    Parameters
    ----------
    xi: ndarray(shape=(3,), dtype=float)
    W: ndarray(shape=(N,N), dtype=complex)

    Returns
    -------
    W_rotated: ndarray(shape=(N,N), dtype=complex)
    """
    N = W.shape[0]
    S1, S2, S3 = so3_generators(N, dtype=W.dtype)
    R = expm(xi[0]*S1 + xi[1]*S2 + xi[2]*S3)
    return R@W@R.T.conj()


def cartesian_generators(N, dtype=np.complex128):
    """
    Return matrices X1, X2, X3 in u(N) corresponding to the Cartesian
    coordinate functions x1, x2, x3 on the sphere.
    The relation to the basis T_lm is that
        T_{1,-1} = \sqrt{3} X_2
        T_{1,0} = \sqrt{3} X_3
        T_{1,1} = \sqrt{3} X_1

    Parameters
    ----------
    N: int
    dtype: array type

    Returns
    -------
    S1, S2, S3: tuple of ndarray
    """
    h = hbar(N)
    S1, S2, S3 = so3_generators(N=N, dtype=dtype)

    return h*S1, h*S2, h*S3


def grad(P):
    """
    Return matrices dP1, dP2, dP3 corresponding to the Cartesian gradient of P.
    """
    # This is a VERY inefficient way to compute the gradient.
    # An optimized version should be implemented.
    X = cartesian_generators(P.shape[-1], P.dtype)
    dP = np.zeros((3,)+P.shape, dtype=P.dtype)
    for Xi, dPi in zip(X, dP):
        dPi[...] = bracket(Xi, P)
    return dP

