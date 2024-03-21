import numpy as np
from numba import njit
from scipy.linalg import expm


@njit(error_model='numpy', fastmath=True)
def hbar(N):
    return 2.0/np.sqrt(N**2-1)


@njit(error_model='numpy', fastmath=True)
def bracket(P, W):
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
    sqN = np.sqrt(W.shape[-1])
    return np.linalg.norm(W, ord='fro')/sqN


# @njit(error_model='numpy', fastmath=True)
def inner_L2(P, W):
    N = W.shape[-1]
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
