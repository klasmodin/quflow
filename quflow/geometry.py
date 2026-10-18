r"""
Geometry of the quantized sphere: brackets, norms, rotations and generators.

Functions on the sphere correspond to :math:`N\times N` matrices (see
:mod:`quflow.quantization`); real-valued functions correspond to
skew-Hermitian matrices. In this correspondence:

* the Poisson bracket corresponds to the scaled commutator
  :math:`\frac{1}{\hbar}[P, W]` (:func:`bracket`), with
  :math:`\hbar = 2/\sqrt{N^2-1}` (:func:`hbar`);
* the :math:`L^2` inner product corresponds to
  :math:`\langle P, W\rangle = \operatorname{Re}\operatorname{tr}(P W^*)/N`
  (:func:`inner_L2`), for which the real quantization basis is orthonormal;
* the coordinate functions :math:`x_1, x_2, x_3` correspond to the matrices
  :math:`X_a = \hbar S_a` (:func:`cartesian_generators`), where :math:`S_a`
  span the irreducible spin-:math:`(N-1)/2` representation of
  :math:`\mathfrak{so}(3)` (:func:`so3_generators`). They satisfy
  :math:`X_1^2 + X_2^2 + X_3^2 = -I`.

Several functions also accept sparse ``scipy.sparse.dia_matrix`` input, as
returned by :func:`quflow.elmr2mat`.
"""
import numpy as np
from numba import njit
from scipy.linalg import expm
from scipy.sparse import isspmatrix_dia, dia_matrix, issparse


@njit(error_model='numpy', fastmath=True)
def hbar(N):
    """
    Quantization parameter ``hbar = 2/sqrt(N**2 - 1)`` for matrix size `N`.

    Parameters
    ----------
    N : int
        Size of the matrices (``N >= 2``).

    Returns
    -------
    float
    """
    return 2.0/np.sqrt(N**2-1)


@njit(error_model='numpy', fastmath=True)
def mult_dia_core(a_data, a_offsets, b_data, b_offsets, N):
    """
    Matrix product of two matrices in diagonal storage (numba kernel).

    Uses the storage of ``scipy.sparse.dia_matrix``: ``data[k, j]`` is the
    entry in column ``j`` of diagonal ``offsets[k]``.

    Parameters
    ----------
    a_data, b_data : ndarray, shape (number of diagonals, N)
        Diagonal data of the two factors.
    a_offsets, b_offsets : ndarray of int
        Diagonal offsets of the two factors.
    N : int
        Size of the (square) matrices.

    Returns
    -------
    c_data : ndarray
        Diagonal data of the product.
    c_offsets : ndarray of int
        Diagonal offsets of the product.
    """
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
    """
    Matrix product of two square ``dia_matrix`` objects, as a ``dia_matrix``.

    Parameters
    ----------
    a, b : scipy.sparse.dia_matrix, shape (N, N)

    Returns
    -------
    scipy.sparse.dia_matrix, shape (N, N)
        The product ``a @ b``.
    """
    c_data, c_offsets = mult_dia_core(a.data, a.offsets, b.data, b.offsets, N=a.shape[-1])
    return dia_matrix((c_data, c_offsets), shape=a.shape)


# @njit(error_model='numpy', fastmath=True)
def bracket(P, W):
    r"""
    Quantized Poisson bracket :math:`\frac{1}{\hbar}[P, W] = (PW - WP)/\hbar`.

    The quantized Euler equations read ``dW/dt = bracket(P, W)``.

    Parameters
    ----------
    P, W : ndarray, shape (..., N, N), or scipy.sparse.dia_matrix
        Matrices (stacks are broadcast as in ``@``). If both are
        ``dia_matrix``, the product is computed in sparse form.

    Returns
    -------
    ndarray or scipy.sparse.dia_matrix
        A new array (``dia_matrix`` if both inputs are).
    """
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

    Computes ``sqrt(sum(|W_ij|**2)/N)``, the norm of :func:`inner_L2`. For
    ``W = shr2mat(omega)`` it equals the Euclidean norm of ``omega``.

    Parameters
    ----------
    W, array
        Matrix of shape (N, N), dense or scipy sparse.

    Returns
    -------
    float
    """
    if issparse(W):
        return np.sqrt(inner_L2(W, W))
    sqN = np.sqrt(W.shape[-1])
    return np.linalg.norm(W, ord='fro')/sqN


# @njit(error_model='numpy', fastmath=True)
def inner_L2(P, W):
    """
    Inner product corresponding to L2: ``Re(sum(P * conj(W)))/N``.

    The real quantization basis (:func:`quflow.elmr2mat`) is orthonormal for
    this inner product, so ``inner_L2(shr2mat(a), shr2mat(b)) == a @ b``.
    For skew-Hermitian matrices it equals ``-Re(tr(P @ W))/N``.

    Parameters
    ----------
    P, W : ndarray, shape (N, N), or scipy sparse matrix
        Dense or sparse (e.g. ``dia_matrix`` from :func:`quflow.elmr2mat`),
        in any combination.

    Returns
    -------
    float
    """
    N = W.shape[-1]
    if issparse(P):
        # Elementwise product (for sparse matrices `*` is the matrix product)
        return np.real(P.multiply(W.conj()).sum())/N
    if issparse(W):
        return np.real(W.conj().multiply(P).sum())/N
    return (P*W.conj()).sum().real/N


# @njit(error_model='numpy', fastmath=True)
def norm_Linf(W):
    """
    Spectral norm of `W`, corresponding to L-infinity.

    The largest singular value of `W`; for normal (e.g. skew-Hermitian)
    matrices, the largest absolute value of an eigenvalue.

    Parameters
    ----------
    W, array
        Matrix of shape (N, N).

    Returns
    -------
    float
    """
    return np.linalg.norm(W, ord=2)


@njit(error_model='numpy', fastmath=True)
def norm_L1(W):
    """
    Scaled nuclear norm of `W`, corresponding to L1.

    Computes ``sum(|eigenvalues of W|)/N``. For normal (e.g. skew-Hermitian)
    matrices this is the nuclear norm divided by `N`; for other matrices it
    is smaller.

    Parameters
    ----------
    W: array
        Complex matrix of shape (N, N).

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

    More precisely, the real number ``Im(tr(W))/N``, since real functions
    correspond to skew-Hermitian matrices. For ``W = shr2mat(omega)`` this is
    the ``el = 0`` coefficient ``omega[0]``.

    Parameters
    ----------
    W: array
        Complex matrix of shape (N, N).

    Returns
    -------
    float
    """
    trW = np.trace(W)
    trW /= W.shape[-1]
    return np.real(-1j*trW)


def so3_generators(N, dtype=np.complex128):
    r"""
    Return a basis S1, S2, S3 for the representation of so(3) in u(N).

    The matrices are skew-Hermitian, span the irreducible representation of
    spin :math:`s = (N-1)/2`, and satisfy
    :math:`[S_1, S_2] = S_3`, :math:`[S_2, S_3] = S_1`, :math:`[S_3, S_1] = S_2`,
    and :math:`S_1^2 + S_2^2 + S_3^2 = -s(s+1) I`. :math:`S_3` is diagonal.

    Parameters
    ----------
    N: int
        Size of the matrices.
    dtype: array type
        Default is ``np.complex128``.

    Returns
    -------
    S1, S2, S3: tuple of ndarray, shape (N, N)
    """
    s = (N-1)/2
    S3 = 1j*np.diag(np.arange(-s, s+1))
    S1 = 1j*np.diag(np.sqrt(s*(s+1)-np.arange(-s, s)*np.arange(-s+1, s+1)), 1)/2 + \
        1j*np.diag(np.sqrt(s*(s+1)-np.arange(-s, s)*np.arange(-s+1, s+1)), -1)/2
    S2 = np.diag(np.sqrt(s*(s+1)-np.arange(-s, s)*np.arange(-s+1, s+1)), 1)/2 - \
        np.diag(np.sqrt(s*(s+1)-np.arange(-s, s)*np.arange(-s+1, s+1)), -1)/2
    return S1.astype(dtype), S2.astype(dtype), S3.astype(dtype)


def rotate(xi, W):
    r"""
    Apply axis-angle (Rodrigues) rotation to vorticity matrix.

    Computes :math:`R W R^*` with
    :math:`R = \exp(\xi_1 S_1 + \xi_2 S_2 + \xi_3 S_3)` (see
    :func:`so3_generators`). This rotates the corresponding function by the
    angle :math:`|\xi|` about the axis :math:`\xi/|\xi|`; for example,
    ``rotate((0, 0, theta), X1) = cos(theta)*X1 + sin(theta)*X2`` for the
    matrices of :func:`cartesian_generators`.

    Parameters
    ----------
    xi: ndarray(shape=(3,), dtype=float)
        Rotation vector (axis times angle).
    W: ndarray(shape=(N,N), dtype=complex)

    Returns
    -------
    W_rotated: ndarray(shape=(N,N), dtype=complex)
        A new array.
    """
    N = W.shape[0]
    S1, S2, S3 = so3_generators(N, dtype=W.dtype)
    R = expm(xi[0]*S1 + xi[1]*S2 + xi[2]*S3)
    return R@W@R.T.conj()


def cartesian_generators(N, dtype=np.complex128):
    r"""
    Return matrices X1, X2, X3 in u(N) corresponding to the Cartesian
    coordinate functions x1, x2, x3 on the sphere.
    The relation to the basis T_lm (see :func:`quflow.elmr2mat`) is that
    :math:`T_{1,-1} = \sqrt{3} X_2`, :math:`T_{1,0} = \sqrt{3} X_3` and
    :math:`T_{1,1} = \sqrt{3} X_1`.

    The matrices are :math:`X_a = \hbar S_a` (see :func:`so3_generators`)
    and satisfy :math:`X_1^2 + X_2^2 + X_3^2 = -I`, the quantized version of
    :math:`x_1^2 + x_2^2 + x_3^2 = 1`.

    Parameters
    ----------
    N: int
        Size of the matrices.
    dtype: array type
        Default is ``np.complex128``.

    Returns
    -------
    X1, X2, X3: tuple of ndarray, shape (N, N)
    """
    h = hbar(N)
    S1, S2, S3 = so3_generators(N=N, dtype=dtype)

    return h*S1, h*S2, h*S3


def sgrad(P):
    r"""
    Return matrices dP1, dP2, dP3 corresponding to the symplectic gradient of P.

    Computes :math:`dP_a = \frac{1}{\hbar}[X_a, P] = [S_a, P]` for
    :math:`a = 1, 2, 3` (see :func:`cartesian_generators`), the quantized
    Poisson brackets :math:`\{x_a, p\}`. In the continuum limit these are the
    Cartesian components of the symplectic gradient (Hamiltonian vector
    field) of :math:`p`, :math:`\nabla p \times x`. For example,
    ``sgrad(X3)`` returns ``(-X2, X1, 0)``, the rotation about the
    :math:`x_3`-axis.

    Parameters
    ----------
    P : ndarray, shape (N, N), complex

    Returns
    -------
    dP : ndarray, shape (3, N, N), complex
    """
    # This is a VERY inefficient way to compute the gradient.
    # An optimized version should be implemented.
    X = cartesian_generators(P.shape[-1], P.dtype)
    dP = np.zeros((3,)+P.shape, dtype=P.dtype)
    for Xi, dPi in zip(X, dP):
        dPi[...] = bracket(Xi, P)
    return dP

