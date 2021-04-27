import numpy as np
import pyssht
from numba import njit


@njit
def ind2elm(ind):
    """
    Convert single index in omega vector to (el, m) indices.

    Parameters
    ----------
    ind: int

    Returns
    -------
    (el, m): tuple of indices
    """
    el = int(np.floor(np.sqrt(ind)))
    m = ind - el * (el + 1)
    return el, m


@njit
def elm2ind(el, m):
    """
    Convert (el,m) spherical harmonics indices to single index
    in `omegacomplex` array.

    Parameters
    ----------
    el: int or ndarray of ints
    m: int or ndarray of ints

    Returns
    -------
    ind: int
    """
    return el*el + el + m


def cart2sph(x, y, z):
    """
    Projection of Cartesian coordinates to spherical coordinates (theta, phi).

    Parameters
    ----------
    x: ndarray
    y: ndarray
    z: ndarray

    Returns
    -------
    (theta, phi): tuple of ndarray
    """
    phi = np.arctan2(y, x)
    theta = np.arctan2(np.sqrt(x * x + y * y), z)
    phi[phi < 0] += 2 * np.pi

    return theta, phi


def sph2cart(theta, phi):
    """
    Spherical coordinates to Cartesian coordinates (assuming radius 1).

    Parameters
    ----------
    theta: ndarray
    phi: ndarry

    Returns
    -------
    (x, y, z): tuple of ndarray
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z


def sphgrid(N):
    """
    Return a mesh grid for spherical coordinates.

    Parameters
    ----------
    N: int
        Bandwidth. In the spherical harmonics expansion we have that
        the wave-number l fulfills 0 <= l <= N-1.

    Returns
    -------
    (theta, phi): tuple of ndarray
        Matrices of shape (N, 2*N-1) such that row-indices corresponds to
        theta variations and column-indices corresponds to phi variations.
        (Notice that phi is periodic but theta is not.)
    """
    theta, phi = pyssht.sample_positions(N, Grid=True)

    return theta, phi


def so3generators(N):
    """
    Return a basis S1, S2, S3 for the representationn of so(3) in u(N).

    Parameters
    ----------
    N: int

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
    return S1, S2, S3


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
    from scipy.linalg import expm
    N = W.shape[0]
    S1, S2, S3 = so3generators(N)
    R = expm(xi[0]*S1 + xi[1]*S2 + xi[2]*S3)
    return R@W@R.T.conj()


def north_blob(N, sigma = 0):
    """
    Return vorticity matrix for blob located at north pole.

    Parameters
    ----------
    N: int
    sigma: float (optional)
        Gaussian sigma for blob. If 0 (default) then give best
        approximation to point vortex

    Returns
    -------
    W: ndarray(shape=(N, N), dtype=complex)
    """
    from .dynamics import solve_heat_flow

    W = np.zeros((N, N), dtype=complex)
    W[-1, -1] = 1.0j

    if sigma != 0:
        W = solve_heat_flow(sigma/4., W)

    return W

