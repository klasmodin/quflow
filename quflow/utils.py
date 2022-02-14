import numpy as np
import pyssht
from numba import njit
from .laplacian.sparse import solve_heat


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


def blob(N, pos=np.array([0.0, 0.0, 1.0]), sigma=0):
    """
    Return vorticity matrix for blob located at 'pos'.

    Parameters
    ----------
    N: int
    pos: ndarray(shape=(3,), dtype=double)
    sigma: float (optional)

    Returns
    -------
    W: ndarray(shape=(N,N), dtype=complex)
    """

    # First find rotation matrix r
    a = np.zeros((3, 3))
    a[:, 0] = pos
    q, r = np.linalg.qr(a)
    if np.dot(q[:, 0], pos) < 0:
        q[:, 0] *= -1
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    q = np.roll(q, 2, axis=-1)

    # Then find axis-angle representation
    from scipy.spatial.transform import Rotation as R
    xi = R.from_matrix(q).as_rotvec()

    # Create north blob
    W = north_blob(N, sigma)

    # Rotate it
    W = rotate(xi, W)

    return W


def north_blob(N, sigma=0):
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

    W = np.zeros((N, N), dtype=complex)
    W[-1, -1] = 1.0j

    if sigma != 0:
        W = solve_heat(sigma/4., W)

    return W


def qtime2seconds(qtime, N):
    """
    Convert quantum time units to seconds.

    Parameters
    ----------
    qtime: float or ndarray
    N: int

    Returns
    -------
    Time in seconds.
    """
    return qtime*np.sqrt(16.*np.pi)/N**(3./2.)


def seconds2qtime(t, N):
    """
    Convert seconds to quantum time unit.

    Parameters
    ----------
    t: float or ndarray
    N: int

    Returns
    -------
    Time in quantum time units.
    """
    return t/np.sqrt(16.*np.pi)*N**(3./2.)
