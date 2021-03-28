import numpy as np
import pyssht
from numba import njit


@njit
def elm2ind(el, m):
    """
    Convert (l,m) spherical harmonics indices to single index
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
