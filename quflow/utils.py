import numpy as np
import pyssht
import os
from numba import njit, prange
from scipy.linalg import expm


def poisson_finite_differences(omegafun, psifun):
    """
    Compute approximation of Poisson bracket using finite differences.
    This is just to test against traditional methods.

    Parameters
    ----------
    omegafun: ndarray, shape=(N,2*N-1)
    psifun: ndarray, shape=(N,2*N-1)

    Returns
    -------
    Approximation to Poisson bracket {omegafun, psifun}.
    """
    N = omegafun.shape[0]
    thetafun, phifun = sphgrid(N)

    dtheta_omega = np.zeros_like(omegafun)
    dphi_omega = np.zeros_like(omegafun)
    dtheta_psi = np.zeros_like(psifun)
    dphi_psi = np.zeros_like(psifun)

    dtheta_omega[1:N, :] = np.diff(omegafun, n=1, axis=0)/np.diff(thetafun, n=1, axis=0)
    dtheta_omega[0, :] = dtheta_omega[1, :]
    dphi_omega[:, :] = np.diff(omegafun, n=1, axis=1, append=omegafun[:, 0].reshape((N, 1)))/(phifun[0, 1] - phifun[0, 0])

    dtheta_psi[1:N, :] = np.diff(psifun, n=1, axis=0)/np.diff(thetafun, n=1, axis=0)
    dtheta_psi[0, :] = dtheta_psi[1, :]
    dphi_psi[:, :] = np.diff(psifun, n=1, axis=1, append=psifun[:, 0].reshape((N, 1)))/(phifun[0, 1] - phifun[0, 0])

    sinth = np.sin(thetafun)
    # sinth[:2, :] = sinth[2, :]
    sinth[-2:, :] = sinth[-2, :]
    br = (dtheta_psi*dphi_omega - dtheta_omega*dphi_psi)/sinth
    br[-2:, :] = br[-2, :]

    return br


# @njit
def ind2elm(ind):
    """
    Convert single index in omega vector to (el, m) indices.

    Parameters
    ----------
    ind: int or array(dtype=int)

    Returns
    -------
    (el, m): tuple of indices
    """
    el = np.floor(np.sqrt(ind)).astype(int)
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
    return S1, S2.astype(np.complex128), S3


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
    S1, S2, S3 = so3generators(N)
    R = expm(xi[0]*S1 + xi[1]*S2 + xi[2]*S3)
    return R@W@R.T.conj()


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


def run_cluster(filename, time, inner_time, step_size):
    """

    Parameters
    ----------
    filename
    time
    inner_time
    step_size

    Returns
    -------

    """
    from . import templates

    # Read run file as string
    with open(templates.__file__.replace("__init__.py","run_TEMPLATE.py"), 'r') as run_file:
        run_str = run_file.read()\
            .replace('_FILENAME_', filename)\
            .replace('_SIMTIME_', str(time))\
            .replace('_INNER_TIME_', str(inner_time))\
            .replace('_STEP_SIZE_', str(step_size))\
            .replace('_SIMULATE_', 'True')\
            .replace('_ANIMATE_', 'True')

    # Read vera file as string
    with open(templates.__file__.replace("__init__.py","vera2_TEMPLATE.sh"), 'r') as vera_file:
        simname = os.path.split(filename)[1].replace(".hdf5", "")
        vera_str = vera_file.read()\
            .replace('$SIMNAME', simname)\
            .replace('$NO_CORES', '16')

    # Write run file
    with open(os.path.join(os.path.split(filename)[0], "run_"+simname+".py"), 'w') as run_file:
        run_file.write(run_str)

    # Write vera file
    with open(os.path.join(os.path.split(filename)[0], "vera2_"+simname+".sh"), 'w') as vera_file:
        vera_file.write(vera_str)
