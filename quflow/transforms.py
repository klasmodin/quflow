import numpy as np
import pyssht
from .utils import elm2ind
from .quantization import mat2shr, mat2shc


def fun2shc(f):
    """
    Transform a theta-phi function to complex spherical harmonics.

    Parameters
    ----------
    f: array_like, shape (N, 2*N-1)
        Matrix representing function values in spherical coordinates
        (theta, phi). Can be either real or complex.

    Returns
    -------
    omega: ndarray
        Complex array of spherical coordinates.
    """

    # Check that the input is correct
    f = np.ascontiguousarray(f)
    N = f.shape[0]
    assert 2*N-1 == f.shape[1], "Shape of input must be (N, 2*N-1)."
    scalar_type = complex if np.iscomplexobj(f) else float
    if f.dtype is not scalar_type:
        f = f.astype(scalar_type)

    # Transform to spherical harmonics
    omega = pyssht.forward(f, N, Reality=True if np.isrealobj(f) else False)

    return omega


def shc2fun(omega, isreal=False, N=-1):
    """
    Transform complex spherical harmonics signal to theta-phi function.

    Parameters
    ----------
    omega: array_like
        Complex array of spherical harmonics.

    isreal: bool (optional, default=True)
        Set to true if the signal corresponds to a real function.

    N: int (optional)
        Bandwidth. If `N == -1` then the bandwidth is automatically inferred.

    Returns
    -------
    f: ndarray, shape (N, 2*N-1)
    """

    # Check that the input is correct
    omega = np.ascontiguousarray(omega)
    if omega.dtype is not complex:
        omega = omega.astype(complex)

    if N == -1:
        # Compute bandwidth
        N = pyssht.ind2elm(omega.shape[0] - 1)[0] + 1
    else:
        # Extend or trim omega
        if omega.shape[0] < N**2:
            omega = np.hstack((omega, np.zeros(N**2 - omega.shape[0], dtype=complex)))
        elif omega.shape[0] > N ** 2:
            omega = omega[:N**2]

    # Make sure things are ok
    assert omega.shape[0] == N**2, "It seems that omega does not have the right length."

    # Transform to function values
    f = pyssht.inverse(omega, N, Reality=isreal)

    return f


def shc2shr(omega_complex):
    """
    Convert from complex to real spherical harmonics.
    (See https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form)
    If `omega_complex` does not corresponds to a real function this is a projection.

    Parameters
    ----------
    omega_complex: ndarray

    Returns
    -------
    omega_real: ndarray
    """
    el = 0
    omega_real = np.zeros(omega_complex.shape[0], dtype=float)

    # el == 0 first
    omega_real[elm2ind(el, 0)] = omega_complex[elm2ind(el, 0)].real

    el += 1
    while elm2ind(el, 0) < omega_complex.shape[0]:
        # m=0
        omega_real[elm2ind(el, 0)] = omega_complex[elm2ind(el, 0)].real

        # Negative m
        ms = np.arange(-el, 0)
        omega_real[elm2ind(el, ms)] = np.sqrt(2) * (-1)**(-ms) * omega_complex[elm2ind(el, -ms)].imag

        # Positive m
        ms = np.arange(1, el + 1)
        omega_real[elm2ind(el, ms)] = np.sqrt(2) * (-1)**ms * omega_complex[elm2ind(el, ms)].real

        # Increase el
        el += 1

    return omega_real


def shr2shc(omega_real):
    """
    Convert from real to complex spherical harmonics.
    (See https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form)

    Parameters
    ----------
    omega_real: ndarray(shape=(N**2,), dtype=float)

    Returns
    -------
    omega_complex: ndarray(shape=(N**2,), dtype=complex)
    """
    el = 0
    omega_complex = np.zeros(omega_real.shape[0], dtype=complex)

    # el == 0 first
    omega_complex[elm2ind(el, 0)] = omega_real[elm2ind(el, 0)]

    el += 1
    while elm2ind(el, 0) < omega_real.shape[0]:
        # m=0
        omega_complex[elm2ind(el, 0)] = omega_real[elm2ind(el, 0)]

        # Negative m
        ms = np.arange(-el, 0)
        omega_complex[elm2ind(el, ms)] = \
            (1. / np.sqrt(2)) * (omega_real[elm2ind(el, -ms)] - 1j*omega_real[elm2ind(el, ms)])

        # Positive m
        ms = np.arange(1, el+1)
        sgn = np.ones(ms.shape[0], dtype=int)
        sgn[::2] = -1
        omega_complex[elm2ind(el, ms)] = \
            (1. / np.sqrt(2)) * sgn * (omega_real[elm2ind(el, ms)] + 1j*omega_real[elm2ind(el, -ms)])

        # Increase el
        el += 1

    return omega_complex


def fun2img(f, lim=np.infty):
    """
    Convert a 2D float array to an 8-bit image.
    Unless given, limits are taken so that the value 128 correspond to 0.0.

    Parameters
    ----------
    f: ndarray(shape=(N,M))
    lim: tuple of float or float (default: automatic)
        The limits of f, corresponding to the values 0 and 255 of the image.
        The default values are balanced, so that f=0.0 correspond to img=128.

    Returns
    -------
    img: ndarray(dtype=uint8)
    """
    if not isinstance(lim, tuple):
        if lim == np.infty:
            lim = np.abs(f).max()
        lim = (-lim, lim)

    fscale = 255*(f-lim[0])/(lim[1]-lim[0])
    fscale[np.where(fscale < 0)] = 0
    fscale[np.where(fscale > 255)] = 255
    img = fscale.astype(np.uint8)

    return img


def img2fun(img, lim=1.0):
    """
    Convert an 8-bit image to a 2D float array.
    Unless given, limits are taken so that the value 0.0 correspond to 128.

    Parameters
    ----------
    img: ndarray(shape=(N,M), dtype=uint8)
    lim: tuple of float or float (default: automatic)
        The limits of f, corresponding to the values 0 and 255 of the image.
        The default values are balanced, so that f=0.0 correspond to img=128.

    Returns
    -------
    img: ndarray(dtype=uint8)
    """
    if not isinstance(lim, tuple):
        lim = (-lim, lim)

    f = img.astype(float)*(lim[1]-lim[0])/255. + lim[0]
    return f


def fun2shr(f):
    """
    Transform a theta-phi function to real spherical harmonics.

    Parameters
    ----------
    f: array_like, shape (N, 2*N-1)
        Matrix representing function values in spherical coordinates
        (theta, phi). Can be either real or complex.

    Returns
    -------
    omega: ndarray
        Real array of spherical coordinates.
    """
    return shc2shr(fun2shc(f))


def shr2fun(omega, N=-1):
    """
    Transform real spherical harmonics signal to theta-phi function.

    Parameters
    ----------
    omega: array_like
        Real array of spherical harmonics.

    N: int (optional)
        Bandwidth. If `N == -1` then the bandwidth is automatically inferred.

    Returns
    -------
    f: ndarray, shape (N, 2*N-1)
    """
    return shc2fun(shr2shc(omega), isreal=True, N=N)


def as_fun(data, N=-1):
    """
    Take data as either `fun`, `img`, `omegar`, `omegac`, or `mat`
    and convert to `fun` (unless already).

    Parameters
    ----------
    data: ndarray
    N: bandwidth (optional)

    Returns
    -------
    fun: ndarray(shape=(N, 2*N-1), dtype=float or complex)
    """
    if data.ndim == 2:
        if data.shape[0] == data.shape[1] and np.iscomplexobj(data):
            # Format is mat
            W = data
            if N == -1:
                N = W.shape[0]
            if np.allclose(W, -W.conj().T):
                fun = shr2fun(mat2shr(W), N)
            else:
                fun = shc2fun(mat2shc(W), N)
        else:
            # Format is fun or img
            if data.dtype == np.uint8:
                # Format is img
                img = data
                fun = img2fun(img)
            else:
                # Format is fun
                fun = data
    else:
        # Format is omegar or omegac
        if np.iscomplexobj(data):
            # Format is omegac
            omegac = data
            fun = shc2fun(omegac) if N == -1 else shc2fun(omegac, N)
        else:
            # Format is omegar
            omegar = data
            fun = shr2fun(omegar) if N == -1 else shr2fun(omegar, N)

    return fun


def as_shr(data):
    """
    Take data as either `fun`, `img`, `omegar`, `omegac`, or `mat`
    and convert to `omegar` (unless already).

    Parameters
    ----------
    data: ndarray

    Returns
    -------
    omegar: ndarray(shape=(N**2,), dtype=float)
    """
    if data.ndim == 2:
        if data.shape[0] == data.shape[1] and np.iscomplexobj(data):
            # Format is mat
            W = data
            N = W.shape[0]
            if N == -1:
                N = W.shape[0]
            omegar = mat2shr(W)
        else:
            # Format is fun or img
            if data.dtype == np.uint8:
                # Format is img
                img = data
                fun = img2fun(img)
            else:
                # Format is fun
                fun = data
            omegar = fun2shr(fun)
    else:
        # Format is omegar or omegac
        if np.iscomplexobj(data):
            # Format is omegac
            omegac = data
            omegar = shc2shr(omegac)
        else:
            # Format is omegar
            omegar = data

    return omegar
