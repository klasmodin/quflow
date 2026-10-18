r"""
Spherical-harmonic transforms between function values and coefficients.

Functions on the sphere are sampled on the grid of :func:`quflow.utils.sphgrid`
(the McEwen-Wiaux sampling: ``N`` values of the inclination ``theta`` and
``2N-1`` values of the azimuth ``phi``), giving arrays of shape
``(N, 2N-1)`` (format ``fun``). For band limit ``N`` (degrees
``0 <= el < N``), the coefficients are stored in vectors of length ``N**2``
with index ``el*(el+1) + m`` (see :func:`quflow.utils.elm2ind`).

Conventions:

* **Complex coefficients** (``shc``) are taken with respect to
  :math:`\sqrt{4\pi}\, Y_\ell^m`, where :math:`Y_\ell^m` are the
  orthonormal spherical harmonics with the Condon-Shortley phase (as in
  :func:`scipy.special.sph_harm_y`):
  :math:`f = \sum_{\ell m} \omega_{\ell m} \sqrt{4\pi}\, Y_\ell^m`.
  With this normalization, :math:`\omega_{00}` is the mean value of
  :math:`f` over the sphere, and :math:`\overline{f^2} = \sum |\omega_{\ell m}|^2`.
* **Real coefficients** (``shr``) are taken with respect to the real
  functions :math:`\sqrt{4\pi}\, \sqrt2 (-1)^m \operatorname{Re} Y_\ell^m`
  for :math:`m > 0`, :math:`\sqrt{4\pi}\, Y_\ell^0` for :math:`m = 0`, and
  :math:`-\sqrt{4\pi}\, \sqrt2 (-1)^m \operatorname{Im} Y_\ell^{|m|}` for
  :math:`m < 0`. For :math:`m < 0` this is the opposite sign of the common
  real form of the spherical harmonics. For example, the real basis
  functions for :math:`\ell = 1` are :math:`\sqrt3 x_1`,
  :math:`\sqrt3 x_3` and :math:`-\sqrt3 x_2` for
  :math:`m = 1, 0, -1`.

:func:`shc2fun` and :func:`shr2fun` by default apply the Berezin
multipliers (:func:`quflow.utils.berezin_multipliers`), which smooth the
function to correspond to the Berezin-Toeplitz quantization. With the
default, :func:`fun2shr` is therefore not the inverse of :func:`shr2fun`;
pass ``berezin=False`` for the plain transform.

The transforms use :mod:`pyssht` if it is installed, and otherwise
:mod:`ducc0`, through functions ``forward`` and ``inverse`` with the
interface of :mod:`pyssht`.
"""
import numpy as np
from numba import njit, prange
from .utils import complex_dtype, real_dtype, berezin_multipliers

try:
    from pyssht import ind2elm, forward, inverse
except ModuleNotFoundError:
    import ducc0

    def _get_theta(L: int, Method: str='MW'):
        if Method == 'MW' or Method == 'MW_pole':
            return np.pi*(2.*np.arange(L)+1) / ( 2.0 * float(L) - 1.0 )
                
        if Method == 'MWSS':
            return np.pi*np.arange(L+1)/float(L)

        if Method == 'DH':
            return np.pi*(2*np.arange(2*L)+1.) / ( 4.0 * float(L) )
            
        if Method == 'GL':
            return ducc0.misc.GL_thetas(L)

    @njit
    def _nalm(lmax: int, mmax:int):
        return ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)

    @njit
    def _get_lidx(L: int):
        res = np.arange(L)
        return res*(res+1)

    @njit
    def _extract_real_alm(flm, L: int):
        res = np.empty((_nalm(L-1, L-1),), dtype=np.complex128)
        myres = res.ravel()
        myflm = flm.ravel()
        ofs=0
        mylidx = _get_lidx(L)
        for m in range(L):
            for i in range(m,L):
                myres[ofs-m+i] = myflm[mylidx[i]+m]
            ofs += L-m
        return res

    @njit
    def _build_real_flm(alm, L: int):
        res = np.empty((L*L), dtype=np.complex128)
        ofs=0
        myres=res.ravel()
        myalm=alm.ravel()
        lidx = _get_lidx(L)
        for m in range(L):
            mfac = (-1)**m
            for i in range(m,L):
                myres[lidx[i]+m] = myalm[ofs-m+i]
                myres[lidx[i]-m] = mfac*(myalm[ofs-m+i].real - 1j*myalm[ofs-m+i].imag)
            ofs += L-m
        return res

    @njit
    def _extract_complex_alm(flm, L: int, Spin: int):
        res = np.empty((2, _nalm(L-1, L-1),), dtype=np.complex128)
        ofs=0
        sfac=(-1)**abs(Spin)
        # myres=res.ravel()
        myres=res
        myflm=flm.ravel()
        lidx = _get_lidx(L)
        if Spin >= 0:
            for m in range(L):
                mfac = (-1)**m
                for i in range(m,L):
                    fp = myflm[lidx[i]+m]
                    fm = mfac * (myflm[lidx[i]-m].real - 1j*myflm[lidx[i]-m].imag)
                    myres[0, ofs-m+i] = 0.5*(fp+fm)
                    myres[1, ofs-m+i] = -0.5j*(fp-fm)
                ofs += L-m
        else:
            for m in range(L):
                mfac = (-1)**m
                for i in range(m,L):
                    fp = mfac*sfac*(myflm[lidx[i]-m].real - 1j*myflm[lidx[i]-m].imag)
                    fm = sfac*myflm[lidx[i]+m]
                    myres[0, ofs-m+i] = 0.5*(fp+fm)
                    myres[1, ofs-m+i] = -0.5j*(fp-fm)
                ofs += L-m
        return res

    @njit
    def _build_complex_flm(alm, L: int, Spin: int):
        res = np.empty((L*L), dtype=np.complex128)
        ofs=0
        myres=res.ravel()
        myalm=np.reshape(alm, (2, _nalm(L-1, L-1),))
        lidx = _get_lidx(L)
        sfac=(-1)**abs(Spin)
        if Spin >= 0:
            for m in range(L):
                mfac = (-1)**m
                for i in range(m,L):
                    fp = myalm[0, ofs-m+i] + 1j*myalm[1, ofs-m+i]
                    fm = myalm[0, ofs-m+i] - 1j*myalm[1, ofs-m+i]
                    myres[lidx[i]+m] = fp
                    myres[lidx[i]-m] = mfac*(fm.real - 1j*fm.imag)
                ofs += L-m
        else:
            for m in range(L):
                mfac = (-1)**m
                for i in range(m,L):
                    fp = myalm[0, ofs-m+i] + 1j*myalm[1, ofs-m+i]
                    fm = myalm[0, ofs-m+i] - 1j*myalm[1, ofs-m+i]
                    myres[lidx[i]+m] = sfac*fm
                    myres[lidx[i]-m] = sfac*mfac*(fp.real -1j *fp.imag)
                ofs += L-m
        return res

    def forward(f, L, Spin=0, Method='MW', Reality=False, nthreads: int=0):
        """
        Forward spherical-harmonic transform, with the interface of
        ``pyssht.forward`` (implemented with :mod:`ducc0`).

        Parameters
        ----------
        f : ndarray
            Function values on the sampling grid of `Method`.
        L : int
            Band limit (degrees ``0 <= el < L``).
        Spin : int, optional
            Spin of the function. Default is 0.
        Method : str, optional
            Sampling scheme: ``'MW'`` (default), ``'MWSS'``, ``'DH'`` or ``'GL'``.
        Reality : bool, optional
            Whether `f` is real. Default is False.
        nthreads : int, optional
            Number of threads (0 means all). Default is 0.

        Returns
        -------
        flm : ndarray, shape (L**2,), complex
            Coefficients with respect to the orthonormal :math:`Y_\\ell^m`.
        """
        gdict = {"DH":"F1", "MW":"MW", "MWSS":"CC", "GL":"GL"}
        theta = _get_theta(L, Method)
        ntheta = theta.shape[0]
        if ntheta != f.shape[0]:
            raise RuntimeError("ntheta mismatch")
        nphi = f.shape[1]
        if Reality:
            return _build_real_flm(ducc0.sht.experimental.analysis_2d(
                map=f.reshape((-1,f.shape[0],f.shape[1])),
                lmax=L-1,
                nthreads=nthreads,
                spin=0,
                geometry=gdict[Method])[0], L)
        elif Spin == 0:
            flmr = forward(f.real, L, Spin, Method, True)
            flmi = forward(f.imag, L, Spin, Method, True)
            alm = np.empty((2,_nalm(L-1, L-1)), dtype=np.complex128)
            alm[0] = _extract_real_alm(flmr, L)
            alm[1] = _extract_real_alm(flmi, L)
            return _build_complex_flm(alm, L, 0)
        else:
            map = f.astype(np.complex128).view(dtype=np.float64).reshape((f.shape[0],f.shape[1],2)).transpose((2,0,1))
            if Spin < 0:
                map[1]*=-1
            res = _build_complex_flm(ducc0.sht.experimental.analysis_2d(
                map=map,
                lmax=L-1,
                nthreads=nthreads,
                spin=abs(Spin),
                geometry=gdict[Method]), L, Spin)
            res *= -1
            return res

    def inverse(flm: np.ndarray, L: int, Spin: int=0, Method: str='MW', Reality: bool=False, nthreads: int=0):
        """
        Inverse spherical-harmonic transform, with the interface of
        ``pyssht.inverse`` (implemented with :mod:`ducc0`).

        Parameters
        ----------
        flm : ndarray, shape (L**2,), complex
            Coefficients with respect to the orthonormal :math:`Y_\\ell^m`.
        L : int
            Band limit.
        Spin, Method, Reality, nthreads
            As for :func:`forward`. With ``Reality=True`` the result is real.

        Returns
        -------
        f : ndarray
            Function values on the sampling grid of `Method`.
        """
        gdict = {"DH":"F1", "MW":"MW", "MWSS":"CC", "GL":"GL"}
        theta = _get_theta(L, Method)
        ntheta = theta.shape[0]
        nphi = 2*L-1
        if Method == 'MWSS':
            nphi += 1
        if Reality:
            return ducc0.sht.experimental.synthesis_2d(
                alm=_extract_real_alm(flm, L).reshape((1,-1)),
                ntheta=ntheta,
                nphi=nphi,
                lmax=L-1,
                nthreads=nthreads,
                spin=0,
                geometry=gdict[Method])[0]
        elif Spin == 0:
            alm = _extract_complex_alm(flm, L,0)
            flmr = _build_real_flm(alm[0], L)
            flmi = _build_real_flm(alm[1], L)
            return inverse(flmr, L, 0, Method, True) + 1j*inverse(flmi, L, 0, Method, True)
        else:
            tmp=ducc0.sht.experimental.synthesis_2d(
                alm=_extract_complex_alm(flm, L, Spin),
                ntheta=ntheta,
                nphi=nphi,
                lmax=L-1,
                nthreads=nthreads,
                spin=abs(Spin),
                geometry=gdict[Method])
            res = -1j*tmp[1] if Spin >=0 else 1j*tmp[1]
            res -= tmp[0]
            return res

from .utils import elm2ind, ind2elm
from .quantization import mat2shr, mat2shc


def fun2shc(f):
    """
    Transform a theta-phi function to complex spherical harmonics.

    Inverse of ``shc2fun(omega, berezin=False)`` (see the module docstring
    for the normalization).

    Parameters
    ----------
    f: array_like, shape (N, 2*N-1)
        Matrix representing function values in spherical coordinates
        (theta, phi), on the grid of :func:`quflow.utils.sphgrid`.
        Can be either real or complex.

    Returns
    -------
    omega: ndarray, shape (N**2,), complex
        Complex spherical-harmonic coefficients.
    """

    # Check that the input is correct
    f = np.ascontiguousarray(f)
    N = f.shape[0]
    assert 2*N-1 == f.shape[1], "Shape of input must be (N, 2*N-1)."
    scalar_type = complex if np.iscomplexobj(f) else float
    if f.dtype is not scalar_type:
        f = f.astype(scalar_type)

    # Transform to spherical harmonics
    omega = forward(f, N, Reality=True if np.isrealobj(f) else False)
    omega /= np.sqrt(4*np.pi)

    return omega


def shc2fun(omega, isreal=False, N=-1, berezin=True):
    """
    Transform complex spherical harmonics signal to theta-phi function.

    Parameters
    ----------
    omega: array_like
        Complex array of spherical harmonics, of length ``N**2``.

    isreal: bool (optional, default=False)
        Set to true if the signal corresponds to a real function; the result
        is then real.

    N: int (optional)
        Bandwidth. If `N == -1` then the bandwidth is automatically inferred.
        Otherwise `omega` is padded with zeros or truncated to length ``N**2``.

    berezin: bool (optional, default=True)
        Use Berezin multipliers instead of Hoppe-Yau, i.e., multiply the
        coefficients by :func:`quflow.utils.berezin_multipliers` before the
        transform. Use ``berezin=False`` for the plain inverse of
        :func:`fun2shc`.

    Returns
    -------
    f: ndarray, shape (N, 2*N-1)
        Function values on the grid of :func:`quflow.utils.sphgrid`.
    """

    # Check that the input is correct
    omega = np.ascontiguousarray(omega, dtype=complex_dtype(omega.dtype))

    if N == -1:
        # Compute bandwidth
        N = ind2elm(omega.shape[0] - 1)[0] + 1
    else:
        # Extend or trim omega
        if omega.shape[0] < N**2:
            omega = np.hstack((omega, np.zeros(N**2 - omega.shape[0], dtype=complex)))
        elif omega.shape[0] > N ** 2:
            omega = omega[:N**2]

    # Make sure things are ok
    assert omega.shape[0] == N**2, "It seems that omega does not have the right length."

    # Berezin multipliers
    if berezin:
        bw = berezin_multipliers(N=N, dtype=real_dtype(omega.dtype))
        omega = omega*bw[:omega.shape[0]]

    # Transform to function values
    f = inverse(omega, N, Reality=isreal)
    f *= np.sqrt(4*np.pi)

    return f


def shc2shr(omega_complex):
    """
    Convert from complex to real spherical harmonics.
    (See https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form,
    but note that the sign for ``m < 0`` is opposite; see the module docstring.)
    If `omega_complex` does not correspond to a real function this is a projection.

    Inverse of :func:`shr2shc`.

    Parameters
    ----------
    omega_complex: ndarray, shape (N**2,), complex

    Returns
    -------
    omega_real: ndarray, shape (N**2,), float
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
    (See https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form,
    but note that the sign for ``m < 0`` is opposite; see the module docstring.)

    The result describes the same (real) function. Inverse of :func:`shc2shr`.

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


def fun2img(f, lim=np.inf):
    """
    Convert a 2D float array to an 8-bit image.
    Unless given, limits are taken so that the value 127 corresponds to 0.0.

    Parameters
    ----------
    f: ndarray(shape=(N,M))
    lim: tuple of float or float (default: automatic)
        The limits of f, corresponding to the values 0 and 255 of the image;
        values outside are clipped. A single float ``lim`` means
        ``(-lim, lim)``. The default is ``max(abs(f))``, so that f=0.0
        corresponds to img=127.

    Returns
    -------
    img: ndarray(dtype=uint8)
    """
    if not isinstance(lim, tuple):
        if lim == np.inf:
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

    Inverse of :func:`fun2img` (up to the 8-bit quantization), given the same
    limits.

    Parameters
    ----------
    img: ndarray(shape=(N,M), dtype=uint8)
    lim: tuple of float or float (default: 1.0)
        The limits of f, corresponding to the values 0 and 255 of the image.
        A single float ``lim`` means ``(-lim, lim)``.

    Returns
    -------
    f: ndarray(shape=(N,M), dtype=float)
    """
    if not isinstance(lim, tuple):
        lim = (-lim, lim)

    f = img.astype(float)*(lim[1]-lim[0])/255. + lim[0]
    return f


def fun2shr(f):
    """
    Transform a theta-phi function to real spherical harmonics.

    Inverse of ``shr2fun(omega, berezin=False)``.

    Parameters
    ----------
    f: array_like, shape (N, 2*N-1)
        Matrix representing function values in spherical coordinates
        (theta, phi), on the grid of :func:`quflow.utils.sphgrid`.
        For complex `f`, the result describes its real part.

    Returns
    -------
    omega: ndarray, shape (N**2,), float
        Real spherical-harmonic coefficients.
    """
    return shc2shr(fun2shc(f))


def shr2fun(omega, N=-1, **kwargs):
    """
    Transform real spherical harmonics signal to theta-phi function.

    Parameters
    ----------
    omega: array_like
        Real array of spherical harmonics.

    N: int (optional)
        Bandwidth. If `N == -1` then the bandwidth is automatically inferred.
    **kwargs
        Passed on to :func:`shc2fun`, e.g. ``berezin=False`` (the default
        applies the Berezin multipliers).

    Returns
    -------
    f: ndarray, shape (N, 2*N-1), float

    Examples
    --------
    >>> import numpy as np
    >>> import quflow as qf
    >>> omega = np.random.default_rng(0).standard_normal(16)
    >>> f = qf.shr2fun(omega, berezin=False)
    >>> f.shape
    (4, 7)
    >>> bool(np.allclose(qf.fun2shr(f), omega))
    True
    """
    return shc2fun(shr2shc(omega), isreal=True, N=N, **kwargs)


def as_fun(data, N=-1, **kwargs):
    """
    Take data as either `fun`, `img`, `omegar`, `omegac`, or `mat`
    and convert to `fun` (unless already).

    The format is determined from the shape and dtype: a square complex
    matrix is `mat` (converted with :func:`shr2fun` if skew-Hermitian,
    otherwise via :func:`mat2shc` and :func:`shc2fun`), another 2D array is
    `img` if of dtype uint8 and `fun` otherwise, a real 1D array is
    `omegar` and a complex 1D array `omegac`.

    Parameters
    ----------
    data: ndarray
    N: bandwidth (optional)
    **kwargs: extra arguments for shc2fun/shr2fun

    Returns
    -------
    fun: ndarray(shape=(N, 2*N-1), dtype=float or complex)

    Notes
    -----
    For a non-skew-Hermitian `mat`, and for `omegac` with `N` given, `N` is
    passed as the `isreal` argument of :func:`shc2fun`, so the result is
    real (the imaginary part is lost).
    """
    if data.ndim == 2:
        if data.shape[0] == data.shape[1] and np.iscomplexobj(data):
            # Format is mat
            W = data
            if N == -1:
                N = W.shape[0]
            if np.allclose(W, -W.conj().T):
                fun = shr2fun(mat2shr(W), N, **kwargs)
            else:
                fun = shc2fun(mat2shc(W), N, **kwargs)
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
            fun = shc2fun(omegac, **kwargs) if N == -1 else shc2fun(omegac, N, **kwargs)
        else:
            # Format is omegar
            omegar = data
            fun = shr2fun(omegar, **kwargs) if N == -1 else shr2fun(omegar, N, **kwargs)

    return fun


def as_shr(data):
    """
    Take data as either `fun`, `img`, `omegar`, `omegac`, or `mat`
    and convert to `omegar` (unless already).

    The format is determined as in :func:`as_fun`. An `img` is converted
    with the default limits of :func:`img2fun`.

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
