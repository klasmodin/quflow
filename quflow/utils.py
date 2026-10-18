"""
Utility functions: dtype helpers, spherical-harmonic index maps, the
sampling grid, coordinate conversions and time units.
"""
import numpy as np
# import pyssht
import os
from numba import njit, prange
from scipy.special import gammaln


def complex_dtype(dt):
	"""
	Return complex dtype corresponding to dt.

	Parameters
	----------
	dt : dtype
		A real or complex floating point dtype.

	Returns
	-------
	numpy scalar type
		The complex type of the same precision, e.g. ``np.cdouble`` for
		``np.float64`` or ``np.complex128``.
	"""
	return {11: np.csingle,
			12: np.cdouble,
			13: np.clongdouble,
			14: np.csingle,
			15: np.cdouble,
			16: np.clongdouble}[np.dtype(dt).num]


def real_dtype(dt):
	"""
	Return real dtype corresponding to dt.

	Parameters
	----------
	dt : dtype
		A real or complex floating point dtype.

	Returns
	-------
	numpy scalar type
		The real type of the same precision, e.g. ``np.double`` for
		``np.float64`` or ``np.complex128``.
	"""
	return {11: np.single,
			12: np.double,
			13: np.longdouble,
			14: np.single,
			15: np.double,
			16: np.longdouble}[np.dtype(dt).num]


def poisson_finite_differences(omegafun, psifun):
	"""
	Compute approximation of Poisson bracket using finite differences.
	This is just to test against traditional methods.
	DO NOT USE THE FUNCTION IN SIMULATIONS.

	Parameters
	----------
	omegafun: ndarray, shape=(N,2*N-1)
		Function values on the grid of :func:`sphgrid`.
	psifun: ndarray, shape=(N,2*N-1)
		Function values on the grid of :func:`sphgrid`.

	Returns
	-------
	ndarray, shape=(N,2*N-1)
		Approximation to Poisson bracket {omegafun, psifun}, computed as
		``(d_theta psi * d_phi omega - d_theta omega * d_phi psi)/sin(theta)``
		with one-sided differences.
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

	Inverse of :func:`elm2ind`: ``el = floor(sqrt(ind))`` and
	``m = ind - el*(el+1)``.

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

	The index is ``el*(el+1) + m``, used for both real and complex
	coefficient vectors. Coefficients with ``el < N`` fill a vector of
	length ``N**2``.

	Parameters
	----------
	el: int or ndarray of ints
		Degree, ``el >= 0``.
	m: int or ndarray of ints
		Order, ``-el <= m <= el``.

	Returns
	-------
	ind: int or ndarray of ints
	"""
	return el*el + el + m


def berezin_multipliers(N, dtype=np.float64, el=None):
	r"""
	Returns the spherical harmonics scalings
	:math:`w_\ell = \sqrt{\prod_{j=1}^\ell \frac{N-j}{N+j}}`
	which converts the Hoppe-Yau quantization T_N to the Berezin-Toeplitz quantization Q_N.

	The multipliers decrease from :math:`w_0 = 1` and act as a smoothing
	filter; they are applied by default in :func:`quflow.shr2fun` and
	:func:`quflow.shc2fun`.

	Parameters
	----------
	N: int
		Size of matrix.
	dtype: dtype of float
		Default is ``np.float64``.
	el: int or array of int, optional
		Wavenumbers to include (defaults to ind2elm(np.arange(N**2))[0], i.e.,
		one multiplier per entry of a coefficient vector).

	Returns
	-------
	ndarray
		The multipliers :math:`w_\ell`.
	"""
	if el is None:
		ells, ms = ind2elm(np.arange(N**2))
		ells = ells.astype(np.float64)
	else:
		ells = np.asarray(el, dtype=np.float64)

	NN = np.float64(N)
	
	log_bw = 0.5 * (
		gammaln(NN + 1)
		+ gammaln(NN)
		- gammaln(NN - ells)
		- gammaln(NN + ells + 1)
	)
	bw = np.exp(log_bw)

	return bw.astype(dtype)


def cart2sph(x, y, z):
	"""
	Projection of Cartesian coordinates to spherical coordinates (theta, phi).

	Parameters
	----------
	x: ndarray
	y: ndarray
	z: ndarray
		Coordinates (arrays; scalar input is not supported).

	Returns
	-------
	(theta, phi): tuple of ndarray
		Inclination ``theta`` in ``[0, pi]`` and azimuth ``phi`` in
		``[0, 2*pi)``. Inverse of :func:`sph2cart` on the unit sphere.
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
		Inclination, ``0 <= theta <= pi``.
	phi: ndarray
		Azimuth.

	Returns
	-------
	(x, y, z): tuple of ndarray
		``(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))``.
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

	Notes
	-----
	This is the McEwen-Wiaux ("MW") sampling used by
	:mod:`quflow.transforms`:
	``theta_j = (2j+1)*pi/(2N-1)`` for ``j = 0, ..., N-1`` (so the last row
	is the south pole, ``theta = pi``), and ``phi_k = 2*pi*k/(2N-1)`` for
	``k = 0, ..., 2N-2``.
	"""
	# theta, phi = pyssht.sample_positions(N, Grid=True)

	# This is the definition of theta and phi according to "MW"
	theta = (2.0*np.arange(N) + 1.0) * np.pi / (2.0*N - 1.0)
	phi = 2.0 * np.arange(2*N-1) * np.pi / (2.0*N - 1.0)
	phig, thetag = np.meshgrid(phi, theta)

	return thetag, phig


def qtime2seconds(qtime, N):
	"""
	Convert quantum time units to seconds.

	Computes ``qtime*hbar(N)``: one quantum time unit corresponds to the
	dimensionless step ``h = dt/hbar`` used by the integrators.

	Parameters
	----------
	qtime: float or ndarray
	N: int

	Returns
	-------
	Time in seconds.
	"""
	# return qtime*np.sqrt(16.*np.pi)/N**(3./2.)
	hbar = 2.0/np.sqrt(N**2-1)
	return qtime*hbar


def seconds2qtime(t, N):
	"""
	Convert seconds to quantum time unit.

	Computes ``t/hbar(N)``; inverse of :func:`qtime2seconds`.

	Parameters
	----------
	t: float or ndarray
	N: int

	Returns
	-------
	Time in quantum time units.
	"""
	# return t/np.sqrt(16.*np.pi)*N**(3./2.)
	hbar = 2.0/np.sqrt(N**2-1)
	return t/hbar


def run_cluster(filename, time, inner_time, step_size):
	"""
	Write a run script and a SLURM batch script for a cluster simulation (legacy).

	Fills in the templates in ``quflow/templates`` and writes
	``run_<simname>.py`` and ``vera2_<simname>.sh`` next to `filename`.
	Superseded by :mod:`quflow.cluster`; the current templates no longer
	contain most of the placeholders filled in here.

	Parameters
	----------
	filename : str
		Simulation file (``*.hdf5``).
	time : float
		Simulation time.
	inner_time : float
		Time between saved states.
	step_size : float
		Time step.

	Returns
	-------
	None
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
