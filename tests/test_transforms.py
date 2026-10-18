import numpy as np
import pytest
import quflow as qf


def get_fun(N=501):
    theta, phi = qf.sphgrid(N)
    f = np.sin(phi)**2*np.sin(theta/2)
    return f


def get_random_omega_real(N=17):
    return np.random.randn(N**2)


def get_random_omega_complex(N=17):
    return qf.shr2shc(get_random_omega_real(N))


@pytest.mark.parametrize("f", [get_fun(), get_fun(N=128), get_fun(N=256)[:128, :255]])
def test_spherical_harmonics_transform_forward(f):
    omega = qf.fun2shc(f)
    f = qf.shc2fun(omega, berezin=False)
    omega2 = qf.fun2shc(f)
    # assert omega == pytest.approx(omega2)
    np.testing.assert_allclose(omega2, omega, atol=1e-14, rtol=1e-4)


@pytest.mark.parametrize("omega_real", [get_random_omega_real(), get_random_omega_real(128)])
def test_real_to_complex_harmonics(omega_real):
    omega_complex = qf.shr2shc(omega_real)
    omega_real2 = qf.shc2shr(omega_complex)
    # assert omega_real == pytest.approx(omega_real2)
    np.testing.assert_allclose(omega_real, omega_real2)


@pytest.mark.parametrize("omega_complex", [get_random_omega_complex(), get_random_omega_complex(128)])
def test_complex_to_real_harmonics(omega_complex):
    omega_real = qf.shc2shr(omega_complex)
    omega_complex2 = qf.shr2shc(omega_real)
    # assert omega_complex == pytest.approx(omega_complex2)
    np.testing.assert_allclose(omega_complex, omega_complex2)


# --- Documented conventions ---------------------------------------------------

from scipy.special import sph_harm_y


def _unit(N, el, m, dtype=float):
    e = np.zeros(N**2, dtype=dtype)
    e[qf.elm2ind(el, m)] = 1
    return e


@pytest.mark.parametrize("el, m", [(0, 0), (1, 1), (1, -1), (2, -2), (3, 2), (4, -3)])
def test_complex_basis_is_scaled_Ylm(el, m):
    N = 7
    theta, phi = qf.sphgrid(N)
    f = qf.shc2fun(_unit(N, el, m, complex), berezin=False)
    np.testing.assert_allclose(f, np.sqrt(4*np.pi)*sph_harm_y(el, m, theta, phi), rtol=0, atol=1e-12)


def test_real_basis_degree_one():
    # Real basis functions for el = 1 (note the sign for m = -1)
    N = 5
    theta, phi = qf.sphgrid(N)
    x1, x2, x3 = qf.sph2cart(theta, phi)
    f = lambda m: qf.shr2fun(_unit(N, 1, m), berezin=False)
    np.testing.assert_allclose(f(1), np.sqrt(3)*x1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(f(0), np.sqrt(3)*x3, rtol=0, atol=1e-12)
    np.testing.assert_allclose(f(-1), -np.sqrt(3)*x2, rtol=0, atol=1e-12)


def test_cartesian_generators_as_functions():
    N = 6
    theta, phi = qf.sphgrid(N)
    x1, x2, x3 = qf.sph2cart(theta, phi)
    X1, X2, X3 = qf.geometry.cartesian_generators(N)
    fun = lambda X: qf.shr2fun(qf.mat2shr(X), berezin=False)
    np.testing.assert_allclose(fun(X1), x1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fun(X2), -x2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fun(X3), x3, rtol=0, atol=1e-12)


def test_normalization_mean_and_parseval(rng):
    N = 8
    np.testing.assert_allclose(qf.shr2fun(_unit(N, 0, 0), berezin=False), 1.0, rtol=0, atol=1e-13)
    omega = rng.standard_normal(N**2)
    f = qf.shr2fun(np.r_[omega, np.zeros(3*N**2)], N=2*N, berezin=False)  # band 2N resolves f**2
    assert abs(qf.fun2shr(f)[0] - omega[0]) < 1e-12          # mean value
    assert abs(qf.fun2shr(f**2)[0] - np.sum(omega**2)) < 1e-10  # Parseval


def test_shr2fun_berezin_default(rng):
    N = 8
    omega = rng.standard_normal(N**2)
    bw = qf.berezin_multipliers(N)
    np.testing.assert_allclose(qf.fun2shr(qf.shr2fun(omega)), omega*bw, rtol=0, atol=1e-12)
    np.testing.assert_allclose(qf.fun2shr(qf.shr2fun(omega, berezin=False)), omega, rtol=0, atol=1e-12)


def test_sphgrid():
    N = 6
    theta, phi = qf.sphgrid(N)
    assert theta.shape == phi.shape == (N, 2*N - 1)
    np.testing.assert_allclose(theta[:, 0], (2*np.arange(N) + 1)*np.pi/(2*N - 1))
    np.testing.assert_allclose(phi[0], 2*np.pi*np.arange(2*N - 1)/(2*N - 1))


def test_img_round_trip(rng):
    f = rng.standard_normal((5, 9))
    lim = np.abs(f).max()
    img = qf.fun2img(f)
    assert img.dtype == np.uint8
    np.testing.assert_allclose(qf.img2fun(img, lim=lim), f, rtol=0, atol=2*lim/255)


def test_as_fun_non_skewh_matrix(rng):
    N = 6
    W = rng.standard_normal((N, N)) + 1j*rng.standard_normal((N, N))
    np.testing.assert_allclose(qf.as_fun(W), qf.shc2fun(qf.mat2shc(W), N=N), rtol=0, atol=1e-12)


def test_utils_dtypes_time_and_berezin():
    assert qf.complex_dtype(np.float32) is np.csingle and qf.real_dtype(np.complex128) is np.double
    N = 9
    assert qf.qtime2seconds(1.0, N) == qf.hbar(N)
    assert abs(qf.seconds2qtime(qf.qtime2seconds(0.7, N), N) - 0.7) < 1e-15
    el = qf.ind2elm(np.arange(N**2))[0]
    expected = [np.sqrt(np.prod([(N - j)/(N + j) for j in range(1, l + 1)])) for l in el]
    np.testing.assert_allclose(qf.berezin_multipliers(N), expected, rtol=1e-12)


def test_as_fun_complex_coefficients_with_N(rng):
    N = 6
    omegac = rng.standard_normal(N**2) + 1j*rng.standard_normal(N**2)  # not a real function
    f = qf.as_fun(omegac, N=N)
    assert np.iscomplexobj(f)
    np.testing.assert_allclose(f, qf.shc2fun(omegac, N=N), rtol=0, atol=1e-12)


@pytest.mark.parametrize("xyz, expected", [
    ((1.0, 0.0, 0.0), (np.pi/2, 0.0)),
    ((0.0, -1.0, 0.0), (np.pi/2, 3*np.pi/2)),
    ((0.0, 0.0, -2.0), (np.pi, 0.0)),
])
def test_cart2sph_scalar(xyz, expected):
    theta, phi = qf.cart2sph(*xyz)
    np.testing.assert_allclose((theta, phi), expected, rtol=0, atol=1e-15)
