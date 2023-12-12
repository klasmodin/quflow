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
    f = qf.shc2fun(omega)
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
