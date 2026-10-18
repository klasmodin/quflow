"""Tests for quflow.integrators.mhd."""
import numpy as np
import pytest
import quflow as qf
from quflow.integrators import isospectral, mhd


N = 8


@pytest.fixture
def state(random_shr):
    """MHD state [W, Theta] of two skew-Hermitian matrices."""
    return np.array([qf.shr2mat(random_shr(N), N=N), qf.shr2mat(random_shr(N), N=N)])


@pytest.fixture
def nonskewh_mode():
    """Switch to general (non skew-Hermitian) mode, and restore the default afterwards."""
    isospectral.select_skewherm(False)
    yield
    isospectral.select_skewherm(True)


@pytest.mark.parametrize("integrator, dt", [
    (mhd.magmp_fixedpoint, 0.1*qf.hbar(N)),
    (mhd.magmp_fixedpoint_old, 0.1),
])
def test_magmp_requires_skewherm_mode(integrator, dt, state, nonskewh_mode):
    # The setting must be read at call time, not at import time
    with pytest.raises(NotImplementedError):
        integrator(state.copy(), dt, 1)


def test_magmp_runs_in_skewherm_mode(state):
    out = mhd.magmp(state.copy(), 0.1*qf.hbar(N), 2)
    assert out.shape == state.shape


def _mhd_state(N=8, seed=1, theta_scale=0.2):
    rng = np.random.default_rng(seed)
    omega = np.zeros(N**2)
    omega[1:16] = rng.standard_normal(15)
    omega_theta = np.zeros(N**2)
    omega_theta[1:16] = theta_scale*rng.standard_normal(15)
    return np.array([qf.shr2mat(omega, N=N), qf.shr2mat(omega_theta, N=N)])


def test_magmp_convergence_order():
    # Small Theta and short time: asymptotic regime (observed order 2.00)
    V0 = _mhd_state()
    N = V0.shape[-1]
    T = 2*qf.hbar(N)
    reference = mhd.magmp(V0.copy(), T/8192, 8192, tol=1e-14, maxit=50)
    errors = [np.abs(mhd.magmp(V0.copy(), T/n, n, tol=1e-14, maxit=50) - reference).max()
              for n in (80, 160)]
    assert abs(np.log2(errors[0]/errors[1]) - 2) < 0.1


def test_magmp_conserved_quantities():
    # Spectrum of Theta and tr(W Theta^k) are conserved up to rounding, as long as
    # the fixed-point iteration converges (it does for this state and step size)
    V0 = _mhd_state(theta_scale=0.2)
    N = V0.shape[-1]
    W0, Theta0 = V0
    stats = {'iterations': 0.0}
    W, Theta = mhd.magmp(V0.copy(), 0.1*qf.hbar(N), 100, tol=1e-14, maxit=50, stats=stats)
    assert stats['maxit'] == 0  # the iteration converged at every step
    spectrum = lambda A: np.sort(np.linalg.eigvalsh(1j*A))
    np.testing.assert_allclose(spectrum(Theta), spectrum(Theta0), rtol=0, atol=1e-12)
    for k in (1, 2, 3):
        c0 = np.trace(W0 @ np.linalg.matrix_power(Theta0, k))
        c1 = np.trace(W @ np.linalg.matrix_power(Theta, k))
        assert abs(c1 - c0) < 1e-10*max(1.0, abs(c0))
