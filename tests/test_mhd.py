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
    assert stats['number_of_maxit'] == 0  # the iteration converged at every step
    spectrum = lambda A: np.sort(np.linalg.eigvalsh(1j*A))
    np.testing.assert_allclose(spectrum(Theta), spectrum(Theta0), rtol=0, atol=1e-12)
    for k in (1, 2, 3):
        c0 = np.trace(W0 @ np.linalg.matrix_power(Theta0, k))
        c1 = np.trace(W @ np.linalg.matrix_power(Theta, k))
        assert abs(c1 - c0) < 1e-10*max(1.0, abs(c0))


# --- Input handling -----------------------------------------------------------

def test_magmp_hamiltonian_stored_arrays(random_skewh):
    # A Hamiltonian returning stored arrays must not have them modified
    V0 = _mhd_state()
    N = V0.shape[-1]
    P0, B0 = random_skewh(N), random_skewh(N)
    P0c, B0c = P0.copy(), B0.copy()
    V_stored = mhd.magmp(V0.copy(), 0.1*qf.hbar(N), 3, hamiltonian=lambda V: (P0, B0))
    V_fresh = mhd.magmp(V0.copy(), 0.1*qf.hbar(N), 3, hamiltonian=lambda V: (P0c.copy(), B0c.copy()))
    np.testing.assert_array_equal(P0, P0c)
    np.testing.assert_array_equal(B0, B0c)
    np.testing.assert_array_equal(V_stored, V_fresh)


def test_magmp_forcing_cached_array():
    # A force cached per time step must give the same result as fresh arrays
    V0 = _mhd_state()
    N = V0.shape[-1]
    F0 = 0.1*_mhd_state(seed=5)
    cache = {"time": None, "force": np.zeros_like(F0)}

    def force_cached(P, V, time):
        if time != cache["time"]:
            np.copyto(cache["force"], F0)
            cache["time"] = time
        return cache["force"]

    def force_fresh(P, V, time):
        return F0.copy()

    V_fresh = mhd.magmp(V0.copy(), 0.1*qf.hbar(N), 3, forcing=force_fresh, time=0.0)
    V_cached = mhd.magmp(V0.copy(), 0.1*qf.hbar(N), 3, forcing=force_cached, time=0.0)
    np.testing.assert_array_equal(V_cached, V_fresh)
    np.testing.assert_array_equal(cache["force"], F0)


@pytest.mark.parametrize("integrator, dt", [
    (mhd.magmp_fixedpoint, 0.1*qf.hbar(N)),
    (mhd.magmp_fixedpoint_old, 0.1),
])
def test_magmp_stats_empty_dict_is_filled(integrator, dt):
    stats = {}
    integrator(_mhd_state(), dt, 3, stats=stats)
    # Same keys as isomp_fixedpoint (these are logged by QuSimulation)
    assert set(stats) == {"tol_auto", "iterations", "number_of_maxit"}


def test_magmp_old_time_advances_from_zero():
    times = []

    def hamiltonian(V, time):
        times.append(time)
        return mhd.solve_mhd(V)

    mhd.magmp_fixedpoint_old(_mhd_state(), 0.1, 3, hamiltonian=hamiltonian, time=0.0, maxit=1)
    # One evaluation per step, at the midpoint time of each step
    np.testing.assert_allclose(times, [0.05, 0.15, 0.25])


def test_magmp_stats_logged_by_qusimulation(tmp_path):
    from quflow.simulation import QuSimulation
    V0 = _mhd_state()
    N = V0.shape[-1]
    sim = QuSimulation(tmp_path/"mhd.hdf5", overwrite=True, state=V0)
    sim['dt'] = 0.1*qf.hbar(N)
    sim['simtime'] = 0.5*qf.hbar(N)
    sim['inner_time'] = 0.25*qf.hbar(N)
    sim['integrator'] = mhd.magmp
    sim['hamiltonian'] = mhd.solve_mhd
    qf.solve(sim, progress_bar=False)
    # All statistics fields are logged at every output time
    n = len(sim['time'])
    assert n >= 2
    for name in ['tol_auto', 'iterations', 'number_of_maxit']:
        assert len(sim[name]) == n, name
    assert np.all(sim['tol_auto'][1:] > 0)
    assert np.all(sim['iterations'][1:] >= 1)
