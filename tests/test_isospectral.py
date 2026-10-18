"""Tests for quflow.integrators.isospectral (input handling and options)."""
import numpy as np
import pytest
import quflow as qf
import quflow.physics
import quflow.integrators.erk
from quflow.integrators import isospectral


N = 12


@pytest.fixture
def W0(random_shr):
    return qf.shr2mat(random_shr(N), N=N)


@pytest.fixture
def dt():
    return 0.1*qf.hbar(N)


@pytest.fixture
def nonskewh_mode():
    """Switch to general (non skew-Hermitian) mode, and restore the default afterwards."""
    isospectral.select_skewherm(False)
    yield
    isospectral.select_skewherm(True)


def test_forcing_cached_array(W0, dt, random_skewh):
    # A force cached per time step (as in the example notebook) must give the
    # same result as one returning a new array, and must not be modified.
    F0 = random_skewh(N)
    cache = {"time": None, "force": np.zeros_like(F0)}

    def force_cached(P, W, time):
        if time != cache["time"]:
            np.copyto(cache["force"], F0)
            cache["time"] = time
        return cache["force"]

    def force_fresh(P, W, time):
        return F0.copy()

    W_fresh = qf.isomp(W0.copy(), dt, 5, forcing=force_fresh, time=0.0)
    W_cached = qf.isomp(W0.copy(), dt, 5, forcing=force_cached, time=0.0)
    np.testing.assert_array_equal(W_cached, W_fresh)
    np.testing.assert_array_equal(cache["force"], F0)


def test_hamiltonian_stored_array(W0, dt, random_skewh):
    # A Hamiltonian returning a stored stream matrix (here constant) must not
    # have its array modified by the integrator.
    P0 = random_skewh(N)
    P0_copy = P0.copy()

    W_stored = qf.isomp(W0.copy(), dt, 5, hamiltonian=lambda W: P0)
    W_fresh = qf.isomp(W0.copy(), dt, 5, hamiltonian=lambda W: P0_copy.copy())
    np.testing.assert_array_equal(P0, P0_copy)
    np.testing.assert_array_equal(W_stored, W_fresh)


def test_forcing_receives_unscaled_stream_matrix(W0, dt):
    # The forcing must see the stream matrix itself, not a scaled version
    received = []

    def force(P, W):
        received.append(P.copy())
        return np.zeros_like(W)

    qf.isomp(W0.copy(), dt, 1, forcing=force, maxit=1, reinitialize=True)
    # With one iteration from W_tilde = W0, the stream matrix is that of W0
    np.testing.assert_allclose(received[-1], qf.solve_poisson(W0.copy()), rtol=0, atol=1e-13)


@pytest.mark.parametrize("integrator", [isospectral.isomp_quasinewton, isospectral.isomp_simple])
def test_forcing_not_supported_raises(integrator, W0, dt):
    W = W0.copy()
    with pytest.raises(NotImplementedError):
        integrator(W, dt, 1, forcing=lambda P, W: np.zeros_like(W))
    np.testing.assert_array_equal(W, W0)


def test_quasinewton_nonskewh_raises(W0, dt, nonskewh_mode):
    with pytest.raises(NotImplementedError):
        isospectral.isomp_quasinewton(W0.copy(), dt, 1)


def test_compsum_with_forcing_raises_before_stepping(W0, dt):
    W = W0.copy()
    with pytest.raises(NotImplementedError):
        qf.isomp(W, dt, 2, compsum=True, forcing=lambda P, W: np.zeros_like(W))
    np.testing.assert_array_equal(W, W0)


@pytest.mark.parametrize("tol", ["auto", 1e-10])
def test_stats_empty_dict_is_filled(W0, dt, tol):
    stats = {}
    qf.isomp(W0.copy(), dt, 3, stats=stats, tol=tol)
    assert stats["iterations"] >= 1
    assert 0 <= stats["number_of_maxit"] <= 1
    assert ("tol_auto" in stats) == (tol == "auto")


def test_commutator_follows_select_skewherm(random_skewh, rng, nonskewh_mode):
    # Non-skew-Hermitian matrices: only the generic commutator is correct
    A = rng.standard_normal((N, N)) + 1j*rng.standard_normal((N, N))
    B = rng.standard_normal((N, N)) + 1j*rng.standard_normal((N, N))
    expected = A@B - B@A
    # All references to `commutator` must follow the global setting,
    # also those imported by name in other modules.
    for comm in [isospectral.commutator, qf.integrators.commutator,
                 quflow.physics.commutator, quflow.integrators.erk.commutator]:
        np.testing.assert_allclose(comm(A, B), expected, rtol=0, atol=1e-12)


def test_commutator_default_skewherm(random_skewh):
    A, B = random_skewh(N), random_skewh(N)
    np.testing.assert_allclose(qf.integrators.commutator(A, B), A@B - B@A, rtol=0, atol=1e-12)
