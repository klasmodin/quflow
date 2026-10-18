"""Tests for quflow.laplacian.cpu (trace handling, caches and kernels)."""
import numpy as np
import pytest
import quflow as qf
from scipy.sparse import dia_matrix
import quflow.laplacian.cpu as cpu


N = 9


@pytest.fixture
def W_with_trace(random_shr):
    """Skew-Hermitian matrix with nonzero trace (el = 0 component)."""
    W = qf.shr2mat(random_shr(N, zero_mean=False), N=N)
    assert abs(np.trace(W)) > 0.1
    return W


@pytest.fixture(params=[True, False], ids=["skewh", "nonskewh"])
def skewh_mode(request):
    """Run in both solver modes, and restore the default afterwards."""
    cpu.select_skewherm(request.param)
    yield request.param
    cpu.select_skewherm(True)


@pytest.fixture
def fresh_caches():
    """Clear the operator and buffer caches before and after the test."""
    caches = [cpu._cpu_heat_cache, cpu._cpu_helmholtz_cache, cpu._cpu_globalqg_cache,
              cpu._cpu_viscdamp_cache, cpu._cpu_buffer_cache]
    for c in caches:
        c.clear()
    yield
    for c in caches:
        c.clear()


def max_abs(A):
    return np.abs(A).max()


# --- Non-singular operators keep the trace of the solution -------------------

def test_solve_heat_keeps_trace(W_with_trace, skewh_mode):
    hnu = 0.05
    Wh = cpu.solve_heat(hnu, W_with_trace).copy()
    assert max_abs(Wh - hnu*cpu.laplace(Wh) - W_with_trace) < 1e-12
    # The heat flow conserves the mean (trace)
    assert abs(np.trace(Wh) - np.trace(W_with_trace)) < 1e-12


def test_solve_helmholtz_keeps_trace(W_with_trace, skewh_mode):
    alpha = 0.1
    P = cpu.solve_helmholtz(W_with_trace, alpha=alpha).copy()
    assert max_abs(P - alpha*cpu.laplace(P) - W_with_trace) < 1e-12


@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_solve_viscdamp_keeps_trace(W_with_trace, theta, skewh_mode):
    h, nu, alpha = 0.1, 1e-2, 0.5
    F = 0.3*W_with_trace
    Wh = cpu.solve_viscdamp(h, W_with_trace, nu=nu, alpha=alpha, force=F, theta=theta).copy()
    lhs = (1 + h*alpha*theta)*Wh - h*nu*theta*cpu.laplace(Wh)
    rhs = (1 - h*alpha*(1 - theta))*W_with_trace + h*nu*(1 - theta)*cpu.laplace(W_with_trace) + h*F
    assert max_abs(lhs - rhs) < 1e-12


@pytest.mark.parametrize("zero_trace", [True, False])
def test_solve_globalqg_exact(W_with_trace, zero_trace, skewh_mode):
    # The global QG equation  ΔP + (γ/2)(Z²P + PZ²) = W  must hold on all
    # entries, also on the diagonal (the operator is not singular for γ > 0).
    W = W_with_trace - np.trace(W_with_trace)/N*np.eye(N) if zero_trace else W_with_trace
    gamma = 0.7
    Z = qf.geometry.cartesian_generators(N)[-1]
    P = cpu.solve_globalqg(W, gamma=gamma).copy()
    assert max_abs(cpu.laplace(P) + gamma/2*(Z@Z@P + P@Z@Z) - W) < 1e-12


# --- Poisson: trace removed, dense and sparse agree --------------------------

def test_solve_poisson_trace_free(W_with_trace, skewh_mode):
    W = W_with_trace
    P = cpu.solve_poisson(W).copy()
    assert abs(np.trace(P)) < 1e-12
    assert max_abs(cpu.laplace(P) - (W - np.trace(W)/N*np.eye(N))) < 1e-12


def test_solve_poisson_sparse_matches_dense(W_with_trace):
    Wband = sum(np.diag(np.diag(W_with_trace, k), k) for k in (-1, 0, 1))
    P_sparse = cpu.solve_poisson(dia_matrix(Wband)).toarray()
    P_dense = cpu.solve_poisson(Wband).copy()
    assert max_abs(P_sparse - P_dense) < 1e-12


def test_solve_poisson_sparse_pure_trace():
    # The el = 0 mode (identity) is in the kernel: its stream matrix is zero
    P = cpu.solve_poisson(dia_matrix(1j*np.eye(N))).toarray()
    assert max_abs(P) < 1e-12


# --- Caches -------------------------------------------------------------------

def test_solve_viscdamp_theta_not_mixed_up(W_with_trace, fresh_caches):
    h, nu, alpha = 0.1, 1e-2, 0.5
    cpu.solve_viscdamp(h, W_with_trace, nu=nu, alpha=alpha, theta=0.5)
    W1 = cpu.solve_viscdamp(h, W_with_trace, nu=nu, alpha=alpha, theta=1.0).copy()
    cpu._cpu_viscdamp_cache.clear()
    W1_fresh = cpu.solve_viscdamp(h, W_with_trace, nu=nu, alpha=alpha, theta=1.0).copy()
    np.testing.assert_array_equal(W1, W1_fresh)


@pytest.mark.parametrize("solver", [
    lambda W: cpu.solve_heat(0.05, W),
    lambda W: cpu.solve_helmholtz(W, alpha=0.1),
    lambda W: cpu.solve_globalqg(W, gamma=0.7),
    lambda W: cpu.solve_viscdamp(0.1, W, nu=1e-2, alpha=0.5),
], ids=["heat", "helmholtz", "globalqg", "viscdamp"])
def test_solvers_respect_dtype(W_with_trace, solver, fresh_caches):
    W = W_with_trace
    ref = solver(W).copy()
    for c in [cpu._cpu_heat_cache, cpu._cpu_helmholtz_cache, cpu._cpu_globalqg_cache,
              cpu._cpu_viscdamp_cache, cpu._cpu_buffer_cache]:
        c.clear()
    # Single precision first, then double precision
    r64 = solver(W.astype(np.complex64)).copy()
    r128 = solver(W).copy()
    assert r64.dtype == np.complex64
    assert r128.dtype == np.complex128
    np.testing.assert_array_equal(r128, ref)


# --- Alternative kernels ------------------------------------------------------

def test_dot_cpu_skewh2(W_with_trace):
    W = np.zeros_like(W_with_trace)
    cpu._dot_cpu_skewh2(cpu.laplacian(N), W_with_trace, W)
    assert max_abs(W - cpu.laplace(W_with_trace)) < 1e-12


@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_solve_cpu_generic_matches_default(n, random_shr):
    W = qf.shr2mat(random_shr(n, zero_mean=False), N=n)
    lap = cpu.laplacian(n, bc=True)
    P = np.zeros_like(W)
    cpu._solve_cpu_generic(lap, W, P, np.zeros((n, n)), np.zeros_like(W), True)
    assert max_abs(P - cpu.solve_poisson(W)) < 1e-12
