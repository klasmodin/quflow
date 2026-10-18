"""Consistency tests for all Laplacian backends (trace handling and caches).

The reference operator is ``quflow.laplacian.cpu.laplace``, which equals the
double commutator with the so(3) generators (tested in test_laplacian_cpu.py).
"""
import numpy as np
import pytest
import quflow as qf
import quflow.laplacian.cpu as cpu
import quflow.laplacian.direct as direct
import quflow.laplacian.tridiagonal as tridiagonal
import quflow.laplacian.gpu as gpu


N = 9
BACKENDS = [cpu, direct, tridiagonal, gpu]


def backends_with(name):
    return [pytest.param(b, id=b.__name__.split(".")[-1]) for b in BACKENDS if hasattr(b, name)]


@pytest.fixture
def W_with_trace(random_shr):
    """Skew-Hermitian matrix with nonzero trace (el = 0 component)."""
    W = qf.shr2mat(random_shr(N, zero_mean=False), N=N)
    assert abs(np.trace(W)) > 0.1
    return W


def max_abs(A):
    return np.abs(np.asarray(A)).max()


def lap(P):
    return cpu.laplace(np.asarray(P))


@pytest.mark.parametrize("backend", backends_with("solve_poisson"))
def test_solve_poisson_with_trace(backend, W_with_trace):
    # Trace-free P with ΔP = W - tr(W)/N I
    W = W_with_trace
    P = np.array(backend.solve_poisson(W.copy()))
    assert abs(np.trace(P)) < 1e-12
    assert max_abs(lap(P) - (W - np.trace(W)/N*np.eye(N))) < 1e-12


@pytest.mark.parametrize("backend", backends_with("solve_heat"))
def test_solve_heat_keeps_trace(backend, W_with_trace):
    hnu = 0.05
    Wh = np.array(backend.solve_heat(hnu, W_with_trace.copy()))
    assert max_abs(Wh - hnu*lap(Wh) - W_with_trace) < 1e-12


@pytest.mark.parametrize("backend", backends_with("solve_helmholtz"))
def test_solve_helmholtz_keeps_trace(backend, W_with_trace):
    alpha = 0.1
    P = np.array(backend.solve_helmholtz(W_with_trace.copy(), alpha=alpha))
    assert max_abs(P - alpha*lap(P) - W_with_trace) < 1e-12


@pytest.mark.parametrize("backend", backends_with("solve_viscdamp"))
@pytest.mark.parametrize("theta", [1.0, 0.5])
def test_solve_viscdamp_keeps_trace(backend, theta, W_with_trace):
    h, nu, alpha = 0.1, 1e-2, 0.5
    W0 = W_with_trace
    Wh = np.array(backend.solve_viscdamp(h, W0.copy(), nu=nu, alpha=alpha, theta=theta))
    lhs = (1 + h*alpha*theta)*Wh - h*nu*theta*lap(Wh)
    rhs = (1 - h*alpha*(1 - theta))*W0 + h*nu*(1 - theta)*lap(W0)
    assert max_abs(lhs - rhs) < 1e-12


@pytest.mark.parametrize("backend", backends_with("solve_viscdamp"))
def test_solve_viscdamp_theta_not_mixed_up(backend, W_with_trace):
    # Same h, nu, alpha but different theta must not reuse the cached operator
    h, nu, alpha = 0.1, 1e-2, 0.37
    W0 = W_with_trace
    backend.solve_viscdamp(h, W0.copy(), nu=nu, alpha=alpha, theta=0.5)
    Wh = np.array(backend.solve_viscdamp(h, W0.copy(), nu=nu, alpha=alpha, theta=1.0))
    assert max_abs((1 + h*alpha)*Wh - h*nu*lap(Wh) - W0) < 1e-12


@pytest.mark.parametrize("solver", [
    lambda W: gpu.solve_heat(0.05, W),
    lambda W: gpu.solve_helmholtz(W, alpha=0.1),
    lambda W: gpu.solve_viscdamp(0.1, W, nu=1e-2, alpha=0.5),
], ids=["heat", "helmholtz", "viscdamp"])
def test_gpu_solvers_respect_dtype(solver, W_with_trace):
    # Same parameters with a new dtype must work, in that dtype
    W = W_with_trace
    r128 = solver(W).copy()
    r64 = solver(W.astype(np.complex64)).copy()
    r128_again = solver(W).copy()
    assert r64.dtype == np.complex64
    np.testing.assert_array_equal(r128_again, r128)


def test_direct_nonskewh_keeps_trace(W_with_trace):
    old = direct.select_skewherm(False)
    try:
        Wh = direct.solve_heat(0.05, W_with_trace.copy())
        P = direct.solve_poisson(W_with_trace.copy())
    finally:
        direct.select_skewherm(old)
    assert max_abs(Wh - 0.05*lap(Wh) - W_with_trace) < 1e-12
    assert max_abs(lap(P) - (W_with_trace - np.trace(W_with_trace)/N*np.eye(N))) < 1e-12
