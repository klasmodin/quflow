"""Tests for quflow.integrators.erk."""
import numpy as np
import pytest
import quflow as qf
from quflow.integrators import erk


def _initial_state(N=8, seed=1):
    rng = np.random.default_rng(seed)
    omega = np.zeros(N**2)
    omega[1:16] = rng.standard_normal(15)  # band limited initial vorticity (el <= 3)
    return qf.shr2mat(omega, N=N)


@pytest.mark.parametrize("integrator, steps_coarse, order", [
    (erk.euler, 1280, 1),
    (erk.heun, 160, 2),
    (erk.rk4, 40, 4),
])
def test_convergence_order(integrator, steps_coarse, order):
    # Step sizes are chosen in the asymptotic regime (observed orders 1.02, 2.00, 4.17)
    W0 = _initial_state()
    N = W0.shape[-1]
    T = 10*qf.hbar(N)
    reference = qf.isomp(W0.copy(), T/8192, 8192, tol=1e-14, maxit=50)
    errors = [np.abs(integrator(W0.copy(), T/n, n) - reference).max()
              for n in (steps_coarse, 2*steps_coarse)]
    assert abs(np.log2(errors[0]/errors[1]) - order) < 0.25


def test_stack_transported_by_first(random_shr):
    # A stack [W, Theta] is transported by the stream matrix of W
    W0 = _initial_state()
    N = W0.shape[-1]
    Theta0 = qf.shr2mat(random_shr(N), N=N)
    T = 2*qf.hbar(N)
    V = erk.rk4(np.array([W0, Theta0]), T/200, 200)
    V_ref = qf.isomp(np.array([W0, Theta0]), T/4096, 4096, tol=1e-14, maxit=50)
    assert np.abs(V - V_ref).max() < 1e-6


def test_euler_stats():
    stats = {}
    erk.euler(_initial_state(), 0.01, steps=7, stats=stats)
    erk.euler(_initial_state(), 0.01, steps=3, stats=stats)
    assert stats["steps"] == 10
