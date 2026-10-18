r"""
Classical explicit Runge-Kutta integrators for the quantized Euler equations.

The integrators solve

.. math::

    \dot W = \frac{1}{\hbar}[P, W] + F(P, W), \qquad P = H(W),

where :math:`H` is the Hamiltonian (by default
:func:`quflow.laplacian.solve_poisson`), :math:`F` an optional forcing, and
:math:`\hbar = 2/\sqrt{N^2-1}`. The right-hand side is computed with
:func:`quflow.geometry.bracket`.

Unlike the isospectral integrators of :mod:`quflow.integrators.isospectral`,
these methods do not preserve the spectrum of :math:`W` (the Casimirs); the
drift decreases with the order of the method. They are mainly useful for
comparison and testing.

Integrators
-----------
euler
    Explicit Euler method, first order.
heun
    Heun's method, second order (alias ``explicit``).
rk4
    Classical Runge-Kutta method, fourth order.

All integrators take the time step ``dt`` in physical time units, overwrite
their input ``W`` and return it. ``W`` may also be a stack of matrices of
shape ``(k, N, N)``; all of them are then transported with the stream matrix
returned by the Hamiltonian.
"""
import numpy as np
import scipy.linalg
import inspect
import numba as nb

from ..laplacian import solve_poisson, solve_heat
from ..laplacian import select_skewherm as select_skewherm_laplacian
from ..geometry import norm_Linf, bracket
from .isospectral import commutator
from .isospectral import update_stats


# -------------------------------------------------
# CLASSICAL (EXPLICIT, NON-ISOSPECTRAL) INTEGRATORS
# -------------------------------------------------

def euler(W: np.ndarray,
          dt: float,
          steps: int = 100,
          hamiltonian=solve_poisson,
          forcing=None,
          stats: dict = None,
          **kwargs) -> np.ndarray:
    r"""
    Time-stepping by Euler's explicit first order method.

    Each step computes :math:`W_{n+1} = W_n + \Delta t\, f(W_n)`, where
    :math:`f(W) = \frac{1}{\hbar}[H(W), W] + F(H(W), W)`.

    Parameters
    ----------
    W: ndarray, shape (N, N) or (k, N, N), complex
        Initial vorticity (overwritten and returned).
    dt: float
        Time step length in physical time units.
    steps: int
        Number of steps to take. Default is 100.
    hamiltonian: function(W)
        The Hamiltonian returning a stream matrix. Default is
        :func:`quflow.laplacian.solve_poisson`.
    forcing: None or function(P, W)
        Extra force function (to allow non-isospectral perturbations).
    stats: None or dict
        Dictionary with statistics. If given, the number of steps is added to
        its entry ``'steps'``.
    **kwargs: dict
        Extra keyword arguments (accepted and ignored, for interface
        compatibility with the other integrators).

    Returns
    -------
    W: ndarray
        The state after `steps` steps (the same array as the input `W`).
    """
    if forcing is None:
        rhs = bracket
    else:
        def rhs(P, W):
            return bracket(P, W) + forcing(P, W)

    for k in range(steps):
        P = hamiltonian(W)
        VF = rhs(P, W)
        W += dt*VF

    if stats is not None:
        update_stats(stats, steps=steps)

    return W


def heun(W, dt, steps=100, hamiltonian=solve_poisson, forcing=None):
    r"""
    Time-stepping by Heun's second order method.

    Each step computes the predictor :math:`W' = W_n + \Delta t\, f(W_n)` and
    then :math:`W_{n+1} = W_n + \tfrac{\Delta t}{2}\big(f(W_n) + f(W')\big)`,
    where :math:`f(W) = \frac{1}{\hbar}[H(W), W] + F(H(W), W)`.

    Parameters
    ----------
    W: ndarray, shape (N, N) or (k, N, N), complex
        Initial vorticity (overwritten and returned).
    dt: float
        Time step length in physical time units.
    steps: int
        Number of steps to take. Default is 100.
    hamiltonian: function(W)
        The Hamiltonian returning a stream matrix. Default is
        :func:`quflow.laplacian.solve_poisson`.
    forcing: function(P, W) or None (default)
        Extra force function (to allow non-isospectral perturbations).

    Returns
    -------
    W: ndarray
        The state after `steps` steps (the same array as the input `W`).

    Notes
    -----
    Unlike :func:`euler`, this function accepts no other keyword arguments.
    """
    if forcing is None:
        rhs = bracket
    else:
        def rhs(P, W):
            return bracket(P, W) + forcing(P, W)

    for k in range(steps):

        # Evaluate RHS at W
        P = hamiltonian(W)
        F0 = rhs(P, W)

        # Compute Heun predictor
        Wprime = W + dt*F0

        # Evaluate RHS at predictor WP
        P = hamiltonian(Wprime)
        F = rhs(P, Wprime)

        # Compute averaged RHS
        F += F0
        F *= dt/2.0

        # Update W
        W += F

    return W


def rk4(W, dt, steps=100, hamiltonian=solve_poisson, forcing=None):
    r"""
    Time-stepping by the classical Runge-Kutta fourth order method.

    The right-hand side is :math:`f(W) = \frac{1}{\hbar}[H(W), W] + F(H(W), W)`.

    Parameters
    ----------
    W: ndarray, shape (N, N) or (k, N, N), complex
        Initial vorticity (overwritten and returned).
    dt: float
        Time step length in physical time units.
    steps: int
        Number of steps to take. Default is 100.
    hamiltonian: function(W)
        The Hamiltonian returning a stream matrix. Default is
        :func:`quflow.laplacian.solve_poisson`.
    forcing: function(P, W) or None (default)
        Extra force function (to allow non-isospectral perturbations).

    Returns
    -------
    W: ndarray
        The state after `steps` steps (the same array as the input `W`).

    Notes
    -----
    Unlike :func:`euler`, this function accepts no other keyword arguments.

    Examples
    --------
    >>> import numpy as np
    >>> import quflow as qf
    >>> N = 16
    >>> omega = np.random.default_rng(0).standard_normal(N**2)
    >>> omega[0] = 0.0
    >>> W = qf.shr2mat(omega, N=N)
    >>> W = qf.integrators.rk4(W, dt=0.1*qf.hbar(N), steps=10)
    """
    if forcing is None:
        rhs = bracket
    else:
        def rhs(P, W):
            return bracket(P, W) + forcing(P, W)

    for k in range(steps):
        P = hamiltonian(W)
        K1 = rhs(P, W)

        Wprime = W + (dt/2.0)*K1
        P = hamiltonian(Wprime)
        K2 = rhs(P, Wprime)

        Wprime = W + (dt/2.0)*K2
        P = hamiltonian(Wprime)
        K3 = rhs(P, Wprime)

        Wprime = W + dt*K3
        P = hamiltonian(Wprime)
        K4 = rhs(P, Wprime)

        W += (dt/6.0)*(K1+2*K2+2*K3+K4)

    return W


# Default explicit integrator
explicit = heun

