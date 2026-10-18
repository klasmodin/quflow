r"""
Physical quantities for the quantized Euler equations: Sobolev inner
products, energy, enstrophy and sectional curvature.

With :math:`\langle\cdot,\cdot\rangle` the :math:`L^2` inner product
(:func:`quflow.geometry.inner_L2`) and :math:`\Delta_N` the quantized
Laplacian, the stream matrix :math:`P` of the vorticity :math:`W` solves
:math:`\Delta_N P = W` (:func:`quflow.laplacian.solve_poisson`), and

* the energy is
  :math:`E = -\tfrac12\langle W, P\rangle = \tfrac12\|W\|_{H^{-1}}^2 = \tfrac12\|P\|_{H^1}^2`;
  for ``W = shr2mat(omega)``, :math:`E = \tfrac12\sum_{\ell \ge 1} \omega_{\ell m}^2/(\ell(\ell+1))`;
* the enstrophy is
  :math:`\tfrac12\langle W, W\rangle = \tfrac12\sum \omega_{\ell m}^2`.

Both are conserved by the quantized Euler equations.
"""
import numpy as np

from .laplacian import solve_poisson, laplace
from .geometry import norm_Linf, norm_L2, inner_L2
from .integrators import commutator

# Define inner products

def inner_Hm1(W1, W2):
    r"""
    :math:`H^{-1}` inner product, :math:`-\langle W_1, \Delta_N^{-1} W_2\rangle`.

    The trace (:math:`\ell = 0` component) of `W2` is ignored, see
    :func:`quflow.laplacian.solve_poisson`.

    Parameters
    ----------
    W1, W2 : ndarray, shape (N, N), complex
        Skew-Hermitian matrices.

    Returns
    -------
    float
    """
    P2 = solve_poisson(W2)
    return -inner_L2(W1, P2)

def norm_Hm1(W):
    """
    :math:`H^{-1}` norm, ``sqrt(inner_Hm1(W, W))``.

    Parameters
    ----------
    W : ndarray, shape (N, N), complex
        Skew-Hermitian matrix.

    Returns
    -------
    float
    """
    return np.sqrt(inner_Hm1(W, W))

def inner_H1(P1, P2):
    r"""
    :math:`H^1` (semi-)inner product, :math:`-\langle P_1, \Delta_N P_2\rangle`.

    Parameters
    ----------
    P1, P2 : ndarray, shape (N, N), complex
        Skew-Hermitian matrices.

    Returns
    -------
    float
    """
    W2 = laplace(P2)
    return -inner_L2(P1, W2)

def norm_H1(P):
    """
    :math:`H^1` (semi-)norm, ``sqrt(inner_H1(P, P))``.

    Parameters
    ----------
    P : ndarray, shape (N, N), complex
        Skew-Hermitian matrix.

    Returns
    -------
    float
    """
    return np.sqrt(inner_H1(P, P))


# Define common functions (e.g. used as loggers)

def energy_euler(W):
    r"""
    Energy of the state with vorticity matrix `W`
    for the 2-D Euler equations.

    Computes :math:`E = -\tfrac12\langle W, P\rangle` with
    :math:`\Delta_N P = W`; see the module docstring. The trace of `W`
    does not contribute.

    Parameters
    ----------
    W : ndarray, shape (N, N), complex
        Skew-Hermitian vorticity matrix.

    Returns
    -------
    float
    """
    P = solve_poisson(W)
    return -inner_L2(W, P)/2.0

def enstrophy(W):
    r"""
    Enstrophy of the vorticity matrix `W`.

    Computes :math:`\tfrac12\langle W, W\rangle`; see the module
    docstring. Unlike the energy, it includes the trace of `W`.

    Parameters
    ----------
    W : ndarray, shape (N, N), complex
        Vorticity matrix.

    Returns
    -------
    float
    """
    return inner_L2(W, W)/2.0


def sectional_curvature(F, G):
    r"""
    Sectional curvature term for the quantized Euler equations.

    With :math:`[\cdot,\cdot]` the matrix commutator (without
    :math:`1/\hbar`, see :func:`quflow.integrators.isospectral.commutator`),
    :math:`\langle\cdot,\cdot\rangle` the :math:`L^2` inner product and
    :math:`A = [\Delta_N F, G] + [\Delta_N G, F]`, computes

    .. math::

        C = -\tfrac14\langle A, \Delta_N^{-1} A\rangle
            - \tfrac12\langle [F, G], [\Delta_N F, G] - [\Delta_N G, F]\rangle
            + \tfrac34\langle [F, G], \Delta_N [F, G]\rangle
            + \langle [\Delta_N F, F], \Delta_N^{-1}[\Delta_N G, G]\rangle.

    This is the sectional curvature :math:`C_N(F, G)` of Zeitlin's model
    from [1]_, up to the constant factor :math:`\hbar^2/(4\pi)`: the paper
    uses the bracket :math:`\frac{1}{\hbar}[\cdot,\cdot]` and the inner
    product :math:`\frac{4\pi}{N}\operatorname{tr}(A^* B)`, whereas this
    function uses the plain commutator and
    :func:`quflow.geometry.inner_L2`. As in the paper's formula, the result is
    not divided by :math:`\|F\|^2\|G\|^2 - \langle F, G\rangle^2`.

    Parameters
    ----------
    F, G : ndarray, shape (N, N), complex
        Skew-Hermitian matrices spanning the plane.

    Returns
    -------
    float

    References
    ----------
    .. [1] K. Modin and M. Perrot. *Eulerian and Lagrangian stability in
       Zeitlin's model of hydrodynamics*, Comm. Math. Phys., 2024.
    """
    DeltaF = laplace(F)
    DeltaG = laplace(G)
    FGcomm = commutator(F, G) 
    DeltaFGcomm = commutator(DeltaF, G)
    DeltaGFcomm = commutator(DeltaG, F)
    DeltaFFcomm = commutator(DeltaF, F)
    DeltaGGcomm = commutator(DeltaG, G)

    C = -inner_L2(DeltaFGcomm + DeltaGFcomm, solve_poisson(DeltaFGcomm + DeltaGFcomm))/4.0

    C -= inner_L2(FGcomm, DeltaFGcomm - DeltaGFcomm)/2.0

    C += inner_L2(FGcomm, laplace(FGcomm))*(3.0/4.0)

    C += inner_L2(DeltaFFcomm, solve_poisson(DeltaGGcomm))

    return C
