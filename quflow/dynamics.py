import numpy as np
import scipy.linalg

from .laplacian.direct import solve_poisson
from .transforms import as_shr
from .utils import elm2ind
from numba import njit

# ----------------
# GLOBAL VARIABLES
# ----------------


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------

def commutator(W, P):
    """
    Efficient computations of commutator for skew-Hermitian matrices.
    Warnings: only works for skew-Hermitian.

    Parameters
    ----------
    W: ndarray
    P: ndarray

    Returns
    -------
    ndarray
    """
    VF = W@P
    VF -= VF.conj().T
    return VF


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def euler(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson):
    """
    Time-stepping by Euler's explicit first order method.

    Parameters
    ----------
    W: ndarray
        Initial vorticity (overwritten and returned).
    stepsize: float
        Time step length.
    steps: int
        Number of steps to take.
    hamiltonian: function
        The Hamiltonian returning a stream matrix.

    Returns
    -------
    W: ndarray
    """
    for k in range(steps):

        P = hamiltonian(W)
        VF = commutator(P, W)
        W += stepsize*VF

    return W


def heun(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson):
    """
    Time-stepping by Heun's second order method.

    Parameters
    ----------
    W: ndarray
        Initial vorticity (overwritten and returned).
    stepsize: float
        Time step length.
    steps: int
        Number of steps to take.
    hamiltonian: function
        The Hamiltonian returning a stream matrix.

    Returns
    -------
    W: ndarray
    """
    for k in range(steps):

        # Evaluate RHS at W
        P = hamiltonian(W)
        F0 = commutator(P, W)

        # Compute Heun predictor
        Wprime = W + stepsize*F0

        # Evaluate RHS at predictor WP
        P = hamiltonian(Wprime)
        F = commutator(P, Wprime)

        # Compute averaged RHS
        F += F0
        F *= stepsize/2.0

        # Update W
        W += F

    return W


def rk4(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson):
    """
    Time-stepping by the classical Runge-Kutta fourth order method.

    Parameters
    ----------
    W: ndarray
        Initial vorticity (overwritten and returned).
    stepsize: float
        Time step length.
    steps: int
        Number of steps to take.
    hamiltonian: function
        The Hamiltonian returning a stream matrix.

    Returns
    -------
    W: ndarray
    """
    for k in range(steps):
        P = hamiltonian(W)
        K1 = commutator(P, W)

        Wprime = W + (stepsize/2.0)*K1
        P = hamiltonian(Wprime)
        K2 = commutator(P, Wprime)

        Wprime = W + (stepsize/2.0)*K2
        P = hamiltonian(Wprime)
        K3 = commutator(P, Wprime)

        Wprime = W + stepsize*K3
        P = hamiltonian(Wprime)
        K4 = commutator(P, Wprime)

        W += (stepsize/6.0)*(K1+2*K2+2*K3+K4)

    return W


def isomp(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, tol=1e-8, maxit=10, verbatim=False):
    """
    Time-stepping by isospectral midpoint second order method.

    Parameters
    ----------
    W: ndarray
        Initial vorticity (overwritten and returned).
    stepsize: float
        Time step length.
    steps: int
        Number of steps to take.
    hamiltonian: function
        The Hamiltonian returning a stream matrix.
    tol: float
        Tolerance for iterations.
    maxit: int
        Maximum number of iterations.
    verbatim: bool
        Print extra information if True. Default is False.

    Returns
    -------
    W: ndarray
    """
    Id = np.eye(W.shape[0])

    Wtilde = W.copy()

    total_iterations = 0

    for k in range(steps):

        for i in range(maxit):

            # Update iterations
            total_iterations += 1

            # Update Ptilde
            Ptilde = hamiltonian(Wtilde)

            # Compute matrix A
            A = Id - stepsize/2.0*Ptilde

            # Compute LU of A
            luA, piv = scipy.linalg.lu_factor(A)

            # Solve first equation for B
            B = scipy.linalg.lu_solve((luA, piv), W)

            # Solve second equation for Wtilde
            Wtilde_new = scipy.linalg.lu_solve((luA, piv), -B.conj().T)

            # Compute error
            resnorm = scipy.linalg.norm(Wtilde - Wtilde_new, np.inf)

            # Update variables
            Wtilde = (Wtilde_new - Wtilde_new.conj().T)/2.0

            # Check error
            if resnorm < tol:
                break

        else:
            # We used maxit iterations
            if verbatim:
                print("Max iterations {} reached!".format(maxit))

        # Update W before returning
        W_new = A.conj().T@Wtilde@A
        np.copyto(W, W_new)

    if verbatim:
        print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))

    return W


def scale_decomposition(W, P=None, hamiltonian=solve_poisson):
    """
    Perform canonical scale separation.

    Parameters
    ----------
    W: ndarray
        Vorticity matrix.
    P: ndarray (optional)
        Stream matrix. Computed if not given.
    hamiltonian: function
        The Hamiltonian returning a stream matrix.

    Returns
    -------
    (Ws, Wr): tuple of ndarray
    """
    if P is None:
        P = hamiltonian(W)

    D, E = np.linalg.eig(P)
    EWE = E.conj().T@W@E
    D2 = np.diag(np.diag(EWE))
    Ws = E@D2@E.conj().T
    Wr = W - Ws

    return Ws, Wr


def energy_spectrum(data):
    """
    Return energy spectrum for `data` in either W, omegar, omegac, or fun format.

    Parameters
    ----------
    data: ndarray

    Returns
    -------
    energy: ndarray
    """
    omegar = as_shr(data)
    N = round(np.sqrt(omegar.shape[0]))
    energy = np.ones(N-1, dtype=float)
    for el in range(1, N):
        energy[el-1] = (omegar[elm2ind(-el, el):elm2ind(el, el)+1]**2).sum()/(el*(el+1))
    return energy


# -------
# CLASSES
# -------

class Simulation(object):
    """
    Class to keep track of an entire simulation.
    """

    def __init__(self, initial, stepsize='auto',
                 inner_steps=100, outer_steps=100, total_steps=None, N=None):
        pass



