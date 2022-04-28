import numpy as np
import scipy.linalg

from .laplacian.direct import solve_poisson
from .transforms import as_shr
from .utils import elm2ind, seconds2qtime
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


def isomp(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson,
          tol=1e-8, maxit=10, verbatim=False, enforce_hermitian=True):
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
    enforce_hermitian: bool
        Enforce at every step that the vorticity matrix is hermitian.

    Returns
    -------
    W: ndarray
    """
    Id = np.eye(W.shape[0])

    Wtilde = W.copy()

    total_iterations = 0

    for k in range(steps):

        # --- Beginning of step ---

        for i in range(maxit):

            # Update iterations
            total_iterations += 1

            # Update Ptilde
            Ptilde = hamiltonian(Wtilde)

            # Compute matrix A
            A = Id - (stepsize/2.0)*Ptilde

            # Compute LU of A
            luA, piv = scipy.linalg.lu_factor(A)

            # Solve first equation for B
            B = scipy.linalg.lu_solve((luA, piv), W)

            # Solve second equation for Wtilde
            Wtilde_new = scipy.linalg.lu_solve((luA, piv), -B.conj().T)

            # Make sure solution is Hermitian (this removes drift in rounding errors)
            if enforce_hermitian:
                Wtilde_new /= 2.0
                Wtilde_new -= Wtilde_new.conj().T

            # Compute error
            resnorm = scipy.linalg.norm(Wtilde - Wtilde_new, np.inf)

            # Update variables
            Wtilde = Wtilde_new

            # Check error
            if resnorm < tol:
                break

        else:
            # We used maxit iterations
            if verbatim:
                print("Max iterations {} reached at step {}.".format(maxit, k))

        # Update W
        W_new = A.conj().T @ Wtilde @ A
        np.copyto(W, W_new)

        # --- End of step ---

    if verbatim:
        print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))

    return W


def isomp_simple(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson,
                 enforce_hermitian=True):
    """
    Time-stepping by the simplified isospectral midpoint method. This is an isospectral
    but not fully symplectic method. Nor is it reversible.

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
    enforce_hermitian: bool
        Enforce at every step that the vorticity matrix is hermitian.

    Returns
    -------
    W: ndarray
    """
    Id = np.eye(W.shape[0])

    Wtilde = W.copy()

    for k in range(steps):

        # --- Beginning of step ---

        # Update Ptilde
        Ptilde = hamiltonian(Wtilde)

        # Compute matrix A
        A = Id - (stepsize/2.0)*Ptilde

        # Compute LU of A
        luA, piv = scipy.linalg.lu_factor(A)

        # Solve first equation for B
        B = scipy.linalg.lu_solve((luA, piv), W)

        # Solve second equation for Wtilde
        Wtilde_new = scipy.linalg.lu_solve((luA, piv), -B.conj().T)

        # Make sure solution is Hermitian (this removes drift in rounding errors)
        if enforce_hermitian:
            Wtilde_new /= 2.0
            Wtilde_new -= Wtilde_new.conj().T

        # Update variables
        Wtilde = Wtilde_new

        # Update W
        W_new = A.conj().T @ Wtilde @ A
        np.copyto(W, W_new)

        # --- End of step ---

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
        energy[el-1] = (omegar[elm2ind(el, -el):elm2ind(el, el)+1]**2).sum()/(el*(el+1))
    return energy


# ----------------------
# GENERIC SOLVE FUNCTION
# ----------------------

def solve(W, qstepsize=0.1, steps=None, qtime=None, time=None,
          method=rk4, method_kwargs=None,
          callback=None, inner_steps=None, inner_qtime=None, inner_time=None, **kwargs):
    """
    High-level solve function.

    Parameters
    ----------
    W: ndarray(shape=(N, N), dtype=complex)
        Initial vorticity matrix.
    qstepsize: float
        Time step length in qtime units.
    steps: None or int
        Total number of steps to take.
    qtime: None or float
        Total simulation time in qtime.
    time: None or float
        Total simulation time in seconds.
    method: callable(W, qstepsize, steps, **method_kwargs)
        Integration method to carry out the steps.
    method_kwargs: dict
        Extra keyword arguments to send to method at each step.
    callback: callable(W, qtime, **kwargs)
        The callback function evaluated every outer step.
        It uses **kwargs as extra keyword arguments.
        It is not evaluated at the initial time.
    inner_steps: None or int
        Number of steps taken between each callback.
    inner_qtime: None or float
        Approximate qtime between each callback.
    inner_time: None or float
        Approximate time in seconds between each callback.
    """
    N = W.shape[0]

    # Set default hamiltonian if needed
    if method_kwargs is None:
        method_kwargs = {}
    if 'hamiltonian' not in method_kwargs:
        method_kwargs['hamiltonian'] = solve_poisson
    hamiltonian = method_kwargs['hamiltonian']

    # Determine steps
    if np.array([0 if x is None else 1 for x in [steps, qtime, time]]).sum() != 1:
        raise ValueError("One, and only one, of steps, qtime, or time should be specified.")
    if time is not None:
        qtime = seconds2qtime(time, N)
    if steps is None:
        steps = round(qtime/qstepsize)
    if callback is not None and not isinstance(callback, tuple):
        callback = (callback,)

    # Determine inner_steps
    if np.array([0 if x is None else 1 for x in [inner_steps, inner_qtime, inner_time]]).sum() == 0:
        inner_steps = 100  # Default value of inner_steps
    elif inner_steps is None:
        if inner_qtime is not None:
            inner_steps = round(inner_qtime/qstepsize)
        elif inner_time is not None:
            inner_steps = round(seconds2qtime(inner_time, N)/qstepsize)

    # Check if inner_steps is too large
    if inner_steps > steps:
        inner_steps = steps

    # Initiate no steps and simulation q-time
    k = 0
    qt = 0.0

    # Main simulation loop
    while True:
        method(W, qstepsize, steps=inner_steps, hamiltonian=hamiltonian)
        k += inner_steps
        qt += inner_steps*qstepsize
        if callback is not None:
            for cfun in callback:
                cfun(W, qt, **kwargs)
        if k >= steps:
            break
        elif k + inner_steps > steps:
            method(W, qstepsize, steps=steps-k, hamiltonian=hamiltonian)
            break


