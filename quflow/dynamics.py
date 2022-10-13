import numpy as np
import scipy.linalg

from .laplacian.direct import solve_poisson, solve_heat
from .transforms import as_shr
from .utils import elm2ind, seconds2qtime, rotate
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

def euler(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None):
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
    forcing: function(P, W)
        Extra force function (to allow non-isospectral perturbations).

    Returns
    -------
    W: ndarray
    """
    if forcing is None:
        rhs = commutator
    else:
        def rhs(P, W):
            return commutator(P, W) + forcing(P, W)

    for k in range(steps):

        P = hamiltonian(W)
        VF = rhs(P, W)
        W += stepsize*VF

    return W


def heun(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None):
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
    hamiltonian: function(W)
        The Hamiltonian returning a stream matrix.
    forcing: function(P, W)
        Extra force function (to allow non-isospectral perturbations).

    Returns
    -------
    W: ndarray
    """
    if forcing is None:
        rhs = commutator
    else:
        def rhs(P, W):
            return commutator(P, W) + forcing(P, W)

    for k in range(steps):

        # Evaluate RHS at W
        P = hamiltonian(W)
        F0 = rhs(P, W)

        # Compute Heun predictor
        Wprime = W + stepsize*F0

        # Evaluate RHS at predictor WP
        P = hamiltonian(Wprime)
        F = rhs(P, Wprime)

        # Compute averaged RHS
        F += F0
        F *= stepsize/2.0

        # Update W
        W += F

    return W


def rk4(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None):
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
    hamiltonian: function(W)
        The Hamiltonian returning a stream matrix.
    forcing: function(P, W) or None (default)
        Extra force function (to allow non-isospectral perturbations).

    Returns
    -------
    W: ndarray
    """
    if forcing is None:
        rhs = commutator
    else:
        def rhs(P, W):
            return commutator(P, W) + forcing(P, W)

    for k in range(steps):
        P = hamiltonian(W)
        K1 = rhs(P, W)

        Wprime = W + (stepsize/2.0)*K1
        P = hamiltonian(Wprime)
        K2 = rhs(P, Wprime)

        Wprime = W + (stepsize/2.0)*K2
        P = hamiltonian(Wprime)
        K3 = rhs(P, Wprime)

        Wprime = W + stepsize*K3
        P = hamiltonian(Wprime)
        K4 = rhs(P, Wprime)

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


def blob(N, pos=np.array([0.0, 0.0, 1.0]), sigma=0):
    """
    Return vorticity matrix for blob located at 'pos'.

    Parameters
    ----------
    N: int
    pos: ndarray(shape=(3,), dtype=double)
    sigma: float (optional)

    Returns
    -------
    W: ndarray(shape=(N,N), dtype=complex)
    """

    # First find rotation matrix r
    a = np.zeros((3, 3))
    a[:, 0] = pos
    q, r = np.linalg.qr(a)
    if np.dot(q[:, 0], pos) < 0:
        q[:, 0] *= -1
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    q = np.roll(q, 2, axis=-1)

    # Then find axis-angle representation
    from scipy.spatial.transform import Rotation as R
    xi = R.from_matrix(q).as_rotvec()

    # Create north blob
    W = north_blob(N, sigma)

    # Rotate it
    W = rotate(xi, W)

    return W


def north_blob(N, sigma=0):
    """
    Return vorticity matrix for blob located at north pole.

    Parameters
    ----------
    N: int
    sigma: float (optional)
        Gaussian sigma for blob. If 0 (default) then give best
        approximation to point vortex

    Returns
    -------
    W: ndarray(shape=(N, N), dtype=complex)
    """

    W = np.zeros((N, N), dtype=complex)
    W[-1, -1] = 1.0j

    if sigma != 0:
        W = solve_heat(sigma/4., W)

    return W


# ----------------------
# GENERIC SOLVE FUNCTION
# ----------------------

def solve(W, qstepsize=0.1, steps=None, qtime=None, time=None,
          method=rk4, method_kwargs=None,
          callback=None, inner_steps=None, inner_qtime=None, inner_time=None,
          progress_bar=True, progress_file=None, **kwargs):
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
    progress_bar: bool
        Show progress bar (default: True)
    progress_file: TextIOWrapper or None
        File to write progress to (default: None)
    """
    N = W.shape[0]

    # Set default hamiltonian if needed
    if method_kwargs is None:
        method_kwargs = {}
    if 'hamiltonian' not in method_kwargs:
        method_kwargs['hamiltonian'] = solve_poisson

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

    # DEBUGGING:
    # print('steps: ', steps)
    # print('inner_steps: ', inner_steps)
    # print('no output steps: ', steps//inner_steps)
    # assert False, "Aborting!"

    # Initiate no steps and simulation q-time
    k = 0
    qt = 0.0

    # Create progressbar
    if progress_bar:
            try:
                if progress_file is None:
                    from tqdm.auto import tqdm
                    pbar = tqdm(total=steps, unit=' steps')
                else:
                    from tqdm import tqdm
                    pbar = tqdm(total=steps, unit=' steps', file=progress_file, ascii=True, mininterval=10.0)
            except ModuleNotFoundError:
                progress_bar = False

    # Main simulation loop
    for k in range(0, steps, inner_steps):
        if k+inner_steps > steps:
            no_steps = steps-k
        else:
            no_steps = inner_steps
        method(W, qstepsize, steps=no_steps, **method_kwargs)
        qt += no_steps*qstepsize
        if progress_bar:
            pbar.update(no_steps)
        if callback is not None:
            for cfun in callback:
                cfun(W, qt, **kwargs)

    # Close progressbar
    if progress_bar:
        pbar.close()
