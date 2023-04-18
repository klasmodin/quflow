import numpy as np
import scipy.linalg

from .laplacian import solve_poisson, solve_heat
from .laplacian import select_skewherm as select_skewherm_laplacian

# ----------------
# GLOBAL VARIABLES
# ----------------

_SKEW_HERM_ = True  # Is the dynamics skew-Hermitian?
_SKEW_HERM_PROJ_FREQ_ = 100  # How many steps before skew-Hermitian projection, negative = never


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------

def commutator_generic(W, P):
    """
    Commutator for arbitrary matrices.

    Parameters
    ----------
    W: ndarray
    P: ndarray

    Returns
    -------
    ndarray
    """
    return W@P - P@W


def commutator_skewherm(W, P):
    """
    Efficient computations of commutator for skew-Hermitian matrices.

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


# Select default commutator
commutator = commutator_skewherm


# Project to skewherm (to avoid drift)
def project_skewherm(W):
    W /= 2.0
    W -= W.conj().T


# Function to update solver statistics
def update_stats(stats: dict, **kwargs):
    for arg, val in kwargs.items():
        if arg in stats and np.isscalar(val):
            stats[arg] += val
        else:
            stats[arg] = val


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def select_skewherm(flag):
    """
    Select whether integrators should work with
    skew Hermitian matrices.

    Parameters
    ----------
    flag: bool

    Returns
    -------
    None
    """
    global _SKEW_HERM_
    global commutator
    if flag:
        commutator = commutator_skewherm
        _SKEW_HERM_ = True
    else:
        commutator = commutator_generic
        _SKEW_HERM_ = False
    select_skewherm_laplacian(flag)


# -------------------------------------------------
# CLASSICAL (EXPLICIT, NON-ISOSPECTRAL) INTEGRATORS
# -------------------------------------------------

def euler(W: np.ndarray,
          stepsize: float = 0.1,
          steps: int = 100,
          hamiltonian=solve_poisson,
          forcing=None,
          stats: dict = None,
          **kwargs) -> np.ndarray:
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
    hamiltonian: function(W)
        The Hamiltonian returning a stream matrix.
    forcing: None or function(P, W)
        Extra force function (to allow non-isospectral perturbations).
    stats: None or dict
        Dictionary with statistics
    **kwargs: dict
        Extra keyword arguments

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

    if stats is not None:
        update_stats(stats, steps=steps)

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


# Default classical integrator
classical = rk4


# -------------------
# ISOSPECTRAL METHODS
# -------------------

def isomp_quasinewton(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None,
                      tol=1e-8, maxit=10, verbatim=False):
    """
    Time-stepping by isospectral midpoint second order method using
    a quasi-Newton iteration scheme. This scheme preserves the eigen-spectrum
    of `W` up to machine epsilon.

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
    forcing: function(P, W) or None (default)
        Extra force function (to allow non-isospectral perturbations).
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
    if forcing is not None:
        assert NotImplementedError("Forcing for isomp_quasinewton is not implemented yet.")

    if not _SKEW_HERM_:
        assert NotImplementedError("isomp_quasinewton might not work for non-skewherm.")

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

        # Make sure solution is Hermitian (this removes drift in rounding errors)
        if _SKEW_HERM_ and k % _SKEW_HERM_PROJ_FREQ_ == _SKEW_HERM_PROJ_FREQ_ - 1:
            project_skewherm(W)

        # --- End of step ---

    if verbatim:
        print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))

    return W


def isomp_simple(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None,
                 enforce_hermitian=True):
    """
    Time-stepping by the simplified isospectral midpoint method.
    This is an explicit isospectral method but not symplectic. Nor is it reversible.

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
    forcing: function(P, W) or None (default)
        Extra force function (to allow non-isospectral perturbations).
    enforce_hermitian: bool
        Enforce at every step that the vorticity matrix is hermitian.

    Returns
    -------
    W: ndarray
    """
    Id = np.eye(W.shape[0])

    Wtilde = W.copy()

    if forcing is not None:
        assert NotImplementedError("Forcing for isomp_simple is not implemented yet.")

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

        # Make sure solution is Hermitian (this removes drift in rounding errors)
        if _SKEW_HERM_ and k % _SKEW_HERM_PROJ_FREQ_ == _SKEW_HERM_PROJ_FREQ_ - 1:
            project_skewherm(W)

        # --- End of step ---

    return W


def isomp_fixedpoint(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None,
                     tol=1e-8, maxit=5, verbatim=True):
    """
    Time-stepping by isospectral midpoint second order method for skew-Hermitian W
    using fixed-point iterations.

    Parameters
    ----------
    W: ndarray
        Initial skew-Hermitian vorticity matrix (overwritten and returned).
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
    skewherm: bool (default: True)
        Flag if the flow is skew-Hermitian.
    skewherm_proj_freq: int
        Project onto skew-Hermitian every skewherm_proj_freq step.
    forcing: function(P, W) or None (default)
        Extra force function (to allow non-isospectral perturbations).

    Returns
    -------
    W: ndarray
    """

    assert maxit >= 1, "maxit must be at least 1."

    total_iterations = 0

    # Initialize
    dW = np.zeros_like(W)
    dW_old = np.zeros_like(W)
    Whalf = np.zeros_like(W)

    for k in range(steps):

        # --- Beginning of step ---

        for i in range(maxit):

            # Update iterations
            total_iterations += 1

            # Compute Wtilde
            np.copyto(Whalf, W)
            Whalf += dW

            # Update Ptilde
            Phalf = hamiltonian(Whalf)
            Phalf *= stepsize/2.0

            # Compute middle variables
            PWcomm = Phalf @ Whalf
            PWPhalf = PWcomm @ Phalf
            if _SKEW_HERM_:
                PWcomm -= PWcomm.conj().T
            else:
                PWcomm -= Whalf @ Phalf

            # Update dW
            np.copyto(dW_old, dW)
            np.copyto(dW, PWcomm)
            dW += PWPhalf

            # Add forcing if needed
            if forcing is not None:
                # Compute forcing if needed
                FW = forcing(Phalf/(stepsize/2.0), Whalf)
                FW *= stepsize/2.0
                dW += FW

            # Compute error
            resnorm = scipy.linalg.norm(dW - dW_old, np.inf)

            # Check error
            if resnorm < tol:
                break

        else:
            # We used maxit iterations
            if verbatim:
                print("Max iterations {} reached at step {}.".format(maxit, k))

        # Update W
        W += 2.0*PWcomm

        # Make sure solution is Hermitian (this removes drift in rounding errors)
        if _SKEW_HERM_ and k % _SKEW_HERM_PROJ_FREQ_ == _SKEW_HERM_PROJ_FREQ_ - 1:
            project_skewherm(W)

        # --- End of step ---

    if verbatim:
        print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))

    return W


# Default isospectral method
isomp = isomp_fixedpoint
