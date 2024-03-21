import numpy as np
import scipy.linalg
import inspect
import numba as nb

from .laplacian import solve_poisson, solve_heat
from .laplacian import select_skewherm as select_skewherm_laplacian
from .geometry import norm_Linf

# ----------------
# GLOBAL VARIABLES
# ----------------

_SKEW_HERM_ = True  # Is the dynamics skew-Hermitian?
_SKEW_HERM_PROJ_FREQ_ = -1  # How many steps before skew-Hermitian projection, negative = never


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


@nb.njit(error_model='numpy', fastmath=True)
def conj_subtract_(a, out):
    N = a.shape[-1]
    if a.ndim == 2:
        for i in range(N):
            out[i, i] = a[i, i] - np.conj(a[i, i])
            for j in range(i):
                out[i, j] = a[i, j] - np.conj(a[j, i])
                out[j, i] = -np.conj(out[i, j])
    elif a.ndim == 3:
        for k in range(a.shape[0]):
            for i in range(N):
                out[k, i, i] = a[k, i, i] - np.conj(a[k, i, i])
                for j in range(i):
                    out[k, i, j] = a[k, i, j] - np.conj(a[k, j, i])
                    out[k, j, i] = -np.conj(out[k, i, j])


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


def estimate_stepsize(W, P=None, safety_factor=0.1):
    """
    If `lambda_max` is the maximum eigenvalue of `P`, which is
    assumed to vary slowly as compared with `W` (i.e., we assume
    that `P` is constant), then an estimate to the stepsize
    is safety_factor*pi/lambda_max.
    This stepsize is dimension-free, so the same estimate can be
    used for any matrix size `N`.
    The corresponding time-step in seconds is delta_time = stepsize*hbar.

    Parameters
    ----------
    W: ndarray
        Vorticity state.
    P: ndarray
        Stream matrix (assumed to vary slow compared to W).
    safety_factor:
        Multiply by this safety factor.

    Returns
    -------
    stepsize (float)
    """
    if P is None:
        P = solve_poisson(W)
    lambda_max = norm_Linf(P)
    stepsize = safety_factor*np.pi/lambda_max
    return stepsize


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
                      tol="auto", maxit=10, verbatim=False):
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

    # Specify tolerance if needed
    if tol == "auto" or tol < 0:
        tol = np.finfo(W.dtype).eps*stepsize*np.linalg.norm(W, np.inf)

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

        if _SKEW_HERM_:
            # Compute LU of A
            luA, piv = scipy.linalg.lu_factor(A)

            # Solve first equation for X
            X = scipy.linalg.lu_solve((luA, piv), W)

            # Solve second equation for Wtilde
            Wtilde = scipy.linalg.lu_solve((luA, piv), -X.conj().T)

            # Update W
            W_new = A.conj().T @ Wtilde @ A

        else:

            # Solve first equation for X
            X = np.linalg.solve(A, W)

            # Solve second equation for Wtilde
            Aalt = Id + (stepsize/2.0)*Ptilde
            Wtilde = np.linalg.solve(Aalt.conj().T, X.conj().T).conj().T
            # The line above could probably be done faster without conj().T everywhere

            # Update W
            W_new = Aalt @ Wtilde @ A

        np.copyto(W, W_new)

        # Make sure solution is Hermitian (this removes drift in rounding errors)
        if _SKEW_HERM_ and k % _SKEW_HERM_PROJ_FREQ_ == _SKEW_HERM_PROJ_FREQ_ - 1:
            project_skewherm(W)

        # --- End of step ---

    return W


def isomp_fixedpoint(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None,
                     tol=1e-8, maxit=5, verbatim=False):
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
        PWcomm *= 2
        W += PWcomm

        # Make sure solution is Hermitian (this removes drift in rounding errors)
        if _SKEW_HERM_ and k % _SKEW_HERM_PROJ_FREQ_ == _SKEW_HERM_PROJ_FREQ_ - 1:
            project_skewherm(W)

        # --- End of step ---

    if verbatim:
        print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))

    return W


def isomp_fixedpoint2(W, stepsize=0.1, steps=100, hamiltonian=solve_poisson, forcing=None,
                      tol='auto', maxit=10, minit=1, stats=None,
                      verbatim=False, compsum=False, reinitialize=False):
    """
    Time-stepping by isospectral midpoint second order method for skew-Hermitian W
    using fixed-point iterations. This implementation uses compensated summation
    and other tricks to achieve Brouwer's law for reduced accumulative rounding errors.

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
    forcing: function(P, W) or None (default)
        Extra force function (to allow non-isospectral perturbations).
    tol: float or 'auto'
        Tolerance for iterations. Negative value or "auto" means automatic choice.
    maxit: int
        Maximum number of iterations.
    integrals: dict or None
        Dictionary to be filled in with first integrals.
    stats: dict or None
        Dictionary to be filled in with integration statistics.
    verbatim: bool
        Print extra information if True. Default is False.
    compsum: bool
        Use compensated summation.
    reinitialize: bool
        Whether to re-initiate the iteration vector at every step.

    Returns
    -------
    W: ndarray
    """

    # Check input
    assert minit >= 1, "minit must be at least 1."
    assert maxit >= minit, "maxit must be at minit."

    # Check if Hamiltonian accepts 'out' argument
    # hamiltonian_accepts_out = False
    # if 'out' in inspect.getfullargspec(hamiltonian).args:
    #     hamiltonian_accepts_out = True
    #     Phalf = np.zeros_like(W)

    # Check if force function accepts 'out' argument
    force_accepts_out = False
    if forcing is not None and 'out' in inspect.getfullargspec(forcing).args:
        force_accepts_out = True
        FW = np.zeros_like(W)

    # Stats variables
    total_iterations = 0
    number_of_maxit = 0

    # Initialize
    dW = np.zeros_like(W)
    dW_old = np.zeros_like(W)
    Whalf = np.zeros_like(W)
    PWcomm = np.zeros_like(W)
    # PWPhalf = np.zeros_like(W)
    hhalf = stepsize/2.0

    # Specify tolerance if needed
    if tol == "auto" or tol < 0:
        mach_eps = np.finfo(W.dtype).eps
        if not compsum:
            mach_eps = np.sqrt(mach_eps)
        if W.ndim > 2:
            zeroind = (0,)*(W.ndim-2) + (Ellipsis,)
            tol = mach_eps*stepsize*np.linalg.norm(W[zeroind], np.inf)
        else:
            tol = mach_eps*stepsize*np.linalg.norm(W, np.inf)
        if verbatim:
            print("Tolerance set to {}.".format(tol))
        if stats:
            stats['tol'] = tol

    # Variables for compensated summation
    if compsum:
        y_compsum = np.zeros_like(W)
        c_compsum = np.zeros_like(W)
        t_compsum = np.zeros_like(W)
        delta_compsum = np.zeros_like(W)

    # --- Beginning of step loop ---
    for k in range(steps):

        # Per step updates
        resnorm = np.inf
        if reinitialize:
            dW.fill(0.0)

        # --- Beginning of iterations ---
        for i in range(maxit):

            # Update iterations
            total_iterations += 1

            # Compute Wtilde
            np.copyto(Whalf, W)
            Whalf += dW

            # Save dW from previous step
            np.copyto(dW_old, dW)

            # Update Ptilde
            Phalf = hamiltonian(Whalf)
            Phalf *= hhalf

            # Compute middle variables
            # PWcomm = Phalf @ Whalf  # old
            np.matmul(Phalf, Whalf, out=PWcomm)
            # PWPhalf = PWcomm @ Phalf  # old
            # np.matmul(PWcomm, Phalf, out=PWPhalf)
            np.matmul(PWcomm, Phalf, out=dW)  # More efficient, as PWPhalf is not needed
            if _SKEW_HERM_:
                # np.conjugate(PWcomm)
                # PWcomm -= PWcomm.conj().T
                conj_subtract_(PWcomm, PWcomm)
            else:
                PWcomm -= Whalf @ Phalf

            # Update dW
            # np.copyto(dW, PWPhalf)
            dW += PWcomm

            # Add forcing if needed
            if forcing:
                # Compute forcing if needed
                if force_accepts_out:
                    forcing(Phalf/hhalf, Whalf, out=FW)
                else:
                    FW = forcing(Phalf/hhalf, Whalf)
                FW *= hhalf
                dW += FW

            # Check if time to break
            if i+1 >= minit:
                # Compute error
                resnorm_old = resnorm
                dW_old -= dW
                if dW_old.ndim > 2:
                    resnormvec = scipy.linalg.norm(dW_old, ord=np.inf, axis=(-1, -2))
                    if Phalf.ndim == 2:
                        resnorm = resnormvec[0]
                    else:
                        resnorm = resnormvec.max()
                else:
                    resnorm = scipy.linalg.norm(dW_old, ord=np.inf)
                if resnorm <= tol or resnorm >= resnorm_old:
                    break

        else:
            # We used maxit iterations
            number_of_maxit += 1
            if verbatim:
                print("Max iterations {} reached at step {}.".format(maxit, k))
            # if stats:
            #     stats['maxit_reached'] = True

        # Update W
        PWcomm *= 2

        if compsum:
            # Compensated summation for W += PWcomm

            # FROM WIKIPEDIA https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            #
            # var sum = 0.0                    // Prepare the accumulator.
            # var c = 0.0                      // A running compensation for lost low-order bits.
            #
            # for i = 1 to input.length do     // The array input has elements indexed input[1] to input[input.length].
            #     var y = input[i] - c         // 1. c is zero the first time around.
            #     var t = sum + y              // 2. Alas, sum is big, y small, so low-order digits of y are lost.
            #     c = (t - sum) - y            // 3. (t - sum) cancels the high-order part of y; subtracting y recovers negative (low part of y)
            #     sum = t                      // 4. Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
            # next i                           // Next time around, the lost low part will be added to y in a fresh attempt.
            
            # 1.
            # y_compsum = PWcomm - c_compsum  # old
            np.copyto(y_compsum, PWcomm)
            y_compsum -= c_compsum

            # 2.
            # t_compsum = W + y_compsum  # old
            np.copyto(t_compsum, W)
            t_compsum += y_compsum

            # 3.
            # c_compsum = (t_compsum - W) - y_compsum  # old
            np.copyto(delta_compsum, t_compsum)
            delta_compsum -= W
            np.copyto(c_compsum, delta_compsum)
            c_compsum -= y_compsum

            # 4.
            np.copyto(W, t_compsum)
        else:
            W += PWcomm

        # --- End of step ---

    if verbatim:
        print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))
    if stats:
        stats["iterations"] = total_iterations/steps
        stats["maxit"] = number_of_maxit/steps
    # if integrals:
    #     P = hamiltonian(W)
    #     integrals["energy"] = (P*W).sum()
    #     integrals["enstrophy"] = -(W**2).sum()

    return W


# Default isospectral method
isomp = isomp_fixedpoint2
