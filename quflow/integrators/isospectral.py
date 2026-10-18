r"""
Isospectral time integrators for the quantized Euler equations.

The quantized (matrix) Euler equations on the sphere read

.. math::

    \dot W = \frac{1}{\hbar}[P, W], \qquad P = H(W),

where :math:`W` is the :math:`N\times N` skew-Hermitian vorticity matrix,
:math:`H` is the Hamiltonian map returning the stream matrix (by default
:func:`quflow.laplacian.solve_poisson`, i.e. :math:`\Delta_N P = W`), and
:math:`\hbar = 2/\sqrt{N^2-1}` (see :func:`quflow.geometry.hbar`).

The flow is isospectral: the eigenvalues of :math:`W` (the Casimirs) are
conserved. The integrators in this module are based on the *isospectral
midpoint method*, which conserves the spectrum exactly up to the tolerance of
its nonlinear solver and rounding errors. With the dimensionless step
:math:`h = \Delta t/\hbar` and :math:`\tilde P = H(\tilde W)`, one step
:math:`W_n \mapsto W_{n+1}` is

.. math::

    W_n = \big(I - \tfrac{h}{2}\tilde P\big)\,\tilde W\,\big(I + \tfrac{h}{2}\tilde P\big),
    \qquad
    W_{n+1} = \big(I + \tfrac{h}{2}\tilde P\big)\,\tilde W\,\big(I - \tfrac{h}{2}\tilde P\big).

Eliminating :math:`\tilde W` gives :math:`W_{n+1} = C W_n C^{-1}` with the Cayley
transform :math:`C = (I + \tfrac{h}{2}\tilde P)(I - \tfrac{h}{2}\tilde P)^{-1}`,
which is unitary when :math:`\tilde P` is skew-Hermitian. Hence the spectrum and
the skew-Hermitian structure of :math:`W` are preserved.

Integrators
-----------
isomp_fixedpoint (alias ``isomp``, the default)
    Solves the midpoint equations by fixed-point iteration. Supports
    time-dependent Hamiltonians, forcing, Strang splitting, compensated
    summation and stacks of matrices.
isomp_quasinewton
    Solves the midpoint equations by a quasi-Newton iteration on the Cayley
    form above, with an LU factorization per iteration.
isomp_simple
    Explicit first order variant: one Hamiltonian evaluation per step, using
    the previous step's :math:`\tilde W`.

All integrators take the time step ``dt`` in physical time units and work
internally with :math:`h = \Delta t/\hbar`. They overwrite their input ``W``
and return it.

The isospectral midpoint method is introduced in [1]_ and applied to the
quantized Euler equations in [2]_. The compensated summation used to reduce
the accumulation of rounding errors (Brouwer's law) follows [3]_.

References
----------
.. [1] K. Modin and M. Viviani. *Lie-Poisson methods for isospectral flows*,
   Found. Comput. Math., 20, 2020.
.. [2] K. Modin and M. Viviani. *A Casimir preserving scheme for long-time
   simulation of spherical ideal hydrodynamics*, J. Fluid Mech., 884:A22, 2020.
.. [3] E. Hairer, R. I. McLachlan and A. Razakarivony. *Achieving Brouwer's law
   with implicit Runge-Kutta methods*, BIT Numer. Math., 48, 2008.
"""
import numpy as np
import scipy.linalg
import inspect
import numba as nb

from ..laplacian import solve_poisson, solve_heat
from ..laplacian import select_skewherm as select_skewherm_laplacian
from ..geometry import norm_Linf, hbar

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
    Matrix commutator ``[W, P] = W@P - P@W`` for arbitrary square matrices.

    Note that, unlike :func:`quflow.geometry.bracket`, there is no
    scaling by ``1/hbar``.

    Parameters
    ----------
    W : ndarray, shape (N, N)
    P : ndarray, shape (N, N)

    Returns
    -------
    ndarray, shape (N, N)
        The commutator ``W@P - P@W`` (a new array).

    See Also
    --------
    commutator_skewherm : Faster version for skew-Hermitian input.
    """
    return W@P - P@W


def commutator_skewherm(W, P):
    """
    Matrix commutator ``[W, P] = W@P - P@W`` for skew-Hermitian matrices.

    Uses that ``P@W = (W@P)^H`` when both ``W`` and ``P`` are skew-Hermitian
    (``^H`` denotes the conjugate transpose), so only one matrix product
    is computed. The result is wrong if the input is not skew-Hermitian.

    Parameters
    ----------
    W : ndarray, shape (N, N), complex
        Skew-Hermitian matrix.
    P : ndarray, shape (N, N), complex
        Skew-Hermitian matrix.

    Returns
    -------
    ndarray, shape (N, N), complex
        The commutator ``W@P - P@W`` (a new, skew-Hermitian array).

    See Also
    --------
    commutator_generic : Version for arbitrary matrices.
    """
    VF = W@P
    VF -= VF.conj().T
    return VF


def commutator(W, P):
    """
    Matrix commutator ``[W, P] = W@P - P@W``, following :func:`select_skewherm`.

    Dispatches to :func:`commutator_skewherm` in skew-Hermitian mode (the
    default) and to :func:`commutator_generic` otherwise. Since the choice is
    made at each call, modules that import ``commutator`` by name also follow
    the current setting.

    Parameters
    ----------
    W : ndarray, shape (N, N)
    P : ndarray, shape (N, N)

    Returns
    -------
    ndarray, shape (N, N)
        The commutator ``W@P - P@W`` (a new array).
    """
    if _SKEW_HERM_:
        return commutator_skewherm(W, P)
    return commutator_generic(W, P)


# Project to skewherm (to avoid drift)
def project_skewherm(W):
    """
    Project `W` in place onto its skew-Hermitian part, ``(W - W^H)/2``.

    Used to remove drift away from skew-Hermiticity caused by rounding errors.

    Parameters
    ----------
    W : ndarray, shape (N, N), complex
        Matrix to project (modified in place).

    Returns
    -------
    None
    """
    W /= 2.0
    W -= W.conj().T


@nb.njit(error_model='numpy', fastmath=True)
def conj_subtract_(a, out):
    """
    Compute ``out = a - a^H`` (numba kernel).

    The result is skew-Hermitian. ``^H`` is the conjugate transpose of each
    matrix; for a stack it is applied matrix by matrix. Only the lower
    triangle of ``a`` and its mirror are combined, so ``out`` may be the
    same array as ``a`` (in-place operation).

    Parameters
    ----------
    a : ndarray, shape (N, N) or (k, N, N), complex
        Input matrix or stack of matrices.
    out : ndarray, same shape and dtype as `a`
        Output array, overwritten.
    """
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
    """
    Update a solver statistics dictionary in place.

    For each keyword argument, a scalar value is added to an existing entry
    of the same name; otherwise (new key or non-scalar value) the entry is set.

    Parameters
    ----------
    stats : dict
        Statistics dictionary (modified in place).
    **kwargs
        Statistics to accumulate, e.g. ``steps=100``.

    Returns
    -------
    None
    """
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
    Select whether the integrators assume skew-Hermitian matrices.

    Skew-Hermitian mode (the default) allows faster computations, but gives
    wrong results for matrices that are not skew-Hermitian. The setting is
    global: it applies to the integrators and :func:`commutator` of this
    module, and calls :func:`quflow.laplacian.select_skewherm` with the same flag.

    Parameters
    ----------
    flag : bool
        ``True`` for skew-Hermitian mode, ``False`` for general complex matrices.

    Returns
    -------
    None
    """
    global _SKEW_HERM_
    _SKEW_HERM_ = bool(flag)
    select_skewherm_laplacian(flag)


def estimate_stepsize(W, P=None, safety_factor=0.1):
    """
    Estimate a dimensionless step size for the isospectral integrators.

    If `lambda_max` is the maximum eigenvalue of `P`, which is
    assumed to vary slowly as compared with `W` (i.e., we assume
    that `P` is constant), then an estimate to the stepsize
    is ``safety_factor*pi/lambda_max``.
    This stepsize is dimension-free, so the same estimate can be
    used for any matrix size `N`.
    The corresponding time step in physical units, as passed to the
    integrators via `dt`, is ``stepsize*hbar(N)``.

    Parameters
    ----------
    W : ndarray, shape (N, N), complex
        Vorticity matrix. Only used if `P` is not given.
    P : ndarray, shape (N, N), complex, optional
        Stream matrix (assumed to vary slowly compared to `W`). If omitted,
        it is computed as ``solve_poisson(W)``.
    safety_factor : float, optional
        Factor multiplying the estimate. Default is 0.1.

    Returns
    -------
    stepsize : float
        Dimensionless step size ``safety_factor*pi/lambda_max``, where
        ``lambda_max = norm_Linf(P)`` is the spectral norm of `P` (its largest
        eigenvalue in absolute value, since `P` is normal).

    Examples
    --------
    >>> import numpy as np
    >>> import quflow as qf
    >>> N = 16
    >>> omega = np.random.default_rng(0).standard_normal(N**2)
    >>> omega[0] = 0.0
    >>> W = qf.shr2mat(omega, N=N)
    >>> dt = qf.integrators.isospectral.estimate_stepsize(W)*qf.hbar(N)
    """
    if P is None:
        P = solve_poisson(W)
    lambda_max = norm_Linf(P)
    stepsize = safety_factor*np.pi/lambda_max
    return stepsize


# -------------------
# ISOSPECTRAL METHODS
# -------------------

def isomp_quasinewton(W, dt, steps=100, hamiltonian=solve_poisson, forcing=None,
                      tol="auto", maxit=10, verbatim=False, **kwargs):
    r"""
    Time-stepping by isospectral midpoint second order method using
    a quasi-Newton iteration scheme. This scheme preserves the eigen-spectrum
    of `W` up to machine epsilon.

    Each iteration evaluates :math:`\tilde P = H(\tilde W)` at the current
    iterate, forms :math:`A = I - \tfrac{h}{2}\tilde P` with :math:`h = \Delta t/\hbar`,
    and updates

    .. math::

        \tilde W \leftarrow A^{-1} W_n A^{-H},

    using one LU factorization of :math:`A`. When the iteration has converged,
    :math:`W_{n+1} = A^{H} \tilde W A`. For skew-Hermitian :math:`\tilde P`,
    :math:`A^{H} = I + \tfrac{h}{2}\tilde P`, so this is the isospectral midpoint
    method described in the module docstring.

    Compared with plain fixed-point iteration on :math:`\tilde W`, solving
    the Cayley form exactly for a given :math:`\tilde P` takes part of the
    Jacobian of the midpoint equations into account, which makes the
    iteration a quasi-Newton method.

    Parameters
    ----------
    W : ndarray, shape (N, N), complex
        Initial skew-Hermitian vorticity matrix (overwritten and returned).
        Stacks of matrices are not supported.
    dt : float
        Time step length in physical time units.
    steps : int, optional
        Number of steps to take. Default is 100.
    hamiltonian : callable, optional
        Function ``hamiltonian(W)`` returning the stream matrix.
        Default is :func:`quflow.laplacian.solve_poisson`.
    forcing : None
        Not supported; see Notes.
    tol : float or 'auto', optional
        Tolerance for the iterations, measured as the matrix infinity norm
        of the change of :math:`\tilde W` between iterations. ``'auto'``
        (default) or a negative value selects ``eps*h*norm(W, inf)``, with
        ``eps`` the machine epsilon of the dtype of `W`.
    maxit : int, optional
        Maximum number of iterations per step. Default is 10.
    verbatim : bool, optional
        Print extra information if True. Default is False.
    **kwargs
        Accepted and ignored, for interface compatibility with
        :func:`isomp_fixedpoint`.

    Returns
    -------
    W : ndarray, shape (N, N), complex
        The state after `steps` steps (the same array as the input `W`).

    Notes
    -----
    Requires skew-Hermitian `W` and stream matrices: raises
    `NotImplementedError` if skew-Hermitian mode is switched off
    (see :func:`select_skewherm`), and if `forcing` is given.

    See Also
    --------
    isomp_fixedpoint : Default isospectral integrator with more features.
    """
    if forcing is not None:
        raise NotImplementedError("Forcing for isomp_quasinewton is not implemented yet.")

    if not _SKEW_HERM_:
        raise NotImplementedError("isomp_quasinewton requires skew-Hermitian mode (see select_skewherm).")

    stepsize = dt / hbar(N=W.shape[-1])

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


def isomp_simple(W, dt, steps=100, hamiltonian=solve_poisson, forcing=None, **kwargs):
    r"""
    Time-stepping by the simplified isospectral midpoint method.
    This is an explicit isospectral method but not symplectic. Nor is it reversible.
    It is of first order.

    Instead of solving the midpoint equations, each step evaluates the
    Hamiltonian once, at the :math:`\tilde W` of the previous step (at the first
    step, at the initial `W`), and then performs the Cayley update of the
    isospectral midpoint method with that stream matrix:

    .. math::

        \tilde W = A^{-1} W_n B^{-1}, \qquad W_{n+1} = B \tilde W A,
        \qquad A = I - \tfrac{h}{2}\tilde P,\quad B = I + \tfrac{h}{2}\tilde P,

    with :math:`h = \Delta t/\hbar`. Since :math:`W_{n+1}` is conjugate to
    :math:`W_n`, the spectrum is preserved.

    Parameters
    ----------
    W : ndarray, shape (N, N), complex
        Initial vorticity matrix (overwritten and returned). Stacks of
        matrices are not supported.
    dt : float
        Time step length in physical time units.
    steps : int, optional
        Number of steps to take. Default is 100.
    hamiltonian : callable, optional
        Function ``hamiltonian(W)`` returning the stream matrix.
        Default is :func:`quflow.laplacian.solve_poisson`.
    forcing : None
        Not supported; see Notes.
    **kwargs
        Accepted and ignored, for interface compatibility with
        :func:`isomp_fixedpoint`.

    Returns
    -------
    W : ndarray, shape (N, N), complex
        The state after `steps` steps (the same array as the input `W`).

    Notes
    -----
    In skew-Hermitian mode (default, see :func:`select_skewherm`), one LU
    factorization of :math:`A` is used for both linear solves; otherwise two
    general solves are used.

    `forcing` is not implemented: passing a forcing function raises
    `NotImplementedError`.

    See Also
    --------
    isomp_fixedpoint : Default isospectral integrator with more features.
    """
    Id = np.eye(W.shape[0])

    Wtilde = W.copy()

    stepsize = dt/hbar(W.shape[-1])

    if forcing is not None:
        raise NotImplementedError("Forcing for isomp_simple is not implemented yet.")

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


def isomp_fixedpoint(W,
                     dt, 
                     steps=100, 
                     hamiltonian=solve_poisson, 
                     time=None, 
                     forcing=None, 
                     strang_splitting=None,
                     stats=None,
                     callback=None,
                     tol='auto', 
                     maxit=10, 
                     minit=1, 
                     verbatim=False, 
                     compsum=False, 
                     reinitialize=False
                     ):
    r"""
    Time-stepping by isospectral midpoint second order method for skew-Hermitian W
    using fixed-point iterations. This implementation can use compensated summation
    (see `compsum`) and other tricks to achieve Brouwer's law for reduced
    accumulative rounding errors.

    This is the default integrator, also available as ``quflow.isomp``. It
    approximates

    .. math::

        \dot W = \frac{1}{\hbar}[P, W] + F(P, W), \qquad P = H(W),

    where :math:`H` is `hamiltonian` and :math:`F` is the optional `forcing`.
    With :math:`h = \Delta t/\hbar`, each step solves for the midpoint
    :math:`\tilde W` by fixed-point iteration of

    .. math::

        \tilde W = W_n + \tfrac{h}{2}[\tilde P, \tilde W]
                   + \tfrac{h^2}{4}\tilde P \tilde W \tilde P
                   + \tfrac{\Delta t}{2} F(\tilde P, \tilde W),
        \qquad \tilde P = H(\tilde W),

    and then sets

    .. math::

        W_{n+1} = W_n + h[\tilde P, \tilde W] + \Delta t\, F(\tilde P, \tilde W).

    Without forcing, this is the isospectral midpoint method described in the
    module docstring, and the spectrum of `W` is preserved up to the iteration
    tolerance and rounding errors. The iteration variable is the increment
    :math:`\tilde W - W_n`. By default it starts from the value of the
    previous step (see `reinitialize`).

    Parameters
    ----------
    W : ndarray, shape (N, N) or (k, N, N), complex
        Initial skew-Hermitian vorticity matrix (overwritten and returned).
        A stack of `k` matrices is also accepted. `hamiltonian` then receives
        the whole stack. If it returns a single (N, N) stream matrix, all
        matrices in the stack are transported by it (with the default
        `hamiltonian`, the stream matrix of ``W[0]``).
    dt : float
        Time step length in physical time units. The dimensionless step is
        ``dt/hbar(N)``.
    steps : int, optional
        Number of steps to take. Default is 100.
    hamiltonian : callable, optional
        Function ``hamiltonian(W)`` or ``hamiltonian(W, time)`` returning the
        stream matrix. Default is :func:`quflow.laplacian.solve_poisson`.
        The returned array is not modified.
    time : float or None, optional
        Time at the initial state. If `None` (default) the system is assumed to
        be autonomous, and the time parameter is not passed to the hamiltonian and forcing.
        Otherwise, `hamiltonian` and `forcing` are called with the keyword
        argument ``time`` (the midpoint time of each step) if they accept it.
        This is detected by calling them once with ``time=time`` before the
        first step: a `TypeError` means they are treated as autonomous.
    forcing : callable or None, optional
        Extra force function ``forcing(P, W)`` or ``forcing(P, W, time)``
        returning a matrix of the same shape as `W` (to allow non-isospectral
        perturbations). It is evaluated at the midpoint, with the unscaled
        stream matrix. The returned array is not modified, so the function
        may return a stored array (for example, a force cached per time step).
        Not supported together with `compsum` (raises `NotImplementedError`).
    strang_splitting : callable or None, optional
        Function ``strang_splitting(t, W)`` returning the state after
        advancing `W` by time `t` with another flow. It is applied with
        ``t = dt/2`` before and after each isospectral step (Strang splitting).
    stats : dict or None, optional
        Dictionary to be filled in with integration statistics: ``'iterations'``
        (average number of iterations per step), ``'number_of_maxit'`` (fraction
        of steps that reached `maxit`) and, if `tol` is automatic, ``'tol_auto'``.
    callback : callable or None, optional
        Function ``callback(W, dW)`` evaluated at the end of every step,
        just before W is updated W += dW. Here ``dW`` is the isospectral part
        :math:`h[\tilde P, \tilde W]` of the increment (without forcing).
    tol : float or 'auto', optional
        Tolerance for iterations. Negative value or "auto" (default) means
        automatic choice: ``eps*h*norm(W, inf)`` with compensated summation and
        ``sqrt(eps)*h*norm(W, inf)`` without, where ``eps`` is the machine
        epsilon of the dtype of `W` (for a stack, the norm of ``W[0]`` is used).
    maxit : int, optional
        Maximum number of iterations per step. Default is 10.
    minit : int, optional
        Minimum number of iterations per step. Default is 1.
    verbatim : bool, optional
        Print extra information if True. Default is False.
    compsum : bool, optional
        Use compensated (Kahan) summation for the update ``W += dW``.
        Default is False.
    reinitialize : bool, optional
        Whether to restart the iteration from :math:`\tilde W = W_n` at every
        step, instead of from the previous step's increment. Default is False.

    Returns
    -------
    W : ndarray
        The state after `steps` steps. This is the input array `W`, unless
        `strang_splitting` returns new arrays; always use the return value.

    Notes
    -----
    The iterations of a step stop after at least `minit` iterations, as soon
    as the change of the iteration variable (in the matrix infinity norm) is
    at most `tol`, or stops decreasing. For a stack, the change of the first
    matrix is used if the stream matrix is a single matrix, and otherwise the
    largest change in the stack.

    Since the automatic tolerance without `compsum` is based on ``sqrt(eps)``,
    the spectrum is then only conserved to roughly that relative accuracy.
    For conservation close to machine precision, use ``compsum=True`` or a
    smaller `tol` (at the cost of more iterations per step).

    Examples
    --------
    >>> import numpy as np
    >>> import quflow as qf
    >>> N = 16
    >>> omega = np.random.default_rng(0).standard_normal(N**2)
    >>> omega[0] = 0.0
    >>> W = qf.shr2mat(omega, N=N)
    >>> spectrum0 = np.linalg.eigvalsh(1j*W)
    >>> W = qf.isomp(W, dt=0.1*qf.hbar(N), steps=10, compsum=True)
    >>> bool(np.allclose(np.linalg.eigvalsh(1j*W), spectrum0, rtol=0, atol=1e-10))
    True

    See Also
    --------
    isomp_quasinewton, isomp_simple : Alternative isospectral integrators.
    """

    # Check input
    assert minit >= 1, "minit must be at least 1."
    assert maxit >= minit, "maxit must be at minit."
    if compsum and forcing is not None:
        raise NotImplementedError("Compensated sum with forcing is not yet implemented.")

    # Check if force function is autonomous
    if forcing is not None:
        autonomous_force = True
        if time is not None:
            try:
                FW = forcing(W, W, time=time)
            except TypeError:
                pass
            else:
                autonomous_force = False
        FW = np.zeros_like(W)
    
    # Check if autonomous
    autonomous = True
    if time is not None:
        try:
            hamiltonian(W, time=time)
        except TypeError:
            pass
        else:
            autonomous = False
    
    # Stats variables
    total_iterations = 0
    number_of_maxit = 0

    # Initialize
    dW = np.zeros_like(W)
    dW_old = np.zeros_like(W)
    Whalf = np.zeros_like(W)
    PWcomm = np.zeros_like(W)
    Phalf = None  # Scaled stream matrix (own buffer, allocated at first use)
    # PWPhalf = np.zeros_like(W)
    # dthalf = dt/2.0
    hb = hbar(N=W.shape[-1])
    vareps = dt/(2*hb)

    # Specify tolerance if needed
    if (tol == 'auto') or (tol < 0):
        mach_eps = np.finfo(W.dtype).eps
        if not compsum:
            mach_eps = np.sqrt(mach_eps)
        if W.ndim > 2:
            zeroind = (0,)*(W.ndim-2) + (Ellipsis,)
            tol = (mach_eps*dt/hb)*np.linalg.norm(W[zeroind], np.inf)
        else:
            tol = (mach_eps*dt/hb)*np.linalg.norm(W, np.inf)
        if verbatim:
            print("Tolerance set to {}.".format(tol))
        if stats is not None:
            stats['tol_auto'] = tol

    # Variables for compensated summation
    if compsum:
        y_compsum = np.zeros_like(W)
        c_compsum = np.zeros_like(W)
        t_compsum = np.zeros_like(W)
        delta_compsum = np.zeros_like(W)


    # --- Beginning of step loop ---
    for k in range(steps):

        # Apply half a Strang step
        if strang_splitting:
            W = strang_splitting(dt/2, W)

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
            if autonomous:
                P = hamiltonian(Whalf)
            else:
                P = hamiltonian(Whalf, time=time + dt/2)

            # Scale into own buffer (the array returned by the hamiltonian is not modified)
            if Phalf is None or Phalf.shape != P.shape:
                Phalf = np.empty_like(P)
            np.multiply(P, vareps, out=Phalf)

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
                # Compute forcing if needed (scaled into own buffer FW,
                # so the array returned by the forcing is not modified)
                if autonomous_force:
                    F = forcing(P, Whalf)
                else:
                    F = forcing(P, Whalf, time=time + dt/2)
                np.multiply(F, dt/2, out=FW)
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

        # Evaluate callback function
        if callback is not None:
            callback(W, PWcomm)

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

            if forcing:
                FW *= 2
                W += FW
     
        if time is not None:
            time += dt

        # Apply half a Strang step
        if strang_splitting:
            W = strang_splitting(dt/2, W)

        # --- End of step ---

    if verbatim:
        print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))
    if stats is not None:
        stats["iterations"] = total_iterations/steps
        stats["number_of_maxit"] = number_of_maxit/steps

    return W


# Default isospectral method
isomp = isomp_fixedpoint
