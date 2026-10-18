r"""
Integrators for quantized magnetohydrodynamics (MHD) on the sphere.

The state is a stack ``V = [W, Theta]`` of two skew-Hermitian
:math:`N\times N` matrices: the vorticity matrix :math:`W` and the magnetic
potential matrix :math:`\Theta`. The equations are

.. math::

    \dot W = \frac{1}{\hbar}[P, W] + \frac{1}{\hbar}[B, \Theta], \qquad
    \dot \Theta = \frac{1}{\hbar}[P, \Theta],

where :math:`(P, B)` is returned by a Hamiltonian function of the state; for
standard MHD (:func:`solve_mhd`), :math:`\Delta_N P = W` and
:math:`B = \Delta_N \Theta`.

The default integrator :func:`magmp_fixedpoint` (alias ``magmp``) is the
second order magnetic midpoint method of [1]_, a Lie-Poisson integrator
analogous to the isospectral midpoint method of
:mod:`quflow.integrators.isospectral`. It conserves the Casimirs of the
system: the spectrum of :math:`\Theta` and the quantities
:math:`\operatorname{tr}(W\Theta^k)`, up to rounding errors and the
tolerance of the fixed-point iteration.

Both integrators require skew-Hermitian mode (see
:func:`quflow.integrators.isospectral.select_skewherm`).

References
----------
.. [1] K. Modin and M. Roop. *Spatio-temporal Lie-Poisson discretization for
   incompressible magnetohydrodynamics on the sphere*, IMA J. Numer. Anal., 2025.
"""
import numpy as np
import scipy.linalg
import inspect

from ..laplacian import solve_poisson, laplace
from ..geometry import norm_Linf, hbar
from . import isospectral as _isospectral
from .isospectral import commutator, conj_subtract_


def _check_skewherm():
    # Read the setting at call time (a `from ... import _SKEW_HERM_` copy
    # would not follow later calls to select_skewherm)
    if not _isospectral._SKEW_HERM_:
        raise NotImplementedError("MAGMP only works for skew-Hermitian matrices (see select_skewherm).")


def solve_mhd(state):
    r"""
    Hamiltonian function for the standard MHD system.

    Parameters
    ----------
    state : ndarray, shape (2, N, N), complex
        The state ``[W, Theta]``.

    Returns
    -------
    P : ndarray, shape (N, N), complex
        Stream matrix, :math:`\Delta_N P = W` (see
        :func:`quflow.laplacian.solve_poisson`; this is an internal buffer
        that is overwritten by the next call).
    B : ndarray, shape (N, N), complex
        :math:`B = \Delta_N \Theta` (a new array).
    """
    W = state[0,:,:]
    Theta = state[1,:,:]
    P = solve_poisson(W)
    B = laplace(Theta)
    return P, B


def magmp_fixedpoint_old(state, 
                     stepsize, 
                     steps=100, 
                     hamiltonian=solve_mhd, 
                     time=None, 
                     forcing=None, 
                     stats=None,
                     callback=None,
                     tol='auto', 
                     maxit=10, 
                     minit=1, 
                     verbatim=False, 
                     reinitialize=False
                     ):
    r"""
    Time-stepping by magnetic midpoint second order method for skew-Hermitian W
    using fixed-point iterations (older version, kept for reference).

    Same scheme as :func:`magmp_fixedpoint`, but with the dimensionless step
    `stepsize` (that is, ``dt/hbar(N)``) instead of the physical time step, so
    that the equations are effectively
    :math:`\dot W = [P, W] + [B, \Theta]`,
    :math:`\dot\Theta = [P, \Theta]` in dimensionless time.
    Use :func:`magmp_fixedpoint` instead.

    Parameters
    ----------
    state: ndarray, shape = (2,N,N)
        Initial state given as the skew-Hermitian vorticity 
        matrices W = state[0,:,:] and Theta = state[1,:,:]
        (overwritten and returned).
    stepsize: float
        Dimensionless time step length, ``dt/hbar(N)``.
    steps: int
        Number of steps to take.
    hamiltonian: function(state) or function(state, time)
        The Hamiltonian returning (P, B).
    time: float or None (default)
        Time at the initial state. If `None` the system is assumed to
        be autonomous, and the time parameter is not passed to the hamiltonian and forcing.
        The time is in the same (dimensionless) units as `stepsize`.
    forcing: function(P, state) or function(P, state, time) or None (default)
        Extra force function (to allow non-isospectral perturbations),
        returning an array shaped like `state`, in physical time units.
    stats: dict or None
        Dictionary to be filled in with integration statistics
        (``'tol_auto'`` when automatic, ``'iterations'`` and
        ``'number_of_maxit'``, as for
        :func:`quflow.integrators.isospectral.isomp_fixedpoint`).
    callback: function(state, dstate) or None (default)
        Callback function evaluated at the end of every step, 
        just before the state is updated. ``dstate`` contains only the
        :math:`[P, \cdot]` part of the increment.
    tol: float or 'auto' (default)
        Tolerance for iterations. Negative value or "auto" means automatic choice.
    maxit: int
        Maximum number of iterations.
    minit: int
        Minimum number of iterations.
    verbatim: bool
        Print extra information if True. Default is False.
    reinitialize: bool
        Whether to re-initiate the iteration vector at every step.

    Returns
    -------
    state: ndarray
        The state after `steps` steps (the same array as the input).
    """

    # Check input
    assert minit >= 1, "minit must be at least 1."
    assert maxit >= minit, "maxit must be at minit."
    _check_skewherm()

    # Check if force function is autonomous
    if forcing is not None:
        autonomous_force = True
        if time is not None and 'time' in inspect.getfullargspec(forcing).args:
            autonomous_force = False
        FW = np.zeros_like(state)  # Own buffer for the scaled force
    
    # Check if autonomous
    autonomous = True
    if time is not None and 'time' in inspect.getfullargspec(hamiltonian).args:
        autonomous = False

    # Stats variables
    total_iterations = 0
    number_of_maxit = 0

    # Initialize
    dstate = np.zeros_like(state)
    dstate_old = np.zeros_like(state)
    statehalf = np.zeros_like(state)
    Pstatecomm = np.zeros_like(state)
    BThetacomm = np.zeros_like(state[0,:,:])
    BThetaPhalf = np.zeros_like(BThetacomm)
    Phalf, Bhalf = None, None  # Scaled (P, B) (own buffers, allocated at first use)
    # PWPhalf = np.zeros_like(W)
    hhalf = stepsize/2.0
    hb = hbar(N=state.shape[-1])

    # Specify tolerance if needed
    if tol == "auto" or tol < 0:
        mach_eps = np.finfo(state.dtype).eps
        if state.ndim > 2:
            zeroind = (0,)*(state.ndim-2) + (Ellipsis,)
            tol = np.sqrt(mach_eps)*stepsize*np.linalg.norm(state[zeroind], np.inf)
        else:
            tol = np.sqrt(mach_eps)*stepsize*np.linalg.norm(state, np.inf)
        if verbatim:
            print("Tolerance set to {}.".format(tol))
        if stats is not None:
            stats['tol_auto'] = tol

    # --- Beginning of step loop ---
    for k in range(steps):

        # Per step updates
        resnorm = np.inf
        if reinitialize:
            dstate.fill(0.0)

        # --- Beginning of iterations ---
        for i in range(maxit):

            # Update iterations
            total_iterations += 1

            # Compute Wtilde
            np.copyto(statehalf, state)
            statehalf += dstate
            Thetahalf = statehalf[1,:,:]

            # Save dW from previous step
            np.copyto(dstate_old, dstate)

            # Update Ptilde
            if autonomous:
                P, B = hamiltonian(statehalf)
            else:
                P, B = hamiltonian(statehalf, time=time + hhalf)
            # Scale into own buffers (the arrays returned by the hamiltonian are not modified)
            if Phalf is None or Phalf.shape != P.shape:
                Phalf = np.empty_like(P)
            if Bhalf is None or Bhalf.shape != B.shape:
                Bhalf = np.empty_like(B)
            np.multiply(P, hhalf, out=Phalf)
            np.multiply(B, hhalf, out=Bhalf)

            # Compute middle variables
            # PWcomm = Phalf @ Whalf  # old
            np.matmul(Phalf, statehalf, out=Pstatecomm)
            np.matmul(Bhalf, Thetahalf, out=BThetacomm)

            # PWPhalf = PWcomm @ Phalf  # old
            # np.matmul(PWcomm, Phalf, out=PWPhalf)
            np.matmul(Pstatecomm, Phalf, out=dstate)  # More efficient, as PWPhalf is not needed
            conj_subtract_(Pstatecomm, Pstatecomm)
            np.matmul(BThetacomm, Phalf, out=BThetaPhalf)
            conj_subtract_(BThetacomm, BThetacomm)

            # Update dW
            dstate += Pstatecomm
            dstate[0,:,:] += BThetaPhalf
            dstate[0,:,:] -= BThetaPhalf.T.conj()
            dstate[0,:,:] += BThetacomm

            # Add forcing if needed
            if forcing is not None:
                if autonomous_force:
                    F = forcing(P, statehalf)
                else:
                    F = forcing(P, statehalf, time=time + hhalf)
                np.multiply(F, hhalf*hb, out=FW)  # The array returned by the forcing is not modified
                dstate += FW

            # Check if time to break
            if i+1 >= minit:
                # Compute error
                resnorm_old = resnorm
                dstate_old -= dstate
                if dstate_old.ndim > 2:
                    resnormvec = scipy.linalg.norm(dstate_old, ord=np.inf, axis=(-1, -2))
                    if Phalf.ndim == 2:
                        resnorm = resnormvec[0]
                    else:
                        resnorm = resnormvec.max()
                else:
                    resnorm = scipy.linalg.norm(dstate_old, ord=np.inf)
                if resnorm <= tol or resnorm >= resnorm_old:
                    break

        else:
            # We used maxit iterations
            number_of_maxit += 1
            if verbatim:
                print("Max iterations {} reached at step {}.".format(maxit, k))
            # if stats:
            #     stats['maxit_reached'] = True

        # Rescale commutators (as they are divided by 2)
        Pstatecomm *= 2
        BThetacomm *= 2

        # Evaluate callback function
        if callback is not None:
            callback(state, Pstatecomm)

        # Update state
        state += Pstatecomm
        state[0,:,:] += BThetacomm   
        if forcing is not None:
            FW *= 2
            state += FW

        if time is not None:
            time += stepsize

        # --- End of step ---

    if verbatim:
        print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))
    if stats is not None:
        stats["iterations"] = total_iterations/steps
        stats["number_of_maxit"] = number_of_maxit/steps

    return state


def magmp_fixedpoint(W,
                     dt, 
                     steps=100, 
                     hamiltonian=solve_mhd, 
                     time=None, 
                     forcing=None, 
                     stats=None,
                     callback=None,
                     tol='auto', 
                     maxit=10, 
                     minit=1, 
                     verbatim=False,  
                     reinitialize=False
                     ):
    r"""
    Time-stepping by the magnetic midpoint second order method for MHD,
    using fixed-point iterations.

    This is the default MHD integrator, also available as
    ``quflow.integrators.magmp``. With :math:`h = \Delta t/\hbar`,
    :math:`\varepsilon = h/2` and :math:`(\tilde P, \tilde B)` the output of
    `hamiltonian` at the midpoint :math:`(\tilde W, \tilde\Theta)`, each step
    solves by fixed-point iteration

    .. math::

        \tilde W &= W_n + \varepsilon[\tilde P, \tilde W]
                    + \varepsilon^2 \tilde P\tilde W\tilde P
                    + \varepsilon[\tilde B, \tilde\Theta]
                    + \varepsilon^2(\tilde B\tilde\Theta\tilde P
                                    + \tilde P\tilde\Theta\tilde B), \\
        \tilde\Theta &= \Theta_n + \varepsilon[\tilde P, \tilde\Theta]
                    + \varepsilon^2 \tilde P\tilde\Theta\tilde P,

    and then sets

    .. math::

        W_{n+1} = W_n + h\big([\tilde P, \tilde W] + [\tilde B, \tilde\Theta]\big),
        \qquad
        \Theta_{n+1} = \Theta_n + h[\tilde P, \tilde\Theta].

    This is the magnetic midpoint method of Modin and Roop (IMA J. Numer.
    Anal., 2025). It conserves the spectrum of :math:`\Theta` (for
    :math:`\Theta` it is the isospectral midpoint method of
    :mod:`quflow.integrators.isospectral`) and the quantities
    :math:`\operatorname{tr}(W\Theta^k)`. The forcing, if given, enters as in
    :func:`quflow.integrators.isospectral.isomp_fixedpoint`.

    Parameters
    ----------
    W: ndarray, shape (2, N, N), complex
        Initial state ``[W, Theta]`` of skew-Hermitian matrices (overwritten
        and returned).
    dt: float
        Time step length in physical time units.
    steps: int
        Number of steps to take. Default is 100.
    hamiltonian: function(state) or function(state, time)
        The Hamiltonian returning the pair ``(P, B)``. Default is
        :func:`solve_mhd`. The returned arrays are not modified.
    time: float or None (default)
        Time at the initial state. If `None` the system is assumed to
        be autonomous, and the time parameter is not passed to the hamiltonian and forcing.
    forcing: function(P, state) or function(P, state, time) or None (default)
        Extra force function (to allow non-isospectral perturbations),
        returning an array shaped like the state. The returned array is not
        modified, so the function may return a stored array.
    stats: dict or None
        Dictionary to be filled in with integration statistics
        (``'tol_auto'`` when automatic, ``'iterations'`` and
        ``'number_of_maxit'``, as for
        :func:`quflow.integrators.isospectral.isomp_fixedpoint`).
    callback: function(state, dstate) or None (default)
        Callback function evaluated at the end of every step, 
        just before the state is updated. ``dstate`` contains only the
        :math:`h[\tilde P, \cdot]` part of the increment.
    tol: float or 'auto' (default)
        Tolerance for iterations. Negative value or "auto" means automatic
        choice, ``sqrt(eps)*h*norm(W[0], inf)``.
    maxit: int
        Maximum number of iterations. Default is 10.
    minit: int
        Minimum number of iterations. Default is 1.
    verbatim: bool
        Print extra information if True. Default is False.
    reinitialize: bool
        Whether to re-initiate the iteration vector at every step.

    Returns
    -------
    W: ndarray, shape (2, N, N), complex
        The state after `steps` steps (the same array as the input).

    Raises
    ------
    NotImplementedError
        In non-skew-Hermitian mode.

    Notes
    -----
    The fixed-point iteration converges only if the step is small enough
    compared with the magnitude of :math:`\tilde P` and :math:`\tilde B`.
    Since :math:`B = \Delta_N\Theta` grows with the wavenumbers of
    :math:`\Theta`, strong or small-scale magnetic fields need small steps.
    If the iteration does not converge within `maxit` iterations, the
    conservation properties hold only up to the iteration error; the fraction
    of such steps is reported in ``stats['number_of_maxit']``.

    Examples
    --------
    >>> import numpy as np
    >>> import quflow as qf
    >>> N = 8
    >>> rng = np.random.default_rng(0)
    >>> W = qf.shr2mat(np.r_[0.0, rng.standard_normal(N**2 - 1)], N=N)
    >>> Theta = qf.shr2mat(np.r_[0.0, 0.2*rng.standard_normal(N**2 - 1)], N=N)
    >>> state = qf.integrators.magmp(np.array([W, Theta]), dt=0.1*qf.hbar(N), steps=10)
    """

    # Check input
    assert minit >= 1, "minit must be at least 1."
    assert maxit >= minit, "maxit must be at minit."
    _check_skewherm()

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
    BThetacomm = np.zeros_like(W[0,:,:])
    BThetaPhalf = np.zeros_like(BThetacomm)
    Phalf, Bhalf = None, None  # Scaled (P, B) (own buffers, allocated at first use)

    # PWPhalf = np.zeros_like(W)
    # dthalf = dt/2.0
    hb = hbar(N=W.shape[-1])
    vareps = dt/(2*hb)

    # Specify tolerance if needed
    if (tol == 'auto') or (tol < 0):
        mach_eps = np.finfo(W.dtype).eps
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
            Thetahalf = Whalf[1,:,:]

            # Save dW from previous step
            np.copyto(dW_old, dW)

            # Update Ptilde
            if autonomous:
                P, B = hamiltonian(Whalf)
            else:
                P, B = hamiltonian(Whalf, time=time + dt/2)
            # Scale into own buffers (the arrays returned by the hamiltonian are not modified)
            if Phalf is None or Phalf.shape != P.shape:
                Phalf = np.empty_like(P)
            if Bhalf is None or Bhalf.shape != B.shape:
                Bhalf = np.empty_like(B)
            np.multiply(P, vareps, out=Phalf)
            np.multiply(B, vareps, out=Bhalf)

            # Compute middle variables
            # PWcomm = Phalf @ Whalf  # old
            np.matmul(Phalf, Whalf, out=PWcomm)
            np.matmul(Bhalf, Thetahalf, out=BThetacomm)
            # PWPhalf = PWcomm @ Phalf  # old
            # np.matmul(PWcomm, Phalf, out=PWPhalf)
            np.matmul(PWcomm, Phalf, out=dW)  # More efficient, as PWPhalf is not needed
            np.matmul(BThetacomm, Phalf, out=BThetaPhalf)
            conj_subtract_(PWcomm, PWcomm)
            conj_subtract_(BThetacomm, BThetacomm)
            
            # Update dW
            # np.copyto(dW, PWPhalf)
            dW += PWcomm
            dW[0,:,:] += BThetaPhalf
            dW[0,:,:] -= BThetaPhalf.T.conj()
            dW[0,:,:] += BThetacomm

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
        BThetacomm *= 2

        # Evaluate callback function
        if callback is not None:
            callback(W, PWcomm)

        W += PWcomm
        W[0,:,:] += BThetacomm    

        if forcing:
            FW *= 2
            W += FW
     
        if time is not None:
            time += dt

        # --- End of step ---

    if verbatim:
        print("Average number of iterations per step: {:.2f}".format(total_iterations/steps))
    if stats is not None:
        stats["iterations"] = total_iterations/steps
        stats["number_of_maxit"] = number_of_maxit/steps

    return W

# Default magnetic integrator
magmp = magmp_fixedpoint
