import numpy as np

from .laplacian import solve_poisson, solve_heat
from .integrators import isomp
from .utils import seconds2qtime, rotate
from .quantization import get_basis
from numba import njit, prange

# ----------------
# GLOBAL VARIABLES
# ----------------


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------


@njit(parallel=False)
def project_el_(basis, el, W, W_out, multiplier=1.0):
    """
    Low-level implementation of `project_momentum`.

    Parameters
    ----------
    basis: ndarray, shape (np.sum(np.arange(N)**2),)
    el: int
    W: ndarray, shape (N,N)
    W_out: ndarray, shape (N,N)
    multiplier: float
    """
    N = W.shape[0]
    basis_break_indices = np.zeros((N+1,), dtype=np.int32)
    basis_break_indices[1:] = (np.arange(N, 0, -1, dtype=np.int32)**2).cumsum()

    for m in prange(el+1):
        bind0 = basis_break_indices[m]
        bind1 = basis_break_indices[m+1]
        basis_m_mat = basis[bind0:bind1].reshape((N-m, N-m)).astype(np.complex128)

        # Lower diagonal
        diag_m = basis_m_mat[:, el-m]
        project_lower_diag_(diag_m, m, W, W_out, multiplier)

        # Upper diagonal
        if m != 0:
            sgn = 1 if m % 2 == 0 else -1
            diag_m = sgn*basis_m_mat[:, el-m]
            project_upper_diag_(diag_m, m, W, W_out, multiplier)


@njit
def project_lower_diag_(diag_m, m, W, W_out, multiplier=1.0):
    N = W.shape[0]

    # Project m:th diagonal onto diag_m
    a = 0.0
    for i in range(N-m):
        a += W[i+m, i]*diag_m[i]

    # Assign m:th diagonal
    a *= multiplier
    for i in range(N-m):
        W_out[i+m, i] += a*diag_m[i]


@njit
def project_upper_diag_(diag_m, m, W, W_out, multiplier=1.0):
    N = W.shape[0]

    # Project m:th diagonal onto diag_m
    a = 0.0
    for i in range(N-m):
        a += W[i, i+m]*diag_m[i]

    # Assign m:th diagonal
    a *= multiplier
    for i in range(N-m):
        W_out[i, i+m] += a*diag_m[i]


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def project_el(W, el=1, complement=False):
    """
    Projection of `W` onto eigenspace for `el`.

    Parameters
    ----------
    W: ndarray
        Vorticity matrix to project.
    el: int or list of ints
        Positive int 0<=el<=N-1 defining the eigenspace.
    complement: bool
        If `True`, project onto the orthogonal complement instead.

    Returns
    -------
    ndarray:
        Projected matrix.
    """
    if complement:
        multiplier = -1.0
        W_out = W.copy()
    else:
        multiplier = 1.0
        W_out = np.zeros_like(W)

    N = W.shape[0]
    basis = get_basis(N)

    if np.isscalar(el):
        el = [el]

    for eli in el:
        if eli < 0:
            eli = N+eli
        # Call low-level projection function
        project_el_(basis, eli, W, W_out, multiplier)

    return W_out


# ----------------------
# GENERIC SOLVE FUNCTION
# ----------------------

def solve(W, qstepsize=0.1, steps=None, qtime=None, time=None,
          method=isomp, method_kwargs=None,
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


# --------------------------------
# HELPERS TO GENERATE INITIAL DATA
# --------------------------------

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