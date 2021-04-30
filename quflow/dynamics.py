import numpy as np
from .utils import elm2ind
from numba import njit
import scipy.sparse.linalg

# ----------------
# GLOBAL VARIABLES
# ----------------

_lu_laplacian_cache = dict()
_sparse_laplacian_cache = dict()
_lu_heat_flow_cache = dict()

_use_umfpack = True


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------

def compute_sparse_laplacian_alt(N):
    s = (N - 1)/2
    mvals = np.linspace(-s, s, N)
    m1 = mvals[:, np.newaxis]
    m2 = mvals[np.newaxis, :]

    # Set diagonal elements
    coeff1 = (2*(s*(s+1)-m1*m2)).ravel()
    ivals = ((m1 + s)*N + m2 + s).ravel().astype(int)
    jvals = ivals.copy()
    values = -coeff1

    # Set first off diagonal
    m1 = mvals[:-1, np.newaxis]
    m2 = mvals[np.newaxis, :-1]
    coeff2 = (-np.sqrt(s*(s+1)-m1*(m1+1))*np.sqrt(s*(s+1)-m2*(m2+1))).ravel()
    ivals = np.hstack((ivals, ((m1 + s)*N + m2 + s).ravel().astype(int)))
    jvals = np.hstack((jvals, ((m1 + s + 1)*N + m2 + s + 1).ravel().astype(int)))
    values = np.hstack((values, -coeff2))

    # Set second off diagonal
    m1 = mvals[1:, np.newaxis]
    m2 = mvals[np.newaxis, 1:]
    coeff3 = (-np.sqrt(s*(s+1)-m1*(m1-1))*np.sqrt(s*(s+1)-m2*(m2-1))).ravel()
    ivals = np.hstack((ivals, ((m1 + s)*N + m2 + s).ravel().astype(int)))
    jvals = np.hstack((jvals, ((m1 + s - 1)*N + m2 + s - 1).ravel().astype(int)))
    values = np.hstack((values, -coeff3))

    # Set BC


@njit
def compute_sparse_laplacian_ind_(N, values, ivals, jvals, bc=True):
    s = (N - 1)/2
    mvals = np.linspace(-s, s, N)
    count = 0
    for m1 in mvals:
        for m2 in mvals:
            coeff1 = 2*(s*(s+1)-m1*m2)
            if abs(coeff1) > 1e-10:
                ivals[count] = round((m1 + s)*N + m2 + s)
                jvals[count] = round((m1 + s)*N + m2 + s)
                values[count] = -coeff1
                count += 1

            if m1 < s and m2 < s:
                coeff2 = -np.sqrt(s*(s+1)-m1*(m1+1))*np.sqrt(s*(s+1)-m2*(m2+1))
                if abs(coeff2) > 1e-10:
                    ivals[count] = round((m1 + s)*N + m2 + s)
                    jvals[count] = round((m1 + s + 1)*N + m2 + s + 1)
                    values[count] = -coeff2
                    count += 1

            if m1 > -s and m2 > -s:
                coeff3 = -np.sqrt(s*(s+1)-m1*(m1-1))*np.sqrt(s*(s+1)-m2*(m2-1))
                if abs(coeff3) > 1e-10:
                    ivals[count] = round((m1 + s)*N + m2 + s)
                    jvals[count] = round((m1 + s - 1)*N + m2 + s - 1)
                    values[count] = -coeff3
                    count += 1

    # Make sure matrix is invertible (corresponds to adding BC in Poisson equations)
    if bc:
        for h in range(N):
            ivals[count] = 0
            jvals[count] = h*(N+1)
            values[count] = -1/N
            count += 1


def compute_sparse_laplacian(N, bc=True):
    """
    Return the sparse laplacian for a specific bandwidth `N`.

    Parameters
    ----------
    N: int
    bc: bool
        Whether to add conditions to exclude singular matrix.

    Returns
    -------
    A: scipy.sparse.spmatrix
        Sparse matrix in some scipy format (typically `csc_matrix`).
    """
    values = np.zeros(3*N**2-4*N+2+N, dtype=float)  # Used to be 'complex' but no need for that
    ivals = np.zeros(values.shape, dtype=int)
    jvals = np.zeros(values.shape, dtype=int)

    compute_sparse_laplacian_ind_(N, values, ivals, jvals, bc=bc)

    # Create sparse matrix
    from scipy.sparse import coo_matrix
    A = coo_matrix((values, (ivals, jvals)), shape=(N**2, N**2)).tocsc()

    return A


def compute_lu_laplacian(A):
    if _use_umfpack:
        scipy.sparse.linalg.use_solver(useUmfpack=True)
    else:
        scipy.sparse.linalg.use_solver(useUmfpack=False)
    return scipy.sparse.linalg.splu(A)


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def laplacian(N):
    """
    Return quantized laplacian (as a sparse matrix).

    Parameters
    ----------
    N: int

    Returns
    -------
    A : sparse matrix
    """
    global _sparse_laplacian_cache

    if N not in _sparse_laplacian_cache:
        A = compute_sparse_laplacian(N)
        _sparse_laplacian_cache[N] = A

    return _sparse_laplacian_cache[N]


def laplace(P):
    """
    Return quantized laplacian applied to stream function `P`.

    Parameters
    ----------
    P: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    W: ndarray(shape=(N, N), dtype=complex)
    """
    N = P.shape[0]
    A = laplacian(N)

    W = A.dot(P.ravel()).reshape((N, N))
    return W


def solve_poisson(W):
    """
    Return stream matrix `P` for `W`.

    Parameters
    ----------
    W: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    P: ndarray(shape=(N, N), dtype=complex)
    """

    global _sparse_laplacian_cache
    global _lu_laplacian_cache

    N = W.shape[0]

    if N not in _lu_laplacian_cache:
        if N not in _sparse_laplacian_cache:
            _sparse_laplacian_cache[N] = compute_sparse_laplacian(N)
        _lu_laplacian_cache[N] = compute_lu_laplacian(_sparse_laplacian_cache[N])

    P = _lu_laplacian_cache[N].solve(W.ravel()).reshape((N, N))
    return P


def solve_heat_flow(h_times_nu, W0):
    """
    Solve quantized heat equation.

    Parameters
    ----------
    h_times_nu: float
        Time-step times viscosity.
    W0: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    Wh: ndarray(shape=(N, N), dtype=complex)
    """
    global _lu_heat_flow_cache

    N = W0.shape[0]

    if (N, h_times_nu) not in _lu_heat_flow_cache:
        if _use_umfpack:
            scipy.sparse.linalg.use_solver(useUmfpack=True)
        else:
            scipy.sparse.linalg.use_solver(useUmfpack=False)
        A = laplacian(N)
        _lu_heat_flow_cache[(N, h_times_nu)] = \
            scipy.sparse.linalg.splu((scipy.sparse.eye(N**2) - h_times_nu*A).tocsc())

    Wh = _lu_heat_flow_cache[(N, h_times_nu)].solve(W0.ravel()).reshape((N, N))
    return Wh


def scale_decomposition(W, P=None):
    """
    Perform canonical scale separation.

    Parameters
    ----------
    W: ndarray
        Vorticity
    P: ndarray (optional)
        Stream matrix. Computed if not given.

    Returns
    -------
    (Ws, Wr): tuple of ndarray
    """
    if P is None:
        P = solve_poisson(W)

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
    from .transforms import as_shr
    from .utils import elm2ind
    omegar = as_shr(data)
    N = round(np.sqrt(omegar.shape[0]))
    energy = np.ones(N-1, dtype=float)
    for el in range(1, N):
        energy[el-1] = (omegar[elm2ind(-el, el):elm2ind(el, el)+1]**2).sum()/(el*(el+1))
    return energy
