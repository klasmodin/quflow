import numpy as np
from numba import njit
import scipy.sparse.linalg
from scipy.sparse import coo_matrix

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
def compute_sparse_laplacian_ind_(N, values, ivals, jvals, bc=False):
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


def compute_sparse_laplacian(N, bc=False):
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
    values = np.zeros(3*N**2-4*N+2+N, dtype=complex)  # Used to be 'complex' but no need for that
    ivals = np.zeros(values.shape, dtype=int)
    jvals = np.zeros(values.shape, dtype=int)

    compute_sparse_laplacian_ind_(N, values, ivals, jvals, bc=bc)

    # Create sparse matrix
    A = coo_matrix((values, (ivals, jvals)), shape=(N**2, N**2)).tocsc()

    return A


def compute_lu(A):
    if _use_umfpack:
        scipy.sparse.linalg.use_solver(useUmfpack=True)
    else:
        scipy.sparse.linalg.use_solver(useUmfpack=False)
    return scipy.sparse.linalg.splu(A)


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def laplacian(N, bc=False):
    """
    Return quantized laplacian (as a sparse matrix).

    Parameters
    ----------
    N: int
    bc: bool
        Whether to add boundary conditions to remove singularity.

    Returns
    -------
    A : sparse matrix
    """
    global _sparse_laplacian_cache

    if (N, bc) not in _sparse_laplacian_cache:
        A = compute_sparse_laplacian(N)
        _sparse_laplacian_cache[(N, bc)] = A

    return _sparse_laplacian_cache[(N, bc)]


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

    global _lu_laplacian_cache

    N = W.shape[0]

    if N not in _lu_laplacian_cache:
        # Get sparse laplacian
        A = laplacian(N, bc=True)

        # Compute sparse LU
        _lu_laplacian_cache[N] = compute_lu(A)

    P = _lu_laplacian_cache[N].solve(W.ravel()).reshape((N, N))
    P.ravel()[::N+1] -= np.trace(P)/N
    return P


def solve_heat(h_times_nu, W0):
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
        # Get sparse laplacian
        A = laplacian(N, False)

        # Using backward Euler method
        T = scipy.sparse.eye(N**2) - h_times_nu*A

        # Compute sparse LU
        _lu_heat_flow_cache[(N, h_times_nu)] = compute_lu(T.tocsc())

    Wh = _lu_heat_flow_cache[(N, h_times_nu)].solve(W0.ravel()).reshape((N, N))
    return Wh
