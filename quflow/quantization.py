import numpy as np
from scipy.io import loadmat
from .utils import elm2ind
from numba import njit
import os
import os.path
import h5py
import appdirs

# ----------------
# GLOBAL VARIABLES
# ----------------

# _basis_path_default = "/Users/moklas/Documents/Coding/eulersph/qvflow/precomputations"
# _basis_path_default = os.path.join(os.path.expanduser("~"), "quflow")
_app_name = "quflow"
_basis_path_default = appdirs.user_data_dir(_app_name, False)
# Typical user data directories are:
#     macos:        ~/Library/Application Support/<AppName>
#     Unix:         ~/.local/share/<AppName>    # or in $XDG_DATA_HOME, if defined
#     Win XP:       C:\Documents and Settings\<username>\Application Data\<AppAuthor>\<AppName>
#     Win 7/10:     C:\Users\<username>\AppData\Local\<AppAuthor>\<AppName>

_basis_filename_default = "quflow_basis.hdf5"
_save_computed_basis_default = True

_file_version = 0.1
_basis_cache = dict()


# ---------------------
# LOWER LEVEL FUNCTIONS
# ---------------------

def load_basis_mat(filename):
    # Load matlab basis
    basis_ind = np.squeeze(loadmat(filename)['BASIS'])

    # Compute N
    N = round(np.sqrt(basis_ind.shape[0]+1))

    # Convert to list
    basis_ind = [np.ones((1, N), dtype=float)/np.sqrt(N)]+list(basis_ind)

    # Compute indices
    break_indices_last = (np.arange(N, 0, -1)**2).cumsum()
    break_indices_first = np.hstack((0, break_indices_last[:-1]))

    # Allocate
    basis_flat = np.zeros(break_indices_last[-1], dtype=float)

    # Convert to flat basis representation
    for m, bind0, bind1 in zip(range(N), break_indices_first, break_indices_last):
        basis_m_mat = basis_flat[bind0:bind1].reshape((N-m, N-m))
        for el in range(m, N):
            ind = elm2ind(el, m)
            basis_m_mat[:, el-m] = np.squeeze(basis_ind[ind])

    return basis_flat


def load_basis_hdf5(filename, N):
    with h5py.File(filename, "a") as f:
        if "BASIS_%s" % str(N) in f:
            basis_set = f["BASIS_%s" % str(N)]
            assert (basis_set.attrs['QUFLOW_FILE_VERSION'] == _file_version)
            basis = basis_set[:]
        else:
            basis = None
    return basis


def load_basis_npy(filename):
    return np.load(filename)


def get_N_for_basis(basis):
    x = basis if isinstance(basis, int) else basis.shape[0]
    N = -1-round((-1+1/(3**(1/3)*(108*x+np.sqrt(3)*np.sqrt(-1+3888*x**2))**(1/3))
                  - (108*x+np.sqrt(3)*np.sqrt(-1+3888*x**2))**(1/3)/3**(2/3))/2)
    assert ((np.arange(1, N+1)**2).sum() == x)  # Check that we solved the correct equation
    return N


def save_basis_hdf5(filename, basis):
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    N = get_N_for_basis(basis)
    with h5py.File(filename, "a") as f:
        if not "BASIS_%s" % str(N) in f:
            basis_set = f.create_dataset("BASIS_%s" % str(N), basis.shape, dtype=str(basis.dtype))
            basis_set[:] = basis[:]
            basis_set.attrs['QUFLOW_FILE_VERSION'] = _file_version
        else:
            basis = None
    return basis


def get_basis_dirs():
    """
    Return list of possible basis directories.
    """
    # Create list of possible paths
    basis_paths = []
    if 'QUFLOW_BASIS_PATH' in os.environ:
        basis_paths += os.environ['QUFLOW_BASIS_PATH'].split(":")
    basis_paths.append(_basis_path_default)

    # Create list of actually available dirs
    basis_paths_valid = []
    for path in basis_paths:
        if os.path.isdir(path):
            basis_paths_valid.append(path)

    return basis_paths_valid


def get_basis_files(basis_filename=None):
    """
    Return list of basis files found in the basis directories.
    """

    # Check for filename if not given. Prefer environment variable over default.
    if basis_filename is None:
        if 'QUFLOW_BASIS_FILENAME' in os.environ:
            basis_filename = os.environ['QUFLOW_BASIS_FILENAME']
        else:
            basis_filename = _basis_filename_default

    # Create list of actually available files
    basis_filenames_valid = []
    for path in get_basis_dirs():
        filepath = os.path.join(path, basis_filename)
        if os.path.isfile(filepath):
            basis_filenames_valid.append(filepath)

    return basis_filenames_valid


def convert_mat_to_hdf5_basis(filename_mat, filename_hdf5=None):
    if filename_hdf5 is None:
        filename_hdf5 = get_basis_files()[0]
    basis = load_basis_mat(filename_mat)
    save_basis_hdf5(filename_hdf5, basis)


def compute_basis(N):
    raise NotImplementedError("Basis computation implementation still lacking.")


@njit
def assign_lower_diag_(diag_m, m, W_out):
    N = W_out.shape[0]
    for i in range(N-m):
        W_out[i+m, i] = diag_m[i]


@njit
def assign_upper_diag_(diag_m, m, W_out):
    N = W_out.shape[0]
    for i in range(N-m):
        W_out[i, i+m] = diag_m[i]


def shr2mat_(omega, basis, W_out):
    """
    Low-level implementation of `shr2mat`.

    Parameters
    ----------
    omega: ndarray, dtype=float, shape=(N*(N+1)/2,)
    basis: ndarray, dtype=float, shape=(np.sum(np.arange(N)**2),)
    W_out: ndarray, dtype=complex, shape=(N,N)
    """
    N = W_out.shape[0]
    basis_break_indices = np.hstack((0, (np.arange(N, 0, -1)**2).cumsum()))

    for m in range(N):
        bind0 = basis_break_indices[m]
        bind1 = basis_break_indices[m+1]
        basis_m_mat = basis[bind0:bind1].reshape((N-m, N-m))

        if m == 0:  # Diagonal
            omega_zero_ind = elm2ind(np.arange(0, N), 0)
            diag = basis_m_mat@omega[omega_zero_ind]
            assign_lower_diag_(diag, 0, W_out)
        else:
            # Lower diagonal
            omega_minus_m_ind = elm2ind(np.arange(m, N), -m)
            omega_plus_m_ind = elm2ind(np.arange(m, N), m)
            omega_complex = (1./np.sqrt(2))*(omega[omega_plus_m_ind]-1j*omega[omega_minus_m_ind])
            sgn = 1 if m % 2 == 0 else -1
            diag_m = sgn*basis_m_mat@omega_complex
            assign_lower_diag_(diag_m.conj(), m, W_out)

            # Upper diagonal
            assign_upper_diag_(diag_m, m, W_out)

    W_out *= 1.0j


def mat2shr_(W, basis, omega_out):
    N = W.shape[0]
    basis_break_indices = np.hstack((np.array([0]), (np.arange(N, 0, -1)**2).cumsum()))

    for m in range(N):
        bind0 = basis_break_indices[m]
        bind1 = basis_break_indices[m+1]
        basis_m_mat = basis[bind0:bind1].reshape((N-m, N-m))

        if m == 0:  # Diagonal
            omega_zero_ind = elm2ind(np.arange(0, N), 0)
            diag_m = np.diagonal(W, 0)  # np.diagonal is more efficient than np.diag, but doesn't work with njit
            omega_out[omega_zero_ind] = ((diag_m@basis_m_mat)/1.0j).real

        else:
            # Lower diagonal
            omega_pos_m_ind = elm2ind(np.arange(m, N), m)
            diag_m = np.diagonal(W, -m)  # np.diagonal is more efficient than np.diag, but doesn't work with njit
            omega_partial_complex = diag_m@basis_m_mat
            sgn = 1 if m % 2 == 0 else -1
            omega_out[omega_pos_m_ind] = np.sqrt(2)*sgn*omega_partial_complex.imag

            # Upper diagonal
            omega_neg_m_ind = elm2ind(np.arange(m, N), -m)
            omega_out[omega_neg_m_ind] = -np.sqrt(2)*sgn*omega_partial_complex.real


def shc2mat_(omega, basis, W_out):
    """
    Low-level implementation of `shc2mat`.

    Parameters
    ----------
    omega: ndarray, shape (N*(N+1)/2,)
    basis: ndarray, shape (np.sum(np.arange(N)**2),)
    W_out: ndarray, shape (N,N)
    """
    N = W_out.shape[0]
    basis_break_indices = np.hstack((0, (np.arange(N, 0, -1)**2).cumsum()))

    for m in range(N):
        bind0 = basis_break_indices[m]
        bind1 = basis_break_indices[m+1]
        basis_m_mat = basis[bind0:bind1].reshape((N-m, N-m))

        # Lower diagonal
        omega_m_ind = elm2ind(np.arange(m, N), m)
        diag_m = basis_m_mat@omega[omega_m_ind]
        assign_lower_diag_(diag_m, m, W_out)

        # Upper diagonal
        if m != 0:
            omega_m_ind = elm2ind(np.arange(m, N), -m)
            sgn = 1 if m % 2 == 0 else -1
            diag_m = sgn*basis_m_mat@omega[omega_m_ind]
            assign_upper_diag_(diag_m, m, W_out)

    W_out *= 1.0j


def mat2shc_(W, basis, omega_out):
    N = W.shape[0]
    basis_break_indices = np.hstack((np.array([0]), (np.arange(N, 0, -1)**2).cumsum()))

    for m in range(N):
        bind0 = basis_break_indices[m]
        bind1 = basis_break_indices[m+1]
        basis_m_mat = basis[bind0:bind1].reshape((N-m, N-m))

        # Lower diagonal
        omega_m_ind = elm2ind(np.arange(m, N), m)
        diag_m = np.diagonal(W, -m)  # np.diagonal is more efficient than np.diag, but doesn't work with njit
        omega_out[omega_m_ind] = diag_m@basis_m_mat

        # Upper diagonal
        if m != 0:
            omega_m_ind = elm2ind(np.arange(m, N), -m)
            diag_m = np.diagonal(W, m)
            sgn = 1 if m % 2 == 0 else -1
            omega_out[omega_m_ind] = sgn*diag_m@basis_m_mat

    omega_out /= 1.0j


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def get_basis(N):
    """
    Return a quantization basis for band limit N.
    The basis is obtained as follows:
    - First look in memory cache.
    - Second look in storage cache.
    - Third compute basis from scratch (this is computationally heavy).

    Parameters
    ----------
    N: int

    Returns
    -------
    basis: ndarray
    """
    global _basis_cache

    basis = None

    # First look in the cache and quickly return if found
    if N in _basis_cache:
        return _basis_cache[N]

    # Next look for a precomputed saved HDF5 basis
    if basis is None:
        for basis_filename in get_basis_files():
            basis = load_basis_hdf5(basis_filename, N)
            if basis is not None:
                break

    # Next look for a precomputed saved NPY basis
    if basis is None:
        for basis_filename in get_basis_files("BASIS_%s.npy" % N):
            basis = load_basis_npy(basis_filename)
            if basis is not None:
                break

    # Next look for a precomputed saved NPZ basis
    if basis is None:
        for basis_filename in get_basis_files("BASIS_%s.npz" % N):
            basis = load_basis_npy(basis_filename)
            if basis is not None:
                break

    # Next look for a precomputed saved MAT basis
    if basis is None:
        for basis_filename in get_basis_files("BASIS_%s.mat" % N):
            basis = load_basis_mat(basis_filename)
            if basis is not None:
                break

    # Finally, if no precomputed basis is to be found, compute it
    if basis is None:
        basis = compute_basis(N)
        if 'QUFLOW_SAVE_COMPUTED_BASIS' in os.environ:
            save_computed_basis = False if os.environ['QUFLOW_SAVE_COMPUTED_BASIS'] \
                                           in ("0", "false", "False", "FALSE") else True
        else:
            save_computed_basis = _save_computed_basis_default
        if save_computed_basis:
            save_basis_hdf5(_basis_filename_default, basis)

    # Save basis to cache
    _basis_cache[N] = basis

    return basis


def shr2mat(omega, N=-1):
    """
    Convert real spherical harmonics to matrix.

    Parameters
    ----------
    omega: ndarray(shape=(N**2,), dtype=float)
    N : int (optional)
        Size of matrix (automatic if not specified).

    Returns
    -------
    W : ndarray(shape=(N, N), dtype=complex)
    """

    # Process input depending on N
    if N == -1:
        N = round(np.sqrt(omega.shape[0]))
    else:
        if omega.shape[0] < N**2:
            omega = np.hstack((omega, np.zeros(N**2-omega.shape[0])))
        else:
            omega = omega[:N**2]

    W_out = np.zeros((N, N), dtype=complex)
    basis = get_basis(N)
    shr2mat_(omega, basis, W_out)

    return W_out


def mat2shr(W):
    """
    Convert NxN complex matrix to real spherical harmonics.

    Parameters
    ----------
    W: ndarray(shape=(N, N), dtype=complex)

    Returns
    -------
    omega: ndarray(shape=(N**2,), dtype=float)
    """
    N = W.shape[0]
    omega = np.zeros(N**2, dtype=float)
    basis = get_basis(N)
    mat2shr_(W, basis, omega)

    return omega


def shc2mat(omega, N=-1):
    """
    Convert complex spherical harmonics to matrix.

    Parameters
    ----------
    omega: complex ndarray, shape (N**2,)
    N : (optional) size of matrix (automatic if not specified)

    Returns
    -------
    W : complex ndarray, shape (N, N)
    """

    # Process input depending on N
    if N == -1:
        N = round(np.sqrt(omega.shape[0]))
    else:
        if omega.shape[0] < N**2:
            omega = np.hstack((omega, np.zeros(N**2-omega.shape[0])))
        else:
            omega = omega[:N**2]

    W_out = np.zeros((N, N), dtype=complex)
    basis = get_basis(N)
    shc2mat_(omega, basis, W_out)

    return W_out


def mat2shc(W):
    """
    Convert NxN complex matrix to complex spherical harmonics.

    Parameters
    ----------
    W: complex ndarray, shape (N, N)

    Returns
    -------
    omega: complex ndarray, shape (N**2,)
    """
    N = W.shape[0]
    omega = np.zeros(N**2, dtype=complex)
    basis = get_basis(N)
    mat2shc_(W, basis, omega)

    return omega