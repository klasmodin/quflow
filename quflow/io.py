import numpy as np
from scipy.io import loadmat
from .utils import elm2ind
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
_file_version = 0.1


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


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

def save(filename, data, qtype="shr"):
    """
    Save `data` in HDF5 file `filename`. The HDF5 file is created if
    it does not exist already.
    The data is stored in format `qtype` which defaults to `shr`.

    Parameters
    ----------
    filename: str
    data: ndarray
    qtype: str
    """
    with h5py.File(filename, "a") as f:
        if qtype == "shr":
            from .transforms import as_shr
            omegar = as_shr(data)
            datapath = "omegar"
            if datapath not in f:
                dset = f.create_dataset(datapath, (1,)+omegar.shape,
                                        dtype=omegar.dtype,
                                        maxshape=(None,)+omegar.shape
                                        )
                dset.attrs["N"] = round(np.sqrt(omegar.shape[0]))
                dset.attrs["qtype"] = "shr"
            elif f[datapath].shape[-1] != omegar.shape[0] or f[datapath].ndim != 2:
                raise ValueError("The file qtype does not seem to be correct.")
            else:
                dset = f[datapath]
            dset.resize((dset.shape[0]+1,) + (dset.shape[1],))
            dset[-1, :] = omegar
        else:
            raise ValueError("Format %s is not supported yet."%qtype)


def load(filename, qtype="shr"):
    """
    Load data saved in either MATLAB or HDF5 format.

    Parameters
    ----------
    filename: str
    qtype: str

    Returns
    -------
    data: h5py.Dataset or ndarray
    """
    if filename[-4:] == "hdf5":
        f = h5py.File(filename, "r")
        if qtype == "shr":
            return f["omegar"]
        else:
            raise ValueError("qtype = '%s' is not supported (yet)." % qtype)
    elif filename[-3:] == "mat":
        from scipy.io import loadmat
        W = np.squeeze(loadmat(filename)['W0'])
        return W


def load_basis(N):
    """
    Return a quantization basis from disk for band limit N.

    Parameters
    ----------
    N: int

    Returns
    -------
    basis: ndarray
    """
    basis = None

    # Look for a precomputed saved HDF5 basis
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

    return basis


def save_basis(basis):
    save_basis_hdf5(_basis_filename_default, basis)

