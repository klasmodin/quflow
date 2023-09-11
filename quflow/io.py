import numpy as np
from scipy.io import loadmat
from .utils import elm2ind, qtime2seconds
import os
import os.path
import h5py
import appdirs
import datetime
import time

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
_basis_file_version = 0.1


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
            assert (basis_set.attrs['QUFLOW_FILE_VERSION'] == _basis_file_version)
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
            basis_set.attrs['QUFLOW_FILE_VERSION'] = _basis_file_version
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


def determine_qtype(data, N=None):
    """
    Determine the qtype of state data and whether it is a sequence of states.

    Parameters
    ----------
    data: array_like
        Data to be determined.
    N: int (optional)
        Specify N is some cases where dim is not unique.

    Returns
    -------
    (qtype, issequence)
    """
    data = np.asarray(data)
    issequence = None
    qtype = None
    if data.ndim == 3:
        # Must be sequence of mat, fun, or img
        issequence = True
        if np.iscomplexobj(data):
            # Must be mat
            qtype = 'mat'
        elif np.isrealobj(data) and data.dtype == np.dtype('uint8'):
            # Must be img
            qtype = 'img'
        else:
            # Must be fun
            qtype = 'fun'
    elif data.ndim == 1:
        # Must be single shr or shc
        issequence = False
        if np.iscomplexobj(data):
            # Must be shc
            qtype = 'shc'
        elif np.isrealobj(data):
            # Must be shr
            qtype = 'shr'
    elif data.ndim == 2 and data.dtype == np.dtype('uint8'):
        # Must be single img
        issequence = False
        qtype = 'img'
    elif data.ndim == 2 and N is not None:  # This case requires the parameter N
        # Must be (i) sequence of shc or shr, or (ii) single mat, fun or img
        if data.shape == (N, N) and np.iscomplexobj(data):
            # Must be single mat
            issequence = False
            qtype = 'mat'
        elif data.shape[-1] == N**2:
            # Must be sequence of shr or shc
            issequence = True
            qtype = 'shr' if np.isrealobj(data) else 'shc'
        elif np.isrealobj(data):
            # Must be single fun
            issequence = False
            qtype = 'fun'

    return qtype, issequence


# ----------------------
# HIGHER LEVEL FUNCTIONS
# ----------------------

class QuData(object):

    def __init__(self, filename, datapath="/", cache_size=1, verbatim=False, max_wait=3600.0, qtype="shr"):
        self.filename = filename
        if len(datapath) == 0 or datapath[-1] != '/':
            datapath += '/'
        self.datapath = datapath
        self.verbatim = verbatim
        self.max_wait = max_wait  # Maximum time to wait before flushing
        self.last_write_time = time.time()

        # Set default values
        attrs = dict()
        attrs['qtime_last'] = 0.0
        attrs['qtime_start'] = 0.0
        attrs['W_cache'] = None
        attrs['qtime_cache'] = None
        attrs['cache_steps'] = 0
        attrs['total_steps'] = 0
        attrs['qtype'] = qtype
        assert cache_size >= 1, "Cache size must be larger than 1."
        attrs['cache_size'] = cache_size

        try:
            f = h5py.File(filename, "r")
        except IOError or KeyError:
            pass
        else:
            try:
                # attrs.update(f[datapath].attrs)
                attrs['qtime_last'] = f[datapath+'qtime'][-1]
                attrs['qtime_start'] = attrs['qtime_last']
                attrs['cache_steps'] = 0
                if self.verbatim:
                    print("Found data in file {} at qtime = {}.".format(self.filename, attrs['qtime_start']))
            except KeyError:
                pass
            else:
                if datapath+'W_cache' in f \
                        and datapath+'qtime_cache' in f \
                        and f[datapath+'W_cache'].shape[0] == cache_size:
                    attrs['W_cache'] = f[datapath+'W_cache'][...]
                    attrs['qtime_cache'] = f[datapath+'qtime_cache'][...]
                    attrs['cache_steps'] = f[datapath].attrs['cache_steps']
                else:
                    assert attrs['cache_steps'] == 0, "W_cache is not saved and cache_steps is still non-zero."
            f.close()

        # Set attributes
        for key in attrs:
            setattr(self, key, attrs[key])

    def __call__(self, W, inner_time, inner_steps=None, **kwargs):

        # Update total steps
        self.total_steps += 1

        # Update qtime
        qtime = inner_time
        if qtime + self.qtime_start < self.qtime_last:  # Hack to fix that time is always increasing. TODO: Fix this!
            self.qtime_start = self.qtime_last
            self.last_write_time = time.time()  # Reset this counter
        qtime += self.qtime_start

        # Update cache steps and initiate cache if needed
        if self.cache_size == 1:
            self.W_cache = np.array([W])
            # self.W_cache = W.reshape((1,)+W.shape)  # Might be a bug
            self.qtime_cache = np.array([qtime])
            self.cache_steps = 1
        else:
            if self.W_cache is None:
                self.W_cache = np.zeros((self.cache_size,) + W.shape, dtype=W.dtype)
                self.qtime_cache = np.zeros(self.cache_size, dtype=float)
                self.cache_steps = 0
            self.W_cache[self.cache_steps, :, :] = W
            self.qtime_cache[self.cache_steps] = qtime
            self.cache_steps += 1

        if self.verbatim:
            print("qtime = {}, time = {}, output steps = {}".format(qtime, qtime2seconds(qtime, W.shape[-1]),
                                                                    self.total_steps))
        now = time.time()
        if self.cache_steps == self.cache_size or now - self.last_write_time > self.max_wait:
            # Time to write to disk
            self.flush()

        # Save last output time
        self.qtime_last = qtime

    def __del__(self):
        self.flush()

    def flush(self):
        self.last_write_time = time.time()
        if self.cache_steps != 0:
            save(self.filename, self.W_cache[:self.cache_steps], qtime=self.qtime_cache[:self.cache_steps],
                 N=self.W_cache[0].shape[0], datapath=self.datapath, qtype=self.qtype)
            self.cache_steps = 0
            if self.verbatim:
                print("Cached data saved to file {}".format(self.filename))

    def _save_attrs(self, save_cache=False):
        with h5py.File(self.filename, "a") as f:
            attrs = dict()
            attrs['qtime_last'] = self.qtime_last
            attrs['qtime_start'] = self.qtime_start
            attrs['total_steps'] = self.total_steps
            attrs['cache_size'] = self.cache_size

            if save_cache:
                attrs['cache_steps'] = self.cache_steps
                if self.datapath+'W_cache' not in f:
                    f.create_dataset(self.datapath+'W_cache', self.W_cache.shape, dtype=self.W_cache.dtype)
                if self.datapath+'qtime_cache' not in f:
                    f.create_dataset(self.datapath+'qtime_cache', self.qtime_cache.shape, dtype=self.qtime_cache.dtype)
                f[self.datapath+'W_cache'][...] = self.W_cache

            f[self.datapath].attrs.update(attrs)


def save(filename, data, qtime=None, qstepsize=None, N=None, qtype="shr", datapath="/", attrs=None):
    """
    Save `data` in HDF5 file `filename`. The HDF5 file is created if
    it does not exist already.
    The data is stored in format `qtype` which defaults to `shr`.

    Parameters
    ----------
    filename: str
    data: ndarray
    qtime: float or 1D ndarray of floats
    qstepsize: float
        Time step in q-time. Only used if qtime is not specified.
    N: int
    qtype: str
        Either of 'shr' (default) or 'shc'
    datapath: str
        HDF5-file prefix of the datapath. Default: "".
    attrs: dict
        Attributes to add to the data file.
    """

    with h5py.File(filename, "a") as f:
        from .transforms import as_shr
        from .quantization import mat2shc
        from . import __version__

        # Process the datapath
        if len(datapath) == 0 or datapath[-1] != '/':
            datapath += '/'

        # Check for N attribute
        if 'N' in f[datapath].attrs:
            if N is not None:
                assert f[datapath].attrs['N'] == N, "Saved N and specified parameter N are different."
            N = f[datapath].attrs['N']

        # Try to determine qtype of data
        data_qtype, is_seq = determine_qtype(data, N=N)
        if data_qtype is None or is_seq is None:
            raise ValueError("Could not determine qtype of data. Try specifying the N parameter.")

        if qtype == "shr" or qtype == "shc":

            # Process data (convert to shr if needed)
            if not is_seq:
                # data = data.reshape((1,)+data.shape)  # Might be a bug
                data = np.array([data])
            if data_qtype == "shr" or data_qtype == "shc":
                omega = data
            else:
                omega = []
                for d in data:
                    if qtype == "shc":
                        if d.ndim == 2:
                            omega.append(mat2shc(d))
                        else:
                            raise ValueError("Something wrong with the input data.")
                    else:
                        omega.append(as_shr(d))
                omega = np.array(omega)

            # Check input
            if qtime is not None and qstepsize is not None:
                raise ValueError("Cannot specify both qtime and qstepsize.")
            if is_seq and qtime is not None:
                qtime = np.asarray(qtime)
                assert qtime.shape[0] == omega.shape[0], "Length of qtime and data are not corresponding."
            if not is_seq and np.isscalar(qtime):
                qtime = np.array([qtime])

            statepath = datapath + "state"
            timepath = datapath + "time"
            qtimepath = datapath + "qtime"

            # Check if state data should be allocated
            if statepath not in f:
                stateset = f.create_dataset(statepath, (0, omega.shape[-1]),
                                            dtype=omega.dtype,
                                            maxshape=(None, omega.shape[-1]),
                                            chunks=(1, omega.shape[-1])
                                            )
                if N is not None:
                    assert N == round(np.sqrt(omega.shape[-1]))
                else:
                    N = round(np.sqrt(omega.shape[-1]))
                f[datapath].attrs["N"] = N
                f[datapath].attrs["version"] = __version__
                f[datapath].attrs["created"] = datetime.datetime.now().isoformat()
                stateset.attrs["qtype"] = qtype
            elif f[statepath].shape[-1] != omega.shape[-1] or f[statepath].ndim != 2:
                raise ValueError("The file qtype does not seem to be correct.")

            # Check if qtime and time data should be allocated
            if qtimepath not in f:
                qtimeset = f.create_dataset(qtimepath, (0,),
                                            dtype=float,
                                            maxshape=(None,)
                                            )
                timeset = f.create_dataset(timepath, (0,),
                                           dtype=qtimeset.dtype,
                                           maxshape=(None,)
                                           )
                if qtimeset.shape[0] > 0:
                    qtimeset[:] = np.arange(f[qtimepath].shape[0], dtype=f[qtimepath].dtype)
                    timeset[:] = qtime2seconds(f[qtimepath][:], N)
            elif f[qtimepath].ndim != 1:
                raise ValueError("The qtimes data does not seem to be correct.")

            # Resize data sets
            f[statepath].resize(f[statepath].shape[0]+omega.shape[0], axis=0)
            f[qtimepath].resize(f[statepath].shape[0], axis=0)
            f[timepath].resize(f[statepath].shape[0], axis=0)

            # Assign newly allocated state data
            f[statepath][-omega.shape[0]:, :] = omega

            # Assign newly allocated qtime and time data
            if qtime is None:
                if qstepsize is None:
                    qstepsize = 1.0
                if f[qtimepath].shape[0] == omega.shape[0]:
                    qtime = qstepsize*np.arange(omega.shape[0], dtype=float)
                else:
                    qtime = f[qtimepath][-omega.shape[0]-1]+qstepsize*np.arange(1, 1+omega.shape[0], dtype=float)

            f[qtimepath][-omega.shape[0]:] = qtime
            f[timepath][-omega.shape[0]:] = qtime2seconds(f[qtimepath][-omega.shape[0]:], N)

            # Add attributs to stateset if there are any
            if attrs:
                f[datapath].attrs.update(attrs)

            # Update timestamp for modified data (this allows us to check if qtime data is up-to-date.
            modtime = datetime.datetime.now()
            f[statepath].attrs["modified"] = modtime.isoformat()
            f[qtimepath].attrs["modified"] = modtime.isoformat()
            f[timepath].attrs["modified"] = modtime.isoformat()
        else:
            raise ValueError("Format %s is not supported yet." % qtype)


def load(filename, datapath="state", qtype="auto"):
    """
    Load data saved in either MATLAB or HDF5 format.

    Parameters
    ----------
    filename: str
    qtype: str
        Either of 'auto' (default), 'shr', or 'shc'

    Returns
    -------
    data: h5py.Dataset or ndarray
    """
    if filename[-4:] == "hdf5":
        f = h5py.File(filename, "r")

        # TODO: This is a slight bug, since we're not checking what
        if qtype == 'auto':
            qtype = f[datapath].attrs["qtype"]
        if qtype == "shr" or qtype == "shc":
            if f[datapath].attrs["qtype"] == qtype:
                return f[datapath]
            else:
                raise ValueError("Not possible to convert hdf5 data between shr and shc.")
        else:
            raise ValueError("qtype = '%s' is not supported (yet)." % qtype)
    elif filename[-3:] == "mat":
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

