"""
Running simulations and storing them in HDF5 files.

:func:`solve` advances a vorticity matrix in time with an integrator (by
default :func:`quflow.isomp`) and calls callbacks at regular output times.
:class:`QuSimulation` is such a callback: it stores the states, in several
representations, together with time, step counts and logged quantities, in
an HDF5 file. It also stores the arguments of :func:`solve`, so that a
simulation can be continued with ``qf.solve(sim)``, also on another computer
(see :func:`create_runfile`).

Typical use::

    sim = qf.QuSimulation("mysim.hdf5", overwrite=True, state=W0,
                          loggers={'energy': qf.energy_euler})
    sim['dt'] = 0.1*qf.hbar(N)
    sim['simtime'] = 10.0
    sim['dt_out'] = 0.5
    qf.solve(sim)
    sim['fun', -1]   # last state as function values
"""
import numpy as np
import os
import os.path
import h5py
import pickle
import datetime
import time
import inspect
from .quantization import mat2shc
from .quantization import mat2shr
from .transforms import shc2fun, shr2fun
from .laplacian import solve_poisson
from .integrators import isomp
from .utils import elm2ind, berezin_multipliers
from .geometry import hbar
import warnings


# from quflow import solve_poisson

# ----------------
# Helper functions
# ----------------
def in_notebook():
    """
    Return True if running in a Jupyter (IPython kernel) notebook.

    Used to choose the notebook version of the progress bar.

    Returns
    -------
    bool
    """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


# ----------------
# GLOBAL VARIABLES
# ----------------

_default_qutypes = {'mat': None, 'fun': np.float32, 'funL2': np.float32}
_default_qutype2varname = {'mat': 'mat', 'fun': 'fun', 'shr': 'shr', 'shc': 'shc', 'funhalf': 'fun', 'funL2': 'funL2', 'funL2half': 'funL2'}
_pickled_argnames = ['qutypes', 'hamiltonian', 'forcing', 'integrator', 'callback', 'integrator_callback', 'strang_splitting']
_info_args = ['info']

# ----------------------
# QUSIMULATION CLASS DEF
# ----------------------

class QuSimulation(object):
    """
    An object from here can be used as a callback for the `quflow.solve` function.
    It represents output data from a simulation, stored on disk as an HDF5 file.
    The data may be stored in either of the following formats
    (specified by a string referred to as `qutype`):

    ``'mat'``
        Vorticity matrix, shape ``(..., N, N)``, complex.
    ``'shr'``
        Real spherical harmonics (:func:`quflow.mat2shr`), shape
        ``(..., N**2)``, real.
    ``'shc'``
        Complex spherical harmonics (:func:`quflow.mat2shc`), shape
        ``(..., N**2)``, complex.
    ``'fun'``
        Function values on the grid of :func:`quflow.sphgrid`, shape
        ``(..., N, 2*N-1)``, computed with the Berezin multipliers (the
        default of :func:`quflow.shr2fun`). Real or complex depending on the
        dtype.
    ``'funL2'``
        As ``'fun'``, but without the Berezin multipliers.
    ``'funhalf'``, ``'funL2half'``
        As ``'fun'`` and ``'funL2'``, but from the coefficients with
        ``el < N//2`` only, on the grid of size ``(N//2, 2*(N//2)-1)``.
        Stored under the names ``'fun'`` and ``'funL2'``.

    The leading ``...`` dimensions are those of a stacked state. Notice that
    the same object can hold data for several `qutype`.

    The file contains one dataset per stored field, with the time index as
    first dimension: one per qutype, ``'time'``, ``'step'``, one per logger,
    the solver statistics ``'tol_auto'``, ``'iterations'`` and
    ``'number_of_maxit'``, and any extra fields given at creation. Arguments
    for :func:`solve` set with ``sim[name] = value`` are stored as attributes
    of the group ``<datapath>args/``.

    Parameters
    ----------
    filename : str
        Name of the HDF5 file.
    qutypes : dict, optional
        Representations to store, as a dict ``{qutype: dtype}``; a dtype of
        None means the dtype of the state. Default is
        ``{'mat': None, 'fun': np.float32, 'funL2': np.float32}``. Only for a
        new file.
    datapath : str, optional
        HDF5 group to store the simulation in; must end with ``"/"``.
        Default is ``"/"``.
    overwrite : bool, optional
        Whether to overwrite an existing file. Default is False: an existing
        file is opened and continued.
    loggers : dict, optional
        Functions ``{name: logger}``, where ``logger(W)`` returns a scalar or
        array that is stored at every output time (e.g.
        ``{'energy': qf.energy_euler}``). The loggers are pickled in the file;
        those that cannot be pickled (e.g. local functions) must be passed
        again when reopening the file.
    state : ndarray, shape (N, N) or (k, N, N), complex, optional
        Initial state. Required for a new file; not allowed for an existing
        one.
    time : float, optional
        Initial time. Default is 0.0.
    **kwargs
        Extra fields to store at every output time, with their initial
        values (scalars or arrays).

    Raises
    ------
    ValueError
        If `datapath` does not end with ``"/"``, if `state` is missing for a
        new file, if `state` or `qutypes` are given for an existing file, or
        if both a full and a half version of the same function output are
        requested.

    Notes
    -----
    The file is opened only for the duration of each operation.

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> import quflow as qf
    >>> N = 8
    >>> W0 = qf.shr2mat(np.r_[0.0, np.random.default_rng(0).standard_normal(N**2 - 1)])
    >>> filename = os.path.join(tempfile.mkdtemp(), "example.hdf5")
    >>> sim = qf.QuSimulation(filename, state=W0, loggers={'energy': qf.energy_euler})
    >>> sim['dt'] = 0.1*qf.hbar(N)
    >>> sim['steps'] = 10
    >>> sim['dt_out'] = 0.5*qf.hbar(N)
    >>> qf.solve(sim, progress_bar=False)
    >>> sim['time'].shape, sim['fun'].shape
    ((3,), (3, 8, 15))
    """

    def __init__(self,
                 filename: str,
                 qutypes: dict = None,
                 datapath: str = "/",
                 overwrite: bool = False,
                 loggers: dict = None,
                 state: np.ndarray = None,
                 time=None,
                 **kwargs): 
        """
        Create a new simulation file, or open an existing one.

        See the class docstring for the parameters.

        Notes
        -----
        Physical units, for reference:

        * velocity v [m/s]
        * vorticity omega = curl v [1/s]
        * stream function psi = Delta^{-1} omega [m^2/s]
        * energy = integral of omega*psi [m^4/s^2]
        * enstrophy = integral of omega^2 [m^2/s^2]
        * sqrt(energy)/enstrophy [s]
        """
        from . import __version__

        self.filename = filename
        if datapath[-1] != "/":
            raise ValueError("Datapath must end with /")
        self.datapath = datapath
        self.fieldnames = dict()
        self.loggers = loggers if loggers is not None else dict()
        self._isopen = False

        if not os.path.exists(filename) or overwrite:
            if state is None:
                raise ValueError("At least `state` must be provided to initialize a QuSimulation.")

            if qutypes is None:
                self.qutypes = _default_qutypes
            else:
                self.qutypes = qutypes
            
            if 'fun' in self.qutypes and 'funhalf' in self.qutypes:
                raise ValueError("Cannot have both fun and funhalf outputs.")

            if 'funL2' in self.qutypes and 'funL2half' in self.qutypes:
                raise ValueError("Cannot have both funL2 and funL2half outputs.")

            # Create or overwrite file
            try:
                f = self._open("w")
            # with h5py.File(self.filename, "w") as f:
                if self.datapath != "/":
                    f.create_group(self.datapath)
                f[self.datapath].attrs["version"] = __version__
                f[self.datapath].attrs["created"] = datetime.datetime.now().isoformat()
                f[self.datapath].attrs["qutypes"] = np.array([pickle.dumps(self.qutypes)])
                try:
                    myp = pickle.dumps(self.loggers)
                except (AttributeError, pickle.PicklingError) as e:
                    # AttributeError up to Python 3.13, PicklingError from 3.14
                    warnings.warn("Loggers could not be pickled and are not saved in {} ({}). "
                                  "Pass them again via `loggers` when reopening the simulation."
                                  .format(self.filename, e))
                else:
                    f[self.datapath].attrs['loggers'] = np.array([myp])
                self.args_datapath = self.datapath + "args/"
                f.create_group(self.args_datapath)
            finally:
                self._close()

            # Add fields
            self.initialize_field(W=state, time=time if time is not None else 0.0, **kwargs)

        else:
            try:
                f = self._open("r")
            # with h5py.File(self.filename, "r") as f:
                # if "prerun" in f[self.datapath].attrs:
                    # exec(f[self.datapath].attrs["prerun"], globals())
                self.qutypes = pickle.loads(f[self.datapath].attrs["qutypes"][0])
                if "N" in f[self.datapath].attrs and state is not None:
                    raise ValueError(self.filename + " has already been initialized with W.")
                if 'loggers' in f[self.datapath].attrs:
                    myp = f[self.datapath].attrs['loggers'][0]
                    self.loggers = pickle.loads(myp)
                elif loggers is None:
                    # Loggers could not be pickled when the file was created
                    warnings.warn("No loggers are stored in {}. Pass them via `loggers` "
                                  "to continue logging.".format(self.filename))
                self.args_datapath = self.datapath + "args/"
                if qutypes is not None:
                    raise ValueError(self.filename + " has already been initialized with qutypes.")
            finally:
                self._close()
        # Update fieldnames
        self._update_fieldnames()


    def _open(self, mode="r"):
        "Open HDF5 file."
        if not self._isopen:
            self._hdf5file = h5py.File(self.filename, mode)
            self._isopen = True
        elif mode != "r" and self._hdf5file.mode == "r":
            self._close()
            return self._open(mode)
        return self._hdf5file
    

    def _close(self):
        "Close HDF5 file."
        if self._isopen:
            self._hdf5file.close()
            self._isopen = False


    def __enter__(self):
        """
        Open the file for reading, for use in a ``with`` block.

        Note that most methods close the file when they finish.
        """
        self._open("r")
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        """Close the file; exceptions are printed and propagated."""
        self._close()
        if exc_type:
            print(f'Exception type: {exc_type}')
            print(f'Exception value: {exc_value}')
        # Return False to propagate the exception, True to suppress it
        return False    


    def __setitem__(self, name, value):
        """
        Store an argument for :func:`solve` (or other metadata) in the file.

        * ``'hamiltonian'``, ``'forcing'``, ``'integrator'``, ``'callback'``,
          ``'integrator_callback'``, ``'strang_splitting'`` and
          ``'qutypes'`` are pickled. Functions are pickled by reference
          (module and name); functions that cannot be pickled (e.g. local
          functions or lambdas) are stored by name only (see
          :meth:`__getitem__`).
        * ``'prerun'`` is a string of Python code (e.g. definitions of the
          functions above), used by :func:`create_runfile`. It is stored,
          not executed.
        * ``'info'`` is stored as an attribute of the simulation group.
        * Anything else (e.g. ``'dt'``, ``'simtime'``, ``'dt_out'``,
          ``'tol'``) is stored as an attribute of the ``args`` group.

        Setting a value to None removes it.

        Parameters
        ----------
        name : str
        value : object
        """
        try:
            f = self._open("r+")
        # with h5py.File(self.filename, "r+") as f:
            if name in _pickled_argnames:
                if value is None:
                    f[self.args_datapath].attrs.pop(name)
                else:
                    try:
                        myp = pickle.dumps(value)
                    except (AttributeError, pickle.PicklingError):
                        myp = value.__name__
                        f[self.args_datapath].attrs[name] = myp
                    else:
                        f[self.args_datapath].attrs[name] = np.array([myp])
            elif name == "prerun":
                prerun = "\n".join([l for l in value.strip().split("\n") if "In[len" not in l])
                f[self.datapath].attrs[name] = prerun
                value = prerun
                # exec(prerun, globals())
            elif name in _info_args:
                if value is None:
                    f[self.datapath].attrs.pop(name)
                else:    
                    f[self.datapath].attrs[name] = value
            else:
                if value is None:
                    f[self.args_datapath].attrs.pop(name)
                else:    
                    f[self.args_datapath].attrs[name] = value
        finally:
            self._close()

    def __getitem__(self, name):
        """
        Return stored data: a dataset, an argument, or an attribute.

        ``sim[name]`` returns the whole dataset `name` (e.g. ``'time'``,
        ``'fun'``, a logger) as an array, or a stored argument or attribute.
        ``sim[name, index]`` returns part of a dataset, e.g.
        ``sim['mat', -1]`` (the last state) or ``sim['fun', :, 0]``.
        ``sim[index]`` is short for ``sim['mat', index]``.

        Functions stored by name only (see :meth:`__setitem__`) are evaluated
        by name in the namespace of :mod:`quflow.simulation`, which only works
        for names available there (such as ``solve_poisson`` and ``isomp``).

        Raises
        ------
        KeyError
            If there is no such dataset, argument or attribute.
        """
        ind = None
        if isinstance(name, tuple):
            if isinstance(name[0], str):
                if len(name) > 2:
                    ind = name[1:]
                else:
                    ind = name[1]
                name = name[0]
        if not isinstance(name, str):
            # Assume it is an index
            ind = name 
            name = "mat"
        try:
            f = self._open("r")
        # with h5py.File(self.filename, 'r') as f:
            if self.datapath + name in f:
                if ind is not None:
                    value = f[self.datapath + name][ind]
                else:
                    value = f[self.datapath + name][:]
            elif name in f[self.args_datapath].attrs:
                if name in _pickled_argnames:
                    if isinstance(f[self.args_datapath].attrs[name], str):
                        value = eval(f[self.args_datapath].attrs[name])
                    else:
                        myp = f[self.args_datapath].attrs[name][0]
                        value = pickle.loads(myp)
                else:
                    value = f[self.args_datapath].attrs[name]
            elif name in f[self.datapath].attrs:
                if name == "qutypes":
                    myp = f[self.datapath].attrs[name][0]
                    value = pickle.loads(myp)
                else:
                    value = f[self.datapath].attrs[name]
            else:
                raise KeyError("There is no dataset or attribute '{}'.".format(name))
        finally:
            self._close()
        return value
    
    def args(self):
        """
        Iterate over the stored arguments for :func:`solve`.

        Yields
        ------
        (name, value) : tuple
        """
        try:
            f = self._open("r")
        # with h5py.File(self.filename, 'r') as f:
            for name in f[self.args_datapath].attrs:
                yield name, self[name]
        finally:
            self._close()

    def qutypes_iterator(self, W, qutype2varname=None):
        """
        Convert a state to each stored representation.

        Parameters
        ----------
        W : ndarray, shape (N, N) or (k, N, N), complex
            State.
        qutype2varname : dict, optional
            Dataset name for each qutype. Default maps ``'funhalf'`` and
            ``'funL2half'`` to ``'fun'`` and ``'funL2'``, and other qutypes
            to themselves.

        Yields
        ------
        (varname, arr, qutype) : tuple
            Dataset name, converted data and qutype.

        Notes
        -----
        A complex function output (``'fun'`` with a complex dtype) is only
        computed correctly if ``'shc'`` comes before it in `qutypes`;
        otherwise real coefficients (:func:`quflow.mat2shr`) are transformed
        as complex ones.
        """
        N = W.shape[-1]
        if qutype2varname is None:
            qutype2varname = _default_qutype2varname
        for qutype, dtype in self.qutypes.items():
            isreal = np.isrealobj(np.array([], dtype=dtype))
            if qutype == 'mat':
                if dtype is None:
                    dtype = W.dtype
                arr = W.astype(dtype)
            elif qutype == 'shr':
                if dtype is None:
                    dtype = W.ravel()[:1].real.dtype
                omegar = []
                for Wi in W.reshape((-1, N, N)):
                    omegar.append(mat2shr(Wi))
                omegar = np.squeeze(np.array(omegar))
                arr = omegar.astype(dtype)
            elif qutype == 'shc':
                if dtype is None:
                    dtype = W.dtype
                omegac = []
                for Wi in W.reshape((-1, N, N)):
                    omegac.append(mat2shc(Wi))
                omegac = np.squeeze(np.array(omegac))
                arr = omegac.astype(dtype)
            elif 'fun' in qutype:
                if isreal:
                    try:
                        omega = omegar
                    except NameError:
                        omegar = []
                        for Wi in W.reshape((-1, N, N)):
                            omegar.append(mat2shr(Wi))
                        omegar = np.squeeze(np.array(omegar))
                        omega = omegar
                else:
                    try:
                        omega = omegac
                    except NameError:
                        omegac = []
                        for Wi in W.reshape((-1, N, N)):
                            omegac.append(mat2shr(Wi))
                        omegac = np.squeeze(np.array(omegac))
                        omega = omegac
                arr = []
                for omegai in omega.reshape((-1, omega.shape[-1])):
                    sh2fun = shr2fun if isreal else shc2fun
                    sh2fun_args = dict()
                    if 'half' in qutype:
                        omegai = omegai[...,:(N//2)**2]
                    if 'funL2' in qutype:
                        sh2fun_args['berezin'] = False
                        # omegai = omegai*berezin_multipliers(N=N)[:omegai.shape[-1]]
                    arr.append(sh2fun(omegai, **sh2fun_args))
                arr = np.squeeze(np.array(arr, dtype=dtype))

            yield qutype2varname[qutype], arr, qutype

    def _update_fieldnames(self):
        """Update ``self.fieldnames``, a dict ``{name: (shape, dtype)}`` of the datasets."""
        try:
            f = self._open("r")
        # with h5py.File(self.filename, 'r') as f:
            for name in f[self.datapath].keys():
                dataset = f[self.datapath + name]
                if isinstance(dataset, h5py.Dataset):
                    self.fieldnames.update({name: (dataset.shape, dataset.dtype)})
        finally:
            self._close()

    def initialize_field(self, W, time=0.0, **kwargs):
        """
        Create the datasets of a new simulation file, with the initial values.

        Called by the constructor for a new file.

        Parameters
        ----------
        W : ndarray, shape (N, N) or (k, N, N), complex, or None
            Initial state. If None, no state datasets are created.
        time : float, optional
            Initial time. Default is 0.0.
        **kwargs
            Extra fields and their initial values, which must be scalars or
            numpy arrays. Values of other types are not rejected, but give
            wrong data. The fields ``'tol_auto'``, ``'iterations'`` and
            ``'number_of_maxit'`` are always created (initially 0.0).

        Raises
        ------
        ValueError
            If an extra field is named ``'time'`` or ``'step'``.
        """
        try:
            f = self._open("r+")
        #     f = h5py.File(self.filename, "r+")
        # except IOError:
        #     raise IOError("Error while trying to write to file {}.".format(self.filename))
        # else:
            if W is not None:
                N = W.shape[-1]

                # Create datasets for all the qutype representations
                for varname, arr, qutype in self.qutypes_iterator(W):
                    varset = f.create_dataset(self.datapath + varname, (1,) + arr.shape,
                                              dtype=arr.dtype,
                                              maxshape=(None,) + arr.shape,
                                              chunks=(1,) + arr.shape
                                              )
                    varset[0, ...] = arr
                    varset.attrs["qutype"] = qutype

                # Add attributes
                f[self.datapath].attrs["N"] = N

            # Create datasets for time
            timeset = f.create_dataset(self.datapath + "time", (1,),
                                       dtype=np.float64,
                                       maxshape=(None,)
                                       )
            timeset[0] = time

            # Create datasets for step
            stepset = f.create_dataset(self.datapath + "step", (1,),
                                       dtype=int,
                                       maxshape=(None,)
                                       )
            stepset[0] = 0

            # Create datasets for loggers
            for name, logger in self.loggers.items():
                value = logger(W)
                if np.isscalar(value):
                    arr = np.array(value)
                elif isinstance(value, np.ndarray):
                    arr = value
                else:
                    ValueError("Data {} is not a scalar or a numpy array.".format(value))
                varset = f.create_dataset(self.datapath + name, (1,) + arr.shape,
                                          dtype=arr.dtype,
                                          maxshape=(None,) + arr.shape
                                          )
                varset[0, ...] = arr

            # Add some default fields (this is temporary until a better integration class)
            for name in ['tol_auto', 'iterations', 'number_of_maxit']:
                if name not in kwargs:
                    kwargs[name] = 0.0

            # Create datasets for kwargs
            for name, value in kwargs.items():
                if name in ("time", "step"):
                    raise ValueError("{} is not a valid field name.".format(name))
                if np.isscalar(value):
                    arr = np.array(value)
                elif isinstance(value, np.ndarray):
                    arr = value
                else:
                    ValueError("Data {} is not a scalar or a numpy array.".format(value))
                varset = f.create_dataset(self.datapath + name, (1,) + arr.shape,
                                          dtype=arr.dtype,
                                          maxshape=(None,) + arr.shape
                                          )
                varset[0, ...] = arr

        finally:
            self._close()

    def __call__(self, W, delta_time, delta_steps=1, **kwargs):
        """
        Append a state to the file (the callback interface of :func:`solve`).

        Stores `W` in each qutype, the time and step count (incremented by
        `delta_time` and `delta_steps`), the values of the loggers, and the
        values in `kwargs` for which a dataset exists (e.g. the solver
        statistics); other keyword arguments are ignored.

        Parameters
        ----------
        W: ndarray, shape (N, N) or (k, N, N), complex
            State.
        delta_time: float
            Time since the previous stored state.
        delta_steps: int
            Number of steps since the previous stored state. Default is 1.
        kwargs: other variables to save
        """
        try:
            f = self._open("r+")
        # with h5py.File(self.filename, "r+") as f:

            # Update state sets
            for varname, arr, qutype in self.qutypes_iterator(W):
                varset = f[self.datapath + varname]
                varset.resize(varset.shape[0]+1, axis=0)
                varset[-1, ...] = arr

            # Update time
            timeset = f[self.datapath + "time"]
            timeset.resize(timeset.shape[0]+1, axis=0)
            timeset[-1] = timeset[-2] + delta_time

            # Update step
            stepset = f[self.datapath + "step"]
            stepset.resize(stepset.shape[0]+1, axis=0)
            stepset[-1] = stepset[-2] + delta_steps

            # Update other fields
            for varname, value in kwargs.items():
                if self.datapath + varname in f and varname not in self.loggers:
                    varset = f[self.datapath + varname]
                    varset.resize(varset.shape[0]+1, axis=0)
                    varset[-1, ...] = value

            # Update logger fields
            for name, logger in self.loggers.items():
                varset = f[self.datapath + name]
                varset.resize(varset.shape[0]+1, axis=0)
                value = logger(W)
                varset[-1, ...] = value
        
        finally:
            self._close()


# -------------------------
# RUNFILE CREATION FUNCTION
# -------------------------

def create_runfile(sim, runfilename: str = None):
    """
    Create cluster runfile from QuSimulation object.

    Writes a Python script that loads the simulation file and continues it
    with ``qf.solve``, and then creates an animation of the ``'fun'`` data.
    The code stored under ``sim['prerun']`` is inserted near the top of the
    script, so that functions it defines (e.g. a custom Hamiltonian) are
    available when the stored arguments are unpickled. The script accepts
    command line options ``--filename``, ``--simtime``, ``--tol``,
    ``--compsum``, ``--animate`` (only animate) and ``--simulate`` (only
    simulate), and uses the CUDA implementation if :mod:`cupy` and a GPU are
    available.

    Parameters
    ----------
    sim: QuSimulation or str
        If str, interpret as filename of QuSimulation file. The simulation
        must have a ``'prerun'`` entry.
    runfilename: str
        Name of output file. Should end with .py. Default is to
        base the name of the simfile name (``<name>_runfile.py``).

    Notes
    -----
    In the generated script, the default value of ``--filename`` is the
    file name without quotes, so the script fails with a `NameError` when
    run; and the GPU branch refers to an undefined name ``complex_type``.
    """
    if not isinstance(sim, QuSimulation):
        if isinstance(sim, str):
            # Assume it is the filename
            sim = QuSimulation(sim)
    
    filestr = """

import numpy as np
import quflow as qf
import argparse

try:
    import cupy as cp
except ImportError:
    cuda_available = False
else:
    cuda_available = True if cp.is_available() else False

if cuda_available:
    from quflow.experimental.cuda import DiagTriDiagOp
    from quflow.experimental.isospectral_cuda import IsompCUDA
else:
    print("Running on CPU")

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="HDF5 simfile", type=str, default={}) 
parser.add_argument("-a", "--animate", help="Only create animation.", action="store_true")
parser.add_argument("-s", "--simulate", help="Only simulate.", action="store_true")
parser.add_argument("-t", "--simtime", help="Total simulation time.", type=float)
parser.add_argument("--compsum", help="Use compensated summation.", action="store_true")
parser.add_argument("--tol", help="Tolerance for solver iterations.", type=float)
args = parser.parse_args()

filename = args.filename
if filename is None:
    raise ValueError("filename must be specified (valid hdf5 file)")

# ---------- Externally defined code ----------

{}

# ---------------------------------------------

# Load simulation from file
mysim = qf.QuSimulation(filename)

solve_kwargs = dict()

if args.simtime is not None:
    solve_kwargs['simtime'] = np.float64(args.simtime)

if args.tol is not None:
    solve_kwargs['tol'] = np.float64(args.tol)

if args.compsum:
    solve_kwargs['compsum'] = True

if cuda_available and mysim['hamiltonian'] is qf.solve_poisson and mysim['integrator'] is qf.isomp:
    print("Running on GPU")
    W0 = mysim['mat', 0] 
    N = W0.shape[-1]
    isomp = IsompCUDA(N, dtype=W0.dtype)
    solve_poisson = DiagTriDiagOp(N, dtype=np.dtype(complex_type))

    solve_kwargs['hamiltonian'] = solve_poisson
    solve_kwargs['integrator'] = isomp

# Run simulation
if not args.animate:
    qf.solve(mysim, **solve_kwargs)

# Create animation
if not args.simulate:
    qf.create_animation(args.filename.replace(".hdf5", ".mp4"), mysim['fun'])


""".format(os.path.basename(sim.filename), sim['prerun'])
    if runfilename is None:
        runfilename = sim.filename.replace(".hdf5", "_runfile.py")
    with open(runfilename, "w") as text_file:
        text_file.write(filestr)
    

# ----------------------
# SOLVE FUNCTION DEF
# ----------------------

def solve(W,
          dt=None, 
          stepsize=None, 
          steps=None, 
          simtime=None,
          endtime=None,
          steps_out=None, 
          dt_out=None,
          integrator=None,
          callback=None, 
          callback_kwargs=None,
          integrator_callback=None,
          progress_bar=True, 
          progress_file=None, 
          **kwargs):
    """
    High-level solve function.

    Advances `W` by repeated calls of `integrator`, in chunks of
    `steps_out` steps, and calls the callbacks after each chunk.

    If `W` is a :class:`QuSimulation`, the simulation is continued from its
    last stored state and time, the simulation object is added to the
    callbacks, and the arguments stored in it (``sim[name] = value``) are
    used for the parameters not given explicitly. Stored arguments not
    listed below (e.g. ``'hamiltonian'``, ``'forcing'``, ``'tol'``) are
    passed to the integrator. A stored ``'callback'`` is used as
    `integrator_callback`; the deprecated names ``'inner_steps'`` and
    ``'inner_time'`` are accepted for `steps_out` and `dt_out`.

    Parameters
    ----------
    W: ndarray(shape=(N, N) or (k, N, N), dtype=complex) or QuSimulation
        Initial vorticity matrix (updated in place by in-place integrators
        such as :func:`quflow.isomp`), or a simulation to continue.
    dt: None or float
        Time step in seconds.
    stepsize: None or float
        Stepsize parameter, related to the actual time step length
        by `dt = hbar * stepsize`.
        Not used when `dt` is specified.
    steps: None or int
        Total number of steps to take.
        Only one of `steps`, `simtime`, or `endtime` should be specified
        (a warning is issued otherwise). If several are given, `endtime`
        takes precedence over `simtime`, which takes precedence over `steps`.
    simtime: None or float
        Total simulation time in seconds. The number of steps is
        ``round(simtime/dt)``.
        Only one of `steps`, `simtime`, or `endtime` should be specified.
    endtime: None or float
        End simulation time in seconds.
        Only one of `steps`, `simtime`, or `endtime` should be specified.
    steps_out: None or int
        Number of steps taken between each callback. Default is 100 (and at
        most `steps`).
    dt_out: None or float
        Approximate time in seconds between each callback; ``steps_out =
        round(dt_out/dt)``.
        Not used when `steps_out` is specified.
    integrator: callable(W, dt, steps, **kwargs)
        Integration method to carry out the steps. Default is
        :func:`quflow.isomp`. If it accepts an argument ``stats``, the
        statistics it reports are passed on to the callbacks.
    callback: callable or tuple of callables
        Called as ``callback(W, delta_time=..., delta_steps=...,
        **callback_kwargs)`` after every chunk of steps (as
        :meth:`QuSimulation.__call__`). It is not evaluated at the initial
        time.
    callback_kwargs: dict
        Extra keyword arguments to send to callback at each output step.
        The integrator statistics are added to it.
    integrator_callback: callable
        Callback for integrator (called once per step), passed as its
        `callback` argument.
    progress_bar: bool
        Show progress bar (default: True). Requires :mod:`tqdm`.
    progress_file: TextIOWrapper or None
        File to write progress to (default: None)
    **kwargs: dict
        Extra keyword arguments to send to integrator at each step, e.g.
        ``hamiltonian`` (default :func:`quflow.solve_poisson`), ``forcing``
        or ``tol``. The argument ``time`` sets the initial time (default 0.0)
        and is passed to the integrator, advanced after each chunk.

    Returns
    -------
    None
        The final state is available through `W` (for in-place integrators)
        and the callbacks.

    Raises
    ------
    ValueError
        If neither `dt` nor `stepsize` is given, or if `endtime` is earlier
        than the current time.

    Notes
    -----
    If none of `steps`, `simtime` and `endtime` is given, a warning is issued
    and the function then fails with a `TypeError`. A ``'progress_bar'``
    argument stored in a :class:`QuSimulation` has no effect.
    """

    time = 0.0 if 'time' not in kwargs else kwargs['time']

    # Preprocess attributes from QuSimulation object
    if isinstance(W, QuSimulation):
        sim = W
        W = sim['mat', -1]
        time = sim['time', -1]
        if callback is None:
            callback = sim
        elif isinstance(callback, tuple):
            callback += (sim,)
        else:
            callback = (callback, sim)
        for name, value in sim.args():
            if name == 'dt':
                if dt is None:
                    dt = value
            elif name == 'stepsize':
                if stepsize is None:
                    stepsize = value
            elif name == 'steps':
                if steps is None:
                    steps = value
            elif name == 'simtime':
                if simtime is None:
                    simtime = value
            elif name == 'endtime':
                if endtime is None:
                    endtime = value
            elif name == 'steps_out':
                if steps_out is None:
                    steps_out = value
            elif name == 'inner_steps':
                if steps_out is None:
                    steps_out = value
            elif name == 'dt_out':
                if dt_out is None:
                    dt_out = value
            elif name == 'inner_time':
                if dt_out is None:
                    dt_out = value
            elif name == 'integrator':
                if integrator is None:
                    integrator = value
            elif (name == 'integrator_callback' or name == 'callback'):
                if integrator_callback is None:
                    integrator_callback = value
            elif name == 'callback_kwargs':
                if callback_kwargs is None:
                    callback_kwargs = value
            elif name == 'progress_bar':
                if progress_bar is None:
                    progress_bar = value
            elif name == 'progress_file':
                if progress_file is None:
                    progress_file = value
            else:
                # Add to kwargs
                if name not in kwargs:
                    kwargs[name] = value

    # Size of state matrix
    N = W.shape[-1]

    # Set dt if not selected
    if dt is None:
        if stepsize is None:
            raise ValueError("Either `dt` or `stepsize` must be specified.")
        dt = stepsize*hbar(N=N)

    # Set default integrator if needed
    if integrator is None:
        integrator = isomp

    # Set default hamiltonian if needed
    integrator_kwargs = kwargs
    integrator_kwargs['time'] = time
    if 'hamiltonian' not in integrator_kwargs:
        integrator_kwargs['hamiltonian'] = solve_poisson
    if 'stats' in inspect.getfullargspec(integrator).args:
        integrator_kwargs['stats'] = {'iterations':0.0}
    if integrator_callback is not None:
        integrator_kwargs['callback'] = integrator_callback

    # Determine steps
    if np.array([0 if x is None else 1 for x in [steps, simtime, endtime]]).sum() != 1:
        warnings.warn("One, and only one, of `steps`, `simtime`, or `endtime` should be specified.")
    if endtime is not None:
        if endtime < time:
            raise ValueError("Specified `endtime`={} is smaller than current `time`={}.".format(endtime, time))
        simtime = endtime - time
    if simtime is not None:
        # qtime = seconds2qtime(simtime, N)
        # steps = round(qtime / np.abs(stepsize))
        steps = round(simtime / np.abs(dt))
    if callback is not None and not isinstance(callback, tuple):
        callback = (callback,)
    if callback_kwargs is None:
        callback_kwargs = dict()

    # Determine inner_steps
    if np.array([0 if x is None else 1 for x in [steps_out, dt_out]]).sum() == 0:
        steps_out = 100  # Default value of inner_steps
    elif steps_out is None:
        if dt_out is not None:
            # inner_steps = round(seconds2qtime(inner_time, N) / np.abs(stepsize))
            steps_out = round(dt_out / np.abs(dt))

    # Check if inner_steps is too large
    if steps_out > steps:
        steps_out = steps

    # Create progressbar
    if progress_bar:
        try:
            if progress_file is None:
                if 'verbatim' in integrator_kwargs and integrator_kwargs['verbatim']:
                    progress_bar = False
                else:
                    if in_notebook():
                        from tqdm.notebook import tqdm
                    else:
                        from tqdm import tqdm
                    pbar = tqdm(total=steps, unit=' steps')
            else:
                from tqdm import tqdm
                pbar = tqdm(total=steps, unit=' steps', file=progress_file, ascii=True, mininterval=10.0)
        except ModuleNotFoundError:
            progress_bar = False

    # Main simulation loop
    for k in range(0, steps, steps_out):

        if k+steps_out > steps:
            no_steps = steps-k
        else:
            no_steps = steps_out
        W = integrator(W, dt, steps=no_steps, **integrator_kwargs)
        # delta_time = qtime2seconds(no_steps*stepsize, N=N)
        delta_time = no_steps*dt
        integrator_kwargs['time'] += delta_time
        if progress_bar:
            pbar.update(no_steps)
        if callback is not None:
            for cfun in callback:
                if 'stats' in integrator_kwargs:
                    callback_kwargs.update(integrator_kwargs['stats'])
                cfun(W, delta_time=delta_time, delta_steps=no_steps, **callback_kwargs)

    # Close progressbar
    if progress_bar:
        pbar.close()


