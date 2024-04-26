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
from .utils import elm2ind, qtime2seconds, seconds2qtime


# from quflow import solve_poisson

# ----------------
# GLOBAL VARIABLES
# ----------------

_default_qutypes = {'mat': None, 'fun': np.float32}
_default_qutype2varname = {'mat': 'mat', 'fun': 'fun', 'shr': 'shr', 'shc': 'shc'}
_pickled_argnames = ['qutypes', 'hamiltonian', 'forcing', 'integrator']
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

    'mat'
    Vorticity matrix.
    shape = (..., N, N)
    dtype = complex

    'shr'
    Real spherical harmonics.
    shape = (..., N**2)
    dtype = real

    'shc'
    Spherical harmonics
    shape = (..., N**2)
    dtype = complex

    'fun'
    Function values in spherical coordinates
    shape = (..., 2*N, N)
    dtype = real or complex

    Notice that the same object can hold data for several `qutype`.
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

        # v [m/s]
        # omega = curl v = [1/s]
        # psi = Delta^{-1}omega = [m^2/s]
        # energy = omega psi dx = [m^4/s^2]
        # enstrophy = omega^2 dx = [m^2/s^2]
        # \sqrt{energy}/enstrophy = [s]

        Parameters
        ----------
        filename
        filemode
        qtype
        """
        from . import __version__

        self.filename = filename
        if datapath[-1] != "/":
            raise ValueError("Datapath must end with /")
        self.datapath = datapath
        self.fieldnames = dict()
        self.loggers = loggers if loggers is not None else dict()

        if not os.path.exists(filename) or overwrite:
            if state is None:
                raise ValueError("At least `state` must be provided to initialize a QuSimulation.")

            if qutypes is None:
                self.qutypes = _default_qutypes
            else:
                self.qutypes = qutypes

            # Create or overwrite file
            with h5py.File(self.filename, "w") as f:
                if self.datapath != "/":
                    f.create_group(self.datapath)
                f[self.datapath].attrs["version"] = __version__
                f[self.datapath].attrs["created"] = datetime.datetime.now().isoformat()
                f[self.datapath].attrs["qutypes"] = np.array([pickle.dumps(self.qutypes)])
                try:
                    myp = pickle.dumps(self.loggers)
                except AttributeError:
                    pass
                else:
                    f[self.datapath].attrs['loggers'] = np.array([myp])
                self.args_datapath = self.datapath + "args/"
                f.create_group(self.args_datapath)

            # Add fields
            self.initialize_field(W=state, time=time if time is not None else 0.0, **kwargs)

        else:
            with h5py.File(self.filename, "r") as f:
                if "prerun" in f[self.datapath].attrs:
                    exec(f[self.datapath].attrs["prerun"], globals())
                self.qutypes = pickle.loads(f[self.datapath].attrs["qutypes"][0])
                if "N" in f[self.datapath].attrs and state is not None:
                    raise ValueError(self.filename + " has already been initialized with W.")
                myp = f[self.datapath].attrs['loggers'][0]
                self.loggers = pickle.loads(myp)
                self.args_datapath = self.datapath + "args/"
                if qutypes is not None:
                    raise ValueError(self.filename + " has already been initialized with qutypes.")
        
        # Update fieldnames
        self._update_fieldnames()

    def __setitem__(self, name, value):
        with h5py.File(self.filename, "r+") as f:
            if name in _pickled_argnames:
                try:
                    myp = pickle.dumps(value)
                except AttributeError:
                    myp = value.__name__
                    f[self.args_datapath].attrs[name] = myp
                else:
                    f[self.args_datapath].attrs[name] = np.array([myp])
            elif name == "prerun":
                prerun = "\n".join([l for l in value.strip().split("\n") if "In[" not in l])
                f[self.datapath].attrs[name] = prerun
                value = prerun
                # exec(prerun, globals())
            elif name in _info_args:
                f[self.datapath].attrs[name] = value
            else:
                f[self.args_datapath].attrs[name] = value

    def __getitem__(self, name):
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
        with h5py.File(self.filename, 'r') as f:
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
        return value
    
    def args(self):
        with h5py.File(self.filename, 'r') as f:
            for name in f[self.args_datapath].attrs:
                yield name, self[name]

    def qutypes_iterator(self, W, qutype2varname=None):
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
            elif qutype == 'fun':
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
                    arr.append(sh2fun(omegai))
                arr = np.squeeze(np.array(arr, dtype=dtype))

            yield qutype2varname[qutype], arr, qutype

    def _update_fieldnames(self):
        with h5py.File(self.filename, 'r') as f:
            for name in f[self.datapath].keys():
                dataset = f[self.datapath + name]
                if isinstance(dataset, h5py.Dataset):
                    self.fieldnames.update({name: (dataset.shape, dataset.dtype)})

    def initialize_field(self, W, time=0.0, **kwargs):
        try:
            f = h5py.File(self.filename, "r+")
        except IOError:
            raise IOError("Error while trying to write to file {}.".format(self.filename))
        else:
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
            for name in ['tol', 'iterations', 'maxit']:
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
            f.close()

    def __call__(self, W, delta_time, delta_steps=1, **kwargs):
        """

        Parameters
        ----------
        W: ndarray
        delta_time: float
        delta_steps: int
        kwargs: other variables to save
        """
        with h5py.File(self.filename, "r+") as f:

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


# ----------------------
# SOLVE FUNCTION DEF
# ----------------------

def solve(W, stepsize=None, timestep=None,
          steps=None, simtime=None,
          inner_steps=None, inner_time=None,
          integrator=None,
          callback=None, callback_kwargs=None,
          progress_bar=True, progress_file=None, **kwargs):
    """
    High-level solve function.

    Parameters
    ----------
    W: ndarray(shape=(N, N) or (k, N, N), dtype=complex)
        Initial vorticity matrix.
    stepsize: None or float
        Stepsize parameter, related to the actual time step length
        by `timestep = hbar * stepsize`.
        If neither `stepsize` nor `timestep` are specified,
        `stepsize` will be automatically selected.
    timestep: None or float
        Time step in seconds.
        Not used when `stepsize` is specified.
    steps: None or int
        Total number of steps to take.
    simtime: None or float
        Total simulation time in seconds.
        Not used when `steps` is specified.
    inner_steps: None or int
        Number of steps taken between each callback.
    inner_time: None or float
        Approximate time in seconds between each callback.
        Not used when `inner_steps` is specified.
    integrator: callable(W, stepsize, steps, **kwargs)
        Integration method to carry out the steps.
    callback: callable(W, inner_steps, inner_time, **callback_kwargs)
        The callback function evaluated every outer step.
        It uses **callback_kwargs as extra keyword arguments.
        It is not evaluated at the initial time.
    callback_kwargs: dict
        Extra keyword arguments to send to callback at each output step.
    progress_bar: bool
        Show progress bar (default: True)
    progress_file: TextIOWrapper or None
        File to write progress to (default: None)
    **kwargs: dict
        Extra keyword arguments to send to integrator at each step.
    """

    # Preprocess attributes from QuSimulation object
    if isinstance(W, QuSimulation):
        sim = W
        W = sim['mat',-1]
        if callback is None:
            callback = sim
        elif isinstance(callback, tuple):
            callback += (sim,)
        else:
            callback = (callback, sim)
        for name, value in sim.args():
            if name == 'stepsize' and stepsize is None:
                stepsize = value
            elif name == 'timestep' and timestep is None:
                timestep = value
            elif name == 'steps' and steps is None:
                steps = value
            elif name == 'simtime' and simtime is None:
                simtime = value
            elif name == 'inner_steps' and inner_steps is None:
                inner_steps = value
            elif name == 'inner_time' and inner_time is None:
                inner_time = value
            elif name == 'integrator' and integrator is None:
                integrator = value
            elif name == 'callback_kwargs' and callback_kwargs is None:
                callback_kwargs = value
            elif name == 'progress_bar' and progress_bar is None:
                progress_bar = value
            elif name == 'progress_file' and progress_file is None:
                progress_file = value
            else:
                # Add to kwargs
                if name not in kwargs:
                    kwargs[name] = value

    # Size of state matrix
    N = W.shape[-1]

    # Set default integrator if needed
    if integrator is None:
        integrator = isomp

    # Set default hamiltonian if needed
    integrator_kwargs = kwargs
    if 'hamiltonian' not in integrator_kwargs:
        integrator_kwargs['hamiltonian'] = solve_poisson
    if 'stats' in inspect.getfullargspec(integrator).args:
        integrator_kwargs['stats'] = {'iterations':0.0}

    # Determine steps
    if np.array([0 if x is None else 1 for x in [steps, simtime]]).sum() != 1:
        raise ValueError("One, and only one, of steps or simtime should be specified.")
    if simtime is not None:
        qtime = seconds2qtime(simtime, N)
        steps = round(qtime / np.abs(stepsize))
    if callback is not None and not isinstance(callback, tuple):
        callback = (callback,)
    if callback_kwargs is None:
        callback_kwargs = dict()

    # Determine inner_steps
    if np.array([0 if x is None else 1 for x in [inner_steps, inner_time]]).sum() == 0:
        inner_steps = 100  # Default value of inner_steps
    elif inner_steps is None:
        if inner_time is not None:
            inner_steps = round(seconds2qtime(inner_time, N) / np.abs(stepsize))

    # Check if inner_steps is too large
    if inner_steps > steps:
        inner_steps = steps

    # Create progressbar
    if progress_bar:
        try:
            if progress_file is None:
                if 'verbatim' in integrator_kwargs and integrator_kwargs['verbatim']:
                    progress_bar = False
                else:
                    from tqdm import tqdm
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
        W = integrator(W, stepsize, steps=no_steps, **integrator_kwargs)
        delta_time = qtime2seconds(no_steps*stepsize, N=N)
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


