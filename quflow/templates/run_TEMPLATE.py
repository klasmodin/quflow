import numpy as np
import quflow as qf
import h5py

filename = "_FILENAME_"  # Should be *.hdf5 file
cache_size = 20
time = _SIMTIME_  # [s], total simulation time
inner_time = _INNER_TIME_  # [s], simulation time between outputs
qstep_size = _STEP_SIZE_  # [qtime], stepsize in qtime units
max_wait = 3600.  # In seconds, max wait between outputs

simulate = _SIMULATE_ # Whether to carry out the simulation
animate = _ANIMATE_ # Whether to save final animation

# Simulate system
if simulate:
    # Callback data object
    mysim = qf.QuData(filename, cache_size=cache_size, max_wait=max_wait, verbatim=False)

    # Load initial data from last step
    try:
        f = h5py.File(filename, "r")
    except IOError or KeyError:
        raise IOError("Something wrong with the input file. Does it exist?")
    else:
        W = qf.shr2mat(f['state'][-1, :])
        N = W.shape[0]
        f.close()

    # Create progress file
    with open(filename.replace(".hdf5", "-progress.txt"), 'w') as progress_file:
        # Run simulation
        qf.solve(W, qstepsize=qstep_size, time=time, inner_time=inner_time, callback=mysim,
                 progress_bar=True, progress_file=progress_file)

    # Flush cache data
    mysim.flush()

# Animate results
if animate:
    with h5py.File(filename, "r") as data, open(filename.replace(".hdf5", "-anim-progress.txt"), 'w') as progress_file:
        qf.create_animation2(filename.replace(".hdf5", ".mp4"),
                             data['state'],
                             projection='hammer', N=np.max([256,N]),
                             progress_bar=True, progress_file=progress_file)
