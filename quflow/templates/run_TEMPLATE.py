import numpy as np
import quflow as qf
import h5py
import os
import pickle
from quflow import cluster

############ PRERUN CODE BEGIN #############
#_PRERUNCODE_
############ PRERUN CODE END ###############

filename = "_FILENAME_"  # Should be *.hdf5 file
cache_size = 20
max_wait = 30*60.0  # In seconds, max wait between outputs

# Load cluster data
with open(cluster.get_clusterfile(filename), 'rb') as f:
    clusterargs = pickle.load(f)

# Load kwargs data
with open(cluster.get_argsfile(filename), 'rb') as f:
    kwargs = pickle.load(f)

simulate = clusterargs['simulate']
animate = clusterargs['animate']

# Initialize progress files
if simulate:
    with open(cluster.get_progressfile(filename, remote=False), 'w') as f:
        f.write("  0%|waiting...|")
if animate:
    with open(cluster.get_progressfile(filename, remote=False, anim=True), 'w') as f:
        f.write("  0%|waiting...|")

# Simulate system
if simulate:
    # Callback data object
    mysim = qf.QuData(filename, cache_size=cache_size, max_wait=max_wait, verbatim=False)
    kwargs['callback'] = mysim

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
    with open(cluster.get_progressfile(filename, remote=False), 'w') as f:
        # Run simulation
        qf.solve(W, **kwargs, progress_bar=True, progress_file=f)

    # Flush cache data
    mysim.flush()

# Animate results
if animate:
    with h5py.File(filename, "r") as data, \
            open(cluster.get_progressfile(filename, remote=False, anim=True), 'w') as progress_file:
        N = round(np.sqrt(data['state'][0].shape[0]))
        qf.create_animation2(cluster.get_animfile(filename),
                             data['state'],
                             projection='hammer', N=np.max([256, N]),
                             progress_bar=True, progress_file=progress_file)

# Finish
print("Runfile {} finished.".format(os.path.basename(__file__)))
