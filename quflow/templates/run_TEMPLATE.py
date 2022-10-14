import numpy as np
import quflow as qf
import h5py
import os
import pickle


filename = "_FILENAME_"  # Should be *.hdf5 file
cache_size = 20
max_wait = 30*60.0  # In seconds, max wait between outputs

# Load pickled data
simname = os.path.splitext(os.path.basename(filename))[0]
args_file = os.path.join(os.path.dirname(filename), simname+'_args.pickle')
with open(args_file, 'rb') as f:
    (kwargs, clusterargs) = pickle.load(f)

simulate = clusterargs['simulate']
animate = clusterargs['animate']

# Initialize progress files
if simulate:
    with open(os.path.splitext(filename)[0]+"_progress.txt", 'w') as progress_file:
        progress_file.write("0% (not started yet)")
if animate:
    with open(os.path.splitext(filename)[0]+"_anim_progress.txt", 'w') as progress_file:
        progress_file.write("0% (not started yet)")

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
    with open(os.path.splitext(filename)[0]+"_progress.txt", 'w') as progress_file:
        # Run simulation
        qf.solve(W, **kwargs, progress_bar=True, progress_file=progress_file)

    # Flush cache data
    mysim.flush()

# Animate results
if animate:
    with h5py.File(filename, "r") as data, \
            open(os.path.splitext(filename)[0]+"_anim_progress.txt", 'w') as progress_file:
        N = round(np.sqrt(data['state'][0].shape[0]))
        qf.create_animation2(os.path.splitext(filename)[0]+".mp4",
                             data['state'],
                             projection='hammer', N=np.max([256, N]),
                             progress_bar=True, progress_file=progress_file)

# Finish
print("Runfile {} finished.".format(os.path.basename(__file__)))
