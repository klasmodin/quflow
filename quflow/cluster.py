import numpy as np
import os
import pickle
import subprocess
import quflow as qf

# ----------------
# GLOBAL VARIABLES
# ----------------
_DEFAULT_SERVER_ = "vera2"
# _DEFAULT_SERVER_ = "moklas@vera2.c3se.chalmers.se"
_RSYNC_COMMAND_ = 'rsync'
_SSH_COMMAND_ = 'ssh'
_RSYNC_ARGS_ = '-auv'
_BASH_SHEBANG_ = '#!/bin/bash'

# ----------------
# HELP FUNCTIONS
# ----------------


def get_simname(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def create_script_files(filename, run_template_str, bash_template_str, cores):
    """

    Parameters
    ----------
    filename
    run_template_str
    bash_template_str
    cores

    Returns
    -------
    tuple (run_filename, bash_filename)
    """

    # Get simname
    simname = get_simname(filename)

    # Prepare run string
    run_str = run_template_str.replace('_FILENAME_', os.path.basename(filename))

    # Write run file
    runfile = os.path.join(os.path.dirname(filename), "{}_run.py".format(simname))
    with open(runfile, 'w') as f:
        f.write(run_str)

    # Prepare bash string
    bash_str = bash_template_str\
        .replace('$SIMNAME', simname)\
        .replace('$NO_CORES', str(cores))\
        .replace('$RUNFILE', os.path.basename(runfile))

    # Write bash file
    bashfile = os.path.join(os.path.dirname(filename),"{}_submit.sh".format(simname))
    with open(bashfile, 'w') as f:
        f.write(bash_str)

    return runfile, bashfile


# --------------------
# HIGH LEVEL FUNCTIONS
# --------------------

def solve(filename,
          upload=True,
          submit=True,
          animate=True,
          simulate=True,
          server=None,
          server_prefix='simulations',
          run_template=None,
          bash_template=None,
          upload_quflow=True,
          cores=2,
          **kwargs):
    """

    Parameters
    ----------
    filename: str
        Name of hdm5 file containing the initial data.
    upload: bool
        Upload to server or not (default: True).
    upload_quflow: bool
        If upload=True this will also upload the quflow module (default: True).
    server: str or None
        Address to ssh server where to run script. (default: cluster._DEFAULT_SERVER_)
    server_prefix: str
        Remote folder prefix on server.
    submit: bool
        Run simulation script on server (default: True).
    animate: bool
        Create animation on server (default: True).
    simulate: bool
        Run simulation on server (default: True).
    run_template: str or file or None
        Template for run file (default: None).
    bash_template: str or file or None
        Template for submit script (default: None).
    cores: int
        Number of cores to use (default: 16).
    kwargs
        Named arguments to send to dynamics.solve.

    """

    # Check if file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError("Couldn't find file {}.".format(filename))

    # Check server
    if server is None:
        server = _DEFAULT_SERVER_

    # Get simname
    simname = get_simname(filename)

    # Check if simulation is currently running
    if upload or submit:
        if os.path.isfile(os.path.join(os.path.basename(filename),simname+'_args.pickle')):
            status_sim, status_anim = status(filename, verbatim=False)
            if (status_sim is not None and "100%" not in status_sim) or \
                    (status_anim is not None and "100%" not in status_anim):
                raise RuntimeError("It appears that file {} is currently running on {}. Aborting.".format(filename, server))

    # Pickle dynamics.solve arguments
    clusterargs = dict()
    clusterargs['animate'] = animate
    clusterargs['submit'] = submit
    clusterargs['simulate'] = simulate
    clusterargs['server'] = server
    clusterargs['server_prefix'] = server_prefix
    clusterargs['cores'] = cores

    args_file = os.path.join(os.path.dirname(filename), simname+'_args.pickle')
    if 'callback' in kwargs:
        RuntimeWarning('cluster.solve(...) does not allow callbacks. Ignoring.')
        kwargs.pop('callback')
    with open(args_file, 'wb') as f:
        pickle.dump((kwargs, clusterargs), f)

    # Create run template string
    if run_template is None:
        from . import templates
        with open(templates.__file__.replace("__init__.py", "run_TEMPLATE.py")) as f:
            run_template_str = f.read()

    # Create bash template string
    if bash_template is None:
        from . import templates
        with open(templates.__file__.replace("__init__.py", "vera2_TEMPLATE.sh")) as f:
            bash_template_str = f.read()

    # Create run and bash files
    runfile, bashfile = create_script_files(filename, run_template_str, bash_template_str, cores)

    # ----------------------
    # Upload files on server
    # ----------------------

    # Create list of local files to sync with server
    local_files = [filename, runfile, bashfile, args_file]

    # Create string for remote folder and files on server
    remote_folder = os.path.join(server_prefix, simname, '')
    remote_files = []
    remote_files += [os.path.join(remote_folder, os.path.basename(filename))]
    # remote_files += [os.path.join(remote_folder, simname+"_progress.txt")]
    # remote_files += [os.path.join(remote_folder, simname+"_anim_progress.txt")]
    remote_files += [os.path.join(remote_folder, simname+".mp4")]

    # If upload flag, then upload
    if upload:
        print('Uploading files to server.')
        for file in local_files:
            # Check that all files exist
            if not os.path.isfile(file):
                raise FileNotFoundError("The file {} to be uploaded was not found.".format(file))
        cmd = [_RSYNC_COMMAND_, _RSYNC_ARGS_]
        cmd += local_files + [server+":"+remote_folder]
        print("> " + " ".join(cmd))
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            raise RuntimeError("Could not upload files to server.")
        # Upload quflow module
        if upload_quflow:
            print('Uploading quflow to server.')
            cmd = [_RSYNC_COMMAND_, _RSYNC_ARGS_]
            cmd += [os.path.dirname(qf.__file__)] + [server+":"+remote_folder]
            print("> " + " ".join(cmd))
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError:
                raise RuntimeError("Could not upload quflow to server.")

    # Create upload script
    upload_str = _BASH_SHEBANG_ + "\n"
    upload_str += _RSYNC_COMMAND_
    upload_str += " " + _RSYNC_ARGS_
    upload_str += " " + " ".join([os.path.basename(s) for s in local_files])
    upload_str += " " + remote_folder
    upload_str += "\n"
    upload_script_file = os.path.join(os.path.dirname(filename), simname+"_upload.sh")
    with open(upload_script_file, 'w') as f:
        f.write(upload_str)

    # Create download script
    download_str = _BASH_SHEBANG_ + "\n"
    download_str += _RSYNC_COMMAND_
    download_str += " " + _RSYNC_ARGS_
    download_str += " " + server + ":'" + " ".join(remote_files) + "'"
    download_str += " ./"
    download_str += "\n"
    download_script_file = os.path.join(os.path.dirname(filename), simname+"_download.sh")
    with open(download_script_file, 'w') as f:
        f.write(download_str)

    # -----------------
    # Run submit script
    # -----------------

    if submit:
        # Check if needed files exist on server
        for file in remote_files:
            if "progress" in file or ".mp4" in file:
                continue
            try:
                subprocess.check_call([_SSH_COMMAND_, server, 'test', '-f', file])
            except subprocess.CalledProcessError:
                raise FileNotFoundError("The file {} does not exist on server {}.".format(file, server))

        # Check if quflow exists on server
        quflow_on_server = True
        try:
            subprocess.check_call([_SSH_COMMAND_, server, 'test', '-d', os.path.join(remote_folder, "quflow")])
        except subprocess.CalledProcessError:
            quflow_on_server = False

        # Create command str
        cmd_str = "\""
        cmd_str += "cd {} ;".format(remote_folder)
        if quflow_on_server:
            cmd_str += "export PYTHONPATH=:`pwd`/quflow$PYTHONPATH ;"
        cmd_str += "sbatch {} ;".format(os.path.basename(bashfile))
        cmd_str += "\""

        print("Submitting job on server.")
        print("> " + " ".join([_SSH_COMMAND_, server, cmd_str]))
        ret = os.system(" ".join([_SSH_COMMAND_, server, cmd_str]))
        if ret != 0:
            raise RuntimeError("Could not submit job on {}.".format(server))


def run_script(filename, subname):
    simname = get_simname(filename)
    script_file = os.path.join(os.path.dirname(filename), simname+"_{}.sh".format(subname))
    if not os.path.isfile(script_file):
        FileNotFoundError("Script file {} does not exist.".format(script_file))
    try:
        subprocess.check_call(['bash', script_file])
    except subprocess.CalledProcessError:
        RuntimeError("Not able to run {}.".format(script_file))


def retrieve(filename):
    run_script(filename, "download")


def status(filename, verbatim=True):
    simname = get_simname(filename)
    args_file = os.path.join(os.path.dirname(filename), simname+'_args.pickle')
    with open(args_file, 'rb') as f:
        (kwargs, clusterargs) = pickle.load(f)
    server = clusterargs['server']
    server_prefix = clusterargs['server_prefix']
    remote_folder = os.path.join(server_prefix, simname, '')

    status_str = None
    status_anim_str = None

    # Check ssh connection to server
    try:
        subprocess.check_call([_SSH_COMMAND_, server, 'echo', 'hej'])
    except subprocess.CalledProcessError:
        RuntimeError("Connection to server {} failed.".format(server))

    # Check progress of simulation
    remote_progress_file = os.path.join(remote_folder, simname+"_progress.txt")
    progress_exists = True
    try:
        subprocess.check_call([_SSH_COMMAND_, server, 'test', '-f', remote_progress_file])
    except subprocess.CalledProcessError:
        progress_exists = False
    if progress_exists:
        try:
            subprocess.check_call([_RSYNC_COMMAND_, server+":"+remote_progress_file, os.path.dirname(filename)])
            with open(os.path.join(os.path.dirname(filename), os.path.basename(remote_progress_file))) as f:
                for line in f:
                    pass
                status_str = line.strip()
            if verbatim:
                print("Simulation progress: "+status_str)
        except subprocess.CalledProcessError:
            RuntimeError("Unable to check progress.")

    # Check progress of animation
    remote_anim_progress_file = os.path.join(remote_folder, simname+"_anim_progress.txt")
    anim_progress_exists = True
    try:
        subprocess.check_call([_SSH_COMMAND_, server, 'test', '-f', remote_anim_progress_file])
    except subprocess.CalledProcessError:
        anim_progress_exists = False
    if anim_progress_exists:
        try:
            subprocess.check_call([_RSYNC_COMMAND_, server+":"+remote_anim_progress_file, os.path.dirname(filename)])
            with open(os.path.join(os.path.dirname(filename), os.path.basename(remote_anim_progress_file))) as f:
                for line in f:
                    pass
                status_anim_str = line.strip()
            if verbatim:
                print("Animation progress: "+status_anim_str)
        except subprocess.CalledProcessError:
            RuntimeError("Unable to check animation progress.")

    if not anim_progress_exists and not progress_exists:
        if verbatim:
            print("No progress reported on server {}.".format(server))

    if not verbatim:
        return status_str, status_anim_str

