import numpy as np
import os
import pickle
import subprocess
import quflow as qf

# ----------------
# GLOBAL VARIABLES
# ----------------
_DEFAULT_SERVER_ = "vera2"
_SERVER_ = _DEFAULT_SERVER_
_SERVER_PREFIX_ = "simulations"
# _DEFAULT_SERVER_ = "moklas@vera2.c3se.chalmers.se"
_RSYNC_COMMAND_ = 'rsync'
_SSH_COMMAND_ = 'ssh'
_RSYNC_UPLOAD_ARGS_ = '-auv'
_RSYNC_DOWNLOAD_ARGS_ = '-auv'
_BASH_SHEBANG_ = '#!/bin/bash'


# ----------------
# HELP FUNCTIONS
# ----------------


def get_simname(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def get_file(filename, ending=None, remote=False):
    basestr = "{}{}".format(get_simname(filename), ending) if ending is not None else os.path.basename(filename)
    if remote:
        return os.path.join(_SERVER_PREFIX_, basestr)
    else:
        return os.path.join(os.path.dirname(filename), basestr)


def get_runfile(filename, remote=False):
    return get_file(filename, "_run.py", remote=remote)


def get_submitfile(filename, remote=False):
    return get_file(filename, "_submit.sh", remote=remote)


def get_uploadfile(filename, remote=False):
    return get_file(filename, "_upload.sh", remote=remote)


def get_downloadfile(filename, remote=False):
    return get_file(filename, "_download.sh", remote=remote)


def get_argsfile(filename, remote=False):
    return get_file(filename, "_args.pickle", remote=remote)


def get_jobsfile(filename, remote=False):
    return get_file(filename, "_jobs.txt", remote=remote)


def get_animfile(filename, remote=False):
    return get_file(filename, ".mp4", remote=remote)


def get_progressfile(filename, remote=True, anim=False):
    ending = "_progress.txt"
    if anim:
        ending = "_anim" + ending
    return get_file(filename, ending, remote=remote)


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
    tuple (runfile_name, submitfile_name)
    """

    # Prepare run string
    run_str = run_template_str.replace('_FILENAME_', os.path.basename(filename))

    # Write run file
    runfile = get_runfile(filename)
    with open(runfile, 'w') as f:
        f.write(run_str)

    # Prepare bash string
    bash_str = bash_template_str\
        .replace('$SIMNAME', get_simname(filename))\
        .replace('$NO_CORES', str(cores))\
        .replace('$RUNFILE', os.path.basename(runfile))

    # Write bash file
    submitfile = get_submitfile(filename)
    with open(submitfile, 'w') as f:
        f.write(bash_str)

    return runfile, submitfile


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

    # Set global variables
    global _SERVER_, _SERVER_PREFIX_
    _SERVER_ = server
    _SERVER_PREFIX_ = server_prefix

    # Check if simulation is currently running
    if upload or submit:
        if os.path.isfile(get_argsfile(filename)):
            status_sim, status_anim = status(filename, verbatim=False)
            if (status_sim is not None and "status: running" in status_sim) or \
                    (status_anim is not None and "status: running" in status_anim):
                raise RuntimeError("The file {} is currently running on {}. Aborting.".format(filename, server))

    # Print header
    print("#### CREATING RUN FILES ####")

    # Get remote folder
    remote_folder = os.path.dirname(get_file(filename, remote=True))

    # Pickle arguments
    clusterargs = dict()
    clusterargs['animate'] = animate
    clusterargs['submit'] = submit
    clusterargs['simulate'] = simulate
    clusterargs['server'] = server
    clusterargs['remote_folder'] = remote_folder
    clusterargs['cores'] = cores

    if 'callback' in kwargs:
        RuntimeWarning('cluster.solve(...) does not allow callbacks. Ignoring.')
        kwargs.pop('callback')
    with open(get_argsfile(filename), 'wb') as f:
        pickle.dump((kwargs, clusterargs), f)

    # Create run template string
    if run_template is None:
        from . import templates
        with open(os.path.join(os.path.dirname(templates.__file__), "run_TEMPLATE.py"), 'r') as f:
            run_template_str = f.read()

    # Create bash template string
    if bash_template is None:
        from . import templates
        with open(templates.__file__.replace("__init__.py", "vera2_TEMPLATE.sh")) as f:
            bash_template_str = f.read()

    # Create run and bash files
    runfile, submitfile = create_script_files(filename, run_template_str, bash_template_str, cores)

    # Print header
    print("########### DONE ###########\n")

    # ----------------------
    # Upload files on server
    # ----------------------

    # Create list of local files to sync with server
    upload_files = [filename, runfile, submitfile, get_argsfile(filename)]

    # Create string for remote files to sync with local
    download_files = [get_file(filename, remote=True), get_animfile(filename, remote=True)]

    # If upload flag, then upload
    if upload:

        # Print header
        print("#### UPLOADING FILES TO SERVER ####")

        for file in upload_files:
            # Check that all files exist
            if not os.path.isfile(file):
                raise FileNotFoundError("The file {} to be uploaded was not found.".format(file))
        cmd = [_RSYNC_COMMAND_, _RSYNC_UPLOAD_ARGS_]
        cmd += upload_files + [server+":"+remote_folder]
        print("> " + " ".join(cmd))
        subprocess.run(cmd, check=True, text=True)

        # Upload quflow module
        if upload_quflow:
            print('Uploading quflow to server.')
            cmd = [_RSYNC_COMMAND_, _RSYNC_UPLOAD_ARGS_]
            cmd += [os.path.dirname(qf.__file__)] + [server+":"+remote_folder]
            print("> " + " ".join(cmd))
            subprocess.run(cmd, check=True, text=True)

        # Print header
        print("############### DONE ##############\n")


    # Create upload script
    upload_str = _BASH_SHEBANG_ + "\n"
    upload_str += _RSYNC_COMMAND_
    upload_str += " " + _RSYNC_UPLOAD_ARGS_
    upload_str += " " + " ".join([os.path.basename(s) for s in upload_files])
    upload_str += " " + remote_folder
    upload_str += "\n"
    with open(get_uploadfile(filename), 'w') as f:
        f.write(upload_str)

    # Create download script
    download_str = _BASH_SHEBANG_ + "\n"
    download_str += _RSYNC_COMMAND_
    download_str += " " + _RSYNC_DOWNLOAD_ARGS_
    download_str += " " + server + ":'" + " ".join(download_files) + "'"
    download_str += " ./"
    download_str += "\n"
    with open(get_downloadfile(filename), 'w') as f:
        f.write(download_str)

    # -----------------
    # Run submit script
    # -----------------

    if submit:

        # Print header
        print("#### SUBMITTING JOB ON SERVER ####")

        # Check if needed files exist on server
        for file in upload_files:
            remote_file = get_file(file, remote=True)
            cp = subprocess.run([_SSH_COMMAND_, server, 'test', '-f', remote_file], text=True)
            if cp.returncode != 0:
                raise FileNotFoundError("File {} not found on server {}.".format(remote_file, server))

        # Check if quflow exists on server
        quflow_on_server = True
        cp = subprocess.run([_SSH_COMMAND_, server, 'test', '-d', os.path.join(remote_folder, "quflow")])
        if cp.returncode != 0:
            quflow_on_server = False

        # Create command str to run on server
        cmd_str = "\""

        # change to run folder
        cmd_str += "cd {} ;".format(remote_folder)

        if quflow_on_server:
            # set PYTHONPATH to include quflow
            cmd_str += "export PYTHONPATH=:`pwd`/quflow$PYTHONPATH ;"

        # remove old progress files
        cmd_str += "rm -f {} {} ;".format(os.path.basename(get_progressfile(filename, anim=False)),
                                        os.path.basename(get_progressfile(filename, anim=True)))
        cmd_str += "sbatch {} ;".format(os.path.basename(get_submitfile(filename, remote=True)))

        cmd_str += "\""

        # Create command
        cmd = [_SSH_COMMAND_, server, cmd_str]

        # Run command
        print("> " + " ".join(cmd))
        cp = subprocess.run(" ".join(cmd), text=True, capture_output=True, shell=True)

        # Check output from command
        if cp.returncode != 0:
            print(cp.stdout)
            print(cp.stderr)
            raise RuntimeError("Could not submit job on {}.".format(server))
        else:
            print(cp.stdout.strip())
            jobid = [int(s) for s in cp.stdout.split() if s.isdigit()]
            if len(jobid) == 0:
                print("Could not extract jobid from submitted job.")
            else:
                jobid = jobid[0]
                print("Extracted jobid {}.".format(str(jobid)))
                with open(get_jobsfile(filename), 'w') as f:
                    f.write(str(jobid))

        # Print header
        print("############## DONE ##############")


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


def ssh_connection(server):
    cp = subprocess.run([_SSH_COMMAND_, server, 'exit', '0'], text=True)
    return True if cp.returncode == 0 else False


def jobstatus(server=None, verbatim=True):
    if server is None:
        server = _SERVER_
    # Run squeue command on server
    cp = subprocess.run([_SSH_COMMAND_, server,  'squeue',  '-u', '$USER'], text=True, capture_output=True, check=True)
    if verbatim:
        print(cp.stdout.strip())
    else:
        return cp.stdout.strip()


def status(filename, verbatim=True):

    # This indicates no status can be reported.
    status_str = None
    status_anim_str = None

    # Check if args file exists.
    if os.path.isfile(get_argsfile(filename)):

        # Load args
        with open(get_argsfile(filename), 'rb') as f:
            (kwargs, clusterargs) = pickle.load(f)

        # Assign variables
        server = clusterargs['server']
        remote_folder = clusterargs['remote_folder']

        # Check if ssh connection works
        if ssh_connection(server):

            # Check if job is still running
            try:
                with open(get_jobsfile(filename), 'r') as f:
                    jobid = f.read().strip()
            except FileNotFoundError:
                job_str = "n/a"
                Warning("Could not establish jobstatus.")
            else:
                # Set job str
                job_str = "running" if jobid in jobstatus(server, verbatim=False) else "not running"

            for anim in (False, True):

                # Run tail -1 command on progress file on server
                cp = subprocess.run([_SSH_COMMAND_, server, 'tail', '-1',
                                     get_progressfile(filename, remote=True, anim=anim)],
                                    text=True, capture_output=True)

                # Check output from stdout
                if cp.returncode == 0:
                    if anim:
                        status_anim_str = cp.stdout.strip().split('\n')[-1] + " (jobstatus: {})".format(job_str)
                    else:
                        status_str = cp.stdout.strip().split('\n')[-1] + " (jobstatus: {})".format(job_str)

        else:
            Warning("Could not establish ssh connection.")
    if verbatim:
        print("Simulation: {}".format("n/a" if status_str is None else status_str))
        print(" Animation: {}".format("n/a" if status_anim_str is None else status_anim_str))
    else:
        return status_str, status_anim_str

