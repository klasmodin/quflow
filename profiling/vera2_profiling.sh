#!/bin/env bash
#SBATCH -A C3SE2023-1-2 -p vera # find your project with the "projinfo" command
#SBATCH -t 0-01:00:00 # maximum simulation time is 1 hour
#SBATCH -J quflow-profiling # name of job
#SBATCH -n 4 # number of cores to use
#SBATCH -N 1 # Use maximum 1 node
#SBATCH -C SKYLAKE # Use SKYLAKE (slower) or ICELAKE (faster)

basedir=$(dirname "$0")
if command -v module &> /dev/null
then
    module load Anaconda3
    # module load SciPy-bundle/2022.05-intel-2022a
    # module load LLVM/14.0.3-GCCcore-11.3.0
    export PYTHONPATH=$PYTHONPATH:${HOME}/quflow 
    # basedir="${HOME}/quflow/profiling"
fi


# today=`date '+%Y-%m-%d'`;
# arch=`uname -m`;
# filename="${basedir}/profile_results_${arch}_${today}.txt"

# echo "Running profiling, output in file ${filename}"

python run_profiling.py "$@"
