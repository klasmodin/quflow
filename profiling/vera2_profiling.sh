#!/bin/env bash
#SBATCH -A C3SE2023-1-2 -p vera # find your project with the "projinfo" command
#SBATCH -t 0-01:00:00 # maximum simulation time is 1 hour
#SBATCH -J quflow-profiling # name of job
#SBATCH -n 4 # number of cores to use
#SBATCH -N 1 # Use maximum 1 node
#SBATCH -C SKYLAKE # Use SKYLAKE (slower) or ICELAKE (faster)

# if command -v module &> /dev/null
# then
#     module load Anaconda3
#     export PYTHONPATH=$PYTHONPATH:`pwd`/../quflow ;
# fi

if type module > /dev/null 2>&1
then
    echo "Loading Anaconda3 module"
    module load Anaconda3
    export PYTHONPATH=$PYTHONPATH:`pwd`/../quflow
fi


today=`date '+%Y-%m-%d'`;
arch=`uname -m`;
filename="profile_results_${arch}_${today}.txt"

echo "Running profiling, output in file ${filename}"

python ./run_profiling.py > $filename
