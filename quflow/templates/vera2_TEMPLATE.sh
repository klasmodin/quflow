#!/bin/env bash
#SBATCH -A C3SE2023-1-2 -p vera # find your project with the "projinfo" command
#SBATCH -t 4-00:00:00 # maximum simulation time is 4 days
#SBATCH -J $SIMNAME # name of job
#SBATCH -n $NO_CORES # number of cores to use
#SBATCH -N 1 # Use maximum 1 node
#SBATCH -C $ARCH # Use SKYLAKE (slower) or ICELAKE (faster)

if command -v module &> /dev/null
then
    # module load GCCcore/11.3.0 
    module load FFmpeg
    # module load FFmpeg/4.4.2-GCCcore-11.3.0
    # module load GEOS/3.10.3-GCC-11.3.0
    module load Anaconda3
fi

python ./$RUNFILE > $SIMNAME_results.out
