#!/bin/env bash
#SBATCH -A C3SE2022-1-15 -p vera # find your project with the "projinfo" command
#SBATCH -t 4-00:00:00 # maximum simulation time is 4 days
#SBATCH -J $SIMNAME # name of job
#SBATCH -n $NO_CORES # number of cores to use
#SBATCH -N 1 # Use maximum 1 node
#SBATCH -C SKYLAKE # Use SKYLAKE (slower) or ICELAKE (faster)

module load GCCcore
module load FFmpeg
module load Anaconda3

python ./$RUNFILE > $SIMNAME_results.out
