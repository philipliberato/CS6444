#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --time=00:05:00 

#SBATCH --partition=training
#SBATCH --account=parallelcomputing

#SBATCH --job-name=tsp2
#SBATCH --output="out/out_tsp2.txt"

module load openmpi/gcc

mpiexec ./tsp