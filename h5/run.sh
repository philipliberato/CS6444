#!/bin/bash


module load openmpi/gcc

# Re-compile the program
mpiCC -O3 tsp.cpp -o tsp

# Submit job
sbatch execute.sh
