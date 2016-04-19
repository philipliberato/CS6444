#!/bin/bash

# Grab dimensions from command line
dims=$1

# Export dimensions so execute.sh can use them
export dims

# Get the compiler
module load cuda

# Re-compile the program
nvcc -O3 mmm.cu -o mmm

# Submit job
sbatch execute.sh
