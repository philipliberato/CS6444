#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=00:05:00 

#SBATCH --partition=training
#SBATCH --account=parallelcomputing

# Change the below values to be for udc-ba30-5 or -7
#SBATCH --nodelist=udc-ba30-5
#SBATCH --job-name=test5
#SBATCH --output="out/out_cuda5.txt"


# Execute using parameters exported from run.sh
./mmm $dims $dims $dims $dims