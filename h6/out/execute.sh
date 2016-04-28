#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=00:20:00 

#SBATCH --partition=training
#SBATCH --account=parallelcomputing

# Change the below values to be for udc-ba30-5 or -7
#SBATCH --nodelist=udc-ba30-7
#SBATCH --job-name=test7
#SBATCH --output="out/out_cuda7.txt"


# Execute using parameters exported from run.sh
./mmm $dims $dims $dims $dims