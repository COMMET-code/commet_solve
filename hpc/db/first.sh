#!/bin/sh
#
#SBATCH --job-name="test"
#SBATCH --partition=compute
#SBATCH --time=00:04:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Research-ME-MSE

# module load 2023r1
# module load openmpi
module load 2024r1 cmake openmpi petsc trilinos hdf5 gmsh parmetis zlib

srun ./bin/commet_solve
