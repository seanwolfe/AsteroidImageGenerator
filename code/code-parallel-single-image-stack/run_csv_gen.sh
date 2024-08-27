#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=3:00:0
#SBATCH --job-name snr-3-20-t
#SBATCH --output=mpi_fullspeed_output_%j.txt
#SBATCH --mail-type=FAIL

module load NiaEnv/2022a
module load python/3.11.5
module load intel/2022u2
module load openmpi/4.1.4+ucx-1.11.2
export MPLCONFIGDIR=$SCRATCH/matplotlib
source ~/.virtualenvs/myenv/bin/activate
mpirun python ImageGen_par.py
