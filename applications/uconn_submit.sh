#!/bin/bash
#SBATCH -J littlecooling
#SBATCH -o littlecooling.o%j
#SBATCH --ntasks=10
#SBATCH --partition=general
#SBATCH --time=4:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jiangce.chen@uconn.edu
module load gcc/12.2.0
source /home/jic17022/.bashrc
source activate anvil
mpirun python /home/jic17022/hpc_parallel_fem_process.py
