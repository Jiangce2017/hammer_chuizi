#!/bin/bash
#SBATCH -J littlecooling
#SBATCH -o littlecooling.o%j
#SBATCH --ntasks=32
#SBATCH --partition=general
#SBATCH --time=5:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jiangce.chen@uconn.edu
module load gcc/12.2.0
source /home/jic17022/.bashrc
source activate anvil
python /home/jic17022/hammer/hpc_fem_process.py
