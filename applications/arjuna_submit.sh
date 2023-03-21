#!/bin/bash
#SBATCH -J littlecooling
#SBATCH -o littlecooling.o%j
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --partition=debug
#SBATCH --time=00:05:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jiangcec@andrew.cmu.edu
source /home/jiangcec/.bashrc
source activate anvil
python /home/jiangcec/hammer/hpc_fem_process.py
