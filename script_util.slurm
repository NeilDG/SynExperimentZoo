#!/bin/bash
#SBATCH -J UTILITY
#SBATCH --partition=serial
#SBATCH --qos=84c-1d_serial
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=script_util.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

# prepare working environment
module load anaconda
module load cuda
source activate NeilZoo

srun python util_script_main.py

conda deactivate