#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --output=script_install.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

#About this script:

#python3 "util_script_main.py"
python3 "ccs2_main.py"