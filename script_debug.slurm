#!/bin/bash
#SBATCH -J ZDB
#SBATCH --partition=debug
#SBATCH --qos=240c-1h_debug
#SBATCH --ntasks=1
#SBATCH --output=script_debug.out
#SBATCH --mail-user=neil.delgallego@dlsu.edu.ph
#SBATCH --mail-type=END

NETWORK_VERSION=$1
echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

# prepare working environment
module load anaconda
module load cuda
source activate NeilZoo

srun python util_script_main.py

srun python train_sr_main.py --server_config=8 --img_to_load=-1 \
--plot_enabled=0 --save_per_iter=500 --network_version=$NETWORK_VERSION

conda deactivate

